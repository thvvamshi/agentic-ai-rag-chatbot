import os
import re
from collections import Counter
from pathlib import Path

from ftfy import fix_text
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)


PDF_PATH = Path(__file__).resolve().parent.parent / "data" / "Ebook-Agentic-AI.pdf"

PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")

# graph.py keeps a local copy of all chunks here; it is deleted after
# every ingestion so the two can never disagree.
CHUNKS_CACHE_PATH = Path(os.getenv("CHUNKS_CACHE_PATH", "data/chunks.json"))

MIN_CHUNK_CHARS = 40

# A short line that repeats on at least this share of pages is treated
# as a running header/footer/page number and removed.
REPEATED_LINE_RATIO = 0.5

# Numbered sub-section headings such as
#   "1.5 Agentic AI Use cases"
#   "2.1 The Core Pillars: From Perception to Execution"
# Adjust this pattern if your PDF numbers its headings differently.
HEADING_PATTERN = re.compile(
    r"^\d{1,2}\.\d{1,2}(?:\.\d{1,2})?\.?\s+[A-Za-z].{2,120}$"
)

BOILERPLATE_PATTERNS = (
    "table of contents",
    "scan this qr",
    "download an e-copy",
)

# A heading ending in one of these words continues on the next line.
HEADING_CONTINUATION_WORDS = {
    "of", "and", "for", "the", "in", "to", "with",
    "a", "an", "on", "or", "by", "at", "as",
}


# --- text cleanup helpers ---------------------------------------------------
def _to_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _normalize_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line.strip().lower())

    return re.sub(r"\d+", "#", line)


def find_repeated_lines(texts: list[str]) -> set[str]:
    """Running headers, footers and page numbers that repeat across pages."""
    if len(texts) < 6:
        return set()

    counts: Counter = Counter()

    for text in texts:
        lines = {
            _normalize_line(line)
            for line in text.splitlines()
            if line.strip()
        }

        counts.update(line for line in lines if len(line) <= 80)

    threshold = max(3, int(len(texts) * REPEATED_LINE_RATIO))

    return {line for line, count in counts.items() if count >= threshold}


def is_heading(line: str) -> bool:
    if not HEADING_PATTERN.match(line):
        return False

    # Table-of-contents entries: dot leaders or a trailing page number.
    if re.search(r"\.{3,}", line) or re.search(r"\s\d{1,3}$", line):
        return False

    # Sentences that merely start with a number.
    if line.endswith((".", ",", ";")):
        return False

    return True


def is_boilerplate(text: str) -> bool:
    lowered = text.lower()

    return any(pattern in lowered for pattern in BOILERPLATE_PATTERNS)


# --- section-aware chunking -------------------------------------------------
def build_chunks(pages) -> list[dict]:
    """Split pages into chunks that each carry their section heading.

    `pages` are LangChain documents (page_content + metadata["page"]).
    """
    texts = [fix_text(page.page_content) for page in pages]

    repeated = find_repeated_lines(texts)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks: list[dict] = []
    headings: list[tuple[int, str]] = []
    skipped = 0
    section = ""
    seen_headings: set[str] = set()

    for page_doc, text in zip(pages, texts):
        page = _to_int(page_doc.metadata.get("page", 0))

        if "table of contents" in text.lower():
            continue

        # Break the page wherever a new heading starts, so no chunk
        # straddles two sections. The current section carries over
        # from the previous page.
        segments: list[tuple[str, str]] = []
        buffer: list[str] = []

        lines = text.splitlines()
        i = 0

        while i < len(lines):
            stripped = lines[i].strip()
            i += 1

            if not stripped:
                buffer.append("")
                continue

            if _normalize_line(stripped) in repeated:
                continue

            if is_heading(stripped):
                # A heading that wraps onto a second line ends on a
                # connector word ("... Strategies of"); pull the rest in.
                joins = 0

                while (
                    stripped.split()[-1].lower() in HEADING_CONTINUATION_WORDS
                    and joins < 2
                    and i < len(lines)
                ):
                    next_line = lines[i].strip()
                    i += 1

                    if next_line:
                        stripped = f"{stripped} {next_line}"
                        joins += 1

                # A title seen before is a running page header, not a
                # new section, so it must not change the current section.
                if stripped in seen_headings:
                    continue

                seen_headings.add(stripped)

                segments.append((section, "\n".join(buffer)))
                buffer = []
                section = stripped
                headings.append((page, stripped))
                continue

            buffer.append(stripped)

        segments.append((section, "\n".join(buffer)))

        for segment_section, segment_text in segments:
            segment_text = segment_text.strip()

            if not segment_text:
                continue

            for piece in splitter.split_text(segment_text):
                piece = piece.strip()

                if len(piece) < MIN_CHUNK_CHARS or is_boilerplate(piece):
                    skipped += 1
                    continue

                body = (
                    f"Section: {segment_section}\n\n{piece}"
                    if segment_section
                    else piece
                )

                chunks.append(
                    {
                        "text": body,
                        "section": segment_section,
                        "page": page,
                        "source": PDF_PATH.name,
                    }
                )

    for chunk_idx, chunk in enumerate(chunks):
        chunk["chunk_idx"] = chunk_idx

    print(f"Detected {len(headings)} section headings:")

    for page, heading in headings:
        print(f"  page {page}: {heading}")

    if not headings:
        print(
            "WARNING: no headings detected. "
            "Check HEADING_PATTERN against your PDF's heading format."
        )

    print(f"Removed {len(repeated)} repeated header/footer line patterns:")

    for pattern in sorted(repeated)[:15]:
        print(f"  {pattern!r}")
    print(f"Skipped {skipped} tiny or boilerplate chunks")

    return chunks


def load_documents() -> list[dict]:
    """Load the source PDF and split it into section-aware chunks."""

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    pages = PyPDFLoader(str(PDF_PATH)).load()

    chunks = build_chunks(pages)

    print(f"Loaded {len(pages)} pages")
    print(f"Created {len(chunks)} chunks")

    return chunks


# --- Pinecone ---------------------------------------------------------------
def get_pinecone_index():
    """Create the Pinecone index if needed and return it."""

    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is missing")

    pinecone = Pinecone(api_key=PINECONE_API_KEY)

    index_names = [index["name"] for index in pinecone.list_indexes()]

    if PINECONE_INDEX_NAME not in index_names:
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")

        pinecone.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        print("Pinecone index created")

    else:
        print(f"Using existing index: {PINECONE_INDEX_NAME}")

    return pinecone.Index(PINECONE_INDEX_NAME)


def ingest():
    """Load, embed and store the PDF chunks in Pinecone."""

    chunks = load_documents()

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Generating embeddings...")

    embeddings = embedding_model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    vectors = [
        {
            "id": f"chunk-{chunk['chunk_idx']}",
            "values": embedding.tolist(),
            "metadata": {
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_idx": chunk["chunk_idx"],
                "section": chunk["section"],
            },
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]

    index = get_pinecone_index()

    # Remove vectors from any earlier ingestion, otherwise old chunks
    # with higher ids would stay searchable.
    try:
        index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
        print("Cleared previous vectors")
    except Exception as error:
        print(f"Nothing to clear ({error})")

    batch_size = 100

    print(f"Uploading {len(vectors)} vectors...")

    for start in range(0, len(vectors), batch_size):
        batch = vectors[start:start + batch_size]

        index.upsert(vectors=batch, namespace=PINECONE_NAMESPACE)

        print(
            f"Uploaded {min(start + batch_size, len(vectors))}"
            f"/{len(vectors)}"
        )

    if CHUNKS_CACHE_PATH.exists():
        CHUNKS_CACHE_PATH.unlink()
        print(f"Deleted stale cache: {CHUNKS_CACHE_PATH}")

    print("Ingestion completed successfully.")


if __name__ == "__main__":
    ingest()