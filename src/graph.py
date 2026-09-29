import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    TOP_K,
)

logger = logging.getLogger(__name__)

REFUSAL_MESSAGE = "I cannot answer based on the provided document."

DEFAULT_SOURCE = "Ebook-Agentic-AI.pdf"

# Local copy of every chunk, used by BM25. ingestion.py deletes it after
# each ingestion, and it is also rebuilt automatically if it looks stale.
CHUNKS_CACHE_PATH = Path(os.getenv("CHUNKS_CACHE_PATH", "data/chunks.json"))

PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")

FINAL_CONTEXT_SIZE = 6
RRF_K = 60

# If even the best vector match is below this cosine score, the question
# is treated as out of scope. Tune against a few off-topic questions.
MIN_SEMANTIC_SCORE = float(os.getenv("MIN_SEMANTIC_SCORE", "0.30"))

CONFIDENCE_WEIGHTS = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2]

STOPWORDS = {
    "the", "and", "are", "for", "what", "which", "who", "how", "does", "did",
    "with", "that", "this", "from", "into", "about", "their", "its", "was",
    "were", "can", "list", "name", "give", "tell", "explain", "document",
    "ebook", "according", "there", "have", "has", "not", "you", "your",
}


class AgentState(TypedDict, total=False):
    question: str
    context: list[str]
    retrieved_context_chunks: list[dict]
    answer: str
    score: float             # retrieval confidence (0-1), kept as "score" for the UI
    answer_support: float    # share of answer terms found in the context


# --- cached clients ---------------------------------------------------------
@lru_cache
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache
def get_pinecone_index():
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is missing")

    pinecone = Pinecone(api_key=PINECONE_API_KEY)

    return pinecone.Index(PINECONE_INDEX_NAME)


@lru_cache
def get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing")

    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
    )


# --- text helpers -----------------------------------------------------------
def tokenize(text: str) -> list[str]:
    tokens = []

    for word in re.findall(r"[a-zA-Z0-9]+", text.lower()):
        if len(word) < 3 or word in STOPWORDS:
            continue

        if word.endswith("ies") and len(word) > 5:
            word = word[:-3] + "y"
        elif word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]

        tokens.append(word)

    return tokens


def normalize_terms(text: str) -> set[str]:
    return set(tokenize(text))


def _to_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _natural_key(text: str) -> list:
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", text)
    ]


# --- local corpus (for BM25) ------------------------------------------------
def _chunk_from_metadata(chunk_id: str, metadata: dict) -> dict | None:
    text = str(metadata.get("text", "")).strip()

    if not text:
        return None

    return {
        "id": chunk_id,
        "text": text,
        "source": metadata.get("source", DEFAULT_SOURCE),
        "page": _to_int(metadata.get("page", 0)),
        "chunk_idx": _to_int(metadata.get("chunk_idx", -1), -1),
        "section": metadata.get("section", ""),
    }


def _download_corpus() -> list[dict]:
    """Pull every chunk out of Pinecone once (serverless indexes only)."""
    index = get_pinecone_index()
    chunks = []

    for id_batch in index.list(namespace=PINECONE_NAMESPACE):
        fetched = index.fetch(
            ids=id_batch,
            namespace=PINECONE_NAMESPACE,
        )

        for chunk_id, vector in fetched.vectors.items():
            chunk = _chunk_from_metadata(chunk_id, vector.metadata or {})

            if chunk:
                chunks.append(chunk)

    return chunks


def _expected_vector_count() -> int | None:
    """How many vectors Pinecone holds, used to detect a stale cache."""
    try:
        stats = get_pinecone_index().describe_index_stats()

        if PINECONE_NAMESPACE:
            namespace = stats.namespaces.get(PINECONE_NAMESPACE)
            return namespace.vector_count if namespace else 0

        return stats.total_vector_count
    except Exception:
        logger.warning("Could not read index stats; skipping cache check")
        return None


@lru_cache
def get_corpus() -> tuple[list[dict], dict[str, int]]:
    """Returns (chunks in document order, chunk_id -> position)."""
    chunks = None

    if CHUNKS_CACHE_PATH.exists():
        try:
            chunks = json.loads(
                CHUNKS_CACHE_PATH.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            logger.warning("Unable to read chunk cache; rebuilding")

        expected = _expected_vector_count()

        if chunks and expected is not None and len(chunks) != expected:
            logger.info(
                "Chunk cache is stale (%d cached vs %d in Pinecone); rebuilding",
                len(chunks),
                expected,
            )
            chunks = None

    if not chunks:
        try:
            chunks = _download_corpus()
        except Exception:
            logger.exception(
                "Could not build the local chunk corpus; "
                "using vector-only retrieval"
            )
            return [], {}

        if chunks:
            CHUNKS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            CHUNKS_CACHE_PATH.write_text(
                json.dumps(chunks),
                encoding="utf-8",
            )

    if not chunks:
        return [], {}

    chunks.sort(
        key=lambda c: (
            str(c["source"]),
            c["chunk_idx"],
            _natural_key(c["id"]),
        )
    )

    positions = {chunk["id"]: i for i, chunk in enumerate(chunks)}

    return chunks, positions


@lru_cache
def get_bm25() -> BM25Okapi | None:
    chunks, _ = get_corpus()

    if not chunks:
        return None

    return BM25Okapi([tokenize(c["text"]) or ["_"] for c in chunks])


def bm25_search(question: str, limit: int) -> list[str]:
    bm25 = get_bm25()

    if bm25 is None:
        return []

    query_tokens = tokenize(question)

    if not query_tokens:
        return []

    chunks, _ = get_corpus()
    scores = bm25.get_scores(query_tokens)

    ranked = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    return [chunks[i]["id"] for i in ranked[:limit] if scores[i] > 0]


def reciprocal_rank_fusion(rank_lists: list[list[str]]) -> dict[str, float]:
    fused: dict[str, float] = {}

    for ranks in rank_lists:
        for rank, chunk_id in enumerate(ranks):
            fused[chunk_id] = (
                fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            )

    return fused


# --- graph nodes ------------------------------------------------------------
def _empty_retrieval() -> dict:
    return {"context": [], "retrieved_context_chunks": []}


def retrieve(state: AgentState) -> dict:
    question = state["question"]

    embedding_model = get_embedding_model()
    index = get_pinecone_index()

    query_vector = embedding_model.encode(
        question,
        normalize_embeddings=True,
    )

    # 1) semantic retrieval
    result = index.query(
        vector=query_vector.tolist(),
        top_k=TOP_K,
        include_metadata=True,
        namespace=PINECONE_NAMESPACE,
    )

    vector_chunks: dict[str, dict] = {}
    vector_ids: list[str] = []

    for match in result.matches:
        chunk = _chunk_from_metadata(match.id, match.metadata or {})

        if chunk is None:
            continue

        chunk["semantic_score"] = round(float(match.score), 4)
        vector_chunks[chunk["id"]] = chunk
        vector_ids.append(chunk["id"])

    # Relevance gate: nothing close to the question means out of scope.
    if not vector_ids or max(
        chunk["semantic_score"] for chunk in vector_chunks.values()
    ) < MIN_SEMANTIC_SCORE:
        return _empty_retrieval()

    # 2) lexical retrieval over the whole document, then fuse with RRF
    corpus, positions = get_corpus()
    by_id = {chunk["id"]: chunk for chunk in corpus}

    lexical_ids = bm25_search(question, TOP_K)

    fused = reciprocal_rank_fusion([vector_ids, lexical_ids])

    ranked_ids = sorted(fused, key=fused.get, reverse=True)

    # 3) take the top chunks
    selected: list[dict] = []

    for chunk_id in ranked_ids:
        base = vector_chunks.get(chunk_id) or by_id.get(chunk_id)

        if base is None:
            continue

        selected.append({**base, "rrf_score": round(fused[chunk_id], 5)})

        if len(selected) == FINAL_CONTEXT_SIZE:
            break

    if not selected:
        return _empty_retrieval()

    # Lexical-only hits have no Pinecone score yet, so embed them locally.
    unscored = [c for c in selected if "semantic_score" not in c]

    if unscored:
        vectors = embedding_model.encode(
            [c["text"] for c in unscored],
            normalize_embeddings=True,
        )

        for chunk, vector in zip(unscored, vectors):
            chunk["semantic_score"] = round(
                max(0.0, float(vector @ query_vector)),
                4,
            )

    # Present the chunks in document order so lists and steps read naturally.
    selected.sort(key=lambda c: positions.get(c["id"], 10**9))

    return {
        "context": [chunk["text"] for chunk in selected],
        "retrieved_context_chunks": selected,
    }


SYSTEM_PROMPT = """
You are a document question-answering assistant.

Answer the user's question using ONLY the provided context
from the Agentic AI eBook. The context passages are given in
document order, and each may begin with its section heading.

Rules:

1. Use only information present in the context.
2. Do not use outside knowledge.
3. Do not invent facts or add details the context does not state.
4. Do not assume the user's premise is correct.
5. If the question states a number, name, or category that
   conflicts with the document, say the premise is not
   supported, then state what the document actually says,
   including the exact count and the complete list.
6. Preserve the exact terminology stated in the document.
7. When the context contains a list or sequence, report every
   item, even if the items are spread across several passages.
8. If the context does not support the answer, say exactly:

"I cannot answer based on the provided document."

Answer clearly and directly.
"""

HUMAN_PROMPT = """
Question:
{question}

Document context:
{context}

Answer:
"""


def generate(state: AgentState) -> dict:
    context = state["context"]

    if not context:
        return {"answer": REFUSAL_MESSAGE}

    formatted_context = "\n\n---\n\n".join(context)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )

    response = (prompt | get_llm()).invoke(
        {
            "question": state["question"],
            "context": formatted_context,
        }
    )

    answer = response.content

    if isinstance(answer, list):
        answer = "".join(
            item.get("text", "")
            for item in answer
            if isinstance(item, dict)
        )

    return {"answer": str(answer).strip()}


def answer_support(question: str, answer: str, context: list[str]) -> float:
    """Share of the answer's content terms that appear in the context.

    Terms echoed from the question are ignored, so correcting a false
    premise ("there are not seven...") is not penalized.
    """
    answer_terms = normalize_terms(answer) - normalize_terms(question)

    if not answer_terms:
        return 0.0

    context_terms = normalize_terms(" ".join(context))

    return round(len(answer_terms & context_terms) / len(answer_terms), 4)


def grounding_check(state: AgentState) -> dict:
    chunks = state["retrieved_context_chunks"]
    answer = state["answer"]

    if not chunks or answer == REFUSAL_MESSAGE:
        return {"score": 0.0, "answer_support": 0.0}

    scores = sorted(
        (
            chunk["semantic_score"]
            for chunk in chunks
            if isinstance(chunk.get("semantic_score"), (int, float))
        ),
        reverse=True,
    )

    if not scores:
        return {"score": 0.0, "answer_support": 0.0}

    # Weight the strongest evidence more heavily.
    weights = CONFIDENCE_WEIGHTS[: len(scores)]

    confidence = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    confidence = max(0.0, min(1.0, confidence))

    return {
        "score": round(confidence, 4),  # retrieval confidence, not correctness
        "answer_support": answer_support(
            state["question"],
            answer,
            state["context"],
        ),
    }


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("grounding_check", grounding_check)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "grounding_check")
    workflow.add_edge("grounding_check", END)

    return workflow.compile()


rag_graph = build_graph()