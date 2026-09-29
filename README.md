# Agentic AI RAG Chatbot

A document-grounded Retrieval-Augmented Generation (RAG) chatbot built with Python, LangGraph, Pinecone, FastAPI, and Streamlit.

The chatbot answers questions using the provided **Agentic AI eBook**, retrieves relevant context, and returns a confidence score. It implements document ingestion, vector retrieval, LangGraph orchestration, grounded generation, and API/UI output.

## Features

- PDF parsing with `PyPDFLoader`
- Section-aware text chunking
- Text cleanup and repeated header/footer removal
- Sentence Transformer embeddings
- Pinecone vector storage
- BM25 lexical retrieval
- Reciprocal Rank Fusion (RRF)
- LangGraph RAG workflow
- Strict context-grounded generation
- Grounding and confidence scoring
- FastAPI `/chat` endpoint
- Lightweight Streamlit interface
- Validation and out-of-scope testing

## Architecture

```text
Ebook-Agentic-AI.pdf
        │
        ▼
   PDF Extraction
   (PyPDFLoader)
        │
        ▼
 Cleaning + Section
     Detection
        │
        ▼
      Chunking
        │
        ├───────────────┐
        ▼               ▼
  Embeddings          BM25
        │               │
        ▼               │
    Pinecone            │
        │               │
        └───────┬───────┘
                ▼
          RRF Hybrid
           Retrieval
                │
                ▼
          Relevant Chunks
                │
                ▼
            LangGraph
                │
        ┌───────┴────────┐
        ▼                ▼
    Retrieve         Generate
                         │
                         ▼
                  Grounding Check
                         │
                ┌────────┴────────┐
                ▼                 ▼
             FastAPI          Streamlit
````

The graph-based RAG workflow covers ingestion, chunking, retrieval, grounded generation, and confidence/grounding verification.

## Tech Stack

| Component         | Technology                     |
| ----------------- | ------------------------------ |
| Language          | Python 3.11+                   |
| Package Manager   | uv                             |
| PDF Loader        | PyPDFLoader                    |
| Text Splitter     | RecursiveCharacterTextSplitter |
| Text Cleanup      | ftfy                           |
| Embeddings        | Sentence Transformers          |
| Vector Database   | Pinecone                       |
| Lexical Retrieval | BM25                           |
| Hybrid Retrieval  | Reciprocal Rank Fusion         |
| LLM               | Groq                           |
| Orchestration     | LangGraph                      |
| API               | FastAPI                        |
| UI                | Streamlit                      |

The reference implementation uses OpenAI embeddings. This project instead uses a local Sentence Transformer model while retaining Pinecone for vector storage; no specific embedding provider is required.

## Project Structure

```text
rag-agentic-ai/
│
├── data/
│   └── Ebook-Agentic-AI.pdf
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── graph.py
│   └── config.py
│
├── app.py
├── streamlit_app.py
├── tests_sample_queries.py
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

## How It Works

### 1. Ingestion

The source PDF is loaded and cleaned using `PyPDFLoader` and `ftfy`.

The document is divided into manageable overlapping chunks and enriched with section, page, and chunk metadata.

```text
PDF
 ↓
PyPDFLoader
 ↓
Text cleanup
 ↓
Section detection
 ↓
Chunking
 ↓
Embeddings
 ↓
Pinecone
```

The recommended chunk size is 500–1,000 characters with overlap, and source text and page metadata are stored in Pinecone.

### 2. Retrieval

For every user query:

```text
Question
   │
   ├──► Pinecone semantic search
   │
   └──► BM25 lexical search
             │
             ▼
        RRF Fusion
             │
             ▼
       Top relevant chunks
```

Semantic retrieval handles meaning-based matching, while BM25 improves exact-term retrieval. RRF combines the ranked results into a single retrieval order.

### 3. LangGraph

The workflow contains:

```text
START
  ↓
retrieve
  ↓
generate
  ↓
grounding_check
  ↓
END
```

The LangGraph state contains fields such as `question`, `context`, `answer`, and `score`, with retrieval and generation represented as graph nodes.

### 4. Grounded Generation

The LLM is instructed to:

* use only retrieved document context
* avoid outside knowledge
* avoid invented facts
* correct unsupported premises using the document
* preserve terminology and complete lists
* refuse when the context does not contain enough information

For unsupported questions, the chatbot returns:

```text
I cannot answer based on the provided document.
```

## Installation

### Prerequisites

* Python 3.11+
* uv
* Pinecone account and API key
* Groq API key

### Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd rag-agentic-ai
```

### Install Dependencies

```bash
uv sync
```

## Environment Variables

Create `.env` in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b

PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=10

PINECONE_NAMESPACE=
MIN_SEMANTIC_SCORE=0.30
```

Never commit `.env` or API credentials to the repository.

## Document Ingestion

Place the source document at:

```text
data/Ebook-Agentic-AI.pdf
```

Run:

```bash
uv run python -m src.ingestion
```

The ingestion process:

1. Loads the PDF.
2. Cleans extracted text.
3. Detects document sections.
4. Creates overlapping chunks.
5. Generates embeddings.
6. Clears previous vectors.
7. Uploads the new vectors to Pinecone.
8. Stores page, section, source, and chunk metadata.

## Run the FastAPI Application

```bash
uv run uvicorn app:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

## API

### Endpoint

```http
POST /chat
```

### Request

```json
{
  "query": "What is Agentic AI according to the eBook?"
}
```

### Response

```json
{
  "query": "What is Agentic AI according to the eBook?",
  "final_answer": "Agentic AI refers to systems capable of autonomous decision-making and action in pursuit of specific objectives.",
  "retrieved_context_chunks": [
    "Relevant chunk from the eBook...",
    "Another relevant chunk..."
  ],
  "confidence_score": 0.82
}
```

The response exposes the query, final answer, retrieved context, and confidence score.

### PowerShell Example

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/chat" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"query":"What is Agentic AI according to the eBook?"}'
```

## Streamlit UI

Run:

```bash
uv run streamlit run streamlit_app.py
```

The UI provides:

* Question input
* Generated answer
* Retrieval confidence
* Retrieved context
* Source page information

The Streamlit interface displays the answer, retrieved context, and score.

## Validation & Testing

The benchmark includes six queries covering definition, architecture, use cases, comparison, challenges, and an out-of-scope question.

Run:

```bash
uv run python tests_sample_queries.py
```

### Benchmark Queries

```text
1. What is Agentic AI according to the eBook?

2. What are the main architectural components required
   to build agentic systems?

3. What real-world industry use cases for Agentic AI
   are discussed in the document?

4. How does Agentic AI differ from traditional generative
   AI chatbots according to the text?

5. What key challenges or limitations of Agentic AI are
   mentioned in the document?

6. What is the capital of France?
```

For the out-of-scope question, the expected behavior is to refuse or state that the information is not available in the eBook context.

## Additional Validation

The implementation also validates false-premise and retrieval behavior with queries such as:

```text
What are the six core pillars of Agentic AI?

According to the document, what are the seven core pillars
of Agentic AI?

According to the document, what are the ten core pillars
of Agentic AI?

How do memory, learning, and autonomy contribute to
continuous improvement in Agentic AI?

How do agents collaborate during a supply-chain crisis?
```

These tests are used to verify that the model follows retrieved document context instead of blindly accepting unsupported premises.

## Confidence Score

`confidence_score` represents **retrieval confidence** based on the similarity of the retrieved evidence.

It indicates how strongly the retrieved context matches the query. It is not a guarantee of answer correctness.

The implementation also calculates an internal answer-support value by comparing answer terminology with the retrieved context.

## Key Design Decisions

### Section-Aware Chunking

Document headings are preserved with chunks so retrieval retains useful section context.

### Pinecone

Pinecone stores the generated vector representations and associated metadata for semantic retrieval.

### Hybrid Retrieval

Combining semantic retrieval with BM25 improves retrieval for both conceptual queries and exact document terminology.

### LangGraph

LangGraph separates retrieval, generation, and grounding into explicit workflow nodes, matching the stateful graph-based design expected by the assignment. 

### Strict Grounding

The generation step is constrained to the retrieved document context to reduce unsupported answers.

## Requirement Mapping

| Assignment Requirement       | Implementation                           |
| ---------------------------- | ---------------------------------------- |
| Custom Python implementation | Python                                   |
| Agentic AI eBook             | `data/Ebook-Agentic-AI.pdf`              |
| PDF parsing                  | PyPDFLoader                              |
| Chunking                     | RecursiveCharacterTextSplitter           |
| Embeddings                   | Sentence Transformers                    |
| Vector DB                    | Pinecone                                 |
| Metadata                     | Text, source, page, section, chunk index |
| LangGraph                    | Retrieve → Generate → Grounding Check    |
| Top-k retrieval              | Pinecone + BM25                          |
| Grounded generation          | Context-only LLM prompt                  |
| Confidence / grounding       | Retrieval confidence + answer support    |
| API                          | FastAPI `/chat`                          |
| UI                           | Streamlit                                |
| Validation                   | Benchmark and grounding queries          |

## Submission Checklist

```text
[ ] Public GitHub repository
[ ] Agentic AI eBook included
[ ] Functional ingestion pipeline
[ ] Pinecone vectors successfully created
[ ] Functional LangGraph RAG workflow
[ ] FastAPI or Streamlit interface
[ ] Structured response with:
    [ ] final_answer
    [ ] retrieved_context_chunks
    [ ] confidence_score
[ ] 5–6 validation queries tested
[ ] Out-of-scope query tested
[ ] README includes setup instructions
[ ] README includes architecture
[ ] No API keys committed
```

The submission checklist covers a public repository, README setup and architecture, functional ingestion and LangGraph pipeline, API/UI, structured response fields, and 5–6 grounding tests.

## Quick Start

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd rag-agentic-ai

uv sync

# Configure .env

uv run python -m src.ingestion

# Terminal 1
uv run uvicorn app:app --reload

# Terminal 2
uv run streamlit run streamlit_app.py
```

## License

This project is developed as part of an AI Engineer interview assignment.
