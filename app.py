from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import rag_graph

app = FastAPI(
    title="Agentic AI RAG Assistant",
    description="LangGraph and Pinecone RAG chatbot for Agentic AI",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {
        "status": "ok",
         "message": "Agentic AI RAG Assistant is running",
        }


@app.post("/chat")
def chat(request: ChatRequest):
    result = rag_graph.invoke(
        {
            "question": request.query,
            "context": [],
            "retrieved_context_chunks": [],
            "answer": "",
            "score": 0.0,
        }
    )

    return {
        "query": request.query,
        "final_answer": result["answer"],
        "retrieved_context_chunks": result["retrieved_context_chunks"],
        "confidence_score": result["score"],
    }