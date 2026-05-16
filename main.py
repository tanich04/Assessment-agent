"""
main.py — FastAPI service for the SHL Conversational Recommender.

Endpoints:
    GET  /health   → {"status": "ok"}
    POST /chat     → AgentResponse

Startup:
    Loads FAISS index + metadata + BM25 index once.
    All per-request state is rebuilt from the messages[] payload (stateless).

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Deploy (Render / Fly.io):
    Set GROQ_API_KEY (or GEMINI_API_KEY) as an environment variable.
    Dockerfile should COPY data/ (faiss_index.bin + assessments_metadata.pkl) into image.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from agent import AgentResponse, ChatRequest, run_agent
from retriever import SHLRetriever

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Shared retriever instance — loaded once at startup, reused across requests.
# SHLRetriever is thread-safe for reads (FAISS search, BM25 retrieve).
# ---------------------------------------------------------------------------
_retriever: SHLRetriever | None = None

TIMEOUT_SECONDS = 28   # hard stop before the evaluator's 30 s wall clock


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: load retriever at startup, release at shutdown."""
    global _retriever
    logger.info("Loading SHLRetriever (FAISS + BM25)…")
    _retriever = SHLRetriever(
        top_k_semantic=50,
        top_k_final=10,
        use_cross_encoder=False,   # keep p95 latency < 5 s; enable for offline eval
    )
    logger.info("SHLRetriever ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent that recommends SHL Individual Test Solutions.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://yourdomain.com"],  # adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Middleware: request timing log
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - t0
    logger.info("%s %s → %d (%.2fs)", request.method, request.url.path, response.status_code, elapsed)
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
def health():
    """Readiness probe. Returns 200 as soon as the service is up."""
    return {"status": "ok"}


@app.post("/chat", response_model=AgentResponse, tags=["chat"])
def chat(body: ChatRequest):
    """
    Stateless chat endpoint. Accepts full conversation history; returns next agent turn.

    Request body:
        {
            "messages": [
                {"role": "user",      "content": "…"},
                {"role": "assistant", "content": "…"},
                {"role": "user",      "content": "…"}
            ]
        }

    Response:
        {
            "reply": "…",
            "recommendations": [{"name": "…", "url": "…", "test_type": "…"}],
            "end_of_conversation": false
        }
    """
    t0 = time.perf_counter()

    try:
        response = run_agent(messages=body.messages, retriever=_retriever)
    except Exception as exc:
        logger.exception("Unhandled error in run_agent: %s", exc)
        raise HTTPException(status_code=500, detail="Internal agent error. Please retry.")

    elapsed = time.perf_counter() - t0
    if elapsed > TIMEOUT_SECONDS:
        # Response is already computed; just log the breach so we can tune later
        logger.warning("Response took %.2fs — over the %.0fs soft limit.", elapsed, TIMEOUT_SECONDS)

    return response


# ---------------------------------------------------------------------------
# Global exception handler — return JSON, not HTML, for all unhandled errors
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "reply": "An unexpected error occurred. Please try again.",
            "recommendations": [],
            "end_of_conversation": False,
        },
    )