"""
retriever.py – Hybrid retrieval with RRF, bm25s, and external embedding API.
No local embedding model – uses Hugging Face Inference API.
Memory usage: FAISS (memory‑mapped) + BM25 (~50 MB total).
"""

import os
import gc
import pickle
from collections import defaultdict
from typing import List, Dict, Any, Optional

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import faiss
import bm25s
import requests

# ----------------------------------------------------------------------
# Load artefacts once at startup (module level)
# ----------------------------------------------------------------------
INDEX_PATH = "data/faiss_index.bin"
METADATA_PATH = "data/assessments_metadata.pkl"

def load_index_and_metadata():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            "Missing retrieval artefacts at 'data/faiss_index.bin' / 'data/assessments_metadata.pkl'.\n"
            "Run scraper.py then embedder.py to generate them before starting the server."
        )
    # Memory‑mapped FAISS index – stays on disk
    index = faiss.read_index(INDEX_PATH, faiss.IO_FLAG_MMAP)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

FAISS_INDEX, ASSESSMENTS = load_index_and_metadata()

# Build BM25 corpus and index
BM25_CORPUS = [
    f"{a['name']} {' '.join(a['tags'])} {a['test_type']}"
    for a in ASSESSMENTS
]
print("Building BM25 index...")
corpus_tokens = bm25s.tokenize(BM25_CORPUS, stopwords="en")
BM25_INDEX = bm25s.BM25()
BM25_INDEX.index(corpus_tokens)
print("BM25 index ready.")

# ----------------------------------------------------------------------
# External embedding via Hugging Face Inference API
# ----------------------------------------------------------------------
HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.environ.get("HF_TOKEN")

def get_embedding(text: str) -> np.ndarray:
    """Call Hugging Face API to get embedding vector."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    try:
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=10)
        if resp.status_code == 200:
            embedding = resp.json()[0]  # list of floats
            return np.array(embedding, dtype=np.float32)
        else:
            raise RuntimeError(f"HF API error: {resp.status_code} - {resp.text}")
    except Exception as e:
        raise RuntimeError(f"Embedding API call failed: {e}")

# ----------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ----------------------------------------------------------------------
def reciprocal_rank_fusion(rankings: List[List[int]], k: int = 60) -> Dict[int, float]:
    rrf_scores = defaultdict(float)
    for rank_list in rankings:
        for rank, doc_idx in enumerate(rank_list, start=1):
            rrf_scores[doc_idx] += 1.0 / (k + rank)
    return dict(rrf_scores)

# ----------------------------------------------------------------------
# Main Retriever class
# ----------------------------------------------------------------------
class SHLRetriever:
    def __init__(self,
                 top_k_semantic: int = 50,
                 top_k_final: int = 10,
                 use_cross_encoder: bool = False):
        self.top_k_semantic = top_k_semantic
        self.top_k_final = top_k_final
        self.use_cross_encoder = use_cross_encoder
        self._cross_encoder = None

    def _get_cross_encoder(self):
        if self.use_cross_encoder and self._cross_encoder is None:
            from sentence_transformers import CrossEncoder  # light, no embedding model
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._cross_encoder

    def _semantic_search(self, query: str) -> List[int]:
        """Get embedding via external API and search FAISS."""
        try:
            emb = get_embedding(query)
            # FAISS expects float32 and normalised for IP (cosine)
            emb = emb.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(emb)
            scores, indices = FAISS_INDEX.search(emb, self.top_k_semantic)
            return indices[0].tolist()
        except Exception as exc:
            print(f"Embedding API failed, falling back to BM25 only: {exc}")
            # Fallback: return BM25 results as semantic results
            return self._bm25_search(query)

    def _bm25_search(self, query: str) -> List[int]:
        query_tokens = bm25s.tokenize([query], stopwords="en")
        indices, _ = BM25_INDEX.retrieve(query_tokens, k=self.top_k_semantic)
        return indices[0].tolist()

    def _cross_rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates or not self.use_cross_encoder:
            return candidates
        cross = self._get_cross_encoder()
        pairs = [(query, c['name']) for c in candidates]
        scores = cross.predict(pairs)
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored]

    def retrieve(self,
                 query: str,
                 test_type: Optional[str] = None,
                 tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        expanded = query
        semantic_ranks = self._semantic_search(expanded)
        bm25_ranks = self._bm25_search(expanded)
        rrf_scores = reciprocal_rank_fusion([semantic_ranks, bm25_ranks])
        candidates = [(idx, rrf_scores[idx]) for idx in rrf_scores]
        candidates.sort(key=lambda x: x[1], reverse=True)

        filtered = []
        for idx, score in candidates:
            item = ASSESSMENTS[idx]
            if test_type:
                item_types = [t.strip().upper() for t in item.get("test_type", "").split(",")]
                if test_type.upper() not in item_types:
                    continue
            if tags:
                item_tags = [t.lower() for t in item.get("tags", [])]
                if not any(req.lower() in item_tags for req in tags):
                    continue
            filtered.append((idx, score))

        final_idx_scores = filtered[:self.top_k_final]
        results = []
        for idx, score in final_idx_scores:
            item = ASSESSMENTS[idx].copy()
            item['relevance_score'] = score
            results.append(item)

        if self.use_cross_encoder and results:
            results = self._cross_rerank(expanded, results)
        return results

    def retrieve_diverse(self,
                         base_query: str,
                         requested_types: List[str],
                         skills: Optional[List[str]] = None,
                         top_k_per_type: int = 15,
                         top_k_final: int = 20) -> List[Dict[str, Any]]:
        if not requested_types:
            return self.retrieve(query=base_query, test_type=None, tags=None)

        type_expansion = {
            "P": "personality behavioral opq motivation interpersonal team collaboration",
            "K": "knowledge technical skill proficiency",
            "A": "cognitive ability aptitude reasoning",
            "B": "behavioral situational",
            "C": "competency",
            "S": "situational judgment",
        }

        all_candidates = []
        seen_urls = set()

        for t in requested_types:
            bucket_parts = [base_query]
            if skills:
                bucket_parts.append(" ".join(skills))
            if t in type_expansion:
                bucket_parts.append(type_expansion[t])
            bucket_query = " ".join(bucket_parts)

            bucket_results = self.retrieve(query=bucket_query, test_type=t, tags=None)
            for item in bucket_results[:top_k_per_type]:
                url = item["url"]
                if url not in seen_urls:
                    seen_urls.add(url)
                    item["_bucket"] = t
                    all_candidates.append(item)

        buckets = {t: [c for c in all_candidates if c.get("_bucket") == t] for t in requested_types}
        interleaved = []
        max_len = max((len(buckets[t]) for t in requested_types), default=0)
        for i in range(max_len):
            for t in requested_types:
                if i < len(buckets[t]):
                    interleaved.append(buckets[t][i])

        if self.use_cross_encoder and interleaved:
            full_query = base_query + (" " + " ".join(skills) if skills else "")
            interleaved = self._cross_rerank(full_query, interleaved)

        return interleaved[:top_k_final]

    def context_assembler(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "No relevant assessments found."
        lines = ["Catalog assessments (only these are allowed):"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. {r['name']} – Type: {r['test_type']} – Tags: {', '.join(r['tags'])} – URL: {r['url']}"
            )
        return "\n".join(lines)

# ----------------------------------------------------------------------
# Quick test (won't run on server because of missing HF_TOKEN)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing SHLRetriever with external embedding API...")
    if not HF_TOKEN:
        print("Set HF_TOKEN environment variable first.")
        exit(1)
    retriever = SHLRetriever(use_cross_encoder=False)
    query = "Java developer assessment with personality test"
    print(f"Query: {query}")
    results = retriever.retrieve(query, test_type="P", tags=["technical", "personality"])
    print(f"Retrieved {len(results)} items:")
    for r in results:
        print(f" - {r['name']} (RRF score: {r['relevance_score']:.4f})")