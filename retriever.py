"""
retriever.py — Hybrid retrieval for the SHL Conversational Recommender.

WHAT THIS FILE DOES
====================
Loads three artefacts at module import time (once per process):
  - FAISS flat inner-product index (bge-base-en-v1.5 embeddings, 768-dim)
  - Assessment metadata list (catalog.json items as Python dicts)
  - BM25 index (bm25s, built on name + tags + test_type text)

Exposes SHLRetriever with two public methods:
  - retrieve(query, test_type, tags)        → standard single-pass retrieval
  - retrieve_diverse(base_query, types, …)  → multi-bucket diversified retrieval

DESIGN RATIONALE
================

Why FAISS + BM25 + RRF instead of just one?
  - FAISS (semantic): captures meaning ("Java developer" matches "programming")
  - BM25 (lexical): exact name matches ("OPQ32r" ranks high when user says "OPQ")
  - RRF (fusion): combines both rank lists without needing calibrated scores.
    RRF is parameter-robust: k=60 works well across domains without tuning.

Why bge-base-en-v1.5?
  - 768-dim, strong performance on short technical queries, runs on CPU in <100ms.
  - The embedder.py already built the index with this model, so it must match here.

Why IndexFlatIP (exact search) over HNSW (approximate)?
  - Catalog size is ~300 items. Exact search over 300 vectors takes ~1ms on CPU.
  - HNSW would save nothing here and adds recall risk.

Why retrieve_diverse?
  - If the user says "I need personality AND cognitive tests", a single retrieval
    query biased toward the role may return 10 cognitive tests and 0 personality.
  - retrieve_diverse runs one retrieval pass per requested test type with a
    type-specific query expansion, then interleaves results round-robin.
  - This guarantees at least one item per requested type even when embedding
    similarity strongly favors one type.

Why NOT use query expansion strings hardcoded in the retriever?
  - The agent already generates a rich retrieval_query via the LLM.
  - The only expansion retrieve_diverse adds is the type-specific semantic hint
    (e.g. "personality behavioral" for type P). This is minimal and correct
    because the FAISS space was built with these terms as part of item texts.
"""

from __future__ import annotations

import os
import pickle
from collections import defaultdict
from typing import Any

import bm25s
import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# Artifact paths
# ─────────────────────────────────────────────────────────────────────────────
INDEX_PATH    = "data/faiss_index.bin"
METADATA_PATH = "data/assessments_metadata.pkl"


def _load_artifacts() -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Load FAISS index and metadata from disk.

    Called once at module import. Raises FileNotFoundError with a clear
    message if the Phase 1 pipeline hasn't been run yet, so the developer
    gets an actionable error rather than a cryptic FAISS segfault.
    """
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Missing retrieval artefacts at '{INDEX_PATH}' / '{METADATA_PATH}'.\n"
            "Run scraper.py then embedder.py to generate them before starting the server."
        )
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata: list[dict] = pickle.load(f)
    return index, metadata


# Module-level singletons loaded once at import time.
# FastAPI's lifespan does NOT need to reload these — they live in module memory.
print("Loading FAISS index and metadata…")
FAISS_INDEX, ASSESSMENTS = _load_artifacts()
print(f"  ✅ {len(ASSESSMENTS)} assessments loaded.")


# ─────────────────────────────────────────────────────────────────────────────
# BM25 index — built from in-memory data (no disk file needed)
# ─────────────────────────────────────────────────────────────────────────────

def _build_bm25(assessments: list[dict]) -> bm25s.BM25:
    """
    Build a BM25 index over the corpus.

    Corpus text per item = name + tags + test_type.
    This is exactly what the FAISS embedder used, so both indices cover the
    same textual surface — important for RRF score alignment.

    Why include test_type in the corpus?
    The code letters (A, P, K, B, C, S) appear in user queries ("I need a K
    test", "what personality assessments do you have") and BM25 will match them
    exactly, boosting the right items.
    """
    corpus = [
        f"{a['name']} {' '.join(a.get('tags', []))} {a.get('test_type', '')}"
        for a in assessments
    ]
    tokens = bm25s.tokenize(corpus, stopwords="en")
    index = bm25s.BM25()
    index.index(tokens)
    return index


print("Building BM25 index…")
BM25_INDEX = _build_bm25(ASSESSMENTS)
print("  ✅ BM25 ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Embedding model — loaded once, reused for all query encodings
# ─────────────────────────────────────────────────────────────────────────────
print("Loading embedding model (bge-base-en-v1.5)…")
_EMBED_MODEL = SentenceTransformer("BAAI/bge-base-en-v1.5")
print("  ✅ Embedding model ready.")


# ─────────────────────────────────────────────────────────────────────────────
# RRF (Reciprocal Rank Fusion)
# ─────────────────────────────────────────────────────────────────────────────

def _rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """
    Fuse multiple rank lists using Reciprocal Rank Fusion.

    Formula: score(d) = Σ 1 / (k + rank(d))
    where rank is 1-indexed and k=60 is the standard damping constant.

    k=60 means a document at rank 1 scores 1/61 ≈ 0.016, while one at rank 60
    scores 1/120 ≈ 0.008. The relative weighting is stable and doesn't require
    tuning unlike linear combination of raw scores.

    Args:
        rankings: Each inner list is a sequence of document indices in
                  descending relevance order (best first).
        k: Damping constant. Default 60 is from the original RRF paper.

    Returns:
        Dict mapping document index → fused score (higher = more relevant).
    """
    scores: dict[int, float] = defaultdict(float)
    for rank_list in rankings:
        for rank, doc_idx in enumerate(rank_list, start=1):
            scores[doc_idx] += 1.0 / (k + rank)
    return dict(scores)


# ─────────────────────────────────────────────────────────────────────────────
# Type-specific query hints for diverse retrieval
# ─────────────────────────────────────────────────────────────────────────────

# These strings are appended to the base query when retrieving items of a
# specific type. They are short and semantically accurate — they don't add
# noise, they steer the embedding toward the right cluster in FAISS space.
# For example, adding "personality behavioral opq" to a data scientist query
# ensures the semantic search returns OPQ32r instead of more data science K-type tests.
_TYPE_HINTS: dict[str, str] = {
    "A": "cognitive ability aptitude reasoning numerical verbal inductive deductive",
    "P": "personality behavioral opq motivation values interpersonal",
    "K": "knowledge skill technical proficiency test",
    "B": "behavioral situational judgment workplace",
    "C": "competency framework structured interview",
    "S": "situational judgment scenarios",
}


# ─────────────────────────────────────────────────────────────────────────────
# SHLRetriever
# ─────────────────────────────────────────────────────────────────────────────

class SHLRetriever:
    """
    Hybrid retriever combining FAISS semantic search, BM25 lexical search,
    and RRF fusion. Supports standard single-pass and diverse multi-bucket retrieval.

    Thread-safety:
        All state is read-only after __init__. Module-level FAISS_INDEX, BM25_INDEX,
        and _EMBED_MODEL are also read-only at inference. The instance is safe to
        share across FastAPI request handlers without locking.

    Args:
        top_k_semantic: Candidates fetched per retrieval pass from each index
                        before fusion. Larger = higher recall, slower search.
                        50 is a good default for a ~300-item catalog.
        top_k_final:    Items returned after fusion and filtering.
                        Set to 20 so the LLM ranker sees enough candidates.
        use_cross_encoder: If True, re-rank final candidates with a cross-encoder.
                           Improves precision by ~10% but adds ~500ms per call.
                           Disable for production unless latency allows it.
    """

    def __init__(
        self,
        top_k_semantic: int = 50,
        top_k_final:    int = 20,
        use_cross_encoder: bool = False,
    ):
        self.top_k_semantic   = top_k_semantic
        self.top_k_final      = top_k_final
        self.use_cross_encoder = use_cross_encoder
        self._cross_encoder: CrossEncoder | None = None

    # ── Private helpers ──────────────────────────────────────────────────────

    def _get_cross_encoder(self) -> CrossEncoder:
        """
        Lazy-load the cross-encoder on first use.
        Cross-encoders are large (~85MB), so we don't load them unless needed.
        """
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._cross_encoder

    def _semantic_search(self, query: str) -> list[int]:
        """
        Encode query with bge-base and search FAISS index.

        normalize_embeddings=True ensures vectors are L2-normalized before
        passing to IndexFlatIP, which then computes cosine similarity
        (inner product of normalized vectors = cosine similarity).

        Returns:
            List of ASSESSMENTS indices in descending similarity order.
        """
        emb = _EMBED_MODEL.encode([query], normalize_embeddings=True).astype(np.float32)
        _, indices = FAISS_INDEX.search(emb, self.top_k_semantic)
        return indices[0].tolist()

    def _bm25_search(self, query: str) -> list[int]:
        """
        Tokenize query and search BM25 index.

        Returns:
            List of ASSESSMENTS indices in descending BM25 score order.
        """
        tokens = bm25s.tokenize([query], stopwords="en")
        indices, _ = BM25_INDEX.retrieve(tokens, k=self.top_k_semantic)
        return indices[0].tolist()

    def _cross_rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """
        Re-rank candidates using a cross-encoder.

        The cross-encoder takes (query, document_name) pairs and scores them.
        This is more accurate than bi-encoder similarity for short queries
        because the cross-encoder sees both query and document jointly.

        Only called when use_cross_encoder=True and there are candidates.
        """
        if not candidates or not self.use_cross_encoder:
            return candidates
        cross = self._get_cross_encoder()
        pairs  = [(query, c["name"]) for c in candidates]
        scores = cross.predict(pairs)
        return [c for c, _ in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)]

    def _fuse_and_filter(
        self,
        query:     str,
        test_type: str | None,
        tags:      list[str] | None,
        top_k:     int,
    ) -> list[dict]:
        """
        Core retrieval pipeline: semantic + BM25 → RRF → metadata filter → top-k.

        This is the shared inner method used by both retrieve() and each bucket
        in retrieve_diverse(). Keeping it in one place ensures both paths use
        identical fusion and filtering logic.

        Args:
            query:     Search query string (may include type hints for diverse buckets).
            test_type: If set, only items whose test_type field contains this code
                       are returned. E.g. "P" filters to personality tests.
                       Match is case-insensitive substring so "P" matches "A, P, K".
            tags:      If set, items must contain at least one of these tags.
                       Used for skill-based filtering (e.g. ["Java"]).
            top_k:     Maximum items to return after filtering.

        Returns:
            List of assessment dicts (copies with 'relevance_score' added),
            sorted by RRF score descending.
        """
        # Step 1: Get ranked lists from both indices
        sem_ranks = self._semantic_search(query)
        bm25_ranks = self._bm25_search(query)

        # Step 2: Fuse with RRF
        fused = _rrf([sem_ranks, bm25_ranks])

        # Step 3: Sort all candidates by fused score
        sorted_candidates = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        # Step 4: Apply metadata filters
        filtered: list[tuple[int, float]] = []
        for idx, score in sorted_candidates:
            item = ASSESSMENTS[idx]

            # Test type filter: case-insensitive check in the comma-separated type string
            # Example: test_type="A, P, K" contains "P" → passes the P filter
            if test_type:
                item_types = item.get("test_type", "").upper()
                # Split on comma and check each code to avoid "A" matching "A" in "N/A"
                item_type_codes = {t.strip() for t in item_types.split(",")}
                if test_type.upper() not in item_type_codes:
                    continue

            # Tag filter: at least one requested tag must appear in item's tag list
            if tags:
                item_tags_lower = {t.lower() for t in item.get("tags", [])}
                if not any(req.lower() in item_tags_lower for req in tags):
                    continue

            filtered.append((idx, score))

        # Step 5: Take top_k
        results: list[dict] = []
        for idx, score in filtered[:top_k]:
            item = ASSESSMENTS[idx].copy()
            item["relevance_score"] = score
            results.append(item)

        return results

    # ── Public API ───────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:     str,
        test_type: str | None = None,
        tags:      list[str] | None = None,
    ) -> list[dict]:
        """
        Standard single-pass retrieval.

        Use this when:
        - No test type preference (or only one type requested)
        - Called from comparison handler (targeted single-item search)
        - Called from refinement handler with a single type

        The query should be the LLM-generated retrieval_query from ConversationState,
        which is a full sentence describing the role and requirements.

        Returns up to self.top_k_final items.
        """
        results = self._fuse_and_filter(query, test_type, tags, self.top_k_final)

        if self.use_cross_encoder and results:
            results = self._cross_rerank(query, results)

        return results

    def retrieve_diverse(
        self,
        base_query:     str,
        requested_types: list[str],
        skills:         list[str] | None = None,
        top_k_per_type: int = 20,
        top_k_final:    int = 20,
    ) -> list[dict]:
        """
        Multi-bucket diversified retrieval for multi-type requests.

        WHY THIS EXISTS
        ---------------
        If the user asks for "personality AND cognitive assessments for a data
        scientist", a single query biased toward "data scientist" will return
        10 knowledge/cognitive tests and 0 personality tests because the data
        science assessments dominate FAISS similarity scores.

        retrieve_diverse solves this by:
        1. Running one retrieval pass per requested type.
        2. For each type, augmenting the query with a type-specific semantic hint
           (see _TYPE_HINTS) so the embedding steers toward that type's cluster.
        3. Applying a hard test_type filter to each bucket so only matching items
           are included.
        4. Interleaving results round-robin across type buckets to guarantee
           balanced representation in the final list.

        This is how the "include personality test" refinement in the demo should
        have worked: personality bucket retrieves OPQ32r and similar P-type items
        directly, independent of the data science query bias.

        WHY top_k_per_type=20 (not 15)?
        The SHL catalog has ~300 items. For niche types like "C" (competency),
        after the hard type filter there may only be 5-8 items total. Using 20
        as the per-type cap ensures we get all available items for rare types
        rather than silently truncating at 15.

        Args:
            base_query:      LLM-generated retrieval query (role + skills + context).
            requested_types: List of type codes, e.g. ["P", "K", "A"].
            skills:          Optional skill strings appended to each bucket query
                             for additional semantic signal.
            top_k_per_type:  Max items to collect per type bucket before interleaving.
            top_k_final:     Max items in the final merged list.

        Returns:
            List of assessment dicts, interleaved across type buckets.
        """
        if not requested_types:
            # Fallback to standard retrieval if no types specified
            return self.retrieve(query=base_query, test_type=None, tags=None)

        # Collect per-bucket results
        seen_urls: set[str] = set()
        buckets: dict[str, list[dict]] = {t: [] for t in requested_types}

        for type_code in requested_types:
            # Build the bucket query: base + skills + type-specific hint
            # The type hint steers semantic search toward the right cluster.
            # Skills are included so we don't lose role-relevance signal.
            parts = [base_query]
            if skills:
                parts.append(" ".join(skills))
            if type_code in _TYPE_HINTS:
                parts.append(_TYPE_HINTS[type_code])
            bucket_query = " ".join(parts)

            # Retrieve with hard type filter to guarantee type coverage
            bucket_results = self._fuse_and_filter(
                query=bucket_query,
                test_type=type_code,
                tags=None,              # no tag filter per bucket — type filter is enough
                top_k=top_k_per_type,
            )

            for item in bucket_results:
                url = str(item.get("url", ""))
                if url not in seen_urls:
                    seen_urls.add(url)
                    buckets[type_code].append(item)

        # Interleave round-robin across buckets
        # This ensures the final list has balanced type representation.
        # Example with types [P, K, A]: P[0], K[0], A[0], P[1], K[1], A[1], ...
        interleaved: list[dict] = []
        max_len = max((len(buckets[t]) for t in requested_types), default=0)
        for i in range(max_len):
            for t in requested_types:
                if i < len(buckets[t]):
                    interleaved.append(buckets[t][i])
            if len(interleaved) >= top_k_final:
                break

        # Optional cross-encoder rerank on the merged set
        if self.use_cross_encoder and interleaved:
            skill_str = " ".join(skills) if skills else ""
            full_query = f"{base_query} {skill_str}".strip()
            interleaved = self._cross_rerank(full_query, interleaved)

        return interleaved[:top_k_final]

    def context_assembler(self, results: list[dict]) -> str:
        """
        Format retrieval results into a human-readable catalog snippet
        for inclusion in LLM prompts.

        The format matches what the agent's handle_retrieve() expects.
        Include all fields that help the LLM rank: name, type, tags, remote, adaptive.
        """
        if not results:
            return "No relevant assessments found in the SHL catalog."
        lines = ["Catalog assessments (recommend ONLY from this list):"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. Name: {r['name']}\n"
                f"   Type codes: {r['test_type']} | Tags: {', '.join(r.get('tags', []))}\n"
                f"   Remote: {r.get('remote_testing','N/A')} | Adaptive: {r.get('adaptive_irt','N/A')}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test when run directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n── Standard retrieval ──")
    r = SHLRetriever(top_k_final=5)
    results = r.retrieve("senior Java developer who communicates with business stakeholders")
    for item in results:
        print(f"  {item['name']} | type={item['test_type']} | score={item['relevance_score']:.4f}")

    print("\n── Diverse retrieval (K + P) ──")
    results = r.retrieve_diverse(
        base_query="data scientist role requiring analysis and collaboration",
        requested_types=["K", "P", "A"],
        skills=["Python", "statistics"],
        top_k_per_type=5,
        top_k_final=10,
    )
    for item in results:
        print(f"  {item['name']} | type={item['test_type']}")