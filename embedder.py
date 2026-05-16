"""
embedder.py
Build embeddings and FAISS index from catalog.json using bge-base-en-v1.5.
Run this file directly to generate data/faiss_index.bin and data/assessments_metadata.pkl.
"""

import json
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Any

class EmbeddingBuilder:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        print(f"🔧 Loading embedding model: {model_name} (768 dim)")
        self.model = SentenceTransformer(model_name)
        self.dim = 768   # bge-base-en-v1.5

    def _make_text(self, item: Dict[str, Any]) -> str:
        """
        Create a single text string from assessment fields.
        Combines name, tags, and test type for rich semantic representation.
        """
        name = item.get("name", "")
        tags = " ".join(item.get("tags", []))
        test_type = item.get("test_type", "")
        remote = "remote" if item.get("remote_testing") == "Yes" else ""
        adaptive = "adaptive" if item.get("adaptive_irt") == "Yes" else ""
        parts = [name, tags, test_type, remote, adaptive]
        return " ".join(p for p in parts if p).strip()

    def build_embeddings(self, assessments: List[Dict[str, Any]]) -> np.ndarray:
        """Return float32 numpy array of embeddings (N x dim)."""
        texts = [self._make_text(a) for a in assessments]
        print(f"📝 Generating embeddings for {len(texts)} items...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.astype(np.float32)

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """
        Build FAISS index using inner product (cosine similarity after L2 normalisation).
        IndexFlatIP is exact and fast for small to medium datasets.
        """
        # Normalise vectors for cosine similarity (inner product = cosine)
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(self.dim)
        index.add(embeddings)
        print(f"✅ FAISS index built with {index.ntotal} vectors, dimension {self.dim}")
        return index

    def save(self, index: faiss.IndexFlatIP, assessments: List[Dict[str, Any]],
             index_path: str = "data/faiss_index.bin",
             metadata_path: str = "data/assessments_metadata.pkl"):
        """Save FAISS index and metadata to disk."""
        faiss.write_index(index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(assessments, f)
        print(f"💾 Saved FAISS index to {index_path}")
        print(f"💾 Saved metadata to {metadata_path}")

# ----------------------------------------------------------------------
# Main: load catalog, build embeddings + index, save artefacts
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting Phase 1 (embedding + FAISS index building) with bge-base-en-v1.5...")
    # Load validated catalog
    catalog_path = "data/catalog.json"
    if not os.path.exists(catalog_path):
        print(f"❌ {catalog_path} not found. Please run scraper.py first.")
        exit(1)

    with open(catalog_path, "r", encoding="utf-8") as f:
        assessments = json.load(f)

    print(f"📚 Loaded {len(assessments)} assessments from catalog.json")

    builder = EmbeddingBuilder()
    embeddings = builder.build_embeddings(assessments)
    print(f"📊 Embeddings shape: {embeddings.shape}")

    index = builder.build_faiss_index(embeddings)
    builder.save(index, assessments, "data/faiss_index.bin", "data/assessments_metadata.pkl")

    print("✅ Phase 1 (embedding + FAISS) complete.")