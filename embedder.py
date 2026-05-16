"""
embedder.py
Build embeddings and FAISS index from catalog.json using bge-small-en-v1.5.
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
        self.dim = 768   # same dimension as bge-small

    def _make_text(self, item: Dict[str, Any]) -> str:
        name = item.get("name", "")
        tags = " ".join(item.get("tags", []))
        test_type = item.get("test_type", "")
        remote = "remote" if item.get("remote_testing") == "Yes" else ""
        adaptive = "adaptive" if item.get("adaptive_irt") == "Yes" else ""
        parts = [name, tags, test_type, remote, adaptive]
        return " ".join(p for p in parts if p).strip()

    def build_embeddings(self, assessments: List[Dict[str, Any]]) -> np.ndarray:
        texts = [self._make_text(a) for a in assessments]
        print(f"📝 Generating embeddings for {len(texts)} items...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.astype(np.float32)

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(self.dim)
        index.add(embeddings)
        print(f"✅ FAISS index built with {index.ntotal} vectors, dimension {self.dim}")
        return index

    def save(self, index: faiss.IndexFlatIP, assessments: List[Dict[str, Any]],
             index_path: str = "data/faiss_index.bin",
             metadata_path: str = "data/assessments_metadata.pkl"):
        faiss.write_index(index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(assessments, f)
        print(f"💾 Saved FAISS index to {index_path}")
        print(f"💾 Saved metadata to {metadata_path}")

if __name__ == "__main__":
    print("🚀 Starting Phase 1 (embedding + FAISS) with bge-small-en-v1.5...")
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

    print("✅ Phase 1 complete.")