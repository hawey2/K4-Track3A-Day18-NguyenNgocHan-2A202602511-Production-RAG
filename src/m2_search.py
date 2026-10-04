from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys, re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    # ⚠️ LƯU Ý: underthesea nối từ ghép bằng "_" (VD: "nghỉ_phép").
    # BM25 tokenize bằng split(" ") -> "nghỉ_phép" thành 1 token,
    # nhưng query "nghỉ phép" thành 2 token -> KHÔNG khớp.
    # Phải replace("_", " ") để BM25 hoạt động đúng.
    try:
        from underthesea import word_tokenize

        segmented = word_tokenize(text, format="text")
        # Hạ chữ thường + bỏ dấu câu để BM25 robust với viết hoa/dấu.
        segmented = segmented.replace("_", " ").lower()
        return re.sub(r"[^\w\s]", " ", segmented)
    except Exception:
        # Fallback khi underthesea lỗi (model chưa tải, mất mạng...).
        return re.sub(r"[^\w\s]", " ", text.lower())


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = chunks
        self.corpus_tokens = [
            segment_vietnamese(c["text"]).split() for c in chunks
        ]
        # Chunk không còn token nào (VD: PDF scan rỗng) sẽ làm BM25Okapi
        # crash vì vocabulary rỗng -> thay bằng placeholder để giữ chỉ số.
        self.corpus_tokens = [toks if toks else ["\x00empty"] for toks in self.corpus_tokens]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents:
            return []

        tokenized_query = segment_vietnamese(query).split()
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        # Lọc score == 0 để bỏ docs không liên quan — giữ RRF sạch,
        # tránh dense list bị BM25 tuỳ nhiên bù lấp điểm hạng.
        ranked = sorted(
            ((float(s), i) for i, s in enumerate(scores) if s > 0),
            key=lambda x: x[0], reverse=True,
        )[:top_k]

        return [
            SearchResult(
                text=self.documents[i]["text"],
                score=score,
                metadata=self.documents[i].get("metadata", {}),
                method="bm25",
            )
            for score, i in ranked
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(":memory:")
        self._encoder = None
        # dim thực tế của từng collection, suy ra lúc index.
        self._collection_dims: dict[str, int] = {}

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct

        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=False)

        # Suy ra dim tu model thay vi tin config: tranh hardcode sai
        # (bge-m3=1024, MiniLM=384) lam Qdrant reject moi upsert.
        dim = len(vectors[0])

        self.client.recreate_collection(
            collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        # id phai la uint64/UUID cho Qdrant -> dung index +1 (0 khong dung).
        points = [
            PointStruct(
                id=i + 1,
                vector=v.tolist(),
                payload={**c.get("metadata", {}), "text": c["text"]},
            )
            for i, (c, v) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection, points)
        self._collection_dims[collection] = dim

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        # qdrant-client >= 2.0 dung query_points(), KHONG phai search().
        try:
            self.client.get_collection(collection)
        except Exception:
            # Chua index collection nay -> dense khong co gi de tra ve.
            return []

        query_vector = self._get_encoder().encode(query, show_progress_bar=False).tolist()
        response = self.client.query_points(collection, query=query_vector, limit=top_k)

        results = []
        for pt in response.points:
            payload = pt.payload or {}
            metadata = {k: v for k, v in payload.items() if k != "text"}
            results.append(SearchResult(
                text=payload.get("text", ""),
                score=float(pt.score),
                metadata=metadata,
                method="dense",
            ))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = SUM 1/(k + rank)."""
    rrf_scores: dict[str, dict] = {}

    for result_list in results_list:
        for rank, result in enumerate(result_list):
            entry = rrf_scores.setdefault(
                result.text, {"score": 0.0, "result": result, "methods": []}
            )
            entry["score"] += 1.0 / (k + rank + 1)
            # Ghi lai method de biet chunk nao hit ca BM25 lan dense
            # (agreement = tin hieu do tin cay cao).
            if result.method not in entry["methods"]:
                entry["methods"].append(result.method)

    ranked = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)

    fused = []
    for entry in ranked[:top_k]:
        original = entry["result"]
        fused.append(SearchResult(
            text=original.text,
            score=entry["score"],
            metadata={**original.metadata, "rrf_methods": entry["methods"]},
            method="hybrid",
        ))
    return fused


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
