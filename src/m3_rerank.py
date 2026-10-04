from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from flashrank import RerankRequest
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K, RERANKER_MODEL, RERANKER_BACKEND


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    """Cross-encoder reranker: doc nen 20 -> top-k.

    Backend theo thu tu uu tien trong config.RERANKER_BACKEND:
      - "flashrank"     : FlashRank ms-marco-TinyBERT-L-2-v2 (nhanh, nhe)
      - "cross-encoder": sentence_transformers.CrossEncoder voi bge-reranker-v2-m3
      - "auto"          : thu cross-encoder truoc, fallback flashrank neu load fail
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self._model = None
        self._fallback = None
        self.backend = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        if RERANKER_BACKEND == "flashrank":
            return None  # di thang flashrank trong rerank()

        # ⚠️ LUU Y: Dung sentence_transformers.CrossEncoder, KHONG dung FlagEmbedding.
        # FlagReranker crash voi transformers>=5.0 (XLMRobertaTokenizer loi).
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            self.backend = "cross-encoder"
            print(f"  ✓ CrossEncoder loaded: {self.model_name}")
            return self._model
        except Exception as e:
            print(f"  ⚠️  CrossEncoder {self.model_name} unavailable ({e})")
            print("  → Fallback sang FlashRank")
            self._model = None
            return None

    def _get_flashrank(self):
        if self._fallback is None:
            from flashrank import Ranker

            self._fallback = Ranker()
            self.backend = "flashrank"
        return self._fallback

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 -> top-k."""
        if not documents:
            return []

        model = self._load_model()
        if model is not None:
            pairs = [(query, doc["text"]) for doc in documents]
            scores = model.predict(pairs)
            if isinstance(scores, (int, float)):
                scores = [scores]
            scores = [float(s) for s in scores]
        else:
            # FlashRank tra ve danh sach da sap xep theo score giam dan.
            ranker = self._get_flashrank()
            passages = [{"id": i, "text": d["text"]} for i, d in enumerate(documents)]
            # flashrank 0.2.x khong nhan top_k -> tra ve toan bo passages,
            # tu sap xep lai ben duoi.
            ranked = ranker.rerank(RerankRequest(query=query, passages=passages))
            order = [r["id"] for r in ranked]
            score_by_id = {r["id"]: float(r["score"]) for r in ranked}
            order += [i for i in range(len(documents)) if i not in score_by_id]
            scores = [score_by_id.get(i, 0.0) for i in order]
            documents = [documents[i] for i in order]

        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [
            RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None
        self.backend = "flashrank"

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []

        from flashrank import Ranker

        if self._model is None:
            self._model = Ranker()

        passages = [{"id": i, "text": d["text"]} for i, d in enumerate(documents)]
        results = self._model.rerank(RerankRequest(query=query, passages=passages))
        results = sorted(results, key=lambda r: r["score"], reverse=True)[:top_k]

        return [
            RerankResult(
                text=documents[r["id"]]["text"],
                original_score=float(documents[r["id"]].get("score", 0.0)),
                rerank_score=float(r["score"]),
                metadata=documents[r["id"]].get("metadata", {}),
                rank=i,
            )
            for i, r in enumerate(results)
        ]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
