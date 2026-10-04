from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, OPENAI_API_KEY


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    empty = {"faithfulness": 0.0, "answer_relevancy": 0.0,
             "context_precision": 0.0, "context_recall": 0.0, "per_question": []}

    # Wrap trong try/except — RAGAS cần OPENAI_API_KEY và Python 3.11+.
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall])
        df = result.to_pandas()

        def _f(row, key) -> float:
            val = row.get(key)
            # NaN (RAGAS trả về NaN khi metric không tính được) -> 0.0
            try:
                return float(val)
            except (TypeError, ValueError):
                return 0.0

        def _contexts(row) -> list[str]:
            # pandas trả về numpy array cho cột "contexts" -> không dùng `or []`
            # (numpy array truth-value gây ValueError ambiguous).
            val = row.get("contexts")
            if val is None:
                return []
            if isinstance(val, str):
                return [val]
            try:
                return [str(c) for c in list(val)]
            except TypeError:
                return [str(val)]

        per_question = [
            EvalResult(
                question=str(row.get("question", "")),
                answer=str(row.get("answer", "")),
                contexts=_contexts(row),
                ground_truth=str(row.get("ground_truth", "")),
                faithfulness=_f(row, "faithfulness"),
                answer_relevancy=_f(row, "answer_relevancy"),
                context_precision=_f(row, "context_precision"),
                context_recall=_f(row, "context_recall"),
            )
            for _, row in df.iterrows()
        ]

        # Aggregate = mean cua per-question (dung 20 mau de bang tong hop,
        # tranh sai lech khi RAGAS dien gia tri trung binh noi bo).
        def _mean(key) -> float:
            vals = [getattr(r, key) for r in per_question]
            return sum(vals) / len(vals) if vals else 0.0

        return {
            "faithfulness": _mean("faithfulness"),
            "answer_relevancy": _mean("answer_relevancy"),
            "context_precision": _mean("context_precision"),
            "context_recall": _mean("context_recall"),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {type(e).__name__}: {e}")
        if not OPENAI_API_KEY:
            print("     → Thiếu OPENAI_API_KEY. Copy .env.example ra .env rồi điền key.")
        return empty


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    # Diagnostic Tree: metric thấp nhất → nguyên nhân gốc → cách sửa.
    diagnostic_tree = {
        "faithfulness": (
            "LLM đang suy diễn ngoài context (hallucination)",
            "Siết prompt 'trả lời CHỈ dựa trên context', hạ temperature=0, "
            "giảm top_k context",
        ),
        "context_recall": (
            "Thiếu chunk chứa câu trả lời (retrieval miss)",
            "Cải thiện chunking (thử hierarchical/structure-aware), thêm BM25 "
            "vào hybrid search, giảm child_size",
        ),
        "context_precision": (
            "Có nhiều chunk nhiễu chen giữa chunk đúng (retrieval noise)",
            "Thêm reranking, lọc theo metadata, giới hạn top_k context",
        ),
        "answer_relevancy": (
            "Câu trả lời lệch so với câu hỏi (semantic mismatch)",
            "Cải thiện prompt template, thêm HyQA để bridge vocabulary, "
            "kiểm tra query rewrite",
        ),
    }

    if not eval_results:
        return []

    metric_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    scored = []
    for r in eval_results:
        metrics = {k: getattr(r, k, 0.0) or 0.0 for k in metric_keys}
        avg = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=lambda k: metrics[k])
        scored.append((avg, worst_metric, metrics, r))

    scored.sort(key=lambda x: x[0])

    failures = []
    for avg, worst_metric, metrics, r in scored[:bottom_n]:
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        failures.append({
            "question": r.question,
            "ground_truth": r.ground_truth,
            "answer": r.answer,
            "avg_score": round(avg, 4),
            "worst_metric": worst_metric,
            "worst_score": round(metrics[worst_metric], 4),
            "metric_scores": {k: round(v, 4) for k, v in metrics.items()},
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
            "error_tree": (
                f"Answer sai/cao → context đúng? "
                f"recall={metrics['context_recall']:.2f} "
                f"precision={metrics['context_precision']:.2f} → "
                f"query ok? → node hỏng: {worst_metric}"
            ),
        })
    return failures


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
