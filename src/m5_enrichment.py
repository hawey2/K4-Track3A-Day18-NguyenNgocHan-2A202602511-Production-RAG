from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, json, re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, LLM_MODEL


_CLIENT = None


def _get_client():
    """Lazy OpenAI client — None nếu thiếu API key."""
    global _CLIENT
    if not OPENAI_API_KEY:
        return None
    if _CLIENT is None:
        from openai import OpenAI

        _CLIENT = OpenAI()
    return _CLIENT


def _chat(system: str, user: str, max_tokens: int) -> str | None:
    """1 LLM call. Trả về None nếu không có key hoặc call lỗi."""
    client = _get_client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠️  LLM call failed: {type(e).__name__}: {e}")
        return None


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    out = _chat(
        "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.",
        text,
        max_tokens=150,
    )
    if out:
        return out

    # Extractive fallback (không cần API):
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]) + "." if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    out = _chat(
        f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
        "Trả về mỗi câu hỏi trên 1 dòng.",
        text,
        max_tokens=200,
    )
    if out:
        questions = [
            re.sub(r"^[\s\d.)\-–—]+", "", line).strip()
            for line in out.split("\n") if line.strip()
        ]
        questions = [q for q in questions if q]
        if questions:
            return questions[:n_questions]

    # Extractive fallback:
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    out = _chat(
        "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về "
        "chủ đề gì. Chỉ trả về 1 câu.",
        f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
        max_tokens=80,
    )
    if out:
        return f"{out}\n\n{text}"

    # Simple fallback:
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    out = _chat(
        'Trích xuất metadata từ đoạn văn. Trả về DUY NHẤT JSON hợp lệ, không có '
        'markdown fence: {"topic": "...", "entities": ["..."], '
        '"category": "policy|hr|it|finance", "language": "vi|en"}',
        text,
        max_tokens=150,
    )
    if out:
        try:
            # Model có thể bọc trong ```json ... ``` — bóc fence trước khi parse.
            cleaned = re.sub(r"^```(?:json)?|```$", "", out.strip(), flags=re.MULTILINE).strip()
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError) as e:
            print(f"  ⚠️  Metadata JSON parse failed: {e}")

    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    out = _chat(
        """Phân tích đoạn văn và trả về DUY NHẤT JSON hợp lệ, không markdown fence:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}""",
        f"Tài liệu: {source}\n\nĐoạn văn:\n{text}",
        max_tokens=400,
    )
    if not out:
        return {}

    try:
        cleaned = re.sub(r"^```(?:json)?|```$", "", out.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError) as e:
        print(f"  ⚠️  Combined enrichment JSON parse failed: {e}")
        return {}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            # Không có API key / call lỗi / JSON hỏng → rơi về các kỹ thuật
            # extractive để pipeline vẫn chạy được offline.
            if not result:
                summary = summarize_chunk(text)
                questions = generate_hypothesis_questions(text)
                enriched_text = contextual_prepend(text, source)
                auto_meta = extract_metadata(text)
                method = "combined-fallback"
            else:
                summary = result.get("summary", "")
                questions = result.get("questions", [])
                context_line = result.get("context", "")
                enriched_text = f"{context_line}\n\n{text}" if context_line else text
                auto_meta = result.get("metadata", {})
                method = "combined"
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}
            method = "+".join(methods)

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method=method,
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
