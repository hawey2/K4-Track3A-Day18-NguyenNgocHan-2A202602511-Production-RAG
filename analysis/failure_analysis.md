# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyen Ngoc Han  
**Khóa:** K4 - Track 3A · **Ngày:** 04/10/2026

> Nguồn số liệu: `reports/ragas_report.json` (20 câu hỏi) và `reports/naive_baseline_report.json`.
> Lưu ý: RAGAS chấm bằng LLM judge nên điểm có dao động ±0.05–0.10 giữa các lần chạy.

---

## 1. RAGAS Scores (Naive Baseline vs Production)

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.4833 | **0.8142** | **+0.3309** |
| Answer Relevancy | 0.5474 | **0.8154** | **+0.2680** |
| Context Precision | 0.6292 | **0.9333** | **+0.3041** |
| Context Recall | 0.5500 | **0.8667** | **+0.3167** |

Production đạt **cả 4 metric ≥ 0.70**. Đóng góp chính:
BM25 + Dense + RRF (M2) → Cross-encoder rerank (M3) → Contextual prepend (M5).

### Latency breakdown (bonus rubric)

| Bước | Thời gian | Ghi chú |
|------|-----------|---------|
| BM25 index (100 chunks) | 1589 ms | underthesea segmentation là phần lớn |
| BM25 search | 6.5 ms | |
| Rerank 20→3 (cold) | 69 126 ms | nạp model lần đầu |
| Rerank 20→3 (warm) | 389 ms | 20 cặp query-doc qua cross-encoder |
| Index toàn bộ (BM25+Dense, 107 chunks) | 17.8 s | |
| Enrichment (107 chunks × 1 call) | ~2.5 s | combined mode, gpt-4o-mini |
| RAGAS (20 câu × 4 metrics) | 45 s | 80 LLM judge call |
| **Toàn pipeline** | **493 s** | |

---

## 2. So sánh 4 chiến lược chunking (M1)

Chạy trên 26 documents (25 `.md` + 1 PDF có text layer; 2 PDF scan bị bỏ qua vì cần OCR):

| Strategy | Chunks | Avg len | Min | Max |
|----------|--------|---------|-----|-----|
| basic | 51 | 410 | 273 | 565 |
| semantic | 208 | 99 | 6 | 354 |
| hierarchical | 102 (+11 parents) | 204 | 65 | 354 |
| structure-aware | 106 | 196 | 86 | 788 |

**Quan sát:** threshold semantic mặc định 0.85 quá cao với `all-MiniLM-L6-v2` — cosine similarity giữa câu tiếng Việt
tiếp tục thường chỉ ~0.3–0.6, nên gần như **mọi câu đều bị tách** (208 chunks, có chunk chỉ 6 ký tự).
Hierarchical cho balance tốt nhất (102 children, avg 204 chars) nên được chọn cho production.

---

## 3. Bottom-5 Failures

### #1 — "Bao lâu phải đổi mật khẩu một lần?" (avg 0.396)
- **Expected:** theo v2.0 → 120 ngày (v1.0 90 ngày đã bị thay thế)
- **Got:** `Không tìm thấy.`
- **Scores:** faith 0.0 · relevancy 0.0 · precision 0.5833 · **recall 1.0**
- **Error Tree:**
  1. Output sai → **2. Context đúng? CÓ — recall = 1.0** (chunk `mat_khau_v2.md` có trong top-3)
  3. Query OK? OK → 4. **Node hỏng: prompt — LLM từ chối trả lời khi context mâu thuẫn**
- **Root cause:** Câu hỏi nằm ở **corpus mâu thuẫn** (v1 = 90 ngày, v2 = 120 ngày). Cả hai chunk cùng
  xếp hạng cao → context mâu thuẫn → LLM không dám chọn và bỏ cuộc, trả lời "Không tìm thấy".
  Đây là lỗi ở **prompt**, không phải retrieval.
- **Suggested fix:** thêm chỉ dẫn vào prompt *"Nếu nhiều chính sách mâu thuẫn, ưu tiên bản có ngày hiệu lực
  mới nhất và nêu rõ bản cũ đã bị thay thế"*, kèm metadata `effective_date` để lọc trước khi đưa vào context.

### #2 — "Senior 9 năm thâm niên → ngày phép và lương?" (avg 0.673)
- **Expected:** 15 + 3 (9÷3) = 18 ngày; lương Senior (P3-P4) 20–35 triệu
- **Got:** "… được nghỉ 18 ngày phép năm (15 + 3). **Lương của nhân viên này là có lương.**"
- **Scores:** **faith 0.3333** · relevancy 0.8583 · precision 1.0 · recall 0.5
- **Error Tree:**
  1. Output sai → 2. Context đúng? **Một phần — recall 0.5** (thiếu chunk `bang_luong_2024.md`)
  3. Query OK? OK → 4. **Node hỏng: multi-hop — cần 2 nguồn (nghi_phép_nam_v2024 + bang_luong_2024),
  top-3 chỉ đủ cho 1 nguồn** → LLM tự suy diễn phần lương → hallucination
- **Root cause:** đây là câu **multi-hop** (2 tài liệu). `RERANK_TOP_K=3` không đủ context cho câu hỏi
  cần đồng thời thông tin từ 2 file khác nhau.
- **Suggested fix:** tăng `RERANK_TOP_K` lên 5, hoặc thêm bước **multi-hop query decomposition**
  (tách "ngày phép" và "lương Senior" thành 2 sub-query, retrieve riêng, rồi union).

### #3 — "Tạm ứng 15 triệu, trả sau 20 ngày, phạt bao nhiêu?" (avg 0.731)
- **Expected:** hạn 15 ngày → quá hạn 5 ngày; 2%/tháng × 15tr = 300.000/tháng → pro-rata ≈ 50.000
- **Got:** "… 2% × 15.000.000 = 300.000 VNĐ. Vậy nhân viên bị phạt **300.000 VNĐ**."
- **Scores:** **faith 0.2** · relevancy 0.8923 · precision 0.8333 · recall 1.0
- **Error Tree:**
  1. Output sai → 2. Context đúng? CÓ (recall 1.0) → 3. Query OK? OK
  4. **Node hỏng: reasoning — LLM tính sai mốc thời gian**
- **Root cause:** context có đủ dữ kiện (hạn 15 ngày, phí 2%/tháng) nhưng **không nêu cách quy đổi pro-rata**.
  LLM hiểu "quá hạn 5 ngày" là "1 tháng" nên trả về con số 300.000 thay vì 50.000.
  Đây là lỗi **arithmetic + thiếu hướng dẫn tính toán**, không phải lỗi retrieval.
- **Suggested fix:** bổ sung vào prompt một mẫu tính pro-rata
  (`phí mỗi ngày = 2% × số tiền ÷ 30`), và trong corpus nên ghi rõ cách tính theo ngày.

### #4 — "Mua laptop 30 triệu, ai phê duyệt + cần gì từ CNTT?" (avg 0.732)
- **Expected:** Giám đốc phòng ban (5–50tr) + xác nhận cấu hình CNTT + **≥3 báo giá** (>10tr)
- **Got:** "Cần Giám đốc phòng ban (Director) phê duyệt… cần xác nhận của phòng CNTT về cấu hình
  kỹ thuật trước khi đề xuất." — **thiếu yêu cầu 3 báo giá**
- **Scores:** faith 0.5 · relevancy 0.7604 · precision 1.0 · **recall 0.6667**
- **Error Tree:**
  1. Output thiếu → 2. Context đúng? **Một phần — recall 0.667** (thiếu phần "3 báo giá")
  3. Query OK? OK → 4. **Node hỏng: context recall chưa đủ các điều kiện phụ**
- **Root cause:** câu hỏi có **3 điều kiện** nằm rải rác ở 2 section (`## Lưu ý đặc biệt` và phần ngưỡng
  phê duyệt). Chunking theo câu đã tách "3 báo giá" ra khỏi chunk chứa "xác nhận CNTT" → khi rerank top-3
  chỉ giữ được một phần.
- **Suggested fix:** dùng **structure-aware chunking** thay vì tách theo câu, để giữ nguyên section;
  hoặc tăng `RERANK_TOP_K`.

### #5 — "Nghỉ phép không lương 20 ngày cần ai phê duyệt?" (avg 0.822)
- **Expected:** 16–30 ngày → **CEO**; lưu ý >14 ngày phải tự đóng BH
- **Got:** "Cần phê duyệt của Giám đốc điều hành (CEO)." — đúng, nhưng thiếu điều kiện tự đóng BH
- **Scores:** faith 1.0 · relevancy 0.786 · precision 1.0 · **recall 0.5**
- **Error Tree:**
  1. Output thiếu 1 phần → 2. Context đúng? **Một phần — recall 0.5** (thiếu câu ">14 ngày tự đóng BH")
  3. Query OK? OK → 4. **Node hỏng: recall — điều kiện phụ nằm ở chunk khác**
- **Root cause:** đáp án chính đúng nhưng thiếu "note" đi kèm → context recall chỉ 0.5.
- **Suggested fix:** giống #4 — structure-aware chunking để gom bảng ngưỡng + ghi chú vào cùng chunk.

---

## 4. Điểm đáng chú ý: Reranker tiếng Anh phá hỏng ranking tiếng Việt

Trong quá trình chạy, **3/4 failure nặng nhất** ban đầu đều trả về `Không tìm thấy.` Điều tra cho thấy
`BM25 + Dense` **đã trả về đúng chunk ở rank 1**, nhưng bước rerank lại đẩy nó ra khỏi top-3.

**Nguyên nhân gốc:** FlashRank chỉ cung cấp model `ms-marco-TinyBERT-L-2-v2` — **English-only**.
Test kiểm chứng:

| Query | Passage | FlashRank score |
|-------|---------|-----------------|
| *tiếng Anh* "laptop purchase approval IT quotes" | passage tiếng Việt | **0.0000** (cả 4 passage) |
| *tiếng Việt* "mua một chiếc laptop 30 triệu…" | passage tiếng Việt | 0.9835 / 0.6962 / 0.0818 |

Khi dùng cho tiếng Việt, mọi score bị nén vào khoảng **0.93–0.99** → thứ hạng gần như ngẫu nhiên.

**Cách sửa đã áp dụng:** đổi `RERANKER_BACKEND` từ `"flashrank"` sang `"auto"` để dùng
`BAAI/bge-reranker-v2-m3` (đa ngôn ngữ). Phân bố score trở lại có ý nghĩa: **0.749 / 0.369 / 0.016**.

**Ảnh hưởng đo được:**

| Metric | FlashRank (English-only) | bge-reranker-v2-m3 | Δ |
|--------|------------------------|--------------------|---|
| Faithfulness | 0.8250 | 0.8142 | −0.011 |
| Answer Relevancy | 0.6491 | **0.8154** | **+0.166** |
| Context Precision | 0.7917 | **0.9333** | **+0.142** |
| Context Recall | 0.6583 | **0.8667** | **+0.208** |

Reranker mạnh làm **faithfulness giảm nhẹ** (0.825 → 0.814) vì context đúng hơn thì LLM có nhiều thông tin
hơn để diễn đạt, đồng thời các câu trả lời nhiều phần dễ bị judge quy cho "không trung thành" hơn.
Nhưng 3 metric retrieval tăng mạnh, tổng thể tốt hơn rõ rệt.

---

## 5. Case Study (cho presentation)

**Câu hỏi:** *"Nhân viên được nghỉ bao nhiêu ngày phép năm?"* — corpus có **2 phiên bản mâu thuẫn**
(`nghi_phep_nam_v2023.md` = 12 ngày, `nghi_phep_nam_v2024.md` = 15 ngày), ground truth yêu cầu
trả lời theo bản **hiện hành** và nêu rõ bản cũ đã bị thay thế.

**Kết quả quan sát được trong lần chạy đầu:** hệ thống trả lời *"Nhân viên có 9 năm thâm niên được
18 ngày phép (15 + 3)"* — đúng 1 phần nhưng **sai**, vì context bị nhiễu bởi câu hỏi khác (multi-hop
#2 lọt vào) và không có cơ chế ưu tiên phiên bản.

**Error Tree walkthrough:**
1. **Output đúng?** → Không (15 ngày cơ bản bị bỏ qua, nhảy thẳng sang case 9 năm thâm niên)
2. **Context đúng?** → Một phần — có cả 2 phiên bản 2023/2024 cùng tồn tại, không có tín hiệu
   nào để ưu tiên (recall 0.0, precision 0.5)
3. **Query rewrite OK?** → OK, query rõ ràng
4. **Fix ở bước:** **Chunking metadata + prompt** — gắn `effective_date` / `status: superseded` vào
   metadata khi index, lọc theo điều kiện trước, và chỉ dẫn LLM ưu tiên bản hiện hành.

**Nếu có thêm 1 giờ, sẽ optimize:**
- **Metadata versioning** (`effective_date`, `supersedes`) để giải quyết dứt điểm mâu thuẫn văn bản —
  đây là lỗi xuất hiện ở 4/20 câu hỏi (v1 vs v2 mật khẩu, v2023 vs v2024 nghỉ phép).
- **Multi-hop decomposition** cho câu hỏi cần ≥2 tài liệu (tăng `RERANK_TOP_K` 3 → 5).
- **Pro-rata template** trong prompt cho câu hỏi tính toán.
- **Semantic threshold** hạ 0.85 → 0.5 cho corpus tiếng Việt, hoặc dùng lại `chunk_semantic`
  với model đa ngôn ngữ (`bge-m3`).
- **OCR cho 2 file PDF scan** (`BCTC.pdf`, `Nghi_dinh_so_13-2023...pdf`) — hiện bị bỏ qua hoàn toàn,
  làm mất nguồn dữ liệu luật về bảo vệ dữ liệu cá nhân.
