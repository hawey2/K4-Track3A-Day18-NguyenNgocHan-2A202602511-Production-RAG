# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyen Ngoc Han  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| **Semantic chunking** | M1 | `chunk_semantic()` | Threshold 0.85 (mặc định trong `config.py`) tạo **208 chunks** so với basic **51** — tức là *nhiều hơn gấp 4*, và có chunk chỉ **6 ký tự**. Nguyên nhân: `all-MiniLM-L6-v2` là model đa ngôn ngữ nhưng chưa fine-tune cho tiếng Việt, cosine similarity giữa hai câu tiếng Việt liên tiếp thường chỉ ~0.3–0.6, thấp hơn nhiều so với giả định của threshold 0.85 (con số này thường được chọn cho model tiếng Anh đã tune). Hệ quả: semantic chunking *phá vỡ* thay vì *bảo toàn* ranh giới câu ở corpus này. Với threshold 0.5, số chunk giảm còn hợp lý và test `test_semantic_groups_by_topic` xanh. → **Threshold phải calibrate theo model + ngôn ngữ, không dùng mặc định có sẵn.** |
| **Hierarchical chunking** | M1 | `chunk_hierarchical()` | Parent (2048) + Child (256) cho **102 children từ 11 parents**, avg 204 chars. Đây là lựa chọn cho production vì cân bằng giữa *precision khi retrieve* (child nhỏ, ít nhiễu) và *context khi sinh câu trả lời* (parent lớn, đủ ngữ cảnh). Chi tiết quan trọng: tôi tách child theo **câu** (`re.split(r'(?<=[.!?])\s+\|\n\n')`) chứ không cắt cứng theo ký tự — tránh cắt giữa chừng làm mất nghĩa. Đây chính là nguyên nhân gốc của failure #4 và #5 (xem Phần 3). |
| **Structure-Aware chunking** | M1 | `chunk_structure_aware()` | **106 chunks**, giữ nguyên header trong text và gắn `section` vào metadata. Regex `^#{1,3}\s+` với `re.MULTILINE` + capturing group cho phép tách theo section. Chunk lớn nhất 788 chars (bảng lương) — lớn hơn `child_size` nhưng **đúng ý**: bảng markdown phải giữ nguyên, cắt giữa bảng sẽ hỏng. Nhận ra: với corpus có bảng + danh sách điều kiện rải rác, structure-aware **vượt trội** semantic về mặt giữ điều kiện phụ. |
| **BM25 + Dense fusion (RRF)** | M2 | `reciprocal_rank_fusion()` | `score(d) = Σ 1/(k + rank + 1)` với `k=60`. RRF giải quyết **bất khả biến thang điểm**: BM25 (0–25, không chuẩn hoá) và cosine similarity (0–1) không thể cộng trực tiếp. RRF chỉ dùng *thứ hạng* nên vô hiệu hoá hoàn toàn vấn đề scale. Quan sát thực nghiệm: BM25 thắng ở **từ khóa chính xác** ("nghỉ_phép", "120 ngày"), Dense thắng ở **paraphrase** ("bao lâu phải đổi mật khẩu" ↔ "thay đổi mật khẩu định kỳ"). Chunk nào hit **cả hai** (metadata `rrf_methods`) là tín hiệu độ tin cậy cao nhất. Một điểm tôi chủ động thêm: lọc `score > 0` ở BM25 để BM25 không lấp điểm hạng cho tài liệu không liên quan và làm nhiễu RRF. |
| **Vietnamese word segmentation** | M2 | `segment_vietnamese()` | Đây là chi tiết **nhỏ nhưng quyết định**. underthesea nối từ ghép bằng dấu `_`: "nghỉ_phép". Nếu giữ nguyên, BM25 tokenize bằng `split(" ")` sẽ coi "nghỉ_phép" là **1 token**, trong khi query "nghỉ phép" là **2 token** → **không bao giờ khớp**. `replace("_", " ")` sửa đúng lỗi này. Ngoài ra tôi thêm `.lower()` + bỏ dấu câu để BM25 robust với viết hoa. |
| **Cross-encoder reranking** | M3 | `CrossEncoderReranker.rerank()` | Latency **389 ms** cho 20→3 (warm), **69 s** cold (nạp model). Đây là bài học lớn nhất của lab — xem Phần 2. Ban đầu tôi dùng FlashRank: **389 ms nhưng hoàn toàn sai** vì model `ms-marco-TinyBERT-L-2-v2` chỉ hiểu tiếng Anh. Sau khi chuyển sang `bge-reranker-v2-m3`, context recall tăng **0.658 → 0.867** (+0.208) và precision **0.792 → 0.933**. Bài học: **biểu diễn nhanh không đồng nghĩa với đúng**; phải kiểm chứng model có hỗ trợ ngôn ngữ đích trước khi dùng. |
| **RAGAS 4 metrics** | M4 | `evaluate_ragas()` | Faithfulness 0.814, Answer Relevancy 0.815, Context Precision 0.933, Context Recall 0.867 — **cả 4 ≥ 0.70**. Metric **thấp nhất là Faithfulness (0.814)** — không phải vì hallucination nhiều, mà vì 3 câu hỏi multi-hop/numeric bị thiếu context khiến LLM suy diễn (failure #2, #3). Điểm kỹ thuật: RAGAS trả `NaN` cho metric không tính được nên tôi chuẩn hoá về `0.0`, và **tính aggregate trên 20 mẫu** thay vì tin giá trị trung bình nội bộ của RAGAS (tránh sai lệch). |
| **Failure analysis / Diagnostic Tree** | M4 | `failure_analysis()` | Sort theo mean của 4 metric, lấy bottom-N, map `worst_metric` → nguyên nhân → cách sửa. Giá trị lớn nhất: nó biến "điểm thấp" thành **hành động cụ thể**. Failure #1 có recall = 1.0 nhưng faithfulness = 0.0 → **loại trừ ngay lỗi retrieval**, biết vấn đề nằm ở prompt/mâu thuẫn corpus. Nếu chỉ nhìn điểm tổng sẽ không bao giờ tìm ra điều đó. |
| **Contextual embeddings / Prepend** | M5 | `contextual_prepend()` + `_enrich_single_call()` | Chạy ở **combined mode: đúng 1 API call/chunk** (107 chunks) thay vì 4 call → tiết kiệm 75% chi phí. Chunk 256 chars rất "trần trụi" khi tách khỏi tài liệu; prepend 1 câu định vị ("Đoạn văn nằm trong phần chính sách lương của tài liệu bang_luong_2024.md") giúp embedding nắm được *chunk này nói về cái gì*, cải thiện match với query không trùng từ khóa. Anthropic benchmark ghi nhận giảm 49% retrieval failure khi dùng độc lập. |
| **Auto metadata** | M5 | `extract_metadata()` | LLM trả JSON `{topic, entities, category, language}` — làm nền cho **metadata filtering** ở bước sau (ví dụ lọc `category=hr`). Chưa khai thác trong lab này nhưng là hạ tầng cho bước lọc theo chính sách hiện hành (xem Phần 3). |
| **Fallback / resilience** | M1–M5 | tất cả | Mọi hàm LLM đều có fallback **extractive** (không cần API): `summarize_chunk` lấy 2 câu đầu, `generate_hypothesis_questions` đổi câu thành câu hỏi, `extract_metadata` trả dict mặc định. Nhờ vậy `pytest` chạy được **không cần API key** và `DenseSearch` tự rơi về `QdrantClient(":memory:")` khi không có Docker. Đây là điều kiện tiên quyết để học lab mà không bị kẹt ở hạ tầng. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

### 2.1 FlashRank phá hỏng toàn bộ ranking tiếng Việt (lỗi nghiêm trọng nhất)

**Exact error / symptom:** không có exception — RAGAS vẫn chạy và trả điểm, nhưng **3/4 câu hỏi
nặng nhất trả về `Không tìm thấy.`** với điểm 0.0. Đây là loại lỗi nguy hiểm nhất: **hệ thống báo cáo
thành công nhưng cho kết quả sai**.

**Quá trình debug:**
1. Đầu tiên nghi ngờ M5 enrichment hoặc prompt → in thử `enriched_text`, thấy enrichment hoạt động bình thường.
2. Đo từng tầng riêng: chạy `BM25 + Dense + RRF` trên câu "mua một chiếc laptop 30 triệu..." → **chunk
   `mua_sam.md` đúng nằm ở rank 1**. Vậy lỗi **không ở retrieval**.
3. Chạy tiếp qua rerank → chunk đúng **bị đẩy ra khỏi top-3**, thay bằng `dao_tao_noi_bo.md`. Nghi phạm
4. **Test kiểm chứng có quyết định:** cho FlashRank chấm 4 passage tiếng Việt với **query tiếng Anh**
   → cả 4 trả về **0.0000**. Kết luận: `ms-marco-TinyBERT-L-2-v2` **English-only**, không hiểu tiếng Việt.
5. Xác nhận thêm: với query tiếng Việt, score bị nén vào **0.93–0.99** → không có phân táo, thứ hạng gần
   như ngẫu nhiên.
6. Sửa: đổi `RERANKER_BACKEND` từ `"flashrank"` → `"auto"` để dùng `bge-reranker-v2-m3` đa ngôn ngữ.
   Phân bố score trở lại có ý nghĩa: **0.749 / 0.369 / 0.016**.

**Kết quả:** context recall **+0.208**, context precision **+0.142**, answer relevancy **+0.166**.

**Bài học:** phải **kiểm chứng khả năng ngôn ngữ của model với một test tối thiểu** trước khi tin vào
điểm số. Và khi metric giảm mạnh đồng loạt trên nhiều câu hỏi → nghi ngờ **hạ tầng/model** trước, không
phải logic prompt.

### 2.2 RAGAS crash với numpy array trong pandas DataFrame

**Exact error message:**
```
```

**Nguyên nhân:** `try/except` trong scaffold đã che lỗi (chỉ in cảnh báo rồi trả về 0.0), nên RAGAS **chạy
mất 9 phút rồi trả toàn 0.0** mà không ai biết vì sao. Sau khi bỏ comment để debug, lỗi lộ ra ở
`list(row.get("contexts") or [])`: pandas biểu diễn cột `contexts` (danh sách văn bản) bằng **numpy array**,

**Cách sửa:** thay bằng hàm `_contexts()` xử lý riêng 3 trường hợp (`None` / `str` / iterable) và ép từng
phần tử về `str`. Đồng thời bỏ `val != val` (NaN check) vì toán tử này cũng lỗi với array.

**Bài học:** `except Exception` rộng như vậy **giấu lỗi thay vì xử lý**. Phải in **exception type + message**
(`{type(e).__name__}: {e}`) để tự báo đúng nguyên nhân — tôi đã sửa luôn ở M5 theo cùng nguyên tắc.

### 2.3 Xung đột phiên bản numpy/scipy/torch

**Exact error messages:**
```

**Nguyên nhân & cách giải quyết:**
1. `sentence-transformers` mới cài kéo theo **transformers 5.18** → hỏng với torch 2.2.2. Pin
   `transformers>=4.49,<5` + `sentence-transformers>=3.3,<4`.
2. **numpy 2.5** không tương thích torch 2.2.2 (`Numpy is not available`) → pin `numpy>=1.26,<2`.
3. `scipy` bản mới yêu cầu numpy≥2 → pin `scipy>=1.11,<1.14`.

**Bài học:** trong ML, **một dependency kéo theo cả chuỗi**. Cài "bản mới nhất" theo thói quen sẽ vỡ.
Phải đọc `requirements.txt` của thư viện cha trước khi nâng version, và kiểm tra tương thích numpy/torch

### 2.4 Máy thiếu bốn thứ: Python 3.11+, Docker, model cache, embedding dim

- **Python:** hệ thống là 3.9.6, RAGAS cần 3.11+ → tạo venv bằng Python 3.12 của miniconda.
- **Docker/Qdrant:** không cài Docker, nhưng scaffold đã có fallback `QdrantClient(":memory:")` — dùng luôn.
  Chi tiết cần lưu ý: **id của Qdrant phải là `uint64`/UUID**, nên tôi dùng `id=i+1` (bỏ qua 0).
- **Model cache:** máy chỉ có sẵn `all-MiniLM-L6-v2`, nên dùng cho embedding. Nhưng `config.py` hardcode
  `EMBEDDING_DIM = 1024` (kích thước của bge-m3) → sẽ khiến Qdrant **reject mọi upsert**. Sửa bằng cách
  **suy ra dim từ `len(vectors[0])`** thay vì tin config — tránh hardcode sai.
- **Reranker:** đây là lỗi tốn nhiều thời gian nhất và tạo ra kết quả sai âm thầm (mục 2.1).

### 2.5 Kiến thức còn thiếu & cách bổ sung

- **Hiểu về corpus mâu thuẫn.** Lab không nói trước, nhưng 4/20 câu hỏi đều dính phiên bản v1/v2 hoặc
  2023/2024. Đây là đặc thù rất phổ biến của corpus doanh nghiệp. → Học cách gắn metadata versioning
  (`effective_date`, `status: superseded`) để lọc ở tầng retrieval thay vì để LLM tự phân giải.
- **Phân biệt lỗi retrieval vs lỗi generation.** Trước lab này tôi có xu hướng thấy điểm thấp là "model
  yếu". Diagnostic Tree chỉ ra context recall = 1.0 trong khi faithfulness = 0.0 → lỗi **không nằm ở
  retrieval**. → Đây là kỹ năng chẩn đoán tôi đoạt được nhiều nhất.
- **Hiệu năng thực tế của RAG.** Latency cross-encoder 389 ms/query là con số tôi cần đưa vào thiết kế
  hệ thống thật (cache kết quả rerank, chỉ rerank top-10 thay vì top-20).

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Trợ lý tra cứu tương tác thuốc (Drug Interaction Lookup Assistant)

#### 1. Hiện trạng

- **Pipeline hiện tại:** chưa có hệ thống nào chạy được — đây là project mới, thiết kế dựa trên những gì
  lab hôm nay đã chứng minh. Khác biệt cốt lõi so với lab: truy vấn không phải câu hỏi tự nhiên mà là
  **cặp thuốc** (A + B → có tương tác gì, mức độ nào?), và corpus là nhãn thuốc (SmPC), cơ sở dữ liệu
  tương tác, tài liệu hướng dẫn lâm sàng.
- **Known issues / rủi ro đã thấy rõ trong lab — sẽ lặp lại y hệt nếu không xử lý trước:**
  - **Corpus mâu thuẫn phiên bản.** Lab hỏng 4/20 câu vì có cả v1 và v2 cùng tồn tại. Với thuốc, điều này
    nghiêm trọng hơn nhiều: nhãn thuốc được cập nhật, cảnh báo mới được thêm. Trả lời theo nhãn cũ là
    thông tin lâm sàng sai.
  - **Reranker hỏng âm thầm.** FlashRank English-only làm hỏng thứ hạng mà hệ thống vẫn báo thành công
    (context recall mất 0.208). Ở bài toán tổng quát đó là điểm số tệ; ở đây nó là thứ hạng sai
    trong danh sách tương tác.
  - **Mất điều kiện phụ.** Lab rơi điều kiện "3 báo giá" vì chunking cắt sai. Với thuốc, rơi điều kiện
    "giảm liều ở người suy thận" hoặc "chống chỉ định ở phụ nữ mang thai" là rủi ro cho bệnh nhân.
  - **Câu hỏi multi-hop hỏng** vì `RERANK_TOP_K=3` không đủ — tương tác thuốc đòi hỏi thông tin từ nhiều
    nguồn (nhãn A, nhãn B, cơ chế CYP).
  - **LLM tính sai số học.** Lab chứng minh LLM quy đổi pro-rata sai (failure #3). Liều dùng, khoảng cách
    giữa hai liều, tính theo cân nặng — phải có công cụ kiểm tra, không để LLM tự suy luận.

#### 2. Kế hoạch áp dụng

1. [ ] **Chunking strategy: structure-aware theo mục thuốc, không theo câu**
   - Chia theo mục cố định của nhãn thuốc (Chỉ định, Chống chỉ định, Tương tác thuốc, Thận trọng, Tác dụng
     phụ, Liều dùng). Mỗi mục là một đơn vị ngữ nghĩa, không cắt lẫn nhau.
   - Quan trọng nhất: **mục "Tương tác thuốc" phải tách thành từng interaction record riêng** — mỗi bản
     ghi là một cặp (thuốc A, thuốc B) + cơ chế + mức độ. Đây là đơn vị nhỏ nhất có nghĩa; chunk theo câu sẽ
     gộp nhiều cặp vào một chunk và làm RRF/rerank mất tác dụng.
   - Metadata bắt buộc: `generic_name`, `brand_names`, `atc_code`, `drug_class`, `cyp_enzyme`,
     `severity` (contraindicated/major/moderate/minor), `evidence_level`, `label_version`, `source`.
   - Thêm `effective_date` / `supersedes` để loại nhãn cũ ngay ở tầng index — bài học trực tiếp từ lab.

2. [ ] **Search: Hybrid BM25 + Dense + RRF, nhưng bắt buộc có entity linking trước**
   - Giữ hybrid + RRF — lab đo được context recall +0.32 so với baseline.
   - **Chuẩn hoá tên thuốc trước khi search.** "Paracetamol" = "Acetaminophen" = "Panadol". Nếu không
     ánh xạ về generic name, BM25 không match giữa cách viết của người dùng và nhãn thuốc — cùng lỗi
     "nghỉ_phép" (1 token) vs "nghỉ phép" (2 token) đã gặp ở M2.
   - Index **cả ba lớp tên**: hoạt chất gốc (INN), tên thương mại, tên lớp thuốc.
   - Metadata filter theo `severity` và `effective_date` trước khi rerank.

3. [ ] **Reranking: có, cross-encoder đa ngôn ngữ + truy vấn theo cặp**
   - Dùng `bge-reranker-v2-m3`. **Không dùng FlashRank** — chỉ hỗ trợ tiếng Anh, lab đã đo mất 0.208
     context recall.
   - Với truy vấn cặp A+B: **retrieve riêng theo từng thuốc rồi giao (intersect)**, không dựa vào một
     thứ hạng phẳng cho cả cặp. Nếu không, chunk của A ngấn ngang đẩy chunk của B ra khỏi top-k.
   - Top-5 thay vì top-3 — lab đã chứng minh `RERANK_TOP_K=3` làm hỏng câu hỏi multi-hop.

4. [ ] **Evaluation: RAGAS + chỉ số lâm sàng riêng, tất yếu có kiểm thử phủ định**
   - RAGAS 4 metrics làm chỉ số hệ thống, nhưng **chưa đủ cho y tế**. Thêm:
     - **Pair recall** — có retrieve đúng bản ghi tương tác của cặp A-B không?
     - **Severity accuracy** — phân loại mức độ có đúng không?
     - **Abstention calibration** (chỉ số quyết định an toàn): với cặp **không có tương tác**, hệ thống
       phải trả lời "không ghi nhận" chứ không bịa. Đo **false-negative rate** — tỉ lệ bỏ sót một tương
       tác nguy hiểm. Chỉ số này quan trọng hơn điểm trung bình.
   - Test set bắt buộc phải có **cả cặp âm** (không tương tác). Nếu chỉ test cặp có tương tác, hệ thống
     học cách đoán "có" cho mọi thứ — và điểm trung bình vẫn đẹp trong khi hành vi sai.
   - Đối chiếu với nhãn thuốc chính thức; một phần bộ test cần dược sĩ review.
   - CI gate: chặn deploy nếu false-negative rate vượt ngưỡng, **kể cả khi RAGAS vẫn đẹp**.

5. [ ] **Enrichment: combined single-call, chạy offline lúc ingest**
   - 1 call/chunk thay vì 4 — tiết kiệm 75% chi phí, giữ nguyên như lab.
   - Extract metadata phục vụ an toàn: `drug_class`, `cyp_enzyme`, `mechanism`, `severity`,
     `evidence_level`. Dùng để lọc, và để **hiển thị nguồn trích dẫn cho người dùng** — mỗi khẳng định
     phải truy được về nhãn thuốc cụ thể.
   - **Contextual prepend** để chunk 256 chars "trần trụi" nắm được nó đang nói về cặp thuốc nào.
   - Enrich lúc **ingest**, không phải mỗi query.
   - Bổ sung **công cụ tính liều** có kiểm tra thay vì để LLM tự nhân — lab đã chứng minh LLM tính sai.

#### 3. Timeline triển khai

- **Tuần 1 — dữ liệu & chuẩn hoá:** lấy nguồn nhãn thuốc + cơ sở dữ liệu tương tác; xây bảng ánh xạ
  generic ↔ brand ↔ ATC; gắn `label_version` / `supersedes`; dựng test set ~150 cặp (**có cả cặp âm**).
  *Bước quyết định — không chuẩn hoá tên thuốc thì mọi tối ưu sau đều vô nghĩa.*
- **Tuần 2 — chất lượng retrieval:** structure-aware chunking theo mục; tách interaction record;
  entity linking; truy vấn cặp (retrieve A, retrieve B, intersect); nâng top-k lên 5.
- **Tuần 3 — lớp an toàn:** abstention và phát hiện "không có tương tác"; công cụ tính liều; trích dẫn
  nguồn cho từng khẳng định; hiển thị rõ mức độ nghiêm trọng; `temperature=0`.
- **Tuần 4 — đánh giá & vận hành:** chạy RAGAS + chỉ số lâm sàng; dược sĩ review một phần bộ test; CI gate
  theo false-negative rate; log câu hỏi thật (đã ẩn danh) để bổ sung vào test set.

#### 4. Bài học mang về

RAG không phải "cứ dùng model mạnh là xong". Trong lab này **thứ hạng bị phá hỏng bởi một model không hỗ trợ
tiếng Việt** — và nó suýt bị bỏ qua vì hệ thống vẫn exit code 0 và vẫn in ra điểm số. Nếu không **đo từng
tầng riêng biệt**, mình sẽ đi tối ưu sai (thêm prompt, đổi chunking) trong khi nguyên nhân thật nằm ở model.

Áp vào bài toán tương tác thuốc, bài học này nặng hơn hẳn: **một hệ thống trông bình thường nhưng xếp hạng
sai thì nguy hiểm hơn một hệ thống báo lỗi**. Trong y tế, thứ tệ nhất không phải là "hỏng", mà là **trả lời
tự tin nhưng sai**.

Bốn nguyên tắc mình rút ra:
1. **Đo từng tầng, không đo tổng** — retrieval đúng + rerank sai vẫn ra câu trả lời sai.
2. **Kiểm chứng giả định bằng test nhỏ** — 10 dòng test English-only tiết kiệm hàng giờ debug mù.
3. **Đừng nuôi dữ liệu bẩn, và đừng chỉ nhìn điểm trung bình** — 4/20 câu hỏi hỏng vì corpus có hai phiên
   bản mâu thuẫn; tương tác thuốc cần chỉ số **phủ định** (bỏ sót bao nhiêu tương tác nguy hiểm), không
   chỉ điểm trung bình.
4. **Thiết kế để hệ thống được phép nói "không biết"** — trong lab, khi context mâu thuẫn, LLM đã tự trả
   lời "Không tìm thấy" và mình đánh dấu đó là lỗi. Sang dự án này mình đánh giá ngược lại: đó là hành vi
   an toàn. Trong y tế, khả năng từ chối trả lời là một tính năng, không phải điểm trừ.
