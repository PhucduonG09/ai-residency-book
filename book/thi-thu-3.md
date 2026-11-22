# Đề thi thử 3 — Mô phỏng đủ các vòng

## Cách làm

Đây là đề cuối của sách. Bạn làm nó ở tuần 48, vào hoặc quanh Thứ Năm 30/9/2027 (ngày ghi trong `book/chapters.yaml`), sau khi đã học xong chương 0–14. Lộ trình giao cho đề việc `w48-run`, 4 giờ: "Làm Đề thi thử 3 theo từng vòng, đúng giờ; sau đó chấm và ghi lỗi vào sổ lỗi". Đề có bốn vòng, xếp theo đúng thứ tự công khai của quy trình tuyển: entrance test, coding interview, video presentation, technical interview. Đề có hai mục đích. Một là đo lần cuối cả kiến thức lẫn cách trình bày khi bị bấm giờ. Hai là giúp bạn quyết định dùng mấy tuần còn lại trước hạn nộp hồ sơ vào việc gì.

**Lưu ý trung thực.** Tin tuyển Batch 17 của Qualcomm RRP (đăng 13/10/2026, mức B) chỉ liệt kê năm vòng: Application → Entrance test → Coding interview → Video presentation → Technical interview. Nội dung, thời lượng, định dạng và tiêu chí chấm của bốn vòng sau Application đều không công khai (mức D; xem mục 13.1 và 14.1). Vì vậy đề này là **bản mô phỏng của sách**. Cơ cấu từng vòng, số phút, loại câu, thang điểm và mọi ngưỡng đọc kết quả đều do người soạn chọn hoặc ước lượng, và chưa được hiệu chỉnh trên người làm thật. Đề không mô phỏng vòng Application; vòng đó là gói hồ sơ bạn đã chuẩn bị ở chương 13. Danh sách vòng của đợt 2027 có thể khác Batch 17 (mục 13.1 ghi Batch 14 có thêm vòng Screening), nên hãy đọc lại tin tuyển của chính đợt bạn nộp.

**Lịch hai buổi** [lựa chọn của người soạn; số phút chấm là ước lượng, chưa đo]. Cả đề kể cả lượt hai và chấm cần khoảng 365 phút, không vừa một buổi, nên chia hai buổi. Vòng 4 cần người phỏng vấn, nên buổi 2 đặt vào cuối tuần.

| Buổi | Mốc | Việc | Phút |
| --- | --- | --- | --- |
| 1 — Thứ Năm 30/9/2027 | 0:00–1:15 | Vòng 1 — Entrance test (giấy bút) | 75 |
| 1 | 1:15–1:25 | Nghỉ | 10 |
| 1 | 1:25–2:35 | Vòng 2 — Coding interview (máy tính, ghi âm) | 70 |
| 1 | 2:35–2:45 | Nghỉ | 10 |
| 1 | 2:45–3:25 | Lượt hai của Vòng 1 và Vòng 2 (tối đa 20 phút mỗi vòng) | 40 |
| 1 | 3:25–3:55 | Chấm Vòng 1 theo Lời giải, gắn nhãn lỗi | 30 |
| 1 | 3:55–4:10 | Chạy bộ chấm, chấm tay Vòng 2 từ bản ghi âm | 15 |
| 2 — Thứ Bảy 2/10/2027 | 0:00–0:30 | Vòng 3 — Video presentation (quay một lần, tự xem lại, đánh dấu tiêu chí) | 30 |
| 2 | 0:30–0:35 | Nghỉ, người phỏng vấn vào | 5 |
| 2 | 0:35–1:20 | Vòng 4 — Technical interview (với người khác, ghi hình) | 45 |
| 2 | 1:20–1:30 | Người phỏng vấn điền phiếu chấm | 10 |
| 2 | 1:30–1:50 | Xem lại bản ghi Vòng 4, đếm từ/phút và từ đệm (Lab 3 chương 14) | 20 |
| 2 | 1:50–1:55 | Mở file người phỏng vấn, đối chiếu ý đáp án, ghi sổ lỗi | 5 |
| **Tổng** | | Buổi 1: 250; buổi 2: 115 | **365** |

**Thời gian có đủ không.** Bốn vòng làm bài (không kể nghỉ và chấm) hết 75 + 70 + 30 + 45 = 220 phút. Cả đề hết 365 phút, tức khoảng 6.1 giờ, vượt 4 giờ của `w48-run` khoảng 2.1 giờ [ước lượng, chưa đo]. Theo `plan/weeks.yaml`, tuần 48 đang có 12.5 giờ; thay 4 giờ của `w48-run` bằng 6.1 giờ thì thành khoảng 14.6 giờ, vẫn dưới trần 15 giờ mỗi tuần mà Đề thi thử 2 dùng. Nếu tính bản quay ở Vòng 3 là lần quay đầu của việc `w48-c` (quay video bản cuối), `w48-c` bớt khoảng nửa giờ và tuần 48 còn khoảng 14.1 giờ. Lượt ba (làm nốt không tính giờ) nằm ngoài ngân sách này. Đề không khuyến nghị làm cả bốn vòng một buổi: riêng bốn vòng và ba lần nghỉ đã hết 245 phút, cộng phần chấm thành hơn 6 giờ liền. Việc `w48-c` nên làm **sau** Vòng 3, ví dụ Chủ nhật 3/10/2027, và dùng kết quả của Vòng 3 để sửa. Việc hẹn người phỏng vấn và gửi file cho họ nên làm từ tuần 47, trong việc `w47-plan`.

**Quy tắc chung.**

- Mỗi vòng bấm giờ riêng. Hết giờ vòng nào thì dừng vòng đó, kể cả khi chưa xong, rồi nghỉ đúng số phút trong lịch. Không dùng tài liệu và không dùng trợ lý AI ở mọi vòng. Thứ được dùng của từng vòng ghi ở đầu vòng đó.
- **Không đọc trước:** đề của các vòng sau, mục Lời giải, bộ chấm `labs/thi-thu-3/cham_coding.py` và file người phỏng vấn `labs/thi-thu-3/nguoi-phong-van.md`. Lời giải Vòng 1–2 đọc sau lượt hai của buổi 1; Lời giải Vòng 3 đọc sau khi tự chấm Vòng 3 ở buổi 2. Câu hỏi, ý đáp án và câu trả lời mẫu của Vòng 4 chỉ có trong file người phỏng vấn: file đó để in hoặc gửi cho người khác, và bạn chỉ mở nó sau khi buổi 2 kết thúc.
- **Hai con số điểm cho Vòng 1 và Vòng 2**, như hai đề trước. Lượt một là đúng giờ của vòng. Hết giờ, gạch ngang chỗ đang dừng; ở Vòng 2 thì lưu file thành `bai_lam_luot1.py`. Lượt hai làm cuối buổi 1, tối đa 20 phút cho mỗi vòng, viết bằng bút khác màu. Câu nào vẫn trống thì làm nốt không tính giờ (lượt ba). Ghi **điểm trong giờ** (chỉ lượt một) và **điểm cuối** (mọi lượt). Vòng 3 và Vòng 4 chỉ có một con số, vì một buổi nói không có lượt hai.
- **Gắn nhãn lỗi** cho mỗi ý mất điểm ở Vòng 1 và Vòng 2, như hai đề trước: **K** (khái niệm), **T** (tính sai), **L** (đọc đề sai, thiếu thuật ngữ), **H** (chỉ làm được ở lượt hai hoặc ba). Với Vòng 3 và Vòng 4, ghi vào sổ lỗi tên dòng của phiếu chấm bị trượt.

**Cách chấm.**

| Vòng | Thang | Ai chấm |
| --- | --- | --- |
| 1 | 100 điểm: Phần A 15 câu × 4 điểm, Phần B 2 bài × 20 điểm | Bạn, theo Lời giải |
| 2 | 20 điểm: mỗi bài 7 điểm do bộ chấm tự động cho, 3 điểm chấm tay phần nói to | Bộ chấm, rồi bạn chấm tay từ bản ghi âm |
| 3 | Số tiêu chí đạt trên 12, mỗi tiêu chí lấy từ một chỗ có thật của chương 13 | Bạn, từ bản quay |
| 4 | Thang 30 điểm cho phỏng vấn thử của chương 14 (mục 14.5); chỉ dòng Hành vi được điều chỉnh, vì đề có một câu chuyện | Người phỏng vấn, theo phiếu trong file người phỏng vấn |

Chỉ cho điểm nguyên ở Vòng 1 và Vòng 2. **Độ khó** của Vòng 1: ★ hiểu định nghĩa, ★★ mức đề thi đầu vào, ★★★ câu phân loại. Các ý ★★★ cộng lại 14 điểm.

## Bản đồ đề

### Vòng 1 — Entrance test

Thứ tự trong bảng là thứ tự làm bài. Cột Phút là ước lượng cho người học mục tiêu, tức người vừa học xong chương 0–14 và làm đề lần đầu (cộng ra 78 phút). Đây là ước lượng sau khi đề đã được cắt bớt cho gần giờ thi [chưa đo trên người thật]; hai ước lượng độc lập cho bản trước khi cắt là 80 và 90 phút.

| Câu | Chương, mục | Kỹ năng | Độ khó | Điểm | Phút [ước lượng] |
| --- | --- | --- | --- | --- | --- |
| A1 | Chương 1, mục 1.3 | Số điều kiện và sai số làm tròn của bình phương tối thiểu | ★ | 4 | 2 |
| A2 | Chương 3, mục 3.3 | Gauss có điều kiện, phân rã Cholesky | ★★ | 4 | 3 |
| A3 | Chương 4, mục 4.1; chương 6, mục 6.5 | Cắt gradient theo chuẩn gộp và theo từng phần tử | ★★ | 4 | 3 |
| A4 | Chương 5, mục 5.2 | AUC bằng đếm cặp, precision khi tỉ lệ lớp đổi | ★★ | 4 | 3 |
| A5 | Chương 7, mục 7.5 | Patch của ViT khi đổi độ phân giải | ★ | 4 | 2.5 |
| A6 | Chương 8, mục 8.3 | RoPE và vị trí tương đối | ★★ | 4 | 3 |
| A7 | Chương 8, mục 8.5; chương 9, mục 9.1 | Dịch nhãn của mô hình ngôn ngữ nhân quả | ★ | 4 | 2 |
| A8 | Chương 8, mục 8.2 và 8.5 | Cross-attention: shape, mặt nạ, cache | ★ | 4 | 3 |
| A9 | Chương 9, mục 9.1 | Hai phép gộp BPE bằng tay | ★ | 4 | 3.5 |
| A10 | Chương 9, mục 9.4 | int8 absmax và một giá trị ngoại lệ | ★★ | 4 | 3 |
| A11 | Chương 10, mục 10.3 | Khuếch tán: quá trình tiến, dự đoán x₀ từ nhiễu | ★★ | 4 | 3.5 |
| A12 | Chương 11, mục 11.2 | Ablation 2²: hiệu ứng chính và tương tác | ★★ | 4 | 2.5 |
| A13 | Chương 12, mục 12.1 | BM25: idf và chuẩn hóa độ dài | ★★ | 4 | 3 |
| A14 | Chương 12, mục 12.2 | Chunk có chồng lấn: cắt ngang và chi phí | ★ | 4 | 2.5 |
| A15 | Chương 12, mục 12.4 | Cohen κ ba nhãn của giám khảo LLM | ★★ | 4 | 4 |
| B1 | Chương 9, mục 9.1; chương 3, mục 3.5; chương 5, mục 5.5 | Nội suy hai mô hình ngôn ngữ: NLL, Jensen, EM | ★★ đến ★★★ | 20 | 16 |
| B2 | Chương 12, mục 12.2 và 12.4; chương 11, mục 11.2 và 11.3 | RAG: phân rã lỗi, từ chối có phạt, so và kết hợp hai phương án | ★★ đến ★★★ | 20 | 16.5 |
| | | Đọc đề | | | 2 |
| **Tổng** | | | | **100** | **78** |

Phút theo phần [ước lượng]: A 43.5, B 32.5, đọc đề 2. Mốc giờ trong 75 phút: đọc đề 2, Phần A đến phút 45, Phần B từ phút 45 đến 75. Mốc cho Phần A 43 phút và Phần B 30 phút, nên theo bảng người học mục tiêu thiếu khoảng 3 phút, phần lớn rơi vào ý cuối của B2.

### Vòng 2 — Coding interview

| Việc | Nội dung | Điểm | Phút [ước lượng] |
| --- | --- | --- | --- |
| Đọc đề, chép khung, bật ghi âm | | | 3 |
| Bài 1 | Cấu trúc dữ liệu và thuật toán, cỡ LeetCode medium | 10 (7 tự động + 3 nói to) | 30 |
| Bài 2 | ML coding: một độ đo của NLP | 10 (7 tự động + 3 nói to) | 35 |
| Chạy lại cả file, lưu | | | 2 |
| **Tổng** | | **20** | **70** |

### Vòng 3 — Video presentation

| Việc | Phút |
| --- | --- |
| Chuẩn bị: thẻ ý chính, thử tiếng 10 giây | 3 |
| Quay một lần (bài nói 3 phút) | 4 |
| Xem lại, bấm giờ từng đoạn, điền bảng | 11 |
| Đánh dấu 12 tiêu chí (chạy Lab 1 và Lab 5, thử 60 giây cho mỗi con số) | 12 |
| **Tổng** | **30** |

### Vòng 4 — Technical interview

| Phần | Phút |
| --- | --- |
| Mở đầu: giới thiệu bản thân 60 giây | 3 |
| 2 chuỗi kỹ thuật × 12 phút, người phỏng vấn bốc 2 trong 4 chuỗi mới | 24 |
| Đào sâu dự án | 8 |
| 1 câu hành vi (kể 2 phút, nối tiếp 4 phút) | 6 |
| Ứng viên hỏi lại | 3 |
| Đệm, chuyển phần | 1 |
| **Tổng** | **45** |

Khung này rút từ khung 60 phút của mục 14.1 [quy tắc của đề này]: giữ nguyên 2 chuỗi × 12 phút; phần dự án giảm từ 12 xuống 8 phút; phần hành vi giảm từ 2 câu xuống 1; mở đầu, phần hỏi lại và phần đệm đều ngắn hơn.

## Vòng 1 — Entrance test (75 phút, 100 điểm)

**Cách làm vòng này.** 75 phút liền, bàn học, giấy nháp, điện thoại để xa. Không tài liệu, không máy tính bỏ túi. Bảng hằng số dưới đây là thứ duy nhất được tra. Làm Phần A rồi đến Phần B. Câu nào vượt số phút ở Bản đồ đề cộng 2 phút thì ghi "bỏ" và làm tiếp. Đáp số được để dạng biểu thức và vẫn đủ điểm, kiểu 4/11, ln 13, √6/5 (đây chỉ là ví dụ dạng, không trùng đáp số câu nào). Đáp số thập phân cần đúng đến 2 chữ số có nghĩa, như Đề thi thử 2. Chỉ ghi lý do khi đề hỏi "vì sao" hoặc "một dòng".

**Bảng hằng số (được tra).** ln 2 = 0.6931; ln 3 = 1.0986; ln 5 = 1.6094; ln 7 = 1.9459; √2 = 1.4142; √3 = 1.7321. Hệ số 95% hai phía của phân phối chuẩn là 1.96. Ở mức 5% hai phía và độ mạnh 0.8, tổng z bằng 2.80. Sai số làm tròn đơn vị (một nửa machine epsilon): float32 u ≈ 6·10⁻⁸; float64 u ≈ 1.1·10⁻¹⁶.

### Phần A — Câu ngắn (15 câu, 60 điểm, đến phút 45)

Mỗi câu 4 điểm, mỗi ý (a), (b) 2 điểm.

**A1.** *(Chương 1 · ★ · 4 điểm)* Ma trận A ∈ ℝ¹⁰⁰⁰ˣ² có hai giá trị kỳ dị σ₁ = 50 và σ₂ = 0.05. Vector b = Ax với một x chưa biết, nên phần dư bằng 0. (a) Tính κ(A) và κ(AᵀA). (b) Dùng quy tắc ngón tay cái "sai số tương đối của nghiệm cỡ κ·u", với u lấy từ bảng hằng số. Giải bằng phương trình chuẩn AᵀAx̂ = Aᵀb trong float32 thì sai cỡ bao nhiêu? Giải bằng QR hoặc `lstsq` trong float32 thì sai cỡ bao nhiêu? Một dòng: vì sao phương trình chuẩn mất độ chính xác?

**A2.** *(Chương 3 · ★★ · 4 điểm)* Vector x = (x₁, x₂) có phân phối N(μ, Σ), với μ = (1, −1) và Σ có các hàng (9, 3) và (3, 5). (a) Phân phối của x₂ khi biết x₁ = 4: trung bình và phương sai. (b) Tìm ma trận tam giác dưới L với Σ = LLᵀ (Cholesky). Tính log det Σ từ đường chéo của L.

**A3.** *(Chương 4, 6 · ★★ · 4 điểm)* Một mô hình có hai tensor tham số. Gradient của chúng là g₁ = (2, −1) và g₂ = (2, 4). (a) `clip_grad_norm_` với `max_norm` = 1 tính một chuẩn L2 gộp trên mọi tensor. Chuẩn đó bằng bao nhiêu, và hai gradient sau khi cắt là gì? (b) `clip_grad_value_` với ngưỡng 1 cắt từng phần tử về đoạn [−1, 1]. Gradient sau khi cắt là gì? Tính cosin của góc giữa nó và gradient gốc, coi cả bốn thành phần là một vector. Cách cắt nào giữ hướng? Một dòng giải thích.

**A4.** *(Chương 5 · ★★ · 4 điểm)* Một bộ phân loại cho ba mẫu dương các điểm 0.9, 0.6, 0.4 và cho bốn mẫu âm các điểm 0.7, 0.35, 0.3, 0.2. (a) Tính AUC bằng đếm cặp. Tính precision và recall ở ngưỡng 0.5 (dự đoán dương khi điểm ≥ 0.5). (b) Nhân mỗi mẫu âm thành 10 bản cùng điểm (40 mẫu âm), giữ nguyên ba mẫu dương. AUC và precision ở ngưỡng 0.5 đổi thành bao nhiêu? Một dòng: khi lớp dương hiếm, nên báo chỉ số nào cạnh AUC?

**A5.** *(Chương 7 · ★ · 4 điểm)* Một ViT cắt ảnh thành các patch 14 × 14 không chồng nhau. Bỏ qua token phân loại. (a) Ảnh 224 × 224 và ảnh 336 × 336 cho bao nhiêu token? (b) Chuyển từ ảnh 224 sang ảnh 336 thì chi phí phần ma trận điểm attention tăng bao nhiêu lần, và chi phí các lớp tuyến tính tính theo token tăng bao nhiêu lần? Bảng vị trí học được, huấn luyện ở ảnh 224, cần xử lý thế nào?

**A6.** *(Chương 8 · ★★ · 4 điểm)* RoPE trong 2 chiều: q ở vị trí m được quay một góc mθ, k ở vị trí n được quay một góc nθ, và điểm attention là tích vô hướng của hai vector đã quay. Cho q = (2, 0), k = (0, 1), θ = π/3. (a) Tính điểm cho (m, n) = (1, 2). Nếu không quay thì điểm bằng bao nhiêu? (b) Tính điểm cho (m, n) = (2, 1). Không tính lại, cho biết điểm của (m, n) = (4, 5). Điểm phụ thuộc vào đại lượng nào của hai vị trí? Nó có giữ nguyên khi đổi chỗ m với n không?

**A7.** *(Chương 8, 9 · ★ · 4 điểm)* Bạn huấn luyện một mô hình ngôn ngữ nhân quả trên một batch 8 chuỗi, mỗi chuỗi 513 token. (a) Viết bằng chỉ số cách cắt đầu vào và mục tiêu từ mỗi chuỗi. Batch cho bao nhiêu vị trí dự đoán? (b) Một lỗi: mục tiêu ở vị trí i là chính token i (quên dịch), còn mặt nạ nhân quả vẫn cho mỗi vị trí nhìn thấy chính nó. Loss train đi về đâu, vì sao? Mô hình sinh văn bản ra sao?

**A8.** *(Chương 8 · ★ · 4 điểm)* Một Transformer encoder–decoder có d = 512 và 8 head. Câu nguồn dài 40 token; decoder đã sinh 25 token. (a) Trong cross-attention của một head, với 25 vị trí phía decoder, Q, K, V và ma trận điểm có shape gì? Cross-attention có cần mặt nạ nhân quả không? Khi các câu nguồn trong batch có độ dài khác nhau thì cần mặt nạ gì? (b) Khi giải mã từng token có cache (một câu nguồn, batch 1), K và V của cross-attention ở một lớp chiếm bao nhiêu byte ở fp16, và chúng khác cache của self-attention ở điểm nào? Cache self-attention của lớp đó sau 25 token chiếm bao nhiêu byte?

**A9.** *(Chương 9 · ★ · 4 điểm)* Một kho đếm từ, ký hiệu cách nhau bằng dấu cách, không có ký hiệu hết từ, và cặp chỉ tính bên trong từng từ: `t h a n h` xuất hiện 4 lần, `t h a n` 3 lần, `n h a n h` 5 lần. (a) Đếm mọi cặp ký hiệu kề nhau, có nhân với tần số. Phép gộp đầu tiên của BPE chọn cặp nào? Tổng số token của kho trước và sau phép gộp đó là bao nhiêu? (b) Phép gộp thứ hai chọn cặp nào? Vì sao số đếm của cặp `a n` tụt mạnh sau phép gộp đầu?

**A10.** *(Chương 9 · ★★ · 4 điểm)* Lượng tử hóa int8 absmax đối xứng: mã = làm tròn của x·127/max|x|, giải mã = mã·max|x|/127. Cho x = (0.254, −0.381, 0.07, 1.27). (a) Tính mã int8 và sai số tuyệt đối lớn nhất sau khi giải mã. (b) Thêm phần tử ngoại lệ 12.7 vào cùng tensor. Tính mã mới của bốn phần tử cũ, giá trị giải mã của 0.07 và sai số tuyệt đối lớn nhất của bốn phần tử cũ. Nêu một cách sửa.

**A11.** *(Chương 10 · ★★ · 4 điểm)* Quá trình tiến của mô hình khuếch tán: xₜ = √ᾱ·x₀ + √(1 − ᾱ)·ε. (a) Với √ᾱ = 0.8 (tức ᾱ = 0.64), x₀ = −1.2 và ε = 0.5, tính xₜ và SNR = ᾱ/(1 − ᾱ). Mạng dự đoán nhiễu ε̂ = 0.25. Dự đoán x̂₀ suy từ ε̂ bằng bao nhiêu, và lệch x₀ bao nhiêu? (b) Ở một bước t lớn hơn (nhiễu hơn), ᾱ = 25/169. Với cùng sai số |ε − ε̂| = 0.25, x̂₀ lệch bao nhiêu? Viết hệ số khuếch đại sai số theo SNR và nêu hệ quả.

**A12.** *(Chương 11 · ★★ · 4 điểm)* Ablation 2² cho hai thành phần A và B. Mỗi ô là độ chính xác dev trung bình trên 5 seed (%): không có thành phần nào 70.0; chỉ A 71.0; chỉ B 70.5; cả A và B 73.0. (a) Tính hiệu ứng chính của A và của B (mỗi hiệu ứng lấy trung bình trên hai mức của thành phần kia), và tính tương tác. (b) Bảng "bỏ từng thành phần khỏi hệ đầy đủ" báo A đóng góp bao nhiêu, B đóng góp bao nhiêu? Tổng của hai con số đó so với mức tăng toàn phần 3.0 thế nào? Nên báo bảng nào?

**A13.** *(Chương 12 · ★★ · 4 điểm)* BM25 với idf kiểu Lucene như mục 12.1, k₁ = 1.2, b = 0.75. Kho có N = 49 đoạn. (a) Tính idf của một từ có mặt ở 2 đoạn và của một từ có mặt ở cả 49 đoạn. (b) Truy vấn chỉ gồm từ hiếm ở (a). Đoạn D₁ dài bằng nửa độ dài trung bình và chứa từ đó 1 lần; đoạn D₂ dài gấp đôi độ dài trung bình và chứa từ đó 2 lần. Tính phần tf của hai đoạn. Đoạn nào xếp trên? Nếu đặt b = 0, đoạn nào xếp trên? Ý cuối chỉ cần một dòng lý do, không cần tính.

**A14.** *(Chương 12 · ★ · 4 điểm)* Chia tài liệu thành chunk theo câu, mỗi chunk c = 5 câu. Một sự thật cần cho câu trả lời gồm m = 3 câu liền nhau. (a) Theo công thức của mục 12.2, tính xác suất sự thật bị cắt ngang khi chồng lấn 0, 1 và 2 câu. (b) Với chồng lấn 2 câu, số chunk gấp xấp xỉ bao nhiêu lần so với khi không chồng lấn? Nếu top-3 là ba chunk liền nhau thì chúng chứa bao nhiêu câu khác nhau (không chồng lấn thì là 15)?

**A15.** *(Chương 12 · ★★ · 4 điểm)* 200 câu trả lời được một người và một giám khảo LLM gán một trong ba nhãn Đúng, Sai, Không rõ. Bảng đếm, hàng là nhãn của người, cột là nhãn của giám khảo theo thứ tự Đúng, Sai, Không rõ: hàng Đúng (90, 10, 0); hàng Sai (15, 45, 10); hàng Không rõ (5, 5, 20). (a) Tính p₀, pₑ và Cohen κ. (b) Theo người, tỉ lệ câu Đúng là bao nhiêu? Theo giám khảo là bao nhiêu? Giám khảo lệch về phía nào (dẫn một con số từ bảng)? Hệ quả cho một bảng xếp hạng chấm bằng giám khảo này là gì?

### Phần B — Suy dẫn (2 bài, 40 điểm, phút 45–75)

Viết đủ các bước chính. Điểm từng ý ghi trong ngoặc.

**B1. Nội suy hai mô hình ngôn ngữ.** *(Chương 9, 3, 5 · 20 điểm · 16 phút)* Hai mô hình ngôn ngữ A và B dùng **cùng một tokenizer**. Trên một chuỗi kiểm tra 4 token, xác suất mỗi mô hình gán cho token đúng ở vị trí t = 1, …, 4 (cho cùng tiền tố) là aₜ và bₜ:

$$
a = (0.5,\ 0.1,\ 0.4,\ 0.05), \qquad b = (0.2,\ 0.4,\ 0.1,\ 0.25), \qquad m_t(w) = w\,a_t + (1 - w)\,b_t ,\quad w \in [0, 1].
$$

Mô hình hỗn hợp gán xác suất mₜ(w) cho token đúng. NLL là trung bình theo token của −ln xác suất (đơn vị nat); perplexity bằng e mũ NLL. Cho sẵn: −ln 0.35 ≈ 1.050 và −ln 0.15 ≈ 1.897.

(a) *(6 điểm, ★★)* Tính NLL của A, của B và của hỗn hợp với w = 0.5 (không cần perplexity). So sánh ba mô hình bằng NLL và giải thích kết quả.

(b) *(7 điểm, ★★)* (i) Chứng minh với mọi w ∈ [0, 1]: NLL của hỗn hợp ≤ w·NLL(A) + (1 − w)·NLL(B). Viết hệ quả cho perplexity. (4 điểm) (ii) Đặt ℓ(w) = ∑ₜ ln mₜ(w). Tính ℓ′ và ℓ″, rồi chứng minh ℓ lõm. Tính ℓ′(0.5) và nói nên tăng hay giảm w. (3 điểm)

(c) *(7 điểm, ★★★)* (i) Coi "mô hình nào sinh ra token t" là một biến ẩn. Viết một vòng EM cho w: E-step tính trách nhiệm rₜ của A, M-step cho w mới. Tính một vòng từ w = 0.5. Kết quả có cùng chiều với dấu của ℓ′(0.5) không? (4 điểm) (ii) Nêu hai điều kiện để phép nội suy và phép so NLL (hay perplexity) ở (a) có nghĩa: một điều về tokenizer, một điều về tập dữ liệu dùng để chọn w. Với mỗi điều kiện, nói cái gì hỏng nếu vi phạm. (3 điểm)

**B2. RAG: trả lời hay từ chối.** *(Chương 12, 11 · 20 điểm · 16.5 phút)* Một hệ hỏi đáp RAG trên kho ArXiv chạy trên 400 câu dev có đoạn vàng gán tay, và hệ luôn trả lời. Ở 260 câu, đoạn vàng nằm trong ngữ cảnh: 208 câu trả lời đúng, 52 câu sai. Ở 140 câu còn lại: 35 câu đúng, 105 câu sai.

(a) *(6 điểm, ★★)* Tính R, a₁, a₀ và độ chính xác Acc theo công thức phân rã của mục 12.2. Có hai phương án cải thiện. P1: thêm reranker, đưa R lên 0.75, a₁ và a₀ giữ nguyên. P2: đổi bộ đọc, đưa a₁ lên 0.88, R và a₀ giữ nguyên. Tính Acc của mỗi phương án. Tính ∂Acc/∂R và ∂Acc/∂a₁ tại điểm hiện tại, rồi so sánh hai phương án và giải thích bằng hai đạo hàm.

(b) *(7 điểm, ★★)* Đổi cách chấm: +1 nếu đúng, 0 nếu từ chối, −λ nếu sai, với λ = 1. (i) Chứng minh: với một câu mà hệ trả lời đúng với xác suất p, trả lời có lợi hơn từ chối khi và chỉ khi p > λ/(1 + λ). (2 điểm) (ii) Hệ luôn trả lời thì đúng 243 câu, sai 157 câu. Tính điểm trung bình mỗi câu nếu hệ luôn trả lời. Tính điểm trung bình nếu hệ từ chối đúng ở mọi câu không có đoạn vàng trong ngữ cảnh (cách này không làm được trong thực tế, chỉ dùng để so). (2 điểm) (iii) Cách làm được: từ chối khi điểm truy hồi của đoạn đầu thấp hơn một ngưỡng. Ở ngưỡng đã chọn, hệ từ chối 120 câu. Trong đó 100 câu không có đoạn vàng (nếu trả lời thì đúng 20 câu) và 20 câu có đoạn vàng (nếu trả lời thì đúng 14 câu). Tính độ phủ, độ chính xác trên phần được trả lời và điểm trung bình mỗi câu. Nếu cách chấm chỉ tính đúng/sai (λ = 0) thì từ chối có lợi không? (3 điểm)

(c) *(7 điểm, ★★★)* (i) Bạn chạy cả P1 và P2 trên cùng 400 câu. Giả sử 15% số câu là cặp bất đồng (đúng ở phương án này, sai ở phương án kia), và sai số chuẩn ghép cặp của hiệu hai độ chính xác xấp xỉ √(0.15/n). Chênh kỳ vọng giữa P1 và P2 ở (a) bằng bao nhiêu lần sai số chuẩn khi n = 400? Cần bao nhiêu câu để có độ mạnh 0.8 ở mức 5% hai phía (tổng z 2.80)? Bậc độ lớn là đủ. Từ đó, nên chọn giữa P1 và P2 thế nào? (4 điểm) (ii) Làm cả hai: thêm reranker và đổi bộ đọc, nên R = 0.75, a₁ = 0.88, a₀ giữ nguyên. Tính Acc, rồi so với Acc hiện tại cộng hai mức tăng riêng của P1 và P2. Phần chênh đến từ số hạng nào của công thức phân rã? Một bảng ablation 2² trên 400 câu có thấy được phần chênh này không? (3 điểm)

## Vòng 2 — Coding interview (70 phút, 20 điểm)

**Cách làm vòng này.** Vòng này mô phỏng một buổi coding interview [giả định của sách: nội dung vòng thật không công khai]. Máy tính chỉ dùng trình soạn thảo và Python (thư viện chuẩn và NumPy). Không internet, không tài liệu, không trợ lý AI, không tự hoàn thành code. Trước khi bấm giờ, chép `labs/thi-thu-3/bai_lam_mau.py` thành `bai_lam.py` ở thư mục gốc của sách và bật ghi âm trên điện thoại. File khung chỉ có chữ ký và docstring của hai bài, giống hai khối code dưới đây. Làm mỗi bài theo ba bước, như khi có người phỏng vấn ngồi cạnh:

1. **Nói to trước khi gõ** (khoảng 3 phút, ghi âm): cách tiếp cận, một ví dụ nhỏ chạy bằng tay, độ phức tạp thời gian và bộ nhớ của từng thao tác, các trường hợp biên.
2. **Viết code và tự thử** với dữ liệu tự nghĩ ra, trong khối `if __name__ == "__main__":`.
3. **Trả lời câu hỏi nối tiếp** in dưới bài, bằng lời, ghi âm, tối đa 2 phút.

Mỗi bài 10 điểm. Bộ chấm cho 7 điểm bằng test ẩn có seed; mỗi test chạy trong một tiến trình riêng có giới hạn thời gian. Bạn tự chấm 3 điểm từ bản ghi âm theo Lời giải. Mốc giờ: đọc đề và chép khung 3 phút; Bài 1 đến phút 33; Bài 2 từ phút 33 đến 68; 2 phút cuối chạy lại cả file và lưu. Hết 70 phút, lưu file thành `bai_lam_luot1.py`. Chỉ sau đó mới chạy `python labs/thi-thu-3/cham_coding.py --student bai_lam_luot1.py` (điểm trong giờ), rồi chạy lại trên `bai_lam.py` sau lượt hai (điểm cuối).

**Bài 1 — Sổ ghi chỉ số theo step.** *(10 điểm · 30 phút)* Bạn lưu loss dev của nhiều lần chạy huấn luyện, và cần trả lời nhanh câu hỏi "ở step s, giá trị gần nhất đã ghi của lần chạy r là bao nhiêu?".

```python
class MetricLog:
    """Sổ ghi một chỉ số (ví dụ loss trên tập dev) của nhiều lần chạy huấn luyện.

    record(run, step, value): ghi giá trị `value` (float) của lần chạy `run` (str) tại bước `step` (int ≥ 0).
        Với mỗi `run`, các lần gọi record có `step` TĂNG NGẶT (lần sau lớn hơn lần trước).
        Các lần chạy khác nhau được ghi xen kẽ nhau.
    at(run, step): trả về `value` của bản ghi có step lớn nhất mà vẫn ≤ `step` (int ≥ 0) trong lần chạy `run`;
        trả về None nếu lần chạy đó chưa có bản ghi nào với step ≤ `step` (kể cả khi `run` chưa từng được ghi).

    Tổng số lần gọi record và at có thể tới 2·10⁵, phần lớn vào cùng một lần chạy; bộ chấm có giới hạn
    thời gian cho cả chuỗi lời gọi."""
    def __init__(self): ...
    def record(self, run, step, value): ...
    def at(self, run, step): ...
```

*Ví dụ:* sau `record("a", 10, 0.9)` và `record("a", 20, 0.8)`, `at("a", 15)` trả 0.9, `at("a", 20)` trả 0.8, `at("a", 5)` trả `None`.

*Câu hỏi nối tiếp (trả lời sau khi code xong):* nếu đề bỏ cam kết "step tăng ngặt", tức bản ghi có thể đến không theo thứ tự, bạn đổi cấu trúc dữ liệu thế nào, và độ phức tạp của `record`, `at` thành bao nhiêu?

**Bài 2 — BLEU cấp corpus.** *(10 điểm · 35 phút)* Viết độ đo BLEU cho bài dịch máy, theo đúng định nghĩa trong docstring.

```python
def corpus_bleu(hyps, refs, max_n=4):
    """BLEU cấp corpus, mỗi câu MỘT câu tham chiếu, không làm trơn.

    hyps: list các câu máy dịch; mỗi câu là list các token (str), có thể rỗng.
    refs: list cùng độ dài với hyps; refs[i] là list token của câu tham chiếu cho hyps[i].
    Trả về (bleu, precisions, bp):
      precisions: list gồm max_n số thực; precisions[n − 1] là precision n-gram đã cắt của cả corpus:
          tử số = tổng trên mọi câu i, trên mọi n-gram g của hyps[i], của min(số lần g có trong hyps[i],
                  số lần g có trong refs[i]);
          mẫu số = tổng số n-gram của mọi hyps[i] (một câu dài L có max(0, L − n + 1) n-gram).
          Nếu mẫu số bằng 0 thì precision đó là 0.0.
      bp: brevity penalty với c = tổng độ dài mọi hyps[i], r = tổng độ dài mọi refs[i]:
          bp = 1.0 nếu c > r; bp = exp(1 − r/c) nếu 0 < c ≤ r; bp = 0.0 nếu c = 0.
      bleu = bp · exp((1/max_n) · Σₙ ln precisions[n − 1]); nếu có precision bằng 0 thì bleu = 0.0.
    bleu, bp và mọi phần tử của precisions là float."""
```

*Ví dụ:* hyps = [["the", "cat", "sat", "on"]], refs = [["the", "cat", "sat", "on", "the", "mat"]] và `max_n` = 2 cho precisions [1.0, 1.0], bp = e mũ (1 − 6/4) ≈ 0.6065 và bleu ≈ 0.6065.

*Câu hỏi nối tiếp (trả lời sau khi code xong):* nếu mỗi câu có **nhiều** câu tham chiếu, hàm của bạn phải đổi ở những chỗ nào?

## Vòng 3 — Video presentation (30 phút)

**Cách làm vòng này.** Vòng này mô phỏng vòng video, mà độ dài, chủ đề, ngôn ngữ, hạn nộp, nền tảng và tiêu chí chấm của nó đều không công khai (mục 13.1, bảng các vòng, mức D). Sách dùng bản thủ 3 phút bằng tiếng Anh, sáu đoạn, theo mục 13.5. Bạn quay **một lần, không cắt, không quay lại** [quy tắc của đề này], vì mục đích là đo bản nói khi chỉ có một cơ hội, chưa phải để làm ra bản nộp. Được dùng một tấm thẻ ghi sáu ý chính, mỗi đoạn một dòng. Không được đọc nguyên kịch bản. Đặt một đồng hồ bấm giờ trong khung hình hoặc ngay cạnh, để lúc xem lại đọc được mốc giờ.

Theo bảng ở Bản đồ đề: chuẩn bị 3 phút, quay 4 phút, xem lại và điền bảng dưới đây 11 phút, đánh dấu 12 tiêu chí 12 phút (tiêu chí 5–6 cần chạy Lab 1, tiêu chí 9 cần Lab 5, tiêu chí 10 cần thử 60 giây cho mỗi con số). Cột "Số từ" lấy từ kịch bản bạn đã chốt nếu bạn nói sát kịch bản. Nếu có bản chép lời, đếm trên bản chép lời. Nếu đã xuất được CSV `start,end,word` từ Whisper, đọc nó bằng `read_words_csv` ở ô cuối Lab 4 chương 13.

**Bảng giờ bấm từng đoạn.** Ngân sách giây lấy từ "Cấu trúc sáu đoạn, 180 giây" của mục 13.5. Mốc kết thúc mục tiêu là tổng cộng dồn: 0:20, 0:45, 1:30, 2:15, 2:40, 3:00.

| Đoạn | Ngân sách (giây) | Mốc kết thúc mục tiêu | Mốc kết thúc thật | Giây thật | Lệch so với ngân sách (%) | Số từ | Từ/phút |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1. Hook và kết luận một câu | 20 | 0:20 | | | | | |
| 2. Vấn đề và vì sao khó | 25 | 0:45 | | | | | |
| 3. Cách làm | 45 | 1:30 | | | | | |
| 4. Bằng chứng | 45 | 2:15 | | | | | |
| 5. Vai trò của tôi và giới hạn | 25 | 2:40 | | | | | |
| 6. Kết: vì sao chương trình này | 20 | 3:00 | | | | | |
| **Tổng** | **180** | | | | | | |

Lệch (%) = 100·(giây thật − ngân sách)/ngân sách. Từ/phút = số từ·60/giây thật. Đây là cách bấm giờ theo đoạn mà ví dụ giải mẫu 13.6 gọi là cách dự phòng khi không có công cụ.

**12 tiêu chí, lấy từ chương 13.** Chương 13 **không có thang điểm cho video**: nó có các quy tắc kiểm và các danh sách "đúng/sai" ở những chỗ ghi trong ngoặc. Vòng này vì thế ghi **số tiêu chí đạt trên 12** và không quy ra điểm. Mỗi tiêu chí đánh Đạt hoặc Trượt.

| # | Tiêu chí | Chỗ trong chương 13 |
| --- | --- | --- |
| 1 | Tổng thời lượng từ 160 đến 190 giây | Ví dụ giải mẫu 13.5, "Quy tắc kiểm của sách" |
| 2 | Không đoạn nào lệch quá 30% so với ngân sách giây của nó | Ví dụ giải mẫu 13.5, cùng quy tắc |
| 3 | Đủ sáu đoạn, đúng thứ tự | Mục 13.5, "Cấu trúc sáu đoạn, 180 giây" |
| 4 | Tốc độ cả bài trong vùng 130–150 từ/phút, và mỗi đoạn dài từ 45 giây trở lên (đoạn 3, 4) trong vùng 120–160 từ/phút | Mục 13.5 (chuẩn tính 140 từ/phút); bài tính nhanh 11 (vùng 130–150); vùng 120–160 lấy từ dòng Cách nói của thang 30 ở mục 14.5 chương 14 |
| 5 | Lab 1 chạy trên kịch bản (hoặc bản chép lời) không báo viết tắt nào chưa định nghĩa | Mục 13.5, "Ngôn ngữ"; Lab 1 |
| 6 | Lab 1 không báo câu nào dài quá 25 từ | Lab 1 (cột câu dài); bài tính nhanh 8 |
| 7 | Có **một** con số trung tâm, nói kèm baseline, số lần chạy và giới hạn | Mục 13.5, "Người chấm đọc gì ở đây" |
| 8 | Nói rõ phần việc của chính bạn trong dự án | Mục 13.5, "Nội dung" |
| 9 | Mọi con số trong video khớp CV và statement | Lab 5; ví dụ giải mẫu 13.7 |
| 10 | Giải thích được mỗi con số trong video trong 60 giây (thử ngay sau khi quay, bấm giờ) | Ví dụ giải mẫu 13.8 |
| 11 | Không mở đầu bằng giai thoại cá nhân hay câu sáo | Bẫy thường gặp; mini-test câu 5 |
| 12 | Không khẳng định điều chưa xác minh về chương trình, nhất là ở đoạn kết | Mini-test câu 6 |

Thêm hai số **đo để so lần quay sau, không chấm Đạt/Trượt**, vì chương 13 không đặt ngưỡng cho chúng: số quãng ngắt từ 1 giây trở lên và số từ đệm trên 100 từ (định nghĩa như Lab 4). Tốc độ của bốn đoạn ngắn (1, 2, 5, 6) cũng chỉ ghi để so, không chấm riêng [quy tắc của đề này]: với đoạn 20 giây nói đúng 47 từ, bấm tay lệch ±1 giây đã đổi tốc độ khoảng ±7 từ/phút.

## Vòng 4 — Technical interview (45 phút, 30 điểm)

**Cách làm vòng này.** Vòng này mô phỏng một buổi phỏng vấn kỹ thuật bằng tiếng Anh [giả định của sách: bộ câu hỏi, thời lượng và tiêu chí chấm của vòng thật không công khai, mục 14.1]. Người phỏng vấn dùng **4 chuỗi mới** mà chương 1–14 chưa hỏi, và bốc ngẫu nhiên 2 chuỗi để hỏi. Sau đó là một câu đào sâu dự án và một câu hành vi. **Người phỏng vấn cầm file `labs/thi-thu-3/nguoi-phong-van.md`**: file đó chứa hướng dẫn, phiếu câu hỏi, ý đáp án, phiếu chấm, câu trả lời mẫu và số liệu đã kiểm của Vòng 4. Đề này không in câu hỏi hay đáp án nào của Vòng 4.

- **Một ngày trước (hẹn người từ tuần 47):** in hoặc gửi file `labs/thi-thu-3/nguoi-phong-van.md` cho người sẽ hỏi bạn (bạn học, anh chị khóa trên, giảng viên). In thẳng từ thư mục hoặc gửi đường dẫn; bạn **không mở** file đó trước khi buổi 2 kết thúc.
- **Mang theo:** CV, một tờ giấy trắng để viết công thức, điện thoại để ghi hình. Nói tiếng Anh. Muốn hỏi lại cho rõ câu hỏi thì cứ hỏi: điều đó được tính điểm.
- **Không có ai hỏi (chế độ tự ghi hình):** nhờ một người bất kỳ in hoặc chép câu hỏi của 2 chuỗi bốc ngẫu nhiên từ file người phỏng vấn lên thẻ, mỗi thẻ một tầng, rồi úp xuống. Nếu không nhờ được ai, chạy `python -c "import random; print(random.sample([1, 2, 3, 4], 2))"`, mở file người phỏng vấn, chỉ đọc mục Phiếu câu hỏi tới đúng hai chuỗi đã bốc và lật từng tầng một; không cuộn xuống mục Ý đáp án. Ở chế độ này, bạn tự điền phiếu chấm từ bản ghi, sau buổi ít nhất một giờ. Điểm tự chấm thường cao hơn điểm người khác chấm [nhận định của người soạn, chưa đo], nên ghi rõ "tự chấm" cạnh con số.
- **Sau buổi:** xin lại phiếu chấm và ba nhận xét viết của người phỏng vấn. Chép lời phần kỹ thuật bằng Whisper (ô cuối Lab 4 chương 13 đọc được CSV của nó); nếu không có Whisper, chỉ chép tay 3 câu trả lời dài khoảng 2 phút, để vừa 20 phút của lịch. Đếm từ/phút và từ đệm bằng Lab 3 chương 14 cho dòng "Cách nói". Sau đó mới mở file người phỏng vấn để đọc ý đáp án và câu trả lời mẫu.

**Cách chấm.** Thang 30 điểm cho phỏng vấn thử của mục 14.5 (thang của chương 14, không phải tiêu chí chính thức của chương trình nào). Năm dòng giữ nguyên chương 14: Kỹ thuật 12, Con số 3, Làm rõ và "không biết" 3, Đào sâu dự án 4, Cách nói 4. Chỉ dòng Hành vi được điều chỉnh, vì đề này có một câu chuyện thay vì hai; đây là **điều chỉnh của đề**, chương 14 không có:

| Phần | Tiêu chí đếm được | Điểm |
| --- | --- | --- |
| Hành vi (1 câu chuyện; điều chỉnh của đề này) | Có số (1), "I" ≥ "we" (1), có bài học (1), ≤ 2.5 phút (1); tương đương chấm câu chuyện đó hai lần theo mức 0.5 của chương 14 | 4 |

Ngưỡng của mục 14.5: **≥ 24** sẵn sàng; **18 đến dưới 24** luyện lại phần yếu nhất; **dưới 18** quay về ngân hàng chuỗi. Dòng Kỹ thuật chia 2 nên tổng có thể lẻ 0.5; ba vùng phủ liên tục. Phiếu chấm đầy đủ, kể cả thang tự chấm 4 tầng của mục 14.1, nằm trong file người phỏng vấn.

## Bảng chấm theo chương

Bảng này chỉ dành cho Vòng 1. Ghi **điểm cuối** vào cột "Điểm của bạn" rồi tra mức. Ngưỡng theo chương là heuristic của sách, như hai đề trước: vững từ 75% điểm tối đa, cần ôn từ 50% đến dưới 75%, yếu dưới 50% (làm tròn lên số nguyên). Các ngưỡng chưa được hiệu chỉnh trên người làm thật. Chương 7, 10, 11 mỗi chương chỉ có một câu 4 điểm nên được gộp thành một dòng; dòng đó và dòng Phần I chỉ báo động chung, không chẩn đoán được từng chương. B1 và B2 được tách theo ý cho đúng chương: B1(b) (Jensen, đạo hàm) và B1(c)(i) (EM) tính cho Phần I; B2(c)(i) (sai số chuẩn ghép cặp, cỡ mẫu) tính cho chương 11.

| Chương | Câu (điểm) | Tối đa | Điểm của bạn | Vững | Cần ôn | Yếu | Mục không được hỏi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Phần I | A1 (chương 1), A2 (chương 3), A3 (chương 4, 6), A4 (chương 5) (16); B1(b) (chương 2, 3) (7); B1(c)(i) (chương 5) (4) | 27 | | 21–27 | 14–20 | 0–13 | Chương 2 chỉ qua đạo hàm ở B1(b)(ii); chương 6 chỉ qua A3 |
| 7, 10, 11 | A5, A11, A12 (12); B2(c)(i) (chương 11) (4) | 16 | | 12–16 | 8–11 | 0–7 | CNN kinh điển (7.1–7.4); VAE, GAN, học tự giám sát (10.1, 10.2, 10.4–10.5); đọc paper, rò rỉ, viết kết quả (11.1, 11.4–11.5) |
| 8 | A6, A7, A8 (12) | 12 | | 9–12 | 6–8 | 0–5 | RNN (8.1); vì sao chia √dₖ (8.2); pre-LN và đếm FLOPs (8.4–8.5) |
| 9 | A9, A10 (8); B1(a), B1(c)(ii) (9) | 17 | | 13–17 | 9–12 | 0–8 | Quy luật quy mô (9.2); hậu huấn luyện (9.3); pass@k (9.5) |
| 12 | A13, A14, A15 (12); B2 trừ (c)(i) (16) | 28 | | 21–28 | 14–20 | 0–13 | Graph-RAG và PPR (12.3); trung thành của câu trả lời (12.4); công bằng, riêng tư, bản quyền (12.5) |
| **Tổng** | | **100** | | Xem mục Tổng kết | | | |

**Điểm ★★★.** Cộng riêng B1(c) và B2(c), tối đa 14 điểm. Con số này không dùng để chọn chương ôn. Nó đo khả năng đi xa hơn đáp số, thứ mà Vòng 4 cũng đo.

## Tổng kết và đọc kết quả

**Bảng điểm bốn vòng.** Ghi vào trang Tiến độ, cạnh số của Đề thi thử 1 và 2.

| Vòng | Thang | Điểm trong giờ | Điểm cuối | Mốc đọc [heuristic của sách, trừ Vòng 4] |
| --- | --- | --- | --- | --- |
| 1 — Entrance test | 100 | | | 75–100 vững; 60–74 cần ôn có mục tiêu; dưới 60 hoặc có chương "yếu" thì ưu tiên số một (cùng mốc với hai đề trước) |
| 2 — Coding interview | 20 | | | 15–20 vững; 10–14 cần luyện; dưới 10 yếu (75% và 50%, như bảng chương) |
| 3 — Video | 12 tiêu chí | — | | Không có mốc: mỗi tiêu chí trượt là một việc sửa |
| 4 — Technical interview | 30 | — | | ≥ 24 sẵn sàng; 18 đến dưới 24 luyện phần yếu nhất; dưới 18 quay về ngân hàng chuỗi (mốc của mục 14.5) |

**Thời gian còn lại.** `plan/weeks.yaml` kết thúc ở tuần 48 (đến 3/10/2027). Hạn nộp hồ sơ của đợt 2027 chưa công bố tại thời điểm viết đề. Lịch ngược của chương 13 dùng hạn **giả định** 12/11/2027 (chưa xác minh, suy từ Batch 17). Nếu hạn đó đúng, từ 4/10 đến hạn còn 39 ngày, khoảng 5.6 tuần. Lịch của các vòng sau Application không công khai (mục 13.1); nếu chúng diễn ra sau hạn nộp [suy luận của người soạn, không có nguồn], thời gian luyện cho Vòng 2–4 thật có thể dài hơn. Mọi số giờ dưới đây là **đề xuất ngoài lộ trình**: bạn tự quyết, và nên giữ dưới trần 15 giờ mỗi tuần.

**Quy tắc một dòng.** Sửa vòng yếu nhất trước, nhưng không bỏ vòng nào quá một tuần: các vòng thật diễn ra liên tiếp, và mỗi vòng loại một số ứng viên [suy luận của người soạn từ cấu trúc nhiều vòng, không có số liệu công khai].

- **Vòng 1 yếu** (dưới 60, hoặc có chương "yếu"): học lại theo bảng Câu sai → ôn gì ở dưới, chương "yếu" trước. Mỗi tuần làm lại một bộ bài tính nhanh bấm giờ của một chương (25 phút) và mini-test của chương đó. Chương 9 hoặc chương 12 yếu là ưu tiên số một của nhánh NLP. Khoảng 3 giờ mỗi tuần trong 3 tuần. Nếu điểm cuối trừ điểm trong giờ từ 15 điểm trở lên, vấn đề chính là tốc độ (như hai đề trước): mỗi ngày làm một bộ bài tính nhanh bấm giờ.
- **Vòng 2 yếu** (dưới 10): xem test nào trượt. Trượt test tốc độ: luyện phân tích độ phức tạp trước khi gõ, mỗi tuần 2 bài có ràng buộc n lớn (luồng B). Trượt test đúng sai ở biên (`b1_bang`, `b1_khong_co`, `b2_bp`): tập viết test biên trước, theo bước "ví dụ nhỏ chạy bằng tay". Mất điểm nói to: mỗi bài luồng B đều ghi âm 3 phút nói trước khi gõ. Khoảng 2.5 giờ mỗi tuần. Bộ test của Bài 2 dùng lại được: viết lại `corpus_bleu` sau 3 ngày, không nhìn lời giải.
- **Vòng 3 có tiêu chí trượt:** tiêu chí 1–4 (thời lượng, nhịp): sửa kịch bản theo ngân sách từ của mục 13.5, chạy lại Lab 1, quay lại và bấm giờ từng đoạn. Tiêu chí 5–6, 11 (ngôn ngữ): sửa câu và viết tắt bằng Lab 1. Tiêu chí 7–10, 12 (nội dung, nhất quán): đối chiếu với CV bằng Lab 5, tập giải thích từng con số trong 60 giây. Quay lại trong việc `w48-c` rồi một lần mỗi tuần cho đến khi đủ 12 tiêu chí. Khoảng 1.5 giờ mỗi tuần.
- **Vòng 4 dưới 24:** xem dòng thấp nhất của thang 30. Kỹ thuật thấp: luyện 4 chuỗi mỗi tuần từ ngân hàng của chương 14, cộng 2 chuỗi trong file người phỏng vấn mà buổi 2 chưa bốc. Làm rõ hoặc "không biết" thấp: luyện bốn bước của mục 14.1 với chuỗi C18. Dự án thấp: chuỗi C15 và P4. Hành vi thấp: rút 6 câu chuyện STAR (mục 14.5) về 2 phút và thêm số. Cách nói thấp: đếm lại bằng Lab 3 sau mỗi buổi. Một buổi phỏng vấn thử với người khác mỗi tuần. Khoảng 3 giờ mỗi tuần.

**Câu sai ở Vòng 1 → ôn gì.** "Bài ★2" nghĩa là bài 2 trong mục Bài tập giấy bút của chương đó; "tính nhanh 3" là bài 3 trong mục Bài tính nhanh.

| Câu | Đọc lại | Làm lại |
| --- | --- | --- |
| A1 | Mục 1.3 (đoạn Thực tế tính toán, κ(AᵀA) = κ(A)²) | Chương 1: Lab 2; bài ★★★14 |
| A2 | Mục 3.3 (đoạn Biên và điều kiện vẫn là Gauss; đoạn Lấy mẫu bằng Cholesky) | Chương 3: bài ★★9 |
| A3 | Mục 4.1, ví dụ 4.3; mục 6.5 (đoạn Theo dõi gradient) | Chương 4: bài ★★5 |
| A4 | Mục 5.2, ví dụ 5.7 | Chương 5: bài ★★9 |
| A5 | Mục 7.5, ví dụ 7.17 | Chương 7: bài ★★★14; tính nhanh 11 |
| A6 | Mục 8.3, ví dụ 8.6 | Chương 8: bài ★★★13; mini-test câu 4 |
| A7 | Mục 9.1 (đoạn đầu); mục 8.5 (đoạn Mặt nạ) | Chương 8: Bài săn bug |
| A8 | Mục 8.2 (đoạn Self-attention và cross-attention); mục 8.5, ví dụ 8.12 | Chương 8: Lab 4 |
| A9 | Mục 9.1, ví dụ 9.1 | Chương 9: bài ★2; Lab 1 |
| A10 | Mục 9.4, ví dụ 9.11 | Chương 9: bài ★★8 |
| A11 | Mục 10.3, ví dụ 10.7 | Chương 10: bài ★3; tính nhanh 7, 8 |
| A12 | Mục 11.2, ví dụ 11.5 | Chương 11: bài ★★★14; viết lại bảng 2³ của ví dụ 11.5 thành bảng "bỏ từng thành phần" rồi so hai bảng |
| A13 | Mục 12.1, ví dụ 12.1 | Chương 12: bài ★1; tính nhanh 1–3 |
| A14 | Mục 12.2, ví dụ 12.4 | Chương 12: bài ★★5; tính nhanh 5 |
| A15 | Mục 12.4, ví dụ 12.8 | Chương 12: bài ★★7, ★★★12; tính nhanh 10 |
| B1 | Mục 9.1, ví dụ 9.3; mục 5.5 (đoạn EM, ví dụ 5.15); mục 3.5 | Chương 5: bài ★★★11; chương 3: bài ★★★11 |
| B2 | Mục 12.2 (đoạn Phân rã lỗi, ví dụ 12.6); mục 12.4, ví dụ 12.9 (d); mục 11.2, ví dụ 11.5 | Chương 12: bài ★4, ★★8, ★★★13; chương 14: bài ★★★13 |

**Đây là đề cuối.** Sách không có đề thi thử thứ tư. Sau đề này là nộp hồ sơ theo lịch **chính thức** của chương trình: kiểm tra trang tuyển dụng của Qualcomm và VinAI như các việc `w46-apply`, `w47-apply` đã làm, và chỉ ghi ngày mở, ngày đóng sau khi thấy công bố thật. Gói hồ sơ, video và repo đã chốt ở tuần 45–46. Từ nay, việc của bạn là giữ phong độ và sửa đúng vòng yếu, không học thêm chủ đề mới.

## Lời giải

Lời giải Vòng 1–2: đọc sau lượt hai của buổi 1. Lời giải Vòng 3: đọc sau khi tự chấm Vòng 3 ở buổi 2. Lời giải Vòng 4 không nằm trong file này: ý đáp án, câu trả lời mẫu và số liệu của bốn chuỗi ở `labs/thi-thu-3/nguoi-phong-van.md`, chỉ mở sau buổi 2. Thang điểm ghi ở cuối mỗi lời giải.

### Vòng 1 — Phần A

**A1.** (a) κ(A) = σ₁/σ₂ = 50/0.05 = 1000. AᵀA có giá trị kỳ dị σ₁², σ₂², nên κ(AᵀA) = κ(A)² = 10⁶. (b) Phương trình chuẩn trong float32: 10⁶·6·10⁻⁸ = 6·10⁻², tức sai khoảng 6%. QR hoặc `lstsq` trong float32: 1000·6·10⁻⁸ = 6·10⁻⁵. Tạo AᵀA đã bình phương số điều kiện trước khi kịp giải, nên phương trình chuẩn mất gấp đôi số chữ số so với QR: khoảng 6 trong 7 chữ số của float32, so với khoảng 3. Script đo bằng float32 thật (QR của PyTorch) trên 20 seed, ma trận đúng như đề: sai số tương đối trung vị khoảng 2·10⁻² (phương trình chuẩn) và 4·10⁻⁶ (QR); tỉ số trung vị cỡ 10³–10⁴ và dao động lớn theo hướng của x (từ khoảng 10 đến khoảng 2·10⁵). Quy tắc κ·u cho bậc độ lớn, không phải dự báo chính xác. Lưu ý: `np.linalg.lstsq` nhận float32 nhưng tính bên trong bằng float64, nên không dùng nó để đo "lstsq trong float32". *Thang:* 1 điểm mỗi số ở (a); (b) 1 điểm hai cỡ sai số, 1 điểm lý do "bình phương số điều kiện".

**A2.** (a) Trung bình −1 + (3/9)(4 − 1) = 0; phương sai 5 − 3·3/9 = 4. Vậy x₂ | x₁ = 4 ~ N(0, 4). (b) L₁₁ = √9 = 3, L₂₁ = 3/3 = 1, L₂₂ = √(5 − 1) = 2, nên L có các hàng (3, 0) và (1, 2). log det Σ = 2(ln 3 + ln 2) = 2 ln 6 ≈ 3.584; kiểm lại: det Σ = 45 − 9 = 36 = 6². L dùng để lấy mẫu: x = μ + Lz với z ~ N(0, I); ví dụ z = (0.5, −1) cho x = (2.5, −2.5) (đề không hỏi). *Thang:* (a) 1 điểm trung bình, 1 điểm phương sai; (b) 1 điểm L, 1 điểm log det.

**A3.** (a) Chuẩn gộp √(4 + 1 + 4 + 16) = 5. Hệ số cắt 1/5, nên g₁ thành (0.4, −0.2), g₂ thành (0.4, 0.8). (b) Cắt theo giá trị: (1, −1) và (1, 1). Tích vô hướng với gradient gốc 2 + 1 + 2 + 4 = 9; chuẩn 5 và 2; cosin = 9/10, tức góc khoảng 25.8°. Cắt theo chuẩn gộp giữ hướng: nó là phép chiếu lên quả cầu ‖g‖₂ ≤ `max_norm` (ví dụ 4.3, điều kiện KKT), chỉ co độ dài. Cắt theo giá trị co từng tọa độ một cách khác nhau nên đổi hướng. *Thang:* (a) 1 điểm chuẩn, 1 điểm hai vector; (b) 1 điểm vector và cosin, 1 điểm "theo chuẩn gộp giữ hướng" kèm lý do.

**A4.** (a) 12 cặp dương–âm. Điểm 0.9 thắng cả 4 mẫu âm; 0.6 thắng 0.35, 0.3, 0.2 (3 cặp); 0.4 thắng 0.35, 0.3, 0.2 (3 cặp). Tổng 10/12 = 5/6 ≈ 0.833. Ở ngưỡng 0.5, dự đoán dương là {0.9, 0.7, 0.6}: TP = 2, FP = 1, FN = 1, nên precision = recall = 2/3. (b) Mỗi cặp dương–âm nhân 10 lần, nên AUC giữ 5/6 (recall cũng giữ 2/3, vì không đụng tới mẫu dương). FP thành 10, nên precision = 2/12 = 1/6. ROC và AUC không phụ thuộc tỉ lệ lớp; precision thì phụ thuộc. Khi lớp dương hiếm, báo precision–recall (hoặc AP) cạnh AUC. *Thang:* (a) 1 điểm AUC, 1 điểm precision và recall; (b) 1 điểm AUC và precision mới, 1 điểm kết luận.

**A5.** (a) 224/14 = 16 nên 16·16 = 256 token; 336/14 = 24 nên 24·24 = 576 token. (b) Ma trận điểm tỉ lệ với số token bình phương: (576/256)² = 2.25² = 5.0625 lần. Các lớp tuyến tính (QKV, FFN) tỉ lệ với số token: 2.25 lần. Bảng vị trí học được có 256 vị trí xếp thành lưới 16 × 16; phải nội suy 2 chiều lên lưới 24 × 24, rồi thường tinh chỉnh thêm ở độ phân giải mới. *Thang:* (a) 1 điểm mỗi số token; (b) 1 điểm hai tỉ số, 1 điểm "nội suy 2D bảng vị trí".

**A6.** (a) Điểm = qᵀR(mθ)ᵀR(nθ)k = qᵀR((n − m)θ)k. Với φ = (n − m)θ, R(φ)k = (−sin φ, cos φ), nên điểm = −2 sin φ. (1, 2) có φ = π/3, cho điểm −2·(√3/2) = −√3 ≈ −1.732. Không quay: q·k = 0. (b) (2, 1) có φ = −π/3, cho +√3. Điểm chỉ phụ thuộc hiệu n − m, nên attention thấy vị trí tương đối, và không cần tính lại: (4, 5) có cùng n − m = 1 với (1, 2), cho −√3. Nó không đối xứng khi đổi chỗ m với n: dấu của hiệu đổi, nên mô hình phân biệt được "trước" với "sau". *Thang:* (a) 1 điểm −√3, 1 điểm số khi không quay; (b) 1 điểm +√3 và (4, 5) cho −√3, 1 điểm "chỉ phụ thuộc n − m, có hướng".

**A7.** (a) Với chuỗi x gồm 513 token, đầu vào là x[0 : 512] (token 0 đến 511) và mục tiêu là x[1 : 513] (token 1 đến 512). Ở vị trí i, mô hình đọc token 0 đến i và dự đoán token i + 1. Mỗi chuỗi có 512 vị trí, batch có 8·512 = 4 096 vị trí dự đoán. (b) Mục tiêu ở vị trí i là token i, mà mặt nạ lại cho vị trí i nhìn thấy chính nó, nên mô hình chỉ cần chép đầu vào ra. Loss train tụt nhanh về gần 0. Lúc sinh, mô hình lặp lại token cuối thay vì dự đoán token kế tiếp, nên văn bản vô nghĩa hoặc lặp một token mãi. Triệu chứng: loss thấp bất thường so với ln của cỡ từ vựng ngay từ đầu. *Thang:* (a) 1 điểm cách cắt, 1 điểm số vị trí; (b) 1 điểm "loss về gần 0 vì chép", 1 điểm hậu quả lúc sinh.

**A8.** (a) dₕ = 512/8 = 64. Q: 25 × 64 (từ decoder); K, V: 40 × 64 (từ encoder); ma trận điểm 25 × 40. Không cần mặt nạ nhân quả: câu nguồn đã có đủ từ đầu, mọi vị trí decoder được nhìn cả câu nguồn. Cần mặt nạ đệm trên các key là token đệm của câu nguồn. (b) K và V của cross-attention chỉ tính **một lần** từ đầu ra encoder rồi dùng lại ở mọi bước giải mã: 2·40·512 = 40 960 giá trị, tức 81 920 byte ở fp16 cho một lớp, không đổi theo số bước. Cache self-attention thì tăng thêm một vị trí sau mỗi bước: sau 25 token là 2·25·512·2 = 51 200 byte. *Thang:* (a) 1 điểm shape, 1 điểm hai câu về mặt nạ; (b) 1 điểm 81 920 byte và "tính một lần", 1 điểm 51 200 byte.

**A9.** (a) Đếm cặp: `t h` 4 + 3 = 7; `h a` 4 + 3 + 5 = 12; `a n` 4 + 3 + 5 = 12; `n h` 4 (cuối `thanh`) + 5 + 5 (đầu và cuối `nhanh`) = 14. Phép gộp đầu chọn `n h` (14). Số token: 5·4 + 4·3 + 5·5 = 57 trước, 4·4 + 4·3 + 3·5 = 43 sau; giảm đúng 14, bằng số lần gộp. (b) Đếm lại: `t h` 7, `h a` 7, `a nh` 4 + 5 = 9, `nh a` 5, `a n` 3. Phép gộp thứ hai chọn `a nh` (9); sau đó kho còn 4·3 + 4·3 + 2·5 = 34 token (đề không hỏi). Cặp `a n` tụt từ 12 xuống 3 vì trong `thanh` và `nhanh`, ký hiệu `n` đã bị gộp vào `nh`. Các cặp chồng nhau tranh cùng một ký hiệu, nên phải đếm lại sau mỗi phép gộp. *Thang:* (a) 1 điểm cặp, 1 điểm 57 và 43; (b) 1 điểm cặp thứ hai, 1 điểm lý do.

**A10.** (a) Hệ số 127/1.27 = 100. Mã: 0.254 → 25.4 → 25; −0.381 → −38.1 → −38; 0.07 → 7; 1.27 → 127. Giải mã: 0.25, −0.38, 0.07, 1.27. Sai số lớn nhất 0.004 (ở 0.254). (b) Hệ số mới 127/12.7 = 10. Mã: 0.254 → 2.54 → 3; −0.381 → −3.81 → −4; 0.07 → 0.7 → 1; 1.27 → 12.7 → 13. 0.07 giải mã thành 1/10 = 0.1, lệch 43%. Sai số lớn nhất của bốn phần tử cũ 0.046 (ở 0.254), gấp khoảng 11.5 lần. Cách sửa: lượng tử theo khối nhỏ (ngoại lệ chỉ làm hỏng khối chứa nó), hoặc tách ngoại lệ ra giữ ở độ chính xác cao. *Thang:* (a) 1 điểm mã, 1 điểm sai số; (b) 1 điểm mã mới và 0.1, 1 điểm sai số lớn nhất kèm một cách sửa.

**A11.** (a) xₜ = 0.8·(−1.2) + 0.6·0.5 = −0.66. SNR = 0.64/0.36 ≈ 1.778. x̂₀ = (xₜ − 0.6·ε̂)/0.8 = (−0.66 − 0.15)/0.8 = −1.0125; lệch 0.1875 = (0.6/0.8)·0.25. (b) √ᾱ = 5/13, √(1 − ᾱ) = 12/13. Sai số của x̂₀ = (12/5)·0.25 = 0.6, gấp 3.2 lần. Tổng quát, sai số x̂₀ = √((1 − ᾱ)/ᾱ)·|ε − ε̂| = |ε − ε̂|/√SNR. Ở bước t lớn, SNR nhỏ (25/144 ≈ 0.174), nên dự đoán x₀ rất nhiễu: một sai số nhỏ ở ε bị khuếch đại thành sai số lớn ở x₀. *Thang:* (a) 1 điểm xₜ và SNR, 1 điểm x̂₀ và sai số; (b) 1 điểm 0.6, 1 điểm hệ số 1/√SNR và hệ quả.

**A12.** (a) A: ((71.0 − 70.0) + (73.0 − 70.5))/2 = 1.75. B: ((70.5 − 70.0) + (73.0 − 71.0))/2 = 1.25. Tương tác: (73.0 − 70.5) − (71.0 − 70.0) = 1.5 (A có lợi hơn 1.5 điểm khi đã có B). (b) Bỏ A khỏi hệ đầy đủ mất 73.0 − 70.5 = 2.5; bỏ B mất 73.0 − 71.0 = 2.0. Cộng lại 4.5, lớn hơn mức tăng toàn phần 3.0, vì phần tương tác 1.5 bị tính cho cả hai. Thêm từng thành phần vào hệ gốc thì cho 1.0 và 0.5, cộng 1.5, nhỏ hơn 3.0. Khi có tương tác, "mỗi thành phần đóng góp bao nhiêu" phụ thuộc thứ tự, nên báo đủ bảng 2² kèm sai số theo seed. *Thang:* (a) 1 điểm hai hiệu ứng chính, 1 điểm tương tác; (b) 1 điểm 2.5 và 2.0, 1 điểm lý do và bảng nên báo.

**A13.** (a) idf = ln(1 + (N − n + 0.5)/(n + 0.5)) = ln((N + 1)/(n + 0.5)). n = 2: ln(50/2.5) = ln 20 = 2 ln 2 + ln 5 ≈ 2.996. n = 49: ln(50/49.5) = ln(100/99) ≈ 0.0101. Từ có mặt ở mọi đoạn gần như không đóng góp. (b) Phần tf = f(k₁ + 1)/(f + k₁(1 − b + b·dl/avgdl)). D₁: 2.2/(1 + 1.2·(0.25 + 0.375)) = 2.2/1.75 ≈ 1.257. D₂: 4.4/(2 + 1.2·(0.25 + 1.5)) = 4.4/4.1 ≈ 1.073. D₁ xếp trên (điểm khoảng 3.77 so với 3.21), dù chỉ chứa từ đó 1 lần: chuẩn hóa độ dài phạt đoạn dài. Với b = 0 không còn chuẩn hóa độ dài: phần tf chỉ còn f(k₁ + 1)/(f + k₁), tăng theo f, nên D₂ (chứa từ đó 2 lần) lên trên; tính ra D₁ 2.2/2.2 = 1, D₂ 4.4/3.2 = 1.375. *Thang:* (a) 1 điểm mỗi idf; (b) 1 điểm hai phần tf và thứ hạng, 1 điểm trường hợp b = 0 có lý do.

**A14.** (a) Công thức max(0, m − 1 − o)/(c − o): o = 0 cho 2/5 = 0.4; o = 1 cho 1/4 = 0.25; o = 2 cho 0. Chồng lấn từ m − 1 = 2 câu trở lên thì không còn cắt ngang. (b) Bước nhảy giảm từ 5 xuống 3 câu, nên số chunk gấp khoảng 5/3 ≈ 1.67 lần. Ba chunk liền nhau chồng 2 câu với nhau chứa 5 + 3 + 3 = 11 câu khác nhau, tức có 4 câu lặp trong ngữ cảnh: tốn token cho nội dung trùng. *Thang:* (a) 1 điểm ba xác suất, 1 điểm kết luận chồng lấn từ m − 1; (b) 1 điểm 5/3, 1 điểm 11 câu.

**A15.** (a) p₀ = (90 + 45 + 20)/200 = 0.775. Biên của người: (100, 70, 30)/200 = (0.5, 0.35, 0.15). Biên của giám khảo: (110, 60, 30)/200 = (0.55, 0.3, 0.15). pₑ = 0.5·0.55 + 0.35·0.3 + 0.15·0.15 = 0.4025. κ = (0.775 − 0.4025)/(1 − 0.4025) ≈ 0.623. (b) Người: 0.50; giám khảo: 0.55. Giám khảo dễ dãi: trong 70 câu người chấm Sai, nó chấm Đúng 15 câu (21%), còn chiều ngược lại chỉ 10/100. Precision của nhãn Đúng của giám khảo là 90/110 ≈ 0.82. Một bảng xếp hạng chấm bằng giám khảo này sẽ thổi độ chính xác lên khoảng 5 điểm, và có thể đổi thứ hạng nếu mức dễ dãi khác nhau giữa các hệ. Phải báo κ, cỡ mẫu, và đối chiếu với nhãn người trên một tập con. *Thang:* (a) 1 điểm p₀ và pₑ, 1 điểm κ; (b) 1 điểm hai tỉ lệ và hướng lệch có số, 1 điểm hệ quả.

### Vòng 1 — Phần B

**B1.** (a) NLL(A) = −¼(ln 0.5 + ln 0.1 + ln 0.4 + ln 0.05) = ¼ ln(1/0.001) = ¾ ln 10 ≈ 1.727 nat. NLL(B) = ¼ ln(1/0.002) = ¼ ln 500 ≈ 1.554. Hỗn hợp có xác suất (0.35, 0.25, 0.25, 0.15). Dùng hai giá trị cho sẵn và −ln 0.25 = 2 ln 2 = 1.386: NLL = (1.050 + 2·1.386 + 1.897)/4 ≈ 1.430. Xếp hạng: hỗn hợp (1.430) tốt hơn B (1.554), B tốt hơn A (1.727). A và B sai ở **những token khác nhau** (A yếu ở token 2 và 4, B yếu ở token 1 và 3), nên trung bình hai xác suất không bao giờ để một token rơi xuống mức rất thấp, và −ln phạt nặng nhất đúng các xác suất rất thấp đó. (Perplexity tương ứng ⁴√1000 ≈ 5.62, ⁴√500 ≈ 4.73 và khoảng 4.18; đề không hỏi.) *Thang:* 3 điểm ba NLL (1 điểm mỗi cái), 1 điểm xếp hạng đúng, 2 điểm giải thích "sai ở token khác nhau, −ln phạt xác suất nhỏ".

(b) (i) Với mỗi t, ln là hàm lõm, nên ln(w·aₜ + (1 − w)·bₜ) ≥ w·ln aₜ + (1 − w)·ln bₜ (Jensen cho hai điểm). Đổi dấu rồi lấy trung bình theo t: NLL(hỗn hợp) ≤ w·NLL(A) + (1 − w)·NLL(B). Lấy e mũ hai vế: perplexity(hỗn hợp) ≤ perplexity(A) mũ w nhân perplexity(B) mũ (1 − w), tức trung bình nhân có trọng số của hai perplexity. Với số ở (a): 1.430 ≤ ½(1.727 + 1.554) = 1.640 (đề không hỏi phép kiểm này). *Thang:* 2 điểm Jensen đúng chiều cho từng t, 1 điểm lấy trung bình ra bất đẳng thức NLL, 1 điểm hệ quả cho perplexity. (ii) ℓ′(w) = ∑ₜ (aₜ − bₜ)/mₜ(w); ℓ″(w) = −∑ₜ (aₜ − bₜ)²/mₜ(w)² ≤ 0, nên ℓ lõm và có cực đại duy nhất trên [0, 1] (hoặc ở biên). ℓ′(0.5) = 0.3/0.35 − 0.3/0.25 + 0.3/0.25 − 0.2/0.15 = 6/7 − 4/3 = −10/21 ≈ −0.476 < 0, nên giảm w. Tối ưu số cho w* ≈ 0.410, NLL ≈ 1.425: chỉ tốt hơn w = 0.5 một chút. *Thang:* 1 điểm ℓ′, 1 điểm ℓ″ và kết luận lõm, 1 điểm ℓ′(0.5) và chiều.

(c) (i) E-step: rₜ = w·aₜ/mₜ(w), là xác suất hậu nghiệm token t "đến từ A". M-step: cực đại ∑ₜ [rₜ ln w + (1 − rₜ) ln(1 − w)], được w mới = (1/4)∑ₜ rₜ. Từ w = 0.5: r = (0.25/0.35, 0.05/0.25, 0.2/0.25, 0.025/0.15) = (5/7, 0.2, 0.8, 1/6) ≈ (0.714, 0.2, 0.8, 0.167), nên w mới ≈ 0.470. w giảm, cùng chiều với ℓ′(0.5) < 0. Lặp tiếp thì hội tụ về w* ≈ 0.410, và log-likelihood không giảm sau mỗi vòng (mục 5.5). *Thang:* 2 điểm hai bước EM đúng công thức, 1 điểm số 0.470, 1 điểm cùng chiều. (ii) Tokenizer: A và B phải chia văn bản thành **cùng** các token. Nếu không, aₜ và bₜ là xác suất của những sự kiện khác nhau, cộng chúng không ra một phân phối, và perplexity theo token không so được (Đề thi thử 2, câu A5). Khi đó phải nội suy hoặc so ở mức xác suất của cả chuỗi, quy ra bits per byte. Tập chọn w: chọn trên dev, báo trên test. Chọn w trên chính tập báo cáo là tinh chỉnh trên test, kết quả lạc quan (chương 11). Với 4 token như ở đây, w* gần như chỉ là nhiễu. *Thang:* 1 điểm điều kiện tokenizer kèm hậu quả, 1 điểm dev/test kèm hậu quả, 1 điểm ý "4 token quá ít" hoặc "quy về mức chuỗi hoặc byte".

**B2.** (a) R = 260/400 = 0.65; a₁ = 208/260 = 0.8; a₀ = 35/140 = 0.25. Acc = 0.65·0.8 + 0.35·0.25 = 0.52 + 0.0875 = 0.6075 (243/400). P1: 0.75·0.8 + 0.25·0.25 = 0.6625. P2: 0.65·0.88 + 0.0875 = 0.6595. ∂Acc/∂R = a₁ − a₀ = 0.55; ∂Acc/∂a₁ = R = 0.65. P1 nhân 0.55 với mức tăng 0.10 (được 0.055); P2 nhân 0.65 với mức tăng 0.08 (được 0.052). Hai tích gần bằng nhau (P1 hơn P2 0.003), nên công thức không chọn hộ; chọn theo chi phí và độ khó. *Thang:* 2 điểm R, a₁, a₀ và Acc; 2 điểm hai phương án; 2 điểm hai đạo hàm và giải thích bằng tích.

(b) (i) Kỳ vọng khi trả lời là p·1 + (1 − p)·(−λ) = p − λ(1 − p); khi từ chối là 0. Trả lời có lợi khi p − λ(1 − p) > 0, tức p > λ/(1 + λ). Với λ = 1 ngưỡng là 0.5. *Thang:* 2 điểm. (ii) Luôn trả lời: (243 − 157)/400 = 0.215. Từ chối đúng ở mọi câu không có đoạn vàng: chỉ trả lời 260 câu, (208 − 52)/400 = 0.39. Khớp với (i): a₁ = 0.8 > 0.5 nên nên trả lời, a₀ = 0.25 < 0.5 nên nên từ chối. *Thang:* 1 điểm mỗi số. (iii) Hệ trả lời 280 câu, độ phủ 280/400 = 0.70. Số câu đúng 243 − 20 − 14 = 209, số câu sai 71. Độ chính xác trên phần được trả lời 209/280 ≈ 0.746. Điểm (209 − 71)/400 = 0.345. Cách nhìn khác: từ chối 100 câu thiếu đoạn vàng bỏ đi 20 đúng và 80 sai (+60 điểm), từ chối 20 câu có đoạn vàng bỏ đi 14 đúng và 6 sai (−8 điểm), nên 86 + 60 − 8 = 138. Với λ = 0, từ chối không bao giờ có lợi: độ chính xác tính trên cả 400 câu tụt từ 0.6075 xuống 209/400 = 0.5225. Cách chấm quyết định hệ có nên biết nói "không biết" hay không (ví dụ giải mẫu 12.9 (d)). *Thang:* 1 điểm độ phủ và độ chính xác trên phần trả lời, 1 điểm 0.345, 1 điểm trường hợp λ = 0.

(c) (i) Chênh kỳ vọng 0.6625 − 0.6595 = 0.003. Sai số chuẩn √(0.15/400) ≈ 0.0194, nên chênh chỉ bằng khoảng 0.155 lần sai số chuẩn (ngưỡng 5% là 1.96·0.0194 ≈ 0.038). Độ mạnh 0.8: n = 0.15·(2.80/0.003)² ≈ 130 667, tức cỡ 1.3·10⁵ câu; chỉ để chênh đó vừa chạm mức 5% cũng đã cần 0.15·(1.96/0.003)² ≈ 64 027 câu. Không bộ dev gán tay nào tách được hai phương án này. Mô phỏng 20 000 lần ở n = 400 cho xác suất phát hiện gần bằng mức α. Chọn theo chi phí, độ trễ hoặc độ ổn định, và báo rằng hai phương án ngang nhau trong phạm vi đo được. *Thang:* 1 điểm 0.155 lần, 2 điểm cỡ mẫu (nhận bậc độ lớn 10⁵), 1 điểm kết luận. (ii) Acc = 0.75·0.88 + 0.25·0.25 = 0.66 + 0.0625 = 0.7225. Hiện tại cộng hai mức tăng riêng: 0.6075 + 0.055 + 0.052 = 0.7145. Phần chênh 0.008 bằng đúng ΔR·Δa₁ = 0.10·0.08. Lý do: Acc = R·a₁ + (1 − R)·a₀ là hàm song tuyến tính của (R, a₁), nên Acc(R + ΔR, a₁ + Δa₁) − Acc(R, a₁) = (a₁ − a₀)·ΔR + R·Δa₁ + ΔR·Δa₁. Số hạng chéo là tương tác của bảng 2², cùng khái niệm với A12 và ví dụ giải mẫu 11.5: reranker đưa thêm đoạn vàng vào ngữ cảnh, và mỗi đoạn vàng thêm vào lại được bộ đọc tốt hơn dùng. Trên 400 câu thì không thấy được: tương tác là hiệu của hai hiệu ghép cặp; nếu mỗi hiệu có tỉ lệ bất đồng cỡ 15% như ở (i) và hai hiệu gần độc lập [giả định], sai số chuẩn của tương tác cỡ √2·0.0194 ≈ 0.027, và 0.008 chỉ khoảng 0.3 lần mức đó. *Thang:* 1 điểm 0.7225 và so với 0.7145; 1 điểm chỉ ra số hạng chéo ΔR·Δa₁ = 0.008; 1 điểm "không thấy được trên 400 câu" có lý do.

### Vòng 2

**Lời giải mẫu** (chỉ có ở đây và trong `work/mock3/grader_tests/ok_loi_giai_mau.py`; thư mục `labs/thi-thu-3` không chứa lời giải Vòng 2):

```python
import bisect
import math
from collections import Counter


class MetricLog:
    def __init__(self):
        self.steps = {}                      # run -> list step, tăng ngặt nhờ cam kết của đề
        self.vals = {}                       # run -> list value, song song với steps

    def record(self, run, step, value):      # O(1) trung bình: append vào cuối danh sách
        self.steps.setdefault(run, []).append(step)
        self.vals.setdefault(run, []).append(value)

    def at(self, run, step):                 # O(log n): đếm số bản ghi có step ≤ truy vấn
        s = self.steps.get(run)
        if not s:
            return None
        i = bisect.bisect_right(s, step)
        return self.vals[run][i - 1] if i > 0 else None


def _ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def corpus_bleu(hyps, refs, max_n=4):
    num, den = [0] * max_n, [0] * max_n
    for h, r in zip(hyps, refs):
        for n in range(1, max_n + 1):
            hc, rc = _ngrams(h, n), _ngrams(r, n)
            num[n - 1] += sum(min(c, rc[g]) for g, c in hc.items())   # cắt theo câu tham chiếu của CHÍNH câu này
            den[n - 1] += max(0, len(h) - n + 1)
    precisions = [num[i] / den[i] if den[i] else 0.0 for i in range(max_n)]
    c, r = sum(map(len, hyps)), sum(map(len, refs))                   # độ dài của cả corpus
    bp = 0.0 if c == 0 else (1.0 if c > r else math.exp(1 - r / c))
    if min(precisions) == 0.0:
        return 0.0, precisions, bp
    bleu = bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return bleu, precisions, bp
```

Bài 1 là biến thể của bài LeetCode 981 (Time Based Key-Value Store), đổi bối cảnh sang sổ ghi chỉ số. Nhờ cam kết step tăng ngặt, danh sách step của mỗi lần chạy luôn đã sắp xếp mà không cần sắp lại, nên `bisect_right` cho số bản ghi có step ≤ truy vấn.

*Thang tự động, 7 điểm mỗi bài* (do `labs/thi-thu-3/cham_coding.py` chấm). Bài 1: truy vấn chặt giữa hai bản ghi và sau bản ghi cuối 2; truy vấn đúng bằng step đã ghi (kể cả step 0) 1; trước bản ghi đầu hoặc lần chạy chưa ghi trả `None` 1; ba lần chạy ghi xen kẽ 1; tốc độ 2 (2·10⁵ lời gọi trong 5 giây: 7·10⁴ lần `record`, 3·10⁴ cặp `record` và `at` xen kẽ, rồi khoảng 7·10⁴ lần `at`; đáp án phải đúng). Bài 2: định dạng 1; cắt số đếm 2; cộng dồn cấp corpus 1; brevity penalty 2; kết hợp thành BLEU 1 (so bleu với chính precisions và bp mà hàm trả về, ở `max_n` = 4 và 2, trên corpus 3 000 câu sinh với seed 17; precision bằng 0, mẫu số bằng 0 và c = 0 phải cho 0.0).

*Thang chấm tay, 3 điểm mỗi bài* (từ bản ghi âm):

- 1 điểm: trước khi gõ, nói cách tiếp cận **và** chạy tay một ví dụ nhỏ, nêu ít nhất hai trường hợp biên. Bài 1: truy vấn trước bản ghi đầu, step bằng nhau, lần chạy chưa ghi. Bài 2: câu ngắn hơn n, precision bằng 0, c > r, câu rỗng.
- 1 điểm: độ phức tạp đúng, có lý do. Bài 1: `record` O(1) trung bình, `at` O(log n) với n là số bản ghi của lần chạy đó, bộ nhớ O(tổng số bản ghi). Bài 2: thời gian O(N·L) với N = `max_n` và L là tổng độ dài corpus, khi coi việc băm một n-gram là O(1); nhận cả O(N²·L) nếu ứng viên tính chi phí băm tuple độ dài n. Bộ nhớ O(N·ℓ) với ℓ là độ dài câu dài nhất, cho các bộ đếm.
- 1 điểm: câu hỏi nối tiếp. Bài 1: khi bản ghi đến không theo thứ tự, chèn vào đúng chỗ bằng `bisect.insort` (`record` thành O(n) vì phải dời phần tử, `at` vẫn O(log n)), hoặc dùng cây cân bằng hay skip list (cả hai thao tác O(log n)), hoặc gom bản ghi rồi sắp xếp trước truy vấn khi đọc nhiều hơn ghi; nói được cách xử lý step trùng. Bài 2: số đếm cắt lấy max trên các câu tham chiếu, tức min(số lần g trong câu máy dịch, max trên k của số lần g trong câu tham chiếu thứ k); r lấy độ dài câu tham chiếu gần độ dài câu máy dịch nhất (hòa độ dài thì lấy câu tham chiếu ngắn hơn), cộng trên corpus; mẫu số của precision không đổi.

**Các chỗ hay sai** (bộ chấm đã được thử với đúng các bài làm này, bảng ở mục Số liệu đã kiểm):

- Bài 1, quét tuyến tính, lùi từng step trong dict, hoặc dựng lại mảng (ví dụ mảng NumPy) sau mỗi lần ghi: đúng kết quả nhưng chậm, mất 2 điểm tốc độ. Cách cuối chỉ chậm khi `record` và `at` xen kẽ.
- Bài 1, dùng `bisect_left` nên chỉ lấy step nhỏ hơn hẳn truy vấn: mất 1 điểm "đúng bằng step".
- Bài 1, chỉ trả về khi step khớp đúng: còn 2 điểm. Lấy bản ghi ngay **sau** truy vấn: còn 1 điểm.
- Bài 1, một danh sách chung cho mọi lần chạy: còn 3 điểm (test tốc độ cũng có một lần chạy ghi xen kẽ). Trả 0.0 thay vì `None`: mất 1 điểm. Trả `None` khi truy vấn sau bản ghi cuối: còn 5. Ném `KeyError` với lần chạy chưa ghi: còn 6.
- Bài 2, không cắt (đếm mọi n-gram của câu máy dịch có mặt trong câu tham chiếu): mất 2 điểm. Cắt bằng số đếm gộp của cả corpus, hoặc lấy trung bình precision của từng câu: mất 1 điểm "corpus". Duyệt n-gram bằng `range(len − n)` (sót n-gram cuối): mất 3 điểm.
- Bài 2, BP đảo chiều exp(1 − c/r), tính BP từng câu rồi lấy trung bình, hoặc quên BP: mất 2 điểm. Lấy trung bình cộng thay trung bình nhân, gọi `math.log(0)` khi có precision bằng 0, hoặc chia cho 0 khi mẫu số bằng 0 hay c = 0: mất 1 điểm.
- Bài 2, hai lỗi khái niệm chạm nhiều thành phần nên bị trừ ở mỗi thành phần: tính precision, BP và BLEU theo từng câu rồi lấy trung bình còn 3; làm trơn cộng 1 ở mọi n còn 3, chỉ từ n ≥ 2 còn 4 (docstring ghi "không làm trơn").

### Vòng 3

Chương 13 không có thang điểm cho video, nên lời giải của vòng này là cách đọc từng tiêu chí trượt (việc sửa nằm ở mục Tổng kết).

- **Tiêu chí 1–2 trượt** (thời lượng): nếu lệch nằm ở đoạn 1 và đoạn 3, thường là mở đầu nói nhanh và phần cách làm dài; ví dụ giải mẫu 13.6 cho thấy trung bình đẹp vẫn che được nhịp lệch. Sửa ngân sách từ trước, không cắt ở lúc quay.
- **Tiêu chí 4 trượt** (nhịp): tổng ngoài vùng 130–150, hoặc đoạn 3, 4 ngoài vùng 120–160. Ví dụ giải mẫu 13.6 cho thấy kiểu lỗi hay gặp: trung bình 130 trông ổn nhưng mở đầu 172 và cuối 106–112 từ/phút. Đoạn ngắn không chấm riêng, nhưng nếu đoạn 1 nhanh hơn hẳn 150 thì vẫn sửa: đó thường là chỗ hồi hộp.
- **Tiêu chí 7–10 trượt** (nội dung): một video có ba con số mà không có baseline là "danh sách chỉ số", đúng điều mục 13.5 dặn tránh. Chọn một con số trung tâm, giữ nó khớp ở CV, statement và video; nếu không giải thích nổi một con số trong 60 giây thì bỏ con số đó khỏi video (ví dụ giải mẫu 13.8).
- **Tiêu chí 12 trượt:** câu kiểu "Qualcomm requires…" hay "the program looks for…" mà không có nguồn. Thay bằng lý do của chính bạn (mini-test câu 6 chương 13).

### Vòng 4

Ý đáp án, câu trả lời mẫu tầng 1–2 và số liệu đã kiểm của bốn chuỗi nằm trong `labs/thi-thu-3/nguoi-phong-van.md`, không nằm trong file này, để người làm đề không đọc thấy trước buổi 2. Sau buổi 2: đối chiếu phiếu chấm với mục Ý đáp án của file đó, đọc câu trả lời mẫu, rồi luyện lại hai chuỗi người phỏng vấn chưa bốc.

## Số liệu đã kiểm

Mọi số trong đề và lời giải được tính lại bằng script, kiểm lần đầu ngày 2026-11-17 và kiểm lại sau phản biện ngày 2026-11-19, với Python 3.13, NumPy 2.5, SciPy 1.18, scikit-learn, PyTorch 2.14 (CPU, float64). PyTorch và scikit-learn chỉ dùng trong script kiểm của người soạn. Bộ chấm của người học chỉ cần thư viện chuẩn của Python.

- `work/mock3/check_numbers.py`: bảng hằng số, Vòng 1 (A1–A15, B1, B2), các con số của bốn chuỗi Vòng 4 (so với file người phỏng vấn), ngân sách giây của Vòng 3, lịch hai buổi và các ngày trong lịch. Đạo hàm kiểm bằng autograd; các ý mô phỏng có ghi seed.
- `work/mock3/check_coding.py`: Vòng 2. Chạy bộ chấm trên lời giải mẫu (7 + 7) và trên khung trống (0, không lỗi), rồi trên 31 bài làm cố ý trong `work/mock3/grader_tests/`: 8 bài đúng viết khác cách, 23 bài sai cố ý, trong đó có 3 bài đúng kết quả nhưng quá chậm. Script cũng kiểm: bộ chấm chỉ import thư viện chuẩn và không chứa cách giải; test tốc độ gọi đúng 2·10⁵ lần như docstring; file có BOM vẫn chấm đúng và file không phải UTF-8 không làm bộ chấm văng; khung `bai_lam_mau.py` không gợi ý thuật toán; các đáp số tính tay của bộ chấm khớp một cài đặt BLEU độc lập.
- `work/mock3/check_exam.py`: rà cấu trúc đề: tổng điểm và tổng phút của từng vòng và của cả ngày, bảng chấm theo chương, ký hiệu (không `^`, `_`, `$` hay lệnh LaTeX ngoài khối công thức và code), mọi chỗ trỏ tới chương 13 và 14 (mục, ví dụ, bài, lab, thang) có thật, mọi chỗ trỏ trong bảng Câu sai → ôn gì có thật và đúng nhóm sao. Script quét lộ đáp án bằng `work/mock3/scan_leaks.py`: phần đọc trước khi bấm giờ (Cách làm, Bản đồ đề) không chứa đáp số nào của Lời giải; đề Vòng 2 không chứa lời giải; file đề không chứa câu hỏi, ý đáp án, câu trả lời mẫu hay con số đặc trưng nào của Vòng 4 (so với file người phỏng vấn). Script quét trùng câu chữ: 8-gram theo từ giữa đề Vòng 1 và chương 1–14 cho 1 cụm 8 từ trùng ("số câu là cặp bất đồng, đúng ở…", cụm định nghĩa của bài ★★★13 chương 12); 6-gram giữa câu hỏi tiếng Anh của bốn chuỗi mới (trong file người phỏng vấn) và chương 0–14 cho 0 cụm. Script quét trùng bộ số của 17 câu Vòng 1 với toàn văn chương 0–14 (kể cả bài chẩn đoán). Cuối cùng, nó kiểm khái niệm trung tâm của mỗi chuỗi mới chưa có trong tiêu đề hay câu hỏi của chuỗi nào ở chương 1–14.

**Trùng khuôn có chủ đích** (bộ số khác, cách hỏi gần sách): A2 cùng khuôn bài ★★9 chương 3 (bản nháp đầu dùng đúng Σ của bài đó, đã đổi sang hàng (9, 3), (3, 5)); A3 dùng lại ý phép chiếu của ví dụ 4.3; A4 cùng khuôn ví dụ 5.7 và bài ★★9 chương 5; A5 cùng chủ đề với bài ★★★14 và tính nhanh 11 chương 7 (patch 16, ảnh 224 và 384) nhưng patch 14 và ảnh 336; A6 cùng khuôn ví dụ 8.6 và mini-test câu 4 chương 8; A9 cùng thủ tục ví dụ 9.1; A10 cùng khuôn ví dụ 9.11; A11 cùng công thức với bài ★3 chương 10 và câu 22 của bài chẩn đoán chương 0 (bản nháp đầu dùng đúng x₀ = 1.5, ε = −0.5 của câu 22, đã đổi sang x₀ = −1.2, ε = 0.5, ᾱ = 0.64); A13, A14, A15 cùng khuôn ví dụ 12.1, 12.4, 12.8; B2(a) cùng công thức ví dụ 12.6. B2(b)(i) trùng ví dụ giải mẫu 12.9 (d) và bài ★★8 chương 12 (ngưỡng λ/(1 + λ)), nên ý này chỉ 2 điểm và đo trí nhớ nhiều hơn suy luận. B2(c)(i) cùng xấp xỉ √(d/n) với bài ★★★13 chương 12, và cùng khuôn bài ★★★13 chương 14 cùng câu hỏi "one-point gain with 5% disagreement" của chương 14. B2(c)(ii) bản đầu trùng ý ví dụ giải mẫu 12.6 (câu đúng mà không có đoạn vàng là cận trên) nên đã được thay bằng ý tương tác của hai phương án; ý mới cùng khái niệm tương tác với A12 và ví dụ giải mẫu 11.5, nhưng chương 12 chưa có. Phần "Giao với sách" của bốn chuỗi Vòng 4 nằm trong file người phỏng vấn.

Tái lập: từ thư mục gốc repo, chạy bốn script trên bằng `/home/claude/venv/bin/python`; mỗi script in "ĐẠT" ở dòng cuối. `check_coding.py` mất khoảng 3–4 phút vì có bài làm quá chậm bị chạy tới giới hạn thời gian.

| Câu | Giá trị | Cách kiểm | Kết quả |
| --- | --- | --- | --- |
| Hằng số | ln 2, ln 3, ln 5, ln 7, √2, √3, 1.96, 2.80; u của float32 = 5.96·10⁻⁸ | `math`, `scipy.stats`, `np.finfo` | khớp |
| A1 | 1000, 10⁶; 6·10⁻², 6·10⁻⁵; đo: trung vị 2·10⁻² và 4·10⁻⁶, tỉ số cỡ 10³–10⁴ | quy tắc κ·u; đo float32 thật (QR của PyTorch) trên 20 seed, ma trận 1000 × 2 đúng hai giá trị kỳ dị | khớp |
| A2 | N(0, 4); L có các hàng (3, 0), (1, 2); mẫu (2.5, −2.5); log det 2 ln 6 = 3.584 | `cholesky`, `det`; hiệp phương sai của 200 000 mẫu (seed 1) | khớp |
| A3 | chuẩn 5; (0.4, −0.2), (0.4, 0.8); (1, −1), (1, 1); cosin 9/10, 25.8° | `clip_grad_norm_`, `clip_grad_value_` | khớp |
| A4 | AUC 5/6; precision, recall 2/3; nhân âm 10 lần: AUC 5/6, precision 1/6 | đếm cặp, `roc_auc_score` | khớp |
| A5 | 256, 576 token; 5.0625 và 2.25 lần | `nn.Conv2d` cho lưới 16 × 16 và 24 × 24 | khớp |
| A6 | −√3, 0; +√3; (4, 5) cho −√3 | ma trận quay | khớp |
| A7 | 4 096 vị trí | tính trực tiếp | khớp |
| A8 | 25 × 40; 81 920 byte; 51 200 byte | `nn.MultiheadAttention` cho trọng số (1, 8, 25, 40) | khớp |
| A9 | cặp 7, 12, 12, 14; 57 → 43 → 34 token | cài BPE theo định nghĩa | khớp |
| A10 | mã (25, −38, 7, 127), sai số 0.004; có 12.7: (3, −4, 1, 13), 0.1, 43%, 0.046, gấp 11.5 lần | NumPy | khớp |
| A11 | −0.66; 1.778; −1.0125, 0.1875; 0.6, 3.2 lần; 1/√SNR | tính trực tiếp | khớp |
| A12 | 1.75, 1.25, 1.5; 2.5, 2.0 | tính trực tiếp | khớp |
| A13 | ln 20 = 2.9957; ln(100/99) = 0.0101; tf 1.2571 và 1.0732; b = 0: 1 và 1.375 | công thức mục 12.1 | khớp |
| A14 | 0.4, 0.25, 0; 5/3; 11 câu | công thức; đếm trên tài liệu dài 6 000 câu | khớp |
| A15 | p₀ 0.775, pₑ 0.4025, κ 0.6234; 0.818; 15/70 | `cohen_kappa_score` | khớp |
| B1 | NLL 1.7269, 1.5537, 1.4299; −ln 0.35 = 1.0498 và −ln 0.15 = 1.8971 (cho sẵn 1.050 và 1.897); perplexity 5.62, 4.73, 4.18; cận 1.6403; ℓ′(0.5) = −10/21; EM 0.4702; w* 0.410, NLL 1.4246 | NumPy, autograd cho ℓ′, `minimize_scalar` và EM 500 vòng hội tụ cùng chỗ; tính lại bằng bảng hằng số lệch dưới 2·10⁻⁴ | khớp |
| B2 | 0.65, 0.8, 0.25; 0.6075; 0.6625, 0.6595; 0.215, 0.39; 209, 71, 0.70, 0.746, 0.345; 0.5225; SE 0.0194; 0.155; 64 027 và 130 667; kết hợp 0.7225, cộng riêng 0.7145, tương tác 0.008, SE tương tác 0.027 | tính trực tiếp; độ mạnh ở n = 400 mô phỏng 20 000 lần (seed 11) dưới 0.07 | khớp |
| Vòng 3 | 20, 25, 45, 45, 25, 20 cộng 180; mốc 0:20 … 3:00; 47, 58, 105, 105, 58, 47 từ | cộng dồn; 140 từ/phút | khớp |
| Lịch | 30/9/2027 Thứ Năm, 2/10/2027 Thứ Bảy, 3/10/2027 Chủ nhật; 39 ngày từ 4/10 đến 12/11/2027; hai buổi 250 + 115 = 365 phút; tuần 48: 12.5 − 4 + 365/60 ≈ 14.6 giờ | `datetime`, `plan/weeks.yaml` | khớp |

**Bộ chấm bám thang điểm.** Mỗi bài làm cố ý chỉ viết một bài (bài kia để `NotImplementedError`), trừ lời giải mẫu.

| Bài làm | Thang dự kiến | Bộ chấm cho |
| --- | --- | --- |
| Lời giải mẫu (cả hai bài) | 7 và 7 | 7 và 7 |
| Khung trống | 0 và 0 | 0 và 0, không lỗi |
| Bài 1 đúng, viết khác cách (hai danh sách với `bisect_right`; tìm kiếm nhị phân tự viết; `bisect` trên tuple với (step, inf); `bisect` có tham số `key`) | 7 | 7, 7, 7, 7 |
| Bài 1 quét tuyến tính / lùi từng step trong dict (đúng nhưng chậm) | 5 (mất tốc độ) | 5, 5 |
| Bài 1 dùng `bisect_left` (bỏ step bằng) | 6 | 6 |
| Bài 1 chỉ trả khi khớp đúng step | 2 | 2 |
| Bài 1 một danh sách chung cho mọi lần chạy | 3 | 3 |
| Bài 1 trả 0.0 thay vì `None` | 6 | 6 |
| Bài 1 trả `None` khi truy vấn sau bản ghi cuối | 5 | 5 |
| Bài 1 `KeyError` với lần chạy chưa ghi | 6 | 6 |
| Bài 1 dựng lại mảng NumPy sau mỗi lần ghi (đúng, chỉ chậm khi xen kẽ) | 5 (mất tốc độ) | 5 |
| Bài 1 lấy bản ghi ngay sau truy vấn | 1 | 1 |
| Bài 2 đúng, viết khác cách (`Counter` và min; phép giao `Counter`; dict tự đếm; NumPy cộng dồn và `math.fsum`) | 7 | 7, 7, 7, 7 |
| Bài 2 không cắt | 5 | 5 |
| Bài 2 cắt bằng số đếm gộp cả corpus / trung bình precision từng câu | 6 | 6, 6 |
| Bài 2 BP đảo chiều / BP từng câu rồi lấy trung bình / quên BP (bp luôn 1) | 5 | 5, 5, 5 |
| Bài 2 trung bình cộng / `math.log(0)` khi có precision bằng 0 | 6 | 6, 6 |
| Bài 2 sót n-gram cuối mỗi câu | 4 | 4 |
| Bài 2 chia cho 0 khi mẫu số bằng 0 hoặc c = 0 | 6 | 6 |
| Bài 2 tính precision, BP và BLEU theo từng câu rồi lấy trung bình | 3 | 3 |
| Bài 2 làm trơn cộng 1 ở mọi n / chỉ từ n ≥ 2 | 3 / 4 | 3, 4 |
| Lời giải mẫu lưu kèm BOM (Notepad) | 7 và 7 | 7 và 7 |

Trên máy kiểm, lời giải mẫu chạy test tốc độ hết khoảng 0.1 giây, dưới giới hạn 5 giây khoảng 40 lần. Bản lùi từng step hết khoảng 14 giây; bản quét tuyến tính và bản dựng lại mảng NumPy bị dừng ở giới hạn 20 giây của tiến trình. **Sửa sau phản biện vòng 1:** (1) test "nhiều lần chạy" bỏ ba truy vấn sau bản ghi cuối, nên lỗi trả `None` sau bản ghi cuối chỉ mất 2 điểm ở test "giữa" (trước đó mất 3); (2) test tốc độ thêm pha 3·10⁴ cặp `record`/`at` xen kẽ và giữ đúng 2·10⁵ lời gọi (trước đó 200 100 lời gọi, ghi hết rồi mới hỏi); (3) test "kết hợp" thêm hai mệnh đề của docstring trước đó chưa test (mẫu số 0, c = 0; với c = 0 chỉ so bleu, để bài quên BP không bị trừ lần hai); (4) bộ chấm đọc file bằng `utf-8-sig` và không văng với file không phải UTF-8. Bài làm cố ý của bản trước không đổi điểm. Trước khi chốt, test "giữa" có một truy vấn ở step 10⁹ làm bản lùi từng step chạy quá giờ và mất thêm 2 điểm đúng/sai. Truy vấn đó đã được đổi thành step 1000 để lỗi chậm chỉ bị trừ ở test tốc độ. Test định dạng của Bài 2 ban đầu đòi bp ≤ 1, nên bản BP đảo chiều bị trừ hai lần; nay test đó chỉ kiểm kiểu và tính hữu hạn.

**Ước lượng thời gian** [ước lượng của người phản biện cho bản đã cắt, cộng phần cắt thêm do người soạn ước lượng; chưa đo trên người thật]. "Người học" là người vừa học xong chương 0–14 và làm đề lần đầu; "giải nhanh" là người vừa ôn kỹ. Các chỗ cắt so với bản đầu, không đổi điểm: A10 đổi sang số dễ tính tay (hệ số 100 và 10); A2(b) bỏ phép tính mẫu; A5 bỏ đếm tham số; A9(b) bỏ đếm token sau phép gộp hai; B2(b)(ii) cho sẵn 243 đúng, 157 sai; B2(c)(i) chỉ hỏi một cỡ mẫu và nhận bậc độ lớn; B1(a) bỏ perplexity; B1(b)(i) bỏ phép kiểm bằng số (theo phản biện, khoảng 8 phút). Cắt thêm: B1 cho sẵn −ln 0.35 và −ln 0.15; B2(c)(ii) thay bằng ý ngắn hơn; A6 tính một cặp vị trí ít hơn; A13 hỏi trường hợp b = 0 bằng lý lẽ; A4 bỏ recall (khoảng 4 phút).

| Phần | Người học | Giải nhanh | Ghi chú |
| --- | --- | --- | --- |
| Vòng 1, đọc đề | 2 | 2 | |
| Vòng 1, A1 và A7 | 2 mỗi câu (4) | 1 mỗi câu (2) | |
| Vòng 1, A5, A12, A14 | 2.5 mỗi câu (7.5) | 1.5 mỗi câu (4.5) | |
| Vòng 1, A2, A3, A4, A6, A8, A10, A13 | 3 mỗi câu (21) | 2 mỗi câu (14) | |
| Vòng 1, A9 và A11 | 3.5 mỗi câu (7) | 2 mỗi câu (4) | A9 phải đếm lại cặp sau phép gộp |
| Vòng 1, A15 | 4 | 2.5 | ba biên; κ để dạng phân số được |
| Vòng 1, B1 | 16 ((a) 4, (b) 6, (c) 6) | 11 | hai giá trị ln cho sẵn |
| Vòng 1, B2 | 16.5 ((a) 5, (b) 6, (c) 5.5) | 11.5 | |
| **Vòng 1** | **78** | **51.5** | giờ thi 75 |
| Vòng 2 | 70 (Bài 1 30, Bài 2 35, đọc đề 3, lưu 2) | 45 | giờ thi 70; người học dùng hết giờ |
| Vòng 3 | 30 | 30 | theo bảng giờ; chấm 12 tiêu chí 12 phút |
| Vòng 4 | 45 | 45 | theo khung |
| Lượt hai và chấm | buổi 1: 85; buổi 2: 35 | | theo lịch hai buổi |

**Cần bạn xác nhận.**

1. Cơ cấu bốn vòng, số phút, loại câu và các thang (100, 20, 12 tiêu chí, 30) là lựa chọn của người soạn. Định dạng thật của cả bốn vòng không công khai. Nếu đợt 2027 công bố gì về định dạng, đề này phải được đọc lại theo đó.
2. Cả đề cần khoảng 365 phút kể cả lượt hai và chấm (khoảng 6.1 giờ), vượt 4 giờ của `w48-run`. Đề chia hai buổi (250 và 115 phút), làm tuần 48 thành khoảng 14.6 giờ, hoặc 14.1 giờ nếu bản quay ở Vòng 3 được tính là lần quay đầu của `w48-c`. Sửa `plan/weeks.yaml` cho khớp thuộc bước merge. Nếu muốn giữ đúng 4 giờ thì phải bỏ lượt hai và dời phần chấm sang tuần sau.
3. Chương 13 không có thang điểm cho video. Vòng 3 vì thế ghi số tiêu chí đạt trên 12, mỗi tiêu chí trỏ tới một chỗ có thật của chương 13, và không có ngưỡng đạt. Nếu muốn có điểm, có thể dùng thêm hai dòng "Đào sâu dự án" và "Cách nói" của thang 30 điểm (chương 14 đã dùng chúng cho khung trình bày 5 phút ở mục 14.4), nhưng đó là thang của chương 14, không phải của chương 13.
4. Vòng 4 dùng thang 30 điểm của mục 14.5 với một điều chỉnh của đề: dòng Hành vi chấm một câu chuyện, mỗi tiêu chí 1 điểm thay vì hai câu chuyện mỗi tiêu chí 0.5. Bốn chuỗi mới nằm trong file người phỏng vấn; người phỏng vấn chỉ bốc 2 để khớp dòng "Kỹ thuật (2 chuỗi)"; 2 chuỗi còn lại để luyện sau.
5. Mốc 15 và 10 điểm của Vòng 2 và các số giờ ôn ở mục Tổng kết là heuristic của sách. Thời điểm của các vòng sau Application chưa công khai, nên "khoảng 5.6 tuần" chỉ đúng nếu hạn nộp giả định 12/11/2027 đúng.
6. Đề chưa được đăng ký trong `book/chapters.yaml` và `book/_toc.yml` (phiên này không được sửa hai file đó), nên app chưa có trang "Sách → Đề thi thử 3". Việc này thuộc bước merge. File người phỏng vấn `labs/thi-thu-3/nguoi-phong-van.md` cố ý không đăng ký vào app, để người làm đề không lướt thấy nó.
7. Ước lượng 78 phút của Vòng 1 vẫn vượt giờ thi khoảng 4%. Nếu đo thật vẫn vượt nhiều, chỗ nên cắt tiếp là B2(b)(iii).
