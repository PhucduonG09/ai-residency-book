# Đề thi thử 2 — Phần II (chương 7–11)

## Cách làm

Đề này làm vào cuối tuần 24 (khoảng 18/4/2027), sau khi bạn học xong chương 7–11. Nó mô phỏng một vòng **entrance test** theo hình dung của người soạn [giả định; định dạng đề thật không công khai]: một buổi bấm giờ, deep learning và NLP ở mức đề thi đầu vào, có phần lập trình. Khoảng 13 điểm nhắc lại Phần I (chương 1–6), như Đề thi thử 1 đã hứa. Mục đích là đo lại Phần II dưới áp lực thời gian, rồi quyết định ôn gì trước khi sang Phần III (chương 12, tuần 25).

**Lưu ý trung thực.** Đề entrance test của Qualcomm AI Residency và VinAI không công khai (xem `huong-dan.md`, dòng "Lưu ý trung thực"). Đây là **đề mô phỏng** do sách soạn, bám nội dung chương 7–11 và các chủ đề chung của các khóa deep learning và NLP (Stanford CS231n, CS224N ở mức chủ đề, không chép bài). Cơ cấu điểm, thời lượng, thứ tự các phần, số phút từng câu và mọi ngưỡng điểm là lựa chọn hoặc ước lượng của người soạn, chưa hiệu chỉnh trên người làm thật.

**Quy tắc.**

- 120 phút liền, bấm giờ, bàn học, giấy nháp, điện thoại để xa. Không tài liệu, không máy tính bỏ túi. Bảng hằng số ở đầu phần Đề bài là thứ duy nhất được tra.
- **Thứ tự làm: Phần A → Phần C → Phần B**, như Đề thi thử 1. Phần C (lập trình) đặt giữa đề để điểm lập trình được đo trong giờ. Phần B làm cuối, trên giấy.
- **Phần C được dùng máy tính**: chỉ trình thông dịch Python với NumPy, không internet, không tài liệu, không trợ lý AI, không tự hoàn thành code. Trước khi bấm giờ, chép `labs/thi-thu-2/bai_lam_mau.py` thành `bai_lam.py` ở thư mục gốc của sách; file này chỉ có chữ ký hai hàm và docstring. Trong giờ, điền hai hàm vào đó và được chạy thử với dữ liệu tự nghĩ ra. Riêng câu C2(b) được dùng Python để tính số. **Chỉ chạy bộ chấm `labs/thi-thu-2/cham_phan_c.py` sau khi hết giờ, và đừng mở file đó trước**: nó không chứa lời giải nhưng có dữ liệu test và vài đáp số tính tay.
- Đáp số được để dạng biểu thức (kiểu √3·10⁸, ln 21, 5/12; đây chỉ là ví dụ dạng, không trùng đáp số câu nào) và vẫn đủ điểm. Đáp số thập phân cần đúng đến 2 chữ số có nghĩa, trừ khi đề nói khác (khác Đề thi thử 1, vốn đòi 2 chữ số thập phân: với các số rất nhỏ, chữ số có nghĩa mới là thứ cần đúng); số đếm lớn (tham số, byte) viết dạng tích chưa nhân ra vẫn đủ điểm nếu biểu thức đúng.
- **Mốc giờ gợi ý** [lựa chọn của người soạn]: đọc đề 2 phút; Phần A đến phút 47; Phần C đến phút 73; Phần B từ phút 73 đến 120. Cột Phút ở Bản đồ câu hỏi là ước lượng cho từng câu; câu nào vượt số phút ở đó cộng 2 phút thì ghi "bỏ" và làm tiếp. Hết mốc của một phần mà chưa xong thì vẫn sang phần sau, câu còn lại để lượt hai.
- **Hai con số điểm.** Lượt một là 120 phút. Hết giờ: gạch ngang chỗ đang dừng, đổi bút khác màu, làm tiếp tối đa 30 phút (lượt hai); câu nào vẫn trống thì làm nốt không tính giờ (lượt ba, bút màu thứ ba). Phần C làm trên máy thì lưu bản lượt một thành `bai_lam_luot1.py` trước khi sửa tiếp. Ghi **điểm trong giờ** (chỉ phần làm ở lượt một) và **điểm cuối** (mọi lượt). Bảng chấm theo chương dùng **điểm cuối** (đo kiến thức); điểm trong giờ đo tốc độ, đọc ở mục Đọc kết quả.

**Thời gian có đủ không.** Bản đầu của đề có hai ước lượng độc lập cho người học mục tiêu (sinh viên năm 3 vừa học xong chương 7–11, làm đề lần đầu): 131 phút của người soạn và 145 phút của người phản biện. Đề đã được cắt bớt các bước tính lặp (không đổi điểm; danh sách ở mục Số liệu đã kiểm). Theo thang ước lượng của người phản biện, người học mục tiêu nay cần khoảng **133 phút**; một người giải nhanh, vừa ôn xong Phần II, cần khoảng **88 phút** [ước lượng, chưa đo trên người thật; sai số có thể ±15 phút]. Như vậy đề dài hơn 120 phút khoảng 13 phút với người học mục tiêu, bằng Đề thi thử 1, và phần thiếu thường rơi vào bài B cuối. Độ dài này là lựa chọn của người soạn, để tập bỏ câu dưới áp lực giờ [không có nguồn về độ dài đề thật]. Ngưỡng để coi chênh lệch giữa hai con số điểm là vấn đề tốc độ ghi ở mục Đọc kết quả.

**Cách chấm.** Tổng 100 điểm. Phần A 20 câu × 2 điểm; Phần C 2 bài × 10 điểm; Phần B 5 bài × 8 điểm. Lời giải ghi rõ ý nào đáng bao nhiêu điểm. Chỉ cho điểm nguyên. Đáp số đúng mà thiếu lý do ở ý đề yêu cầu giải thích thì mất điểm của ý giải thích. Phần C: chạy `python labs/thi-thu-2/cham_phan_c.py --student bai_lam.py` từ thư mục gốc của sách để chấm phần tự động (16 điểm, chỉ cần NumPy), rồi chấm tay 4 điểm còn lại theo lời giải.

**Gắn nhãn lỗi** cho mỗi ý mất điểm, như Đề thi thử 1: **K** (khái niệm), **T** (tính sai), **L** (đọc đề sai, thiếu thuật ngữ), **H** (chỉ làm được ở lượt hai hoặc ba). Ghi vào sổ lỗi.

**Độ khó.** ★ hiểu định nghĩa, ★★ mức đề thi đầu vào, ★★★ câu phân loại (mức phỏng vấn nghiên cứu). Các ý ★★★ cộng lại 12 điểm.

## Bản đồ câu hỏi

Thứ tự trong bảng là thứ tự làm bài. Cột Phút là ước lượng cho người học mục tiêu theo thang của người phản biện, sau khi cắt (cộng ra 133 phút). "Phần I" là phần nhắc lại chương 1–6.

| Câu | Chương, mục | Kỹ năng | Độ khó | Điểm | Phút [ước lượng] |
| --- | --- | --- | --- | --- | --- |
| A1 | Chương 7, mục 7.1 | Kích thước, tham số, MAC của một conv có stride | ★ | 2 | 2.5 |
| A2 | Chương 7, mục 7.2 | Max-pool và phép dịch: equivariance theo bội của stride | ★★ | 2 | 2.5 |
| A3 | Chương 7, mục 7.4 | BatchNorm2d: trục thống kê, gộp BN vào conv | ★★ | 2 | 2.5 |
| A4 | Chương 9, mục 9.1 | Nhiệt độ và top-p, hai token cùng logit | ★★ | 2 | 2.5 |
| A5 | Chương 9, mục 9.1 | Perplexity theo token so với bits-per-byte, NLL đo bằng bit | ★★ | 2 | 2.5 |
| A6 | Chương 9, mục 9.2 | 6ND và tỉ lệ token mỗi tham số | ★★ | 2 | 3 |
| A7 | Chương 9, mục 9.3 | Advantage nhóm của GRPO, nhóm không có tín hiệu | ★★ | 2 | 3 |
| A8 | Chương 9, mục 9.4 | Đếm LoRA, bộ nhớ 16Ψ | ★ | 2 | 2 |
| A9 | Chương 9, mục 9.5 | pass@k không chệch và ngây thơ | ★★ | 2 | 2.5 |
| A10 | Chương 9, mục 9.3 | Nghiệm đóng của bài toán có phạt KL, phần thưởng ngầm của DPO | ★★ | 2 | 2.5 |
| A11 | Chương 10, mục 10.1 | KL Gauss chéo, logvar và reparameterization | ★ | 2 | 2.5 |
| A12 | Chương 10, mục 10.4 | InfoNCE: sụp đổ, cận thông tin tương hỗ | ★★ | 2 | 2 |
| A13 | Chương 10, mục 10.5 | FID một chiều, thiên lệch theo cỡ mẫu | ★★ | 2 | 2.5 |
| A14 | Chương 11, mục 11.1 | Suy cỡ tập test từ phần trăm báo cáo | ★ | 2 | 3 |
| A15 | Chương 11, mục 11.3 | Best-of-k là thiên lệch | ★★ | 2 | 2 |
| A16 | Chương 11, mục 11.4 | Nhận ra rò rỉ theo nhóm | ★ | 2 | 1 |
| A17 | Phần I, chương 1, mục 1.5 | PCA từ ma trận hiệp phương sai | ★★ | 2 | 3 |
| A18 | Phần I, chương 2–3, mục 2.5 và 3.5A | Label smoothing: gradient p − mục tiêu, logit tối ưu hữu hạn | ★★ | 2 | 3 |
| A19 | Phần I, chương 4, mục 4.3 | Nhiễu minibatch, quy tắc scale tuyến tính | ★ | 2 | 2 |
| A20 | Phần I, chương 6, mục 6.4 | Inverted dropout | ★ | 2 | 1.5 |
| C1 | Chương 8, mục 8.2 và 8.5 | Causal self-attention một head, ổn định số, KV cache | ★★ đến ★★★ | 10 | 15 |
| C2 | Chương 11, mục 11.3; Phần I, chương 3, mục 3.5B | Paired bootstrap CI cho hiệu accuracy (làm lại Lab 5 chương 3 dưới áp lực giờ, không xem code), McNemar | ★★ | 10 | 13 |
| B1 | Chương 8, mục 8.2; chương 2, mục 2.5 | Gradient qua softmax-attention một query, bão hòa | ★★ đến ★★★ | 8 | 12.5 |
| B2 | Chương 8, mục 8.4–8.5; chương 9, mục 9.2 | Đếm tham số khối có GQA và SwiGLU, FLOPs, KV cache | ★★ đến ★★★ | 8 | 11 |
| B3 | Chương 10, mục 10.1 | ELBO dạng đóng, q tối ưu là posterior, β-VAE | ★★ đến ★★★ | 8 | 11.5 |
| B4 | Chương 7, mục 7.1–7.2 | Backward của conv 2D, receptive field, conv chuyển vị | ★★ đến ★★★ | 8 | 10.5 |
| B5 | Chương 11, mục 11.3 | Seed ghép cặp, CI t, số seed theo độ mạnh, đa kiểm định | ★★ đến ★★★ | 8 | 9.5 |
| | | Đọc đề | | | 2 |
| **Tổng** | | | | **100** | **133** |

Phút theo phần [ước lượng]: A 48, C 28, B 55, đọc đề 2. Mốc giờ trong 120 phút (giữ như Đề thi thử 1): đọc đề 2, A 45, C 26, B 47.

## Đề bài

**Bảng hằng số (được tra).** ln 2 = 0.6931; ln 3 = 1.0986; e = 2.718; e⁻¹ = 0.3679; e² = 7.389; e⁻² = 0.1353; e¹·¹ = 3.004; 2¹·¹ = 2.144; e⁶ = 403.4; e⁷ = 1096.6; e⁸ = 2981; σ(2) = 0.8808, nên σ(−2) = 0.1192; σ(6) = 0.99753, nên σ(−6) = 0.002473; √2 = 1.4142; √5 = 2.2361; √6 = 2.4495; √7 = 2.6458. Hệ số 95% hai phía: phân phối chuẩn 1.96; phân phối t với 5 bậc tự do 2.571, với 10 bậc tự do 2.228. Mức 5% hai phía và độ mạnh 0.8: tổng z bằng 2.80. Ngưỡng χ² một bậc tự do ở mức 5%: 3.84. Kỳ vọng của max của 10 biến N(0, 1) độc lập: 1.539. Ở đây σ(z) = 1/(1 + e⁻ᶻ).

### Phần A — Trắc nghiệm và điền số (20 câu, 40 điểm, đến phút 47)

Mỗi câu 2 điểm. Chỉ ghi đáp số và một dòng lý do khi đề hỏi "vì sao".

**A1.** *(Chương 7 · ★ · 2 điểm)* `nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)` có bias, nhận tensor 16 × 40 × 40 (bỏ trục batch). (a) Kích thước đầu ra (kênh × cao × rộng). (b) Số tham số và số MAC của lớp conv này.

**A2.** *(Chương 7 · ★★ · 2 điểm)* x = (1, 3, 0, 2, 5, 1, 0, 4). Max-pool một chiều cửa sổ 2, stride 2. (a) Tính đầu ra cho x, cho x dịch phải 1 bước (thêm 0 bên trái, bỏ phần tử cuối) và cho x dịch phải 2 bước. (b) Phép dịch nào cho đầu ra là bản dịch của đầu ra gốc, và vì sao? Nếu sau pool có global max pooling thì ba đầu vào cho gì?

**A3.** *(Chương 7 · ★★ · 2 điểm)* (a) `nn.BatchNorm2d(32)` nhận tensor (16, 32, 7, 7) theo thứ tự NCHW. Mỗi kênh lấy trung bình trên bao nhiêu giá trị? Có bao nhiêu tham số học được và bao nhiêu giá trị running (mean cộng var)? (b) Ở chế độ eval, gộp BN vào conv 1×1 một kênh có w = 3, b = 0; BN có running mean 2, running var 9, γ = 0.6, β = 1, bỏ ε. Tính w′ và b′.

**A4.** *(Chương 9 · ★★ · 2 điểm)* Logit của bốn token là z = (1, 1, 0, −1). Giải mã bằng nhiệt độ T rồi top-p với p = 0.9 (giữ tập nhỏ nhất có tổng xác suất ≥ 0.9, rồi chuẩn hóa lại). (a) Ở T = 1 và ở T = 0.5, top-p giữ bao nhiêu token? (b) Phân phối sau khi cắt và chuẩn hóa lại ở mỗi nhiệt độ.

**A5.** *(Chương 9 · ★★ · 2 điểm)* Cùng một đoạn văn tiếng Việt 600 byte. Mô hình P đi với tokenizer của nó cắt đoạn thành 150 token, NLL trung bình 2.0 **bit** mỗi token (log cơ số 2). Mô hình Q đi với tokenizer khác, 300 token, NLL trung bình 1.1 bit mỗi token. (a) Perplexity theo token của P và Q. (b) Bits-per-byte của P và Q. Mô hình nào mô hình hóa đoạn văn tốt hơn?

**A6.** *(Chương 9 · ★★ · 2 điểm)* Một mô hình 10⁹ tham số được huấn luyện trên 10¹¹ token. (a) Ngân sách 6ND bằng bao nhiêu FLOPs? Cùng ngân sách đó, quy tắc khoảng 20 token mỗi tham số cho N và D bằng bao nhiêu? (b) Nêu một lý do thực tế để vẫn chọn mô hình 10⁹ tham số.

**A7.** *(Chương 9 · ★★ · 2 điểm)* GRPO: G = 6 đầu ra cho cùng một prompt, phần thưởng (1, 1, 0, 0, 0, 0). Advantage = (r − trung bình nhóm)/(độ lệch chuẩn nhóm), độ lệch chuẩn chia G. (a) Advantage của một đầu ra đúng, của một đầu ra sai, và tổng advantage của cả nhóm. (b) Nếu với prompt này mô hình trả lời đúng với xác suất 0.5, độc lập giữa các đầu ra, thì xác suất cả 6 đầu ra cùng đúng hoặc cùng sai là bao nhiêu, và khi đó prompt này đóng góp gradient gì?

**A8.** *(Chương 9 · ★ · 2 điểm)* (a) Mô hình có 28 khối, mỗi khối có `W_Q` và `W_V` cỡ 3072 × 3072. LoRA hạng r = 8 chỉ trên hai ma trận này: bao nhiêu tham số học được? (b) Tinh chỉnh toàn phần một mô hình Ψ = 3·10⁹ tham số bằng Adam với độ chính xác hỗn hợp (16Ψ byte, chưa tính activation) cần bao nhiêu GB (10⁹ byte)? Với LoRA, phần nào của con số này gần như biến mất và phần nào vẫn còn?

**A9.** *(Chương 9 · ★★ · 2 điểm)* Sinh n = 8 lời giải cho một bài lập trình, c = 2 lời giải qua test. (a) pass@4 theo ước lượng không chệch. (b) Công thức ngây thơ 1 − (1 − c/n)ᵏ cho bao nhiêu, và nó chệch về phía nào?

**A10.** *(Chương 9 · ★★ · 2 điểm)* Bài toán căn chỉnh có phạt KL: với một prompt có ba đầu ra, cực đại 𝔼 theo π của r − β·KL(π ‖ πref), với πref = (0.2, 0.4, 0.4), phần thưởng r = (1, 0, 0) và β = 1. (a) Nghiệm π* tỉ lệ với πref·exp(r/β): tính π*. (b) Phần thưởng ngầm β ln(π*/πref) của ba đầu ra bằng bao nhiêu? Vì sao từ π* chỉ khôi phục được r sai khác một hằng số, và vì sao điều đó không cản DPO?

**A11.** *(Chương 10 · ★ · 2 điểm)* Encoder của một VAE xuất μ = (0, 1.5) và `logvar` = (−2 ln 2, 0), với `logvar` là ln σ² của từng chiều. (a) KL(q(z|x) ‖ N(0, I)) bằng bao nhiêu? (b) Dòng `z = mu + logvar.exp() * eps` chạy không lỗi nhưng sai ở đâu? Viết lại cho đúng.

**A12.** *(Chương 10 · ★★ · 2 điểm)* InfoNCE với batch N = 512 (mỗi truy vấn có 1 khóa dương, 511 khóa âm). (a) Nếu embedding sụp đổ, mọi tương đồng bằng nhau, loss bằng bao nhiêu? Huấn luyện xong loss là 1.9: cận dưới của thông tin tương hỗ là bao nhiêu nat? (b) Muốn cận này có thể vượt 7 nat thì N nhỏ nhất là bao nhiêu?

**A13.** *(Chương 10 · ★★ · 2 điểm)* (a) FID giữa hai Gauss một chiều N(1, 0.5²) và N(0, 1). (b) Mục 10.5 cho xấp xỉ: hai tập, mỗi tập N mẫu, cùng lấy từ N(0, I) trong ℝᵈ, có FID kỳ vọng xấp xỉ (d(d + 1)/2 + 2d)/N. Tính cho d = 32, N = 500. Một bảng so FID 2.9 (đo bằng 1 000 mẫu) với FID 2.5 (đo bằng 10 000 mẫu): rút được kết luận gì?

**A14.** *(Chương 11 · ★ · 2 điểm)* Một paper báo độ chính xác 77.78% trên một tập test, một lần chạy. (a) Liệt kê mọi cỡ tập n ≤ 30 có số câu đúng k với k/n viết dạng phần trăm, làm tròn 2 chữ số thập phân, bằng 77.78%. (b) Phụ lục ghi n = 50. Số liệu có tự nhất quán không?

**A15.** *(Chương 11 · ★★ · 2 điểm)* Hai phương pháp thật sự ngang nhau; độ lệch chuẩn giữa các seed là σ = 0.4 điểm. Nhóm X chạy 10 seed và báo seed tốt nhất; nhóm Y báo trung bình của 10 seed. (a) Kỳ vọng X hơn Y bao nhiêu điểm? (b) Nếu cả hai nhóm cùng báo seed tốt nhất trong 10 thì chênh kỳ vọng bằng 0. Vì sao đó vẫn là cách báo cáo sai?

**A16.** *(Chương 11 · ★ · 2 điểm)* Quy trình nào gây **rò rỉ dữ liệu**? Chọn một, nêu lý do một dòng.
(A) Tách train/test ngẫu nhiên theo câu, trong khi mỗi tài liệu có khoảng 20 câu, nên câu của cùng một tài liệu nằm ở cả hai phía. (B) Chuẩn hóa z-score các đặc trưng bằng trung bình và độ lệch chuẩn tính trên train, áp nguyên cho test. (C) Chọn checkpoint và learning rate theo tập dev, chỉ chạy test một lần ở cuối. (D) Huấn luyện tokenizer BPE chỉ trên phần train.

**A17.** *(Phần I, chương 1 · ★★ · 2 điểm)* Dữ liệu 2 chiều đã trừ trung bình có ma trận hiệp phương sai C với hàng (5, 2) và (2, 2). (a) Hướng của thành phần chính thứ nhất, và tỉ lệ phương sai nó giữ lại. (b) Tọa độ của điểm (1, 3) trên thành phần chính đó.

**A18.** *(Phần I, chương 2–3 · ★★ · 2 điểm)* Phân loại 4 class, class đúng là class 1, label smoothing ε = 0.1: mục tiêu = (1 − ε)·one-hot + ε/4. (a) Tại logit (0, 0, 0, 0): cross-entropy với mục tiêu mềm, và gradient của nó theo logit. (b) Logit tối ưu có hữu hạn không? Tại tối ưu, logit của class đúng hơn logit của mỗi class khác bao nhiêu?

**A19.** *(Phần I, chương 4 · ★ · 2 điểm)* Huấn luyện bằng SGD + momentum với batch B = 64 và η = 0.02 đang ổn định. Bạn đổi sang B = 512. (a) Độ lệch chuẩn của nhiễu trong gradient minibatch giảm bao nhiêu lần, và chi phí tính một bước tăng bao nhiêu lần? (b) Theo quy tắc scale tuyến tính, η mới là bao nhiêu? Một dòng: quy tắc này gãy khi nào?

**A20.** *(Phần I, chương 6 · ★ · 2 điểm)* Inverted dropout với xác suất tắt p = 0.25, đầu vào (2, 4, 0, 6), mặt nạ giữ lại (1, 0, 1, 1). (a) Đầu ra ở chế độ train và ở chế độ eval. (b) Một dòng: vì sao chia cho 1 − p lúc train?

### Phần C — Lập trình NumPy (2 bài, 20 điểm, phút 47–73, làm trên máy)

Chỉ dùng NumPy. Viết cả hai hàm vào `bai_lam.py` (chép từ `labs/thi-thu-2/bai_lam_mau.py`, đã có sẵn chữ ký và docstring dưới đây). Câu (b) của mỗi bài trả lời trên giấy.

**C1. Causal self-attention một head.** *(Chương 8 · 10 điểm · 14 phút)* Không vòng lặp Python theo mẫu trong batch hay theo vị trí.

```python
def causal_attention(X, Wq, Wk, Wv):
    """Self-attention một head có mặt nạ nhân quả.
    X: mảng (B, T, d) float64. Wq, Wk: (d, dk). Wv: (d, dv).
    Q = X Wq, K = X Wk, V = X Wv (nhân theo từng mẫu trong batch).
    Trả về (O, A):
      A: mảng (B, T, T); A[b, i, j] là trọng số query i dành cho key j, bằng 0 khi j > i,
         mỗi hàng cộng bằng 1 (softmax của q_i·k_j/√dk trên các j ≤ i);
      O: mảng (B, T, dv) = A V.
    dk là số cột của Wq. Giữ float64. Phải đúng cả khi q·k/√dk cỡ ±10⁴ (không inf, không nan)."""
```

(a) *(8 điểm)* Viết hàm.

(b) *(2 điểm, ★★★, không chạy máy)* (i) Hàng đầu của A và đầu ra ở vị trí 1 bằng gì, với mọi X? Vì sao? Nếu một hàng của ma trận điểm bị che hết thì softmax của hàng đó ra gì khi che bằng −∞, và khi điền −1e9? Hậu quả với token đó là gì? (1 điểm) (ii) Lúc sinh tự hồi quy với KV cache, ở bước t ta chỉ tính q, k, v cho token mới xₜ, còn K và V của t − 1 token trước lấy từ cache (chỉ có một lớp attention này). Viết đầu ra oₜ theo qₜ và cache. Vì sao bước này không cần ma trận mặt nạ, và vì sao oₜ trùng hàng t của O khi chạy hàm (a) trên cả chuỗi? (1 điểm)

**C2. Paired bootstrap cho hiệu accuracy.** *(Chương 11, Phần I chương 3 · 10 điểm · 13 phút)* Không vòng lặp Python theo câu test; vòng lặp theo lần bootstrap thì được phép.

```python
def paired_bootstrap_ci(correct_a, correct_b, n_boot=10000, seed=0):
    """correct_a, correct_b: mảng (n,) kiểu int gồm 0 và 1; phần tử i cho biết mô hình A (hoặc B)
    trả lời đúng câu test i hay không. Hai mô hình được chấm trên CÙNG n câu.
    Trả về (diff, lo, hi), ba số thực:
      diff = accuracy(A) − accuracy(B), dạng tỉ lệ (0.012, không phải 1.2);
      [lo, hi] = khoảng tin cậy 95% của diff bằng paired bootstrap phân vị: n_boot lần, mỗi lần
                 lấy n chỉ số câu CÓ HOÀN LẠI và dùng cùng các chỉ số đó cho A và B.
    Dùng rng = np.random.default_rng(seed) cho mọi phép ngẫu nhiên: cùng seed phải cho cùng kết quả."""
```

(a) *(8 điểm)* Viết hàm.

(b) *(2 điểm, ★★, được dùng Python để tính số)* Hai checkpoint A và B của một mô hình hỏi đáp tiếng Việt được chấm trên cùng 1 000 câu: A đúng 812 câu, B đúng 790 câu; có 52 câu A đúng mà B sai và 30 câu A sai mà B đúng. Gọi b = 52 và c = 30 là hai ô bất đồng, diff là hiệu accuracy. (i) Coi A và B là hai mẫu độc lập: tính SE của hiệu và CI 95% dạng hiệu ± 1.96·SE. Rồi tính SE ghép cặp √((b + c)/n − diff²)/√n và CI 95% tương ứng (đây là xấp xỉ chuẩn của khoảng mà hàm (a) trả về). Hai khoảng khác nhau thế nào, vì sao? (1 điểm) (ii) Kiểm định McNemar có hiệu chỉnh liên tục, χ² = (|b − c| − 1)²/(b + c): tính, so với ngưỡng 3.84 và kết luận. Vì sao 918 câu hai checkpoint cùng đúng hoặc cùng sai không đi vào thống kê? (1 điểm)

### Phần B — Tự luận suy dẫn (5 bài, 40 điểm, phút 73–120)

Viết đủ các bước chính. Mỗi bài 8 điểm; điểm từng ý ghi trong ngoặc.

**B1. Gradient qua softmax-attention của một query.** *(Chương 8, 2 · 8 điểm · 12 phút)* Một query q ∈ ℝᵈᵏ chú ý vào n key k₁, …, kₙ ∈ ℝᵈᵏ (các hàng của K ∈ ℝⁿˣᵈᵏ) với value v₁, …, vₙ ∈ ℝᵈᵛ (các hàng của V):

$$
s_j = \frac{q \cdot k_j}{\sqrt{d_k}}, \qquad a = \operatorname{softmax}(s), \qquad o = \sum_{j=1}^{n} a_j v_j = V^{\top} a, \qquad L = \tfrac{1}{2}\,\lVert o - t \rVert_2^2 .
$$

Đặt g = ∂L/∂o = o − t.

(a) *(3 điểm, ★★)* Suy dẫn ∂L/∂vⱼ, ∂L/∂sⱼ, ∂L/∂q và ∂L/∂kⱼ theo a, g, o, q, K, V. Kiểm shape của từng gradient.

(b) *(3 điểm, ★★)* Số: dₖ = 4, n = 2, q = (1, 1, 0, 0), k₁ = (2, 2, 0, 0), k₂ = (0, 0, 1, −1), v₁ = (1, 0), v₂ = (0, 1), mục tiêu t = (0, 1) (tức muốn query chú ý vào key 2). Tính s, a, o, L; rồi ∂L/∂s và ∂L/∂q.

(c) *(2 điểm, ★★★)* (i) Chứng minh ∑ⱼ ∂L/∂sⱼ = 0 với mọi số liệu, và suy ra ∂L/∂q chỉ phụ thuộc hiệu giữa các key; với hai key, ∂L/∂q nằm trên đường thẳng nào? (1 điểm) (ii) Giữ nguyên mọi thứ ở (b) trừ q được nhân 3, nên logit thành (6, 0) (cùng hiệu ứng với việc quên chia √dₖ khi dₖ lớn). Tính L và ∂L/∂s₁, so với (b), và nói điều đó có nghĩa gì cho việc học. (1 điểm)

**B2. Đếm một mô hình kiểu Llama có GQA.** *(Chương 8, 9 · 8 điểm · 11 phút)* Mô hình decoder-only: L = 24 khối, d = 1024, 16 head query với dₕ = 64, `n_kv` = 4 head K/V (grouped-query attention), FFN SwiGLU có ba ma trận với kích thước ẩn `d_ff` = 2816, mỗi khối hai RMSNorm (chỉ có vector gain d chiều), một RMSNorm cuối, không bias ở đâu cả. Từ vựng V = 32 000, bảng embedding đầu vào và `lm_head` **không** dùng chung trọng số. Ngữ cảnh huấn luyện n = 2048.

(a) *(3 điểm, ★★)* Số tham số phần attention của một khối, phần FFN cộng hai norm, và tổng số tham số của mô hình.

(b) *(3 điểm, ★★)* Theo quy ước của chương 8 (FLOPs huấn luyện mỗi token ≈ 6N + 12·L·d·n, N là số tham số tham gia nhân ma trận: gồm `lm_head`, không gồm bảng embedding đầu vào), tính FLOPs mỗi token. Huấn luyện trên D = 6.7·10⁹ token (khoảng 20 token mỗi tham số): tổng FLOPs.

(c) *(2 điểm, ★★★)* (i) KV cache fp16 mỗi token, và cho batch 16 chuỗi dài 2048 token. Nếu dùng attention đầy đủ (16 head K/V) thì bao nhiêu? (1 điểm) (ii) Ở pha decode, GQA không giảm FLOPs của phép QKᵀ và AV, vậy vì sao nó vẫn tăng thông lượng phục vụ? Nó trả giá bằng gì? (1 điểm)

**B3. ELBO một chiều và β-VAE.** *(Chương 10 · 8 điểm · 11 phút)* p(z) = N(0, 1), p(x|z) = N(z, s²) với s² = 1/3 cố định, quan sát x = 2. Họ phân phối xấp xỉ q(z|x) = N(μ, σ²). Được dùng KL giữa hai Gauss một chiều:

$$
\mathrm{KL}\big(\mathcal{N}(\mu_1, \sigma_1^2)\,\|\,\mathcal{N}(\mu_2, \sigma_2^2)\big) = \ln\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}.
$$

(a) *(3 điểm, ★★)* Chứng minh

$$
\mathrm{ELBO}(\mu, \sigma^2) = -\tfrac{1}{2}\ln(2\pi s^2) - \frac{(x - \mu)^2 + \sigma^2}{2 s^2} - \tfrac{1}{2}\big(\sigma^2 + \mu^2 - 1 - \ln \sigma^2\big).
$$

(b) *(3 điểm, ★★)* Cực đại ELBO theo μ và σ²; so (μ*, σ²*) với posterior thật p(z|x) và giải thích vì sao khe hở bằng 0 tại đó. Tính khe hở log p(x) − ELBO khi q = N(1, 1).

(c) *(2 điểm, ★★★)* β-VAE cực đại 𝔼 theo q của log p(x|z) − β·KL(q ‖ p(z)). Tìm q tối ưu theo β, tính cho β = 4, rồi mô tả giới hạn β → ∞ và β → 0 theo đánh đổi rate–distortion.

**B4. Conv 2D: backward, receptive field, conv chuyển vị.** *(Chương 7 · 8 điểm · 12 phút)*

(a) *(3 điểm, ★★)* Một lớp cross-correlation 2D như `nn.Conv2d` (một kênh vào, một kênh ra, stride 1, không padding, không bias): Y[i, j] = ∑ᵤ ∑ᵥ W[u, v]·X[i + u, j + v]. Cho X có các hàng (1, 0, 2), (−1, 3, 1), (2, 1, 0); W có các hàng (1, −1), (2, 0); và G = ∂L/∂Y có các hàng (1, 2), (0, −1). Tính Y, ∂L/∂W, và hai ô của ∂L/∂X: ô giữa ∂L/∂X[1, 1] và ô góc ∂L/∂X[0, 2] (chỉ số từ 0).

(b) *(3 điểm, ★★)* Mạng cho ảnh 3 × 64 × 64, mọi conv có bias: conv1 = `Conv2d(3, 32, 3, stride=2, padding=1)`, ReLU, conv2 = `Conv2d(32, 32, 3, stride=1, padding=1)`, ReLU, `MaxPool2d(2)` (stride 2), conv3 = `Conv2d(32, 64, 3, stride=1, padding=2, dilation=2)`. Tính kích thước không gian sau mỗi conv và sau pool, số tham số của conv3, và receptive field cùng jump của một điểm đầu ra conv3 (theo một chiều).

(c) *(2 điểm, ★★★)* (i) Với số của (a), tính hai tích trong ⟨G, Y⟩ và ⟨∂L/∂W, W⟩ (tổng các tích từng phần tử). Không tính hết ∂L/∂X, cho biết ⟨∂L/∂X, X⟩ bằng bao nhiêu, và giải thích vì sao ba tích trong bằng nhau. (1 điểm) (ii) conv1 đưa ảnh 64 × 64 về 32 × 32. Ảnh 63 × 63 qua conv1 ra kích thước bao nhiêu? `ConvTranspose2d` cùng kernel, stride, padding cho kích thước ra (n − 1)·s − 2p + k + `output_padding`. Gradient theo đầu vào của conv1 chính là một conv chuyển vị: để nó có kích thước 64 thì `output_padding` bằng bao nhiêu, và vì sao chỉ từ kích thước 32 không suy ra được điều đó? (1 điểm)

**B5. So hai phương pháp bằng seed ghép cặp.** *(Chương 11 · 8 điểm · 10 phút)* Bạn so phương pháp M với baseline B trên bài NLI tiếng Việt. Cả hai được tune với cùng ngân sách, rồi chạy 6 seed; cùng một seed cho cùng khởi tạo phần chung và cùng thứ tự dữ liệu; cùng một tập test 2 000 câu. Độ chính xác (%):

| Seed | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| B | 71.0 | 72.4 | 70.2 | 73.1 | 71.8 | 70.9 |
| M | 71.6 | 72.6 | 70.7 | 74.0 | 72.1 | 71.4 |

Độ lệch chuẩn mẫu (chia n − 1) của từng cột cho sẵn: cột B khoảng 1.07, cột M khoảng 1.14.

(a) *(3 điểm, ★★)* Tính hiệu M − B theo từng seed, trung bình, độ lệch chuẩn mẫu, SE và CI 95% bằng phân vị t. Nếu coi hai cột là hai mẫu độc lập thì SE của hiệu là √((1.07² + 1.14²)/6) ≈ 0.64 (cho sẵn): tính CI 95% với hệ số 2.228 (xấp xỉ Welch, khoảng 10 bậc tự do). Vì sao hai kết luận khác nhau?

(b) *(3 điểm, ★★)* Muốn phát hiện chênh lệch thật δ = 0.3 điểm ở mức 5% hai phía với độ mạnh 0.8. Theo công thức xấp xỉ chuẩn, cần bao nhiêu seed nếu so ghép cặp (độ lệch chuẩn của hiệu lấy từ (a))? Bao nhiêu seed mỗi bên nếu so như hai mẫu độc lập với độ lệch chuẩn mỗi bên khoảng 1.1? Giải thích chỗ khác nhau giữa hai công thức, và công thức xấp xỉ chuẩn lệch về phía nào khi n nhỏ.

(c) *(2 điểm, ★★★)* (i) CI ở (a) chỉ phản ánh nguồn phương sai nào, và bỏ sót nguồn nào? Nêu cách đo thêm nguồn bị bỏ sót cho **hiệu** giữa M và B. (1 điểm) (ii) Thật ra bạn đã thử 4 biến thể của M và chỉ đưa vào bảng biến thể tốt nhất trên tập test. Điều này làm hỏng kết luận thế nào, và bạn sửa quy trình ra sao? (1 điểm)

## Bảng chấm theo chương

Điểm của mỗi câu được chia về chương theo cột "Câu (điểm)". Ghi **điểm cuối** vào cột "Điểm của bạn" rồi tra mức. Ngưỡng theo chương là heuristic của sách: vững từ 75% điểm tối đa, cần ôn từ 50% đến dưới 75%, yếu dưới 50% (làm tròn lên số nguyên); chưa hiệu chỉnh trên người làm thật. Tổng điểm có ngưỡng riêng, ở bảng "Theo tổng điểm cuối".

| Chương | Câu (điểm) | Tối đa | Điểm của bạn | Vững | Cần ôn | Yếu | Mục không được hỏi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | A1–A3 (6), B4 (8) | 14 | | 11–14 | 7–10 | 0–6 | Kiến trúc kinh điển, bottleneck, conv tách theo chiều sâu (7.3); augmentation, transfer learning (7.4); conv 1D, TextCNN, ViT (7.5) |
| 8 | C1 (10), B1 (8), B2(a)(c) (5) | 23 | | 18–23 | 12–17 | 0–11 | RNN, seq2seq, attention cộng tính (8.1); mã hóa vị trí và RoPE (8.3); pre-LN so với post-LN (8.4) chỉ qua đếm tham số; mặt nạ đệm chỉ qua C1(b)(i) |
| 9 | A4–A10 (14), B2(b) (3) | 17 | | 13–17 | 9–12 | 0–8 | BPE, beam search (9.1); reward model, loss DPO, PPO (9.3) chỉ qua nghiệm đóng ở A10 và GRPO ở A7; lượng tử hóa (9.4); RAG, agent (9.5) |
| 10 | A11–A13 (6), B3 (8) | 14 | | 11–14 | 7–10 | 0–6 | GAN (10.2); diffusion (10.3); masked modeling, BERT so với GPT (10.4) |
| 11 | A14–A16 (6), C2 định dạng, seed, ghép cặp (5), B5 (8) | 19 | | 15–19 | 10–14 | 0–9 | Baseline mạnh, ngân sách tune, ablation (11.2); nhiễm benchmark (11.4); viết, trình bày và tái lập (11.5) |
| Phần I | A17 (chương 1), A18 (chương 2–3), A19 (chương 4), A20 (chương 6) (8), C2 hiệu, mức 95% và C2(b) (5, chương 3) | 13 | | 10–13 | 7–9 | 0–6 | Chương 5 không có câu nào; mỗi chương 1, 2, 4, 6 chỉ có 2 điểm, nên dòng này chỉ báo động chung, không chẩn đoán từng chương |
| **Tổng** | | **100** | | Xem bảng "Theo tổng điểm cuối" | | | |

Cách chia điểm Phần C về chương (bộ chấm in sẵn phần tự động theo chương): C1 cả 10 điểm thuộc chương 8 (shape 1, mặt nạ 2, giá trị 3, ổn định số 2, câu (b) 2). C2 gồm định dạng 1, seed và lấy mẫu lại thật 1, ghép cặp 3 (chương 11); hiệu đúng 1, mức 95% 2 và câu (b) 2 (Phần I, chương 3).

**Điểm ★★★.** Cộng riêng các ý ★★★: C1(b), B1(c), B2(c), B3(c), B4(c), B5(c), tối đa 12 điểm. Con số này không dùng để quyết định ôn chương nào; nó đo khả năng đi xa hơn đáp số, thứ technical interview hay hỏi.

## Đọc kết quả

**Quy tắc một dòng.** Tra bằng điểm cuối; điểm trong giờ chỉ đọc tốc độ. Chương yếu thì học lại theo mục trước khi sang phần phụ thuộc vào nó; chương cần ôn thì ôn đúng câu sai, không đọc lại cả chương.

**Theo tổng điểm cuối** [ngưỡng là heuristic của sách, cùng mốc với Đề thi thử 1]. Có ít nhất một chương "yếu" (kể cả dòng Phần I) thì dùng dòng cuối, bất kể tổng.

| Tổng | Việc nên làm |
| --- | --- |
| 75–100, không chương nào yếu | Sang Phần III đúng lịch (tuần 25, chương 12, bắt đầu 19/4/2027). Dành 1–2 giờ trong tuần 25 làm lại các câu sai theo bảng Câu sai → ôn gì ở dưới. |
| 60–74, không chương nào yếu | Sang Phần III đúng lịch, nhưng ôn có mục tiêu: mỗi chương "cần ôn" khoảng 2 giờ, rải trong tuần 25–26. Theo `plan/weeks.yaml` hai tuần này đã có 13.75 giờ mỗi tuần, nên dưới trần 15 giờ chỉ còn khoảng 1.25 giờ mỗi tuần; phần còn lại theo người soạn nên lấy từ luồng B (giảm từ 3 bài xuống 2 bài mỗi tuần trong hai tuần đó). Đây là đề xuất, bạn quyết. |
| 0–59, hoặc có chương "yếu" | Cân nhắc lùi chương 12 một tuần và dùng tuần đó học lại chương yếu theo mục (đọc lại mục, tự làm lại ví dụ giải mẫu, làm bài ★ rồi ★★, làm lại mini-test). Lùi một tuần kéo theo cả lịch Phần III và Đề thi thử 3 (30/9/2027 theo `book/chapters.yaml`); quyết định thuộc về bạn. |

**Chương yếu kéo theo chương nào.** Các việc ôn dưới đây là giờ **vượt kế hoạch** nếu không lấy từ luồng B như dòng 60–74 ở trên.

- Chương 8 yếu: ưu tiên số một. Chương 12 (truy hồi, RAG, đánh giá LLM) và chương 14 (technical interview, chuỗi tuần 36 về deep learning hiện đại) giả định bạn viết được attention có mặt nạ và đếm được tham số, FLOPs, KV cache. Làm lại Lab 1 và Lab 4 chương 8 cùng mini-test chương 8 (đạt từ 24/30) trong tuần lùi nếu bạn lùi lịch; nếu không, trong tuần 25, trước buổi đọc retrieval của tuần 26 (theo `plan/weeks.yaml`, tuần 25 chỉ đọc mở đầu chương 12 và paper 1–3).
- Chương 9 yếu: chương 12 dùng recall@k và pass@k của mục 9.5 để đo truy hồi và RAG; ôn mục 9.5 (và 9.1, 9.3 nếu sai A4, A5, A10) trong tuần lùi, hoặc trong tuần 25 trước buổi đọc retrieval của tuần 26.
- Chương 11 yếu: chương 12 bắt đầu bằng việc đọc 10 paper theo phiếu ba lượt (mục 11.1) và dự án luồng A chạy nhiều seed; ôn mục 11.1 và 11.3 trong tuần 25, rải 30 phút mỗi ngày.
- Chương 10 yếu: truy hồi dense ở mục 12.1 huấn luyện bằng InfoNCE; ôn mục 10.4 trong tuần lùi, hoặc trong tuần 25 trước buổi đọc retrieval của tuần 26. Phần VAE, diffusion, FID ôn sau, trước chuỗi phỏng vấn tuần 36 của chương 14.
- Chương 7 yếu: Phần III (nhánh NLP) ít dùng trực tiếp CNN; ôn mục 7.1–7.2 trước chuỗi phỏng vấn tuần 36 của chương 14, không cần dừng lịch.
- Phần I yếu (dưới 7/13): xem câu sai thuộc chương nào và làm lại bài tính nhanh của chương đó; chương 14 có chuỗi tổng hợp tuần 35 về chương 1–6.

**Tốc độ.** Nếu điểm cuối trừ điểm trong giờ từ 15 điểm trở lên [heuristic, giữ như Đề thi thử 1; với độ dài đề này, người học mục tiêu dự kiến chênh khoảng 5–10 điểm], vấn đề chính là tốc độ chứ không phải kiến thức: mỗi ngày làm một bộ bài tính nhanh bấm giờ (khoảng 25 phút, giờ vượt kế hoạch như trên; chọn chương có nhiều nhãn H), và tập viết lời giải Phần B theo khung "công thức → thế số → kiểm shape". So với Đề thi thử 1 (cùng độ dài ước lượng, khoảng 133 phút): nếu khoảng chênh giảm, tốc độ đang lên. Nếu ở Đề thi thử 1 bạn rơi vào trường hợp điểm trong giờ dưới 50 nhưng điểm cuối từ 75 (đề 1 hẹn đo lại ở đây): điểm trong giờ của đề này từ 60 trở lên nghĩa là tốc độ đã đủ [heuristic].

**Theo nhãn lỗi.** Nhiều **K** ở một chương: đọc lại mục theo bảng dưới, làm bài ★. Nhiều **T** rải nhiều chương: làm bài tính nhanh 15 phút mỗi ngày và viết đủ bước trung gian. Nhiều **L**: đọc chậm đề, gạch chân điều kiện (√dₖ hay √d, tỉ lệ hay phần trăm, std chia n hay n − 1, ghép cặp hay độc lập). Nhiều **H**: xem mục Tốc độ.

**Câu sai → ôn gì.** "Bài ★2" nghĩa là bài 2 trong mục Bài tập giấy bút của chương đó; "tính nhanh 3" là bài 3 trong mục Bài tính nhanh.

| Câu | Đọc lại | Làm lại |
| --- | --- | --- |
| A1 | Mục 7.1, ví dụ 7.2 | Chương 7: bài ★1, ★2; tính nhanh 1–3; mini-test câu 1 |
| A2 | Mục 7.2, ví dụ 7.7 | Chương 7: bài ★★5; tính nhanh 14 |
| A3 | Mục 7.4, ví dụ 7.12 và 7.13 | Chương 7: bài ★★8; tính nhanh 9, 10, 15; mini-test câu 3 |
| A4 | Mục 9.1 (đoạn Giải mã), ví dụ 9.4 | Chương 9: bài ★3; tính nhanh 13; mini-test câu 2 |
| A5 | Mục 9.1 (đoạn Perplexity), ví dụ 9.3 | Chương 9: bài ★1; tính nhanh 2, 14; mini-test câu 1 |
| A6 | Mục 9.2, ví dụ 9.5 | Chương 9: bài ★★5; tính nhanh 3, 4; mini-test câu 3 |
| A7 | Mục 9.3 (đoạn RLVR và GRPO), ví dụ 9.8 | Chương 9: bài ★★★14; tính nhanh 9 |
| A8 | Mục 9.4, ví dụ 9.9 và 9.10 | Chương 9: bài ★★7; tính nhanh 5, 6; mini-test câu 5 |
| A9 | Mục 9.5, ví dụ 9.12 | Chương 9: bài ★★9; tính nhanh 8 |
| A10 | Mục 9.3 (đoạn DPO), ví dụ 9.7 | Chương 9: bài ★4, ★★★12; mini-test câu 4 |
| A11 | Mục 10.1 (công thức KL và đoạn Thực tế tính toán), ví dụ 10.1 | Chương 10: bài ★1; tính nhanh 1–3 |
| A12 | Mục 10.4, ví dụ 10.9 | Chương 10: bài ★4; tính nhanh 10, 11 |
| A13 | Mục 10.5, ví dụ 10.12 | Chương 10: tính nhanh 14 |
| A14 | Mục 11.1, ví dụ 11.2 | Chương 11: bài ★★6; tính nhanh 5, 9 |
| A15 | Mục 11.3 (đoạn Best-of-k là thiên lệch) | Chương 11: bài ★★9; tính nhanh 6 |
| A16 | Mục 11.4 | Chương 11: bài ★4; mini-test câu 4 |
| A17 | Mục 1.5 (đoạn PCA chính là SVD) | Chương 1: bài ★★★11 |
| A18 | Mục 3.5A, ví dụ 3.10; mục 2.5 | Chương 2: bài ★★★11 |
| A19 | Mục 4.3, ví dụ 4.6 và 4.7, đoạn "η/B như một nhiệt độ"; mục 4.4 (momentum) | Chương 4: bài ★★★11; tính nhanh 9; mini-test câu 5; Kiểm tra nhanh câu 3 |
| A20 | Mục 6.4 (đoạn Dropout) | Chương 6: bài ★★8; tính nhanh 10 |
| C1 | Mục 8.2 (softmax ổn định, ví dụ 8.3); mục 8.5 (mặt nạ, ví dụ 8.13 và 8.14; KV cache, ví dụ 8.12) | Chương 8: Lab 1; mini-test câu 5; viết lại hàm không nhìn lời giải sau 3 ngày |
| C2 | Mục 3.5B (bootstrap, paired bootstrap, ví dụ 3.11); mục 11.3 | Chương 3: Lab 5; viết lại hàm sau 3 ngày |
| B1 | Mục 8.2 (vì sao chia √dₖ, ví dụ 8.4); mục 2.5 (Jacobian của softmax) | Chương 8: bài ★★★14; mini-test câu 3 |
| B2 | Mục 8.4–8.5, ví dụ 8.8, 8.9, 8.11, 8.12; mục 9.2, ví dụ 9.5 | Chương 8: bài ★4, ★★8; mini-test câu 2, 6 |
| B3 | Mục 10.1, ví dụ 10.2 | Chương 10: bài ★★5, ★★★12 |
| B4 | Mục 7.1 (đoạn Backward của conv cũng là conv), ví dụ 7.3 và 7.4; mục 7.2, ví dụ 7.5 | Chương 7: bài ★3, ★★9; mini-test câu 2 |
| B5 | Mục 11.3, ví dụ 11.6 và 11.7 | Chương 11: bài ★3, ★★8, ★★★13; mini-test câu 1, 5 |

**Đo lại.** Sách không có bản thứ hai của đề này. Mini-test từng chương đo lại chương đó; Đề thi thử 3 (30/9/2027) là mốc đo tổng tiếp theo. Ghi điểm từng chương, điểm trong giờ và điểm cuối vào trang Tiến độ, đặt cạnh số của Đề thi thử 1 để so.

## Lời giải

Chỉ xem sau khi đã làm hết đề và tự chấm Phần C bằng bộ chấm. Thang điểm ghi ở cuối mỗi lời giải. Lời giải xếp theo thứ tự làm bài: A, C, B.

### Phần A

**A1.** (a) Theo mỗi chiều: ⌊(40 + 2·2 − 5)/2⌋ + 1 = ⌊39/2⌋ + 1 = 20, nên đầu ra 32 × 20 × 20. (b) Tham số 32·16·5·5 + 32 = 12 832. MAC = 20·20·32·16·25 = 5 120 000 (bỏ bias). *Thang:* 1 điểm (a); 1 điểm (b) đủ hai số.

**A2.** (a) x cho (3, 2, 5, 4). Dịch 1 bước, (0, 1, 3, 0, 2, 5, 1, 0), cho (1, 3, 5, 1). Dịch 2 bước, (0, 0, 1, 3, 0, 2, 5, 1), cho (0, 3, 2, 5). (b) Chỉ dịch 2 bước (bội của stride) cho đầu ra là bản dịch 1 bước của (3, 2, 5, 4): các cửa sổ vẫn gom đúng các cặp cũ. Dịch 1 bước làm các cặp bị cắt ở chỗ khác, nên đầu ra đổi hẳn (aliasing, ví dụ 7.7). Global max của cả ba đầu ra đều là 5: ở ví dụ này global pooling khôi phục được bất biến, nhưng nói chung chỉ là bất biến xấp xỉ (phần tử bị đẩy ra khỏi biên có thể là max). *Thang:* 1 điểm ba đầu ra; 1 điểm kết luận "bội của stride" có lý do.

**A3.** (a) Mỗi kênh lấy thống kê trên N·H·W = 16·7·7 = 784 giá trị. Tham số học γ và β: 2·32 = 64; running mean và running var: 64 giá trị (buffer, không học bằng gradient). (b) Hệ số γ/√σ² = 0.6/3 = 0.2. w′ = 3·0.2 = 0.6; b′ = (0 − 2)·0.2 + 1 = 0.6. Tự kiểm (không bắt buộc) với x = 2: conv cho 6, BN cho (6 − 2)/3·0.6 + 1 = 1.8; conv đã gộp cho 0.6·2 + 0.6 = 1.8. *Thang:* 1 điểm (a) đủ ba số; 1 điểm (b) đủ w′ và b′.

**A4.** (a) T = 1: p tỉ lệ với (e, e, 1, e⁻¹), tổng 2e + 1 + e⁻¹ ≈ 6.805, nên p ≈ (0.400, 0.400, 0.147, 0.054). Cộng dồn sau khi sắp giảm dần: 0.400, 0.799, 0.946; hai token đầu chưa đủ 0.9, token thứ ba làm tổng vượt 0.9 và phải được giữ: **3 token**. T = 0.5: chia logit cho T được (2, 2, 0, −2), p tỉ lệ với (e², e², 1, e⁻²), tổng ≈ 15.91, nên p ≈ (0.464, 0.464, 0.063, 0.008); cộng dồn 0.464, 0.929 ≥ 0.9: **2 token**. (b) T = 1: (e, e, 1)/(2e + 1) ≈ (0.422, 0.422, 0.155, 0). T = 0.5: (0.5, 0.5, 0, 0). Hạ nhiệt độ làm phân phối nhọn hơn nên top-p giữ ít token hơn; hai token cùng logit thì sắp xếp theo thứ tự nào cũng cho cùng tập. Lỗi hay gặp: chia T trên xác suất thay vì trên logit, hoặc bỏ token làm tổng vượt p. *Thang:* 1 điểm (a) hai số token; 1 điểm (b) hai phân phối (dạng biểu thức đủ điểm).

**A5.** (a) NLL đo bằng bit nên perplexity = 2 mũ (số bit mỗi token): P = 2² = 4, Q = 2¹·¹ ≈ 2.14. Viết e² và e¹·¹ là lẫn cơ số (lỗi L). (b) Tổng NLL: P 150·2.0 = 300 bit, Q 300·1.1 = 330 bit. Bits-per-byte: P 300/600 = 0.50, Q 330/600 = 0.55. P tốt hơn: nó gán xác suất cao hơn cho đúng đoạn văn (tổng NLL nhỏ hơn). Perplexity theo token của Q thấp chỉ vì Q cắt nhỏ hơn, mỗi token dễ đoán hơn; perplexity theo token không so được giữa hai tokenizer. *Thang:* 1 điểm hai perplexity đúng cơ số; 1 điểm hai bpb và kết luận P tốt hơn kèm lý do.

**A6.** (a) C = 6·10⁹·10¹¹ = 6·10²⁰ FLOPs. Với D = 20N: 6·N·20N = 6·10²⁰, nên N² = 5·10¹⁸, N = √5·10⁹ ≈ 2.24·10⁹ tham số, D ≈ 4.47·10¹⁰ token. Mô hình 10⁹ trên 10¹¹ token (100 token mỗi tham số) là "huấn luyện lâu hơn" so với điểm Chinchilla. (b) Chi phí suy luận: nếu mô hình sẽ phục vụ rất nhiều yêu cầu, mô hình nhỏ huấn luyện lâu hơn rẻ hơn về tổng chi phí (Sardana và cộng sự, mục 9.2). Lý do khác cũng nhận: phải vừa bộ nhớ thiết bị triển khai, độ trễ. *Thang:* 1 điểm C, N, D; 1 điểm lý do.

**A7.** (a) Trung bình 1/3, phương sai (1/3)(2/3) = 2/9, độ lệch chuẩn √2/3 ≈ 0.471. Đầu ra đúng: (2/3)/(√2/3) = √2 ≈ 1.414; đầu ra sai: (−1/3)/(√2/3) = −1/√2 ≈ −0.707. Tổng 2·1.414 − 4·0.707 = 0. (b) 2·0.5⁶ = 1/32 ≈ 0.031. Khi đó độ lệch chuẩn nhóm bằng 0, mọi advantage bằng 0 (cài đặt thường thêm ε vào mẫu số hoặc bỏ nhóm đó), nên prompt này không đóng góp gradient từ phần thưởng. *Thang:* 1 điểm (a) đủ ba số; 1 điểm (b) số và "không có gradient".

**A8.** (a) Mỗi ma trận r·(3072 + 3072) = 8·6144 = 49 152; hai ma trận × 28 khối: 2 752 512. (b) 16·3·10⁹ = 48·10⁹ byte = 48 GB. Với LoRA, gradient và trạng thái Adam (m, v, bản fp32) chỉ còn cho 2.75 triệu tham số adapter, nên gần như biến mất; trọng số nền đóng băng vẫn phải nằm trong bộ nhớ (2Ψ = 6 GB ở bf16, hoặc ít hơn nếu lượng tử hóa như QLoRA), cộng activation. *Thang:* 1 điểm (a); 1 điểm (b) đủ số và phần còn lại.

**A9.** (a) 1 − C(6, 4)/C(8, 4) = 1 − 15/70 = 11/14 ≈ 0.786. (b) 1 − 0.75⁴ = 1 − 0.316 = 0.684, chệch xuống: công thức ngây thơ coi k mẫu là lấy có hoàn lại từ tỉ lệ c/n, trong khi ước lượng không chệch đếm đúng xác suất một tập con k mẫu (không hoàn lại) chứa ít nhất một mẫu đúng. *Thang:* 1 điểm mỗi ý.

**A10.** (a) Z = 0.2e + 0.4 + 0.4 = 0.2·2.718 + 0.8 ≈ 1.344. π* = (0.2e, 0.4, 0.4)/Z ≈ (0.405, 0.298, 0.298): đầu ra có thưởng tăng từ 0.2 lên khoảng 0.4, hai đầu ra còn lại giảm cùng tỉ lệ. (b) β ln(π*/πref) = r − β ln Z = (1 − ln Z, −ln Z, −ln Z) ≈ (0.705, −0.295, −0.295) (dạng biểu thức đủ điểm). Cộng một hằng c vào mọi phần thưởng của cùng prompt thì mọi exp(r/β) nhân cùng một thừa số, Z hấp thụ nó, π* không đổi: từ π* chỉ suy ra được r + hằng số theo prompt. DPO chỉ dùng **hiệu** phần thưởng ngầm giữa hai đầu ra của cùng prompt (mô hình Bradley-Terry), nên số hạng β ln Z(x) triệt tiêu và không cần tính Z. *Thang:* 1 điểm π*; 1 điểm phần thưởng ngầm bằng r − β ln Z kèm lý do triệt tiêu trong hiệu.

**A11.** (a) σ² = (e⁻²ˡⁿ², 1) = (0.25, 1). Chiều 1: ½(0.25 + 0 − 1 + 2 ln 2) = ½(0.25 − 1 + 1.3863) = 0.318. Chiều 2: ½(1 + 2.25 − 1 − 0) = 1.125. Tổng ≈ 1.443 nat (cộng theo chiều, không lấy trung bình). (b) `logvar.exp()` là σ², không phải σ: ở chiều 1 nhiễu bị nhân 0.25 thay vì 0.5. Đúng là `z = mu + (0.5 * logvar).exp() * eps`. *Thang:* 1 điểm mỗi ý.

**A12.** (a) Mọi logit bằng nhau thì loss = ln 512 = 9 ln 2 ≈ 6.238. Cận I ≥ ln N − L = 6.238 − 1.9 ≈ 4.34 nat. (b) Cận không bao giờ vượt ln N, nên cần ln N > 7, tức N > e⁷ ≈ 1096.6: N nhỏ nhất là 1 097. *Thang:* 1 điểm (a) hai số; 1 điểm (b).

**A13.** (a) Một chiều: (m₁ − m₂)² + (σ₁ − σ₂)² = 1 + 0.25 = 1.25. (b) (32·33/2 + 64)/500 = (528 + 64)/500 ≈ 1.18: hai mẫu từ **cùng** một phân phối vẫn có FID cỡ 1.2. Thiên lệch giảm theo 1/N, nên con số đo bằng 1 000 mẫu bị đẩy lên nhiều hơn con số đo bằng 10 000 mẫu; chênh 0.4 giữa hai con số đo với cỡ mẫu khác nhau không cho kết luận nào. Phải đo cùng cỡ mẫu (và cùng bộ trích đặc trưng). *Thang:* 1 điểm (a); 1 điểm (b) có số và kết luận "không so được".

**A14.** (a) 77.78% ≈ 7/9 = 0.77778. Với n ≤ 30, chỉ các bội của 9 cho được: n = 9, 18, 27 (k = 7, 14, 21). (b) Với n = 50, k/n là bội của 2%, chỉ ra 76.00% hoặc 78.00%, nên "77.78% trên 50 câu, một lần chạy" không tự nhất quán. Có thể là trung bình nhiều seed, hoặc n khác; đó là câu để hỏi tác giả, không phải bằng chứng gian lận. *Thang:* 1 điểm mỗi ý.

**A15.** (a) Seed tốt nhất có kỳ vọng μ + 1.539σ; trung bình 10 seed có kỳ vọng μ. Chênh ≈ 1.539·0.4 ≈ 0.62 điểm, không có thật. (b) Con số báo cáo của cả hai đều bị thổi lên khoảng 0.6 điểm so với hiệu năng kỳ vọng của một lần chạy, và max của 10 seed vẫn dao động; người đọc không biết phương sai, không tái lập được. Nên báo trung bình, độ lệch chuẩn (nói rõ chia n − 1), số seed, và hiệu kèm CI. *Thang:* 1 điểm mỗi ý.

**A16.** (A). Câu của cùng một tài liệu rất giống nhau (chủ đề, tên riêng, văn phong), nên test đo một phần khả năng nhớ tài liệu đã thấy; phải tách theo tài liệu (group split). (B), (C), (D) chỉ dùng thông tin của train hoặc dev. *Thang:* 1 điểm chọn đúng, 1 điểm lý do.

**A17.** (a) tr C = 7, det C = 6: trị riêng 6 và 1. Vector riêng của 6: (5 − 6)u₁ + 2u₂ = 0, tức u = (2, 1)/√5. Tỉ lệ phương sai 6/7 ≈ 0.857. (b) Tọa độ (1, 3)·(2, 1)/√5 = 5/√5 = √5 ≈ 2.236. Thành phần chính chỉ xác định đến dấu: hướng (−2, −1)/√5 và tọa độ −√5 cũng đủ điểm. *Thang:* 1 điểm mỗi ý.

**A18.** (a) Mục tiêu (0.925, 0.025, 0.025, 0.025). Logit 0 cho p đều 0.25, nên CE = −∑ tᵢ ln 0.25 = ln 4 ≈ 1.386. Gradient p − mục tiêu = (−0.675, 0.225, 0.225, 0.225). (b) Gradient bằng 0 khi p = mục tiêu, đạt được với logit hữu hạn: logit class đúng hơn mỗi class khác ln(0.925/0.025) = ln 37 ≈ 3.61. Khác với one-hot, nơi loss chỉ tiến về 0 khi các logit tách xa vô hạn. *Thang:* 1 điểm mỗi ý.

**A19.** (a) Độ lệch chuẩn tỉ lệ 1/√B; B tăng 8 lần nên nhiễu giảm √8 = 2√2 ≈ 2.83 lần; chi phí mỗi bước tăng 8 lần. (b) η = 0.02·8 = 0.16. Quy tắc gãy khi η chạm trần ổn định (ηλₘₐₓ < 2 với SGD thường, < 2(1 + β) với momentum hệ số β, bài ★★★11 chương 4), lúc đó không tăng η được nữa dù B tăng; ở đầu huấn luyện còn cần warmup để không phân kỳ (Goyal và cộng sự kiểm chứng cho SGD + momentum). *Thang:* 1 điểm mỗi ý.

**A20.** (a) Train: (2/0.75, 0, 0, 6/0.75) = (2.667, 0, 0, 8). Eval: dropout là hàm đồng nhất, đầu ra (2, 4, 0, 6). (b) Mỗi phần tử được giữ với xác suất 1 − p, nên chia 1 − p làm 𝔼 đầu ra lúc train bằng đầu vào, khớp với lúc eval không cần nhân gì. *Thang:* 1 điểm mỗi ý.

### Phần C

**C1.** (a) Lời giải mẫu (chỉ có ở đây và trong `work/mock2/check_partC.py`; thư mục `labs/thi-thu-2` không chứa lời giải):

```python
import numpy as np

def causal_attention(X, Wq, Wk, Wv):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv                        # (B, T, dk), (B, T, dk), (B, T, dv)
    S = Q @ K.swapaxes(-1, -2) / np.sqrt(Wq.shape[1])      # (B, T, T), chia √dk
    T = X.shape[1]
    S = np.where(np.tril(np.ones((T, T), dtype=bool)), S, -np.inf)   # che j > i TRƯỚC softmax
    S = S - S.max(axis=-1, keepdims=True)                   # max trên các ô được nhìn; đường chéo luôn mở
    E = np.exp(S)                                           # exp(−inf) = 0
    A = E / E.sum(axis=-1, keepdims=True)
    return A @ V, A
```

Các chỗ hay sai, kèm số điểm bộ chấm trừ (khớp thang dưới):

- Quên mặt nạ, hoặc che **sau** softmax (nhân A với `tril` mà không chuẩn hóa lại): mẫu số đã chứa các key tương lai, hàng không còn cộng bằng 1, và đổi token tương lai làm đổi đầu ra quá khứ. Mất 2 (mặt nạ).
- Chia √d (số chiều của X) thay vì √dₖ, chia dₖ, hoặc không chia. Mất 3 (giá trị).
- Không trừ max (tràn exp), trừ max của cả tensor thay vì từng hàng (hàng có điểm thấp hơn nhiều bị tràn dưới thành 0/0), hoặc lấy max theo hàng **trước** khi che (max gồm key tương lai; nếu key tương lai có điểm rất lớn thì mọi ô được phép tràn dưới, ra 0/0). Mất 2 (ổn định số).
- `K.T` trên tensor 3 chiều đảo cả ba trục, không phải hai trục cuối; với B ≠ dₖ phép nhân báo lỗi và hàm được 0. Dùng `K.swapaxes(-1, -2)` hoặc `np.transpose(K, (0, 2, 1))`.
- Softmax theo trục −2, hoặc che cả đường chéo bằng −∞ (hàng đầu bị che hết, ra nan): sai cả mặt nạ, giá trị và ổn định số, chỉ còn 1 điểm shape.
- Che cả đường chéo nhưng điền −1e9: hàng 1 thành phân phối đều, không nan; hàng 1 nhìn cả tương lai và mọi hàng mất chính mình. Mất mặt nạ 2 và giá trị 3, còn 3.
- Đổi sang float32: mất giá trị 3 và ổn định số 2, vì dung sai 10⁻¹⁰ và test cộng hằng cỡ 10⁴ chủ ý đòi float64.

*Thang (a), 8 điểm:* 1 điểm shape (O là (B, T, dᵥ), A là (B, T, T)); 2 điểm mặt nạ nhân quả (ô j > i bằng 0, mỗi hàng cộng 1, đổi token tương lai không đổi đầu ra quá khứ); 3 điểm giá trị (hàng cuối của A và O khớp tham chiếu, gồm một ví dụ tay có dₖ ≠ d); 2 điểm ổn định số (cộng một hằng cỡ ±10⁴ vào từng hàng điểm không đổi kết quả; key tương lai có điểm cỡ 1 400 không làm ra nan). Trừ 2 điểm nếu có vòng lặp Python theo mẫu trong batch hay theo vị trí (điểm (a) không âm). Phần 8 điểm này do `labs/thi-thu-2/cham_phan_c.py` chấm; test giá trị chỉ so hàng cuối (hàng nhìn mọi key), nên lỗi mặt nạ không bị trừ lần nữa ở đó.

(b) (i) Hàng 1 chỉ được nhìn key 1, nên softmax trên đúng một phần tử bằng 1: hàng đầu của A là (1, 0, …, 0) và o₁ = 1·v₁ bất kể X. Một hàng bị che hết xảy ra khi viết nhầm mặt nạ che cả đường chéo (hàng 1) hoặc ở hàng của token đệm khi padding trái (ví dụ 8.14). Với −∞, mọi exp bằng 0 (sau khi trừ max, −∞ − (−∞) đã là nan), hàng ra nan và nan lan sang mọi phép tính dùng hàng đó, kể cả gradient của cả batch. Với −1e9 thì mọi ô của hàng bằng nhau, softmax ra phân phối đều 1/T: token đó nhìn đều mọi vị trí, **cả tương lai** và cả token đệm, một lỗi rò rỉ im lặng (nếu do che cả đường chéo thì các hàng khác còn mất khả năng nhìn chính mình). Xử lý: quy ước hàng bị che hết cho trọng số 0 (đầu ra 0) và bỏ vị trí đó khỏi loss. (ii)

$$
o_t = \sum_{j=1}^{t} \frac{\exp\!\big(q_t \cdot k_j / \sqrt{d_k}\big)}{\sum_{l=1}^{t} \exp\!\big(q_t \cdot k_l / \sqrt{d_k}\big)}\, v_j .
$$

Cache chỉ chứa các key j ≤ t, nên không có gì cần che. Với một lớp attention, kⱼ = xⱼWₖ và vⱼ = xⱼWᵥ chỉ phụ thuộc xⱼ, nên giá trị trong cache đúng bằng giá trị khi chạy cả chuỗi; và hàng t của A có mặt nạ cũng là softmax trên đúng các j ≤ t. Hai biểu thức trùng nhau. (Với nhiều lớp cần quy nạp theo lớp: kⱼ ở lớp ℓ chỉ phụ thuộc x₁…xⱼ nhờ mặt nạ ở các lớp dưới; bài ★★★12 chương 8.) *Thang:* 1 điểm mỗi ý; ở (i) cần hàng đầu (1, 0, …, 0) và o₁ = v₁ kèm lý do, và hai trường hợp −∞, −1e9 kèm hậu quả.

**C2.** (a) Lời giải mẫu:

```python
import numpy as np

def paired_bootstrap_ci(correct_a, correct_b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = correct_a.astype(np.float64) - correct_b.astype(np.float64)   # hiệu theo từng câu: −1, 0 hoặc 1
    n = d.size
    idx = rng.integers(0, n, size=(n_boot, n))            # cùng chỉ số cho A và B: ghép cặp
    boot = d[idx].mean(axis=1)                            # n_boot giá trị của hiệu accuracy
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)
```

Lấy mẫu lại hiệu dᵢ của từng câu tương đương lấy mẫu lại cặp (aᵢ, bᵢ) rồi tính accuracy(A) − accuracy(B) trên cùng chỉ số. Ma trận chỉ số cỡ (`n_boot`, n) với n = 2 000 và `n_boot` = 4 000 chiếm khoảng 64 MB; nếu máy yếu, lặp theo lần bootstrap (được phép). Hai cách khác bộ chấm chấp nhận: bootstrap "basic" [2·diff − q₉₇.₅, 2·diff − q₂.₅] và diff ± 1.96·(độ lệch chuẩn của các giá trị bootstrap); với n cỡ hàng trăm trở lên, ba cách gần như trùng nhau.

Các chỗ hay sai, kèm số điểm bộ chấm trừ:

- Lấy mẫu lại A và B bằng hai bộ chỉ số độc lập: CI rộng như coi hai mô hình độc lập (gấp khoảng 1.9 lần trên bộ test của bộ chấm). Mất 3 (ghép cặp).
- Trả phần trăm (×100) hoặc tính B − A. Mất 1 (hiệu đúng); các test về CI chia cho chính diff của bạn nên không trừ thêm.
- Lấy phân vị 5 và 95 (khoảng 90%). Mất 2 (mức 95%).
- Nhầm thang của `np.percentile` (viết 0.025 và 0.975, tức phân vị 0.025% và 0.975%): khoảng nằm hẳn ở đuôi trái, không chứa chính diff. Mất 3 (ghép cặp) và 2 (mức), còn 3.
- `np.random.default_rng()` không truyền seed, hoặc không bootstrap mà dùng công thức chuẩn: mất 1 (seed và lấy mẫu lại thật).
- Lấy mẫu **không** hoàn lại (hoán vị): mọi lần bootstrap cho cùng một hiệu, khoảng suy biến thành một điểm. Mất 1 + 3 + 2, còn 2.
- Tạo lại `rng = np.random.default_rng(seed)` bên trong vòng lặp bootstrap: mọi lần bootstrap rút cùng một bộ chỉ số, khoảng suy biến thành một điểm (CI hẹp giả, trông như rất chắc chắn). Mất 1 + 3 + 2, còn 2.

*Thang (a), 8 điểm:* 1 điểm định dạng (ba số hữu hạn, lo ≤ hi); 1 điểm tất định theo seed và có lấy mẫu lại thật (đổi seed thì khoảng đổi); 1 điểm hiệu đúng (A − B, dạng tỉ lệ); 3 điểm ghép cặp (khoảng chứa diff, và tỉ số độ rộng CI giữa một bộ dữ liệu hai mô hình tương quan mạnh với một bộ độc lập có cùng tỉ lệ đúng khớp tỉ số SE ghép cặp, 0.53, sai lệch tối đa 15%); 2 điểm mức 95% (trên bộ độc lập, độ rộng trong khoảng ±7% quanh 2·1.96·SE và tâm cách hiệu không quá 0.25 SE). Trừ 2 điểm nếu có vòng lặp Python theo câu test (điểm (a) không âm). Phần 8 điểm này do `labs/thi-thu-2/cham_phan_c.py` chấm.

(b) (i) Hiệu 0.812 − 0.790 = 0.022. Coi độc lập: SE = √(0.812·0.188/1000 + 0.79·0.21/1000) ≈ 0.0178, CI ≈ [−0.013, 0.057], chứa 0. Ghép cặp: SE = √((82/1000 − 0.022²)/1000) ≈ 0.0090, CI = 0.022 ± 1.96·0.0090 = [0.004, 0.040], không chứa 0. Hàm (a) chạy trên hai mảng 0/1 dựng từ bảng chéo cho gần như cùng khoảng (10 000 lần, 5 seed: cận dưới 0.004 đến 0.005, cận trên 0.039 đến 0.040). Khoảng ghép cặp hẹp gần một nửa vì hai checkpoint đồng ý trên 918/1 000 câu: phần lớn dao động do "câu dễ hay khó" là chung cho cả hai và triệt tiêu trong hiệu. Coi độc lập là bỏ đi tương quan dương đó. (ii) b = 52, c = 30: χ² = (22 − 1)²/82 = 441/82 ≈ 5.38 > 3.84, bác bỏ "hai checkpoint có cùng accuracy" ở mức 5% (bản không hiệu chỉnh cho 5.90, cùng kết luận). Câu hai bên cùng đúng hoặc cùng sai không cho thông tin về việc bên nào tốt hơn: dưới giả thuyết không, mỗi câu bất đồng nghiêng về A hay B với xác suất ½, và chỉ số câu bất đồng mới là phép thử. *Thang:* 1 điểm mỗi ý; ở (i) cần hai khoảng và lý do tương quan.

**Chạy bộ chấm.** Sau khi hết giờ, từ thư mục gốc của sách: `python labs/thi-thu-2/cham_phan_c.py --student bai_lam.py`. Bộ chấm chỉ cần NumPy: tham chiếu của C1 viết bằng vòng lặp theo định nghĩa trên dữ liệu có điểm nhỏ, các ca điểm cực lớn so với đáp số tính tay; tham chiếu của C2 là SE ghép cặp dạng đóng. Nó in điểm tự động trên 16, chia sẵn theo chương, và nhắc nếu thấy vòng lặp trong hàm; cộng tay 4 điểm của C1(b), C2(b) và trừ điểm vòng lặp nếu đó là vòng lặp bị cấm.

### Phần B

**B1.** (a) Đi ngược từng bước.

- o = ∑ⱼ aⱼvⱼ nên ∂L/∂vⱼ = aⱼg ∈ ℝᵈᵛ, và ∂L/∂aⱼ = vⱼ·g.
- Jacobian của softmax là diag(a) − aaᵀ (chương 2, mục 2.5), nên ∂L/∂sⱼ = aⱼ(vⱼ·g − ∑ₗ aₗ vₗ·g) = aⱼ(vⱼ·g − o·g) = aⱼ(vⱼ − o)·g.
- sⱼ = q·kⱼ/√dₖ nên

$$
\frac{\partial L}{\partial q} = \frac{1}{\sqrt{d_k}} \sum_{j} \frac{\partial L}{\partial s_j}\, k_j = \frac{1}{\sqrt{d_k}}\, K^{\top} \frac{\partial L}{\partial s} \in \mathbb{R}^{d_k}, \qquad \frac{\partial L}{\partial k_j} = \frac{1}{\sqrt{d_k}}\, \frac{\partial L}{\partial s_j}\, q \in \mathbb{R}^{d_k}.
$$

Shape: ∂L/∂s là vector n chiều (cùng s); Kᵀ cỡ dₖ × n nhân vector n chiều ra dₖ, khớp q; mỗi ∂L/∂kⱼ cùng shape kⱼ (gộp lại thành ma trận n × dₖ như K, bằng (∂L/∂s)qᵀ/√dₖ); ∂L/∂V = a gᵀ cỡ n × dᵥ như V. *Thang:* 1 điểm ∂L/∂sⱼ có qua Jacobian của softmax; 1 điểm ∂L/∂q và ∂L/∂kⱼ; 1 điểm ∂L/∂vⱼ và kiểm shape cho cả bốn.

(b) q·k₁ = 4, q·k₂ = 0; chia √4 = 2: s = (2, 0). a = (σ(2), σ(−2)) = (0.8808, 0.1192). o = a = (0.8808, 0.1192) vì v₁, v₂ là hai vector đơn vị. g = o − t = (0.8808, −0.8808), L = ½·2·0.8808² ≈ 0.776. Với hai key, ∂L/∂s₁ = a₁(v₁ − o)·g = a₁a₂(v₁ − v₂)·g (vì v₁ − o = a₂(v₁ − v₂)) = 0.8808·0.1192·(0.8808 + 0.8808) ≈ 0.185, và ∂L/∂s₂ ≈ −0.185. ∂L/∂q = ½(0.185·k₁ − 0.185·k₂) = 0.0925·(2, 2, −1, 1) ≈ (0.185, 0.185, −0.092, 0.092). (Không hỏi: ∂L/∂k₂ = ½·(−0.185)·q ≈ (−0.092, −0.092, 0, 0).) Dấu hợp lý: gradient descent đi ngược ∂L/∂q, tức q dịch về phía k₂ − k₁, làm s₂ tăng, s₁ giảm, chú ý chuyển sang key 2. (Chấp nhận lệch 0.002 do làm tròn.) *Thang:* 1 điểm s, a, o, L; 1 điểm ∂L/∂s; 1 điểm ∂L/∂q.

(c)(i) ∑ⱼ ∂L/∂sⱼ = ∑ⱼ aⱼ(vⱼ·g) − (o·g)∑ⱼ aⱼ = o·g − o·g = 0. Hệ quả: với mọi vector u, ∂L/∂q = (1/√dₖ)∑ⱼ (∂L/∂sⱼ)(kⱼ − u), nên chỉ hiệu giữa các key quan trọng (cộng cùng một vector vào mọi key làm mọi sⱼ cộng cùng một hằng, softmax không đổi). Với hai key, ∂L/∂q = (∂L/∂s₁)(k₁ − k₂)/√dₖ: nằm trên đường thẳng theo hướng k₁ − k₂, như số ở (b). *Thang:* 1 điểm cho chứng minh và hướng k₁ − k₂.

(c)(ii) s = (6, 0), a = (0.99753, 0.002473), L = a₁² ≈ 0.995 (gần mức tệ nhất 1), nhưng ∂L/∂s₁ = 2a₁²a₂ ≈ 2·0.995·0.00247 ≈ 0.0049, nhỏ hơn khoảng 38 lần so với 0.185 ở (b). Softmax bão hòa: mô hình đang chú ý sai gần như tuyệt đối mà gradient gần như bằng 0, nên học rất chậm để sửa. Đây là lý do chia √dₖ (mục 8.2, ví dụ 8.4): giữ logit cỡ 1 lúc khởi tạo để softmax chưa bão hòa. *Thang:* 1 điểm khi có L, ∂L/∂s₁ đúng cỡ và kết luận bão hòa.

**B2.** (a) Attention: `W_Q` và `W_O` mỗi cái d·d = 1 048 576; `W_K`, `W_V` mỗi cái d·(`n_kv`·dₕ) = 1024·256 = 262 144. Tổng 2·1 048 576 + 2·262 144 = 2 621 440. FFN SwiGLU: 3·d·`d_ff` = 3·1024·2816 = 8 650 752; hai RMSNorm 2·1024 = 2 048. Một khối: 11 274 240. Toàn mô hình: 24·11 274 240 + 2·32 000·1024 (embedding và `lm_head`) + 1 024 (norm cuối) = 270 581 760 + 65 536 000 + 1 024 = **336 118 784** (khoảng 336M). *Thang:* 1 điểm attention (đúng chiều của K, V khi có GQA); 1 điểm FFN và norm; 1 điểm tổng.

(b) N = 336 118 784 − 32 768 000 = 303 350 784. 6N ≈ 1.820·10⁹; 12·L·d·n = 12·24·1024·2048 ≈ 6.04·10⁸. FLOPs mỗi token ≈ 2.42·10⁹ (attention chiếm khoảng 25%). Tổng ≈ 2.42·10⁹·6.7·10⁹ ≈ 1.62·10¹⁹ FLOPs (không hỏi: ở 10¹⁴ FLOP/s là 1.62·10⁵ giây, khoảng 45 giờ). GQA không đổi số hạng 12·L·d·n: QKᵀ và AV vẫn tính cho đủ 16 head query. *Thang:* 1 điểm N đúng quy ước; 1 điểm FLOPs mỗi token; 1 điểm tổng FLOPs.

(c)(i) KV cache mỗi token = 2·L·`n_kv`·dₕ·2 byte = 2·24·4·64·2 = 24 576 byte (24 KiB). Batch 16, T = 2048: 24 576·2048·16 = 805 306 368 byte = 768 MiB. Attention đầy đủ (16 head K/V): gấp 4, tức 3 GiB, lớn hơn cả trọng số bf16 (khoảng 641 MiB). (ii) Decode xử lý một token mỗi bước nên bị chặn bởi băng thông bộ nhớ: mỗi bước phải đọc toàn bộ KV cache của mọi chuỗi, và với ngữ cảnh dài phần này chiếm phần lớn số byte phải đọc; tăng batch không giúp phần này vì mỗi chuỗi có cache riêng (mục 8.5). GQA giảm byte phải đọc và bộ nhớ chiếm 4 lần, nên chứa được batch lớn hơn và mỗi bước nhanh hơn. Cái giá: các head query trong một nhóm buộc dùng chung K, V, năng lực biểu diễn giảm; chất lượng có thể kém MHA chút ít, nên thường phải huấn luyện với GQA từ đầu hoặc huấn luyện thêm (uptraining) khi chuyển một checkpoint MHA sang GQA. Ainslie và cộng sự (2023) uptrain với khoảng 5% compute tiền huấn luyện; trên T5-XXL, GQA-8 đạt trung bình 47.1 so với 47.2 của MHA và 46.6 của MQA, với thời gian mỗi mẫu 0.28 s so với 1.51 s của MHA (Bảng 1 của bản HTML arXiv 2305.13245v3, người phản biện mở ngày 17/11/2026). Đó là mô hình encoder–decoder, chưa phải LLM decoder-only; chương 8 không đo mức kém. *Thang:* 1 điểm mỗi ý.

**B3.** (a) ELBO = 𝔼 theo q của log p(x|z) − KL(q ‖ p(z)). Với log p(x|z) = −½ ln(2πs²) − (x − z)²/(2s²) và z = μ + σε:

$$
\mathbb{E}_q\big[(x - z)^2\big] = (x - \mu)^2 - 2(x - \mu)\,\mathbb{E}[z - \mu] + \mathbb{E}\big[(z - \mu)^2\big] = (x - \mu)^2 + \sigma^2 .
$$

KL tới N(0, 1) lấy từ công thức đã cho với μ₂ = 0, σ₂ = 1: ln(1/σ) + (σ² + μ²)/2 − ½ = ½(σ² + μ² − 1 − ln σ²). Ghép lại ra công thức đề cho. *Thang:* 1 điểm kỳ vọng của số hạng tái tạo có bước (x − μ)² + σ²; 1 điểm KL; 1 điểm ghép đúng dấu.

(b) ∂/∂μ: (x − μ)/s² − μ = 0, nên μ* = x/(1 + s²) = 2/(4/3) = 1.5. ∂/∂σ²: −1/(2s²) − ½ + 1/(2σ²) = 0, nên 1/σ² = 1 + 1/s² = 4, σ²* = s²/(1 + s²) = 1/4. Posterior thật (Gauss liên hợp: độ chính xác 1 + 1/s² = 4, trung bình (x/s²)/4 = 1.5) là N(1.5, 1/4): trùng q*. Vì log p(x) = ELBO + KL(q ‖ p(z|x)) và họ Gauss chứa đúng posterior, cực đại ELBO là kéo KL đó về 0, nên khe hở bằng 0 tại q*. Khe hở tại q = N(1, 1) bằng KL(N(1, 1) ‖ N(1.5, 0.25)) = ln(0.5/1) + (1 + 0.25)/(2·0.25) − ½ = −0.6931 + 2.5 − 0.5 = 2 − ln 2 ≈ 1.307 nat. (Cách khác cũng đủ điểm: ELBO(q*) − ELBO(1, 1) = (C − 2.193) − (C − 3.5) ≈ 1.307, với C là hằng số chung.) *Thang:* 1 điểm μ*; 1 điểm σ²* kèm so với posterior và lý do khe hở 0; 1 điểm khe hở 1.307.

(c) Cực tiểu ((x − μ)² + σ²)/(2s²) + β·½(σ² + μ² − 1 − ln σ²). Theo μ: −(x − μ)/s² + βμ = 0, nên μ = x/(1 + βs²). Theo σ²: 1/(2s²) + β/2 − β/(2σ²) = 0, nên σ² = βs²/(1 + βs²). Với β = 4: μ = 2/(1 + 4/3) = 6/7 ≈ 0.857, σ² = (4/3)/(7/3) = 4/7 ≈ 0.571, gần prior hơn posterior N(1.5, 0.25). β → ∞: μ → 0, σ² → 1, q → p(z): KL (rate) về 0, z không mang thông tin gì về x, tái tạo tệ nhất (distortion cao), cùng cơ chế với posterior collapse. β → 0: σ² → 0, μ → x: q thành một điểm, mã mang tối đa thông tin, tái tạo tốt nhất nhưng KL → ∞, mã không còn nằm trong vùng prior nên lấy mẫu từ prior cho rác (autoencoder thường). *Thang:* 1 điểm công thức theo β và số với β = 4; 1 điểm hai giới hạn diễn giải đúng theo rate–distortion.

**B4.** (a) Y[0, 0] = 1·1 + 0·(−1) + (−1)·2 + 3·0 = −1; tương tự Y có các hàng (−1, 4), (0, 4). ∂L/∂W[u, v] = ∑ᵢⱼ G[i, j]·X[i + u, j + v] (tương quan của X với G): các hàng (−2, 3), (4, 5); ví dụ ∂L/∂W[0, 0] = 1·1 + 2·0 + 0·(−1) + (−1)·3 = −2. ∂L/∂X[m, n] = ∑ᵢⱼ G[i, j]·W[m − i, n − j] (tương quan đầy đủ của G với W lật, mục 7.1). Ô giữa: ∂L/∂X[1, 1] = G₀₀W₁₁ + G₀₁W₁₀ + G₁₀W₀₁ + G₁₁W₀₀ = 0 + 4 + 0 − 1 = 3. Ô góc: chỉ một số hạng hợp lệ, ∂L/∂X[0, 2] = G₀₁W₀₁ = 2·(−1) = −2. (Cả ma trận, không hỏi: các hàng (1, 1, −2), (2, 3, 1), (0, −2, 0).) *Thang:* 1 điểm Y; 1 điểm ∂L/∂W; 1 điểm hai ô của ∂L/∂X.

(b) conv1: ⌊(64 + 2 − 3)/2⌋ + 1 = 32; conv2: 32; pool: 16; conv3: 16 + 4 − 2·2 − 1 + 1 = 16. Tham số của conv3: 32·64·9 + 64 = 18 496 (cả mạng, không hỏi: 896 + 9 248 + 18 496 = 28 640; pool không có tham số). Receptive field theo r ← r + d(k − 1)j, j ← j·s, khởi đầu r = j = 1: conv1 r = 3, j = 2; conv2 r = 3 + 2·2 = 7, j = 2; pool r = 7 + 1·2 = 9, j = 4; conv3 r = 9 + 2·2·4 = 25, j = 4. Vậy (RF, jump) = (25, 4). *Thang:* 1 điểm kích thước; 1 điểm tham số; 1 điểm RF và jump (cập nhật r bằng j cũ, dilation nhân vào).

(c)(i) ⟨G, Y⟩ = −1 + 8 + 0 − 4 = 3; ⟨∂L/∂W, W⟩ = −2 − 3 + 8 + 0 = 3; nên ⟨∂L/∂X, X⟩ cũng bằng 3 mà không cần tính (kiểm bằng ma trận đầy đủ: 1 + 0 − 4 − 2 + 9 + 1 + 0 − 2 + 0 = 3). Lý do: đặt L = ⟨G, Y⟩ với G cố định. Y tuyến tính theo X (W cố định) và tuyến tính theo W (X cố định), nên L là hàm tuyến tính của X và ⟨∇ₓL, X⟩ = L (định lý Euler cho hàm thuần nhất bậc 1); tương tự với W. Nói cách khác, ánh xạ G ↦ ∂L/∂X là ánh xạ liên hợp của X ↦ Y, đúng nghĩa "conv chuyển vị". *Thang:* 1 điểm hai số và ⟨∂L/∂X, X⟩ = 3, kèm lý do tuyến tính hoặc liên hợp.

(c)(ii) Ảnh 63: ⌊(63 + 2 − 3)/2⌋ + 1 = 31 + 1 = 32, cũng 32. ConvTranspose từ 32: (32 − 1)·2 − 2 + 3 = 63 khi `output_padding` = 0; muốn 64 phải đặt `output_padding` = 1. Phép lấy sàn trong công thức kích thước của conv có stride 2 làm hai kích thước vào (63 và 64) cho cùng kích thước ra, nên ánh xạ ngược từ 32 không biết phải trả về 63 hay 64; backward của conv1 biết vì nó nhớ kích thước đầu vào thật (64), còn một lớp `ConvTranspose2d` đứng riêng thì phải được báo bằng `output_padding`. *Thang:* 1 điểm cho 32, `output_padding` = 1 và lý do lấy sàn.

**B5.** (a) Hiệu M − B: (0.6, 0.2, 0.5, 0.9, 0.3, 0.5); trung bình 0.5; độ lệch so với trung bình (0.1, −0.3, 0, 0.4, −0.2, 0), tổng bình phương 0.30, phương sai mẫu 0.30/5 = 0.06, độ lệch chuẩn √0.06 ≈ 0.245; SE = 0.245/√6 = 0.1. CI 95% = 0.5 ± 2.571·0.1 = [0.24, 0.76], không chứa 0. Coi độc lập: SE = √((1.07² + 1.14²)/6) = √0.407 ≈ 0.64 (cho sẵn); CI = 0.5 ± 2.228·0.64 ≈ [−0.92, 1.92], chứa 0. Khác nhau vì hai cột tương quan rất mạnh (cùng seed cho cùng khởi tạo và thứ tự dữ liệu; hệ số tương quan khoảng 0.98): phần lớn dao động giữa các seed là chung và triệt tiêu trong hiệu. So ghép cặp là đúng ở đây; coi độc lập là vứt thông tin. *Thang:* 1 điểm hiệu, độ lệch chuẩn và SE; 1 điểm CI ghép cặp; 1 điểm CI độc lập kèm lý do tương quan.

(b) Ghép cặp: n ≥ (2.80·σd/δ)² = (2.80·0.245/0.3)² ≈ 2.29² ≈ 5.2, tức 6 seed. Độc lập: mỗi bên n ≥ 2·(2.80·1.1/0.3)² ≈ 2·10.3² ≈ 211 seed. Khác nhau ở hai chỗ: hiệu của hai trung bình độc lập có phương sai 2σ²/n (hệ số 2), còn hiệu ghép cặp có phương sai σd²/n, với σd là độ lệch chuẩn của hiệu; và σd (0.245) nhỏ hơn σ (1.1) rất nhiều nhờ tương quan. Công thức dùng z nên lạc quan khi n nhỏ: với n = 6, phân vị t (2.571) lớn hơn 1.96 nên độ mạnh thật chỉ khoảng 0.67 (mô phỏng trong script kiểm), cần thêm vài seed để thật sự đạt 0.8. *Thang:* 1 điểm n ghép cặp; 1 điểm n độc lập; 1 điểm giải thích hai khác biệt và hướng lệch.

(c)(i) CI theo seed chỉ gom phương sai huấn luyện (khởi tạo, thứ tự dữ liệu) trên **một** tập test cố định. Nó bỏ sót phương sai do lấy mẫu tập test: riêng accuracy cỡ 72% trên 2 000 câu đã có SE nhị thức khoảng 1.0 điểm, lớn gấp nhiều lần SE 0.1 ở (a). Với **hiệu** trên cùng tập test thì phần dao động chung triệt tiêu một phần, nhưng phần còn lại phải đo: paired bootstrap trên các câu test (như C2), gộp dự đoán qua các seed, hoặc McNemar trên cùng tập; tốt hơn nữa là đánh giá trên một tập test thứ hai. (Phương sai do chọn tập train là nguồn thứ ba, mục 11.3, nếu kết luận muốn khái quát ra dữ liệu train khác.) (ii) Chọn biến thể tốt nhất trên test là best-of-4: số báo cáo bị thổi lên (thiên lệch chọn lọc, A15) và là đa kiểm định, nên mức 5% không còn đúng. Sửa: chọn biến thể trên tập dev, chỉ chạy test một lần cho biến thể đã chọn; hoặc báo cả 4 biến thể với hiệu chỉnh đa kiểm định (Bonferroni: 0.05/4 = 0.0125 cho mỗi phép so); và ghi trong bài số biến thể đã thử. *Thang:* 1 điểm mỗi ý.

## Số liệu đã kiểm

Mọi số trong đề và lời giải được tính lại bằng script, kiểm ngày 2026-11-17 với Python 3.13, NumPy 2.x, SciPy 1.18, PyTorch 2.14 (CPU, float64). PyTorch chỉ dùng trong script kiểm của người soạn; bộ chấm của người học chỉ cần NumPy.

- `work/mock2/check_numbers.py`: bảng hằng số, Phần A, Phần B, C1(b), C2(b). Đạo hàm kiểm bằng torch autograd và sai phân trung tâm; các ý mô phỏng có ghi seed.
- `work/mock2/check_partC.py`: Phần C. Chạy lời giải mẫu qua bộ chấm `labs/thi-thu-2/cham_phan_c.py` (16/16, chia chương đúng thang), chạy khung trống `labs/thi-thu-2/bai_lam_mau.py` (0/16, không lỗi), và chạy 30 bài làm cố ý trong `work/mock2/grader_tests/` (8 bài đúng viết khác cách, 2 bài đúng nhưng có vòng lặp, 20 bài sai cố ý) so với điểm thang dự kiến.
- `work/mock2/check_exam.py`: rà cấu trúc đề: tổng điểm, tổng phút, bảng chấm theo chương, ký hiệu (không `^`, `_`, `$` hay lệnh LaTeX ngoài khối công thức và code); dòng quy tắc và mọi phần trước Đề bài không chứa đáp số nào của Lời giải (quét bằng `work/mock2/scan_leaks.py`); câu (b) của C1 không nêu sẵn hàng đầu của A; trùng câu chữ với chương 1–11 (quét 8-gram theo từ trên phần Đề bài, bỏ số và code: 19 cụm trùng, đều là cụm định nghĩa như quy ước đếm FLOPs của chương 8, định nghĩa top-p và nghiệm đóng KL của chương 9, "cận dưới của thông tin tương hỗ", "một lần chạy trên một tập test"); và trùng bộ số: bộ số đặc trưng của cả 27 câu được tìm trong toàn văn chương 1–11 (gồm ví dụ, bài tập, tính nhanh, mini-test), không bộ nào xuất hiện. Phép kiểm bộ số so khớp chuỗi, nên chỉ bắt trùng đúng con số; trùng khuôn thì ghi tay ở dưới.

**Trùng khuôn có chủ đích** (bộ số khác, cách hỏi gần sách): A14 cùng khuôn ví dụ 11.2 và bài ★★6 chương 11; C2(a) là Lab 5 chương 3 làm lại trong giờ (ghi ở Bản đồ); C1(a) là Lab 1 chương 8 thêm trục batch; B3 cùng thiết lập với ví dụ 10.2 (prior N(0, 1), likelihood Gauss quanh z) nhưng khác phương sai, quan sát và câu hỏi; A4 cùng loại với ví dụ 9.4 và mini-test câu 2 chương 9 nhưng logit khác và có hai token cùng logit. Bản trước của A3(a) (N = 8, bản đồ 14 × 14) và A19 (B = 32 → 256) trùng đúng bộ số của tính nhanh 10 chương 7 và mini-test câu 5 chương 4; đã đổi.

Tái lập: từ thư mục gốc repo, chạy ba script trên bằng `/home/claude/venv/bin/python`; mỗi script in "ĐẠT" ở dòng cuối.

| Câu | Giá trị | Cách kiểm | Kết quả |
| --- | --- | --- | --- |
| Hằng số | ln 2, ln 3, e, e⁻¹, e², e⁻², e¹·¹, 2¹·¹, e⁶, e⁷, e⁸, σ(±2), σ(±6), các căn, z, t₅, t₁₀, 2.80, 3.84; E[max của 10 biến chuẩn] = 1.539 (tích phân số 1.53875) | `math`, `scipy.stats`, tích phân số | khớp |
| A1 | 32 × 20 × 20; 12 832 tham số; 5 120 000 MAC | `nn.Conv2d` | khớp |
| A2 | (3, 2, 5, 4); (1, 3, 5, 1); (0, 3, 2, 5) | `F.max_pool1d` | khớp |
| A3 | 784; 64 và 64; w′ = b′ = 0.6; x = 2 cho 1.8 | `nn.BatchNorm2d`, tính trực tiếp | khớp |
| A4 | T = 1: p = (0.3995, 0.3995, 0.1470, 0.0541), giữ 3, chuẩn hóa (0.4223, 0.4223, 0.1554); T = 0.5: cộng dồn 0.4643, 0.9287, giữ 2, chuẩn hóa (0.5, 0.5) | NumPy, cài top-p theo định nghĩa | khớp |
| A5 | perplexity 4 và 2¹·¹ = 2.1435; bpb 0.50 và 0.55 | `math` | khớp |
| A6 | 6·10²⁰; N = √5·10⁹; D = 4.47·10¹⁰ | `math` | khớp |
| A7 | √2, −1/√2, tổng 0; 1/32 | NumPy | khớp |
| A8 | 2 752 512; 48 GB; nền bf16 6 GB | tính trực tiếp | khớp |
| A9 | 11/14 = 0.7857; 0.6836 | `math.comb`, mô phỏng 200 000 lần (seed 0) cho 0.786 | khớp |
| A10 | Z = 1.3437; π* = (0.4046, 0.2977, 0.2977); phần thưởng ngầm (0.7046, −0.2954, −0.2954) = r − ln Z; cộng hằng vào r không đổi π* | NumPy; π* cũng là nghiệm cực đại 𝔼[r] − KL tìm bằng tối ưu số | khớp |
| A11 | KL 1.4431; σ = (0.5, 1) | `torch.distributions.kl_divergence` | khớp |
| A12 | ln 512 = 6.238; cận 4.338; N = 1 097 | `F.cross_entropy` trên ma trận hằng | khớp |
| A13 | 1.25; xấp xỉ 1.184 | 300 lần mô phỏng FID với d = 32, N = 500 (seed 1) cho trung bình lệch dưới 0.03 | khớp |
| A14 | n ∈ {9, 18, 27}; n = 50 không khớp | duyệt mọi k/n | khớp |
| A15 | 0.4·1.539 ≈ 0.616 | mô phỏng 200 000 lần (seed 2) | khớp |
| A17 | trị riêng 6, 1; PC1 ±(2, 1)/√5; 6/7; tọa độ ±√5 | `eigh` | khớp |
| A18 | CE ln 4; gradient (−0.675, 0.225, 0.225, 0.225); gradient 0 tại hiệu logit ln 37 | `F.cross_entropy(label_smoothing=0.1)` | khớp |
| A19 | √8 = 2.83; chi phí 8 lần; η = 0.16; heavy-ball β = 0.9 với ηλ = 3 vẫn hội tụ, trần 2(1 + β) = 3.8 | mô phỏng 20 000 lần (seed 3); bán kính phổ | khớp |
| A20 | (2.667, 0, 0, 8); eval đồng nhất; 𝔼 giữ nguyên | `nn.Dropout` | khớp |
| C1(b) | hàng đầu (1, 0, …); hàng bị che hết: nan với −∞, đều 1/T với −1e9; KV cache trùng từng hàng | NumPy, T = 4 | khớp |
| C2(b) | hiệu 0.022; SE ghép cặp 0.00903, CI [0.0043, 0.0397]; SE độc lập 0.01785, CI [−0.0130, 0.0570]; McNemar 5.378 (không hiệu chỉnh 5.902), bản chính xác p < 0.05; bootstrap mẫu cho [0.004 đến 0.005, 0.039 đến 0.040] với 5 seed | `scipy.stats.binomtest`, lời giải mẫu | khớp |
| B1 | s = (2, 0); L = 0.7758; ∂L/∂s = (±0.1850); ∂L/∂q = (0.185, 0.185, −0.0925, 0.0925); ∂L/∂k₂ = (−0.0925, −0.0925, 0, 0); ở logit (6, 0): L = 0.9951, ∂L/∂s₁ = 0.00492, nhỏ hơn 37.6 lần | autograd và sai phân trung tâm | khớp |
| B2 | khối 11 274 240 (attention 2 621 440, FFN 8 650 752, norm 2 048); tổng 336 118 784; N = 303 350 784; 2 424 084 480 FLOPs mỗi token; 1.62·10¹⁹; 45.1 giờ; KV 24 576 byte, 768 MiB, MHA 3 GiB; trọng số bf16 641 MiB | đếm bằng `nn.Linear(bias=False)` | khớp |
| B3 | ELBO dạng đóng khớp Monte Carlo 2 triệu mẫu (seed 5) tại N(1, 1); μ* = 1.5, σ²* = 0.25 (LBFGS); posterior trên lưới N(1.5, 0.25); khe hở 2 − ln 2 = 1.3069; β = 4: (6/7, 4/7) từ đúng hàm mục tiêu viết trong lời giải (c) | autograd, LBFGS, tích phân số | khớp |
| B4 | Y, ∂L/∂W như lời giải; ∂L/∂X[1, 1] = 3, ∂L/∂X[0, 2] = −2; ba tích trong bằng 3; kích thước 32, 32, 16, 16; conv3 18 496 tham số (cả mạng 28 640); (RF, jump) = (25, 4); ảnh 63 cũng ra 32; ConvTranspose ra 63, thêm `output_padding` = 1 thì trùng gradient theo đầu vào của conv1 | `F.conv2d`, autograd, `F.conv_transpose2d` | khớp |
| B5 | hiệu trung bình 0.5, phương sai 0.06, SE 0.1, CI [0.243, 0.757]; độ lệch chuẩn hai cột 1.071 và 1.145; SE độc lập 0.638, CI [−0.92, 1.92]; Welch 9.96 bậc tự do, p = 0.45; t ghép cặp p < 0.01; tương quan 0.978; n = 5.23 và 210.8; độ mạnh thật của t ghép cặp ở n = 6 khoảng 0.67 (mô phỏng 40 000 lần, seed 6) | `scipy.stats` | khớp |

A16 là câu chọn, không có số để kiểm. Điểm: A 40, C 20, B 40, tổng 100; bảng chấm theo chương cộng 14 + 23 + 17 + 14 + 19 + 13 = 100; mọi chương 7–11 từ 14 điểm trở lên.

**Bộ chấm bám thang điểm.** Chạy `labs/thi-thu-2/cham_phan_c.py` trên các bài làm trong `work/mock2/grader_tests/` (trong `work/mock2/check_partC.py`). Mỗi bài chỉ viết một hàm, điểm là phần tự động của hàm đó trên 8:

| Bài làm | Thang dự kiến | Bộ chấm cho |
| --- | --- | --- |
| C1 đúng, viết khác cách (`np.where` với −inf; thay ô bị che bằng −1e9 và log-sum-exp với `einsum`; max trên ô được phép rồi nhân mặt nạ 0/1; `scipy.special.softmax` với mặt nạ cộng) | 8 | 8, 8, 8, 8 |
| C1 lặp theo vị trí | 8 tự động, trừ tay 2 | 8, có in cảnh báo vòng lặp |
| C1 quên mặt nạ / che sau softmax | 6 (mất mặt nạ) | 6, 6 |
| C1 chia √d | 5 (mất giá trị) | 5 |
| C1 không trừ max / trừ max cả tensor / max trước khi che | 6 (mất ổn định số) | 6, 6, 6 |
| C1 softmax theo trục −2 / che cả đường chéo bằng −∞ | 1 (chỉ còn shape) | 1, 1 |
| C1 che cả đường chéo, điền −1e9 | 3 (mất mặt nạ và giá trị) | 3 |
| C1 đổi sang float32 | 3 (mất giá trị và ổn định số) | 3 |
| C1 dùng `K.T` trên tensor 3 chiều | 0 (báo lỗi) | 0 |
| C2 đúng, viết khác cách (ma trận chỉ số; lặp theo lần bootstrap với `rng.choice`; bootstrap basic; diff ± 1.96·std bootstrap) | 8 | 8, 8, 8, 8 |
| C2 lặp theo câu test | 8 tự động, trừ tay 2 | 8, có in cảnh báo vòng lặp |
| C2 lấy mẫu A và B độc lập | 5 (mất ghép cặp) | 5 |
| C2 phần trăm / B − A | 7 (mất hiệu đúng) | 7, 7 |
| C2 phân vị 5 và 95 | 6 (mất mức) | 6 |
| C2 nhầm thang `np.percentile` | 3 (mất ghép cặp và mức) | 3 |
| C2 không truyền seed / công thức chuẩn thay bootstrap | 7 (mất seed và lấy mẫu lại) | 7, 7 |
| C2 lấy mẫu không hoàn lại / tạo lại `default_rng(seed)` trong vòng lặp | 2 | 2, 2 |

Độ bền của dung sai C2 (cũng trong `check_partC.py`): với lời giải mẫu chạy 60 seed khác nhau, tỉ số độ rộng của test ghép cặp nằm trong 0.95–1.04 lần giá trị kỳ vọng (dung sai ±15%), độ rộng của test mức nằm trong 0.97–1.03 lần 2·1.96·SE (dung sai ±7%); bản phân vị 5 và 95 cho 0.83–0.87 và bản lấy mẫu độc lập cho tỉ số 1.83–2.00, đều ngoài dung sai. Test "tất định theo seed" nay đòi thêm khoảng có độ rộng dương, để bắt lỗi tạo lại bộ sinh trong vòng lặp (trước khi vá, bài đó được 3 thay vì 2).

**Ước lượng thời gian từng câu** [theo thang của người phản biện (vòng 1: 145 phút cho bản đầu), trừ phần đã cắt; chưa đo trên người thật]. "Người học" là sinh viên năm 3 vừa học xong chương 7–11, làm lần đầu; "giải nhanh" là người vừa ôn kỹ Phần II. Các chỗ đã cắt, không đổi điểm: B4(a) chỉ hỏi hai ô của ∂L/∂X và B4(c)(i) không cần cả ma trận; B4(b) chỉ đếm tham số conv3; C2(b)(i) dùng SE ghép cặp dạng đóng thay vì dựng mảng và chạy hàm; A14 giới hạn n ≤ 30; B1(b) bỏ ∂L/∂k₂; B2(b) bỏ đổi ra giờ; B5(a) cho sẵn SE coi độc lập; A5 cho NLL theo bit nên không phải chia cho ln 2; A3(b) bỏ bước kiểm bằng x = 2.

| Câu | Người học | Giải nhanh | Ghi chú |
| --- | --- | --- | --- |
| Đọc đề | 2 | 2 | |
| A1, A2, A3, A4, A5, A9, A10, A11, A13 | 2.5 mỗi câu (22.5) | 1.5 mỗi câu (13.5) | A4, A10 có vài phép chia; dạng biểu thức đủ điểm |
| A6, A7, A14, A17, A18 | 3 mỗi câu (15) | 2 mỗi câu (10) | A6 có căn tay; A14 phải tự thuyết phục không còn n nào khác |
| A8, A12, A15, A19 | 2 mỗi câu (8) | 1.5 mỗi câu (6) | |
| A16, A20 | 1 và 1.5 (2.5) | 0.5 và 1 (1.5) | |
| **Phần A** | **48** | **31** | |
| C1 | 15 ((a) 12 gồm chạy thử với batch; (b) 3) | 10 | khung chữ ký có sẵn |
| C2 | 13 ((a) 9.5 gồm chạy thử; (b) 3.5) | 9 | (b) tính bằng Python |
| **Phần C** | **28** | **19** | |
| B1 | 12.5 ((a) 5, (b) 4.5, (c) 3) | 8.5 | |
| B2 | 11 ((a) 4.5, (b) 3.5, (c) 3) | 7 | nhân số lớn bằng tay, được để dạng tích |
| B3 | 11.5 ((a) 3.5, (b) 5, (c) 3) | 7 | |
| B4 | 10.5 ((a) 4, (b) 4, (c) 2.5) | 7.5 | |
| B5 | 9.5 ((a) 3, (b) 3, (c) 3.5) | 6 | |
| **Phần B** | **55** | **36** | |
| **Tổng** | **133** | **88** | giờ thi 120 |

**Cần bạn xác nhận.**

1. Cơ cấu 40/20/40, thứ tự A → C → B và 120 phút giữ như Đề thi thử 1, là lựa chọn của người soạn; định dạng thật không công khai.
2. Bảng chương 14/23/17/14/19/13 theo đề xuất của người phản biện: A4 (mặt nạ nhân quả với padding trái, trùng ví dụ 8.14 và trùng hiện tượng với C1(b)(i)) và A11 cũ (một bước DDPM) được thay bằng hai câu chương 9 (giải mã, nghiệm đóng KL và DPO); câu VAE chuyển từ A10 sang A11. Diffusion không còn được hỏi. Câu giải mã dùng logit (1, 1, 0, −1) chứ không phải (2, 1, 0, −1) như phản biện đề xuất, vì (2, 1, 0, −1) ở T = 0.5 cho đúng các số của ví dụ 9.4 (0.8647, 0.9817) và gần mini-test câu 2 chương 9. Nếu muốn giữ diffusion, có thể thay A13 (FID) bằng câu DDPM cũ.
3. Ước lượng 133 phút cho người học mục tiêu và 88 phút cho người giải nhanh chưa đo trên người thật; cắt thêm để xuống dưới 130 sẽ phải bỏ ý có điểm. Sau lần làm đầu, báo số phút thật từng phần để chỉnh cột Phút và mốc giờ.
4. Ngưỡng 75% và 50% theo chương, mốc 75 và 60 theo tổng, mốc chênh 15 điểm cho tốc độ là heuristic, giữ nguyên từ Đề thi thử 1 để hai lần đo so được với nhau. w24 đã có 3 giờ cho làm và chấm đề (`w24-mock`); tổng w24 là 13.5 giờ. Lịch w25–w26 (13.75 giờ mỗi tuần) chỉ còn khoảng 1.25 giờ mỗi tuần dưới trần 15 giờ cho việc ôn khi tổng 60–74.
5. Phần nhắc lại Phần I chỉ 13 điểm và không có chương 5; dòng "Phần I" của bảng chấm chỉ báo động chung. Mỗi chương 1, 2, 4, 6 có đúng một câu 2 điểm, không đủ để chẩn đoán từng chương.
6. Bảng "Câu sai → ôn gì" trỏ tới số ví dụ, số bài, mini-test của chương 1–11 tại ngày kiểm (2026-11-17); nếu các chương được sửa và đánh số lại, bảng cần rà lại.
7. Đề chưa được đăng ký trong `book/chapters.yaml` và `book/_toc.yml`, nên trang "Sách → Đề thi thử 2" mà `w24-mock` trỏ tới chưa có trong app. Việc này thuộc bước merge, không thuộc file đề.
