# Chương 0 — Bài chẩn đoán đầu vào

## Cách dùng bài chẩn đoán

Bài này có 30 câu, làm trong 120 phút (bằng thời lượng Đề thi thử 1), trước khi bạn học sâu (hoặc ở tuần đầu). Nó không phải đề thi và không có điểm đậu rớt. Việc duy nhất nó làm là cho bạn biết chương nào đã vững, chương nào cần đọc kỹ, chương nào phải học từ đầu, để chia 16 giờ của mỗi chương cho đúng chỗ. Phủ chương 1–11, và chỉ một phần mục của mỗi chương (xem cột "Mục không được hỏi" ở bảng chấm). Chương 12–14 không được đo: mỗi chương đó sẽ có bài tự kiểm riêng ở đầu chương.

**Cách làm.**

- Một buổi liền 120 phút, bàn học, giấy nháp, điện thoại để xa. Không tra sách, không máy tính bỏ túi. Phần N (code) được chạy Python để tự kiểm, nhưng chỉ sau khi đã viết xong dự đoán và code ra giấy. **Chấm Phần N trên bản viết tay, trước khi chạy**; kết quả chạy chỉ để bạn thấy mình sai chỗ nào (gắn nhãn lỗi theo bản viết tay). Thời gian chạy tính trong 17 phút của phần N.
- Đáp số được phép để dạng biểu thức, vẫn đủ điểm: 2 mũ 1.75, σ(1) = 1/(1 + e⁻¹), 24·10⁹/2³⁰ GiB. Chỉ khi đề hỏi so sánh hai số thì mới cần xấp xỉ đến 2 chữ số.
- Gợi ý thời gian, theo ước lượng của người soạn, chưa đo trên người làm thật [ước lượng]: phần A (câu 1–24) 87 phút, phần N (câu 25–27) 17 phút, phần E (câu 28–30) 14 phút, 2 phút đọc đề, tổng 120 phút. Cột Phút ở bản đồ câu hỏi cộng đúng ra các số này. Câu nào quá thời gian ghi ở cột đó cộng thêm 2 phút thì bỏ, ghi "bỏ" rồi làm tiếp.
- **Ba lượt, hai con số.** Lượt một là 120 phút. Hết giờ mà còn câu chưa làm hoặc làm dở: gạch một vạch ngang ở chỗ đang dừng, đổi sang bút khác màu, làm tiếp tối đa 30 phút (lượt hai). Câu nào vẫn chưa làm sau lượt hai thì làm nốt, không tính giờ, bằng bút màu thứ ba (lượt ba). Ghi riêng hai số: **điểm trong giờ** (chỉ bút lượt một) và **điểm cuối** (mọi bút, sau cả ba lượt). **Chỉ điểm cuối được dùng để tra bảng chấm theo chương**, vì câu bị bỏ do hết giờ thường nằm ở cuối đề (chương 9, 10, 11, phần N và E) và không có nghĩa là chương đó yếu. Điểm trong giờ chỉ để đo tốc độ: hiệu "điểm cuối trừ điểm trong giờ" là tín hiệu tốc độ, tách khỏi tín hiệu kiến thức (xem mục Đọc theo nhãn lỗi).
- **Đừng học trước khi làm.** Bài chỉ cho tín hiệu thật khi bạn chưa ôn. Lịch hai tuần đầu (w00–w01 trong `plan/weeks.yaml`) xếp chương 1 mục 1.1–1.4 vào đúng khoảng thời gian bạn làm bài này, nên nếu bạn đã đọc mục nào của chương nào, vẫn làm đủ 30 câu nhưng ghi "đã học" cạnh nhóm đó; nhóm "đã học" được xử lý riêng ở mục Đọc kết quả.

**Thang điểm.** Mỗi câu 2 điểm, tổng 60. Lời giải ghi rõ ý nào đáng 1 điểm. Chấm nghiêm: có đáp số mà không có lý do (khi đề yêu cầu giải thích) thì mất điểm của ý giải thích đó (ý tính đúng vẫn được 1 điểm); phân vân thì cho điểm thấp hơn. Không cho nửa điểm.

**Gắn nhãn lỗi.** Với mỗi ý mất điểm, ghi một nhãn vào sổ lỗi: **K** (khái niệm sai hoặc không biết), **T** (hiểu đúng nhưng tính sai), **L** (ngôn ngữ, đọc đề sai, thiếu thuật ngữ). Nhãn K cần đọc lại lý thuyết; nhãn T cần bài tính nhanh; nhãn L cần luồng C. Ý không có điểm trong giờ vì chưa làm kịp nhưng được điểm ở lượt hai hoặc ba: ghi **H** (hết giờ), không ghi K, để không nhầm với kiến thức. Ý vẫn mất điểm ở điểm cuối thì ghi K, T hoặc L như trên.

**Lưu ý trung thực.** Đây là bài mô phỏng do sách soạn, không phải đề của Qualcomm hay VinAI (đề thật không công khai, theo hiểu biết của người soạn). Ngưỡng điểm, số giờ gợi ý và số phút mỗi câu là heuristic hoặc ước lượng, chưa được hiệu chỉnh trên người làm thật. Các điểm này cần bạn xác nhận (xem cuối chương).

## Bản đồ câu hỏi

| Câu | Chương, mục | Chủ đề | Phút [ước lượng] |
| --- | --- | --- | --- |
| 1 | Chương 1, mục 1.2 | Hạng, không gian null, hệ có nghiệm | 4 |
| 2 | Chương 1, mục 1.4–1.5 | Trị riêng, giá trị kỳ dị, số điều kiện | 3 |
| 3 | Chương 2, mục 2.1 | Gradient, Hessian, xấp xỉ bậc 1 | 4 |
| 4 | Chương 2, mục 2.3 và 2.5 | Backprop một neuron, mặt nạ ReLU | 4 |
| 5 | Chương 3, mục 3.1 | Bayes, tỉ lệ nền, cập nhật hai lần | 3 |
| 6 | Chương 3, mục 3.4–3.5 | MLE, cross-entropy, KL | 4 |
| 7 | Chương 4, mục 4.2 | GD trên bài bậc hai, số điều kiện | 5 |
| 8 | Chương 4, mục 4.4 | Momentum và bước đầu của Adam | 3 |
| 9 | Chương 5, mục 5.1 | Gradient của logistic regression | 2 |
| 10 | Chương 5, mục 5.2 | Ridge, chẩn đoán overfit | 3 |
| 11 | Chương 6, mục 6.1–6.2 | Đếm tham số, shape, loss ban đầu | 2 |
| 12 | Chương 6, mục 6.4–6.5 | Khởi tạo He, debug loss lạ | 3 |
| 13 | Chương 7, mục 7.1 và 7.3 | Đếm conv, conv tách theo chiều sâu | 5 |
| 14 | Chương 7, mục 7.2 và 7.5 | Receptive field, patch embedding | 3 |
| 15 | Chương 8, mục 8.2 | Một bước attention tính tay | 4 |
| 16 | Chương 8, mục 8.2–8.3 | Mã hóa vị trí sinusoidal, hoán vị | 3 |
| 17 | Chương 8, mục 8.5 | Tham số và KV cache | 4 |
| 18 | Chương 9, mục 9.1 | BPE, perplexity | 4 |
| 19 | Chương 9, mục 9.2–9.3 | Ngân sách 6ND, Bradley–Terry | 4 |
| 20 | Chương 9, mục 9.4 | LoRA, bộ nhớ tinh chỉnh | 4 |
| 21 | Chương 10, mục 10.1–10.2 | KL Gauss, reparameterization, D* của GAN | 4 |
| 22 | Chương 10, mục 10.3 | Nhiễu hóa và khử nhiễu DDPM | 3 |
| 23 | Chương 11, mục 11.3 | Nhiều seed, khoảng tin cậy | 4 |
| 24 | Chương 11, mục 11.1–11.2 | Phê bình một khẳng định | 5 |
| 25 | Phần N (luồng B; NumPy nền, không có mục riêng trong sách) | Broadcasting, khoảng cách không vòng lặp | 6 |
| 26 | Phần N (Chương 3, mục 3.3) | log-softmax ổn định | 5 |
| 27 | Phần N (luồng B; NumPy nền, không có mục riêng trong sách) | Ba lỗi im lặng | 6 |
| 28 | Phần E (Luồng C; Chương 11, mục 11.1) | Đọc hiểu một abstract | 5 |
| 29 | Phần E (Luồng C; Chương 8, mục 8.2) | Dịch hai câu kỹ thuật sang tiếng Anh | 4 |
| 30 | Phần E (Luồng C; Chương 11, mục 11.4) | Giải thích ngắn bằng tiếng Anh | 5 |

## Đề bài

### Phần A — Toán, ML và deep learning (câu 1–24)

**1.** *(Chương 1, mục 1.2)* A có các hàng (1, 2, 3), (2, 4, 6), (0, 1, 1). (a) Tìm hạng của A và một vector khác 0 trong N(A). (b) Hệ Ax = b có nghiệm không khi b = (5, 10, 2)? Khi b = (1, 1, 1)? Nếu có nghiệm, có bao nhiêu?

**2.** *(Chương 1, mục 1.4–1.5)* S có các hàng (5, 2) và (2, 2). (a) Tìm các trị riêng và hai vector riêng. (b) Từ (a), đọc ra các giá trị kỳ dị, ‖S‖₂, κ(S), và nói S có xác định dương không. Vì sao đọc được mà không cần tính SVD?

**3.** *(Chương 2, mục 2.1)* f(x, y) = x²y + 3y. (a) Tính ∇f và Hessian tại (2, 1). Dùng xấp xỉ bậc 1 ước lượng f(2.1, 0.9) và so với giá trị đúng. (b) (2, 1) có phải điểm dừng không? Hessian ở đó nói được gì về dạng của f?

**4.** *(Chương 2, mục 2.3 và 2.5)* Một neuron ẩn: z = w₁x, h = max(z, 0), ŷ = w₂h, L = (ŷ − t)². Cho x = 2, w₂ = 3, t = 1. (a) Với w₁ = 0.5: chạy forward, rồi backward để có ∂L/∂ŷ, ∂L/∂w₂, ∂L/∂h, ∂L/∂z, ∂L/∂w₁. (b) Với w₁ = −0.5: ∂L/∂w₁ bằng bao nhiêu và điều đó ảnh hưởng thế nào đến huấn luyện?

**5.** *(Chương 3, mục 3.1)* Một xét nghiệm: tỉ lệ mắc bệnh trong dân số 0.5%, độ nhạy 95%, độ đặc hiệu 98%. (a) Người có kết quả dương thì xác suất mắc bệnh là bao nhiêu? (b) Người đó xét nghiệm lại lần hai (độc lập khi biết tình trạng bệnh) và lại dương: xác suất mới? (c) Một câu: vì sao (a) thấp dù xét nghiệm "đúng 95–98%"?

**6.** *(Chương 3, mục 3.4–3.5)* (a) Ba quan sát 1, 3, 8 lấy từ N(μ, σ²). Tính MLE của μ và σ², và ước lượng không chệch của σ². (b) Phân phối thật p = (0.5, 0.25, 0.25), mô hình q = (0.25, 0.25, 0.5). Tính theo bit: H(p), cross-entropy H(p, q), KL(p‖q), và perplexity (bằng 2 mũ cross-entropy tính bằng bit).

**7.** *(Chương 4, mục 4.2)* f(x) = ½(x₁² + 9x₂²), gradient descent với bước η cố định. (a) Ngưỡng trên của η để GD còn ổn định, η tối ưu, và hệ số co sai số mỗi bước ở η tối ưu. (b) Từ x = (1, 1) với η = 0.1: x sau một bước? Hướng nào hội tụ chậm? (c) Xuất phát từ x = (1, 1), ước lượng số bước để sai số của thành phần chậm nhất (|x₁|, hướng λ nhỏ) giảm còn một phần nghìn so với ban đầu, với η tối ưu và với η = 0.1.

**8.** *(Chương 4, mục 4.4)* Quy ước momentum: v ← βv + g, w ← w − ηv. (a) β = 0.9, gradient hằng g = 1, v bắt đầu từ 0: tính v sau 1, 2, 3 bước và giới hạn khi t → ∞. Hệ quả cho learning rate hiệu dụng? (b) Adam bước đầu (t = 1, α = 0.001, ε = 10⁻⁸, có bias correction): độ dời của hai tham số có g = 0.01 và g = −100. Điều này nói gì về thang của gradient?

**9.** *(Chương 5, mục 5.1)* Logistic regression, w = (1, 1), b = 0, mẫu x = (2, −1), nhãn y = 1. Tính z, p = σ(z), log-loss và gradient của log-loss theo w. Dấu của từng thành phần gradient cho biết gì?

**10.** *(Chương 5, mục 5.2)* (a) Hồi quy một chiều không hệ số chặn, x = (1, 2, 3), y = (2, 3, 7), với hàm mất mát sau. Tìm w khi λ = 0 và khi λ = 6.

$$
J(w) = \sum_{i=1}^{3} (y_i - w x_i)^2 + \lambda w^2
$$

(b) Một mô hình đạt accuracy train 99% và validation 70%. Chẩn đoán, nêu hai biện pháp có ích và một biện pháp sẽ không giúp.

**11.** *(Chương 6, mục 6.1–6.2)* MLP 784 → 256 → 10, ReLU ở lớp ẩn, mọi lớp có bias. (a) Số tham số. (b) Batch 32: shape của h (sau ReLU) và của ∂L/∂W₁. (c) Cross-entropy lúc vừa khởi tạo (logit gần 0) xấp xỉ bao nhiêu?

**12.** *(Chương 6, mục 6.4–6.5)* (a) Khởi tạo He cho lớp có fan-in 784: độ lệch chuẩn bằng bao nhiêu? (b) Bài phân loại 10 lớp, loss in ở bước 0 là 7.2. Nêu hai nguyên nhân khả dĩ và một phép thử nhanh để phân biệt chúng.

**13.** *(Chương 7, mục 7.1 và 7.3)* `nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)` có bias, nhận ảnh 16×28×28. (a) Kích thước đầu ra, số tham số, số MAC của trọng số (bỏ qua bias). (b) Thay bằng conv depthwise 5×5 (16 kênh, stride 2, padding 2) rồi conv pointwise 1×1 (16 → 32), cả hai không bias: tham số, MAC, và tỉ lệ MAC so với (a).

**14.** *(Chương 7, mục 7.2 và 7.5)* (a) Chồng lớp: conv 3×3 s1, conv 3×3 s1, max-pool 2×2 s2, conv 3×3 s1. Receptive field và jump của một điểm đầu ra. (b) ViT với ảnh 224×224×3, patch 16×16, d = 768: bao nhiêu token (không tính CLS)? Lớp patch embedding (có bias) có bao nhiêu tham số?

**15.** *(Chương 8, mục 8.2)* dₖ = 4; q = (2, 2, 0, 0); hai key k₁ = (1, 1, 0, 0), k₂ = (0, 0, 1, 1); V có hàng (10, 0) và (0, 5). (a) Tính điểm sau khi chia √dₖ và trọng số attention (cho e² ≈ 7.389). (b) Đầu ra. (c) Nếu quên chia, trọng số là bao nhiêu và vì sao điều đó hại huấn luyện? (Tối đa hai câu.)

**16.** *(Chương 8, mục 8.2–8.3)* Mã hóa vị trí sinusoidal với d = 4, hai cặp tần số 1 và 0.01, xếp theo thứ tự (sin, cos, sin, cos):

$$
PE(pos, 2i) = \sin(\omega_i \, pos), \qquad PE(pos, 2i+1) = \cos(\omega_i \, pos)
$$

(a) Tính PE ở vị trí 2. (b) Self-attention không causal, không có mã hóa vị trí. Hoán vị hai token trong đầu vào thì đầu ra tại mỗi token thay đổi thế nào? Hệ quả cho câu "mèo cắn chó" và "chó cắn mèo"?

**17.** *(Chương 8, mục 8.5)* Transformer d = 768, 12 khối, FFN có độ rộng 4d. Bỏ qua bias, LayerNorm và embedding. (a) Số tham số của một khối (bốn ma trận d × d cho attention, hai ma trận d × 4d cho FFN) và của cả 12 khối. (b) KV cache fp16 mỗi token và cho T = 2048, batch 1 (tính MiB). (c) Một câu: bộ nhớ nào lớn lên theo T² khi tính attention theo cách đơn giản?

**18.** *(Chương 9, mục 9.1)* Kho văn bản gồm "ab" 3 lần, "abc" 2 lần, "bc" 1 lần; mỗi từ khởi đầu tách thành ký tự (tổng 14 token), **không dùng ký hiệu hết từ** `</w>`. (a) Hai phép gộp BPE đầu tiên là cặp nào, và tổng số token sau mỗi phép? (b) Một mô hình gán xác suất 0.5, 0.25, 0.125, 0.125 cho bốn token đúng liên tiếp. NLL trung bình (bit mỗi token) và perplexity?

**19.** *(Chương 9, mục 9.2–9.3)* (a) Mô hình 2·10⁹ tham số huấn luyện trên 4·10¹⁰ token: FLOPs theo 6ND và số token mỗi tham số. (b) Giả sử một GPU duy trì 10 TFLOP/s hiệu dụng (10¹³ FLOP/s; con số giả định để minh họa, không phải số đo của GPU nào): bao nhiêu ngày? Một câu kết luận cho kế hoạch của bạn. (c) Reward model kiểu Bradley–Terry cho r(thắng) = 1.5 và r(thua) = 0.5: xác suất mô hình ưa câu thắng?

**20.** *(Chương 9, mục 9.4)* (a) LoRA hạng 16 trên ma trận 2048 × 4096: số tham số học được và tỉ lệ so với tinh chỉnh toàn phần ma trận đó. (b) Tinh chỉnh toàn phần mô hình 1.5·10⁹ tham số bằng AdamW fp32 (trọng số, gradient, m, v, tức 16 byte mỗi tham số, chưa tính activation): bao nhiêu GiB? (c) Trọng số 4-bit của mô hình đó (0.5 byte mỗi tham số, bỏ qua phần phụ) cỡ bao nhiêu GiB? Hai con số này nói gì về một GPU 6 GiB?

**21.** *(Chương 10, mục 10.1–10.2)* Encoder trả q(z | x) = N(μ = 2, σ = 0.5) một chiều; prior N(0, 1). (a) Tính KL(q‖prior), cho ln 0.25 = −1.3863. (b) Reparameterization với ε = −1: z bằng bao nhiêu? Vì sao code không lấy mẫu z trực tiếp từ N(μ, σ²)? (c) Bài GAN, generator cố định: tại điểm x có pdata(x) = 0.6 và pg(x) = 0.2, discriminator tối ưu D*(x) bằng bao nhiêu?

**22.** *(Chương 10, mục 10.3)* DDPM với ᾱₜ = 0.36. (a) x₀ = 1.5 và ε = −0.5: tính xₜ. (b) Từ xₜ đó, mạng đoán ε̂ = −0.3: x̂₀ bằng bao nhiêu? (c) Mạng "lười" luôn đoán ε̂ = 0: MSE kỳ vọng với ε ~ N(0, 1) là bao nhiêu, và con số đó giúp gì khi đọc đường cong loss?

**23.** *(Chương 11, mục 11.3)* Sáu seed cho accuracy 79.0, 81.0, 80.5, 79.5, 80.0, 80.0. (a) Mean, std mẫu, SE, CI 95% (hệ số t với 5 bậc tự do là 2.571). (b) Baseline được báo cáo 79.2 từ một lần chạy. Có thể nói phương pháp của bạn hơn baseline không? Vì sao?

**24.** *(Chương 11, mục 11.1–11.2)* Một paper viết: "Phương pháp X cải thiện khả năng suy luận": accuracy 82% so với 80% trên benchmark công khai gồm 250 câu, một seed, một mẫu prompt, X sinh nhiều gấp 3 lần số token. (a) SE của accuracy 80% trên 250 câu; chênh lệch 2 điểm lớn hay nhỏ hơn con số đó? (b) Nêu ít nhất ba vấn đề khác về bằng chứng, mỗi vấn đề kèm một thí nghiệm bổ sung.

### Phần N — Lập trình và NumPy nền (câu 25–27)

**25.** *(Phần N, luồng B)* (a) Không chạy, dự đoán shape hoặc báo lỗi: `np.zeros((5,1)) + np.zeros((1,4))`; `np.zeros(3) + np.zeros((3,1))`; `np.zeros((4,3)) + np.zeros(4)`. (b) Viết `sq_dists(X, C)` nhận X cỡ n × d, C cỡ k × d, trả ma trận n × k bình phương khoảng cách Euclid, không dùng vòng lặp Python. Kiểm với X có hàng (0, 0), (3, 4) và C có hàng (0, 0), (1, 1).

**26.** *(Chương 3, mục 3.3)* (a) `np.exp(np.array([1000., 1001., 1002.]))` in ra gì và vì sao? (b) Viết `log_softmax(z)` ổn định số cho vector z. Kết quả với z trên là gì, và tổng exp của nó bằng bao nhiêu?

**27.** *(Phần N, luồng B)* Mỗi đoạn dưới đây có một lỗi chạy không báo lỗi. Với mỗi đoạn: nêu lỗi, giá trị in ra, cách sửa.

```python
# A
logits = np.array([[2.0, 0.1, 0.1], [0.0, 3.0, 0.0], [0.2, 5.0, 0.1]])
y = np.array([0, 1, 1])
print((logits.argmax(axis=0) == y).mean())          # mong đợi: 1.0

# B
X = np.array([[1, 100], [2, 200], [3, 300], [4, 400.]])
Xs = (X - X.mean()) / X.std()
print(Xs.mean(axis=0))                               # mong đợi: [0, 0]

# C
w = np.array([1.0, 2.0])
w_old = w
w += 0.5
print(w_old)                                         # mong đợi: [1. 2.]
```

### Phần E — Tiếng Anh kỹ thuật (câu 28–30)

**28.** *(Luồng C; Chương 11, mục 11.1)* Đoạn dưới là abstract bịa để làm bài. Tên KelpMix do sách đặt; lúc soạn (2026-11-04) tìm không thấy bài nào trùng tên này, nhưng chưa loại trừ hết. Trả lời bằng tiếng Anh, mỗi ý một đến hai câu.

> *[Fabricated abstract for this exercise.] We propose KelpMix, a data-mixing method for fine-tuning small language models. On three classification benchmarks, KelpMix improves average accuracy from 71.2 to 73.0 over a tuned baseline. The gains are largest on the smallest dataset (+3.1) and vanish on the largest (+0.1). All numbers are averaged over three random seeds; variance is not reported.*

(a) What is the main claim? (b) Where is the effect concentrated, and what might that suggest? (c) Name one thing missing from the evidence.

**29.** *(Luồng C; Chương 8, mục 8.2)* Dịch sang tiếng Anh, một câu mỗi ý, dùng thuật ngữ đúng: (a) "Ta chia tích vô hướng cho căn bậc hai của số chiều key để phương sai của điểm không tăng theo số chiều." (b) "Kết quả này chỉ có một seed nên chưa đủ để kết luận mô hình mới tốt hơn."

**30.** *(Luồng C; Chương 11, mục 11.4)* Người phỏng vấn hỏi: "Why do we need a separate test set if we already have a validation set?" Trả lời bằng tiếng Anh trong 3–4 câu, nói như thể đang nói chuyện.

## Bảng chấm theo chương

Cộng điểm các câu (mỗi câu 0, 1 hoặc 2) theo dòng, điền vào cột "Điểm của bạn", rồi tra mức. Số điền vào cột này là **điểm cuối** (sau lượt hai và lượt ba, nếu có). Điểm trong giờ ghi riêng ở một cột phụ tự kẻ, chỉ để đo tốc độ.

| Nhóm | Câu | Điểm tối đa | Điểm của bạn | Vững | Cần đọc kỹ | Học từ đầu | Mục không được hỏi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chương 1 | 1, 2 | 4 | | 3–4 | 2 | 0–1 | 1.1 (ma trận là phép biến đổi), 1.3 (chuẩn, phép chiếu, QR, bình phương tối thiểu) |
| Chương 2 | 3, 4 | 4 | | 3–4 | 2 | 0–1 | 2.2, 2.4 (đồng nhất thức ma trận, forward/reverse mode) |
| Chương 3 | 5, 6 | 4 | | 3–4 | 2 | 0–1 | 3.2 (kỳ vọng, hiệp phương sai); 3.3 chỉ qua câu 26 |
| Chương 4 | 7, 8 | 4 | | 3–4 | 2 | 0–1 | 4.1, 4.3, 4.5 (lồi và KKT, SGD, bậc hai) |
| Chương 5 | 9, 10 | 4 | | 3–4 | 2 | 0–1 | 5.3, 5.4, 5.5 (SVM, cây và ensemble, k-means/GMM/EM) |
| Chương 6 | 11, 12 | 4 | | 3–4 | 2 | 0–1 | 6.3 (PyTorch); 6.5 chỉ qua câu 12(b) (danh mục lỗi im lặng không được hỏi) |
| Chương 7 | 13, 14 | 4 | | 3–4 | 2 | 0–1 | 7.3 (kiến trúc kinh điển; chỉ hỏi conv tách theo chiều sâu, ở câu 13(b)), 7.4 (huấn luyện CNN) |
| Chương 8 | 15, 16, 17 | 6 | | 5–6 | 3–4 | 0–2 | 8.1, 8.4 (RNN, khối Transformer); trong 8.3 chỉ hỏi sinusoidal, không hỏi RoPE; mặt nạ nhân quả chưa hỏi |
| Chương 9 | 18, 19, 20 | 6 | | 5–6 | 3–4 | 0–2 | 9.5 (đánh giá LLM); giải mã và nhiệt độ trong 9.1; SFT, DPO, RLVR trong 9.3 (chỉ hỏi Bradley–Terry) |
| Chương 10 | 21, 22 | 4 | | 3–4 | 2 | 0–1 | 10.2 (GAN; chỉ hỏi D* ở câu 21(c), không hỏi minimax hay mode collapse), 10.4, 10.5 (tự giám sát, đánh giá mô hình sinh) |
| Chương 11 | 23, 24 | 4 | | 3–4 | 2 | 0–1 | 11.4 chỉ qua câu 30; 11.5 (viết và tái lập) |
| Phần N (luồng B) | 25, 26, 27 | 6 | | 5–6 | 3–4 | 0–2 | Không đo cấu trúc dữ liệu và thuật toán (thuộc luồng B); NumPy chỉ đo broadcasting, ổn định số, ba lỗi im lặng. Các câu 25 và 27 không đo mục nào của chương 1 hay 6: broadcasting không có mục riêng trong sách (chỉ gặp trong bài tập chương 2), và ba lỗi của câu 27 không nằm trong danh mục lỗi im lặng của mục 6.5; luyện qua lab và bài săn bug |
| Phần E (luồng C) | 28, 29, 30 | 6 | | 5–6 | 3–4 | 0–2 | Không đo nghe và nói thật (câu 30 tự chấm), viết dài, đọc paper dài |
| **Tổng** | 1–30 | 60 | | | | | Chương 12–14: không đo |

Chương 8 và 9 có ba câu vì là trọng tâm của nhánh NLP. "Vững" ứng với từ 75% điểm trở lên; "học từ đầu" ứng với dưới 50%; đúng 50% (2 trên 4) xếp vào "cần đọc kỹ".

**Ý nghĩa của "vững".** Mỗi nhóm chỉ có hai hoặc ba câu, nên lệch một ý là nhảy một mức, và "vững" chỉ nói về các mục được hỏi, không nói về cột "Mục không được hỏi". Vì vậy: (1) điểm sát một ngưỡng (nhóm 4 điểm: 1, 2 hoặc 3; nhóm 6 điểm: 2, 3, 4 hoặc 5) thì làm Kiểm tra nhanh của chương đó. Điểm nằm ngay trên ngưỡng (nhóm 4 điểm: 2 hoặc 3; nhóm 6 điểm: 3 hoặc 5): Kiểm tra nhanh từ 4 trên 5 câu thì giữ mức bảng cho, từ 3 trên 5 trở xuống thì hạ một mức. Điểm nằm ngay dưới ngưỡng (nhóm 4 điểm: 1 hoặc 2; nhóm 6 điểm: 2 hoặc 4): đúng 5 trên 5 thì nâng một mức, còn lại giữ mức bảng cho [heuristic, chưa hiệu chỉnh; để quy tắc đối xứng, vì chỉ hạ mà không bao giờ nâng sẽ đẩy mọi sai lệch về phía thêm giờ]; (2) chương xếp "vững" vẫn phải qua Kiểm tra nhanh của chương ở các mục trong cột "Mục không được hỏi" trước khi rút giờ; (3) chương 12 (NLP chuyên sâu) không được đo ở đây, nên không có kết luận nào về nó.

## Đọc kết quả

**Quy tắc một dòng.** Vững thì rút giờ nhưng không bỏ bước; cần đọc kỹ thì theo đúng kế hoạch; học từ đầu thì thêm giờ và học theo thứ tự các mục.

| Mức | Giờ gợi ý cho chương (kế hoạch gốc: 16 giờ) | Cách học |
| --- | --- | --- |
| Vững | 11–13 giờ | Làm Kiểm tra nhanh của chương: mục nào đúng thì đọc lướt. Làm đủ bài ★★★, lab, mini-test. Bài ★ và ★★ chỉ làm những bài bạn thấy lạ. Tiết kiệm 3–5 giờ (chương 8, 9: tối đa 2 giờ, xem ngoại lệ dưới bảng). |
| Cần đọc kỹ | 16 giờ | Theo kế hoạch tuần trong app. |
| Học từ đầu | 20–24 giờ | Thêm 4–8 giờ. Đọc tuần tự từng mục, tự làm lại mọi ví dụ giải mẫu trước khi xem lời giải, làm đủ bài ★ trước khi sang ★★, xem video Manim của từng mục, làm bài tính nhanh hằng ngày. Gặp chỗ không hiểu thì quay về chương nền (bảng bên dưới). |

Ngoại lệ: chương 8 và 9 là trọng tâm phỏng vấn NLP, nên dù điểm cao cũng không rút xuống dưới 14 giờ (tức rút tối đa 2 giờ).

**Câu bị bỏ vì hết giờ, và chương đã học.** Số duy nhất dùng để tra bảng là điểm cuối, tức điểm sau khi mọi câu đã được làm (lượt ba là bắt buộc nếu còn câu trống); điểm trong giờ không dùng để tra bảng. Nếu tra bằng điểm trong giờ, chương nằm cuối đề (9, 10, 11) bị hạ mức oan. Chương nào bạn đã học trước khi làm bài (nhãn "đã học"; chương 1 thường thuộc trường hợp này vì lịch w00–w01 xếp mục 1.1–1.4 vào khoảng thời gian làm bài, nên cả câu 2, thuộc mục 1.4–1.5, cũng có thể đã bị học trước) thì tính 16 giờ trong quỹ giờ, bất kể điểm: điểm cao có thể chỉ vì vừa đọc xong. Điểm của nhóm đó vẫn dùng để xem mục nào hổng.

**Quỹ giờ cho Phần I (chương 1–6).** Kế hoạch gốc là 6 × 16 = 96 giờ. Quy đổi mỗi chương: vững 12 giờ, cần đọc kỹ 16 giờ, học từ đầu 22 giờ [heuristic]. Gọi s là số chương học từ đầu và v là số chương vững trong chương 1–6; chênh lệch so với 96 giờ là

$$
D = 6s - 4v \quad \text{(giờ)}
$$

Mỗi tuần dành cho chương khoảng 8 giờ và luồng A không bị cắt, nên mỗi tuần lùi mốc hấp thụ tối đa 8 giờ. Có thể bù thêm tối đa 4 giờ bằng một hoặc hai buổi Thứ Bảy, nhưng đó là **giờ vượt kế hoạch**: các tuần w01–w12 trong `plan/weeks.yaml` đã ở khoảng 14.25–15.25 giờ mỗi tuần (cộng giờ các việc trong tuần, kiểm ngày 2026-11-05), tức đã sát hoặc vượt trần 12–15 giờ mỗi tuần của `huong-dan.md`. Vì vậy chỉ bù Thứ Bảy khi bạn chủ động đồng ý tăng giờ. Một công thức cho mọi dải: số tuần lùi là ⌈max(D − 4, 0)/8⌉ nếu bù 4 giờ Thứ Bảy, hoặc ⌈max(D, 0)/8⌉ nếu không bù. Tra D:

| D (giờ) | Lùi Đề thi thử 1 (31/1/2027), có bù 4 giờ Thứ Bảy | Lùi, không bù Thứ Bảy |
| --- | --- | --- |
| Từ −24 đến 0 | 0 tuần (giữ mốc). D âm nghĩa là có giờ dư; đừng dùng giờ dư đó để cắt luồng A, B, C (xem Phần II bên dưới) | 0 tuần |
| +1 đến +4 | 0 tuần | 1 tuần |
| +5 đến +12 | 1 tuần | 1 tuần nếu D ≤ 8, 2 tuần nếu D từ 9 |
| +13 đến +20 | 2 tuần | 2 tuần nếu D ≤ 16, 3 tuần nếu D từ 17 |
| +21 đến +28 | 3 tuần | 3 tuần nếu D ≤ 24, 4 tuần nếu D từ 25 |
| +29 đến +36 | 4 tuần | 4 tuần nếu D ≤ 32, 5 tuần nếu D từ 33 |

Từ 3 tuần lùi trở lên, hoặc khi chọn tăng số giờ mỗi tuần dành cho chương thay vì lùi, quyết định ảnh hưởng cả lịch Phần II và hồ sơ; bạn phải tự cân nhắc và quyết, không nên coi là mặc định.

Ví dụ: 2 chương học từ đầu, 0 chương vững: D = +12, lùi 1 tuần (có bù Thứ Bảy) hoặc 2 tuần (không bù). 3 chương học từ đầu và 3 chương vững: D = +6, lùi 1 tuần. 3 chương học từ đầu, 0 vững: D = +18, lùi 2 hoặc 3 tuần. Cả 6 chương học từ đầu: D = +36, lùi 4 tuần (có bù) hoặc 5 tuần (không bù). Nếu coi ba mức của mỗi chương là khả dĩ như nhau (một giả định đơn giản, không phải xác suất thật của bạn), trong 729 tổ hợp có 415 (57%) rơi vào D ≤ +4 (không lùi nếu bù Thứ Bảy), 161 (22%) vào +5 đến +12, 125 (17%) vào +13 đến +20, 21 (3%) vào +21 đến +28, và 7 (1%) vào +29 đến +36 [tính bằng `work/ch00/round3_sim.py`, ngày 2026-11-05].

**Phần II (chương 7–11): giờ quyết lúc đến chương, không tính từ điểm hôm nay.** Bài chẩn đoán làm trước khi học Phần II, nên nhiều chương 7–11 ra mức thấp chỉ vì chưa học, không vì bạn yếu. Vì vậy mức Phần II hôm nay chỉ dùng để thấy mục hổng và chọn chương nền cần ôn, **không dùng để cộng hay trừ giờ**. Giờ của từng chương 7–11 quyết bằng Kiểm tra nhanh ngay đầu chương, theo cùng bảng mức: vững 12 giờ, cần đọc kỹ 16, học từ đầu 22; chương 8 và 9 không dưới 14 [heuristic]. Khi đó gọi D′ là tổng chênh giờ của năm chương so với 16 giờ mỗi chương:

$$
D' = 6s - 4v - 2w \quad \text{(giờ)}
$$

với s là số chương học từ đầu trong chương 7–11, v là số chương vững trong {7, 10, 11}, w là số chương vững trong {8, 9}. Đề thi thử 2 (18/4/2027) lùi theo cùng công thức và cùng hai cách của bảng D, áp dụng cho D + D′: ⌈max(D + D′ − 4, 0)/8⌉ tuần nếu bù 4 giờ Thứ Bảy, ⌈max(D + D′, 0)/8⌉ tuần nếu không bù (bù Thứ Bảy tính một lần cho cả hai mốc, vì không có cơ sở để cộng dồn giờ vượt trần; heuristic). Vì D + D′ gộp cả hai phần, **giờ dư của Phần I (D < 0) tự chuyển sang các chương Phần II yếu**, và việc lùi mốc 1 mà Phần II còn dư giờ có thể không làm mốc 2 lùi.

Hôm nay chỉ nói được D′ nằm trong khoảng từ −16 (cả năm chương vững) đến +30 (cả năm học từ đầu), nên D + D′ hôm nay là một khoảng, không phải một số. Để thấy vì sao không dùng mức hôm nay: một hồ sơ trung bình (D = +12) mà năm chương Phần II đều ra "học từ đầu" thì D + D′ = +42, tức lùi 5 tuần (có bù) hoặc 6 tuần (không bù); quy tắc cũ "D − S" chỉ trừ giờ rút và bỏ qua giờ thêm, nên báo thiếu 12 giờ trong khi thiếu thật tới 42 nếu mức đó còn nguyên lúc đến chương [mô phỏng `work/ch00/round3_sim.py`, 2026-11-05]. Sau khi học, mức thường khá hơn hôm nay [dự đoán, chưa đo]; Kiểm tra nhanh đầu chương sẽ cho con số thật.

**Chương nền kéo theo chương nào.** Nếu một chương nền yếu, các chương dùng nó sẽ chậm hơn dự kiến dù điểm riêng của chúng có thể khá.

| Yếu ở | Ảnh hưởng tới | Việc nên làm thêm |
| --- | --- | --- |
| Chương 1 | 2, 4, 8, 10 | Thêm bài tính nhanh về nhân ma trận, trị riêng, SVD 2×2 mỗi ngày. Chưa vững mục 1.4–1.5 thì chương 4 (κ) và 8 (attention) sẽ khó. |
| Chương 2 | 6, 8 | Làm lại ví dụ backprop bằng tay; làm lab gradient check trước khi vào chương 6. |
| Chương 3 | 5, 9, 10, 11 | Chắc MLE, cross-entropy, KL, CI trước khi vào chương 9–11; những chương đó dùng chúng liên tục. |
| Chương 4 | 6, 9 | Chắc phần số điều kiện và cập nhật Adam; chương 6 và 9 dựa vào chúng khi debug và tính bộ nhớ. |
| Chương 5 | 6, 11 | Chắc metric và leakage; chương 11 nâng chúng lên mức thiết kế thí nghiệm. |
| Chương 6 | 7–10 | Từ chương 7 trở đi lab dùng PyTorch; nếu chương 6 yếu, đừng chuyển sang chương 7 trước khi qua mini-test chương 6. |
| Chương 8 hoặc 9 | 12 (NLP chuyên sâu) | Chương 12 giả định bạn nắm attention và vòng đời LLM. Yếu thì dành tuần đệm trước chương 12. |

**Khuyến nghị theo luồng.**

- **Luồng B (coding), theo phần N.** 5–6: giữ kế hoạch, thêm một bài bấm giờ mỗi tuần từ sớm. 3–4: mỗi tuần thêm 1 giờ NumPy (axis, broadcasting, view và copy), chủ yếu qua lab. 0–2: đầu tuần 1 dành 3–4 giờ ôn NumPy từ đầu, và làm lab chương 1 và 2 chậm rãi trước; nhiều lỗi code sẽ tưởng là lỗi toán. Bài săn bug của mỗi chương là cách luyện tốt nhất cho câu kiểu 27.
- **Luồng C (tiếng Anh), theo phần E.** 5–6: giữ 1 giờ mỗi tuần, đọc to câu trả lời mẫu và ghi âm. 3–4: nói to (không chỉ đọc) câu trả lời mẫu, và chép 5 cụm hay dùng mỗi tuần vào sổ từ vựng. 0–2: học phần từ vựng cuối mỗi chương từ chương 1, thêm 30 phút mỗi tuần viết lại câu trả lời mẫu bằng lời mình trước khi ghi âm; chương 13–14 sẽ nặng tiếng Anh hơn nhiều.
- **Luồng A (dự án nghiên cứu).** Nếu chương 3 hoặc 11 ở mức "học từ đầu", những tuần đầu của luồng A nên tập trung vào baseline và câu hỏi nghiên cứu; đọc mục 11.2 và 11.3 sớm để thiết kế thí nghiệm đúng ngay từ đầu.

**Đọc theo nhãn lỗi.** Nhiều **K** ở một chương: đọc lại lý thuyết theo mục, làm bài ★. Nhiều **T** rải ở nhiều chương: không cần đọc lại, làm bài tính nhanh 15 phút mỗi ngày và viết đủ bước trung gian. Nhiều nhãn **H**, hoặc điểm cuối hơn điểm trong giờ cỡ 6 điểm trở lên [heuristic, chưa hiệu chỉnh] (hiểu nhưng chậm): luyện bài tính nhanh có bấm giờ; đó là vấn đề tốc độ, đừng đọc thành chương yếu. Nhiều **L** ở phần E hoặc ở các ý giải thích bằng lời: dùng luồng C. Đúng đáp số mà không viết được lý do: đó là kỹ năng phỏng vấn, dùng chuỗi câu hỏi 4 tầng của từng chương.

**Đo lại.** Sách không có bản thứ hai của bài này: mini-test mỗi chương đo lại chương đó, Đề thi thử 1 đo lại Phần I. Ghi điểm các nhóm hôm nay, cả điểm trong giờ và điểm cuối, để so. Lịch tuần 39 (`plan/weeks.yaml`) có xếp một lần làm lại chính bài này; nếu làm lại, hiểu đó là so thô, vì bạn nhớ đề ở mức nào đó, và 120 phút làm bài cộng chấm không vừa 1.5 giờ (xem mục cuối).

## Lời giải

Chỉ xem sau khi đã làm hết 30 câu. Thang điểm (1 + 1) ghi ngay sau mỗi lời giải.

**Câu 1.** (a) Hàng 2 gấp đôi hàng 1; hàng 3 độc lập với hai hàng kia, nên hạng = 2. Cột 3 = cột 1 + cột 2, nên một vector null là (−1, −1, 1); kiểm: hàng 1 cho −1 − 2 + 3 = 0, hàng 3 cho −1 + 1 = 0. dim N(A) = 3 − 2 = 1. (b) Cột 1 là (1, 2, 0), cột 2 là (2, 4, 1), nên mọi b trong C(A) có thành phần thứ hai gấp đôi thành phần thứ nhất. b = (5, 10, 2) thỏa: x = (1, 2, 0) là một nghiệm (1·(1,2,0) + 2·(2,4,1) = (5, 10, 2)), nên có vô số nghiệm x = (1, 2, 0) + t(−1, −1, 1). b = (1, 1, 1) không thỏa nên vô nghiệm. *Thang:* 1 điểm cho hạng và vector null đúng; 1 điểm cho kết luận đúng cả hai hệ kèm lý do.

**Câu 2.** (a) Đa thức đặc trưng λ² − 7λ + 6 = 0 (vết 7, định thức 6) cho λ = 6 và λ = 1. Vector riêng: (2, 1) cho 6, (1, −2) cho 1; hai vector vuông góc. (b) S đối xứng và các trị riêng đều dương, nên S xác định dương, và SVD của nó chính là phân rã riêng: giá trị kỳ dị bằng trị riêng, σ = 6 và 1. ‖S‖₂ = 6, κ = 6. *Thang:* 1 điểm cho (a); 1 điểm cho σ, ‖S‖₂, κ, xác định dương và lý do (đối xứng, trị riêng dương).

**Câu 3.** (a) ∇f = (2xy, x² + 3) = (4, 7). Hessian có hàng (2y, 2x) và (2x, 0), tại (2, 1) là hàng (2, 4) và (4, 0). Xấp xỉ: f(2, 1) = 7, nên f(2.1, 0.9) ≈ 7 + 4(0.1) + 7(−0.1) = 6.7; giá trị đúng 6.669 (sai khác 0.031, cỡ bậc hai). (b) ∇f = (4, 7) ≠ 0 nên không phải điểm dừng, và Hessian ở đây không dùng để phân loại cực trị. Hessian có trị riêng 1 ± √17, tức khoảng 5.12 và −3.12: bất định, hàm cong lên theo một hướng và cong xuống theo hướng khác (độ cong khác dấu). *Thang:* 1 điểm cho gradient, Hessian và ước lượng đúng; 1 điểm cho nhận ra không phải điểm dừng và đọc đúng Hessian bất định.

**Câu 4.** (a) z = 1, h = 1, ŷ = 3, L = 4. ∂L/∂ŷ = 2(ŷ − t) = 4; ∂L/∂w₂ = 4·h = 4; ∂L/∂h = 4·w₂ = 12; ReLU cho qua vì z > 0, nên ∂L/∂z = 12; ∂L/∂w₁ = 12·x = 24. (b) z = −1, h = 0, ŷ = 0, L = 1, ∂L/∂ŷ = −2, nhưng ReLU chặn: ∂L/∂z = 0 nên ∂L/∂w₁ = 0 (và ∂L/∂w₂ = 0 vì h = 0). Neuron "chết" với mẫu này: không nhận tín hiệu học, w₁ đứng yên; nếu nó chết với mọi mẫu thì không bao giờ sống lại (hệ quả này được dạy kỹ ở mục 6.1; câu chỉ đòi gradient bị mặt nạ ReLU chặn). *Thang:* 1 điểm cho (a) đủ năm đạo hàm; 1 điểm cho (b) có con số 0 và nêu được hệ quả.

**Câu 5.** (a) P = 0.005·0.95 / (0.005·0.95 + 0.995·0.02) = 0.00475 / 0.02465 ≈ 0.193. (b) Dạng odds: odds ban đầu 0.005/0.995 ≈ 0.00503; mỗi lần dương nhân tỉ số hợp lý 0.95/0.02 = 47.5. Sau hai lần: 0.00503 · 47.5² ≈ 11.34, tức xác suất 11.34/12.34 ≈ 0.919. (c) Tỉ lệ nền thấp: trong 1000 người có 5 người mắc (khoảng 4.75 dương thật) nhưng 995 người khỏe cho khoảng 19.9 dương giả; số dương giả lấn át số dương thật. *Thang:* 1 điểm cho (a) đúng; 1 điểm cho (b) đúng và (c) nêu được tỉ lệ nền.

**Câu 6.** (a) MLE của μ là trung bình = 4. MLE của σ² chia cho n: (9 + 1 + 16)/3 = 26/3 ≈ 8.667. Ước lượng không chệch chia cho n − 1: 26/2 = 13. (b) H(p) = 0.5·1 + 0.25·2 + 0.25·2 = 1.5 bit. H(p, q) = 0.5·2 + 0.25·2 + 0.25·1 = 1.75 bit. KL(p‖q) = 1.75 − 1.5 = 0.25 bit. Perplexity = 2 mũ 1.75 ≈ 3.36. *Thang:* 1 điểm cho (a) (cả ba số); 1 điểm cho (b) (bốn số, trong đó KL = cross-entropy − entropy).

**Câu 7.** (a) Hessian diag(1, 9): λₘₐₓ = 9, λₘᵢₙ = 1. Ngưỡng trên của η để ổn định: η < 2/λₘₐₓ = 0.222 (cận trên không đạt, không gồm biên). η tối ưu = 2/(λₘₐₓ + λₘᵢₙ) = 0.2; hệ số co (κ − 1)/(κ + 1) = 8/10 = 0.8. (b) Mỗi thành phần nhân với (1 − ηλ): hướng λ = 1 nhân 0.9, hướng λ = 9 nhân 0.1; x = (0.9, 0.1). Hướng λ nhỏ (x₁) chậm. (c) Chuẩn sai số là |x₁|, thành phần chậm nhất, nên sau k bước nó bằng (hệ số)ᵏ. Với η tối ưu cả hai thành phần cùng co 0.8 mỗi bước: ln 1000 / ln(1/0.8) ≈ 30.96, tức 31 bước. Với η = 0.1: ln 1000 / ln(1/0.9) ≈ 65.6, tức 66 bước. Mô phỏng từ x = (1, 1) cho đúng 31 và 66. Nếu bạn đo sai số bằng chuẩn Euclid ‖x‖₂ so với ‖x‖₂ ban đầu (√2), kết quả là 31 và 63 vì lúc đầu phần x₂ còn góp vào chuẩn; cũng chấp nhận nếu bạn nêu rõ chuẩn đã dùng. *Thang:* 1 điểm cho (a); 1 điểm cho (b) và (c) (nêu chuẩn đã dùng và số bước khớp với chuẩn đó).

**Câu 8.** (a) v₁ = 1, v₂ = 0.9·1 + 1 = 1.9, v₃ = 0.9·1.9 + 1 = 2.71; giới hạn 1/(1 − β) = 10. Với gradient ổn định, bước hiệu dụng lớn gấp khoảng 10 lần so với SGD thường cùng η; vì vậy đổi β thì thường phải đổi η. (b) Bước đầu: m̂ = g, v̂ = g², nên w đổi một lượng −α·g/(|g| + ε), có dấu ngược dấu của g và độ lớn xấp xỉ α = 0.001 cho cả hai: g = 0.01 thì w giảm 0.000999999; g = −100 thì w tăng 0.0009999999999 (hai độ lớn lệch nhau cỡ 10⁻⁹, đều ≈ α). Adam gần như không phụ thuộc thang gradient ở bước này (và nói chung chuẩn hóa theo từng tọa độ). *Thang:* 1 điểm cho (a) (ba số và giới hạn); 1 điểm cho (b) (độ dời ≈ α và nhận xét).

**Câu 9.** z = 2 − 1 = 1; p = σ(1) ≈ 0.7311; log-loss = −ln p ≈ 0.3133. Gradient = (p − y)x = −0.2689·(2, −1) = (−0.538, +0.269). Dấu âm ở w₁ nghĩa là gradient descent sẽ tăng w₁ (vì y = 1 và x₁ > 0, cần tăng logit); dấu dương ở w₂ nghĩa là giảm w₂ (x₂ < 0, giảm w₂ làm logit tăng). *Thang:* 1 điểm cho z, p, loss; 1 điểm cho gradient đúng dấu và giải thích hướng.

**Câu 10.** (a) Đặt đạo hàm bằng 0: w = ∑xy / (∑x² + λ) = 29/(14 + λ). λ = 0: w ≈ 2.071. λ = 6: w = 29/20 = 1.45 (bị co về 0). (b) Phương sai cao (overfit): train cao nhiều, validation thấp. Có ích: thêm dữ liệu, regularization mạnh hơn (weight decay, dropout), early stopping, giảm độ phức tạp mô hình, augmentation. Không giúp: huấn luyện thêm nhiều epoch mà không có early stopping, hoặc tăng độ phức tạp của mô hình mà không đổi gì khác. Chấp nhận thêm: người trả lời nêu cả khả năng lệch phân phối giữa train và validation, miễn chẩn đoán chính vẫn là overfit và có phép thử phân biệt (ví dụ so loss trên một phần train giữ lại). *Thang:* 1 điểm cho (a); 1 điểm cho (b) (chẩn đoán đúng, hai biện pháp đúng, một biện pháp không giúp).

**Câu 11.** (a) 784·256 + 256 + 256·10 + 10 = 203 530. (b) h có shape (32, 256); ∂L/∂W₁ cùng shape với W₁, tức (784, 256) (hoặc (256, 784) nếu W₁ lưu theo quy ước ngược; điều quan trọng là bằng shape của tham số). (c) Logit gần 0 thì softmax gần đều: loss ≈ ln 10 ≈ 2.303. *Thang:* 1 điểm cho (a); 1 điểm cho (b) và (c).

**Câu 12.** (a) √(2/784) ≈ 0.0505. (b) Loss ban đầu 7.2 lớn hơn nhiều so với ln 10 ≈ 2.30, nghĩa là logit khởi đầu quá lớn và sai chỗ. Nguyên nhân khả dĩ: khởi tạo trọng số quá lớn, đầu vào chưa chuẩn hóa (giá trị cỡ hàng trăm), hoặc lớp cuối có scale lớn. Phép thử nhanh để phân biệt: chuẩn hóa đầu vào (trừ mean, chia std) rồi đo lại loss bước 0. Nếu loss về khoảng 2.3 thì lỗi ở dữ liệu; nếu vẫn lớn, nhân trọng số lớp cuối với 0.01 và đo lại: loss về khoảng 2.3 thì lỗi ở khởi tạo. *Thang:* 1 điểm cho (a); 1 điểm cho (b) (hai nguyên nhân hợp lý, một phép thử cụ thể).

**Câu 13.** (a) Đầu ra: ⌊(28 + 4 − 5)/2⌋ + 1 = 14, nên 32 × 14 × 14. Tham số: 32·16·25 + 32 = 12 832. MAC: 32·14·14·16·25 = 2 508 800. (b) Depthwise: 16·25 = 400 tham số, MAC 16·14·14·25 = 78 400. Pointwise: 16·32 = 512 tham số, MAC 32·14·14·16 = 100 352. Tổng 912 tham số và 178 752 MAC. Tỉ lệ MAC: 2 508 800 / 178 752 ≈ 14.0 (xấp xỉ 1/(1/32 + 1/25)). *Thang:* 1 điểm cho (a); 1 điểm cho (b).

**Câu 14.** (a) Bắt đầu r = 1, j = 1. Conv 3×3: r = 3. Conv 3×3: r = 5. Pool 2×2 s2: r = 5 + 1·1 = 6, j = 2. Conv 3×3: r = 6 + 2·2 = 10. Receptive field 10, jump 2. (b) (224/16)² = 196 token. Patch embedding là conv 16×16 stride 16 (hay Linear từ 16·16·3 = 768 chiều tới 768): 768·768 + 768 = 590 592 tham số. *Thang:* 1 điểm cho (a); 1 điểm cho (b).

**Câu 15.** (a) q·k₁ = 4, q·k₂ = 0; chia √4 = 2 được (2, 0). Softmax: e²/(e² + 1) ≈ 7.389/8.389 ≈ 0.881, và 0.119. (b) Đầu ra = 0.881·(10, 0) + 0.119·(0, 5) ≈ (8.81, 0.60). (c) Không chia: điểm (4, 0), trọng số (0.982, 0.018), softmax gần như one-hot. Khi dₖ lớn điểm có phương sai tỉ lệ dₖ nên softmax bão hòa, đạo hàm p(1 − p) gần 0, gradient biến mất qua attention. *Thang:* 1 điểm cho (a) và (b); 1 điểm cho (c) (số và lý do bão hòa).

**Câu 16.** (a) Tần số 1 và 0.01, vị trí 2: (sin 2, cos 2, sin 0.02, cos 0.02) ≈ (0.9093, −0.4161, 0.0200, 0.9998). (b) Self-attention không PE là equivariant với hoán vị: hoán vị hai token thì đầu ra tại hai token đó cũng hoán vị theo, nội dung từng đầu ra không đổi. Nên mô hình thấy hai câu "mèo cắn chó" và "chó cắn mèo" như cùng một tập từ (nếu gộp bằng trung bình hay tổng thì hai biểu diễn trùng nhau); thứ tự phải được đưa vào bằng mã hóa vị trí. *Thang:* 1 điểm cho (a); 1 điểm cho (b) (equivariance, hệ quả mất thứ tự).

**Câu 17.** (a) Attention 4d² + FFN 2·d·4d = 8d²; tổng 12d² = 12·768² = 7 077 888 mỗi khối; 12 khối: 84 934 656 (khoảng 85 triệu). (b) Mỗi token lưu K và V cho mỗi khối: 2·12·768·2 byte = 36 864 byte. Với T = 2048: 36 864·2048 = 75 497 472 byte = 72 MiB. (c) Ma trận điểm attention T × T mỗi đầu mỗi khối (2048² phần tử; ở fp16 là 8 MiB mỗi đầu mỗi lớp), nếu tính đơn giản mà không dùng kernel gộp. *Thang:* 1 điểm cho (a); 1 điểm cho (b) và (c).

**Câu 18.** (a) Cặp (a, b) có tần số 3 + 2 = 5, cặp (b, c) có 2 + 1 = 3; gộp (a, b) trước: token còn 9. Sau đó cặp (ab, c) có tần số 2, cặp (b, c) chỉ còn 1; gộp (ab, c): token còn 7. Điều này đúng với đề đã nêu (không có ký hiệu hết từ). Nếu thêm `</w>` vào cuối mỗi từ như ví dụ 9.1 của chương 9, bài có 20 token, gộp đầu vẫn là (a, b) (tần số 5, còn 15 token), nhưng gộp thứ hai hòa: (ab, `</w>`) và (c, `</w>`) cùng tần số 3, nên chọn một trong hai (cách chọn khi hòa) và còn 12 token; đó là một bài khác, không chấm theo đáp án này. (b) NLL = (1 + 2 + 3 + 3)/4 = 2.25 bit mỗi token; perplexity = 2 mũ 2.25 ≈ 4.76 (cũng bằng (1/512) mũ −1/4). *Thang:* 1 điểm cho (a) (hai cặp và 9, 7); 1 điểm cho (b).

**Câu 19.** (a) 6ND = 6·2·10⁹·4·10¹⁰ = 4.8·10²⁰ FLOPs; tỉ lệ 20 token mỗi tham số. (b) 4.8·10²⁰ / 10¹³ = 4.8·10⁷ giây, khoảng 556 ngày trên một GPU. Kết luận (khuyến nghị của sách, không phải kết quả tính): tiền huấn luyện không nằm trong tầm một GPU cá nhân; kế hoạch nên dùng mô hình pretrained rồi tinh chỉnh nhẹ (LoRA), và tập trung vào hiểu quy luật quy mô để đọc paper. Con số 10 TFLOP/s chỉ là giả định minh họa. (c) σ(1.5 − 0.5) = σ(1) ≈ 0.731. *Thang:* 1 điểm cho (a) và (b) (kể cả kết luận); 1 điểm cho (c) cùng việc nhận ra hiệu reward đi vào sigmoid.

**Câu 20.** (a) 16·(2048 + 4096) = 98 304 tham số, so với 2048·4096 = 8 388 608: khoảng 1.17%, tức xấp xỉ 1/85. (b) 1.5·10⁹ · 16 byte = 2.4·10¹⁰ byte ≈ 22.35 GiB. (c) 1.5·10⁹ · 0.5 byte = 7.5·10⁸ byte ≈ 0.70 GiB. Tinh chỉnh toàn phần (22 GiB) vượt xa 6 GiB; trọng số 4-bit chỉ 0.7 GiB nên LoRA trên nền lượng tử hóa là khả thi (khuyến nghị của sách), dù activation và adapter vẫn tốn thêm. *Thang:* 1 điểm cho (a); 1 điểm cho (b), (c) và nhận xét.

**Câu 21.** (a) KL(N(μ, σ²)‖N(0, 1)) = ½(σ² + μ² − 1 − ln σ²) = ½(0.25 + 4 − 1 + 1.3863) = ½·4.6363 ≈ 2.318 nat. (b) z = μ + σε = 2 + 0.5·(−1) = 1.5. Lấy mẫu trực tiếp là phép ngẫu nhiên không có đường đạo hàm tới μ và σ; viết z = μ + σε đẩy tính ngẫu nhiên sang ε (không phụ thuộc tham số) nên gradient chảy qua μ, σ. (c) D*(x) = pdata(x)/(pdata(x) + pg(x)) = 0.6/(0.6 + 0.2) = 0.75 (và D* = ½ khi pg = pdata, tức generator hoàn hảo). *Thang:* 1 điểm cho (a); 1 điểm cho (b) và (c) khi cả hai đúng (z = 1.5 kèm lý do; D* = 0.75).

**Câu 22.** (a) xₜ = √0.36·1.5 + √0.64·(−0.5) = 0.6·1.5 − 0.8·0.5 = 0.9 − 0.4 = 0.5. (b) x̂₀ = (xₜ − √(1 − ᾱₜ) ε̂)/√ᾱₜ = (0.5 + 0.24)/0.6 ≈ 1.233, lệch 0.267 so với x₀ = 1.5 vì ε̂ chưa đúng; với ε̂ = −0.5 sẽ ra đúng 1.5. (c) E[ε²] = 1, nên MSE kỳ vọng của mạng luôn đoán 0 bằng 1. Đây là mức nền: loss chưa xuống dưới khoảng 1 nghĩa là mạng chưa học được gì hơn "đoán 0". *Thang:* 1 điểm cho (a) và (b); 1 điểm cho (c) (số 1 và vai trò mức nền).

**Câu 23.** (a) Mean = 80.0. Độ lệch −1, 1, 0.5, −0.5, 0, 0 có tổng bình phương 2.5; phương sai mẫu 2.5/5 = 0.5, std ≈ 0.707. SE = 0.707/√6 ≈ 0.289. Nửa độ rộng = 2.571·0.289 ≈ 0.742, CI 95% khoảng [79.26, 80.74]. (b) Không kết luận được. Baseline chỉ có một lần chạy nên chưa biết độ dao động của nó (có thể cỡ 0.7 như phương pháp của bạn); 79.2 nằm ngay dưới cận dưới 79.26 của CI của bạn, nên nhìn có vẻ baseline kém hơn, nhưng đó là một lần chạy: vẫn chưa kết luận được. Cần chạy baseline với nhiều seed (cùng ngân sách tune) rồi so có cặp hoặc so CI. *Thang:* 1 điểm cho (a); 1 điểm cho (b) (không kết luận, lý do là phương sai của baseline).

**Câu 24.** (a) SE = √(0.8·0.2/250) ≈ 0.0253, tức 2.5 điểm phần trăm. SE của hiệu hai accuracy (coi như độc lập) là √(0.8·0.2/250 + 0.82·0.18/250) ≈ 0.035, tức 3.5 điểm phần trăm; chênh 2 điểm chỉ bằng khoảng 0.57 SE. Chênh 2 điểm nhỏ hơn cả SE của một accuracy (2.5 điểm), nên nằm trong nhiễu. Trên cùng một tập câu hỏi thì phép so có cặp (McNemar) có thể cho nhiễu nhỏ hơn, nhưng với một seed thì vẫn chưa đủ để kết luận. (b) Ví dụ: (1) chỉ một seed và một mẫu prompt: chạy nhiều seed và nhiều mẫu prompt, báo mean ± std; (2) X sinh nhiều gấp 3 lần token, chênh về ngân sách tính toán: so với baseline có cùng ngân sách (như self-consistency với cùng số token); (3) benchmark công khai có thể nhiễm trong dữ liệu huấn luyện: kiểm trùng lặp hoặc thử bộ câu mới/biến thể; (4) không có ablation cho thấy thành phần nào tạo ra cải thiện, và "suy luận" là kết luận rộng hơn một benchmark: thêm bộ đánh giá khác. *Thang:* 1 điểm cho (a); 1 điểm cho (b) khi có ít nhất ba vấn đề hợp lý, mỗi vấn đề có thí nghiệm kèm.

**Câu 25.** (a) (5, 4); (3, 3); (4, 3) cộng với (4,) bị lỗi vì so shape từ phải sang trái: 3 so với 4. (b)

```python
def sq_dists(X, C):
    return (X**2).sum(1)[:, None] - 2 * X @ C.T + (C**2).sum(1)[None, :]
```

Kết quả với dữ liệu kiểm: hàng (0, 2) và (25, 13). Thực tế khác lý thuyết: khai triển ‖x‖² − 2x·c + ‖c‖² có thể ra số âm cỡ 10⁻¹⁶ (hoặc lớn hơn nếu giá trị lớn) khi hai điểm trùng nhau hoặc rất gần nhau, rồi căn bậc hai báo nan; code thật nên kẹp bằng `np.maximum(·, 0)`. *Thang:* 1 điểm cho (a) đúng cả ba; 1 điểm cho (b) (đúng shape n × k và đúng giá trị kiểm). Chấp nhận cách khác, như `((X[:, None] - C[None])**2).sum(-1)`.

**Câu 26.** (a) Ba giá trị đều tràn thành inf (exp 1000 vượt giới hạn float64, khoảng e mũ 709), nên tính log của tổng ra inf. (b)

```python
def log_softmax(z):
    m = z.max()
    return z - m - np.log(np.exp(z - m).sum())
```

Kết quả: (−2.4076, −1.4076, −0.4076); tổng exp bằng 1. Mẹo: trừ giá trị lớn nhất không đổi softmax nhưng chặn tràn. *Thang:* 1 điểm cho (a) (inf và lý do tràn); 1 điểm cho (b) (trừ max, giữ log-sum-exp, đúng số).

**Câu 27.** A: `argmax(axis=0)` lấy theo cột (trên mẫu) thay vì theo lớp (axis=1); vì ma trận vuông 3×3 nên không báo lỗi shape. In ra 0.333 (dự đoán sai (0, 2, 0)), trong khi đúng là 1.0. Sửa: `axis=1`. B: `X.mean()` và `X.std()` lấy trên toàn bộ phần tử, không phải theo từng đặc trưng; Xs.mean(axis=0) khoảng (−0.84, 0.84), không bằng 0. Sửa: `X.mean(axis=0)` và `X.std(axis=0)`. C: `w_old = w` chỉ gán tham chiếu, `w += 0.5` sửa tại chỗ nên `w_old` cũng đổi; in ra [1.5 2.5]. Sửa: `w_old = w.copy()` (hoặc dùng `w = w + 0.5`). *Thang (ba mốc, vì có ba đoạn mà chỉ 2 điểm):* mỗi lỗi "đúng" nghĩa là nêu lỗi, giá trị in ra và cách sửa. Giá trị in ra của A và C phải đúng số (0.333; [1.5 2.5]); với B chấp nhận dạng định tính (khác 0, hai cột đối dấu, cỡ ±0.8), vì tính std của tám số bằng tay không phải điều bài đo. 2 điểm khi đúng ít nhất hai lỗi; 1 điểm khi đúng một lỗi; 0 điểm nếu không đúng lỗi nào. Nếu đúng cả ba, vẫn 2 điểm nhưng ghi "tốt" vào sổ. Việc nghĩ `argmax` hòa trả chỉ số cuối không đổi kết quả in ra của A.

**Câu 28.** Câu trả lời mẫu: (a) *KelpMix raises the average accuracy of small language models by 1.8 points over a tuned baseline on three classification benchmarks.* (b) *The gain shrinks as the dataset gets larger: +3.1 on the smallest, +0.1 on the largest, and about +2.2 on the middle one if the 1.8-point average is a plain mean over the three datasets (3 × 1.8 − 3.1 − 0.1 = 2.2, give or take rounding). This suggests that the method acts like a regularizer or a low-data helper, so it may not scale; it also means the average hides an uneven effect.* (c) *Variance is not reported: with only three seeds and no confidence intervals we cannot tell whether a 1.8-point average gain is larger than the noise. The full per-dataset table is also missing (only two of the three gains are given), and so is an ablation.* *Thang.* Nội dung, 1 điểm khi đạt ít nhất hai trong ba ý kiểm được: (i) claim đúng (tăng trung bình 1.8 điểm accuracy so với baseline đã tune); (ii) nhận ra hiệu quả giảm theo cỡ tập, lớn nhất ở tập nhỏ nhất, gần 0 ở tập lớn nhất; (iii) nêu thiếu phương sai hoặc khoảng tin cậy, hoặc thiếu ablation, hoặc thiếu bảng đầy đủ. Ai suy ra được mức giữa ≈ 2.2 thì ghi "tốt", không cộng điểm. Ngôn ngữ, 1 điểm theo danh sách kiểm ngôn ngữ chung ở cuối câu 30.

**Câu 29.** Mẫu: (a) *We divide the dot product by the square root of the key dimension so that the variance of the scores does not grow with the dimension.* (b) *This result comes from a single seed, so it is not enough to conclude that the new model is better.* *Thang.* Nội dung, 1 điểm khi cả hai ý đúng (chia √dₖ để chuẩn hóa phương sai; một seed không đủ kết luận). Ngôn ngữ, 1 điểm khi đạt (i) và (ii) trong danh sách kiểm chung (thuật ngữ cần có: dot product, key dimension, variance, seed, đủ cả bốn).

**Câu 30.** Mẫu (trang trọng hơn): *We tune hyperparameters and choose models on the validation set, so after many choices it is no longer an unbiased estimate of performance on new data; we have partly overfit to it. A separate test set is touched only once, at the end, so it gives an honest estimate. If we report validation accuracy, the number is optimistically biased.* Mẫu (nói chuyện, ngắn hơn): *Because we pick the model based on the validation set, its score is a bit optimistic. The test set stays untouched until the end, so it tells us how the model really does on new data.* *Thang.* Nội dung, 1 điểm khi đạt ít nhất hai trong ba ý: (i) chọn mô hình và tinh chỉnh siêu tham số trên validation làm điểm validation lạc quan; (ii) test set chỉ dùng một lần, ở cuối; (iii) vì vậy báo cáo điểm test chứ không phải điểm validation. Ngôn ngữ, 1 điểm khi đạt (i) và (ii) trong danh sách kiểm chung; thêm: 3–4 câu, nối ý, nói được mà không phải đọc từ giấy.

**Danh sách kiểm ngôn ngữ chung cho câu 28–30.** Người học tự chấm bài tiếng Anh của mình nên nhiễu lớn, vì vậy chấm theo mục kiểm được, không theo cảm giác: (i) mỗi câu có chủ ngữ và động từ chia đúng; (ii) có đủ thuật ngữ cần thiết của câu đó (câu 28: accuracy, baseline, seed, variance, đủ ít nhất ba; câu 29: dot product, key dimension, variance, seed (đủ cả bốn); câu 30: validation, test); (iii) không dịch từng chữ từ tiếng Việt, người đọc hiểu mà không phải đoán. Điểm ngôn ngữ của mỗi câu: 1 điểm khi đạt cả (i) và (ii). Mục (iii) khó tự chấm nên không tính điểm, nhưng nếu bạn thấy mình vi phạm thì ghi nhãn **L** vào sổ lỗi; nếu có người đọc hộ (bạn, giảng viên), nhờ họ chấm (iii).

## Số liệu đã kiểm

Mọi số trong lời giải được tính lại bằng ba script trong repo: `work/ch00/check.py` và `work/ch00/audit2.py` (tác giả), `work/ch00/opus_audit0.py` (kiểm độc lập), cộng `work/ch00/round3_sim.py` (quy tắc giờ và BPE câu 18). Tái lập: từ thư mục gốc repo, chạy `python work/ch00/<tên script>.py` bằng Python của dự án (NumPy, SciPy; thêm PyTorch cho câu 13 và 14), kiểm ngày 2026-11-05 với NumPy 2.5.4, SciPy 1.18.1, PyTorch 2.14.1. Ngữ nghĩa NumPy dùng ở câu 25–27 (broadcasting, `argmax` chọn chỉ số đầu khi hòa, tham chiếu so với bản sao) ổn định giữa các phiên bản. Dưới đây là các giá trị chính và kết quả đối chiếu.

| Câu | Giá trị | Kết quả kiểm |
| --- | --- | --- |
| 1 | hạng 2; (−1, −1, 1) ∈ N(A); b = (5, 10, 2) giải được, b = (1, 1, 1) dư 0.447 | khớp |
| 2 | trị riêng 6 và 1; κ = 6.0; det = 6 | khớp |
| 3 | ∇f = (4, 7); trị riêng Hessian 5.123 và −3.123; xấp xỉ 6.7, đúng 6.669 | khớp |
| 4 | w₁ = 0.5: ∂L/∂w₁ = 24 (sai phân trung tâm 23.99999); w₂: 4; w₁ = −0.5: 0 | khớp |
| 5 | 0.1927; sau hai lần dương 0.9189 (cả bằng odds và bằng Bayes trực tiếp) | khớp |
| 6 | MLE 4 và 8.667; không chệch 13; H = 1.5, CE = 1.75, KL = 0.25, perplexity 3.364 | khớp |
| 7 | 2/9 = 0.2222; η = 0.2; hệ số 0.8; (c) từ x = (1, 1), sai số = giá trị tuyệt đối của x₁ giảm còn 1/1000: 31 và 66 bước (giải tích 30.96 và 65.56); chuẩn ∞: 31 và 66; chuẩn Euclid so với √2 ban đầu: 31 và 63 | khớp, đã nêu chuẩn trong đề |
| 8 | v: 1, 1.9, 2.71; giới hạn 10; w đổi −α·g/(abs(g) + ε): −0.000999999 (g = 0.01) và +0.0009999999999 (g = −100) | khớp |
| 9 | z = 1; p = 0.7311; loss 0.3133; gradient (−0.5379, 0.2689) | khớp |
| 10 | 2.0714; 1.45 | khớp |
| 11 | 203 530; ln 10 = 2.3026 | khớp |
| 12 | 0.0505 | khớp |
| 13 | PyTorch cho (32, 14, 14) và 12 832 tham số; MAC 2 508 800; depthwise + pointwise 178 752, tỉ lệ 14.04 | khớp |
| 14 | receptive field 10, jump 2; PyTorch cho patch embedding 590 592 và lưới 14 × 14 = 196 | khớp |
| 15 | e² = 7.389; trọng số (0.8808, 0.1192); đầu ra (8.808, 0.596); không chia (0.982, 0.018) | khớp |
| 16 | (0.9093, −0.4161, 0.0200, 0.9998), khớp bảng ở ví dụ giải mẫu 8.5 | khớp |
| 17 | 7 077 888; 84 934 656; 36 864 byte; 75 497 472 byte = 72 MiB | khớp |
| 18 | BPE mô phỏng, không `</w>`: 14, 9, 7 token (cặp 5 rồi 2); có `</w>`: 20, 15, 12 token (gộp thứ hai hòa ở tần số 3); 2.25 bit; perplexity 4.7568 | khớp, đề đã loại `</w>` |
| 19 | 4.8·10²⁰; 20 token mỗi tham số; 555.6 ngày; σ(1) = 0.7311 | khớp |
| 20 | 98 304; 8 388 608; 1.17%; 22.35 GiB; 0.70 GiB | khớp |
| 21 | KL = 2.3181; z = 1.5; D* = 0.6/(0.6 + 0.2) = 0.75 | khớp |
| 22 | xₜ = 0.5; x̂₀ = 1.2333, lệch 0.2667 so với 1.5 (ε̂ = −0.5 cho 1.5); MSE nền 1 | khớp |
| 23 | mean 80.0; std 0.7071; SE 0.2887; t(5) = 2.5706; CI [79.258, 80.742] | khớp |
| 24 | SE = 0.02530; SE hiệu hai accuracy độc lập 0.0351 (p = 0.80 và 0.82), chênh 2 điểm = 0.57 SE | khớp |
| 25 | broadcasting (5, 4), (3, 3), lỗi cho (4, 3) + (4,); khoảng cách (0, 2; 25, 13) bằng cả hai cách; với dữ liệu lớn, đường chéo của công thức khai triển lệch cỡ ±1.5·10⁻⁸ so với 0 | khớp |
| 26 | exp ra inf; log_softmax (−2.4076, −1.4076, −0.4076), tổng exp 1.0 | khớp |
| 27 | A: 0.3333 (đúng 1.0 với axis=1); B: (−0.8427, 0.8427) (sửa thì 0); C: [1.5 2.5] (copy thì [1. 2.]) | khớp |

Câu 28–30 viết tiếng Anh; các số trong abstract câu 28 là số bịa cho bài (hiệu 73.0 − 71.2 = 1.8; nếu 1.8 là trung bình thường của ba tập thì mức thứ ba ≈ 2.2, trong khoảng 1.8 đến 2.6 khi tính cả sai số làm tròn).

Thời gian và quỹ giờ: cột Phút cộng ra phần A 87, phần N 17, phần E 14, cộng 2 phút đọc đề là 120; D = 6s − 4v được liệt kê trên 729 tổ hợp mức của 6 chương (415 / 161 / 125 / 21 / 7 tổ hợp theo năm dải của bảng D, tính với 4 giờ Thứ Bảy).

**Cần bạn xác nhận.**

1. Ngưỡng 75% (vững) và 50% (cần đọc kỹ), số giờ gợi ý (12 / 16 / 22 giờ mỗi chương) và số phút mỗi câu là heuristic hoặc ước lượng, chưa thử trên người thật. Sau lần làm đầu, nếu bạn thấy kết quả không khớp cảm nhận của mình về chương nào khó, hoặc 120 phút quá chặt hay quá rộng, hãy báo để chỉnh.
2. Mốc 31/1/2027 của Đề thi thử 1 lấy từ danh mục chương; số tuần lùi trong bảng D ở mục Đọc kết quả là gợi ý (giả định 8 giờ mỗi tuần cho chương), quyết định thuộc về bạn, nhất là khi phải lùi trên hai tuần. Cột "có bù 4 giờ Thứ Bảy" của bảng D vượt trần 12–15 giờ mỗi tuần của hướng dẫn, và các tuần w01–w12 đã gần trần; bạn cần quyết có chấp nhận hay không.
3. Bài chẩn đoán chỉ hỏi D* của mục 10.2 (GAN, câu 21(c)) và không có câu cho 10.5 (đánh giá mô hình sinh), và cũng không hỏi RoPE, giải mã và nhiệt độ, khối Transformer (xem cột "Mục không được hỏi"); nếu bạn muốn phủ thêm, cần thêm hoặc đổi câu và tính lại bảng chấm.
4. Làm bài (120 phút) cộng chấm và ghi sổ lỗi (khoảng 1 giờ, ước lượng) là khoảng 3 giờ mà lịch tuần hiện chưa có chỗ; bạn cần quyết lấy từ luồng nào hoặc dời sang tuần nào.
5. Câu 7(c) dùng |x₁| làm chuẩn sai số; nếu bạn thích chuẩn Euclid hơn, đáp số là 31 và 63 và lời giải đã ghi cả hai.
6. Lịch `plan/weeks.yaml` tuần 39 xếp 1.5 giờ cho "làm lại bài chẩn đoán chương 0" và "so với kết quả tuần 0", nhưng bài là 120 phút cộng chấm, và bài hôm nay chưa được xếp vào tuần nào. Cần quyết lấy giờ từ đâu, và tuần 0 hiện đã là w00–w01.
