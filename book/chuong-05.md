# Chương 5 — ML cổ điển

## Mục tiêu và kiểm tra nhanh

Chương này là cầu nối giữa toán (chương 1–4) và mạng nơ-ron (chương 6). Mỗi mô hình theo cùng một khung: **giả thuyết → hàm mất mát → cách tối ưu → thất bại ở đâu**, cũng là cách trả lời phỏng vấn tốt nhất. Phần lớn những gì sống sót sang deep learning nằm ở đây: softmax + cross-entropy, weight decay, early stopping, metric cho dữ liệu lệch, leakage, và EM như tiền thân của VAE. Thời lượng: khoảng 2 tuần, 8 giờ mỗi tuần.

**Sau chương này, bạn phải làm được:**

- ☐ Viết giả thuyết, loss và gradient của hồi quy tuyến tính, logistic và softmax; giải thích vì sao log-loss là negative log-likelihood.
- ☐ Phân rã sai số thành bias² + variance + nhiễu, giải thích L1 khác L2, và thiết kế cross-validation không rò rỉ.
- ☐ Tính precision, recall, F1, AUC, AP bằng tay, và chọn ngưỡng theo chi phí khi lớp dương hiếm.
- ☐ Giải thích margin, hinge loss, đối ngẫu và kernel trick của SVM, và nói vì sao SVM kernel không scale.
- ☐ Tính Gini/entropy cho một phép chia, chạy tay vài vòng gradient boosting, và chạy tay k-means và một vòng EM.

**Kiểm tra nhanh (5 phút, không máy tính).** Câu i kiểm tra mục 5.i: đúng thì đọc lướt, sai thì đọc kỹ. Đúng cả 5 thì làm thẳng bài ★★★ và mini-test.

1. Logistic regression với một mẫu có p = σ(wᵀx) = 0.8 và nhãn y = 0. Gradient của log-loss theo w bằng bao nhiêu (theo x)?
2. Dữ liệu có 1% mẫu dương. Mô hình "luôn đoán âm" có accuracy và recall bằng bao nhiêu?
3. Một điểm có nhãn y = +1 và f(x) = wᵀx + b = 0.4. Hinge loss của nó bằng bao nhiêu? Nó được phân loại đúng hay sai?
4. Thêm cây vào random forest có gây overfit không? Thêm vòng vào gradient boosting thì sao?
5. Hàm mục tiêu J của k-means có thể tăng sau một vòng lặp gán–cập nhật không? k-means có luôn tìm được cực tiểu toàn cục không?

Đáp án ở mục Lời giải cuối chương.

## 5.1 Mô hình tuyến tính: hồi quy, logistic, softmax, GLM

Ba mô hình của mục này có cùng cấu trúc: tổ hợp tuyến tính θᵀx, đi qua một hàm "link ngược" (đồng nhất, sigmoid, softmax), và loss là negative log-likelihood của một phân phối (Gauss, Bernoulli, categorical). Nắm cấu trúc đó thì lớp cuối của mọi mạng phân loại trở nên hiển nhiên.

**Hồi quy tuyến tính.** Giả thuyết hθ(x) = θᵀx (thêm cột 1 vào X để có hệ số chặn). Hàm mất mát là tổng bình phương sai số. Đây chính là bài toán bình phương tối thiểu của chương 1, mục 1.3:

$$
J(\theta) = \tfrac{1}{2}\lVert X\theta - y \rVert_2^2, \qquad \nabla_\theta J = X^{\top}(X\theta - y), \qquad \hat{\theta} = (X^{\top}X)^{-1}X^{\top}y
$$

Khi tính, dùng `np.linalg.lstsq` hoặc QR, không nghịch đảo XᵀX (chương 1); với dữ liệu lớn dùng gradient descent: θ ← θ − η Xᵀ(Xθ − y)/N. Từ đây ta viết w cho vector trọng số; θ chỉ dùng ở hồi quy tuyến tính để khớp với CS229.

**Diễn giải xác suất.** Nếu y = θᵀx + ε với ε ~ 𝒩(0, σ²) độc lập, log-likelihood của N mẫu là hằng số trừ (1/(2σ²))∑(yᵢ − θᵀxᵢ)², nên cực đại likelihood ⇔ cực tiểu bình phương tối thiểu (MLE, chương 3): MSE là giả định nhiễu Gauss viết dưới dạng loss.

**Ví dụ giải mẫu 5.1 — phương trình chuẩn.** x = (0, 1, 2, 3), y = (1, 3, 4, 7), thêm cột 1.

- XᵀX có hàng (4, 6), (6, 14); Xᵀy = (15, 32); det = 20. θ̂ = (14·15 − 6·32, −6·15 + 4·32)/20 = (0.9, 1.9), tức ŷ = 0.9 + 1.9x.
- Phần dư (0.1, 0.2, −0.7, 0.4): tổng bằng 0 (vuông góc cột 1) và 0·0.1 + 1·0.2 + 2·(−0.7) + 3·0.4 = 0 (vuông góc cột x), đúng hình học phép chiếu của chương 1.
- SSE = 0.70; tổng bình phương quanh trung bình là 18.75, nên R² = 1 − 0.70/18.75 ≈ 0.963.

**Logistic regression.** Phân loại nhị phân y ∈ {0, 1}. Giả thuyết p = σ(wᵀx + b) với σ(z) = 1/(1 + e⁻ᶻ). Giả định y ~ Bernoulli(p), negative log-likelihood trung bình là log-loss (binary cross-entropy):

$$
L(w) = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i\log p_i + (1-y_i)\log(1-p_i)\Big], \qquad \nabla_w L = \frac{1}{N}\sum_{i=1}^{N}(p_i - y_i)\,x_i
$$

Gradient có dạng "sai số × đầu vào", giống hệt hồi quy tuyến tính, nhờ σ′(z) = σ(z)(1 − σ(z)) triệt tiêu với đạo hàm của log. Hessian là (1/N)∑ pᵢ(1 − pᵢ)xᵢxᵢᵀ, nửa xác định dương, nên L lồi: mọi cực tiểu địa phương là toàn cục. Nhưng không có nghiệm đóng; phải dùng gradient descent hoặc Newton (chương 4).

Vì sao không dùng MSE cho logistic? (1) (σ(z) − y)² theo z không lồi. (2) Gradient của nó có thêm thừa số σ′(z), gần 0 khi mô hình sai tự tin (z rất âm mà y = 1), nên học chậm đúng lúc cần học nhất. Log-loss phạt sai tự tin gần như tuyến tính theo z.

**Ví dụ giải mẫu 5.2 — một bước gradient, và chuyện gì xảy ra khi dữ liệu tách được.** x = (1, 2, 3, 4), y = (0, 0, 1, 1), khởi tạo w = b = 0, η = 0.5.

- Tại 0: mọi pᵢ = 0.5, loss = ln 2 ≈ 0.6931; ∂L/∂w = mean((p − y)x) = (0.5 + 1 − 1.5 − 2)/4 = −0.5; ∂L/∂b = 0.
- Sau một bước: w = 0.25, b = 0; p = (0.5622, 0.6225, 0.6792, 0.7311); loss còn 0.6250.
- Chạy tiếp, không regularize: (w, b) ≈ (1.716, −3.959) ở bước 100, (8.925, −22.135) ở bước 10 000, (10.312, −25.606) ở bước 20 000. Ranh giới −b/w = 2.3071 (bước 100), 2.4803 (10 000), 2.4830 (20 000), 2.4871 (10⁵), 2.4904 (10⁶), 2.4917 (4 × 10⁶): vẫn tăng, không dừng ở 2.48. Dữ liệu tách được nên nhân (w, b) lên luôn giảm loss thêm chút nữa: loss → 0, ‖w‖ → ∞, w không hội tụ. Nhưng **hướng** của (w, b) thì hội tụ, rất chậm (cỡ 1/log t theo bước t), về nghiệm SVM lề cực đại trên đặc trưng (x, 1): w = 2, b = −5, tức ranh giới 2.5, đúng điểm giữa hai lớp. Đây là **implicit bias** của gradient descent trên dữ liệu tách được (Soudry, Hoffer, Nacson, Gunasekar, Srebro 2018, JMLR): không cần regularization tường minh, GD vẫn chọn nghiệm lề lớn. Số liệu ở trên tự chạy lại với η = 0.5, 4 × 10⁶ bước (khoảng nửa phút bằng NumPy).
- Thêm L2 **chỉ phạt w, không phạt b** (gradient theo w cộng λw, theo b giữ nguyên; λ = 0.1): hội tụ về (1.508, −3.769), ranh giới đúng 2.5 (nhờ dữ liệu đối xứng quanh 2.5). Nếu phạt cả b thì ra (0.617, −1.062), ranh giới 1.72: phạt b kéo ranh giới về gốc. Regularization biến bài toán "vô nghiệm" thành có nghiệm duy nhất. Đây là cầu sang mục 5.2.

% manim: LogisticTachDuoc

**Softmax regression.** Với K lớp, mỗi lớp có một vector trọng số; logit z = Wᵀx ∈ ℝᴷ, xác suất là softmax, loss là cross-entropy với nhãn one-hot Y:

$$
p_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \qquad L = -\log p_{y}, \qquad \frac{\partial L}{\partial z} = p - Y
$$

Logistic là trường hợp K = 2. Softmax bất biến khi cộng cùng một hằng vào mọi logit (eᶜ triệt tiêu ở tử và mẫu), nên W không duy nhất (chỉ hiệu giữa các cột xác định được) và ta luôn tính softmax trên z − max(z), nếu không e¹⁰⁰⁰ tràn thành inf và kết quả ra nan.

**Ví dụ giải mẫu 5.3 — softmax và gradient.** Logit z = (2, 1, 0.1).

- eᶻ ≈ (7.389, 2.718, 1.105), tổng 11.212 ⇒ p ≈ (0.659, 0.242, 0.099). Nhãn đúng là lớp 0: CE = −ln 0.659 ≈ 0.417.
- Gradient theo logit p − Y = (−0.341, 0.242, 0.099), tổng bằng 0: tăng logit lớp đúng, giảm các lớp khác, tỉ lệ với xác suất đang gán sai.
- Cộng 100 vào mọi logit: p không đổi; cộng 1000 thì e¹⁰⁰⁰ tràn, còn trừ max thì luôn an toàn.

**GLM ở mức trực giác.** Chọn phân phối của y trong họ mũ, liên kết tham số tự nhiên của nó với θᵀx; loss là negative log-likelihood: Gauss + link đồng nhất = hồi quy tuyến tính (MSE); Bernoulli + logit = logistic (log-loss); categorical = softmax (cross-entropy); Poisson + link log = hồi quy đếm (chỉ nêu tên). Với link chuẩn, gradient luôn là (dự đoán − nhãn) × x.

**Vì sao ML cần cái này.**

- Lớp cuối của mọi mô hình ngôn ngữ là softmax regression trên hidden state: logit = Wh, loss = cross-entropy với token tiếp theo. Gradient p − Y ở đây chính là tín hiệu bắt đầu backprop (chương 6).
- Hiện tượng ‖w‖ → ∞ trên dữ liệu tách được có bản sao trong deep learning: mạng nội suy được tập train thì cross-entropy cứ đẩy logit to ra, mô hình tự tin quá mức. Weight decay, early stopping và label smoothing đều là cách hãm lại.
- Ngay cả khi không hãm, GD không chọn nghiệm tùy tiện: trên dữ liệu tách được, hướng của (w, b) hội tụ về nghiệm lề cực đại (implicit bias, ví dụ 5.2). Đó là một mảnh của câu hỏi nghiên cứu "vì sao mạng quá tham số hóa vẫn tổng quát hóa".

**Thực tế tính toán.** Đừng viết `log(sigmoid(z))` (z rất âm thì ra −inf): dùng `np.logaddexp(0, -z)` hoặc hàm loss "with logits"; cross-entropy tính bằng log-softmax, không lấy log của softmax.

## 5.2 Tổng quát hóa: bias–variance, regularization, CV, metric, leakage

Mục tiêu của học máy là sai số trên dữ liệu chưa thấy, không phải trên train. Regularization, cross-validation, metric và chống leakage đều phục vụ một việc: ước lượng và cải thiện sai số đó một cách trung thực. Đây là mục bị hỏi nhiều nhất.

**Train, validation, test.** Train để fit tham số, validation để chọn siêu tham số và dừng sớm, test dùng một lần ở cuối. Underfitting: sai cả train lẫn validation; overfitting: train tốt, validation tệ.

**Phân rã bias–variance.** Cố định một điểm x, nhãn y = f(x) + ε với 𝔼[ε] = 0, Var ε = σ². Mô hình f̂ được huấn luyện trên một tập dữ liệu ngẫu nhiên, nên f̂(x) là biến ngẫu nhiên. Với squared loss:

$$
\mathbb{E}\big[(y - \hat{f}(x))^2\big] = \underbrace{\sigma^2}_{\text{nhiễu}} + \underbrace{\big(f(x) - \mathbb{E}\hat{f}(x)\big)^2}_{\text{bias}^2} + \underbrace{\mathbb{E}\big[(\hat{f}(x) - \mathbb{E}\hat{f}(x))^2\big]}_{\text{variance}}
$$

Kỳ vọng lấy trên cả nhiễu test lẫn tập train. Bias: mô hình trung bình lệch sự thật bao nhiêu (quá đơn giản). Variance: mô hình dao động bao nhiêu khi đổi tập train (quá linh hoạt). σ² không giảm được.

**Ví dụ giải mẫu 5.4 — bias–variance bằng mô phỏng.** f(x) = sin(2πx), σ = 0.3 (σ² = 0.09), 20 điểm x cách đều trên \[0, 1\], lấy lại nhãn nhiễu 2000 lần, khớp đa thức, đo trên 100 điểm trong \[0.05, 0.95\] (seed `default_rng(0)`):

| Bậc đa thức | bias² | variance | tổng (+ σ²) | chính xác (bias² / variance) |
| --- | --- | --- | --- | --- |
| 1 | 0.1809 | 0.0078 | 0.2787 | 0.1816 / 0.0079 |
| 3 | 0.0054 | 0.0136 | 0.1090 | 0.0054 / 0.0138 |
| 9 | 0.0000 | 0.0380 | 0.1280 | 0.0000 / 0.0377 |

Cột cuối tính bằng phép chiếu tuyến tính, không phụ thuộc rng (sai số mô phỏng cỡ 10⁻³). Bảng mô phỏng chỉ tái hiện đúng đến chữ số thứ tư nếu dùng **một** `default_rng(0)` chung cho cả ba bậc theo thứ tự 1 → 3 → 9; code ở Lab 2 (tiếp). Bậc 1 không uốn được theo hình sin: bias lớn. Bậc 9 trung bình gần trùng sự thật nhưng dao động mạnh: variance gấp gần 3 lần bậc 3. Bậc 3 cân bằng nhất. Mô phỏng này cố định x. Nếu mỗi lần lấy lại, 20 giá trị x cũng được rút ngẫu nhiên đều trên [0, 1] và đo cả ở biên, thì ma trận Vandermonde bậc 9 có điều kiện rất xấu khi x có khoảng trống, và variance bậc 9 tăng hàng trăm đến hàng trăm nghìn lần so với 0.038 của thiết kế cố định: trung vị theo điểm khoảng 30 (gấp gần 800 lần), trung bình khoảng 1.7 × 10⁵ vì vài bộ dữ liệu làm đa thức bùng nổ ở biên (2000 lần lấy lại, giá trị tính lại ngày 04/11/2026). Bài học: bậc cao cộng thiết kế x ngẫu nhiên, thưa ở biên, là công thức cho variance khổng lồ; đây cũng là lý do ví dụ cố định x.

% manim: BiasVarianceDaThuc

**Regularization L2 và L1.** Thêm số hạng phạt độ lớn trọng số:

$$
\text{ridge: } \min_w \tfrac{1}{2}\lVert y - Xw\rVert_2^2 + \tfrac{\lambda}{2}\lVert w\rVert_2^2 \qquad\qquad \text{lasso: } \min_w \tfrac{1}{2}\lVert y - Xw\rVert_2^2 + \lambda\lVert w\rVert_1
$$

- **L2 (ridge, weight decay).** Nghiệm (XᵀX + λI)⁻¹Xᵀy; tương đương MAP với prior Gauss, λ = σ²/τ² (bài 6). Co mọi hệ số về 0 nhưng hiếm khi đúng bằng 0.
- **L1 (lasso, Tibshirani 1996).** Tương đương MAP với prior Laplace. Cho nghiệm thưa: nhiều hệ số đúng bằng 0, tức là tự chọn đặc trưng. Không có nghiệm đóng tổng quát; giải bằng coordinate descent (lab 2).
- **Hình học.** Tương đương cực tiểu loss trong "quả cầu" ‖w‖ ≤ t: đồng mức elip phình ra đến khi chạm quả cầu. Quả cầu L2 tròn, điểm chạm thường có mọi tọa độ khác 0; quả cầu L1 là hình thoi có đỉnh trên trục, điểm chạm thường là một đỉnh, tức một số tọa độ bằng 0. (Elastic net, chỉ nêu tên, trộn cả hai.)

**Ví dụ giải mẫu 5.5 — ridge và lasso trên thiết kế trực chuẩn.** Khi XᵀX = I, nghiệm không phạt là z = Xᵀy và hai bài toán tách thành từng tọa độ: ridge cho wⱼ = zⱼ/(1 + λ), lasso cho wⱼ = sign(zⱼ)·max(|zⱼ| − λ, 0) (soft-thresholding, chứng minh ở bài 7).

- z = (3, 0.8, −2), λ = 1. Ridge: (1.5, 0.4, −1.0) — chia đều cho 2, không số nào bằng 0.
- Lasso: (2, 0, −1) — trừ đều λ về phía 0, số nào có |z| ≤ λ bị cắt đúng về 0.
- Một chiều không trực chuẩn: x = (1, 2, 3), y = (2, 4, 7), w = 31/(14 + λ): λ = 0, 1, 10 → 2.2143, 2.0667, 1.2917.

% manim: L1L2HinhHoc

**Cross-validation.** k-fold: chia dữ liệu thành k phần, lần lượt dùng mỗi phần làm validation, lấy trung bình. Các biến thể bắt buộc biết:

- **Stratified** (giữ tỉ lệ lớp mỗi fold, quan trọng khi lớp hiếm); **group** (mọi mẫu của một người dùng, tài liệu hay bệnh nhân nằm chung một fold); **time series** (train quá khứ, validate tương lai).
- **Nested CV**: điểm CV dùng để chọn siêu tham số là ước lượng lạc quan, vì ta đã chọn cái tốt nhất trong nhiều lần thử. Muốn trung thực thì chọn siêu tham số bằng CV bên trong mỗi fold train, đánh giá trên fold ngoài (Cawley & Talbot 2010).

**Metric phân loại.** Từ ma trận nhầm lẫn TP, FP, FN, TN:

Precision = TP/(TP + FP) (trong các cảnh báo, bao nhiêu là thật); recall = TPR = TP/(TP + FN) (bắt được bao nhiêu ca dương); FPR = FP/(FP + TN) (báo nhầm bao nhiêu ca âm); F1 = 2PR/(P + R), trung bình điều hòa của P và R; balanced accuracy = (TPR + TNR)/2.

ROC là đường TPR theo FPR khi quét ngưỡng; AUC là diện tích dưới nó, cũng bằng xác suất một mẫu dương ngẫu nhiên có điểm cao hơn một mẫu âm ngẫu nhiên (cặp hòa tính 0.5). PR curve là precision theo recall; average precision (AP) tóm tắt nó bằng ∑(Rₙ − Rₙ₋₁)Pₙ qua các ngưỡng.

**Ví dụ giải mẫu 5.6 — lớp dương 1%.** N = 10 000, 100 dương. Mô hình cho TP = 90, FN = 10, FP = 495, TN = 9405.

- Recall 0.90, FPR = 495/9900 = 0.05, precision = 90/585 ≈ 0.154, F1 ≈ 0.263, balanced accuracy 0.925, accuracy 0.9495. "Luôn đoán âm" có accuracy 0.99, cao hơn, nhưng recall 0: accuracy vô nghĩa khi lớp lệch.
- Giữ TPR 0.9 và FPR 0.05, chỉ đổi tỉ lệ dương π: precision = 0.9π/(0.9π + 0.05(1 − π)). π = 50% → 0.947; π = 10% → 0.667; π = 1% → 0.154. ROC (và AUC) không đổi vì TPR, FPR chỉ tính trong từng lớp; precision trộn hai lớp nên phụ thuộc π. Khi lớp hiếm, nhìn PR/AP.
- Ngưỡng theo chi phí: FN tốn 10, FP tốn 1, xác suất đã hiệu chỉnh ⇒ báo dương khi 10p > 1·(1 − p) ⇔ p > 1/11 ≈ 0.091, không phải 0.5.

**Ví dụ giải mẫu 5.7 — AUC bằng đếm cặp, và AP.** Điểm (0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.2), nhãn (1, 1, 0, 1, 0, 0, 1, 0).

- 16 cặp: dương 0.9 và 0.8 thắng cả 4 âm (8); dương 0.6 thắng 3; dương 0.3 thắng 1. Tổng 12/16 ⇒ AUC = 0.75.
- Sắp theo điểm giảm dần, precision tại các vị trí có mẫu dương: 1/1, 2/2, 3/4, 4/7. AP = (1 + 1 + 0.75 + 0.5714)/4 ≈ 0.830.

% manim: NguongVaPrecision

**Data leakage.** Thông tin từ tập đánh giá lọt vào quá trình huấn luyện, khiến điểm số đẹp hơn thực tế. Các dạng phổ biến:

1. **Tiền xử lý trên toàn bộ dữ liệu trước khi chia** (chuẩn hóa, điền thiếu, chọn đặc trưng, PCA): mọi bước học từ dữ liệu phải fit trên fold train rồi chỉ *áp* lên validation.
2. **Trùng lặp**: một mẫu (hoặc bản augment của nó) nằm ở cả train và test.
3. **Theo thời gian hoặc theo nhóm**: đặc trưng tính từ tương lai, chia ngẫu nhiên chuỗi thời gian, hoặc một tài liệu bị cắt thành đoạn nằm ở cả hai phía.
4. **NLP hiện đại — contamination**: dữ liệu test của benchmark lẫn vào dữ liệu pretraining của LM.

*Thí nghiệm nhỏ.* n = 100, d = 5000 đặc trưng ngẫu nhiên, nhãn ngẫu nhiên (đáp án đúng ≈ 0.5). Chọn 20 đặc trưng tương quan mạnh nhất với y trên **toàn bộ** dữ liệu rồi 5-fold CV với nearest-centroid: accuracy khoảng 0.90. Chọn đặc trưng **trong từng fold train**: khoảng 0.50. Cụ thể: dữ liệu sinh bằng `default_rng(s)` (X chuẩn tắc, y tung đồng xu), thứ tự fold lấy từ `permutation` của chính `default_rng(s)`, k = 20 cố định, độ chọn bằng |tương quan|. Với s = 1 ra 0.89 và 0.52; trung bình 20 seed (s = 0…19) là 0.90 ± 0.02 và 0.50 ± 0.05 (độ lệch chuẩn giữa các seed). Đáp án đúng ≈ 0.5, nên 0.45 hay 0.52 đều không có nghĩa là tệ hơn hay tốt hơn tung đồng xu. Với 5000 đặc trưng, luôn có vài chục đặc trưng tình cờ tương quan với nhãn trên chính 100 mẫu đó; chọn chúng bằng cả dữ liệu test là "nhìn đáp án". Bài săn bug dựa trên hiện tượng này.

**Vì sao ML cần cái này.**

- Weight decay trong huấn luyện mạng là L2 (với SGD thuần hai cách viết trùng nhau; với Adam thì khác, AdamW tách riêng weight decay: chương 4, mục 4.4).
- Early stopping là regularization ngầm, gần giống L2 trong trường hợp đơn giản: mô hình tuyến tính (hoặc bậc hai), GD khởi tạo từ 0, khi đó η·t đóng vai 1/λ. Với mạng sâu đây chỉ là trực giác.
- Benchmark NLP thường lệch lớp: báo F1 hoặc AP, nói rõ ngưỡng, và kiểm contamination khi đánh giá LM.

**Thực tế khác lý thuyết — double descent.** Chữ U cổ điển không phải toàn bộ câu chuyện: với mô hình quá tham số hóa, sai số test có thể đạt đỉnh quanh ngưỡng nội suy (số tham số ≈ số mẫu) rồi giảm lại khi tăng tiếp kích thước (CS229 main notes). Chỉ giới thiệu; đừng dùng nó để bỏ qua validation.

## 5.3 SVM: margin, hinge loss, kernel trick

SVM tìm siêu phẳng phân cách có lề rộng nhất. Viết dưới dạng loss, nó là hinge loss + L2, cùng họ với logistic regression; điểm khác biệt thật nằm ở dạng đối ngẫu, nơi dữ liệu chỉ xuất hiện qua tích vô hướng và kernel trick khả thi.

**Margin.** Nhãn y ∈ {−1, +1}, bộ phân loại f(x) = wᵀx + b. Functional margin là m = y·f(x): dương khi phân loại đúng, càng lớn càng chắc. Geometric margin là khoảng cách có dấu từ điểm tới siêu phẳng: y(wᵀx + b)/‖w‖₂. Nhân w và b với cùng một số không đổi siêu phẳng, nên ta chuẩn hóa để điểm gần nhất có functional margin bằng 1; khi đó bề rộng lề giữa hai mặt wᵀx + b = ±1 là 2/‖w‖₂.

**Hard-margin và soft-margin.** Tối đa hóa 2/‖w‖₂ ⇔ tối thiểu ½‖w‖₂²:

$$
\text{hard: } \min_{w,b} \tfrac{1}{2}\lVert w\rVert_2^2 \;\;\text{s.t. } y_i(w^{\top}x_i + b) \ge 1 \qquad\qquad \text{soft: } \min_{w,b} \tfrac{1}{2}\lVert w\rVert_2^2 + C\sum_{i=1}^{N}\max\big(0,\, 1 - y_i(w^{\top}x_i + b)\big)
$$

Soft-margin (Cortes & Vapnik 1995) cho phép vi phạm lề với giá C mỗi đơn vị. Chia cho C: tổng hinge loss cộng (1/(2C))‖w‖₂², nên **C lớn = regularization yếu**; nếu muốn dùng hinge trung bình thì hệ số phạt là 1/(2CN). Support vector là các điểm có αᵢ > 0 (mục đối ngẫu bên dưới): nằm đúng trên lề hoặc vi phạm lề (hinge > 0); chỉ chúng quyết định w. Điểm nằm trên lề vẫn có thể có αᵢ = 0 (suy biến), nên "trên lề" chưa đủ để kết luận.

**Ví dụ giải mẫu 5.8 — margin và hinge.** w = (1, 1), b = −3. Năm điểm và nhãn: (1, 1)/−1, (2, 2)/−1, (3, 3)/+1, (1, 3)/+1, (2.5, 1)/+1.

- wᵀx + b lần lượt là −1, 1, 3, 1, 0.5; nhân với y được margin m = (1, −1, 3, 1, 0.5).
- Hinge = max(0, 1 − m) = (0, 2, 0, 0, 0.5).
- (2, 2) sai (m < 0, hinge 2); (2.5, 1) đúng phía nhưng trong lề (0 < m < 1, hinge 0.5); (1, 1) và (1, 3) đúng trên lề (m = 1); (3, 3) ngoài lề, không đóng góp.
- Bề rộng lề 2/‖w‖₂ = 2/√2 = √2 ≈ 1.414.
- Hard-margin với hai điểm (1, 1)/−1 và (3, 3)/+1: w ∥ (1, 1) theo đối xứng, giải được w = (0.5, 0.5), b = −2. Lề 2√2 ≈ 2.828, đúng bằng khoảng cách giữa hai điểm.

**So sánh ba loss trên trục margin.** 0-1 loss là bậc thang (1 nếu m < 0): không lồi, gradient bằng 0 hầu khắp nơi. Hinge max(0, 1 − m) và log-loss ln(1 + e⁻ᵐ) (logistic theo nhãn ±1) đều lồi và, sau khi chia log-loss cho ln 2, đều nằm trên 0-1 loss: "cận trên lồi" (convex surrogate). Hinge bằng đúng 0 khi m ≥ 1 (điểm xa lề bị bỏ qua ⇒ nghiệm thưa theo mẫu); log-loss không bao giờ bằng 0 (mọi điểm kéo nhẹ ⇒ cho xác suất).

% manim: HamMatMatSoSanh

**Đối ngẫu ở mức trực giác.** Dùng nhân tử Lagrange αᵢ ≥ 0 cho từng ràng buộc (chương 4). Điều kiện tối ưu cho w = ∑ αᵢyᵢxᵢ (bài 12 tự suy ra), và bài toán đối ngẫu là

$$
\max_{\alpha}\; \sum_{i=1}^{N}\alpha_i - \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_j y_i y_j\, x_i^{\top}x_j \qquad \text{s.t. } 0 \le \alpha_i \le C,\;\; \sum_{i=1}^{N}\alpha_i y_i = 0
$$

Đây là dạng **soft-margin**: cận trên C đến từ giá C của vi phạm lề. Với **hard-margin** (dữ liệu tách được, không có biến lỏng) cận trên biến mất, chỉ còn αᵢ ≥ 0 (tương ứng C = ∞), và đó là bài toán của bài 12 và của ví dụ hai điểm dưới đây (hai nhân tử đều nhỏ hơn mọi C đủ lớn nên cận không ảnh hưởng).

αᵢ > 0 chỉ với support vector (điều kiện bù KKT). Với hai điểm ở trên: α = (0.25, 0.25) cho w = 0.25·(3, 3) − 0.25·(1, 1) = (0.5, 0.5), ∑αᵢyᵢ = 0, giá trị đối ngẫu 0.5 − ½·0.5 = 0.25 = ½‖w‖₂² của bài toán gốc. Dự đoán f(x) = ∑ αᵢyᵢ xᵢᵀx + b. Điểm then chốt: **dữ liệu chỉ xuất hiện qua tích vô hướng xᵢᵀxⱼ**.

**Kernel trick.** Thay mọi tích vô hướng bằng K(x, z) = φ(x)ᵀφ(z), với φ là một ánh xạ đặc trưng (có thể vô hạn chiều) mà ta không bao giờ phải tính tường minh. K hợp lệ khi và chỉ khi K **đối xứng** và mọi ma trận Gram Kᵢⱼ = K(xᵢ, xⱼ) đều nửa xác định dương (điều kiện Mercer; nối với PSD ở chương 1, mục 1.4).

- Đa thức: K(x, z) = (xᵀz + c)ᵈ. RBF: K(x, z) = exp(−γ‖x − z‖₂²); γ lớn ⇒ mỗi điểm chỉ ảnh hưởng vùng rất gần ⇒ ranh giới ngoằn ngoèo, variance cao.

**Ví dụ giải mẫu 5.9 — kernel trick bằng số.** x = (1, 2), z = (3, −1).

- K(x, z) = (xᵀz)² = (3 − 2)² = 1. Với φ(v) = (v₁², √2·v₁v₂, v₂²): φ(x) = (1, 2√2, 4), φ(z) = (9, −3√2, 1), tích = 9 − 12 + 4 = 1: khớp, mà không cần dựng φ.
- RBF với γ = 0.5: ‖x − z‖₂² = 13, K = e⁻⁶·⁵ ≈ 0.0015: hai điểm xa nhau gần như "không giống nhau".
- XOR: (1, 1), (−1, −1) nhãn +1; (1, −1), (−1, 1) nhãn −1. Không đường thẳng nào tách được, nhưng đặc trưng x₁x₂ tách hoàn toàn, và x₁x₂ là một tọa độ (chia √2) của φ ở trên.

**Vì sao ML cần cái này.**

- Hinge và margin sống tiếp trong ranking, contrastive và triplet loss: "cặp đúng phải điểm cao hơn cặp sai ít nhất một lề".
- Ma trận Gram và kernel xuất hiện lại trong attention (ở mức trực giác: exp(qᵀk/√d) có dạng mũ của tích vô hướng, giống một kernel, nhưng q và k là hai phép chiếu tuyến tính khác nhau của x (hai ma trận trọng số khác nhau) nên hàm của x, x′ không đối xứng và không phải kernel Mercer; softmax chuẩn hóa theo hàng, giống ước lượng Nadaraya–Watson. Hướng phân tích attention như kernel: Tsai và cộng sự 2019 [cần kiểm chứng tên và năm trước khi trích]) và trong lý thuyết neural tangent kernel.
- Câu phỏng vấn kinh điển: "SVM trên dữ liệu không tách tuyến tính thì sao?" Trả lời: soft-margin với C, rồi kernel, rồi chi phí.

**Thực tế tính toán.** SVM kernel cần ma trận Gram N × N: N = 100 000 là 10¹⁰ phần tử, 80 GB ở float64; tài liệu scikit-learn nói thời gian huấn luyện SVC tăng nhanh hơn bậc hai theo N, ước chừng giữa O(d·N²) và O(d·N³) tùy dữ liệu và C [cần kiểm chứng: chưa mở lại được trang tài liệu ngày 04/11/2026]. Với N lớn: SVM tuyến tính (SGD trên bài toán gốc), random features, hoặc đổi mô hình. SVM không cho xác suất: `predict_proba` của scikit-learn là Platt scaling (logistic regression trên điểm SVM, fit bằng CV nội bộ) và có thể mâu thuẫn với `predict`. Luôn chuẩn hóa đặc trưng trước SVM, vì RBF dùng khoảng cách.

## 5.4 Cây quyết định và ensemble: bagging, random forest, boosting

Cây quyết định là mô hình linh hoạt nhưng không ổn định: bias thấp, variance cao. Ensemble sửa đúng điểm yếu đó theo hai hướng ngược nhau: bagging lấy trung bình nhiều cây sâu để giảm variance; boosting cộng dồn nhiều cây nông để giảm bias.

**Cây quyết định.** Chia không gian đặc trưng thành các hình hộp bằng câu hỏi "xⱼ < t?"; mỗi lá dự đoán lớp đa số hoặc trung bình. Tìm cây tối ưu là NP-khó, nên chia **tham lam**: tại mỗi nút, thử mọi (j, t) và chọn phép chia giảm tạp chất nhiều nhất. Với tỉ lệ lớp pₖ tại một nút:

$$
\text{Gini} = 1 - \sum_{k} p_k^2, \qquad H = -\sum_{k} p_k\log_2 p_k, \qquad \Delta = I(\text{cha}) - \frac{n_L}{n}I(L) - \frac{n_R}{n}I(R)
$$

Với hồi quy, tạp chất là phương sai (MSE). Dừng chia bằng độ sâu tối đa, số mẫu tối thiểu ở lá, hoặc tỉa cây. Cây sâu khớp được cả nhiễu.

**Ví dụ giải mẫu 5.10 — chọn phép chia bằng Gini và entropy.** Nút cha 10 mẫu, 5 dương 5 âm: Gini 0.5, entropy 1 bit.

- Chia A: trái (4 dương, 1 âm), phải (1, 4). Gini mỗi bên 0.32, giảm Gini 0.5 − 0.32 = 0.18; entropy mỗi bên 0.7219, information gain 0.2781.
- Chia B: trái (5, 2), phải (0, 3). Gini trái 1 − (5/7)² − (2/7)² = 0.4082, phải 0; giảm Gini 0.5 − 0.7·0.4082 = 0.2143. Information gain 0.3958.
- Cả hai tiêu chí chọn B: một nút thuần khiết được thưởng lớn, dù bên kia còn lẫn.

**Bagging (bootstrap aggregating).** Huấn luyện B cây trên B mẫu bootstrap (rút N mẫu có hoàn lại), rồi lấy trung bình hoặc bỏ phiếu. Mỗi mẫu có xác suất (1 − 1/N)ᴺ ≈ 1/e ≈ 0.368 không được rút vào một bootstrap: các mẫu **out-of-bag (OOB)** là validation miễn phí.

Nếu mỗi cây có phương sai σ² và hai cây bất kỳ có hệ số tương quan ρ, thì phương sai của trung bình B cây là

$$
\operatorname{Var}\Big(\frac{1}{B}\sum_{b=1}^{B}\hat{f}_b(x)\Big) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2
$$

Tăng B chỉ xóa số hạng thứ hai; sàn ρσ² còn nguyên.

**Ví dụ giải mẫu 5.11 — vì sao cần giảm tương quan.**

- ρ = 0.3: B = 10 → 0.37σ²; B = 100 → 0.307σ². Thêm 90 cây chỉ giảm thêm 0.063σ².
- ρ = 0.7, B = 100 → 0.703σ²: chỉ giảm 30% so với một cây, và B = 10 đã đạt 0.73σ², nên thêm 90 cây gần như vô ích. (σ² ở đây là phương sai của một cây, không phải phương sai nhiễu của mục 5.2.)
- ρ = 0.2: để nằm trong 5% của sàn 0.2σ² cần 0.8/B ≤ 0.01, tức B ≥ 80.
- Tỉ lệ OOB với N = 1000: (1 − 1/1000)¹⁰⁰⁰ ≈ 0.3677, rất gần 1/e ≈ 0.3679.

**Random forest** (Breiman 2001) = bagging + tại mỗi lần chia chỉ xét một tập con ngẫu nhiên các đặc trưng (thường khoảng √d cho phân loại). Mỗi cây kém hơn một chút nhưng các cây khác nhau hơn nhiều: ρ giảm, sàn ρσ² giảm. Sai số tổng quát hóa hội tụ khi số cây tăng (Breiman), nên **thêm cây không gây overfit**, chỉ tốn thời gian.

**Boosting.** AdaBoost (chỉ nêu tên) huấn luyện tuần tự các bộ phân loại yếu, tăng trọng số mẫu bị sai, và cho mỗi bộ trọng số α = ½ ln((1 − ε)/ε) (ε = 0.3 ⇒ α ≈ 0.424). Gradient boosting (Friedman 2001) tổng quát hóa ý tưởng: **gradient descent trong không gian hàm**.

$$
F_0 = \arg\min_c \sum_i \ell(y_i, c), \qquad r_i^{(m)} = -\frac{\partial \ell(y_i, F)}{\partial F}\Big|_{F = F_{m-1}(x_i)}, \qquad F_m = F_{m-1} + \eta\, h_m
$$

Mỗi vòng, khớp một cây nhỏ hₘ với âm gradient rᵢ của loss theo dự đoán hiện tại, rồi đi một bước η (shrinkage). Với MSE ½(y − F)², âm gradient là phần dư y − F; với log-loss, rᵢ = yᵢ − pᵢ.

**Ví dụ giải mẫu 5.12 — gradient boosting bằng tay (MSE, stump, η = 0.5).** x = (1, …, 6), y = (1.0, 1.2, 1.1, 3.0, 3.2, 2.9).

- F₀ = trung bình = 2.0667, MSE ban đầu 0.9456.
- Vòng 1: phần dư (−1.067, −0.867, −0.967, 0.933, 1.133, 0.833); stump tốt nhất chia tại x < 3.5, dự đoán ∓0.967. F = 2.0667 ± 0.5·0.967 = 1.583 (ba điểm đầu), 2.550 (ba điểm sau); MSE 0.2447.
- Vòng 2: phần dư trung bình −0.483 (trái), 0.483 (phải); F = 1.342 / 2.792, MSE 0.0695.
- Vòng 3: F = 1.221 / 2.913, MSE 0.0257. Mỗi vòng xóa nửa khoảng cách còn lại tới trung bình mỗi nhóm (1.1 và 3.033).
- η = 1 tới đích ngay vòng 1. η nhỏ đi chậm hơn nhưng mỗi cây chỉ đóng góp một phần, các cây sau có cơ hội sửa sai và mô hình thường tổng quát hóa tốt hơn; đổi lại cần nhiều vòng và early stopping.

% manim: BoostingPhanDu

**Bagging và boosting, tóm tắt.** Bagging/random forest giảm variance, dùng cây sâu, huấn luyện song song, thêm cây không overfit; siêu tham số chính là số đặc trưng mỗi lần chia. Gradient boosting giảm bias, dùng cây nông (độ sâu 3–8), huấn luyện tuần tự, thêm vòng có thể overfit (cần early stopping); siêu tham số chính là η, số vòng, độ sâu, subsample.

**Feature importance.** Importance theo giảm tạp chất tính trên train và thiên vị cột nhiều giá trị (một cột ID ngẫu nhiên có thể xếp hạng cao); dùng **permutation importance** trên held-out.

**Vì sao ML cần cái này.**

- Gradient boosting là ví dụ cụ thể nhất của "tối ưu trong không gian hàm" và là trực giác cho residual connection: mỗi khối chỉ học phần sửa thêm.
- Ensemble nhiều checkpoint hoặc seed tăng độ chính xác và ước lượng bất định của mạng nơ-ron; ρσ² giải thích vì sao ensemble các mô hình quá giống nhau lợi rất ít. Trên dữ liệu bảng, GBDT là baseline phải đánh bại.

**Thực tế khác lý thuyết — GBDT là baseline khó đánh trên dữ liệu bảng (mức chắc chắn: trung bình, đang bị thách thức).** Grinsztajn, Oyallon, Varoquaux (2022) so sánh 45 bộ dữ liệu bảng cỡ vừa (~10 000 mẫu) với ngân sách tìm siêu tham số lớn: mô hình dựa trên cây (XGBoost, random forest) vẫn tốt hơn mạng nơ-ron huấn luyện từ đầu. Ba lý giải của paper: mạng nơ-ron thiên về nghiệm quá trơn nên khó khớp hàm mục tiêu nhiều bậc thang; mạng kiểu MLP nhạy với đặc trưng vô ích (thêm cột vô ích làm khoảng cách với cây rộng ra); và bất biến theo phép quay là bất lợi vì mỗi cột của dữ liệu bảng có nghĩa riêng (xoay ngẫu nhiên dữ liệu làm đảo thứ hạng). Paper **không** quy cho giá trị thiếu hay biến phân loại: nó xóa các hàng và cột thiếu khi tiền xử lý, và nói biến phân loại không phải điểm yếu chính của mạng (đối chiếu bản ar5iv ngày 04/11/2026). Việc GBDT hiện đại xử lý biến phân loại và giá trị thiếu thuận tiện là một ưu điểm thực hành riêng, không phải lý giải của paper. Kết luận gắn với điều kiện của paper (45 bộ, ~10 000 mẫu, huấn luyện từ đầu). Mô hình nền cho dữ liệu bảng thách thức nó ở quy mô nhỏ: TabPFN v2 (Hollmann và cộng sự, Nature 2025) báo cáo (phân loại) vượt các phương pháp trước, kể cả ensemble baseline tinh chỉnh 4 giờ, trên dữ liệu đến 10 000 mẫu (đối chiếu phần tóm tắt ngày 04/11/2026; chưa kiểm số liệu trong thân bài, và có thể đã có phiên bản mới với giới hạn kích thước lớn hơn [cần kiểm chứng]). Vì vậy GBDT vẫn là baseline phải đánh bại, nhưng đừng nói "luôn thắng". Thư viện: XGBoost (Chen & Guestrin 2016), LightGBM, CatBoost.

## 5.5 Học không giám sát: k-means, GMM và EM

Không có nhãn, ta tìm cấu trúc: các cụm. k-means gán cứng mỗi điểm cho một tâm; GMM gán mềm theo xác suất; EM là thuật toán tối ưu chung cho mô hình có biến ẩn, và lập luận chứng minh nó đúng (cận dưới ELBO) là nền tảng của VAE. (PCA, phương pháp không giám sát còn lại, nằm ở chương 1.)

**k-means.** Cho K tâm μ₁, …, μₖ và gán c(i) của mỗi điểm. Hàm biến dạng (distortion):

$$
J(c, \mu) = \sum_{i=1}^{N}\lVert x_i - \mu_{c(i)}\rVert_2^2
$$

Lặp hai bước: (1) **gán** mỗi điểm cho tâm gần nhất — tối ưu J theo c khi cố định μ; (2) **cập nhật** mỗi tâm thành trung bình các điểm của nó — tối ưu J theo μ khi cố định c. Đây là coordinate descent trên J, nên J không bao giờ tăng; có hữu hạn cách gán nên thuật toán dừng sau hữu hạn bước, nhưng chỉ tại một **cực tiểu địa phương**.

**Ví dụ giải mẫu 5.13 — k-means kẹt.** Dữ liệu 1 chiều (1, 2, 3, 8, 9, 10, 25), K = 2.

- Khởi tạo (1, 10): gán {1, 2, 3} và {8, 9, 10, 25}; tâm (2, 13); gán lại không đổi. J = (1 + 0 + 1) + (25 + 16 + 9 + 144) = 196.
- Khởi tạo (1, 25): gán {1, …, 10} và {25}; tâm (5.5, 25); dừng. J = 77.5.
- Cả hai là điểm dừng, J chênh gần 2.5 lần. Cách chữa: chạy nhiều khởi tạo, giữ J nhỏ nhất. Điểm ngoại lai (25) cũng có thể chiếm hẳn một cụm vì bình phương khoảng cách phạt điểm xa rất nặng. Chú ý: J = 77.5 là tối ưu toàn cục (duyệt hết 63 cách chia thành hai nhóm), nhưng nó chính là nghiệm dành cả một cụm cho ngoại lai; J nhỏ hơn không có nghĩa là cụm có ý nghĩa hơn.

**Ví dụ giải mẫu 5.14 — khởi tạo trải đều.** (0, 1, 2, 10, 11, 12, 20, 21, 22), K = 3. Khởi tạo (0, 1, 2), cả ba tâm dồn vào một cụm, dừng ở tâm (0, 1.5, 16), J = 154.5; khởi tạo trải đều (0, 11, 21) cho tâm (1, 11, 21), J = 6. Đó là ý tưởng của **k-means++** (Arthur & Vassilvitskii 2007): chọn tâm tiếp theo với xác suất tỉ lệ bình phương khoảng cách tới tâm gần nhất đã có (lab 5).

**Chọn K và các giả định.** J luôn giảm khi tăng K (K = N cho J = 0), nên không chọn K bằng J trên train; dùng khuỷu tay, silhouette hoặc mục tiêu thực tế. k-means ngầm giả định cụm hình cầu, cỡ tương đương, và nhạy với thang đo và ngoại lai.

**Gaussian mixture model.** Mỗi điểm sinh ra bằng cách chọn thành phần ẩn z ~ Categorical(π), rồi x ~ 𝒩(μₖ, Σₖ) với k = z:

$$
p(x) = \sum_{k=1}^{K}\pi_k\,\mathcal{N}(x;\mu_k,\Sigma_k), \qquad \gamma_{ik} = p(z_i = k \mid x_i) = \frac{\pi_k\mathcal{N}(x_i;\mu_k,\Sigma_k)}{\sum_{j}\pi_j\mathcal{N}(x_i;\mu_j,\Sigma_j)}
$$

γᵢₖ (responsibility) là gán **mềm**: điểm nằm giữa hai cụm thuộc mỗi cụm một nửa. Log-likelihood có tổng nằm trong log, nên không có nghiệm đóng.

**EM.** Lặp hai bước:

- **E-step:** với tham số hiện tại, tính γᵢₖ.
- **M-step:** coi γ là trọng số, cập nhật bằng MLE có trọng số: Nₖ = ∑ᵢ γᵢₖ, πₖ = Nₖ/N, μₖ = ∑ᵢ γᵢₖxᵢ/Nₖ, Σₖ = ∑ᵢ γᵢₖ(xᵢ − μₖ)(xᵢ − μₖ)ᵀ/Nₖ.

**Vì sao EM đúng: Jensen và ELBO.** Với bất kỳ phân phối q(z) nào, vì log lõm, bất đẳng thức Jensen cho

$$
\log p(x;\theta) = \log\sum_{z} q(z)\frac{p(x,z;\theta)}{q(z)} \;\ge\; \sum_{z} q(z)\log\frac{p(x,z;\theta)}{q(z)} = \text{ELBO}(q,\theta)
$$

Khoảng cách giữa hai vế đúng bằng KL(q ‖ p(z | x; θ)) ≥ 0 (chương 3). E-step chọn q = p(z | x; θ) để KL = 0: cận chặt, ELBO bằng log-likelihood. M-step tăng ELBO theo θ với q cố định. Do đó log-likelihood mới ≥ ELBO mới ≥ ELBO cũ = log-likelihood cũ: **log-likelihood không bao giờ giảm**. Nhưng, như k-means, EM chỉ tới điểm dừng địa phương. VAE dùng cùng cận dưới này, chỉ khác là q do mạng nơ-ron (encoder) tham số hóa vì không tính được p(z | x) chính xác.

**Ví dụ giải mẫu 5.15 — một vòng EM.** Dữ liệu (0, 1, 2.5, 4, 5), K = 2, khởi tạo μ = (0, 5), σ² = (1, 1), π = (0.5, 0.5).

- E-step: γ của thành phần 1 = (1.000, 0.9994, 0.5, 0.0006, 0.000). Điểm 2.5 cách đều hai tâm nên chia đúng 0.5.
- M-step: N₁ = N₂ = 2.5 ⇒ π = (0.5, 0.5); μ₁ ≈ 0.901, μ₂ ≈ 4.099 (đối xứng); σ² ≈ (0.842, 0.842).
- Log-likelihood: −11.491 trước vòng 1, −9.425 sau vòng 1, −9.424 sau vòng 2. Không giảm, đúng như chứng minh.

% manim: KMeansVaEM

**k-means là giới hạn của GMM:** cố định Σₖ = σ²I, πₖ bằng nhau, cho σ → 0 thì E-step thành gán cứng, M-step thành lấy trung bình (bài 14).

**Suy biến.** Mật độ Gauss tại chính tâm là 1/√(2πσ²): σ² = 1 → 0.399; 0.01 → 3.99; 10⁻⁴ → 39.9. Một thành phần co về một điểm dữ liệu có thể đẩy likelihood → ∞: MLE của GMM không bị chặn trên. Thực tế cộng ε nhỏ vào đường chéo hiệp phương sai (`reg_covar` của scikit-learn); chọn số thành phần bằng BIC hoặc likelihood trên validation.

**Vì sao ML cần cái này.**

- ELBO và chiến lược "chọn q, rồi tối ưu θ" là cốt lõi của VAE, diffusion (dạng biến phân) và nhiều mô hình biến ẩn.
- k-means dùng trực tiếp trong deep learning: lượng tử hóa vector (VQ-VAE: codebook cập nhật giống k-means trực tuyến, bằng gradient hoặc EMA), product quantization, và rời rạc hóa âm thanh thành "token" (HuBERT dùng k-means trên đặc trưng; nhận định ở mức mô tả).
- Phân cụm embedding câu là bước khám phá dữ liệu phổ biến trong NLP (kiểm trùng lặp, tìm chủ đề, lọc dữ liệu).

**Thực tế tính toán.** Tính E-step trong miền log với log-sum-exp (nếu không, mật độ nhiều chiều làm tròn về 0 và γ ra 0/0); với d lớn dùng hiệp phương sai chéo (lab 5).

## Bài tập giấy bút

14 bài, làm theo thứ tự: hết ★ trong 1 buổi, hết ★★ trong 2 buổi, ★★★ dành cho Thứ Bảy. Bài sai thì ghi vào sổ lỗi kèm lý do.

**★ Hiểu định nghĩa**

1. Chứng minh σ(−z) = 1 − σ(z) và σ′(z) = σ(z)(1 − σ(z)). Từ đó viết log-loss của một mẫu theo z = wᵀx khi y = 1 và y = 0, và nói nó tiến tới đâu khi z → ±∞.
2. Một bộ phân loại trên 1000 mẫu cho TP = 40, FP = 10, FN = 20, TN = 930. Tính precision, recall, F1, accuracy và FPR. Mô hình "luôn đoán âm" có accuracy bao nhiêu?
3. SVM tuyến tính với w = (2, −1), b = 1. Điểm x = (2, 1) có nhãn y = +1. Tính functional margin, geometric margin, hinge loss của điểm và bề rộng lề 2/‖w‖₂.
4. Tính Gini và entropy (bit) của một nút có 6 mẫu dương và 2 mẫu âm. Tính Gini của một nút 3 lớp với số mẫu (2, 1, 1).

**★★ Mức đề thi đầu vào**

5. Chứng minh gradient của log-loss trung bình theo w là (1/N)∑(pᵢ − yᵢ)xᵢ, và Hessian là (1/N)∑ pᵢ(1 − pᵢ)xᵢxᵢᵀ. Suy ra log-loss lồi. Hessian có xác định dương không, và trị riêng của nó đi về đâu khi dữ liệu tách được tuyến tính và ‖w‖ → ∞?
6. Với y = Xw + ε, ε ~ 𝒩(0, σ²I) và prior w ~ 𝒩(0, τ²I), chứng minh MAP là nghiệm ridge và tìm λ theo σ², τ². Prior nào cho lasso?
7. Chứng minh nghiệm của min ½(z − w)² + λ|w| (λ > 0) là sign(z)·max(|z| − λ, 0). *Gợi ý: xét w > 0, w < 0, w = 0; tại 0 dùng dưới vi phân.*
8. Chứng minh 𝔼(y − f̂(x))² = σ² + bias² + variance tại một điểm x cố định; nói rõ giả thiết dùng để triệt tiêu các số hạng chéo.
9. Điểm số (0.8, 0.6, 0.6, 0.4, 0.3), nhãn (1, 1, 0, 0, 1). Tính AUC bằng đếm cặp (hòa tính 0.5) và giải thích vì sao quy ước đó khớp diện tích hình thang trên ROC.
10. Với x = (1, 2), z = (3, −1), tính K(x, z) = (1 + xᵀz)², tìm φ: ℝ² → ℝ⁶ để K(x, z) = φ(x)ᵀφ(z) và kiểm bằng số. Chứng minh mọi ma trận Gram của kernel dạng φ(x)ᵀφ(z) đều PSD.

**★★★ Mức phỏng vấn nghiên cứu**

11. Chứng minh EM không làm giảm log-likelihood: Jensen cho ELBO, khoảng cách tới log-likelihood là KL(q ‖ p(z | x; θ)), rồi nối E-step và M-step. Nếu M-step chỉ tăng ELBO chút ít (không cực đại), kết luận còn đúng không?
12. Hard-margin SVM. Viết Lagrangian với nhân tử αᵢ ≥ 0, đặt đạo hàm theo w và b bằng 0, thế lại để ra bài toán đối ngẫu. Chỉ ra vì sao dữ liệu chỉ xuất hiện qua xᵢᵀxⱼ, và kiểm lại bằng hai điểm (1, 1)/−1, (3, 3)/+1 của ví dụ 5.8.
13. Logistic regression cho lớp dương hiếm, giữ mọi mẫu dương và chỉ giữ ngẫu nhiên tỉ lệ β mẫu âm. Chứng minh odds dự đoán bị nhân với 1/β và viết công thức hiệu chỉnh về phân phối gốc. Với β = 0.1 và p′ = 0.5, xác suất thật là bao nhiêu? Vì sao điều này quan trọng khi chọn ngưỡng theo chi phí?
14. GMM 1 chiều, hai thành phần cùng phương sai σ², π bằng nhau, μ = (0, 5). Tính γ của thành phần 1 tại x = 2 khi σ² = 1, 0.25 và σ² → 0. Chứng minh khi σ → 0, EM thành k-means. Điều gì xảy ra nếu σ² của từng thành phần được học tự do?

## Bài tính nhanh

15 bài, mỗi bài tối đa 2 phút, giấy bút, bấm giờ cả bộ. Mục tiêu: xong trong 25 phút, sai không quá 1 bài; làm lại sau 3 ngày và sau 1 tuần.

1. σ(ln 3) bằng bao nhiêu?
2. Logit z = (0, ln 2, ln 5). Tính softmax và cross-entropy khi nhãn đúng là lớp thứ ba.
3. Ridge một chiều không hệ số chặn: x = (1, 1, 2), y = (1, 2, 2), λ = 1. Tính w = xᵀy/(xᵀx + λ).
4. Soft-threshold z = (−0.5, 1.5, 4) với λ = 1.
5. TP = 30, FP = 20, FN = 10. Tính precision, recall, F1.
6. Dữ liệu có 2% mẫu dương. Accuracy của mô hình luôn đoán âm?
7. Bỏ sót tốn 5, báo nhầm tốn 1, xác suất đã hiệu chỉnh. Ngưỡng báo dương là bao nhiêu?
8. y = −1, f(x) = 0.3. Margin và hinge loss?
9. w = (3, 4). Bề rộng lề 2/‖w‖₂?
10. RBF với γ = 0.5 và ‖x − z‖₂² = 4. K(x, z) ≈ ?
11. Gini của nút có số mẫu ba lớp (2, 1, 1)?
12. N = 5. Xác suất một mẫu không được rút vào một bootstrap?
13. ρ = 0.5, B = 10. Phương sai của trung bình theo σ²?
14. AdaBoost: lỗi có trọng số ε = 0.2. Tính α.
15. Dữ liệu (0, 2, 10, 12), tâm (1, 11). Tính J của k-means.

## Lab code và bài săn bug

Năm lab đã chạy với Python 3.13, NumPy 2.5, SciPy 1.18; chỉ dùng NumPy, **không cần scikit-learn**, seed cố định; notebook lời giải chạy khoảng 6 giây trên CPU, số liệu có thể lệch nhẹ theo phiên bản. Tự viết từng lab trước, chỉ đối chiếu code mẫu khi kết quả lệch với cột "kết quả mong đợi".

| Lab | Việc phải làm | Kết quả mong đợi |
| --- | --- | --- |
| 1. Softmax regression từ đầu | `softmax` ổn định số, `ce_loss_grad` vectorized (có L2), GD trên 3 cụm Gauss 600 điểm; logistic 1 chiều của ví dụ 5.2 | softmax (1000, 1001) = (0.2689, 0.7311), không nan; gradient lệch sai phân hữu hạn 8.6 × 10⁻¹⁰; loss 1.0986 → 0.1599, accuracy train 0.943; không L2: w = 8.925 → 10.312; L2 λ = 0.1: w = 1.508, ranh giới 2.500 |
| 2. Ridge, lasso, k-fold CV | Ridge qua `solve`, lasso bằng coordinate descent, `kfold_indices`, chọn λ bằng 5-fold CV trên dữ liệu 100 × 50 có 5 hệ số thật khác 0; ô tiếp theo tái hiện bảng bias²–variance của ví dụ 5.4 | Ridge (1.5, 0.4, −1), lasso (2, 0, −1); số hệ số khác 0 theo λ = 1, 3, 10, 30, 100: 46, 37, 18, 6, 5; CV chọn λ = 10; MSE test: OLS 1.791, lasso 1.176, lasso giữ đủ 5 hệ số thật; bias² / variance 0.1809 / 0.0078, 0.0054 / 0.0136, 0.0000 / 0.0380 |
| 3. Metric tự viết | `precision_recall_f1`, AUC bằng đếm cặp và hình thang (xử lý điểm hòa), `average_precision` | Ví dụ 5.7: AUC 0.7500 cả hai cách, AP 0.8304; ví dụ 5.6: precision 0.1538, F1 0.2628; dương 50% → 1%: ROC-AUC 0.920 → 0.916, AP 0.921 → 0.295 |
| 4. Gradient boosting với stump | `fit_stump` tìm ngưỡng tối ưu MSE bằng tổng tích lũy, vòng boosting có MSE train/test | Ví dụ 5.12: MSE 0.2447, 0.0695, 0.0257; sin nhiễu 80 điểm: η = 0.5 có MSE test nhỏ nhất 0.1052 ở vòng 34, vòng 400 test 0.1313; η = 0.05 đạt 0.1041 ở vòng 372 |
| 5. k-means, k-means++, EM | k-means theo dõi J, k-means++, EM cho GMM chéo trong miền log | Ví dụ 5.13: J = 196 và 77.5, tốt nhất trong 10 lần k-means++ = 77.5; ví dụ 5.15: μ = (0.9007, 4.0993), σ² = 0.8421; 3 cụm 2 chiều: J không tăng, log-likelihood −1560.61 → −1290.36 không giảm, tìm lại ba tâm (0, 0), (4, 1), (1, 4) |

Code nằm trong `labs/chuong05/lab.ipynb` (có chỗ TODO và ô kiểm tra tự động) và `labs/chuong05/loigiai.ipynb` (lời giải đã chạy kiểm). Mở từ trang **Lab** của app.

**Câu hỏi sau lab.**

- Lab 2: lasso với λ = 10 giữ 18 hệ số khác 0 trong khi chỉ có 5 hệ số thật. Vì sao CV (tối ưu sai số dự đoán) thường chọn λ "quá nhỏ" so với mục tiêu chọn đúng đặc trưng?
- Lab 3 (d): vì sao ROC-AUC gần như không đổi còn AP sụp từ 0.921 xuống 0.295 khi lớp dương từ 50% xuống 1%?
- Lab 4: vì sao η = 0.05 cần khoảng 11 lần số vòng nhưng đạt MSE test hơi tốt hơn và gần như không tăng lại sau đáy?

**Bài săn bug.** Hàm dưới đây chạy không báo lỗi. Trên n = 100 mẫu, d = 5000 đặc trưng ngẫu nhiên và nhãn **tung đồng xu** (đáp án đúng ≈ 0.5), nó báo accuracy CV **1.00** với k = 50 đặc trưng. Tìm cả **ba** lỗi rò rỉ trước khi xem lời giải.

```python
def cv_buggy(X, y, ks=(5, 10, 20, 50), n_folds=5, seed=0):
    X = (X - X.mean(0)) / X.std(0)
    folds = np.array_split(np.random.default_rng(seed).permutation(len(y)), n_folds)
    scores = {}
    for k in ks:
        sel = top_k_corr(X, y, k)
        accs = []
        for i in range(n_folds):
            te = folds[i]; tr = np.setdiff1d(np.arange(len(y)), te)
            accs.append(np.mean(nearest_centroid(X[tr][:, sel], y[tr], X[te][:, sel]) == y[te]))
        scores[k] = np.mean(accs)
    best_k = max(scores, key=scores.get)
    return best_k, scores[best_k]
```

`top_k_corr(X, y, k)` trả về chỉ số k cột có |tương quan| với y lớn nhất; `nearest_centroid` gán mỗi mẫu test cho lớp có trung bình train gần nhất. Hai hàm này đúng.

## Bài dựng cảnh Manim

Muốn vẽ được một khái niệm, bạn phải hiểu từng bước của nó. Giới hạn **1 giờ** (tính vào luồng C), không chỉnh màu mè.

**Đề bài.** Dùng `manim/chuong05_scenes.py` làm mẫu, viết cảnh `BaggingPhuongSai` vào thư mục `manim/cua-toi/`:

1. Vẽ trục hoành là số cây B từ 1 đến 100, trục tung là Var/σ² từ 0 đến 1. Nhãn trục dùng `Text` (`include_numbers` cần LaTeX).
2. Lần lượt vẽ đường ρ + (1 − ρ)/B cho ρ = 0, 0.3, 0.7, mỗi đường kèm đường nét đứt ở sàn ρ.
3. Với mỗi ρ hiện giá trị tại B = 10 và B = 100, rồi phụ đề kết luận vì sao random forest phải giảm ρ.

**Tiêu chí tự chấm.** (1) Đúng toán: các số hiện ra là 0.100/0.010, 0.370/0.307, 0.730/0.703 (ví dụ 5.11). (2) Có tiêu đề và phụ đề nói người xem phải thấy điều gì; chữ không chồng lên đường cong hay lên nhau. (3) Dài không quá 30 giây.

Render bằng nút **Render cảnh của tôi** ở trang Manim. Lời giải tham khảo: `manim/loigiai/chuong05_bagging_phuong_sai.py` (chỉ mở sau khi đã tự làm).

## Phỏng vấn, bẫy, thực tế khác lý thuyết

**Cách luyện.** Mỗi chuỗi có 4 tầng câu hỏi, đào sâu dần như người phỏng vấn thật. Trả lời theo khung **giả thuyết → loss → tối ưu → thất bại ở đâu**: nói tiếng Việt trước, rồi tiếng Anh, 2 phút mỗi tầng, ghi âm và nghe lại. Model answer chỉ phủ tầng 1–2; tầng 3–4 bạn tự soạn rồi đối chiếu lời giải các bài liên quan.

**Chuỗi 1 — Logistic regression**

1. Vì sao logistic regression dùng log-loss mà không dùng MSE?
2. Gradient của nó là gì, và vì sao có dạng đẹp như vậy?
3. Dữ liệu tách được tuyến tính thì điều gì xảy ra với trọng số? Bạn xử lý thế nào? (Ví dụ 5.2.)
4. Liên hệ với softmax + cross-entropy ở lớp cuối của một LM. Label smoothing (xem chương 3, ví dụ 3.10) sửa vấn đề gì của tầng 3? GD thuần chọn nghiệm nào trong vô số nghiệm khi dữ liệu tách được? (Xem implicit bias, mục 5.1.)

> *Model answer:* Logistic regression models y as Bernoulli with p = σ(wᵀx), so maximizing the likelihood is minimizing the log-loss. That loss is convex in w, while MSE composed with a sigmoid is not, and the MSE gradient carries a factor σ′(z) that vanishes exactly when the model is confidently wrong. With the log-loss that factor cancels and the gradient is (p − y)x averaged over the data: error times input, the same form as linear regression, as for any GLM with the canonical link.

**Chuỗi 2 — Bias–variance và regularization**

1. Định nghĩa bias và variance, và viết phân rã sai số.
2. L1 và L2 khác nhau thế nào? Vì sao L1 cho nghiệm thưa?
3. L2 tương đương với prior nào? Weight decay có luôn bằng L2 không? (Xem chương 4, mục 4.4.)
4. Double descent có mâu thuẫn với chữ U cổ điển không?

> *Model answer:* For squared loss at a fixed input, expected test error splits into irreducible noise, squared bias and variance (how much the fit changes when we redraw the training set). Regularization trades a little bias for a large drop in variance. In an orthonormal design, L2 shrinks every weight proportionally, while L1 subtracts a constant and clips small weights to exactly zero: the L1 ball has corners on the axes, so the loss contours usually touch it at a sparse point.

**Chuỗi 3 — Đánh giá trên dữ liệu lệch**

1. Vì sao accuracy là metric tệ khi lớp dương chiếm 1%?
2. ROC-AUC và PR-AUC khác nhau thế nào, cái nào thay đổi theo tỉ lệ lớp?
3. Chọn ngưỡng quyết định thế nào? Resampling ảnh hưởng gì tới xác suất? (Bài 13.)
4. Thiết kế đánh giá cho bộ phân loại văn bản khi một tài liệu bị cắt thành nhiều đoạn, và khi đánh giá một LM đã pretrain trên web. Kể các dạng leakage.

> *Model answer:* With 1% positives, always predicting negative gives 99% accuracy and catches nothing. TPR and FPR are each computed within one class, so the ROC curve does not change with the class ratio. Precision mixes the classes: at TPR 0.9 and FPR 0.05 it drops from about 0.95 at a balanced ratio to about 0.15 at 1% positives. For rare events I report average precision and choose the threshold from the cost of each error on calibrated probabilities.

**Chuỗi 4 — Ensemble và GBDT**

1. Bagging và boosting khác nhau thế nào?
2. Vì sao random forest chỉ xét một tập con đặc trưng mỗi lần chia? (Công thức ρσ².)
3. Giải thích gradient boosting như gradient descent trong không gian hàm; learning rate đóng vai trò gì?
4. GBDT có còn thắng mạng nơ-ron trên dữ liệu bảng không? Nêu bằng chứng ủng hộ, bằng chứng thách thức, và mức chắc chắn của bạn.

> *Model answer:* Bagging averages many deep trees trained independently on bootstrap samples, mainly reducing variance; boosting adds shallow trees sequentially, each correcting the current ensemble, mainly reducing bias. The variance of an average of B trees with correlation ρ is ρσ² + (1 − ρ)σ²/B, so more trees never push it below ρσ². Random forests lower that floor by letting each split see a random subset of features. More trees in a forest do not overfit, but more boosting rounds can, so boosting needs early stopping.

**Chuỗi dự phòng — k-means, GMM, EM.** k-means có hội tụ không, tới đâu? k-means khác GMM ở đâu? Vì sao EM không làm giảm likelihood? Liên hệ với ELBO trong VAE. (Ví dụ 5.13–5.15, bài 11 và 14.)

**Từ vựng nên nói trôi chảy:** maximum likelihood, cross-entropy, canonical link, linearly separable, bias–variance trade-off, shrinkage, sparsity, soft-thresholding, nested cross-validation, data leakage, class imbalance, calibration, functional / geometric margin, hinge loss, dual problem, support vectors, kernel trick, Gram matrix, impurity, out-of-bag, decorrelation, responsibilities, evidence lower bound (ELBO).

**Bẫy thường gặp.**

- Mặc định của thư viện: `LogisticRegression` **có regularization L2** (C = 1), nên so hệ số với statsmodels sẽ lệch; `SVC.predict_proba` là Platt scaling và có thể mâu thuẫn với `predict`.
- Quên chuẩn hóa trước L1/L2, SVM, k-means, k-NN: phạt và khoảng cách phụ thuộc đơn vị đo. Ngược lại, chuẩn hóa bằng thống kê của toàn bộ dữ liệu trước khi chia là leakage.
- Tin lasso "chọn đúng đặc trưng": với đặc trưng tương quan nó giữ một và bỏ phần còn lại gần như tùy ý; lab 2: CV còn giữ thêm 13 đặc trưng thừa.
- Báo AUC cao cho lớp hiếm (một điểm ROC rất đẹp, TPR 0.9 và FPR 0.05, vẫn chỉ có precision 0.154 ở 1% dương: ví dụ 5.6; video NguongVaPrecision với hai Gauss, AUC ≈ 0.98, cũng chỉ có precision 0.154 ở một ngưỡng), hoặc báo điểm CV đã dùng để chọn siêu tham số như sai số tổng quát hóa (lạc quan; dùng nested CV hoặc test chưa đụng tới).
- Cây: thêm cây vào random forest không overfit, thêm vòng boosting thì có (lab 4: 0.1052 → 0.1313); importance theo tạp chất thiên vị cột nhiều giá trị, nên dùng permutation importance trên held-out.
- k-means không "tìm cụm đúng": nó chỉ tìm cụm cầu, cùng cỡ, kẹt ở cực tiểu địa phương, và không chọn K bằng J trên train (J luôn giảm khi K tăng).

**Thực tế khác lý thuyết.**

- **Nghiệm đóng hiếm khi dùng nguyên dạng.** Không ai nghịch đảo XᵀX; dùng QR/SVD/Cholesky (chương 1), hoặc SGD cho dữ liệu lớn. Ridge với λ > 0 còn cải thiện số điều kiện.
- **Resampling và class weight làm hỏng hiệu chỉnh xác suất.** Bài 13: β = 0.1 biến xác suất thật 0.091 thành 0.5. Thường tốt hơn là giữ xác suất trên phân phối thật và dời ngưỡng theo chi phí; nếu đã resample thì hiệu chỉnh ngược.
- **SVM gần như biến mất khỏi NLP hiện đại** vì không scale theo N và không học biểu diễn; margin sống tiếp trong contrastive loss, còn TF-IDF + logistic regression vẫn là baseline văn bản rẻ và mạnh, nên chạy trước.

## Mini-test 30 phút

Làm sau bài tập và lab, bấm giờ đúng 30 phút, không xem sách, không máy tính (câu 5 chỉ đọc code, không chạy). Tổng 30 điểm. **Từ 24 điểm:** sang chương 6. **18 đến dưới 24:** ôn lại mục ứng với câu sai rồi làm lại sau 3 ngày. **Dưới 18:** học lại chương.

1. **(4 điểm)** Khớp ŷ = θ₀ + θ₁x qua ba điểm (x, y) = (0, 1), (1, 1), (2, 4) bằng phương trình chuẩn. Kiểm tra phần dư vuông góc với hai cột của X.
2. **(4 điểm)** Softmax hai lớp với logit z = (0, ln 3), nhãn đúng là lớp 0. Tính xác suất, cross-entropy và gradient theo z. Nếu cộng 10 vào cả hai logit thì ba đại lượng đó thay đổi thế nào?
3. **(5 điểm)** 1000 mẫu, 50 dương. Mô hình gắn cờ 100 mẫu, trong đó 40 là dương thật. Tính precision, recall, FPR, F1, accuracy. Giữ nguyên TPR và FPR, tính precision nếu tỉ lệ dương là 10% và 50%.
4. **(5 điểm)** Gradient boosting với MSE và stump trên x = (1, 2, 3, 4), y = (0, 0, 2, 4). Tính F₀, phần dư, phép chia tốt nhất (thử ngưỡng 1.5, 2.5, 3.5), và MSE sau vòng 1 với η = 1 và η = 0.5.
5. **(6 điểm)** Đọc code và trả lời không chạy máy:

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((500, 20)); y = rng.integers(0, 2, 500)
X = (X - X.mean(0)) / X.std(0)
Xa = np.concatenate([X, X + 0.01 * rng.standard_normal(X.shape)])   # "augmentation"
ya = np.concatenate([y, y])
idx = rng.permutation(1000); tr, te = idx[:800], idx[800:]
d = ((Xa[te][:, None] - Xa[tr][None]) ** 2).sum(-1)
print(np.mean(ya[tr][d.argmin(1)] == ya[te]))                         # 1-NN
```

(a) Accuracy đúng phải xấp xỉ bao nhiêu, vì sao? (b) Code in ra khoảng 0.91. Giải thích cơ chế, và ước lượng tỉ lệ mẫu test có "bản sao" trong train. (c) Sửa thế nào? Còn lỗi rò rỉ nào khác trong đoạn code?

6. **(6 điểm)** Tối đa 6 câu: một công ty có bảng 50 000 dòng, 30 cột (một nửa là biến phân loại, có giá trị thiếu), cần dự đoán khách rời bỏ (8% dương). Bạn chọn mô hình đầu tiên nào, đánh giá bằng metric gì, thiết kế CV thế nào, và khi nào mới cân nhắc mạng nơ-ron?

## Flashcard

18 thẻ của chương nằm trong `flashcards/chuong05.csv`. Ôn ở trang **Ôn tập** của app (lịch Leitner: 1 → 3 → 7 → 14 ngày, quên thì về hộp 1); file CSV cũng nhập thẳng được vào Anki.

## Lời giải

Chỉ đọc sau khi đã tự thử ít nhất 20 phút. Với bài chứng minh, so ý chính chứ không so từng chữ. Mọi đáp số bằng số trong chương đã được kiểm lại bằng NumPy.

**Kiểm tra nhanh**

1. ∇w = (p − y)x = 0.8x: mô hình tự tin sai (gán 0.8 cho lớp 1 khi nhãn là 0), nên bước GD giảm wᵀx.
2. Accuracy 0.99, recall 0. Đây là lý do không dùng accuracy khi lớp lệch.
3. m = 0.4, hinge = 0.6. Phân loại đúng (m > 0) nhưng nằm trong lề nên vẫn bị phạt.
4. Random forest: không, sai số hội tụ khi số cây tăng (Breiman 2001). Gradient boosting: có thể, vì mỗi vòng tiếp tục khớp phần dư kể cả nhiễu, nên cần early stopping.
5. Không tăng: cả hai bước đều tối ưu J theo một nhóm biến khi cố định nhóm kia. Không luôn tới cực tiểu toàn cục: ví dụ 5.13 dừng ở J = 196 trong khi tồn tại J = 77.5.

**Bài tập giấy bút**

1. σ(−z) = 1/(1 + eᶻ) = e⁻ᶻ/(e⁻ᶻ + 1) = 1 − σ(z); σ′(z) = e⁻ᶻ/(1 + e⁻ᶻ)² = σ(z)(1 − σ(z)). y = 1: loss = log(1 + e⁻ᶻ) → 0 khi z → +∞, ≈ −z khi z → −∞. y = 0: loss = log(1 + eᶻ) → 0 khi z → −∞, ≈ z khi z → +∞. Sai tự tin bị phạt tuyến tính, không bão hòa.
2. Precision 0.8; recall 40/60 ≈ 0.667; F1 ≈ 0.727; accuracy 0.97; FPR = 10/940 ≈ 0.0106. Luôn đoán âm: accuracy 0.94.
3. Functional margin = 4 − 1 + 1 = 4; ‖w‖₂ = √5 ⇒ geometric margin 4/√5 ≈ 1.789; hinge 0; lề 2/√5 ≈ 0.894.
4. Gini = 1 − 0.5625 − 0.0625 = 0.375; entropy ≈ 0.311 + 0.5 = 0.811 bit. Ba lớp (0.5, 0.25, 0.25): Gini 0.625.
5. Một mẫu: ∂ℓ/∂p = (p − y)/(p(1 − p)), nhân ∂p/∂z = p(1 − p) được ∂ℓ/∂z = p − y, nhân ∂z/∂w = x được (p − y)x. Đạo hàm tiếp cho H = (1/N)∑ pᵢ(1 − pᵢ)xᵢxᵢᵀ và vᵀHv = (1/N)∑ pᵢ(1 − pᵢ)(vᵀxᵢ)² ≥ 0: L lồi. Với mọi w hữu hạn, pᵢ ∈ (0, 1) nên H vẫn xác định dương nếu X đủ hạng cột; nhưng khi ‖w‖ → ∞ (dữ liệu tách được) mọi pᵢ(1 − pᵢ) → 0 nên H → 0, tức mọi trị riêng → 0, không riêng λₘᵢₙ: hàm lồi mà không lồi mạnh đều, gần như phẳng theo hướng w, nên GD đi mãi và Newton gần suy biến ("trượt ra vô cực" của ví dụ 5.2). Thêm (λ/2)‖w‖² cộng λI vào H, đảm bảo λₘᵢₙ ≥ λ. (X thiếu hạng cột thì H suy biến dù chưa tách được.)
6. log p(w | y) = −‖y − Xw‖₂²/(2σ²) − ‖w‖₂²/(2τ²) + hằng; cực đại ⇔ cực tiểu ½‖y − Xw‖₂² + (σ²/(2τ²))‖w‖₂²: ridge với λ = σ²/τ². Lasso ứng với prior Laplace p(wⱼ) ∝ exp(−|wⱼ|/b), λ = σ²/b.
7. w > 0: f′ = w − z + λ = 0 ⇒ w = z − λ, hợp lệ khi z > λ. w < 0: w = z + λ, hợp lệ khi z < −λ. Tại 0, dưới vi phân của |w| là \[−1, 1\], nên 0 là cực tiểu ⇔ 0 ∈ −z + λ\[−1, 1\] ⇔ |z| ≤ λ. Gộp lại w = sign(z)·max(|z| − λ, 0); f lồi chặt nên nghiệm duy nhất.
8. Đặt f̄ = 𝔼f̂: y − f̂ = ε + (f − f̄) + (f̄ − f̂). Bình phương, lấy kỳ vọng; ba số hạng chéo triệt tiêu: 𝔼[ε(f − f̄)] = 0 vì 𝔼ε = 0; 𝔼[ε(f̄ − f̂)] = 0 vì nhiễu của điểm test **độc lập** với tập train (nên với f̂); 𝔼[(f − f̄)(f̄ − f̂)] = (f − f̄)·𝔼[f̄ − f̂] = 0 vì f − f̄ là hằng. Còn σ² + bias² + variance. Giả thiết: 𝔼ε = 0 và nhiễu test độc lập với dữ liệu train; dạng cộng chỉ đúng cho squared loss.
9. Dương {0.8, 0.6, 0.3}, âm {0.6, 0.4}: 6 cặp. 0.8 thắng cả hai (2); 0.6 hòa 0.6 (0.5), thắng 0.4 (1); 0.3 thua cả hai (0). AUC = 3.5/6 ≈ 0.583. Hạ ngưỡng qua 0.6 thì một dương và một âm vào cùng lúc, ROC đi **chéo**; diện tích hình thang dưới đoạn chéo đúng bằng nửa ô, khớp quy ước cặp hòa tính 0.5.
10. K = (1 + 1)² = 4. Khai triển (1 + x₁z₁ + x₂z₂)² cho φ(v) = (1, √2v₁, √2v₂, v₁², √2v₁v₂, v₂²); φ(x)ᵀφ(z) = 1 + 6 − 4 + 9 − 12 + 4 = 4. PSD: cᵀKc = ∑ᵢ∑ⱼ cᵢcⱼφ(xᵢ)ᵀφ(xⱼ) = ‖∑ᵢ cᵢφ(xᵢ)‖₂² ≥ 0.
11. Với q(z) > 0 bất kỳ, Jensen (log lõm): log p(x; θ) ≥ ∑ q(z) log(p(x, z; θ)/q(z)) = ELBO(q, θ) = log p(x; θ) − KL(q ‖ p(z | x; θ)) (viết p(x, z; θ) = p(z | x; θ)p(x; θ)). E-step đặt qₜ = p(z | x; θₜ) ⇒ KL = 0 ⇒ ELBO(qₜ, θₜ) = log p(x; θₜ). M-step chọn θₜ₊₁ với ELBO(qₜ, θₜ₊₁) ≥ ELBO(qₜ, θₜ). Vậy log p(x; θₜ₊₁) ≥ ELBO(qₜ, θₜ₊₁) ≥ ELBO(qₜ, θₜ) = log p(x; θₜ); N mẫu độc lập thì cộng theo mẫu. M-step chỉ cần **không làm giảm** ELBO (generalized EM).
12. Lagrangian ℒ(w, b, α) = ½‖w‖² − ∑αᵢ\[yᵢ(wᵀxᵢ + b) − 1\], αᵢ ≥ 0. ∂ℒ/∂w = w − ∑αᵢyᵢxᵢ = 0 ⇒ w = ∑αᵢyᵢxᵢ; ∂ℒ/∂b = −∑αᵢyᵢ = 0. Thế lại: ½‖w‖² = ½∑∑αᵢαⱼyᵢyⱼxᵢᵀxⱼ; ∑αᵢyᵢwᵀxᵢ = ‖w‖², số hạng b triệt tiêu vì ∑αᵢyᵢ = 0. Vậy ℒ = ∑αᵢ − ½∑∑αᵢαⱼyᵢyⱼxᵢᵀxⱼ, cực đại theo α ≥ 0 với ∑αᵢyᵢ = 0. Bài toán gốc lồi với ràng buộc tuyến tính và khả thi (dữ liệu tách được) nên đối ngẫu mạnh (soft-margin chỉ thêm cận αᵢ ≤ C); điều kiện bù αᵢ\[yᵢ(wᵀxᵢ + b) − 1\] = 0 cho αᵢ > 0 chỉ với support vector. x chỉ vào qua xᵢᵀxⱼ (và f(x) = ∑αᵢyᵢxᵢᵀx + b), nên thay được bằng K(xᵢ, xⱼ). Kiểm hai điểm: α₁ = α₂ = α, w = α(3, 3) − α(1, 1) = α(2, 2), giá trị đối ngẫu 2α − 4α², cực đại tại α = 0.25 bằng 0.25 = ½‖w‖² với w = (0.5, 0.5). Khôi phục b từ support vector: b = yᵢ − wᵀxᵢ = 1 − 3 = −2 (điểm (3, 3)), đúng với ví dụ 5.8.
13. Gọi s = 1 nếu mẫu được giữ: P(s = 1 | y = 1) = 1, P(s = 1 | y = 0) = β. Theo Bayes, odds trên dữ liệu đã lấy mẫu = P(y = 1 | x)/(β P(y = 0 | x)): odds thật nhân 1/β (với logistic, chỉ hệ số chặn dời thêm ln(1/β)). Hiệu chỉnh: p = βp′/(βp′ + 1 − p′); β = 0.1, p′ = 0.5 cho p = 0.05/0.55 ≈ 0.091. Ngưỡng chi phí (ví dụ 1/11 ≈ 0.091) chỉ có nghĩa trên xác suất đã hiệu chỉnh; áp lên p′ sẽ báo dương quá nhiều.
14. γ₁(x) = 1/(1 + exp(−((x − 5)² − x²)/(2σ²))) = 1/(1 + exp(−(25 − 10x)/(2σ²))). Tại x = 2: γ₁ = 1/(1 + exp(−2.5/σ²)). σ² = 1: ≈ 0.924; σ² = 0.25: exp(−10) ⇒ ≈ 0.99995; σ² → 0: → 1. Tổng quát, Σ = σ²I và π bằng nhau cho γᵢₖ ∝ exp(−‖xᵢ − μₖ‖₂²/(2σ²)); khi σ → 0, tỉ số giữa tâm gần nhất và các tâm khác → ∞ (nếu không hòa), γ thành one-hot: E-step = bước gán của k-means, M-step = lấy trung bình. Nếu σ² tự do: cụm có độ rộng khác nhau, nhưng một thành phần co về một điểm làm likelihood → ∞; cần regularize phương sai.

**Bài tính nhanh**

1. σ(ln 3) = 1/(1 + 1/3) = 0.75.
2. eᶻ = (1, 2, 5), tổng 8 ⇒ p = (0.125, 0.25, 0.625). CE = −ln 0.625 ≈ 0.470.
3. xᵀy = 1 + 2 + 4 = 7, xᵀx = 6, w = 7/(6 + 1) = 1.
4. (0, 0.5, 3).
5. P = 30/50 = 0.6, R = 30/40 = 0.75, F1 = 0.9/1.35 ≈ 0.667.
6. 0.98.
7. 5p > 1 − p ⇔ p > 1/6 ≈ 0.167.
8. m = −0.3, hinge = 1.3 (phân loại sai).
9. ‖w‖₂ = 5, lề 2/5 = 0.4.
10. e⁻² ≈ 0.135.
11. 1 − (0.25 + 0.0625 + 0.0625) = 0.625.
12. (1 − 1/5)⁵ = 0.8⁵ ≈ 0.328.
13. 0.5 + 0.5/10 = 0.55σ².
14. α = ½ ln(0.8/0.2) = ½ ln 4 = ln 2 ≈ 0.693.
15. 1 + 1 + 1 + 1 = 4.

**Câu hỏi sau lab**

- Lab 2: CV cực tiểu sai số dự đoán. Bỏ sót một đặc trưng thật tốn hơn giữ thêm vài đặc trưng thừa nhỏ, và λ đủ lớn để loại hết đặc trưng thừa cũng co mạnh các hệ số thật (lasso trừ đều λ, nên có bias). Muốn chọn đặc trưng thì dùng λ lớn hơn (quy tắc "một độ lệch chuẩn") hoặc fit lại không phạt trên tập đã chọn.
- Lab 3 (d): bộ phân loại giữ nguyên phân phối điểm trong từng lớp, nên mọi cặp (TPR, FPR) không đổi và ROC-AUC chỉ dao động do lấy mẫu (0.920 so với 0.916). Khi lớp âm đông gấp 99 lần, cùng một FPR sinh ra nhiều FP hơn hẳn TP, nên precision ở mọi mức recall giảm, kéo AP xuống 0.295.
- Lab 4: bước nhỏ hơn 10 lần nên cần cỡ 10 lần số vòng (thực đo: 372 so với 34 vòng, khoảng 11 lần). Mỗi cây chỉ đóng góp một phần, đáy rộng và phẳng nên vượt đáy ít gây hại (0.1043 ở vòng 400 so với đáy 0.1041); với η = 0.5, mô hình khớp nhiễu nhanh và MSE test tăng rõ sau vòng 34.

**Mini-test (thang 30 điểm)**

1. XᵀX có hàng (3, 3), (3, 5); Xᵀy = (6, 9) (1.5 điểm). det = 6, θ = (0.5, 1.5) (1.5 điểm). Phần dư (0.5, −1, 0.5): tổng 0 và 0·0.5 + 1·(−1) + 2·0.5 = 0 (1 điểm).
2. p = (1, 3)/4 = (0.25, 0.75) (1 điểm). CE = ln 4 ≈ 1.386 (1 điểm). Gradient p − Y = (−0.75, 0.75) (1 điểm). Cộng 10: cả ba không đổi vì softmax bất biến khi cộng hằng (1 điểm).
3. TP = 40, FP = 60, FN = 10, TN = 890. Precision 0.4, recall 0.8, FPR = 60/950 ≈ 0.063, F1 = 0.64/1.2 ≈ 0.533, accuracy 0.93 (3 điểm, 0.5 mỗi đại lượng, thêm 0.5 cho TN đúng). Precision = 0.8π/(0.8π + 0.063(1 − π)): π = 10% → ≈ 0.585; π = 50% → ≈ 0.927 (2 điểm).
4. F₀ = 1.5, phần dư (−1.5, −1.5, 0.5, 2.5), MSE 2.75 (1 điểm). SSE của stump: ngưỡng 1.5 → 8, 2.5 → 2, 3.5 → 2.667; chọn 2.5, giá trị −1.5 (trái) và 1.5 (phải) (2 điểm). η = 1: F = (0, 0, 3, 3), MSE = 0.5 (1 điểm). η = 0.5: F = (0.75, 0.75, 2.25, 2.25), MSE = 1.0625 (1 điểm).
5. (a) ≈ 0.5: nhãn ngẫu nhiên độc lập với X (1 điểm). (b) Mỗi mẫu gốc có hai bản gần trùng nhau; chia ngẫu nhiên sau khi nhân đôi nên phần lớn mẫu test có "anh em" trong train và 1-NN chép nhãn của nó. Xác suất có bản sao ≈ 800/999 ≈ 0.80 (chạy thật: 81%), khi đó gần như chắc đúng; còn lại đúng một nửa: 0.8 + 0.2·0.5 = 0.9, khớp 0.91 (3 điểm). (c) Chia theo **mẫu gốc** trước (group split), rồi chỉ augment phần train; chạy thật cho ≈ 0.5 (trung bình nhiều seed; một seed cho 0.46). Lỗi thứ hai: chuẩn hóa bằng thống kê của toàn bộ X trước khi chia; phải tính mean/std trên train (2 điểm).
6. (2 điểm) Mô hình đầu tiên là GBDT (LightGBM/XGBoost/CatBoost): xử lý biến phân loại và giá trị thiếu tự nhiên, không cần chuẩn hóa; kèm baseline logistic regression. (2 điểm) Metric: PR-AUC/AP và recall ở mức precision chấp nhận được (hoặc chi phí kỳ vọng ở ngưỡng theo chi phí), không dùng accuracy vì 8% dương; kiểm hiệu chỉnh xác suất. (1 điểm) CV stratified; chia theo thời gian là mặc định ở đây (nhãn rời bỏ định nghĩa trên cửa sổ tương lai), thêm group theo khách nếu một khách có nhiều dòng; mọi tiền xử lý trong fold. (1 điểm) Mạng nơ-ron chỉ khi có lý do cụ thể (dữ liệu rất lớn, văn bản/ảnh/chuỗi hành vi) và phải thắng GBDT trên cùng giao thức đánh giá.

**Săn bug — ba lỗi**

1. **Chọn đặc trưng trên toàn bộ dữ liệu** (`top_k_corr(X, y, k)` ngoài vòng fold). Lỗi này tạo ra gần như toàn bộ con số đẹp (đặc trưng tình cờ tương quan với nhãn của *cả* mẫu test). Chỉ sửa lỗi này, accuracy tụt về 0.45–0.53 tùy k.
2. **Chuẩn hóa bằng mean/std của toàn bộ X** trước khi chia fold. Leakage thật nhưng ảnh hưởng nhỏ ở đây: chỉ sửa lỗi này, k = 50 vẫn cho 0.99. Với điền giá trị thiếu hay PCA, nó có thể lớn.
3. **Chọn k theo chính điểm của các fold test rồi báo cáo điểm đó.** Sau khi sửa hai lỗi trên, lấy max qua 4 giá trị k vẫn lạc quan: trên 30 bộ ngẫu nhiên, max đạt 0.545 thay vì 0.5. Sửa bằng nested CV: chọn k bằng CV bên trong từng fold train.

```python
def _fit_predict(Xtr, ytr, Xte, k):
    mu, sd = Xtr.mean(0), Xtr.std(0)                  # sửa lỗi 2: thống kê của train
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    sel = top_k_corr(Xtr, ytr, k)                     # sửa lỗi 1: chọn đặc trưng trên train
    return nearest_centroid(Xtr[:, sel], ytr, Xte[:, sel])


def cv_fixed(X, y, ks=(5, 10, 20, 50), n_folds=5, seed=0):
    rngf = np.random.default_rng(seed)
    folds = np.array_split(rngf.permutation(len(y)), n_folds)
    accs = []
    for i in range(n_folds):
        te = folds[i]; tr = np.setdiff1d(np.arange(len(y)), te)
        inner = np.array_split(rngf.permutation(tr), 4)   # sửa lỗi 3: nested CV để chọn k
        def inner_score(k):
            return np.mean([np.mean(_fit_predict(X[np.setdiff1d(tr, f)], y[np.setdiff1d(tr, f)], X[f], k) == y[f])
                            for f in inner])
        k_best = max(ks, key=inner_score)
        accs.append(np.mean(_fit_predict(X[tr], y[tr], X[te], k_best) == y[te]))
    return np.mean(accs)
```

Đã chạy kiểm: bản sửa cho accuracy CV 0.530 (mức tung đồng xu); bản lỗi báo 1.00 ở k = 50.

## Tài liệu đào sâu và nguồn

Nội dung chương được viết lại, đối chiếu phạm vi kiến thức với các nguồn dưới đây, đọc theo thứ tự ưu tiên:

| Tài liệu | Dùng để làm gì | Phần nên đọc |
| --- | --- | --- |
| [Stanford CS229 Lecture Notes](https://cs229.stanford.edu/main_notes.pdf) (Ng, Ma) và [syllabus](https://cs229.stanford.edu/syllabus.html) | Khung lý thuyết chính, cùng thứ tự với chương | Hồi quy, logistic, GLM; kernel và SVM; bias–variance, double descent; regularization, CV; k-means; EM, Jensen, ELBO |
| scikit-learn: [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html), [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html), [SVM](https://scikit-learn.org/stable/modules/svm.html), [Ensembles](https://scikit-learn.org/stable/modules/ensemble.html), [Gaussian mixture](https://scikit-learn.org/stable/modules/mixture.html) | Leakage, kiểu chia CV, mặc định dễ gây bẫy | Data leakage; Platt scaling; OOB; reg_covar |
| [Grinsztajn, Oyallon, Varoquaux (2022)](https://arxiv.org/abs/2207.08815) và [Hollmann và cộng sự (2025), Nature](https://pubmed.ncbi.nlm.nih.gov/39780007/) | Hai phía của tranh luận GBDT và dữ liệu bảng | Benchmark và ba lý giải (2022); phần tóm tắt TabPFN (2025) |
| [Chen & Guestrin (2016), XGBoost](https://arxiv.org/abs/1603.02754), [Friedman (2001)](https://doi.org/10.1214/aos/1013203451), [Breiman (2001)](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) | Paper gốc gradient boosting, random forest | Steepest descent trong không gian hàm; sai số hội tụ khi thêm cây, OOB |
| [Soudry và cộng sự (2018), JMLR](https://jmlr.org/papers/v19/18-188.html) (implicit bias, ví dụ 5.2) [cần kiểm chứng link], [Cortes & Vapnik (1995)](https://link.springer.com/article/10.1007/BF00994018), [Tibshirani (1996)](https://academic.oup.com/jrsssb/article/58/1/267/7027929), [Dempster, Laird, Rubin (1977)](https://academic.oup.com/jrsssb/article/39/1/1/7027539), [Cawley & Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html) | Paper gốc soft-margin SVM, lasso, EM; selection bias của điểm CV | Phần giới thiệu và định nghĩa thuật toán |

**Từ khóa để tự tìm thêm:** exponential family GLM, elastic net, nested cross-validation, probability calibration (Platt, isotonic), Mercer kernel, random features, permutation importance, k-means++, double descent, tabular foundation model. Sách nâng cao: *The Elements of Statistical Learning* (Hastie, Tibshirani, Friedman).

**Trước khi sang chương 6:** đạt từ 24/30 điểm mini-test, làm lại các bài ★★★ sai, và tự viết được lab 1 và lab 5 không nhìn code mẫu; chương 6 bắt đầu từ softmax regression và gradient p − Y.

Nguồn tra cứu ngày 24/10/2026; riêng bài Nature 2025 (chỉ phần tóm tắt) mở ngày 28/10/2026.
