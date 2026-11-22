# Chương 3 — Xác suất, thống kê, lý thuyết thông tin

## Mục tiêu và kiểm tra nhanh

Chương này dạy bạn đọc mọi mô hình ML như một phân phối xác suất có tham số. Huấn luyện là MLE hoặc MAP, hàm mất mát cross-entropy bằng KL từ dữ liệu tới mô hình cộng một hằng số không phụ thuộc tham số, còn việc báo cáo "model A tốt hơn B" là một bài toán thống kê suy diễn. Người phỏng vấn track Research hỏi rất kỹ cả hai đầu: suy ra loss từ giả định xác suất, và kiểm tra một kết quả thực nghiệm có đáng tin không. Thời lượng: khoảng 2 tuần, 8 giờ mỗi tuần.

**Sau chương này, bạn phải làm được:**

- ☐ Áp dụng định lý Bayes cho bài toán có tỉ lệ nền thấp, giải thích vì sao precision đổi theo tỉ lệ nền còn recall thì không.
- ☐ Tính kỳ vọng, phương sai, hiệp phương sai của tổ hợp tuyến tính; giải thích vì sao ma trận hiệp phương sai luôn nửa xác định dương.
- ☐ Viết mật độ Gauss nhiều chiều, tính khoảng cách Mahalanobis, phân phối điều kiện, lấy mẫu bằng Cholesky, và tính softmax ổn định bằng log-sum-exp.
- ☐ Suy ra MSE và cross-entropy từ MLE, suy ra weight decay từ MAP với prior Gauss.
- ☐ Tính entropy, cross-entropy, KL, perplexity; so sánh hai model đúng cách bằng kiểm định có cặp, McNemar, bootstrap và nhiều seed.

**Kiểm tra nhanh (5 phút, không dùng máy tính).** Câu i kiểm tra mục 3.i: đúng thì đọc lướt mục đó, sai thì đọc kỹ. Đúng cả 5 câu thì làm thẳng bài ★★★ và mini-test.

1. Bộ lọc bình luận độc hại: tỉ lệ nền 2%, recall 90%, tỉ lệ dương tính giả (FPR) 5%. Một bình luận bị gắn cờ. Xác suất nó thật sự độc hại khoảng bao nhiêu?
2. X phân phối đều trên {−1, 0, 1} và Y = X². Cov(X, Y) bằng bao nhiêu? X và Y có độc lập không?
3. Mật độ của N(0, 0.1²) tại 0 bằng khoảng 3.99. Vì sao điều này không vi phạm tiên đề xác suất?
4. Tung đồng xu 3 lần, được 3 ngửa. MLE của xác suất ngửa p là bao nhiêu? Với prior Beta(2, 2) thì MAP là bao nhiêu?
5. KL(P‖Q) có bằng KL(Q‖P) không? Cho ví dụ bằng hai phân phối Bernoulli.

Đáp án ở mục Lời giải cuối chương.

## 3.1 Xác suất có điều kiện và định lý Bayes

Định lý Bayes chỉ là cách đảo chiều một xác suất có điều kiện: từ P(dữ liệu ∣ giả thuyết), thứ ta dễ đo, sang P(giả thuyết ∣ dữ liệu), thứ ta cần. Bước đảo chiều này buộc phải dùng tỉ lệ nền P(giả thuyết), và bỏ quên tỉ lệ nền là lỗi suy luận phổ biến nhất trong đánh giá model.

**Định nghĩa.** Với không gian mẫu Ω và các biến cố A, B ⊆ Ω, P(B) > 0:

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \qquad P(A \cap B) = P(A \mid B)\,P(B) \quad\text{(quy tắc nhân)}
$$

Lặp lại quy tắc nhân cho n biến ta được **quy tắc chuỗi**, nền tảng của mọi mô hình ngôn ngữ tự hồi quy: P(x₁, …, xₙ) = P(x₁)P(x₂ ∣ x₁)⋯P(xₙ ∣ x₁, …, xₙ₋₁). Nếu B₁, …, Bₖ chia Ω thành các phần rời nhau thì **xác suất toàn phần** là P(A) = ∑ᵢ P(A ∣ Bᵢ)P(Bᵢ).

**Định lý Bayes.** Gọi H là giả thuyết, D là dữ liệu quan sát:

$$
\underbrace{P(H \mid D)}_{\text{posterior}} = \frac{\overbrace{P(D \mid H)}^{\text{likelihood}}\;\overbrace{P(H)}^{\text{prior}}}{\underbrace{P(D)}_{\text{evidence}}}, \qquad P(D) = \sum_{h} P(D \mid h)\,P(h)
$$

**Dạng odds — dạng dễ tính nhẩm nhất.** Odds của H là P(H)/P(¬H). Chia Bayes cho H và ¬H, mẫu số P(D) triệt tiêu:

$$
\frac{P(H \mid D)}{P(\neg H \mid D)} = \frac{P(D \mid H)}{P(D \mid \neg H)} \times \frac{P(H)}{P(\neg H)} \qquad \text{(posterior odds = tỉ số likelihood × prior odds)}
$$

Lấy log hai vế: log-odds posterior = log-odds prior + log tỉ số likelihood. Đây chính là logit của hồi quy logistic, và là lý do đổi tỉ lệ nền chỉ làm cộng thêm một hằng số vào logit (bài tập 12).

**Độc lập.** A và B độc lập khi P(A ∩ B) = P(A)P(B). A và B **độc lập có điều kiện** theo C khi P(A ∩ B ∣ C) = P(A ∣ C)P(B ∣ C). Hai khái niệm không suy ra nhau: hai triệu chứng của cùng một bệnh thường tương quan với nhau, nhưng có thể độc lập khi đã biết bệnh. Chiều ngược lại cũng xảy ra (explaining away): hai nguyên nhân độc lập của cùng một báo động thì độc lập với nhau, nhưng khi đã biết báo động kêu, biết một nguyên nhân có mặt làm nguyên nhân kia bớt khả dĩ, tức hai nguyên nhân trở nên phụ thuộc.

**Ví dụ giải mẫu 3.1 — bộ lọc độc hại và tỉ lệ nền.** Tỉ lệ bình luận độc hại là 2%. Bộ lọc có recall P(cờ ∣ độc) = 0.90 và FPR P(cờ ∣ sạch) = 0.05.

- Xác suất toàn phần: P(cờ) = 0.90 × 0.02 + 0.05 × 0.98 = 0.018 + 0.049 = 0.067.
- Posterior: P(độc ∣ cờ) = 0.018/0.067 ≈ 0.269. Precision chỉ khoảng 27% dù recall 90% và FPR 5%. Nghĩ theo 1000 bình luận: 20 độc, 18 trong đó bị gắn cờ; 980 sạch, 49 trong đó bị gắn cờ oan. Trong 67 cờ chỉ có 18 cờ đúng.
- Dạng odds: prior odds 2/98, tỉ số likelihood 0.90/0.05 = 18, posterior odds ≈ 0.367, tức xác suất 0.367/1.367 ≈ 0.269.
- Một bộ lọc thứ hai, độc lập có điều kiện với bộ lọc đầu khi biết nhãn thật, cùng thông số, cũng gắn cờ: nhân odds với 18 lần nữa, được 6.61, tức posterior ≈ 0.869. Posterior của bước trước trở thành prior của bước sau.
- Bài học: precision phụ thuộc tỉ lệ nền, recall thì không (recall là xác suất có điều kiện theo nhãn thật). Model đạt precision cao trên tập test cân bằng sẽ tụt precision khi deploy vào dữ liệu thật có rất ít mẫu dương.

% manim: LuoiBayes

**Naive Bayes — bộ phân loại đầu tiên dựng từ Bayes.** Với đặc trưng x gồm các đặc trưng x₁, x₂, … và lớp y, giả định các đặc trưng độc lập có điều kiện khi biết y:

$$
P(y \mid x) \propto P(y) \prod_{j=1}^{d} P(x_j \mid y) \qquad\Rightarrow\qquad \hat{y} = \arg\max_y \Big[\log P(y) + \sum_{j} \log P(x_j \mid y)\Big]
$$

**Ví dụ giải mẫu 3.2 — Naive Bayes tính tay.** P(spam) = 0.4. Email chứa hai từ "free" và "meeting". P(free ∣ spam) = 0.3, P(meeting ∣ spam) = 0.05; P(free ∣ ham) = 0.02, P(meeting ∣ ham) = 0.2.

- Điểm spam ∝ 0.4 × 0.3 × 0.05 = 0.006. Điểm ham ∝ 0.6 × 0.02 × 0.2 = 0.0024.
- Chuẩn hóa: P(spam ∣ email) = 0.006/(0.006 + 0.0024) ≈ 0.714.
- Giả định độc lập có điều kiện nằm ở bước nhân P(free ∣ spam) với P(meeting ∣ spam). Trong thực tế các từ tương quan mạnh ("free" và "offer"), nên xác suất Naive Bayes thường quá tự tin, dù thứ hạng giữa các lớp vẫn hay đúng.
- Email thật có hàng trăm từ; tích hàng trăm số cỡ 10⁻³ tràn dưới (underflow) về 0 trong float64. Vì vậy luôn cộng log thay vì nhân xác suất.

**Vì sao ML cần cái này.**

- **Đánh giá trên dữ liệu mất cân bằng:** precision, false discovery rate và mọi chỉ số "có điều kiện theo dự đoán" đều phụ thuộc tỉ lệ nền. Khi tỉ lệ dương ở tập test khác lúc deploy, phải tính lại hoặc hiệu chỉnh ngưỡng.
- **Mô hình ngôn ngữ** là quy tắc chuỗi: mỗi bước dự đoán P(xₜ ∣ x₁, …, xₜ₋₁). Loss của cả chuỗi là tổng log của các thừa số.
- **Bộ phân loại sinh (generative) và phân biệt (discriminative):** Naive Bayes mô hình P(x ∣ y)P(y) rồi dùng Bayes; hồi quy logistic mô hình thẳng P(y ∣ x). Dạng odds cho thấy log-odds của Naive Bayes với đặc trưng nhị phân là hàm tuyến tính của vector đặc trưng x, cùng dạng với hồi quy logistic.

## 3.2 Biến ngẫu nhiên, kỳ vọng, phương sai, hiệp phương sai

Kỳ vọng là trung bình có trọng số theo xác suất, và nó tuyến tính với mọi biến, kể cả các biến phụ thuộc nhau. Phương sai thì không tuyến tính: phương sai của tổng chứa thêm số hạng hiệp phương sai, và chính số hạng đó quyết định ensemble hay nhiều seed giúp được bao nhiêu.

**Biến ngẫu nhiên** là một hàm X: Ω → ℝ. Biến rời rạc có hàm khối xác suất (PMF) p(x) = P(X = x); biến liên tục có hàm mật độ (PDF) p(x), với P(a ≤ X ≤ b) bằng tích phân của p từ a tới b. Hàm phân phối tích lũy (CDF) F(x) = P(X ≤ x) dùng được cho cả hai loại.

**Kỳ vọng và phương sai.**

$$
\mathbb{E}[X] = \sum_x x\,p(x) \;\;\text{hoặc}\;\; \int x\,p(x)\,dx, \qquad \operatorname{Var} X = \mathbb{E}\big[(X - \mathbb{E}X)^2\big] = \mathbb{E}[X^2] - (\mathbb{E}X)^2
$$

𝔼[aX + bY + c] = a𝔼X + b𝔼Y + c luôn đúng, không cần độc lập. Cov(X, Y) = 𝔼[(X − 𝔼X)(Y − 𝔼Y)] = 𝔼[XY] − 𝔼X𝔼Y, và hệ số tương quan ρ = Cov(X, Y)/√(Var X · Var Y) ∈ [−1, 1].

$$
\operatorname{Var}(aX + bY) = a^2 \operatorname{Var} X + b^2 \operatorname{Var} Y + 2ab\,\operatorname{Cov}(X, Y)
$$

**Không tương quan không có nghĩa là độc lập.** Độc lập ⇒ Cov = 0, nhưng chiều ngược lại sai. Tương quan chỉ đo quan hệ *tuyến tính*. Phản ví dụ kinh điển: X đều trên {−1, 0, 1}, Y = X². Khi đó 𝔼[XY] = 𝔼[X³] = 0 và 𝔼X = 0, nên Cov(X, Y) = 0, trong khi Y là một hàm của X. Ngoại lệ quan trọng: nếu (X, Y) đồng thời có phân phối Gauss thì Cov = 0 ⇔ độc lập (mục 3.3).

**Ma trận hiệp phương sai.** Với vector ngẫu nhiên x ∈ ℝᵈ có trung bình μ:

$$
\Sigma = \mathbb{E}\big[(x - \mu)(x - \mu)^{\top}\big], \qquad v^{\top} \Sigma v = \mathbb{E}\big[(v^{\top}(x - \mu))^2\big] = \operatorname{Var}(v^{\top} x) \ge 0
$$

Vậy Σ đối xứng và nửa xác định dương (PSD) với mọi phân phối: vᵀΣv là phương sai của dữ liệu chiếu lên hướng v, và phương sai không thể âm. Đây chính là thương Rayleigh của chương 1: hướng có phương sai lớn nhất là vector riêng lớn nhất của Σ (PCA). Σ suy biến (có trị riêng 0) nghĩa là dữ liệu nằm trọn trong một không gian con.

**Ước lượng từ mẫu.** Với n mẫu, trung bình mẫu x̄ và hiệp phương sai mẫu chia cho n − 1:

$$
\hat{\Sigma} = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(x_i - \bar{x})^{\top}
$$

Chia n − 1 vì x̄ đã "ăn" mất một bậc tự do: các độ lệch xᵢ − x̄ có tổng bằng 0, và chia n sẽ cho ước lượng chệch xuống (bài tập 7).

**Luật số lớn và định lý giới hạn trung tâm (CLT).** Trung bình của n mẫu độc lập cùng phân phối hội tụ về 𝔼X (luật số lớn). Hơn nữa, với n đủ lớn, x̄ xấp xỉ N(𝔼X, σ²/n) bất kể X có phân phối gì, miễn phương sai hữu hạn (CLT). Đại lượng σ/√n gọi là **sai số chuẩn** (standard error, SE). Muốn SE giảm một nửa phải tăng n gấp 4. Mục 3.5 dùng CLT để dựng khoảng tin cậy cho độ chính xác.

**Ví dụ giải mẫu 3.3 — ensemble và tương quan.** Có K model, lỗi của mỗi model là biến ngẫu nhiên có phương sai σ² = 1, tương quan từng cặp là ρ. Phương sai của lỗi trung bình:

$$
\operatorname{Var}\Big(\frac{1}{K}\sum_{k=1}^{K} X_k\Big) = \frac{1}{K^2}\Big[K\sigma^2 + K(K-1)\rho\sigma^2\Big] = \rho\,\sigma^2 + \frac{1-\rho}{K}\,\sigma^2
$$

- ρ = 0, K = 5: 0.2. ρ = 0.5, K = 5: 0.6. ρ = 0.9, K = 5: 0.92.
- ρ = 0.5, K = 1000: 0.5005. Phương sai không bao giờ xuống dưới ρσ², dù thêm bao nhiêu model.
- Bài học: ensemble và trung bình nhiều seed chỉ giảm được phần phương sai "riêng" của từng model. Random forest chọn ngẫu nhiên tập đặc trưng ở mỗi nút chính là để giảm ρ giữa các cây.

% manim: PhuongSaiEnsemble

**Ví dụ giải mẫu 3.4 — hiệp phương sai mẫu 2 chiều.** Bốn điểm (1, 2), (2, 3), (3, 5), (4, 4).

- Trung bình (2.5, 3.5). Độ lệch theo trục 1: (−1.5, −0.5, 0.5, 1.5); theo trục 2: (−1.5, −0.5, 1.5, 0.5).
- Tổng bình phương mỗi trục là 5, tổng tích chéo là 2.25 + 0.25 + 0.75 + 0.75 = 4. Chia n − 1 = 3: Σ̂ có hàng (1.667, 1.333) và (1.333, 1.667).
- Hệ số tương quan 1.333/1.667 = 0.8.
- Σ̂ có dạng hàng (a, b), (b, a), nên trị riêng là a ± b = 3 và 1/3, vector riêng (1, 1)/√2 và (1, −1)/√2. Thành phần chính đầu tiên nằm dọc đường chéo, giải thích 3/(3 + 1/3) = 90% phương sai.

**Vì sao ML cần cái này.**

- **Nhiều seed và ensemble:** công thức ví dụ 3.3 cho biết lợi ích tối đa trước khi chạy. Các seed chung dữ liệu và kiến trúc có ρ cao, nên trung bình 5 seed giảm phương sai ít hơn nhiều so với 5 lần.
- **Khởi tạo trọng số:** Xavier/He chọn phương sai trọng số sao cho Var của đầu ra mỗi lớp giữ nguyên; phép tính dựa trên Var của tổng các biến độc lập (chương 6).
- **Mini-batch gradient** là trung bình mẫu của gradient từng ví dụ; SE của nó giảm như 1/√B theo kích thước batch B. Đó là lý do tăng batch gấp 4 chỉ giảm nhiễu gradient một nửa.

## 3.3 Các phân phối hay dùng trong ML

Một vài phân phối xuất hiện ở khắp nơi: Bernoulli và categorical cho nhãn, Gauss cho nhiễu và cho biến ẩn. Hiểu Gauss nhiều chiều qua trị riêng của Σ là hiểu luôn khoảng cách Mahalanobis, phát hiện bất thường, và cách các thư viện lấy mẫu.

**Bernoulli và categorical.** Bernoulli(p): X ∈ {0, 1}, P(X = 1) = p, 𝔼X = p, Var X = p(1 − p). Categorical(π) trên K lớp: viết nhãn dạng one-hot y ∈ {0, 1}ᴷ, thì

$$
P(y) = \prod_{k=1}^{K} \pi_k^{\,y_k}
$$

Mạng nơ-ron không xuất π trực tiếp mà xuất **logit** z ∈ ℝᴷ, rồi tham số hóa bằng softmax: πₖ = exp(zₖ)/∑ⱼ exp(zⱼ). Binomial(n, p) là tổng n biến Bernoulli độc lập, có kỳ vọng np và phương sai np(1 − p).

**Gauss một chiều.**

$$
\mathcal{N}(x;\, \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\Big(-\frac{(x-\mu)^2}{2\sigma^2}\Big)
$$

Khoảng 68.3%, 95.4%, 99.7% khối xác suất nằm trong 1, 2, 3 độ lệch chuẩn. **Mật độ không phải xác suất:** N(0, 0.1²) có mật độ tại 0 bằng 1/(0.1√(2π)) ≈ 3.99. Chỉ tích phân của mật độ trên một khoảng mới là xác suất, và tích phân trên cả trục bằng 1. Vì thế log-likelihood liên tục có thể dương, và "loss âm" trong mô hình liên tục không phải lỗi.

**Gauss nhiều chiều.** Với μ ∈ ℝᵈ và Σ đối xứng xác định dương:

$$
\mathcal{N}(x;\, \mu, \Sigma) = \frac{1}{(2\pi)^{d/2} (\det \Sigma)^{1/2}} \exp\Big(-\tfrac{1}{2}\,\Delta^2\Big), \qquad \Delta^2 = (x-\mu)^{\top} \Sigma^{-1} (x-\mu)
$$

Δ là **khoảng cách Mahalanobis**. Viết Σ = QΛQᵀ (định lý phổ, chương 1) thì Δ² = ∑ᵢ (qᵢᵀ(x − μ))²/λᵢ: đo khoảng cách theo từng hướng riêng, chia cho phương sai theo hướng đó. Đường đồng mức Δ = const là elip có trục theo các vector riêng qᵢ, bán trục tỉ lệ √λᵢ. Mật độ có det Σ ở mẫu số: log det Σ xuất hiện trong mọi log-likelihood Gauss.

**Biên và điều kiện vẫn là Gauss.** Chia x = (x₁, x₂) và chia μ, Σ thành khối tương ứng. Biên của x₁ là N(μ₁, Σ₁₁): chỉ cần xóa hàng và cột. Điều kiện:

$$
x_2 \mid x_1 \sim \mathcal{N}\Big(\mu_2 + \Sigma_{21}\Sigma_{11}^{-1}(x_1 - \mu_1),\;\; \Sigma_{22} - \Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12}\Big)
$$

Trung bình điều kiện là một hàm tuyến tính của x₁ (đây là hồi quy tuyến tính "miễn phí"), còn phương sai điều kiện không phụ thuộc giá trị x₁ và không bao giờ lớn hơn Σ₂₂ theo nghĩa ma trận (Σ₂₂ trừ đi một ma trận nửa xác định dương): biết thêm thông tin thì bất định không tăng. Với Gauss, Σ₁₂ = 0 làm phân phối điều kiện bằng biên, tức độc lập.

**Lấy mẫu bằng Cholesky.** Phân rã Σ = LLᵀ với L tam giác dưới. Nếu z ~ N(0, I) thì x = μ + Lz có Cov(x) = L·I·Lᵀ = Σ. Cùng phân rã đó cho log det Σ = 2∑ᵢ log Lᵢᵢ và cho Δ² = ‖L⁻¹(x − μ)‖₂², tính bằng thế tiến, không cần nghịch đảo Σ.

**Ví dụ giải mẫu 3.5 — softmax và log-sum-exp.** Logit z = (2, 1, 0).

- exp(z) = (7.389, 2.718, 1), tổng 11.107, nên softmax ≈ (0.665, 0.245, 0.090).
- Với z = (1000, 999, 998), exp(1000) tràn số thành inf trong float64, và inf/inf cho nan ở cả ba vị trí.
- Softmax không đổi khi cộng cùng một hằng số vào mọi logit. Trừ m = max zᵢ = 1000 được (0, −1, −2), lại ra đúng (0.665, 0.245, 0.090).

$$
\log \sum_{i} e^{z_i} = m + \log \sum_{i} e^{z_i - m}, \qquad m = \max_i z_i, \qquad \log \operatorname{softmax}(z)_k = z_k - \log\sum_i e^{z_i}
$$

Với z = (1000, 999, 998), log-sum-exp ≈ 1000.408. Sau khi trừ m, số mũ lớn nhất là e⁰ = 1, nên không tràn; tổng ít nhất bằng 1 nên log không bị −∞.

**Ví dụ giải mẫu 3.6 — Gauss 2 chiều, Mahalanobis, điều kiện.** μ = 0, Σ có hàng (2, 1) và (1, 2) (chính ma trận ví dụ 1.6).

- Trị riêng 3 theo hướng (1, 1) và 1 theo hướng (1, −1): elip đồng mức nằm dọc đường chéo, tỉ lệ bán trục √3 : 1. det Σ = 3.
- Hai điểm (1, 1) và (1, −1) cùng cách gốc √2 theo Euclid. Σ⁻¹ có hàng (2, −1)/3 và (−1, 2)/3. Δ² của (1, 1) là 2/3 ≈ 0.667, mật độ ≈ 0.0658. Δ² của (1, −1) là 2.0, mật độ ≈ 0.0338. Điểm đi ngược hướng tương quan có mật độ chỉ bằng khoảng một nửa.
- Điều kiện x₂ ∣ x₁ = 2: trung bình (1/2) × 2 = 1.0, phương sai 2 − 1 × 1/2 = 1.5 < 2.
- Cholesky: L có hàng (1.414, 0) và (0.707, 1.225); lấy mẫu x = Lz. Lab 2 ước lượng lại Σ từ 200 000 mẫu, sai lệch dưới 0.002.

% manim: ElipGauss

**Vì sao ML cần cái này.**

- **Lớp đầu ra:** sigmoid cho Bernoulli, softmax cho categorical, đầu ra tuyến tính cho Gauss với phương sai cố định (muốn phương sai học được thì thêm một đầu ra log σ², dẫn tới heteroscedastic loss). Chọn phân phối đầu ra là chọn loss (mục 3.4).
- **Phát hiện bất thường và dữ liệu ngoài phân phối:** khớp một Gauss lên embedding của dữ liệu huấn luyện, rồi dùng Δ² làm điểm bất thường. Euclid bỏ qua cấu trúc tương quan, Mahalanobis thì không.
- **VAE và diffusion:** biến ẩn và nhiễu là Gauss; lấy mẫu dạng μ + σz (reparameterization) là trường hợp chéo của x = μ + Lz. Gaussian process dùng đúng công thức điều kiện ở trên để dự đoán.

**Thực tế tính toán.** Không bao giờ tính softmax rồi mới lấy log: hãy tính log-softmax trực tiếp bằng log-sum-exp. Các framework gộp hai bước này (ví dụ hàm cross-entropy của PyTorch nhận logit chưa chuẩn hóa, không nhận xác suất). SciPy có `scipy.special.logsumexp`. Với Gauss nhiều chiều, dùng Cholesky thay cho `inv` và `det`: det của ma trận 100 × 100 có trị riêng cỡ 10⁻⁴ là 10⁻⁴⁰⁰, tràn dưới về 0, trong khi tổng log của đường chéo L vẫn tính được.

## 3.4 MLE và MAP

Huấn luyện một mô hình xác suất là chọn tham số θ làm dữ liệu quan sát "dễ xảy ra" nhất: đó là MLE. Thêm một niềm tin ban đầu về θ thì thành MAP, và log của niềm tin đó chính là số hạng regularization. Hai câu này giải thích nguồn gốc của gần như mọi loss bạn dùng.

**Likelihood không phải xác suất của θ.** Với dữ liệu D = {x₁, …, xₙ} độc lập cùng phân phối, likelihood L(θ) = p(D ∣ θ) = ∏ᵢ p(xᵢ ∣ θ) là một hàm của θ khi D cố định. Nó không tích phân bằng 1 theo θ, nên không phải phân phối của θ. Ta cực đại log-likelihood thay vì likelihood: log biến tích thành tổng (tránh underflow, dễ lấy đạo hàm) và không đổi vị trí cực đại.

$$
\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \sum_{i=1}^{n} \log p(x_i \mid \theta), \qquad \hat{\theta}_{\text{MAP}} = \arg\max_{\theta} \Big[\sum_{i=1}^{n} \log p(x_i \mid \theta) + \log p(\theta)\Big]
$$

MAP là mode của posterior p(θ ∣ D) ∝ p(D ∣ θ)p(θ); evidence p(D) không phụ thuộc θ nên bỏ được.

**MLE của Bernoulli.** Với h lần ngửa trong n lần tung, log L(p) = h log p + (n − h) log(1 − p). Đạo hàm h/p − (n − h)/(1 − p) = 0 cho p̂ = h/n: MLE chỉ là đếm tỉ lệ.

**MLE của Gauss.** Lấy đạo hàm log-likelihood theo μ và σ² rồi cho bằng 0: μ̂ = x̄ và σ̂² = (1/n)∑(xᵢ − x̄)². Ước lượng phương sai của MLE chia cho n, nên chệch: kỳ vọng của nó bằng (n − 1)σ²/n. Ví dụ dữ liệu (2, 4, 4, 4, 5, 5, 7, 9): μ̂ = 5, σ̂² (MLE, chia n) = 4.0, ước lượng không chệch (chia n − 1) = 4.571. Không chệch chưa chắc tốt hơn: ước lượng chia n có sai số bình phương trung bình nhỏ hơn với dữ liệu Gauss.

**Loss là negative log-likelihood.**

- *Hồi quy với nhiễu Gauss.* y = f(x; w) + ε, ε ~ N(0, σ²). Khi đó −log p(y ∣ x, w) = (y − f(x; w))²/(2σ²) + const. Cực đại likelihood ⇔ cực tiểu MSE.
- *Phân loại nhị phân.* y ∈ {0, 1}, mô hình dự đoán q = sigmoid(f(x)). −log p(y ∣ x) = −y log q − (1 − y) log(1 − q): binary cross-entropy.
- *Phân loại K lớp.* −log p(y ∣ x) = −log softmax(z)ᵧ: cross-entropy với nhãn one-hot.
- *Nhiễu Laplace* p(ε) ∝ exp(−|ε|/b) cho MAE; MAE bền với ngoại lai hơn MSE vì phạt tuyến tính chứ không phạt bình phương.

$$
-\log p(y \mid x, w) = \frac{1}{2\sigma^2}\big(y - f(x; w)\big)^2 + \frac{1}{2}\log(2\pi\sigma^2) \qquad\Rightarrow\qquad \arg\max_w \text{likelihood} = \arg\min_w \sum_i \big(y_i - f(x_i; w)\big)^2
$$

**MAP với prior Gauss là ridge (weight decay).** Đặt prior w ~ N(0, τ²I). Lấy −log posterior và nhân cả hai vế với 2σ²:

$$
-\log p(w \mid \mathcal{D}) = \frac{1}{2\sigma^2}\sum_i \big(y_i - w^{\top} x_i\big)^2 + \frac{1}{2\tau^2}\lVert w\rVert_2^2 + \text{const} \;\;\propto\;\; \sum_i \big(y_i - w^{\top}x_i\big)^2 + \lambda \lVert w \rVert_2^2, \quad \lambda = \frac{\sigma^2}{\tau^2}
$$

Điều kiện: λ = σ²/τ² chỉ đúng khi loss là *tổng* bình phương trên n mẫu, đúng như dạng trên. Nếu loss là *trung bình* (mặc định của framework) thì hệ số phạt tương ứng là σ²/(nτ²) = λ/n, và hệ số weight decay phải chia cho n tương ứng; đây là câu bẫy phỏng vấn hay gặp. Còn một lệch hệ số 2 nữa: tham số `weight_decay = wd` của SGD trong PyTorch ứng với phạt (wd/2)‖w‖₂², trong khi MSE trung bình (1/n)∑(y − wᵀx)² không có hệ số ½. Với loss trung bình (1/n)∑(y − wᵀx)² + (λ/n)‖w‖₂² (cùng nghiệm với dạng tổng ở trên) thì wd = 2λ/n [quy ước của SGD theo tài liệu PyTorch; cần kiểm chứng lại trên docs hiện hành]. Prior càng hẹp (τ² nhỏ), λ càng lớn. Prior Laplace cho phạt L1 (lasso). Nghiệm đóng (XᵀX + λI)⁻¹Xᵀy là nghiệm ridge của chương 1.

**Beta–Bernoulli: prior là "đếm giả".** Prior Beta(a, b) có mật độ ∝ pᵃ⁻¹(1 − p)ᵇ⁻¹. Nhân với likelihood pʰ(1 − p)ⁿ⁻ʰ được posterior Beta(a + h, b + n − h): cùng họ với prior (prior liên hợp, conjugate prior). Có hai cách đọc "đếm giả", tùy ta lấy MAP hay trung bình posterior. Theo MAP, prior Beta(a, b) giống như đã thấy a − 1 lần ngửa và b − 1 lần sấp trước khi có dữ liệu. Theo trung bình posterior, số đếm giả là a và b.

$$
\hat{p}_{\text{MAP}} = \frac{h + a - 1}{n + a + b - 2}, \qquad \mathbb{E}[p \mid \mathcal{D}] = \frac{h + a}{n + a + b}
$$

**Ví dụ giải mẫu 3.7 — đồng xu: MLE và MAP.** Tung 10 lần, được 7 ngửa. Prior Beta(2, 2).

- MLE p̂ = 0.7. Posterior Beta(9, 5). MAP = (7 + 1)/(10 + 2) = 8/12 ≈ 0.667. Trung bình posterior = 9/14 ≈ 0.643.
- Ca cực đoan: 3 lần tung đều ngửa. MLE = 1, tức gán xác suất 0 cho mặt sấp: một lần sấp tiếp theo sẽ có log-likelihood −∞. MAP = (3 + 1)/(3 + 2) = 0.8.
- Add-one (Laplace) smoothing trong mô hình n-gram, cộng 1 vào mọi bộ đếm để không từ nào có xác suất 0, ứng với hai cách đọc. (i) Luật kế tiếp của Laplace (rule of succession), gốc của tên gọi, là trung bình posterior với prior Beta(1, 1) (đều): (h + 1)/(n + 2). (ii) MAP với Beta(2, 2) cũng cho (h + 1)/(n + 2), như phép tính 8/12 ở trên. Với K lớp: trung bình posterior của Dirichlet(1, …, 1), hoặc MAP của Dirichlet(2, …, 2), đều cho (đếm + 1)/(tổng + K).

**Ví dụ giải mẫu 3.8 — ridge là MAP, một tham số.** Mô hình y = wx + ε, ε ~ N(0, σ²), dữ liệu x = (1, 2, 3), y = (2, 4, 7).

- MLE (bình phương tối thiểu không hệ số chặn): ŵ = ∑xᵢyᵢ/∑xᵢ² = (2 + 8 + 21)/(1 + 4 + 9) = 31/14 ≈ 2.214.
- Prior w ~ N(0, τ²) với σ² = 1, τ² = 0.1, nên λ = 10. ŵ (MAP) = 31/(14 + 10) = 31/24 ≈ 1.292: bị kéo về 0.
- Với τ² = 1 thì λ = 1 và ŵ (MAP) = 31/15 ≈ 2.067; với τ² = 0.01 thì λ = 100 và ŵ (MAP) ≈ 0.272.
- Nhiều dữ liệu làm ∑xᵢ² lớn lên trong khi λ giữ nguyên, nên prior bị lấn át. Lab 4 đo: khoảng cách giữa MAP và MLE giảm từ 1.738 (n = 10) xuống 0.0022 (n = 10 000).

% manim: PriorKeoVe0

**Vì sao ML cần cái này.**

- **Chọn loss có nguyên tắc:** đổi giả định nhiễu là đổi loss. Dữ liệu có ngoại lai nặng thì nhiễu Gauss là giả định tồi, nên MSE bị kéo lệch; MAE hoặc Huber loss hợp hơn.
- **Weight decay** là MAP với prior Gauss trong SGD thường. Với Adam, cộng λ‖w‖₂² vào loss không còn tương đương với co trọng số trực tiếp, vì gradient của phạt bị chia theo từng tọa độ; đó là động lực của AdamW (chương 4).
- **Smoothing trong mô hình ngôn ngữ:** add-one, label smoothing và việc tránh xác suất 0 đều là cách đưa prior vào để mô hình không quá tự tin.

**Thực tế tính toán.** MAP phụ thuộc cách tham số hóa: đổi biến θ = g(φ) làm mật độ nhân thêm Jacobian, nên mode có thể dời đi, trong khi MLE thì bất biến. Trong deep learning, "MAP" chỉ là cách đọc của regularization; không ai dùng nó làm ước lượng posterior thật, vì mode của một posterior hàng tỉ chiều gần như không nói gì về độ bất định.

## 3.5 Lý thuyết thông tin và thống kê suy diễn cho người làm thực nghiệm

Mục này có hai nửa nối với nhau bằng một ý: loss bạn tối thiểu (cross-entropy) là thước đo thông tin, còn con số bạn báo cáo (accuracy, BLEU) là một ước lượng có sai số. Nửa A giải thích loss đo cái gì; nửa B giải thích khi nào hiệu số giữa hai model là thật.

### A. Entropy, cross-entropy, KL, mutual information

**Self-information và entropy.** Một sự kiện có xác suất p mang −log p đơn vị thông tin: sự kiện càng hiếm càng "bất ngờ". Entropy là độ bất ngờ trung bình, đồng thời là cận dưới của độ dài mã trung bình khi mã hóa theo đúng p: mã nhị phân độ dài nguyên (như Huffman) cho độ dài trong khoảng [H, H + 1), đạt đúng H khi mọi p(x) là lũy thừa của ½, và tiệm cận H khi mã hóa theo khối. Dùng log cơ số 2 thì đơn vị là bit; log tự nhiên thì đơn vị là nat (1 nat ≈ 1.443 bit).

$$
H(p) = -\sum_x p(x)\log p(x), \qquad H(p, q) = -\sum_x p(x)\log q(x), \qquad \mathrm{KL}(p\,\|\,q) = \sum_x p(x)\log\frac{p(x)}{q(x)}
$$

H(p, q) là **cross-entropy**: độ dài mã trung bình (lý tưởng, tiệm cận) khi dữ liệu theo p nhưng ta mã hóa theo q. Phần dôi ra so với mã tối ưu chính là KL:

$$
H(p, q) = H(p) + \mathrm{KL}(p\,\|\,q), \qquad \mathrm{KL}(p\,\|\,q) \ge 0, \;\; \text{dấu bằng} \Leftrightarrow p = q
$$

KL ≥ 0 suy ra từ bất đẳng thức Jensen (bài tập 11).

**Cross-entropy loss = MLE = cực tiểu KL tới dữ liệu.** Gọi p̂ là phân phối thực nghiệm của tập huấn luyện. Cross-entropy trung bình H(p̂, q(·; θ)) = −(1/n)∑ᵢ log q(·; θ)(xᵢ), đúng là negative log-likelihood chia n. H(p̂) không phụ thuộc θ, nên:

$$
\arg\min_\theta H(\hat{p}, q_\theta) = \arg\min_\theta \mathrm{KL}(\hat{p}\,\|\,q_\theta) = \arg\max_\theta \sum_i \log q_\theta(x_i) = \hat{\theta}_{\text{MLE}}
$$

Ba cách nói, một bài toán. Trong phân loại, loss trung bình là (1/n)∑ᵢ H(tᵢ, qᵢ) ≥ (1/n)∑ᵢ H(tᵢ), với tᵢ là phân phối mục tiêu của mẫu i và qᵢ là dự đoán của mô hình. Cận dưới bằng 0 khi mọi tᵢ là one-hot, và dương khi dùng nhãn mềm. Cận này chỉ đúng cho biến rời rạc: với biến liên tục, NLL có thể âm (mục 3.3).

**Perplexity** của mô hình ngôn ngữ là exp của cross-entropy trung bình theo token (tính bằng nat): PPL = exp(−(1/T)∑ₜ log q(xₜ ∣ x₁, …, xₜ₋₁)). Đọc trực giác: mô hình bối rối như thể phải chọn đều giữa PPL token ở mỗi bước. Perplexity chỉ so sánh được giữa các mô hình dùng cùng tokenizer và cùng tập test, vì "một token" ở hai tokenizer là hai đơn vị khác nhau.

**KL không đối xứng.** KL(p‖q) lấy trung bình theo p: nó phạt nặng (tới ∞) khi q ≈ 0 ở chỗ p > 0, nên cực tiểu theo q cho một q trải rộng phủ mọi vùng p có khối lượng (**mode-covering**). KL(q‖p) lấy trung bình theo q: nó phạt khi q đặt khối lượng ở chỗ p ≈ 0, nên q được phép bỏ qua một đỉnh của p và bám chặt một đỉnh khác (**mode-seeking**). Ví dụ: p = ½N(−2, 0.6²) + ½N(2, 0.6²), fit một Gauss q. Cực tiểu KL(p‖q) cho q = N(0, 2.09²), khớp đúng trung bình và phương sai của p (bài tập 14). Cực tiểu KL(q‖p) bằng tối ưu số cho q ≈ N(2.0, 0.60²), tức chỉ một đỉnh, với KL ≈ 0.692 ≈ ln 2. Variational inference và VAE tối thiểu KL(q‖p) theo q, nên xấp xỉ posterior thường quá hẹp. KL cũng không thỏa bất đẳng thức tam giác: nó không phải khoảng cách.

% manim: KLKhongDoiXung

**Mutual information** đo X cho biết bao nhiêu về Y: I(X; Y) = KL(p(x, y)‖p(x)p(y)) = H(X) + H(Y) − H(X, Y) = H(Y) − H(Y ∣ X). I = 0 ⇔ X và Y độc lập, kể cả với quan hệ phi tuyến mà tương quan bỏ sót.

**Ví dụ giải mẫu 3.9 — cross-entropy là NLL, perplexity.**

- Model dự đoán q = (0.7, 0.2, 0.1), nhãn đúng là lớp 1. Loss = −ln 0.7 ≈ 0.357 nat (0.515 bit). Chỉ xác suất của lớp đúng tham gia.
- Chuỗi 4 token có xác suất token đúng (0.5, 0.25, 0.1, 0.8). NLL trung bình = (0.693 + 1.386 + 2.303 + 0.223)/4 ≈ 1.151 nat. Perplexity = e¹·¹⁵¹ ≈ 3.16.
- Model đoán đều trên từ vựng 50 000 token có cross-entropy ln 50 000 ≈ 10.82 nat, perplexity 50 000. Đây là mốc để kiểm loss lúc khởi tạo: loss ban đầu lệch xa ln(cỡ từ vựng) thường là dấu hiệu lỗi khởi tạo hoặc lỗi nhãn.

**Ví dụ giải mẫu 3.10 — entropy, KL, MI, label smoothing.**

- H(0.5, 0.5) = 1 bit; H(0.9, 0.1) ≈ 0.469 bit; phân phối đều trên 4 lớp: 2 bit.
- P = (0.5, 0.5), Q = (0.9, 0.1): KL(P‖Q) ≈ 0.737 bit, KL(Q‖P) ≈ 0.531 bit (tính bằng nat: 0.511 và 0.368). Kiểm: H(P, Q) = −0.5 log₂ 0.9 − 0.5 log₂ 0.1 ≈ 1.737 = H(P) + KL(P‖Q).
- Phân phối đồng thời p(x, y) có hàng (0.4, 0.1) và (0.1, 0.4), hai biên đều đều. H(X, Y) ≈ 1.722 bit, nên I(X; Y) = 1 + 1 − 1.722 ≈ 0.278 bit.
- Label smoothing với K = 3, ε = 0.1: mục tiêu (1 − ε)·one-hot + ε/K = (0.933, 0.033, 0.033). Với q = (0.7, 0.2, 0.1), CE = 0.463 nat. CE nhỏ nhất đạt được (khi q bằng đúng mục tiêu) là H(mục tiêu) ≈ 0.291 nat, không phải 0. Đó là lý do loss train có "sàn" khi bật label smoothing.

### B. Thống kê suy diễn: báo cáo kết quả sao cho đáng tin

**Sai số chuẩn và khoảng tin cậy.** Độ chính xác trên n câu test là trung bình mẫu của n biến Bernoulli, nên SE ≈ √(p̂(1 − p̂)/n). Theo CLT, khoảng Wald p̂ ± 1.96·SE là khoảng tin cậy (CI) 95% xấp xỉ.

**Diễn giải đúng CI 95%:** nếu lặp lại toàn bộ thủ tục (lấy mẫu test mới, tính khoảng) rất nhiều lần, khoảng 95% số khoảng tính được sẽ chứa giá trị thật. Con số 95% thuộc về *thủ tục*, không phải về một khoảng cụ thể đã tính. Khoảng Wald kém khi n nhỏ hoặc p gần 0 hay 1; khi đó dùng khoảng Wilson hoặc bootstrap.

**Kiểm định giả thuyết và p-value.** Đặt giả thuyết không H₀ (ví dụ: hai model có cùng độ chính xác thật). p-value là xác suất, *khi H₀ đúng*, thu được thống kê cực đoan ít nhất như dữ liệu đã thấy. p-value không phải P(H₀ đúng), và "không có ý nghĩa thống kê" không có nghĩa "hai model bằng nhau", chỉ là dữ liệu chưa đủ để phân biệt.

**Có cặp và không cặp.** Hai model chạy trên *cùng* tập test cho dữ liệu có cặp: mỗi câu cho một cặp (đúng/sai của A, đúng/sai của B). Phương sai của hiệu là Var(a − b) = Var a + Var b − 2Cov(a, b). Hai model mạnh thường cùng đúng ở câu dễ và cùng sai ở câu khó, nên Cov lớn và dương. Kiểm định không cặp bỏ qua số hạng Cov, nên khi Cov(a, b) > 0 (trường hợp thường gặp) nó đánh giá quá cao phương sai và mất độ mạnh.

**McNemar.** Chỉ những câu hai model bất đồng mới mang thông tin về hiệu số. Gọi b là số câu A đúng B sai, c là số câu A sai B đúng. Dưới H₀, mỗi câu bất đồng nghiêng về A hay B với xác suất ½, nên b ~ Binomial(b + c, ½). Bản xấp xỉ χ² có hiệu chỉnh liên tục:

$$
\chi^2 = \frac{(|b - c| - 1)^2}{b + c} \quad (\text{1 bậc tự do}), \qquad \text{bản chính xác: } b \sim \operatorname{Binomial}\big(b + c, \tfrac{1}{2}\big)
$$

**Bootstrap.** Khi thống kê không có SE dạng đóng (BLEU, F1 toàn tập, chỉ số trên cả corpus), lấy mẫu lại n câu test *có hoàn lại* hàng nghìn lần, tính lại thống kê mỗi lần, rồi lấy phân vị 2.5% và 97.5% làm CI. **Paired bootstrap** (Koehn, 2004) dùng *cùng* một bộ chỉ số được lấy mẫu lại cho cả hai hệ thống, nên giữ được cấu trúc cặp.

**Ví dụ giải mẫu 3.11 — hai model trên cùng 1000 câu test.** A đúng 850 câu, B đúng 835 câu. Bảng chéo: cả hai đúng 820, chỉ A đúng 30, chỉ B đúng 15, cả hai sai 135.

- CI 95% Wald cho A: SE = √(0.85 × 0.15/1000) ≈ 0.0113, CI ≈ [0.828, 0.872]. Nếu chỉ có 100 câu: [0.780, 0.920], rộng gấp √10 ≈ 3.2 lần. Muốn nửa độ rộng 1 điểm % ở mức 85% cần 1.96² × 0.85 × 0.15/0.01² ≈ 4 898 câu.
- Không cặp: SE của hiệu ≈ √(0.85 × 0.15/1000 + 0.835 × 0.165/1000) ≈ 0.0163. Hiệu 0.015 chỉ bằng 0.92 SE, trông như không có ý nghĩa.
- Có cặp: SE ≈ 0.0067, nhỏ hơn khoảng 2.4 lần (bài tập 13), vì 955/1000 câu hai model đồng ý.
- McNemar với b = 30, c = 15: χ² = (15 − 1)²/45 ≈ 4.36, p ≈ 0.037; bản chính xác p ≈ 0.036.
- Paired bootstrap 10 000 lần (lab 5, seed 0): CI 95% của hiệu ≈ [0.002, 0.028], không chứa 0; tỉ lệ lần hiệu ≤ 0 khoảng 0.012. Lấy mẫu lại độc lập cho A và B (sai) cho CI ≈ [−0.017, 0.046], vắt qua 0 (hàm `unpaired_bootstrap_ci` trong lab 5 cho cận trên 0.047). Các số bootstrap đổi nhẹ theo seed và thứ tự dữ liệu.

% manim: PairedBootstrap

**Ví dụ giải mẫu 3.12 — 5 seed và đa kiểm định.** Độ chính xác (%) của A qua 5 seed: 84.1, 85.3, 83.7, 84.9, 85.0; của B: 83.6, 84.2, 83.9, 84.1, 84.3. Mỗi cặp dùng cùng seed và cùng cách chia dữ liệu.

- Trung bình 84.60 so với 84.02, hiệu 0.58, độ lệch chuẩn của hiệu theo cặp 0.487.
- Paired t-test (4 bậc tự do): t = 0.58/(0.487/√5) ≈ 2.66, p ≈ 0.056. Welch không cặp: p ≈ 0.130 (`ttest_ind` mặc định gộp phương sai nên cho p ≈ 0.112; cả hai đều bỏ qua cặp nên đều sai chỗ). CI 95% t của hiệu: 0.58 ± 2.776 × 0.487/√5 ≈ [−0.02, 1.18], chứa 0.
- Đừng "chữa" cỡ mẫu nhỏ bằng kiểm định phi tham số. Wilcoxon signed-rank chính xác với n = 5 có p hai phía nhỏ nhất là 2/32 = 0.0625, nên không bao giờ bác bỏ được ở α = 0.05 (trên dữ liệu này `scipy.stats.wilcoxon` cho p = 0.125). Paired bootstrap trên 5 hiệu seed cho CI quá hẹp: trên dữ liệu này cỡ [0.2, 0.9] (đổi nhẹ theo seed), không chứa 0, mâu thuẫn với CI t; mô phỏng dữ liệu Gauss với n = 5 cho thấy CI percentile bootstrap chỉ phủ khoảng 84% thay vì 95%. Cách đúng: thêm seed (từ 10 trở lên), báo CI t của hiệu như trên, hoặc paired bootstrap trên các câu test gộp qua seed (như ví dụ 3.11). Coi p = 0.056 là gợi ý yếu chứ không phải kết luận, và "không có ý nghĩa" ở đây không có nghĩa là A và B bằng nhau: CI của hiệu vẫn cho phép hiệu tới hơn 1 điểm.
- Báo cáo "84.6 so với 84.0" một mình là không đủ: với 5 seed, hiệu 0.6 điểm vẫn ở ranh giới. Lấy seed tốt nhất của mỗi bên (85.3 so với 84.3) còn phóng đại hiệu thêm.
- Đa kiểm định: 20 so sánh độc lập ở α = 0.05, nếu mọi H₀ đúng thì xác suất có ít nhất một "phát hiện" giả là 1 − 0.95²⁰ ≈ 0.64. Hiệu chỉnh Bonferroni dùng ngưỡng α/20 = 0.0025 cho mỗi so sánh.

**Vì sao ML cần cái này.**

- **Bảng kết quả trong paper:** hiệu 0.5 điểm trên một tập test vài nghìn câu thường nằm trong sai số. Người phản biện và người phỏng vấn sẽ hỏi bạn đã kiểm định chưa.
- **NLP dùng bootstrap:** BLEU không phải trung bình theo câu nên không có SE dạng đóng; Koehn (2004) đề xuất bootstrap và paired bootstrap cho dịch máy. Dror và cộng sự (ACL 2018) khảo sát bài ACL/TACL 2017 và thấy kiểm định ý nghĩa thường bị bỏ qua hoặc dùng sai.
- **Chọn siêu tham số trên test** là đa kiểm định trá hình: thử 20 cấu hình rồi báo cái tốt nhất trên test cho một ước lượng lạc quan. Chọn trên tập validation, báo cáo trên test một lần.

**Thực tế tính toán.** Kết quả deep learning dao động theo seed, thứ tự dữ liệu, khởi tạo và lựa chọn siêu tham số. Bouthillier và cộng sự (MLSys 2021) đo các nguồn phương sai này và khuyến nghị ngẫu nhiên hóa nhiều nguồn khi so sánh. Quy tắc thực dụng: ít nhất 3–5 seed, báo cáo mean ± std, so sánh có cặp. Dietterich (1998) chỉ ra t-test trên nhiều lần chia train/test ngẫu nhiên có tỉ lệ sai lầm loại I cao, và khuyến nghị McNemar hoặc 5×2cv. `scipy.stats.bootstrap` mặc định dùng phương pháp BCa với 9 999 lần lấy mẫu lại và có tùy chọn `paired`.

## Bài tập giấy bút

14 bài, làm theo thứ tự. Mục tiêu: hết ★ trong 1 buổi, hết ★★ trong 2 buổi, ★★★ dành cho Thứ Bảy. Bài nào sai ghi vào sổ lỗi kèm lý do sai.

**★ Hiểu định nghĩa**

1. Tung hai xúc xắc cân đối. Biết có ít nhất một con ra 6, tính xác suất cả hai con đều ra 6.
2. X là kết quả tung một xúc xắc cân đối. Tính 𝔼X, 𝔼[X²] và Var X.
3. Chứng minh Var X = 𝔼[X²] − (𝔼X)² và Var(aX + b) = a²Var X.
4. Tính entropy (bit) của phân phối (0.5, 0.25, 0.25) và perplexity tương ứng. Cho một mã nhị phân có độ dài trung bình đúng bằng entropy đó.

**★★ Mức đề thi đầu vào**

5. Một bệnh có tỉ lệ 1%, xét nghiệm có độ nhạy 99% và FPR 5%. (a) Tính P(bệnh ∣ dương). (b) Làm thêm một xét nghiệm độc lập có điều kiện, cùng thông số, cũng dương: posterior mới bằng bao nhiêu? (c) FPR phải bằng bao nhiêu để sau một lần dương, P(bệnh ∣ dương) = 0.5?
6. (a) Chứng minh mọi ma trận hiệp phương sai đều nửa xác định dương. (b) Suy ra công thức phương sai của trung bình K model trong ví dụ 3.3 từ công thức Var của tổng.
7. Với n mẫu độc lập từ N(μ, σ²), suy ra MLE của μ và σ². Chứng minh kỳ vọng của ước lượng σ² theo MLE bằng (n − 1)σ²/n.
8. Suy ra loss từ MLE cho ba trường hợp: (a) hồi quy với nhiễu Gauss, (b) phân loại nhị phân với đầu ra sigmoid, (c) hồi quy với nhiễu Laplace. Với (c), hằng số c nào cực tiểu tổng |yᵢ − c|?
9. x ~ N(0, Σ) với Σ có hàng (4, 2) và (2, 3). (a) Tìm phân phối của x₂ ∣ x₁ = 2. (b) Tính Mahalanobis² của điểm (2, 0). (c) Tìm trị riêng của Σ và ma trận Cholesky L.
10. Với hai Gauss một chiều, KL(N(μ₁, σ₁²)‖N(μ₂, σ₂²)) = ln(σ₂/σ₁) + (σ₁² + (μ₁ − μ₂)²)/(2σ₂²) − ½. Tính KL(N(0, 1)‖N(1, 2²)) và KL(N(1, 2²)‖N(0, 1)). Vì sao chiều thứ hai lớn hơn nhiều?

**★★★ Mức phỏng vấn nghiên cứu**

11. Chứng minh KL(p‖q) ≥ 0 với dấu bằng khi và chỉ khi p = q. Từ đó chứng minh: cực tiểu cross-entropy trên tập huấn luyện tương đương MLE, và tương đương cực tiểu KL từ phân phối thực nghiệm tới mô hình.
12. Một bộ phân loại được huấn luyện trên dữ liệu cân bằng (50% dương) và cho xác suất 0.8 với một mẫu. Khi deploy, tỉ lệ dương thật chỉ là 5%. Giả sử P(x ∣ y) không đổi giữa hai môi trường. Posterior đúng là bao nhiêu? Viết phép hiệu chỉnh dưới dạng cộng một hằng số vào logit.
13. Trong ví dụ 3.11, gọi dᵢ = aᵢ − bᵢ ∈ {−1, 0, 1}. Suy ra SE có cặp theo b, c, n và kiểm con số 0.0067. Giải thích bằng Cov(a, b) vì sao SE không cặp lớn gấp khoảng 2.4 lần. Khi nào kiểm định không cặp là đúng?
14. Chứng minh rằng trong họ Gauss một chiều, q cực tiểu KL(p‖q) có trung bình và phương sai bằng đúng của p (moment matching). Áp dụng cho p = ½N(−2, 0.6²) + ½N(2, 0.6²). Vì sao cực tiểu KL(q‖p) lại cho một nghiệm hoàn toàn khác, và vì sao giá trị tối ưu của nó xấp xỉ ln 2?

## Bài tính nhanh

15 bài, mỗi bài tối đa 2 phút, làm bằng giấy bút và bấm giờ cả bộ. Mục tiêu: xong cả 15 bài trong 25 phút, sai không quá 1 bài. Làm lại bộ này sau 3 ngày và sau 1 tuần.

1. A, B độc lập, P(A) = 0.3, P(B) = 0.5. Tính P(A ∩ B) và P(A ∪ B).
2. P(B ∣ A) = 0.5, P(A) = 0.2, P(B) = 0.25. Tính P(A ∣ B).
3. Var X = 1, Var Y = 4, Cov(X, Y) = 1. Tính Var(2X − 3Y).
4. Với số liệu bài 3, hệ số tương quan của X và Y là bao nhiêu?
5. Bernoulli(p) có phương sai lớn nhất tại p nào, và giá trị lớn nhất là bao nhiêu?
6. Z ~ N(0, 1). P(Z > 2) xấp xỉ bao nhiêu?
7. Tính softmax của logit (0, ln 3).
8. Tính log-sum-exp của (1000, 1000).
9. Prior Beta(2, 2), tung 10 lần được 3 ngửa. Tính MAP của p.
10. Dữ liệu (1, 3, 5). Tính ước lượng phương sai theo MLE và theo công thức không chệch.
11. Model cho lớp đúng xác suất 0.6. Cross-entropy (nat) là bao nhiêu?
12. Entropy (bit) của phân phối đều trên 8 lớp, và perplexity tương ứng?
13. Cross-entropy trung bình theo token là 2.0 nat. Perplexity xấp xỉ bao nhiêu?
14. Độ chính xác 0.90 trên 400 câu test. Tính SE và CI 95% Wald.
15. 10 kiểm định độc lập ở α = 0.05, mọi H₀ đều đúng. Xác suất có ít nhất một dương tính giả? Ngưỡng Bonferroni?

## Lab code và bài săn bug

Năm lab dưới đây đã chạy ngày 04/11/2026 với Python 3, NumPy 2.5 và SciPy 1.18; số liệu có thể lệch nhẹ theo phiên bản. Mọi phần ngẫu nhiên dùng seed cố định. Cách học đúng: tự viết từng lab trước, chỉ đối chiếu code mẫu khi kết quả lệch với cột "kết quả mong đợi".

| Lab | Việc phải làm | Kết quả mong đợi |
| --- | --- | --- |
| 1. Bayes và tỉ lệ nền | Viết `posterior(prior, tpr, fpr)`; mô phỏng 10⁶ bình luận và đếm tỉ lệ độc trong nhóm bị gắn cờ | Công thức 0.2687; mô phỏng 0.2696; sau bộ lọc thứ hai 0.8686 |
| 2. Gauss nhiều chiều từ đầu | `mvn_logpdf` bằng Cholesky và `scipy.linalg.solve_triangular` (không dùng `inv`), `sample_mvn` theo x = μ + Lz, ước lượng lại Σ | logpdf khớp SciPy tới 10⁻¹⁰; Σ̂ từ 200 000 mẫu có hàng (2.0013, 1.0003), (1.0003, 2.0007); Mahalanobis² 0.6667 và 2.0 |
| 3. log-sum-exp và cross-entropy | `logsumexp`, `log_softmax`, `cross_entropy(logits, y, eps)` có label smoothing | Softmax ngây thơ ra nan, bản ổn định ra (0.665, 0.245, 0.090); CE 0.3567; có smoothing 0.4633; sàn H(mục tiêu) 0.2911 |
| 4. MLE, MAP, ridge | MLE/MAP Bernoulli với prior Beta; ridge giải đóng so với tối ưu −log posterior bằng `scipy.optimize.minimize` | 0.7/0.667 và 1.0/0.8; ŵ 2.2143 (MLE), 1.2917 (MAP); hai cách giải khớp 10⁻⁵; ‖MAP − MLE‖₂ = 1.738, 0.223, 0.0204, 0.0022 khi n = 10, 100, 1000, 10 000 (rng cục bộ, seed 40) |
| 5. So sánh hai model đúng cách | `wald_ci`, `mcnemar_exact`, `paired_bootstrap_ci`, `unpaired_bootstrap_ci`; mô phỏng tỉ lệ phủ của CI Wald | Wald [0.828, 0.872]; McNemar chính xác p = 0.0357; paired bootstrap [0.002, 0.028]; SE có cặp 0.0067, không cặp 0.0163; Wald ở n = 100, p = 0.85 chỉ phủ 93.30% thay vì 95% (cộng xác suất binomial chính xác: lệch phía trên 5.51%, phía dưới 1.19%; 200 000 lần mô phỏng cho 93.4%, 5.47%, 1.15%) |

Code nằm trong notebook `labs/chuong03/lab.ipynb` (bản có chỗ TODO để bạn tự viết, kèm ô kiểm tra tự động) và `labs/chuong03/loigiai.ipynb` (bản lời giải đã chạy kiểm). Mở từ trang **Lab** của app.

**Câu hỏi sau lab.**

- Lab 2: vì sao dùng Cholesky lại tốt hơn tính `inv(Sigma)` và `det(Sigma)`, cả về độ chính xác lẫn chi phí?
- Lab 4: khoảng cách MAP–MLE giảm khoảng 10 lần mỗi khi n tăng 10 lần. Giải thích tốc độ đó.
- Lab 5: tỉ lệ phủ thực của CI Wald 95% chỉ là 93.30%. Vì sao nó thấp hơn danh nghĩa, và nó thiếu phủ ở phía nào?

**Bài săn bug.** Hàm `compare_buggy` trong notebook chạy không báo lỗi, chỉ in ra một dict. Trên dữ liệu của ví dụ 3.11 (5% hàng có logit bị cộng thêm 1000), nó báo loss 0.26 cho A, CI của hiệu [−0.017, 0.047] (cận trên đổi nhẹ theo seed và thứ tự dữ liệu) và p-value giữa các seed 0.112: kết luận "hai model không khác nhau". Hai triệu chứng đáng ngờ: xáo trộn ngẫu nhiên vector nhãn y thì loss **không đổi**, và CI rất rộng dù hai model chỉ bất đồng ở 45/1000 câu. Tìm cả **bốn** lỗi trước khi xem lời giải. Lưu ý: ngay khi sửa đủ bốn lỗi, p giữa các seed vẫn là 0.056, lớn hơn 0.05.

```python
def compare_buggy(logits_A, logits_B, y, seeds_A, seeds_B, n_boot=10_000):
    out, correct = {}, {}
    for name, L in [("A", logits_A), ("B", logits_B)]:
        with np.errstate(over="ignore", invalid="ignore"):
            p = np.exp(L) / np.exp(L).sum(axis=1, keepdims=True)
            out["loss_" + name] = -np.nanmean(np.log(p.max(axis=1)))
        correct[name] = (L.argmax(axis=1) == y).astype(float)
        out["acc_" + name] = correct[name].mean()
    r = np.random.default_rng(0)
    i = r.integers(0, len(y), (n_boot, len(y)))
    j = r.integers(0, len(y), (n_boot, len(y)))
    d = correct["A"][i].mean(axis=1) - correct["B"][j].mean(axis=1)
    out["ci_diff"] = np.percentile(d, [2.5, 97.5])
    out["p_seeds"] = stats.ttest_ind(seeds_A, seeds_B).pvalue
    return out
```

## Bài dựng cảnh Manim

Mỗi chương có một bài tự dựng cảnh: muốn vẽ được một khái niệm, bạn phải hiểu từng bước của nó. Giới hạn **1 giờ** (tính vào luồng C), không chỉnh màu mè.

**Đề bài.** Dùng `manim/chuong03_scenes.py` làm mẫu, viết cảnh `BetaCapNhat` vào thư mục `manim/cua-toi/`:

1. Vẽ mật độ prior Beta(2, 2) trên [0, 1]. Dùng `scipy.stats.beta` để tính mật độ; nhãn trục viết bằng `Text`.
2. Cập nhật posterior sau từng lần tung theo dãy: 3 lần ngửa, rồi xen kẽ thêm 7 lần sao cho tổng cộng 7 ngửa trên 10 lần. Mỗi bước hiện Beta(a, b), MLE và MAP.
3. Sau lần tung thứ 3, vẽ hai vạch nét đứt tại MLE = 1 và MAP = 0.8, kèm một dòng giải thích.

**Tiêu chí tự chấm.** (1) Đúng toán: sau 3 ngửa là Beta(5, 2) với MAP 0.8; cuối cùng là Beta(9, 5) với MAP 8/12 ≈ 0.667, khớp ví dụ 3.7. (2) Có tiêu đề và phụ đề nói người xem phải thấy điều gì (MLE nhảy về 1, MAP thì không; posterior hẹp dần). (3) Dài không quá 30 giây, chữ không chồng nhau.

Render bằng nút **Render cảnh của tôi** ở trang Manim. Lời giải tham khảo nằm ở `manim/loigiai/chuong03_beta_cap_nhat.py` — chỉ mở sau khi đã tự làm.

## Phỏng vấn, bẫy, thực tế khác lý thuyết

**Cách luyện.** Mỗi chuỗi có 4 tầng câu hỏi, đào sâu dần như người phỏng vấn thật. Trả lời to bằng tiếng Việt trước, sau đó nói lại bằng tiếng Anh, 2 phút mỗi tầng, ghi âm và tự nghe lại. Câu trả lời mẫu tiếng Anh bên dưới chỉ phủ tầng 1–2; tầng 3–4 bạn tự soạn, rồi đối chiếu với lời giải các bài tập liên quan. Các câu hỏi là dạng phổ biến trong phỏng vấn ML, soạn lại từ nội dung chuẩn; không phải đề thật của công ty nào.

**Chuỗi 1 — Bayes và tỉ lệ nền**

1. Tính P(bệnh ∣ dương) với tỉ lệ 1%, độ nhạy 99%, FPR 5%.
2. Vì sao precision thay đổi theo tỉ lệ nền còn recall thì không?
3. Model deploy sang môi trường có tỉ lệ dương thấp hơn 10 lần. Precision thay đổi ra sao, và bạn hiệu chỉnh ngưỡng hay logit thế nào? (Bài tập 12.)
4. Naive Bayes sai giả định độc lập có điều kiện mà vẫn phân loại tốt. Vì sao? (Phân loại chỉ cần thứ hạng đúng; xác suất thì quá tự tin, thiếu hiệu chỉnh.)

> *Model answer:* By Bayes' rule, P(disease ∣ positive) = 0.99 × 0.01 / (0.99 × 0.01 + 0.05 × 0.99), which is exactly 1/6, so only about 17% of positives are real. The test looks accurate, but the disease is so rare that false positives from the healthy 99% outnumber the true positives. Recall is conditioned on the true label, so it does not depend on how common the positive class is. Precision is conditioned on the prediction, so it mixes in the base rate through the denominator, and it drops whenever positives become rarer.

**Chuỗi 2 — MLE và loss**

1. Vì sao MSE ứng với giả định nhiễu Gauss?
2. Vì sao cross-entropy ứng với MLE của phân phối categorical?
3. Nếu nhiễu là Laplace thì loss là gì? Nếu nhiễu có đuôi nặng thì sao?
4. Weight decay là MAP với prior nào? Với Adam, cộng phạt L2 vào loss có còn tương đương weight decay không? (Không; xem AdamW ở chương 4.)

> *Model answer:* If I model y = f(x; w) + ε with Gaussian noise of fixed variance, the negative log-likelihood of one example is the squared error divided by 2σ², plus a constant, so maximizing the likelihood is the same as minimizing the MSE. For classification, the network outputs logits, softmax turns them into a categorical distribution, and the negative log-probability of the true class is exactly the cross-entropy with a one-hot target. So both standard losses are just negative log-likelihoods under different output distributions. Changing the noise model changes the loss: Laplace noise gives the absolute error, which is more robust to outliers.

**Chuỗi 3 — Lý thuyết thông tin**

1. Định nghĩa entropy, cross-entropy và KL, và cho biết quan hệ giữa chúng.
2. Vì sao cực tiểu cross-entropy tương đương cực tiểu KL từ phân phối dữ liệu tới mô hình?
3. KL(p‖q) và KL(q‖p): cái nào mode-covering, cái nào mode-seeking? Variational inference dùng cái nào?
4. Perplexity liên hệ với cross-entropy thế nào? So sánh perplexity giữa hai mô hình dùng hai tokenizer khác nhau có công bằng không?

> *Model answer:* Entropy H(p) is the expected surprise, the expectation of −log p(x) under p. By Shannon's source coding theorem it is also the minimum achievable average code length per symbol, reached asymptotically with block coding. Cross-entropy H(p, q) is the (asymptotic) average code length when the data come from p but I encode with q, and the gap between the two is the KL divergence, so H(p, q) = H(p) + KL(p‖q). When p is the empirical data distribution, H(p) does not depend on the model parameters. Minimizing the cross-entropy loss is therefore the same as minimizing KL from the data to the model, and also the same as maximum likelihood.

**Chuỗi 4 — Thực nghiệm và ý nghĩa thống kê**

1. Model A hơn B 0.5 điểm trên tập test. Bạn có tin không?
2. Bạn kiểm tra bằng cách nào? (Paired bootstrap, McNemar, nhiều seed với kiểm định có cặp.)
3. p = 0.04 nghĩa là gì, và không nghĩa là gì?
4. Một nhóm chạy 20 cấu hình rồi báo cáo cấu hình tốt nhất trên test. Sai ở đâu, và nên làm thế nào?

> *Model answer:* Not without more information, because a 0.5-point gap on a few thousand examples is often within the sampling noise of the test set, and it can also be within seed-to-seed variance. Since both models are evaluated on the same examples, I would use a paired test: McNemar's test on the examples where they disagree, or a paired bootstrap that resamples the same indices for both systems, which also works for corpus-level metrics like BLEU. I would also train several seeds per model and compare them with a paired test. Then I would report the mean, the standard deviation and a confidence interval for the difference, not just two point estimates.

**Từ vựng nên nói trôi chảy:** base rate, posterior odds, conditional independence, covariance matrix, positive semi-definite, Mahalanobis distance, log-sum-exp trick, maximum likelihood, conjugate prior, negative log-likelihood, cross-entropy, KL divergence, mode-seeking, perplexity, standard error, confidence interval, paired bootstrap, McNemar's test, multiple comparisons.

**Bẫy thường gặp.**

- Nhầm P(A ∣ B) với P(B ∣ A), bỏ quên tỉ lệ nền: "recall 90%" không có nghĩa "90% cờ là đúng" (ví dụ 3.1 chỉ có 27%).
- Coi Cov = 0 là độc lập. Sai, trừ khi đồng thời Gauss; phản ví dụ Y = X².
- Coi mật độ là xác suất: PDF có thể lớn hơn 1, và log-likelihood liên tục có thể dương.
- Coi KL là khoảng cách: nó không đối xứng, không thỏa bất đẳng thức tam giác, và bằng ∞ khi q gán 0 cho chỗ p > 0.
- Đọc CI 95% là "95% khả năng tham số nằm trong khoảng này", và đọc p-value là "xác suất H₀ đúng". Cả hai đều sai.
- Dùng t-test không cặp hoặc bootstrap độc lập cho hai model chạy trên cùng tập test: SE bị thổi lên khoảng 2.4 lần trong ví dụ 3.11, và một hiệu thật biến thành "không có ý nghĩa".

**Thực tế khác lý thuyết.**

- Sách tính softmax rồi lấy log; code thật tính log-softmax trực tiếp bằng log-sum-exp và gộp vào hàm loss nhận logit. Tính riêng hai bước là nguồn nan và loss âm vô lý hay gặp nhất.
- Lý thuyết nói "chia n − 1 cho ước lượng không chệch", nhưng BatchNorm và LayerNorm chuẩn hóa bằng phương sai chia n. Riêng BatchNorm, `running_var` (dùng lúc inference) trong PyTorch được cập nhật bằng ước lượng không chệch chia n − 1 [cần kiểm chứng trên docs hiện hành; môi trường soạn sách không mở được pytorch.org], nên lúc train và lúc inference dùng hai quy ước khác nhau. Với n lớn chênh lệch không đáng kể; đây là câu hỏi mẹo, không phải lỗi.
- Label smoothing (Szegedy và cộng sự, 2016, mục 7, arXiv 1512.00567: ε = 0.1, top-1 và top-5 error trên ImageNet cùng giảm khoảng 0.2 điểm phần trăm tuyệt đối) làm mục tiêu không còn one-hot, nên loss train có sàn H(mục tiêu) > 0. Müller, Kornblith và Hinton (2019) cho thấy nó cải thiện calibration nhưng làm distillation kém hơn, vì nó xóa thông tin về độ giống nhau giữa các lớp.

## Mini-test 30 phút

Làm sau khi xong bài tập và lab, bấm giờ đúng 30 phút, không xem sách, không dùng máy tính. Tổng 30 điểm. **Từ 24 điểm:** sang chương 4. **18 đến dưới 24:** ôn lại mục ứng với câu sai rồi làm lại sau 3 ngày. **Dưới 18:** học lại chương.

1. **(4 điểm)** 10% email là spam. Từ "free" xuất hiện trong 60% email spam và 1% email thường. Một email chứa "free": tính P(spam ∣ free) bằng cả công thức Bayes và dạng odds.
2. **(4 điểm)** X đều trên {0, 1, 2, 3, 4} và Y = 3X − 1. Tính 𝔼X, Var X, 𝔼Y, Var Y, Cov(X, Y) và hệ số tương quan.
3. **(5 điểm)** x ~ N(0, Σ), Σ có hàng (1, 0.8) và (0.8, 1). Tính Mahalanobis² của (1, 1) và (1, −1); điểm nào bất thường hơn? Tìm phân phối của x₂ ∣ x₁ = 1.
4. **(5 điểm)** Với prior Beta(a, b) và h ngửa trên n lần tung, suy ra MAP của p. Với a, b nào thì MAP trùng MLE? Add-one smoothing (h + 1)/(n + 2) ứng với a, b nào nếu đọc theo MAP, và với a, b nào nếu đọc theo trung bình posterior?
5. **(6 điểm)** P = (0.5, 0.5), Q = (0.99, 0.01). Tính KL(P‖Q) và KL(Q‖P) (nat). Nếu Q = (1, 0) thì hai giá trị đó bằng bao nhiêu? Liên hệ với "mode-covering" và "mode-seeking".
6. **(6 điểm)** Hai model trên cùng 500 câu test: 12 câu chỉ A đúng, 4 câu chỉ B đúng, các câu còn lại hai model cho cùng kết quả. Thực hiện McNemar (bản χ² có hiệu chỉnh liên tục; biết P(χ² với 1 bậc tự do > 3.84) = 0.05). Kết luận gì? Bạn sẽ viết gì trong paper, và làm gì tiếp?

## Flashcard

18 thẻ của chương nằm trong `flashcards/chuong03.csv`. Ôn ở trang **Ôn tập** của app: app tự lên lịch theo hộp Leitner (thẻ nhớ được sẽ giãn lịch ôn 1 → 3 → 7 → 14 ngày, thẻ quên quay lại hộp 1). File CSV này cũng nhập thẳng được vào Anki nếu bạn thích dùng Anki.

## Lời giải

Chỉ đọc sau khi đã tự thử ít nhất 20 phút. Với bài chứng minh, so ý chính chứ không so từng chữ. Mọi đáp số bằng số trong chương đã được kiểm lại bằng NumPy/SciPy.

**Kiểm tra nhanh**

1. P(cờ) = 0.9 × 0.02 + 0.05 × 0.98 = 0.067, nên P(độc ∣ cờ) = 0.018/0.067 ≈ 0.27.
2. Cov(X, Y) = 𝔼[X³] − 𝔼X·𝔼[X²] = 0 − 0 = 0. Không độc lập: P(Y = 0 ∣ X = 0) = 1 trong khi P(Y = 0) = 1/3.
3. Mật độ không phải xác suất. Xác suất là tích phân của mật độ trên một khoảng, và tích phân trên cả trục vẫn bằng 1; một phân phối hẹp phải có mật độ cao để bù độ rộng nhỏ.
4. MLE = 3/3 = 1. MAP = (3 + 1)/(3 + 2) = 0.8.
5. Không. P = (0.5, 0.5), Q = (0.9, 0.1): KL(P‖Q) ≈ 0.511 nat, KL(Q‖P) ≈ 0.368 nat.

**Bài tập giấy bút**

1. "Ít nhất một con ra 6" có 11 kết quả trong 36; chỉ (6, 6) có cả hai ra 6. Xác suất 1/11 ≈ 0.091.
2. 𝔼X = 21/6 = 3.5. 𝔼[X²] = 91/6 ≈ 15.17. Var X = 91/6 − 49/4 = 35/12 ≈ 2.917.
3. Khai triển: 𝔼[(X − μ)²] = 𝔼[X²] − 2μ𝔼X + μ² = 𝔼[X²] − μ². Với Y = aX + b thì 𝔼Y = aμ + b, nên Y − 𝔼Y = a(X − μ) và Var Y = a²𝔼[(X − μ)²]. Hằng số b chỉ dịch phân phối, không đổi độ phân tán.
4. H = 0.5 × 1 + 0.25 × 2 + 0.25 × 2 = 1.5 bit. Perplexity = 2¹·⁵ ≈ 2.83. Mã 0, 10, 11 có độ dài trung bình 0.5 × 1 + 0.25 × 2 + 0.25 × 2 = 1.5 bit, đúng bằng entropy vì mọi xác suất đều là lũy thừa của ½.
5. (a) 0.0099/(0.0099 + 0.0495) = 1/6 ≈ 0.167. (b) Prior mới 1/6: 0.99 × (1/6)/(0.99 × (1/6) + 0.05 × (5/6)) ≈ 0.798. Dạng odds: odds ban đầu 1/99, tỉ số likelihood 19.8; hai lần dương cho odds 19.8²/99 = 3.96, tức 0.798. (c) Cần 0.99 × 0.01 = FPR × 0.99, tức FPR = 0.01. Muốn precision 50% ở tỉ lệ nền 1%, FPR phải giảm 5 lần.
6. (a) vᵀΣv = 𝔼[vᵀ(x − μ)(x − μ)ᵀv] = 𝔼[(vᵀ(x − μ))²] ≥ 0 với mọi v. (b) Var(∑ₖXₖ) = ∑ₖVar Xₖ + ∑ₖ≠ₗCov(Xₖ, Xₗ) = Kσ² + K(K − 1)ρσ². Chia cho K²: ρσ² + (1 − ρ)σ²/K.
7. log L = −(n/2)log(2πσ²) − ∑(xᵢ − μ)²/(2σ²). Đạo hàm theo μ: ∑(xᵢ − μ)/σ² = 0 ⇒ μ̂ = x̄. Đạo hàm theo σ²: −n/(2σ²) + ∑(xᵢ − x̄)²/(2σ⁴) = 0 ⇒ σ̂² = (1/n)∑(xᵢ − x̄)². Kỳ vọng: ∑(xᵢ − x̄)² = ∑(xᵢ − μ)² − n(x̄ − μ)². Kỳ vọng của số hạng đầu là nσ², của số hạng sau là n · σ²/n = σ². Vậy kỳ vọng của tổng là (n − 1)σ², chia n được (n − 1)σ²/n. Đó là cực đại: log L lõm theo μ (đạo hàm bậc hai −n/σ² < 0), còn theo σ² đạo hàm đổi dấu từ dương sang âm tại σ̂².
8. (a) −log N(y; f, σ²) = (y − f)²/(2σ²) + const ⇒ MSE. (b) −log[qʸ(1 − q)¹⁻ʸ] = −y log q − (1 − y)log(1 − q) ⇒ binary cross-entropy. (c) −log[(1/2b)exp(−|y − f|/b)] = |y − f|/b + const ⇒ MAE. Tổng |yᵢ − c| có đạo hàm (số yᵢ < c) − (số yᵢ > c), bằng 0 khi c là trung vị. MSE cho trung bình, MAE cho trung vị, nên MAE bền với ngoại lai.
9. (a) Trung bình (2/4) × 2 = 1, phương sai 3 − 2²/4 = 2: x₂ ∣ x₁ = 2 ~ N(1, 2). (b) det Σ = 8, Σ⁻¹ có hàng (3, −2)/8 và (−2, 4)/8. Mahalanobis² của (2, 0) = 3 × 4/8 = 1.5. (c) λ² − 7λ + 8 = 0 ⇒ λ = (7 ± √17)/2 ≈ 5.56 và 1.44. L có hàng (2, 0) và (1, √2); kiểm LLᵀ có hàng (4, 2) và (2, 1 + 2) = Σ.
10. KL(N(0, 1)‖N(1, 4)) = ln 2 + (1 + 1)/8 − 0.5 ≈ 0.443. KL(N(1, 4)‖N(0, 1)) = ln(1/2) + (4 + 1)/2 − 0.5 ≈ 1.307. Chiều thứ hai lấy trung bình theo N(1, 4), một phân phối rộng, và phạt mọi chỗ nó đặt khối lượng mà N(0, 1) gần bằng 0 (đuôi xa). Phân phối hẹp không "phủ" được phân phối rộng, nên KL lớn.
11. Dùng Jensen cho hàm lõm log: −KL(p‖q) = ∑ p log(q/p) ≤ log ∑ p · (q/p) = log ∑ q ≤ log 1 = 0 (tổng lấy trên tập p > 0). Dấu bằng của Jensen cần q/p là hằng số trên tập p > 0; kết hợp ∑q = 1 thì q = p. Với p̂ là phân phối thực nghiệm, H(p̂, q(·; θ)) = −(1/n)∑ᵢ log q(·; θ)(xᵢ) nên argmin cross-entropy = argmax log-likelihood. Và H(p̂, q(·; θ)) = H(p̂) + KL(p̂‖q(·; θ)) với H(p̂) không phụ thuộc θ, nên cũng là argmin KL. Giá trị loss nhỏ nhất là H(p̂), đạt khi q(·; θ) = p̂ (nếu họ mô hình chứa được p̂).
12. Odds posterior = tỉ số likelihood × odds prior. Ở môi trường huấn luyện, odds prior = 1, nên tỉ số likelihood = 0.8/0.2 = 4. Ở môi trường mới, odds prior = 0.05/0.95 = 1/19, nên odds posterior = 4/19 ≈ 0.211, tức p ≈ 0.174. Theo logit: logit mới = logit cũ + ln(odds prior mới/odds prior cũ) = ln 4 + ln(1/19) ≈ 1.386 − 2.944 = −1.558, sigmoid ra 0.174. Ngưỡng 0.5 trên điểm hiệu chỉnh tương đương ngưỡng 0.95 trên điểm cũ. Giả định P(x ∣ y) không đổi ("label shift") là then chốt; nếu chính P(x ∣ y) cũng đổi thì hiệu chỉnh này không đủ.
13. dᵢ = 1 với b câu, −1 với c câu, 0 với các câu còn lại. d̄ = (b − c)/n = 0.015, 𝔼[d²] = (b + c)/n = 0.045, nên Var d ≈ 0.045 − 0.015² = 0.044775 và SE = √(0.044775/1000) ≈ 0.0067. Không cặp: Var a + Var b = 0.1275 + 0.1378 = 0.2653, SE = √(0.2653/1000) ≈ 0.0163. Hiệu số là Var(a − b) = Var a + Var b − 2Cov(a, b); ở đây Cov(a, b) ≈ 0.110, triệt tiêu gần hết tổng phương sai. Tỉ số SE ≈ 0.0163/0.0067 ≈ 2.4. Kiểm định không cặp đúng khi hai model được đánh giá trên hai mẫu độc lập (hai tập test khác nhau), hoặc khi không ghép được các cặp seed.
14. KL(p‖q) = −H(p) − 𝔼ₚ[log q]. Với q = N(μ, σ²): −𝔼ₚ[log q] = ½log(2πσ²) + 𝔼ₚ[(x − μ)²]/(2σ²). Theo μ: 𝔼ₚ[(x − μ)²] nhỏ nhất tại μ = 𝔼ₚx. Thay vào, theo σ²: đạo hàm 1/(2σ²) − Varₚ/(2σ⁴) = 0 ⇒ σ² = Varₚ. Với hỗn hợp: trung bình 0, phương sai = 0.36 + 4 = 4.36 (phương sai trong từng thành phần cộng phương sai giữa các tâm), σ ≈ 2.088, khớp nghiệm số. KL(q‖p) = 𝔼_q[log q − log p] phạt q đặt khối lượng ở vùng giữa, nơi p ≈ 0, nên nghiệm tốt nhất đặt q lên đúng một đỉnh: q ≈ N(2.0, 0.60²) (hoặc đối xứng tại −2). Khi đó p ≈ ½ × (thành phần đó) trên vùng q sống, nên log q − log p ≈ log 2: KL ≈ ln 2 ≈ 0.693, tối ưu số cho 0.692. Bài học: cùng một họ xấp xỉ, hai chiều KL cho hai câu trả lời khác hẳn nhau.

**Bài tính nhanh**

1. 0.15; 0.3 + 0.5 − 0.15 = 0.65.
2. 0.5 × 0.2/0.25 = 0.4.
3. 4 × 1 + 9 × 4 − 2 × 2 × 3 × 1 = 28.
4. 1/(1 × 2) = 0.5.
5. p = 0.5, phương sai 0.25.
6. Khoảng 0.023 (95.4% nằm trong ±2σ, chia đôi phần còn lại).
7. exp: (1, 3), chia tổng 4: (0.25, 0.75).
8. 1000 + ln 2 ≈ 1000.693.
9. (3 + 1)/(10 + 2) = 1/3 ≈ 0.333.
10. Trung bình 3; tổng bình phương độ lệch 8. MLE 8/3 ≈ 2.667; không chệch 8/2 = 4.
11. −ln 0.6 ≈ 0.511 nat.
12. 3 bit; perplexity 8.
13. e² ≈ 7.39.
14. SE = √(0.9 × 0.1/400) = 0.015; CI ≈ [0.871, 0.929].
15. 1 − 0.95¹⁰ ≈ 0.401; ngưỡng 0.005.

**Câu hỏi sau lab**

- Lab 2: Cholesky tốn khoảng d³/3 phép tính, rẻ hơn nghịch đảo, và giữ được tính đối xứng xác định dương. Từ L ta có cả Δ² (thế tiến bằng `scipy.linalg.solve_triangular(L, x - mu, lower=True)`, chi phí d² và không tạo Σ⁻¹) lẫn log det Σ = 2∑log Lᵢᵢ, tránh được tràn dưới của det. Cholesky còn thất bại ngay khi Σ không xác định dương, nên nó là phép kiểm miễn phí.
- Lab 4: ŵ (MAP) = (XᵀX + λI)⁻¹Xᵀy và XᵀX tăng tuyến tính theo n (cỡ nI với đặc trưng chuẩn hóa), trong khi λ = 10 cố định. Hiệu ŵ (MAP) − ŵ (MLE) ≈ −λ(XᵀX)⁻¹ŵ (MLE), nên có độ lớn cỡ λ‖w‖₂/n: tăng n 10 lần thì khoảng cách giảm khoảng 10 lần.
- Lab 5: khoảng Wald dùng p̂ thay cho p thật trong SE, và giả định phân phối của p̂ đối xứng. Ở p = 0.85, n = 100, lỗi phủ không chia đều hai phía: tính chính xác bằng cách cộng pmf binomial (lab 5 in ra số này; 200 000 lần mô phỏng cho kết quả sát bên) cho thấy khoảng nằm trọn *phía trên* p thật 5.51% số lần, còn nằm trọn phía dưới chỉ 1.19%. Lý do: khi p̂ lớn hơn p, √(p̂(1 − p̂)) lại nhỏ hơn, nên khoảng vừa lệch phải vừa bị co hẹp đúng lúc cần rộng. Khoảng Wilson cho n = 85/100 là [0.767, 0.907], lệch xuống so với Wald [0.780, 0.920].

**Mini-test (thang 30 điểm)**

1. Bayes: 0.6 × 0.1/(0.6 × 0.1 + 0.01 × 0.9) = 0.06/0.069 ≈ 0.870 (2 điểm). Odds: prior odds 1/9, tỉ số likelihood 60, posterior odds 60/9 ≈ 6.67, xác suất 6.67/7.67 ≈ 0.870 (2 điểm).
2. 𝔼X = 2, Var X = (4 + 1 + 0 + 1 + 4)/5 = 2 (1 điểm). 𝔼Y = 5, Var Y = 9 × 2 = 18 (1 điểm). Cov(X, Y) = 3Var X = 6 (1 điểm). Tương quan = 6/(√2 × √18) = 1, vì Y là hàm tuyến tính tăng của X (1 điểm).
3. det Σ = 0.36, Σ⁻¹ có hàng (1, −0.8)/0.36 và (−0.8, 1)/0.36 (1 điểm). Mahalanobis² của (1, 1) = 0.4/0.36 ≈ 1.11; của (1, −1) = 3.6/0.36 = 10 (2 điểm). (1, −1) bất thường hơn nhiều, vì nó đi ngược tương quan 0.8 (1 điểm). x₂ ∣ x₁ = 1 ~ N(0.8, 1 − 0.64) = N(0.8, 0.36) (1 điểm).
4. log posterior = (h + a − 1)log p + (n − h + b − 1)log(1 − p) + const. Đạo hàm bằng 0 ⇒ p = (h + a − 1)/(n + a + b − 2) (3 điểm). a = b = 1 (prior đều) thì MAP = h/n = MLE (1 điểm). Add-one smoothing (h + 1)/(n + 2) ứng với a = b = 2 nếu đọc theo MAP, hoặc a = b = 1 nếu đọc theo trung bình posterior (h + a)/(n + a + b), tức luật Laplace gốc; chấp nhận một trong hai cách đọc, nêu cả hai thì tốt hơn nhưng không cộng thêm điểm (1 điểm).
5. KL(P‖Q) = 0.5 ln(0.5/0.99) + 0.5 ln(0.5/0.01) ≈ −0.342 + 1.956 = 1.614 nat (2 điểm). KL(Q‖P) = 0.99 ln 1.98 + 0.01 ln 0.02 ≈ 0.676 − 0.039 = 0.637 nat (2 điểm). Với Q = (1, 0): KL(P‖Q) = ∞ vì P đặt khối lượng ở chỗ Q bằng 0; KL(Q‖P) = 1 × ln 2 ≈ 0.693, hữu hạn (1 điểm). Cực tiểu KL(P‖Q) theo Q buộc Q phủ mọi chỗ P > 0 (mode-covering); KL(Q‖P) cho phép Q bỏ hẳn một phần của P (mode-seeking) (1 điểm).
6. χ² = (|12 − 4| − 1)²/16 = 49/16 ≈ 3.06 < 3.84, nên p ≈ 0.08 > 0.05 (bản chính xác p ≈ 0.077) (3 điểm). Kết luận: chưa đủ bằng chứng là hai model khác nhau; không có nghĩa chúng bằng nhau. Hiệu 8/500 = 1.6 điểm chỉ dựa trên 16 câu bất đồng (1 điểm). Trong paper: báo cáo hiệu kèm CI (paired bootstrap) và p-value, không viết "A tốt hơn B" (1 điểm). Làm tiếp: tăng tập test, chạy nhiều seed và so sánh có cặp; quyết định số câu và kiểm định trước khi nhìn kết quả (1 điểm).

**Săn bug — bốn lỗi**

1. **Softmax không trừ max.** Với 5% hàng có logit cỡ 1000, exp tràn thành inf và inf/inf cho nan. `np.nanmean` lặng lẽ bỏ các hàng đó, nên loss trông bình thường thay vì báo lỗi.
2. **Không lấy cột nhãn đúng.** `p.max(axis=1)` là độ tự tin của dự đoán, không phải xác suất của nhãn đúng. Vì vậy xáo trộn y không đổi loss, và loss báo 0.26 thay vì 0.62 (sai số gấp khoảng 2.4 lần).
3. **Bootstrap lấy hai bộ chỉ số độc lập `i` và `j`.** Cấu trúc cặp bị phá, phương sai của hiệu bị thổi lên: CI thành [−0.017, 0.047] thay vì [0.002, 0.028].
4. **`ttest_ind` cho dữ liệu có cặp.** 5 seed ghép cặp phải dùng `ttest_rel`: p = 0.056 thay vì 0.112 (bản gộp phương sai `equal_var=True`, mặc định của `ttest_ind`; Welch cho 0.130 như ví dụ 3.12; cả hai bỏ qua cặp). Lưu ý p = 0.056 vẫn lớn hơn 0.05: sửa lỗi 4 không tự đổi kết luận ở α = 0.05, nó chỉ cho thấy kiểm định cũ đã làm mất độ mạnh. Bằng chứng mạnh hơn nằm ở so sánh có cặp trên câu test (McNemar p = 0.036; CI paired bootstrap [0.002, 0.028]).

```python
def compare_fixed(logits_A, logits_B, y, seeds_A, seeds_B, n_boot=10_000):
    out, correct, n = {}, {}, len(y)
    for name, L in [("A", logits_A), ("B", logits_B)]:
        logp = L - special.logsumexp(L, axis=1, keepdims=True)    # sửa lỗi 1
        out["loss_" + name] = -logp[np.arange(n), y].mean()       # sửa lỗi 2
        correct[name] = (L.argmax(axis=1) == y).astype(float)
        out["acc_" + name] = correct[name].mean()
    r = np.random.default_rng(0)
    i = r.integers(0, n, (n_boot, n))
    d = correct["A"][i].mean(axis=1) - correct["B"][i].mean(axis=1)  # sửa lỗi 3: cùng chỉ số
    out["ci_diff"] = np.percentile(d, [2.5, 97.5])
    out["p_seeds"] = stats.ttest_rel(seeds_A, seeds_B).pvalue     # sửa lỗi 4
    return out
```

Đã chạy kiểm: bản sửa cho loss 0.620 (A) và 0.648 (B), khớp `scipy.stats.multinomial.logpmf` tới 10⁻⁸; CI của hiệu [0.002, 0.028]; p giữa các seed 0.056.

## Tài liệu đào sâu và nguồn

Nội dung chương được viết lại, đối chiếu phạm vi kiến thức với các nguồn dưới đây. Đọc theo thứ tự ưu tiên:

| Tài liệu | Dùng để làm gì | Phần nên đọc |
| --- | --- | --- |
| [CS229 Review of Probability Theory](https://cs229.stanford.edu/section/cs229-prob.pdf) (Stanford) | Ôn ngắn gọn toàn bộ nền xác suất của mục 3.1–3.3 | Toàn bộ; đặc biệt phần biến ngẫu nhiên nhiều chiều |
| [CS229 The Multivariate Gaussian Distribution](https://cs229.stanford.edu/section/gaussians.pdf) (Chuong B. Do) | Gauss nhiều chiều qua trị riêng của Σ | Phần hình dạng đường đồng mức và phép biến đổi tuyến tính |
| [CS229 Main Notes](https://cs229.stanford.edu/main_notes.pdf) | Suy ra loss từ likelihood | Diễn giải xác suất của bình phương tối thiểu (§1.3), logistic (§2.1), regularization và Bayes (§9.4) |
| [Deep Learning, chương 3](https://www.deeplearningbook.org/contents/prob.html) và [chương 5](https://www.deeplearningbook.org/contents/ml.html) (Goodfellow, Bengio, Courville) | Xác suất, thông tin và MLE theo góc nhìn deep learning | Ch.3: §3.9–3.13, hình KL không đối xứng; ch.5: §5.5 MLE, §5.6.1 MAP và weight decay |
| [Mathematics for Machine Learning](https://mml-book.github.io/) (Deisenroth, Faisal, Ong) | Bài tập thêm, chứng minh đầy đủ | Chương 6 (xác suất và phân phối), chương 8–9 (MLE, MAP, hồi quy Bayes) |
| [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) (C. Olah, 2015) | Trực giác bằng hình về entropy, cross-entropy, KL, MI | Toàn bộ, đọc trước nửa A của mục 3.5 |
| [MIT 18.05, Spring 2022](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/) | Thống kê suy diễn có bài tập | [Reading 23a](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class23-prep-a.pdf) (diễn giải CI), [Reading 24](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class24-prep.pdf) (bootstrap) |
| [Raschka (2018), Model Evaluation, Model Selection, and Algorithm Selection](https://arxiv.org/pdf/1811.12808) | Hướng dẫn thực hành so sánh model | §2.4 bootstrap CI, §4.3–4.4 McNemar, §4.11 5×2cv |
| [Koehn (2004), Statistical Significance Tests for MT Evaluation](https://aclanthology.org/W04-3250.pdf) | Paired bootstrap gốc trong NLP | Toàn bộ (ngắn) |
| [Dror và cộng sự (2018), Hitchhiker's Guide to Testing Statistical Significance in NLP](https://aclanthology.org/P18-1128/) | Chọn kiểm định phù hợp cho bài toán NLP | Phần quy trình chọn kiểm định và phần khảo sát |
| [Dietterich (1998), Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms](https://mlanthology.org/neco/1998/dietterich1998neco-approximate/) | Vì sao t-test trên nhiều lần chia dữ liệu sai lầm loại I cao | Kết luận và khuyến nghị McNemar, 5×2cv |
| [Bouthillier và cộng sự (2021), Accounting for Variance in Machine Learning Benchmarks](https://arxiv.org/abs/2103.03098v1) | Các nguồn phương sai trong benchmark và cách báo cáo | Phần phân tích nguồn phương sai và khuyến nghị |
| [Szegedy và cộng sự, Rethinking the Inception Architecture](https://arxiv.org/pdf/1512.00567) và [Müller, Kornblith, Hinton (2019), When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629v1) | Label smoothing: nguồn gốc và tác dụng phụ | §7 của Szegedy; phần calibration và distillation của Müller |
| [SciPy `logsumexp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html) và [SciPy `bootstrap`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) | API dùng trong lab | Tham số `axis`, `b`; `method`, `paired`, `n_resamples` |

**Từ khóa để tự tìm thêm:** base rate fallacy, conjugate prior, Gaussian conditioning, Wilson score interval, forward vs reverse KL, evidence lower bound (ELBO), label shift, Berg-Kirkpatrick và cộng sự (EMNLP 2012) về ý nghĩa thống kê trong NLP, multiple comparisons (Holm–Bonferroni), 5×2cv paired t-test.

**Trước khi sang chương 4:** đạt từ 24/30 điểm mini-test, làm lại các bài ★★★ sai, và tự viết được lab 3 (cross-entropy ổn định) và lab 5 (paired bootstrap) mà không nhìn code mẫu.

Nguồn tra cứu ngày 24/10/2026.
