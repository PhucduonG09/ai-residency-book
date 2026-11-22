# Chương 10 — Mô hình sinh và học tự giám sát

## Mục tiêu và kiểm tra nhanh

Chương này trả lời hai câu hỏi nằm sau gần như mọi mô hình NLP hiện đại: học phân phối p(x) của dữ liệu thế nào khi không tính được nó (VAE, GAN, diffusion), và lấy tín hiệu học từ đâu khi không có nhãn (contrastive, masked LM, causal LM). Phỏng vấn track Research hay hỏi dẫn xuất ELBO, vì sao GAN khó train, vì sao diffusion đoán nhiễu, vì sao BERT che 15%. Thời lượng: khoảng 2 tuần, 8 giờ mỗi tuần. Mọi thí nghiệm là đồ chơi 1–3 chiều, chạy một seed: chúng kiểm cơ chế, không chứng minh kết luận thực nghiệm (xem chương 11).

**Sau chương này, bạn phải làm được:**

- ☐ Dẫn xuất ELBO theo hai cách, tính KL Gauss chéo bằng tay, giải thích reparameterization và posterior collapse.
- ☐ Viết trò chơi minimax của GAN, tìm discriminator tối ưu, giải thích vì sao gradient bão hòa và vì sao mode collapse không hiện trên loss.
- ☐ Viết quá trình nhiễu hóa và khử nhiễu của DDPM, nói vì sao đoán nhiễu tương đương đoán score, tính classifier-free guidance và đánh đổi chất lượng–đa dạng.
- ☐ Viết InfoNCE, giải thích vai trò của nhiệt độ τ và cận thông tin tương hỗ, so masked LM với causal LM.
- ☐ Tính FID tay cho hai Gauss, nêu ba giới hạn của FID và vì sao không có thước đo duy nhất cho mô hình sinh.

**Kiểm tra nhanh (5 phút, không dùng máy tính).** Câu i kiểm tra mục 10.i: đúng thì đọc lướt mục đó, sai thì đọc kỹ. Đúng cả 5 câu thì làm thẳng bài ★★★ và mini-test.

1. Vì sao không tối đa trực tiếp log p(x) trong VAE? Số hạng nào của ELBO kéo q(z|x) về prior?
2. Với generator cố định, D*(x) bằng bao nhiêu tại điểm mà pdata(x) = pg(x)?
3. DDPM có ᾱₜ = 0.25. Với x₀ = 2 và ε = 1, tính xₜ.
4. Batch N = 256, mọi embedding bằng nhau. InfoNCE bằng bao nhiêu?
5. Hai tập mẫu lấy từ cùng một phân phối nhưng chỉ có vài trăm điểm. FID giữa chúng lớn hơn, nhỏ hơn hay bằng 0?

Đáp án ở mục Lời giải cuối chương.

## 10.1 Autoencoder, VAE, ELBO và reparameterization

VAE là autoencoder có thêm một phân phối trên mã z, nhờ đó lấy mẫu được. Cái giá là bạn tối ưu một cận dưới của log p(x) (ELBO) chứ không phải chính nó.

**Autoencoder (AE).** Bộ mã hóa đưa x về mã z có chiều nhỏ, bộ giải mã đưa z về x̂, hàm mất mát là sai số tái tạo. AE tuyến tính với loss bình phương học đúng không gian con của PCA (chương 1): trên dữ liệu cuộn bánh 3D của Lab 1, k = 2 cho sai số kiểm thử 0.33289 với PCA và 0.33288 với AE tuyến tính. AE phi tuyến (MLP) cùng k = 2 cho 0.0565. Thí nghiệm này không tách được phi tuyến khỏi độ sâu (MLP có cả hai); lý do nằm ở lý thuyết: AE tuyến tính sâu bao nhiêu tầng thì cả bộ mã hóa lẫn bộ giải mã vẫn là một ánh xạ tuyến tính, nên không thể vượt PCA. Cái làm AE vượt PCA là các hàm kích hoạt phi tuyến. Nhưng AE **không phải mô hình sinh**: không có gì ép các mã z của dữ liệu lấp đầy một vùng mà bạn biết cách lấy mẫu. Lấy z ngẫu nhiên đưa vào bộ giải mã của AE thường ra rác (nhận định chung, không đo ở đây).

**VAE.** Giả sử câu chuyện sinh: lấy z ~ p(z) = N(0, I), rồi x ~ p(x|z). Likelihood p(x) = ∫ p(z)p(x|z) dz không tính được. Ta thêm bộ mã hóa q(z|x) = N(μ(x), diag σ²(x)) làm phân phối xấp xỉ cho posterior p(z|x), một mạng dùng chung cho mọi x (amortized inference). Hai cách dẫn xuất ELBO nên biết cả hai:

1. *Jensen.* log p(x) = log 𝔼 theo q của [p(x, z)/q(z|x)] ≥ 𝔼 theo q của log[p(x, z)/q(z|x)], vì log lõm.
2. *Phân rã chính xác.* log p(x) không phụ thuộc z nên bằng 𝔼 theo q của log[p(x, z)/p(z|x)]. Nhân và chia cho q rồi tách:

$$
\log p(x) = \underbrace{\mathbb{E}_{q(z|x)}\big[\log p(x\mid z)\big] - \mathrm{KL}\big(q(z|x)\,\|\,p(z)\big)}_{\text{ELBO}} + \mathrm{KL}\big(q(z|x)\,\|\,p(z|x)\big)
$$

Cách 2 mạnh hơn: nó nói khe hở giữa ELBO và log p(x) **đúng bằng** KL từ q tới posterior thật. Tối đa ELBO theo q là kéo q về posterior; và VAE bị chặn bởi họ phân phối của q (Gauss chéo không khớp posterior nhiều đỉnh thì khe hở vẫn còn).

Hai số hạng của ELBO kéo ngược chiều nhau. *Tái tạo* buộc z mang thông tin về x. *KL* buộc q gần prior, tức mã "ngắn": đây là đánh đổi rate–distortion. β-VAE (Higgins và cộng sự, 2017) nhân KL với β: Lab 1 cho β = 0, 1, 30 lần lượt KL = 13.40, 4.719, 0.475 nat và sai số tái tạo 0.0424, 0.0471, 0.6018. Với q Gauss chéo và prior N(0, I), KL có dạng đóng, cộng theo từng chiều ẩn:

$$
\mathrm{KL}\big(\mathcal{N}(\mu, \operatorname{diag}\sigma^2)\,\|\,\mathcal{N}(0, I)\big) = \sum_{j} \tfrac12\big(\sigma_j^2 + \mu_j^2 - 1 - \ln \sigma_j^2\big)
$$

**Ví dụ giải mẫu 10.1 — KL Gauss chéo.** q có μ = (0.5, −1), σ = (0.8, 1.5).

- Chiều 1: ½(0.64 + 0.25 − 1 − ln 0.64) = 0.1681. Chiều 2: ½(2.25 + 1 − 1 − ln 2.25) = 0.7195. Tổng **0.8877**.
- Kiểm bằng Monte Carlo (2 triệu mẫu, trung bình log q − log p): 0.8866 trong Lab 1, lệch 0.1%.
- Mốc nhớ nhanh: μ = 0, σ = 1 cho 0; μ = 2, σ = 1 cho 2.0; μ = 0, σ = 0.5 cho 0.3181.

**Ví dụ giải mẫu 10.2 — ELBO tay một chiều.** p(z) = N(0, 1), p(x|z) = N(z, 1), quan sát x = 1, q = N(0.5, 0.8²).

- 𝔼 theo q của log p(x|z) = −½ ln 2π − ½((x − μ)² + σ²) = −0.9189 − ½(0.25 + 0.64) = −1.3639. KL(q‖p) = 0.1681 (cùng công thức, chiều 1 của ví dụ 10.1). ELBO = **−1.5321**.
- Bài này giải tích được: x ~ N(0, 2), nên log p(x) = −½ ln 4π − x²/4 = **−1.5155**. Khe hở 0.0166.
- Posterior thật là N(0.5, 0.5) (phương sai 0.5). KL(q‖posterior) = ½(0.64/0.5 − 1 − ln(0.64/0.5)) = 0.0166: **khớp khe hở**. Lấy q đúng bằng posterior thì ELBO = −1.5155 = log p(x).

% manim: ELBOvaKhoangHo

**Reparameterization.** Cần gradient của 𝔼 theo q của f(z) đối với tham số của q, trong khi lấy mẫu z ~ q là bước không khả vi. Viết z = μ + σ ⊙ ε với ε ~ N(0, I): ngẫu nhiên nằm ở ε (không phụ thuộc tham số), z là hàm khả vi của (μ, σ), nên gradient đi qua. Bộ ước lượng score-function (REINFORCE), 𝔼 của f(z)·∇ log q(z), cũng không chệch nhưng nhiễu hơn nhiều.

**Ví dụ giải mẫu 10.3 — cùng kỳ vọng, khác phương sai.** Mục tiêu 𝔼[z²], z ~ N(μ, σ²), μ = 0.5, σ = 0.8. Đáp án đúng: đạo hàm theo μ bằng 2μ = 1.0.

- Reparam: đạo hàm theo μ của (μ + σε)² là 2z. Trung bình 0.9983, độ lệch chuẩn mỗi mẫu **1.600** (đúng bằng 2σ).
- Score function: z²(z − μ)/σ². Trung bình 0.9972 nhưng độ lệch chuẩn **3.639**, gấp 2.3 lần, nên cần khoảng 5 lần số mẫu cho cùng độ chính xác (số đo trong Lab 1, 2 triệu mẫu).

% manim: TaiThamSoHoa

**Posterior collapse.** Nếu bộ giải mã đủ mạnh để tự mô hình hóa x mà bỏ qua z, ELBO tối ưu có q(z|x) = p(z), tức KL = 0 và z không mang thông tin (bài tập 12). Với VAE văn bản có decoder LSTM đây là hiện tượng kinh điển (Bowman và cộng sự, 2016): decoder học cách bỏ qua z. Cách chữa: KL annealing (tăng β từ 0 lên 1), word dropout (làm yếu decoder), free bits (không phạt KL dưới một ngưỡng mỗi chiều). Đừng lẫn với một hiện tượng họ hàng nhưng khác cơ chế: *chiều không hoạt động* (inactive units). Trong Lab 1, với k = 6 chiều ẩn, KL từng chiều là (1.932, 1.675, 1.721, 0.001, 0.001, 0.001): ba chiều có KL gần 0. Đây không phải posterior collapse theo nghĩa Bowman: dữ liệu của lab chỉ có 3 chiều và decoder là một MLP nhỏ, không mạnh. Mỗi chiều ẩn dư tốn KL mà không giảm sai số tái tạo, nên mô hình tỉa chúng về prior. Phân biệt bằng câu hỏi "decoder có tự mô hình hóa được x mà không cần z không": collapse là có (toàn bộ z bị bỏ), chiều không hoạt động là không (z vẫn được dùng, chỉ thừa vài chiều).

**Vì sao ML cần cái này.**

- **ELBO là khung chung.** Diffusion (mục 10.3) cũng được dẫn xuất như ELBO của một chuỗi biến ẩn; hiểu khe hở "KL tới posterior" là hiểu vì sao các mô hình này cho cận chứ không cho likelihood chính xác.
- **Reparameterization là kỹ thuật chuẩn** để cho gradient đi qua lấy mẫu liên tục. Với biến rời rạc (token) nó không áp dụng trực tiếp: đây là một lý do sinh văn bản khó hơn sinh ảnh với các mô hình Gauss.
- **Posterior collapse** là chỗ phỏng vấn NLP hay chạm tới, vì mô hình ngôn ngữ tự hồi quy mạnh đến mức bỏ qua mọi biến ẩn.

**Thực tế tính toán.**

- Cho mạng xuất `logvar` thay vì σ để σ² = exp(logvar) luôn dương, và nhớ σ = exp(0.5 · logvar). Viết `mu + logvar.exp() * eps` chạy được nhưng sai.
- KL **cộng** theo chiều ẩn. Lấy trung bình thay vì cộng tương đương β = 1/k.
- `torch.normal(mu, std)` chạy không lỗi nhưng gradient theo `mu` và `std` bằng 0 (Lab 1 in ra 0.0). Phải viết `mu + std * torch.randn_like(mu)` hoặc dùng `Normal.rsample()`.
- Với dữ liệu liên tục, −log p(x|z) có thể âm (mật độ lớn hơn 1): Lab 1 cho −0.928. Phương sai cố định của decoder (0.1 trong lab) quy định tỉ lệ giữa tái tạo và KL, nên cũng là một "β ẩn".

## 10.2 GAN: trò chơi minimax, discriminator tối ưu, mode collapse

GAN bỏ hẳn việc viết ra p(x): một generator G(z) tạo mẫu, một discriminator D(x) ∈ (0, 1) phân biệt thật với giả, hai mạng chơi một trò chơi đối kháng. Lý thuyết đẹp (cực tiểu là phân phối dữ liệu), thực hành khó (gradient biến mất, mode collapse).

Ký hiệu: pdata là mật độ dữ liệu, pg là mật độ của mẫu G(z) (viết liền, vì Unicode không có chỉ số dưới cho các chữ này). Goodfellow và cộng sự (2014):

$$
\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}\big[\log D(x)\big] + \mathbb{E}_{z \sim p(z)}\big[\log\big(1 - D(G(z))\big)\big]
$$

**Discriminator tối ưu.** Với G cố định, mỗi điểm x cực đại hóa riêng pdata(x)·log D + pg(x)·log(1 − D), cho D*(x) = pdata(x)/(pdata(x) + pg(x)). Thay lại:

$$
D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}, \qquad \max_D V(D, G) = -\ln 4 + 2\,\mathrm{JSD}(p_{\text{data}} \,\|\, p_g)
$$

Vậy GAN lý tưởng tối thiểu hóa khoảng cách Jensen–Shannon (JSD là trung bình hai KL tới mixture M = (P + Q)/2), cực tiểu −ln 4 đạt khi pg = pdata, lúc đó D* ≡ ½.

**Ví dụ giải mẫu 10.4 — D* giữa hai Gauss.** pdata = N(0, 1), pg = N(2, 1).

- Tỉ số pg/pdata = exp(2x − 2), nên D*(x) = 1/(1 + exp(2x − 2)) = σ(2 − 2x) (σ là sigmoid).
- x = −1, 0, 1, 2, 3 cho D* = 0.9820, 0.8808, 0.5000, 0.1192, 0.0180. Kiểm x = 0: σ(2) = 0.8808.
- JSD = 0.3368 (tích phân số), nên max V = 2 · 0.3368 − ln 4 = **−0.7126**. Ở cân bằng pg = pdata: D* ≡ ½, V = −ln 4 = −1.3863, và loss BCE của D bằng 2 ln 2 = 1.3863.

% manim: DiscriminatorToiUu

**Vì sao khó huấn luyện.** Arjovsky và Bottou (2017) chỉ ra hai điều có định lý. (a) Khi pdata và pg nằm trên các đa tạp thấp chiều không khớp nhau (ảnh, văn bản embedding), tồn tại discriminator hoàn hảo, JSD là hằng số log 2 và gradient trên giá đỡ bằng 0: generator không nhận được hướng đi. (b) Với D gần tối ưu, gradient của log(1 − D(G(z))) đối với generator tiến về 0. Goodfellow đã đề xuất mẹo *non-saturating*: G cực đại hóa log D(G(z)) thay vì cực tiểu hóa log(1 − D(G(z))). Cùng điểm cố định, gradient mạnh hơn lúc đầu. Arjovsky và Bottou chỉ ra mẹo này có vấn đề riêng (ước lượng gradient nhiễu). Nói đúng: giải pháp không hoàn hảo.

**Ví dụ giải mẫu 10.5 — JSD bão hòa, gradient bão hòa.**

- Dịch tâm pg = N(d, 1): JSD = 0.0303 (d = 0.5), 0.3368 (d = 2), 0.6759 (d = 5), **0.6931 = ln 2** (d = 10), trong khi KL(pdata‖pg) = d²/2 = 0.125, 2, 12.5, 50 tăng không giới hạn. JSD "không còn thấy" khoảng cách khi hai phân phối rời nhau.
- Gradient theo logit của D khi D(G(z)) = D: loss bão hòa log(1 − D) cho |gradient| = D; loss non-saturating −log D cho 1 − D. Tại D = 0.01: 0.01 so với 0.99. Tại D = 0.5: cùng 0.5 (Lab 2 tính bằng autograd).

**Mode collapse.** Generator ánh xạ nhiều z về cùng một vùng x (Goodfellow gọi là "Helvetica scenario"). Lab 2 huấn luyện GAN non-saturating trên hỗn hợp N(−3, 0.3²) + N(3, 0.3²), ba seed, đếm mẫu nằm trong khoảng ±1 quanh mỗi mode:

| Seed | Gần −3 | Gần +3 | Loss D cuối |
| --- | --- | --- | --- |
| 0 | 0.985 | 0.000 | 0.935 |
| 1 | 0.452 | 0.501 | 1.377 |
| 2 | 0.491 | 0.496 | 1.385 |

Cùng kiến trúc và siêu tham số, seed 0 chỉ giữ một mode; seed 1 và 2 phủ cả hai (cân bằng lý thuyết của loss D là 1.386). Với ba seed bạn không rút ra được tần suất collapse, chỉ thấy nó phụ thuộc seed. Loss D của seed 0 thấp hơn hẳn 1.386, nghĩa là D đang thắng; nhưng tín hiệu này không cho biết G thiếu mode nào, cũng không phân biệt collapse với những kiểu thất bại khác. Muốn biết phải nhìn mẫu và đếm mode.

**Vị trí hiện nay.** Dhariwal và Nichol (2021) đặt tên bài báo "Diffusion Models Beat GANs on Image Synthesis": với ảnh, diffusion thay GAN vì train ổn định hơn và phủ mode tốt hơn. Với văn bản GAN gần như không dùng, vì token rời rạc làm gradient không đi qua generator (suy luận kỹ thuật, không đo ở đây). WGAN, gradient penalty, spectral normalization, StyleGAN là các từ khóa để tự tìm (cuối chương).

**Vì sao ML cần cái này.**

- **Hiểu khi nào một "loss học được" có gradient:** tín hiệu từ một bộ phân biệt chỉ hữu ích khi hai phân phối còn chồng lấn; bài học áp dụng cho mọi thành phần học được đóng vai trò hàm mất mát.
- **Nền để đọc tài liệu sinh ảnh/âm thanh** và để trả lời "vì sao diffusion thay GAN" trong phỏng vấn.
- **Mẫu điển hình của "lý thuyết đẹp, thực hành khó":** bài cân bằng ở đỉnh yên ngựa thay vì giảm một loss.

**Thực tế tính toán.** Dùng `F.binary_cross_entropy_with_logits` (không sigmoid rồi BCE). Khi cập nhật D phải `detach()` mẫu giả. Cập nhật D và G ở hai lượt forward riêng. Theo dõi mẫu và số mode, không chỉ theo dõi loss.

## 10.3 Diffusion và score matching: DDPM, lịch nhiễu, classifier-free guidance

Diffusion học cách đảo ngược một quá trình làm hỏng dữ liệu dần dần bằng nhiễu Gauss. Mạng chỉ làm một việc đơn giản và có thể huấn luyện bằng hồi quy: đoán nhiễu đã cộng vào.

**Quá trình tiến.** Mỗi bước thu nhỏ tín hiệu một chút và cộng nhiễu: q(xₜ|xₜ₋₁) = N(√(1 − βₜ) xₜ₋₁, βₜ I). Đặt αₜ = 1 − βₜ và ᾱₜ = ∏ₛ≤ₜ αₛ. Vì tổng các Gauss là Gauss, ta nhảy thẳng tới bước t được (Ho và cộng sự, 2020):

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)
$$

DDPM dùng T = 1000 và βₜ tăng tuyến tính từ 10⁻⁴ đến 0.02, nên ᾱ ở bước cuối gần 0 và x ở bước cuối gần như N(0, I).

**Huấn luyện và lấy mẫu.** Chọn t đều trên 1..T, nhiễu ε, tạo xₜ như trên, mạng ε̂(xₜ, t) đoán ε:

$$
L_{\text{simple}} = \mathbb{E}_{x_0,\,t,\,\varepsilon}\,\big\lVert \varepsilon - \hat\varepsilon(x_t, t)\big\rVert^2, \qquad x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\hat\varepsilon(x_t, t)\Big) + \sigma_t z
$$

với z ~ N(0, I) ở mọi bước trừ bước cuối (z = 0). Lý do dạng này: posterior thật q(xₜ₋₁|xₜ, x₀) là Gauss có trung bình đúng bằng vế đầu của công thức lấy mẫu khi ε̂ được thay bằng ε thật, nên khớp trung bình quy về hồi quy trên ε. Phương sai σₜ² lấy bằng βₜ hoặc β̃ₜ = (1 − ᾱₜ₋₁)/(1 − ᾱₜ)·βₜ; hai cách cho kết quả gần nhau (Lab 3 dùng β̃ₜ). Chú ý: hàm mất mát đơn giản hóa (L simple) là ELBO có trọng số khác, không phải ELBO (xem "Thực tế khác lý thuyết").

**Liên hệ với score.** Score của một mật độ là ∇ₓ log q(x), vector chỉ hướng tăng mật độ nhanh nhất. Bắt đầu từ q(xₜ|x₀) ở trên. Score **có điều kiện** theo x₀ là:

$$
\nabla_{x_t}\log q(x_t \mid x_0) = -\frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{1-\bar\alpha_t} = -\frac{\varepsilon}{\sqrt{1-\bar\alpha_t}}
$$

Đến đây chưa được kết luận "mạng đoán nhiễu là mạng đoán score". Vế phải phụ thuộc ε, tức phụ thuộc x₀ cụ thể mà lúc lấy mẫu ta không có; thứ bộ lấy mẫu cần là score của **phân phối biên** của dữ liệu đã thêm nhiễu, qₜ(xₜ) = ∫ q(x₀) q(xₜ|x₀) dx₀. Bước nối hai thứ này là ý của denoising score matching (Vincent, 2011), gồm hai nửa.

*Nửa 1: score biên là trung bình của score có điều kiện.* Đạo hàm dưới dấu tích phân rồi viết ∇q = q·∇log q:

$$
\nabla_{x_t}\log q_t(x_t) = \frac{1}{q_t(x_t)}\int q(x_0)\,\nabla_{x_t} q(x_t \mid x_0)\,dx_0 = \mathbb{E}_{q(x_0 \mid x_t)}\Big[\nabla_{x_t}\log q(x_t \mid x_0)\Big] = -\frac{\mathbb{E}\,[\varepsilon \mid x_t]}{\sqrt{1-\bar\alpha_t}}
$$

*Nửa 2: nghiệm tối ưu của hồi quy nhiễu là kỳ vọng có điều kiện.* Mạng chỉ nhận (xₜ, t). Với bình phương sai số, hàm tốt nhất theo xₜ là ε̂*(xₜ, t) = 𝔼[ε|xₜ]: với mọi hàm f, 𝔼‖ε − f(xₜ)‖² = 𝔼‖ε − 𝔼[ε|xₜ]‖² + 𝔼‖𝔼[ε|xₜ] − f(xₜ)‖², số hạng đầu không phụ thuộc f. Ghép hai nửa:

$$
\hat\varepsilon^{*}(x_t, t) = \mathbb{E}[\varepsilon \mid x_t] = -\sqrt{1-\bar\alpha_t}\;\nabla_{x_t}\log q_t(x_t)
$$

Vậy mạng đoán nhiễu, khi huấn luyện đủ tốt, ước lượng score của phân phối biên có nhiễu (cùng hệ số −√(1 − ᾱₜ)), chứ không phải score có điều kiện. Kiểm bằng trường hợp Gauss: nếu x₀ ~ N(0, 1) thì xₜ ~ N(0, 1) với mọi t (vì ᾱ + (1 − ᾱ) = 1), score biên là −xₜ; mặt khác ε và xₜ đồng Gauss với hiệp phương sai √(1 − ᾱₜ), nên 𝔼[ε|xₜ] = √(1 − ᾱₜ)·xₜ, đúng bằng −√(1 − ᾱₜ)·(−xₜ). Bài tập 11 làm đầy đủ phép chứng minh. Kết luận này là nền cho các tên gọi score-based model (nhiều mức nhiễu và Langevin, Song và Ermon 2019) và cho việc coi DDPM là rời rạc hóa của một SDE (Song và cộng sự 2021). Trực giác: dữ liệu thật nằm trên đa tạp thấp chiều nên score ở chỗ trống không xác định; thêm nhiễu làm mật độ có mặt khắp nơi, và khử nhiễu từng nấc là đi theo score từ vùng nhiễu về vùng dữ liệu.

**Lịch nhiễu.** Mức nhiễu tương đối được đo bằng SNR = ᾱₜ/(1 − ᾱₜ).

**Ví dụ giải mẫu 10.6 — lịch tuyến tính, T = 1000.**

| t | ᾱₜ | √ᾱₜ | √(1 − ᾱₜ) | SNR |
| --- | --- | --- | --- | --- |
| 1 | 0.9999 | 0.99995 | 0.0100 | 9999 |
| 100 | 0.8970 | 0.9471 | 0.3209 | 8.71 |
| 250 | 0.5241 | 0.7239 | 0.6899 | 1.10 |
| 500 | 0.07859 | 0.2803 | 0.9599 | 0.0853 |
| 750 | 0.003351 | 0.0579 | 0.9983 | 0.00336 |
| 1000 | 4.04 × 10⁻⁵ | 0.00635 | 0.99998 | 4.04 × 10⁻⁵ |

Lịch cosine của Nichol và Dhariwal (2021) đặt ᾱₜ = f(t)/f(0) với f(t) = cos²(((t/T + s)/(1 + s))·π/2), s = 0.008. Bước đầu tiên có ᾱ < 0.5 là t = 260 với lịch tuyến tính, t = 497 với cosine; số bước có SNR < 0.01 là 326 so với 64. Nói cách khác, một phần ba lịch tuyến tính gần như là nhiễu thuần: lãng phí chứ không làm hỏng (DDPM với lịch tuyến tính vẫn cho mẫu tốt).

% manim: LichNhieuKhuechTan

**Ví dụ giải mẫu 10.7 — một bước tiến, một bước lùi, và score (t = 500).** x₀ = 2, ε = 0.5, ᾱ₅₀₀ = 0.078587.

- x₅₀₀ = 0.2803 · 2 + 0.9599 · 0.5 = 0.5607 + 0.4800 = **1.0406**. Nếu ε đoán đúng, x̂₀ = (xₜ − √(1 − ᾱ)·ε)/√ᾱ = 2.0000.
- Score có điều kiện ∇ log q(xₜ|x₀) = −ε/√(1 − ᾱ) = −0.5/0.9599 = **−0.5209** (đây là score theo x₀ đã cho, không phải score biên mà mạng học). Dấu âm: score đẩy xₜ = 1.04 xuống, về trung bình có điều kiện √ᾱ·x₀ = 0.2803 · 2 = 0.56 của q(xₜ|x₀) (kiểm: −(1.0406 − 0.5607)/0.9214 = −0.5209), ngược hướng với x₀ = 2 nằm phía trên xₜ. Tín hiệu x₀ chỉ khôi phục sau khi chia cho √ᾱ (x̂₀ = 2.0000), nên bộ lấy mẫu phải phóng đại lại bằng 1/√αₜ qua từng bước.
- Với β₅₀₀ = 0.01004, posterior q(x₄₉₉|x₅₀₀, x₀) có trung bình 1.04063 và độ lệch chuẩn 0.1002; dạng theo ε, (xₜ − β/√(1 − ᾱ)·ε)/√α, cho cùng 1.04063.

**Classifier-free guidance (CFG).** Muốn sinh có điều kiện nhãn c. Huấn luyện một mạng duy nhất, nhưng với xác suất p (0.1 trong lab) thay nhãn bằng token rỗng ∅, nên mạng học cả ε̂(x, c) lẫn ε̂(x, ∅). Lúc lấy mẫu dùng (Ho và Salimans, 2021):

$$
\tilde\varepsilon = (1 + w)\,\hat\varepsilon(x, c) - w\,\hat\varepsilon(x, \varnothing) = \hat\varepsilon(x, \varnothing) + (1 + w)\big(\hat\varepsilon(x, c) - \hat\varepsilon(x, \varnothing)\big)
$$

Hiểu: đi từ dự đoán không điều kiện về phía dự đoán có điều kiện, rồi ngoại suy thêm w lần. Vì p(c|x) ∝ p(x|c)/p(x), hiệu hai score chính là ∇ log p(c|x): CFG có thể diễn giải như cộng gradient của một bộ phân loại ngầm (implicit classifier), không cần huấn luyện bộ phân loại riêng (classifier guidance của Dhariwal và Nichol dùng bộ phân loại riêng). Giới hạn, theo chính Ho và Salimans: hai score đến từ mạng không bị ràng buộc nên biểu thức không nhất thiết là gradient của một bộ phân loại thật; đó là cách hiểu, không phải đẳng thức. Hai quy ước hay gặp: εc + w(εc − εu) chính là dạng trên (bài tập 9 chứng minh); dạng εu + s(εc − εu), cách viết của nhiều thư viện, dùng s = 1 + w (s = 0 là không điều kiện, s = 1 là có điều kiện thuần, ứng với w = 0). Đọc code hay bài báo khác thì kiểm xem w hay s đang được dùng.

**Ví dụ giải mẫu 10.8 — CFG bằng số.** ε̂(x, c) = 0.50, ε̂(x, ∅) = 0.20. Khi đó ε̃ = (1 + w)·0.50 − w·0.20, với w = 0, 1, 3, 7.5 cho **0.50, 0.80, 1.40, 2.75**.

Lab 3 huấn luyện DDPM (T = 100, 3000 bước) trên hai mặt trăng có nhãn, lớp 0 là nửa vành trên tâm (−1, −0.5) bán kính 2. Mỗi dòng dùng 2000 mẫu mỗi lớp; Fréchet (FID 2D, tính trực tiếp trên tọa độ) so toàn bộ mẫu với dữ liệu thử; "hướng tâm" là sai lệch trung bình của khoảng cách tới tâm so với 2 (nhỏ hơn là mẫu nằm sát vành hơn):

| w | Fréchet | Hướng tâm lớp 0 | Độ lệch chuẩn (x, y) của lớp 0 |
| --- | --- | --- | --- |
| 0 | 0.0303 | 0.1253 | (1.338, 0.622) |
| 1 | 0.1096 | 0.0891 | (1.115, 0.507) |
| 3 | 0.2807 | 0.0653 | (0.853, 0.386) |
| 7 | 0.6527 | 0.0492 | (0.550, 0.212) |

Dữ liệu thật của lớp 0: hướng tâm 0.130 (do nhiễu 0.08 trong dữ liệu), độ lệch chuẩn (1.42, 0.62). Tăng w làm mẫu sát vành hơn (mỗi mẫu "đẹp" hơn) nhưng co lại về giữa vành và Fréchet tăng: đa dạng giảm. Đây là đánh đổi chất lượng–đa dạng. Ho và Salimans (2021) đo nó cho CFG dưới dạng đánh đổi giữa FID và Inception Score khi tăng w; Dhariwal và Nichol (2021) đo đánh đổi precision–recall trên ImageNet, nhưng cho *classifier guidance* (bộ phân loại riêng), không phải CFG. Bảng của lab chỉ là bản đồ chơi của cùng hiện tượng.

% manim: HuongDanKhongPhanLoai

**Vì sao ML cần cái này.**

- **Diffusion là nền của sinh ảnh/âm thanh hiện nay**, và hai ý "đoán nhiễu" cùng "CFG" xuất hiện trong mọi pipeline text-to-image.
- **Sinh văn bản rời rạc không chạy trực tiếp với nhiễu Gauss.** Hai hướng nghiên cứu: diffusion trên embedding liên tục (Diffusion-LM) và diffusion trên trạng thái rời rạc (D3PM). Dùng làm từ khóa.
- **Chi phí lấy mẫu:** T lần chạy mạng (gấp đôi khi dùng CFG). DDIM, distillation, latent diffusion là các hướng giảm chi phí.

**Thực tế tính toán.** Lập bảng `abar` một lần rồi lấy chỉ số theo t, và kiểm chỉ số bắt đầu từ 0 hay 1 (`abar[t − 1]` với t = 0 quấn về phần tử cuối trong Python). Lấy mẫu với CFG gọi mạng hai lần mỗi bước.

## 10.4 Học tự giám sát: contrastive, InfoNCE, masked modeling, BERT và GPT

Học tự giám sát (self-supervised learning, SSL) tự tạo nhãn từ chính dữ liệu: che một phần rồi đoán lại, đoán phần tiếp theo, hoặc nhận ra hai góc nhìn của cùng một mẫu. Nhãn miễn phí nên dùng được lượng văn bản không nhãn gần như vô hạn, rồi tinh chỉnh cho tác vụ thật. Đây là nền của BERT, GPT và mọi LLM (chương 9).

**Contrastive và InfoNCE.** Với mỗi truy vấn có một khóa dương và N − 1 khóa âm, hàm mất mát là bài toán phân loại N lớp "khóa nào khớp với truy vấn". CPC (van den Oord và cộng sự, 2018) đưa ra cận dưới thông tin tương hỗ cho nó; phép dẫn gốc dùng một bước xấp xỉ, còn chứng minh chặt thấy trong Poole và cộng sự (2019) [cần kiểm chứng]. Dạng cài đặt hay dùng là cosine chia nhiệt độ τ (SimCLR gọi là NT-Xent):

$$
\ell_i = -\log \frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{N}\exp(s_{ij}/\tau)}, \quad s_{ij} = \cos(a_i, b_j), \qquad I(x; c) \ge \ln N - L_N
$$

Cài đặt: `F.cross_entropy(a_norm @ b_norm.T / tau, arange(N))`, nhãn là đường chéo. Hai hệ quả: embedding sụp đổ (mọi vector bằng nhau) cho loss đúng bằng ln N (4.8520 với N = 128); và cận thông tin không bao giờ vượt ln N, nên muốn "thấy" I nat thông tin cần N ≥ eᴵ mẫu. SimCLR đếm 2(N − 1) mẫu âm trong batch 2N góc nhìn, CPC đếm N − 1.

**Ví dụ giải mẫu 10.9 — InfoNCE trên ma trận tương đồng 4×4.** Hàng là truy vấn, cột là khóa, cặp dương trên đường chéo:

S = [[0.9, 0.1, 0.0, 0.2], [0.2, 0.8, 0.3, 0.1], [0.0, 0.2, 0.7, 0.4], [0.1, 0.1, 0.3, 0.6]]

- τ = 1.0: loss từng hàng (0.8555, 0.9753, 1.0452, 1.0831), trung bình **0.9898**; xác suất đúng (0.4251, 0.3771, 0.3516, 0.3385).
- τ = 0.5: trung bình **0.6816**. τ = 0.1: **0.0322** (xác suất đúng 0.9986, 0.99, 0.9457, 0.9405).
- Mọi tương đồng bằng nhau thì loss = ln 4 = 1.3863. Cận MI với τ = 0.5: I ≥ ln 4 − 0.6816 = 0.7047 nat.
- τ quyết định mức loss ngay cả khi embedding hoàn hảo: ma trận đơn vị 128×128 (cos = 1 trên đường chéo, 0 ngoài) cho loss 0.0057 ở τ = 0.1 nhưng 2.9007 ở τ = 0.5.

% manim: MaTranInfoNCE

τ nhỏ làm softmax nhọn, phạt nặng các mẫu âm "khó"; τ lớn làm loss tiến về ln N, hết tín hiệu. Trong Lab 4 (dữ liệu 24 chiều, hai góc nhìn là hai lần cộng nhiễu, 600 bước):

| τ | Loss bước đầu | Retrieval@1 | Linear probe |
| --- | --- | --- | --- |
| 0.05 | 1.960 | 0.426 | 0.644 |
| 0.5 | 4.258 | 0.343 | 0.965 |

Retrieval@1 là tỉ lệ truy vấn tìm đúng khóa trong 2000 ứng viên; linear probe là độ chính xác của một lớp tuyến tính trên đặc trưng đóng băng, cho nhãn nhị phân. Trong hai giá trị τ này, τ = 0.05 cho retrieval cao hơn nhưng probe thấp hơn. Hai con số đối chứng cho thấy probe 0.644 còn tệ hơn: encoder ngẫu nhiên chưa huấn luyện đã có probe 0.928 và đầu vào thô 0.993, vì nhãn của đồ chơi này quá dễ. Nghĩa là ở τ = 0.05 huấn luyện contrastive đã làm mất bớt thông tin nhãn so với không huấn luyện (nguyên nhân chưa kiểm; một giả thuyết là mục tiêu ép phân biệt từng mẫu bằng cả chi tiết không liên quan nhãn). Ở τ = 0.5 probe 0.965 vẫn thấp hơn đầu vào thô. Chỉ hai giá trị τ, một seed, một nhãn dễ: kết luận duy nhất rút được là retrieval và probe không đi cùng chiều ở đây; đừng khái quát thành "SSL phụ thuộc tác vụ" từ đồ chơi này. Về nguồn: SimCLR (Chen và cộng sự, Bảng 5) thấy τ = 0.1 tốt nhất khi quét {0.05, 0.1, 0.5, 1} có chuẩn hóa ℓ2, nhưng ở CIFAR-10 cùng bài chọn 0.5 (phụ lục B.9); SimCSE (Gao và cộng sự, Bảng D.1) chọn 0.05 cho câu. Cả hai số đã đối chiếu bản gốc. Vậy không có τ phổ quát. SimCSE còn cho thấy "tăng cường dữ liệu" có thể chỉ là nhiễu dropout: hai lần chạy cùng một câu qua encoder với mặt nạ dropout khác nhau là cặp dương.

**Masked modeling (BERT).** Chọn 15% token (WordPiece); trong số chọn, 80% thay bằng [MASK], 10% thay bằng token ngẫu nhiên, 10% giữ nguyên; loss chỉ tính trên các vị trí được chọn (Devlin và cộng sự, 2019). Lý do 80/10/10: [MASK] không xuất hiện lúc tinh chỉnh, nên nếu luôn thay bằng [MASK] thì mạng chỉ học đoán khi thấy ký hiệu đó; lẫn token ngẫu nhiên và giữ nguyên buộc nó giữ biểu diễn tốt cho mọi vị trí.

**Ví dụ giải mẫu 10.10 — đếm tín hiệu, và palindrome.**

- Chuỗi 512 token: chọn 15% là 76.8 vị trí dự đoán, gồm 61.44 [MASK], 7.68 ngẫu nhiên, 7.68 giữ nguyên. Causal LM dự đoán 511 vị trí mỗi chuỗi, gấp 6.7 lần. Mô phỏng 200 000 token (Lab 4): chọn 14.98%; trong số chọn, 79.7% [MASK], 9.9% ngẫu nhiên, 10.4% giữ nguyên.
- Palindrome: từ vựng 10 token, chuỗi dài 8, nửa sau là đảo của nửa đầu (token thứ 9 − i bằng token thứ i, i = 1..4). Cùng một Transformer 2 lớp, 400 bước Adam, hai mục tiêu. *Causal LM:* loss 0.9966 trên 7 vị trí dự đoán; ba vị trí đầu của nửa đầu ngẫu nhiên (ln 10 mỗi vị trí), bốn vị trí sau xác định, nên sàn lý thuyết là (3/7)·ln 10 = **0.9868**. Đo theo nhóm: nửa đầu 2.324, nửa sau 0.0014. *Masked LM* (che 25% khi huấn luyện, che đúng một vị trí khi đo): loss **0.0023**, vì vị trí đối xứng luôn nhìn thấy.

% manim: MatNaMaskedVsCausal

Bài học: cùng kiến trúc, khác mặt nạ thì khác khả năng. Palindrome là trường hợp dựng sẵn để thấy hiệu ứng, không chứng minh BERT tốt hơn GPT.

| | Masked LM (kiểu BERT) | Causal LM (kiểu GPT) |
| --- | --- | --- |
| Mặt nạ attention | Không có: thấy cả hai phía | Nhân quả: chỉ thấy bên trái |
| Vị trí học trên mỗi chuỗi | Khoảng 15% | 100% (trừ token đầu) |
| Likelihood của chuỗi | Không định nghĩa | Có, nên có perplexity |
| Sinh văn bản | Không tự nhiên | Tự hồi quy |
| Mạnh ở | Biểu diễn ngữ cảnh hai chiều cho phân loại, gán nhãn | Sinh và mô hình ngôn ngữ |

Phần đi sâu pretraining quy mô lớn thuộc chương 9.

**Vì sao ML cần cái này.**

- **Mọi LLM đều là SSL:** masked và causal chính là loss huấn luyện, nên hiểu chúng là hiểu mô hình đang được dạy gì.
- **Biểu diễn câu bằng contrastive** (SimCSE) là bài toán NLP điển hình; negative sampling của word2vec và NCE là tiền thân của InfoNCE (nhớ từ literature, chưa mở lại).
- **Cái bẫy false negative:** batch lấy ngẫu nhiên từ dữ liệu có nhãn thì nhiều "mẫu âm" thật ra cùng lớp với truy vấn (bài tính nhanh 13 trả lời bằng số).

**Thực tế tính toán.** Chuẩn hóa L2 trước khi nhân, chia cho τ (không nhân), nhãn là `arange(N)`. Với masked LM đặt nhãn −100 ở vị trí không che và `ignore_index = −100`.

## 10.5 Đánh giá mô hình sinh: likelihood, perplexity, FID, và vì sao không có một thước đo

Không có một con số nào đo "mô hình sinh tốt". Theis, van den Oord và Bethge (2016) chỉ ra log-likelihood, ước lượng Parzen và chất lượng mẫu nhìn bằng mắt gần như độc lập với nhau, nên phải đánh giá theo ứng dụng. Ví dụ trong bài: trộn mô hình tốt (xác suất 1/100) với mô hình xấu (99/100) làm 99% mẫu đến từ mô hình xấu, mà log-likelihood giảm tối đa ln 100 = 4.61 nat, nhỏ so với chênh lệch giữa các mô hình trên ảnh.

**Likelihood.** Mô hình tự hồi quy cho likelihood chính xác; VAE và diffusion cho cận dưới (ELBO); GAN không có likelihood. Với dữ liệu liên tục dùng bits/dim = NLL (nat)/(số chiều · ln 2), và cần "khử lượng tử hóa" (dequantization) khi so mô hình trên dữ liệu rời rạc như ảnh 8 bit. Likelihood cao không chắc mẫu đẹp, và ngược lại: bài DDPM báo FID rất thấp (3.17 trên CIFAR-10) nhưng codelength không cạnh tranh với mô hình likelihood tốt nhất (≤ 3.75 bits/dim, cũng trên CIFAR-10; Ho và cộng sự, Bảng 1, đã đối chiếu bản gốc). Diffusion không chỉ cho cận ELBO: probability-flow ODE (Song và cộng sự, 2021) cho phép tính likelihood chính xác của mô hình, với chi phí tính toán lớn hơn (nhớ từ literature, chưa mở lại).

**Perplexity** (chương 3 đã dạy: exp của cross-entropy trung bình, và vì sao so giữa hai tokenizer khác nhau là sai) có hai điểm mới ở đây. (1) Masked LM không định nghĩa likelihood của chuỗi nên không có perplexity chuẩn (có "pseudo-perplexity", một xấp xỉ khác). (2) Perplexity đo độ khớp phân phối, không đo độ hữu ích.

**Ví dụ giải mẫu 10.11 — bits/dim và perplexity.**

- NLL 6400 nat trên ảnh 32×32×3 = 3072 chiều: 6400/3072/ln 2 = **3.0056 bits/dim**. Mô hình đoán đều trên 256 mức có 8 bits/dim, tức 17 034.8 nat mỗi ảnh.
- Phân phối (0.5, 0.25, 0.125, 0.125) có entropy 1.75 bit, perplexity = exp(1.75 · ln 2) = **3.364**; đoán đều cho 4.0.
- Mô hình có loss 2.9 nat/token và 4.2 byte/token: 2.9/ln 2/4.2 = **0.996 bit/byte**, con số so được giữa các tokenizer.

**FID** (Heusel và cộng sự, 2017) khớp một Gauss (trung bình m, hiệp phương sai C) cho đặc trưng Inception-v3 của ảnh thật và một Gauss (mw, Cw) cho ảnh sinh, rồi tính khoảng cách Fréchet giữa hai Gauss:

$$
\mathrm{FID} = \lVert m - m_w \rVert_2^2 + \operatorname{Tr}\big(C + C_w - 2\,(C\,C_w)^{1/2}\big)
$$

**Ví dụ giải mẫu 10.12 — FID tay và ba giới hạn.**

- Một chiều: FID = (m₁ − m₂)² + (σ₁ − σ₂)². Hai Gauss N(0, 1) và N(3, 2²) cho 9 + 1 = 10. Hai Gauss 2D có hiệp phương sai chéo: phần hiệp phương sai là ∑(σ₁ − σ₂)².
- Hai Gauss 2D: m₁ = (0, 0), C₁ = [[1, 0.5], [0.5, 2]]; m₂ = (1, 2), C₂ = diag(2, 1). Phần trung bình 5.0, phần hiệp phương sai (tính bằng `sqrtm`) 0.4352, **FID = 5.4352**.
- *Mù hình dạng:* 20 000 điểm N(0, I₂) so với 20 000 điểm đều trên vòng tròn bán kính √2 (cùng trung bình 0 và hiệp phương sai I): FID = **0.00018**.
- *Thiên lệch theo cỡ mẫu:* d = 64, hai mẫu cùng N(0, I). N = 100, 500, 2000, 10 000 cho FID 21.6, 4.45, 1.09, 0.217; N · FID luôn quanh 2170. Xấp xỉ bậc hai cho thiên lệch (dẫn xuất gần đúng, không chặt): hai tập mẫu N điểm từ cùng N(0, I). Phần trung bình: m₁ − m₂ có phương sai 2/N mỗi chiều, nên 𝔼‖m₁ − m₂‖² = 2d/N. Phần hiệp phương sai: khi C₁, C₂ gần I, Tr(C₁ + C₂ − 2·√(C₁C₂)) ≈ ¼‖C₁ − C₂‖² (√ là căn bậc hai ma trận) (chuẩn Frobenius); mỗi phần tử ngoài đường chéo của C₁ − C₂ có phương sai 2/N, mỗi phần tử trên đường chéo 4/N, nên 𝔼‖C₁ − C₂‖² = (2d² + 2d)/N, và phần này có kỳ vọng d(d + 1)/(2N). Tổng: N · FID ≈ d(d + 1)/2 + 2d, tức thiên lệch tăng như d²/N. Mô phỏng nhiều lần lặp (N = 2000; d = 4, 16, 32, 64) cho 17.0, 164.9, 591.1, 2205.7 so với công thức 18, 168, 592, 2208. Bỏ số hạng d/2 (tức d²/2 + 2d: 16, 160, 576, 2176) lệch hệ thống về phía thấp.
- *Bỏ một mode vẫn bị phạt, nhưng Gauss khớp moment gần như không bị phạt:* dữ liệu 50% N(−3, 0.3²) + 50% N(3, 0.3²). Mô hình chỉ có mode phải: FID = **16.35**. Mô hình là một Gauss khớp hai moment đầu (phương sai 9.09): FID = **0.0029**, dù hai bên khác hẳn.

Hệ quả: so FID giữa các bài báo chỉ có nghĩa khi cùng cỡ mẫu, cùng bản Inception, cùng cách resize (hai điều sau là kiến thức nhớ từ literature, chưa đo). FID thấp không có nghĩa là đa dạng. Precision và recall cho mô hình sinh (Kynkäänniemi và cộng sự, 2019; nhớ, chưa mở) tách "chất lượng" khỏi "độ phủ", dùng để nhìn đánh đổi chất lượng–đa dạng như ở mục 10.3.

**Vì sao ML cần cái này.**

- **Đọc bảng kết quả bài báo:** biết một con số FID, perplexity hay bits/dim đo được gì và bỏ sót gì, là kỹ năng phản biện bài báo (chương 11).
- **Tự đánh giá mô hình của mình:** luôn kết hợp nhiều chỉ số và nhìn mẫu.

**Thực tế tính toán.** `scipy.linalg.sqrtm` có thể trả số phức với phần ảo rất nhỏ: lấy phần thực và kiểm phần ảo. Cỡ mẫu nhỏ thì hiệp phương sai ước lượng kém (d = 2048 cần nhiều hơn 2048 mẫu để ma trận không suy biến).

## Bài tập giấy bút

14 bài, làm theo thứ tự. Mục tiêu: hết ★ trong 1 buổi, hết ★★ trong 2 buổi, ★★★ dành cho Thứ Bảy. Bài nào sai ghi vào sổ lỗi kèm lý do sai.

**★ Hiểu định nghĩa**

1. Tính KL của N(μ, σ²) so với N(0, 1) cho (μ, σ) = (0, 1), (2, 1) và (0, 0.5). Chuyện gì xảy ra với KL khi σ → 0, và khi |μ| lớn?
2. Tại một điểm x có pdata(x) = 0.3 và pg(x) = 0.1. D*(x) bằng bao nhiêu? Nếu pg(x) = 0 thì sao? Nếu pg(x) = pdata(x)?
3. DDPM có ᾱ = 0.36. Với x₀ = (1, −1) và ε = (0.5, 1), tính xₜ và SNR.
4. Batch N = 5, mọi tương đồng bằng nhau. InfoNCE bằng bao nhiêu? Nếu loss quan sát được là 0.5 thì cận dưới của thông tin tương hỗ là bao nhiêu nat?

**★★ Mức đề thi đầu vào**

5. Chứng minh log p(x) = ELBO + KL(q(z|x)‖p(z|x)) từ định nghĩa. Suy ra ELBO ≤ log p(x) và nói dấu bằng xảy ra khi nào.
6. z ~ N(μ, σ²), f(z) = z². Tính tay: (a) đạo hàm của 𝔼[f(z)] theo μ; (b) kỳ vọng và phương sai của bộ ước lượng reparameterization 2z (số với σ = 0.8); (c) kỳ vọng của bộ ước lượng score-function z²(z − μ)/σ².
7. Dẫn xuất D*(x) và chứng minh max V(D, G) = −ln 4 + 2 JSD(pdata‖pg).
8. Từ q(xₜ|x₀) = N(√ᾱ x₀, (1 − ᾱ)I), tính ∇ log q(xₜ|x₀) theo xₜ rồi theo ε. Số với ᾱ = 0.078587 và ε = 0.5. Nếu mạng đoán đúng ε, khôi phục x̂₀ thế nào? (c) Mạng không biết x₀, vậy đại lượng ∇ log q(xₜ|x₀) vừa tính có phải là thứ nó học được không? Nó học gì, và vì sao đó là score của một mật độ cụ thể nào?
9. Chứng minh (1 + w)εc − w·εu = εu + (1 + w)(εc − εu) = εc + w(εc − εu). Với εc = 0.5, εu = 0.2, w nào cho ε̃ = 2.0? w = −1 và w = 0 cho gì?
10. N = 4, τ = 0.5, truy vấn có tương đồng (0.9, 0.1, 0.0, 0.2) với bốn khóa, khóa dương là khóa đầu. Tính loss của hàng này và xác suất mô hình gán cho khóa đúng. Điều gì xảy ra khi τ → 0 và τ → ∞?

**★★★ Mức phỏng vấn nghiên cứu**

11. Denoising score matching. Cho x₀ ~ q(x₀) bất kỳ, xₜ = √ᾱ x₀ + √(1 − ᾱ) ε. (a) Chứng minh ∇ log qₜ(xₜ) = 𝔼[∇ log q(xₜ|x₀) | xₜ] với qₜ là phân phối biên của xₜ, rồi suy ra 𝔼[ε|xₜ] = −√(1 − ᾱ)·∇ log qₜ(xₜ). (b) Chứng minh nghiệm của min_f 𝔼‖ε − f(xₜ)‖² là f* = 𝔼[ε|xₜ]. (c) Kiểm lại cho x₀ ~ N(0, 1) và cho x₀ ∈ {−1, +1} đều xác suất (một chiều; chỉ cần ra công thức cho 𝔼[ε|xₜ] và so với −√(1 − ᾱ)·score). (d) Vì sao mạng huấn luyện hồi quy ε trên cặp (xₜ, ε) mà không cần biết qₜ?

12. Posterior collapse. Giả sử decoder p(x|z) = r(x) không phụ thuộc z và r đủ mạnh. Chứng minh ELBO = log r(x) − KL(q(z|x)‖p(z)) và nó cực đại tại q = p(z). VAE chỉ dùng z khi nào? Giải thích cơ chế của KL annealing và free bits.
13. pdata là khối điểm tại 0 và pg là khối điểm tại θ ≠ 0. Tính JSD, KL(pdata‖pg) và khoảng cách Wasserstein-1 theo θ. Gradient của từng đại lượng theo θ là gì, và điều đó nói gì về GAN dạng gốc so với WGAN?
14. Palindrome (mục 10.4). (a) Dẫn xuất sàn (3/7)·ln 10 của causal LM. (b) Khi huấn luyện masked LM che mỗi vị trí với xác suất 0.25, loss huấn luyện có thể xuống gần 0 không? Vì sao lúc đo, che đúng một vị trí lại cho 0.0023?

## Bài tính nhanh

15 bài, mỗi bài tối đa 2 phút, làm bằng giấy bút và bấm giờ cả bộ. Mục tiêu: xong cả 15 bài trong 25 phút, sai không quá 1 bài. Làm lại bộ này sau 3 ngày và sau 1 tuần.

1. KL của N(1, 1) so với N(0, 1).
2. KL của N(0, 2²) so với N(0, 1).
3. VAE có k = 20 chiều ẩn, mỗi chiều KL = 0.5 nat. KL tổng bằng bao nhiêu? Nếu lấy trung bình theo chiều thay vì cộng thì tương đương β bằng bao nhiêu?
4. pdata(x) = 0.2 và pg(x) = 0.6. D*(x)?
5. Hai phân phối có giá đỡ rời nhau. JSD bằng bao nhiêu, và max V(D, G) bằng bao nhiêu?
6. D(G(z)) = 0.05. |Gradient theo logit| của loss bão hòa log(1 − D) và của loss non-saturating −log D?
7. ᾱ = 0.81, x₀ = 1, ε = −1. Tính xₜ.
8. ᾱ = 0.75, ε = 1. Tính SNR và score −ε/√(1 − ᾱ).
9. CFG với εc = 1.0, εu = 0.4, w = 2: ε̃?
10. InfoNCE với N = 64 mẫu và mọi embedding bằng nhau: loss?
11. Loss InfoNCE đo được 1.2 với N = 100. Cận dưới thông tin tương hỗ (nat)?
12. Chuỗi 200 token, mặt nạ BERT: kỳ vọng bao nhiêu token được chọn, bao nhiêu [MASK], bao nhiêu thay ngẫu nhiên, bao nhiêu giữ nguyên?
13. Batch 128 từ dữ liệu 10 lớp cân bằng. Một truy vấn có kỳ vọng bao nhiêu "mẫu âm" thật ra cùng lớp với nó?
14. FID một chiều giữa N(0, 1) và N(3, 2²)?
15. Cross-entropy 4.6 nat/token: perplexity? Nếu trung bình 3 byte/token thì bao nhiêu bit/byte?

## Lab code và bài săn bug

Năm lab dưới đây đã chạy với Python 3.13, PyTorch 2.14.1 (bản CPU là đủ) và scipy. Toàn bộ notebook lời giải (gồm cả phân tích từng lỗi) chạy khoảng 1 phút trên CPU 2 lõi đang rảnh (đo được 60 giây đồng hồ, 54 giây CPU); khi máy bận việc khác có thể chậm đi nhiều lần (một lần đo lúc máy bận: 167 giây). Hãy đặt `torch.set_num_threads(1)` như ở đầu notebook; đừng đặt 2 luồng trên CPU ít lõi vì thời gian có thể tăng gấp nhiều lần do tranh chấp luồng. Giảm số bước huấn luyện DDPM để chạy nhanh hơn sẽ đổi kết luận (thử 2000 bước: độ lệch chuẩn của lớp 0 không còn giảm đơn điệu theo w như bảng ở mục 10.3), nên lab giữ 3000 bước. Số liệu tái lập được với seed cố định nhưng có thể lệch nhẹ theo phiên bản thư viện.

| Lab | Việc phải làm | Kết quả mong đợi |
| --- | --- | --- |
| 1. AE, VAE, reparameterization | PCA và AE (tuyến tính, MLP) trên cuộn bánh 3D; `kl_gauss`, `reparam`, `nll_rec`; thử β và k = 6; gradient hai bộ ước lượng | PCA 0.33289, AE tuyến tính 0.33288, AE phi tuyến 0.0565. KL ví dụ 10.1 = 0.8877 (Monte Carlo 0.8866). Gradient: reparam 0.9983 ± 1.600, score function 0.9972 ± 3.639; `torch.normal` cho gradient 0.0. VAE β = 1: −log p(x\|z) = −0.928, KL = 4.719, −ELBO = 3.791; β = 0: KL 13.401; β = 30: KL 0.475, sai số tái tạo 0.6018; k = 6: KL từng chiều (1.932, 1.675, 1.721, 0.001, 0.001, 0.001), −ELBO 2.716 |
| 2. GAN một chiều | `d_star`, `jsd_shift`, `gen_grad_mag`, `train_gan` (non-saturating) trên hai mode ±3 | D*(0) = 0.8808; JSD 0.0303 / 0.3368 / 0.6759 / 0.6931 khi d = 0.5 / 2 / 5 / 10; gradient tại D = 0.01: 0.01 và 0.99. Ba seed như bảng ở mục 10.2: seed 0 sụp về mode −3 |
| 3. Fréchet và DDPM có CFG | `frechet`, `make_schedule`, `q_sample`, `cfg_combine`, huấn luyện với nhãn rỗng (3000 bước, T = 100), lấy mẫu có hướng dẫn | FID tay 5.4352; x₅₀₀ = 1.0406; ᾱ cuối = 0.00536; CFG 0.5 / 0.8 / 1.4 / 2.75; bảng w = 0, 1, 3, 7 ở mục 10.3 (lab dùng w = 7, ví dụ 10.8 dùng 7.5) (Fréchet 0.0303 → 0.6527) |
| 4. InfoNCE, mặt nạ BERT, palindrome | `infonce`, `bert_mask`, `linear_probe` (PyTorch), `causal_lm_loss`, `masked_lm_loss` | Ví dụ 10.9: 0.9898 / 0.6816 / 0.0322; sụp đổ 4.8520 = ln 128; trực giao 0.0057 / 2.9007. τ = 0.05: retrieval 0.426, probe 0.644; τ = 0.5: 0.343, 0.965. Mặt nạ: chọn 0.1498, 80/10/10 trong 0.01. Palindrome: causal 0.9966 (sàn 0.9868), masked 0.0023 |
| 5. Đo đạc | FID theo N, Gauss so với vòng tròn, bỏ mode, `perplexity`, `bits_per_dim` | N · FID: 2162, 2224, 2186, 2168 (N = 100, 500, 2000, 10 000); N · FID theo d = 4, 16, 32, 64 (N = 2000; 200, 40, 20, 10 lần lặp): 17.0, 164.9, 591.1, 2205.7 so với d(d + 1)/2 + 2d = 18, 168, 592, 2208; vòng tròn 0.00018; bỏ mode 16.35, Gauss khớp moment 0.0029; perplexity đoán đều = 50 257.0; 3.0056 bits/dim |

Code nằm trong notebook `labs/chuong10/lab.ipynb` (bản có chỗ TODO để bạn tự viết, kèm ô kiểm tra tự động) và `labs/chuong10/loigiai.ipynb` (bản lời giải đã chạy kiểm). Mở từ trang **Lab** của app.

**Câu hỏi sau lab.**

- Lab 1: AE tuyến tính đạt sai số bằng PCA, nhưng các cột của bộ giải mã có nhất thiết trùng các thành phần chính không? Vì sao k = 6 chỉ dùng ba chiều (đây có phải posterior collapse không?), và vì sao −ELBO của k = 6 (2.716) lại tốt hơn k = 2 (3.791)? *Gợi ý: nhiễu 3D của dữ liệu là 0.02 trong khi σ của decoder cố định là 0.1; giải thích này chưa kiểm, đừng suy rộng.*
- Lab 2: loss D của seed 0 là 0.935, thấp hơn 2 ln 2 = 1.386. Đó có phải là bằng chứng chắc chắn của mode collapse không? Kể một tình huống khác cũng làm loss D thấp.
- Lab 3: nếu huấn luyện không bao giờ dùng nhãn rỗng, vì sao w = 0 vẫn trông ổn nhưng w > 0 hỏng? Vì sao Fréchet tăng theo w trong khi hướng tâm giảm?
- Lab 4: vì sao causal LM không thể xuống dưới 0.9868 trên palindrome? Vì sao τ = 0.05 cho retrieval cao hơn nhưng probe thấp hơn τ = 0.5, và probe 0.644 so với 0.928 của encoder ngẫu nhiên nói gì về thí nghiệm?
- Lab 5: N · FID ≈ 2170 với d = 64. Cần bao nhiêu mẫu để thiên lệch dưới 0.01? Dùng d(d + 1)/2 + 2d ước lượng N · FID cho d = 2048 và so với d = 64.

**Bài săn bug.** Vòng huấn luyện và bộ lấy mẫu DDPM có CFG dưới đây (cùng mô hình, dữ liệu và đánh giá như Lab 3) chạy không báo lỗi, loss huấn luyện giảm bình thường (0.703 ở 100 bước đầu, 0.362 ở 100 bước cuối; bản đúng 0.309). Triệu chứng đo được: với w = 0, Fréchet 0.4319 và hướng tâm lớp 0 là 0.4374 (bản đúng 0.0303 và 0.1253); bật w = 3 thì Fréchet vọt lên 7.7373 và hướng tâm 2.8009 (bản đúng 0.2807 và 0.0653), nhóm lớp 0 bay khỏi hình mặt trăng. Tìm cả **bốn** lỗi trước khi xem lời giải.

```python
def train_buggy(steps=3000, seed=0):
    torch.manual_seed(seed); m = Net(); opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    losses = []
    for i in range(steps):
        idx = torch.randint(0, 6000, (256,)); x0 = X[idx]; c = C[idx]
        t = torch.randint(0, T, (256,)); eps = torch.randn(1, 2)
        a = abar[t].unsqueeze(1)
        xt = a.sqrt() * x0 + (1 - a) * eps
        loss = ((m(xt, t, c) - eps) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return m, losses

@torch.no_grad()
def sample_buggy(m, n, c, w=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g); cc, cu = torch.full((n,), c), torch.full((n,), 2)
    for t in reversed(range(T)):
        tt = torch.full((n,), t)
        eps_hat = (1 + w) * m(x, tt, cc) - w * m(x, tt, cu)
        mean = x - beta[t] / (1 - abar[t]).sqrt() * eps_hat
        if t > 0:
            var = (1 - abar[t - 1]) / (1 - abar[t]) * beta[t]
            x = mean + var.sqrt() * torch.randn(n, 2, generator=g)
        else:
            x = mean
    return x
```

## Bài dựng cảnh Manim

Mỗi chương có một bài tự dựng cảnh: muốn vẽ được một khái niệm, bạn phải hiểu từng bước của nó. Giới hạn **1 giờ** (tính vào luồng C), không chỉnh màu mè.

**Đề bài.** Dùng `manim/chuong10_scenes.py` làm mẫu, viết cảnh `FIDHaiGauss` vào thư mục `manim/cua-toi/`:

1. Vẽ hai elip Gauss 2D của ví dụ 10.12: m₁ = (0, 0), C₁ = [[1, 0.5], [0.5, 2]] (màu xanh) và m₂ = (1, 2), C₂ = diag(2, 1) (màu đỏ). Elip là đường đồng mức cách trung bình một đơn vị Mahalanobis; có thể lấy trục bằng `np.linalg.eigh`.
2. Vẽ mũi tên từ m₁ đến m₂ và ghi "phần trung bình = 5.0" (bình phương độ dài mũi tên).
3. Hiện phần hiệp phương sai 0.4352 (tính bằng `scipy.linalg.sqrtm` trong code, không gõ tay) rồi cộng hai phần thành FID = 5.4352.

**Tiêu chí tự chấm.** (1) Đúng toán: hai phần là 5.0 và 0.4352, tổng 5.4352, và con số do code tính chứ không gõ tay. (2) Có tiêu đề và phụ đề nói người xem phải thấy điều gì (hai phần của FID đo hai thứ khác nhau). (3) Dài không quá 30 giây, chữ không chồng chữ.

Render bằng nút **Render cảnh của tôi** ở trang Manim. Lời giải tham khảo nằm ở `manim/loigiai/chuong10_fid_hai_gauss.py` — chỉ mở sau khi đã tự làm.

## Phỏng vấn, bẫy, thực tế khác lý thuyết

**Cách luyện.** Mỗi chuỗi có 4 tầng câu hỏi, đào sâu dần như người phỏng vấn thật. Trả lời to bằng tiếng Việt trước, sau đó nói lại bằng tiếng Anh, 2 phút mỗi tầng, ghi âm và tự nghe lại. Câu trả lời mẫu tiếng Anh bên dưới chỉ phủ tầng 1–2; tầng 3–4 bạn tự soạn, rồi đối chiếu với lời giải các bài tập liên quan. Các câu hỏi là câu điển hình, diễn đạt lại; người viết không có nguồn đề phỏng vấn cụ thể cho chương này.

**Chuỗi A — VAE**

1. ELBO từ đâu ra, hai số hạng nghĩa là gì?
2. Vì sao cần reparameterization, và nó giảm phương sai ở đâu? (Số của ví dụ 10.3.)
3. Posterior collapse là gì, nhận ra thế nào, chữa thế nào? (KL từng chiều, KL annealing, free bits, word dropout.)
4. Khe hở giữa ELBO và log p(x) bằng gì? Vì sao mẫu VAE hay mờ? (Câu sau là nhận định nhớ từ literature, nói rõ mức chắc chắn.)

> *Model answer:* We cannot maximize log p(x) directly because the integral over z is intractable. Instead we introduce an encoder q(z|x) and maximize the ELBO: the expected reconstruction log-likelihood minus the KL divergence from q(z|x) to the prior. The first term makes the code informative about x, the second keeps the code close to the prior so we can sample from it. The gap to log p(x) is exactly the KL from q to the true posterior. Reparameterization writes z as mu plus sigma times noise, so gradients flow through the sampling step; the score-function estimator is also unbiased but in my toy test its per-sample standard deviation was about 2.3 times larger.

**Chuỗi B — GAN**

1. Viết trò chơi minimax. Discriminator tối ưu là gì?
2. Vì sao điều đó dẫn tới Jensen–Shannon, và vì sao gradient của generator có thể biến mất?
3. Non-saturating loss sửa gì và còn thiếu gì? Làm sao nhận ra mode collapse?
4. FID đo gì và không đo gì? Vì sao diffusion thay GAN cho ảnh?

> *Model answer:* The discriminator maximizes the expected log-probability of classifying real and fake samples correctly, and the generator minimizes that same quantity. For a fixed generator the optimal discriminator is pdata over pdata plus pg, and plugging it back gives minus log 4 plus twice the Jensen–Shannon divergence. The trouble is that when the two distributions have disjoint supports, JSD is a constant log 2 and gives no gradient. The non-saturating loss keeps the same fixed point with a stronger gradient early on, but it is not a complete fix. Mode collapse does not show up in the loss: in my toy run one seed put almost all samples in one mode while the discriminator loss stayed below the equilibrium value, so you have to look at samples.

**Chuỗi C — Diffusion**

1. Viết xₜ theo x₀. Mạng được huấn luyện để làm gì?
2. Vì sao đoán nhiễu tương đương đoán score? Hệ số là gì?
3. Lịch nhiễu tuyến tính so với cosine; SNR đóng vai trò gì? Classifier-free guidance làm gì, w lớn gây hậu quả gì?
4. Chi phí lấy mẫu; vì sao L simple cho mẫu đẹp nhưng codelength kém hơn ELBO đúng; vì sao diffusion Gauss khó cho văn bản?

> *Model answer:* The forward process has a closed form: x at step t equals the square root of alpha-bar times x at step 0 plus the square root of one minus alpha-bar times Gaussian noise. The network predicts that noise from the noisy input and t with a plain mean squared error. The score of the Gaussian conditional is minus the noise divided by the square root of one minus alpha-bar, but the network does not see x at step 0. Its optimal output is the conditional expectation of the noise given the noisy input, and averaging the conditional score over the posterior of x at step 0 gives the score of the noisy marginal. So a noise predictor estimates the marginal score up to the known scale, which is exactly what the sampler needs. Classifier-free guidance trains one network with the label randomly dropped, and at sampling time extrapolates from the unconditional prediction toward the conditional one. In my toy experiment a larger guidance weight made each sample sit closer to the data manifold but shrank the spread of the samples, so the Frechet distance went up.

**Chuỗi D — Học tự giám sát**

1. Viết InfoNCE. Nó liên hệ thế nào với thông tin tương hỗ?
2. Vai trò của nhiệt độ τ và số mẫu âm? Khi nào loss bằng ln N?
3. BERT: vì sao 15% và 80/10/10? So với causal LM về tín hiệu học, likelihood và sinh văn bản.
4. Vì sao SSL là nền của NLP hiện đại? SimCSE làm gì? Vì sao không so perplexity giữa tokenizer và giữa masked với causal LM?

> *Model answer:* InfoNCE is a softmax cross-entropy over one positive and N minus one negatives, with similarities divided by a temperature. Minimizing it maximizes a lower bound on mutual information, and that bound can never exceed log N. A small temperature sharpens the softmax and punishes hard negatives, a large one pushes the loss toward log N. BERT masks about 15 percent of tokens; of those, 80 percent become the mask symbol, 10 percent a random token and 10 percent stay unchanged, because the mask symbol never appears at fine-tuning time. Compared with a causal language model, a masked model sees both sides but learns from far fewer positions and does not define a sequence likelihood.

**Từ vựng nên nói trôi chảy:** evidence lower bound, amortized inference, reparameterization trick, posterior collapse, minimax game, mode collapse, non-saturating loss, noise schedule, signal-to-noise ratio, classifier-free guidance, score function, InfoNCE, temperature, false negative, masked language modeling, linear probe, Fréchet distance, bits per dimension.

**Bẫy thường gặp.** Danh sách dưới đây nhắc lại đúng các lỗi của bài săn bug ở mục Lab; chỉ đọc sau khi đã tự làm bài đó.

- **VAE:** `torch.normal(mu, std)` thay cho `mu + std * randn_like`: chạy không lỗi, gradient bằng 0, KL = 0.0 và mô hình chỉ đoán trung bình. Dùng `exp(logvar)` làm σ thay vì `exp(0.5 * logvar)`. Lấy trung bình KL theo chiều ẩn thay vì cộng. Lấy z ngẫu nhiên đưa vào AE thường rồi gọi đó là mô hình sinh.
- **GAN:** quên `detach()` khi cập nhật D; cập nhật D và G trên cùng một lượt forward; dùng sigmoid rồi BCE thay vì `BCEWithLogits`; nhìn loss thay vì nhìn mẫu để phát hiện collapse.
- **Diffusion:** viết (1 − ᾱ)·ε thay vì √(1 − ᾱ)·ε; bỏ hệ số 1/√αₜ khi lấy mẫu; dùng một ε chung cho cả batch; lệch chỉ số t; không bao giờ thay nhãn bằng ∅ rồi dùng w > 0.
- **CFG:** nhầm quy ước w giữa (1 + w)εc − wεu và εu + w(εc − εu).
- **InfoNCE:** nhân τ thay vì chia; quên chuẩn hóa L2; nhãn không phải `arange`; coi mẫu cùng lớp trong batch là âm; đếm sai số âm (N − 1 hay 2(N − 1)).
- **Masked LM và đánh giá:** tính loss cả token không che (phải `ignore_index = −100`); tính perplexity cho masked LM; so FID khác cỡ mẫu; `sqrtm` trả số phức mà không lấy phần thực.

**Thực tế khác lý thuyết.**

- **VAE có k chiều ẩn không dùng k chiều (chiều không hoạt động, không phải posterior collapse).** Lab 1 với k = 6: ba chiều có KL ≈ 1.7–1.9 nat, ba chiều ≈ 0.001. Dữ liệu chỉ có 3 chiều và decoder không mạnh, nên KL tỉa các chiều thừa; z vẫn mang thông tin qua ba chiều còn lại. −ELBO của k = 6 (2.716) tốt hơn k = 2 (3.791). Giải thích có thể là chiều thứ ba mô hình hóa độ dày nhiễu; chưa kiểm, đừng suy rộng.
- **Cân bằng D = ½ của GAN không phải thứ bạn thấy.** Seed 0 của Lab 2 có loss D 0.935, thấp hơn 1.386, và chỉ giữ một mode. Ba seed không đủ để nói tần suất collapse; chỉ đủ để nói nó phụ thuộc seed. Đồng thời, trên đồ chơi này hai phân phối chồng lấn một phần, nên không tái hiện "gradient biến mất hoàn toàn" của định lý (định lý nói về discriminator hoàn hảo).
- **Diffusion không tối ưu đúng ELBO, và đó là chủ ý.** Ho và cộng sự (2020, Bảng 1, CIFAR-10): tối ưu đúng ELBO (L, phương sai Σ cố định) cho ≤ 3.70 bits/dim nhưng FID 13.51; L simple (ELBO bỏ trọng số theo t) cho FID 3.17 nhưng ≤ 3.75 bits/dim, tức codelength không cạnh tranh. Bài báo gọi L simple là một weighted variational bound. Đã đối chiếu bản gốc; một cấu hình, không suy rộng sang mô hình khác.
- **Một thước đo không đủ.** Trong bản lỗi thiếu căn, Fréchet 0.0055 tốt hơn bản đúng (0.0303) trong khi sai số hướng tâm xấu gấp 1.9 lần (0.2359 so với 0.1253). Fréchet chỉ nhìn hai moment. CFG cũng không miễn phí: w tăng thì Fréchet tăng và độ lệch chuẩn co lại.
- **Contrastive:** trong đồ chơi, τ = 0.05 cho retrieval cao hơn nhưng probe thấp hơn (và thấp hơn cả encoder ngẫu nhiên); không có τ phổ quát. BERT "hai chiều" cũng không tốt hơn mọi chỗ: cho tác vụ sinh, causal LM là mục tiêu tự nhiên. Palindrome chỉ là thí nghiệm dựng sẵn.
- **Mọi số trong chương** là đồ chơi một seed (nối chương 11).

## Mini-test 30 phút

Làm sau khi xong bài tập và lab, bấm giờ đúng 30 phút, không xem sách, không dùng máy tính. Tổng 30 điểm. **Từ 24 điểm:** sang chương 11. **18 đến dưới 24:** ôn lại mục ứng với câu sai rồi làm lại sau 3 ngày. **Dưới 18:** học lại chương.

1. **(4 điểm)** Một VAE hai chiều ẩn có q với μ = (1, 0), σ = (0.5, 2). Tính KL so với N(0, I). Cho ln 2 = 0.6931.
2. **(5 điểm)** pdata = N(0, 1), pg = N(1, 1). Viết D*(x) dạng sigmoid, tính D*(0) và D*(1.5), tìm x để D*(x) = ½. Nếu JSD(pdata, pg) = 0.1114, max V(D, G) bằng bao nhiêu?
3. **(5 điểm)** Một bước lấy mẫu DDPM: ᾱₜ = 0.64, ᾱₜ₋₁ = 0.8, xₜ = 1.2, mạng đoán ε̂ = 0.6. Tính αₜ, βₜ, trung bình của xₜ₋₁, và độ lệch chuẩn của nhiễu cộng thêm (phương sai β̃ₜ). Nếu ε̂(x, c) = 0.6, ε̂(x, ∅) = 0.1, w = 2 thì ε̃ bằng bao nhiêu?
4. **(5 điểm)** N = 3, S = [[0.8, 0.2, 0.0], [0.1, 0.6, 0.5], [0.3, 0.0, 0.9]] (cặp dương trên đường chéo), τ = 0.5. Tính loss trung bình, cận dưới thông tin tương hỗ, và nói hàng nào khó nhất.
5. **(5 điểm)** Chuỗi 1000 token. (a) Masked LM kiểu BERT và causal LM dự đoán bao nhiêu vị trí mỗi chuỗi? (b) Nếu 100% vị trí được chọn đều thay bằng [MASK], có vấn đề gì lúc tinh chỉnh? (c) Vì sao masked LM không có perplexity chuẩn?
6. **(6 điểm)** Tối đa 5 câu: mô hình A có FID 5.0 tính trên 1000 mẫu, mô hình B có FID 8.0 tính trên 50 000 mẫu. Có kết luận A tốt hơn B được không? Nêu ít nhất hai lý do FID không đủ.

## Flashcard

18 thẻ của chương nằm trong `flashcards/chuong10.csv`. Ôn ở trang **Ôn tập** của app: app tự lên lịch theo hộp Leitner (thẻ nhớ được sẽ giãn lịch ôn 1 → 3 → 7 → 14 ngày, thẻ quên quay lại hộp 1). File CSV này cũng nhập thẳng được vào Anki nếu bạn thích dùng Anki.

## Lời giải

Chỉ đọc sau khi đã tự thử ít nhất 20 phút. Với bài chứng minh, so ý chính chứ không so từng chữ. Mọi đáp số bằng số trong chương đã được kiểm lại bằng NumPy hoặc PyTorch.

**Kiểm tra nhanh**

1. Tích phân ∫ p(z)p(x|z) dz không tính được. Số hạng KL(q(z|x)‖p(z)) kéo q về prior.
2. D* = ½ (pdata = pg nên tỉ số bằng 1).
3. xₜ = √0.25 · 2 + √0.75 · 1 = 1 + 0.8660 = 1.8660.
4. ln 256 = 5.5452, vì softmax đều trên 256 ứng viên.
5. Lớn hơn 0. Ước lượng Gauss từ ít mẫu làm FID dương, thiên lệch cỡ 1/N (Lab 5: 2170/N khi d = 64).

**Bài tập giấy bút**

1. KL = 0, 2, 0.3181. Khi σ → 0, số hạng −ln σ² → ∞ nên KL → ∞ (q co về một điểm, không thể khớp prior trải rộng). Khi |μ| lớn, KL ≈ μ²/2 tăng bậc hai.
2. D* = 0.3/(0.3 + 0.1) = 0.75. Nếu pg(x) = 0 thì D* = 1 (chỉ có thể là dữ liệu thật). Nếu pg = pdata thì D* = ½.
3. √ᾱ = 0.6, √(1 − ᾱ) = 0.8. xₜ = 0.6 · (1, −1) + 0.8 · (0.5, 1) = (1.0, 0.2). SNR = 0.36/0.64 = 0.5625.
4. Loss = ln 5 = 1.6094. Cận dưới thông tin tương hỗ = ln 5 − 0.5 = 1.1094 nat (nhỏ hơn ln 5 = 1.6094, nhất quán với cận không vượt ln N).
5. log p(x) = 𝔼 theo q của [log p(x, z) − log p(z|x)] vì log p(x) không phụ thuộc z. Cộng và trừ log q: = 𝔼[log p(x, z) − log q] + 𝔼[log q − log p(z|x)] = ELBO + KL(q‖p(z|x)). Vì p(x, z) = p(x|z)p(z), 𝔼[log p(x, z) − log q] = 𝔼[log p(x|z)] − KL(q‖p(z)). KL ≥ 0 nên ELBO ≤ log p(x); dấu bằng khi q(z|x) = p(z|x) hầu khắp nơi.
6. (a) ∂/∂μ 𝔼[z²] = ∂/∂μ (μ² + σ²) = 2μ = 1.0. (b) Ước lượng 2z có kỳ vọng 2μ = 1 và phương sai 4σ² = 2.56, độ lệch chuẩn 1.6. (c) ∇μ 𝔼[f] = ∇μ ∫ f q = ∫ f q ∇μ log q = 𝔼[f(z)(z − μ)/σ²], nên kỳ vọng cũng bằng 1.0 (không chệch), nhưng phương sai lớn hơn (độ lệch chuẩn 3.64 đo trong Lab 1).
7. Với mỗi x, cực đại hóa a log y + b log(1 − y) theo y ∈ (0, 1), a = pdata(x), b = pg(x), cho y = a/(a + b). Thay lại, đặt M = (pdata + pg)/2: V = ∫ pdata log(pdata/2M) + pg log(pg/2M) = −ln 4 + KL(pdata‖M) + KL(pg‖M) = −ln 4 + 2 JSD.
8. ∇ log q = −(xₜ − √ᾱ x₀)/(1 − ᾱ) = −ε/√(1 − ᾱ). Số: −0.5/0.9599 = −0.5209. Nếu mạng đoán đúng ε: x̂₀ = (xₜ − √(1 − ᾱ)ε)/√ᾱ. (c) Không. Đó là score có điều kiện theo x₀, mạng không có x₀ làm đầu vào. Mạng chỉ nhìn (xₜ, t) nên đầu ra tối ưu là 𝔼[ε|xₜ] (bài 11b), và 𝔼[ε|xₜ] = −√(1 − ᾱ)·∇ log qₜ(xₜ) với qₜ là phân phối biên của xₜ (bài 11a). Vậy mạng ước lượng score của qₜ, mật độ của dữ liệu đã thêm nhiễu; đó mới là thứ bộ lấy mẫu cần.
9. (1 + w)εc − wεu = εu + (1 + w)(εc − εu) (khai triển) = εc + w(εc − εu) (cộng trừ εc). ε̃ = 0.2 + 0.3(1 + w) = 2.0 cho 1 + w = 6, w = 5. w = 0 cho εc = 0.5; w = −1 cho εu = 0.2 (không điều kiện).
10. Loss = ln(e¹·⁸ + e⁰·² + 1 + e⁰·⁴) − 1.8 = 0.4786; xác suất khóa đúng 0.6197. τ → 0: softmax thành argmax, vì khóa dương có tương đồng lớn nhất nên loss → 0 (xác suất → 1). τ → ∞: softmax đều, loss → ln 4 = 1.3863.
11. (a) qₜ(xₜ) = ∫ q(x₀) q(xₜ|x₀) dx₀. Đạo hàm theo xₜ dưới dấu tích phân và dùng ∇q = q·∇log q: ∇qₜ(xₜ) = ∫ q(x₀) q(xₜ|x₀) ∇ log q(xₜ|x₀) dx₀. Chia cho qₜ(xₜ) và nhận ra q(x₀) q(xₜ|x₀)/qₜ(xₜ) = q(x₀|xₜ) (Bayes): ∇ log qₜ(xₜ) = 𝔼[∇ log q(xₜ|x₀) | xₜ] = −𝔼[ε|xₜ]/√(1 − ᾱ), vì score có điều kiện bằng −ε/√(1 − ᾱ) (bài 8). Nhân hai vế với −√(1 − ᾱ): 𝔼[ε|xₜ] = −√(1 − ᾱ)·∇ log qₜ(xₜ). (b) Cộng và trừ 𝔼[ε|xₜ]: 𝔼‖ε − f‖² = 𝔼‖ε − 𝔼[ε|xₜ]‖² + 2𝔼[(ε − 𝔼[ε|xₜ])ᵀ(𝔼[ε|xₜ] − f(xₜ))] + 𝔼‖𝔼[ε|xₜ] − f‖². Số hạng giữa bằng 0: lấy kỳ vọng có điều kiện theo xₜ trước, ε − 𝔼[ε|xₜ] có kỳ vọng 0 trong khi thừa số kia là hàm của xₜ. Số hạng đầu không phụ thuộc f nên cực tiểu khi f(xₜ) = 𝔼[ε|xₜ]. (c) Gauss: xₜ ~ N(0, 1), score biên −xₜ. Cov(ε, xₜ) = √(1 − ᾱ), Var(xₜ) = 1, nên 𝔼[ε|xₜ] = √(1 − ᾱ)·xₜ = −√(1 − ᾱ)·(−xₜ): khớp. Hai điểm ±1 (a = √ᾱ, s² = 1 − ᾱ): qₜ là hỗn hợp hai Gauss, score biên = −xₜ/s² + (a/s²)·tanh(a·xₜ/s²). Mặt khác 𝔼[x₀|xₜ] = tanh(a·xₜ/s²) nên 𝔼[ε|xₜ] = (xₜ − a·tanh(a·xₜ/s²))/s, và −s·score biên cho đúng biểu thức đó: khớp. Ở ví dụ này score biên khác hẳn score có điều kiện, và chỉ score biên ra khỏi hàm của xₜ. (d) Kỳ vọng của hàm mất mát lấy trên cặp (xₜ, ε) rút từ quá trình nhiễu hóa, mà ta mô phỏng được; nghiệm tối ưu f* chỉ phụ thuộc phân phối đồng thời đó. Không bước nào cần đánh giá qₜ (không tính được), nên có thể huấn luyện bằng hồi quy bình thường.
12. Nếu p(x|z) = r(x): 𝔼 theo q của log r(x) = log r(x), nên ELBO = log r(x) − KL(q‖p(z)) ≤ log r(x), cực đại khi KL = 0, tức q = p(z): z không mang thông tin. VAE chỉ dùng z khi dùng nó giảm loss tái tạo nhiều hơn chi phí KL, tức decoder không tự mô hình hóa nổi x. KL annealing đặt β nhỏ lúc đầu để encoder kịp đưa thông tin vào z trước khi bị phạt; free bits không phạt KL dưới một ngưỡng mỗi chiều nên không còn lực đẩy các chiều đó về 0.
13. JSD = ln 2 với mọi θ ≠ 0 (hai khối điểm rời nhau); KL(pdata‖pg) = ∞; Wasserstein-1 = |θ|. Gradient theo θ: JSD bằng 0, KL không xác định, W₁ có độ lớn 1 và hướng về 0. GAN dạng gốc (với discriminator gần hoàn hảo) không có tín hiệu kéo hai khối lại; WGAN có. Đây là trực giác, không phải bằng chứng mọi trường hợp.
14. (a) Causal LM dự đoán token 2..8 (7 vị trí) từ tiền tố. Token 2, 3, 4 là nửa đầu, ngẫu nhiên đều trên 10 giá trị nên mỗi cái tối thiểu ln 10; token 5..8 xác định từ token 4..1. Sàn = (3/7) ln 10 = 0.9868. (b) Không: khi huấn luyện, với xác suất 0.25 vị trí đối xứng cũng bị che, lúc đó chỉ đoán ngẫu nhiên (ln 10), nên loss huấn luyện có sàn cỡ 0.25 · ln 10 ≈ 0.58 (xấp xỉ, bỏ qua điều kiện "ít nhất một vị trí bị che"). Lúc đo, che đúng một vị trí nên vị trí đối xứng luôn thấy được: 0.0023.

**Bài tính nhanh**

1. ½(1 + 1 − 1 − 0) = 0.5.
2. ½(4 + 0 − 1 − ln 4) = ½(3 − 1.3863) = 0.8069.
3. 20 · 0.5 = 10 nat; trung bình thay cho tổng tương đương β = 1/20 = 0.05.
4. 0.2/(0.2 + 0.6) = 0.25.
5. JSD = ln 2 = 0.6931; max V = −ln 4 + 2 ln 2 = 0.
6. Bão hòa: 0.05; non-saturating: 0.95.
7. xₜ = 0.9 · 1 + √0.19 · (−1) = 0.9 − 0.4359 = 0.4641.
8. SNR = 0.75/0.25 = 3; score = −1/0.5 = −2.
9. 3 · 1.0 − 2 · 0.4 = 2.2.
10. ln 64 = 4.1589.
11. ln 100 − 1.2 = 4.6052 − 1.2 = 3.4052 nat.
12. 30 token được chọn; 24 [MASK], 3 ngẫu nhiên, 3 giữ nguyên.
13. Khoảng 127 · 0.1 = 12.7 mẫu âm cùng lớp (giả định nhãn cân bằng và độc lập).
14. (0 − 3)² + (1 − 2)² = 10.
15. e⁴·⁶ = 99.5. Bit/byte = 4.6/ln 2/3 = 2.21.

**Câu hỏi sau lab**

- Lab 1: không nhất thiết: AE tuyến tính chỉ cần cùng không gian con, nên bộ giải mã có thể là PCA nhân một ma trận khả nghịch bất kỳ k × k; sai số tái tạo không đổi. k = 6 dùng ba chiều vì dữ liệu chỉ có 3 chiều: ba chiều đủ mô hình hóa nó và mỗi chiều dư tốn KL mà không giảm tái tạo, nên bị tỉa về prior (chiều không hoạt động). Đây không phải posterior collapse: decoder là MLP nhỏ, không tự mô hình hóa được x nếu thiếu z, và z vẫn mang thông tin qua ba chiều còn sống. −ELBO của k = 6 tốt hơn có thể do chiều thứ ba bắt độ dày nhiễu (chưa kiểm).
- Lab 2: không chắc chắn. Loss D thấp chỉ nói D đang thắng; G cũng có thể đơn giản chưa học, hoặc D quá mạnh nên gradient của G yếu. Phải đếm mode trên mẫu.
- Lab 3: khi không bao giờ có ∅, ε̂(x, ∅) chưa được huấn luyện, nhưng w = 0 không dùng nó nên vẫn ổn; w > 0 trừ đi một dự đoán rác nhân w. Fréchet đo cả độ phủ nên tăng khi mẫu co lại; hướng tâm chỉ đo độ sát vành nên giảm.
- Lab 4: sau ba vị trí đầu ngẫu nhiên, causal LM không có thông tin nào khác nên không thể tốt hơn ln 10 ở ba vị trí đó, suy ra sàn (3/7) ln 10. Giả thuyết (chưa kiểm): τ nhỏ ép phân biệt từng mẫu một, kể cả bằng chi tiết vô ích cho nhãn, nên retrieval cao hơn nhưng đặc trưng kém cho nhãn thô. Probe 0.928 của encoder ngẫu nhiên nghĩa là nhãn quá dễ, và probe 0.644 của τ = 0.05 thấp hơn mốc đó: huấn luyện contrastive ở τ này đã làm mất bớt thông tin nhãn so với không huấn luyện. Thí nghiệm này kiểm cơ chế, không kiểm chất lượng SSL.
- Lab 5: N · FID ≈ 2170 nên thiên lệch < 0.01 cần N > 217 000. Với d = 2048, d(d + 1)/2 + 2d = 2 102 272 ≈ 2.1 triệu, gấp khoảng 950 lần d = 64 (2208), nên cần cỡ 2.1 × 10⁸ mẫu: số 2170 chỉ đúng cho d = 64. Công thức dẫn xuất cho hai tập cùng N(0, I) và N lớn hơn d nhiều; phân phối đặc trưng thật (hiệp phương sai khác I, d = 2048 mà N chỉ cỡ vài chục nghìn) sẽ lệch khỏi nó.

**Mini-test (thang 30 điểm)**

1. Chiều 1: ½(0.25 + 1 − 1 − ln 0.25) = ½(0.25 + 1.3863) = 0.8181 (1.5 điểm). Chiều 2: ½(4 + 0 − 1 − ln 4) = 0.8069 (1.5 điểm). Tổng 1.625 (1 điểm).
2. D*(x) = 1/(1 + exp(x − 0.5)) = σ(0.5 − x) (2 điểm). D*(0) = σ(0.5) = 0.6225, D*(1.5) = σ(−1) = 0.2689 (1.5 điểm). D* = ½ tại x = 0.5 (0.5 điểm). max V = 2 · 0.1114 − ln 4 = −1.1635 (1 điểm).
3. αₜ = ᾱₜ/ᾱₜ₋₁ = 0.8, βₜ = 0.2 (1 điểm). Trung bình = (1.2 − (0.2/0.6) · 0.6)/√0.8 = 1.0/0.8944 = 1.1180 (2 điểm). β̃ₜ = (1 − 0.8)/(1 − 0.64) · 0.2 = 0.1111, độ lệch chuẩn 0.3333 (1.5 điểm). ε̃ = 3 · 0.6 − 2 · 0.1 = 1.6 (0.5 điểm).
4. Loss từng hàng ở τ = 0.5: 0.4075, 0.7824, 0.3829; trung bình 0.5242 (3 điểm). Cận I ≥ ln 3 − 0.5242 = 0.5744 nat (1 điểm). Hàng 2 khó nhất: dương 0.6 chỉ hơn âm 0.5 một chút (1 điểm).
5. (a) Masked: khoảng 150 vị trí; causal: 999 (2 điểm). (b) [MASK] không xuất hiện lúc tinh chỉnh, nên biểu diễn học được chỉ tối ưu cho đầu vào có ký hiệu đó: lệch pretrain–finetune (2 điểm). (c) Masked LM không định nghĩa xác suất của cả chuỗi (không có phân rã theo quy tắc chuỗi), nên không có exp của cross-entropy trên chuỗi (1 điểm).
6. Không kết luận được (2 điểm): FID thiên lệch theo cỡ mẫu (cùng phân phối vẫn dương, giảm theo 1/N), nên 1000 mẫu bị thổi phồng so với 50 000; hơn nữa còn cần cùng bản Inception và cách resize. (2 điểm) FID chỉ nhìn hai moment nên mù hình dạng (vòng tròn so với Gauss: 0.00018) và có thể thấp dù bỏ mode hoặc có cấu trúc sai. (2 điểm) Cần đo thêm nhiều chỉ số (precision/recall, đánh giá theo ứng dụng) và nhìn mẫu.

**Săn bug — bốn lỗi**

1. **Thiếu căn khi tạo xₜ.** `(1 - a) * eps` thay vì `(1 - a).sqrt() * eps`. Phương sai của xₜ không còn đúng lịch; mạng học trên phân phối sai ở mọi t. Chạy riêng lỗi này cho Fréchet 0.0055 (tốt hơn bản đúng 0.0303) nhưng hướng tâm 0.2359 (bản đúng 0.1253).
2. **Không bao giờ thay nhãn bằng ∅.** Thiếu `c[torch.rand(256) < 0.1] = 2` nên nhánh không điều kiện chưa được huấn luyện. Chạy riêng: w = 0 hoàn toàn ổn (Fréchet 0.0009), w = 3 sụp đổ (Fréchet 6.3826, hướng tâm 2.4035).
3. **Một ε dùng chung cho cả batch.** `torch.randn(1, 2)` thay vì `torch.randn(256, 2)`: mạng không thấy nhiễu đa dạng trong một bước cập nhật, gradient tương quan. Loss vẫn giảm. Chạy riêng: w = 0 cho Fréchet 0.4858, hướng tâm 0.2667, std lớp 0 (1.301, 0.622); w = 3 cho Fréchet 1.4440, hướng tâm 0.2237. Lỗi 3 gây hại hơn riêng lỗi 1 và 4 ở w = 0, và bốn lỗi cộng gộp (0.4319 / 0.4374 ở w = 0) không phải tổng các lỗi riêng: chúng tương tác.
4. **Lấy mẫu thiếu chia √αₜ.** `mean = x - ...` thay vì `(x - ...) / alpha[t].sqrt()`. Chạy riêng: Fréchet 0.2379, std lớp 0 (1.144, 0.561) thay vì (1.338, 0.622).

Tổ hợp cả bốn: w = 0 cho Fréchet 0.4319, hướng tâm 0.4374; w = 3 cho 7.7373 và 2.8009. Bài học: so nhiều chỉ số (lỗi 1 chỉ lộ ở hướng tâm), và kiểm cả w > 0 (lỗi 2 chỉ lộ ở đó).

```python
def train_fixed(steps=3000, seed=0):
    torch.manual_seed(seed); m = Net(); opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for i in range(steps):
        idx = torch.randint(0, 6000, (256,)); x0 = X[idx]; c = C[idx].clone()
        t = torch.randint(0, T, (256,)); eps = torch.randn(256, 2)            # sửa lỗi 3
        a = abar[t].unsqueeze(1)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps                            # sửa lỗi 1
        c[torch.rand(256) < 0.1] = 2                                         # sửa lỗi 2
        loss = ((m(xt, t, c) - eps) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m

# trong sample_fixed:
mean = (x - beta[t] / (1 - abar[t]).sqrt() * eps_hat) / alpha[t].sqrt()      # sửa lỗi 4
```

Đã chạy kiểm: bản sửa cho w = 0 Fréchet 0.0303, hướng tâm 0.1253; w = 3 Fréchet 0.2807, hướng tâm 0.0653.


## Tài liệu đào sâu và nguồn

Nội dung chương được viết lại, đối chiếu phạm vi kiến thức với các nguồn dưới đây. **Mức xác minh:** (A+) đã mở bản gốc và đối chiếu các khẳng định dùng trong chương (kể cả số bảng, số phương trình); (A) chỉ đọc qua bản tóm tắt trang, nên số mục, số bảng, số phương trình trong các bài đó **chưa đối chiếu bản gốc**; (B) chỉ nhớ từ literature hoặc chỉ thấy tiêu đề, chưa mở. Sách chỉ dẫn "tác giả (năm)", không dẫn số trang. Mở lại bài gốc trước khi trích số ngoài các bài (A+).

| Tài liệu | Dùng để làm gì | Phần nên đọc |
| --- | --- | --- |
| [Weng (2021), What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models) (A) | Bài đọc nhẹ nhất về DDPM, score, CFG | Toàn bài; phần liên hệ với score và phần tăng tốc |
| [Kingma và Welling (2019), An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691) (A) | Khe hở ELBO, collapse, free bits | Phần ELBO và phần mẹo huấn luyện |
| [Kingma và Welling (2014), Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (A) | Paper gốc VAE, reparameterization | Phần ELBO và bộ ước lượng; phụ lục KL Gauss chưa đọc (công thức KL được kiểm bằng Monte Carlo trong Lab 1) |
| [Bowman và cộng sự (2016), Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349) (A) | Posterior collapse trong VAE văn bản | Phần decoder bỏ qua z, KL annealing, word dropout |
| [Goodfellow và cộng sự (2014), Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) (A) | Paper gốc GAN, D* tối ưu, non-saturating | Phần phương pháp, phần lý thuyết, phần hạn chế |
| [Arjovsky và Bottou (2017), Towards Principled Methods for Training GANs](https://arxiv.org/abs/1701.04862) (A) | Vì sao gradient GAN biến mất | Các định lý về giá đỡ rời nhau |
| [Ho, Jain, Abbeel (2020), Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (A+) | Paper gốc DDPM | Quá trình tiến, thuật toán huấn luyện và lấy mẫu, bảng so FID và codelength |
| [Nichol và Dhariwal (2021), Improved DDPM](https://arxiv.org/abs/2102.09672) (A+) | Lịch cosine, phương sai học được | Phần lịch nhiễu |
| [Vincent (2011), A Connection Between Score Matching and Denoising Autoencoders](https://www.gwern.net/doc/ai/nn/diffusion/2011-vincent.pdf) (B) | Denoising score matching: vì sao hồi quy trên nhiễu ước lượng score của phân phối biên (mục 10.3, bài 11) | Phần kết nối score matching với khử nhiễu (bản PDF lưu ở gwern.net, chỉ thấy trong kết quả tìm kiếm, chưa đọc) |
| [Song và Ermon (2019), Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) (B) | Score nhiều mức nhiễu và lấy mẫu Langevin | Phần nhiều mức nhiễu |
| [Song và cộng sự (2021), Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) (A) | DDPM và score matching là một | Phần SDE thuận và ngược |
| [Ho và Salimans (2021), Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) (A+) | CFG, đánh đổi FID–IS theo w (arXiv đăng 2022; bản workshop NeurIPS 2021, nên sách ghi 2021) | Thuật toán và phần chọn xác suất bỏ nhãn |
| [Dhariwal và Nichol (2021), Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) (A) | Classifier guidance, precision và recall | Phần guidance và bảng precision/recall |
| [van den Oord, Li, Vinyals (2018), Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) (A) | InfoNCE và cận thông tin tương hỗ | Phần hàm mất mát và phép dẫn cận (có một bước xấp xỉ; chứng minh chặt: Poole và cộng sự, 2019, [cần kiểm chứng]) |
| [Chen và cộng sự (2020), SimCLR](https://arxiv.org/abs/2002.05709) (A+ cho τ) | NT-Xent, nhiệt độ, kích thước batch | Phần hàm mất mát; Bảng 5 (τ) và phụ lục B.9 (CIFAR-10) |
| [Gao, Yao, Chen (2021), SimCSE](https://arxiv.org/abs/2104.08821) (A+ cho τ) | Contrastive cho câu, dropout làm tăng cường | Phần học không giám sát; phụ lục D, Bảng D.1 (τ) |
| [Devlin và cộng sự (2019), BERT](https://arxiv.org/abs/1810.04805) (A) | Masked LM, quy tắc 80/10/10 | Phần tiền huấn luyện |
| [Heusel và cộng sự (2017), GANs Trained by a Two Time-Scale Update Rule](https://arxiv.org/abs/1706.08500) (A) | Định nghĩa FID | Phần định nghĩa FID |
| [Theis, van den Oord, Bethge (2016), A note on the evaluation of generative models](https://arxiv.org/abs/1511.01844) (A) | Vì sao không có một thước đo | Toàn bài |
| [Higgins và cộng sự (2017), β-VAE](https://mlanthology.org/iclr/2017/higgins2017iclr-beta) (B) | β-VAE | Phần phương pháp |
| [Kynkäänniemi và cộng sự (2019), Improved Precision and Recall Metric](https://arxiv.org/abs/1904.06991) (B) | Precision và recall cho mô hình sinh | Phần định nghĩa |

**Từ khóa để tự tìm thêm:** amortized inference, β-VAE, VQ-VAE, free bits, Helvetica scenario, Wasserstein GAN, spectral normalization, denoising score matching, noise schedule, v-prediction, DDIM, latent diffusion, Diffusion-LM, D3PM, alignment and uniformity, false negatives, span corruption, replaced token detection (ELECTRA), pseudo-perplexity, dequantization, precision and recall for generative models.

**Trước khi sang chương 11:** đạt từ 24/30 điểm mini-test, làm lại các bài ★★★ sai, tự viết được `kl_gauss` cùng `reparam` và `infonce` không nhìn code mẫu, và tìm được cả bốn lỗi của bài săn bug.

Nguồn tra cứu ngày 30/10/2026, bổ sung ngày 04/11/2026.
