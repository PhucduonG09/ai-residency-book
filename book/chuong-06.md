# Chương 6 — Mạng nơ-ron từ con số 0

## Mục tiêu và kiểm tra nhanh

Chương này ghép những gì bạn đã có thành một mạng nơ-ron chạy được. Chương 2 đã dạy backprop dạng ma trận, chương 4 đã dạy SGD, momentum, Adam và chuyện L2 khác AdamW. Ở đây bạn tự viết một MLP bằng NumPy, viết lại nó bằng PyTorch, làm cho mạng sâu huấn luyện được (khởi tạo, normalization, regularization), rồi học cách chẩn đoán khi vòng huấn luyện chạy không báo lỗi mà vẫn sai. Phần cuối là phần người phỏng vấn hay đào nhất: không ai hỏi bạn viết `loss.backward()`, họ hỏi bạn làm gì khi loss đứng yên ở 2.30. Thời lượng: khoảng 2 tuần, 8 giờ mỗi tuần.

**Sau chương này, bạn phải làm được:**

- ☐ Giải thích vì sao mạng cần phi tuyến, mạng ReLU biểu diễn loại hàm gì, và định lý xấp xỉ phổ quát nói gì, không nói gì.
- ☐ Tự viết forward, backward và gradient check cho MLP softmax–cross-entropy theo batch bằng NumPy, đúng tới sai số máy.
- ☐ Viết vòng huấn luyện PyTorch chuẩn (`Dataset`, `DataLoader`, `nn.Module`, train/eval, `no_grad`) và giải thích từng dòng.
- ☐ Dẫn xuất khởi tạo Xavier và He bằng lập luận phương sai; so sánh BatchNorm với LayerNorm theo trục chuẩn hóa và theo hành vi train/eval; dùng đúng dropout và weight decay.
- ☐ Chạy checklist debug: loss ban đầu, overfit một batch, grad norm, quét learning rate, và nhận ra các lỗi im lặng kinh điển.

**Kiểm tra nhanh (5 phút, không dùng máy tính).** Câu i kiểm tra mục 6.i: đúng thì đọc lướt mục đó, sai thì đọc kỹ. Đúng cả 5 câu thì làm thẳng bài ★★★ và mini-test.

1. Mạng 100 → 5 → 100 gồm hai lớp Linear, không có hàm kích hoạt. Nó tương đương với phép biến đổi nào, và hạng tối đa là bao nhiêu?
2. Một lớp ReLU có z = (−1, 0.5, 2) và gradient đổ về từ phía trên là ∂L/∂h = (3, 3, 3). Tính ∂L/∂z.
3. Trong PyTorch, bạn gọi forward + `backward()` hai lần trên cùng một batch mà không gọi `zero_grad()`. `w.grad` khi đó là gì?
4. Lớp Linear có fan-in nᵢₙ = 512, theo sau là ReLU. Khởi tạo He cho độ lệch chuẩn trọng số bằng bao nhiêu?
5. Bài phân loại 100 lớp, mạng vừa khởi tạo. Loss cross-entropy ban đầu nên xấp xỉ bao nhiêu?

Đáp án ở mục Lời giải cuối chương.

## 6.1 MLP, hàm kích hoạt và vì sao cần phi tuyến

Một MLP (multilayer perceptron, mạng nhiều lớp kết nối đầy đủ) là hợp của các lớp affine xen kẽ với hàm phi tuyến. Bỏ phi tuyến đi thì bao nhiêu lớp cũng chỉ bằng một lớp.

**Neuron, lớp, mạng.** Một neuron tính tích vô hướng của đầu vào với vector trọng số, cộng bias, rồi đưa qua hàm kích hoạt φ. Gom m neuron lại là một ma trận W ∈ ℝᵐˣⁿ: mỗi hàng là một neuron (cách đọc theo hàng của chương 1). Mạng hai lớp là:

$$
h = \varphi(W_1 x + b_1), \qquad \hat{y} = W_2 h + b_2
$$

**Sụp đổ tuyến tính.** Nếu φ là hàm đồng nhất thì

$$
W_2(W_1 x + b_1) + b_2 = (W_2 W_1)\,x + (W_2 b_1 + b_2)
$$

tức đúng một lớp affine với trọng số W₂W₁. Thêm nữa, rank(W₂W₁) ≤ min(rank W₁, rank W₂) (chương 1): lớp hẹp nhất là nút cổ chai. Xếp thêm lớp tuyến tính không thêm năng lực biểu diễn, chỉ thêm tham số.

**Ví dụ giải mẫu 6.1 — hai lớp tuyến tính sụp thành một.** W₁ ∈ ℝ³ˣ² có các hàng (1, 2), (0, 1), (−1, 1); W₂ = (1, 0, 2) (một hàng).

- W₂W₁ = 1·(1, 2) + 0·(0, 1) + 2·(−1, 1) = (−1, 4).
- Mạng 2 → 3 → 1 không kích hoạt đúng bằng một neuron tuyến tính có trọng số (−1, 4). Ba neuron ẩn không giúp gì.
- Kiểm bằng thực nghiệm: mạng Linear 1 → 64 → 64 → 1 không kích hoạt, huấn luyện 2000 bước Adam để khớp sin(2x) trên lưới 256 điểm cách đều trong \[−π, π\], đạt MSE 0.4233 — trùng tới 4 chữ số với MSE của đường thẳng bình phương tối thiểu trên cùng lưới (0.4233; lưới 512 điểm cho 0.4236, nên đừng so số giữa hai lưới). Hơn 4 000 tham số mà làm được đúng bằng 2 tham số.

% manim: SupDoTuyenTinh

**Các hàm kích hoạt bạn phải nói được ưu nhược.**

| Hàm | Công thức | Đạo hàm | Ưu | Nhược |
| --- | --- | --- | --- | --- |
| Sigmoid σ | 1/(1 + e⁻ˣ) | σ(1 − σ) ≤ 0.25 | Ra xác suất (0, 1) | Bão hòa hai đầu, không tâm 0; qua nhiều lớp gradient nhân với số ≤ 0.25 |
| tanh | 2σ(2x) − 1 | 1 − tanh² ≤ 1 | Tâm 0 | Vẫn bão hòa khi \|x\| lớn |
| ReLU | max(0, x) | 1 nếu x > 0, 0 nếu x < 0 | Rẻ, không bão hòa phía dương, gradient không bị co | "Neuron chết": z < 0 với mọi mẫu thì gradient luôn 0 |
| Leaky ReLU | max(αx, x), α nhỏ như 0.01 | α hoặc 1 | Không có neuron chết hoàn toàn | Thêm một siêu tham số |
| GELU | x·Φ(x), Φ là CDF chuẩn | trơn | Dùng trong BERT, GPT-2 | Đắt hơn ReLU một chút |
| SwiGLU | Swish(W₁x) ⊙ (W₃x), Swish(x) = x·σ(x) | trơn, có cổng | Dùng trong FFN của nhiều LLM mở hiện nay (vd họ LLaMA) | Cần 3 ma trận thay vì 2 trong khối FFN |

ReLU không khả vi tại 0. Framework chọn một subgradient (gradient dưới, bất kỳ giá trị nào trong \[0, 1\] đều hợp lệ); PyTorch lấy giá trị 0. Trong thực tế z rơi đúng vào 0 rất hiếm với số thực, nên lựa chọn này hầu như không ảnh hưởng — trừ khi bạn khởi tạo bias và trọng số bằng 0.

**Mạng ReLU là hàm tuyến tính từng khúc.** Mỗi neuron ReLU một chiều ReLU(wx + b) gãy tại x = −b/w. Tổng có trọng số của m neuron như vậy là một đường gấp khúc có tối đa m điểm gãy. Với đầu vào nhiều chiều, các siêu phẳng wᵢᵀx + bᵢ = 0 chia không gian thành các vùng, và trên mỗi vùng mạng là một hàm affine.

**Ví dụ giải mẫu 6.2 — XOR bằng 2 neuron ReLU.** XOR không tách tuyến tính được: (0, 1) và (1, 0) có nhãn 1, (0, 0) và (1, 1) có nhãn 0. Lấy

$$
h_1 = \mathrm{ReLU}(x_1 + x_2), \qquad h_2 = \mathrm{ReLU}(x_1 + x_2 - 1), \qquad y = h_1 - 2h_2
$$

- (0, 0): h = (0, 0), y = 0.
- (0, 1) và (1, 0): h = (1, 0), y = 1.
- (1, 1): h = (2, 1), y = 2 − 2 = 0.
- Trong không gian (h₁, h₂), hai điểm nhãn 1 trùng nhau tại (1, 0), còn (1, 1) bị đẩy tới (2, 1). Đường y = 0.5 tách được. Lớp ẩn bẻ cong không gian để lớp cuối chỉ cần tuyến tính.

% manim: XORHaiNeuron

**Định lý xấp xỉ phổ quát (universal approximation).** Cybenko (1989) chứng minh cho hàm kích hoạt sigmoid: với mọi hàm liên tục trên một miền compact và mọi sai số ε > 0, **tồn tại** một mạng một lớp ẩn đủ rộng xấp xỉ hàm đó trong phạm vi ε. Kết quả không dừng ở sigmoid: Leshno, Lin, Pinkus và Schocken (1993) chứng minh mạng một lớp ẩn xấp xỉ phổ quát khi và chỉ khi hàm kích hoạt không phải đa thức (với điều kiện kỹ thuật nhẹ), nên ReLU cũng được phủ.

Trực giác với ReLU, do người viết dựng lại: ReLU(x − a) − ReLU(x − b) với a < b là một **đoạn dốc** (ramp), bằng 0 trước a, tăng tuyến tính từ a tới b, rồi phẳng ở mức b − a. Lấy một ramp lên và một ramp xuống (bốn ReLU) thì được một bướu hình thang; chồng đủ nhiều bướu hẹp thì xấp xỉ được hàm liên tục bất kỳ. Khi a và b rất sát nhau và hệ số đủ lớn, ramp gần như một bậc thang. Cách dựng bằng bậc thang thì Nielsen (chương 4) làm với sigmoid có trọng số lớn; bản ReLU ở trên là tương tự của người viết [cần kiểm chứng: chưa mở lại trang Nielsen để xem ông có bài tập ReLU hay không].

Định lý **không nói** ba điều: cần bao nhiêu neuron, gradient descent có tìm ra trọng số đó không, và mạng có tổng quát hóa ra ngoài dữ liệu huấn luyện không.

**Ví dụ giải mẫu 6.3 — tuyến tính từng khúc xấp xỉ tốt đến đâu, và tối ưu tìm được đến đâu.** Nội suy tuyến tính sin x trên \[0, π\] bằng K khúc đều nhau (mỗi điểm gãy ứng với khoảng một neuron ReLU):

| K | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- |
| Sai số lớn nhất | 0.2105 | 0.0704 | 0.0188 | 0.0048 | 0.0012 |

Mỗi lần gấp đôi K, sai số giảm khoảng 4 lần, đúng bậc O(1/K²) của nội suy tuyến tính hàm trơn. Đó là chuyện **tồn tại**. Giờ xem chuyện **tìm được**: huấn luyện thật một lớp ẩn m neuron ReLU khớp sin(2x) (Adam lr 1e-2, 2000 bước; `nn.Sequential(nn.Linear(1, m), nn.ReLU(), nn.Linear(m, 1))` với init mặc định của `nn.Linear`, đặt `torch.manual_seed(seed)` ngay trước khi tạo mạng; 256 điểm dữ liệu cách đều trên \[−π, π\]). Với seed 0: m = 2 → MSE 0.498; m = 8 → 0.332; m = 32 → 0.00177; m = 128 → 0.00018. Với m = 8, MSE 0.332 không tốt hơn đường thẳng bao nhiêu (0.4233 trên cùng lưới), trong khi chỉ cần cố định 7 điểm gãy cách đều ở bên trong khoảng rồi giải bình phương tối thiểu cho hệ số (cộng thêm số hạng hằng và x) đã được MSE 0.0065. Chạy m = 8 với 5 seed (0 đến 4): MSE 0.332, 0.073, ≈ 0.005, 0.073, 0.073; seed 0 có 3 trong 8 neuron chết (không bật với điểm nào trên miền). Với dữ liệu hoặc cách khởi tạo khác, kết quả theo seed sẽ khác, nên "3 trong 8 neuron chết" là kết luận có điều kiện cho đúng cấu hình này. Kết luận đã đo: cùng một kiến trúc, kết quả phụ thuộc mạnh vào khởi tạo; neuron chết và điểm gãy nằm sai chỗ là một phần nguyên nhân.

% manim: XapXiTungKhuc

**Vì sao ML cần cái này.**

- Mọi kiến trúc bạn sẽ gặp (Transformer, CNN) đều là các khối affine xen phi tuyến. Khối feed-forward (FFN) của Transformer là một MLP hai lớp: ReLU ở bản gốc 2017, GELU ở BERT và GPT-2, còn nhiều LLM mở hiện nay dùng biến thể có cổng SwiGLU (ba ma trận thay vì hai).
- Câu "mạng sâu thì mạnh hơn" cần sửa lại: mạng sâu có phi tuyến mới mạnh hơn. Hai lớp Linear liền nhau (không có gì xen giữa) chỉ là phân tích nhân tử hạng thấp, như LoRA ở chương 1.
- Khoảng cách "tồn tại" và "tìm được" là lý do của cả mục 6.4 và 6.5: phần lớn công sức thực tế dành cho việc làm cho tối ưu tìm được nghiệm.

## 6.2 Tự viết forward/backward cho MLP bằng NumPy

Backprop cho MLP chỉ là chương 2 áp theo thứ tự ngược: cache những gì forward tính ra, rồi đi từ loss ngược về, mỗi bước nhân với Jacobian cục bộ. Cách nhanh nhất để không đặt sai chuyển vị là kiểm shape.

**Quy ước batch.** Mỗi hàng là một mẫu: X ∈ ℝᴺˣᵈ, W₁ ∈ ℝᵈˣᵐ, W₂ ∈ ℝᵐˣᶜ với m là độ rộng lớp ẩn, còn H dưới đây là ma trận activation (đây là chuyển vị của quy ước "mỗi hàng một neuron" ở 6.1; `nn.Linear` lưu `weight` theo quy ước 6.1 và tính XWᵀ). Y ∈ ℝᴺˣᶜ là nhãn one-hot.

$$
Z_1 = XW_1 + b_1,\quad H = \mathrm{ReLU}(Z_1),\quad Z_2 = HW_2 + b_2,\quad P = \mathrm{softmax}(Z_2),\quad L = -\frac{1}{N}\sum_{n=1}^{N} \log P_{n, y_n}
$$

**Backward**, theo đúng thứ tự ngược và kèm shape:

$$
\begin{aligned}
dZ_2 &= (P - Y)/N &&(N \times C) \\
dW_2 &= H^{\top} dZ_2,\quad db_2 = \textstyle\sum_n dZ_2[n,:] &&(m \times C),\ (C) \\
dH &= dZ_2 W_2^{\top} &&(N \times m) \\
dZ_1 &= dH \odot \mathbb{1}[Z_1 > 0] &&(N \times m) \\
dW_1 &= X^{\top} dZ_1,\quad db_1 = \textstyle\sum_n dZ_1[n,:] &&(d \times m),\ (m)
\end{aligned}
$$

Ba điểm cần nhớ. (1) Gradient gọn P − Y chỉ có được khi gộp softmax với cross-entropy; tính riêng Jacobian của softmax rồi nhân là đúng nhưng chậm và kém ổn định. (2) Phép chia N đặt một lần ở dZ₂ là đủ, vì mọi gradient sau đều tuyến tính theo nó. (3) Bias nhận tổng theo batch vì nó được "phát sóng" (broadcast) cho mọi mẫu.

**Softmax ổn định số.** softmax(z) = softmax(z − c) với mọi hằng số c. Trừ max của mỗi hàng trước khi lấy exp thì số lớn nhất là e⁰ = 1, không bao giờ tràn. Với logit cỡ 1000, exp trực tiếp ra inf và loss ra NaN.

**Ví dụ giải mẫu 6.4 — forward/backward bằng tay cho mạng 2–3–2 (một mẫu).** Dùng quy ước cột cho dễ đọc: x = (1, 2), nhãn y = 0. W₁ có các hàng (0.5, −1), (1, 0.5), (−0.5, 0.5); b₁ = (0, 0, 1). W₂ có các hàng (1, −1, 0.5), (0.5, 1, −1); b₂ = 0.

- z₁ = W₁x + b₁ = (0.5 − 2, 1 + 1, −0.5 + 1 + 1) = (−1.5, 2, 1.5); h = ReLU(z₁) = (0, 2, 1.5).
- z₂ = W₂h = (0 − 2 + 0.75, 0 + 2 − 1.5) = (−1.25, 0.5).
- p = softmax(z₂) = (0.148, 0.852), vì e⁻¹·²⁵/(e⁻¹·²⁵ + e⁰·⁵) = 1/(1 + e¹·⁷⁵) ≈ 0.148. L = −ln 0.148 = 1.9102.
- dz₂ = p − e₀ = (−0.852, 0.852). dW₂ = dz₂hᵀ có các hàng (0, −1.7039, −1.2779) và (0, 1.7039, 1.2779).
- dh = W₂ᵀdz₂ = (−0.426, 1.7039, −1.2779). dz₁ = dh ⊙ 1\[z₁ > 0\] = (0, 1.7039, −1.2779): neuron 1 có z = −1.5 nên chặn gradient.
- dW₁ = dz₁xᵀ có các hàng (0, 0), (1.7039, 3.4078), (−1.2779, −2.5559). Hàng của neuron tắt bằng 0: nếu neuron này tắt với **mọi** mẫu, nó không bao giờ học lại — đó là neuron chết.
- Gradient check (bên dưới) với δ = 1e-5, float64: sai số tương đối 1.8e-11.

% manim: LanNguocMLP

**Gradient check.** So gradient giải tích với sai phân trung tâm, theo sai số tương đối:

$$
\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta + \delta e_i) - L(\theta - \delta e_i)}{2\delta}, \qquad \text{rel} = \frac{\lvert g_{\text{giải tích}} - g_{\text{số}} \rvert}{\max(\lvert g_{\text{giải tích}} \rvert, \lvert g_{\text{số}} \rvert)}
$$

Sai phân trung tâm với bước δ có sai số O(δ²), còn sai phân một phía chỉ O(δ). Dùng float64: với float32, sai số làm tròn chia cho δ đã cỡ 1e-7/1e-5 = 1e-2, lẫn hết vào tín hiệu. Ngưỡng tham khảo của CS231n: rel khoảng 1e-7 trở xuống là tốt; trên 1e-2 gần như chắc chắn sai; khoảng 1e-4 chỉ chấp nhận được khi hàm có điểm gãy (ReLU), vì sai phân có thể "nhảy qua" điểm gãy. Mẹo: kiểm trên batch nhỏ, ít chiều, và tắt dropout.

**Ví dụ giải mẫu 6.5 — đếm tham số và bộ nhớ.** MLP 784–256–10 (MNIST):

- 784·256 + 256 + 256·10 + 10 = 200 704 + 256 + 2 560 + 10 = 203 530 tham số.
- float32 (4 byte): 814 120 byte ≈ 0.81 MB cho trọng số.
- Huấn luyện bằng Adam cần 4 mảng cùng kích thước: trọng số, gradient và hai moment (bậc nhất, bậc hai) ⇒ 3 256 480 byte ≈ 3.26 MB, chưa tính activation (tỉ lệ với batch size). Cách nhân "16 byte mỗi tham số" này dùng để ước lượng VRAM cho mô hình lớn ở chương sau.

**Vì sao ML cần cái này.**

- Câu phỏng vấn kinh điển "viết forward/backward cho mạng 2 lớp bằng NumPy" kiểm đúng mục này. Viết được trong 15 phút, không tra cứu, là mức tối thiểu.
- Khi tự viết một lớp mới (attention biến thể, loss lạ) mà không có autograd sẵn, gradient check là cách duy nhất để tin code.
- Hiểu cache là hiểu bộ nhớ: activation phải giữ lại cho backward, nên batch lớn tốn VRAM chủ yếu vì activation chứ không phải vì trọng số.

**Thực tế tính toán.** Trong thư viện thật, softmax và cross-entropy được gộp thành log-softmax (log-sum-exp ổn định) rồi lấy phần tử của nhãn; không ai tính P rồi mới lấy log. Gradient check chỉ chạy trên mạng tí hon, vì mỗi tham số cần hai lần forward.

## 6.3 Chuyển sang PyTorch: tensor, autograd, nn.Module, DataLoader, vòng lặp chuẩn

PyTorch làm đúng những gì bạn vừa viết tay, chỉ khác là nó tự ghi lại đồ thị tính toán lúc forward và tự lan ngược. Phải nắm vài quy tắc của autograd, nếu không sẽ gặp lỗi im lặng.

**Tensor và autograd.** Tensor là ndarray cộng thêm thiết bị (CPU/GPU) và khả năng ghi đồ thị. Tensor có `requires_grad=True` là lá (leaf) của đồ thị; mỗi phép toán trên nó tạo một nút có `grad_fn`. Gọi `.backward()` trên một **scalar** sẽ lan ngược và **cộng** gradient vào `.grad` của các lá (tensor trung gian không giữ `.grad` trừ khi gọi `retain_grad()`). Đồ thị là động: xây lại ở mỗi lần forward, và mặc định bị giải phóng sau một lần backward (muốn backward lần hai trên cùng đồ thị phải truyền `retain_graph=True`).

**Ví dụ giải mẫu 6.6 — gradient cộng dồn.** w = 2 (`requires_grad=True`), x = 3, L = (wx − 1)².

- dL/dw = 2(wx − 1)x = 2·5·3 = 30. Sau `L.backward()` lần đầu, `w.grad` = 30.
- Tính lại L rồi `backward()` lần nữa mà không xóa: `w.grad` = 60.
- Đây chính là lỗi "quên `zero_grad()`": mỗi bước dùng tổng gradient của mọi bước trước. Cũng chính cơ chế này là **gradient accumulation**: khi VRAM chỉ chứa nổi batch 8 mà muốn batch hiệu dụng 32, gọi backward 4 lần (loss chia 4) rồi mới `step()` và `zero_grad()`.

**nn.Module.** Khai báo các lớp con trong `__init__`, mô tả tính toán trong `forward`. Mọi `nn.Parameter` của lớp con được đăng ký tự động, nên `model.parameters()` đưa thẳng vào optimizer và `model.state_dict()` lưu được toàn bộ trọng số. Với mạng chỉ là một chuỗi lớp, `nn.Sequential` là đủ.

**Dữ liệu.** Một `Dataset` chỉ cần `__len__` và `__getitem__`; `TensorDataset` bọc sẵn các tensor có cùng số hàng. `DataLoader` cắt thành batch và, với `shuffle=True`, xáo trộn lại mỗi epoch. Nếu dữ liệu xếp theo lớp mà quên shuffle, mỗi batch chỉ có một lớp và mạng bị kéo qua lại giữa các lớp.

**Vòng lặp chuẩn.** Thuộc lòng năm dòng giữa:

```python
for epoch in range(E):
    model.train()                       # bật dropout, BN dùng thống kê batch
    for xb, yb in loader:
        opt.zero_grad()                 # xóa gradient cũ
        loss = F.cross_entropy(model(xb), yb)   # model trả LOGIT thô
        loss.backward()                 # lan ngược, CỘNG vào .grad
        opt.step()                      # cập nhật theo .grad
    model.eval()                        # tắt dropout, BN dùng running stats
    with torch.no_grad():               # không ghi đồ thị: nhanh hơn, ít bộ nhớ
        val_loss = F.cross_entropy(model(Xva), yva).item()
```

`nn.CrossEntropyLoss` (và `F.cross_entropy`) nhận **logit thô**: bên trong nó đã là LogSoftmax + NLLLoss, mặc định lấy trung bình theo batch (`reduction='mean'`). Đưa xác suất vào là lỗi im lặng (mục 6.5).

**Ví dụ giải mẫu 6.7 — tái hiện ví dụ 6.4 bằng PyTorch.** Tạo `nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 2))`, chép đúng W₁, b₁, W₂, b₂ của ví dụ 6.4 vào bằng `with torch.no_grad(): m[0].weight.copy_(...)`. Gọi `F.cross_entropy(m(x), y)` với x = \[\[1, 2\]\], y = \[0\] rồi `backward()`:

- loss = 1.9102, khớp tay.
- `m[0].weight.grad` có các hàng (0, 0), (1.7039, 3.4078), (−1.2779, −2.5559), khớp dW₁ của ví dụ 6.4 từng chữ số. Shape là (3, 2): `nn.Linear` lưu trọng số theo quy ước "mỗi hàng một neuron".
- Lab 2 làm phép đối chiếu này trên mạng 20–128–10 đã huấn luyện, batch 64: loss NumPy và PyTorch trùng nhau tới 10 chữ số thập phân.

**Vì sao ML cần cái này.**

- Gần như mọi bug thực tế của người mới nằm ở năm dòng của vòng lặp: thứ tự `zero_grad`/`backward`/`step`, train/eval, `no_grad`.
- Gradient accumulation là cách chuẩn để fine-tune mô hình ngôn ngữ trên GPU 6 GB của bạn.
- Đọc được `grad_fn` và kiểm `requires_grad` giúp thấy tensor còn dính vào đồ thị (vd `total_loss` có `requires_grad=True`); thói quen đúng là `loss.item()` hoặc `loss.detach()` khi chỉ cần ghi lại số.

**Thực tế tính toán.** `torch.no_grad()` chỉ tiết kiệm bộ nhớ và thời gian; quên nó khi đánh giá không làm sai con số. Quên `model.eval()` thì có làm sai con số. Hai việc này khác nhau và hay bị nhầm.

## 6.4 Khởi tạo trọng số và regularization: Xavier/He, BatchNorm, LayerNorm, dropout, weight decay

Mạng sâu chỉ huấn luyện được khi tín hiệu forward và gradient backward không tắt cũng không nổ qua các lớp. Khởi tạo đúng và normalization là hai cách giữ thang đo đó; dropout và weight decay là chuyện khác: chống overfitting.

**Vì sao không khởi tạo toàn 0 (hay cùng một hằng số).** Nếu mọi neuron trong một lớp có cùng trọng số vào và cùng trọng số ra, chúng tính ra cùng activation, nhận cùng gradient, được cập nhật giống nhau, và mãi mãi giống nhau (bài tập 9 chứng minh bằng quy nạp). Lớp m neuron khi đó chỉ làm việc của một neuron. Khởi tạo ngẫu nhiên là để **phá đối xứng**. Riêng init toàn 0 còn tệ hơn: W₂ = 0 làm gradient đổ về lớp ẩn dH = 0, nên W₁ không nhận gradient ngay từ đầu (đúng với cả tanh; với ReLU, đạo hàm tại 0 bằng 0 chỉ là lý do phụ) (lab 3 dùng hằng số 0.3 để thấy rõ hiện tượng đối xứng mà gradient vẫn khác 0).

**Lập luận phương sai.** Xét z = ∑ wᵢxᵢ với nᵢₙ số hạng, wᵢ độc lập với xᵢ, 𝔼\[wᵢ\] = 0:

$$
\operatorname{Var}(z) = n_{\text{in}} \cdot \operatorname{Var}(w) \cdot \mathbb{E}[x^2]
$$

- Lớp tuyến tính (hoặc tanh quanh 0), đầu vào tâm 0: 𝔼\[x²\] = Var(x). Giữ Var(z) = Var(x) ⇒ Var(w) = 1/nᵢₙ.
- Backward có cùng lập luận nhưng với nₒᵤₜ. Xavier/Glorot (2010) lấy trung bình hai điều kiện: Var(w) = 2/(nᵢₙ + nₒᵤₜ); bản uniform có biên √(6/(nᵢₙ + nₒᵤₜ)).
- ReLU: nếu z của lớp trước đối xứng quanh 0 thì x = ReLU(z) có 𝔼\[x²\] = ½Var(z) — ReLU giữ lại đúng một nửa năng lượng. Để bù, He/Kaiming (2015) chọn Var(w) = 2/nᵢₙ, tức gain √2.

$$
\text{Xavier: } \operatorname{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}, \qquad \text{He (ReLU): } \operatorname{Var}(w) = \frac{2}{n_{\text{in}}}, \qquad \texttt{nn.Linear}\text{ mặc định: } w \sim U\!\left(-\tfrac{1}{\sqrt{n_{\text{in}}}}, \tfrac{1}{\sqrt{n_{\text{in}}}}\right)
$$

Mặc định của `nn.Linear` có Var(w) = 1/(3nᵢₙ), tức std ≈ 1/√(3·nᵢₙ). Đây **không phải** He. Với mạng nông thì không sao; với mạng sâu không có normalization thì tín hiệu co dần: mỗi lớp ReLU nhân std với khoảng √(nᵢₙ · 1/(3nᵢₙ) · ½) = √(1/6) ≈ 0.41. `nn.init.calculate_gain('relu')` trả √2 ≈ 1.414, `'tanh'` trả 5/3.

**Ví dụ giải mẫu 6.8 — tín hiệu qua 10 lớp ReLU.** Đầu vào chuẩn 1000 × 512, 10 lớp rộng 512, không bias, NumPy seed 0. Std của activation sau lớp 1 / lớp 5 / lớp 10:

| Khởi tạo | Lớp 1 | Lớp 5 | Lớp 10 |
| --- | --- | --- | --- |
| std w = 0.01 | 0.132 | 8.3e-5 | 8.0e-9 |
| Xavier, Var = 2/(512 + 512) = 1/512 | 0.585 | 0.153 | 0.0276 |
| He, Var = 2/512 | 0.827 | 0.743 | 0.758 |

- std 0.01: mỗi lớp nhân std với √(512 · 10⁻⁴ · ½) = 0.16, nên sau 10 lớp tín hiệu gần như bằng 0. Chú ý: "std" trong bảng là độ lệch chuẩn thật của activation (nhỏ hơn RMS vì đầu ra ReLU có trung bình khác 0), còn hệ số ½ trong các phép tính là của 𝔼[x²].
- Xavier: hệ số mỗi lớp đo được là căn bậc 9 của 0.0276/0.585, tức ≈ 0.71 ≈ 1/√2, đúng phần ReLU bỏ đi (ước lượng từ một seed; chạy nhiều seed thì std ở lớp 10 dao động khoảng 0.62–1.04 với He và 0.019–0.032 với Xavier). Xavier được dẫn xuất cho mạng tuyến tính, không cho ReLU.
- He: giữ ổn định.
- Với tanh thì vấn đề ngược lại cũng có: 10 lớp tanh rộng 512 với std w = 1 làm 90.5% activation có \|h\| > 0.99 (bão hòa, đạo hàm gần 0); std 0.01 làm std activation còn 3.4e-7.
- PyTorch, 20 lớp `nn.Linear(256, 256)` + ReLU (lab 3): init mặc định cho std cuối 0.023, `kaiming_normal_` + bias 0 cho 0.878. Con số 0.023 chủ yếu do bias mặc định giữ lại: cùng thí nghiệm mà bỏ bias thì std cuối cỡ 1e-8 (notebook đo 1.0e-8), nghĩa là tín hiệu từ đầu vào đã tắt.

% manim: TinHieuQuaLop

**BatchNorm (Ioffe & Szegedy, 2015).** Với mỗi đặc trưng (mỗi cột của batch), chuẩn hóa theo batch rồi co giãn bằng tham số học được:

$$
\mu_B = \frac{1}{N}\sum_n x_n,\quad \sigma_B^2 = \frac{1}{N}\sum_n (x_n - \mu_B)^2,\quad \hat{x}_n = \frac{x_n - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}},\quad y_n = \gamma \hat{x}_n + \beta
$$

Có hai chế độ. Lúc train: dùng μ và σ² của batch hiện tại (phương sai **lệch**, chia N), đồng thời cập nhật running mean/var với momentum 0.1: running ← 0.9·running + 0.1·giá trị batch, trong đó running_var dùng phương sai **không lệch** (chia N − 1). Lúc eval: dùng running stats, nên đầu ra của một mẫu không phụ thuộc các mẫu khác. γ, β cho phép mạng học lại thang đo nếu chuẩn hóa không có lợi.

**LayerNorm (Ba, Kiros, Hinton, 2016).** Cùng công thức nhưng thống kê lấy trên **các đặc trưng của từng mẫu** (một hàng). Không phụ thuộc batch, nên train và eval giống hệt nhau và batch 1 không có vấn đề gì.

| | BatchNorm | LayerNorm |
| --- | --- | --- |
| Thống kê lấy trên | Batch, cho từng đặc trưng (cột) | Đặc trưng, cho từng mẫu (hàng) |
| Train khác eval | Có (batch stats vs running stats) | Không |
| Batch nhỏ hoặc 1 | Thống kê nhiễu; lúc train báo lỗi khi mỗi kênh chỉ có 1 giá trị (vd `BatchNorm1d` với đầu vào (1, C)) | Không ảnh hưởng |
| Chuỗi độ dài thay đổi, padding | Khó: thống kê trộn lẫn token đệm | Tự nhiên, từng token một |
| Dùng nhiều ở | CNN thị giác | Transformer, RNN |

**Ví dụ giải mẫu 6.9 — một bước BatchNorm và LayerNorm.**

- BN, một đặc trưng, batch (1, 2, 3, 6): μ = 3, phương sai lệch (4 + 1 + 0 + 9)/4 = 3.5, x̂ = (−2, −1, 0, 3)/√3.5 = (−1.069, −0.535, 0, 1.604).
- Running stats sau bước này (khởi đầu mean 0, var 1): running_mean = 0.1·3 = 0.3; phương sai không lệch 14/3 ≈ 4.667, running_var = 0.9·1 + 0.1·4.667 = 1.3667. `nn.BatchNorm1d` cho đúng 0.3 và 1.3667 (lab 4).
- LN trên một mẫu (2, 4, 6, 8): mean 5, phương sai (9 + 1 + 1 + 9)/4 = 5, ra (−1.342, −0.447, 0.447, 1.342).
- BN khi mỗi kênh chỉ có 1 giá trị ở chế độ train (vd `BatchNorm1d` với đầu vào (1, C)): phương sai batch bằng 0 nên x̂ = 0 và mọi đầu ra sụp về β; ước lượng không lệch cho running_var phải chia N − 1 = 0. PyTorch chặn bằng `ValueError: Expected more than 1 value per channel when training`. Đầu vào (1, C, L) hoặc `BatchNorm2d` với (1, C, H, W) vẫn chạy, vì mỗi kênh có L hay H·W giá trị (nhưng thống kê chỉ lấy từ một mẫu). Ở chế độ eval thì dùng running stats nên chạy được.

% manim: BatchVaLayerNorm

Lời giải thích ban đầu của BN là giảm "internal covariate shift" (phân phối đầu vào của mỗi lớp thay đổi khi các lớp trước được cập nhật). Santurkar và cộng sự (2018) cho thấy lời giải thích đó không đứng vững: họ cố tình thêm nhiễu làm dịch phân phối sau BN mà mạng vẫn huấn luyện tốt, và đề xuất rằng tác dụng chính là làm mặt loss trơn hơn (gradient ổn định hơn), nên dùng được lr lớn hơn. Hiệu quả của BN là thật; lý do thì còn tranh luận.

**Dropout (Srivastava và cộng sự, JMLR 2014).** Lúc train, mỗi activation bị đặt về 0 với xác suất p, độc lập (r là mặt nạ 0/1). **Inverted dropout** (cách PyTorch làm) chia phần còn lại cho (1 − p) ngay lúc train, để 𝔼 đầu ra bằng đầu vào; lúc eval dropout là hàm đồng nhất, không cần nhân gì.

$$
\tilde{h} = \frac{r \odot h}{1 - p}, \quad r_i \sim \mathrm{Bernoulli}(1 - p) \;\Rightarrow\; \mathbb{E}[\tilde{h}] = h
$$

`nn.Dropout` mặc định p = 0.5. Diễn giải hay dùng: mỗi bước huấn luyện một mạng con ngẫu nhiên, tất cả dùng chung trọng số; lúc test dùng cả mạng là xấp xỉ trung bình của tập mạng con đó. Một diễn giải khác: dropout là tiêm nhiễu nhân, buộc mỗi neuron không được dựa vào một vài neuron cụ thể.

**Weight decay.** Chương 4 đã chứng minh: với SGD thuần, thêm (λ/2)‖w‖₂² vào loss tương đương với nhân w với (1 − ηλ) mỗi bước. Với Adam thì không: gradient của L2 là λw đi qua phép chia cho √v̂, nên bị chuẩn hóa theo độ lớn gradient của từng tham số. AdamW (Loshchilov & Hutter) tách decay ra khỏi gradient: w ← w − ηλw riêng, rồi mới bước Adam. `torch.optim.Adam(weight_decay=...)` là L2-trong-gradient; `torch.optim.AdamW` mới là decay tách rời, mặc định weight_decay = 0.01.

**Ví dụ giải mẫu 6.10 — L2 trong Adam khác AdamW (một bước, w = 1, λ = 0.01, lr = 1e-3).** Ở bước đầu, Adam sau hiệu chỉnh độ lệch có m̂/√v̂ = sign(g) (m̂, v̂ là moment bậc nhất và bậc hai của Adam, không liên quan độ rộng lớp m) (bỏ qua ε), tức bước có độ lớn đúng lr.

- Gradient dữ liệu g = 0. L2-trong-Adam: gradient hiệu dụng là λw = 0.01, bị chuẩn hóa thành 1, nên w = 1 − 1e-3 = 0.999000 — decay bị phóng to 100 lần. AdamW: w = 1 − lr·λ·w = 0.999990, đúng 1e-5.
- g = 10. L2-trong-Adam: gradient hiệu dụng 10.01, chuẩn hóa vẫn thành 1, w = 0.999000 — phần decay coi như biến mất. AdamW: w = 1 − 1e-5 − 1e-3 = 0.998990.
- Kết luận: với Adam, mức decay thực của L2 phụ thuộc vào độ lớn gradient của từng tham số; AdamW cho decay đồng đều. Đã kiểm bằng `torch.optim.Adam` và `torch.optim.AdamW`.

Thực hành phổ biến (không phải định luật): không áp weight decay cho bias và tham số γ, β của normalization, vì chúng ít tham số và không gây overfitting theo kiểu trọng số lớn.

**Vì sao ML cần cái này.**

- Transformer dùng LayerNorm, không dùng BatchNorm: chuỗi có độ dài khác nhau, batch nhỏ khi fine-tune, và suy luận từng token một. Nhánh NLP của bạn sẽ gặp LN (và biến thể RMSNorm) ở mọi mô hình.
- Mọi công thức fine-tune mô hình ngôn ngữ hiện nay dùng AdamW. Nhầm `Adam(weight_decay=0.01)` với AdamW là nhầm phổ biến khi chép code.
- Khởi tạo đúng thang đo là lý do mạng 20 lớp trong lab 3 truyền được tín hiệu. BN/LN làm tín hiệu forward gần như không phụ thuộc thang của W, nên mạng có normalization **ít nhạy hơn** với thang init. Nhưng init vẫn không hết quan trọng: với BN, learning rate hiệu dụng của một lớp phụ thuộc ‖W‖; Transformer có residual vẫn nhạy với init (GPT-2 co các phép chiếu vào nhánh residual theo 1/√N, N là số lớp residual, để giữ phương sai tổng ổn định); và lớp cuối không đứng trước normalization nào.

**Thực tế tính toán.** Trên dữ liệu tổng hợp của lab (không overfitting đáng kể), AdamW với weight_decay 0, 0.1, 1.0 cho acc val 0.7845, 0.7860, 0.7855 sau 3 epoch — gần như không khác. Regularization chỉ có ích khi có overfitting thật để chữa; đừng chờ đợi hiệu ứng mà dữ liệu không cho phép.

## 6.5 Quy trình debug huấn luyện

Mạng nơ-ron hỏng một cách im lặng: code chạy hết, loss vẫn giảm, accuracy vẫn ra một con số, nhưng kết quả kém hơn nhiều so với lẽ ra (Karpathy, 2019). Vì không có thông báo lỗi, bạn cần một quy trình kiểm theo thứ tự, mỗi bước loại trừ một nhóm lỗi.

**Checklist theo thứ tự.**

1. **Cố định mọi thứ.** Seed cố định, tắt augmentation, mô hình nhỏ. Làm một baseline ngu (ví dụ đầu vào toàn 0): mô hình thật phải thắng nó, nếu không thì mạng không dùng đầu vào.
2. **Kiểm loss ban đầu.** Với C lớp và logit gần bằng nhau, p ≈ 1/C cho mọi lớp nên loss ≈ ln C. Lệch xa ⇒ logit lớp cuối quá lớn, nhãn sai định dạng, hoặc loss bị gọi sai.
3. **Overfit một batch.** Vài chục mẫu, tắt regularization, loss phải xuống gần 0. Không làm được thì lỗi nằm ở mô hình, loss hoặc vòng lặp — đừng chạy cả bộ dữ liệu.
4. **Theo dõi gradient.** Grad norm toàn cục và từng lớp; tỉ lệ ‖Δw‖/‖w‖ mỗi bước (CS231n nêu khoảng 1e-3 như quy tắc kinh nghiệm); tỉ lệ neuron chết. Norm thỉnh thoảng vọt lên ⇒ `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)` như phao cứu sinh (chuẩn hóa lại cả vector gradient khi norm vượt ngưỡng).
5. **Quét learning rate** trên thang log, 1e-5 … 1. Hoặc LR range test (Leslie Smith, 2015): một lần chạy ngắn, tăng lr dần, ghi loss, chọn lr hơi nhỏ hơn chỗ loss bắt đầu tăng. Với Adam, 3e-4 hoặc 1e-3 là điểm xuất phát an toàn.
6. **Đọc đường loss.** lr thấp: loss giảm gần như tuyến tính, chậm. lr cao: giảm nhanh rồi kẹt ở mức cao, dao động, hoặc nổ thành NaN. Train giảm mà val tăng: overfitting — lúc này mới thêm regularization.

**Gradient bùng nổ và neuron chết.** Bùng nổ: grad norm vọt lên nhiều bậc độ lớn rồi loss thành NaN hoặc inf; in grad norm từng lớp để thấy lớp nào nổ trước. Chữa theo thứ tự rẻ đến đắt: giảm lr, `clip_grad_norm_`, warmup, kiểm lại init và việc chuẩn hóa đầu vào. Neuron chết: trên một batch lớn, đếm tỉ lệ neuron có h = 0 với mọi mẫu; tỉ lệ cao (vài chục %) thường do lr quá lớn hoặc bias âm lệch. Phòng: lr nhỏ hơn hoặc warmup, init He, Leaky ReLU/GELU.

**Ví dụ giải mẫu 6.11 — loss ban đầu nói gì.** ln 10 = 2.3026; ln 2 = 0.6931; ln 1000 = 6.9078. Mô phỏng logit ngẫu nhiên N(0, s²) cho 10 lớp, CE trung bình: s = 0.01 → 2.303; s = 1 → 2.730; s = 3 → 5.096; s = 10 → khoảng 15.5. Thấy loss ban đầu cỡ 15 với 10 lớp nghĩa là logit lớp cuối có thang đo khoảng 10: init lớp cuối quá lớn, hoặc đầu vào chưa chuẩn hóa. MLP 20–128–128–10 của lab 5 với init mặc định cho loss ban đầu 2.3103. Với nhãn lệch 1 : 3 (2 lớp), đặt bias lớp cuối bằng log tần suất lớp thì loss ban đầu bằng entropy của nhãn, 0.5623, thay vì ln 2 = 0.6931.

**Ví dụ giải mẫu 6.12 — quét learning rate (lab 5).** Adam, 3 epoch, MLP 20–128–128–10 trên dữ liệu chung của lab (10 cụm Gauss chồng lấn trong ℝ²⁰). Loss val / acc val:

| lr | 1e-5 | 1e-4 | 1e-3 | 1e-2 | 1e-1 | 1 |
| --- | --- | --- | --- | --- | --- | --- |
| Loss val | 2.289 | 1.956 | 0.646 | 0.767 | 1.945 | 2.315 |
| Acc val | 0.120 | 0.573 | 0.789 | 0.759 | 0.319 | 0.105 |

Hình chữ U trên thang log, vùng tốt 1e-3 đến 1e-2. Hai đầu đều gần mức đoán mò ln 10, nhưng vì hai lý do khác nhau: lr 1e-5 chưa kịp học; lr 1 thì nhảy loạn. Cùng lr 1e-3, tỉ lệ ‖Δw‖/‖w‖ của lớp đầu sau bước đầu tiên là 7.8e-3 với Adam nhưng 1.7e-5 với SGD — chênh hơn 400 lần. Learning rate không chép được từ optimizer này sang optimizer khác (bài tập 14 giải thích con số 7.8e-3).

% manim: QuetLearningRate

**Ví dụ giải mẫu 6.13 — overfit một batch.** 32 mẫu, MLP 20–128–10 không regularization, Adam lr 1e-2: loss 2.337 ở bước 0, 1.9e-4 ở bước 50, 7.9e-5 ở bước 199 (lab 5). Mạng có 20·128 + 128 + 128·10 + 10 = 3 978 tham số và chỉ 32 mẫu, nên ghi nhớ được hết; nếu không xuống được, chắc chắn có lỗi.

**Danh mục lỗi im lặng kinh điển** (tất cả đều chạy không báo lỗi):

| Lỗi | Triệu chứng |
| --- | --- |
| Quên `opt.zero_grad()` | Gradient là tổng mọi bước trước; loss giảm chậm, dao động |
| `softmax` rồi mới đưa vào `cross_entropy` | Accuracy gần như bình thường nhưng loss bị chặn dưới, gradient nhỏ |
| Quên `model.eval()` khi đánh giá | Dropout/BN vẫn ở chế độ train; cùng input cho output khác nhau |
| Chuẩn hóa val/test bằng thống kê của chính nó | Rò rỉ dữ liệu; trên dữ liệu thật có dịch phân phối thì sai lệch lớn |
| Xáo X và y bằng hai hoán vị khác nhau | Nhãn mất liên hệ với đầu vào. Xáo lại mỗi epoch: loss train kẹt ở ln C. Lệch cố định một lần: mạng đủ lớn thuộc lòng được nên loss train vẫn giảm, nhưng val ở mức đoán mò (notebook: loss val 2.380, acc 0.084) |
| `total_loss += loss` (tensor) | Sau `backward()` bộ nhớ gần như không tăng, nhưng `total_loss` vẫn giữ `grad_fn`; dùng `.item()`. Rò thật khi giữ loss của lần forward chưa backward (vd đánh giá thiếu `no_grad`) hoặc dùng `retain_graph` |
| `view`/`reshape` sai thứ tự chiều | Trộn các mẫu trong batch với nhau |
| Quên `shuffle=True` khi dữ liệu xếp theo lớp | Mỗi batch một lớp; loss train dao động mạnh theo chu kỳ |
| Quên `torch.no_grad()` khi đánh giá | Chỉ tốn bộ nhớ và thời gian, con số vẫn đúng |

**Vì sao ML cần cái này.**

- Ở giai đoạn nghiên cứu, phần lớn thời gian là debug thí nghiệm chứ không phải nghĩ ý tưởng. Kết luận "phương pháp A không hiệu quả" từ một lần chạy có lỗi im lặng là lỗi nghiêm trọng nhất của người làm nghiên cứu.
- Checklist này dùng nguyên cho fine-tune mô hình ngôn ngữ: loss ban đầu của LM có từ vựng 32 000 token với init ngẫu nhiên là khoảng ln 32 000 ≈ 10.4; mô hình pretrained thì thấp hơn nhiều.

**Thực tế tính toán.** Trên dữ liệu dễ, lỗi gần như vô hình. Thử với 10 cụm Gauss tách bạch: quên `zero_grad` vẫn đạt acc 0.988 và softmax-trước-CE vẫn 0.999. Đó là lý do lab dùng dữ liệu chồng lấn (acc tối đa khoảng 0.79): triệu chứng chỉ hiện ra khi bài toán đủ khó. Một benchmark dễ không kiểm được code của bạn.

## Bài tập giấy bút

14 bài, làm theo thứ tự. Mục tiêu: hết ★ trong 1 buổi, hết ★★ trong 2 buổi, ★★★ dành cho Thứ Bảy. Bài nào sai ghi vào sổ lỗi kèm lý do sai.

**★ Hiểu định nghĩa**

1. Đếm số tham số của MLP 512–1024–1024–10 (có bias ở mọi lớp). Huấn luyện bằng Adam ở float32 thì trọng số, gradient và hai moment chiếm bao nhiêu MB?
2. Chứng minh tanh(x) = 2σ(2x) − 1 và σ′(x) = σ(x)(1 − σ(x)). Suy ra σ′ ≤ 0.25. Qua 10 lớp sigmoid, gradient bị nhân với tối đa bao nhiêu (chỉ tính phần đạo hàm kích hoạt)?
3. Mạng một chiều: z = w₁x + b₁, h = ReLU(z), ŷ = w₂h, L = ½(ŷ − y)². Với x = 2, w₁ = 0.5, b₁ = −0.5, w₂ = 3, y = 1, tính L, ∂L/∂w₂ và ∂L/∂w₁. Nếu b₁ = −1.5 thì ∂L/∂w₁ bằng bao nhiêu?
4. BatchNorm một đặc trưng, ε bỏ qua, γ = 1, β = 0, running mean khởi đầu 0, running var khởi đầu 1, momentum 0.1. Batch train là (2, 4, 6). Tính x̂, running_mean, running_var. Sau đó ở chế độ eval, đầu vào x = 4 ra bao nhiêu?

**★★ Mức đề thi đầu vào**

5. (a) Chứng minh hợp của k lớp affine là một lớp affine, và hạng của ma trận tương đương không vượt quá chiều nhỏ nhất của các lớp ẩn. (b) Chứng minh ReLU(z) − ReLU(−z) = z. Suy ra một MLP ReLU một lớp ẩn có thể biểu diễn chính xác mọi bộ phân loại tuyến tính.
6. Với p = softmax(z) và L = −log pₖ (k là nhãn đúng), chứng minh ∂L/∂zⱼ = pⱼ − 1\[j = k\]. Vì sao tổng các thành phần của gradient này luôn bằng 0, và điều đó nói gì về việc cộng một hằng số vào mọi logit?
7. Dẫn xuất Var(w) = 2/nᵢₙ cho khởi tạo He. Nêu rõ từng giả thiết (độc lập, tâm 0, đối xứng) và chỉ ra bước nào cần ReLU.
8. Inverted dropout với xác suất tắt p. (a) Chứng minh 𝔼\[h̃\] = h. (b) Tính Var(h̃ᵢ) theo hᵢ và p. (c) Nếu cài dropout không chia cho (1 − p) lúc train thì lúc test phải làm gì để khớp kỳ vọng? (d) Viết backward: biết dh̃, tính dh. Cần lưu gì từ forward?
9. MLP một lớp ẩn, khởi tạo sao cho mọi hàng của W₁ (quy ước mỗi hàng một neuron) bằng nhau, b₁ có mọi phần tử bằng nhau, và mọi cột của W₂ bằng nhau. Chứng minh bằng quy nạp rằng sau bất kỳ số bước SGD nào, các neuron ẩn vẫn giống hệt nhau. Dropout có phá được đối xứng này không?
10. Một tham số w = 2, λ = 0.1, lr = 0.01, bước Adam đầu tiên (đã hiệu chỉnh độ lệch, bỏ qua ε). (a) Tính w sau một bước theo hai cách: `Adam(weight_decay=0.1)` và `AdamW(weight_decay=0.1)`, với gradient dữ liệu g = 0 và g = −5. (b) Giải thích vì sao ở trường hợp g = −5, L2-trong-Adam cho kết quả trùng với không có decay. (c) Với g = 0, decay thực của L2-trong-Adam lớn hơn hay nhỏ hơn AdamW, và gấp khoảng bao nhiêu lần? (d) Vì sao L2 dùng ½λ‖w‖² chứ không phải λ‖w‖₂?

**★★★ Mức phỏng vấn nghiên cứu**

11. LayerNorm một mẫu x ∈ ℝⁿ, γ = 1, β = 0, ε bỏ qua: x̂ᵢ = (xᵢ − μ)/σ với σ² là phương sai lệch. Biết dy = ∂L/∂x̂. (a) Dẫn xuất dxᵢ = (1/σ)(dyᵢ − mean(dy) − x̂ᵢ·mean(dy ⊙ x̂)). (b) Chứng minh dx vuông góc với vector toàn 1 và với x̂; nói bằng lời "trừ trung bình rồi chiếu vuông góc" nghĩa là gì. (c) Với x = (2, 4, 6, 8), dy = (1, 0, 0, 0), tính dx. (d) BatchNorm khác ở đâu?
12. Một người viết `F.cross_entropy(F.softmax(logits, 1), y)` với 10 lớp. (a) Chứng minh loss không bao giờ xuống dưới ln(1 + 9/e) ≈ 1.4612. (b) Giải thích vì sao gradient theo logit nhỏ hơn hẳn so với bản đúng, đặc biệt khi mạng đã tự tin. (c) Vì sao accuracy gần như không đổi, và bài học cho việc chọn chỉ số để debug là gì?
13. Bạn fine-tune trên GPU 6 GB, batch 4, gradient accumulation 8 bước để có batch hiệu dụng 32. (a) Nếu mạng có BatchNorm, thống kê batch được tính trên bao nhiêu mẫu, và vì sao kết quả có thể khác huấn luyện batch 32 thật? (b) LayerNorm có bị ảnh hưởng không? (c) Khi huấn luyện mô hình ngôn ngữ, đôi khi loss val thấp hơn loss train. Nêu ba lý do có thể.
14. Lab 5 đo được: sau bước Adam đầu tiên với lr 1e-3, tỉ lệ ‖Δw‖/‖w‖ của lớp `nn.Linear(20, 128)` là 7.8e-3. Dùng (i) tính chất bước đầu của Adam và (ii) init mặc định của `nn.Linear`, giải thích con số này bằng giấy bút. Suy ra lr Adam để tỉ lệ ban đầu cỡ 1e-3 cho một lớp có fan-in 768. Vì sao với SGD tỉ lệ này lại phụ thuộc vào độ lớn gradient, còn với Adam thì gần như không?

## Bài tính nhanh

15 bài, mỗi bài tối đa 2 phút, làm bằng giấy bút và bấm giờ cả bộ. Mục tiêu: xong cả 15 bài trong 25 phút, sai không quá 1 bài. Làm lại bộ này sau 3 ngày và sau 1 tuần.

1. ReLU(−2), ReLU(3), và đạo hàm của ReLU tại 0 theo PyTorch.
2. σ(0) và σ′(0).
3. tanh′(0) bằng bao nhiêu? tanh(x) viết theo σ thế nào?
4. Số tham số của `nn.Linear(768, 3072)`.
5. Loss ban đầu mong đợi khi phân loại 1000 lớp (biết ln 10 ≈ 2.3026).
6. Biên của Xavier uniform cho lớp 784 → 256.
7. Độ lệch chuẩn He cho fan-in 256.
8. Biên của init mặc định `nn.Linear` khi in_features = 100.
9. BatchNorm (chế độ train, γ = 1, β = 0, ε bỏ qua) trên batch (1, 2, 3) của một đặc trưng.
10. Dropout p = 0.2 ở chế độ train, phần tử có giá trị 5 không bị tắt. Đầu ra bằng bao nhiêu?
11. Mô hình 10 triệu tham số, float32, Adam. Bộ nhớ cho trọng số + gradient + m + v.
12. Gradient check: giải tích 1.2e-3, số 1.0e-3. Sai số tương đối? Đúng hay sai?
13. w = 2, x = 3, L = (wx − 1)². Gọi forward + backward 3 lần không zero_grad. `w.grad` bằng bao nhiêu?
14. 4000 mẫu train, batch 64, không bỏ batch cuối. Bao nhiêu bước mỗi epoch, bao nhiêu bước sau 8 epoch?
15. Logit (0, ln 3), nhãn đúng là lớp có logit ln 3 (chỉ số 1, vì chương đánh số lớp từ 0). Tính p và cross-entropy.

## Lab code và bài săn bug

Năm lab dưới đây đã chạy với Python 3, NumPy 2.5 và PyTorch 2.14.1 (bản CPU là đủ). Toàn bộ notebook lời giải đo được 36–39 giây khi máy rảnh; trên máy đang bận một lần chạy mất 137 giây, nên hãy dự trù tới 2–3 phút. Số liệu có thể lệch nhẹ theo phiên bản và số luồng CPU. Dữ liệu chung: 10 cụm Gauss chồng lấn trong ℝ²⁰ (tâm `randn(10, 20) * 0.6`, nhiễu chuẩn 1, rồi nhân 5 cộng 3), 4000 mẫu train và 2000 mẫu val, chuẩn hóa bằng thống kê của train.

| Lab | Việc phải làm | Kết quả mong đợi |
| --- | --- | --- |
| 1. MLP NumPy | `forward`, `softmax_ce`, `backward` theo batch; gradient check sai phân trung tâm | Ví dụ 6.4: L = 1.9102, dW₁ khớp; sai số tương đối lớn nhất 5.7e-11 trên mạng ngẫu nhiên float64 |
| 2. NumPy rồi PyTorch | SGD NumPy (He init, lr 0.1, 8 epoch); viết lại bằng `nn.Module` + `DataLoader` + Adam | NumPy: loss val 0.690, acc 0.771. PyTorch Adam 1e-3: loss val 0.634, acc 0.788. Cùng trọng số: loss trùng tới 10 chữ số |
| 3. Khởi tạo | Std activation qua 20 lớp ReLU rộng 256; init hằng số | std cuối: 0.01 → 1.0e-19; mặc định → 0.023; Xavier → 8.6e-4; He → 0.878. Init hằng số: 5 hàng gradient giống hệt nhau |
| 4. BN, LN, dropout viết tay | `batchnorm_forward` (train/eval, running stats), `layernorm_forward`, inverted dropout; so với `nn.BatchNorm1d`, `nn.LayerNorm` | running_mean 0.3, running_var 1.3667; khớp PyTorch qua 3 bước train + eval; dropout p = 0.5: trung bình 1.0027, tắt 49.9% |
| 5. Hộp công cụ debug | `initial_loss`, `overfit_one_batch`, `grad_norm`, `lr_sweep`, tỉ lệ ‖Δw‖/‖w‖ | Loss ban đầu 2.3103; overfit 32 mẫu 2.337 → 7.9e-5; lr tốt nhất 1e-3 (bảng ví dụ 6.12); tỉ lệ Adam 7.8e-3, SGD 1.7e-5 |

Code nằm trong notebook `labs/chuong06/lab.ipynb` (bản có chỗ TODO để bạn tự viết, kèm ô kiểm tra tự động) và `labs/chuong06/loigiai.ipynb` (bản lời giải đã chạy kiểm). Mở từ trang **Lab** của app.

**Câu hỏi sau lab.**

- Lab 3: vì sao std với Xavier giảm khoảng 1/√2 mỗi lớp ReLU, còn init mặc định giảm nhanh hơn rồi lại "dừng" ở 0.023? *Gợi ý: so với cùng thí nghiệm không có bias.*
- Lab 4: vì sao BN dùng phương sai lệch khi chuẩn hóa nhưng running_var lại lưu phương sai không lệch? Với batch 64 thì chênh lệch này đáng kể không?
- Lab 5: lr 1e-3 tốt cho Adam. Dùng tỉ lệ ‖Δw‖/‖w‖ (7.8e-3 so với 1.7e-5), ước lượng SGD cần lr cỡ nào để bước đầu có cùng độ lớn tương đối, và vì sao ước lượng đó chỉ đúng ở đầu huấn luyện.

**Bài săn bug.** Hàm `train_buggy` dưới đây chạy không báo lỗi. Model: `Linear(20, 128) → BatchNorm1d → ReLU → Dropout(0.5) → Linear(128, 10)`, Adam lr 1e-3, 8 epoch, dữ liệu chung của lab. Triệu chứng đo được khi chạy trong notebook: loss val 1.7485, acc val 0.712, và gọi model hai lần trên cùng 8 mẫu val cho logit chênh nhau tới 35. Bản đúng đạt loss val 0.658 và acc 0.788. Tìm cả **năm** lỗi trước khi xem lời giải.

```python
def train_buggy(Xtr_raw, ytr, Xva_raw, yva, epochs=8):
    mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0)
    xtr = (Xtr_raw - mu) / sd
    xva = (Xva_raw - Xva_raw.mean(0)) / Xva_raw.std(0)
    model = make_bn_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    dl = DataLoader(TensorDataset(xtr, ytr), batch_size=64, shuffle=True)
    total_loss = 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            probs = F.softmax(model(xb), dim=1)
            loss = F.cross_entropy(probs, yb)
            loss.backward()
            opt.step()
            total_loss += loss
    with torch.no_grad():
        out = F.softmax(model(xva), dim=1)
        val_loss = F.cross_entropy(out, yva).item()
        val_acc = (out.argmax(1) == yva).float().mean().item()
    return model, xva, val_loss, val_acc, total_loss
```

## Bài dựng cảnh Manim

Mỗi chương có một bài tự dựng cảnh: muốn vẽ được một khái niệm, bạn phải hiểu từng bước của nó. Giới hạn **1 giờ** (tính vào luồng C), không chỉnh màu mè.

**Đề bài.** Dùng `manim/chuong06_scenes.py` làm mẫu, viết cảnh `InvertedDropout` vào thư mục `manim/cua-toi/`:

1. Một hàng 8 neuron với activation (0.8, 1.2, 0.4, 1.0, 0.6, 1.4, 0.2, 0.4), hiện tổng 6.0, ghi "chế độ eval".
2. Ba lần chạy ở chế độ train với p = 0.5: mỗi lần tắt (tô xám, giá trị 0) đúng 4 neuron do bạn chọn, 4 neuron còn lại hiện giá trị đã nhân 2, cập nhật tổng. Tắt đúng 4 chỉ để minh họa: dropout thật tắt mỗi neuron độc lập, nên số neuron bị tắt là ngẫu nhiên (Binomial(8, 0.5)), không phải luôn bằng p·n.
3. Kết thúc bằng việc chuyển về chế độ eval: mọi neuron sáng lại với giá trị gốc, kèm một dòng nói kỳ vọng của tổng ở chế độ train bằng tổng ở chế độ eval.

**Tiêu chí tự chấm.** (1) Đúng toán: mỗi giá trị giữ lại bằng đúng 2 lần giá trị gốc, và tổng của từng lần train khớp với phép cộng tay. (2) Có tiêu đề và phụ đề nói người xem phải thấy điều gì (vì sao chia cho 1 − p, vì sao eval không nhân). (3) Dài không quá 30 giây, chữ không chồng chữ.

Render bằng nút **Render cảnh của tôi** ở trang Manim. Lời giải tham khảo nằm ở `manim/loigiai/chuong06_dropout.py` — chỉ mở sau khi đã tự làm.

## Phỏng vấn, bẫy, thực tế khác lý thuyết

**Cách luyện.** Mỗi chuỗi có 4 tầng câu hỏi, đào sâu dần như người phỏng vấn thật. Trả lời to bằng tiếng Việt trước, sau đó nói lại bằng tiếng Anh, 2 phút mỗi tầng, ghi âm và tự nghe lại. Câu trả lời mẫu tiếng Anh bên dưới chỉ phủ tầng 1–2; tầng 3–4 bạn tự soạn, rồi đối chiếu với lời giải các bài tập liên quan. Nhiều câu ở đây được diễn đạt lại từ danh sách câu hỏi của Chip Huyen (*ML Interviews Book*, mục 8.3).

**Chuỗi 1 — Phi tuyến và năng lực biểu diễn**

1. Vì sao không đặt hai lớp Linear liền nhau mà không có gì ở giữa?
2. Một MLP ReLU biểu diễn loại hàm gì? Mạng nhỏ nhất giải được XOR là gì?
3. Định lý xấp xỉ phổ quát nói gì, và không nói gì? (Ví dụ 6.3: m = 8 neuron chỉ đạt MSE 0.33 với seed 0.)
4. Nếu một lớp ẩn đủ rộng đã xấp xỉ được mọi hàm, vì sao người ta vẫn dùng mạng sâu?

> *Model answer:* Two stacked linear layers compose into a single linear map, W₂W₁x plus a bias, so the extra layer adds parameters but no expressive power, and its rank is capped by the narrower layer. Inserting a nonlinearity breaks this collapse. With ReLU, the network becomes a continuous piecewise-linear function: each hidden unit contributes a hinge, and the pieces are glued along the hyperplanes where pre-activations change sign. That is why two ReLU units are enough for XOR: the hidden layer folds the input space so that the two positive points land on the same location and a single line separates the classes.

> *Gợi ý tầng 3–4:* định lý nói tồn tại một mạng đủ rộng, không nói bao nhiêu neuron hay GD có tìm được không. Về độ sâu: số vùng tuyến tính tối đa của mạng ReLU có thể tăng theo cấp số nhân với độ sâu, còn với một lớp rộng chỉ tăng theo đa thức (Montúfar và cộng sự, 2014; đây là chặn dưới cho một số họ mạng, không phải mọi mạng), cộng thêm hai lý do thực dụng: mạng sâu hiệu quả tham số hơn với hàm có cấu trúc hợp thành, và thực nghiệm cho thấy dễ tối ưu hơn khi có norm và residual.

**Chuỗi 2 — Khởi tạo**

1. Vì sao không khởi tạo mọi trọng số bằng 0?
2. Dẫn xuất Var(w) = 1/nᵢₙ để giữ phương sai qua một lớp tuyến tính.
3. Vì sao He có hệ số 2? Init mặc định của `nn.Linear` có phải He không?
4. Khi đã có BatchNorm/LayerNorm và kết nối tắt, init còn quan trọng không? (Phân biệt: thang đo tín hiệu forward, learning rate hiệu dụng, và nhánh residual.)

> *Model answer:* If all weights in a layer start equal, every hidden unit computes the same activation and receives the same gradient, so they stay identical forever and the layer behaves like a single unit; random initialization is there to break that symmetry. For the scale, consider z as a sum of n inputs times independent zero-mean weights: its variance is n times Var(w) times the second moment of the input. Keeping the variance constant from layer to layer gives Var(w) = 1/n for a linear or tanh layer near zero. Xavier averages the forward and backward conditions, giving two over fan-in plus fan-out, which is what you want for symmetric activations.

> *Gợi ý tầng 4:* BN/LN làm đầu ra mỗi lớp gần như không phụ thuộc thang đo của W (hệ số thang bị hấp thụ), nên init sai thang ít gây tắt hay nổ tín hiệu forward. Nhưng không thể nói "init hết quan trọng": lr hiệu dụng của lớp có BN phụ thuộc ‖W‖; Transformer vẫn dùng init riêng (GPT-2 co phép chiếu residual theo 1/√N); lớp cuối và nhánh residual không được normalization che. (Bài săn bug chỉ cho thấy BN hấp thụ thang đo của **đầu vào**, không phải của init.)

**Chuỗi 3 — Normalization**

1. BatchNorm làm gì? Tham số γ, β để làm gì?
2. BN khác nhau thế nào giữa train và eval? Batch 1 thì sao?
3. Vì sao Transformer dùng LayerNorm thay vì BatchNorm?
4. "Internal covariate shift" có phải lý do BN hoạt động không? Bằng chứng nào?

> *Model answer:* BatchNorm standardizes each feature using the mean and variance of the current mini-batch, then applies a learned scale γ and shift β so the network can recover any scale it needs. During training it uses batch statistics and keeps exponential running averages; at evaluation it switches to those running averages, so a prediction no longer depends on the other samples in the batch. It also breaks when a channel has only one value per batch, for example an input of shape (1, C): the batch variance is then zero, so every output collapses to β, and the unbiased running estimate would divide by N − 1 = 0, so PyTorch raises an error in training mode. With more values per channel, such as a (1, C, L) input, it runs but the statistics are noisy. LayerNorm normalizes across the features of each sample instead, so it behaves identically in both modes and does not care about batch size.

**Chuỗi 4 — Debug huấn luyện**

1. Loss đứng yên ở khoảng 2.30 với bài 10 lớp. Bạn kiểm tra gì trước?
2. Vì sao "overfit một batch" là bước bắt buộc? Nếu không overfit được thì nghi gì?
3. Làm sao chọn learning rate? Đường loss khi lr quá cao, quá thấp, vừa phải trông thế nào?
4. Model cho output khác nhau mỗi lần chạy trên cùng input; accuracy ổn nhưng loss val không xuống dưới 1.7. Liệt kê các lỗi im lặng có thể gây ra từng triệu chứng.

> *Model answer:* A loss stuck at 2.30 with ten classes means the model is predicting a near-uniform distribution, since ln 10 ≈ 2.30, so it is not learning anything from the input. I would first check that the loss at initialization is indeed about ln C, then try to overfit a single small batch with regularization turned off; if the loss cannot reach almost zero, the bug is in the model, the loss or the training loop rather than in the data size. Typical culprits are a missing optimizer step or zero_grad, or a learning rate that is far too small or so large that the network diverges back to chance; a pipeline bug such as misaligned inputs and labels can slip past a single-batch test, so I would also print a few (x, y) pairs. After that I would log gradient norms per layer and run a short learning-rate sweep on a log scale.

**Từ vựng nên nói trôi chảy:** nonlinearity, piecewise linear, universal approximation, dead ReLU, subgradient, numerically stable softmax, gradient check, central difference, autograd, computational graph, gradient accumulation, symmetry breaking, fan-in, Xavier/Glorot, Kaiming/He initialization, running statistics, inverted dropout, decoupled weight decay, overfit a single batch, learning-rate sweep, gradient clipping, silent bug.

**Bẫy thường gặp.**

- Đưa xác suất (đã softmax) vào `CrossEntropyLoss`. Trong lab: acc val 0.792 so với 0.788 của bản đúng (không thấy gì), nhưng loss val 1.710 so với 0.658, bị chặn dưới bởi 1.4612, và grad norm lúc khởi tạo nhỏ đi khoảng 10 lần (0.174 so với 1.690).
- Nghĩ init mặc định của `nn.Linear` là He. Nó là U(±1/√nᵢₙ), std ≈ 1/√(3·nᵢₙ); qua 20 lớp ReLU không bias, std còn cỡ 1e-8.
- Dùng `torch.optim.Adam(weight_decay=...)` và tưởng là AdamW (ví dụ 6.10).
- BatchNorm với batch nhỏ, hoặc BN kết hợp gradient accumulation: thống kê theo micro-batch, không theo batch hiệu dụng.
- Chuẩn hóa val/test bằng thống kê của chính nó. Trên dữ liệu lab cùng phân phối, triệu chứng nhỏ (acc 0.7845 so với 0.7875); trên dữ liệu thật có dịch phân phối hoặc khi test từng mẫu một, nó sai hẳn. Đây là lỗi quy trình, đừng đánh giá nó bằng con số.
- Debug bằng accuracy thay vì loss: accuracy có thể giữ nguyên trong khi loss báo có lỗi.
- Cộng hay lưu `loss` (tensor) thay vì `loss.item()`: sau `backward()` các tensor lưu cho backward đã được giải phóng nên bộ nhớ gần như không tăng (MLP 100–512–10, batch 64, 1 500 bước: không khác biệt đo được so với `.item()`; bỏ `backward()` thì tăng khoảng 370 MB, dùng `retain_graph=True` khoảng 380 MB), nhưng tensor vẫn giữ `grad_fn`. Rò thật chỉ khi giữ loss của lần forward chưa backward hoặc dùng `retain_graph`. Luôn `.item()` hoặc `.detach()`.

**Thực tế khác lý thuyết.**

- Định lý xấp xỉ phổ quát nói "tồn tại", không nói "tìm được". Ví dụ 6.3: cùng mạng 8 neuron, MSE dao động từ khoảng 0.005 đến 0.33 chỉ vì đổi seed; seed tệ nhất có 3 neuron chết. Trong thực tế, mạng sâu vừa phải dễ tối ưu và hiệu quả tham số hơn mạng một lớp rất rộng.
- Lời giải thích gốc của BatchNorm ("internal covariate shift") bị phản bác một phần (Santurkar và cộng sự, 2018). Hiệu quả là thật; cơ chế vẫn đang tranh luận.
- Xavier được dẫn xuất cho mạng tuyến tính, He cho ReLU, cả hai giả định độc lập và không có normalization. Khi có BN/LN, thang đo của W bị hấp thụ nên init ít gây tắt hay nổ tín hiệu forward, nhưng vẫn quyết định lr hiệu dụng, độ ổn định của nhánh residual và lớp cuối. Đừng trích bài săn bug làm bằng chứng cho init: nó bỏ **chuẩn hóa đầu vào** (acc 0.7865 so với 0.7875, vì BN sau lớp đầu hấp thụ thang đo của đầu vào), mô hình đó không có residual và init trọng số không đổi.

## Mini-test 30 phút

Làm sau khi xong bài tập và lab, bấm giờ đúng 30 phút, không xem sách, không dùng máy tính (câu 5 chỉ đọc code, không chạy). Tổng 30 điểm. **Từ 24 điểm:** sang chương 7. **18 đến dưới 24:** ôn lại mục ứng với câu sai rồi làm lại sau 3 ngày. **Dưới 18:** học lại chương.

1. **(4 điểm)** Mạng 2–2–1: x = (2, −1); W₁ có các hàng (1, 1) và (−1, 2), b₁ = 0; h = ReLU(W₁x); ŷ = w₂ᵀh với w₂ = (2, 3), không bias; L = ½ŷ² (mục tiêu là 0). Tính z₁, h, ŷ, L, ∂L/∂w₂ và ∂L/∂W₁.
2. **(4 điểm)** MLP 1000–500–100 có bias. Đếm tham số. Huấn luyện bằng Adam ở float32 thì trọng số, gradient, m, v chiếm bao nhiêu byte? Vì sao con số thực tế trên GPU luôn lớn hơn?
3. **(5 điểm)** Dẫn xuất khởi tạo He. Sau đó giải thích vì sao Xavier làm std activation giảm khoảng 1/√2 mỗi lớp trong mạng ReLU.
4. **(5 điểm)** So sánh BatchNorm và LayerNorm: trục chuẩn hóa, hành vi train/eval, batch 1. Vì sao Transformer dùng LayerNorm?
5. **(6 điểm)** Đọc code và trả lời không chạy máy:

```python
for ep in range(10):
    perm = torch.randperm(N)
    Xs, ys = X[perm], y[torch.randperm(N)]
    model.train()
    for i in range(0, N, 64):
        opt.zero_grad()
        loss = F.cross_entropy(model(Xs[i:i+64]), ys[i:i+64])
        loss.backward(); opt.step()
        history.append(loss)
acc = (model(Xva).argmax(1) == yva).float().mean()
```

(a) Với 10 lớp cân bằng, loss train và acc val sẽ ra khoảng bao nhiêu, vì sao? (b) Dòng `history.append(loss)` gây vấn đề gì? (c) Model có Dropout; dòng cuối thiếu gì, và thiếu thì sai số hay chỉ tốn bộ nhớ?

6. **(6 điểm)** Tối đa 6 câu: một mô hình 10 lớp có loss đứng ở 2.30 suốt 5 epoch. Viết quy trình kiểm tra theo thứ tự, mỗi bước nói nó loại trừ nhóm lỗi nào.

## Flashcard

18 thẻ của chương nằm trong `flashcards/chuong06.csv`. Ôn ở trang **Ôn tập** của app: app tự lên lịch theo hộp Leitner (thẻ nhớ được sẽ giãn lịch ôn 1 → 3 → 7 → 14 ngày, thẻ quên quay lại hộp 1). File CSV này cũng nhập thẳng được vào Anki nếu bạn thích dùng Anki.

## Lời giải

Chỉ đọc sau khi đã tự thử ít nhất 20 phút. Với bài chứng minh, so ý chính chứ không so từng chữ. Mọi đáp số bằng số trong chương đã được kiểm lại bằng NumPy hoặc PyTorch.

**Kiểm tra nhanh**

1. Tương đương một ma trận 100 × 100 duy nhất W₂W₁ (cộng bias), hạng tối đa 5.
2. ∂L/∂z = (3, 3, 3) ⊙ 1\[z > 0\] = (0, 3, 3).
3. Tổng gradient của hai lần: gấp đôi gradient đúng (gradient được cộng dồn, ví dụ 6.6).
4. Var(w) = 2/512 = 1/256, std = 1/16 = 0.0625.
5. ln 100 = 2 ln 10 ≈ 4.6052.

**Bài tập giấy bút**

1. 512·1024 + 1024 + 1024·1024 + 1024 + 1024·10 + 10 = 524 288 + 1 024 + 1 048 576 + 1 024 + 10 240 + 10 = 1 585 162 tham số. Adam float32: 16 byte mỗi tham số ⇒ 25 362 592 byte ≈ 25.4 MB.
2. 2σ(2x) − 1 = (2 − 1 − e⁻²ˣ)/(1 + e⁻²ˣ) = (1 − e⁻²ˣ)/(1 + e⁻²ˣ); nhân cả tử và mẫu với eˣ được (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ) = tanh x. σ′(x) = e⁻ˣ/(1 + e⁻ˣ)² = σ(x)·e⁻ˣ/(1 + e⁻ˣ) = σ(1 − σ). Tích s(1 − s) với s ∈ (0, 1) lớn nhất khi s = ½, bằng 0.25. Qua 10 lớp: tối đa 0.25¹⁰ ≈ 9.5 × 10⁻⁷.
3. z = 0.5·2 − 0.5 = 0.5, h = 0.5, ŷ = 1.5, L = ½(0.5)² = 0.125. ∂L/∂w₂ = (ŷ − y)h = 0.25. ∂L/∂h = (ŷ − y)w₂ = 1.5, ∂L/∂z = 1.5 (vì z > 0), ∂L/∂w₁ = 1.5·x = 3.0. Với b₁ = −1.5: z = −0.5 < 0 nên ∂L/∂w₁ = 0 (và ∂L/∂w₂ = 0 vì h = 0).
4. μ = 4, phương sai lệch (4 + 0 + 4)/3 = 8/3, x̂ = (−2, 0, 2)/√(8/3) = (−1.2247, 0, 1.2247). Phương sai không lệch 8/2 = 4. running_mean = 0.1·4 = 0.4; running_var = 0.9·1 + 0.1·4 = 1.3. Eval: (4 − 0.4)/√1.3 = 3.157 — rất khác 0 dù 4 là trung bình của batch train, vì running stats sau một bước còn xa thống kê thật. Đó là lý do mô hình BN mới huấn luyện vài bước cho kết quả eval tệ.
5. (a) Quy nạp: giả sử k − 1 lớp đầu bằng x ↦ Ax + a; thêm lớp W(·) + b được (WA)x + (Wa + b), vẫn affine. Ma trận tương đương là tích Wₖ⋯W₁; theo rank(AB) ≤ min(rank A, rank B), hạng không vượt quá số chiều của lớp ẩn hẹp nhất. (b) Nếu z ≥ 0: z − 0 = z; nếu z < 0: 0 − (−z) = z. Bộ phân loại tuyến tính f(x) = wᵀx + c: dùng hai neuron ẩn h₁ = ReLU(wᵀx), h₂ = ReLU(−wᵀx), lớp ra h₁ − h₂ + c = wᵀx + c. Vậy MLP ReLU chứa mọi mô hình tuyến tính như trường hợp riêng.
6. log pₖ = zₖ − log ∑ᵢ exp(zᵢ). ∂/∂zⱼ của −zₖ là −1\[j = k\]; của log ∑ᵢ exp(zᵢ) là exp(zⱼ)/∑ᵢ exp(zᵢ) = pⱼ. Cộng lại: pⱼ − 1\[j = k\]. Tổng theo j: ∑pⱼ − 1 = 0. Gradient vuông góc với vector toàn 1, tức hướng "cộng cùng một hằng số vào mọi logit" — đúng vì softmax bất biến với phép cộng đó (cũng là lý do trừ max để ổn định số không đổi kết quả).
7. Lớp l: z = ∑ᵢ wᵢxᵢ với nᵢₙ số hạng. Giả thiết: các wᵢ độc lập cùng phân phối, 𝔼\[w\] = 0, độc lập với x. Khi đó 𝔼\[z\] = 0 và Var(z) = ∑ Var(wᵢxᵢ) = nᵢₙ·Var(w)·𝔼\[x²\] (vì 𝔼\[w\] = 0 nên Var(wx) = 𝔼\[w²\]𝔼\[x²\]). Bước cần ReLU: x = ReLU(z′) với z′ là tiền kích hoạt lớp trước, giả thiết z′ đối xứng quanh 0 ⇒ 𝔼\[x²\] = ½𝔼\[z′²\] = ½Var(z′). Ghép lại Var(z) = ½·nᵢₙ·Var(w)·Var(z′). Muốn Var(z) = Var(z′) ⇒ Var(w) = 2/nᵢₙ. Nhận xét: 𝔼\[x\] ≠ 0 nên phải dùng 𝔼\[x²\] chứ không phải Var(x) — đây là chỗ hay sai.
8. (a) 𝔼\[mᵢhᵢ/(1 − p)\] = (1 − p)hᵢ/(1 − p) = hᵢ. (b) 𝔼\[h̃ᵢ²\] = (1 − p)hᵢ²/(1 − p)² = hᵢ²/(1 − p), nên Var = hᵢ²/(1 − p) − hᵢ² = hᵢ²·p/(1 − p); với p = 0.5 là hᵢ². (c) Nếu train không chia thì 𝔼 lúc train là (1 − p)h; lúc test phải nhân activation với (1 − p) (dropout "cổ điển" của paper gốc). Inverted dropout chuyển phép co giãn sang lúc train để test không cần sửa gì. (d) h̃ = m ⊙ h/(1 − p) tuyến tính theo h với hệ số mᵢ/(1 − p), nên dh = m ⊙ dh̃/(1 − p). Phải lưu mặt nạ m của forward; neuron bị tắt nhận gradient 0, neuron giữ lại nhận gradient nhân 1/(1 − p), và 𝔼[dh] = dh̃.
9. Giả thiết quy nạp: ở bước t, mọi hàng của W₁ bằng nhau, mọi phần tử của b₁ bằng nhau, mọi cột của W₂ bằng nhau. Khi đó với mọi mẫu, mọi neuron ẩn có cùng z và cùng h. Gradient: dh = W₂ᵀdz₂ có mọi thành phần bằng nhau (mỗi thành phần là tích của dz₂ với một cột của W₂, mà các cột bằng nhau); dz₁ = dh ⊙ 1\[z₁ > 0\] có mọi thành phần bằng nhau; hàng i của dW₁ là dz₁ᵢxᵀ, nên mọi hàng bằng nhau; db₁ mọi phần tử bằng nhau; cột j của dW₂ là dz₂hⱼ, mọi cột bằng nhau. Cập nhật SGD trừ đi các gradient đối xứng đó nên bước t + 1 vẫn thỏa giả thiết. Dropout có phá được: mặt nạ ngẫu nhiên khác nhau cho từng neuron làm gradient khác nhau. Nhưng dựa vào dropout để phá đối xứng là lãng phí; dùng init ngẫu nhiên. (Lưu ý: momentum và Adam cũng giữ đối xứng vì chúng chỉ là hàm của lịch sử gradient đã đối xứng.)
10. Bước đầu Adam: Δw = −lr·sign(g′), với g′ là gradient hiệu dụng. (a) Trường hợp g = 0: L2-trong-Adam có g′ = λw = 0.2 > 0, w = 2 − 0.01 = 1.990; AdamW: w = 2 − lr·λ·w = 2 − 0.002 = 1.998 (bước Adam bằng 0 vì g = 0). Trường hợp g = −5: L2-trong-Adam có g′ = −5 + 0.2 = −4.8, w = 2 + 0.01 = 2.010; AdamW: w = 2 − 0.002 + 0.01 = 2.008. Không decay: w = 2.010. (b) L2-trong-Adam trùng với không decay vì sau chuẩn hóa chỉ còn dấu của gradient hiệu dụng; λw = 0.2 không đủ đổi dấu −5 nên decay mất hẳn. Đã kiểm bằng `torch.optim.Adam` và `torch.optim.AdamW`. (c) Với g = 0, L2-trong-Adam trừ 0.01 còn AdamW trừ 0.002: lớn hơn 5 lần ở đây, vì bước Adam luôn có độ lớn lr bất kể gradient hiệu dụng nhỏ đến đâu. (d) ‖w‖² có gradient λw: mượt, tỉ lệ với w, bằng 0 tại 0, nên co trọng số lớn mạnh và trọng số nhỏ nhẹ. ‖w‖₂ có gradient λw/‖w‖ độ lớn cố định λ và không xác định tại w = 0, nên kéo trọng số về 0 theo bước cố định và gây dao động quanh 0.
11. (a) ∂σ²/∂xᵢ = 2(xᵢ − μ)/n (số hạng chứa μ triệt tiêu vì ∑(xⱼ − μ) = 0) nên ∂σ/∂xᵢ = x̂ᵢ/n. Do đó ∂x̂ⱼ/∂xᵢ = (δᵢⱼ − 1/n)/σ − (xⱼ − μ)x̂ᵢ/(nσ²) = (1/σ)(δᵢⱼ − 1/n − x̂ᵢx̂ⱼ/n). Cộng theo j với dyⱼ: dxᵢ = (1/σ)(dyᵢ − mean(dy) − x̂ᵢ·mean(dy ⊙ x̂)). (b) ∑x̂ᵢ = 0 và ∑x̂ᵢ² = n. Tổng ∑dxᵢ ∝ ∑dy − n·mean(dy) − 0 = 0; ∑dxᵢx̂ᵢ ∝ ∑dyᵢx̂ᵢ − 0 − n·mean(dy ⊙ x̂) = 0. Vậy dx = (1/σ)·(dy sau khi bỏ thành phần theo 1 và theo x̂): trừ trung bình là bỏ thành phần dọc 1, trừ phần theo x̂ là bỏ thành phần dọc x̂ (hai hướng này vuông góc nhau). Lý do: phép chuẩn hóa không đổi khi cộng hằng số hay nhân thang vào x, nên gradient không có thành phần theo hai hướng đó. (c) μ = 5, σ² = 5, σ = 2.2361, x̂ = (−1.3416, −0.4472, 0.4472, 1.3416); mean(dy) = 0.25, mean(dy ⊙ x̂) = −0.3354. dx = (0.3, −0.4, −0.1, 0.2)/√5 = (0.1342, −0.1789, −0.0447, 0.0894): tổng bằng 0 và vuông góc x̂. (d) BN dùng cùng công thức nhưng mean lấy trên trục batch cho từng đặc trưng, nên dx của một mẫu phụ thuộc mọi mẫu khác trong batch; ở eval thống kê là hằng số nên không còn phép chiếu, dx = dy chia cho σ của running stats. Với γ thì thay dy bằng γ·dy.
12. (a) Đầu vào của CE là q = softmax(z): mọi qⱼ ∈ \[0, 1\] và tổng bằng 1. Gọi k là nhãn đúng. CE(q, k) = −qₖ + log ∑ⱼ exp(qⱼ) = log(1 + ∑ⱼ≠ₖ exp(qⱼ − qₖ)). Mỗi qⱼ ≥ 0 và qₖ ≤ 1 nên qⱼ − qₖ ≥ −1, do đó tổng 9 số hạng ≥ 9/e và CE ≥ log(1 + 9/e) ≈ 1.4612; dấu bằng khi q là one-hot tại k. (b) Gradient đi qua thêm Jacobian của softmax thứ nhất, có các phần tử pᵢ(δᵢⱼ − pⱼ) với độ lớn ≤ 0.25 và tiến về 0 khi mạng tự tin (p gần one-hot), nên gradient nhỏ đi nhiều và học chậm dần. Đo được trong notebook: grad norm lúc khởi tạo 0.174 so với 1.690. (c) argmax của softmax bằng argmax của logit, nên dự đoán chỉ phụ thuộc thứ tự, mà thứ tự vẫn học được dù chậm; acc val 0.792 so với 0.788. Bài học: debug bằng loss (đại lượng được tối ưu), đừng chỉ nhìn accuracy.
13. (a) BN tính thống kê trên từng micro-batch 4 mẫu, không phải 32. Thống kê nhiễu hơn nhiều (phương sai ước lượng từ 4 điểm), nhiễu đó đi vào cả forward lẫn gradient, và running stats được cập nhật 8 lần mỗi bước hiệu dụng. Gradient accumulation chỉ làm tương đương batch lớn khi mọi lớp độc lập giữa các mẫu; BN vi phạm điều đó. Cách chữa: dùng LayerNorm/GroupNorm, hoặc đóng băng BN khi fine-tune. (b) Không: LN chỉ dùng thông tin của từng mẫu. (c) Dropout bật lúc train và tắt lúc val, nên loss train được đo trên mạng "yếu hơn"; loss train là trung bình trong cả epoch khi mô hình còn kém ở đầu epoch, còn loss val đo ở cuối epoch; tập val có thể dễ hơn hoặc trùng lặp với dữ liệu pretraining/train (rò rỉ).
14. Bước Adam đầu tiên (hiệu chỉnh độ lệch, bỏ qua ε): m̂ = g, v̂ = g², nên mỗi phần tử thay đổi đúng lr (trừ phần tử có g = 0). Với n phần tử: ‖Δw‖ ≈ lr·√n. Init mặc định: std(w) = 1/√(3·nᵢₙ), nên ‖w‖ ≈ √n/√(3·nᵢₙ). Tỉ lệ ≈ lr·√(3·nᵢₙ) = 1e-3·√60 ≈ 7.75e-3, khớp 7.8e-3 đo được. Với fan-in 768: muốn lr·√(3·768) ≈ 1e-3 thì lr ≈ 1e-3/48 ≈ 2e-5 — cùng cỡ các lr hay dùng khi fine-tune BERT, dù đây chỉ là lập luận cho bước đầu, không phải quy tắc chọn lr. Với SGD, Δw = −lr·g nên tỉ lệ bằng lr·‖g‖/‖w‖, tỉ lệ thuận với độ lớn gradient; Adam chia cho √v̂ nên độ lớn bước gần như chỉ do lr quyết định.

**Bài tính nhanh**

1. 0, 3, và 0.
2. 0.5 và 0.25.
3. tanh′(0) = 1 − tanh²(0) = 1; tanh(x) = 2σ(2x) − 1.
4. 768·3072 + 3072 = 2 362 368.
5. ln 1000 = 3 ln 10 ≈ 6.9078.
6. √(6/1040) ≈ 0.0760.
7. √(2/256) = 1/(8√2) ≈ 0.0884.
8. 1/√100 = 0.1.
9. μ = 2, phương sai lệch 2/3, x̂ = (−1, 0, 1)/√(2/3) = (−1.2247, 0, 1.2247).
10. 5/(1 − 0.2) = 6.25.
11. 10⁷ · 16 byte = 1.6 × 10⁸ byte = 160 MB.
12. 0.2e-3/1.2e-3 ≈ 0.17. Trên 1e-2, gần như chắc chắn sai.
13. 3 · 30 = 90.
14. ⌈4000/64⌉ = 63 bước mỗi epoch (62 batch đủ 64 và một batch 32); 8 epoch là 504 bước.
15. Nhãn là phần tử thứ hai (chỉ số 1). p = (1, 3)/4 = (0.25, 0.75); CE = −ln 0.75 = ln(4/3) ≈ 0.2877.

**Câu hỏi sau lab**

- Lab 3: Xavier có Var(w) = 2/(256 + 256) = 1/256, nên mỗi lớp nhân phương sai với nᵢₙ·Var(w)·½ = ½, tức std nhân 1/√2. Init mặc định có Var(w) = 1/(3·256), mỗi lớp nhân std với √(1/6) ≈ 0.41, nên tín hiệu từ đầu vào tắt rất nhanh; con số "dừng" ở 0.023 là thang đo do bias (mặc định U(±1/16)) tạo ra ở mỗi lớp, cân bằng với hệ số co 0.41. Bỏ bias thì std cuối cỡ 1e-8 (1.0e-8 trong notebook; con số chính xác phụ thuộc phiên bản và số luồng). Tức là ở lớp 20, activation gần như chỉ còn là hàm của bias, không còn thông tin đầu vào.
- Lab 4: chuẩn hóa dùng thống kê của chính batch đó, nên phương sai lệch (chia N) là phương sai đúng của các số đang chuẩn hóa: sau chuẩn hóa, phương sai lệch của x̂ đúng bằng 1. running_var là ước lượng phương sai của **tổng thể** để dùng lúc eval, nên dùng ước lượng không lệch (chia N − 1). Với batch 64, hệ số N/(N − 1) = 1.016, chênh 1.6%, không đáng kể; với batch 4 thì chênh 33%.
- Lab 5: tỉ lệ của SGD bằng lr·‖g‖/‖w‖; để đạt 7.8e-3 thay vì 1.7e-5 cần lr lớn hơn khoảng 450 lần, tức cỡ 0.45 (con số này tính cho mạng 20–128–128–10 của lab 5; lab 2 là mạng và init khác nên không dùng làm đối chứng). Ước lượng chỉ đúng ở bước đầu vì ‖g‖ của SGD thay đổi mạnh trong quá trình huấn luyện (thường giảm), còn bước Adam ở các bước sau nhỏ hơn lr khi gradient đổi dấu liên tục (m̂ bị triệt tiêu một phần).

**Mini-test (thang 30 điểm)**

1. z₁ = (2 − 1, −2 − 2) = (1, −4), h = (1, 0), ŷ = 2·1 + 3·0 = 2, L = 2 (1 điểm). ∂L/∂ŷ = ŷ = 2; ∂L/∂w₂ = ŷh = (2, 0) (1 điểm). ∂L/∂h = ŷw₂ = (4, 6); ∂L/∂z₁ = (4, 0) vì neuron 2 tắt (1 điểm). ∂L/∂W₁ = (∂L/∂z₁)xᵀ có các hàng (8, −4) và (0, 0) (1 điểm).
2. 1000·500 + 500 + 500·100 + 100 = 550 600 tham số (1.5 điểm). 16 byte mỗi tham số ⇒ 8 809 600 byte ≈ 8.8 MB (1.5 điểm). Thực tế lớn hơn vì còn activation cho backward (tỉ lệ với batch size), bộ nhớ tạm của các phép toán, và phần dành riêng của framework/CUDA (1 điểm).
3. Dẫn xuất như bài tập 7: Var(z) = nᵢₙVar(w)𝔼\[x²\], ReLU của biến đối xứng cho 𝔼\[x²\] = ½Var(z′), nên Var(w) = 2/nᵢₙ (3 điểm; thiếu giả thiết độc lập/tâm 0 trừ 1 điểm; dùng Var(x) thay 𝔼\[x²\] trừ 1 điểm). Xavier với nᵢₙ = nₒᵤₜ = n cho Var(w) = 1/n, nên mỗi lớp nhân phương sai với n·(1/n)·½ = ½, std nhân 1/√2 (2 điểm).
4. BN chuẩn hóa từng đặc trưng theo batch; LN chuẩn hóa từng mẫu theo đặc trưng (1.5 điểm). BN: train dùng thống kê batch và cập nhật running stats, eval dùng running stats; LN giống nhau ở hai chế độ (1.5 điểm). Batch 1: BN ở chế độ train, khi mỗi kênh chỉ có 1 giá trị (vd `BatchNorm1d` với (1, C)), có phương sai batch bằng 0 nên đầu ra sụp về β, và PyTorch báo lỗi; với (1, C, L) hay `BatchNorm2d` thì vẫn chạy nhưng thống kê nhiễu. LN không ảnh hưởng (1 điểm). Transformer: chuỗi độ dài thay đổi có padding, batch nhỏ, suy luận tự hồi quy từng token, nên cần chuẩn hóa không phụ thuộc batch (1 điểm).
5. (a) `ys` dùng một hoán vị khác `Xs`, nên nhãn không còn liên hệ với đầu vào: loss train kẹt quanh ln 10 ≈ 2.30, acc val quanh 0.1 (vì nhãn bị xáo lại mỗi epoch; nếu xáo một lần cố định, mạng đủ lớn có thể thuộc lòng và loss train xuống thấp, còn acc val vẫn quanh 0.1). Trong bảng tách lỗi của notebook, xáo nhãn độc lập cho loss val 2.380 và acc 0.084 (2 điểm). (b) `loss` là tensor còn `grad_fn`, nên list giữ tham chiếu tới đồ thị. Vì đã `backward()`, các tensor lưu cho backward được giải phóng và bộ nhớ gần như không tăng (MLP 100–512–10, batch 64, 1 500 bước: không khác biệt đo được so với `.item()`; bỏ `backward()` thì tăng khoảng 370 MB, dùng `retain_graph=True` khoảng 380 MB); chỉ rò thật nếu vòng lặp không backward (vd đánh giá thiếu `no_grad`) hoặc dùng `retain_graph`. Sửa: `history.append(loss.item())` hoặc `loss.detach()` (2 điểm). (c) Thiếu `model.eval()` — dropout vẫn bật nên acc bị sai và thay đổi mỗi lần chạy: đây là sai số. Thiếu `torch.no_grad()` — chỉ tốn bộ nhớ, không sai số (2 điểm).
6. Thang chấm, mỗi ý 1 điểm, tối đa 6: (1) Cố định seed, kiểm dữ liệu: in vài cặp (x, y), kiểm X và y được xáo cùng hoán vị, nhãn đúng định dạng — loại lỗi pipeline dữ liệu. (2) Kiểm loss ban đầu ≈ ln 10 — loại lỗi init lớp cuối/loss sai. (3) Overfit một batch nhỏ không regularization — nếu không được, lỗi ở mô hình/loss/vòng lặp (thiếu `step`, thiếu `zero_grad`, softmax trước CE, gradient bị cắt đứt bởi `detach` hay `no_grad`). (4) In grad norm từng lớp — gradient bằng 0 hoặc NaN chỉ ra lớp bị đứt hoặc neuron chết. (5) Quét lr trên thang log — loại lr quá nhỏ (không học) hoặc quá lớn (nhảy về đoán mò). (6) So với baseline đầu vào bằng 0 — xác nhận mô hình có dùng đầu vào; nếu bằng nhau thì đầu vào không đi vào mạng (sai `view`, sai cột).

**Săn bug — năm lỗi**

1. **Thiếu `opt.zero_grad()`**: gradient cộng dồn qua mọi bước. Riêng lỗi này: acc val 0.690, loss val 0.899.
2. **Softmax trước `cross_entropy`** (cả lúc train lẫn lúc đánh giá): loss bị chặn dưới bởi 1.4612 và gradient nhỏ. Riêng lỗi này: loss val 1.710, acc 0.792 — chính nó gây ra triệu chứng "loss không xuống dưới 1.7".
3. **Không gọi `model.eval()` khi đánh giá**: dropout vẫn tắt ngẫu nhiên một nửa neuron và BN dùng thống kê của batch val; thêm nữa, vì còn ở chế độ train nên BN cập nhật running stats bằng dữ liệu val, một dạng rò rỉ nhỏ. Riêng lỗi này: acc 0.728; cùng input gọi hai lần cho kết quả khác nhau.
4. **Chuẩn hóa val bằng thống kê của chính val**: phải dùng `mu, sd` của train. Trên dữ liệu này triệu chứng nhỏ (acc 0.7845 so với 0.7875) vì train và val cùng phân phối; đây là lỗi quy trình: khi test từng mẫu hoặc khi phân phối test lệch, nó sai hẳn.
5. **`total_loss += loss`** cộng tensor còn `grad_fn` (`total_loss.requires_grad` là `True`). Sau `backward()` các tensor lưu cho backward đã được giải phóng nên bộ nhớ gần như không tăng, nhưng `total_loss` vẫn giữ tham chiếu tới đồ thị; đây là thói quen xấu, không đổi con số nào. Sửa: `loss.item()` hoặc `loss.detach()`. Rò thật xảy ra khi cộng loss của lần forward chưa backward hoặc dùng `retain_graph`.

Bảng tách từng lỗi trong notebook (cùng seed): không lỗi 0.6583 / 0.7875; ba lỗi 1–3 cùng lúc 1.7490 / 0.7090; cả bốn lỗi 1–4 1.7485 / 0.7120; chỉ bỏ bước chuẩn hóa đầu vào (không lỗi nào khác) 0.6581 / 0.7865; chỉ xáo nhãn độc lập với đầu vào (một hoán vị cố định, không thuộc năm lỗi) 2.380 / 0.084 (loss val / acc val).

```python
def train_fixed(Xtr_raw, ytr, Xva_raw, yva, epochs=8):
    mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0)
    xtr = (Xtr_raw - mu) / sd
    xva = (Xva_raw - mu) / sd                            # sửa lỗi 4: thống kê của TRAIN
    model = make_bn_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    dl = DataLoader(TensorDataset(xtr, ytr), batch_size=64, shuffle=True)
    total_loss = 0.0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()                              # sửa lỗi 1
            loss = F.cross_entropy(model(xb), yb)        # sửa lỗi 2: logit thô
            loss.backward()
            opt.step()
            total_loss += loss.item()                    # sửa lỗi 5
    model.eval()                                         # sửa lỗi 3
    with torch.no_grad():
        out = model(xva)
        val_loss = F.cross_entropy(out, yva).item()
        val_acc = (out.argmax(1) == yva).float().mean().item()
    return model, xva, val_loss, val_acc, total_loss
```

Đã chạy kiểm: bản sửa cho loss val 0.6583, acc val 0.7875, và hai lần gọi model trên cùng batch val cho output giống hệt nhau.

## Tài liệu đào sâu và nguồn

Nội dung chương được viết lại, đối chiếu phạm vi kiến thức với các nguồn dưới đây. Đọc theo thứ tự ưu tiên:

| Tài liệu | Dùng để làm gì | Phần nên đọc |
| --- | --- | --- |
| [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) (Karpathy, 2019) | Tinh thần và checklist debug của mục 6.5 | Toàn bài; đặc biệt phần dựng khung huấn luyện, overfit, regularize |
| [CS231n: Neural Networks Part 1](https://cs231n.github.io/neural-networks-1/) | Hàm kích hoạt, neuron chết, năng lực biểu diễn | Mục về activation functions và representational power |
| [CS231n: Neural Networks Part 2](https://cs231n.github.io/neural-networks-2/) | Tiền xử lý, khởi tạo, BN, regularization | Data preprocessing, weight initialization, regularization |
| [CS231n: Neural Networks Part 3](https://cs231n.github.io/neural-networks-3/) | Gradient check, kiểm tra trước khi huấn luyện, theo dõi quá trình học | Gradient checks, sanity checks, babysitting the learning process |
| [CS231n: Backprop, Intuitions](https://cs231n.github.io/optimization-2/) | Backprop theo từng giai đoạn, dùng shape để đặt chuyển vị | Phần gradient cho phép nhân ma trận |
| [Glorot & Bengio (2010), Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html) | Nguồn gốc khởi tạo Xavier | Phần phân tích phương sai và bão hòa sigmoid |
| [He và cộng sự (2015), Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) | Khởi tạo He cho ReLU | Mục khởi tạo (dẫn xuất hệ số 2) |
| [Ioffe & Szegedy (2015), Batch Normalization](https://arxiv.org/abs/1502.03167) | Paper gốc BN | Thuật toán BN, phần train/inference |
| [Ba, Kiros, Hinton (2016), Layer Normalization](https://arxiv.org/abs/1607.06450) | Paper gốc LN | Phần so sánh với BN |
| [Santurkar và cộng sự (2018), How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604) | Phản biện internal covariate shift | Thí nghiệm thêm nhiễu và phần độ trơn của loss |
| [Srivastava và cộng sự (2014), Dropout](https://jmlr.org/papers/v15/srivastava14a.html) | Paper gốc dropout, JMLR 15 | Phần mô hình và diễn giải mạng con |
| [Loshchilov & Hutter, Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | L2 khác weight decay với Adam, AdamW | Phần phân tích và thuật toán AdamW |
| [Smith (2015), Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) | LR range test | Mục ước lượng khoảng lr |
| [d2l.ai: Numerical Stability and Initialization](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html), [Batch Normalization](https://d2l.ai/chapter_convolutional-modern/batch-norm.html), [Dropout](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html) | Giáo trình mở có code chạy được | Ba mục này, làm bài tập cuối mục |
| [Nielsen, Neural Networks and Deep Learning, chương 4](http://neuralnetworksanddeeplearning.com/chap4.html) | Chứng minh trực quan xấp xỉ phổ quát, dựng bậc thang từ sigmoid | Toàn chương (có hình tương tác) |
| Leshno, Lin, Pinkus, Schocken (1993), Multilayer feedforward networks with a nonpolynomial activation function can approximate any function, *Neural Networks* 6(6) | Mở rộng xấp xỉ phổ quát ra mọi hàm kích hoạt không phải đa thức, gồm ReLU | Phát biểu định lý chính (chưa gắn link vì chưa mở được trang) |
| [Montúfar và cộng sự (2014), On the Number of Linear Regions of Deep Neural Networks](https://arxiv.org/abs/1402.1869) | Chặn dưới số vùng tuyến tính theo độ sâu (chuỗi 1) | Phát biểu kết quả chính; lưu ý đây là chặn dưới cho một số họ mạng |
| PyTorch: [Autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html), [Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html), [Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) | Tutorial chính thức cho mục 6.3 | Cả ba bài, gõ lại code |
| PyTorch docs: [CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html), [Linear](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html), [BatchNorm1d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm1d.html), [Dropout](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html), [AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html) | Giá trị mặc định chính xác (init, momentum, p, weight_decay) | Phần mô tả tham số và công thức |
| [Chip Huyen, ML Interviews Book, 8.3 Training neural networks](https://huyenchip.com/ml-interviews-book/contents/8.3-training-neural-networks.html) | Danh sách câu hỏi phỏng vấn | Làm hết các câu, so với chuỗi phỏng vấn của chương |

Định lý của Cybenko (1989) được dẫn qua CS231n; người viết chưa mở trực tiếp bài báo gốc, và cũng chưa mở trực tiếp bài của Leshno và cộng sự (1993) [cần kiểm chứng phát biểu và số trang]. Chưa mở lại được trang Chip Huyen mục 8.3 và trang Nielsen trong lần rà này: [cần kiểm chứng] URL và tiêu đề mục còn đúng.

**Từ khóa để tự tìm thêm:** dying ReLU, Kaiming fan_in và fan_out, pre-LN và post-LN Transformer, RMSNorm, GroupNorm, gradient accumulation với BatchNorm, LR range test, gradient clipping, label smoothing.

**Trước khi sang chương 7:** đạt từ 24/30 điểm mini-test, làm lại các bài ★★★ sai, tự viết lại được lab 1 (forward/backward + gradient check) trong 30 phút không nhìn code mẫu, và tìm được cả năm lỗi của bài săn bug.

Nguồn tra cứu ngày 24/10/2026.
