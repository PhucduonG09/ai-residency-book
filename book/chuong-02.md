# Chương 2 — Giải tích ma trận và backpropagation

## Mục tiêu và kiểm tra nhanh

Chương này dạy bạn tính gradient của bất kỳ hàm nào gặp trong học sâu, bằng tay và bằng code, rồi giải thích vì sao backpropagation tính được chúng rẻ như vậy. Chương 1 đã nói "backprop là chọn đúng thứ tự nhân" (Ví dụ 1.1); chương này chứng minh câu đó. Nắm được chương này thì đọc paper không còn sợ các dòng đạo hàm, và bạn tự kiểm được code của mình bằng gradient check. Thời lượng: khoảng 2 tuần, 8 giờ mỗi tuần.

**Sau chương này, bạn phải làm được:**

- ☐ Tính gradient, Jacobian, Hessian của một hàm nhiều biến; dùng Hessian để nói một điểm dừng là cực tiểu hay yên ngựa.
- ☐ Suy ra ∇(xᵀAx), ∇‖Ax − b‖₂², ∂L/∂W của lớp Y = XW bằng phương pháp vi phân, rồi kiểm lại bằng phân tích chiều.
- ☐ Vẽ đồ thị tính toán của một biểu thức, chạy forward và backward bằng tay, cộng đúng gradient ở chỗ một biến được dùng nhiều lần.
- ☐ Giải thích vì sao huấn luyện dùng reverse mode (VJP), khi nào forward mode (JVP) tốt hơn, và cái giá bộ nhớ của reverse mode.
- ☐ Viết backward cho MLP + softmax-cross-entropy theo batch bằng NumPy và kiểm nó bằng gradient check đúng cách (sai phân trung tâm, float64, sai số tương đối).

**Kiểm tra nhanh (5 phút, không dùng máy tính).** Câu i kiểm tra mục 2.i: đúng thì đọc lướt mục đó, sai thì đọc kỹ. Đúng cả 5 câu thì làm thẳng bài ★★★ và mini-test.

1. f(x, y) = xy. Dùng xấp xỉ bậc 1 quanh (2, 3), ước lượng f(2.1, 2.9).
2. A ∈ ℝⁿˣⁿ không đối xứng. Gradient của f(x) = xᵀAx là gì? Có phải 2Ax không?
3. Trong một đồ thị tính toán, biến y đi vào hai phép toán khác nhau. Gradient của loss theo y được ghép từ hai nhánh thế nào?
4. Hàm f: ℝ¹⁰⁰⁰⁰⁰⁰ → ℝ (một triệu tham số, loss vô hướng). Để có cả gradient, forward mode cần bao nhiêu lượt, reverse mode cần bao nhiêu lượt?
5. Logit z = (0, 0), nhãn đúng là lớp 1. Tính ∂L/∂z của cross-entropy sau softmax.

Đáp án ở mục Lời giải cuối chương.

**Quy ước của cả chương.** Vector là vector cột. Với hàm f: ℝⁿ → ℝᵐ, Jacobian J ∈ ℝᵐˣⁿ có Jᵢⱼ = ∂fᵢ/∂xⱼ (numerator layout: mỗi hàng là một đầu ra). Gradient của một hàm *vô hướng* theo một biến luôn **cùng shape với biến đó**: ∇f ∈ ℝⁿ là vector cột, ∂L/∂W cùng shape với W. Đây là cách viết của CS224n (ghi chú của khóa gọi là *shape convention*) và CS231n. Sách Mathematics for Machine Learning (MML) khác: Định nghĩa 5.5 của sách coi gradient là **vector hàng** 1 × n (numerator layout thuần, chọn vậy để quy tắc chuỗi viết gọn), nên ∇f của MML là chuyển vị của ∇f trong chương này. Với *đạo hàm của hàm vô hướng theo ma trận* thì hầu hết nguồn đều giữ shape của biến; chỗ các nguồn thật sự lệch nhau là đạo hàm vector theo vector (cách xếp Jacobian). Layout cụ thể của Matrix Cookbook ở từng công thức: [cần kiểm chứng], chưa mở lại được tài liệu. Vì vậy khi chép công thức từ nguồn khác, luôn kiểm lại bằng phân tích chiều.

## 2.1 Gradient, Jacobian, Hessian và khai triển Taylor

Đạo hàm là xấp xỉ tuyến tính tốt nhất của một hàm quanh một điểm. Gradient, Jacobian và Hessian chỉ là các cách viết xấp xỉ đó cho các loại hàm khác nhau.

**Đạo hàm riêng và gradient.** Với f: ℝⁿ → ℝ, đạo hàm riêng ∂f/∂xᵢ là tốc độ f thay đổi khi chỉ nhích xᵢ. Gom các đạo hàm riêng thành một vector cột ta có gradient ∇f. Định nghĩa thật sự của gradient là qua Taylor bậc 1:

$$
f(x + d) = f(x) + \nabla f(x)^{\top} d + o(\lVert d \rVert), \qquad \nabla f(x) = \Big(\tfrac{\partial f}{\partial x_1}, \dots, \tfrac{\partial f}{\partial x_n}\Big)^{\top}
$$

**Đạo hàm theo hướng.** Đi một bước nhỏ theo hướng đơn vị u thì f thay đổi với tốc độ ∇fᵀu = ‖∇f‖₂ cos θ, với θ là góc giữa u và ∇f. Từ đó suy ra ngay ba điều:

- Tốc độ lớn nhất khi u cùng hướng ∇f (cos θ = 1), và bằng ‖∇f‖₂. Đó là lý do gradient descent đi theo −∇f.
- Tốc độ bằng 0 khi u ⊥ ∇f. Những hướng đó tiếp xúc với đường đồng mức, nên tại mọi điểm có ∇f ≠ 0, ∇f vuông góc với đường đồng mức.
- Gradient chỉ là thông tin *bậc 1*: nó không thấy độ cong.

**Jacobian.** Với f: ℝⁿ → ℝᵐ, xấp xỉ tuyến tính là f(x + d) ≈ f(x) + J d, với J ∈ ℝᵐˣⁿ. Hàng i của J là (∇fᵢ)ᵀ; cột j cho biết mọi đầu ra thay đổi thế nào khi chỉ nhích xⱼ. Nhìn theo chương 1, J là *ma trận của phép biến đổi tuyến tính* gần đúng nhất với f quanh x. Khi m = n (J vuông), |det J| là hệ số phóng thể tích cục bộ.

**Hessian.** Hessian ∇²f ∈ ℝⁿˣⁿ có phần tử Hᵢⱼ = ∂²f/∂xᵢ∂xⱼ, chính là Jacobian của ∇f. Khi các đạo hàm bậc hai liên tục, H đối xứng (định lý Schwarz). Taylor bậc 2 thêm độ cong vào:

$$
f(x + d) \approx f(x) + \nabla f(x)^{\top} d + \tfrac{1}{2}\, d^{\top} H d
$$

Tại một điểm dừng (∇f = 0), mọi thứ do số hạng ½dᵀHd quyết định, và chương 1 đã cho ta công cụ: H đối xứng nên dᵀHd = ∑ λᵢd̂ᵢ², với d̂ = Qᵀd là tọa độ của d trong cơ sở vector riêng của H (H = QΛQᵀ). Mọi λᵢ > 0 ⇒ cực tiểu cục bộ; mọi λᵢ < 0 ⇒ cực đại; có cả dương lẫn âm ⇒ điểm yên ngựa (saddle point). Nếu có trị riêng bằng 0 (các trị riêng còn lại không âm hoặc không dương) thì Hessian không kết luận được, phải xét số hạng bậc cao hơn.

% manim: GradientDongMuc

**Ví dụ giải mẫu 2.1 — gradient, Hessian, Taylor.** f(x₁, x₂) = x₁²x₂ + 3x₂ tại x₀ = (1, 2).

- f(x₀) = 2 + 6 = 8. ∇f = (2x₁x₂, x₁² + 3) = (4, 4).
- H có hàng (2x₂, 2x₁) và (2x₁, 0), tại x₀ là (4, 2) và (2, 0). Trị riêng: λ² − 4λ − 4 = 0 ⇒ λ = 2 ± 2√2 ≈ 4.828 và −0.828. Có một trị riêng âm, nên f không lồi quanh x₀. Lưu ý x₀ không phải điểm dừng (∇f ≠ 0): ở đây Hessian chỉ cho biết độ cong, chưa nói gì về cực trị.
- Bước nhỏ d = (0.1, 0.2): giá trị thật f(1.1, 2.2) = 9.262. Bậc 1: 8 + ∇fᵀd = 8 + 1.2 = 9.2 (sai 0.062). Bậc 2: thêm ½dᵀHd = ½(0.04 + 0.08 + 0) = 0.06, ra 9.26 (sai 0.002).
- Bước d = (0.1, 0.1): thật f(1.1, 2.1) = 8.841. Bậc 1: 8 + 0.8 = 8.8 (sai 0.041). Bậc 2: thêm ½dᵀHd = ½(0.04 + 0.04) = 0.04, ra 8.84 (sai 0.001).
- Bước lớn d = (0.5, 0.5): thật 13.125, bậc 1 cho 12.0 (sai 1.125), bậc 2 cho 13.0 (sai 0.125).
- Đạo hàm theo hướng u = (0.6, 0.8): ∇fᵀu = 2.4 + 3.2 = 5.6. Lớn nhất là ‖∇f‖₂ = 4√2 ≈ 5.657, đạt khi u = (1, 1)/√2.
- *Một điểm dừng thật:* g(x₁, x₂) = x₁² − x₂² có ∇g = (2x₁, −2x₂) = 0 tại (0, 0), và H = diag(2, −2) có trị riêng 2 và −2, nên đó là yên ngựa: g(t, 0) = t² > 0 nhưng g(0, t) = −t² < 0. Khi có trị riêng 0, Hessian bất lực: x₁⁴ và −x₁⁴ đều có H = 0 tại 0, mà một hàm có cực tiểu, hàm kia có cực đại.

Đọc kết quả: dọc theo cùng hướng (0.5, 0.5), tăng bước 5 lần (từ (0.1, 0.1) lên (0.5, 0.5)) thì sai số bậc 1 tăng từ 0.041 lên 1.125 (khoảng 27 lần), sai số bậc 2 tăng từ 0.001 lên 0.125 (125 lần = 5³). Sai số bậc 1 tỉ lệ ‖d‖², bậc 2 tỉ lệ ‖d‖³ — ở hàm này phần dư bậc 3 là hằng số nên tỉ lệ 125 là chính xác, còn 27 lệch khỏi 25 vì có cả số hạng bậc 3. Taylor chỉ đúng *cục bộ*, và đó là lý do learning rate phải nhỏ: mỗi bước gradient descent tin vào một mô hình tuyến tính chỉ đúng gần điểm hiện tại.

% manim: TaylorXapXi

**Ví dụ giải mẫu 2.2 — Jacobian là ma trận của xấp xỉ tuyến tính.** Đổi tọa độ cực g(r, θ) = (r cos θ, r sin θ) tại r = 2, θ = π/6.

- J có hàng (cos θ, −r sin θ) và (sin θ, r cos θ), tại điểm đã cho là (0.866, −1.000) và (0.500, 1.732).
- det J = r cos²θ + r sin²θ = r = 2. Một ô nhỏ dr × dθ quanh điểm này bị biến thành gần như một hình bình hành có hai cạnh là hai cột của J (nhân dr, dθ), diện tích gấp 2 lần.
- Nối với chương 1: det là hệ số phóng diện tích của một ma trận; ở đây ma trận đó là Jacobian, nên hệ số chỉ đúng *cục bộ* và thay đổi theo điểm.

% manim: JacobianOVuong

**Vì sao ML cần cái này.**

- **Gradient descent** là "đi theo −∇f một bước nhỏ" — hệ quả trực tiếp của đạo hàm theo hướng và của việc Taylor bậc 1 chỉ đúng cục bộ.
- **Điểm yên ngựa:** nhiều phân tích và thực nghiệm (Dauphin và cộng sự, NeurIPS 2014) cho thấy ở không gian nhiều chiều, điểm dừng dạng yên ngựa (Hessian có cả trị riêng dương lẫn âm) thường gặp hơn cực tiểu địa phương xấu. Đây là quan sát và lập luận có điều kiện, không phải định lý cho mọi mạng. Dù vậy, gradient bằng 0 không có nghĩa là đã tối ưu xong.
- **Độ cong và learning rate:** chương 1 cho điều kiện η < 2/λₘₐₓ trên hàm toàn phương. Hessian chính là ma trận của hàm toàn phương đó, nên λₘₐₓ(H) quyết định learning rate tối đa, κ(H) quyết định tốc độ.
- **Normalizing flows** đổi biến mật độ xác suất bằng log|det J| của một phép biến đổi khả nghịch; nhiều kiến trúc flow (coupling, autoregressive) được thiết kế để J tam giác, cho det rẻ; không phải flow nào cũng theo khuôn này (Glow dùng phân rã LU cho phép nhân chập 1 × 1).

## 2.2 Đạo hàm theo vector và ma trận: các đồng nhất thức

Đừng học thuộc bảng công thức. Hãy học một phương pháp — vi phân cộng vết — và một phép kiểm — phân tích chiều. Với hai thứ đó bạn tự suy ra mọi công thức trong chương này trong một phút.

**Phương pháp vi phân.** Viết vi phân df theo dX rồi đưa về dạng tr(Gᵀ dX). Khi đó ∂f/∂X = G (cùng shape với X). Với biến là vector thì dạng tương ứng là df = gᵀdx ⇒ ∇f = g. Các quy tắc cần dùng:

$$
d(AXB) = A\,(dX)\,B, \qquad d(XY) = (dX)\,Y + X\,(dY), \qquad d(x^{\top}x) = 2x^{\top}dx, \qquad \operatorname{tr}(ABC) = \operatorname{tr}(CAB)
$$

Mẹo vết (chương 1, mục 1.4): một số vô hướng bằng vết của chính nó, và vết hoán vị vòng được. Đó là cách "kéo" dX ra cuối biểu thức.

**Bảng đồng nhất thức (tự suy ra được bằng phương pháp trên):**

| Hàm f | Gradient | Ghi chú |
| --- | --- | --- |
| aᵀx | a | Hàm tuyến tính |
| xᵀAx | (A + Aᵀ)x | Bằng 2Ax chỉ khi A đối xứng. Hessian = A + Aᵀ |
| ‖Ax − b‖₂² | 2Aᵀ(Ax − b) | Hessian = 2AᵀA (gradient đã gặp ở chương 1, mục 1.3; Hessian là mới ở đây) |
| tr(BX) | Bᵀ | ∂/∂Xᵢⱼ của ∑ BⱼᵢXᵢⱼ là Bⱼᵢ |
| log det X | X⁻ᵀ | Với det X > 0; mỗi phần tử của X được coi là độc lập |
| L(Y), Y = XW, biết G = ∂L/∂Y | ∂L/∂W = XᵀG, ∂L/∂X = GWᵀ | Công thức của mọi lớp Linear |

Chứng minh hai dòng quan trọng nhất:

$$
d(x^{\top}Ax) = dx^{\top}Ax + x^{\top}A\,dx = x^{\top}A^{\top}dx + x^{\top}A\,dx = \big((A + A^{\top})x\big)^{\top} dx
$$

$$
dL = \operatorname{tr}(G^{\top} dY) = \operatorname{tr}(G^{\top} X\, dW) = \operatorname{tr}\big((X^{\top}G)^{\top} dW\big) \;\Rightarrow\; \frac{\partial L}{\partial W} = X^{\top} G
$$

(Ở dòng đầu, dxᵀAx là số vô hướng nên bằng chuyển vị của nó, xᵀAᵀdx.)

**Phân tích chiều — phép kiểm 10 giây.** Với X ∈ ℝᴺˣᴰ, W ∈ ℝᴰˣᴴ, G ∈ ℝᴺˣᴴ: ∂L/∂W phải có shape D × H. Chỉ có một cách ghép X và G ra D × H là XᵀG. ∂L/∂X phải là N × D, chỉ có GWᵀ. Khi quên công thức, ghép shape là đủ. Khi mọi chiều bằng nhau (ma trận vuông) thì phép kiểm này không còn phân biệt được W với Wᵀ — đó là lúc bug lọt qua (bài săn bug).

**Khi bí: viết ra từng phần tử.** Với Yₙₕ = ∑ₖ XₙₖWₖₕ, ta có ∂L/∂Wₖₕ = ∑ₙ (∂L/∂Yₙₕ) Xₙₖ = (XᵀG)ₖₕ. Chậm nhưng không bao giờ sai, và không cần tensor bốn chiều nào.

**Ví dụ giải mẫu 2.3 — bẫy xᵀAx với A không đối xứng.** A có hàng (1, 2) và (0, 3), x = (1, −1).

- xᵀAx = x₁² + 2x₁x₂ + 3x₂² = 1 − 2 + 3 = 2.
- Công thức đúng: A + Aᵀ có hàng (2, 2), (2, 6), nên ∇f = (2 − 2, 2 − 6) = (0, −4).
- Nếu áp nhầm 2Ax = 2(1 − 2, −3) = (−2, −6). Sai cả hai thành phần.
- Sai phân trung tâm cho (0, −4), xác nhận công thức đúng. Lý do sâu: xᵀAx chỉ "thấy" phần đối xứng ½(A + Aᵀ), vì phần phản đối xứng cho xᵀKx = 0.

**Ví dụ giải mẫu 2.4 — gradient bình phương tối thiểu.** Giữ nguyên ma trận A của ví dụ khớp đường thẳng ở chương 1 nhưng **đổi b** thành (1, 2, 4), nên nghiệm khác chương 1. A có hàng (1, 0), (1, 1), (1, 2); b = (1, 2, 4); w = (0, 1).

- Phần dư r = Aw − b = (0, 1, 2) − (1, 2, 4) = (−1, −1, −2), loss ‖r‖₂² = 6.
- Vi phân: d‖r‖₂² = 2rᵀdr = 2rᵀA dw ⇒ ∇ = 2Aᵀr = 2(−4, −5) = (−8, −10).
- Cho ∇ = 0 được phương trình chuẩn AᵀAw = Aᵀb của chương 1 (phần bình phương tối thiểu). Nghiệm w* = (5/6, 3/2) ≈ (0.8333, 1.5); tại đó gradient bằng 0 (đã kiểm).
- Hessian 2AᵀA không phụ thuộc w — hàm này đúng là toàn phương, Taylor bậc 2 chính xác tuyệt đối.

**Ví dụ giải mẫu 2.5 — log det.** X có hàng (2, 1) và (1, 3). det X = 5, log det X ≈ 1.6094.

- X⁻¹ có hàng (0.6, −0.2) và (−0.2, 0.4); X đối xứng nên X⁻ᵀ = X⁻¹.
- Nhiễu riêng phần tử ở hàng 1 cột 2 (giữ nguyên phần tử đối xứng của nó) một lượng ±10⁻⁶: sai phân cho −0.2000, khớp phần tử tương ứng của X⁻ᵀ.
- Giả định ngầm: mọi phần tử của X là biến độc lập. Nếu X bị ràng buộc đối xứng (ma trận hiệp phương sai), nhiễu một phần tử kéo theo phần tử đối xứng, và gradient theo tham số độc lập khác đi — Matrix Cookbook có mục riêng cho trường hợp này.

**Vì sao ML cần cái này.**

- **Mọi lớp Linear** dùng đúng hai công thức ∂L/∂W = XᵀG và ∂L/∂X = GWᵀ. Attention, MLP, embedding projection đều là các phép nhân ma trận như vậy.
- **Hồi quy tuyến tính, ridge, linear probe:** gradient 2Aᵀ(Aw − b) và Hessian 2AᵀA nối thẳng với số điều kiện ở chương 1 (xem bài tập 5, 6 của chương này).
- **Log-likelihood Gaussian** chứa log det Σ; huấn luyện Gaussian process hay mô hình hỗn hợp cần gradient X⁻ᵀ.

**Thực tế tính toán.** Framework không dùng công thức cho *cả biểu thức* (như ∇‖Ax − b‖₂²); nó ghép VJP của từng phép toán sơ cấp theo quy tắc chuỗi (mục 2.3–2.4), và VJP của phép nhân ma trận chính là dòng cuối của bảng trên (XᵀG và GWᵀ). Công thức gộp dùng để *kiểm* framework và để *suy luận* trên giấy (ví dụ Hessian của least squares). Khi viết kernel backward tay (CUDA, Triton) thì bạn cần đúng các công thức này.

## 2.3 Quy tắc chuỗi và đồ thị tính toán

Backprop chỉ là quy tắc chuỗi, áp lên một đồ thị, theo thứ tự ngược. Mỗi nút chỉ cần biết gradient cục bộ của chính nó; nút không cần biết gì về phần còn lại của mạng.

**Quy tắc chuỗi dạng ma trận.** Nếu h = f ∘ g, với g: ℝⁿ → ℝᵏ và f: ℝᵏ → ℝᵐ, thì Jacobian của hợp là tích các Jacobian, theo đúng thứ tự áp dụng ngược lại:

$$
J_{f \circ g}(x) = J_f\big(g(x)\big)\, J_g(x) \qquad\in \mathbb{R}^{m \times n}
$$

Với một chuỗi dài x → z₁ → z₂ → ⋯ → L (L vô hướng), gradient là một tích dài các Jacobian. Thứ tự tính tích đó là toàn bộ nội dung của mục 2.4.

**Đồ thị tính toán.** Viết biểu thức thành một đồ thị có hướng không chu trình (DAG): nút lá là đầu vào và tham số, mỗi nút trong là một phép toán sơ cấp. Hai lượt quét:

1. *Forward:* đi từ lá tới đầu ra, tính và **lưu** mọi giá trị trung gian.
2. *Backward:* gán ∂L/∂L = 1 ở đầu ra, đi ngược theo thứ tự topo. Ở mỗi nút: gradient đổ về từ trên × gradient cục bộ = gradient gửi cho từng đầu vào.

**Các mẫu cổng cần thuộc** (gradient từ trên là g):

| Cổng | Forward | Gửi về đầu vào | Trực giác |
| --- | --- | --- | --- |
| Cộng | a + b | g cho a, g cho b | Phân phát nguyên vẹn |
| Nhân | a · b | g·b cho a, g·a cho b | Đổi chéo: gradient của a tỉ lệ với b |
| Max | max(a, b) | g cho đầu vào lớn hơn, 0 cho đầu kia | Định tuyến |
| ReLU | max(0, a) | g nếu a > 0, ngược lại 0 | Chặn |
| Sigmoid | s = σ(a) | g · s(1 − s) | Hệ số tối đa 1/4 |

**Fan-out: cộng, không gán.** Nếu một biến đi vào nhiều nút, gradient theo nó là *tổng* các đóng góp qua mọi đường đi. Đây là quy tắc chuỗi nhiều biến, không phải quy ước:

$$
\frac{\partial L}{\partial y} = \sum_{\text{mỗi nút con } c \text{ của } y} \frac{\partial L}{\partial c}\,\frac{\partial c}{\partial y}
$$

Công thức viết cho y vô hướng; nếu y là vector thì mỗi số hạng là một VJP, (∂c/∂y)ᵀ ∂L/∂c, và ta cộng các VJP đó. Trong code, điều này có nghĩa là `grad += ...`, không bao giờ `grad = ...`. Bias trong lớp Linear theo batch là fan-out kinh điển: cùng một b được cộng vào N hàng, nên ∂L/∂b là tổng của G theo trục batch.

**Ví dụ giải mẫu 2.6 — một neuron sigmoid trên đồ thị.** w₁ = 0.5, w₂ = −1, b = 0.2, x₁ = 2, x₂ = 1, s = σ(w₁x₁ + w₂x₂ + b).

- Forward: w₁x₁ = 1, w₂x₂ = −1, z = 1 − 1 + 0.2 = 0.2, s = σ(0.2) ≈ 0.549834.
- Cổng sigmoid: gradient cục bộ s(1 − s) ≈ 0.247517. Bắt đầu với ∂s/∂s = 1, ta có ∂s/∂z ≈ 0.247517.
- Cổng cộng phân phát nguyên: ∂s/∂b ≈ 0.247517, và mỗi nhánh nhân nhận 0.247517.
- Cổng nhân đổi chéo: ∂s/∂w₁ = 0.247517 × x₁ ≈ 0.495033; ∂s/∂x₁ = 0.247517 × w₁ ≈ 0.123758; ∂s/∂w₂ = 0.247517 × x₂ ≈ 0.247517; ∂s/∂x₂ = 0.247517 × w₂ ≈ −0.247517.

Bài học từ cổng nhân: gradient của trọng số tỉ lệ với độ lớn của đầu vào. Một đặc trưng có thang đo gấp 1000 lần đặc trưng khác sẽ làm gradient của trọng số tương ứng lớn gấp 1000 lần, và một learning rate chung không hợp cho cả hai. Đó là một lý do phải chuẩn hóa đầu vào.

% manim: DoThiNeuron

**Ví dụ giải mẫu 2.7 — fan-out phải cộng.** f = (x + y) · y tại x = 3, y = 2.

- Forward: q = x + y = 5, f = q · y = 10.
- Cổng nhân f = q · y: ∂f/∂q = y = 2 và nhánh trực tiếp ∂f/∂y = q = 5.
- Cổng cộng q = x + y gửi nguyên 2 về x và về y.
- ∂f/∂x = 2. ∂f/∂y = 5 (đường trực tiếp) + 2 (qua q) = 7. Kiểm bằng khai triển: f = xy + y², ∂f/∂y = x + 2y = 7.
- Nếu code viết `y.grad = ...` ở mỗi nhánh, kết quả là 5 hoặc 2 tùy nhánh nào chạy sau — sai mà không có lỗi nào được báo.

% manim: FanOutCongDon

**Vì sao ML cần cái này.**

- **Mọi framework autodiff** (PyTorch, JAX, TensorFlow) là cài đặt của đồ thị này: mỗi phép toán định nghĩa forward và một hàm backward cục bộ.
- **Chia sẻ trọng số** (RNN dùng cùng W ở mọi bước thời gian, tied embedding dùng cùng ma trận cho đầu vào và đầu ra) và **residual connection** x + F(x) (activation x đi vào cả F lẫn phép cộng) đều là fan-out: gradient từ mọi chỗ dùng được cộng lại. Lớp embedding là ca đặc biệt cho NLP: chỉ hàng của token xuất hiện trong batch nhận gradient (thưa), và token lặp thì các đóng góp cộng dồn (bài tập 9b). Backward qua softmax-attention dùng đúng Jacobian diag(p) − ppᵀ của mục 2.5, áp lên từng hàng của ma trận điểm attention.
- **Residual connection** là cổng cộng: nó chuyển nguyên gradient về các lớp sớm, một lý do mạng rất sâu có residual vẫn huấn luyện được.
- **Vanishing gradient với sigmoid:** mỗi cổng sigmoid nhân gradient với hệ số ≤ 1/4. Qua 10 lớp sigmoid (bỏ qua trọng số), hệ số tối đa là 0.25¹⁰ ≈ 9.5 × 10⁻⁷.
- **Exploding gradient** là mặt đối ngẫu: gradient là một tích dài các hệ số, nên hệ số mỗi lớp chỉ cần hơi lớn hơn 1 là tích nổ (1.1⁵⁰ ≈ 117), hơi nhỏ hơn 1 là tích tắt (0.9⁵⁰ ≈ 0.005). Cách chữa phổ biến: gradient clipping, khởi tạo hợp lý, normalization, residual connection.

## 2.4 Forward mode, reverse mode và vector–Jacobian product

Backprop rẻ vì nó không bao giờ dựng Jacobian. Nó chỉ nhân một *vector* với Jacobian từ đầu ra ngược về, và với loss vô hướng thì một lượt như vậy cho toàn bộ gradient.

**Hai cách đi qua tích Jacobian.** Với x → z₁ → ⋯ → zₖ = y, Jacobian toàn phần là J = Jₖ ⋯ J₂J₁. Ta có thể nhân nó với một vector theo hai chiều:

$$
\text{JVP (forward mode):}\;\; J v = J_k\big(\cdots\big(J_2\,(J_1 v)\big)\big) \qquad\qquad \text{VJP (reverse mode):}\;\; u^{\top} J = \big(\big((u^{\top} J_k) \cdots\big) J_2\big) J_1
$$

- **JVP** (Jacobian–vector product) đẩy một *tiếp tuyến* v cùng chiều với forward: "nếu nhích đầu vào theo hướng v, mọi đầu ra nhích bao nhiêu?". Mỗi lượt cho một tổ hợp các cột của J; với v = eⱼ là đúng cột j.
- **VJP** (vector–Jacobian product) đẩy một *cotangent* u ngược từ đầu ra: "đầu ra được đánh trọng số u nhạy thế nào với từng đầu vào?". Mỗi lượt cho một tổ hợp các hàng; với u = eᵢ là đúng hàng i, tức ∇fᵢ.

**Đếm chi phí.** Với f: ℝⁿ → ℝᵐ, dựng cả J cần n lượt JVP hoặc m lượt VJP; mỗi lượt tốn cỡ một lần chạy f (một hằng số nhỏ lần). Loss huấn luyện là vô hướng (m = 1), n là số tham số (hàng triệu, hàng tỉ). Reverse mode cho cả gradient trong **một** lượt. Đây là lý do backprop = reverse mode.

**Ví dụ giải mẫu 2.8 — JVP và VJP trên cùng một Jacobian.** f(x₁, x₂, x₃) = (x₁x₂, x₂x₃) tại x = (1, 2, 3).

- J có hàng (x₂, x₁, 0) = (2, 1, 0) và (0, x₃, x₂) = (0, 3, 2). J ∈ ℝ²ˣ³.
- JVP với v = (1, 0, 1): Jv = (2, 2). Một lượt forward, ra một tổ hợp cột.
- VJP với u = (1, −1): uᵀJ = (2, −2, −2). Một lượt backward, ra một tổ hợp hàng.
- Dựng cả J: forward cần 3 lượt (số đầu vào), reverse cần 2 lượt (số đầu ra).
- **Đếm chi phí ở quy mô thật:** chuỗi 10 lớp tuyến tính 1000 × 1000 dẫn tới loss vô hướng. Reverse mode: 10 tích vector × ma trận, mỗi tích 10⁶ phép nhân, tổng 10⁷. Nếu nhân các Jacobian thành ma trận đầy đủ từ phía đầu vào: 10 tích ma trận × ma trận, mỗi tích 10⁹, tổng khoảng 10¹⁰ — chậm hơn khoảng 1000 lần. Đây đúng là Ví dụ 1.1 của chương 1, áp lên cả mạng.

% manim: ForwardVsReverse

**Không bao giờ dựng Jacobian.** Jacobian của phép toán phần tử (ReLU, sigmoid, nhân phần tử) là ma trận đường chéo n × n; VJP qua nó chỉ là nhân phần tử với vector đạo hàm, O(n) thay vì O(n²). Jacobian của Y = XW theo W là một tensor bốn chiều (ma trận (NH) × (DH) nếu duỗi ra), nhưng VJP của nó chỉ là XᵀG. Mỗi phép toán trong framework cài sẵn VJP của nó theo dạng gọn như vậy.

**Forward mode bằng dual numbers.** Thay mỗi số x bằng một cặp (giá trị, đạo hàm) và định nghĩa phép toán trên cặp: (a, a′) + (b, b′) = (a + b, a′ + b′), (a, a′)·(b, b′) = (ab, a′b + ab′), sin(a, a′) = (sin a, a′ cos a). Ví dụ f(x) = x² sin x tại x = 2, bắt đầu từ (2, 1):

- x² = (4, 4); sin x ≈ (0.909297, −0.416147).
- Tích: (4 × 0.909297, 4 × 0.909297 + 4 × (−0.416147)) ≈ (3.637190, 1.972602).
- Đúng bằng f(2) và f′(2) = 2x sin x + x² cos x ≈ 1.972602. Một lượt forward, không cần lưu gì.

**Tích Hessian–vector (HVP).** Hv là JVP của hàm ∇f theo hướng v. Với f ở Ví dụ 2.1, ∇f = (2x₁x₂, x₁² + 3); đạo hàm theo hướng v = (1, 0) tại (1, 2) là (2x₂, 2x₁) = (4, 2), đúng cột đầu của H. Trong thực tế ta tính HVP bằng "forward-over-reverse": một lượt reverse cho ∇f, rồi một JVP qua chính chương trình tính ∇f. Chi phí chỉ cỡ vài lần tính gradient, không cần dựng H ∈ ℝⁿˣⁿ (với n = 10⁹, H có 10¹⁸ phần tử). HVP dùng trong phương pháp Newton–CG, ước lượng λₘₐₓ(H) bằng power iteration (chương 1, lab 1) và phân tích độ sắc của cực tiểu.

**Cái giá của reverse mode: bộ nhớ.** Backward ở lớp k cần các giá trị forward của lớp k (ví dụ X trong XᵀG, mặt nạ A > 0 của ReLU). Nên forward phải *lưu* mọi activation cho tới khi backward dùng xong. Bộ nhớ tỉ lệ với độ sâu tính toán, nhân với batch size. Forward mode không cần lưu gì, nhưng cần n lượt.

**Gradient checkpointing** đổi tính toán lấy bộ nhớ: chỉ lưu activation ở một số "điểm mốc", lúc backward cần đoạn nào thì chạy lại forward của đoạn đó. Chia L lớp thành khoảng √L đoạn thì bộ nhớ activation giảm từ cỡ L xuống cỡ √L, đổi lại khoảng thêm một lượt forward (ý tưởng từ Chen, Xu, Zhang, Guestrin 2016, "Training Deep Nets with Sublinear Memory Cost"; xem bảng nguồn).

**Vì sao ML cần cái này.**

- **Huấn luyện** = một lượt forward + một lượt reverse, cho gradient của hàng tỉ tham số với chi phí chỉ vài lần forward.
- **Vì sao inference tốn ít VRAM hơn training:** trong `torch.no_grad()` hoặc `inference_mode`, framework không lưu activation cho backward.
- **Forward mode có chỗ dùng:** khi số đầu vào nhỏ hơn số đầu ra (phân tích độ nhạy của nhiều đầu ra theo vài siêu tham số), và bên trong HVP.

**Thực tế tính toán.** PyTorch ghi lại một DAG các đối tượng `Function` trong lúc forward (đồ thị dựng lại mỗi lần chạy, nên `if`/`for` của Python vẫn hoạt động); mỗi tensor kết quả trỏ tới `grad_fn` đã tạo ra nó và các "saved tensors" cần cho backward. `loss.backward()` chạy VJP với u = 1. Gọi `backward(v)` trên một output không vô hướng chính là tính VJP vᵀJ — framework không tự dựng Jacobian. Ba chi tiết hay bị hỏi: (1) `.grad` được *cộng dồn* qua các lần `backward()` (đúng tinh thần fan-out), nên mỗi bước huấn luyện phải gọi `zero_grad()`; cũng có thể lợi dụng để tích lũy gradient qua nhiều micro-batch. Gọi `backward()` lần hai trên cùng một đồ thị thì thường báo lỗi vì saved tensors đã được giải phóng (đồ thị chỉ gồm phép cộng thì không có gì để báo), trừ khi dùng `retain_graph=True` (khi đó `.grad` cộng dồn thêm). (2) Phép in-place (`x.add_(1)`, `x *= 2`) có thể ghi đè giá trị mà backward cần; mỗi tensor có version counter, và PyTorch báo lỗi lúc `backward()` nếu một saved tensor đã bị sửa sau khi lưu. (3) `detach()` trả về tensor tách khỏi đồ thị, gradient không chảy qua nó; dùng cho target cố định, và khi ghi log nên dùng `loss.item()` hoặc `loss.detach()` để không giữ cả đồ thị trong bộ nhớ. JAX tách bạch hơn: `jax.jvp`, `jax.vjp`, và `jacfwd` (dựng J bằng JVP, hợp với J "cao": nhiều đầu ra) / `jacrev` (bằng VJP, hợp với J "rộng": nhiều đầu vào).

## 2.5 Backprop cho MLP, softmax + cross-entropy, và gradient checking

Gom ba mục trước vào một mạng thật: backward của MLP theo batch chỉ gồm các công thức Linear, ReLU và một dòng p − y. Sau đó học cách tự kiểm mọi gradient bằng sai phân — kỹ năng quan trọng nhất của chương.

**Softmax và Jacobian của nó.** p = softmax(z), pᵢ = exp(zᵢ)/∑ⱼ exp(zⱼ). Đạo hàm ∂pᵢ/∂zⱼ = pᵢ(δᵢⱼ − pⱼ), với δᵢⱼ = 1 khi i = j và 0 khi khác:

$$
J_{\text{softmax}} = \operatorname{diag}(p) - p\,p^{\top}
$$

Ma trận này đối xứng, bán xác định dương, và *suy biến*: J1 = p − p(pᵀ1) = 0, vì pᵀ1 = 1. Hướng suy biến 1 nói rằng cộng cùng một hằng số vào mọi logit không đổi p.

**Gộp với cross-entropy: p − y.** Với nhãn one-hot y, L = −∑ yᵢ log pᵢ = −log pₖ (k là lớp đúng). Gradient theo p là −y/p (chia phần tử); VJP qua Jacobian của softmax cho:

$$
\frac{\partial L}{\partial z} = J_{\text{softmax}}^{\top}\Big(-\frac{y}{p}\Big) = -y + p\,(\mathbf{1}^{\top} y) = p - y
$$

Phép chia cho p biến mất: gradient gộp không bao giờ chia cho một xác suất nhỏ. Mỗi thành phần của p − y nằm trong [−1, 1], và khi mô hình sai nặng (pₖ ≈ 0) thì gradient ở lớp đúng ≈ −1, không bị bão hòa.

**Log-sum-exp ổn định.** log pₖ = zₖ − log ∑ exp(zⱼ). Với logit lớn, exp tràn số. Vì cộng hằng số vào mọi logit không đổi p, ta trừ m = max zⱼ trước:

$$
\log \sum_j e^{z_j} = m + \log \sum_j e^{z_j - m}, \qquad m = \max_j z_j
$$

**Ví dụ giải mẫu 2.9 — softmax + cross-entropy.** Logit z = (2, 1, 0), nhãn đúng là lớp 2, y = (0, 1, 0).

- p = softmax(z) ≈ (0.665241, 0.244728, 0.090031). Loss = −log 0.244728 ≈ 1.407606.
- ∂L/∂z = p − y ≈ (0.665241, −0.755272, 0.090031): logit lớp đúng bị đẩy lên (gradient âm, gradient descent trừ đi gradient), hai logit sai bị kéo xuống.
- Kiểm qua Jacobian: Jacobian của softmax có hàng ≈ (0.2227, −0.1628, −0.0599), (−0.1628, 0.1848, −0.0220), (−0.0599, −0.0220, 0.0819). Mỗi hàng cộng bằng 0; trị riêng ≈ 0, 0.1186, 0.3709 (PSD, suy biến). VJP của −y/p qua ma trận này cho đúng p − y.
- Ổn định số: với z = (1000, 999, 998), `np.exp(z)` cho `inf` và kết quả chia ra `nan`. Trừ max được (0, −1, −2), ra đúng p ≈ (0.665241, 0.244728, 0.090031) như trên.

% manim: SoftmaxGradient

**Backward của MLP theo batch.** Dữ liệu X ∈ ℝᴺˣᴰ, mỗi hàng một mẫu; loss là *trung bình* cross-entropy:

$$
A = XW_1 + \mathbf{1}b_1^{\top},\quad H = \operatorname{ReLU}(A),\quad S = HW_2 + \mathbf{1}b_2^{\top},\quad L = \frac{1}{N}\sum_{n} \mathrm{CE}(S_n, y_n)
$$

$$
dS = \frac{P - Y}{N},\quad \frac{\partial L}{\partial W_2} = H^{\top} dS,\quad \frac{\partial L}{\partial b_2} = dS^{\top}\mathbf{1},\quad dH = dS\,W_2^{\top},\quad dA = dH \odot [A > 0],\quad \frac{\partial L}{\partial W_1} = X^{\top} dA,\quad \frac{\partial L}{\partial b_1} = dA^{\top}\mathbf{1}
$$

Ở đây dS, dH, dA là cách viết kiểu code của ∂L/∂S, ∂L/∂H, ∂L/∂A (gradient của loss), không phải vi phân d như ở mục 2.2. P là ma trận xác suất, Y là one-hot, ⊙ là nhân phần tử, [A > 0] là mặt nạ 0/1. Hệ số 1/N đến từ việc loss là trung bình; ∂L/∂b là tổng theo trục batch (fan-out). Thêm weight decay ½λ‖W‖² thì cộng λW vào gradient của W.

**Ví dụ giải mẫu 2.10 — backprop MLP 2–2–2 trọn vẹn bằng tay.** Quy ước cột (một mẫu): h = ReLU(W₁x + b₁), o = W₂h + b₂. Cho x = (1, 2); W₁ có hàng (0.1, 0.2), (−0.3, 0.4); b₁ = (0, 0.1); W₂ có hàng (1, −1), (0.5, 0.5); b₂ = 0; nhãn lớp 1 (lớp đầu tiên).

- Forward: a = W₁x + b₁ = (0.5, 0.6); h = (0.5, 0.6) (cả hai dương); o = (0.5 − 0.6, 0.25 + 0.3) = (−0.1, 0.55); p ≈ (0.342990, 0.657010); L = −log 0.342990 ≈ 1.070055.
- δo = p − y ≈ (−0.657010, 0.657010).
- ∂L/∂W₂ = δo hᵀ, hàng ≈ (−0.328505, −0.394206) và (0.328505, 0.394206).
- δh = W₂ᵀδo ≈ (−0.657010 + 0.328505, 0.657010 + 0.328505) = (−0.328505, 0.985516).
- ReLU mở cả hai (a > 0), nên δa = δh. ∂L/∂W₁ = δa xᵀ, hàng ≈ (−0.328505, −0.657010) và (0.985516, 1.971031).
- Gradient check trung tâm (h = 10⁻⁵, float64) trên W₁ cho sai số tương đối cỡ 10⁻¹¹. Lab 5 tái hiện đúng các số này.

**Gradient checking.** Ước lượng đạo hàm bằng sai phân rồi so với gradient giải tích:

$$
\text{tiến: } \frac{f(x + h) - f(x)}{h} = f'(x) + O(h), \qquad \text{trung tâm: } \frac{f(x + h) - f(x - h)}{2h} = f'(x) + O(h^2)
$$

Khai triển Taylor hai phía làm số hạng bậc 2 triệt tiêu, nên sai phân trung tâm chính xác hơn hẳn với cùng h. Nhưng h không thể nhỏ tùy ý: mỗi giá trị f bị làm tròn cỡ ε|f| (ε là machine epsilon), và phép chia cho h khuếch đại sai số này thành cỡ ε/h. Tổng sai số cỡ h² + ε/h nhỏ nhất khi h cỡ ∛ε: khoảng 6 × 10⁻⁶ với float64 (ε ≈ 2.2 × 10⁻¹⁶), khoảng 5 × 10⁻³ với float32 (ε ≈ 1.2 × 10⁻⁷).

So sánh bằng **sai số tương đối** chứ không phải tuyệt đối, vì gradient có thể rất lớn hoặc rất nhỏ:

$$
\text{rel\_err} = \frac{\max |g_{\text{giải tích}} - g_{\text{số}}|}{\max\big(\max|g_{\text{giải tích}}|,\; \max|g_{\text{số}}|\big)}
$$

Ngưỡng tham khảo (theo ghi chú CS231n, chỉ là kinh nghiệm): dưới 10⁻⁷ là tốt; khoảng 10⁻⁴ cần xem lại; trên 10⁻² gần như chắc có bug. Công thức rel_err ở trên chuẩn hóa theo max của *cả mảng*, là một biến thể gọn; các ngưỡng của CS231n được nêu cho sai số tương đối theo từng phần tử [cần kiểm chứng]. Chuẩn hóa toàn cục có thể che lỗi ở phần tử nhỏ: gradient (100, 0.001) so với (100, 0.002) cho 10⁻⁵ theo cách toàn cục nhưng 0.5 theo từng phần tử. Vì vậy hãy kiểm riêng từng tham số (như các lab làm) và xem thêm sai số theo phần tử. Mạng càng sâu thì sai số tích lũy càng lớn, nên ngưỡng nới ra; khi mạng có ReLU, 10⁻⁴ thường vẫn chấp nhận được vì điểm gãy làm sai phân lệch mà không có bug.

**Ví dụ giải mẫu 2.11 — chọn h khi gradient check.** f = sin tại x = 1, đạo hàm thật cos 1. Sai số tuyệt đối (chạy thật trong lab 2):

| h | float64 trung tâm | float64 tiến | float32 trung tâm | float32 tiến |
| --- | --- | --- | --- | --- |
| 10⁻² | 9.0 × 10⁻⁶ | 4.2 × 10⁻³ | 1.0 × 10⁻⁵ | 4.2 × 10⁻³ |
| 10⁻³ | 9.0 × 10⁻⁸ | 4.2 × 10⁻⁴ | 1.4 × 10⁻⁵ | 4.0 × 10⁻⁴ |
| 10⁻⁵ | 1.1 × 10⁻¹¹ | 4.2 × 10⁻⁶ | 8.8 × 10⁻⁴ | 3.9 × 10⁻³ |
| 10⁻⁷ | 1.9 × 10⁻¹⁰ | 4.2 × 10⁻⁸ | 5.6 × 10⁻² | 5.6 × 10⁻² |
| 10⁻⁸ | 2.6 × 10⁻⁹ | 3.0 × 10⁻⁹ | 5.4 × 10⁻¹ | 5.4 × 10⁻¹ |

- Ở h lớn, sai số trung tâm giảm 100 lần khi h giảm 10 lần (O(h²)), sai số tiến giảm 10 lần (O(h)).
- float64: tốt nhất 1.1 × 10⁻¹¹ tại h = 10⁻⁵, gần với ∛ε ≈ 6 × 10⁻⁶. Giảm h tiếp thì sai số làm tròn thắng và sai số tăng lại.
- float32: với h = 10⁻⁷ (giá trị có vẻ "an toàn"), sai số tuyệt đối 0.056, tức khoảng 10% giá trị thật cos 1 ≈ 0.540. Một gradient hoàn toàn đúng sẽ bị báo là sai.
- float32, h ≲ 3 × 10⁻⁸: 1 ± h làm tròn về đúng 1 trong float32 (ε ≈ 1.2 × 10⁻⁷), nên hai giá trị hàm bằng nhau, ước lượng bằng đúng 0 và sai số bằng chính cos 1 = 0.540. Ở vùng này mô hình "sai số làm tròn ∝ ε/h" không còn áp dụng, đường sai số float32 đi ngang ở 0.54 chứ không tiếp tục dốc lên. Kiểm gradient luôn ở float64.

**Điểm gãy (kink).** ReLU tại x = 10⁻⁶ với h = 10⁻⁵: sai phân trung tâm cho (ReLU(1.1 × 10⁻⁵) − ReLU(−0.9 × 10⁻⁵))/(2 × 10⁻⁵) = 0.55, trong khi đạo hàm giải tích là 1. Không có bug nào; khoảng [x − h, x + h] vắt qua điểm gãy. Với nhiều ReLU, chuyện này xảy ra thường xuyên hơn ta nghĩ.

**Vì sao ML cần cái này.**

- **p − y** là gradient đầu tiên của mọi bộ phân loại và mọi language model (cross-entropy trên từ vựng). Nó cũng giải thích vì sao cross-entropy + softmax học nhanh hơn MSE + sigmoid khi mô hình sai nặng (bài tập 10).
- **Gradient check** là cách duy nhất chắc chắn khi bạn tự viết một lớp mới, một loss mới hay một kernel backward tay.
- **Ổn định số** (trừ max, log-softmax gộp) là lý do `CrossEntropyLoss` của PyTorch nhận logit, không nhận xác suất.

**Thực tế tính toán.**

- Kiểm trên ít dữ liệu (vài mẫu) và ít tham số, ở float64, tắt mọi nguồn ngẫu nhiên (dropout, augmentation), kiểm riêng loss dữ liệu rồi mới cộng regularization (regularization lớn có thể lấn át và che bug ở phần dữ liệu).
- Kiểm ở vài điểm ngẫu nhiên, không chỉ tại khởi tạo: tại khởi tạo, nhiều activation gần 0 có thể khiến một gradient sai cũng nhỏ và lọt qua.
- Training thật chạy float32, bf16 hoặc thấp hơn; gradient check chỉ dùng để xác minh code, không phản ánh sai số khi train.

## Bài tập giấy bút

14 bài, làm theo thứ tự. Mục tiêu: hết ★ trong 1 buổi, hết ★★ trong 2 buổi, ★★★ dành cho Thứ Bảy. Mỗi gradient tính tay nên kiểm lại bằng phân tích chiều trước khi xem lời giải. Bài nào sai ghi vào sổ lỗi kèm lý do sai.

**★ Hiểu định nghĩa**

1. f(x, y) = x³ + 2xy² − y. Tính ∇f và Hessian tại (1, 1). Hessian có xác định dương không? Tính đạo hàm theo hướng u = (0.6, −0.8).
2. Một lớp Linear Y = XW với X ∈ ℝ⁶⁴ˣ⁷⁸⁴, W ∈ ℝ⁷⁸⁴ˣ²⁵⁶. Cho shape của ∂L/∂W và ∂L/∂X. Nếu dựng đầy đủ Jacobian ∂Y/∂W (mọi phần tử của Y theo mọi phần tử của W) thì nó có bao nhiêu phần tử?
3. σ(x) áp phần tử lên x ∈ ℝⁿ. Viết Jacobian của nó. Vì sao VJP qua lớp này tốn O(n) chứ không O(n²)?
4. Viết từng phần tử để chứng minh ∇‖x‖₂² = 2x và ∂(aᵀXb)/∂X = abᵀ, với a ∈ ℝᵐ, X ∈ ℝᵐˣⁿ, b ∈ ℝⁿ. Kiểm shape của vế phải.

**★★ Mức đề thi đầu vào**

5. Với A, b của Ví dụ 2.4, Hessian của ‖Aw − b‖₂² là bao nhiêu? Tính trị riêng của nó, learning rate lớn nhất để gradient descent ổn định, và số điều kiện κ.
6. Ridge: f(w) = ‖Aw − b‖₂² + λ‖w‖₂². Tính ∇f, suy ra nghiệm đóng (so với chương 1). Với A của Ví dụ 2.4 và λ = 1, tính trị riêng Hessian và κ. Regularization thay đổi κ thế nào?
7. Lớp Y = XW + 1bᵀ với X có hàng (1, 2), (0, 1); W có hàng (1, 0, −1), (2, 1, 0); b = (0, 1, 0). Biết G = ∂L/∂Y có hàng (1, 0, 1), (0, 1, −1). Tính Y, ∂L/∂W, ∂L/∂X, ∂L/∂b.
8. Lớp kiểu `nn.Linear` của PyTorch: Y = XWᵀ + 1bᵀ, với W ∈ ℝᴴˣᴰ (mỗi hàng là một neuron). Biết G = ∂L/∂Y, dùng phương pháp vi phân để suy ∂L/∂W, ∂L/∂X, ∂L/∂b. Kiểm shape với N = 64, D = 784, H = 256, rồi nối kết quả với quy ước cột y = Wx của một mẫu.
9. f = (x + y) · max(y, z) tại x = 1, y = 2, z = 0. Vẽ đồ thị tính toán, chạy forward và backward, cho ∂f/∂x, ∂f/∂y, ∂f/∂z. Cổng max làm gì với gradient? (b) Lớp embedding E ∈ ℝ⁵⁰⁰⁰ˣ⁶⁴ nhận chuỗi token id (7, 3, 7) và trả về ba hàng E₇, E₃, E₇; gọi g₁, g₂, g₃ ∈ ℝ⁶⁴ là gradient ở ba vị trí đầu ra. Hàng nào của ∂L/∂E khác 0 và bằng gì? Viết kết quả dưới dạng XᵀG với X là ma trận one-hot.
10. Một neuron sigmoid dự đoán s = σ(z), nhãn đúng t = 1, hiện tại z = −5 (mô hình sai nặng). So sánh ∂L/∂z cho cross-entropy L = −log s và cho MSE L = ½(s − 1)². Loss nào học nhanh hơn từ điểm này, nhanh hơn bao nhiêu lần?

**★★★ Mức phỏng vấn nghiên cứu**

11. Đặt ℓ(z) = log ∑ⱼ exp(zⱼ) (log-sum-exp). Chứng minh ∇ℓ = softmax(z) và ∇²ℓ = diag(p) − ppᵀ; chứng minh ma trận này bán xác định dương và có hạng n − 1. Với nhãn y là một phân phối bất kỳ (∑ yᵢ = 1), cross-entropy là L = ℓ(z) − yᵀz: suy ra Hessian của L theo logit không phụ thuộc nhãn, L lồi theo z nhưng không lồi chặt, và giải thích bằng hướng 1 vì sao logit tối ưu không duy nhất.
12. Một MLP có L = 24 lớp Linear d × d (d = 1024, bỏ qua bias), batch B = 32. Đếm số phép nhân của forward và của backward (gồm cả ∂L/∂X và ∂L/∂W mỗi lớp). Cần lưu bao nhiêu số thực activation cho backward, bao nhiêu MB ở float32? Gradient checkpointing với √L đoạn thay đổi gì? Vì sao với Transformer thật, activation thường chiếm bộ nhớ hơn hẳn tham số?
13. Với f của Ví dụ 2.1, tính Hv với v = (0, 1) bằng cách lấy đạo hàm theo hướng của ∇f. Giải thích vì sao HVP bằng forward-over-reverse tốn cỡ vài lần tính gradient, còn dựng H tốn n lần. Nêu một thuật toán chỉ cần HVP để tìm λₘₐₓ(H).
14. Gradient check báo sai số tương đối 0.3 trên W₁ của một mạng có ReLU, trong khi bạn tin code đúng. Nêu ít nhất bốn nguyên nhân có thể (cả "có bug" và "không có bug"), và cho mỗi nguyên nhân một cách kiểm. Ngược lại, nêu một tình huống gradient check *pass* dù code có bug.

## Bài tính nhanh

15 bài, mỗi bài tối đa 2 phút, làm bằng giấy bút và bấm giờ cả bộ. Mục tiêu: xong cả 15 bài trong 25 phút, sai không quá 1 bài. Làm lại bộ này sau 3 ngày và sau 1 tuần.

1. ∇(aᵀx) với a = (1, −2, 3).
2. x = (1, 2, 2). Tính ∇‖x‖₂² và ∇‖x‖₂.
3. A có hàng (2, 1), (1, 3); x = (1, 1). Tính ∇(xᵀAx).
4. Jacobian của f(x₁, x₂) = (x₁ + x₂, x₁x₂) tại (1, 3), và det của nó.
5. Hessian của f = x² + 3xy + y². Điểm (0, 0) là cực tiểu, cực đại hay yên ngựa?
6. σ(0) và σ′(0).
7. tanh′(0) và đạo hàm của ReLU tại −2.
8. softmax(0, ln 3).
9. Xác suất của lớp đúng là 0.25. Cross-entropy bằng bao nhiêu (lấy ln 2 ≈ 0.693)?
10. p = (0.25, 0.75), nhãn đúng lớp 2. Tính ∂L/∂z.
11. y = Wx với một mẫu: x = (1, 2), W ∈ ℝ¹ˣ², ∂L/∂y = 3. Tính ∂L/∂W.
12. f: ℝ³ → ℝ¹⁰⁰. Để dựng cả Jacobian, dùng forward hay reverse mode, bao nhiêu lượt?
13. f(x) = x³ tại x = 1, h = 0.1. Tính sai phân trung tâm và sai phân tiến; mỗi cái sai bao nhiêu?
14. Mười cổng sigmoid nối tiếp, mỗi cổng nhân gradient với tối đa 1/4. Hệ số tối đa là bao nhiêu (bậc lũy thừa của 10)?
15. Bias được broadcast qua batch 3 mẫu; G = ∂L/∂Y có hàng (1, 2), (3, 4), (5, 6). Tính ∂L/∂b.

## Lab code và bài săn bug

Năm lab dưới đây đã chạy với Python 3 và NumPy 2.5; toàn bộ notebook lời giải chạy khoảng 3 giây trên CPU. Không dùng PyTorch: bạn tự viết mọi gradient, framework để dành cho chương 6. Mỗi lab có dùng số ngẫu nhiên đều tự tạo bộ sinh với seed riêng, nên chạy lại một ô hay chạy lệch thứ tự vẫn ra đúng số ở cột "kết quả mong đợi" (các số cuối cùng của sai số cỡ 10⁻¹⁰ có thể lệch chút theo phiên bản NumPy/BLAS). Cách học đúng: tự viết từng lab trước, chỉ đối chiếu code mẫu khi kết quả lệch với cột "kết quả mong đợi".

| Lab | Việc phải làm | Kết quả mong đợi |
| --- | --- | --- |
| 1. Gradient số | `numerical_grad` (sai phân trung tâm cho mảng shape bất kỳ) và `rel_error`; kiểm ∇(xᵀAx) và ∂ log det X/∂X | (A + Aᵀ)x: rel_error 5.3 × 10⁻¹²; công thức sai 2Ax: 0.57; X⁻ᵀ: 1.4 × 10⁻¹⁰ (X không đối xứng); đáp án quên chuyển vị X⁻¹: 0.41, phải bị bắt |
| 2. Quét h | Sai số sai phân tiến và trung tâm của sin tại 1 theo h, float64 và float32 | Bảng Ví dụ 2.11. h tốt nhất: 10⁻⁵ (float64, sai 1.1 × 10⁻¹¹), 10⁻² (float32, sai 1.0 × 10⁻⁵); float32 với h = 10⁻⁷ sai 5.6 × 10⁻² |
| 3. Mini autograd | Lớp `Value` với `+`, `*`, `sigmoid`, `backward()` theo thứ tự topo, dùng `+=` | Ví dụ 2.7: ∂f/∂y = 7; Ví dụ 2.6: ∂s/∂w₁ = 0.495033, ∂s/∂x₁ = 0.123758; so với sai phân 4.1 × 10⁻¹¹ |
| 4. JVP và VJP | Chuỗi 3 lớp tanh ℝ⁶ → ℝ³; dựng J bằng 6 JVP và bằng 3 VJP | Hai Jacobian khớp tới 2.6 × 10⁻¹⁶; so với sai phân 2.4 × 10⁻¹¹; uᵀ(Jv) = (uᵀJ)v tới 12 chữ số |
| 5a. MLP + softmax-CE | `mlp_loss_grad` theo batch, có L2; khớp Ví dụ 2.10 và gradient check | L = 1.070055, ∂L/∂W₁ khớp Ví dụ 2.10; rel_error mọi tham số 3.5 × 10⁻¹¹ đến 1.1 × 10⁻¹⁰ |
| 5b. Huấn luyện | Dữ liệu xoắn ốc 3 lớp, 300 điểm, 100 neuron ẩn, L2 λ = 10⁻³, 3000 bước GD với η = 1 | Loss 1.108 → 0.244; accuracy train 98.7% |

Code nằm trong notebook `labs/chuong02/lab.ipynb` (bản có chỗ TODO để bạn tự viết, kèm ô kiểm tra tự động) và `labs/chuong02/loigiai.ipynb` (bản lời giải đã chạy kiểm). Mở từ trang **Lab** của app.

**Câu hỏi sau lab.**

- Lab 2: vì sao float32 đạt sai số tốt nhất ở h = 10⁻², lớn hơn nhiều so với float64? Sai số tốt nhất của float32 (1.0 × 10⁻⁵) gần với con số lý thuyết nào?
- Lab 4: Jacobian có shape 3 × 6. Nếu đổi thành ℝ⁶ → ℝ¹⁰⁰⁰, bạn sẽ dựng J bằng mode nào? Còn khi đầu ra là một loss vô hướng?
- Lab 5b: loss cuối 0.244 trong khi accuracy 98.7%. Vì sao loss không về gần 0? *Gợi ý: có hai thành phần trong loss.*

**Bài săn bug.** Hàm dưới đây tính gradient cho MLP hai lớp + softmax-CE (loss là trung bình). Nó chạy không báo lỗi và mọi shape đều đúng vì trong bài thử N = H = C = 4. Với learning rate 0.1, loss vẫn giảm (từ 1.35 xuống 0.25 sau 300 bước trên 4 mẫu), nên nhìn đường loss không thấy gì bất thường. Nhưng gradient check cho sai số tương đối 0.80, 1.00, 0.75, 0.75 trên W₁, b₁, W₂, b₂. Với learning rate 0.5, tại bước 50 bản đúng có loss 0.22 còn bản lỗi đã nổ lên 308. **Lưu ý khi tự chạy bản lỗi lâu hơn 50 bước:** loss tiếp tục vọt lên khoảng 2.5 × 10⁷ quanh bước 122, trọng số phình tới cỡ 10⁴, rồi loss rơi về 0.0000 (bản đúng ở bước 300 là 0.0065). Bản lỗi "thắng" bản đúng không có nghĩa gradient của nó đúng: đừng kết luận gì từ giá trị loss cuối; hãy so hai bản ở cùng mốc bước và xem cả quỹ đạo. Tìm cả **ba** lỗi trước khi xem lời giải.

```python
def grads_buggy(W1, b1, W2, b2, X, y):
    N = X.shape[0]
    A = X @ W1 + b1
    Hh = np.maximum(0, A)
    S = Hh @ W2 + b2
    P = np.exp(S - S.max(axis=1, keepdims=True)); P /= P.sum(axis=1, keepdims=True)
    dS = P.copy(); dS[np.arange(N), y] -= 1
    dW2 = Hh.T @ dS
    db2 = dS.sum(axis=0)
    dH = dS @ W2
    dA = dH * (A > 0)
    dW1 = X.T @ dA
    db1 = dA.sum(axis=1)
    return dW1, db1, dW2, db2
```

## Bài dựng cảnh Manim

Mỗi chương có một bài tự dựng cảnh: muốn vẽ được một khái niệm, bạn phải hiểu từng bước của nó. Giới hạn **1 giờ** (tính vào luồng C), không chỉnh màu mè.

**Đề bài.** Dùng `manim/chuong02_scenes.py` làm mẫu, viết cảnh `ChonBuocH` vào thư mục `manim/cua-toi/`:

1. Vẽ hệ trục log–log: trục ngang là h từ 10⁻⁹ đến 10⁻¹, trục dọc là sai số tuyệt đối của sai phân trung tâm cho f = sin tại x = 1. Manim cần LaTeX cho nhãn số tự động, nên tự đặt nhãn bằng `Text`, và tự đổi tọa độ sang log₁₀.
2. Vẽ hai đường: float64 và float32, mỗi đường tính tại khoảng 30 giá trị h.
3. Đánh dấu điểm thấp nhất của mỗi đường và ghi giá trị lý thuyết ∛ε cho từng kiểu số.

**Tiêu chí tự chấm.** (1) Đúng toán: hai đường có đường float64 có hình chữ V trọn vẹn, đường float32 cũng có đáy nhưng có đoạn phẳng ở cos 1 ≈ 0.54 khi h dưới khoảng 3 × 10⁻⁸ (giải thích được vì sao: xem Ví dụ 2.11); nhánh phải dốc 2 (sai số cắt cụt ~ h²), đáy float64 gần h = 10⁻⁵ và đáy float32 gần h = 10⁻²–10⁻³, khớp bảng Ví dụ 2.11. (2) Có tiêu đề và phụ đề nói người xem phải thấy điều gì. (3) Dài không quá 30 giây.

Render bằng nút **Render cảnh của tôi** ở trang Manim. Lời giải tham khảo nằm ở `manim/loigiai/chuong02_chon_buoc_h.py` — chỉ mở sau khi đã tự làm.

## Phỏng vấn, bẫy, thực tế khác lý thuyết

**Cách luyện.** Mỗi chuỗi có 4 tầng câu hỏi, đào sâu dần như người phỏng vấn thật. Đây là các dạng câu hỏi thường gặp trong phỏng vấn ML nói chung, tổng hợp từ nội dung chuẩn của các khóa học ở mục cuối; không có công ty nào công bố bộ đề chính thức. Trả lời to bằng tiếng Việt trước, sau đó nói lại bằng tiếng Anh, 2 phút mỗi tầng, ghi âm và tự nghe lại. Câu trả lời mẫu tiếng Anh chỉ phủ tầng 1–2; tầng 3–4 bạn tự soạn, rồi đối chiếu với lời giải các bài tập liên quan.

**Chuỗi 1 — Vì sao backprop hiệu quả**

1. Backpropagation là gì?
2. Vì sao dùng reverse mode chứ không phải forward mode để huấn luyện?
3. Khi nào forward mode tốt hơn? (Ít đầu vào, nhiều đầu ra; JVP bên trong Hessian–vector product.)
4. Cái giá của reverse mode là gì, và giảm nó thế nào? (Lưu activation, bộ nhớ tỉ lệ độ sâu × batch; gradient checkpointing — bài tập 12. Hỏi thêm: gọi `backward()` hai lần thì sao? `.grad` cộng dồn, và đồ thị đã giải phóng nên lần hai báo lỗi nếu không có `retain_graph`.)

> *Model answer:* Backpropagation is reverse-mode automatic differentiation applied to the computational graph of the loss. The forward pass computes and stores every intermediate value; the backward pass starts from the scalar loss and propagates vector–Jacobian products from the output back to the inputs, so each node only needs its local derivative. For a function from n inputs to m outputs, forward mode needs n passes to get the full Jacobian while reverse mode needs m. Since the training loss is a scalar and n is millions or billions of parameters, reverse mode gives the entire gradient for roughly the cost of a few forward passes, and it never materializes a Jacobian.

**Chuỗi 2 — Softmax và cross-entropy**

1. Đạo hàm của cross-entropy theo logit là gì?
2. Jacobian của riêng softmax là gì, vì sao nó suy biến?
3. Vì sao framework gộp softmax và cross-entropy, và nhận logit thay vì xác suất?
4. Vì sao cross-entropy + softmax không bão hòa như MSE + sigmoid khi mô hình sai nặng? (Bài tập 10.)

> *Model answer:* The gradient of softmax cross-entropy with respect to the logits is simply p − y, the predicted probabilities minus the one-hot label, divided by the batch size if the loss is averaged. Softmax alone has Jacobian diag(p) − ppᵀ, which is symmetric positive semi-definite and singular, because adding the same constant to every logit leaves the probabilities unchanged. When you chain it with −y/p, the gradient of the negative log-likelihood with respect to p, the division by p cancels, which is why the combined gradient is so clean and bounded between −1 and 1 per component.

**Chuỗi 3 — Gradient checking**

1. Bạn kiểm một backward tự viết thế nào?
2. Vì sao dùng sai phân trung tâm?
3. Chọn h thế nào, dùng float32 hay float64, đọc sai số tương đối ra sao?
4. Vì sao gradient check có thể báo sai khi code đúng, hoặc báo đúng khi code sai? (Bài tập 14.)

> *Model answer:* I compare the analytic gradient with a numerical estimate from finite differences, coordinate by coordinate, and report the relative error rather than the absolute error. I use the centered difference, whose error is O(h²) instead of O(h) for the one-sided version, because the second-order Taylor terms cancel. The step size cannot be arbitrarily small, since round-off error grows like machine epsilon divided by h, so the best h is around the cube root of epsilon — roughly 10⁻⁵ in float64. I always run the check in double precision, on a few data points, with dropout and other randomness turned off.

**Chuỗi 4 — Đạo hàm ma trận nhanh**

1. Gradient của xᵀAx là gì?
2. Với Y = XW và G = ∂L/∂Y, ∂L/∂W bằng gì?
3. Chứng minh bằng phương pháp vi phân và kiểm bằng phân tích chiều.
4. Hessian của loss bình phương tối thiểu là gì, và nó nói gì về tốc độ của gradient descent? (2AᵀA; κ(AᵀA) = κ(A)², xem chương 1 và bài tập 5 của chương này.)

> *Model answer:* The gradient of xᵀAx is (A + Aᵀ)x, which reduces to 2Ax only when A is symmetric — the quadratic form only sees the symmetric part of A. For a linear layer Y = XW with upstream gradient G, the weight gradient is XᵀG and the input gradient is GWᵀ. A quick sanity check is shape matching: the gradient must have the same shape as the variable, and with X of size N × D and G of size N × H, XᵀG is the only product that gives D × H. When all dimensions are equal, shape checks stop helping, so I fall back to writing the derivative entry by entry or to a gradient check.

**Từ vựng nên nói trôi chảy:** Jacobian, Hessian, directional derivative, chain rule, computational graph, forward pass / backward pass, upstream gradient, local gradient, vector–Jacobian product, Jacobian–vector product, reverse-mode / forward-mode autodiff, gradient checkpointing, log-sum-exp trick, centered finite difference, relative error, numerator layout.

**Bẫy thường gặp.**

- Dùng ∇(xᵀAx) = 2Ax khi A không đối xứng (Ví dụ 2.3).
- Chép công thức từ nguồn dùng denominator layout rồi ra kết quả chuyển vị. Chữa: gradient phải cùng shape với biến.
- Gán `grad = ...` thay vì `grad += ...` ở fan-out; tính ∂L/∂b bằng tổng theo trục đặc trưng thay vì trục batch.
- Quên `zero_grad()`: `.grad` cộng dồn qua các lần `backward()`, nên gradient của bước này lẫn gradient của mọi bước trước. Hai bẫy cùng họ: phép in-place làm hỏng saved tensor (lỗi version counter), và giữ `loss` thay vì `loss.item()` khi ghi log làm đồ thị không được giải phóng.
- Quên chia N khi loss là trung bình (gradient lớn gấp N lần, learning rate "tốt" thay đổi theo batch size), và ma trận vuông che lỗi: nhầm W với Wᵀ không báo lỗi shape (bài săn bug).
- Tính softmax bằng `exp(z) / exp(z).sum()` không trừ max: `nan` ngay khi logit vượt khoảng 709 (giới hạn của exp ở float64), sớm hơn nhiều ở float16.

**Thực tế khác lý thuyết.**

- **Autograd không tính "đạo hàm đúng" tại điểm không khả vi.** Tại ReLU(0) hay max có hai đầu vào bằng nhau, framework chọn một quy ước (PyTorch ghi rõ trong tài liệu autograd rằng nó ưu tiên gradient nếu có, nếu không thì chọn một subgradient theo quy tắc riêng). Thêm nữa, mask *sau* một phép toán sinh `nan` (ví dụ chia cho 0 rồi dùng `where` để bỏ) vẫn cho gradient `nan`, vì backward vẫn đi qua nhánh bị bỏ; phải mask đầu vào *trước* phép toán.
- **Bộ nhớ, không phải FLOP, mới thường là giới hạn của reverse mode.** Bạn hết VRAM khi tăng batch lúc train thường xuyên hơn lúc inference, vì train phải lưu activation cho backward (với LLM, KV cache khi inference cũng chiếm nhiều bộ nhớ, nhưng đó là chuyện khác với activation lưu cho backward). Gradient checkpointing, mixed precision và giảm batch là các cách xử lý trong thực tế.
- **Gradient check chỉ đáng tin ở float64 và trên ít dữ liệu**, trong khi training chạy float32 hoặc bf16. Hai việc dùng hai độ chính xác khác nhau: kiểm ở float64, train ở độ chính xác thấp. Ngoài ra không ai dựng Hessian đầy đủ của mạng thật; mọi phân tích bậc hai thực tế đều đi qua HVP.

## Mini-test 30 phút

Làm sau khi xong bài tập và lab, bấm giờ đúng 30 phút, không xem sách, không dùng máy tính (câu 5 chỉ đọc code, không chạy). Tổng 30 điểm. **Từ 24 điểm:** sang chương 3. **18 đến dưới 24:** ôn lại mục ứng với câu sai rồi làm lại sau 3 ngày. **Dưới 18:** học lại chương.

1. **(4 điểm)** f(x) = xᵀAx + bᵀx với A có hàng (1, 2), (0, 1) và b = (1, −1). Tính ∇f tại x = (1, 1) và Hessian. f có lồi không?
2. **(4 điểm)** f = (xy + x) · z tại x = 2, y = 1, z = 3. Vẽ đồ thị tính toán, chạy forward và backward, cho ∂f/∂x, ∂f/∂y, ∂f/∂z. Chỉ ra chỗ fan-out.
3. **(5 điểm)** L = ½‖XW + 1bᵀ‖², chuẩn Frobenius, với X có hàng (1, 0), (2, 1), W = I (2 × 2), b = 0. Tính ∂L/∂W và ∂L/∂b bằng công thức của lớp Linear.
4. **(5 điểm)** Logit z = (0, ln 2, ln 5), nhãn đúng lớp 2. Tính p, loss, ∂L/∂z. Nếu một batch gồm 4 bản sao của mẫu này và loss là trung bình, gradient theo logit của mỗi mẫu bằng bao nhiêu?
5. **(6 điểm)** Đọc code và trả lời không chạy máy. X ∈ ℝᴺˣᴰ, W ∈ ℝᴰˣᴴ, dY ∈ ℝᴺˣᴴ là ∂L/∂Y của Y = X @ W + b.

```python
def linear_backward(X, W, dY):
    dW = X.T @ dY
    db = dY.sum(axis=1)
    dX = dY @ W
    return dW, db, dX
```

(a) Dòng nào sai, sửa thế nào? (b) Với điều kiện nào của N, D, H thì code sai vẫn chạy không lỗi shape, kể cả khi đi tiếp vào bước cập nhật tham số? (c) Bạn phát hiện các lỗi đó bằng cách nào ngoài đọc code?

6. **(6 điểm)** Tối đa 5 câu: vì sao huấn luyện một mô hình cần nhiều bộ nhớ GPU hơn hẳn inference với cùng batch, và gradient checkpointing giải quyết điều đó bằng cách nào, đổi lại cái gì?

## Flashcard

20 thẻ của chương nằm trong `flashcards/chuong02.csv`. Ôn ở trang **Ôn tập** của app: app tự lên lịch theo hộp Leitner (thẻ nhớ được sẽ giãn lịch ôn 1 → 3 → 7 → 14 ngày, thẻ quên quay lại hộp 1). File CSV này cũng nhập thẳng được vào Anki nếu bạn thích dùng Anki.

## Lời giải

Chỉ đọc sau khi đã tự thử ít nhất 20 phút. Với bài chứng minh, so ý chính chứ không so từng chữ. Mọi đáp số bằng số trong chương đã được kiểm lại bằng NumPy.

**Kiểm tra nhanh**

1. ∇f = (y, x) = (3, 2) tại (2, 3). f(2.1, 2.9) ≈ 6 + 3 · 0.1 + 2 · (−0.1) = 6.1. Giá trị thật 6.09; sai 0.01 = |d₁d₂|, đúng bằng phần bậc 2 bị bỏ.
2. (A + Aᵀ)x. Chỉ bằng 2Ax khi A đối xứng.
3. Cộng lại: ∂L/∂y = tổng các đóng góp qua mọi nút con của y (quy tắc chuỗi nhiều biến). Trong code là `+=`.
4. Forward mode: 10⁶ lượt (mỗi lượt một đạo hàm riêng). Reverse mode: 1 lượt.
5. p = (0.5, 0.5), y = (1, 0), nên ∂L/∂z = p − y = (−0.5, 0.5).

**Bài tập giấy bút**

1. ∇f = (3x² + 2y², 4xy − 1) = (5, 3). H có hàng (6x, 4y), (4y, 4x) = (6, 4), (4, 4). det = 24 − 16 = 8 > 0 và tr = 10 > 0, nên cả hai trị riêng dương (≈ 0.877 và 9.123): xác định dương. ∇fᵀu = 5 · 0.6 + 3 · (−0.8) = 0.6.
2. ∂L/∂W ∈ ℝ⁷⁸⁴ˣ²⁵⁶ (cùng shape với W), ∂L/∂X ∈ ℝ⁶⁴ˣ⁷⁸⁴. Jacobian đầy đủ: Y có 64 · 256 = 16 384 phần tử, W có 784 · 256 = 200 704 phần tử, tích là 3 288 334 336 ≈ 3.3 × 10⁹ phần tử (khoảng 13 GB ở float32) — trong khi XᵀG chỉ cần 200 704 số. Hầu hết các phần tử của Jacobian đó bằng 0: Yₙₕ chỉ phụ thuộc cột h của W.
3. J = diag(σ′(x₁), …, σ′(xₙ)) với σ′ = σ(1 − σ), vì đầu ra i chỉ phụ thuộc đầu vào i. VJP uᵀJ = u ⊙ σ′(x): n phép nhân. Không ai lưu ma trận n × n đó.
4. ‖x‖₂² = ∑ xₖ² nên ∂/∂xᵢ = 2xᵢ, tức ∇ = 2x. aᵀXb = ∑ᵢ ∑ⱼ aᵢXᵢⱼbⱼ; đạo hàm theo Xₖₗ chỉ còn số hạng i = k, j = l, bằng aₖbₗ = (abᵀ)ₖₗ. Vậy ∂(aᵀXb)/∂X = abᵀ ∈ ℝᵐˣⁿ, cùng shape với X (một tích ngoài, giống δo hᵀ của Ví dụ 2.10).
5. H = 2AᵀA, AᵀA có hàng (3, 3), (3, 5), nên H có hàng (6, 6), (6, 10). Trị riêng 8 ± √40 ≈ 14.325 và 1.675. Ổn định khi η < 2/14.325 ≈ 0.140. κ ≈ 14.325/1.675 ≈ 8.55. Đây là κ(AᵀA) = κ(A)² của chương 1.
6. ∇f = 2Aᵀ(Aw − b) + 2λw. Cho bằng 0: (AᵀA + λI)w = Aᵀb, đúng nghiệm ridge của chương 1. Hessian 2(AᵀA + λI), với λ = 1 có hàng (8, 6), (6, 12), trị riêng 10 ± √40 ≈ 16.325 và 3.675, κ ≈ 4.44. Regularization cộng 2λ vào mọi trị riêng, nên trị riêng nhỏ được nâng lên tương đối nhiều hơn: κ giảm từ 8.55 xuống 4.44 và gradient descent hội tụ nhanh hơn.
7. XW có hàng (1 + 4, 0 + 2, −1 + 0) = (5, 2, −1) và (2, 1, 0); cộng b được Y có hàng (5, 3, −1), (2, 2, 0). ∂L/∂W = XᵀG: Xᵀ có hàng (1, 0), (2, 1), nên XᵀG có hàng (1, 0, 1) và (2, 1, 1). ∂L/∂X = GWᵀ có hàng (0, 2) và (1, 1). ∂L/∂b = tổng các hàng của G = (1, 1, 0). Kiểm shape: 2 × 3, 2 × 2, 3 — đều khớp biến. Đã kiểm bằng sai phân.
8. dY = X dWᵀ + dX Wᵀ + 1 dbᵀ. Phần W: dL = tr(Gᵀ X dWᵀ) = tr(dWᵀ GᵀX) = tr((GᵀX)ᵀ dW) ⇒ ∂L/∂W = GᵀX ∈ ℝᴴˣᴰ, cùng shape với W. Phần X: dL = tr(Gᵀ dX Wᵀ) = tr(WᵀGᵀ dX) = tr((GW)ᵀ dX) ⇒ ∂L/∂X = GW ∈ ℝᴺˣᴰ. Phần b: dL = tr(Gᵀ 1 dbᵀ) = (Gᵀ1)ᵀ db ⇒ ∂L/∂b = Gᵀ1 ∈ ℝᴴ (tổng G theo trục batch). Với N = 64, D = 784, H = 256: 256 × 784, 64 × 784 và 256. Nối với quy ước cột: một mẫu y = Wx cho ∂L/∂W = g xᵀ (tích ngoài); cộng N mẫu được ∑ₙ gₙxₙᵀ = GᵀX, đúng kết quả trên. So với Y = XW (∂L/∂W = XᵀG), W lưu chuyển vị nên ∂L/∂W cũng chuyển vị.
9. Forward: q = x + y = 3, m = max(2, 0) = 2, f = q · m = 6. Backward: cổng nhân cho ∂f/∂q = m = 2, ∂f/∂m = q = 3. Cổng max chuyển toàn bộ 3 cho y (đầu vào lớn hơn) và 0 cho z. Cổng cộng chuyển 2 cho x và 2 cho y. Vậy ∂f/∂x = 2, ∂f/∂y = 2 + 3 = 5 (fan-out), ∂f/∂z = 0. Cổng max *định tuyến*: chỉ đầu vào thắng nhận gradient — đó cũng là cách max-pooling backprop. (b) Hàng 7 của ∂L/∂E bằng g₁ + g₃ (token lặp nên cộng dồn, giống fan-out của Ví dụ 2.7), hàng 3 bằng g₂, 4 998 hàng còn lại bằng 0: gradient thưa. Với X ∈ ℝ³ˣ⁵⁰⁰⁰ one-hot (hàng n có số 1 ở cột id của token n), đầu ra là XE nên ∂L/∂E = XᵀG, đúng công thức lớp Linear. Trong thực tế người ta không dựng X mà chỉ cộng dồn gradient vào các hàng được dùng.
10. s = σ(−5) ≈ 0.006693. Cross-entropy: ∂L/∂z = s − 1 ≈ −0.9933. MSE: ∂L/∂z = (s − 1) · s(1 − s) ≈ −0.006604. Cross-entropy cho gradient lớn hơn khoảng 150 lần (đúng bằng 1/(s(1 − s))). MSE + sigmoid bị nhân với σ′ nhỏ đúng lúc mô hình sai nhất; cross-entropy triệt tiêu hệ số đó.
11. ∂ℓ/∂zᵢ = eᶻⁱ/Z = pᵢ với Z = ∑ eᶻʲ, nên ∇ℓ = p. Đạo hàm tiếp: ∂pᵢ/∂zⱼ = (δᵢⱼeᶻⁱZ − eᶻⁱeᶻʲ)/Z² = pᵢδᵢⱼ − pᵢpⱼ, tức ∇²ℓ = diag(p) − ppᵀ, đối xứng. PSD: với mọi v, vᵀJv = ∑ pᵢvᵢ² − (∑ pᵢvᵢ)² = Varₚ(v) ≥ 0. Bằng 0 ⇔ v hằng trên mọi i có pᵢ > 0; khi mọi pᵢ > 0 không gian null đúng là span{1}, nên hạng n − 1. Với L = ℓ(z) − yᵀz (dùng ∑ yᵢ = 1: −∑ yᵢ(zᵢ − ℓ) = ℓ − yᵀz), số hạng yᵀz tuyến tính nên ∇²L = ∇²ℓ: không phụ thuộc nhãn và PSD, vậy L lồi theo z. Hệ quả: ∇L = p − y cho mọi nhãn mềm (label smoothing, distillation). L không lồi chặt: L(z + c1) = L(z) với mọi c, nên tập logit tối ưu (nếu có) là cả một đường thẳng theo hướng 1; đó là lý do softmax regression cần weight decay hoặc cố định một logit để nghiệm duy nhất. Với nhãn one-hot còn một điều nữa: L > 0 với mọi logit hữu hạn và chỉ tiến về 0 khi các logit tách nhau ra vô hạn, nên cận dưới 0 không đạt; weight decay vì thế giúp nghiệm *tồn tại*, không chỉ duy nhất.
12. Mỗi lớp forward: B · d² = 32 · 1024² ≈ 3.36 × 10⁷ phép nhân; 24 lớp: 805 306 368 ≈ 8.1 × 10⁸. Backward mỗi lớp cần dX = G Wᵀ và dW = Xᵀ G, mỗi cái B · d², nên gấp đôi forward: ≈ 1.6 × 10⁹ (chính xác hơn: lớp đầu không cần dX nên là 47 phép nhân ma trận, ≈ 1.58 × 10⁹; tổng một bước huấn luyện ≈ 3 lần forward). Với ReLU còn phải lưu thêm mặt nạ. Activation cần lưu: đầu vào của mỗi lớp, 24 · 32 · 1024 = 786 432 số ≈ 3 MB ở float32 (ở bài này nhỏ hơn hẳn 24 · 1024² ≈ 2.5 × 10⁷ tham số). Checkpoint √24 ≈ 5 đoạn: chỉ lưu khoảng 5 điểm mốc cộng activation của một đoạn đang xử lý (cỡ 5 + 5 lớp thay vì 24), đổi lại chạy forward thêm khoảng một lần. Với Transformer, activation tỉ lệ với batch × độ dài chuỗi × số lớp, và attention còn có ma trận cỡ (độ dài chuỗi)² cho mỗi head, cộng các tensor trung gian của softmax, LayerNorm, GELU, dropout — nên khi chuỗi dài và batch lớn, activation vượt xa tham số.
13. ∇f = (2x₁x₂, x₁² + 3). Đạo hàm theo hướng v = (0, 1) là (∂/∂x₂)∇f = (2x₁, 0) = (2, 0) tại (1, 2) — đúng cột thứ hai của H. Forward-over-reverse: chương trình tính ∇f (một lượt reverse) có chi phí cỡ hằng số lần f; một JVP qua chương trình đó cũng chỉ thêm hằng số lần. Dựng H cần n HVP (một cho mỗi cột). Power iteration (chương 1, lab 1) chỉ cần nhân H với vector: lặp v ← Hv/‖Hv‖₂ với HVP để ước lượng λₘₐₓ(H) — dùng để đo độ sắc của cực tiểu hoặc chọn learning rate (η < 2/λₘₐₓ). Lanczos cũng chỉ cần HVP.
14. Nguyên nhân "không có bug": (a) h vắt qua điểm gãy của ReLU — kiểm bằng cách đếm số activation có |A| < h, hoặc đổi h; (b) kiểm ở float32 — chuyển sang float64; (c) h quá lớn hoặc quá nhỏ — quét h như lab 2; (d) có nguồn ngẫu nhiên (dropout, mẫu hóa) khiến f(x + h) và f(x − h) dùng hai mặt nạ khác nhau — cố định seed hoặc tắt dropout. Nguyên nhân "có bug": (e) quên chia N, nhầm Wᵀ, sai trục tổng của bias — kiểm từng tham số riêng (lỗi ở b₁ mà W₂ khớp khoanh vùng được chỗ sai); (f) regularization tính sai — kiểm riêng phần dữ liệu và phần regularization. Pass dù có bug: lỗi nằm ở nhánh không được kích hoạt trên dữ liệu thử (ví dụ mọi A > 0 nên lỗi ở mặt nạ ReLU không lộ), hoặc regularization rất lớn lấn át phần dữ liệu nên sai số tương đối trông nhỏ, hoặc mọi chiều bằng nhau và W đối xứng (ví dụ khởi tạo bằng I) khiến W và Wᵀ cho cùng kết quả. Vì vậy kiểm với dữ liệu ngẫu nhiên, shape khác nhau ở mọi chiều, và kiểm riêng từng phần của loss.

**Bài tính nhanh**

1. a = (1, −2, 3).
2. ∇‖x‖₂² = 2x = (2, 4, 4). ∇‖x‖₂ = x/‖x‖₂ = (1, 2, 2)/3.
3. A đối xứng nên 2Ax = 2(3, 4) = (6, 8).
4. J có hàng (1, 1) và (x₂, x₁) = (3, 1). det = 1 − 3 = −2.
5. H có hàng (2, 3), (3, 2); trị riêng 5 và −1. Yên ngựa.
6. σ(0) = 0.5; σ′(0) = 0.5 · 0.5 = 0.25.
7. tanh′(0) = 1 − tanh²(0) = 1; ReLU′(−2) = 0.
8. (1, 3)/(1 + 3) = (0.25, 0.75).
9. −ln 0.25 = ln 4 = 2 ln 2 ≈ 1.386.
10. p − y = (0.25, 0.75) − (0, 1) = (0.25, −0.25).
11. ∂L/∂W = g xᵀ = 3 · (1, 2) = (3, 6), viết thành ma trận 1 × 2.
12. Forward mode, 3 lượt (n = 3 < m = 100).
13. Trung tâm: (1.331 − 0.729)/0.2 = 3.01, sai 0.01 = h². Tiến: (1.331 − 1)/0.1 = 3.31, sai 0.31 ≈ 3h.
14. 0.25¹⁰ = 2⁻²⁰ ≈ 10⁻⁶ (chính xác 9.5 × 10⁻⁷).
15. Tổng theo trục batch: (1 + 3 + 5, 2 + 4 + 6) = (9, 12).

**Câu hỏi sau lab**

- Lab 2: h tối ưu cỡ ∛ε, mà ε của float32 lớn hơn float64 khoảng 5 × 10⁸ lần, nên h tối ưu lớn hơn khoảng 800 lần (∛ε ≈ 5 × 10⁻³ so với 6 × 10⁻⁶). Lưới h của lab bước nhảy 10 lần nên đáy rơi vào 10⁻²; xung quanh đáy, sai số float32 cỡ 10⁻⁵, gần với (∛ε)² ≈ 2.4 × 10⁻⁵ mà phân tích h² + ε/h dự đoán: thay h = ∛ε thì cả hai số hạng đều cỡ (∛ε)².
- Lab 4: ℝ⁶ → ℝ¹⁰⁰⁰ có n = 6 < m = 1000, nên forward mode (6 lượt). Loss vô hướng có m = 1, nên reverse mode (1 lượt).
- Lab 5b: loss gồm cross-entropy và ½λ(‖W₁‖² + ‖W₂‖²). Với λ = 10⁻³, phần regularization không bằng 0, và để phân loại đúng đường xoắn ốc mạng cần trọng số đủ lớn; ngoài ra cross-entropy chỉ về 0 khi xác suất lớp đúng tiến tới 1, điều mà regularization ngăn lại. Đo thật: loss cuối 0.2439 = cross-entropy dữ liệu 0.1119 + regularization 0.1320 (tổng ‖W‖² = 264, nên ½ · 10⁻³ · 264 = 0.132). Tức hơn một nửa loss là regularization, và loss 0.244 là điểm cân bằng giữa hai phần, không phải dấu hiệu học kém. Cách tách này (tính lại với reg = 0) cũng là cách kiểm loss dữ liệu riêng.

**Mini-test (thang 30 điểm)**

1. ∇f = (A + Aᵀ)x + b. A + Aᵀ có hàng (2, 2), (2, 2), nên ∇f(1, 1) = (4, 4) + (1, −1) = (5, 3) (2 điểm). Hessian = A + Aᵀ, trị riêng 0 và 4 (1 điểm). PSD nên f lồi, nhưng không lồi chặt: dọc hướng (1, −1) độ cong bằng 0 (1 điểm). Lỗi hay gặp: dùng 2Ax, ra (6, 2) + b — mất 2 điểm.
2. Forward: u = xy = 2, q = u + x = 4, f = qz = 12 (1 điểm). Backward: ∂f/∂q = z = 3, ∂f/∂z = q = 4; cổng cộng chuyển 3 cho u và cho x; cổng nhân u = xy cho ∂f/∂y = 3x = 6 và đóng góp 3y = 3 cho x (2 điểm). x là fan-out: ∂f/∂x = 3 + 3 = 6. Kết quả (6, 6, 4) (1 điểm).
3. Y = X, nên G = ∂L/∂Y = Y = X (1 điểm). ∂L/∂W = XᵀG = XᵀX, có hàng (1·1 + 2·2, 0·1 + 1·2) = (5, 2) và (2, 1) (3 điểm). ∂L/∂b = tổng các hàng của G = (1 + 2, 0 + 1) = (3, 1) (1 điểm).
4. eᶻ = (1, 2, 5), tổng 8, p = (0.125, 0.25, 0.625) (1.5 điểm). L = −ln 0.25 = ln 4 ≈ 1.386 (1 điểm). ∂L/∂z = (0.125, −0.75, 0.625) (1.5 điểm). Batch 4 bản sao, loss trung bình: mỗi mẫu nhận (p − y)/4 = (0.03125, −0.1875, 0.15625) (1 điểm).
5. (a) `db = dY.sum(axis=0)` (tổng theo trục batch) và `dX = dY @ W.T` (2 điểm); dòng `dW` đúng. (b) `db = dY.sum(axis=1)` chạy được với mọi N, H (ra vector độ dài N); lỗi shape chỉ lộ ở bước cập nhật `b -= lr * db`, và chỉ lộ khi N ≠ H *và* N ≠ 1: nếu N = H thì shape khớp, còn nếu N = 1 thì db có shape (1,) và broadcast im lặng sang (H,), b bị trừ đều cùng một số. N = 1 (debug trên một mẫu) lại là cách làm rất phổ biến. `dX = dY @ W` chạy khi D = H (dY là N × H, W là D × H, nên `dY @ W` cần H = D và ra N × H = N × D). Vậy cả hai lỗi cùng lọt khi (N = H hoặc N = 1) và D = H (2 điểm: 1 điểm cho điều kiện của db gồm cả N = 1, 1 điểm cho điều kiện của dX). (c) Gradient check với N, D, H khác nhau đôi một và lớn hơn 1 — khi đó lỗi shape lộ ngay ở bước cập nhật hoặc so sánh gradient — và so sai số tương đối từng tham số ở float64 (2 điểm).
6. Thang chấm: (2 điểm) backward của mỗi lớp cần giá trị forward của lớp đó (đầu vào X cho XᵀG, mặt nạ ReLU…), nên khi train phải lưu mọi activation tới lúc backward dùng; inference không có backward nên dùng xong là bỏ. (2 điểm) Bộ nhớ activation tỉ lệ với số lớp × batch (× độ dài chuỗi), thường vượt bộ nhớ tham số; cộng thêm gradient và trạng thái optimizer. (2 điểm) Checkpointing chỉ lưu activation ở một số điểm mốc và chạy lại forward của từng đoạn khi backward cần; với √L đoạn, bộ nhớ activation giảm từ cỡ L lớp xuống cỡ √L, đổi lại khoảng thêm một lượt forward (chậm hơn cỡ 20–30% tùy mô hình — con số kinh nghiệm, cần đo).

**Săn bug — ba lỗi**

1. **Quên chia N.** Loss là trung bình nên dS = (P − Y)/N. Thiếu hệ số này, mọi gradient lớn gấp N = 4 lần, tức learning rate 0.5 thành bước thực tế 2.0. Lỗi này một mình không gây nổ: nếu chỉ giữ lỗi 1 (sửa hai lỗi kia), loss vẫn giảm từ 1.35 xuống 0.076 sau 50 bước ở lr 0.5, và còn về 0.009 sau 300 bước ở lr 0.1, tốt hơn cả bản đúng (0.146). Không cặp lỗi nào tự nổ. Hiện tượng nổ (308 ở bước 50) chỉ xuất hiện khi cả ba lỗi cùng tác động: bước lớn gấp 4 lần lại đi theo hướng sai do lỗi 2 và 3. Bài học: đừng kết luận có bug hay không từ đường loss. Ở lr 0.1 bản lỗi vẫn giảm (xuống 0.25 sau 300 bước, trong khi bản đúng xuống 0.146) nên trông bình thường; chỉ gradient check mới lộ ra. Ở lr 0.5 quỹ đạo của bản lỗi là: 308 (bước 50), 2.2 × 10⁵ (bước 100), 2.5 × 10⁷ (bước 122), rồi về 0.0000 từ khoảng bước 150 đến 300, vì trọng số lớn đến mức softmax bão hòa và bốn mẫu đều được phân loại "chắc chắn". Loss thấp ở bước 300 thấp hơn cả bản đúng (0.0065), nhưng đó là kết quả của một quỹ đạo nổ, không phải của gradient đúng: loss thấp không chứng minh gradient đúng.
2. **`dS @ W2` thay vì `dS @ W2.T`.** S = H W₂ nên ∂L/∂H = dS W₂ᵀ. Vì W₂ vuông (H = C = 4) nên không có lỗi shape.
3. **`dA.sum(axis=1)` thay vì `axis=0`.** b₁ được cộng vào mọi hàng (mẫu), nên gradient là tổng theo trục batch. Vì N = H = 4 nên shape vẫn khớp.

```python
def grads_fixed(W1, b1, W2, b2, X, y):
    N = X.shape[0]
    A = X @ W1 + b1
    Hh = np.maximum(0, A)
    S = Hh @ W2 + b2
    P = np.exp(S - S.max(axis=1, keepdims=True)); P /= P.sum(axis=1, keepdims=True)
    dS = P.copy(); dS[np.arange(N), y] -= 1
    dS /= N                                      # sửa lỗi 1
    dW2 = Hh.T @ dS
    db2 = dS.sum(axis=0)
    dH = dS @ W2.T                               # sửa lỗi 2
    dA = dH * (A > 0)
    dW1 = X.T @ dA
    db1 = dA.sum(axis=0)                         # sửa lỗi 3
    return dW1, db1, dW2, db2
```

Đã chạy kiểm: bản sửa cho sai số tương đối 8.8 × 10⁻¹¹, 4.8 × 10⁻¹¹, 1.6 × 10⁻¹⁰, 3.1 × 10⁻¹⁰ trên W₁, b₁, W₂, b₂. Bài học: thử nghiệm với N = H = C là cách tệ nhất để kiểm code; hãy chọn mọi chiều khác nhau.

## Tài liệu đào sâu và nguồn

Nội dung chương được viết lại, đối chiếu phạm vi kiến thức với các nguồn dưới đây. Đọc theo thứ tự ưu tiên:

| Tài liệu | Dùng để làm gì | Phần nên đọc |
| --- | --- | --- |
| [CS231n — Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/) (Stanford) | Đồ thị tính toán, mẫu cổng add/mul/max, `+=` ở fan-out, gradient của phép nhân ma trận bằng phân tích chiều | Toàn bộ; làm lại các ví dụ số trên giấy |
| [CS231n — Neural Networks Part 3](https://cs231n.github.io/neural-networks-3/) | Gradient check: sai phân trung tâm, sai số tương đối, float64, kink, các mẹo thực hành | Mục "Gradient checks" |
| [CS231n — Neural network case study](https://cs231n.github.io/neural-networks-case-study/) | Softmax-CE theo batch, backprop qua ReLU, dữ liệu xoắn ốc của lab 5b | Toàn bộ; lưu ý trang này không trừ max trong softmax |
| [CS224n — Computing Neural Network Gradients](https://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf) (Kevin Clark) | Quy ước Jacobian và shape, các đồng nhất thức, ví dụ mạng một lớp | Toàn bộ, khoảng 7 trang |
| [Vector, Matrix, and Tensor Derivatives](https://cs231n.stanford.edu/vecDerivs.pdf) (Erik Learned-Miller) | Mẹo viết từng phần tử, tránh tensor nhiều chiều | Toàn bộ, ngắn |
| [CS229 Linear Algebra Review](https://cs229.stanford.edu/section/cs229-linalg.pdf) (Kolter, cập nhật bởi Do) | Gradient, Hessian, dạng toàn phương, least squares, gradient của log det | Mục 4 (Matrix Calculus) |
| [Mathematics for Machine Learning](https://mml-book.github.io/) (Deisenroth, Faisal, Ong) | Khung lý thuyết đầy đủ của cả chương, có bài tập | Chương 5: 5.2–5.8. Lưu ý: MML coi gradient là vector hàng (1 × n), ngược với quy ước đầu chương |
| [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/) (Parr, Howard) | Numerator layout, Jacobian của phép toán phần tử là đường chéo, các dạng quy tắc chuỗi | Toàn bộ; đọc nhanh nếu đã nắm 2.1–2.3 |
| [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) (Petersen, Pedersen) | Tra cứu đạo hàm (dạng toàn phương, vết, det, ma trận đối xứng) | Mục 2; nhớ kiểm layout |
| [Identifying and attacking the saddle point problem in high-dimensional non-convex optimization](https://arxiv.org/abs/1406.2572) (Dauphin và cộng sự, NIPS 2014) [link chưa mở lại được, cần kiểm chứng] | Vì sao yên ngựa, không phải cực tiểu địa phương, là chướng ngại chính trong không gian chiều cao | Phần mở đầu |
| [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) (Chen, Xu, Zhang, Guestrin, 2016) [link chưa mở lại được, cần kiểm chứng] | Paper gốc của gradient checkpointing √L | Phần mở đầu và thuật toán |
| [JAX — The Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) | JVP/VJP, `jacfwd`/`jacrev`, HVP forward-over-reverse, bộ nhớ của reverse mode | Toàn bộ phần 1 |
| [PyTorch — Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) | Autograd thật làm gì: DAG, saved tensors, in-place, điểm không khả vi, `no_grad`/`inference_mode` | Toàn bộ |
| [d2l.ai — Automatic Differentiation](https://d2l.ai/chapter_preliminaries/autograd.html) | `backward` trên output không vô hướng, tích lũy gradient, `detach`, control flow | Mục 2.5 |
| [Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/) (Christopher Olah) | Trực giác "tổng theo đường đi", vì sao reverse mode thắng khi một đầu ra | Toàn bộ, đọc đầu tiên nếu thích hình |
| [Automatic Differentiation in Machine Learning: a Survey](https://jmlr.org/papers/v18/17-468.html) (Baydin, Pearlmutter, Radul, Siskind; JMLR 2018) | Paper tổng quan: AD khác sai phân và khác đạo hàm ký hiệu thế nào | Mục 1–3 |
| [Learning representations by back-propagating errors](https://ui.adsabs.harvard.edu/abs/1986Natur.323..533R/abstract) (Rumelhart, Hinton, Williams; Nature 323, 1986) | Paper gốc phổ biến backprop cho mạng neural | Đọc tóm tắt để biết bối cảnh lịch sử |
| [Yes you should understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b) (Andrej Karpathy) | Các bẫy thật: sigmoid bão hòa, dead ReLU, exploding gradient, clip sai cách | Toàn bộ, ngắn |

**Từ khóa để tự tìm thêm:** matrix differential calculus, numerator vs denominator layout, vector-Jacobian product, Hessian-vector product, gradient checkpointing, dual numbers, log-sum-exp trick, micrograd (Karpathy).

**Trước khi sang chương 3:** đạt từ 24/30 điểm mini-test, làm lại các bài ★★★ sai, và tự viết được lab 5a (MLP + softmax-CE qua gradient check) mà không nhìn code mẫu.

Nguồn tra cứu ngày 24/10/2026.
