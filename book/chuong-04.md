# Chương 4 — Tối ưu hóa

## Mục tiêu và kiểm tra nhanh

Huấn luyện một mô hình là giải một bài toán tối ưu. Chương này trả lời ba câu hỏi: khi nào bài toán "dễ" (lồi, điều kiện tối ưu), một thuật toán bậc một đi nhanh hay chậm và vì sao (số điều kiện κ), và các optimizer thật (SGD, momentum, Adam, AdamW, lịch learning rate) sửa được gì, không sửa được gì. Cả chương dùng chung một bài toán f(x) = ½(x₁² + 100x₂²), κ = 100, điểm đầu (1, 1), để bạn thấy mỗi thuật toán trả giá khác nhau cho cùng một κ. Thời lượng: khoảng 2 tuần, 8 giờ mỗi tuần.

Chương 1 (mục 1.4) đã dạy Hessian, PSD và κ = λₘₐₓ/λₘᵢₙ, và đã cho xem cảnh GD đi zig-zag. Ở đây ta không nhắc lại định nghĩa mà chứng minh, đếm bước và so sánh.

**Ghi chú ký hiệu.** Vài ký hiệu mang hai nghĩa tùy mục, đọc theo ngữ cảnh. μ không chỉ số: ở mục 4.1 là nhân tử KKT, từ mục 4.2 là hằng số strongly convex (nhân tử KKT có chỉ số, μᵢ). λᵢ là trị riêng của Hessian; λ không chỉ số ở mục 4.4 là hệ số weight decay. H là Hessian (ở phần BFGS ta dùng B cho xấp xỉ Hessian và H cho xấp xỉ nghịch đảo của nó, có nói rõ tại chỗ). ε là sai số mục tiêu ở mục 4.2 và là hằng số chống chia cho 0 của Adam ở mục 4.4.

**Sau chương này, bạn phải làm được:**

- ☐ Kiểm tra một hàm có lồi không, phân loại điểm dừng bằng Hessian, viết và đọc điều kiện KKT cho một bài có ràng buộc bất đẳng thức.
- ☐ Chứng minh GD trên bài bậc hai nhân sai số theo từng hướng riêng với (1 − ηλᵢ), suy ra η ổn định, η tối ưu và số bước theo κ.
- ☐ Giải thích vì sao SGD với η hằng dừng ở một "sàn nhiễu" và bán kính sàn phụ thuộc η/B thế nào.
- ☐ Viết đúng cập nhật momentum, Adam, AdamW; giải thích bias correction, vì sao L2 ≠ weight decay với Adam, và vì sao Transformer dùng warmup + cosine.
- ☐ So sánh Newton, L-BFGS với GD; giải thích vì sao deep learning gần như chỉ dùng phương pháp bậc một, và vai trò của điểm yên ngựa và độ nhọn.

**Kiểm tra nhanh (5 phút, không dùng máy tính).** Câu i kiểm tra mục 4.i: đúng thì đọc lướt mục đó, sai thì đọc kỹ. Đúng cả 5 câu thì làm thẳng bài ★★★ và mini-test.

1. f lồi, khả vi, và ∇f(x₀) = 0. x₀ có chắc là cực tiểu toàn cục không? Nếu f không lồi thì sao?
2. Hessian của một hàm bậc hai có trị riêng 2 và 50. GD với η cố định: η lớn nhất còn ổn định, η tối ưu và tốc độ hội tụ tương ứng là bao nhiêu?
3. Tăng batch từ 16 lên 64, độ lệch chuẩn của nhiễu gradient minibatch thay đổi thế nào?
4. Bỏ bias correction trong Adam (β₁ = 0.9, β₂ = 0.999). Bước đầu tiên to hơn hay nhỏ hơn α, khoảng bao nhiêu lần?
5. Vì sao Newton giải xong một bài bậc hai lồi chặt trong đúng 1 bước, và vì sao nó có thể bị hút về điểm yên ngựa?

Đáp án ở mục Lời giải cuối chương.

## 4.1 Hàm lồi, điều kiện tối ưu, Lagrange và KKT

Với hàm lồi, mọi cực tiểu địa phương là cực tiểu toàn cục và ∇f = 0 là đủ để kết luận. Với hàm không lồi, ∇f = 0 chỉ là điều kiện cần, và Hessian chỉ cho thông tin địa phương.

**Tập lồi và hàm lồi.** Tập C lồi nếu đoạn thẳng nối hai điểm bất kỳ của C nằm trọn trong C. Hàm f lồi nếu dây cung nối hai điểm trên đồ thị nằm trên đồ thị. Với f khả vi, ba định nghĩa sau tương đương:

$$
f(\theta x + (1-\theta) y) \le \theta f(x) + (1-\theta) f(y) \;\;\forall \theta \in [0,1]
\quad\Longleftrightarrow\quad
f(y) \ge f(x) + \nabla f(x)^{\top}(y - x)
\quad\Longleftrightarrow\quad
\nabla^2 f(x) \succeq 0
$$

Định nghĩa thứ hai (tiếp tuyến nằm dưới đồ thị) là định nghĩa hữu dụng nhất: một thông tin địa phương, gradient tại x, cho ta một cận dưới toàn cục của f. Định nghĩa thứ ba (Hessian PSD tại mọi điểm, xem mục 1.4) là cách kiểm tra nhanh nhất khi f khả vi hai lần.

% manim: DayCungHamLoi

**Phép toán giữ tính lồi.** Tổng có trọng số không âm của các hàm lồi; hợp với một hàm affine (f(Ax + b) lồi nếu f lồi); max theo điểm của các hàm lồi. Nhờ ba quy tắc này: MSE của hồi quy tuyến tính ‖Xw − y‖² là chuẩn bình phương hợp với hàm affine, nên lồi; hinge loss max(0, 1 − y·wᵀx) là max của hai hàm affine, nên lồi.

**Hệ quả quan trọng nhất.** Nếu f lồi và ∇f(x*) = 0, thay vào định nghĩa tiếp tuyến: f(y) ≥ f(x*) + 0 với mọi y. Vậy x* là cực tiểu toàn cục. Không cần kiểm Hessian, không sợ cực tiểu địa phương xấu.

**Điều kiện tối ưu cho hàm tổng quát** (khả vi hai lần):

- Bậc 1, cần: x* là cực tiểu địa phương ⇒ ∇f(x*) = 0. Điểm có ∇f = 0 gọi là điểm dừng (stationary point).
- Bậc 2, cần: x* là cực tiểu địa phương ⇒ ∇²f(x*) PSD.
- Bậc 2, đủ: ∇f(x*) = 0 và ∇²f(x*) PD ⇒ x* là cực tiểu địa phương chặt.
- Điểm dừng mà Hessian có cả trị riêng dương lẫn âm là **điểm yên ngựa** (saddle point): đi theo hướng riêng dương thì f tăng, theo hướng riêng âm thì f giảm.

Khoảng trống giữa "PSD là cần" và "PD là đủ" là có thật: f(x) = x³ tại 0 có f′ = 0, f″ = 0 (PSD) nhưng không phải cực tiểu.

**Ví dụ giải mẫu 4.1 — phân loại điểm dừng.** f(x, y) = x³ − 3x + y².

- ∇f = (3x² − 3, 2y) = 0 ⇒ hai điểm dừng (1, 0) và (−1, 0).
- Hessian = diag(6x, 2). Tại (1, 0): trị riêng 6 và 2, đều dương ⇒ cực tiểu địa phương, f = −2.
- Tại (−1, 0): trị riêng −6 và 2, trái dấu ⇒ điểm yên ngựa, f = 2.
- f không có cực tiểu toàn cục: cho y = 0, x → −∞ thì f → −∞. Hessian tại (1, 0) PD nhưng điều đó chỉ nói về lân cận của (1, 0). Hàm không lồi vì 6x < 0 khi x < 0.

**Hàm mất mát nào lồi trong ML.** ½·MSE tuyến tính (Hessian XᵀX/n; MSE đầy đủ có 2XᵀX/n; cả hai PSD). Logistic regression: Hessian là Xᵀ diag(s(1 − s)) X / n với s = sigmoid(Xw), mỗi số s(1 − s) dương, nên PSD. Softmax cross-entropy lồi theo logit. Ngược lại, mạng có lớp ẩn không lồi theo trọng số: hoán vị hai neuron ẩn (đổi hàng của W₁ và cột của W₂ tương ứng) cho cùng hàm số, nên nếu w* là cực tiểu thì bản hoán vị w′ ≠ w* cũng là cực tiểu. Nếu loss lồi, mọi điểm trên đoạn nối w* và w′ cũng phải là cực tiểu; điều này nói chung không đúng (trung điểm thường cho hai neuron giống nhau, loss cao hơn).

**Lagrange cho ràng buộc đẳng thức.** Cực tiểu f(x) với h(x) = 0. Tại nghiệm, đường mức của f tiếp xúc với mặt ràng buộc, nên ∇f song song ∇h:

$$
\nabla f(x^*) + \nu \nabla h(x^*) = 0, \qquad h(x^*) = 0
$$

Nếu ∇f có thành phần dọc theo mặt ràng buộc, ta còn trượt trên mặt đó để giảm f được, nên x* chưa tối ưu.

**KKT cho ràng buộc bất đẳng thức.** Cực tiểu f(x) với gᵢ(x) ≤ 0. Điều kiện Karush–Kuhn–Tucker gồm bốn phần:

$$
\underbrace{\nabla f(x^*) + \sum_i \mu_i \nabla g_i(x^*) = 0}_{\text{dừng}}, \qquad
\underbrace{g_i(x^*) \le 0}_{\text{khả thi gốc}}, \qquad
\underbrace{\mu_i \ge 0}_{\text{khả thi đối ngẫu}}, \qquad
\underbrace{\mu_i\, g_i(x^*) = 0}_{\text{bù}}
$$

Đọc điều kiện bù: nếu ràng buộc i không chạm (gᵢ < 0) thì μᵢ = 0, ràng buộc như không có. Nếu chạm (gᵢ = 0) thì μᵢ ≥ 0 và −∇f phải chỉ ra ngoài miền khả thi, theo hướng ∇gᵢ: f muốn đi ra ngoài nhưng bị chặn. μᵢ là "giá" của ràng buộc: nới gᵢ ≤ 0 thành gᵢ ≤ δ thì giá trị tối ưu giảm khoảng μᵢδ. Với bài lồi (f, gᵢ lồi) thỏa điều kiện Slater (có một điểm thỏa mọi gᵢ < 0 chặt), KKT vừa cần vừa đủ. Bài toán đối ngẫu của SVM (chương 5) là ứng dụng trực tiếp; chương này không đi sâu vào đối ngẫu.

**Ví dụ giải mẫu 4.2 — KKT với một ràng buộc.** Cực tiểu (x − 2)² + (y − 1)² với x + y ≤ 2, tức g = x + y − 2 ≤ 0.

- Cực tiểu tự do (2, 1) có 2 + 1 = 3 > 2, vi phạm. Vậy ràng buộc phải chạm: x + y = 2.
- Dừng: 2(x − 2) + μ = 0 và 2(y − 1) + μ = 0 ⇒ x − 2 = y − 1. Cùng với x + y = 2: x = 1.5, y = 0.5, μ = 1 > 0. KKT thỏa. SciPy (SLSQP) cho đúng (1.5, 0.5).
- Kiểm ý nghĩa của μ: với vế phải 2 + δ, nghiệm là (1.5 + δ/2, 0.5 + δ/2) và f* = (1 − δ)²/2. Đạo hàm theo δ tại 0 là −1 = −μ.
- Đổi ràng buộc thành x + y ≤ 4: (2, 1) khả thi, g = −1 < 0, nên μ = 0 theo điều kiện bù.

% manim: KKTTiepXuc

**Ví dụ giải mẫu 4.3 — gradient clipping là một phép chiếu thỏa KKT.** Chiếu a = (3, 4) lên quả cầu ‖w‖₂ ≤ 1: cực tiểu ½‖w − a‖² với g = ½(‖w‖² − 1) ≤ 0.

- Dừng: (w − a) + μw = 0 ⇒ w = a/(1 + μ). Vì ‖a‖ = 5 > 1, ràng buộc chạm: ‖w‖ = 1 ⇒ 1 + μ = 5, μ = 4.
- Nghiệm w = a/‖a‖ = (0.6, 0.8). Đây đúng là việc `clip_grad_norm_` làm với max_norm bằng 1: gradient có chuẩn vượt ngưỡng được co về đúng ngưỡng, giữ nguyên hướng. Chính xác hơn: `clip_grad_norm_` tính một chuẩn gộp trên mọi tham số của mô hình và nhân với max_norm/(chuẩn gộp + 10⁻⁶), nên chỉ gần đúng phép chiếu lên quả cầu bán kính max_norm trong không gian ghép mọi tham số.

**Vì sao ML cần cái này.**

- **Biết khi nào được yên tâm.** Hồi quy tuyến tính, logistic, SVM là bài lồi: optimizer nào hội tụ cũng ra cùng một cực tiểu (giá trị loss). Với mạng nơ-ron, khởi tạo và optimizer khác nhau cho nghiệm khác nhau.
- **Weight decay và ràng buộc chuẩn.** Phạt (λ/2)‖w‖² chính là số hạng Lagrange của ràng buộc g = ½(‖w‖² − r²) ≤ 0: mỗi r ứng với một μ = λ (viết g = ‖w‖² − r² thì μ = λ/2). Hai cách viết cho cùng họ nghiệm (với bài lồi).
- **Clipping và phép chiếu.** Gradient clipping, max-norm constraint, projected gradient descent đều là "đi một bước rồi chiếu lại vào tập khả thi".
- **Softmax từ Lagrange.** Bài tập 14: cực đại entropy với ràng buộc kỳ vọng cho nghiệm dạng pᵢ ∝ exp(λ·i), chính là dạng của softmax.

## 4.2 Gradient descent: L-smooth, strongly convex, số điều kiện

Trên bài bậc hai, GD tách thành các bài một chiều độc lập theo từng hướng riêng của Hessian; mỗi bước nhân sai số theo hướng i với (1 − ηλᵢ). Mọi kết luận về η và tốc độ suy ra từ một dòng đó.

**Vì sao bước GD có dạng x − η∇f.** Xấp xỉ f quanh x bằng bậc một, cộng một số hạng phạt đi xa:

$$
x_{k+1} = \arg\min_{y} \Big[\, f(x_k) + \nabla f(x_k)^{\top}(y - x_k) + \frac{1}{2\eta}\lVert y - x_k \rVert_2^2 \,\Big] = x_k - \eta \nabla f(x_k)
$$

η nhỏ là "không tin xấp xỉ bậc một ở xa". Bước Newton (mục 4.5) thay số hạng phạt bằng chính Hessian.

**Hai giả thiết chuẩn.** f là L-smooth nếu gradient Lipschitz với hằng số L; với f khả vi hai lần, điều này tương đương ‖∇²f(x)‖₂ ≤ L ở mọi x, tức −L ≤ λᵢ(∇²f(x)) ≤ L (cả hai phía, vì gradient Lipschitz chặn độ cong âm lẫn dương; nếu f lồi thì mọi λᵢ ≥ 0 nên chỉ cần λₘₐₓ ≤ L). Cận trên bậc hai của bổ đề giảm đủ bên dưới chỉ dùng nửa trên ∇²f ⪯ L·I. f là μ-strongly convex nếu λₘᵢₙ(∇²f) ≥ μ > 0 ở mọi nơi. Số điều kiện κ = L/μ. Với bài bậc hai ½xᵀAx, L = λₘₐₓ(A), μ = λₘᵢₙ(A), và κ trùng với κ của chương 1. Khác biệt: L và μ là cận cho *mọi* điểm. L định nghĩa được cho cả hàm không bậc hai (logistic, cross-entropy); μ > 0 thì chỉ có khi có thêm số hạng (λ/2)‖w‖², hoặc khi chỉ xét trên một tập bị chặn: Hessian của logistic và cross-entropy không regularization tiến về 0 khi |logit| lớn.

**Bổ đề giảm đủ (descent lemma).** L-smooth cho một cận trên bậc hai: f(y) ≤ f(x) + ∇f(x)ᵀ(y − x) + (L/2)‖y − x‖². Thay y = x − η∇f(x):

$$
f(x - \eta \nabla f(x)) \le f(x) - \eta\Big(1 - \frac{L\eta}{2}\Big)\lVert \nabla f(x) \rVert_2^2
\qquad\overset{\eta = 1/L}{\Longrightarrow}\qquad
f(x_{k+1}) \le f(x_k) - \frac{1}{2L}\lVert \nabla f(x_k) \rVert_2^2
$$

Vậy với mọi η ∈ (0, 2/L), f giảm sau mỗi bước, không cần lồi. η = 1/L tối đa hóa cận giảm. Đây là lý do câu "η ≤ 1/L luôn an toàn".

**Phân tích chính xác trên bài bậc hai.** f(x) = ½xᵀAx, A = QΛQᵀ, nghiệm x* = 0. Đặt x̂ = Qᵀx (tọa độ theo hệ trục riêng). Vì ∇f = Ax:

$$
x_{k+1} = (I - \eta A)\, x_k
\quad\Longrightarrow\quad
\hat{x}_{k,i} = (1 - \eta\lambda_i)^{k}\, \hat{x}_{0,i}
$$

Mỗi hướng là một dãy cấp số nhân với công bội 1 − ηλᵢ. Từ đây:

- **Ổn định** ⇔ |1 − ηλᵢ| < 1 với mọi i ⇔ 0 < η < 2/L. Đây là chứng minh của câu chỉ được nêu ở mục 1.4.
- **Tốc độ** là ρ(η) = maxᵢ|1 − ηλᵢ| = max(|1 − ημ|, |1 − ηL|), vì hàm |1 − ηλ| theo λ lớn nhất ở hai đầu.
- **η tối ưu** cân bằng hai đầu: 1 − ημ = ηL − 1 ⇒ η* = 2/(L + μ), ρ* = (L − μ)/(L + μ) = (κ − 1)/(κ + 1).
- Với η = 1/L: ρ = 1 − 1/κ. **Số bước** để sai số còn ε lần ban đầu: ln(1/ε)/(−ln ρ) ≈ κ·ln(1/ε) khi κ lớn (vì −ln(1 − 1/κ) ≈ 1/κ).

**Ví dụ giải mẫu 4.4 — đếm bước GD trên bài κ = 100.** f = ½(x₁² + 100x₂²), x₀ = (1, 1), L = 100, μ = 1. Tiêu chí dừng: ‖xₖ‖ < 10⁻⁶·‖x₀‖.

- η = 1/L = 0.01: hướng x₂ nhân 1 − 1 = 0, xong sau 1 bước. Hướng x₁ nhân 0.99. Sau 100 bước x ≈ (0.366, 0): hướng dốc xong ngay, hướng thoải gần như chưa đi. Cần 0.99ᵏ < √2 × 10⁻⁶ (vì ‖x₀‖ = √2 còn x₂ đã về 0), tức k = 1 341. Chạy thật: 1 341 bước.
- η = 2/(L + μ) = 2/101 ≈ 0.0198: cả hai hướng có |1 − ηλ| = 99/101 ≈ 0.980 (hướng x₂ đổi dấu mỗi bước). Lý thuyết ln(10⁻⁶)/ln(0.980) ≈ 691; chạy thật 691.
- η = 0.0201 > 2/L = 0.02: hướng x₂ có |1 − 2.01| = 1.01 > 1, phân kỳ. Sau 100 bước x₂ ≈ 2.70, đổi dấu mỗi bước, lớn dần.
- Bài học: chọn η tối ưu chỉ lợi gấp khoảng 2 so với η = 1/L. Muốn lợi hơn phải đổi thuật toán (momentum, mục 4.4) hoặc đổi bài toán (tiền xử lý, ví dụ 4.5).

% manim: TocDoTungHuongRieng

**Ba chế độ hội tụ của GD với η = 1/L.**

| Giả thiết | Đảm bảo | Kiểu hội tụ |
| --- | --- | --- |
| L-smooth, μ-strongly convex | ‖xₖ − x*‖² ≤ (1 − μ/L)ᵏ‖x₀ − x*‖² | Tuyến tính (hình học), ≈ κ·ln(1/ε) bước |
| L-smooth, lồi | f(xₖ) − f* ≤ L‖x₀ − x*‖²/(2k) | Dưới tuyến tính, O(1/k) |
| L-smooth, không lồi, f* = inf f > −∞ | min theo j ≤ k của ‖∇f(xⱼ)‖² ≤ 2L(f(x₀) − f*)/k | Chỉ về điểm dừng, ‖∇f‖ = O(1/√k) |

Dòng cuối chứng minh trong một dòng: cộng bổ đề giảm đủ từ bước 0 tới k − 1, vế trái co lại thành f(x₀) − f(xₖ) ≤ f(x₀) − f*, nên tổng k số ‖∇f(xⱼ)‖² ≤ 2L(f(x₀) − f*), và số nhỏ nhất không vượt trung bình. Chú ý: đảm bảo này chỉ nói ‖∇f‖ nhỏ, không nói ta ở gần cực tiểu (có thể là yên ngựa). Để thấy khác biệt giữa tuyến tính và O(1/k): với L = 100, ‖x₀ − x*‖ = 1, cận O(1/k) cho 5.0, 0.5, 0.05 ở k = 10, 100, 1 000, mỗi lần thêm một chữ số tốn gấp 10 lần số bước; tuyến tính thì mỗi chữ số tốn cùng một số bước.

**Ví dụ giải mẫu 4.5 — chuẩn hóa đặc trưng thay đổi κ.** Hồi quy tuyến tính có cột hằng số 1 và một đặc trưng t = (0, 1, 2, 10). Hàm mất mát ½·MSE = ‖Xw − y‖²/(2n) có Hessian XᵀX/n (bỏ hệ số ½ thì Hessian nhân đôi, κ không đổi).

- XᵀX/n có hàng (1, 3.25) và (3.25, 26.25): trung bình của t là 3.25, trung bình của t² là 105/4 = 26.25.
- Trị riêng ≈ 0.588 và 26.66 ⇒ κ ≈ 45.3; η ổn định tối đa 2/26.66 ≈ 0.075.
- Chuẩn hóa t (trừ trung bình, chia độ lệch chuẩn): cột 1 và cột đã chuẩn hóa vuông góc, mỗi cột có trung bình bình phương bằng 1, nên Hessian = I, κ = 1. GD với η = 1 hội tụ trong 1 bước.
- Hai nguồn gây κ lớn ở đây: thang đo của t (trung bình bình phương 26.25 so với 1) và tương quan giữa t và cột hằng (trung bình 3.25 ≠ 0). Trừ trung bình sửa cái thứ hai, chia độ lệch chuẩn sửa cái thứ nhất.

**Vì sao ML cần cái này.**

- **Learning rate có trần.** Trần 2/L là thật: loss tăng vọt rồi NaN ngay đầu huấn luyện thường là η vượt trần theo hướng cong nhất. Mục 4.5 cho thấy mạng thật "tự điều chỉnh" quanh trần này (Edge of Stability).
- **StandardScaler không phải mẹo.** Nó là cách rẻ nhất để giảm κ của bài tuyến tính. BatchNorm, LayerNorm được biện minh một phần bằng trực giác này; với mạng sâu, đó là trực giác chứ không phải chứng minh.
- **κ là đơn vị đo.** Các mục sau đo mỗi thuật toán bằng κ: GD cần ≈ κ, momentum ≈ √κ (chỉ chứng minh được trên bài bậc hai, mục 4.4), Newton không phụ thuộc κ.

**Thực tế tính toán.** Line search (thử η, giảm dần đến khi f giảm đủ theo điều kiện Armijo) cho phép không cần biết L. `scipy.optimize` dùng line search kiểu Wolfe (Armijo cộng điều kiện độ cong) cho BFGS và CG. Deep learning hầu như không dùng, vì mỗi lần thử η cần thêm một lần tính loss trên minibatch, mà loss minibatch nhiễu.

## 4.3 SGD, minibatch, nhiễu gradient, learning rate

SGD thay gradient thật bằng một ước lượng không chệch rẻ hơn nhiều lần. Cái giá: với η hằng, SGD không hội tụ về x* mà dao động trong một vùng nhiễu có phương sai tỉ lệ với η/B (bán kính tỉ lệ với √(η/B)).

**Bài toán tổng hữu hạn.** Loss huấn luyện là trung bình f(w) = (1/n)∑ᵢfᵢ(w) theo n mẫu. Lấy minibatch B mẫu ngẫu nhiên, gradient minibatch g = (1/B)∑ trong batch ∇fᵢ(w) có 𝔼[g] = ∇f(w) (không chệch). Nếu gradient một mẫu có phương sai σ² (theo mỗi tọa độ), thì với lấy mẫu độc lập:

$$
\operatorname{Var}[g] = \frac{\sigma^2}{B}, \qquad \text{độ lệch chuẩn} = \frac{\sigma}{\sqrt{B}}
$$

Chi phí mỗi bước tăng tuyến tính theo B, độ lệch chuẩn chỉ giảm theo √B. Đó là lý do lợi ích của batch lớn giảm dần.

**Ví dụ giải mẫu 4.6 — sàn nhiễu của SGD.** f(w) = ½w², gradient nhiễu g = w + ξ, ξ có trung bình 0, phương sai σ²/B với σ² = 4.

- Cập nhật w ← w − η(w + ξ) = (1 − η)w − ηξ. Ở trạng thái dừng, phương sai V không đổi qua một bước: V = (1 − η)²V + η²σ²/B. Giải ra:

$$
V = \frac{\eta^2 \sigma^2 / B}{1 - (1-\eta)^2} = \frac{\eta\, \sigma^2}{B\,(2 - \eta)}
$$

- η = 0.1, B = 1: V = 0.2105; mô phỏng 200 000 bước cho 0.2104.
- η = 0.1, B = 16: V = 0.01316, nhỏ hơn đúng 16 lần; mô phỏng 0.01315.
- η = 0.01, B = 1: V = 0.02010, nhỏ hơn khoảng 10 lần; mô phỏng 0.02026.
- Bài học: w không bao giờ dừng ở 0. Phương sai vùng nhiễu tỉ lệ với η/B nên bán kính tỉ lệ với √(η/B) (khi η nhỏ). Muốn vào sâu hơn thì giảm η hoặc tăng B.

% manim: SanNhieuSGD

**Ví dụ giải mẫu 4.7 — chi phí của minibatch.** Gradient một mẫu có độ lệch chuẩn 2. B = 64 cho độ lệch chuẩn 2/8 = 0.25, nhỏ hơn 8 lần, nhưng tốn 64 lần tính toán. Muốn giảm nhiễu thêm 2 lần phải tăng B thêm 4 lần (lên 256).

**Muốn hội tụ thật thì phải giảm η.** Điều kiện Robbins–Monro cổ điển cho SGD: tổng các ηₜ bằng vô cùng (đi được đủ xa) và tổng các ηₜ² hữu hạn (nhiễu tích lũy bị chặn), ví dụ ηₜ ∝ 1/t. Thực tế deep learning dùng lịch "giữ rồi giảm" (cosine, linear decay, mục 4.4) chứ không dùng 1/t, nhưng tinh thần giống nhau: η lớn để đi nhanh, η nhỏ cuối cùng để vào sâu.

**η/B như một "nhiệt độ".** Công thức sàn nhiễu cho thấy η và B xuất hiện qua tỉ số η/B. Hai ý sau hay bị gộp làm một nhưng có nguồn khác nhau. (1) **Quy tắc scale tuyến tính** của [Goyal và cộng sự (2017)](https://arxiv.org/abs/1706.02677): tăng batch k lần thì tăng learning rate k lần. Lập luận của họ là: k bước với batch B và learning rate η xấp xỉ một bước với batch kB và learning rate kη, miễn là gradient đổi chậm giữa k bước đó. Họ kiểm chứng thực nghiệm cho SGD + momentum trên ImageNet tới batch 8 192, và cần warmup ở đầu để quy tắc này không làm phân kỳ. (2) Góc nhìn "η/B là nhiệt độ của SGD" (mức nhiễu quyết định SGD đậu ở cực tiểu rộng hay hẹp) là của [Smith và Le (2017)](https://arxiv.org/abs/1710.06451), và của Jastrzębski và cộng sự (2017, arXiv 1711.04623 [cần kiểm chứng ID]). Cả hai cùng cho thấy giữ η/B không đổi giữ mức nhiễu gần như cũ. Quy tắc gãy khi η·λₘₐₓ chạm trần ổn định của mục 4.2: lúc đó không thể tăng η nữa dù B tăng.

**Epoch và xáo trộn.** Thực tế không lấy mẫu độc lập mà xáo trộn dữ liệu mỗi epoch rồi cắt thành các minibatch rời nhau (lấy mẫu không hoàn lại). Không xáo trộn khi dữ liệu được sắp theo nhãn hoặc theo nguồn là lỗi nghiêm trọng: các bước liên tiếp kéo trọng số theo các hướng có hệ thống, không còn là nhiễu trung bình 0. Một lỗi hay gặp khác là lấy cùng một minibatch mãi (bài săn bug).

**Vì sao ML cần cái này.**

- **Đường loss nhiễu là bình thường.** Loss báo theo minibatch dao động quanh giá trị thật; cần làm mượt (trung bình trượt) trước khi kết luận.
- **Giới hạn phần cứng.** Với GPU 6 GB, B thường bị chặn bởi bộ nhớ chứ không bởi lựa chọn. Gradient accumulation (cộng gradient của k minibatch nhỏ rồi mới cập nhật) mô phỏng batch lớn hơn k lần với cùng bộ nhớ, đổi lại chậm hơn k lần mỗi bước.
- **Nhiễu không chỉ có hại.** Nhiễu SGD giúp thoát yên ngựa và có bằng chứng thực nghiệm rằng nó thiên về cực tiểu "phẳng" (mục 4.5). Đây là lĩnh vực còn tranh luận, không phải định lý.

**Thực tế tính toán.** Lab 2 chạy SGD minibatch 32 trên một bài hồi quy κ ≈ 98: với η = 0.5/L hằng, f − f* dừng ở khoảng 9.3 × 10⁻⁵ dù chạy bao lâu; giữ η nửa đầu rồi giảm tuyến tính về η/100 thì xuống 3.0 × 10⁻⁷. GD full-batch cùng bài và cùng η đạt ≈ 10⁻¹⁶ sau 3 125 bước (2.0 × 10⁻³ sau 200 bước), nhưng mỗi bước tốn gấp 500/32 ≈ 16 lần.

## 4.4 Momentum, Nesterov, AdaGrad/RMSProp/Adam/AdamW, warmup và lịch learning rate

Momentum sửa vấn đề κ bằng cách nhớ hướng đi: trên bài bậc hai, nó giảm số bước từ cỡ κ xuống cỡ √κ (với hàm strongly convex tổng quát, đảm bảo tăng tốc thuộc về Nesterov, xem bên dưới). Adam sửa một vấn đề khác: thang đo khác nhau giữa các tọa độ. AdamW sửa một lỗi tinh tế của Adam với weight decay.

**Momentum (heavy ball, Polyak).** Thêm một biến vận tốc v:

$$
v_{k+1} = \beta v_k - \eta \nabla f(x_k), \qquad x_{k+1} = x_k + v_{k+1}
$$

PyTorch viết dạng tương đương v ← βv + g, w ← w − ηv (với η hằng thì hai dạng cho cùng quỹ đạo). v là trung bình rò rỉ (EMA) của gradient với cửa sổ hiệu dụng 1/(1 − β) bước; khi gradient gần như không đổi, v ≈ −η∇f/(1 − β), nên **learning rate hiệu dụng là η/(1 − β)**. β = 0.9 nhân η hiệu dụng lên 10 lần: đổi β mà quên chỉnh η rất dễ phân kỳ.

**Vì sao momentum nhanh hơn: phân tích trên bài bậc hai.** Theo hướng riêng có trị riêng λ, đặt a = ηλ. Cặp (xₖ₊₁, xₖ) biến đổi bằng một ma trận 2 × 2:

$$
\begin{pmatrix} x_{k+1} \\ x_k \end{pmatrix}
= \begin{pmatrix} 1 + \beta - a & -\beta \\ 1 & 0 \end{pmatrix}
\begin{pmatrix} x_k \\ x_{k-1} \end{pmatrix}
$$

- Ổn định khi mọi trị riêng của ma trận này có mô-đun < 1, tương đương 0 < a < 2 + 2β (bài tập 11). Với β = 0.9, miền là a < 3.8, gần gấp đôi miền a < 2 của GD.
- Phương trình đặc trưng t² − (1 + β − a)t + β = 0 có biệt thức (1 + β − a)² − 4β. Nếu âm (a nằm giữa (1 − √β)² và (1 + √β)²) thì hai trị riêng phức và mô-đun của chúng bằng √β bất kể a. Biệt thức bằng 0 đúng ở hai biên a = (1 ± √β)²: nghiệm thực kép t = ±√β. Để mọi λ ∈ [μ, L] đều có mô-đun √β thì đặt hai đầu đúng ở hai biên đó: ημ = (1 − √β)² và ηL = (1 + √β)². Lấy căn rồi cộng: √η·(√L + √μ) = 2, nên η = 4/(√L + √μ)². Chia: √κ = (1 + √β)/(1 − √β), nên √β = (√κ − 1)/(√κ + 1). Kết quả:

$$
\eta^* = \frac{4}{(\sqrt{L} + \sqrt{\mu})^2}, \qquad
\beta^* = \Big(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\Big)^2, \qquad
\text{tốc độ } \sqrt{\beta^*} = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}
$$

So với (κ − 1)/(κ + 1) của GD, κ được thay bằng √κ: số bước ≈ √κ·ln(1/ε). Phân tích này chỉ áp dụng cho bài bậc hai. Hình ảnh: theo hướng dốc, gradient đổi dấu liên tục nên trung bình của nó nhỏ (triệt tiêu dao động); theo hướng thoải, gradient cùng dấu nên được cộng dồn (tăng tốc).

**Ví dụ giải mẫu 4.8 — momentum trên bài κ = 100.** Cùng f và x₀ như ví dụ 4.4.

- η* = 4/(10 + 1)² = 4/121 ≈ 0.0331; β* = (9/11)² ≈ 0.669; tốc độ 9/11 ≈ 0.818.
- Lý thuyết ln(10⁻⁶)/ln(0.818) ≈ 69 bước; chạy thật 93. Phần chênh do hằng số trước tốc độ: ở tham số tối ưu, ma trận 2 × 2 có trị riêng kép và không chéo hóa được, nên sai số cỡ k·(0.818)ᵏ thay vì (0.818)ᵏ.
- So với 691 bước của GD tối ưu: nhanh khoảng 7.4 lần, đúng bậc của √κ = 10 so với κ = 100.
- ‖x‖ sau 50 / 100 / 200 bước: GD tối ưu 0.52 / 0.19 / 0.026; momentum 4.1 × 10⁻³ / 3.5 × 10⁻⁷ / 1.4 × 10⁻¹⁵.
- Momentum không đơn điệu: ở bước 4, x₂ vọt tới ≈ 3.7 trước khi tắt dần. Loss tăng tạm thời khi dùng momentum là bình thường.

% manim: MomentumBongLan

**Nesterov.** Lấy gradient tại điểm "nhìn trước" x + βv thay vì tại x:

$$
v_{k+1} = \beta v_k - \eta \nabla f(x_k + \beta v_k), \qquad x_{k+1} = x_k + v_{k+1}
$$

Ưu điểm đã chứng minh: với hàm lồi L-smooth (không cần strongly convex), Nesterov với lịch β thích hợp đạt f − f* = O(1/k²), so với O(1/k) của GD, và đây là tốc độ tốt nhất có thể cho phương pháp bậc một trên lớp hàm này. Về √κ cho hàm không bậc hai: với f L-smooth và μ-strongly convex tổng quát, chính Nesterov (với β thích hợp, cỡ (√κ − 1)/(√κ + 1)) có đảm bảo tốc độ cỡ (1 − 1/√κ) mỗi bước, còn heavy ball chỉ chứng minh được tăng tốc trên bài bậc hai. Tham số tối ưu của bài bậc hai thậm chí có thể làm heavy ball không hội tụ trên một hàm strongly convex khác: phản ví dụ của [Lessard, Recht và Packard (2016)](https://arxiv.org/abs/1408.3595). Đây là lý do lý thuyết để ưu tiên Nesterov. Trên một bài cụ thể với tham số cố định thì không có thứ tự chắc chắn. Lab 3 đo trên bài κ = 100: với η = 0.01, β = 0.9, heavy ball cần 258 bước, Nesterov 207; với η = 0.01, β = 9/11, heavy ball 146, Nesterov 155. Vậy nói "Nesterov có đảm bảo lý thuyết tốt hơn cho hàm lồi tổng quát, thực tế phụ thuộc tham số", đừng nói "Nesterov luôn nhanh hơn".

**Từ AdaGrad đến Adam.** Ý tưởng chung: chia bước của mỗi tọa độ cho một ước lượng độ lớn gradient của chính tọa độ đó.

- **AdaGrad**: chia cho căn của *tổng* bình phương gradient từ đầu. Tốt cho đặc trưng thưa (tọa độ hiếm khi có gradient giữ được bước lớn). Nhược: tổng chỉ tăng, nên bước giảm đơn điệu và với mạng sâu thường dừng quá sớm.
- **RMSProp**: thay tổng bằng EMA của bình phương gradient, nên bước không tắt dần.
- **Adam** ([Kingma và Ba, 2015](https://arxiv.org/abs/1412.6980)): RMSProp cộng momentum (EMA của gradient) cộng bias correction. Algorithm 1 của paper, với gₜ là gradient ở bước t, mọi phép toán theo từng tọa độ:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2, \qquad
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad
\hat{v}_t = \frac{v_t}{1-\beta_2^t}, \qquad
\theta_t = \theta_{t-1} - \alpha\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

Mặc định trong paper: α = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸, m₀ = v₀ = 0.

**Adam là một tiền điều kiện chéo.** Bước của tọa độ i là α·m̂ᵢ/√v̂ᵢ: gradient chia cho căn trung bình bình phương của chính nó. Nhân gradient của tọa độ i với hằng số c > 0 thì m̂ᵢ nhân c, √v̂ᵢ nhân c, bước không đổi (bỏ qua ε). Vậy Adam bất biến với việc đổi thang gradient theo từng tọa độ. Nó *không* xoay hệ trục: nếu Hessian không chéo (các hướng riêng nghiêng so với trục tọa độ), Adam không sửa được κ do tương quan giữa các tọa độ. Đây là câu trả lời cho tầng 4 của chuỗi phỏng vấn ở chương 1.

**Ví dụ giải mẫu 4.9 — bias correction và bước đầu của Adam.** Gradient hằng g, mặc định của paper.

- Ở t = 1: m₁ = 0.1g, v₁ = 0.001g², m̂₁ = g, v̂₁ = g². Bước = α·g/(|g| + ε) ≈ α·sign(g). Với g = 10⁻⁴, 1, 10³, bước lần lượt là 0.0009999, 0.0010000, 0.0010000 (lệch ở g = 10⁻⁴ do ε). Adam "quên" độ lớn gradient ở bước đầu.
- Bỏ bias correction: bước 1 = α·0.1|g|/√(0.001g²) = α·0.1/√0.001 ≈ 3.16α. Bước 10 ≈ 6.53α. Bước đầu **to hơn** khoảng 3 lần chứ không nhỏ đi: m bị co 10 lần nhưng √v bị co √1000 ≈ 31.6 lần.
- Thừa số 1 − β₂ᵗ bằng 0.001, 0.00996, 0.0952, 0.632 ở t = 1, 10, 100, 1 000: bias của v kéo dài cả nghìn bước. Thừa số 1 − β₁ᵗ bằng 0.1, 0.651, ≈ 1.000 ở t = 1, 10, 100: bias của m hết sau vài chục bước. Khớp với cửa sổ EMA 10 bước và 1 000 bước.
- Kiểm bất biến đổi thang (lab 4): chạy Adam 50 bước trên f = ½(x₁² + 100x₂²) với **α = 0.01** (không phải mặc định 10⁻³; như trong lab 4) và ε = 10⁻¹², lần hai nhân gradient từng tọa độ với (10⁴, 10⁻²). Hai quỹ đạo trùng nhau, cùng ra (0.5368, 0.5368). Với α = 10⁻³ thì kết quả là (0.9503, 0.9503), vẫn trùng nhau giữa hai lần chạy. Chú ý cả hai tọa độ bằng nhau dù độ cong khác nhau 100 lần: Adam bước như nhau theo mỗi tọa độ.

**Weight decay ≠ L2 khi dùng Adam ⇒ AdamW.** Với SGD thuần, thêm (λ/2)‖θ‖² vào loss cho cập nhật θ ← θ − η(∇f + λθ) = (1 − ηλ)θ − η∇f: đúng là "co θ theo hệ số 1 − ηλ" (weight decay). Hai cách viết tương đương (sau khi đổi thang λ theo η). Với Adam thì không: số hạng λθ được cộng vào gradient *trước* khi chia cho √v̂, nên tham số có gradient lớn bị decay ít, tham số có gradient nhỏ bị decay nhiều. [Loshchilov và Hutter](https://arxiv.org/abs/1711.05101) đề xuất tách rời (decoupled): cập nhật Adam chỉ dùng gradient của loss, rồi trừ riêng một số hạng decay.

$$
\text{L2 trong Adam: } g_t \leftarrow \nabla f(\theta_{t-1}) + \lambda\theta_{t-1} \text{ rồi chạy Adam}
$$

$$
\text{AdamW, dạng PyTorch (lr = } \alpha\text{): } \theta_t = \theta_{t-1} - \alpha\Big(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\,\theta_{t-1}\Big)
$$

$$
\text{AdamW, Algorithm 2 của paper: } \theta_t = \theta_{t-1} - \eta_t\Big(\frac{\alpha\,\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\,\theta_{t-1}\Big)
$$

Hai dạng khác nhau ở chỗ nào nhân λ. Trong paper, ηₜ là hệ số lịch (schedule multiplier) và α là learning rate gốc, nên decay là ηₜ·λ·θ, **không** nhân α. PyTorch gộp α·ηₜ thành lr và trừ lr·λ·θ, nên λ của PyTorch tương ứng λ của paper chia α, và đổi lr là đổi luôn mức decay mỗi bước. Ví dụ 4.10, lab 4 và mini-test dùng dạng PyTorch.

Trong PyTorch: `torch.optim.Adam(weight_decay=λ)` là L2 (cộng λθ vào gradient); `torch.optim.AdamW` là bản tách rời (trừ lr·λ·θ), mặc định weight_decay = 0.01. Từ PyTorch 2.7, `Adam` có thêm cờ `decoupled_weight_decay` (mặc định False); `AdamW` chính là `Adam` với cờ này bật (kiểm mã nguồn ngày 04/11/2026). Muốn AdamW thì dùng lớp `AdamW`.

**Ví dụ giải mẫu 4.10 — L2 và AdamW khác nhau bao nhiêu.** α = 10⁻³, λ = 0.01, θ = 1.

- Tham số a có gradient loss 10, tham số b có gradient loss 0.01. "L2 trong Adam": gradient tổng là g + λθ; bước đầu Adam chuẩn hóa nên bước tổng ≈ α. Phần do decay đóng góp tỉ lệ λθ/(g + λθ): a chỉ ≈ 0.1% (decay hiệu dụng ≈ 10⁻⁶), b tới 50% (≈ 5 × 10⁻⁴). AdamW (dạng PyTorch): cả hai đều bị decay đúng α·λ·θ = 10⁻⁵.
- Tham số có gradient loss bằng 0, chạy 1 000 bước: AdamW đưa θ từ 1 về 0.9900 (= (1 − 10⁻⁵)¹⁰⁰⁰, trùng SGD + L2 với η = 10⁻³). "L2 trong Adam" đưa θ về 0.258: gradient duy nhất là λθ, Adam chuẩn hóa nó nên mỗi bước θ giảm khoảng α, gần như bất kể λ, chừng nào λθ còn lớn hơn ε nhiều (ε = 0: đúng 0.2577 với mọi λ > 0; ε = 10⁻⁸, λ = 10⁻⁵ đo được 0.2583). Mức decay phụ thuộc lịch sử gradient, không còn là một siêu tham số độc lập (bài tập 12 chứng minh 0.258 không phụ thuộc λ).

% manim: AdamChuanHoaBuoc

**Lịch learning rate.** Ba họ hay gặp: step decay (chia η cho 10 ở vài mốc), exponential, và cosine (từ [SGDR của Loshchilov và Hutter](https://arxiv.org/abs/1608.03983), dùng không restart là dạng phổ biến nhất). Với Transformer và LLM, công thức chuẩn là warmup tuyến tính W bước rồi cosine (hoặc tuyến tính) về gần 0 ở bước T:

$$
\eta(t) = \eta_{\text{peak}} \cdot \frac{t+1}{W} \;\;(t < W), \qquad
\eta(t) = \tfrac{1}{2}\,\eta_{\text{peak}}\Big(1 + \cos\Big(\pi\,\frac{t - W}{T - W}\Big)\Big) \;\;(W \le t < T)
$$

**Ví dụ giải mẫu 4.11 — warmup + cosine.** η đỉnh 3 × 10⁻⁴, W = 1 000, T = 10 000. η ở bước 0, 499, 999: 3 × 10⁻⁷, 1.5 × 10⁻⁴, 3 × 10⁻⁴. Giữa đoạn cosine (bước 5 500, (t − W)/(T − W) = 0.5): 1.5 × 10⁻⁴. Bước 9 999: ≈ 9 × 10⁻¹².

**Vì sao cần warmup.** Có hai giải thích có bằng chứng, chưa có lý thuyết đóng. (a) Batch lớn và η lớn: ở đầu huấn luyện độ cong có thể lớn và gradient thay đổi nhanh, η đầy đủ ngay bước đầu dễ phân kỳ (Goyal và cộng sự). (b) Riêng với Adam: ở những bước đầu v̂ được ước lượng từ rất ít gradient nên có phương sai lớn, và "learning rate thích nghi" α/√v̂ dao động mạnh ([Liu và cộng sự, RAdam](https://arxiv.org/abs/1908.03265)). Warmup giữ bước nhỏ trong lúc thống kê ổn định lại.

**Bảng chọn nhanh.**

| Bài toán | Lựa chọn thường gặp | Ghi chú |
| --- | --- | --- |
| CNN thị giác (ResNet) | SGD + momentum 0.9, step hoặc cosine | Thường tổng quát hóa tốt, cần tinh chỉnh η |
| Transformer, NLP, LLM | AdamW, warmup + cosine hoặc linear decay | Gần như mặc định; weight decay thường 0.01–0.1 |
| Bài nhỏ, full-batch, lồi | L-BFGS (mục 4.5) | Logistic regression của scikit-learn |
| Đặc trưng rất thưa | AdaGrad hoặc Adam | Tọa độ hiếm vẫn được bước lớn |

**Vì sao ML cần cái này.**

- **Nhánh NLP của bạn dùng AdamW + warmup gần như mọi lúc.** Hiểu nó để biết vì sao loss nổ khi bỏ warmup, vì sao weight decay "không có tác dụng" khi dùng nhầm `Adam(weight_decay=...)`.
- **Momentum không chỉ để làm mượt nhiễu.** Trên bài xác định *bậc hai* nó đổi κ thành √κ (hàm tổng quát: xem Nesterov); trên bài nhiễu nó cũng làm mượt. Hai vai trò khác nhau.
- **ε của Adam không vô hại.** Khi √v̂ nhỏ cỡ ε (gradient rất nhỏ, bf16), ε quyết định bước. Một số nhóm tăng ε để tránh bước bùng nổ ở các tham số có v̂ gần 0; đây là kinh nghiệm thực hành, không phải quy tắc chung.

## 4.5 Phương pháp bậc hai và bối cảnh không lồi

Newton không quan tâm κ: nó giải bài bậc hai lồi chặt trong một bước và hội tụ bậc hai gần nghiệm. Deep learning vẫn không dùng nó vì bộ nhớ, nhiễu minibatch và điểm yên ngựa.

**Bước Newton.** Cực tiểu hóa xấp xỉ bậc hai của f quanh xₖ (bước GD dùng phạt (1/2η)‖Δ‖², Newton dùng chính Hessian):

$$
x_{k+1} = \arg\min_{y} \Big[ f(x_k) + \nabla f(x_k)^{\top}(y-x_k) + \tfrac12 (y-x_k)^{\top} H_k (y-x_k) \Big] = x_k - H_k^{-1} \nabla f(x_k)
$$

- **Bất biến affine.** Đổi biến x = Az làm Hessian và gradient đổi theo, nhưng bước Newton đổi đúng theo. Vì thế κ không ảnh hưởng: trên bài bậc hai, xấp xỉ bậc hai là chính xác, nên một bước tới nghiệm.
- **Hội tụ bậc hai.** Gần nghiệm (H PD, Hessian Lipschitz) sai số thỏa eₖ₊₁ ≤ C·eₖ²: số chữ số đúng gấp đôi mỗi bước.
- **Nhược.** Cần H PD (nếu không, "cực tiểu của xấp xỉ bậc hai" không tồn tại hoặc là điểm yên ngựa). Bộ nhớ O(n²), giải hệ O(n³). Trong code, giải H·Δ = ∇f bằng `np.linalg.solve`, không nghịch đảo H.

**Ví dụ giải mẫu 4.12 — Newton hội tụ bậc hai.** f(x) = eˣ − 2x, nghiệm x* = ln 2 ≈ 0.693147. Bước Newton x ← x − (eˣ − 2)/eˣ, từ x₀ = 0:

| k | xₖ | Sai số \|xₖ − x*\| |
| --- | --- | --- |
| 1 | 1.0 | 0.307 |
| 2 | 0.73576 | 0.0426 |
| 3 | 0.694042 | 8.95 × 10⁻⁴ |
| 4 | 0.6931476 | 4.0 × 10⁻⁷ |
| 5 | 0.69314718056 | 8.0 × 10⁻¹⁴ |

Từ bước 3, số mũ của sai số gần gấp đôi mỗi bước. Trên bài κ = 100 của ví dụ 4.4, Newton đi từ (1, 1) tới (0, 0) trong đúng 1 bước. Đặt cạnh nhau: GD tối ưu 691 bước, momentum tối ưu 93, Newton 1.

**Quasi-Newton: BFGS và L-BFGS.** Không tính Hessian mà xây dần một xấp xỉ từ các cặp sₖ = xₖ₊₁ − xₖ và yₖ = ∇f(xₖ₊₁) − ∇f(xₖ). Xấp xỉ được ép thỏa **phương trình secant**: Bₖ₊₁sₖ = yₖ với Bₖ ≈ ∇²f, tương đương Hₖ₊₁yₖ = sₖ với Hₖ ≈ (∇²f)⁻¹ (BFGS cập nhật trực tiếp Hₖ). Hessian xấp xỉ mới phải giải thích được sự thay đổi gradient vừa quan sát. BFGS lưu ma trận n × n. L-BFGS chỉ giữ m cặp gần nhất (thường 5–20) và tính tích H⁻¹∇f bằng một vòng lặp hai chiều, nên bộ nhớ O(mn). L-BFGS là lựa chọn mạnh cho bài full-batch, gradient chính xác, cỡ vừa: logistic regression trong scikit-learn, `scipy.optimize.minimize`, khớp tham số mô hình vật lý.

**Ví dụ giải mẫu 4.13 — Newton bị hút về yên ngựa, GD thì thoát.** f = x² − y², H = diag(2, −2), điểm yên ngựa (0, 0).

- Newton từ (1, 0.5): ∇f = (2, −1), H⁻¹∇f = (1, 0.5), bước tới đúng (0, 0). Newton tìm điểm dừng; nó không phân biệt cực tiểu với yên ngựa.
- GD với η = 0.1 từ (1, 10⁻³): x nhân 1 − 0.2 = 0.8, y nhân 1 + 0.2 = 1.2 mỗi bước. Sau 38 bước |y| > 1 (1.2³⁸ ≈ 1 021 > 1 000), trong khi x co về khoảng 2 × 10⁻⁴. GD thoát theo hướng trị riêng âm với tốc độ hình học.
- GD cần một thành phần khác 0 theo hướng âm để thoát. Nhiễu SGD cung cấp đúng điều đó. Hai mặt của việc "GD thoát yên ngựa": với khởi tạo ngẫu nhiên, GD thoát các yên ngựa chặt (strict saddle, có trị riêng Hessian âm) với xác suất 1 ([Lee và cộng sự, 2016](https://arxiv.org/abs/1602.04915) [cần kiểm chứng ID]); nhưng thời gian thoát có thể rất dài, thậm chí mũ theo số chiều khi đi qua một chuỗi yên ngựa ([Du và cộng sự, 2017](https://arxiv.org/abs/1705.10412)). Thành phần ban đầu 10⁻⁸ thay vì 10⁻³ cũng chỉ làm GD đi lâu hơn qua vùng gần bằng phẳng.

% manim: NewtonYenNgua

**Ví dụ giải mẫu 4.14 — bộ nhớ và số lần lặp.**

- Hessian fp32 cho n = 10⁶ tham số: 10¹² × 4 byte ≈ 3 725 GiB (khớp con số trong ghi chú CS231n). n = 10⁸: ≈ 3.7 × 10⁷ GiB. Một mô hình 7 × 10⁹ tham số thì không có ý nghĩa để tính.
- L-BFGS với m = 10 (20 vector dài n), fp32: 0.075 GiB với n = 10⁶, 7.45 GiB với n = 10⁸. Vừa với RAM, nhưng vẫn là 20 bản sao của mô hình.
- Rosenbrock (thung lũng cong hẹp) từ (−1.2, 1), SciPy với gradient giải tích: BFGS 32 lần lặp, L-BFGS-B 36, CG 36, Newton-CG 83 (Newton-CG này được truyền cả Hessian giải tích; chỉ có gradient thì 86). GD với η = 10⁻³ cần 22 845 bước để ‖x − (1, 1)‖ < 10⁻⁴. Số lần lặp của SciPy có thể lệch nhẹ theo phiên bản (đo với SciPy 1.18).

**Vì sao deep learning gần như không dùng bậc hai.**

1. **Bộ nhớ và chi phí.** Như ví dụ 4.14. Có thể tính tích Hessian–vector mà không dựng H (chi phí cỡ hai lần backprop), nhưng mỗi bước Newton cần giải một hệ, tức nhiều tích như vậy.
2. **Nhiễu minibatch.** L-BFGS và line search giả định gradient chính xác. Với gradient minibatch, yₖ = Δ∇f chứa hiệu của hai nhiễu độc lập, phương trình secant học từ nhiễu. Ghi chú CS231n coi việc làm L-BFGS chạy tốt trên minibatch là vấn đề nghiên cứu mở.
3. **Yên ngựa.** Hessian của mạng nơ-ron thường không PD, Newton thuần bị hút về yên ngựa (ví dụ 4.13). [Dauphin và cộng sự (2014)](https://arxiv.org/abs/1406.2572) đề xuất "saddle-free Newton" dùng |H| thay cho H.
4. **Bậc một đủ tốt.** [Sutskever và cộng sự (2013)](https://proceedings.mlr.press/v28/sutskever13.html) cho thấy momentum với khởi tạo và lịch tốt đạt kết quả cạnh tranh với Hessian-Free optimization trên các bài khi đó được coi là cần bậc hai.

Các phương pháp "gần bậc hai" như K-FAC, Shampoo (và Muon, optimizer trực giao hóa momentum [cần kiểm chứng mức phổ biến]) vẫn được nghiên cứu; chương này chỉ nêu tên như từ khóa để bạn tự tìm.

**Bối cảnh không lồi: yên ngựa chiếm ưu thế.** Một điểm dừng là cực tiểu chỉ khi *mọi* trị riêng Hessian dương. Trực giác của Dauphin và cộng sự: trong không gian hàng triệu chiều, điều kiện "tất cả cùng dương" hiếm khi xảy ra ở điểm có loss cao, nên phần lớn điểm dừng ở đó là yên ngựa; cực tiểu địa phương thường có loss gần với cực tiểu toàn cục. Đây là lập luận có bằng chứng thực nghiệm và mô hình hóa, không phải định lý cho mạng thật. Hệ quả thực hành: vấn đề không phải "mắc kẹt ở cực tiểu xấu" mà là "đi chậm qua vùng phẳng và yên ngựa".

**Cực tiểu phẳng và nhọn.** Quanh cực tiểu, lệch tham số một lượng δ làm loss tăng ≈ ½δᵀHδ. Cực tiểu "nhọn" (λₘₐₓ(H) lớn) nhạy với nhiễu tham số hơn.

**Ví dụ giải mẫu 4.15 — phẳng và nhọn.** Hai cực tiểu một chiều cùng loss, độ cong h = 1 và h = 100. Nhiễu tham số có độ lệch chuẩn 0.1: loss tăng kỳ vọng ½·h·0.01 = 0.005 và 0.5, gấp 100 lần. Liên hệ với mục 4.2: GD với η chỉ "đậu" được ở cực tiểu có λₘₐₓ < 2/η. Với η = 0.1, cực tiểu có độ cong 100 không ổn định (cần < 20), nên GD bị đẩy ra khỏi nó. Một lý giải vì sao learning rate lớn thiên về cực tiểu phẳng.

[Keskar và cộng sự (2016)](https://arxiv.org/abs/1609.04836) quan sát thực nghiệm: batch rất lớn dẫn tới cực tiểu nhọn hơn và khoảng cách tổng quát hóa lớn hơn. [SAM của Foret và cộng sự (2020)](https://arxiv.org/abs/2010.01412) tối ưu trực tiếp loss xấu nhất trong một lân cận nhỏ để tìm vùng phẳng. Giới hạn đã biết của khái niệm: độ nhọn phụ thuộc cách tham số hóa. Với mạng ReLU, nhân trọng số một lớp với c và chia lớp sau cho c cho cùng hàm số nhưng Hessian khác (bài tập 13 có ví dụ hai tham số). [Dinh và cộng sự (2017), "Sharp Minima Can Generalize For Deep Nets"](https://arxiv.org/abs/1703.04933) dùng đúng phép nhân c / chia c này để chỉ ra rằng cực tiểu nhọn vẫn có thể tổng quát hóa tốt; xem như một câu hỏi mở.

**Edge of Stability.** [Cohen và cộng sự (2021)](https://arxiv.org/abs/2103.00065) chạy GD full-batch trên mạng thật và thấy: λₘₐₓ(H) tăng dần trong lúc huấn luyện ("progressive sharpening") cho tới khoảng 2/η rồi lơ lửng ở đó; loss vẫn giảm về dài hạn nhưng không đơn điệu. Đây là chỗ lý thuyết "η < 2/L" của mục 4.2 gặp thực tế: L không phải hằng số của bài toán mà thay đổi theo quỹ đạo, và quỹ đạo tự điều chỉnh theo η.

**Vì sao ML cần cái này.**

- **Biết khi nào dùng L-BFGS.** Bài nhỏ, full-batch, cần độ chính xác cao (khớp tham số, logistic regression, tối ưu đầu vào cho một mạng cố định): L-BFGS thường hội tụ trong vài chục lần lặp.
- **Đọc paper optimizer mới.** Gần như mọi optimizer mới (Shampoo, Sophia, Lion…) được trình bày bằng ngôn ngữ của chương này: tiền điều kiện hoặc chuẩn hóa bước kiểu gì (Shampoo, Sophia gần với xấp xỉ bậc hai; Lion thì không, nó dùng dấu của momentum), chi phí bao nhiêu.
- **Hiểu cuộc tranh luận về tổng quát hóa.** Batch size, learning rate và độ nhọn liên quan với nhau; người phỏng vấn research thích hỏi chỗ này vì chưa có câu trả lời đóng.

## Bài tập giấy bút

14 bài, làm theo thứ tự. Mục tiêu: hết ★ trong 1 buổi, hết ★★ trong 2 buổi, ★★★ dành cho Thứ Bảy. Bài nào sai ghi vào sổ lỗi kèm lý do sai.

**★ Hiểu định nghĩa**

1. Chứng minh: nếu f lồi thì mọi cực tiểu địa phương là cực tiểu toàn cục. Dùng định nghĩa dây cung, không giả sử f khả vi.
2. Phân loại điểm dừng của f(x, y) = x² + 4xy + y². f có lồi không?
3. f(x) = ½(4x₁² + x₂²). Tìm L, μ, κ, learning rate lớn nhất còn ổn định của GD, η tối ưu, tốc độ tương ứng, và ước lượng số bước để sai số giảm 10⁶ lần.
4. Chạy tay heavy ball trên f(x) = ½x² từ x₀ = 1, v₀ = 0, với η = 0.5, β = 0.5, dùng v ← βv − ηf′(x), x ← x + v. Ghi x sau 4 bước. Dãy có đơn điệu không?

**★★ Mức đề thi đầu vào**

5. Cực tiểu x² + y² với x + 2y ≥ 5. Viết điều kiện KKT, tìm nghiệm và μ. Kiểm rằng nới vế phải từ 5 thành 5 + δ làm giá trị tối ưu tăng khoảng μδ.
6. Từ định nghĩa gradient Lipschitz ‖∇f(x) − ∇f(y)‖ ≤ L‖x − y‖ (không dùng sẵn cận bậc hai), chứng minh f(y) ≤ f(x) + ∇f(x)ᵀ(y − x) + (L/2)‖y − x‖², rồi suy ra f(x − ∇f(x)/L) ≤ f(x) − ‖∇f(x)‖²/(2L). Kiểm bằng số trên f = ½(x₁² + 100x₂²) tại (1, 1).
7. Cho f(x) = ½xᵀAx, A đối xứng PD với λₘᵢₙ = μ, λₘₐₓ = L. GD với η > 0 bất kỳ có tốc độ ρ(η) = maxᵢ|1 − ηλᵢ|. Chứng minh ρ(η) < 1 khi và chỉ khi 0 < η < 2/L, và η tối ưu là 2/(L + μ) với ρ = (κ − 1)/(κ + 1). Với L = 100, μ = 1, η = 0.0201 (ngoài miền ổn định), tính ρ và ‖xₖ‖/‖x₀‖ sau 10 bước khi x₀ nằm dọc hướng riêng λ = 100.
8. Cho f(w) = ½wᵀAw với A = diag(λ₁, λ₂), gradient nhiễu g = Aw + ξ, ξₖ độc lập giữa các bước và độc lập với wₖ, 𝔼ξ = 0, Cov ξ = Σ/B. Chứng minh hiệp phương sai dừng V thỏa V = (I − ηA)V(I − ηA) + η²Σ/B, suy ra Vᵢⱼ = η²Σᵢⱼ/(B(1 − (1 − ηλᵢ)(1 − ηλⱼ))). Tính V với λ = (1, 10), Σ có hàng (4, 2) và (2, 4), η = 0.05, B = 8. Với một chiều, kiểm lại ησ²/(B(2 − η)) của ví dụ 4.6. Điều kiện nào của η cần cho trạng thái dừng tồn tại?
9. Chứng minh rằng với SGD thuần, thêm (λ/2)‖w‖² vào loss tương đương weight decay w ← (1 − ηλ)w − η∇f. Nêu một bước trong lập luận này hỏng đi khi dùng Adam.
10. Tìm phân phối p trên {1, 2, 3} có entropy lớn nhất với ràng buộc kỳ vọng ∑ i·pᵢ = 2.5. Dùng Lagrange (gọi nhân tử là τ) để chứng minh pᵢ ∝ exp(τ·i), rồi giải số τ. So entropy với ln 3. Liên hệ với dạng softmax.

**★★★ Mức phỏng vấn nghiên cứu**

11. Chứng minh heavy ball trên một hướng riêng (ma trận 2 × 2 ở mục 4.4) ổn định khi và chỉ khi 0 < ηλ < 2 + 2β, với 0 ≤ β < 1. *Gợi ý: đa thức đặc trưng z² − (1 + β − a)z + β; dùng điều kiện để cả hai nghiệm nằm trong đường tròn đơn vị.*
12. Trong ví dụ 4.10, tham số có gradient loss bằng 0 bị "L2 trong Adam" kéo về 0.258 sau 1 000 bước. Chứng minh rằng với ε = 0, con số này không phụ thuộc λ > 0. Kết luận gì về ý nghĩa của λ trong `Adam(weight_decay=λ)` đối với tham số ít được cập nhật?
13. Xét mô hình hai tham số f(a, b) = ½(ab − 1)² (một "mạng tuyến tính hai lớp" với một mẫu). Tập cực tiểu là ab = 1. Tính λₘₐₓ của Hessian tại (1, 1) và tại (10, 0.1). Hai điểm cho cùng hàm số đầu vào → đầu ra. Bài này nói gì về việc dùng λₘₐₓ làm thước đo "độ phẳng"?
14. Cho f lồi, L-smooth, có cực tiểu x*, và GD với η = 1/L. (a) Chứng minh f(xₖ) − f* ≤ L‖x₀ − x*‖²/(2k). (b) Nếu f còn μ-strongly convex (không cần là hàm bậc hai), chứng minh ‖xₖ₊₁ − x*‖² ≤ (1 − μ/L)‖xₖ − x*‖². (c) Kiểm số trên f = ½(x₁² + 100x₂²), x₀ = (1, 1), k = 100: cận ở (a), giá trị thật của f(xₖ), cận ở (b) và ‖xₖ‖² thật. *Gợi ý: khai triển ‖xₖ₊₁ − x*‖² = ‖xₖ − x*‖² − (2/L)∇f(xₖ)ᵀ(xₖ − x*) + ‖∇f(xₖ)‖²/L²; dùng tính lồi (hoặc strongly convex) cho số hạng giữa và bổ đề giảm đủ cho số hạng cuối.*

## Bài tính nhanh

15 bài, mỗi bài tối đa 2 phút, làm bằng giấy bút và bấm giờ cả bộ. Mục tiêu: xong cả 15 bài trong 25 phút, sai không quá 1 bài. Làm lại bộ này sau 3 ngày và sau 1 tuần.

1. f(x, y) = x² + y² − 2x. Tìm điểm dừng và giá trị f tại đó.
2. f(x) = x³ có f′(0) = 0 và f″(0) = 0. 0 có phải cực tiểu không?
3. f(x) = ½xᵀAx + bᵀx với A = diag(3, 12). L, μ, κ bằng bao nhiêu?
4. f(x) = x², x₀ = 1, η = 0.1. Tính x₁.
5. Tốc độ hội tụ 0.99 mỗi bước. Cần khoảng bao nhiêu bước để sai số giảm 100 lần?
6. λₘₐₓ của Hessian bằng 50. Learning rate lớn nhất còn ổn định của GD?
7. κ = 100. Tốc độ của GD với η tối ưu và của heavy ball với tham số tối ưu?
8. Momentum β = 0.9, η = 0.1. Learning rate hiệu dụng khi gradient không đổi?
9. Gradient một mẫu có độ lệch chuẩn 2. Độ lệch chuẩn của gradient minibatch 64?
10. Adam, α = 10⁻³, gradient bước đầu là −50. θ thay đổi bao nhiêu ở bước 1?
11. Adam bỏ bias correction, β₁ = 0.9, β₂ = 0.99. Bước đầu bằng bao nhiêu lần α?
12. Warmup 500 bước, tổng 5 500 bước, η đỉnh 10⁻³, cosine về 0. η ở bước 3 000?
13. Một bước Newton cho f(x) = x⁴/4 từ x = 1 cho x bằng bao nhiêu?
14. Hessian fp32 của mô hình 10⁷ tham số chiếm bao nhiêu GiB?
15. Cực tiểu có độ cong 40, nhiễu tham số độ lệch chuẩn 0.05. Loss tăng kỳ vọng bao nhiêu?

## Lab code và bài săn bug

Năm lab dưới đây đã chạy với Python 3.13, NumPy 2.5 và SciPy 1.18; số liệu có thể lệch nhẹ theo phiên bản. Mọi lab dùng NumPy thuần (PyTorch bắt đầu từ chương 6). Cách học đúng: tự viết từng lab trước, chỉ đối chiếu code mẫu khi kết quả lệch với cột "kết quả mong đợi".

| Lab | Việc phải làm | Kết quả mong đợi |
| --- | --- | --- |
| 1. GD và số điều kiện | Cài `gd_steps`, đếm bước trên f = ½(x₁² + 100x₂²) với η = 0.01, 2/101, 0.0201; đo κ trước và sau chuẩn hóa (ví dụ 4.5) | 1 341 / 691 / phân kỳ; κ từ 45.31 về 1.00 |
| 2a. Sàn nhiễu SGD | Mô phỏng ví dụ 4.6 với 4 000 chuỗi song song, so với ησ²/(B(2 − η)) | 0.2111 / 0.0132 / 0.0200, so với lý thuyết 0.2105 / 0.0132 / 0.0201 |
| 2b. SGD minibatch trên hồi quy (n = 500, d = 10, κ ≈ 98) | Cài SGD có xáo trộn mỗi epoch, B = 32, 200 epoch; η hằng và η giảm dần | f − f*: η hằng 9.3 × 10⁻⁵; giữ rồi giảm tuyến tính 3.0 × 10⁻⁷; đối chứng GD full-batch cùng η: 2.0 × 10⁻³ sau 200 bước, 1.1 × 10⁻¹⁶ sau 3 125 bước |
| 3. Momentum và Nesterov | Cài heavy ball và Nesterov, đếm bước trên bài κ = 100 | Heavy ball tối ưu 93 bước (GD tối ưu 691, nhanh 7.4 lần); η = 0.01, β = 0.9: HB 258, Nesterov 207; β = 9/11: HB 146, Nesterov 155 |
| 4. Adam và AdamW từ đầu | Cài `adam_step` theo Algorithm 1, có cả L2 và decoupled | Bước đầu 0.0009999–0.0010000 với g = 10⁻⁴, 1, 10³; hai quỹ đạo đổi thang (lr = 0.01, ε = 10⁻¹²) cùng ra (0.5368, 0.5368); L2 0.2577, AdamW 0.9900; hồi quy (lr = 0.05) ‖w − w*‖ = 7 × 10⁻¹⁵ |
| 5. Newton, L-BFGS, yên ngựa | Newton cho logistic regression (d = 3, n = 300), so với L-BFGS-B; Rosenbrock; x² − y²; logistic trên 4 điểm tách được, không có cột bias: (1, 1), (2, 1), (−1, −1), (−2, −1) | Newton 6 bước, L-BFGS-B 9 lần lặp, cùng nghiệm (−0.513, 2.373, −1.143); Rosenbrock BFGS 32, L-BFGS-B 36, CG 36, GD 22 845; GD thoát yên ngựa sau 38 bước; trên dữ liệu tách được ‖w‖ = 3.41, 4.95, 6.54, 8.16 ở bước 10² … 10⁵ |

Code nằm trong notebook `labs/chuong04/lab.ipynb` (bản có chỗ TODO để bạn tự viết, kèm ô kiểm tra tự động) và `labs/chuong04/loigiai.ipynb` (bản lời giải đã chạy kiểm). Mở từ trang **Lab** của app.

**Câu hỏi sau lab.**

- Lab 1: số bước lý thuyết với η = 0.01 là ln(10⁻⁶)/ln(0.99) ≈ 1 375, nhưng chạy thật 1 341. Giải thích chênh lệch chính xác tới từng bước.
- Lab 3: 93/691 ≈ 1/7.4, trong khi √κ/κ = 1/10. Vì sao momentum không nhanh gấp đúng 10 lần? Nesterov thắng ở một bộ tham số và thua ở bộ kia: điều này có mâu thuẫn với định lý O(1/k²) không?
- Lab 5: Newton cần 6 bước còn L-BFGS-B cần 9 lần lặp. Vì sao trên bài này Newton vẫn là lựa chọn hợp lý, còn với d = 10⁶ thì không?

**Bài săn bug.** Hàm dưới đây huấn luyện hồi quy tuyến tính (n = 1 000, d = 5, w thật = (1, 2, 3, 4, 5), nhiễu 0.5) bằng SGD + momentum + lịch cosine. Nó chạy không báo lỗi, không NaN, loss vẫn giảm, nhưng sau 20 epoch ‖w − w*‖ ≈ 0.334 và f − f* ≈ 0.054, trong khi bản đúng đạt ‖w − w*‖ ≈ 0.0019 và f − f* ≈ 2 × 10⁻⁶. Tìm cả **ba** lỗi trước khi xem lời giải. Một lỗi gần như không thấy được qua loss cuối; hãy nghĩ cách phát hiện nó.

```python
def train_buggy(epochs=20, B=32, lr0=0.05, beta=0.9):
    w, v = np.zeros(db), np.zeros(db)
    T, t = epochs * (nb // B), 0
    r = np.random.default_rng(1)
    perm = r.permutation(nb)
    for ep in range(epochs):
        for i in range(nb // B):
            idx = perm[:B]
            Xi = Xb[idx]
            g = Xi.T @ (Xi @ w - yb[idx]) / B
            lr = 0.5 * lr0 * (1 + np.cos(np.pi * t))
            v = beta * v + g
            w = w - lr * g
            t += 1
    return w
```

## Bài dựng cảnh Manim

Mỗi chương có một bài tự dựng cảnh: muốn vẽ được một khái niệm, bạn phải hiểu từng bước của nó. Giới hạn **1 giờ** (tính vào luồng C), không chỉnh màu mè.

**Đề bài.** Dùng `manim/chuong04_scenes.py` làm mẫu, viết cảnh `ThanhTruotEta` vào thư mục `manim/cua-toi/`:

1. Vẽ một trục số cho η từ 0 đến 0.022, tô đỏ đoạn η > 2/L = 0.02.
2. Một con trỏ chạy lần lượt tới η = 0.01, 2/101 và 0.0201 (dùng `ValueTracker`).
3. Bảng bên dưới cập nhật liên tục: |1 − η| (hướng λ = 1), |1 − 100η| (hướng λ = 100), và số bước tới 10⁻⁶ theo công thức ln(10⁻⁶)/ln(max|1 − ηλ|), hoặc chữ "phân kỳ".

**Tiêu chí tự chấm.** (1) Đúng toán: ở η = 2/101 hai giá trị |1 − ηλ| bằng nhau (0.9802) và số bước là 691; ở η = 0.01 công thức cho 1 375 (khác 1 341 của ví dụ 4.4 vì tiêu chí dừng khác, xem câu hỏi sau lab 1). (2) Có tiêu đề và một dòng kết luận nói người xem phải thấy điều gì. (3) Dài không quá 30 giây, chữ không chồng lên nhau.

Render bằng nút **Render cảnh của tôi** ở trang Manim. Lời giải tham khảo nằm ở `manim/loigiai/chuong04_toc_do_huong_rieng.py` — chỉ mở sau khi đã tự làm.

## Phỏng vấn, bẫy, thực tế khác lý thuyết

**Cách luyện.** Mỗi chuỗi có 4 tầng câu hỏi, đào sâu dần như người phỏng vấn thật. Đây là các câu hỏi tiêu biểu trong phỏng vấn ML research, tổng hợp từ nội dung các nguồn ở cuối chương, không phải đề của một công ty cụ thể. Trả lời to bằng tiếng Việt trước, sau đó nói lại bằng tiếng Anh, 2 phút mỗi tầng, ghi âm và tự nghe lại. Câu trả lời mẫu tiếng Anh chỉ phủ tầng 1–2; tầng 3–4 bạn tự soạn, rồi đối chiếu với lời giải các bài tập liên quan.

**Chuỗi 1 — Gradient descent và số điều kiện**

1. Vì sao learning rate quá lớn làm GD phân kỳ? Điều kiện chính xác trên bài bậc hai là gì?
2. Learning rate tối ưu là bao nhiêu, và số bước phụ thuộc κ thế nào?
3. Momentum cải thiện sự phụ thuộc vào κ ra sao, và vì sao? (Bài tập 11, ví dụ 4.8. Gợi ý: trả lời cho bài bậc hai; với hàm strongly convex tổng quát thì cần Nesterov, vì heavy ball không có đảm bảo.)
4. Trên mạng nơ-ron thật, điều kiện η < 2/L có còn đúng không? (Edge of Stability: λₘₐₓ tự tăng tới ≈ 2/η.)

> *Model answer:* On a quadratic with Hessian eigenvalues λᵢ, one gradient step multiplies the error along each eigenvector by (1 − ηλᵢ). So the iteration is stable only if |1 − ηλᵢ| < 1 for every i, which means 0 < η < 2/λₘₐₓ; above that, the steepest direction oscillates with growing amplitude. The convergence rate is the largest |1 − ηλᵢ|, which is minimized at η = 2/(λₘₐₓ + λₘᵢₙ) and gives a rate of (κ − 1)/(κ + 1). So the number of iterations grows roughly linearly with the condition number, and even the optimal step size only buys a factor of about two over η = 1/L.

**Chuỗi 2 — Adam**

1. Viết cập nhật của Adam.
2. Bias correction để làm gì? Bỏ đi thì bước đầu to hơn hay nhỏ hơn?
3. Adam với L2 khác AdamW thế nào? Vì sao Transformer dùng AdamW?
4. Adam có đảm bảo hội tụ cho bài lồi không? ([Reddi, Kale, Kumar](https://arxiv.org/abs/1904.09237) dựng phản ví dụ lồi và đề xuất AMSGrad.)

> *Model answer:* Adam keeps an exponential moving average of the gradient, m, and of the squared gradient, v, then updates each parameter by α·m̂/(√v̂ + ε), where the hats denote bias-corrected estimates. Bias correction is needed because both averages start at zero, so early on they underestimate the true moments. A common misconception is that without it the first steps are too small; with the default β₂ = 0.999, v is shrunk much more than m, so the first step is actually about 3.16 times larger than α. With correction, the first step has magnitude almost exactly α regardless of the gradient scale.

**Chuỗi 3 — SGD và batch size**

1. Vì sao SGD hội tụ dù gradient nhiễu? Với learning rate cố định, nó dừng ở đâu?
2. Tăng batch 8 lần thì chỉnh learning rate thế nào? (Quy tắc scale tuyến tính, warmup.)
3. Vì sao batch rất lớn có thể tổng quát hóa kém? (Keskar và cộng sự, cực tiểu nhọn.)
4. Vì sao Adam cần warmup ngay cả khi batch nhỏ? (Phương sai của v̂ ở những bước đầu, RAdam.)

> *Model answer:* The minibatch gradient is an unbiased estimate of the full gradient, so on average each step points downhill, and its variance shrinks like 1/B. With a constant learning rate, though, SGD does not converge to the minimizer: it settles into a noise ball whose variance scales with η/B, so its radius scales with √(η/B). On f(w) = ½w² with gradient noise variance σ²/B, the stationary variance is ησ²/(B(2 − η)), so increasing the batch 16 times shrinks it 16 times. To actually converge you need to decay the learning rate, which is why every practical schedule ends with a small learning rate. If I increase the batch size 8 times, the gradient noise drops, so I raise the learning rate 8 times as well; this linear scaling rule of Goyal et al. keeps η/B constant. It is an empirical rule, tested for SGD with momentum up to batch size 8 192, not a theorem. It stops working when η·λₘₐₓ reaches the stability limit of 2, and the curvature is large at the start of training, so I add a linear warmup over the first few epochs. With Adam people usually scale more cautiously.

**Chuỗi 4 — Bậc hai và không lồi**

1. Newton khác GD thế nào? Ưu điểm chính?
2. Vì sao deep learning không dùng Newton hay L-BFGS?
3. Loss của mạng nơ-ron có nhiều cực tiểu địa phương xấu không, hay yên ngựa mới là vấn đề? (Dauphin và cộng sự.)
4. Cực tiểu phẳng có thật sự tổng quát hóa tốt hơn không? Định nghĩa độ phẳng có vấn đề gì? (Phụ thuộc tham số hóa, bài tập 13; SAM.)

> *Model answer:* Newton's method minimizes the local quadratic model, so its step is −H⁻¹∇f instead of −η∇f. It is invariant to affine changes of variables, so the condition number does not matter: it solves a strictly convex quadratic in one step and converges quadratically near a minimum. Deep learning rarely uses it for three reasons: storing and inverting the Hessian is impractical at a million parameters (about 3.7 TB in fp32 just to store it), the curvature estimates break down with noisy minibatch gradients, and since the Hessian is usually indefinite, a pure Newton step is attracted to saddle points rather than minima. Well-tuned first-order methods with momentum are cheap and turn out to be good enough.

**Từ vựng nên nói trôi chảy:** convex, strongly convex, L-smooth, condition number, stationary point, saddle point, KKT conditions, complementary slackness, descent lemma, linear convergence, noise floor, linear scaling rule, heavy-ball momentum, Nesterov acceleration, bias correction, decoupled weight decay, warmup, cosine schedule, diagonal preconditioner, quasi-Newton, secant equation, sharpness, edge of stability.

**Bẫy thường gặp.**

- **`Adam(weight_decay=λ)` là L2, không phải AdamW.** Ví dụ 4.10: tham số không có gradient bị kéo về 0.258 thay vì 0.990. Với Transformer, dùng `AdamW`.
- **"Bỏ bias correction thì bước đầu bé."** Sai với β₂ = 0.999: bước đầu to ≈ 3.16 lần α. Chỉ khi β₂ = 0.99 (với β₁ = 0.9; tổng quát khi 1 − β₂ = (1 − β₁)²) thì bước đầu không đổi (bài tính nhanh 11).
- **"∇f = 0 và Hessian PSD ⇒ cực tiểu."** Sai với hàm không lồi: x³ tại 0. PSD là điều kiện cần; PD mới đủ. Và Newton còn nguy hiểm hơn vì nó chủ động tìm điểm có ∇f = 0 (ví dụ 4.13).
- **"Hàm lồi thì GD hội tụ tuyến tính."** Cần strongly convex. Logistic regression trên dữ liệu tách được tuyến tính không có cực tiểu hữu hạn: lab 5d lấy 4 điểm (1, 1), (2, 1), (−1, −1), (−2, −1), nhãn 1, 1, 0, 0, không có cột bias (đặc trưng thứ hai bằng +1 ở hai điểm lớp 1 và −1 ở hai điểm lớp 0; nếu bạn đổi thành bias thật, tức (−1, 1), (−2, 1), thì ‖w‖ ra 4.02, 6.23, 8.52, 10.82, mỗi lần ≈ +2.2); GD với η = 1 từ w = 0 cho ‖w‖ = 3.41, 4.95, 6.54, 8.16 ở bước 10², 10³, 10⁴, 10⁵: mỗi lần ×10 số bước chỉ cộng thêm ≈ 1.6, tức ‖w‖ tăng như log k và loss giảm về 0 mà không có cực tiểu.
- **Đổi β của momentum mà giữ nguyên η.** Learning rate hiệu dụng là η/(1 − β): tăng β từ 0.9 lên 0.99 nhân bước lên 10 lần.
- **Gradient tổng thay vì trung bình trong minibatch.** Đổi B là đổi learning rate hiệu dụng theo đúng tỉ lệ (mini-test câu 5). Quy tắc scale tuyến tính cũng không phải luật luôn đúng: Goyal và cộng sự kiểm chứng tới batch 8 192 với SGD và warmup; với Adam người ta thường scale thận trọng hơn (kinh nghiệm phổ biến, không có định lý).

**Thực tế khác lý thuyết.**

- **Edge of Stability.** Lý thuyết cổ điển cấm η > 2/L. Trên mạng thật, λₘₐₓ(H) tăng tới ≈ 2/η rồi lơ lửng ở đó, loss giảm không đơn điệu nhưng vẫn giảm (Cohen và cộng sự, 2021).
- **Adam có lý thuyết hội tụ yếu nhưng thắng thực tế.** Reddi và cộng sự có phản ví dụ lồi cho Adam; SGD + momentum có lý thuyết đẹp hơn. Thế nhưng AdamW là mặc định cho Transformer và LLM, và SGD thường kém hơn rõ rệt trên các mô hình này.
- **Lý thuyết lồi gợi ý bậc hai tốt hơn; thực tế dùng bậc một.** Newton giải bài κ = 100 trong 1 bước so với 691. Nhưng bộ nhớ, nhiễu minibatch và yên ngựa khiến deep learning dùng bậc một có momentum (Sutskever và cộng sự, 2013), và các phương pháp gần bậc hai vẫn là hướng nghiên cứu.

## Mini-test 30 phút

Làm sau khi xong bài tập và lab, bấm giờ đúng 30 phút, không xem sách, không dùng máy tính (câu 5 chỉ đọc code, không chạy). Tổng 30 điểm. **Từ 24 điểm:** sang chương 5. **18 đến dưới 24:** ôn lại mục ứng với câu sai rồi làm lại sau 3 ngày. **Dưới 18:** học lại chương.

1. **(4 điểm)** f(x, y) = x² + y⁴ − 2y². Tìm mọi điểm dừng, phân loại từng điểm, cho biết giá trị nhỏ nhất của f. f có lồi không?
2. **(5 điểm)** Cực tiểu (x − 1)² + (y − 2)² với x ≤ 0 và y ≤ 0. Viết KKT, tìm nghiệm và hai nhân tử μ₁, μ₂. Ràng buộc nào đang chạm?
3. **(5 điểm)** f = ½(x₁² + 25x₂²). (a) η tối ưu của GD, tốc độ, và ước lượng số bước để sai số giảm 10⁶ lần. (b) Tốc độ của heavy ball với tham số tối ưu và ước lượng số bước tương ứng. (c) Vì sao số bước thật của heavy ball thường lớn hơn ước lượng ở (b)?
4. **(5 điểm)** Viết cập nhật Adam đầy đủ. Giải thích vì sao bước đầu có độ lớn ≈ α bất kể độ lớn gradient, và AdamW thay đổi chính xác dòng nào.
5. **(5 điểm)** Đọc code và trả lời không chạy máy:

```python
def sgd_step(w, X, y, lr):
    g = X.T @ (X @ w - y)        # X là minibatch B×d
    return w - lr * g
```

(a) Hàm này sai ở đâu so với gradient của MSE trung bình? (b) Một người chỉnh lr cho chạy tốt ở B = 32, rồi đổi sang B = 256 mà giữ nguyên lr. Điều gì có thể xảy ra, và vì sao? (c) Sửa code. Sau khi sửa, theo quy tắc scale tuyến tính thì lr ở B = 256 nên là bao nhiêu so với ở B = 32, và cần thêm gì để an toàn?

6. **(6 điểm)** Tối đa 6 câu: vì sao huấn luyện một Transformer gần như luôn dùng AdamW với warmup và lịch giảm, chứ không dùng Newton, L-BFGS hay SGD với η hằng?

## Flashcard

18 thẻ của chương nằm trong `flashcards/chuong04.csv`. Ôn ở trang **Ôn tập** của app: app tự lên lịch theo hộp Leitner (thẻ nhớ được sẽ giãn lịch ôn 1 → 3 → 7 → 14 ngày, thẻ quên quay lại hộp 1). File CSV này cũng nhập thẳng được vào Anki nếu bạn thích dùng Anki.

## Lời giải

Chỉ đọc sau khi đã tự thử ít nhất 20 phút. Với bài chứng minh, so ý chính chứ không so từng chữ. Mọi đáp số bằng số trong chương đã được kiểm lại bằng NumPy và SciPy.

**Kiểm tra nhanh**

1. Có, nếu f lồi: f(y) ≥ f(x₀) + ∇f(x₀)ᵀ(y − x₀) = f(x₀) với mọi y. Nếu f không lồi thì không: x₀ có thể là cực đại, yên ngựa (x² − y² tại 0) hoặc cực tiểu chỉ địa phương (ví dụ 4.1).
2. η < 2/50 = 0.04. η tối ưu 2/(50 + 2) = 1/26 ≈ 0.0385. κ = 25, tốc độ (25 − 1)/(25 + 1) = 12/13 ≈ 0.923.
3. Giảm một nửa: độ lệch chuẩn tỉ lệ 1/√B, và √(64/16) = 2.
4. To hơn, khoảng 3.16 lần: bước đầu là α(1 − β₁)/√(1 − β₂) = α·0.1/√0.001.
5. Newton cực tiểu hóa xấp xỉ bậc hai, mà với bài bậc hai thì xấp xỉ đó là chính xác, nên bước −H⁻¹∇f tới thẳng nghiệm. Bước này chỉ giải ∇f = 0 cho mô hình bậc hai; nếu H có trị riêng âm, điểm dừng của mô hình là yên ngựa và Newton nhảy vào đó (ví dụ 4.13).

**Bài tập giấy bút**

1. Giả sử x* là cực tiểu địa phương: f(x*) ≤ f(z) với mọi z trong một quả cầu bán kính r quanh x*. Lấy y bất kỳ, và z = θy + (1 − θ)x* với θ > 0 đủ nhỏ để z nằm trong quả cầu. Theo định nghĩa lồi: f(x*) ≤ f(z) ≤ θf(y) + (1 − θ)f(x*). Trừ (1 − θ)f(x*) hai vế rồi chia cho θ > 0: f(x*) ≤ f(y). Vậy x* là cực tiểu toàn cục.
2. ∇f = (2x + 4y, 4x + 2y) = 0 chỉ có nghiệm (0, 0). Hessian có hàng (2, 4) và (4, 2), trị riêng 6 (vector (1, 1)) và −2 (vector (1, −1)). Trái dấu ⇒ yên ngựa. f không lồi vì Hessian không PSD. Kiểm: theo hướng (1, −1), f(t, −t) = t² − 4t² + t² = −2t² < 0.
3. Hessian diag(4, 1): L = 4, μ = 1, κ = 4. Ổn định khi η < 2/4 = 0.5. η tối ưu 2/(4 + 1) = 0.4, tốc độ (4 − 1)/(4 + 1) = 0.6. Số bước ≈ ln(10⁻⁶)/ln(0.6) ≈ 27.
4. Bước 1: v = −0.5, x = 0.5. Bước 2: v = 0.5·(−0.5) − 0.5·0.5 = −0.5, x = 0. Bước 3: v = −0.25 − 0 = −0.25, x = −0.25. Bước 4: v = −0.125 + 0.125 = 0, x = −0.25. Không đơn điệu: x vượt qua 0 sang âm (vận tốc mang x đi quá đích), đặc trưng của momentum.
5. Viết ràng buộc g = 5 − x − 2y ≤ 0. Dừng: (2x, 2y) + μ(−1, −2) = 0 ⇒ x = μ/2, y = μ. Nếu μ = 0 thì (0, 0) vi phạm 0 ≥ 5, nên ràng buộc chạm: μ/2 + 2μ = 5 ⇒ μ = 2. Nghiệm (1, 2), f* = 5, μ = 2 ≥ 0, KKT thỏa. Tổng quát với vế phải b: nghiệm (b/5, 2b/5), f* = b²/5, đạo hàm 2b/5 = 2 tại b = 5 = μ. Với b = 5.01: f* = 5.02002, tăng ≈ 0.02 = μ·0.01.
6. Bước 1, từ định nghĩa. f(y) − f(x) − ∇f(x)ᵀ(y − x) = ∫₀¹(∇f(x + t(y − x)) − ∇f(x))ᵀ(y − x) dt; theo Cauchy–Schwarz và Lipschitz, tích phân này bị chặn bởi ∫₀¹ L·t‖y − x‖² dt = (L/2)‖y − x‖². Vậy f(y) ≤ f(x) + ∇f(x)ᵀ(y − x) + (L/2)‖y − x‖². Bước 2: thay y − x = −∇f/L: f(y) ≤ f(x) − ‖∇f‖²/L + ‖∇f‖²/(2L) = f(x) − ‖∇f‖²/(2L). Kiểm số: tại (1, 1), f = 50.5, ∇f = (1, 100), ‖∇f‖²/(2L) = 10 001/200 = 50.005. Sau bước η = 0.01: x = (0.99, 0), f = 0.49005, giảm 50.00995 ≥ 50.005. Đúng.
7. Như mục 4.2: x̂ₖ,ᵢ = (1 − ηλᵢ)ᵏx̂₀,ᵢ, nên ‖xₖ‖ = ‖x̂ₖ‖ ≤ maxᵢ|1 − ηλᵢ|ᵏ‖x₀‖, với dấu bằng khi x₀ nằm dọc hướng riêng đạt max. Hàm λ ↦ |1 − ηλ| là hàm lồi theo λ nên trên [μ, L] đạt max tại một đầu mút: ρ(η) = max(|1 − ημ|, |1 − ηL|). Với η < 2/(L + μ) thì max là |1 − ημ| = 1 − ημ, giảm theo η; với η > 2/(L + μ) thì max là |1 − ηL| = ηL − 1, tăng theo η (so sánh hai giá trị cho thấy chúng bằng nhau đúng tại 2/(L + μ)); nên max nhỏ nhất khi hai giá trị bằng nhau: 1 − ημ = ηL − 1 ⇒ η = 2/(L + μ), ρ = 1 − 2μ/(L + μ) = (L − μ)/(L + μ) = (κ − 1)/(κ + 1). Với mọi η > 0 thì ρ(η) = max(|1 − ημ|, |1 − ηL|) (đầu mút của [μ, L], vì λₘₐₓ = L đạt thật), nên ρ < 1 ⇔ |1 − ηL| < 1 ⇔ η < 2/L (điều kiện |1 − ημ| < 1 yếu hơn). Với η = 0.0201: ρ = |1 − 2.01| = 1.01 > 1, và dọc hướng λ = 100 thì ‖xₖ‖/‖x₀‖ = 1.01ᵏ: sau 10 bước là 1.01¹⁰ ≈ 1.105, tức phân kỳ (chậm nhưng không dừng).
8. w′ = (I − ηA)w − ηξ, với ξ độc lập với w nên Cov w′ = (I − ηA)V(I − ηA) + η²Σ/B (số hạng chéo bằng 0). Với A chéo, phần tử (i, j): Vᵢⱼ = (1 − ηλᵢ)(1 − ηλⱼ)Vᵢⱼ + η²Σᵢⱼ/B ⇒ Vᵢⱼ = η²Σᵢⱼ/(B(1 − (1 − ηλᵢ)(1 − ηλⱼ))). Một chiều, λ = 1, Σ = σ²: mẫu số 1 − (1 − η)² = η(2 − η), cho V = ησ²/(B(2 − η)), đúng ví dụ 4.6. Số: V₁₁ = 0.0025·4/(8·0.0975) = 0.01282; V₂₂ = 0.0025·4/(8·0.75) = 0.001667; V₁₂ = 0.0025·2/(8·0.525) = 0.001190 (cả ba khớp nghiệm của phương trình Lyapunov tính bằng SciPy). Trạng thái dừng cần mọi |1 − ηλᵢ| < 1, tức η < 2/λₘₐₓ = 0.2; ở η ≥ 2/λₘₐₓ thành phần theo hướng cong nhất không ổn định, phương sai không hội tụ (một chiều: mẫu số 2 − η ≤ 0). Hướng cong nhất có phương sai dừng nhỏ nhất (V₂₂ < V₁₁), nhưng chính nó đặt trần cho η.
9. Gradient của f + (λ/2)‖w‖² là ∇f + λw, nên w ← w − η(∇f + λw) = (1 − ηλ)w − η∇f. Bước hỏng với Adam: "gradient tổng được nhân với cùng một số η cho mọi tọa độ". Adam nhân mỗi tọa độ với α/√v̂ᵢ khác nhau, và v̂ᵢ lại phụ thuộc chính λw, nên số hạng λw bị chia cho một đại lượng riêng của từng tọa độ; không còn là co đều theo hệ số 1 − αλ.
10. Cực đại −∑pᵢ ln pᵢ với ∑pᵢ = 1 và ∑ i·pᵢ = 2.5. Lagrangian: −∑pᵢ ln pᵢ + ν(∑pᵢ − 1) + τ(∑ i·pᵢ − 2.5). Đạo hàm theo pᵢ: −ln pᵢ − 1 + ν + τi = 0 ⇒ pᵢ = exp(ν − 1)·exp(τi) ∝ exp(τi). Chuẩn hóa: pᵢ = exp(τi)/∑ⱼexp(τj), đúng là softmax của các "logit" τi. Giải ∑ i·pᵢ = 2.5 bằng số: τ ≈ 0.834, p ≈ (0.116, 0.268, 0.616), entropy ≈ 0.901 nat, nhỏ hơn ln 3 ≈ 1.099 của phân phối đều (kỳ vọng 2). Ràng buộc làm mất entropy; dạng mũ là dạng "ít giả định nhất" thỏa ràng buộc kỳ vọng.
11. Đa thức đặc trưng của ma trận là p(z) = z² − (1 + β − a)z + β. Hai nghiệm có tích bằng β < 1. Nếu nghiệm phức liên hợp thì cả hai có mô-đun √β < 1: ổn định. Nếu nghiệm thực, cả hai nằm trong (−1, 1) khi và chỉ khi p(1) > 0 và p(−1) > 0 (cùng với tích < 1). p(1) = 1 − (1 + β − a) + β = a > 0. p(−1) = 1 + (1 + β − a) + β = 2 + 2β − a > 0 ⇔ a < 2 + 2β. Ngược lại, nếu a ≤ 0 hoặc a ≥ 2 + 2β thì p(1) ≤ 0 hoặc p(−1) ≤ 0, mà p(z) → +∞ khi |z| → ∞ trên trục thực, nên có một nghiệm thực với |z| ≥ 1: không ổn định. (Trường hợp nghiệm phức tự động thỏa 0 < a < 2 + 2β vì khi đó (1 + β − a)² < 4β ≤ (1 + β)², nên |1 + β − a| < 1 + β.) Kiểm số: β = 0.9 thì a = 3.79 cho mô-đun lớn nhất 0.949, a = 3.81 cho 1.065.
12. Với gradient loss bằng 0, gradient Adam nhìn thấy là gₜ = λwₜ₋₁. Giả sử chạy song song hai bản với λ và λ′ = cλ, cùng w₀. Quy nạp: nếu hai bản có cùng w tới bước t − 1, thì gradient của bản thứ hai bằng c lần bản thứ nhất ở mọi bước đã qua, nên m′ = cm, v′ = c²v, và m̂′/√v̂′ = m̂/√v̂ (ε = 0). Hai bản có cùng bước, nên cùng wₜ. Vậy quỹ đạo không phụ thuộc λ > 0. Đo bằng số: λ = 1, 0.01, 10⁻⁵ đều cho 0.257665. Kết luận: với `Adam(weight_decay=λ)`, tham số có gradient loss nhỏ bị co với tốc độ khoảng α mỗi bước, gần như bất kể λ; λ chỉ có tác dụng tương đối so với gradient loss. Muốn λ có nghĩa độc lập, dùng AdamW.
13. ∇f = (ab − 1)(b, a). Tại ab = 1, Hessian = (b, a)ᵀ(b, a) + (ab − 1)·(ma trận khác) = (b, a)ᵀ(b, a), hạng 1, trị riêng 0 và a² + b². Tại (1, 1): λₘₐₓ = 2. Tại (10, 0.1): λₘₐₓ = 100.01. Hai điểm cho cùng hàm số x ↦ abx = x, cùng loss 0, nhưng "độ nhọn" khác nhau 50 lần. Vậy λₘₐₓ không phải là tính chất của hàm số mô hình học được, mà của cách tham số hóa; mọi lập luận "phẳng ⇒ tổng quát hóa tốt" phải dùng một định nghĩa độ phẳng bất biến với các phép đổi tham số giữ nguyên hàm số, hoặc chấp nhận giới hạn này.
14. Đặt g = ∇f(xₖ), rₖ = ‖xₖ − x*‖. (a) Bổ đề giảm đủ: f(xₖ₊₁) ≤ f(xₖ) − ‖g‖²/(2L). Tính lồi: f(xₖ) − f* ≤ gᵀ(xₖ − x*). Khai triển rₖ₊₁² = rₖ² − (2/L)gᵀ(xₖ − x*) + ‖g‖²/L² ≤ rₖ² − (2/L)(f(xₖ) − f*) + ‖g‖²/L². Bổ đề giảm đủ cho ‖g‖²/L² ≤ (2/L)(f(xₖ) − f(xₖ₊₁)); thế vào: rₖ₊₁² ≤ rₖ² − (2/L)(f(xₖ₊₁) − f*). Cộng từ bước 0 tới n − 1: (2/L)∑ⱼ₌₁ⁿ(f(xⱼ) − f*) ≤ r₀² − rₙ² ≤ r₀². Vì f(xⱼ) không tăng, n(f(xₙ) − f*) ≤ ∑ⱼ(f(xⱼ) − f*), nên f(xₙ) − f* ≤ L r₀²/(2n). (b) Strongly convex cho gᵀ(xₖ − x*) ≥ f(xₖ) − f* + (μ/2)rₖ², nên rₖ₊₁² ≤ (1 − μ/L)rₖ² − (2/L)(f(xₖ) − f*) + ‖g‖²/L². L-smooth cho f* ≤ f(xₖ − g/L) ≤ f(xₖ) − ‖g‖²/(2L), tức f(xₖ) − f* ≥ ‖g‖²/(2L), nên −(2/L)(f(xₖ) − f*) + ‖g‖²/L² ≤ 0. Vậy rₖ₊₁² ≤ (1 − μ/L)rₖ². (c) L = 100, μ = 1, r₀² = 2, k = 100: cận (a) = 100·2/(2·100) = 1.0, giá trị thật f(x₁₀₀) = 0.067 (0.99¹⁰⁰ = 0.366, x₂ = 0 sau bước đầu). Cận (b) = 2·0.99¹⁰⁰ = 0.732, thật ‖x₁₀₀‖² = 0.134. Cả hai cận đúng nhưng lỏng: cận (b) coi mọi hàm μ-strongly convex, còn bài này có x₂ biến mất sau một bước. Điểm cần nhớ: (a) chỉ cần lồi và cho O(1/k), (b) cần thêm μ và cho tốc độ tuyến tính (1 − 1/κ)ᵏ.

**Bài tính nhanh**

1. ∇f = (2x − 2, 2y) = 0 ⇒ (1, 0), f = −1.
2. Không. x³ < 0 với x < 0. Đó là điểm uốn.
3. L = 12, μ = 3, κ = 4 (b không ảnh hưởng Hessian).
4. f′ = 2x, x₁ = 1 − 0.1 · 2 = 0.8.
5. ln(0.01)/ln(0.99) ≈ 458 bước.
6. η < 2/50 = 0.04.
7. GD: 99/101 ≈ 0.980. Heavy ball: 9/11 ≈ 0.818.
8. 0.1/(1 − 0.9) = 1.0.
9. 2/√64 = 0.25.
10. Bước đầu có độ lớn ≈ α và ngược dấu gradient: θ tăng 0.001.
11. 0.1/√(1 − 0.99) = 0.1/0.1 = 1, tức đúng α.
12. (3 000 − 500)/(5 500 − 500) = 0.5, nên η = ½ · 10⁻³ · (1 + cos(π/2)) = 5 × 10⁻⁴.
13. x − x³/(3x²) = 1 − 1/3 = 2/3. Lặp lại thì xₖ = (2/3)ᵏ: chỉ tuyến tính vì f″(0) = 0, khác ví dụ 4.12.
14. (10⁷)² × 4 byte = 4 × 10¹⁴ byte ≈ 3.7 × 10⁵ GiB (372 529 GiB).
15. ½ · 40 · 0.05² = 0.05.

**Câu hỏi sau lab**

- Lab 1: với η = 0.01, x₂ về 0 sau bước đầu, chỉ còn x₁ = 0.99ᵏ. Tiêu chí dừng so ‖x‖ với 10⁻⁶·‖x₀‖ = √2 × 10⁻⁶, chứ không phải 10⁻⁶. Giải 0.99ᵏ < √2 × 10⁻⁶: k > ln(√2 × 10⁻⁶)/ln(0.99) ≈ 1 340.1, nên k = 1 341, khớp đúng. Với η = 2/101, cả hai tọa độ có cùng độ lớn (99/101)ᵏ nên ‖xₖ‖/‖x₀‖ = (99/101)ᵏ, và công thức cho đúng 691.
- Lab 3: tốc độ thật của heavy ball là 9/11 nhưng ở tham số tối ưu ma trận lặp có trị riêng kép không chéo hóa được, nên sai số cỡ k·(9/11)ᵏ; thừa số k làm tốn thêm khoảng 24 bước (93 so với 69). Tỉ lệ 1/7.4 vẫn đúng bậc √κ. Không mâu thuẫn: định lý O(1/k²) là cận trường hợp xấu nhất cho cả lớp hàm lồi L-smooth với một lịch tham số cụ thể; nó không nói Nesterov thắng heavy ball trên mọi bài với mọi bộ (η, β) cố định.
- Lab 5: với d = 3, H là ma trận 3 × 3, giải hệ gần như miễn phí, và 6 bước bậc hai cho độ chính xác ‖∇f‖ < 10⁻¹⁰. Với d = 10⁶, H cần 3 725 GiB và giải hệ O(d³); khi đó L-BFGS (O(md) bộ nhớ) hoặc phương pháp bậc một là lựa chọn duy nhất khả thi.

**Mini-test (thang 30 điểm)**

1. ∇f = (2x, 4y³ − 4y) = 0 ⇒ x = 0, y ∈ {0, 1, −1} (1 điểm). Hessian diag(2, 12y² − 4): tại (0, 0) là diag(2, −4), yên ngựa; tại (0, ±1) là diag(2, 8), cực tiểu địa phương, f = −1 (2 điểm). f ≥ −1 vì y⁴ − 2y² = (y² − 1)² − 1 ≥ −1, nên −1 là giá trị nhỏ nhất toàn cục; f không lồi vì Hessian tại (0, 0) có trị riêng âm (1 điểm).
2. g₁ = x ≤ 0, g₂ = y ≤ 0. Dừng: 2(x − 1) + μ₁ = 0, 2(y − 2) + μ₂ = 0 (1 điểm). Cực tiểu tự do (1, 2) vi phạm cả hai, thử cả hai chạm: x = y = 0 ⇒ μ₁ = 2, μ₂ = 4, đều ≥ 0 (2 điểm). Kiểm khả thi và bù: thỏa (1 điểm). Cả hai ràng buộc chạm; μ₂ > μ₁ nghĩa là nới ràng buộc y có lợi hơn (1 điểm).
3. (a) L = 25, μ = 1: η = 2/26 ≈ 0.0769, tốc độ 24/26 = 12/13 ≈ 0.923, số bước ≈ ln(10⁻⁶)/ln(0.923) ≈ 173 (2 điểm). (b) √κ = 5, tốc độ (5 − 1)/(5 + 1) = 2/3, số bước ≈ ln(10⁻⁶)/ln(2/3) ≈ 34 (2 điểm). (c) Ở tham số tối ưu, ma trận lặp có trị riêng kép, sai số cỡ k·(2/3)ᵏ thay vì (2/3)ᵏ (1 điểm). Chạy thật: GD 173 bước, heavy ball 44.
4. Năm dòng: mₜ, vₜ, m̂ₜ = mₜ/(1 − β₁ᵗ), v̂ₜ = vₜ/(1 − β₂ᵗ), θₜ = θₜ₋₁ − α·m̂ₜ/(√v̂ₜ + ε) (2 điểm). Ở t = 1: m̂₁ = g, v̂₁ = g², nên bước α·g/(|g| + ε) ≈ α theo độ lớn (1.5 điểm). AdamW không cộng λθ vào gₜ, mà thêm −α·λ·θₜ₋₁ vào dòng cập nhật θ, ngoài phép chia cho √v̂ (1.5 điểm). Đây là quy ước PyTorch (lr·λ·θ); trong paper decay nhân hệ số lịch ηₜ chứ không nhân α.
5. (a) Thiếu chia cho B: g là tổng, lớn gấp B lần gradient trung bình (1.5 điểm). (b) Gradient tăng 8 lần nên learning rate hiệu dụng tăng 8 lần. Nếu lr·B·λₘₐₓ (của loss trung bình) vượt 2 thì phân kỳ: loss nổ hoặc NaN; nếu không vượt thì vẫn có thể dao động mạnh (2 điểm). (c) `g = X.T @ (X @ w - y) / len(y)`. Quy tắc scale tuyến tính gợi ý nhân lr lên 8 khi B tăng 8 lần (bản lỗi "vô tình" làm đúng việc đó), kèm warmup và kiểm trần ổn định 2/λₘₐₓ (1.5 điểm).
6. Thang chấm: (2 điểm) Newton, L-BFGS không khả thi: Hessian O(n²) bộ nhớ, nhiễu minibatch phá phương trình secant, Hessian không PD dẫn về yên ngựa. (2 điểm) AdamW: tiền điều kiện chéo xử lý chênh lệch thang gradient giữa các loại tham số (embedding, LayerNorm, attention), weight decay tách rời giữ λ có nghĩa. (2 điểm) Warmup vì v̂ ở đầu nhiễu và độ cong ban đầu lớn; lịch giảm vì η hằng chỉ đưa tới sàn nhiễu, giảm η để vào sâu.

**Săn bug — ba lỗi**

1. **Minibatch cố định.** `idx = perm[:B]` lấy cùng 32 mẫu cho mọi bước, và `perm` chỉ được tạo một lần. Mô hình chỉ học 32 mẫu. Riêng lỗi này đã cho ‖w − w*‖ ≈ 0.334.
2. **Lịch cosine thiếu chia T.** `np.cos(np.pi * t)` với t nguyên bằng ±1, nên lr nhảy giữa lr0 và 0 theo bước chẵn lẻ: một nửa số bước bị bỏ và lr không bao giờ giảm. Riêng lỗi này cho ‖w − w*‖ ≈ 0.035 (bản đúng 0.0019).
3. **Không dùng vận tốc.** v được tính nhưng cập nhật lại dùng g, tức thuật toán thật là SGD thuần: β bị vô hiệu. Lỗi này không làm hỏng kết quả trên bài dễ này: bản chỉ còn lỗi 3 cho ‖w − w*‖ = 0.0019 và f − f* = 1.76 × 10⁻⁶, bản đúng 0.0019 và 1.96 × 10⁻⁶ (hơi tệ hơn, nằm trong nhiễu). Vì vậy nó chỉ lộ ra qua kiểm tra độ nhạy theo β: đổi β từ 0 sang 0.5, 0.9 mà kết quả giống hệt nhau tới mọi chữ số; một siêu tham số không có tác dụng gì là dấu hiệu code không dùng nó.

```python
def train_fixed(epochs=20, B=32, lr0=0.05, beta=0.9):
    w, v = np.zeros(db), np.zeros(db)
    T, t = epochs * (nb // B), 0
    r = np.random.default_rng(1)
    for ep in range(epochs):
        perm = r.permutation(nb)                          # xáo trộn mỗi epoch
        for i in range(nb // B):
            idx = perm[i * B:(i + 1) * B]                 # sửa lỗi 1: chạy theo i
            Xi = Xb[idx]
            g = Xi.T @ (Xi @ w - yb[idx]) / B
            lr = 0.5 * lr0 * (1 + np.cos(np.pi * t / T)) # sửa lỗi 2: chia cho T
            v = beta * v + g
            w = w - lr * v                                # sửa lỗi 3: dùng vận tốc
            t += 1
    return w
```

Đã chạy kiểm: bản sửa đạt ‖w − w*‖ = 0.0019 và f − f* = 2.0 × 10⁻⁶.

## Tài liệu đào sâu và nguồn

Nội dung chương được viết lại, đối chiếu phạm vi kiến thức với các nguồn dưới đây. Đọc theo thứ tự ưu tiên:

| Tài liệu | Dùng để làm gì | Phần nên đọc |
| --- | --- | --- |
| [Why Momentum Really Works](https://distill.pub/2017/momentum/) (Goh, Distill, 2017) | Phân tích GD và momentum theo từng hướng riêng, có hình tương tác | Toàn bộ; đặc biệt phần ma trận 2 × 2 và tham số tối ưu |
| [d2l.ai, chương Optimization Algorithms](https://d2l.ai/chapter_optimization/index.html) | Trình bày có code cho lồi, SGD, minibatch, momentum, Adam, lịch learning rate | Các mục convexity, momentum, adam, lr-scheduler |
| [CS231n — Neural Networks Part 3](https://cs231n.github.io/neural-networks-3/) | Góc nhìn thực hành: Nesterov, Adagrad, RMSProp, Adam, annealing, vì sao không dùng bậc hai | Mục parameter updates và second order methods |
| [Boyd và Vandenberghe, Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) (PDF miễn phí) | Tài liệu chuẩn về lồi, KKT, GD, Newton | Chương 3 (hàm lồi), mục 5.5 (KKT), 9.3 (GD), 9.5 (Newton) |
| [Mathematics for Machine Learning](https://mml-book.github.io) (Deisenroth, Faisal, Ong) | Cầu nối toán và ML ở mức nhập môn | Chương 7 Continuous Optimization |
| [CS229 Main Notes](https://cs229.stanford.edu/main_notes.pdf) (Stanford) | Batch và stochastic GD, Newton cho logistic, Lagrange duality cho SVM | Mục 1.1, 2.4, 6.5–6.6 [cần kiểm chứng số mục, chưa mở được PDF] |
| [Kingma và Ba, Adam](https://arxiv.org/abs/1412.6980) | Paper gốc của Adam | Algorithm 1 và phần bias correction |
| [Loshchilov và Hutter, Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | Paper gốc của AdamW | Phần so sánh L2 và weight decay |
| [Cohen và cộng sự, Edge of Stability](https://arxiv.org/abs/2103.00065) | Thực tế khác lý thuyết của trần 2/L | Phần mở đầu và các hình progressive sharpening |
| [Goyal và cộng sự, Large Minibatch SGD](https://arxiv.org/abs/1706.02677) | Quy tắc scale tuyến tính và warmup | Mục 2 |

Các paper khác được trích trong chương: [SGDR](https://arxiv.org/abs/1608.03983), [RAdam](https://arxiv.org/abs/1908.03265), [Reddi và cộng sự](https://arxiv.org/abs/1904.09237), [Sutskever và cộng sự 2013](https://proceedings.mlr.press/v28/sutskever13.html), [Dauphin và cộng sự 2014](https://arxiv.org/abs/1406.2572), [Keskar và cộng sự 2016](https://arxiv.org/abs/1609.04836), [SAM](https://arxiv.org/abs/2010.01412), [Smith và Le 2017](https://arxiv.org/abs/1710.06451), [Dinh và cộng sự 2017](https://arxiv.org/abs/1703.04933), [Lessard, Recht và Packard 2016](https://arxiv.org/abs/1408.3595), [Du và cộng sự 2017](https://arxiv.org/abs/1705.10412). Tài liệu API: [PyTorch AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html), [PyTorch Adam](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html), [SciPy L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html).

**Từ khóa để tự tìm thêm:** Polyak–Łojasiewicz condition, Nesterov accelerated gradient, AMSGrad, Lion optimizer, Sophia optimizer, Muon [cần kiểm chứng mức dùng trong ngành], warmup–stable–decay (WSD) schedule, schedule-free optimizer, K-FAC, Shampoo, gradient clipping và loss spikes khi huấn luyện LLM, μP (hyperparameter transfer), implicit regularization of SGD, sharp minima can generalize.

**Trước khi sang chương 5:** đạt từ 24/30 điểm mini-test, làm lại các bài ★★★ sai, và tự viết được lab 4 (Adam và AdamW) mà không nhìn code mẫu. Chương 5 dùng ngay KKT (SVM) và logistic regression với L-BFGS.

Nguồn tra cứu ngày 24/10/2026.
