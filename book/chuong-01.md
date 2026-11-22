# Chương 1 — Đại số tuyến tính cho ML

## Mục tiêu và kiểm tra nhanh

Chương này dạy bạn nhìn ma trận như một phép biến đổi, không phải một bảng số. Mọi chương sau (backprop, PCA, attention, LoRA) đều dựa vào cách nhìn đó. Thời lượng: khoảng 2 tuần, 8 giờ mỗi tuần.

**Sau chương này, bạn phải làm được:**

- ☐ Giải thích Ax theo hàng và theo cột, và đọc AB theo đủ bốn cách.
- ☐ Tìm hạng và không gian null của một ma trận, nói được hệ Ax = b có bao nhiêu nghiệm.
- ☐ Suy ra nghiệm bình phương tối thiểu bằng hình học và bằng gradient; giải thích vì sao thư viện dùng QR thay vì phương trình chuẩn.
- ☐ Phân rã một ma trận đối xứng, kiểm tra nó có xác định dương không, và nói ý nghĩa hình học của det A.
- ☐ Tính SVD bằng tay cho ma trận 2×2, giải thích vì sao PCA chính là SVD của dữ liệu đã trừ trung bình.

**Kiểm tra nhanh (5 phút, không dùng máy tính).** Câu i kiểm tra mục 1.i: đúng thì đọc lướt mục đó, sai thì đọc kỹ. Đúng cả 5 câu thì làm thẳng bài ★★★ và mini-test.

1. A, B ∈ ℝ¹⁰⁰ˣ¹⁰⁰ và x ∈ ℝ¹⁰⁰. Tính (AB)x hay A(Bx) rẻ hơn, rẻ hơn khoảng bao nhiêu lần?
2. A ∈ ℝ³ˣ⁵. Phương trình Ax = 0 có luôn có nghiệm khác 0 không? Vì sao?
3. Trong bài toán bình phương tối thiểu, vì sao phần dư b − Ax̂ vuông góc với mọi cột của A?
4. u ∈ ℝⁿ là vector đơn vị. Ma trận P = uuᵀ có những trị riêng nào?
5. Q ∈ ℝⁿˣⁿ là ma trận trực giao. Các giá trị kỳ dị của Q bằng bao nhiêu?

Đáp án ở mục Lời giải cuối chương.

## 1.1 Ma trận là phép biến đổi tuyến tính

Một ma trận A ∈ ℝᵐˣⁿ là một hàm biến vector trong ℝⁿ thành vector trong ℝᵐ. Cột aⱼ của A cho biết vector đơn vị eⱼ bị đưa tới đâu: Aeⱼ = aⱼ. Biết ảnh của các vector đơn vị là biết toàn bộ phép biến đổi.

% manim: BienDangLuoi

**Hai cách đọc Ax.** Theo hàng: mỗi phần tử của kết quả là một tích vô hướng, đo độ "giống nhau" giữa một hàng của A và x. Theo cột: kết quả là tổ hợp tuyến tính các cột của A, với hệ số lấy từ x.

$$
(Ax)_i = r_i^{\top} x \quad\text{(theo hàng)} \qquad\qquad Ax = x_1 a_1 + x_2 a_2 + \cdots + x_n a_n \quad\text{(theo cột)}
$$

Cách đọc theo hàng giải thích một lớp fully-connected y = Wx: mỗi neuron đầu ra là một hàng của W, "so khớp" với đầu vào. Cách đọc theo cột giải thích vì sao Ax = b có nghiệm khi và chỉ khi b nằm trong không gian sinh bởi các cột (mục 1.2).

**Bốn cách đọc C = AB**, với A ∈ ℝᵐˣⁿ và B ∈ ℝⁿˣᵖ. AB là phép hợp: áp B trước, rồi A.

| Cách đọc | Công thức | Dùng khi nào |
| --- | --- | --- |
| Từng phần tử | Cᵢⱼ = rᵢᵀbⱼ (hàng i của A nhân cột j của B) | Tính tay, đếm chi phí |
| Theo cột | Cột j của C = Abⱼ | Mỗi cột của C là A áp lên một cột của B (một mẫu, một vector) |
| Theo hàng | Hàng i của C = rᵢᵀB | Mỗi hàng của C là một hàng của A đi qua B (batch dữ liệu XW) |
| Tổng tích ngoài | C = tổng theo k của aₖ nhân hàng k của B | Phân tích hạng, LoRA, SVD |

$$
AB = \sum_{k=1}^{n} a_k\, \tilde{b}_k^{\top}, \qquad a_k = \text{cột } k \text{ của } A,\;\; \tilde{b}_k^{\top} = \text{hàng } k \text{ của } B
$$

Mỗi số hạng aₖb̃ₖᵀ là một ma trận hạng 1, nên rank(AB) ≤ n. LoRA dùng đúng cách đọc này: cập nhật ΔW = BA, trong đó B có r cột nên ΔW là tổng của r ma trận hạng 1. (A, B ở đây là ký hiệu của paper LoRA, không phải A, B của mục này.)

**Ví dụ giải mẫu 1.1 — thứ tự phép nhân và chi phí.** A, B ∈ ℝ¹⁰⁰⁰ˣ¹⁰⁰⁰ và x ∈ ℝ¹⁰⁰⁰. So sánh (AB)x với A(Bx).

- (AB)x: tính AB mất 1000³ = 10⁹ phép nhân, nhân với x thêm 10⁶. Tổng khoảng 10⁹.
- A(Bx): Bx mất 10⁶, rồi A(Bx) thêm 10⁶. Tổng 2 × 10⁶, nhanh hơn khoảng 500 lần.
- Hai cách cho cùng kết quả (tính kết hợp) nhưng chi phí khác xa nhau. Backprop cùng ý tưởng nhưng nhân từ phía bên kia: với loss vô hướng, nó nhân một vector hàng với các Jacobian từ đầu ra ngược về, vᵀ(J₃J₂J₁) = ((vᵀJ₃)J₂)J₁ (vector–Jacobian product), và không bao giờ dựng cả ma trận Jacobian. Cách nhân A(Bx) ở trên, vector đứng bên phải, tương ứng chế độ forward (Jacobian–vector product).

**Ví dụ giải mẫu 1.2 — AB ≠ BA.** Lấy S = diag(2, 1) (kéo giãn trục thứ nhất) và R là phép quay 90°, có cột (0, 1) và (−1, 0). Áp vào e₁: SRe₁ = S(0, 1) = (0, 1), còn RSe₁ = R(2, 0) = (0, 2). Hai kết quả khác nhau, nên thứ tự các lớp trong mạng là quan trọng.

% manim: ABKhacBA

**Vì sao ML cần cái này.**

- **Batch:** dữ liệu một batch là ma trận X ∈ ℝᴺˣᵈ, mỗi hàng một mẫu. Tích XW với W ∈ ℝᵈˣᵏ áp cùng một phép biến đổi cho cả N mẫu một lúc (cách đọc theo hàng), và đó là lý do GPU làm việc theo ma trận.
- **Attention:** QKᵀ ∈ ℝᴺˣᴺ chứa mọi tích vô hướng giữa query và key, tức cách đọc "từng phần tử" áp lên toàn bộ chuỗi.

## 1.2 Hạng và bốn không gian con

Hạng r của A là số chiều thật sự của ảnh: số cột độc lập tuyến tính, cũng bằng số hàng độc lập. Mọi câu hỏi "Ax = b có nghiệm không, có bao nhiêu nghiệm" đều trả lời được bằng r, m và n.

**Bốn không gian con của A ∈ ℝᵐˣⁿ có hạng r** — bức tranh trung tâm của MIT 18.06:

| Không gian | Nằm trong | Số chiều | Ý nghĩa |
| --- | --- | --- | --- |
| Không gian cột C(A) | ℝᵐ | r | Mọi b mà Ax = b có nghiệm |
| Không gian null N(A) | ℝⁿ | n − r | Mọi hướng x mà A "xóa mất": Ax = 0 |
| Không gian hàng C(Aᵀ) | ℝⁿ | r | Phần của x mà A thật sự nhìn thấy |
| Null trái N(Aᵀ) | ℝᵐ | m − r | Phần của b không bao giờ đạt được |

Hai cặp vuông góc nhau: C(Aᵀ) ⊥ N(A) trong ℝⁿ, và C(A) ⊥ N(Aᵀ) trong ℝᵐ. Hệ quả là định lý hạng–số chiều null: dim N(A) = n − r.

**Bốn trường hợp của Ax = b**, theo việc r có bằng m (số hàng) và n (số cột) hay không:

| | r = n (cột độc lập) | r < n |
| --- | --- | --- |
| **r = m** | Chỉ xảy ra khi m = n: A khả nghịch, đúng một nghiệm x = A⁻¹b | Luôn có nghiệm với mọi b, và có vô số (N(A) khác {0}) |
| **r < m** | Có nghiệm khi và chỉ khi b ∈ C(A), và khi đó nghiệm duy nhất (tối đa một nghiệm) | Nếu b ∈ C(A) thì vô số nghiệm, nếu không thì vô nghiệm |

Nghiệm tổng quát (khi có) = một nghiệm riêng + một vector bất kỳ trong N(A). Khi r < m có những b vô nghiệm (b có thành phần nằm trong N(Aᵀ)); lúc đó ta giải bình phương tối thiểu (mục 1.3).

Với A vuông, các mệnh đề sau tương đương: A khả nghịch ⇔ r = n ⇔ N(A) = {0} ⇔ det A ≠ 0 ⇔ 0 không phải trị riêng của A.

**Ví dụ giải mẫu 1.3.** A ∈ ℝ²ˣ³ có hai hàng (1, 2, 3) và (2, 4, 6).

- Hàng 2 = 2 × hàng 1, nên r = 1. C(A) là đường thẳng sinh bởi (1, 2) trong ℝ².
- dim N(A) = 3 − 1 = 2. Giải x₁ + 2x₂ + 3x₃ = 0: một cơ sở là (−2, 1, 0) và (−3, 0, 1).
- Với b = (1, 2): vô số nghiệm, ví dụ (1, 0, 0) + N(A). Với b = (1, 3): vô nghiệm vì (1, 3) không nằm trên đường sinh bởi (1, 2).

**Vì sao ML cần cái này.**

- **Mô hình quá tham số (n tham số > m mẫu):** hệ nội suy dữ liệu có vô số nghiệm, nên câu hỏi thật là thuật toán chọn nghiệm nào. Xét hồi quy tuyến tính với hệ Ax = b có nghiệm (chẳng hạn A có hạng m) và gradient descent khởi tạo từ 0 với bước đủ nhỏ để hội tụ (η < 2/λₘₐₓ(AᵀA)). Mọi bước cập nhật Aᵀ(Ax − b) đều nằm trong C(Aᵀ), nên điểm hội tụ cũng nằm trong C(Aᵀ). Trong tập nghiệm x₀ + N(A) chỉ có đúng một điểm thuộc C(Aᵀ) (vì C(Aᵀ) ⊥ N(A)), và đó là nghiệm có ‖x‖₂ nhỏ nhất. Đây là ví dụ đơn giản nhất của "implicit bias"; với mạng phi tuyến, kết luận này không còn đúng nguyên dạng.
- **Đặc trưng cộng tuyến:** nếu hai cột của X gần như tỉ lệ, XᵀX gần suy biến và trọng số nhảy lung tung. Regularization (mục 1.3) là cách chữa.
- **Nút cổ chai:** rank(AB) ≤ min(rank A, rank B). Với mạng tuyến tính (không có hàm kích hoạt), một lớp ẩn 64 chiều giữa hai lớp 1024 chiều ép toàn bộ phép biến đổi xuống hạng tối đa 64. Có hàm kích hoạt thì phép biến đổi không còn tuyến tính, và phát biểu đúng là cho Jacobian tại từng điểm: hạng Jacobian ≤ 64. Autoencoder và LoRA dựa vào đúng điều này.

## 1.3 Chuẩn, phép chiếu, QR, bình phương tối thiểu

Bình phương tối thiểu chỉ là một phép chiếu: khi b ∉ C(A), ta lấy điểm gần b nhất trong C(A). Nắm hình học này thì công thức tự nhớ được; nắm QR thì biết thư viện thật sự giải nó thế nào.

**Tích vô hướng và chuẩn.** xᵀy = ‖x‖₂‖y‖₂ cos θ. Cosine similarity giữa hai embedding chính là tích vô hướng sau khi chuẩn hóa. Bất đẳng thức Cauchy–Schwarz |xᵀy| ≤ ‖x‖₂‖y‖₂ đảm bảo cos θ ∈ \[−1, 1\].

| Chuẩn | Định nghĩa | Dùng ở đâu trong ML |
| --- | --- | --- |
| ‖x‖₂ | √(x₁² + … + xₙ²) | Weight decay, khoảng cách Euclid |
| ‖x‖₁ | \|x₁\| + … + \|xₙ\| | Lasso, tạo trọng số thưa |
| ‖x‖∞ | maxᵢ \|xᵢ\| | Tấn công đối kháng giới hạn theo từng pixel |
| Chuẩn Frobenius của A | √(tổng mọi Aᵢⱼ²) = √tr(AᵀA) | Sai số tái tạo, xấp xỉ hạng thấp |

Vì sao L1 cho nghiệm thưa còn L2 thì không: quả cầu L1 có các đỉnh nằm trên trục tọa độ. Đường đồng mức của hàm mất mát có nhiều khả năng chạm quả cầu ở đỉnh hoặc cạnh (nơi một số tọa độ bằng 0), và càng rõ khi số chiều cao.

**Ma trận trực giao.** Q ∈ ℝⁿˣⁿ trực giao khi QᵀQ = I, tức Q⁻¹ = Qᵀ. Nhân với Q không đổi độ dài: ‖Qx‖₂ = ‖x‖₂. Vì vậy thuật toán dựa trên Q không khuếch đại sai số làm tròn. Ở phân rã QR rút gọn bên dưới, Q ∈ ℝᵐˣⁿ chỉ có QᵀQ = I; QQᵀ ≠ I khi m > n.

**Phép chiếu.** Chiếu b lên đường thẳng theo hướng a, rồi tổng quát lên C(A) khi A có các cột độc lập:

$$
\operatorname{proj}_a(b) = \frac{a^{\top} b}{a^{\top} a}\, a \qquad\qquad P = A\,(A^{\top} A)^{-1} A^{\top}, \qquad P^2 = P,\;\; P^{\top} = P
$$

% manim: PhepChieu

**Gram–Schmidt và phân rã QR.** Từ các cột độc lập a₁, …, aₙ, lần lượt trừ đi phần chiếu lên các hướng đã có rồi chuẩn hóa. Kết quả là A = QR, với Q có các cột trực chuẩn và R tam giác trên, Rᵢⱼ = qᵢᵀaⱼ.

$$
v_j = a_j - \sum_{i<j} (q_i^{\top} a_j)\, q_i, \qquad q_j = \frac{v_j}{\lVert v_j \rVert_2}, \qquad A = QR
$$

Gram–Schmidt là cách hiểu QR, nhưng thư viện không tính như vậy: `np.linalg.qr` gọi LAPACK `geqrf`, dùng các phép phản xạ Householder, vì Gram–Schmidt cổ điển mất tính trực giao khi các cột gần phụ thuộc (κ lớn). Biến thể modified Gram–Schmidt ổn hơn nhưng vẫn kém Householder.

**Ví dụ giải mẫu 1.4 — Gram–Schmidt.** a₁ = (1, 1, 0), a₂ = (1, 0, 1).

- q₁ = (1, 1, 0)/√2.
- q₁ᵀa₂ = 1/√2, nên v₂ = a₂ − (1/√2)q₁ = (1, 0, 1) − (½, ½, 0) = (½, −½, 1).
- ‖v₂‖₂ = √(3/2), nên q₂ = (1, −1, 2)/√6. Kiểm tra: q₁ᵀq₂ = (1 − 1 + 0)/√12 = 0.
- R có hàng (√2, 1/√2) và (0, √(3/2)). Đã kiểm bằng `np.linalg.qr`: khớp tới dấu. NumPy dùng phản xạ Householder nên có thể đổi dấu từng cặp (qᵢ, hàng i của R); ở ví dụ này q₁ và hàng 1 của R trái dấu so với tính tay (R₁₁ = −1.414), tích QR vẫn bằng A.

**Bình phương tối thiểu, hai cách suy ra.** Bài toán: tìm x̂ nhỏ nhất hóa ‖Ax − b‖₂².

1. *Hình học:* phần dư b − Ax̂ phải vuông góc với mọi cột của A, tức Aᵀ(b − Ax̂) = 0.
2. *Giải tích:* gradient của ‖Ax − b‖₂² là 2AᵀAx − 2Aᵀb. Cho bằng 0 ra cùng phương trình.

$$
A^{\top} A\, \hat{x} = A^{\top} b \quad\text{(phương trình chuẩn)} \qquad\qquad A = QR \;\Rightarrow\; R\, \hat{x} = Q^{\top} b \qquad\qquad \hat{x}_{\text{ridge}} = (A^{\top} A + \lambda I)^{-1} A^{\top} b
$$

Với A = QR (QR rút gọn, A có các cột độc lập nên R khả nghịch) thì AᵀA = RᵀR và Aᵀb = RᵀQᵀb. Phương trình chuẩn thành RᵀRx̂ = RᵀQᵀb; nhân hai vế với (Rᵀ)⁻¹ được Rx̂ = Qᵀb, giải bằng thế ngược vì R tam giác trên. Cách này không bao giờ tạo AᵀA.

**Ví dụ giải mẫu 1.5 — khớp đường thẳng.** Ba điểm (t, y) = (0, 1), (1, 2), (2, 2), mô hình y = w₀ + w₁t.

- A có các hàng (1, 0), (1, 1), (1, 2); b = (1, 2, 2).
- AᵀA có hàng (3, 3) và (3, 5); Aᵀb = (5, 6); det(AᵀA) = 15 − 9 = 6.
- Giải: w₀ = (5·5 − 3·6)/6 = 7/6, w₁ = (−3·5 + 3·6)/6 = 1/2.
- Kiểm tra: phần dư r = (−1/6, 1/3, −1/6). Tổng các rᵢ = 0 (vuông góc cột 1) và 0·(−1/6) + 1·(1/3) + 2·(−1/6) = 0 (vuông góc cột 2), đúng như hình học dự đoán.

**Vì sao ML cần cái này.**

- **Bình phương tối thiểu ở khắp nơi:** hồi quy tuyến tính, lớp cuối của nhiều mô hình và việc "linear probe" một embedding đều là bình phương tối thiểu.
- **Ridge chữa cộng tuyến:** AᵀA + λI luôn khả nghịch với λ > 0, vì mọi trị riêng của nó ít nhất bằng λ (mục 1.4).

**Thực tế tính toán.** Đừng giải phương trình chuẩn và đừng tính nghịch đảo. Tạo AᵀA làm số điều kiện bị bình phương: κ(AᵀA) = κ(A)². Lab 2 cho thấy với ma trận Vandermonde bậc 12, κ(A) ≈ 6.8 × 10⁸, nên κ(AᵀA) = κ(A)² ≈ 4.6 × 10¹⁷. Con số này lớn hơn 1/ε ≈ 4.5 × 10¹⁵ của float64 (ε ≈ 2.2 × 10⁻¹⁶ là machine epsilon), nên AᵀA suy biến về mặt số học. Tính số trực tiếp κ(AᵀA) cho khoảng 8.5 × 10¹⁷, nhưng chính con số đó cũng không đáng tin vì đã vượt 1/ε. Phần dư của phương trình chuẩn chỉ 4.1 × 10⁻⁷, trông ổn, nhưng nghiệm sai tới 610% (sai số tương đối 6.1); QR sai 1.3 × 10⁻⁹. Bài học: phần dư nhỏ không có nghĩa là nghiệm đúng. Dùng `np.linalg.lstsq` (dựa trên SVD) hoặc QR.

## 1.4 Định thức, vết, trị riêng, ma trận đối xứng

Vector riêng là hướng mà A chỉ kéo giãn, không xoay: Av = λv. Với ma trận đối xứng, các hướng này vuông góc với nhau, nên A chỉ là "xoay về hệ trục riêng, kéo giãn từng trục, xoay lại". Định thức và vết là hai con số tóm tắt các trị riêng đó.

**Định thức — hệ số thay đổi thể tích.** |det A| là tỉ lệ A phóng to hay thu nhỏ thể tích: hình vuông đơn vị biến thành hình bình hành có diện tích |det A|. Dấu của det A cho biết A có lật hướng không gian hay không. det A = 0 nghĩa là A ép không gian xuống ít chiều hơn, nên không khả nghịch. Các tính chất phải thuộc: det(AB) = det A · det B, det Aᵀ = det A, det A⁻¹ = 1/det A, và det A = λ₁λ₂⋯λₙ.

**Vết.** tr A = A₁₁ + A₂₂ + ⋯ + Aₙₙ = λ₁ + λ₂ + ⋯ + λₙ. Tính chất hay dùng nhất là tính hoán vị vòng: tr(AB) = tr(BA) khi cả hai tích đều xác định, tr(ABC) = tr(BCA) = tr(CAB). Hệ quả hữu ích: bình phương chuẩn Frobenius của A bằng tr(AᵀA), và xᵀAx = tr(Axxᵀ).

**Trị riêng.** Là nghiệm của det(A − λI) = 0. Ma trận không đối xứng có thể có trị riêng phức (phép quay 90° không giữ hướng nào) hoặc không chéo hóa được.

**Định lý phổ.** A ∈ ℝⁿˣⁿ đối xứng ⇒ mọi trị riêng là số thực, và có một cơ sở trực chuẩn gồm các vector riêng:

$$
A = Q \Lambda Q^{\top} = \sum_{i=1}^{n} \lambda_i\, q_i q_i^{\top}, \qquad Q^{\top} Q = I, \qquad \Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_n)
$$

**Dạng toàn phương và xác định dương.** Đặt y = Qᵀx (tọa độ của x theo hệ trục riêng):

$$
x^{\top} A x = \sum_{i=1}^{n} \lambda_i\, y_i^{2}
$$

Nên A nửa xác định dương (PSD) ⇔ mọi λᵢ ≥ 0; xác định dương (PD) ⇔ mọi λᵢ > 0. Ma trận Gram XᵀX luôn PSD vì xᵀXᵀXx = ‖Xx‖₂² ≥ 0. Chú ý: các định nghĩa này dành cho ma trận đối xứng.

**Thương Rayleigh.** Giá trị lớn nhất của xᵀAx khi ‖x‖₂ = 1 bằng λₘₐₓ, đạt tại vector riêng tương ứng. Đây là chìa khóa của PCA (mục 1.5) và của power iteration (lab 1).

**Ví dụ giải mẫu 1.6.** A có hàng (2, 1) và (1, 2).

- det(A − λI) = (2 − λ)² − 1 = 0 ⇒ λ = 3 hoặc λ = 1. Kiểm tra: λ₁ + λ₂ = 4 = tr A, λ₁λ₂ = 3 = det A.
- λ = 3 có vector riêng (1, 1)/√2; λ = 1 có vector riêng (1, −1)/√2. Hai vector vuông góc, đúng như định lý phổ.
- Hình vuông đơn vị biến thành hình bình hành diện tích 3 = det A.
- Cả hai trị riêng dương nên A xác định dương. Đường xᵀAx = 1 là một elip có bán trục 1/√3 theo hướng (1, 1) và 1 theo hướng (1, −1).

% manim: VectorRieng

**Vì sao ML cần cái này.**

- **Hessian và tốc độ hội tụ:** gần cực tiểu, hàm mất mát xấp xỉ dạng toàn phương với ma trận Hessian H. Gradient descent ổn định khi learning rate η < 2/λₘₐₓ(H), và số bước cần thiết tỉ lệ với κ = λₘₐₓ/λₘᵢₙ (cỡ κ · log(1/sai số), với η tối ưu). Đây là lý do cần chuẩn hóa đầu vào, normalization và optimizer thích nghi.
- **Định thức trong mô hình xác suất:** log det Σ xuất hiện trong log-likelihood của phân phối Gauss; normalizing flow cần log|det J| của Jacobian để đổi biến mật độ.
- **Ma trận hiệp phương sai và ma trận kernel** luôn PSD. Một "kernel" cho ra ma trận không PSD thì không phải kernel hợp lệ.
- **Lũy thừa ma trận:** Aᵏ = QΛᵏQᵀ. Nhân lặp cùng một ma trận khiến thành phần có |λ| > 1 bùng nổ, |λ| < 1 tiêu biến — trực giác cho exploding/vanishing gradient trong RNN. Ma trận trọng số thật thường không đối xứng, nên đây là trực giác, không phải chứng minh.

% manim: GDSoDieuKien

## 1.5 SVD, chuẩn phổ, xấp xỉ hạng thấp, PCA

Mọi ma trận, kể cả không vuông, đều là "xoay – kéo giãn – xoay". Đó là SVD, công cụ quan trọng nhất của chương: PCA, nén mô hình, LoRA và giả nghịch đảo đều là SVD.

$$
A = U \Sigma V^{\top} = \sum_{i=1}^{r} \sigma_i\, u_i v_i^{\top}, \qquad \sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r > 0
$$

U ∈ ℝᵐˣᵐ và V ∈ ℝⁿˣⁿ trực giao; Σ chỉ có các σᵢ trên đường chéo. Hình học: Vᵀ xoay đầu vào (có thể kèm phản xạ, nếu det V = −1), Σ kéo giãn từng trục, U xoay sang không gian đích (cũng có thể kèm phản xạ). Với A vuông khả nghịch, hình tròn đơn vị biến thành một elip có các bán trục σᵢ.

**Chuẩn phổ.** ‖A‖₂ là mức kéo giãn lớn nhất mà A gây ra cho một vector, và nó bằng σ₁. Số điều kiện κ(A) = σ₁/σₘᵢₙ là tỉ số giữa mức kéo giãn mạnh nhất và yếu nhất, với σₘᵢₙ là giá trị kỳ dị thứ min(m, n); κ lớn nghĩa là bài toán nhạy với sai số. Nếu A thiếu hạng thì σₘᵢₙ = 0 và κ(A) = ∞ (đây cũng là điều `np.linalg.cond` tính). Với A có các cột độc lập, σₘᵢₙ = σₙ. Tỉ số σ₁/σᵣ chỉ là "số điều kiện hiệu dụng" trên C(Aᵀ), không phải κ.

$$
\lVert A \rVert_2 = \max_{x \neq 0} \frac{\lVert A x \rVert_2}{\lVert x \rVert_2} = \sigma_1, \qquad \kappa(A) = \frac{\sigma_1}{\sigma_{\min(m,n)}}
$$

**Liên hệ với trị riêng.** AᵀA = VΣᵀΣVᵀ, nên σᵢ² là trị riêng của AᵀA, các cột của V là vector riêng của AᵀA, các cột của U là vector riêng của AAᵀ. Đây là cách tính SVD bằng tay. Lưu ý dấu: nếu tính V và U riêng rẽ từ hai ma trận AᵀA và AAᵀ thì dấu từng cặp (uᵢ, vᵢ) có thể lệch nhau; hãy tính uᵢ = Avᵢ/σᵢ như ví dụ 1.7 bên dưới.

**Định lý Eckart–Young.** Cắt SVD sau k số hạng cho xấp xỉ hạng k tốt nhất, theo cả chuẩn Frobenius lẫn chuẩn phổ:

$$
A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^{\top}, \qquad \lVert A - A_k \rVert_F^2 = \sum_{i>k} \sigma_i^2, \qquad \lVert A - A_k \rVert_2 = \sigma_{k+1}
$$

**Ví dụ giải mẫu 1.7 — SVD bằng tay.** A có hàng (3, 0) và (4, 5).

1. AᵀA có hàng (25, 20) và (20, 25), trị riêng 45 và 5 ⇒ σ₁ = 3√5, σ₂ = √5. Kiểm tra: σ₁σ₂ = 15 = |det A|.
2. Vector riêng của AᵀA: v₁ = (1, 1)/√2, v₂ = (1, −1)/√2.
3. uᵢ = Avᵢ/σᵢ: u₁ = (1, 3)/√10, u₂ = (3, −1)/√10. Hai vector vuông góc vì 1·3 + 3·(−1) = 0.
4. Xấp xỉ hạng 1: A₁ = σ₁u₁v₁ᵀ có hàng (1.5, 1.5) và (4.5, 4.5).
5. Sai số: A − A₁ có hàng (1.5, −1.5) và (−0.5, 0.5); tổng bình phương 2.25 + 2.25 + 0.25 + 0.25 = 5 = σ₂². Chuẩn phổ của A − A₁ bằng σ₂ = √5. Đúng như Eckart–Young.

% manim: SVDXoayKeoXoay

**PCA chính là SVD.** Cho dữ liệu X ∈ ℝᴺˣᵈ, mỗi hàng một mẫu.

1. Trừ trung bình từng cột: X̃ = X − 1μᵀ, với μ là vector trung bình các cột và 1 là vector toàn số 1. Bỏ bước này là lỗi kinh điển (xem bài săn bug).
2. Ma trận hiệp phương sai C = X̃ᵀX̃/(N − 1). Hướng có phương sai lớn nhất là nghiệm của max wᵀCw với ‖w‖₂ = 1, tức vector riêng lớn nhất của C (thương Rayleigh, mục 1.4).
3. Nếu X̃ = UΣVᵀ thì C = VΣ²Vᵀ/(N − 1). Vậy các thành phần chính là các cột của V, và phương sai giải thích bởi thành phần i là σᵢ²/(N − 1).
4. Chiếu xuống k chiều: Z = X̃Vₖ. Bình phương sai số tái tạo (theo chuẩn Frobenius) bằng tổng σᵢ² với i > k; lab 3 kiểm chứng điều này bằng số.

% manim: PCAXoayTruc

Mặc định nên tính PCA bằng SVD của X̃ chứ không tạo C, vì cùng lý do số điều kiện như ở mục 1.3. Có một ngoại lệ hợp lý: khi N ≫ d và d nhỏ, ma trận C (d × d) rẻ hơn nhiều, và scikit-learn từ phiên bản 1.5 tự chọn cách này (`svd_solver="auto"` dùng `covariance_eigh` khi số đặc trưng ≤ 1000 và số mẫu ≥ 10 lần số đặc trưng; đã đọc mã nguồn bản 1.9.1 ngày 04/11/2026). Đổi lại số điều kiện bị bình phương, nên kém ổn định khi phổ σᵢ trải rộng.

**Vì sao ML cần cái này.**

- **LoRA** ([Hu và cộng sự, 2021](https://arxiv.org/abs/2106.09685)): giả định phần cập nhật khi finetune có hạng nội tại thấp, nên chỉ học ΔW = BA với r nhỏ. Bài tập 13 tính số tham số tiết kiệm được.
- **Nén và tăng tốc:** thay một lớp W bằng tích hai ma trận mỏng UₖΣₖ và Vₖᵀ. Lab 4 cho thấy giới hạn: ma trận có cấu trúc hạng 2 giữ 99% năng lượng chỉ với 2 thành phần, còn một ảnh thật 512 × 512 cần 21 thành phần, và hạng 50 (19.6% số liệu) vẫn sai 6.4%. Phổ σᵢ giảm chậm là chuyện bình thường với dữ liệu thật.
- **Giả nghịch đảo** A⁺ = VΣ⁺Uᵀ cho nghiệm bình phương tối thiểu có ‖x‖₂ nhỏ nhất, kể cả khi A không đủ hạng. Ở đây Σ⁺ là chuyển vị của Σ với mỗi σᵢ ≠ 0 thay bằng 1/σᵢ.

## Bài tập giấy bút

14 bài, làm theo thứ tự. Mục tiêu: hết ★ trong 1 buổi, hết ★★ trong 2 buổi, ★★★ dành cho Thứ Bảy. Bài nào sai ghi vào sổ lỗi kèm lý do sai.

**★ Hiểu định nghĩa**

1. A ∈ ℝ⁴ˣ⁶ có hạng 3. Cho số chiều của C(A), N(A), C(Aᵀ), N(Aᵀ).
2. u = (1, 2, 3), v = (1, −1). Viết ma trận uvᵀ và cho biết hạng của nó.
3. Chứng minh ‖Qx‖₂ = ‖x‖₂ với mọi ma trận trực giao Q.
4. Tìm trị riêng và vector riêng của ma trận có hàng (4, 1) và (2, 3). Đối chiếu tổng và tích các trị riêng với tr A và det A.

**★★ Mức đề thi đầu vào**

5. Chứng minh rank(AᵀA) = rank(A). *Gợi ý: chứng minh N(AᵀA) = N(A).*
6. Cho A có các cột độc lập tuyến tính và P = A(AᵀA)⁻¹Aᵀ. Chứng minh P² = P, Pᵀ = P, và trị riêng của P chỉ có thể là 0 hoặc 1.
7. Chứng minh AᵀA + λI khả nghịch với mọi λ > 0. Sau đó tính nghiệm ridge với λ = 1 cho dữ liệu của ví dụ 1.5 (phạt cả w₀ lẫn w₁).
8. Tìm SVD của ma trận có hàng (0, 2) và (0, 0). Ma trận này có trị riêng gì? Vì sao trị riêng và giá trị kỳ dị khác nhau?
9. Chứng minh rằng với A đối xứng, hai vector riêng ứng với hai trị riêng khác nhau thì vuông góc.
10. Chứng minh det Q = ±1 với mọi ma trận trực giao Q. Giải thích bằng hình học: vì sao Q không thể làm thay đổi thể tích?

**★★★ Mức phỏng vấn nghiên cứu**

11. Chứng minh max xᵀAx trên ‖x‖₂ = 1 bằng λₘₐₓ (A đối xứng). Suy ra vì sao thành phần chính đầu tiên của PCA là vector riêng lớn nhất của ma trận hiệp phương sai.
12. Bạn có N = 100 ảnh, mỗi ảnh d = 10 000 chiều, đã trừ trung bình. Ma trận hiệp phương sai d × d có 10⁸ phần tử. Làm sao tính các thành phần chính mà chỉ cần ma trận 100 × 100?
13. Một lớp có W ∈ ℝ⁴⁰⁹⁶ˣ⁴⁰⁹⁶. LoRA học ΔW = BA với B ∈ ℝ⁴⁰⁹⁶ˣ⁸, A ∈ ℝ⁸ˣ⁴⁰⁹⁶. Đếm số tham số cần học, so với finetune toàn bộ W, và cho biết hạng tối đa của ΔW. Khi nào giả định hạng thấp có thể sai?
14. Tại một cực tiểu, Hessian có λₘₐₓ = 100 và λₘᵢₙ = 0.01. Với gradient descent dùng learning rate cố định: learning rate lớn nhất còn ổn định là bao nhiêu? Vì sao hội tụ theo hướng ứng với λₘᵢₙ rất chậm? Kể hai kỹ thuật thực tế làm giảm vấn đề này.

## Bài tính nhanh

15 bài, mỗi bài tối đa 2 phút, làm bằng giấy bút và bấm giờ cả bộ. Mục tiêu: xong cả 15 bài trong 25 phút, sai không quá 1 bài. Bài thi đầu vào thường bấm giờ chặt, nên tốc độ tính tay cũng quan trọng như hiểu khái niệm. Làm lại bộ này sau 3 ngày và sau 1 tuần.

1. Tính AB với A có hàng (1, 2), (3, 4) và B có hàng (0, 1), (1, 0). B làm gì với các cột của A?
2. Viết tích ngoài uvᵀ với u = (1, 2), v = (3, −1).
3. Tính det của ma trận có hàng (2, 5) và (1, 3).
4. Tính det của ma trận tam giác trên có hàng (2, 7, 1), (0, 3, 5), (0, 0, −1).
5. A ∈ ℝ²ˣ³ có hàng (1, 0, 2), (0, 1, 1); B ∈ ℝ³ˣ² có hàng (1, 1), (0, 2), (3, 0). Tính tr(AB) mà không cần tính cả ma trận AB.
6. Hạng của ma trận có hàng (1, 2, 3), (4, 5, 6), (7, 8, 9) là bao nhiêu?
7. x = (3, −4, 0). Tính ‖x‖₁, ‖x‖₂, ‖x‖∞.
8. Góc giữa (1, 0, 1) và (0, 1, 1) là bao nhiêu độ?
9. Chiếu b = (2, 3) lên đường thẳng theo hướng a = (1, 1).
10. Trị riêng và vector riêng của ma trận có hàng (0, 1), (1, 0).
11. Ma trận có hàng (3, 1), (0, 3) có những trị riêng nào, và bao nhiêu vector riêng độc lập? Nó có chéo hóa được không?
12. A ∈ ℝ²ˣ² có trị riêng 2 và 3. Tính det(A²) và tr(A⁻¹).
13. Các giá trị kỳ dị của diag(−3, 2) là bao nhiêu?
14. Ma trận có hàng (1, 1), (1, 1): trị riêng, hạng, có PSD không?
15. Một lớp Linear từ 512 chiều lên 2048 chiều, có bias. Có bao nhiêu tham số?

## Lab code và bài săn bug

Bốn lab dưới đây đã chạy với Python 3, NumPy 2.5 và scikit-image 0.26. Mỗi lab dùng seed riêng, nên số liệu không phụ thuộc code bạn viết ở lab khác; các sai số cỡ 10⁻⁹ trở xuống có thể lệch nhẹ theo phiên bản và thư viện BLAS. Cách học đúng: tự viết từng lab trước, chỉ đối chiếu code mẫu khi kết quả lệch với cột "kết quả mong đợi".

| Lab | Việc phải làm | Kết quả mong đợi |
| --- | --- | --- |
| 1. Power iteration | Tìm λₘₐₓ bằng cách nhân lặp v ← Av/‖Av‖₂ | Khớp `np.linalg.eigh` tới 6 chữ số; \|cos\| giữa hai vector = 1.0 |
| 2a. Bình phương tối thiểu, hệ không nhiễu | Phương trình chuẩn, QR, `lstsq`; đo cả phần dư lẫn sai số nghiệm ‖x̂ − x‖₂/‖x‖₂ | Bậc 6 (κ ≈ 2 × 10⁴): cả ba đều tốt (sai số nghiệm ≤ 3 × 10⁻¹⁰). Bậc 12 (κ ≈ 6.8 × 10⁸): phương trình chuẩn có phần dư 4.1 × 10⁻⁷ nhưng sai số nghiệm 6.1; QR sai 1.3 × 10⁻⁹, `lstsq` 2.6 × 10⁻⁹ |
| 2b. Thêm nhiễu cỡ 10⁻⁶ (tuyệt đối, mỗi phần tử) vào b | Giải bằng `lstsq`, đo sai số nghiệm | Bậc 6: 8.4 × 10⁻⁴. Bậc 12: 5.3 — sai gấp hơn 5 lần độ lớn nghiệm |
| 3. PCA bằng SVD | Viết PCA từ đầu, kiểm công thức sai số của SVD cắt cụt và so với một phép chiếu hạng k ngẫu nhiên | Bình phương sai số tái tạo = tổng σᵢ² bị bỏ (cùng 35.5344); chiếu ngẫu nhiên sai cỡ 1.3 × 10⁴, lớn hơn rất nhiều |
| 4a. Hạng thấp, ma trận tổng hợp | Cắt SVD ở hạng 1, 2, 5, 20 | Hạng 2 sai 5.0%; tăng hạng gần như không giảm thêm (hạng 20 vẫn 4.3%, phần còn lại là nhiễu) |
| 4b. Hạng thấp, ảnh thật 512 × 512 | Lặp lại với `skimage.data.camera()` | Hạng 20 sai 10.1%, hạng 100 sai 3.9%; giữ 99% năng lượng cần 21 thành phần, so với 2 ở 4a |

Code nằm trong notebook `labs/chuong01/lab.ipynb` (bản có chỗ TODO để bạn tự viết, kèm ô kiểm tra tự động) và `labs/chuong01/loigiai.ipynb` (bản lời giải đã chạy kiểm). Mở từ trang **Lab** của app.

**Câu hỏi sau lab.**

- Lab 2b: ở bậc 12, ngay cả `lstsq` cũng cho nghiệm sai gấp hơn 5 lần. Đó là lỗi của thuật toán hay của bài toán? Thuật toán tốt có cứu được không?
- Lab 4a: vì sao ma trận tổng hợp này có cấu trúc đúng hạng 2? *Gợi ý: sin(aᵢ) + cos(bⱼ) viết được thành tổng hai tích ngoài.*
- Lab 4b: vì sao "giữ 99% năng lượng" lại ứng với sai số tương đối khoảng 10%, chứ không phải 1%?

**Bài săn bug.** Hàm dưới đây chạy không báo lỗi nhưng cho kết quả sai. Trên dữ liệu sinh như lab 3 nhưng chỉ có 2 thành phần thật, nó báo hai thành phần đầu chỉ giải thích 0.01% phương sai, trong khi đáp án đúng là khoảng 85% và 15%. Tìm cả **ba** lỗi trước khi xem lời giải.

```python
def pca_buggy(X, k):
    C = X.T @ X / (len(X) - 1)
    evals, evecs = np.linalg.eigh(C)
    comps = evecs[:k]
    explained_ratio = evals[:k] / evals.sum()
    return comps, explained_ratio
```

## Bài dựng cảnh Manim

Mỗi chương có một bài tự dựng cảnh: muốn vẽ được một khái niệm, bạn phải hiểu từng bước của nó. Giới hạn **1 giờ** (tính vào luồng C), không chỉnh màu mè.

**Đề bài.** Dùng `manim/chuong01_scenes.py` làm mẫu, viết cảnh `PowerIteration` vào thư mục `manim/cua-toi/`:

1. Bắt đầu từ một vector đơn vị v₀ có góc 80° so với trục hoành.
2. Mỗi bước áp A (hàng (2, 1) và (1, 2)) rồi chuẩn hóa: vₖ₊₁ = Avₖ/‖Avₖ‖₂. Vẽ 6 bước.
3. Hiện góc giữa vₖ và vector riêng lớn nhất (1, 1)/√2 sau mỗi bước.

**Tiêu chí tự chấm.** (1) Đúng toán: tan của góc đó giảm đúng 3 lần mỗi bước, vì tỉ số λ₂/λ₁ = 1/3. (2) Có tiêu đề và phụ đề nói người xem phải thấy điều gì. (3) Dài không quá 30 giây.

Render bằng nút **Render cảnh của tôi** ở trang Manim. Lời giải tham khảo nằm ở `manim/loigiai/chuong01_power_iteration.py` — chỉ mở sau khi đã tự làm.

## Phỏng vấn, bẫy, thực tế khác lý thuyết

**Cách luyện.** Mỗi chuỗi có 4 tầng câu hỏi, đào sâu dần như người phỏng vấn thật. Trả lời to bằng tiếng Việt trước, sau đó nói lại bằng tiếng Anh, 2 phút mỗi tầng, ghi âm và tự nghe lại. Câu trả lời mẫu tiếng Anh bên dưới chỉ phủ tầng 1–2; tầng 3–4 bạn tự soạn, rồi đối chiếu với lời giải các bài tập liên quan.

**Chuỗi 1 — PCA**

1. PCA làm gì?
2. Vì sao các thành phần chính là vector riêng của ma trận hiệp phương sai?
3. Vì sao phải trừ trung bình? Nếu không trừ thì thành phần đầu tiên chỉ về đâu?
4. Khi nào PCA thất bại? (Cấu trúc phi tuyến, các đặc trưng khác thang đo, hướng có phương sai lớn không mang thông tin cho nhãn.)

> *Model answer:* PCA finds the orthogonal directions along which the centered data has the largest variance. If X̃ is the centered data matrix, the variance along a unit vector w is wᵀCw, where C is the covariance matrix. Maximizing this Rayleigh quotient gives the top eigenvector of C, so the principal components are the eigenvectors of C — equivalently, the right singular vectors of X̃. By default I would compute them with an SVD of X̃ rather than forming C, because that is more numerically stable. Forming C is cheaper when N is much larger than d and d is small (scikit-learn does this automatically since version 1.5), but it squares the condition number.

**Chuỗi 2 — Bình phương tối thiểu**

1. Viết nghiệm đóng của hồi quy tuyến tính.
2. Giải thích nó bằng hình học.
3. Khi nào AᵀA không khả nghịch, và bạn xử lý thế nào?
4. Vì sao thư viện số không giải bằng phương trình chuẩn? Ridge tương đương với giả định xác suất nào? (Prior Gauss trên trọng số — gặp lại ở chương 3.)

> *Model answer:* Assuming A has full column rank, the closed-form solution is x̂ = (AᵀA)⁻¹Aᵀb. Geometrically, Ax̂ is the orthogonal projection of b onto the column space of A, so the residual is orthogonal to every column of A — and that orthogonality condition is exactly the normal equations. If AᵀA is singular or badly conditioned, I would add ridge regularization or use the pseudo-inverse, and I would solve with QR or SVD rather than inverting AᵀA, since forming AᵀA squares the condition number.

**Chuỗi 3 — SVD và hạng thấp**

1. SVD khác phân rã trị riêng ở điểm nào?
2. Xấp xỉ hạng k tốt nhất là gì, sai số bằng bao nhiêu?
3. Giải thích LoRA bằng ngôn ngữ SVD.
4. Nếu phổ σᵢ của ΔW giảm chậm (không có cấu trúc hạng thấp), LoRA sẽ ra sao?

> *Model answer:* An eigendecomposition exists only for square, diagonalizable matrices, and its eigenvectors need not be orthogonal, while the SVD exists for every matrix and always gives two orthonormal bases. By the Eckart–Young theorem, truncating the SVD after k terms gives the best rank-k approximation; the error in Frobenius norm is the square root of the sum of the discarded σᵢ², and in spectral norm it is σₖ₊₁. LoRA bets that the fine-tuning update ΔW has low intrinsic rank, so it learns a rank-r factorization BA instead of the full matrix.

**Chuỗi 4 — Số điều kiện và tối ưu**

1. Số điều kiện là gì?
2. Nó ảnh hưởng thế nào đến gradient descent?
3. Vì sao chuẩn hóa đầu vào giúp huấn luyện nhanh hơn?
4. Adam "sửa" vấn đề này đến đâu, và không sửa được gì? (Adam chỉ co giãn theo từng tọa độ, không xoay được hệ trục.)

> *Model answer:* The condition number κ = σₘₐₓ/σₘᵢₙ (infinite if the matrix is rank-deficient) measures how much a problem can amplify relative errors in its input. For gradient descent on a quadratic loss, the step size is capped by the largest Hessian eigenvalue, while progress along the flattest direction depends on the smallest one, so the number of iterations grows roughly in proportion to κ. Input normalization, normalization layers and preconditioned optimizers such as Adam all try to reduce the effective condition number.

**Từ vựng nên nói trôi chảy:** column space, null space, rank-deficient, orthogonal projection, normal equations, ill-conditioned, eigendecomposition, singular value decomposition, low-rank approximation, positive semi-definite, Rayleigh quotient.

**Bẫy thường gặp.**

- `np.linalg.eigh` trả trị riêng theo thứ tự **tăng dần**; `np.linalg.svd` trả σᵢ theo thứ tự **giảm dần**. Lấy nhầm đầu là lấy nhầm thành phần.
- `eigh` trả vector riêng theo **cột**; `svd` trả Vᵀ, nên thành phần chính là **hàng** của `Vt`.
- Dữ liệu dạng N × d (mẫu theo hàng) hay d × N (mẫu theo cột): sai quy ước thì bạn trừ trung bình theo sai trục, và các thành phần chính nằm trong ℝᴺ thay vì ℝᵈ.
- Viết `inv(A) @ b` thay vì `solve(A, b)`: chậm hơn và kém chính xác hơn.
- Thấy phần dư nhỏ rồi kết luận nghiệm đúng: lab 2 cho thấy phần dư 4.1 × 10⁻⁷ mà nghiệm sai 610%.
- Kết luận "mọi trị riêng ≥ 0 nên PSD" với ma trận không đối xứng: sai (bài tập 8 có phản ví dụ).

**Thực tế khác lý thuyết.**

- Trên lớp bạn tính trị riêng bằng det(A − λI). Thư viện không bao giờ làm vậy; họ dùng thuật toán lặp (QR, Lanczos), và với ma trận rất lớn thì dùng randomized SVD chỉ lấy k thành phần đầu.
- Mô hình học sâu chạy phần lớn phép tính ở bf16/fp8 (mixed precision), còn trọng số chính và phép cộng dồn thường giữ ở fp32; độ chính xác thấp như vậy làm số điều kiện lớn gây lỗi số rõ hơn nhiều so với float64 trong bài tập.
- Ma trận trọng số và dữ liệu thật thường có phổ σᵢ giảm dần chứ không có "hạng thấp" tuyệt đối (lab 4b). Nén hạng thấp luôn là đánh đổi giữa kích thước và chất lượng, phải đo trên tác vụ thật.

## Mini-test 30 phút

Làm sau khi xong bài tập và lab, bấm giờ đúng 30 phút, không xem sách, không dùng máy tính (trừ câu 5 chỉ đọc code, không chạy). Tổng 30 điểm. **Từ 24 điểm:** sang chương 2. **18 đến dưới 24:** ôn lại mục ứng với câu sai rồi làm lại sau 3 ngày. **Dưới 18:** học lại chương.

1. **(4 điểm)** A có hàng (2, 2) và (1, 1). Tìm hạng, các giá trị kỳ dị, một cơ sở của C(A) và của N(A).
2. **(4 điểm)** Khớp mô hình y = wt (không có hệ số chặn) qua ba điểm (t, y) = (1, 2), (2, 3), (3, 5) theo bình phương tối thiểu. Tìm w.
3. **(5 điểm)** Chứng minh: nếu A² = A thì mọi trị riêng của A là 0 hoặc 1. Cho một ví dụ A² = A nhưng A không đối xứng, và nói ví dụ đó "chiếu" theo kiểu gì.
4. **(5 điểm)** A ∈ ℝⁿˣⁿ đối xứng, có trị riêng λ₁ ≥ ⋯ ≥ λₙ. Tìm trị riêng của A + cI và của A². Với giá trị nào của c thì A + cI xác định dương?
5. **(6 điểm)** Đọc code và trả lời không chạy máy:

```python
X = np.random.randn(1000, 50) @ np.diag(np.linspace(1, 10, 50))
Xc = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
Z = Xc @ Vt[:5].T
```

(a) Kích thước của U, s, Vt, Z? (b) Các cột của Z có tương quan với nhau không, vì sao? (c) Phương sai của cột đầu tiên của Z xấp xỉ bao nhiêu?

6. **(6 điểm)** Tối đa 5 câu: vì sao nén embedding 768 chiều xuống 128 chiều bằng PCA đôi khi gần như không làm giảm chất lượng tìm kiếm, nhưng đôi khi lại làm giảm mạnh? Trả lời bằng ngôn ngữ của các σᵢ.

## Flashcard

17 thẻ của chương nằm trong `flashcards/chuong01.csv`. Ôn ở trang **Ôn tập** của app: app tự lên lịch theo hộp Leitner (thẻ nhớ được sẽ giãn lịch ôn 1 → 3 → 7 → 14 ngày, thẻ quên quay lại hộp 1). File CSV này cũng nhập thẳng được vào Anki nếu bạn thích dùng Anki.

## Lời giải

Chỉ đọc sau khi đã tự thử ít nhất 20 phút. Với bài chứng minh, so ý chính chứ không so từng chữ. Mọi đáp số bằng số trong chương đã được kiểm lại bằng NumPy.

**Kiểm tra nhanh**

1. A(Bx) rẻ hơn: Bx và A(Bx) mỗi bước 10⁴ phép nhân, tổng 2 × 10⁴. (AB)x tốn 10⁶ + 10⁴. Rẻ hơn khoảng 50 lần.
2. Có. Hạng tối đa là 3 < 5 cột, nên dim N(A) ≥ 5 − 3 = 2.
3. Ax̂ là điểm gần b nhất trong C(A), và điểm gần nhất trên một không gian con là chân đường vuông góc. Nếu phần dư còn thành phần dọc theo một cột, dịch x̂ theo cột đó sẽ giảm khoảng cách, mâu thuẫn với tính tối ưu.
4. 1 (một lần, vector riêng u, vì Pu = u(uᵀu) = u) và 0 (n − 1 lần, mọi x vuông góc với u).
5. Đều bằng 1, vì Q = Q · I · Iᵀ là một SVD của Q.

**Bài tập giấy bút**

1. dim C(A) = 3, dim N(A) = 6 − 3 = 3, dim C(Aᵀ) = 3, dim N(Aᵀ) = 4 − 3 = 1.
2. uvᵀ có các hàng (1, −1), (2, −2), (3, −3). Hạng 1, vì mọi hàng là bội của (1, −1).
3. ‖Qx‖₂² = xᵀQᵀQx = xᵀx = ‖x‖₂².
4. λ² − 7λ + 10 = 0 ⇒ λ = 5 với vector riêng (1, 1), λ = 2 với vector riêng (1, −2). Kiểm tra: 5 + 2 = 7 = tr A, 5 · 2 = 10 = det A. Hai vector riêng không vuông góc vì A không đối xứng.
5. Nếu Ax = 0 thì AᵀAx = 0. Ngược lại, nếu AᵀAx = 0 thì xᵀAᵀAx = ‖Ax‖₂² = 0 nên Ax = 0. Hai ma trận cùng không gian null và cùng số cột n, nên cùng hạng.
6. AᵀA khả nghịch vì các cột của A độc lập (bài 5). P² = A(AᵀA)⁻¹(AᵀA)(AᵀA)⁻¹Aᵀ = P. Pᵀ = P vì AᵀA đối xứng nên nghịch đảo của nó cũng đối xứng. Nếu Pv = λv thì P²v = λ²v, mà P²v = Pv = λv, nên λ² = λ, tức λ = 0 hoặc 1.
7. xᵀ(AᵀA + λI)x = ‖Ax‖₂² + λ‖x‖₂² > 0 với mọi x ≠ 0, nên ma trận xác định dương, do đó khả nghịch. Với dữ liệu ví dụ 1.5: AᵀA + I có hàng (4, 3), (3, 6), định thức 15; nghiệm (w₀, w₁) = (0.8, 0.6). So với (7/6, 1/2) ≈ (1.17, 0.5) của bình phương tối thiểu, ridge co chuẩn ‖w‖₂ (từ 1.27 xuống 1.00), nhưng không co từng thành phần: w₀ giảm mạnh (1.17 → 0.8) còn w₁ lại tăng (0.5 → 0.6). Lý do: w₀ bị phạt nặng nên w₁ phải "gánh" thêm phần giải thích dữ liệu, vì hai cột của A tương quan. (Thực tế thường không phạt bias w₀.)
8. A = 2e₁e₂ᵀ, nên σ₁ = 2 với u₁ = e₁, v₁ = e₂; σ₂ = 0 với u₂ = e₂, v₂ = e₁. A là ma trận tam giác trên có đường chéo bằng 0, nên cả hai trị riêng bằng 0. Trị riêng đo hướng được giữ nguyên; giá trị kỳ dị đo mức kéo giãn. A đưa e₂ thành 2e₁: kéo giãn hệ số 2 nhưng không giữ hướng nào. Thêm nữa, A có mọi trị riêng ≥ 0 nhưng xᵀAx = 2x₁x₂ có thể âm, nên "trị riêng không âm" không suy ra PSD khi ma trận không đối xứng.
9. λ₁v₁ᵀv₂ = (Av₁)ᵀv₂ = v₁ᵀAv₂ = λ₂v₁ᵀv₂ (dùng Aᵀ = A). Vậy (λ₁ − λ₂)v₁ᵀv₂ = 0, mà λ₁ ≠ λ₂ nên v₁ᵀv₂ = 0.
10. det(QᵀQ) = det Qᵀ · det Q = (det Q)² = det I = 1, nên det Q = ±1. Hình học: Q giữ nguyên độ dài và góc, nên hình lập phương đơn vị biến thành một hình lập phương đơn vị (có thể bị lật hướng); thể tích không đổi.
11. Đặt y = Qᵀx, thì ‖y‖₂ = 1 và xᵀAx = λ₁y₁² + ⋯ + λₙyₙ² ≤ λₘₐₓ(y₁² + ⋯ + yₙ²) = λₘₐₓ; dấu bằng tại x là vector riêng của λₘₐₓ. Phương sai của dữ liệu chiếu lên hướng w (‖w‖₂ = 1) là wᵀCw, nên hướng có phương sai lớn nhất là vector riêng lớn nhất của C.
12. "Gram trick": tính G = X̃X̃ᵀ ∈ ℝ¹⁰⁰ˣ¹⁰⁰ = UΣ²Uᵀ, rồi khôi phục thành phần chính bằng vᵢ = X̃ᵀuᵢ/σᵢ. Sau khi trừ trung bình có tối đa N − 1 = 99 thành phần khác 0. Trong code, `np.linalg.svd(Xc, full_matrices=False)` làm việc tương đương với chi phí cỡ N²d.
13. LoRA: 4096 · 8 + 8 · 4096 = 65 536 tham số. Finetune toàn bộ: 4096² = 16 777 216. Tỉ lệ khoảng 0.39%. Hạng tối đa của ΔW là 8. Giả định hạng thấp dễ sai khi tác vụ mới khác xa dữ liệu pretraining (ngôn ngữ mới, miền chuyên biệt, cần nạp nhiều kiến thức mới); khi đó tăng r hoặc finetune toàn bộ. Có bằng chứng thực nghiệm: Biderman và cộng sự (2024), *LoRA Learns Less and Forgets Less* (arXiv 2405.09673), báo cáo rằng finetune toàn bộ học các nhiễu loạn trọng số có hạng lớn hơn 10–100 lần cấu hình LoRA thường dùng, và LoRA kém hơn rõ ở các miền code và toán (đã mở trang arXiv ngày 04/11/2026). Dù vậy vẫn cần đo trên từng tác vụ.
14. Ổn định khi η < 2/100 = 0.02. Theo hướng có λₘᵢₙ = 0.01, mỗi bước sai số chỉ nhân với 1 − ηλₘᵢₙ ≈ 1 − 2 × 10⁻⁴, nên cần cỡ 1/(ηλₘᵢₙ) bước để giảm đáng kể. Ở sát biên η ≈ 0.02 thì đó là hơn 5 000 bước (tại đúng η = 0.02, hướng λₘₐₓ có hệ số −1 nên dao động mãi, không hội tụ); với η = 1/λₘₐₓ = 0.01 an toàn hơn thì cần cỡ 10 000 bước. Số điều kiện κ = 10⁴. Kỹ thuật: chuẩn hóa đầu vào và normalization trong mạng (BatchNorm, LayerNorm), momentum, optimizer thích nghi như Adam.

**Bài tính nhanh**

1. AB có hàng (2, 1), (4, 3): B đổi chỗ hai cột của A.
2. uvᵀ có hàng (3, −1), (6, −2).
3. 2 · 3 − 5 · 1 = 1.
4. −6 (tích các phần tử đường chéo).
5. tr(AB) = tổng mọi AᵢⱼBⱼᵢ = 1·1 + 0·0 + 2·3 + 0·1 + 1·2 + 1·0 = 9.
6. 2 (hàng 3 = 2 × hàng 2 − hàng 1).
7. ‖x‖₁ = 7, ‖x‖₂ = 5, ‖x‖∞ = 4.
8. cos θ = 1/(√2 · √2) = 1/2, nên θ = 60°.
9. (aᵀb/aᵀa)a = (5/2)(1, 1) = (2.5, 2.5).
10. λ = 1 với vector riêng (1, 1); λ = −1 với vector riêng (1, −1).
11. λ = 3 (bội 2) nhưng chỉ có một vector riêng độc lập (1, 0), nên không chéo hóa được.
12. det(A²) = (2 · 3)² = 36; tr(A⁻¹) = 1/2 + 1/3 = 5/6.
13. 3 và 2 (giá trị kỳ dị luôn không âm).
14. λ = 2 và 0; hạng 1; có PSD.
15. 512 · 2048 + 2048 = 1 050 624.

**Câu hỏi sau lab**

- Lab 2b: lỗi nằm ở bài toán, không ở thuật toán. Với κ(A) ≈ 6.8 × 10⁸, nhiễu cỡ 10⁻⁶ trong b có thể bị khuếch đại thành sai số lớn trong x. Thuật toán tốt chỉ tránh làm tệ thêm; muốn cứu phải đổi bài toán: regularization (ridge), bậc thấp hơn, hoặc dùng cơ sở đa thức trực giao.
- Lab 4a: sin(aᵢ) + cos(bⱼ) = sin(aᵢ) · 1 + 1 · cos(bⱼ), tức ma trận bằng s1ᵀ + 1cᵀ với s = sin(a), c = cos(b): tổng hai tích ngoài, nên hạng tối đa 2.
- Lab 4b: năng lượng là bình phương. Sai số tương đối = √(1 − tỉ lệ năng lượng giữ lại) = √0.01 = 0.1.

**Mini-test (thang 30 điểm)**

1. Hạng 1 (1 điểm). A = (2, 1)(1, 1)ᵀ nên σ₁ = ‖(2, 1)‖₂ · ‖(1, 1)‖₂ = √5 · √2 = √10, σ₂ = 0 (1.5 điểm). C(A) = span{(2, 1)}, N(A) = span{(1, −1)} (1.5 điểm).
2. Công thức w = (tổng tᵢyᵢ)/(tổng tᵢ²), từ phương trình chuẩn tᵀt · w = tᵀy (2 điểm). Tính tổng: 2 + 6 + 15 = 23 và 1 + 4 + 9 = 14 (1 điểm). w = 23/14 ≈ 1.643 (1 điểm).
3. Av = λv ⇒ A²v = λ²v, mà A²v = Av = λv, nên λ² = λ (3 điểm). Ví dụ A có hàng (1, 1), (0, 0): A² = A nhưng A không đối xứng (1 điểm). A(x, y) = (x + y, 0): chiếu lên trục thứ nhất theo hướng (1, −1), là phép chiếu xiên chứ không phải chiếu trực giao (1 điểm).
4. A + cI có trị riêng λᵢ + c, A² có trị riêng λᵢ², với cùng các vector riêng (3 điểm). A + cI xác định dương ⇔ λₙ + c > 0 ⇔ c > −λₙ (2 điểm).
5. (a) U: (1000, 50), s: (50,), Vt: (50, 50), Z: (1000, 5) (2 điểm). (b) Không tương quan: Z = X̃V₅ = U₅Σ₅ nên ZᵀZ = Σ₅² là ma trận đường chéo, và các cột của Z có trung bình 0 (2 điểm). (c) σ₁²/999, xấp xỉ 100 vì cột có độ lệch chuẩn lớn nhất là 10. Chạy thật (mô phỏng 200 seed) cho khoảng 100–116 tùy seed, trung vị ≈ 108.6: trị riêng lớn nhất của hiệp phương sai mẫu bị lệch lên, vì có nhiều cột thang gần 10 (10, 9.8, 9.6, …) và d/N = 0.05. Thang chấm: ≈ 100 kèm lý do là đủ điểm (2 điểm).
6. Thang chấm: (2 điểm) PCA giữ k hướng có σᵢ lớn nhất; nếu phổ σᵢ của embedding giảm nhanh, 128 thành phần giữ gần hết năng lượng và tích vô hướng gần như không đổi. (2 điểm) Nếu phổ giảm chậm, phần năng lượng bị bỏ lớn; hoặc thông tin phân biệt các văn bản nằm ở hướng có phương sai nhỏ — PCA tối ưu cho tái tạo, không cho tác vụ. (2 điểm) Cách kiểm: vẽ phổ σᵢ và đo recall trên tập kiểm tra trước và sau khi nén.

**Săn bug — ba lỗi**

1. **Không trừ trung bình.** C khi đó là ma trận moment bậc hai, và hướng lớn nhất chủ yếu chỉ về phía vector trung bình (ở đây là 3.0 trên mọi chiều), không phải hướng phương sai.
2. **Thứ tự của `eigh`.** Trị riêng tăng dần, nên `evals[:k]` là k trị riêng *nhỏ nhất* — lý do tỉ lệ giải thích chỉ 0.01%.
3. **Hàng thay vì cột.** Vector riêng là các *cột* của `evecs`; `evecs[:k]` lấy k hàng, không có ý nghĩa gì.

```python
def pca_fixed(X, k):
    Xc = X - X.mean(axis=0)                          # sửa lỗi 1
    C = Xc.T @ Xc / (len(X) - 1)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]                  # sửa lỗi 2: giảm dần
    evals, evecs = evals[order], evecs[:, order]
    return evecs[:, :k].T, evals[:k] / evals.sum()   # sửa lỗi 3: lấy cột
```

Đã chạy kiểm: bản sửa cho tỉ lệ 0.850 và 0.147, mỗi thành phần tìm được có |cos| = 1.00 với hướng thật.

## Tài liệu đào sâu và nguồn

Nội dung chương được viết lại, đối chiếu phạm vi kiến thức với các nguồn dưới đây. Đọc theo thứ tự ưu tiên:

| Tài liệu | Dùng để làm gì | Phần nên đọc |
| --- | --- | --- |
| [CS229 Linear Algebra Review and Reference](https://cs229.stanford.edu/summer2022/cs229-linalg.pdf) (Stanford; Kolter, cập nhật bởi Do và Ma) | Tài liệu ôn chính thức của khóa ML Stanford, khoảng 30 trang | Toàn bộ; mục 4 (giải tích ma trận) là cầu nối sang chương 2 |
| [MIT 18.06 Linear Algebra, Spring 2010](https://www.ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/pages/study-materials) (Gilbert Strang, OCW) | Bài giảng video, bài tập và đề thi có lời giải | Bốn không gian con, phép chiếu, Gram–Schmidt, trị riêng, SVD; làm thêm đề thi cũ |
| [Mathematics for Machine Learning](https://mml-book.github.io) (Deisenroth, Faisal, Ong; Cambridge, 2020; PDF miễn phí) | Sách nối toán với ML, có bài tập và notebook PCA | Chương 2–4 và 10 (PCA); làm notebook PCA |
| [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu và cộng sự, 2021) | Ứng dụng trực tiếp của xấp xỉ hạng thấp | Phần phương pháp và phần phân tích hạng của ΔW |
| [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) (Biderman và cộng sự, 2024) | Bằng chứng thực nghiệm về giới hạn của giả định hạng thấp (bài tập 13) | Phần tóm tắt và phần phân tích phổ của ΔW |

**Từ khóa để tự tìm thêm:** randomized SVD (Halko, Martinsson, Tropp), implicit bias of gradient descent, numerical linear algebra (Trefethen và Bau), condition number và backward stability.

**Trước khi sang chương 2:** đạt từ 24/30 điểm mini-test, làm lại các bài ★★★ sai, và tự viết được lab 3 (PCA) mà không nhìn code mẫu.

Nguồn tra cứu ngày 24/10/2026; trang arXiv 2405.09673, bản CS229 summer2022 và mã nguồn scikit-learn 1.9.1 kiểm lại ngày 04/11/2026. MML book và LoRA (2106.09685) chưa mở lại được lần này [cần kiểm chứng].
