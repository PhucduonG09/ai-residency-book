# Đề thi thử 1 — Phần I (chương 1–6)

## Cách làm

Đề này làm vào cuối tuần 13 (khoảng 31/1/2027), sau khi bạn học xong chương 1–6. Nó mô phỏng một vòng **entrance test** theo hình dung của người soạn [giả định; định dạng đề thật không công khai]: một buổi bấm giờ, toán và ML nền tảng, có phần lập trình. Mục đích là đo lại Phần I trong điều kiện có áp lực thời gian, rồi quyết định ôn chỗ nào trước khi sang Phần II.

**Lưu ý trung thực.** Đề entrance test của Qualcomm AI Residency và VinAI không công khai (xem `huong-dan.md`, dòng "Lưu ý trung thực"). Đây là **đề mô phỏng** do sách soạn, dựa trên yêu cầu công khai "thành thạo lập trình và toán" và nội dung các khóa ML nền tảng (Stanford CS229, CS231n, MIT 18.065 ở mức chủ đề, không chép bài). Cơ cấu điểm, thời lượng, thứ tự các phần, số phút từng câu và mọi ngưỡng điểm là lựa chọn hoặc ước lượng của người soạn, chưa hiệu chỉnh trên người làm thật.

**Quy tắc.**

- 120 phút liền, bấm giờ, bàn học, giấy nháp, điện thoại để xa. Không tài liệu, không máy tính bỏ túi. Bảng hằng số ở đầu phần Đề bài là thứ duy nhất được tra.
- **Thứ tự làm: Phần A → Phần C → Phần B.** Phần C (lập trình) đặt giữa đề để điểm lập trình được đo trong giờ, không bị dồn sang lượt hai. Phần B làm cuối, trên giấy.
- **Phần C được dùng máy tính**: chỉ trình thông dịch Python với NumPy, không internet, không tài liệu, không trợ lý AI, không tự hoàn thành code. Trước khi bấm giờ, chép `labs/thi-thu-1/bai_lam_mau.py` thành `bai_lam.py` ở thư mục gốc của sách; file này chỉ có chữ ký hai hàm và docstring. Trong giờ, điền hai hàm vào đó và được chạy thử với dữ liệu tự nghĩ ra. **Chỉ chạy bộ chấm `labs/thi-thu-1/cham_phan_c.py` sau khi hết giờ, và đừng mở file đó trước**: nó không chứa lời giải nhưng có dữ liệu test và đáp số câu C1(b).
- Đáp số được để dạng biểu thức (kiểu 5/12, 3/√2, ln 11 − ½ln 2; đây chỉ là ví dụ dạng, không trùng đáp số câu nào) và vẫn đủ điểm. Đáp số thập phân cần đúng đến 2 chữ số thập phân, trừ khi đề nói khác.
- **Mốc giờ gợi ý** [lựa chọn của người soạn]: đọc đề 2 phút; Phần A đến phút 47; Phần C đến phút 73; Phần B từ phút 73 đến 120. Cột Phút ở Bản đồ câu hỏi là ước lượng cho từng câu; câu nào vượt số phút ở đó cộng 2 phút thì ghi "bỏ" và làm tiếp. Hết mốc của một phần mà chưa xong thì vẫn sang phần sau, câu còn lại để lượt hai.
- **Hai con số điểm, như bài chẩn đoán.** Lượt một là 120 phút. Hết giờ: gạch ngang chỗ đang dừng, đổi bút khác màu, làm tiếp tối đa 30 phút (lượt hai); câu nào vẫn trống thì làm nốt không tính giờ (lượt ba, bút màu thứ ba). Phần C làm trên máy thì ghi chú giờ lưu file ở lượt một (ví dụ lưu bản lượt một thành `bai_lam_luot1.py`). Ghi **điểm trong giờ** (chỉ phần làm ở lượt một) và **điểm cuối** (mọi lượt). Bảng chấm theo chương dùng **điểm cuối** (đo kiến thức); điểm trong giờ đo tốc độ, đọc ở mục Đọc kết quả.

**Thời gian có đủ không.** Người soạn ước lượng từng câu (bảng ở mục Số liệu đã kiểm): người học mục tiêu, tức sinh viên năm 3 vừa học xong chương 1–6 và làm đề lần đầu, cần khoảng **133 phút**; một người giải nhanh, vừa ôn xong Phần I, cần khoảng **92 phút** [ước lượng, chưa đo trên người thật; sai số có thể ±15 phút]. Như vậy đề dài hơn 120 phút khoảng 13 phút với người học mục tiêu, phần thiếu thường rơi vào bài B cuối cùng. Độ dài này là lựa chọn của người soạn, để tập làm bài và bỏ câu dưới áp lực giờ [không có nguồn về độ dài đề thật]. Một khoảng chênh nhỏ giữa điểm cuối và điểm trong giờ là bình thường; ngưỡng để coi là vấn đề tốc độ ghi ở mục Đọc kết quả.

**Cách chấm.** Tổng 100 điểm. Phần A 20 câu × 2 điểm; Phần C 2 bài × 10 điểm; Phần B 5 bài × 8 điểm. Lời giải ghi rõ ý nào đáng bao nhiêu điểm. Chỉ cho điểm nguyên, không nửa điểm. Đáp số đúng mà thiếu lý do ở ý đề yêu cầu giải thích thì mất điểm của ý giải thích. Phần C: chạy `python labs/thi-thu-1/cham_phan_c.py --student bai_lam.py` từ thư mục gốc của sách để chấm phần tự động (16 điểm, chỉ cần NumPy), rồi chấm tay 4 điểm còn lại theo lời giải.

**Gắn nhãn lỗi** cho mỗi ý mất điểm, giống bài chẩn đoán: **K** (khái niệm), **T** (tính sai), **L** (đọc đề sai, thiếu thuật ngữ), **H** (chỉ làm được ở lượt hai hoặc ba). Ghi vào sổ lỗi.

**Độ khó.** ★ hiểu định nghĩa, ★★ mức đề thi đầu vào, ★★★ câu phân loại (mức phỏng vấn nghiên cứu). Các ý ★★★ cộng lại 12 điểm.

## Bản đồ câu hỏi

Thứ tự trong bảng là thứ tự làm bài. Cột Phút là ước lượng của người soạn cho người học mục tiêu (cộng ra 133 phút, xem Thời gian có đủ không).

| Câu | Chương, mục | Kỹ năng | Độ khó | Điểm | Phút [ước lượng] |
| --- | --- | --- | --- | --- | --- |
| A1 | Chương 1, mục 1.1 và 1.4 | Ma trận của phép biến đổi, định thức là hệ số diện tích | ★ | 2 | 2 |
| A2 | Chương 1, mục 1.2 | Hạng, không gian null | ★ | 2 | 3 |
| A3 | Chương 1, mục 1.3 | Chiếu lên một đường thẳng, khoảng cách | ★ | 2 | 2 |
| A4 | Chương 1, mục 1.4 | Trị riêng của đa thức ma trận và của nghịch đảo | ★★ | 2 | 3 |
| A5 | Chương 1, mục 1.5 | Giá trị kỳ dị khi cột trực giao, κ, chuẩn Frobenius | ★★ | 2 | 2 |
| A6 | Chương 2, mục 2.1–2.2 | Gradient và Hessian của softplus(aᵀx) | ★ | 2 | 3 |
| A7 | Chương 2, mục 2.3 | Backprop tay qua cổng max và fan-out | ★ | 2 | 3 |
| A8 | Chương 2, mục 2.5 | Jacobian của softmax | ★★ | 2 | 2 |
| A9 | Chương 3, mục 3.1 | Bayes, tỉ lệ nền | ★ | 2 | 2 |
| A10 | Chương 3, mục 3.5A | Entropy, cross-entropy, KL bất đối xứng | ★★ | 2 | 3 |
| A11 | Chương 3, mục 3.5B | SE, CI Wald, cỡ tập test | ★★ | 2 | 2 |
| A12 | Chương 4, mục 4.1 | Nhận ra hàm không lồi | ★ | 2 | 1 |
| A13 | Chương 4, mục 4.1 | Lagrange, ý nghĩa nhân tử | ★★ | 2 | 3 |
| A14 | Chương 4, mục 4.4 | Hai bước Adam với gradient đổi dấu | ★★ | 2 | 4 |
| A15 | Chương 5, mục 5.2 | Rò rỉ dữ liệu trong CV | ★ | 2 | 1 |
| A16 | Chương 5, mục 5.3 | Margin, hinge loss | ★★ | 2 | 2 |
| A17 | Chương 5, mục 5.5 | Một vòng k-means | ★★ | 2 | 2 |
| A18 | Chương 6, mục 6.1 và 6.4 | Đếm tham số có BatchNorm | ★ | 2 | 3 |
| A19 | Chương 6, mục 6.4 | Lan truyền phương sai qua lớp ReLU | ★★ | 2 | 3 |
| A20 | Chương 6, mục 6.5 | Loss ban đầu, lỗi im lặng khi đánh giá | ★ | 2 | 2 |
| C1 | Chương 2, mục 2.5; chương 3, mục 3.3 | Softmax + cross-entropy ổn định số, vectorized | ★★ | 10 | 13 |
| C2 | Chương 5, mục 5.1; chương 4, mục 4.4; chương 6, mục 6.5 | Một bước logistic regression có L2, lỗi shape im lặng | ★★ đến ★★★ | 10 | 14 |
| B1 | Chương 2, mục 2.2; chương 1, mục 1.5; chương 4, mục 4.5 | Gradient của phân rã hạng thấp, Eckart–Young, không lồi | ★★ đến ★★★ | 8 | 13 |
| B2 | Chương 3, mục 3.4 và 3.5A; chương 4, mục 4.1 | MAP (và MLE) của unigram bằng Lagrange, perplexity | ★★ đến ★★★ | 8 | 10 |
| B3 | Chương 4, mục 4.2 | Hội tụ của GD trên hàm bậc hai, dao động | ★ đến ★★★ | 8 | 10 |
| B4 | Chương 5, mục 5.2; chương 3, mục 3.2 và 3.4 | Bias–variance của ridge, λ tối ưu | ★★ đến ★★★ | 8 | 10 |
| B5 | Chương 6, mục 6.2; chương 2, mục 2.5 | Backprop MLP 2–2–1 với batch 2, gradient check | ★ đến ★★★ | 8 | 13 |
| | | Đọc đề | | | 2 |
| **Tổng** | | | | **100** | **133** |

Phút theo phần [ước lượng]: A 48, C 27, B 56, đọc đề 2. Mốc giờ trong 120 phút: A 45, C 26, B 47, đọc đề 2.

## Đề bài

**Bảng hằng số (được tra).** ln 2 = 0.6931; ln 3 = 1.0986; ln 5 = 1.6094; ln 7 = 1.9459; ln 10 = 2.3026; σ(2) = 0.8808, nên σ(−2) = 0.1192; ln(1 + e²) = 2.1269; √2 = 1.4142; √3 = 1.7321; √5 = 2.2361; hệ số 95% của phân phối chuẩn 1.96. Ở đây σ(z) = 1/(1 + e⁻ᶻ).

### Phần A — Trắc nghiệm và điền số (20 câu, 40 điểm, đến phút 47)

Mỗi câu 2 điểm. Chỉ ghi đáp số và một dòng lý do khi đề hỏi "vì sao".

**A1.** *(Chương 1 · ★ · 2 điểm)* Phép biến đổi tuyến tính T: ℝ² → ℝ² đưa e₁ thành (2, 1) và e₂ thành (−1, 3). (a) Tính T(1, 2). (b) Ảnh của hình vuông đơn vị qua T có diện tích bao nhiêu?

**A2.** *(Chương 1 · ★ · 2 điểm)* Ba vector đặc trưng (1, 1, 0), (0, 1, 1), (1, 2, 1) xếp thành ba hàng của A. (a) Chúng sinh ra không gian con bao nhiêu chiều? (b) Cho một vector w ≠ 0 sao cho mô hình tuyến tính xᵀw cho đầu ra 0 với cả ba vector.

**A3.** *(Chương 1 · ★ · 2 điểm)* Chiếu b = (3, 0, 3) lên đường thẳng qua gốc theo hướng a = (1, 2, 2). Cho hình chiếu và khoảng cách từ b tới đường thẳng đó.

**A4.** *(Chương 1 · ★★ · 2 điểm)* A có hàng (1, 2) và (3, 2). (a) Trị riêng của A. (b) Trị riêng của A² − A, và det(A⁻¹).

**A5.** *(Chương 1 · ★★ · 2 điểm)* A ∈ ℝ³ˣ² có hai cột (1, 2, 2) và (4, −4, 2). Không tính AᵀA đầy đủ, cho σ₁, σ₂, κ(A) và chuẩn Frobenius của A. Một dòng: vì sao đọc được ngay?

**A6.** *(Chương 2 · ★ · 2 điểm)* f(x) = ln(1 + exp(aᵀx)) với a = (1, −1, 2). Tại x = (1, 1, 0): tính ∇f và Hessian; Hessian có hạng bao nhiêu?

**A7.** *(Chương 2 · ★ · 2 điểm)* f(x, y) = max(x, y)·(x − y) tại x = 3, y = 1. Tính f, ∂f/∂x, ∂f/∂y bằng đồ thị tính toán (ghi rõ biến nào có fan-out).

**A8.** *(Chương 2 · ★★ · 2 điểm)* p = softmax(z) với z ∈ ℝ³. Tại z = (0, 0, 0), cho phần tử đường chéo và phần tử ngoài đường chéo của Jacobian ∂p/∂z. Một dòng: vì sao tổng mỗi hàng của Jacobian bằng 0?

**A9.** *(Chương 3 · ★ · 2 điểm)* Một bộ phát hiện văn bản do máy sinh có TPR 0.8 và FPR 0.1. (a) Trong kho có 10% văn bản do máy sinh; một văn bản bị gắn cờ, xác suất nó do máy sinh? (b) Nếu tỉ lệ đó là 50% thì sao?

**A10.** *(Chương 3 · ★★ · 2 điểm)* p = (½, ½, 0, 0), q đều trên 4 giá trị. Tính theo bit: H(p), H(p, q), KL(p‖q). Còn KL(q‖p) bằng bao nhiêu, vì sao?

**A11.** *(Chương 3 · ★★ · 2 điểm)* Accuracy 0.80 trên 625 câu test. (a) SE và CI 95% Wald. (b) Muốn nửa độ rộng CI còn một nửa (giữ accuracy 0.80) thì cần bao nhiêu câu?

**A12.** *(Chương 4 · ★ · 2 điểm)* Hàm nào **không** lồi trên miền đã ghi? Chọn một và nêu lý do một dòng.
(A) ln(1 + eˣ) trên ℝ. (B) ‖x‖₁ trên ℝⁿ. (C) max(x², 1 − x) trên ℝ. (D) f(x₁, x₂) = x₁x₂ trên ℝ².

**A13.** *(Chương 4 · ★★ · 2 điểm)* Cực tiểu x² + 2y² với ràng buộc x + y = 3. (a) Nghiệm (x*, y*), giá trị f*, và nhân tử ν trong ∇f = ν∇g. (b) Không giải lại: ước lượng f* khi ràng buộc thành x + y = 3.1.

**A14.** *(Chương 4 · ★★ · 2 điểm)* Adam một tham số, β₁ = 0.9, β₂ = 0.999, ε = 0, có bias correction, learning rate α. Gradient ở bước 1 là g₁ = 1, ở bước 2 là g₂ = −1. Độ dời của tham số ở bước 1 và ở bước 2 (viết theo α)? Một dòng nhận xét.

**A15.** *(Chương 5 · ★ · 2 điểm)* Bạn có 200 mẫu, 5 000 đặc trưng. Quy trình nào cho ước lượng hiệu năng **lạc quan sai**? Chọn một, nêu lý do một dòng.
(A) Trong mỗi fold của 5-fold CV, chọn 50 đặc trưng tương quan mạnh nhất với nhãn trên phần train của fold rồi khớp mô hình. (B) Chọn 50 đặc trưng tương quan mạnh nhất với nhãn trên toàn bộ 200 mẫu, rồi chạy 5-fold CV trên 50 đặc trưng đó. (C) Dùng nested CV: vòng trong chọn số đặc trưng, vòng ngoài báo hiệu năng. (D) Giữ riêng 40 mẫu làm test, mọi lựa chọn làm trên 160 mẫu còn lại.

**A16.** *(Chương 5 · ★★ · 2 điểm)* SVM tuyến tính với w = (1, −2), b = 0.5. Ba điểm: (2, 0) nhãn +1; (1, 1) nhãn −1; (0, −1) nhãn −1. Tổng hinge loss? Điểm nào bị phân loại sai, điểm nào phân loại đúng nhưng nằm trong lề?

**A17.** *(Chương 5 · ★★ · 2 điểm)* k-means một chiều, K = 2, dữ liệu (0, 1, 3, 7, 8, 12), tâm khởi tạo (0, 8). (a) Sau một vòng gán và cập nhật, hai tâm mới là gì? (b) Vòng tiếp theo có đổi phép gán không? Vì sao?

**A18.** *(Chương 6 · ★ · 2 điểm)* `nn.Sequential(Linear(300, 100), BatchNorm1d(100), ReLU(), Linear(100, 100), BatchNorm1d(100), ReLU(), Linear(100, 5))`, mọi Linear có bias. (a) Số tham số học được, và số giá trị running mean cộng running var. (b) Một dòng: vì sao bias của Linear ngay trước BatchNorm là thừa?

**A19.** *(Chương 6 · ★★ · 2 điểm)* Mạng sâu, mọi lớp có fan-in 400, ReLU, không bias, trọng số i.i.d. N(0, 0.01²). Đầu vào có các thành phần i.i.d., trung bình 0, phương sai 1. (a) Độ lệch chuẩn của pre-activation lớp 1. (b) Từ lớp 2 trở đi, mỗi lớp nhân độ lệch chuẩn của pre-activation với bao nhiêu? (c) Độ lệch chuẩn khởi tạo He cho fan-in này?

**A20.** *(Chương 6 · ★ · 2 điểm)* (a) Phân loại 5 lớp (5 class), mạng vừa khởi tạo cho logit gần 0: cross-entropy bước 0 xấp xỉ bao nhiêu? (b) Bạn đánh giá **cùng một** mô hình (đã train xong, không cập nhật thêm) trên **cùng** tập val hai lần liên tiếp, được 71.3% và 70.8%. Nguyên nhân hợp lý nhất: (A) quên `model.eval()` mà mô hình có dropout; (B) learning rate quá lớn; (C) quên `opt.zero_grad()`; (D) khởi tạo với seed khác.

### Phần C — Lập trình NumPy (2 bài, 20 điểm, phút 47–73, làm trên máy)

Chỉ dùng NumPy. Không vòng lặp Python theo mẫu hay theo lớp. Viết cả hai hàm vào `bai_lam.py` (chép từ `labs/thi-thu-1/bai_lam_mau.py`, đã có sẵn chữ ký và docstring dưới đây). Câu (b) của mỗi bài trả lời trên giấy.

**C1. Softmax + cross-entropy ổn định số.** *(Chương 2, 3 · 10 điểm · 13 phút)*

```python
def softmax_cross_entropy(Z, y):
    """Z: mảng (N, K) logit, float64. y: mảng (N,) nhãn nguyên trong [0, K).
    Trả về (loss, dZ):
      loss: số vô hướng, cross-entropy TRUNG BÌNH trên N mẫu;
      dZ:   mảng (N, K) = ∂loss/∂Z.
    Giữ float64. Không được sửa Z tại chỗ. Phải đúng cả khi logit cỡ ±1000."""
```

(a) *(8 điểm)* Viết hàm.

(b) *(2 điểm)* Không chạy máy: với Z = [[1000., 1000., 1000.]] và y = [2], (i) bản ngây thơ `np.exp(Z) / np.exp(Z).sum(1, keepdims=True)` cho ra gì, vì sao; (ii) hàm của bạn trả loss và dZ bằng bao nhiêu?

**C2. Một bước logistic regression có L2.** *(Chương 5, 4, 6 · 10 điểm · 14 phút)* Hàm mục tiêu với z = Xw + b (không phạt b):

$$
J(w, b) = \frac{1}{N}\sum_{i=1}^{N}\Big[\ln\big(1 + e^{z_i}\big) - y_i z_i\Big] + \frac{\lambda}{2}\|w\|_2^2 .
$$

```python
def logreg_step(w, b, X, y, lr, lam):
    """w: (d,), b: float, X: (N, d), y: (N,) nhãn 0/1, lr: learning rate, lam: λ.
    Trả về (w_moi, b_moi, J) với J tính tại (w, b) CŨ, sau đúng một bước gradient descent.
    Giữ float64. Phải đúng cả khi |z| cỡ 800 (không inf, không nan)."""
```

(a) *(8 điểm)* Viết hàm.

(b) *(2 điểm, ★★★)* Một bạn viết một bước **hồi quy tuyến tính** với MSE và phạt L2, J = mean((z − y)²) + (λ/2)‖w‖₂², như sau. Code đúng khi y có shape (N,), nhưng bạn ấy gọi hàm với y có shape (N, 1) (lấy bằng `df[["target"]].values`), N = 4, d = 3.

```python
z = X @ w + b                                     # (N,)
r = z - y
loss = np.mean(r ** 2) + 0.5 * lam * w @ w
db = 2 * np.mean(r)
dw = 2 * X.T @ r / len(y) + lam * w
```

Dòng nào báo lỗi, dòng nào chạy im lặng mà sai, dòng nào tình cờ vẫn đúng? Nếu N = d = 4 thì sao? Sửa ở đâu là gọn nhất?

### Phần B — Tự luận suy dẫn (5 bài, 40 điểm, phút 73–120)

Viết đủ các bước chính. Mỗi bài 8 điểm; điểm từng ý ghi trong ngoặc.

**B1. Phân rã hạng thấp.** *(Chương 2, 1, 4 · 8 điểm · 13 phút)* Cho M ∈ ℝᵐˣⁿ, U ∈ ℝᵐˣʳ, V ∈ ℝⁿˣʳ và hàm mất mát (chuẩn Frobenius)

$$
L(U, V) = \tfrac{1}{2}\,\big\|UV^{\top} - M\big\|_F^2 .
$$

Đây là bài toán gốc của matrix factorization và là dạng đơn giản nhất của việc học một cập nhật hạng thấp kiểu LoRA.

(a) *(3 điểm, ★★)* Suy dẫn ∂L/∂U và ∂L/∂V. Kiểm shape.

(b) *(2 điểm, ★★)* m = n = 2, r = 1, M có hàng (4, 1) và (1, 4), U = (1, 1) và V = (1, 0) (vector cột). Tính L, ∂L/∂U, ∂L/∂V; rồi U₁, V₁ sau một bước gradient descent đồng thời với η = 0.1 (cả U và V cập nhật từ gradient tại điểm cũ).

(c) *(3 điểm, ★★★)* (i) Với r = 1, giá trị nhỏ nhất của L trên mọi U, V là bao nhiêu, đạt tại ma trận UVᵀ nào? Nêu định lý bạn dùng. (2 điểm) (ii) Chứng minh L không lồi theo (U, V). *Gợi ý: hàm lồi khả vi có tính chất gì tại điểm dừng?* (1 điểm)

**B2. Unigram: MAP, MLE và perplexity.** *(Chương 3, 4 · 8 điểm · 10 phút)* Mô hình unigram trên từ vựng 4 token {a, b, c, d} có tham số θ = (θ₁, θ₂, θ₃, θ₄), θₖ ≥ 0, ∑θₖ = 1. Tập train có N = 10 token với số đếm n = (5, 3, 2, 0). Đặt prior Dirichlet đối xứng với α ≥ 1:

$$
p(\theta) \propto \prod_{k=1}^{4} \theta_k^{\,\alpha - 1}.
$$

(a) *(3 điểm, ★★)* Viết log posterior (bỏ hằng) và suy dẫn θ MAP bằng nhân tử Lagrange.

(b) *(3 điểm, ★★)* α nào cho lại MLE, α nào cho add-one smoothing? Tính θ MLE và θ MAP với α = 2. Với MLE, chuyện gì xảy ra với token có số đếm 0, và vì sao điểm tối ưu nằm ở biên?

(c) *(2 điểm, ★★★)* Chuỗi test là "a b d a". Tính cross-entropy (nat mỗi token) và perplexity của mô hình MLE và mô hình MAP với α = 2. So với mô hình đều (θₖ = ¼, perplexity 4) và giải thích trong hai câu.

**B3. Gradient descent trên hàm bậc hai.** *(Chương 4 · 8 điểm · 10 phút)*

$$
f(w) = \tfrac{1}{2}\, w^{\top} A w - b^{\top} w, \qquad A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}, \quad b = \begin{pmatrix} 4 \\ 0 \end{pmatrix}.
$$

Cho sẵn nghiệm w* = (1.5, −0.5) của Aw = b.

(a) *(2 điểm, ★)* Trị riêng và vector riêng của A; L, μ và κ.

(b) *(4 điểm, ★★)* GD: wₖ₊₁ = wₖ − η∇f(wₖ). Cho biết sai số eₖ = wₖ − w* thỏa eₖ₊₁ = (I − ηA)eₖ (vì ∇f(w) = A(w − w*)). Suy ra: điều kiện của η để hội tụ từ mọi w₀; η tối ưu và hệ số co tương ứng; với η tối ưu đó, số bước k nhỏ nhất bảo đảm ‖eₖ‖₂ ≤ 10⁻⁶‖e₀‖₂ với mọi w₀.

(c) *(2 điểm, ★★★)* Với η = 0.5 và w₀ = 0: tính w₁, w₂, mô tả quỹ đạo, và nói f(wₖ) có tiến về f* không. Giải thích bằng cách phân tích w₀ − w* theo hai vector riêng.

**B4. Bias–variance của ridge một tham số.** *(Chương 5, 3 · 8 điểm · 10 phút)* Dữ liệu sinh theo yᵢ = θxᵢ + εᵢ, i = 1, …, n, với xᵢ cố định (không ngẫu nhiên), εᵢ i.i.d. trung bình 0, phương sai σ². Đặt S = ∑xᵢ². Ước lượng ridge và công thức đóng của nó (đạo hàm theo w bằng 0):

$$
\hat{w}_\lambda = \arg\min_{w} \; \sum_{i=1}^{n} (y_i - w x_i)^2 + \lambda w^2 = \frac{\sum_{i} x_i y_i}{S + \lambda}, \qquad \lambda \ge 0 .
$$

(a) *(4 điểm, ★★)* Tính 𝔼[ŵ], bias, Var(ŵ) và MSE(λ) = 𝔼[(ŵ − θ)²] theo S, σ², θ, λ.

(b) *(3 điểm, ★★)* Tìm λ* cực tiểu MSE(λ). Với S = 10, σ² = 4, θ = 1: tính MSE(0), λ* và MSE(λ*).

(c) *(1 điểm, ★★★)* λ* phụ thuộc θ, thứ ta không biết. Thực tế chọn λ thế nào? Nối λ* với cách đọc ridge như MAP với prior w ~ N(0, τ²).

**B5. Backprop qua MLP 2–2–1 với batch 2.** *(Chương 6, 2 · 8 điểm · 13 phút)* Quy ước cột cho từng mẫu: a = W₁x + b₁, h = ReLU(a), z = w₂ᵀh + b₂, p = σ(z). Loss là binary cross-entropy trung bình trên batch:

$$
L = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i \ln p_i + (1 - y_i)\ln(1 - p_i)\Big].
$$

Tham số: W₁ có hàng (1, −1) và (0.5, 1); b₁ = (0, 0); w₂ = (1, −2); b₂ = 1. Batch N = 2: x⁽¹⁾ = (2, 1) với y₁ = 1; x⁽²⁾ = (−1, 1) với y₂ = 0. Dùng bảng hằng số.

(a) *(2 điểm, ★)* Forward: a, h, z, p của từng mẫu và L.

(b) *(4 điểm, ★★)* Backward. Nêu vì sao ∂L/∂zᵢ = (pᵢ − yᵢ)/N, rồi tính ∂L/∂w₂ và ∂L/∂W₁ (chấp nhận lệch 0.01 do làm tròn).

(c) *(2 điểm, ★★★)* (i) Mẫu 2 đóng góp gì vào hàng 1 của ∂L/∂W₁, vì sao? Nếu điều đó đúng với mọi mẫu của tập train thì neuron ẩn 1 sẽ thế nào trong suốt quá trình huấn luyện? (1 điểm) (ii) Bạn kiểm (b) bằng gradient check: viết công thức sai phân trung tâm cho một phần tử của W₁ và giải thích vì sao phải tránh điểm có một pre-activation aⱼ⁽ⁱ⁾ rất gần 0. (1 điểm)

## Bảng chấm theo chương

Điểm của mỗi câu được chia về chương theo cột "Câu (điểm)". Ghi **điểm cuối** vào cột "Điểm của bạn" rồi tra mức. Ngưỡng theo chương là heuristic của sách: vững từ 75% điểm tối đa, cần ôn từ 50% đến dưới 75%, yếu dưới 50%; chưa hiệu chỉnh trên người làm thật. Tổng điểm có ngưỡng riêng, ở bảng "Theo tổng điểm cuối".

| Chương | Câu (điểm) | Tối đa | Điểm của bạn | Vững | Cần ôn | Yếu | Mục không được hỏi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A1–A5 (10), B1(c)(i) (2) | 12 | | 9–12 | 6–8 | 0–5 | QR và Gram–Schmidt (1.3), PCA (1.5) |
| 2 | A6–A8 (6), B1(a)(b) (5), B5(c)(ii) (1), C1 shape, không sửa Z và giá trị (5) | 17 | | 13–17 | 9–12 | 0–8 | Forward và reverse mode, VJP (2.4) chỉ hỏi gián tiếp |
| 3 | A9–A11 (6), B2 (8), C1 ổn định số và C1(b) (5) | 19 | | 15–19 | 10–14 | 0–9 | Kỳ vọng và phương sai (3.2) chỉ dùng làm công cụ trong B4, điểm B4 tính cho chương 5; Gauss nhiều chiều, Mahalanobis (3.3); McNemar, đa kiểm định (3.5B) |
| 4 | A12–A14 (6), B3 (8), B1(c)(ii) (1), C2 không phạt b (1) | 16 | | 12–16 | 8–11 | 0–7 | SGD, nhiễu minibatch (4.3); momentum, AdamW, lịch learning rate (4.4) chỉ qua A14; Newton (4.5) |
| 5 | A15–A17 (6), B4 (8), C2 giá trị và ổn định số (5) | 19 | | 15–19 | 10–14 | 0–9 | Metric (precision, recall, AUC) (5.2); cây và ensemble (5.4); GMM và EM (5.5) |
| 6 | A18–A20 (6), B5(a)(b)(c)(i) (7), C2 shape và C2(b) (4) | 17 | | 13–17 | 9–12 | 0–8 | Vòng lặp PyTorch (6.3) chỉ qua A20; LayerNorm, dropout (6.4) |
| **Tổng** | | **100** | | Xem bảng "Theo tổng điểm cuối" | | | |

Cách chia điểm Phần C về chương (bộ chấm in sẵn phần tự động theo chương): C1 gồm shape 1 điểm, không sửa Z tại chỗ 1 điểm, giá trị kèm gradient 3 điểm (chương 2); ổn định số 3 điểm và câu (b) 2 điểm (chương 3). C2 gồm shape 2 điểm và câu (b) 2 điểm (chương 6, lỗi im lặng); giá trị 3 điểm và ổn định số 2 điểm (chương 5); không phạt b 1 điểm (chương 4).

**Điểm ★★★.** Cộng riêng các ý ★★★: B1(c), B2(c), B3(c), B4(c), B5(c), C2(b), tối đa 12 điểm. Con số này không dùng để quyết định ôn chương nào; nó đo khả năng đi xa hơn đáp số, thứ technical interview hay hỏi.

## Đọc kết quả

**Quy tắc một dòng.** Tra bằng điểm cuối; điểm trong giờ chỉ đọc tốc độ. Chương yếu thì học lại theo mục trước khi sang chương phụ thuộc vào nó; chương cần ôn thì ôn đúng câu sai, không đọc lại cả chương.

**Theo tổng điểm cuối** [ngưỡng là heuristic của sách]. Có ít nhất một chương "yếu" thì dùng dòng cuối, bất kể tổng.

| Tổng | Việc nên làm |
| --- | --- |
| 75–100, không chương nào yếu | Sang Phần II đúng lịch (tuần 14, chương 7, bắt đầu 1/2/2027). Dành 1–2 giờ trong tuần 14 làm lại các câu sai theo bảng Câu sai → ôn gì ở dưới. |
| 60–74, không chương nào yếu | Sang Phần II đúng lịch, nhưng ôn có mục tiêu: mỗi chương "cần ôn" khoảng 2 giờ, rải trong tuần 14–15. Lịch hai tuần này đã 14.5 và 15 giờ, sát trần 12–15 giờ, nên phải lấy giờ từ đâu đó; cách ít hại nhất theo người soạn là giảm luồng B từ 3 bài xuống 2 bài mỗi tuần trong hai tuần đó. Đây là đề xuất, bạn quyết. |
| 0–59, hoặc có chương "yếu" | Cân nhắc lùi chương 7 một tuần và dùng tuần đó học lại chương yếu theo mục (đọc lại mục, tự làm lại ví dụ giải mẫu, làm bài ★ rồi ★★, làm lại mini-test). Lùi một tuần kéo theo cả lịch Phần II và Đề thi thử 2 (18/4/2027); quyết định thuộc về bạn. |

**Chương yếu kéo theo chương nào** (cùng logic với bài chẩn đoán). Các việc ôn dưới đây là giờ **vượt kế hoạch** nếu không lấy từ luồng B như dòng 60–74 ở trên.

- Chương 6 yếu: đừng bắt đầu chương 7 trước khi làm lại mini-test chương 6 và đạt từ 24/30; lab chương 7 trở đi dùng PyTorch và giả định bạn debug được vòng huấn luyện.
- Chương 1 hoặc 2 yếu: chương 8 (attention) cần nhân ma trận, softmax và Jacobian của nó, và theo `plan/weeks.yaml` bắt đầu ngày 15/2/2027, tức chỉ 2 tuần sau đề này; ôn rải 30 phút mỗi ngày bằng bài tính nhanh chương 1–2 trong hai tuần chương 7 (khoảng 3.5 giờ mỗi tuần), thay vì dừng lịch.
- Chương 3 yếu: chương 9–11 dùng cross-entropy, perplexity, CI liên tục; ôn mục 3.4–3.5 trước chương 9.
- Chương 4 yếu: các lab Phần II huấn luyện bằng Adam, và chương 8 dùng lịch learning rate có warmup; ôn mục 4.2 (điều kiện ổn định của learning rate) và 4.4 trước chương 8.
- Chương 5 yếu: chương 11 nâng metric và leakage lên mức thiết kế thí nghiệm; ôn mục 5.2 trước chương 11.

**Tốc độ.** Nếu điểm cuối trừ điểm trong giờ từ 15 điểm trở lên [heuristic; với độ dài đề này, người học mục tiêu dự kiến chênh khoảng 5–10 điểm], vấn đề chính là tốc độ chứ không phải kiến thức: mỗi ngày làm một bộ bài tính nhanh bấm giờ (khoảng 25 phút, giờ vượt kế hoạch như trên; chọn chương có nhiều nhãn H), và tập viết lời giải Phần B theo khung "công thức → thế số → kiểm shape" để không phải nghĩ cách trình bày lúc thi. Nếu điểm trong giờ dưới 50 nhưng điểm cuối từ 75, bạn đủ kiến thức mà chưa đủ nhanh cho một bài thi bấm giờ chặt; Đề thi thử 2 sẽ đo lại.

**Theo nhãn lỗi.** Nhiều **K** ở một chương: đọc lại mục theo bảng dưới, làm bài ★. Nhiều **T** rải nhiều chương: không cần đọc lại, làm bài tính nhanh 15 phút mỗi ngày và viết đủ bước trung gian. Nhiều **L**: đọc chậm đề, gạch chân điều kiện (trung bình hay tổng, nat hay bit, có phạt b không). Nhiều **H**: xem mục Tốc độ.

**Câu sai → ôn gì.** "Bài ★2" nghĩa là bài 2 trong mục Bài tập giấy bút của chương đó; "tính nhanh 3" là bài 3 trong mục Bài tính nhanh.

| Câu | Đọc lại | Làm lại |
| --- | --- | --- |
| A1 | Mục 1.1, ví dụ 1.2; định thức ở mục 1.4 | Tính nhanh chương 1: 1–4 |
| A2 | Mục 1.2, ví dụ 1.3 | Chương 1: bài ★1, ★★5; tính nhanh 6; mini-test câu 1 |
| A3 | Mục 1.3, đoạn "Phép chiếu" (chiếu lên đường thẳng); ví dụ 1.5 | Chương 1: bài ★★6; tính nhanh 9 |
| A4 | Mục 1.4, ví dụ 1.6 | Chương 1: bài ★4; tính nhanh 10–12; mini-test câu 4 |
| A5 | Mục 1.5, ví dụ 1.7 | Chương 1: bài ★★8; tính nhanh 13 |
| A6 | Mục 2.1–2.2, ví dụ 2.1 | Chương 2: bài ★1, ★4; tính nhanh 1–5 |
| A7 | Mục 2.3, ví dụ 2.7 | Chương 2: bài ★★9; mini-test câu 2 |
| A8 | Mục 2.5, ví dụ 2.9 | Chương 2: bài ★★★11; tính nhanh 8, 10 |
| A9 | Mục 3.1, ví dụ 3.1 | Chương 3: bài ★★5; mini-test câu 1 |
| A10 | Mục 3.5A, ví dụ 3.10 | Chương 3: bài ★4, ★★10; mini-test câu 5 |
| A11 | Mục 3.5B, ví dụ 3.11 | Chương 3: tính nhanh 14 |
| A12 | Mục 4.1 | Chương 4: bài ★1, ★2; Kiểm tra nhanh câu 1 |
| A13 | Mục 4.1, ví dụ 4.2 | Chương 4: bài ★★5; mini-test câu 2 |
| A14 | Mục 4.4, ví dụ 4.9 | Chương 4: tính nhanh 10–11; mini-test câu 4 |
| A15 | Mục 5.2 (CV và leakage, đoạn Thí nghiệm nhỏ) | Chương 5: Bài săn bug; mini-test câu 5 |
| A16 | Mục 5.3, ví dụ 5.8 | Chương 5: bài ★3; tính nhanh 8–9 |
| A17 | Mục 5.5, ví dụ 5.13 | Chương 5: tính nhanh 15 |
| A18 | Mục 6.1, ví dụ 6.5; mục 6.4, ví dụ 6.9 | Chương 6: bài ★1; tính nhanh 4 |
| A19 | Mục 6.4, ví dụ 6.8 | Chương 6: bài ★★7; mini-test câu 3 |
| A20 | Mục 6.5, ví dụ 6.11 và danh mục lỗi im lặng | Chương 6: mini-test câu 5, 6 |
| C1 | Mục 2.5, ví dụ 2.9; mục 3.3, ví dụ 3.5 | Viết lại hàm không nhìn lời giải sau 3 ngày |
| C2 | Mục 5.1, ví dụ 5.2; mục 6.5 (lỗi im lặng) | Chương 5: bài ★★5; viết lại hàm sau 3 ngày |
| B1 | Mục 2.2 (phương pháp vi phân cộng vết); Eckart–Young ở mục 1.5 | Chương 2: bài ★★7, ★★8; chương 1: bài ★★★13; chương 4: bài ★★★13 |
| B2 | Mục 3.4, ví dụ 3.7 (đoạn add-one); Lagrange ở mục 4.1 | Chương 3: bài ★★7, mini-test câu 4; chương 4: bài ★★10 |
| B3 | Mục 4.2, ví dụ 4.4 | Chương 4: bài ★3, ★★7; mini-test câu 3 |
| B4 | Mục 5.2, ví dụ 5.4; ví dụ 3.8 (ridge là MAP) | Chương 5: bài ★★6, ★★8 |
| B5 | Mục 2.5, ví dụ 2.10 và 2.11; mục 6.2, ví dụ 6.4 | Chương 6: bài ★3; mini-test câu 1 |

**Đo lại.** Sách không có bản thứ hai của đề này. Mini-test từng chương đo lại chương đó; Đề thi thử 2 (18/4/2027) phủ Phần II và có một phần nhỏ nhắc lại Phần I [dự kiến, đề 2 chưa viết]. Ghi điểm từng chương, điểm trong giờ và điểm cuối vào trang Tiến độ để so.

## Lời giải

Chỉ xem sau khi đã làm hết đề và tự chấm Phần C bằng bộ chấm. Thang điểm ghi ở cuối mỗi lời giải. Lời giải xếp theo thứ tự làm bài: A, C, B.

### Phần A

**A1.** Cột của A là ảnh của e₁, e₂, nên A có hàng (2, −1) và (1, 3). (a) T(1, 2) = 1·(2, 1) + 2·(−1, 3) = (0, 7). (b) Diện tích bằng |det A| = |2·3 − (−1)·1| = 7. *Thang:* 1 điểm mỗi ý.

**A2.** Hàng 3 = hàng 1 + hàng 2, hai hàng đầu không tỉ lệ, nên hạng 2: ba vector sinh ra một mặt phẳng (2 chiều). w ∈ N(A): w₁ + w₂ = 0 và w₂ + w₃ = 0 cho w = (1, −1, 1); kiểm hàng 3: 1 − 2 + 1 = 0. Ý nghĩa: cộng bội của w vào trọng số không đổi đầu ra trên ba mẫu này, nên dữ liệu không xác định được w duy nhất. *Thang:* 1 điểm hạng, 1 điểm vector null (bội bất kỳ khác 0 đều đúng).

**A3.** aᵀb = 3 + 0 + 6 = 9, aᵀa = 9, nên hình chiếu là (9/9)a = (1, 2, 2). Phần dư r = b − Pb = (2, −2, 1), kiểm aᵀr = 2 − 4 + 2 = 0; khoảng cách ‖r‖₂ = √9 = 3. *Thang:* 1 điểm hình chiếu, 1 điểm khoảng cách.

**A4.** (a) tr A = 3, det A = 2 − 6 = −4: λ² − 3λ − 4 = 0, λ = 4 và λ = −1. (b) A² − A có cùng vector riêng, trị riêng λ² − λ: 16 − 4 = 12 và 1 + 1 = 2. det(A⁻¹) = 1/det A = −1/4. *Thang:* 1 điểm (a), 1 điểm (b) đủ cả hai số.

**A5.** Hai cột trực giao: 1·4 + 2·(−4) + 2·2 = 0. Khi đó AᵀA = diag(‖a₁‖², ‖a₂‖²) = diag(9, 36), nên giá trị kỳ dị là độ dài các cột: σ₁ = 6, σ₂ = 3. κ(A) = 6/3 = 2. Chuẩn Frobenius √(36 + 9) = √45 = 3√5 ≈ 6.71. *Thang:* 1 điểm σ₁, σ₂ kèm lý do cột trực giao; 1 điểm κ và chuẩn Frobenius.

**A6.** Đặt u = aᵀx; f = ln(1 + eᵘ) có df/du = σ(u). Quy tắc chuỗi: ∇f = σ(aᵀx)a. Tại x = (1, 1, 0): aᵀx = 1 − 1 + 0 = 0, σ(0) = 0.5, ∇f = (0.5, −0.5, 1). Hessian = σ(u)(1 − σ(u))aaᵀ = 0.25aaᵀ, có hàng (0.25, −0.25, 0.5), (−0.25, 0.25, −0.5), (0.5, −0.5, 1); hạng 1 (tích ngoài). *Thang:* 1 điểm gradient, 1 điểm Hessian và hạng.

**A7.** Nút m = max(x, y) = 3, nút d = x − y = 2, f = m·d = 6. Backward: ∂f/∂m = d = 2, ∂f/∂d = m = 3. Cổng max chuyển gradient cho đầu vào lớn hơn (x): x nhận 2 từ nhánh max. Cả x và y đều fan-out (vào cả max và hiệu), nên cộng hai nhánh: ∂f/∂x = 2·1 + 3·1 = 5; ∂f/∂y = 2·0 + 3·(−1) = −3. *Thang:* 1 điểm f và ∂f/∂x; 1 điểm ∂f/∂y kèm chỉ ra fan-out và cổng max chặn nhánh y.

**A8.** ∂pᵢ/∂zⱼ = pᵢ(δᵢⱼ − pⱼ), tức Jacobian = diag(p) − ppᵀ. Tại z = 0, p = (⅓, ⅓, ⅓): đường chéo ⅓ − 1/9 = 2/9 ≈ 0.222, ngoài đường chéo −1/9 ≈ −0.111. Tổng hàng i là tổng theo j của ∂pᵢ/∂zⱼ, bằng pᵢ(1 − ∑ⱼpⱼ) = 0, vì cộng cùng một hằng vào mọi logit (hướng (1, 1, 1)) không đổi softmax. *Thang:* 1 điểm hai số; 1 điểm lý do.

**A9.** (a) P = 0.1·0.8 / (0.1·0.8 + 0.9·0.1) = 0.08/0.17 = 8/17 ≈ 0.47. (b) 0.5·0.8 / (0.5·0.8 + 0.5·0.1) = 0.4/0.45 = 8/9 ≈ 0.89. Cùng một bộ phát hiện nhưng độ tin của một lần gắn cờ phụ thuộc mạnh vào tỉ lệ nền. *Thang:* 1 điểm mỗi ý.

**A10.** H(p) = 2·½·1 = 1 bit. H(p, q) = −∑pᵢ log₂ qᵢ = 2·½·2 = 2 bit. KL(p‖q) = 2 − 1 = 1 bit. KL(q‖p) = +∞: q gán xác suất ¼ cho hai giá trị mà p gán 0, nên có số hạng ¼·log₂(¼/0). KL không đối xứng; trong ML điều này giải thích vì sao mô hình gán xác suất 0 cho thứ có thật bị phạt vô hạn khi tính cross-entropy theo dữ liệu. *Thang:* 1 điểm ba số đầu; 1 điểm KL(q‖p) = ∞ kèm lý do support.

**A11.** (a) SE = √(0.8·0.2/625) = 0.4/25 = 0.016. Nửa độ rộng 1.96·0.016 = 0.0314, CI ≈ [0.769, 0.831]. (b) Nửa độ rộng tỉ lệ với 1/√n, nên giảm một nửa cần n gấp 4: 2 500 câu. *Thang:* 1 điểm (a), 1 điểm (b).

**A12.** (D). Hessian của x₁x₂ có hàng (0, 1) và (1, 0), trị riêng ±1, không nửa xác định dương: dọc hướng (1, −1) hàm là −t², cong xuống. (A) có đạo hàm bậc hai σ(x)(1 − σ(x)) > 0; (B) là chuẩn, lồi theo bất đẳng thức tam giác; (C) là max của hai hàm lồi nên lồi (bẫy hay gặp: nghĩ "có góc" là không lồi). *Thang:* 1 điểm chọn đúng, 1 điểm lý do.

**A13.** (a) ∇f = (2x, 4y) = ν(1, 1) cho x = 2y; thế vào x + y = 3: y = 1, x = 2, f* = 4 + 2 = 6, ν = 2x = 4. (b) Nhân tử là độ nhạy của giá trị tối ưu theo vế phải: f* tăng khoảng ν·0.1 = 0.4, tức f* ≈ 6.4 (đúng chính xác 2·3.1²/3 ≈ 6.407). *Thang:* 1 điểm (a) đủ nghiệm, f*, ν; 1 điểm (b).

**A14.** Bước 1: m₁ = 0.1, v₁ = 0.001, m̂₁ = 1, v̂₁ = 1, độ dời −α. Bước 2: m₂ = 0.9·0.1 + 0.1·(−1) = −0.01, chia 1 − 0.9² = 0.19 cho m̂₂ = −1/19; v₂ = 0.999·0.001 + 0.001·1 = 0.001999, chia 1 − 0.999² = 0.001999 cho v̂₂ = 1. Độ dời −α·m̂₂/√v̂₂ = +α/19 ≈ 0.053α. Nhận xét: khi gradient đổi dấu, moment bậc nhất tự triệt tiêu nên bước nhỏ lại khoảng 19 lần; moment bậc hai không đổi dấu nên vẫn bằng 1. Đó là cách Adam giảm bước ở hướng dao động. *Thang:* 1 điểm bước 1 và m̂₂, v̂₂; 1 điểm độ dời α/19 đúng dấu kèm nhận xét.

**A15.** (B). Chọn đặc trưng bằng nhãn của toàn bộ dữ liệu là đã nhìn nhãn của các fold kiểm tra: với 5 000 đặc trưng ngẫu nhiên và 200 mẫu, luôn có 50 đặc trưng tương quan tình cờ với nhãn, và CV sau đó báo hiệu năng cao giả. (A), (C), (D) đều giữ phần đánh giá tách khỏi mọi lựa chọn. *Thang:* 1 điểm chọn đúng, 1 điểm lý do (nhãn của phần đánh giá đã được dùng).

**A16.** f(x) = wᵀx + b: (2, 0) → 2.5; (1, 1) → 1 − 2 + 0.5 = −0.5; (0, −1) → 2 + 0.5 = 2.5. Margin y·f: 2.5, 0.5, −2.5. Hinge max(0, 1 − margin): 0, 0.5, 3.5; tổng 4. Điểm (0, −1) bị phân loại sai (margin âm); điểm (1, 1) đúng nhưng nằm trong lề (0 < margin < 1). *Thang:* 1 điểm tổng hinge; 1 điểm phân loại đúng hai điểm.

**A17.** (a) Gán: 0, 1, 3 gần 0 hơn (3 cách 0 là 3, cách 8 là 5); 7, 8, 12 gần 8. Tâm mới 4/3 ≈ 1.333 và 9. (b) Vòng sau: 3 cách 4/3 là 1.67, cách 9 là 6; 7 cách 9 là 2, cách 4/3 là 5.67; các điểm còn lại càng rõ hơn. Không đổi gán, thuật toán dừng. (Không bắt buộc: J tại tâm mới bằng 42/9 + 14 = 56/3 ≈ 18.67.) *Thang:* 1 điểm hai tâm; 1 điểm kết luận không đổi kèm so khoảng cách của các điểm biên 3 và 7.

**A18.** (a) Linear: 300·100 + 100 = 30 100; 100·100 + 100 = 10 100; 100·5 + 5 = 505. BatchNorm: γ và β, 2·100 mỗi lớp, tổng 400. Tham số học được 30 100 + 10 100 + 505 + 400 = 41 105. Running mean và running var: 2·100 mỗi lớp, tổng 400 (buffer, không học bằng gradient; PyTorch còn một bộ đếm `num_batches_tracked` mỗi lớp). (b) BatchNorm trừ trung bình theo batch của từng đặc trưng, nên một hằng cộng thêm (bias) bị trừ mất; β của BatchNorm làm đúng việc đó. Vì vậy thường đặt `bias=False` cho Linear hay Conv đứng trước BN. *Thang:* 1 điểm (a) cả hai số; 1 điểm (b).

**A19.** Var(z) = fan-in · Var(w) · 𝔼[đầu vào²]. (a) Lớp 1: 400·10⁻⁴·1 = 0.04, độ lệch chuẩn 0.2. (b) Sau ReLU của một biến đối xứng tâm 0, 𝔼[h²] = Var(z)/2, nên Var(zₗ₊₁) = 400·10⁻⁴·Var(zₗ)/2 = 0.02·Var(zₗ): độ lệch chuẩn nhân √0.02 ≈ 0.141 mỗi lớp (sau 10 lớp nhân (√0.02)¹⁰ = 0.02⁵ ≈ 3·10⁻⁹, tín hiệu chết). (c) He: √(2/400) ≈ 0.0707, khi đó hệ số mỗi lớp đúng bằng 1. *Thang:* 1 điểm cho (a) và (b) cùng đúng (quên chia 2 ở (b) là mất điểm này); 1 điểm (c).

**A20.** (a) Logit gần 0 thì softmax gần đều trên 5 class, loss ≈ ln 5 ≈ 1.609. (b) (A). Ở chế độ train, dropout tắt ngẫu nhiên neuron ở mỗi lần forward, nên cùng input cho kết quả khác nhau; (C) và (B) chỉ ảnh hưởng khi có bước cập nhật, (D) không đổi gì khi mô hình đã cố định. *Thang:* 1 điểm mỗi ý.

### Phần C

**C1.** (a) Lời giải mẫu (chỉ có ở đây và trong `work/mock1/check_partC.py`; thư mục `labs/thi-thu-1` không chứa lời giải):

```python
import numpy as np

def softmax_cross_entropy(Z, y):
    N = Z.shape[0]
    Zs = Z - Z.max(axis=1, keepdims=True)                 # trừ max theo hàng: chặn tràn, không đổi softmax
    lse = np.log(np.exp(Zs).sum(axis=1, keepdims=True))   # log-sum-exp của Zs
    logP = Zs - lse                                       # log-softmax, không bao giờ là log(0)
    loss = -logP[np.arange(N), y].mean()
    dZ = np.exp(logP)                                     # softmax
    dZ[np.arange(N), y] -= 1.0                            # P − one-hot
    return loss, dZ / N                                   # chia N vì loss là trung bình
```

Các chỗ hay sai:

- Trừ max của cả ma trận thay vì từng hàng. Về toán vẫn đúng (cộng một hằng vào mọi logit không đổi softmax), nhưng hàng nào thấp hơn hàng cao nhất trên khoảng 700 sẽ có mọi exp tràn dưới về 0, ra log 0.
- Thiếu `keepdims=True` khi trừ max theo hàng: `Z.max(axis=1)` có shape (N,), broadcast theo trục cuối của (N, K), nên báo lỗi khi N ≠ K và **sai im lặng** khi N = K.
- Tính `np.log(softmax)`: ra −inf khi xác suất làm tròn về 0.
- Quên chia N trong dZ.
- Viết `Z -= Z.max(...)`: sửa tại chỗ đầu vào của người gọi. `Zs = Z - ...` tạo mảng mới nên Z không đổi.

*Thang (a), 8 điểm:* 1 điểm shape trả về (loss vô hướng, dZ cùng shape Z); 1 điểm không sửa Z tại chỗ; 3 điểm đúng giá trị và gradient so với tham chiếu; 3 điểm ổn định số (logit ±1000 và ±10⁴, các hàng lệch nhau xa, không ra nan hay inf, loss vẫn đúng). Kết quả phải giữ float64: dung sai 10⁻¹⁰ chủ ý đòi float64, nên đổi sang float32 sẽ trượt cả phần giá trị lẫn ổn định số. Trừ 2 điểm nếu có vòng lặp Python theo mẫu hay theo lớp (điểm (a) không âm). Phần 8 điểm này do `labs/thi-thu-1/cham_phan_c.py` chấm; mỗi lỗi chỉ bị trừ ở đúng một mục (ví dụ quên chia N chỉ mất 3 điểm giá trị).

(b) (i) exp(1000) tràn thành inf (giới hạn float64 cỡ e mũ 709), nên mỗi phần tử là inf/inf = nan. (ii) Ba logit bằng nhau nên softmax đều ⅓: loss = −ln ⅓ = ln 3 ≈ 1.0986; dZ = (⅓, ⅓, ⅓ − 1)/1 = (0.333, 0.333, −0.667). *Thang:* 1 điểm mỗi ý.

**C2.** (a) Lời giải mẫu:

```python
import numpy as np

def logreg_step(w, b, X, y, lr, lam):
    N = X.shape[0]
    z = X @ w + b                                         # (N,)
    loss = np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * lam * (w @ w)
    p = 0.5 * (1.0 + np.tanh(0.5 * z))                    # sigmoid ổn định, tanh không tràn
    r = p - y                                             # ∂(log-loss)/∂z của từng mẫu
    dw = X.T @ r / N + lam * w                            # có phạt L2
    db = r.mean()                                         # không phạt b
    return w - lr * dw, b - lr * db, loss
```

Vì sao như vậy: ln(1 + eᶻ) viết thẳng sẽ tràn khi z ≈ 800 (eᶻ = inf), còn `np.logaddexp(0, z)` tính ổn định. Dạng `ln(1 + eᶻ) − y·z` gộp sẵn hai trường hợp y = 0 và y = 1, nên không phải tính ln p hay ln(1 − p) (dễ ra ln 0). σ(z) = ½(1 + tanh(z/2)) là đồng nhất thức đúng và không tràn; cách khác hợp lệ là tách theo dấu của z. Gradient theo z của log-loss là p − y (cùng phép tính như B5).

*Thang (a), 8 điểm:* 2 điểm shape (w mới có shape (d,), b và J là số); 3 điểm J, dw, db đúng so với tham chiếu; 1 điểm không phạt b; 2 điểm ổn định với |z| = 800. Kết quả phải giữ float64. Trừ 2 điểm nếu có vòng lặp Python theo mẫu (điểm (a) không âm). Phần 8 điểm này do `labs/thi-thu-1/cham_phan_c.py` chấm; các test ổn định số dùng từng mẫu một, nên lấy tổng thay trung bình chỉ mất 3 điểm giá trị, và phạt cả b chỉ mất 1 điểm "không phạt b".

(b) Với y shape (4, 1) và z shape (4,): `r = z - y` broadcast thành bảng (4, 4) với phần tử zⱼ − yᵢ, không báo lỗi.

- Dòng `loss`: chạy im lặng mà **sai**. Trung bình của bảng (zⱼ − yᵢ)² bằng mean(z²) − 2·mean(z)·mean(y) + mean(y²), còn bản đúng là mean(z²) − 2·mean(y·z) + mean(y²); hai số chỉ trùng khi y và z không tương quan trên batch.
- Dòng `db`: chạy im lặng và **tình cờ đúng**: 2·mean(zⱼ − yᵢ) trên cả bảng bằng 2(mean z − mean y).
- Dòng `dw`: `X.T @ r` có shape (3, 4), cộng `lam * w` shape (3,) thì broadcast so chiều cuối 4 với 3 nên **báo lỗi** ValueError.
- Nếu N = d = 4: phép cộng ở dòng `dw` lại broadcast được, không có dòng nào báo lỗi; dw thành ma trận (4, 4) và `w - lr * dw` biến w thành ma trận (4, 4). Mọi thứ chạy tiếp, chỉ có số sai. Đây là kiểu lỗi trong danh mục lỗi im lặng của mục 6.5.

Sửa gọn nhất: ở đầu hàm, `y = y.reshape(-1)` (hoặc `assert y.shape == (X.shape[0],)` để lỗi lộ ra sớm). Lỗi này giống hệt nếu thay MSE bằng log-loss của C2(a): mọi phép trừ hay nhân giữa vector (N,) và cột (N, 1) đều thành bảng (N, N). *Thang:* 1 điểm chỉ đúng ba dòng (sai im lặng, tình cờ đúng, báo lỗi); 1 điểm trường hợp N = d và cách sửa.

**Chạy bộ chấm.** Sau khi hết giờ, từ thư mục gốc của sách: `python labs/thi-thu-1/cham_phan_c.py --student bai_lam.py`. Bộ chấm chỉ cần NumPy: tham chiếu là loss viết thẳng theo định nghĩa trên dữ liệu logit nhỏ, gradient tham chiếu lấy bằng sai phân trung tâm, còn các ca logit cực lớn so với đáp số tính tay. Nó in điểm tự động trên 16, chia sẵn theo chương, và nhắc nếu thấy vòng lặp trong hàm; cộng tay 4 điểm của C1(b), C2(b) và trừ điểm vòng lặp nếu đó là vòng lặp theo mẫu hay theo lớp.

### Phần B

**B1.** (a) Đặt R = UVᵀ − M. Vi phân: dL = tr(Rᵀ dR), dR = dU Vᵀ + U dVᵀ.

$$
dL = \operatorname{tr}\big(R^{\top} dU\, V^{\top}\big) + \operatorname{tr}\big(R^{\top} U\, dV^{\top}\big) = \operatorname{tr}\big((RV)^{\top} dU\big) + \operatorname{tr}\big((R^{\top}U)^{\top} dV\big),
$$

dùng hoán vị vòng của vết (số hạng đầu: tr(VᵀRᵀdU)) và tr(Xᵀ) = tr X (số hạng sau: tr(RᵀU dVᵀ) = tr(dV UᵀR)). Đọc ra

$$
\frac{\partial L}{\partial U} = R V \in \mathbb{R}^{m\times r}, \qquad \frac{\partial L}{\partial V} = R^{\top} U \in \mathbb{R}^{n\times r}.
$$

Shape: R là m × n, nên RV là m × r (khớp U) và RᵀU là n × r (khớp V). *Thang:* 1 điểm mỗi gradient có lập luận (vi phân hoặc từng phần tử); 1 điểm kiểm shape tường minh cho cả hai.

(b) UVᵀ có hàng (1, 0) và (1, 0); R có hàng (−3, −1) và (0, −4); L = ½(9 + 1 + 0 + 16) = 13. ∂L/∂U = RV = (−3, 0) (cột đầu của R, vì V = e₁). ∂L/∂V = RᵀU = tổng các hàng của R = (−3, −5). Bước GD: U₁ = (1, 1) − 0.1(−3, 0) = (1.3, 1.0); V₁ = (1, 0) − 0.1(−3, −5) = (1.3, 0.5). (Không bắt buộc: L tại (U₁, V₁) là 8.8993, giảm từ 13.) *Thang:* 1 điểm L và hai gradient; 1 điểm U₁, V₁.

(c)(i) Với r = 1, UVᵀ có hạng tối đa 1, nên min L = ½ · min ‖X − M‖² trên mọi X hạng ≤ 1 (chuẩn Frobenius). Theo Eckart–Young, cực tiểu đạt tại phần cắt SVD hạng 1, sai số bình phương bằng σ₂². M đối xứng, trị riêng 5 (vector (1, 1)/√2) và 3 (vector (1, −1)/√2), đều dương nên σ₁ = 5, σ₂ = 3. Vậy min L = ½·3² = 4.5, đạt tại UVᵀ = 5·qqᵀ với q = (1, 1)/√2, tức ma trận có mọi phần tử 2.5 (ví dụ U = (1, 1), V = (2.5, 2.5)). Chạy GD tiếp từ điểm ở (b) hội tụ đúng về đó. *Thang:* 1 điểm giá trị 4.5 kèm tên định lý; 1 điểm ma trận tối ưu.

(c)(ii) Với hàm lồi khả vi, mọi điểm dừng là cực tiểu toàn cục. Tại U = 0, V = 0: R = −M nên RV = 0 và RᵀU = 0, đây là điểm dừng, nhưng L = ½‖M‖² = ½(16 + 1 + 1 + 16) = 17 > 4.5. Mâu thuẫn, nên L không lồi. (Cách khác: L(cU, V/c) = L(U, V) với mọi c ≠ 0, nên tập nghiệm tối ưu không lồi.) *Thang:* 1 điểm cho lập luận đầy đủ.

**B2.** (a) Log posterior = log-likelihood + log prior = ∑ₖ nₖ ln θₖ + ∑ₖ (α − 1) ln θₖ + hằng = ∑ₖ (nₖ + α − 1) ln θₖ + hằng. Lagrangian ∑ₖ (nₖ + α − 1) ln θₖ − ν(∑θₖ − 1). Với nₖ + α − 1 > 0: (nₖ + α − 1)/θₖ = ν, nên θₖ = (nₖ + α − 1)/ν; cộng lại cho ν = N + K(α − 1):

$$
\hat{\theta}_k^{\text{MAP}} = \frac{n_k + \alpha - 1}{N + K(\alpha - 1)} .
$$

*Thang:* 1 điểm log posterior; 1 điểm lập Lagrangian và điều kiện bậc nhất; 1 điểm công thức.

(b) α = 1 (prior đều) cho lại MLE θ̂ₖ = nₖ/N = (0.5, 0.3, 0.2, 0). α = 2 cho (nₖ + 1)/(N + K), tức add-one: θ = (6, 4, 3, 1)/14 ≈ (0.429, 0.286, 0.214, 0.071). Với MLE và n₄ = 0, hệ số của ln θ₄ bằng 0 nên log-likelihood không phụ thuộc θ₄, còn ràng buộc thì "tốn" khối lượng; tối ưu đẩy θ₄ về biên 0 (KKT: đạo hàm theo θ₄ của Lagrangian là −ν < 0 tại mọi θ₄ > 0), và điều kiện "θₖ = nₖ/ν" ở (a) chỉ áp dụng cho nₖ > 0. *Thang:* 1 điểm hai trường hợp α; 1 điểm hai bộ số; 1 điểm xử lý đúng nₖ = 0 (θ₄ = 0 ở biên, có lý do).

(c) MLE: θ̂(d) = 0 nên token "d" có −ln 0 = +∞; cross-entropy và perplexity đều vô hạn. MAP (α = 2): xác suất của bốn token test là 3/7, 2/7, 1/14, 3/7. Cross-entropy

$$
-\tfrac{1}{4}\Big[2\ln\tfrac{3}{7} + \ln\tfrac{2}{7} + \ln\tfrac{1}{14}\Big] = \ln 7 - \tfrac{1}{2}\ln 3 = 1.9459 - 0.5493 = 1.3966 \text{ nat},
$$

perplexity = e mũ 1.3966 = 7/√3 ≈ 4.04. Mô hình đều có cross-entropy ln 4 = 1.3863, perplexity 4, tức tốt hơn một chút trên chuỗi này. Giải thích: 1/4 số token test là "d", thứ chưa từng xuất hiện trong train, và chuỗi test chỉ có 4 token nên con số rất nhiễu; smoothing cứu được khỏi vô hạn nhưng một tập test nhỏ không đủ để kết luận mô hình nào tốt hơn. *Thang:* 1 điểm MLE vô hạn và cross-entropy MAP đúng; 1 điểm perplexity, so sánh và giải thích.

**B3.** (a) Kiểm: Aw* = (4.5 − 0.5, 1.5 − 1.5) = (4, 0) = b. Trị riêng 4 với vector (1, 1)/√2 và 2 với vector (1, −1)/√2 (tr A = 6, det A = 8). L = 4, μ = 2, κ = 2. *Thang:* 1 điểm trị riêng và vector riêng; 1 điểm L, μ, κ.

(b) Theo hướng riêng i, sai số nhân với 1 − ηλᵢ mỗi bước, nên hội tụ từ mọi w₀ khi và chỉ khi |1 − ηλᵢ| < 1 với mọi i, tức 0 < η < 2/L = 0.5. η tối ưu cân bằng hai đầu: 1 − ημ = −(1 − ηL), η = 2/(L + μ) = 1/3; hệ số co (κ − 1)/(κ + 1) = 1/3 (cả hai hướng: |1 − 4/3| = |1 − 2/3| = 1/3). Với η = 1/3 thì I − ηA có hai trị riêng ±1/3 và đối xứng, nên ‖eₖ‖₂ = 3⁻ᵏ‖e₀‖₂ đúng bằng với mọi w₀. Cần 3ᵏ ≥ 10⁶: k ≥ 6 ln 10 / ln 3 = 13.8155/1.0986 ≈ 12.58, tức k = 13. *Thang:* 1 điểm điều kiện η; 1 điểm η tối ưu; 1 điểm hệ số co 1/3 cho cả hai hướng; 1 điểm k = 13.

(c) w₁ = 0 − 0.5(A·0 − b) = 0.5b = (2, 0). ∇f(w₁) = (6, 2) − (4, 0) = (2, 2), w₂ = (2, 0) − (1, 1) = (1, −1). w₃ = (2, 0) lại, và cứ thế nhảy qua lại giữa (2, 0) và (1, −1) mãi. Phân tích: w₀ − w* = (−1.5, 0.5) = −0.5·(1, 1) − 1·(1, −1). Hướng λ = 2 nhân 1 − 0.5·2 = 0: bị xóa sau một bước. Hướng λ = 4 nhân 1 − 0.5·4 = −1: biên độ giữ nguyên, chỉ đổi dấu. Vậy wₖ = w* ± 0.5(1, 1). f(wₖ) = −2 với mọi k ≥ 1, còn f* = f(w*) = −3: loss đứng yên ở mức cao hơn tối ưu, không giảm. η = 2/L đúng là biên giới hạn của (b). *Thang:* 1 điểm w₁, w₂ và mô tả dao động; 1 điểm phân tích theo vector riêng và f không tiến về f*.

**B4.** (a) ∑xᵢyᵢ = θS + ∑xᵢεᵢ. Do đó

$$
\mathbb{E}[\hat{w}] = \frac{S\theta}{S + \lambda}, \qquad \text{bias} = -\frac{\lambda\theta}{S + \lambda}, \qquad \operatorname{Var}(\hat{w}) = \frac{\sigma^2 S}{(S + \lambda)^2},
$$

vì Var(∑xᵢεᵢ) = σ²∑xᵢ² = σ²S (các εᵢ độc lập). MSE = bias² + Var:

$$
\operatorname{MSE}(\lambda) = \frac{\lambda^2\theta^2 + \sigma^2 S}{(S + \lambda)^2}.
$$

*Thang:* 1 điểm kỳ vọng; 1 điểm bias; 1 điểm phương sai có lý do độc lập; 1 điểm MSE.

(b) Đạo hàm:

$$
\frac{d\,\operatorname{MSE}}{d\lambda} = \frac{2S(\lambda\theta^2 - \sigma^2)}{(S + \lambda)^3},
$$

âm khi λ < σ²/θ² và dương khi λ > σ²/θ², nên λ* = σ²/θ². Số: MSE(0) = σ²/S = 0.4 (ước lượng không chệch, toàn phương sai). λ* = 4/1 = 4. MSE(4) = (16 + 40)/196 = 2/7 ≈ 0.286. Chấp nhận một chút bias làm MSE giảm khoảng 29%. (Không bắt buộc: MSE(4) gồm bias² = (4/14)² ≈ 0.082 và variance = 40/196 ≈ 0.204.) *Thang:* 1 điểm đạo hàm; 1 điểm λ* có lập luận dấu; 1 điểm ba số.

(c) Thực tế chọn λ bằng cross-validation (hoặc tập validation): đo lỗi dự đoán theo một lưới λ (thường theo thang log) và lấy λ tốt nhất. Theo cách đọc MAP, ridge với prior w ~ N(0, τ²) và nhiễu phương sai σ² cho λ = σ²/τ²; λ* = σ²/θ² ứng với τ² = θ², tức prior có độ lớn đúng bằng tham số thật. Prior càng sát thực tế thì regularization càng đúng liều. *Thang:* 1 điểm khi có cả CV và liên hệ τ² = θ².

**B5.** (a) Mẫu 1: a = (2 − 1, 1 + 1) = (1, 2), h = (1, 2), z = 1 − 4 + 1 = −2, p = σ(−2) = 0.1192. Mẫu 2: a = (−1 − 1, −0.5 + 1) = (−2, 0.5), h = (0, 0.5), z = 0 − 1 + 1 = 0, p = 0.5. L = ½[−ln σ(−2) − ln 0.5] = ½[ln(1 + e²) + ln 2] = ½(2.1269 + 0.6931) = 1.410. *Thang:* 1 điểm forward hai mẫu; 1 điểm L.

(b) Với một mẫu, ℓ = −y ln p − (1 − y) ln(1 − p) và dp/dz = p(1 − p), nên dℓ/dz = −y(1 − p) + (1 − y)p = p − y; chia N vì L là trung bình. δ₁ = (0.1192 − 1)/2 = −0.4404; δ₂ = (0.5 − 0)/2 = 0.25.

- ∂L/∂w₂ = δ₁h⁽¹⁾ + δ₂h⁽²⁾ = (−0.4404, −0.8808) + (0, 0.125) = (−0.4404, −0.7558).
- ∂L/∂h⁽ⁱ⁾ = δᵢw₂: mẫu 1 (−0.4404, 0.8808); mẫu 2 (0.25, −0.5).
- Qua ReLU (mặt nạ a > 0): mẫu 1 giữ cả hai, g⁽¹⁾ = (−0.4404, 0.8808); mẫu 2 chặn neuron 1, g⁽²⁾ = (0, −0.5).
- ∂L/∂W₁ = g⁽¹⁾x⁽¹⁾ᵀ + g⁽²⁾x⁽²⁾ᵀ. Hạng tử 1 có hàng (−0.8808, −0.4404) và (1.7616, 0.8808); hạng tử 2 có hàng (0, 0) và (0.5, −0.5). Tổng: hàng (−0.8808, −0.4404) và (2.2616, 0.3808).

Dùng σ(−2) ≈ 0.12 cho δ₁ = −0.44 và các số làm tròn tương ứng (lệch dưới 0.002). Kiểm nhanh: ∂L/∂W₁ cùng shape với W₁. (Không hỏi nhưng nên biết: ∂L/∂b₂ = δ₁ + δ₂ = −0.1904 và ∂L/∂b₁ = g⁽¹⁾ + g⁽²⁾ = (−0.4404, 0.3808).) *Thang:* 1 điểm δ có lý do (p − y)/N; 1 điểm ∂L/∂w₂; 1 điểm mặt nạ ReLU đúng cho cả hai mẫu; 1 điểm ∂L/∂W₁.

(c)(i) Mẫu 2 có a₁ = −2 < 0, ReLU chặn, nên đóng góp 0 vào hàng 1 của ∂L/∂W₁ (và vào thành phần 1 của ∂L/∂b₁). Nếu với mọi mẫu train a₁ đều âm thì hàng 1 của W₁ và b₁ không bao giờ nhận gradient: neuron "chết" vĩnh viễn (dead ReLU), mạng mất một đơn vị ẩn. Hay gặp khi learning rate quá lớn đẩy b₁ về âm sâu. *Thang:* 1 điểm cả hai ý.

(c)(ii) Với mỗi phần tử θ của W₁: g̃ = (L(θ + h) − L(θ − h))/(2h), so với g bằng sai số tương đối |g − g̃| / max(|g| + |g̃|, ε nhỏ), chấp nhận cỡ 10⁻⁷ trở xuống ở float64 với h cỡ 10⁻⁵ đến 10⁻⁶. Nếu một aⱼ⁽ⁱ⁾ nằm trong khoảng cỡ h quanh 0 thì θ ± h làm ReLU đổi trạng thái: sai phân đi qua điểm gãy, ra giá trị trung gian không bằng đạo hàm một phía nào, và check báo sai dù code đúng. Ở bài này mọi |a| ≥ 0.5 nên an toàn. *Thang:* 1 điểm khi có công thức sai phân trung tâm và lý do điểm gãy (sai số tương đối và cách chọn h là phần thưởng, không bắt buộc).

## Số liệu đã kiểm

Mọi số trong đề và lời giải được tính lại bằng hai script, kiểm ngày 2026-11-16 với Python 3.13.17, NumPy 2.5.4, SciPy 1.18.1, PyTorch 2.14.1 (CPU, float64). PyTorch chỉ dùng trong script kiểm của người soạn; bộ chấm của người học chỉ cần NumPy.

- `work/mock1/check_numbers.py`: Phần A và B. Đạo hàm kiểm bằng torch autograd và sai phân trung tâm; các ý dùng mô phỏng (A19, B4) có ghi seed.
- `work/mock1/check_partC.py`: Phần C. Chạy lời giải mẫu qua bộ chấm `labs/thi-thu-1/cham_phan_c.py` (16/16 điểm tự động, chia chương đúng thang), chạy khung trống `labs/thi-thu-1/bai_lam_mau.py` (0/16, không lỗi), và kiểm C1(b), C2(b).

Tái lập: từ thư mục gốc repo, `python work/mock1/check_numbers.py` và `python work/mock1/check_partC.py`; cả hai in "ĐẠT" ở dòng cuối, chạy dưới 30 giây trên CPU.

| Câu | Giá trị | Cách kiểm | Kết quả |
| --- | --- | --- | --- |
| Hằng số | ln 2, 3, 5, 7, 10; σ(2) = 0.8808; ln(1 + e²) = 2.1269; √2 = 1.4142 (A19 dùng √0.02 = √2/10) | `math` | khớp |
| A1 | T(1, 2) = (0, 7); det 7 | NumPy | khớp |
| A2 | hạng 2; A(1, −1, 1) = 0 | `matrix_rank` | khớp |
| A3 | hình chiếu (1, 2, 2); phần dư (2, −2, 1), độ dài 3 | NumPy | khớp |
| A4 | trị riêng −1 và 4; của A² − A là 2 và 12; det A⁻¹ = −0.25 | `eigvals` | khớp |
| A5 | σ = (6, 3); κ = 2; chuẩn Frobenius √45 = 6.708 | `svd`, `cond` | khớp |
| A6 | ∇f = (0.5, −0.5, 1); Hessian 0.25aaᵀ, hạng 1 | autograd, `hessian` | khớp |
| A7 | f = 6; 5; −3 | autograd | khớp |
| A8 | đường chéo 2/9, ngoài −1/9; tổng hàng 0 | `jacobian` của softmax | khớp |
| A9 | 8/17 = 0.4706; 8/9 = 0.8889 | tính trực tiếp | khớp |
| A10 | 1; 2; 1 bit; KL(q‖p) = inf | NumPy | khớp |
| A11 | SE 0.016; CI [0.7686, 0.8314]; 2 500 câu | tính trực tiếp | khớp |
| A12 | Hessian x₁x₂ trị riêng ±1; max(x², 1 − x) có sai phân bậc hai không âm trên lưới | NumPy | khớp |
| A13 | (2, 1); f* = 6; ν = 4; f*(3.1) = 6.4067, xấp xỉ 6.4 | công thức và tìm trên lưới | khớp |
| A14 | độ dời −α và +α/19 = 0.0526α | `torch.optim.Adam` với ε = 0 | khớp |
| A16 | f = (2.5, −0.5, 2.5); hinge (0, 0.5, 3.5), tổng 4 | NumPy | khớp |
| A17 | tâm (4/3, 9); gán lại không đổi; J = 56/3 (không còn hỏi) | NumPy | khớp |
| A18 | 41 105 tham số; 400 running stats; đổi bias trước BN không đổi đầu ra BN | PyTorch | khớp |
| A19 | std lớp 1: 0.1999; tỉ số các lớp sau 0.1413, 0.1405, 0.1427 (lý thuyết 0.2 và 0.1414); He 0.0707; 0.02⁵ = 3.2·10⁻⁹ | mô phỏng 30 lần khởi tạo, seed 0 | khớp |
| A20 | ln 5 = 1.6094; dropout ở train mode cho hai output khác nhau, eval thì trùng | PyTorch | khớp |
| C1 | khớp loss định nghĩa và gradient sai phân trung tâm đến 10⁻⁸; logit ±1000, ±10⁴ và hàng lệch 4 800 hữu hạn, đúng đáp số tay; C1(b): bản ngây thơ ra nan, hàm mẫu cho ln 3 và (⅓, ⅓, −⅔) | `check_partC.py` qua bộ chấm NumPy | khớp |
| C2 | khớp J định nghĩa và gradient sai phân đến 10⁻⁹; z = ±800 đúng đáp số tay (J = 800, w mới −7); C2(b) với đoạn MSE: dòng dw báo ValueError khi N = 4, d = 3, loss sai im lặng (1.3448 so với 1.5458 trên dữ liệu kiểm, đúng công thức mean(z²) − 2·mean(z)·mean(y) + mean(y²)), db trùng bản đúng; N = d = 4 chạy hết, dw shape (4, 4); thêm `y.reshape(-1)` thì khớp | `check_partC.py` | khớp |
| B1 | L = 13; ∂L/∂U = (−3, 0); ∂L/∂V = (−3, −5); U₁ = (1.3, 1.0); V₁ = (1.3, 0.5); L₁ = 8.8993 (không còn hỏi); σ(M) = (5, 3), min L = 4.5, GD 20 000 bước (η = 0.02) hội tụ về ma trận toàn 2.5; tại 0 gradient 0 và L = 17 | công thức kiểm bằng sai phân trên ma trận ngẫu nhiên 4 × 3, r = 2 | khớp |
| B2 | MLE (0.5, 0.3, 0.2, 0), cross-entropy inf; MAP (6, 4, 3, 1)/14; cross-entropy 1.3966 nat; perplexity 4.0415 = 7/√3; tính bằng bảng hằng số lệch dưới 10⁻⁴ | NumPy; MAP kiểm bằng `scipy.optimize.minimize` có ràng buộc | khớp |
| B3 | w* = (1.5, −0.5); trị riêng 2 và 4; η* = 1/3, hệ số 1/3; k = 13 (chạy thật từ w₀ = 0 cũng 13); η = 0.5 dao động (2, 0) ↔ (1, −1), f = −2, f* = −3 | NumPy | khớp |
| B4 | MSE(0) = 0.4; λ* = 4; MSE(4) = 0.2857 = bias² 0.0816 + var 0.2041; Monte Carlo 400 000 lần với x = (1, 1, 2, 2) cho 0.4006 và 0.2860 | công thức, tìm trên lưới, mô phỏng seed 0 | khớp |
| B5 | z = (−2, 0); p = (0.1192, 0.5); L = 1.4100; ∂L/∂w₂ = (−0.4404, −0.7558); ∂L/∂W₁ hàng (−0.8808, −0.4404), (2.2616, 0.3808); ∂L/∂b₂ = −0.1904, ∂L/∂b₁ = (−0.4404, 0.3808) (không còn hỏi); bản làm tròn σ(−2) ≈ 0.12 lệch tối đa 0.0016 | autograd với `binary_cross_entropy_with_logits`; sai phân trung tâm cho W₁ | khớp |

A15 là câu chọn, không có số để kiểm. Điểm: A 40, C 20, B 40, tổng 100; bảng chấm theo chương cộng 12 + 17 + 19 + 16 + 19 + 17 = 100.

**Bộ chấm bám thang điểm.** Chạy `labs/thi-thu-1/cham_phan_c.py` trên các bài làm cố ý (trong `work/mock1/review_check.py`, Phần 2). Mỗi bài chỉ viết một hàm, điểm là phần tự động của hàm đó trên 8:

| Bài làm | Thang dự kiến | Bộ chấm cho |
| --- | --- | --- |
| C1 đúng, viết khác cách (logsumexp của SciPy; one-hot và trả `float`; mặt nạ và trả mảng 0 chiều) | 8 | 8, 8, 8 |
| C1 sửa Z tại chỗ | 7 (mất "không sửa Z") | 7 |
| C1 quên chia N trong dZ | 5 (mất giá trị) | 5 |
| C1 không trừ max / log của softmax / trừ max của cả ma trận | 5 (mất ổn định số) | 5, 5, 5 |
| C1 đổi sang float32 | 2 (mất giá trị và ổn định số, vì đề đòi float64) | 2 |
| C1 có vòng lặp theo mẫu | 8 tự động, trừ tay 2 | 8, có in cảnh báo vòng lặp |
| C2 đúng, viết khác cách (`expit`; tách theo dấu và đổi y thành cột bên trong; sigmoid ngây thơ kèm loss `logaddexp`) | 8 | 8, 8, 8 |
| C2 phạt cả b | 7 (mất "không phạt b") | 7 |
| C2 lấy tổng thay trung bình | 5 (mất giá trị) | 5 |
| C2 J tại tham số mới / quên λw trong dw | 5 (mất giá trị) | 5, 5 |
| C2 `log(1 + exp(z))` / clip p rồi lấy log | 6 (mất ổn định số) | 6, 6 |
| C2 trả w shape (d, 1) | 3 (mất shape và giá trị) | 3 |

**Ước lượng thời gian từng câu** [ước lượng của người soạn theo số bước tính tay, chưa đo trên người thật]. "Người học" là sinh viên năm 3 vừa học xong chương 1–6, làm lần đầu; "giải nhanh" là người vừa ôn kỹ Phần I. Bản trước khi sửa, theo ước lượng của người phản biện, cần khoảng 155 và 105 phút.

| Câu | Người học | Giải nhanh | Ghi chú |
| --- | --- | --- | --- |
| Đọc đề | 2 | 2 | |
| A1, A3, A5, A8, A9, A11, A16, A17, A20 | 2 mỗi câu (18) | 1.5 mỗi câu, A11 2, A20 1 (13.5) | A17 đã bỏ phần tính J |
| A2, A4, A6, A7, A10, A13, A18, A19 | 3 mỗi câu (24) | 2 mỗi câu, A4 và A7 1.5 (15) | |
| A12, A15 | 1 mỗi câu (2) | 1 mỗi câu (2) | |
| A14 | 4 | 2.5 | hai bước Adam có bias correction |
| **Phần A** | **48** | **33** | |
| C1 | 13 ((a) 8 viết, 2 chạy thử; (b) 3) | 9 | khung chữ ký có sẵn |
| C2 | 14 ((a) 8 viết, 2 chạy thử; (b) 4) | 10 | đoạn code (b) ngắn hơn bản cũ |
| **Phần C** | **27** | **19** | |
| B1 | 13 ((a) 4, (b) 3.5, (c)(i) 3, (c)(ii) 2.5) | 8.5 | bỏ tính L₁ (bình phương số thập phân bằng tay) |
| B2 | 10 ((a) 3.5, (b) 2, (c) 4.5) | 7 | gộp suy dẫn MLE vào MAP (α = 1) |
| B3 | 10 ((a) 2, (b) 4, (c) 4) | 6 | cho sẵn w* và hệ thức sai số; bỏ Newton |
| B4 | 10 ((a) 4, (b) 4.5, (c) 1.5) | 7 | cho sẵn công thức đóng; bỏ tách bias² và variance |
| B5 | 13 ((a) 3.5, (b) 6, (c) 3.5) | 9 | bỏ ∂L/∂b₁, ∂L/∂b₂ |
| **Phần B** | **56** | **37.5** | |
| **Tổng** | **133** | **91.5** | giờ thi 120 |

**Cần bạn xác nhận.**

1. Cơ cấu 40/20/40, thứ tự A → C → B và 120 phút là lựa chọn của người soạn để mô phỏng một entrance test; định dạng thật không công khai. Nếu sau này có thông tin đáng tin về định dạng (ví dụ trắc nghiệm thuần, hoặc chỉ coding), cơ cấu nên đổi.
2. Ước lượng 133 phút cho người học mục tiêu và 92 phút cho người giải nhanh chưa đo trên người thật. Để đề còn dài hơn 120 phút khoảng 13 phút là chủ ý; sau lần làm đầu, báo số phút thật từng phần để chỉnh cột Phút và mốc giờ.
3. Ngưỡng 75% và 50% theo chương, mốc 75 và 60 theo tổng, mốc chênh 15 điểm cho tốc độ là heuristic. w13 đã có 3 giờ cho làm và chấm đề (`w13-mock`); tổng w13 là 13.5 giờ. Lịch w14–w15 (14.5 và 15 giờ) chưa có chỗ cho việc ôn khi tổng 60–74; đề xuất bớt luồng B là ý của người soạn.
4. Các mục không được hỏi (cột cuối bảng chấm), đáng chú ý nhất là cây và ensemble (5.4), GMM và EM (5.5), metric phân loại (5.2), SGD (4.3), Newton (4.5, bỏ khi rút ngắn B3), LayerNorm (6.4). Muốn phủ thì phải thay câu Phần A và tính lại bảng chấm.
5. Phần C cho dùng máy nhưng không tài liệu; nếu bạn muốn mô phỏng đề viết tay (một số đề đầu vào không cho chạy code), hãy viết ra giấy trước và chấm bản giấy, như Phần N của bài chẩn đoán.
6. Bảng "Câu sai → ôn gì" trỏ tới số ví dụ và số bài của chương 1–6 tại ngày kiểm (2026-11-16); nếu các chương được sửa và đánh số lại, bảng cần rà lại.
7. Đề chưa được đăng ký trong `book/chapters.yaml` và `book/_toc.yml`, nên trang "Sách → Đề thi thử 1" mà `w13-mock` trỏ tới chưa có trong app. Việc này thuộc bước merge, không thuộc file đề.
