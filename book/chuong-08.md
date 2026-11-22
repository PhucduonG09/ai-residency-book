# Chương 8 — Attention và Transformer

## Mục tiêu và kiểm tra nhanh

Đây là chương trọng tâm của nhánh NLP. Bạn dùng backprop ma trận (chương 2), Adam (chương 4), LayerNorm, khởi tạo và quy trình debug (chương 6) để hiểu và tự viết Transformer: vì sao attention ra đời, nó tính gì, vì sao chia √dₖ, vị trí vào bằng cách nào, một khối gồm gì, và nó tốn bao nhiêu tham số, phép tính, bộ nhớ. Người phỏng vấn NLP gần như chắc chắn hỏi một chuỗi câu ở đây, thường theo kiểu "viết đi, rồi giải thích từng dòng". Tokenization, pretraining, scaling law, fine-tuning thuộc chương 9. Thời lượng: khoảng 2 tuần, 8 giờ mỗi tuần.

**Sau chương này, bạn phải làm được:**

- ☐ Giải thích vì sao seq2seq RNN có nút thắt thông tin và đường gradient dài, attention của Bahdanau sửa điều gì, và tính tay một bước attention cộng tính lẫn dot-product.
- ☐ Dẫn xuất scaled dot-product attention, chứng minh Var(q·k) = dₖ, giải thích multi-head (hình dạng tensor, số tham số không đổi theo số đầu) và tự viết nó bằng NumPy và PyTorch, có mặt nạ.
- ☐ So sánh sinusoidal, learned và RoPE; chứng minh điểm RoPE chỉ phụ thuộc hiệu n − m của hai vị trí; nói đúng khi nào thiếu mã hóa vị trí thì mất thứ tự và khi nào không.
- ☐ Vẽ khối Transformer pre-LN và post-LN, giải thích residual, LayerNorm, FFN, LN cuối và cách khởi tạo; không nhầm bản triển khai với bản trong paper.
- ☐ Đếm tham số, FLOPs, bộ nhớ ma trận điểm và KV cache của một cấu hình bất kỳ, và kiểm tra một mô hình không nhìn thấy tương lai.

**Kiểm tra nhanh (5 phút, không dùng máy tính).** Câu i kiểm tra mục 8.i: đúng thì đọc lướt mục đó, sai thì đọc kỹ. Đúng cả 5 câu thì làm thẳng bài ★★★ và mini-test.

1. Một RNN có ‖∂hₜ/∂hₜ₋₁‖₂ ≤ 0.9 ở mọi bước. Chặn trên của ‖∂h₁₀₀/∂h₀‖₂ là bao nhiêu?
2. q và k có các thành phần độc lập, trung bình 0, phương sai 1, và dₖ = 64. Độ lệch chuẩn của q·k là bao nhiêu, và sau khi chia √dₖ?
3. Self-attention không causal, không có mã hóa vị trí. Đổi chỗ hai token trong đầu vào thì đầu ra thay đổi thế nào?
4. Viết công thức cập nhật dòng residual của một nhánh con (attention hoặc FFN) trong post-LN và trong pre-LN.
5. Mô hình 12 lớp, 12 đầu, độ rộng mỗi đầu dₕ = 64, KV cache lưu ở fp16. Mỗi token tốn bao nhiêu byte cache?

Đáp án ở mục Lời giải cuối chương.

## 8.1 RNN, seq2seq và vì sao attention ra đời

Attention ra đời để chữa ba bệnh của seq2seq dùng RNN: một vector cố định phải chứa cả câu nguồn, gradient đi qua một chuỗi dài các phép nhân, và tính toán buộc phải tuần tự.

**RNN trong ba dòng.** Một RNN đọc chuỗi x₁, …, xₙ và giữ một trạng thái ẩn hₜ, cập nhật lần lượt:

$$
h_t = \tanh\!\left(W_h h_{t-1} + W_x x_t + b\right)
$$

Cùng một bộ trọng số dùng cho mọi bước. LSTM và GRU thêm các cổng (gate) quyết định giữ hay quên trạng thái cũ; cổng giúp gradient đi xa hơn nhưng không loại bỏ vấn đề dưới đây.

**Seq2seq (encoder–decoder).** Sutskever và cộng sự (2014) và Cho và cộng sự (2014) dùng hai RNN: encoder nén câu nguồn thành một vector cố định c = hₙ, decoder sinh câu đích từ c. Lúc huấn luyện dùng teacher forcing (đầu vào decoder là từ đúng của đáp án); lúc sinh thật, mô hình tự hồi quy (từ vừa sinh được đưa lại làm đầu vào).

**Ba bệnh của kiến trúc này.**

1. *Nút thắt thông tin.* Câu 8 từ hay 80 từ đều bị nén vào cùng một vector c.
2. *Đường gradient dài.* Gradient từ cuối chuỗi về đầu chuỗi là tích của n Jacobian: ∂hₙ/∂h₀ = ∏ₜ diag(1 − hₜ²)·`W_h`. Nếu mỗi thừa số co lại thì tích tiêu biến theo cấp số nhân (mục 1.4).
3. *Tuần tự.* hₜ cần hₜ₋₁ nên n bước không song song được; GPU thích phép nhân ma trận lớn, không thích chuỗi n phép nhân nhỏ.

**Bahdanau, Cho và Bengio (2014, ICLR 2015): nhìn lại thay vì nén.** Thay một c duy nhất bằng một ngữ cảnh riêng cho từng bước giải mã i, là tổ hợp có trọng số của mọi trạng thái encoder h₁, …, hₙ:

$$
c_i = \sum_j \alpha_{ij}\, h_j, \qquad \alpha_{i\cdot} = \operatorname{softmax}(e_{i\cdot}), \qquad e_{ij} = v^{\top}\tanh\!\left(W s_{i-1} + U h_j\right)
$$

Ở đây sᵢ₋₁ là trạng thái decoder, eᵢⱼ là điểm "bước i quan tâm từ nguồn j bao nhiêu" (additive attention: một mạng nhỏ chấm điểm). Đây là tra cứu mềm và khả vi: softmax cho trọng số không âm, tổng 1, nên gradient chảy được về mọi hⱼ. Ma trận α là bảng căn chỉnh (alignment) giữa từ đích và từ nguồn. Luong và cộng sự (2015) đơn giản hóa hàm điểm thành dot-product e = hⱼ·sᵢ (hoặc dạng general sᵢᵀWhⱼ): rẻ hơn vì cả bảng điểm là một phép nhân ma trận.

**Ví dụ giải mẫu 8.1 — một bước attention, hai cách chấm điểm.** Một bước giải mã với s = (1, 0), hai trạng thái encoder h₁ = (1, 1), h₂ = (0, 1). Để dễ tính lấy W = U = I và v = (1, 1) (bản gốc dùng các ma trận học được).

- Cộng tính: s + h₁ = (2, 1), tanh cho (0.9640, 0.7616), nên e₁ = 1.7256. s + h₂ = (1, 1), tanh cho (0.7616, 0.7616), nên e₂ = 1.5232.
- softmax(1.7256, 1.5232) = (0.5504, 0.4496). Ngữ cảnh c = 0.5504·(1, 1) + 0.4496·(0, 1) = (0.5504, 1.0000).
- Dot-product: e = (h₁·s, h₂·s) = (1, 0), α = (0.7311, 0.2689), c = (0.7311, 1.0000).
- Cùng một cơ chế (điểm, softmax, trung bình có trọng số), chỉ khác hàm điểm; c nằm trong bao lồi của h₁, h₂.

**Ví dụ giải mẫu 8.2 — số đo cho ba bệnh.** Tất cả số dưới đây lấy từ Lab 6 (tùy chọn, cuối notebook), một seed, bài toán đồ chơi.

(a) *Gradient qua thời gian.* RNN tanh 64 chiều: hₜ = tanh(Whₜ₋₁ + nhiễu), mỗi bước cộng nhiễu 0.3·N(0, 1) vào đầu vào của tanh, h₀ ~ 0.5·N(0, 1), W ngẫu nhiên rescale về bán kính phổ ρ. Đo ‖∂hₜ/∂h₀‖₂ bằng cách nhân các Jacobian diag(1 − h²)·W (hai chi tiết nhiễu và h₀ quyết định số đo; thiếu chúng bạn không tái lập được bảng):

| ρ | T = 5 | T = 20 | T = 50 | T = 100 |
| --- | --- | --- | --- | --- |
| 0.9 | 7.33e-1 | 1.22e-2 | 6.05e-6 | 9.63e-12 |
| 1.0 | 1.11 | 5.75e-2 | 3.30e-4 | 1.98e-8 |

Ngay cả ρ = 1.0 vẫn giảm mạnh, và nguyên nhân là **đạo hàm của tanh**, không phải tính không chuẩn tắc của W: 1 − h² < 1 ở hầu hết các bước (trung bình 0.82), và 0.82¹⁰⁰ ≈ 2e-9, chênh khoảng một bậc so với 1.98e-8 (ước lượng thô, vì tích thật phụ thuộc cả hướng của các vector, nhưng đủ cho thấy sự co theo cấp số nhân). Lab 6 tách từng ứng viên ở T = 100: ‖W‖₂ = 1.97 (> 1); bỏ tanh thì chuẩn của tích là ‖W¹⁰⁰‖₂ (W lũy thừa 100) = 2.62, không co; thay W bằng ma trận trực giao (ρ = 1, hoàn toàn chuẩn tắc) vẫn cho 1.2e-7. Hiệu ứng ròng là co theo cấp số nhân dù chuẩn từng thừa số có thể lớn hơn 1. Đừng suy ra "ρ > 1 thì nổ": tanh bão hòa kéo đạo hàm nhỏ lại.

(b) *Nút thắt trên bài chép chuỗi.* Seq2seq GRU (ẩn 32, từ vựng 10 ký tự), teacher forcing, Adam 3e-3, 300 bước, accuracy theo token trên 1000 chuỗi mới:

| Độ dài chuỗi | Không attention | Có attention (dot-product trên trạng thái encoder) |
| --- | --- | --- |
| 8 | 0.667 | 0.994 |
| 16 | 0.350 | 0.986 |

Hai mô hình chỉ khác một dòng: bản có attention nối thêm ngữ cảnh A·(trạng thái encoder) vào đầu ra của decoder. Đây là bài đồ chơi, một seed, mô hình nhỏ, 300 bước: chưa phân biệt được "nút thắt dung lượng" với "chưa hội tụ". Lab 6 chạy thêm độ dài 16, không attention, h = 64, 600 bước: 0.630 (so với 0.350), khá hơn nhiều nhưng vẫn kém xa 0.986. Đừng kết luận "RNN không nhớ được chuỗi dài": GRU hay LSTM lớn hơn, huấn luyện lâu hơn, sẽ khá hơn.

% manim: NutThatCoChai

**So sánh theo từng lớp** (diễn đạt lại Bảng 1 của Vaswani và cộng sự, đã đối chiếu paper).

| Loại lớp | Số phép tính mỗi lớp | Số bước tuần tự | Đường đi dài nhất giữa hai vị trí |
| --- | --- | --- | --- |
| Self-attention | O(n²·d) | O(1) | O(1) |
| RNN | O(n·d²) | O(n) | O(n) |

Hai điểm cần nói được: self-attention nối trực tiếp mọi cặp vị trí và song song hoàn toàn; và n²d so với nd² nghĩa là nó rẻ hơn khi n < d, đắt hơn khi n > d. Với d = 512: n = 50 cho 1.28e6 so với 1.31e7 (rẻ hơn khoảng 10 lần); n = 1000 cho 5.12e8 so với 2.62e8 (đảo chiều). Cái giá O(n²) là có thật; mục 8.5 đếm nó.

**Vì sao ML cần cái này.**

- Câu "attention giải quyết vấn đề gì của seq2seq" là câu mở đầu quen thuộc của buổi phỏng vấn; mọi mô hình ngôn ngữ hiện đại là hậu duệ của ý "tra cứu mềm trên trạng thái trước đó".
- Bài chép chuỗi là phép thử nhanh nhất cho một cơ chế truy cập bộ nhớ: không chép được thì càng không làm được việc khó hơn.

## 8.2 Scaled dot-product attention và multi-head

Attention là một phép tra cứu mềm: mỗi vị trí đặt một câu hỏi (query), so với nhãn (key) của mọi vị trí, rồi lấy trung bình có trọng số các nội dung (value). Hệ số √dₖ, mặt nạ và cách tách đầu là chỗ người ta hay hỏi và hay viết sai.

**Công thức.** X ∈ ℝⁿˣᵈ là n vector đầu vào. Ba phép chiếu tuyến tính cho Q = `XW_Q`, K = `XW_K`, V = `XW_V`, với Q, K ∈ ℝⁿˣᵈᵏ và V ∈ ℝⁿˣᵈᵛ:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V, \qquad Q K^{\top} \in \mathbb{R}^{n \times n}
$$

Phần tử (i, j) của QKᵀ là qᵢ·kⱼ, độ khớp giữa query i và key j. Softmax theo từng hàng cho A, mỗi hàng là một phân phối. Đầu ra của vị trí i là ∑ⱼ Aᵢⱼvⱼ, một tổ hợp lồi của các hàng của V (chương 1). Cả bảng tính bằng vài phép nhân ma trận, đó là lý do GPU thích nó.

**Softmax ổn định.** softmax(z) = softmax(z − c) với mọi hằng c, vì tử và mẫu cùng nhân e⁻ᶜ. Trừ c = max(z) thì mọi số mũ ≤ 0 nên exp không tràn. Ví dụ softmax(1000, 1001) tính trực tiếp ra inf/inf = NaN; sau khi trừ max còn softmax(0, 1) = (0.2689, 0.7311). Đây cũng là mẹo log-sum-exp của cross-entropy ở chương 6. Lab 1 ép điều này bằng assert, nên bản `exp(Z)/sum` ngây thơ sẽ không qua.

**Self-attention và cross-attention.** Self-attention: Q, K, V cùng từ một chuỗi X. Cross-attention: Q từ decoder, K và V từ encoder (attention của Bahdanau, với dot-product thay mạng cộng tính).

**Ví dụ giải mẫu 8.3 — attention tay cho 3 token, dₖ = 2.** Lấy Q = K có các hàng (1, 0), (0, 1), (1, 1) và V có các hàng (1, 2), (3, 4), (5, 6).

- QKᵀ có các hàng (1, 0, 1), (0, 1, 1), (1, 1, 2). Chia √2: hàng 1 là (0.7071, 0, 0.7071).
- Hàng 1: exp(0.7071) = 2.0281, exp(0) = 1, tổng 5.0562 nên A₁ = (0.4011, 0.1978, 0.4011). Hàng 2: (0.1978, 0.4011, 0.4011). Hàng 3: (0.2483, 0.2483, 0.5035).
- O = AV: O₁ = (3.0000, 4.0000), O₂ = (3.4067, 4.4067), O₃ = (3.5105, 4.5105).
- Với mặt nạ nhân quả (che ô j > i bằng −∞ trước softmax): A₁ = (1, 0, 0) nên O₁ = (1, 2); A₂ = (0.3302, 0.6698, 0) nên O₂ = (2.3395, 3.3395); hàng 3 không đổi vì nó vốn được nhìn cả ba vị trí.
- Nếu không chia √2, hàng 3 là (0.2119, 0.2119, 0.5761), nhọn hơn (0.2483, 0.2483, 0.5035); với dₖ = 2 hiệu ứng nhỏ, ví dụ 8.4 cho thấy nó lớn khi dₖ lớn.
- Lab 1 tái hiện các số này và khớp `F.scaled_dot_product_attention` (sai khác 9e-16 ở float64).

% manim: AttentionBaTu

**Vì sao chia √dₖ.** Giả sử các thành phần của q và k độc lập, trung bình 0, phương sai 1. Khi đó q·k = ∑ᵢ qᵢkᵢ là tổng của dₖ số hạng độc lập, mỗi số hạng có trung bình 0 và phương sai 𝔼[qᵢ²]𝔼[kᵢ²] = 1. Vậy:

$$
\mathbb{E}[q \cdot k] = 0, \qquad \operatorname{Var}(q \cdot k) = d_k, \qquad \operatorname{std}\!\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = 1
$$

Không chia, std của logit tăng theo √dₖ và softmax bão hòa gần one-hot; đạo hàm pᵢ(δᵢⱼ − pⱼ) gần 0 ở **đa số** query, còn gradient dồn vào số ít query có hai logit gần bằng nhau. Đo ở ví dụ 8.4 cho thấy đây không phải "gradient biến mất ở mọi nơi" (trung bình gradient theo q thậm chí lớn hơn khi không chia) mà là gradient thưa và phương sai rất lớn: đa số query gần như không học, vài query nhận gradient lớn, nên học chậm và bất ổn. Chính Vaswani và cộng sự chỉ nêu lý do này như một phỏng đoán ("we suspect", chú thích ở mục 3.2.1 [cần kiểm chứng: số mục và câu trích chưa mở lại]). Chia √dₖ giữ logit cỡ 1 bất kể dₖ. Chia cho dₖ thì std còn 1/√dₖ, softmax gần phân phối đều, mất chọn lọc. Lập luận chỉ đúng lúc khởi tạo với giả thiết độc lập, không bảo đảm gì khi q, k đã tương quan sau huấn luyện (xem "Thực tế khác lý thuyết").

**Ví dụ giải mẫu 8.4 — phương sai của q·k và bão hòa softmax (đo ở Lab 1).**

std(q·k) với 20 000 mẫu là 2.01, 3.99, 7.98, 15.93 ở dₖ = 4, 16, 64, 256 (so với √dₖ = 2, 4, 8, 16). Bão hòa: một query với 10 key ngẫu nhiên chuẩn, lặp 4000 lần, kèm một gradient ngược g ~ N(0, I) theo p. "max p" là trung bình của xác suất lớn nhất; ‖J‖F là chuẩn Frobenius của Jacobian softmax diag(p) − ppᵀ theo logit **đưa vào** softmax; ‖∂L/∂q‖ là chuẩn gradient theo query gốc q của L = g·p. "Không chia" dùng logit thô q·k, "có chia" dùng q·k/√dₖ. Mỗi đại lượng báo cả trung bình lẫn trung vị, vì phân phối của gradient có đuôi nặng.

| dₖ | Cách | max p | ‖J‖F trung bình | ‖J‖F trung vị | ‖∂L/∂q‖ trung bình | ‖∂L/∂q‖ trung vị |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | không chia | 0.489 | 0.339 | 0.358 | 0.524 | 0.441 |
| 4 | có chia | 0.303 | 0.341 | 0.340 | 0.290 | 0.256 |
| 64 | không chia | 0.867 | 0.157 | 0.0723 | 1.047 | 0.297 |
| 64 | có chia | 0.322 | 0.347 | 0.347 | 0.321 | 0.303 |
| 512 | không chia | 0.950 | 0.066 | 0.00016 | 1.205 | 0.0017 |
| 512 | có chia | 0.323 | 0.348 | 0.347 | 0.322 | 0.300 |

Đọc ở dₖ = 512. Không chia thì max p trung bình 0.95; Jacobian có trung bình 0.066 nhưng **trung vị chỉ 0.00016**: đa số query có Jacobian gần 0. Gradient theo q có trung bình 1.2 (lớn hơn 0.32 của bản có chia, vì logit thô lớn gấp √dₖ lần bù lại Jacobian nhỏ) nhưng trung vị chỉ 0.0017 so với 0.30. Tức là đa số query gần như không nhận gradient, còn trung bình bị kéo lên bởi vài query "gần hòa" có gradient rất lớn. Trung bình nhạy với seed (một seed khác cho 1.09; trung vị 0.0019), nên bài học là trung vị, không phải con số trung bình. Có chia thì cả hai thống kê gần như không đổi theo dₖ. Tính tay cho dₖ = 64: hai logit (8, 0) cho p = (0.9997, 0.0003) và đạo hàm p₁(1 − p₁) = 3.35e-4. Sau khi chia √64 = 8 còn logit (1, 0), p = (0.7311, 0.2689), đạo hàm 0.1966, lớn gấp gần 600 lần.

% manim: BaoHoaSoftmax

**Multi-head.** Một softmax chỉ cho một phân phối chú ý mỗi vị trí; muốn đồng thời chú ý "từ ngay trước" và "chủ ngữ" thì cần nhiều phân phối. h đầu chạy song song, mỗi đầu chiếu riêng xuống không gian dₖ = dᵥ = d/h, rồi nối kết quả và chiếu lại bằng `W_O` ∈ ℝᵈˣᵈ:

$$
\operatorname{head}_i = \operatorname{Attention}(X W_Q^{(i)}, X W_K^{(i)}, X W_V^{(i)}), \qquad \operatorname{MHA}(X) = \left[\operatorname{head}_1; \dots; \operatorname{head}_h\right] W_O
$$

Trong code, các đầu là một chiều batch. Hình dạng tensor (B mẫu, T token, d chiều, h đầu, dₖ = d/h):

| Bước | Phép toán | Hình dạng |
| --- | --- | --- |
| Đầu vào | x | (B, T, d) |
| Chiếu một lần | `qkv(x)` rồi `split` | 3 tensor (B, T, d) |
| Tách đầu | `view(B, T, h, dk).transpose(1, 2)` | (B, h, T, dk) |
| Điểm | q @ kᵀ / √dₖ | (B, h, T, T) |
| Softmax theo trục cuối, nhân v | A @ v | (B, h, T, dk) |
| Gộp đầu | `transpose(1, 2).reshape(B, T, d)` | (B, T, d) |
| Chiếu ra | `o(·)` | (B, T, d) |

Dòng tách đầu dễ sai nhất: `reshape(B, h, T, dk)` trực tiếp (không transpose) cho cùng shape nhưng trộn các vị trí vào các đầu. Bài săn bug gài đúng lỗi này.

**Số tham số và chi phí không đổi theo số đầu.** Bốn ma trận d × d (`W_Q`, `W_K`, `W_V`, `W_O`) cho 4d² tham số, cộng 4d bias: 4d² + 4d. Vì h·dₖ = d, số tham số và số phép tính của phép chiếu, QKᵀ và AV không phụ thuộc h (Lab 2: d = 512 với h = 1, 8, 16 đều 1 050 624 tham số). Điều đổi là cách chia d chiều: nhiều đầu hẹp thay vì ít đầu rộng.

**Hoán vị-đẳng biến.** Nếu không có thông tin vị trí thì self-attention không có khái niệm thứ tự. Với ma trận hoán vị P:

$$
\operatorname{Attn}(PX) = P \cdot \operatorname{Attn}(X)
$$

Vì sao: điểm của đầu vào đã hoán vị là PSPᵀ; softmax theo hàng đẳng biến với hoán vị cột (đổi chỗ các cột thì đầu ra đổi chỗ cùng cách) nên A mới là PAPᵀ; đầu ra PAPᵀ·PV = PAV (Lab 1: sai khác 2e-16). Hệ quả: bộ mã hóa hai chiều thuần túy xem câu như túi từ; muốn phân biệt "chó cắn người" với "người cắn chó" phải đưa vị trí vào (mục 8.3).

**Vì sao ML cần cái này.**

- Viết được attention có mặt nạ trong 10 phút và nêu đúng hình dạng từng tensor là câu hỏi lập trình tiêu chuẩn.
- Lập luận phương sai của √dₖ cùng kiểu với khởi tạo He ở chương 6: giữ độ lớn tín hiệu không đổi khi kích thước đổi. Và đầu ra là tổ hợp lồi của V nên không ra ngoài bao lồi của V.

## 8.3 Mã hóa vị trí: sinusoidal, learned, RoPE

Self-attention không có thứ tự, nên thứ tự phải được đưa vào riêng. Có ba họ: cộng vector vị trí tuyệt đối vào embedding (sinusoidal của Vaswani; learned của GPT-2 và BERT), mã hóa vị trí tương đối (Shaw và cộng sự 2018, bias của T5, ALiBi của Press và cộng sự 2021), và xoay q và k (RoPE của Su và cộng sự 2021).

**Sinusoidal (Vaswani 2017).** Mỗi vị trí pos nhận một vector d chiều, cộng vào embedding của token:

$$
PE(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad PE(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

Mỗi cặp (sin, cos) là một "kim đồng hồ" quay với tần số ωᵢ = 10000^(−2i/d); bước sóng chạy theo dải hình học từ 2π đến khoảng 10000·2π. Cặp đầu quay nhanh (phân biệt vị trí gần nhau), cặp cuối quay chậm (phân biệt vị trí xa nhau), như các chữ số của một bộ đếm. Hai tính chất, đã kiểm ở Lab 3:

1. *Dịch vị trí là phép biến đổi tuyến tính:* PE(pos + k) = R(−ωᵢk)·PE(pos) cho từng cặp, R(φ) là ma trận quay ngược chiều kim đồng hồ góc φ (định nghĩa ở phần RoPE bên dưới); dấu trừ vì thứ tự cặp là (sin, cos), nên PE quay theo chiều kim đồng hồ (công thức cộng góc của sin và cos, đã kiểm bằng số). Mô hình có thể học "nhìn lùi k vị trí" bằng một phép biến đổi tuyến tính cố định.
2. *Tích vô hướng chỉ phụ thuộc khoảng cách:* PE(p)·PE(p + k) = ∑ᵢ cos(ωᵢk), không phụ thuộc p, nhưng *không đơn điệu* theo k.

**Ví dụ giải mẫu 8.5 — bảng sinusoidal d = 4, vị trí 0 đến 3.** Hai cặp: tần số 1 và 0.01.

| pos | PE |
| --- | --- |
| 0 | (0, 1, 0, 1) |
| 1 | (0.8415, 0.5403, 0.0100, 1.0000) |
| 2 | (0.9093, −0.4161, 0.0200, 0.9998) |
| 3 | (0.1411, −0.9900, 0.0300, 0.9996) |

Cặp chậm (0.01 rad mỗi bước) gần như không đổi trong 4 vị trí đầu, chỉ có ích khi so các vị trí cách nhau hàng chục hoặc hàng trăm. Với d = 512, bước sóng cặp đầu là 6.28 và cặp cuối khoảng 60 611. Tích vô hướng (d = 64): PE(p)·PE(p + 5) = 23.504 tại p = 0, 10, 100, còn |PE|² = d/2 = 32; theo k = 0, 1, 2, 5, 10, 20, 50, 100 là 32.000, 30.917, 28.304, 23.504, 21.052, 18.879, 15.674, 17.875 (k = 100 lớn hơn k = 50: không đơn điệu).

% manim: SongSinViTri

Paper nhân embedding của token với √d trước khi cộng PE (d là kích thước mô hình), và Annotated Transformer làm đúng vậy (`self.lut(x) * math.sqrt(self.d_model)`); nanoGPT không làm. Paper thử learned PE và báo kết quả "nearly identical"; chọn sinusoidal vì nó "may allow" ngoại suy là giả thuyết của tác giả, không phải kết quả đã chứng minh (Press và cộng sự 2021 nhận xét các phương pháp hiện có không ngoại suy hiệu quả).

**Learned absolute.** Bảng embedding `wpe` kích thước `block_size` × d, học cùng mô hình (GPT-2 small: 1024 × 768 = 786 432 tham số). Đơn giản, nhưng không có vị trí ngoài `block_size`: nanoGPT có `assert t <= block_size`.

**RoPE (Su và cộng sự 2021).** Xoay q và k theo vị trí thay vì cộng vào embedding. Chia vector d chiều thành d/2 cặp; ở vị trí m, quay cặp j một góc mθⱼ, θⱼ = `base^(−2j/d)`, base = 10 000 (mặc định trong mã Llama). Quay 2D một góc φ là nhân với R(φ); R trực giao và R(a)ᵀR(b) = R(b − a). Nên điểm giữa query ở vị trí m và key ở vị trí n là:

$$
\left(R(m\theta)\,q\right)^{\top}\left(R(n\theta)\,k\right) = q^{\top} R(-m\theta) R(n\theta)\, k = q^{\top} R\big((n - m)\theta\big)\, k
$$

Chỉ phụ thuộc n − m: attention thấy vị trí *tương đối* dù q và k mang vị trí *tuyệt đối*. RoPE chỉ áp lên q và k (những thứ vào tích vô hướng), không áp lên v (nội dung được trộn), không cộng vào embedding, và bảo toàn chuẩn ‖R(φ)q‖ = ‖q‖. Bước sóng chậm nhất (d = 128) là 2π·`base^((d−2)/d)`: 54 410 vị trí với base 10 000, 2 559 196 với base 500 000 (tính từ công thức). Llama 3 tăng base lên 500 000 (báo cáo Llama 3, mục 3.2).

**Cảnh báo quy ước ghép cặp.** Ghép liền kề (2j, 2j + 1), như mã Llama gốc (`view_as_complex`) và lab của sách; hoặc ghép i với i + d/2 (`rotate_half` trong mã Hugging Face, nơi checkpoint đã được hoán vị trọng số cho khớp). Cả hai đều cho điểm chỉ phụ thuộc n − m, nhưng nạp trọng số theo quy ước này vào mã theo quy ước kia thì mô hình vẫn chạy mà kém: một bug im lặng.

**Ví dụ giải mẫu 8.6 — RoPE tay trong 2D.** q = (1, 2), k = (3, 1), θ = π/6 mỗi vị trí. Với (m, n) = (2, 0): quay q thêm 60°, k đứng yên. R(60°)q = (cos 60°·1 − sin 60°·2, sin 60°·1 + cos 60°·2) = (−1.2321, 1.8660). Điểm: (−1.2321)·3 + 1.8660·1 = −1.8301 = 5/2 − 5√3/2. Cùng giá trị cho (5, 3) và (12, 10) vì n − m = −2 (cùng hiệu). Không quay thì q·k = 5; góc giữa hai vector tăng từ 45° lên 105° và cos 105°·‖q‖‖k‖ = −1.830, khớp. Lab 3 kiểm ở d = 8: các cặp (5, 2), (15, 12), (105, 102), (1005, 1002) đều cho −1.5865 (không quay: −1.4680), ‖q‖ không đổi.

% manim: RoPEQuay

**Ví dụ giải mẫu 8.7 — thiếu vị trí thì mù thứ tự (Lab 3).** Chuỗi 8 token từ vựng 10, nhãn 1 nếu token đầu lớn hơn token cuối, 0 nếu ngược lại. Tỉ lệ lớp 1 là 0.450, đoán lớp đa số đạt 0.550. Bộ mã hóa hai chiều 2 khối (d = 32, 2 đầu), mean-pool, Adam 3e-3, 300 bước:

- Có learned PE: accuracy **1.000**.
- Không PE: **0.549**, bằng đoán lớp đa số: mean-pool là hàm của *tập* token, không phân biệt "đầu" và "cuối".
- Kiểm không cần huấn luyện: mô hình hai chiều không PE, đổi thứ tự đầu vào thì đầu ra đổi thứ tự tương ứng (sai khác 4.8e-7). Mô hình **causal** không PE: sai khác **0.807**, không còn đẳng biến.

Mặt nạ nhân quả tự mang thông tin vị trí: token ở vị trí i chỉ thấy i token, "số token tôi thấy" ngầm là tín hiệu vị trí. Kazemnejad và cộng sự (2023, "NoPE") báo rằng decoder-only không PE vẫn học được và tổng quát hóa độ dài tốt hơn nhiều PE tường minh trong các bài của họ. Ở bài săn bug, mô hình causal không PE đạt loss 0.676 trên bài chép token ngay trước (có PE: 0.014, cùng điều kiện): học được một phần nhưng kém rõ rệt. Vậy "bỏ PE thì mất thứ tự" đúng cho bộ mã hóa hai chiều, chỉ đúng một phần cho GPT.

**Vì sao ML cần cái này.**

- Chọn mã hóa vị trí quyết định mô hình xử lý được chuỗi dài đến đâu; phải giải thích được vì sao các mô hình mới chuyển sang RoPE.
- Bẫy phỏng vấn: "attention không biết thứ tự" chưa đủ; nó đúng với mô hình không causal và chỉ đúng một phần với causal.

## 8.4 Khối Transformer: residual, LayerNorm, pre-LN, FFN

Một khối Transformer là hai nhánh con (attention, rồi FFN), mỗi nhánh bọc bởi một đường tắt residual và một phép chuẩn hóa. Điều phải nói được là *nhánh con nằm ở đâu so với chuẩn hóa*, vì paper gốc và hầu hết mã bạn đọc làm khác nhau.

**Khối pre-LN (kiểu GPT-2, nanoGPT; Radford và cộng sự 2019).**

$$
x \leftarrow x + \operatorname{MHA}(\operatorname{LN}(x)), \qquad x \leftarrow x + \operatorname{FFN}(\operatorname{LN}(x)), \qquad \operatorname{FFN}(x) = W_2\,\operatorname{GELU}(W_1 x)
$$

FFN mở rộng từ d lên kích thước ẩn `d_ff` = 4d, qua GELU rồi thu về d (Vaswani base: 512 lên 2048; paper gốc dùng ReLU, GPT-2 dùng GELU). Sau các khối có một LN cuối (`ln_f`) trước lớp ra. FFN áp lên từng vị trí độc lập; attention là chỗ duy nhất các vị trí trao đổi thông tin.

**Residual.** x + f(x) có đạo hàm I + f′: dù f′ nhỏ, gradient vẫn đi thẳng qua phần I (chương 2, ResNet). Cách nhìn "residual stream" (Elhage và cộng sự 2021): một dòng chung x chạy suốt mô hình, mỗi nhánh con đọc từ nó và *cộng thêm* kết quả. Cách nhìn này chỉ gợi ý vì sao cắt bớt một lớp mô hình không hỏng (quan sát thực nghiệm, không suy ra được).

**Post-LN (paper gốc) so với pre-LN.**

$$
\text{post-LN: } x_2 = \operatorname{LN}\!\big(x + \operatorname{Sub}(x)\big), \qquad \text{pre-LN: } x_2 = x + \operatorname{Sub}\!\big(\operatorname{LN}(x)\big)
$$

Vaswani dùng post-LN. Xiong và cộng sự (2020, ICML) phân tích rằng với post-LN, gradient lúc khởi tạo ở các lớp gần đầu ra lớn nên cần warmup, còn pre-LN ổn định hơn và có thể bỏ warmup. Lab của chương **không** tái hiện kết luận này (xem "Thực tế khác lý thuyết"). Bẫy khi đối chiếu code: lớp `SublayerConnection` của Annotated Transformer ghi "for code simplicity the norm is first as opposed to last", tức là **pre-LN**, không phải post-LN như paper.

**Hệ quả của pre-LN: dòng residual không được chuẩn hóa**, nên độ lớn có thể tăng theo độ sâu; đó là lý do cần LN cuối. Lab 2 đo (d = 256, 24 khối, khởi tạo mặc định PyTorch, đầu vào N(0, 1)): std của dòng residual sau 1, 6, 12, 24 khối là 1.020, 1.128, 1.257, 1.555 với pre-LN và đúng 1.0 với post-LN. Con số 1.0 là hằng đẳng thức (đầu ra LN có gain 1, bias 0), không phải phát hiện; điều đáng nói là pre-LN tăng.

**LayerNorm, BatchNorm, RMSNorm.** LN chuẩn hóa theo các đặc trưng của *từng token*, không phụ thuộc batch và giống nhau lúc train lẫn eval (chương 6); BatchNorm khó dùng với chuỗi độ dài thay đổi, padding và batch nhỏ. RMSNorm (Zhang & Sennrich, 2019) bỏ việc trừ trung bình và bias: chia cho căn trung bình bình phương; Llama dùng nó.

**FFN chiếm phần lớn tham số.** Attention 4d², FFN 8d² (d·4d + 4d·d): FFN là 2/3 số tham số của khối. Geva và cộng sự (arXiv 2020, đăng EMNLP 2021 [cần kiểm chứng]) nhìn FFN như bộ nhớ key–value (chỉ là một cách nhìn). *Vì sao rộng gấp 4?* Tỉ lệ 4 là lựa chọn thực nghiệm của Vaswani, không có chứng minh. FFN là chỗ có phi tuyến thật theo từng vị trí (attention chỉ trộn các vector bằng trọng số) và giữ 2/3 tham số. SwiGLU có ba ma trận nên dùng ẩn ≈ 8d/3 để giữ cùng số tham số: int(2·4d/3) làm tròn lên bội 256, thành 11 008 khi d = 4096 (`w2(silu(w1(x)) * w3(x))` trong mã Llama; Shazeer 2020). Llama còn dùng RMSNorm, RoPE, không bias. Mã Llama hỗ trợ grouped-query attention (`n_kv_heads`, `repeat_kv`), nhưng không phải cỡ nào cũng bật nó: Llama 1 và Llama-2-7B/13B dùng MHA đầy đủ; Llama 2 chỉ dùng GQA ở 34B và 70B; Llama 3 dùng 8 đầu K/V ở mọi cỡ.

**Khởi tạo kiểu nanoGPT (`model.py`).** Mọi Linear và Embedding lấy N(0, 0.02); các lớp chiếu ra của attention và FFN (`c_proj`) dùng 0.02/√(2·`n_layer`); `wte` buộc trọng số với `lm_head`. Lý do của 1/√(2·`n_layer`): mỗi khối cộng hai nhánh vào dòng residual nên phương sai cộng dồn theo 2·`n_layer`; chia như vậy giữ phương sai tổng cỡ hằng số. Mã nanoGPT ghi đây là cách làm "per GPT-2 paper" (`model.py` dòng 142, kiểm lại 04/11/2026); bài GPT-2 (Radford và cộng sự 2019) nhân trọng số các lớp residual với 1/√N, N là số lớp residual [cần kiểm chứng: số mục của paper; link chưa xác minh]. Lập luận phương sai ở trên là cách giải thích của sách, không phải trích từ paper.

**Huấn luyện theo paper gốc:** Adam β = (0.9, 0.98), ε = 1e-9, label smoothing 0.1, dropout 0.1, warmup 4000 bước, learning rate

$$
\mathrm{lr}=d^{-1/2}\min\left(\text{step}^{-1/2},\ \text{step}\cdot\text{warmup}^{-3/2}\right)
$$

Với d = 512 lịch đạt đỉnh 6.99e-4 tại bước 4000.

**Loss ban đầu.** Mô hình ngôn ngữ vừa khởi tạo cho loss cỡ ln V (chương 6): ln 50 257 = 10.825. Mô hình nhỏ của Lab 2 (V = 50, khởi tạo mặc định PyTorch) cho 4.135 so với ln 50 = 3.912. Chênh lệch không đến từ embedding (có `ln_f` phía sau; nhân embedding với 10 vẫn cho 4.07) mà từ `lm_head`: logit có std σ ≈ 0.58, và với logit độc lập với nhãn thì loss kỳ vọng ≈ ln V + σ²/2 = 3.912 + 0.17 = 4.08 (trung bình 20 seed: 4.07).

**Ví dụ giải mẫu 8.8 — đếm tham số một khối tí hon, và LN so với RMSNorm.**

- d = 4, `d_ff` = 16, 2 đầu, có bias. Attention: 4·16 + 16 = 80. FFN: 4·16 + 16 + 16·4 + 4 = 148. Hai LN: 2·(2·4) = 16. Tổng **244** (Lab 2 đếm bằng PyTorch: 80, 148, 8 mỗi LN).
- Tổng quát cho khối có bias: 12d² + 13d. Với d = 768: 7 077 888 + 9 984 = 7 087 872.
- Vector (2, 4, 6, 8). LN: trung bình 5, phương sai 5, nên (−3, −1, 1, 3)/√5 = (−1.342, −0.447, 0.447, 1.342). RMSNorm: RMS = √30 = 5.477 nên (0.365, 0.730, 1.095, 1.461); không đưa trung bình về 0.
- Pre-LN so với post-LN trên cùng một nhánh: bài tập 10.

% manim: PreLNvsPostLN

**Vì sao ML cần cái này.**

- Chỉ cần 8–10 con số (d, L, h, `d_ff`, V, T, vị trí đặt chuẩn hóa) là dựng lại cả mô hình và đếm được tham số (mục 8.5).
- Pre-LN hay post-LN, LN cuối, kiểu khởi tạo là chi tiết người phỏng vấn dùng để phân biệt người đã tự viết với người mới đọc.

## 8.5 Đếm tham số, FLOPs, KV cache và mặt nạ

Chi phí của Transformer đếm được bằng vài công thức; các số này quyết định cái gì chạy được trên RTX 4050 6 GB của bạn.

**Tham số của một khối (có bias).** Attention 4d² + 4d; FFN với `d_ff` = 4d là 8d² + 5d; hai LN là 4d. Tổng 12d² + 13d.

**Ví dụ giải mẫu 8.9 — GPT-2 small.** V = 50 257, T = 1024, d = 768, 12 khối.

- `wte` 38 597 376 + `wpe` 786 432 + 12 khối (85 054 464) + LN cuối 1536 = **124 439 808**, khớp bản dựng bằng đúng `model.py` của nanoGPT. nanoGPT in "123.65M" vì trừ `wpe` (123 653 376): nhầm 124M với 123.65M là bẫy quen thuộc.
- Không cộng `lm_head` vì buộc trọng số với `wte`; không buộc thì thêm 38 597 376, được 163 037 184.
- Các cỡ khác (Lab 5): medium (d = 1024, 24 khối) 354 823 168; large (1280, 36) 774 030 080; xl (1600, 48) 1 557 611 200.

**Ví dụ giải mẫu 8.10 (tùy chọn) — Transformer gốc của Vaswani, và vì sao lệch với paper.** Cách đếm (Lab 5): d = 512, `d_ff` = 2048, 6 lớp encoder và 6 lớp decoder, embedding dùng chung với V = 37 000 (paper dùng BPE dùng chung cỡ 37 000 cho En–De), có bias và LN, post-LN nên không có LN cuối. Lớp encoder 3 152 384, lớp decoder 4 204 032 (hai khối attention, FFN, ba LN); 6·(3 152 384 + 4 204 032) + 37 000·512 = **63 082 496** (63.08M). Bản big (d = 1024, `d_ff` = 4096): 214 245 376 (214.2M). Table 3 của paper báo 65M và 213M: cách đếm trên lệch −3.0% và +0.6%, chưa giải thích được (có thể do embedding, từ vựng, bias hay LN). Sách dùng số đếm kèm cách đếm, không coi là phép kiểm khớp paper. Với Llama-2-7B (vocab 32 000, d = 4096, 32 khối, SwiGLU ẩn 11 008, không buộc trọng số, không bias) công thức cho 6 738 415 616 (Lab 5; chưa đối chiếu checkpoint công bố).

**FLOPs.** Nhân ma trận m × k với k × n tốn 2mkn FLOPs. Forward cỡ 2N FLOPs mỗi token, backward gấp đôi, nên huấn luyện cỡ 6N; N là số tham số *tham gia nhân ma trận* (gồm `lm_head`; không gồm bảng chỉ để tra như `wpe`, hay bảng embedding đầu vào khi không buộc trọng số). Thêm số hạng attention theo ngữ cảnh n: forward 4·L·d·n mỗi token (QKᵀ và AV, mỗi cái 2·n·d), huấn luyện gấp ba. Đây là công thức trong `estimate_mfu` của nanoGPT:

$$
\text{FLOPs huấn luyện mỗi token} \approx 6N + 12\,L\,d\,n
$$

**Ví dụ giải mẫu 8.11 — FLOPs và điểm giao của attention.** GPT-2 small, n = T = 1024.

- Forward mỗi token: các khối 2·85 054 464 = 170 108 928; `lm_head` 77 194 752; attention (đếm đủ n × n, không giảm một nửa do mặt nạ) 37 748 736.
- Huấn luyện: 6·123 653 376 + 12·12·768·1024 = 741 920 256 + 113 246 208 = **855 166 464** FLOPs mỗi token; attention chiếm 13.2%.
- Attention so với (chiếu + FFN) mỗi lớp mỗi token: 4nd so với 24d², tỉ số n/(6d). Với d = 768: n = 512, 1024, 4096, 8192 cho 0.111, 0.222, 0.889, 1.778; attention chỉ vượt khi n > 6d = 4608.
- 6ND với D = 300e9 cho 2.23e20 FLOPs, cỡ 26% của 8.63e20 (4 ngày × 8 A100 × 312 TFLOPS, theo README nanoGPT; ước lượng, vì README ghi 4 ngày còn config ghi 5).

**Bộ nhớ của ma trận điểm.** n × n mỗi đầu, mỗi mẫu, mỗi lớp. GPT-2 small, B = 8, T = 1024, 12 đầu, fp32: 0.375 GiB mỗi lớp, 4.5 GiB cho 12 lớp nếu lưu cho backward; 32 đầu, T = 4096, fp16: 1 GiB mỗi lớp mỗi mẫu (16 GiB ở T = 16 384). Vì thế cần kernel không dựng ma trận n × n, như FlashAttention (Dao và cộng sự, 2022) và `F.scaled_dot_product_attention` (số là phép tính, không đo trên GPU). Adam fp32 cho GPT-2 small: trọng số + gradient + m + v = 1.85 GiB, chưa tính activation.

**KV cache.** Lúc sinh, token thứ t chỉ cần K và V của các token trước, và chúng không đổi (mặt nạ nhân quả), nên tính một lần rồi giữ lại. Số byte mỗi token:

$$
\text{byte mỗi token} = 2 \cdot L \cdot n_{kv} \cdot d_h \cdot \text{(byte mỗi phần tử)}
$$

với `n_kv` là số đầu K/V, dₕ là độ rộng một đầu, hệ số 2 là K và V.

**Ví dụ giải mẫu 8.12 — KV cache.**

- GPT-2 small, fp16: 2·12·12·64·2 = 36 864 byte (36 KiB) mỗi token, 36 MiB ở T = 1024.
- Llama-2-7B (32 lớp, 32 đầu, dₕ = 128), fp16: 524 288 byte (512 KiB) mỗi token, 2 GiB ở T = 4096 với batch 1, 16 GiB với batch 8 (trọng số fp16: 12.55 GiB).
- Grouped-query attention (GQA, Ainslie và cộng sự 2023): nhiều đầu query dùng chung một số ít đầu K/V. 8 đầu K/V cho 0.5 GiB (nhỏ hơn 4 lần); multi-query attention (MQA, Shazeer 2019) với 1 đầu cho 0.0625 GiB. Llama-2-70B (80 lớp, 8 đầu K/V, dₕ = 128; báo cáo Llama 2 xác nhận chỉ 34B và 70B dùng GQA, riêng 80 lớp chưa kiểm): 320 KiB mỗi token, 1.25 GiB ở T = 4096.
- Lab 4 (pre-LN, L = 3, d = 64, h = 4, T = 32): sinh bằng cache cho cùng logit với forward đầy đủ, sai lệch lớn nhất 6.0e-7; cache có 12 288 phần tử = 2·L·h·T·dₖ = 2·3·4·32·16.
- Tốc độ: không cache thì mỗi bước phải xử lý lại cả chuỗi, tổng n(n + 1)/2 lượt xử lý một token qua cả mạng (n = 1000: 500 500); có cache thì mỗi bước chỉ xử lý token mới, tổng n (1000). Lab 4 đo trên mô hình rất nhỏ (31 token, 7 lần, máy 2 lõi) chỉ thấy tỉ lệ thời gian khoảng 1 đến 2 lần tùy lần chạy (các lần chạy ghi nhận 1.0 đến 2.0), vì overhead Python lấn át; đừng dán con số đó làm bằng chứng cho tỉ lệ lý thuyết.
- **Prefill so với decode.** Prefill xử lý cả prompt T token một lượt: với các phép nhân với trọng số, (T × d)(d × d′), mỗi trọng số đọc một lần và dùng cho T token, cường độ số học cỡ T FLOP/byte (fp16). Chỉ khi T vượt "điểm gãy" của phần cứng (FLOPs đỉnh chia băng thông bộ nhớ, cỡ hàng trăm FLOP/byte với GPU trung tâm dữ liệu [con số cụ thể cần kiểm chứng theo datasheet]) prefill mới bị chặn bởi FLOPs; prompt ngắn thì prefill cũng bị chặn bởi bộ nhớ. Decode batch 1 là nhân ma trận với vector: 2dd′ FLOPs trên 2dd′ byte, cỡ 1 FLOP/byte, thấp xa điểm gãy, nên bị chặn bởi băng thông bộ nhớ. Tăng batch B nâng cường độ của các phép nhân với trọng số lên cỡ B (một lần đọc trọng số phục vụ B chuỗi), nhưng **không** nâng cường độ của attention trên KV cache: mỗi chuỗi có cache riêng nên phải đọc cache riêng của nó, cỡ 1 FLOP/byte dù batch lớn, và khi ngữ cảnh dài phần này chiếm phần lớn byte phải đọc (cũng là lý do GQA, MQA giúp tăng thông lượng). (Suy luận từ số liệu trên; chưa đo băng thông thiết bị.)

% manim: KVCache

**Mặt nạ (mask).** (1) Nhân quả: đặt −∞ vào các ô nằm phía trên đường chéo chính (j > i, không kể đường chéo) **trước** softmax; đường chéo giữ nguyên để mỗi token còn nhìn thấy chính nó. (2) Padding: che các cột của token đệm. (3) Kết hợp: ô bị che nếu bất kỳ mặt nạ nào che. (4) Cross-attention chỉ cần padding của nguồn.

**Ví dụ giải mẫu 8.13 — các bẫy của mặt nạ bằng số.** Dùng A của ví dụ 8.3.

- Che **sau** softmax: hàng 1 thành (0.4011, 0, 0), tổng 0.4011; hàng 2 tổng 0.5989; hàng 3 tổng 1 (che đúng: mọi hàng tổng 1). Mẫu số của softmax đã chứa các ô tương lai nên thông tin rò qua nó: ô Phân tích từng lỗi của notebook đo rò rỉ 6.4471 (bản đúng: 0.0).
- fp16: số hữu hạn nhỏ nhất là −65 504. Ép kiểu −1e9 sang fp16 (`torch.tensor(-1e9, dtype=torch.float16)`) cho −∞, và một hàng toàn −∞ cho softmax NaN (Lab 1 xác nhận cả hai). Nhưng `masked_fill(mask, -1e9)` trực tiếp trên tensor fp16 thì PyTorch 2.14.1 **báo lỗi** (đã chạy; thông báo nói giá trị tràn kiểu Half), không tràn im lặng; −1e9 chỉ tràn im lặng khi mặt nạ được cộng ở fp32 rồi mới đổi kiểu. Cách an toàn là `torch.finfo(dtype).min` hoặc −∞ thật. Ở fp32, hàng ba giá trị −1e9 hữu hạn lại cho phân phối **đều** (1/3, 1/3, 1/3): hai cách che cho hai hành vi khác nhau với hàng bị che hết (ví dụ hàng toàn padding).
- `F.scaled_dot_product_attention(..., is_causal=True)` khớp bản tay tới 9e-16 ở float64 (Lab 1).
- *Phép thử rò rỉ tương lai* dùng được cho mọi mô hình: đổi các token ở vị trí t ≥ t₀ rồi so logit ở vị trí < t₀; mô hình đúng cho đúng 0.0. Lab 4: mô hình đúng 0.0, mô hình quên mặt nạ 0.0992 (chưa huấn luyện).

**Ví dụ giải mẫu 8.14 — mặt nạ padding và mặt nạ kết hợp.** 4 token, dₖ = 2; ba token đầu như ví dụ 8.3, token 4 là đệm với q = k = (2, 2), v = (100, 100) (giá trị lớn để lộ lỗi). Không che padding, hàng 1 của A là (0.2212, 0.1091, 0.2212, 0.4486): token đệm chiếm gần nửa trọng số và kéo O₁ về (46.5, 47.1). Che cột 4 bằng −∞ trước softmax thì hàng 1 là (0.4011, 0.1978, 0.4011, 0), và O₁, O₂, O₃ đúng bằng kết quả ví dụ 8.3 tính trên 3 token thật. Kết hợp với mặt nạ nhân quả: đệm **phải** thì mặt nạ nhân quả đã che sẵn cột 4 cho ba hàng đầu; đệm **trái** (batch sinh văn bản) thì mọi hàng của các token đệm đều bị che hết (hàng của token đệm thứ hai chỉ còn thấy các ô đệm, và các ô đó đã bị mặt nạ padding che), và softmax của các hàng đó phải trả về một quy ước (toàn 0) chứ không phải NaN. Lab 1 kiểm cả hai.

**Vì sao ML cần cái này.**

- Trả lời "cần bao nhiêu bộ nhớ, tốn bao nhiêu FLOPs" bằng giấy bút trong 2 phút phân biệt người đã làm việc với LLM với người chỉ gọi API; KV cache giải thích vì sao phục vụ LLM bị giới hạn bởi bộ nhớ và vì sao GQA, MQA ra đời.
- Phép thử rò rỉ và kiểm "cache khớp forward đầy đủ" là hai kiểm thử nên có cho mọi mô hình tự viết; chúng bắt được lỗi mà đường cong loss không bắt được.

## Bài tập giấy bút

14 bài, làm theo thứ tự. Mục tiêu: hết ★ trong 1 buổi, hết ★★ trong 2 buổi, ★★★ dành cho Thứ Bảy. Bài nào sai ghi vào sổ lỗi kèm lý do sai.

**★ Hiểu định nghĩa**

1. Bước giải mã có s = (0, 1), h₁ = (1, 0), h₂ = (1, 1), W = U = I, v = (1, 1). Tính α và c bằng attention cộng tính, rồi bằng dot-product e = h·s.
2. X ∈ ℝ¹⁰ˣ⁵¹², h = 8. Viết hình dạng của `W_Q`, Q, Q một đầu, ma trận điểm một đầu, A, đầu ra sau gộp đầu và `W_O`. Tổng tham số MHA có bias?
3. Tính PE của vị trí 5 với d = 4 (hai cặp, tần số 1 và 0.01).
4. Đếm số tham số của một khối pre-LN có bias với d = 8, 2 đầu, `d_ff` = 32. Tách thành attention, FFN, LN.

**★★ Mức đề thi đầu vào**

5. Chứng minh Var(q·k) = dₖ khi các thành phần của q, k độc lập, trung bình 0, phương sai 1. Nếu phương sai là σ² và τ² thì sao? Vì sao chia √dₖ mà không chia dₖ?
6. Chứng minh Attn(PX) = P·Attn(X) với ma trận hoán vị P. Chỉ ra bước nào trong chứng minh hỏng khi thêm mặt nạ nhân quả.
7. Q có các hàng (2, 0), (0, 2); K = V = I₂ (ma trận đơn vị); dₖ = 2. Tính A và O, rồi tính lại với mặt nạ nhân quả. Hàng nào không đổi và vì sao?
8. d = 768, h = 12. (a) Số tham số của MHA có bias. (b) Có thay đổi khi h = 1 không? (c) Với B = 8, T = 1024, fp16, bộ nhớ của các ma trận điểm của một lớp. (d) Vậy dùng nhiều đầu có "miễn phí" không?
9. Chứng minh (R(mθ)q)ᵀ(R(nθ)k) = qᵀR((n − m)θ)k. Tính với q = (1, 0), k = (1, 1), θ = π/2 và (m, n) = (3, 1), bằng cả hai vế.
10. Một nhánh con có Sub(y) = 2y. Với x = (1, 3) tính đầu ra của khối post-LN và pre-LN (bỏ ε). Cái nào giữ được thông tin về độ lớn của x?

**★★★ Mức phỏng vấn nghiên cứu**

11. Mỗi nhánh con cộng vào mỗi thành phần của dòng residual một số ngẫu nhiên độc lập, trung bình 0, phương sai σ²; đầu vào phương sai 1. Tìm phương sai và std sau L khối pre-LN (2L nhánh). Dùng std 1.020 sau 1 khối (Lab 2) để ước σ², dự đoán std sau 24 khối rồi so với 1.555. Chênh lệch nói gì về giả thiết độc lập? Từ đó giải thích hệ số 0.02/√(2·`n_layer`) của nanoGPT.
12. Giả thiết: dropout tắt, chỉ số vị trí đúng, mặt nạ nhân quả. Chứng minh giải mã có KV cache cho đúng cùng logit với forward đầy đủ (phát biểu quy nạp theo lớp ℓ). Đếm số lượt xử lý một token qua cả mạng của hai cách ở n = 1000 và dung lượng cache của GPT-2 small khi n = 1000. Khi nào cache không đáng giá?
13. (a) Tính bước sóng chậm nhất của RoPE với d = 128 ở base 10 000 và 500 000, và giải thích vì sao RoPE chỉ áp lên q, k mà không áp lên v. (b) *Online softmax.* Tách các logit thành hai khối; mỗi khối giữ (m, Z, o) với m = max logit, Z = ∑exp(s − m), o = ∑exp(s − m)·v. Chứng minh công thức gộp hai khối ra đúng softmax toàn cục, rồi tính với logit (1, 3 | 2, 0) và giá trị vô hướng v = (1, 5 | 2, 4). Đây là phép gộp mà FlashAttention dùng để không dựng ma trận n × n.
14. *Backward của softmax-attention.* Với S = QKᵀ/√dₖ, A = softmax(S) theo hàng, O = AV và dO cho trước: (a) dẫn xuất dV, dA, dS, dQ, dK, trong đó dS = A ⊙ (dA − (dA ⊙ A)·1) (nhân từng phần tử, tổng theo hàng); (b) với một hàng A = (0.7311, 0.2689) và dA = (1, 0) tính dS và đối chiếu với Jacobian diag(p) − ppᵀ; (c) chứng minh ∑ⱼ dAᵢⱼAᵢⱼ = dOᵢ·Oᵢ, tức số hạng trừ chỉ cần dO và O, không cần dựng lại A (mẹo của FlashAttention).

## Bài tính nhanh

15 bài, mỗi bài tối đa 2 phút, làm bằng giấy bút và bấm giờ cả bộ. Mục tiêu: xong cả 15 bài trong 25 phút, sai không quá 1 bài. Làm lại bộ này sau 3 ngày và sau 1 tuần.

1. Softmax của (0, ln 3).
2. q = (1, 2, 3, 4), k = (4, 3, 2, 1). Tính q·k và q·k/√dₖ.
3. Với dₖ = 16, std của q·k (các thành phần chuẩn độc lập) là bao nhiêu?
4. Số tham số MHA có bias, d = 256.
5. d = 1024, 16 đầu: dₖ bằng bao nhiêu?
6. Số tham số khối có bias, d = 100 (`d_ff` = 4d).
7. FFN không bias, d = 512, `d_ff` = 2048: số tham số.
8. LN của (1, 3), bỏ ε.
9. RMSNorm của (3, 4), bỏ ε.
10. KV cache fp16 mỗi token: 24 lớp, 16 đầu K/V, dₕ = 64.
11. FLOPs của phép nhân (128 × 768)(768 × 3072).
12. 6ND với N = 1.5e9 tham số và D = 3e10 token.
13. Mặt nạ nhân quả n = 5: có bao nhiêu ô được phép nhìn (kể cả đường chéo)?
14. Loss ban đầu của mô hình ngôn ngữ có từ vựng 32 000.
15. Ma trận điểm fp16 của 16 đầu, T = 2048, một mẫu: bao nhiêu MiB?

## Lab code và bài săn bug

Sáu lab đã chạy với Python 3, NumPy 2.5, PyTorch 2.14.1 (bản CPU đủ). Notebook lời giải chạy khoảng 80 giây trên máy build 2 lõi khi rảnh, tới 190 giây khi máy đang bận (in ở ô cuối); vượt mức dưới 2 phút của quy chuẩn chung vì Lab 6 và bài săn bug huấn luyện nhiều mô hình nhỏ, và bạn có thể bỏ Lab 6 (tùy chọn) để ngắn hơn; số liệu lệch nhẹ theo phiên bản và số luồng. Dữ liệu tự sinh, seed cố định. Lab 6 là tùy chọn và nằm cuối notebook, nên bỏ qua nó vẫn Run All được phần còn lại.

| Lab | Việc phải làm | Kết quả mong đợi |
| --- | --- | --- |
| 1. Attention từ đầu (NumPy) | `softmax` ổn định (phải qua softmax(1000, 1001)), `attention(Q, K, V, causal, pad)`; hoán vị-đẳng biến, std(q·k), mặt nạ padding, bẫy mặt nạ | O₃ = (3.5105, 4.5105), O₂ có mặt nạ = (2.3395, 3.3395); khớp PyTorch tới 9e-16; std(q·k) = 7.98 (dₖ = 64); max p 0.867 so với 0.322; dₖ = 512, trung vị ‖∂L/∂q‖ 0.0017 (không chia) so với 0.30 (có chia), trung bình 1.2 so với 0.32; mask sau softmax: tổng hàng (0.4011, 0.5989, 1.0); −1e9 ép sang fp16 thành −inf, `masked_fill(-1e9)` trên fp16 báo lỗi; token đệm không ảnh hưởng token thật |
| 2. Multi-head và khối pre-LN | `MHA`, `Block`; đếm tham số; so với `nn.MultiheadAttention`; đo std dòng residual | Khối d = 4: 244 tham số; MHA d = 512: 1 050 624 với h = 1, 8, 16; sai khác PyTorch 1.2e-7; std residual pre-LN 1.020 / 1.128 / 1.257 / 1.555 sau 1 / 6 / 12 / 24 khối, post-LN 1.0 |
| 3. Mã hóa vị trí | `sinusoidal`, `rope_thetas`, `rope`; tích vô hướng chỉ phụ thuộc khoảng cách; bài "token đầu > token cuối" | Bảng d = 4 khớp ví dụ 8.5; PE(p)·PE(p + 5) = 23.504; RoPE đều −1.8301; hoán vị: 4.8e-7 (không causal), 0.807 (causal); acc có PE 1.000, không PE 0.549 |
| 4. KV cache và rò rỉ tương lai | `decode_step` có cache; `cached_logits`; `leak_test` | Sai lệch 6.0e-7; 12 288 phần tử cache; rò rỉ 0.0 (đúng) và 0.0992 (quên mặt nạ); tốc độ cache: khoảng 7 lần đo |
| 5. Máy tính chi phí | `params`, `flops_per_token`, `kv_bytes`, `params_vaswani` | GPT-2 small 124 439 808; 855 166 464 FLOPs/token; Llama-2-7B 6 738 415 616; KV 2 / 0.5 / 0.0625 GiB (MHA, GQA 8, MQA); Vaswani base 63 082 496, big 214 245 376 |
| 6 (tùy chọn, cuối notebook). RNN so với attention | `jac_norm`; seq2seq GRU có và không có attention trên bài chép chuỗi | ρ = 0.9: 7.33e-1 → 9.63e-12 (T = 5 → 100); ρ = 1: ‖W‖₂ = 1.97, bỏ tanh thì ‖W¹⁰⁰‖₂ = 2.62, W trực giao 1.2e-7; chép chuỗi độ dài 8: 0.667 so với 0.994; độ dài 16: 0.350 so với 0.986 (0.630 nếu h = 64, 600 bước) |

Code nằm trong `labs/chuong08/lab.ipynb` (có chỗ TODO và ô kiểm tra tự động) và `labs/chuong08/loigiai.ipynb` (lời giải đã chạy kiểm); mở từ trang **Lab** của app. Làm theo thứ tự: Lab 3, 4 và săn bug dùng lại `MHA`, `Block`, `LM` của Lab 2.

**Câu hỏi sau lab.**

- Lab 1: vì sao mọi hàng của A có tổng 1 và đầu ra nằm trong bao lồi của các hàng V? Vì sao std(q·k) ≈ √dₖ vẫn đúng khi q, k là Gauss nhưng có thể sai sau huấn luyện?
- Lab 2: vì sao số tham số của MHA không đổi khi tăng h, và vì sao std residual của pre-LN tăng còn của post-LN thì không?
- Lab 3: vì sao mô hình causal không PE không còn hoán vị-đẳng biến (0.807), trong khi mô hình không causal thì có (4.8e-7)?
- Lab 4: vì sao cache khớp forward đầy đủ nhưng không bắt buộc nhanh hơn trên mô hình bé? Vì sao GQA 8 đầu K/V nhỏ hơn đúng 4 lần so với 32 đầu?
- Lab 6: vì sao ngay cả ρ = 1.0 vẫn có gradient biến mất (1.98e-8 ở T = 100), và tính không chuẩn tắc của W đóng vai trò gì?

**Bài săn bug.** Mô hình ngôn ngữ decoder-only 2 khối (d = 64, 4 đầu, T = 16, từ vựng 50, Adam 1e-3, 300 bước, dropout 0.1) chạy không báo lỗi. Hai bài kiểm: (1) dự đoán token kế tiếp trên chuỗi **ngẫu nhiên**, nên loss không thể xuống dưới ln 50 = 3.912; (2) bài `lag1`: mục tiêu tại vị trí t là token ngay trước đó, học được trong 300 bước nếu có thông tin vị trí.

Triệu chứng đo được (bản lỗi, mọi lỗi cùng lúc):

| Bài kiểm | loss train | loss trên batch mới | hai lần gọi cùng input chênh | rò rỉ tương lai |
| --- | --- | --- | --- | --- |
| Nhãn ngẫu nhiên | 3.148 | 3.086 | 4.592 | 6.1084 |
| `lag1` | 1.000 | 0.984 | 6.470 | 0.6419 |

Bản đúng cho 3.921 / 3.918 / 0.000 / 0.0000 (nhãn ngẫu nhiên) và 0.019 / 0.011 / 0.000 / 0.0000 (`lag1`). "Rò rỉ" là thay đổi lớn nhất của logit ở các vị trí < 10 khi đổi các token ở vị trí ≥ 10 (phải đúng 0.0). Tóm lại: loss trên nhãn ngẫu nhiên xuống dưới ln 50, `lag1` kẹt ở khoảng 1.0 thay vì 0.01, đổi tương lai làm quá khứ thay đổi, và gọi model hai lần cùng input cho kết quả khác nhau. Tìm cả **bốn** lỗi rồi viết `train_fixed` trong notebook. Công cụ: phép so với `nn.MultiheadAttention` (Lab 2) và phép thử rò rỉ (Lab 4).

Phần còn lại (`BuggyBlock` pre-LN dropout 0.1, `BuggyLM` có bảng vị trí học được `pe`, `train_buggy`) nằm ở ô đầu của phần Săn bug trong `lab.ipynb`. Đoạn cần đọc:

```python
class BuggyAttention(nn.Module):
    # __init__: self.h = h; self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
    def forward(self, x):
        B, T, d = x.shape
        dk = d // self.h
        q, k, v = self.qkv(x).split(d, dim=2)
        q = q.reshape(B, self.h, T, dk)
        k = k.reshape(B, self.h, T, dk)
        v = v.reshape(B, self.h, T, dk)
        A = (q @ k.transpose(-2, -1) / math.sqrt(dk)).softmax(-1)
        A = A * torch.tril(torch.ones(T, T))
        return self.o((A @ v).reshape(B, T, d))

# BuggyLM.forward:   x = self.te(idx)      (self.pe được tạo ở __init__ nhưng không dùng)
# train_buggy, đoạn đo:   xe, ye = task(512, ...);  with torch.no_grad(): o1, o2 = model(xe), model(xe)   (không gọi model.eval() trước)
```

## Bài dựng cảnh Manim

Mỗi chương có một bài tự dựng cảnh: muốn vẽ được một khái niệm, bạn phải hiểu từng bước của nó. Giới hạn **1 giờ** (tính vào luồng C), không chỉnh màu mè.

**Đề bài.** Dùng `manim/chuong08_scenes.py` làm mẫu, viết cảnh `MatNaNhanQua` vào thư mục `manim/cua-toi/`:

1. Lấy ma trận điểm của ví dụ 8.3 (QKᵀ/√2, các hàng (0.707, 0, 0.707), (0, 0.707, 0.707), (0.707, 0.707, 1.414)).
2. Vẽ hai cột cạnh nhau. Cột trái: che các ô j > i bằng −∞ **trước** softmax. Cột phải: softmax trước, rồi nhân với mặt nạ 0/1 (**sau** softmax). Hiện ma trận A của từng cột.
3. Hiện tổng của từng hàng: cột trái cho (1, 1, 1); cột phải cho (0.4011, 0.5989, 1.0), tô đỏ các hàng không còn tổng 1.

**Tiêu chí tự chấm.** (1) Đúng toán: các hàng của cột trái là (1, 0, 0), (0.3302, 0.6698, 0), (0.2483, 0.2483, 0.5035), và tổng của cột phải khớp phép cộng tay. (2) Có tiêu đề và phụ đề nói người xem phải thấy điều gì (hàng nào mất khối lượng, vì sao mẫu số của softmax vẫn chứa các ô tương lai). (3) Dài không quá 30 giây, chữ không chồng chữ.

Render bằng nút **Render cảnh của tôi** ở trang Manim. Lời giải tham khảo nằm ở `manim/loigiai/chuong08_mat_na.py` — chỉ mở sau khi đã tự làm.

## Phỏng vấn, bẫy, thực tế khác lý thuyết

**Cách luyện.** Mỗi chuỗi có 4 tầng câu hỏi, đào sâu dần. Trả lời to bằng tiếng Việt, rồi nói lại bằng tiếng Anh, 2 phút mỗi tầng, ghi âm và tự nghe. Câu trả lời mẫu chỉ phủ tầng 1–2; tầng 3–4 bạn tự soạn rồi đối chiếu với lời giải các bài tập liên quan. Các câu hỏi dưới đây là câu điển hình về chủ đề này, do người viết diễn đạt lại, không lấy từ một bộ đề phỏng vấn cụ thể.

**Chuỗi 1 — Vì sao attention**

1. Vì sao attention ra đời? Nó chữa vấn đề gì của seq2seq RNN?
2. Nút thắt thông tin và gradient dài thể hiện thế nào, có số liệu gì? Attention của Bahdanau tính gì (eᵢⱼ, αᵢⱼ), khác dot-product thế nào?
3. So sánh self-attention và RNN về số phép tính, số bước tuần tự, độ dài đường đi. Khi nào self-attention đắt hơn?
4. Vì sao O(n²) vẫn đáng giá, và FlashAttention với KV cache nằm ở đâu trong bức tranh đó?

> *Model answer:* In a plain RNN encoder–decoder the whole source sentence is squeezed into one fixed-size vector, so long inputs lose information, and the gradient from the end of the sequence to the start is a product of n Jacobians that tends to vanish. Bahdanau attention replaces that vector with a context vector, a softmax-weighted average of all encoder states: a differentiable soft lookup with a direct path from every source position to every target step. The original scores come from a small MLP; Luong showed a plain dot product works and is just a matrix multiplication. In a toy copy task I measured token accuracy of 0.35 without attention versus 0.99 with it at length 16.

**Chuỗi 2 — Scaled dot-product và multi-head**

1. Viết scaled dot-product attention và hình dạng các tensor.
2. Vì sao chia √dₖ? Chứng minh Var(q·k) = dₖ. Điều gì xảy ra với softmax và gradient nếu không chia? Vì sao không chia cho dₖ hay √d?
3. Vì sao nhiều đầu? Số tham số có tăng theo số đầu không? Một đầu với dₖ = d thì khác gì? Vì sao code dùng view rồi transpose?
4. Khi nào √dₖ không đủ? Nếu không chia, gradient trung bình có nhỏ không (hãy nhìn trung bình lẫn trung vị)? (Từ khóa: QK-norm, logit growth, attention entropy collapse.) Người phỏng vấn phản biện: "nhưng mô hình nhỏ của bạn không chia √dₖ vẫn học tốt": trả lời bằng mục Thực tế khác lý thuyết. Trọng số attention có phải là lời giải thích không?

> *Model answer:* Attention computes softmax(QKᵀ/√dₖ)V: Q and K are (B, h, T, dₖ), the scores are (B, h, T, T), and the output is (B, h, T, dₖ), merged back to (B, T, d). If the entries of q and k are independent with mean 0 and variance 1, q·k is a sum of dₖ terms of variance 1, so its variance is dₖ. Without the scaling the softmax saturates toward one-hot and its Jacobian diag(p) − ppᵀ is nearly zero for most queries, so most of them get almost no gradient while a few near-ties get large ones; Vaswani et al. only say they suspect this is why scaling helps. I measured an average max probability of 0.32 with scaling versus 0.95 without it at dₖ = 512, and a median gradient norm with respect to q of 0.30 versus 0.0017. Multi-head splits d into h subspaces, so the parameter count stays 4d² plus biases whatever h is.

**Chuỗi 3 — Mã hóa vị trí**

1. Vì sao cần mã hóa vị trí?
2. Sinusoidal có công thức và tính chất gì? RoPE hoạt động thế nào và vì sao điểm chỉ phụ thuộc n − m?
3. Vì sao RoPE chỉ áp lên q và k? So sánh learned, sinusoidal, RoPE, ALiBi.
4. Mô hình causal không có PE có biết thứ tự không? Còn việc ngoại suy độ dài (position interpolation, đổi base, NoPE) thì sao?

> *Model answer:* Self-attention is permutation-equivariant, so without position information a bidirectional encoder treats a sentence as a bag of words: in my experiment a two-block encoder without positions scored 0.549 on a "first token greater than last token" task, the majority-class rate, versus 1.000 with learned positions. Sinusoidal encodings use sines and cosines at geometric frequencies, so a shift by k is a fixed rotation of each pair. RoPE rotates q and k by angles proportional to position; since rotations are orthogonal, (R(mθ)q)ᵀ(R(nθ)k) = qᵀR((n−m)θ)k depends only on the offset n − m.

**Chuỗi 4 — Khối Transformer và chi phí**

1. Vẽ một khối pre-LN và nói vai trò từng phần.
2. Pre-LN và post-LN khác gì? Vì sao Transformer dùng LayerNorm thay BatchNorm? Vì sao cần LN cuối?
3. Đếm tham số GPT-2 small. Vì sao FFN rộng gấp 4? FLOPs huấn luyện ≈ 6ND đến từ đâu?
4. KV cache lớn bao nhiêu, MQA và GQA tiết kiệm gì? Vì sao giải mã ở batch 1 bị chặn bởi băng thông bộ nhớ còn prefill thường thì không? Prefill có luôn bị chặn bởi FLOPs không, và tăng batch giúp được phần nào của decode?

> *Model answer:* A pre-LN block computes x + MHA(LN(x)), then x + FFN(LN(x)): the norm sits inside the branch and the residual stream is left unnormalized. The original paper normalizes after the addition (post-LN); Xiong et al. argue this makes gradients near the output large at initialization and training sensitive to warmup. LayerNorm works per token, so it does not depend on batch size or padding. Because the pre-LN stream is unnormalized its scale grows with depth, hence the final LayerNorm; I measured the std rising from 1.02 to 1.56 over 24 blocks. A block has 12d² + 13d parameters; adding the token and position embeddings and the final LayerNorm gives 124 439 808 for GPT-2 small.

**Từ vựng nên nói trôi chảy:** scaled dot-product attention, query/key/value, multi-head, softmax saturation, permutation-equivariant, causal mask, positional encoding, rotary embedding, residual stream, pre-LN, KV cache, grouped-query attention, memory-bound, FLOPs per token.

**Bẫy thường gặp.**

- **Chia sai hệ số**: chia dₖ thay vì √dₖ, hoặc √d (kích thước mô hình) thay vì √dₖ của một đầu.
- **Mặt nạ sai**: che sau softmax, softmax sai chiều (`dim=-2` chuẩn hóa theo query; shape vẫn đúng), −1e9 ở fp16 (ép kiểu thành −∞ rồi NaN ở hàng bị che hết; `masked_fill` trực tiếp thì báo lỗi, nên dùng `torch.finfo(dtype).min`), hàng bị che hết, quên mặt nạ padding. Dấu hiệu chung: loss tốt một cách vô lý. Phép thử: nhãn ngẫu nhiên không thể giảm dưới ln V; đổi tương lai thì quá khứ không đổi.
- **Tách đầu sai**: `reshape(B, h, T, dk)` thay cho `view(B, T, h, dk).transpose(1, 2)`: shape đúng, loss bình thường, không rò rỉ (ô Phân tích từng lỗi: 3.918 và 0.0000); chỉ so với bản chuẩn mới bắt được (chênh 0.894 với cùng trọng số).
- **Vị trí**: quên cộng PE; learned PE gặp chuỗi dài hơn `block_size`; KV cache dùng chỉ số vị trí sai; **quy ước ghép cặp RoPE** (liền kề hay i với i + d/2) lệch giữa mã và checkpoint. Phải có test cache khớp forward đầy đủ (sai khác cỡ 6e-7).
- **Đối chiếu với paper và đếm tham số**: Annotated Transformer là pre-LN còn paper là post-LN; nanoGPT không nhân embedding với √d còn paper có; nhiều đầu không thêm tham số; đừng đếm hai lần `wte` buộc với `lm_head`; 124M khác 123.65M ở bảng `wpe`.

**Thực tế khác lý thuyết.**

- **"Không có PE thì mất thứ tự" chỉ đúng với mô hình không causal.** Mô hình causal không PE không hoán vị-đẳng biến (0.807, Lab 3) và học được một phần việc phụ thuộc thứ tự (loss 0.676 ở `lag1` so với 0.014 khi có PE, cùng điều kiện).
- **Attention O(n²) chưa chắc là nút thắt FLOPs, nhưng là nút thắt thật về bộ nhớ.** Về FLOPs nó (đếm đủ n × n, không giảm một nửa do causal) chỉ vượt FFN cộng phép chiếu khi n > 6d (4608 với d = 768); ở n = 1024 là 22% của phần còn lại (13.2% tổng). Về bộ nhớ, ma trận điểm tốn 1 GiB mỗi lớp mỗi mẫu ở n = 4096 với 32 đầu fp16. KV cache giảm số lượt xử lý token từ n(n + 1)/2 xuống n nhưng đo trên mô hình bé chỉ thấy khoảng 1–2 lần vì overhead Python.
- **Con số công bố không tái hiện chính xác, và vài hiệu ứng không thấy ở quy mô nhỏ.** Đếm Transformer base ra 63.08M so với 65M của paper (−3.0%, chưa giải thích được); nanoGPT ghi "khoảng 4 ngày" ở README và "khoảng 5 ngày" ở config. Lab của chương **không** tái hiện "post-LN cần warmup" và "không chia √dₖ thì không huấn luyện được": các hiệu ứng đó được báo cáo ở quy mô lớn hơn (dịch máy, mô hình sâu, lr lớn, độ chính xác thấp), chưa đo ở đây, nên đừng dùng bài lab làm bằng chứng phản đối.

## Mini-test 30 phút

Làm sau khi xong bài tập và lab, bấm giờ đúng 30 phút, không xem sách, không dùng máy tính (câu 5 chỉ đọc code, không chạy). Tổng 30 điểm. **Từ 24 điểm:** sang chương 9. **18 đến dưới 24:** ôn lại mục ứng với câu sai rồi làm lại sau 3 ngày. **Dưới 18:** học lại chương.

1. **(4 điểm)** Hai token, dₖ = 2, Q = K có các hàng (1, 0) và (1, 1), V có các hàng (2, 0) và (0, 4). Tính A và O. Rồi tính lại với mặt nạ nhân quả. (Cho exp(0.7071) ≈ 2.028, exp(1.4142) ≈ 4.113.)
2. **(4 điểm)** (Chỉ cần biểu thức, không cần nhân ra từng số lớn.) Mô hình GPT nhỏ: V = 10 000, T = 256 (vị trí học được), d = 512, 6 khối, 8 đầu, có bias, trọng số `wte` buộc với `lm_head`. Đếm tổng số tham số. Huấn luyện bằng Adam fp32 thì trọng số, gradient, m, v tốn bao nhiêu byte? KV cache fp16 mỗi token là bao nhiêu byte?
3. **(5 điểm)** Chứng minh Var(q·k) = dₖ. Với dₖ = 64, nếu logit của hai key là (8, 0) thì softmax và đạo hàm p₁(1 − p₁) bằng bao nhiêu, và sau khi chia √dₖ thì bằng bao nhiêu? Vì sao không chia cho dₖ?
4. **(5 điểm)** Chứng minh RoPE cho điểm chỉ phụ thuộc n − m. Tính với q = (1, 1), k = (1, 0), θ = π/4 và (m, n) = (4, 1). Vì sao không áp RoPE lên v?
5. **(6 điểm)** Đọc code và trả lời không chạy máy:

```python
q, k, v = self.qkv(x).split(d, dim=2)
q = q.reshape(B, h, T, d // h)
k = k.reshape(B, h, T, d // h)
v = v.reshape(B, h, T, d // h)
a = (q @ k.transpose(-2, -1)) / math.sqrt(d)
a = a.softmax(dim=-2)
a = a * torch.tril(torch.ones(T, T))
return self.o((a @ v).reshape(B, T, d))
```

(a) Tìm **bốn** lỗi. (b) Nêu hai phép thử tự động sẽ lộ ra các lỗi này mà không cần biết đáp án.

6. **(6 điểm)** Tối đa 6 câu: vì sao giải mã ở batch 1 bị chặn bởi băng thông bộ nhớ chứ không phải FLOPs? KV cache lớn bao nhiêu với Llama-2-7B ở T = 4096, và GQA 8 đầu K/V thay đổi điều đó thế nào? Tăng batch có giải quyết được nút thắt băng thông ở decode không?

## Flashcard

20 thẻ của chương nằm trong `flashcards/chuong08.csv`. Ôn ở trang **Ôn tập** của app: app tự lên lịch theo hộp Leitner (thẻ nhớ được sẽ giãn lịch ôn 1 → 3 → 7 → 14 ngày, thẻ quên quay lại hộp 1). File CSV này cũng nhập thẳng được vào Anki nếu bạn thích dùng Anki.

## Lời giải

Chỉ đọc sau khi đã tự thử ít nhất 20 phút. Với bài chứng minh, so ý chính chứ không so từng chữ. Đáp số bằng số đã được kiểm lại bằng NumPy hoặc PyTorch; số đo huấn luyện lấy từ lần chạy của `loigiai.ipynb`.

**Kiểm tra nhanh**

1. ‖∂h₁₀₀/∂h₀‖₂ ≤ 0.9¹⁰⁰ = 2.66e-5 (chặn trên, theo tính chất chuẩn của tích).
2. 8 (√64); sau khi chia √64 còn 1.
3. Các hàng đầu ra đổi chỗ y như đầu vào: Attn(PX) = P·Attn(X).
4. Post-LN: x₂ = LN(x + Sub(x)). Pre-LN: x₂ = x + Sub(LN(x)).
5. 2·12·12·64·2 = 36 864 byte (36 KiB): K và V, L = 12, 12 đầu, dₕ = 64, 2 byte (fp16).

**Bài tập giấy bút**

1. Cộng tính: s + h₁ = (1, 1) cho tanh (0.7616, 0.7616), e₁ = 1.5232; s + h₂ = (1, 2) cho tanh (0.7616, 0.9640), e₂ = 1.7256. α = (0.4496, 0.5504), c = 0.4496·(1, 0) + 0.5504·(1, 1) = (1.0000, 0.5504). Dot-product: e = (h₁·s, h₂·s) = (0, 1), α = (0.2689, 0.7311), c = (1.0000, 0.7311).
2. `W_Q`: 512 × 512 (8 phép chiếu 512 × 64 ghép lại); Q: 10 × 512; Q một đầu: 10 × 64; điểm và A một đầu: 10 × 10 (8 × 10 × 10 cho cả 8 đầu); sau gộp đầu: 10 × 512; `W_O`: 512 × 512. Tham số: 4·512² + 4·512 = 1 050 624.
3. Vị trí 5: (sin 5, cos 5, sin 0.05, cos 0.05) = (−0.9589, 0.2837, 0.0500, 0.9988).
4. Attention: 4·64 + 32 = 288. FFN: 8·64 + 5·8 = 552 (d·4d + 4d + 4d·d + d = 256 + 32 + 256 + 8). LN: 4d = 32. Tổng 872, khớp 12d² + 13d = 768 + 104.
5. q·k = ∑ qᵢkᵢ; 𝔼[qᵢkᵢ] = 𝔼[qᵢ]𝔼[kᵢ] = 0 và Var(qᵢkᵢ) = 𝔼[qᵢ²]𝔼[kᵢ²] = 1 (độc lập); các số hạng độc lập nên phương sai cộng thành dₖ. Với σ² và τ²: Var(q·k) = dₖσ²τ². Chia dₖ cho std 1/√dₖ: logit nhỏ, softmax gần đều, mất chọn lọc; chia √dₖ cho đúng std 1.
6. Điểm của đầu vào hoán vị là S′ = PSPᵀ (vì (PXW_Q)(PXW_K)ᵀ = P(XW_Q)(XW_K)ᵀPᵀ). Softmax theo hàng đổi thứ tự cột cùng cách nên A′ = PAPᵀ, và O′ = A′PV = PAPᵀPV = PAV (PᵀP = I). Với mặt nạ nhân quả M cần M′ = PMPᵀ, nhưng mô hình dùng đúng M: PSPᵀ + M ≠ P(S + M)Pᵀ vì M (tam giác) không bất biến với hoán vị. Đầu ra phụ thuộc chỉ số vị trí: đẳng biến bị phá.
7. QKᵀ/√2 = ((1.4142, 0), (0, 1.4142)), exp(1.4142) = 4.113: A = ((0.8044, 0.1956), (0.1956, 0.8044)); V = I nên O = A. Có mặt nạ nhân quả: hàng 1 = (1, 0), hàng 2 = (0.1956, 0.8044) không đổi vì vị trí cuối vốn nhìn được cả hai.
8. (a) 4·768² + 4·768 = 2 362 368. (b) Không đổi khi h = 1. (c) 8·12·1024²·2 byte = 192 MiB. (d) Không miễn phí: tham số và phép chiếu không đổi, nhưng bộ nhớ ma trận điểm tăng tuyến tính theo h, và mỗi đầu hẹp hơn (dₖ = 64 so với 768).
9. R(a)ᵀ = R(−a) và R(−a)R(b) = R(b − a), nên (R(mθ)q)ᵀ(R(nθ)k) = qᵀR(−mθ)R(nθ)k = qᵀR((n − m)θ)k. Số: R(3π/2)q = (0, −1), R(π/2)k = (−1, 1), tích vô hướng −1. Vế phải: n − m = −2, R(−π) = −I, qᵀ(−k) = −1. Khớp.
10. Post-LN: x + Sub(x) = (1, 3) + (2, 6) = (3, 9); trung bình 6, phương sai 9, đầu ra (−1, 1). Pre-LN: LN(x) = (−1, 1), Sub = (−2, 2), x₂ = (1, 3) + (−2, 2) = (−1, 5). Pre-LN giữ x trong dòng residual nên còn thông tin độ lớn: với x = (10, 30) cho x₂ = (8, 32) (LN(x) vẫn (−1, 1), Sub = (−2, 2), cộng với x), còn post-LN vẫn cho (−1, 1) với mọi x cùng hướng, tức LN xóa thông tin độ lớn. Ở d = 2 còn mạnh hơn: LN cho (−1, 1) với **mọi** x có x₁ < x₂ (hai chiều sau khi trừ trung bình và chia độ lệch chuẩn chỉ còn một bit: x₁ < x₂ hay ngược lại).
11. Phương sai sau L khối: 1 + 2Lσ². Một khối: 1.020² = 1.0404 nên σ² = 0.0202. Sau 24 khối (48 nhánh): 1 + 48·0.0202 = 1.970, std 1.403. Đo được 1.555 (phương sai 2.418), lớn hơn dự đoán: các nhánh không độc lập (đọc dòng đã chứa kết quả các nhánh trước, có thể cộng cùng hướng); đây là giải thích hợp lý, chưa tách riêng nguyên nhân. Hệ số 0.02/√(2·`n_layer`): phương sai thêm vào của 2·`n_layer` nhánh tỉ lệ với 2·`n_layer`, nên chia độ lệch chuẩn của mỗi lớp chiếu ra cho √(2·`n_layer`) giữ tổng phương sai thêm vào cỡ hằng số theo độ sâu. Hệ số này nanoGPT lấy từ bài GPT-2 (dòng 142 của `model.py` ghi "per GPT-2 paper"); phần giải thích bằng phương sai là lập luận của sách.
12. Quy nạp: với mọi lớp ℓ, đầu vào của lớp ℓ tại các vị trí ≤ t chỉ phụ thuộc token ≤ t (và chỉ số vị trí của chúng). Lớp 0: đúng. Bước ℓ → ℓ + 1: K, V tại s ≤ t là hàm theo từng vị trí của đầu vào lớp ℓ tại s; đầu ra tại t là hàm của đầu vào tại t (phần dư, LN, FFN theo từng vị trí) và của K, V tại các vị trí ≤ t (mặt nạ nhân quả). Hệ quả: K, V của vị trí t không đổi khi thêm token sau, nên cache cho đúng logit của forward đầy đủ (Lab 4: 6.0e-7). Số lượt xử lý token: n(n + 1)/2 = 500 500 so với 1000; cache GPT-2 small ở n = 1000: 36 864 000 byte ≈ 35.2 MiB. Không đáng giá khi n rất nhỏ, khi bộ nhớ không chứa nổi cache, hoặc khi chỉ chấm điểm một chuỗi đã biết (một forward song song là đủ).
13. (a) 2π·`base^((d−2)/d)` = 54 410 (base 10 000) và 2 559 196 (base 500 000). RoPE không áp lên v vì phép quay chỉ cần có trong tích vô hướng để điểm phụ thuộc n − m; quay v làm đầu ra của vị trí m quay theo m, trộn vị trí tuyệt đối vào nội dung. (b) Mỗi khối giữ m = max s, Z = ∑exp(s − m), o = ∑exp(s − m)v; softmax toàn cục cho o/Z. Gộp: m = max(m₁, m₂), Z = Z₁exp(m₁ − m) + Z₂exp(m₂ − m), o = o₁exp(m₁ − m) + o₂exp(m₂ − m). Chứng minh: o₁exp(m₁ − m) = ∑exp(s − m)v trên khối 1, tương tự khối 2 và Z; nên o, Z sau gộp là ∑exp(s − m)v và ∑exp(s − m) trên mọi phần tử, và o/Z = softmax toàn cục (e⁻ᵐ triệt tiêu). Số: khối 1 (logit 1, 3): m₁ = 3, Z₁ = 1.1353, o₁ = 5.1353; khối 2 (logit 2, 0): m₂ = 2, Z₂ = 1.1353, o₂ = 2.5413. Gộp: m = 3, Z = 1.1353 + 1.1353e⁻¹ = 1.5530, o = 5.1353 + 2.5413e⁻¹ = 6.0702, o/Z = 3.9087, bằng softmax trực tiếp (0.0871, 0.6439, 0.2369, 0.0321)·(1, 5, 2, 4).
14. (a) O = AV nên dV = AᵀdO, dA = dO·Vᵀ. Mỗi hàng p = softmax(s) có Jacobian J = diag(p) − ppᵀ (đối xứng), nên dS = J·dA = A ⊙ (dA − (dA ⊙ A)·1). Vì S = QKᵀ/√dₖ: dQ = dS·K/√dₖ, dK = dSᵀ·Q/√dₖ. (b) p·dA = 0.7311, dS = (0.7311·(1 − 0.7311), 0.2689·(0 − 0.7311)) = (0.1966, −0.1966), bằng cột đầu của J = (p₁(1 − p₁), −p₁p₂). (c) dAᵢⱼ = dOᵢ·Vⱼ (tích vô hướng của hai hàng) nên ∑ⱼ dAᵢⱼAᵢⱼ = ∑ⱼ (dOᵢ·Vⱼ)Aᵢⱼ = dOᵢ·(∑ⱼ AᵢⱼVⱼ) = dOᵢ·Oᵢ: số hạng trừ tính được từ dO và O, không cần dựng lại hàng A (mẹo của FlashAttention backward).

**Bài tính nhanh**

1. e⁰ = 1, exp(ln 3) = 3: (1/4, 3/4).
2. 4 + 6 + 6 + 4 = 20; chia √4 = 2 cho 10.
3. 4.
4. 4·256² + 4·256 = 262 144 + 1024 = 263 168.
5. 1024/16 = 64.
6. 12·100² + 13·100 = 120 000 + 1300 = 121 300.
7. 2·512·2048 = 2 097 152.
8. Trung bình 2, phương sai 1: (−1, 1).
9. RMS = √((9 + 16)/2) = √12.5 = 3.5355: (0.8485, 1.1314).
10. 2·24·16·64·2 = 98 304 byte.
11. 2·128·768·3072 = 603 979 776 ≈ 6.04e8.
12. 6·1.5e9·3e10 = 2.7e20.
13. 5·6/2 = 15.
14. ln 32 000 = 10.37.
15. 16·2048²·2 byte = 134 217 728 = 128 MiB.

**Câu hỏi sau lab**

- Lab 1: mỗi hàng của A là softmax nên không âm, tổng 1; đầu ra là tổ hợp lồi các hàng của V. std(q·k) = √dₖ dựa vào giả thiết độc lập, trung bình 0, phương sai 1; sau huấn luyện q và k thường tương quan, giả thiết không còn đúng.
- Lab 2: các phép chiếu luôn là bốn ma trận d × d vì h·dₖ = d. Pre-LN cộng các nhánh vào dòng không được chuẩn hóa nên chuẩn tích lũy; post-LN có LN sau mỗi phép cộng (nên std = 1.0 là hằng đẳng thức).
- Lab 3: mô hình không causal xử lý mọi vị trí như nhau nên chỉ hoán vị đầu ra. Với causal, vị trí i chỉ thấy i + 1 token đầu, nên đổi thứ tự đổi tập token mà mỗi vị trí thấy.
- Lab 4: cache khớp forward vì K, V không đổi khi thêm token sau (bài 12). Mô hình bé bị overhead Python lấn át (tỉ lệ khoảng 1–2 lần tùy lần chạy). Cache tỉ lệ với số đầu K/V nên GQA 8 đầu nhỏ hơn đúng 32/8 = 4 lần.
- Lab 6: nguyên nhân là đạo hàm tanh: 1 − h² < 1 (trung bình 0.82), nên tích 100 thừa số co như 0.82¹⁰⁰ ≈ 2e-9, chênh khoảng một bậc so với 1.98e-8 (ước lượng thô). Tính không chuẩn tắc của W không làm tích co: ‖W‖₂ = 1.97 > 1, bỏ tanh thì ‖W¹⁰⁰‖₂ = 2.62, và W trực giao (hoàn toàn chuẩn tắc) vẫn cho 1.2e-7.

**Mini-test (thang 30 điểm)**

1. QKᵀ = ((1, 1), (1, 2)), chia √2 cho ((0.7071, 0.7071), (0.7071, 1.4142)); A = ((0.5, 0.5), (0.3302, 0.6698)) (1.5 điểm). O₁ = (1, 2), O₂ = (0.6605, 2.6790) (1.5 điểm). Mặt nạ nhân quả: O₁ = (2, 0), O₂ không đổi vì vị trí cuối vốn nhìn được cả hai (1 điểm).
2. `wte` 5 120 000; `wpe` 131 072; 6 khối × (12·512² + 13·512 = 3 152 384) = 18 914 304; LN cuối 1024. Tổng **24 166 400** (2 điểm; trừ 0.5 điểm nếu cộng thêm `lm_head`). Adam fp32: 4 bản · 4 byte · 24 166 400 = 386 662 400 byte, khoảng 0.36 GiB (1 điểm). KV cache: 2·6·8·64·2 = 12 288 byte (1 điểm; số đầu 8, dₕ = 512/8 = 64).
3. Chứng minh như bài 5 (2 điểm). Logit (8, 0): p = (0.9997, 0.0003), đạo hàm p₁(1 − p₁) = 3.35e-4; sau chia √64 = 8: logit (1, 0), p = (0.7311, 0.2689), đạo hàm 0.1966 (2 điểm). Chia dₖ thì std logit còn 0.125, softmax gần đều (1 điểm).
4. (R(mθ)q)ᵀ(R(nθ)k) = qᵀR(−mθ)R(nθ)k = qᵀR((n − m)θ)k (2 điểm). n − m = −3, góc −3π/4: R(−135°)k = (−0.7071, −0.7071), q·(…) = −1.4142 (2 điểm). Không áp lên v vì quay v đưa vị trí tuyệt đối vào đầu ra; tính chất n − m cần cặp q, k trong tích vô hướng (1 điểm).
5. (a) Mỗi lỗi 1 điểm: (i) `reshape(B, h, T, d // h)` không có `view(...).transpose(1, 2)`, và gộp đầu cũng thiếu transpose; (ii) chia `math.sqrt(d)` thay vì `math.sqrt(d // h)`; (iii) `softmax(dim=-2)` chuẩn hóa theo query thay vì theo key; (iv) nhân mặt nạ 0/1 **sau** softmax (đúng phải điền −∞ trước softmax) (4 điểm). (b) Phép thử rò rỉ: đổi các token ở vị trí ≥ t₀, các logit ở vị trí < t₀ phải không đổi (1 điểm); so với `nn.MultiheadAttention` hoặc `F.scaled_dot_product_attention` bằng cách chép trọng số, hoặc kiểm tổng mỗi hàng của A bằng 1 (1 điểm; phép kiểm tổng này chỉ bắt được lỗi (iii) và (iv), không bắt (i) và (ii), nên khi chấm cần thêm phép so với bản chuẩn hoặc phép thử rò rỉ).
6. Thang chấm: (1.5 điểm) mỗi token phải đọc toàn bộ trọng số (2N byte, fp16) nhưng chỉ làm cỡ 2N FLOPs, tức 1 FLOP/byte, thấp xa điểm gãy của phần cứng: tốc độ đọc bộ nhớ quyết định (prefill chỉ bị chặn bởi FLOPs khi T đủ lớn, vượt điểm gãy đó). (1.5 điểm) Cache Llama-2-7B: 2·32·32·128·2 = 524 288 byte/token, 2 GiB ở T = 4096, batch 1. (1.5 điểm) GQA 8 đầu K/V: 0.5 GiB, nhỏ hơn 4 lần, chạy được batch hoặc ngữ cảnh lớn hơn. (1.5 điểm) Tăng batch chỉ nâng cường độ của các phép nhân với trọng số (một lần đọc trọng số phục vụ cả batch); attention trên KV cache đọc cache riêng của từng chuỗi nên vẫn cỡ 1 FLOP/byte, và cache còn tăng tuyến tính theo batch.

**Săn bug — bốn lỗi**

1. **Tách và gộp đầu sai.** `reshape(B, h, T, dk)` không có `transpose` (và gộp lại bằng `reshape(B, T, d)` thiếu transpose) trộn các vị trí vào các đầu: mỗi "hàng" của một đầu là một mảnh dₖ chiều của một token khác. Lỗi im lặng nhất: nhãn ngẫu nhiên vẫn cho loss 3.918, rò rỉ 0.0000 (hai phép reshape sai nhất quán nên mặt nạ vẫn chặn đúng tương lai); chỉ lộ ở bài `lag1` (0.755 thay vì 0.014) và khi so với bản chuẩn (cùng trọng số, đầu ra chênh 0.894).
2. **Mặt nạ sau softmax.** `A * tril(...)`: hàng không còn tổng 1, mẫu số vẫn chứa ô tương lai nên rò rỉ. Chỉ lỗi này đã đưa loss nhãn ngẫu nhiên xuống 2.972, rò rỉ 6.4471.
3. **Quên mã hóa vị trí.** `self.pe` được tạo nhưng không được cộng vào embedding. Bài `lag1` chỉ đạt 0.676 thay vì 0.014 (mô hình causal nên vẫn học được một phần).
4. **Quên `model.eval()`.** Dropout vẫn bật lúc đo: hai lần gọi cùng input chênh nhau (4.592 và 6.470 ở bản lỗi).

Bảng tách từng lỗi (ô Phân tích từng lỗi của `loigiai.ipynb`; mọi hàng cùng điều kiện: dropout 0, cùng seed, cùng dữ liệu, chỉ khác lỗi được bật):

| Cấu hình | Loss trên batch mới | Rò rỉ |
| --- | --- | --- |
| không lỗi, nhãn ngẫu nhiên | 3.918 | 0.0000 |
| mask sau softmax, nhãn ngẫu nhiên | 2.972 | 6.4471 |
| tách đầu sai, nhãn ngẫu nhiên | 3.918 | 0.0000 |
| không lỗi, bài `lag1` | 0.014 | 0.0000 |
| không PE, bài `lag1` | 0.676 | 0.0000 |
| tách đầu sai, bài `lag1` | 0.755 | 0.0000 |

Cả bốn lỗi cùng lúc cho `lag1` 0.984 (tệ hơn từng lỗi riêng) và nhãn ngẫu nhiên 3.086. Bản sửa:

```python
class FixedAttention(nn.Module):       # __init__ giữ nguyên
    def forward(self, x):
        B, T, d = x.shape
        dk = d // self.h
        q, k, v = [t.view(B, T, self.h, dk).transpose(1, 2) for t in self.qkv(x).split(d, dim=2)]   # sửa lỗi 1
        S = q @ k.transpose(-2, -1) / math.sqrt(dk)
        S = S.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float('-inf'))          # sửa lỗi 2
        return self.o((S.softmax(-1) @ v).transpose(1, 2).reshape(B, T, d))                          # sửa lỗi 1

# trong BuggyLM.forward:  x = self.te(idx) + self.pe(torch.arange(idx.shape[1]))                      # sửa lỗi 3
# trong train_buggy, trước khi đo:  model.eval()                                                      # sửa lỗi 4
```

Đã chạy kiểm: bản sửa cho 3.918 trên nhãn ngẫu nhiên (không dưới ln 50 = 3.912), 0.011 trên `lag1`, rò rỉ đúng 0.0, hai lần gọi giống hệt nhau.

## Tài liệu đào sâu và nguồn

Viết lại bằng lời của người viết. **Đã kiểm ngày 29/10/2026 (riêng dòng 142 "per GPT-2 paper" của nanoGPT kiểm lại 04/11/2026):** mã nguồn nanoGPT, Annotated Transformer, Llama; nội dung Vaswani (Bảng 1, 3), Xiong, Bahdanau, Kazemnejad, Press, Pope, Llama 2 (Bảng 1), Llama 3 (mục 3.2) trên arXiv. **Chưa kiểm nội dung** (chỉ tìm kiếm tiêu đề hoặc nhớ mã arXiv): bài GPT-2 (Radford và cộng sự 2019), chú thích "we suspect" ở mục 3.2.1 của Vaswani, RoFormer, GQA, Luong, Sutskever, Cho, Zhang & Sennrich, Shazeer, FlashAttention, Michel, Voita, Jain & Wallace, Wiegreffe & Pinter, Geva, Elhage, Shaw, T5; mở thử trước khi dựa vào.

| Tài liệu | Dùng để làm gì | Phần nên đọc |
| --- | --- | --- |
| [nanoGPT `model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py) và [README](https://github.com/karpathy/nanoGPT) (Karpathy); [build-nanogpt](https://github.com/karpathy/build-nanogpt) và video "Let's build GPT" (tìm theo tên) | Khối pre-LN, khởi tạo, weight tying, `estimate_mfu`; dựng lại GPT-2 từng commit | Gõ lại `CausalSelfAttention`, `Block`, `GPT` |
| [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) và [mã nguồn](https://github.com/harvardnlp/annotated-transformer); [Meta Llama `model.py`](https://github.com/meta-llama/llama/blob/main/llama/model.py) | Triển khai bám paper (norm đặt trước, nhân embedding với √d); RMSNorm, SwiGLU, RoPE, GQA | `SublayerConnection`, `PositionalEncoding`; `Attention`, `FeedForward` |
| [Vaswani và cộng sự (2017)](https://arxiv.org/abs/1706.03762) | Attention, sinusoidal PE, post-LN, Bảng 1 (độ phức tạp), Bảng 3 (65M, 213M); chú thích về √dₖ ở mục 3.2.1 | Mục mô hình và mục vì sao self-attention; tự đếm bằng Lab 5 |
| Radford và cộng sự (2019), "Language Models are Unsupervised Multitask Learners" (OpenAI; tìm theo tên, link chưa xác minh [cần kiểm chứng]) | Nguồn của khối pre-LN kiểu GPT-2 và của hệ số khởi tạo 1/√N cho lớp residual mà nanoGPT dẫn ("per GPT-2 paper") | Phần mô tả mô hình và khởi tạo [số mục cần kiểm chứng] |
| [Bahdanau và cộng sự (2014)](https://arxiv.org/abs/1409.0473); Luong (2015, 1508.04025); Sutskever (2014, 1409.3215); Cho (2014, 1406.1078) | Attention cộng tính, hàm điểm dot, seq2seq, GRU | Phần mô hình |
| [RoFormer](https://arxiv.org/abs/2104.09864); [ALiBi](https://arxiv.org/abs/2108.12409); [Kazemnejad và cộng sự (2023)](https://arxiv.org/abs/2305.19466) | RoPE; ngoại suy độ dài; NoPE | Dẫn xuất tính chất tương đối; thí nghiệm ngoại suy |
| [Xiong và cộng sự (2020)](https://arxiv.org/abs/2002.04745); Zhang & Sennrich (2019, 1910.07467); Shazeer (2020, 2002.05202) | Pre-LN và post-LN, warmup; RMSNorm; SwiGLU | Phân tích gradient lúc khởi tạo |
| [GQA](https://arxiv.org/abs/2305.13245); [Pope và cộng sự (2022)](https://arxiv.org/abs/2211.05102); Shazeer (2019, 1911.02150); FlashAttention (2022, 2205.14135) | MQA, GQA, chi phí suy luận, không dựng ma trận n × n | Phần động cơ và phân tích bộ nhớ |
| [Llama 3](https://arxiv.org/abs/2407.21783); [Llama 2](https://arxiv.org/abs/2307.09288); [Elhage và cộng sự (2021)](https://transformer-circuits.pub/2021/framework/index.html); [CS224n](https://web.stanford.edu/class/cs224n/); Jay Alammar, "The Illustrated Transformer" (tìm theo tên) | Base RoPE 500 000, cấu hình GQA; residual stream; bài giảng; hình minh họa | Mục 3.2 và Bảng 3 (Llama 3), Bảng 1 (Llama 2) |

**Từ khóa để tự tìm thêm:** additive attention, alignment matrix, exposure bias, teacher forcing, QK-norm, attention entropy collapse, multi-query attention, rotary embedding base frequency, position interpolation, YaRN, NTK-aware scaling, NoPE, pre-LN vs post-LN, DeepNorm, RMSNorm, SwiGLU, weight tying, FlashAttention, PagedAttention, prefill vs decode, arithmetic intensity, roofline, online softmax; các bài về cắt tỉa đầu attention (Michel 2019, Voita 2019) và về việc trọng số attention có phải lời giải thích không (Jain & Wallace 2019, Wiegreffe & Pinter 2019; mã arXiv chưa kiểm).

**Trước khi sang chương 9:** đạt từ 24/30 điểm mini-test, làm lại các bài ★★★ sai, tự viết được Lab 1 (attention có mặt nạ) và `MHA` của Lab 2 trong 20 phút không nhìn code mẫu, và tìm được cả bốn lỗi của bài săn bug.

Nguồn tra cứu ngày 29/10/2026; bổ sung và sửa ngày 04/11/2026 (vòng 3).
