# Chương 7 — CNN và thị giác máy tính

## Mục tiêu và kiểm tra nhanh

Chương này dạy kiến trúc có cấu trúc đầu tiên sau MLP: mạng tích chập (CNN). Bạn sẽ tự viết phép tích chập và đạo hàm của nó, đếm đúng kích thước, tham số và số phép tính của một lớp conv, hiểu vì sao cục bộ và chia sẻ trọng số cho ra thiên kiến cấu trúc (inductive bias) hợp với ảnh, và đi qua các kiến trúc kinh điển như một chuỗi ý tưởng. Phần cuối nối sang NLP: conv 1D, vì sao Transformer thay phần lớn conv, và vì sao residual cộng chuẩn hóa là nền của Transformer. Chương 6 đã dạy BatchNorm, LayerNorm, khởi tạo và debug; chương này dùng lại. Thời lượng: khoảng 2 tuần, 8 giờ mỗi tuần.

**Sau chương này, bạn phải làm được:**

- ☐ Tính kích thước đầu ra, số tham số và số phép nhân-cộng (MAC) của một lớp conv (kể cả stride, padding, dilation, 1×1, depthwise); nói được `nn.Conv2d` tính cross-correlation; viết conv bằng vòng lặp và bằng im2col.
- ☐ Tính receptive field của một chồng lớp, giải thích vì sao receptive field hiệu dụng nhỏ hơn con số lý thuyết, và phân biệt equivariance với invariance (conv, pooling, stride, Flatten, global pooling).
- ☐ Giải thích từng kiến trúc kinh điển bằng ý tưởng của nó (3×3 xếp chồng, 1×1 nén kênh, residual, bottleneck, conv tách theo chiều sâu) và đếm tham số, MAC của chúng.
- ☐ Nêu BatchNorm2d lấy thống kê trên trục nào, gộp BN vào conv ở chế độ suy luận, chọn augmentation bảo toàn nhãn, và fine-tune backbone pretrained mà không dính bẫy BN.
- ☐ Giải thích conv 1D (TextCNN, causal, dilated), patch embedding của ViT, chi phí conv so với attention, và vì sao khối Transformer là phiên bản "residual + chuẩn hóa" của ResNet.

**Kiểm tra nhanh (5 phút, không dùng máy tính).** Câu i kiểm tra mục 7.i: đúng thì đọc lướt mục đó, sai thì đọc kỹ. Đúng cả 5 câu thì làm thẳng bài ★★★ và mini-test.

1. `nn.Conv2d(3, 16, 3)` có bias nhận ảnh 3×32×32 với padding 1, stride 1. Đầu ra có kích thước gì, lớp có bao nhiêu tham số?
2. (a) Ba lớp conv 3×3 stride 1 xếp liên tiếp: một điểm đầu ra nhìn thấy vùng ảnh cỡ nào? (b) CNN có bất biến với phép dịch chuyển ảnh không?
3. Mạng 56 lớp có train error cao hơn mạng 20 lớp cùng loại. Đó có phải overfitting không? Với khối residual y = x + F(x), đạo hàm ∂y/∂x bằng gì?
4. `nn.BatchNorm2d(64)` lấy thống kê trên những trục nào của tensor N × C × H × W, và có bao nhiêu tham số học được?
5. TextCNN với embedding 300 chiều và một nhóm 100 bộ lọc cỡ 3 (có bias) có bao nhiêu tham số? Max-over-time pooling để làm gì?

Đáp án ở mục Lời giải cuối chương.

## 7.1 Phép tích chập: cục bộ, chia sẻ trọng số và cách đếm cho đúng

Tích chập là một lớp tuyến tính như `nn.Linear` của chương 6, nhưng bị ép hai ràng buộc: mỗi đầu ra chỉ nhìn một vùng nhỏ của đầu vào (**cục bộ**), và mọi vị trí dùng chung một bộ trọng số (**chia sẻ trọng số**). Hai ràng buộc này làm số tham số giảm hàng nghìn lần so với lớp kết nối đầy đủ (ví dụ ngay dưới: khoảng 4 300 lần) và cho mạng tính chất equivariance với phép dịch.

**Vì sao không dùng lớp kết nối đầy đủ cho ảnh.** Ảnh 224×224×3 có 150 528 giá trị. Chỉ một lớp Linear từ ảnh sang 1 000 neuron ẩn đã có 150 528 000 trọng số, và nó học riêng từng vị trí: mẫu "cạnh dọc" ở góc trái và ở góc phải phải học lại từ đầu. Ảnh tự nhiên có cấu trúc: điểm gần nhau liên quan nhau hơn, và cùng một mẫu cục bộ xuất hiện ở nhiều chỗ; conv mã hóa đúng hai quan sát đó. Số đếm được: conv1 của AlexNet (3 → 96 kênh, kernel 11×11) có 34 944 tham số, khoảng 1/4 300 lớp Linear trên; nếu mỗi vị trí đầu ra 55×55 có bộ lọc riêng (cục bộ nhưng không chia sẻ) thì cần 105 705 600, gấp 3 025 (= 55²) lần.

**Cross-correlation, không phải convolution.** Với một kênh vào, một kênh ra, kernel k×k, stride s, thứ `nn.Conv2d` tính là:

$$
y[i, j] = b + \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} w[u, v]\; x[\,i s + u,\; j s + v\,]
$$

Kernel **không bị lật**. Convolution toán học (giáo trình tín hiệu) dùng x[i·s − u, j·s − v], tức lật kernel theo cả hai chiều. Với CNN không có khác biệt về năng lực, vì kernel được học nên tự thích nghi với cách đánh chỉ số. Khác biệt chỉ lộ ra khi so với `scipy.signal.convolve2d` hoặc lý thuyết tín hiệu: với kernel Sobel phản đối xứng, hai kết quả khác nhau đúng một dấu âm (ví dụ 7.1).

Với nhiều kênh, mỗi kênh ra là tổng trên **mọi** kênh vào, và mỗi kênh ra có kernel riêng:

$$
\text{tensor vào } x \in \mathbb{R}^{N \times C_{in} \times H \times W}, \qquad \text{kernel } w \in \mathbb{R}^{C_{out} \times C_{in} \times k \times k}, \qquad \text{tham số} = C_{out} C_{in} k^{2} + C_{out}
$$

PyTorch dùng thứ tự **NCHW** (batch, kênh, cao, rộng). Kích thước đầu ra theo từng chiều (n = kích thước vào, p = padding mỗi phía, d = dilation, s = stride) và số phép nhân-cộng (MAC, multiply-accumulate; 1 MAC ≈ 2 FLOP) của cả lớp:

$$
n_{out} = \left\lfloor \frac{n + 2p - d(k-1) - 1}{s} \right\rfloor + 1, \qquad \text{MAC} = H_{out} W_{out} \cdot C_{out} C_{in} k^{2}
$$

Phép chia lấy sàn nên hai kích thước vào khác nhau có thể cho cùng kích thước ra (hàng, cột cuối bị bỏ qua). Padding "same" (s = 1, k lẻ) giữ nguyên kích thước khi p = d(k − 1)/2, tức (k − 1)/2 nếu không giãn. Stride lớn hơn 1 hạ độ phân giải. Dilation d lớn hơn 1 cách đều các điểm của kernel d bước: vùng nhìn mở rộng mà không thêm tham số.

**Ví dụ giải mẫu 7.1 — một lớp cross-correlation 3×3, rồi so với convolution thật.** Ảnh X (5×5) có các hàng (1, 2, 0, 1, 3), (0, 1, 2, 3, 1), (2, 1, 0, 1, 0), (1, 0, 1, 2, 2), (3, 1, 0, 0, 1). Kernel Sobel dọc K có các hàng (1, 0, −1), (2, 0, −2), (1, 0, −1). Stride 1, không padding, nên đầu ra 3×3 (5 − 3 + 1 = 3).

- Ô (0, 0): cửa sổ có các hàng (1, 2, 0), (0, 1, 2), (2, 1, 0). Nhân từng phần tử với K rồi cộng: 1·1 + 2·0 + 0·(−1) + 0·2 + 1·0 + 2·(−2) + 2·1 + 1·0 + 0·(−1) = 1 − 4 + 2 = −1.
- Làm tương tự cho 9 vị trí: đầu ra có các hàng (−1, −3, −1), (2, −4, 0), (5, −3, −3). `F.conv2d` và `scipy.signal.correlate2d` cho đúng ma trận này.
- `scipy.signal.convolve2d` (lật kernel) cho các hàng (1, 3, 1), (−2, 4, 0), (−5, 3, 3), đúng bằng **âm** ma trận trên, vì Sobel phản đối xứng qua tâm. Lab 1 kiểm điều này bằng assert.

% manim: TruotKernel

**Ví dụ giải mẫu 7.2 — đếm kích thước, tham số, MAC.**

- Kích thước: 224 vào, kernel 7, p = 3, s = 2 cho ⌊(224 + 6 − 6 − 1)/2⌋ + 1 = 112; rồi 112 vào, kernel 3, p = 1, s = 2 cho 56 (đúng phần đầu của ResNet). Kernel 3, d = 2, vào 32, p = 0 cho 32 − 4 = 28.
- Làm tròn xuống: AlexNet dùng ảnh vào 227 (kernel 11, stride 4, p = 0) ra 55; với 224 và p = 0 ra 54: paper ghi "224×224" nhưng số đó không khớp phép tính, còn `torchvision` dùng p = 2.
- Tham số: conv 3 → 64, kernel 3×3, có bias: 3·3·3·64 + 64 = 1 792 (không bias: 1 728). Conv 64 → 128, 3×3: 64·128·9 + 128 = 73 856.
- MAC của lớp 64 → 128, 3×3, đầu ra 56×56: 56·56·128·64·9 = 231 211 008, tức khoảng 0.23 GMAC cho một lớp.
- Chia sẻ trọng số: cùng một kernel cho 56·56 = 3 136 vị trí nên MAC lớn hơn số tham số hàng nghìn lần (231 211 008 so với 73 856); lớp Linear thì mỗi tham số chỉ dùng một lần cho mỗi mẫu.

**Conv 1×1** là một lớp Linear áp dụng độc lập tại từng vị trí không gian: trộn kênh mà không nhìn hàng xóm (đã kiểm ở ô "Kiểm nhanh" của lab: `nn.Conv2d(c, c2, 1)` khớp `nn.Linear(c, c2)` khi hoán trục kênh về cuối, sai khác dưới 10⁻⁵). Dùng để nén hay mở rộng số kênh (mục 7.3).

**Conv là phép nhân ma trận có cấu trúc.** Trong 1D, conv là nhân với một ma trận Toeplitz, mỗi hàng là kernel dịch đi một bước. Trong 2D, thư viện dựng ma trận cột (`im2col`, trong PyTorch là `F.unfold`): mỗi cột là một vùng nhìn duỗi thẳng, rồi nhân một lần với ma trận kernel Cₒᵤₜ × (Cᵢₙk²). Đây là một trong những cách tính thật sự (cùng FFT và Winograd); đổi lại bộ nhớ tăng khoảng k² lần.

**Ví dụ giải mẫu 7.3 — conv là nhân ma trận.**

- 1D: x = (1, 2, 3, 4, 5), w = (1, 0, −1). Ma trận T cỡ 3×5 có các hàng (1, 0, −1, 0, 0), (0, 1, 0, −1, 0), (0, 0, 1, 0, −1). Tx = (1 − 3, 2 − 4, 3 − 5) = (−2, −2, −2), khớp `np.correlate(x, w, 'valid')`. Số hàng của T là số vị trí ra, số cột là số điểm vào.
- 2D: ảnh 3×3 các hàng (1, 2, 0), (0, 1, 3), (2, 0, 1), kernel 2×2 các hàng (1, 2), (3, 4): 4 vị trí ra nên ma trận M cỡ 4×9, mỗi hàng đặt 4 giá trị kernel vào 4 điểm ảnh của một cửa sổ, còn lại bằng 0. Mx = (9, 17, 8, 11); vị trí đầu: 1·1 + 2·2 + 0·3 + 1·4 = 9, khớp `correlate2d`.

% manim: ConvLaMaTran

**Backward của conv cũng là conv.** Giả sử y[i] = ∑ᵤ w[u]·x[i + u] (1D) và biết g = ∂L/∂y từ lớp phía trên. Mỗi trọng số được dùng ở mọi vị trí, nên gradient của nó là **tổng** đóng góp từ mọi vị trí. Mỗi điểm vào x[m] chỉ ảnh hưởng những y[i] có i + u = m:

$$
\frac{\partial L}{\partial w[u]} = \sum_{i} g[i]\, x[i + u], \qquad\qquad \frac{\partial L}{\partial x[m]} = \sum_{i} g[i]\, w[m - i]
$$

Công thức đầu là cross-correlation của x với g. Công thức sau là convolution "đầy đủ" của g với w, tức cross-correlation của g (đệm k − 1 số 0 mỗi phía) với kernel **đã lật**. Chiều 2D giống hệt, lật theo cả hai chiều; với nhiều kênh, dx của kênh vào c là tổng trên các kênh ra o của (gradient kênh o) ⋆ (kernel lật của cặp (o, c)).

**Ví dụ giải mẫu 7.4 — backward của conv 1D.** x = (1, 2, 3, 4, 5), w = (1, 0, −1), y = (−2, −2, −2), gradient từ phía trên g = (1, 2, 3).

- dw[0] = 1·1 + 2·2 + 3·3 = 14, dw[1] = 1·2 + 2·3 + 3·4 = 20, dw[2] = 1·3 + 2·4 + 3·5 = 26.
- dx[m] = ∑ᵢ g[i]·w[m − i] (chỉ lấy chỉ số hợp lệ): dx[0] = g₀w₀ = 1; dx[1] = g₁w₀ + g₀w₁ = 2; dx[2] = g₂w₀ + g₁w₁ + g₀w₂ = 3 + 0 − 1 = 2; dx[3] = g₂w₁ + g₁w₂ = −2; dx[4] = g₂w₂ = −3. Vậy dx = (1, 2, 2, −2, −3).

**Vì sao ML cần cái này.**

- **Inductive bias.** Cục bộ và chia sẻ là giả định về dữ liệu: quan hệ quan trọng nằm gần nhau, và cùng một mẫu có thể xuất hiện ở bất kỳ đâu. Khi đúng (ảnh, âm thanh, chuỗi), mạng học được từ ít dữ liệu hơn MLP nhiều; khi sai (dữ liệu bảng, thứ tự cột tùy ý), conv không có lý do để hợp.
- **Đếm đúng là kỹ năng phỏng vấn:** câu hỏi tính kích thước, tham số, MAC, receptive field xuất hiện trong hầu hết đề về CNN; nhầm MAC với FLOP hoặc làm tròn là mất điểm.

**Thực tế tính toán.** Hãy tự đếm tham số bằng code: `sum(p.numel() for p in net.parameters())`. Công thức MAC **không** tính BatchNorm, ReLU, pooling, bias: các phép đó rẻ về số phép tính nhưng đọc ghi bộ nhớ nhiều (xem "Thực tế khác lý thuyết").

## 7.2 Pooling, receptive field và chuyện "bất biến" dịch chuyển

Pooling hạ độ phân giải và mở rộng vùng nhìn; conv cho equivariance với phép dịch, nhưng phân loại cần invariance, và khoảng cách giữa hai khái niệm này là chỗ bị hỏi nhiều nhất. Mục này gồm receptive field (lý thuyết và hiệu dụng), aliasing do stride, và vì sao `Flatten` + `Linear` phá cả equivariance.

**Pooling.** Max pooling lấy giá trị lớn nhất trong cửa sổ (thường 2×2, stride 2), average pooling lấy trung bình; cả hai không có tham số học được. Max giữ phản ứng mạnh nhất, hợp với câu hỏi "đặc trưng này có xuất hiện không"; average giữ thống kê cả vùng. Hai tác dụng chung: giảm kích thước (giảm MAC các lớp sau, tăng receptive field) và thêm chút bất biến cục bộ. **Global pooling** (trung bình hoặc max trên cả H × W) cho mỗi kênh một số, thay `Flatten` + `Linear` ở đầu ra: bỏ ràng buộc kích thước ảnh và, như ví dụ 7.8, giữ được tính dịch chuyển tới cuối mạng.

**Receptive field (RF)** của một điểm đầu ra là tập các điểm đầu vào ảnh hưởng nó. Theo dõi hai số khi đi từ đầu vào ra đầu ra: r (RF hiện tại) và j (jump: khoảng cách trên ảnh gốc giữa hai điểm liền nhau của bản đồ hiện tại). Khởi đầu r = j = 1. Qua một lớp có kernel k, stride s, dilation d:

$$
r \leftarrow r + d\,(k-1)\, j, \qquad\qquad j \leftarrow j \cdot s
$$

Thứ tự quan trọng: cập nhật r bằng j **cũ**, rồi mới nhân j với s. Pooling cũng là một lớp (k, s). Lab 3 viết hàm này.

**Ví dụ giải mẫu 7.5 — receptive field và vì sao xếp chồng 3×3.**

- Ba lớp 3×3 stride 1: r = 3, 5, 7; hai lớp: RF 5. Tham số với C kênh vào và ra (bỏ bias): hai lớp 3×3 là 18C², một lớp 5×5 là 25C² (tỉ lệ 0.72); ba lớp 3×3 là 27C², một lớp 7×7 là 49C² (tỉ lệ 0.551). Với C = 64: 73 728 so với 102 400, và 110 592 so với 200 704. Tổng quát L lớp 3×3 có RF 2L + 1 với 9L·C² tham số, so với (2L + 1)²·C² của kernel phẳng, lại thêm phi tuyến ở giữa.
- Chồng kiểu VGG (conv 3, conv 3, pool 2, conv 3, conv 3, pool 2, conv 3): sau hai conv đầu r = 5, j = 1; pool: r = 6, j = 2; hai conv: r = 10, rồi 14; pool: r = 16, j = 4; conv cuối: r = 16 + 2·4 = 24, tức (RF, jump) = (24, 4). Phần đầu của ResNet (conv 7 s2, pool 3 s2): r = 7, j = 2, rồi r = 7 + 2·2 = 11, j = 4: (11, 4).

% manim: ChongBaLop

**Receptive field hiệu dụng (ERF).** RF lý thuyết nói điểm nào *có thể* ảnh hưởng. Luo, Li, Urtasun và Zemel (2016) đo điểm nào *thực sự* ảnh hưởng, và thấy phân bố ảnh hưởng gần Gauss, chỉ chiếm một phần nhỏ của RF lý thuyết. Lý do có thể tính tay ở khởi tạo: nếu kernel không âm và chuẩn hóa tổng bằng 1, ta đọc nó như phân phối xác suất của một độ dịch; gradient lan ngược từ một điểm đầu ra qua L lớp là phép chập lặp L lần, tức phân phối của tổng L độ dịch độc lập, nên phương sai cộng lại, và theo định lý giới hạn trung tâm phân phối tiến về Gauss (bỏ qua ReLU và hiệu ứng biên).

**Ví dụ giải mẫu 7.6 — độ rộng ERF khi mọi kernel đều 1/9.** Kernel 3×3 đều có ảnh hưởng một chiều (1/3, 1/3, 1/3) trên các độ lệch (−1, 0, 1), phương sai (1/3)·1 + 0 + (1/3)·1 = 2/3. Cộng L lớp độc lập: phương sai 2L/3, độ lệch chuẩn √(2L/3) = 0.8165·√L điểm ảnh. Đo bằng autograd (Lab 3): với L = 1, 4, 16, 64 (RF lý thuyết 3, 9, 33, 129) độ lệch chuẩn ERF là 0.816, 1.633, 3.266, 6.532, luôn bằng 0.8165·√L.

RF lý thuyết tăng tuyến tính theo L, ERF chỉ tăng như √L: ở L = 64, vùng "±2 độ lệch chuẩn" rộng khoảng 26 điểm trong khi RF lý thuyết là 129. Đây là kết quả ở khởi tạo với kernel đều; mạng đã huấn luyện có phân bố khác (Luo và cộng sự khảo sát thêm ReLU, dropout, subsampling, kết nối tắt) nhưng kết luận định tính giữ nguyên: mạng sâu "nhìn" ít hơn con số RF, một lý do cần pooling, dilation, hoặc cơ chế nhìn toàn cục (attention, chương 8).

**Equivariance và invariance.** Gọi Tₐ là phép dịch ảnh a điểm. Hàm f là *equivariant* nếu f(Tₐx) = Tₐf(x): dịch đầu vào thì đầu ra dịch theo. Hàm f là *invariant* nếu f(Tₐx) = f(x). Một lớp conv stride 1 là equivariant (với biên phù hợp, ví dụ circular padding, hoặc bỏ qua rìa ảnh), vì cùng một kernel áp ở mọi vị trí. Phân loại cần invariant (con mèo ở đâu cũng là con mèo): conv (equivariant) rồi pooling toàn cục tạo ra nó.

Hai thứ phá điều đó. Thứ nhất là **stride**: lấy mẫu con (subsampling) chỉ giữ một vị trí trong s vị trí, nên dịch một điểm thay đổi những vị trí được giữ lại. Đây là aliasing trong lý thuyết lấy mẫu (Zhang, 2019; Azulay và Weiss, 2019). Thứ hai là `Flatten` + `Linear`: mỗi vị trí đặc trưng có trọng số riêng, nên mẫu hình ở vị trí chưa thấy bao giờ gặp trọng số chưa được học.

**Ví dụ giải mẫu 7.7 — aliasing khi stride 2 (1D).** x = (1, 0, 1, 0, 1, 0, 1, 0). Lấy mẫu với stride 2 (giữ vị trí 0, 2, 4, 6) cho x[::2] = (1, 1, 1, 1). Dịch x sang phải một bước thành (0, 1, 0, 1, 0, 1, 0, 1), cùng lấy mẫu cho (0, 0, 0, 0). Một bước dịch đảo toàn bộ đầu ra.

Max-pool 2 (stride 2) trên xung đơn độ dài 8: xung ở vị trí 1 cho (1, 0, 0, 0); ở vị trí 2 và 3 đều cho (0, 1, 0, 0), nên dịch xung một bước đôi khi làm đầu ra dịch một ô, đôi khi không đổi: chỉ dịch theo bội của stride mới cho đầu ra dịch đều. Đo trên chồng conv 1D + ReLU + max-pool 2 với circular padding (ô "Kiểm nhanh" của lab, một seed): dịch đầu vào 2 điểm thì đầu ra dịch đúng 1 ô; dịch 1 điểm thì không khớp phép dịch nào.

% manim: AliasingStride

**Ví dụ giải mẫu 7.8 — `Flatten` phá equivariance (lab 4, số đo thật).** Bài toán đồ chơi: phân loại 6 mẫu hình 5×5 (thanh ngang, thanh dọc, dấu cộng, khung vuông, đường chéo, phản chéo) trong ảnh 12×12 có nhiễu Gauss. Huấn luyện chỉ với mẫu hình đặt ở vị trí góc trên-trái (hàng, cột) ∈ {2, 3, 4}², kiểm tra ở vị trí chưa từng thấy {0, 1, 6, 7}². Đoán ngẫu nhiên đạt 1/6 = 0.167. Kết quả (Adam 12 epoch, 2 000 mẫu train, trung bình 2 seed):

| Mô hình | Tham số | Acc trong vùng đã thấy | Acc ở vị trí lạ |
| --- | --- | --- | --- |
| MLP (Flatten, Linear 144 → 128, ReLU, Linear 128 → 6) | 19 334 | 0.858 | 0.183 |
| CNN, hai lần (conv, ReLU, maxpool 2), rồi Flatten + Linear | 2 118 | 0.952 | 0.242 |
| CNN, ba conv, rồi global max pooling + Linear | 3 670 | 0.996 | 0.996 |

- MLP ngoài vùng đã thấy gần như đoán ngẫu nhiên (trọng số nối với điểm ảnh ở rìa chưa từng nhận tín hiệu có ích). CNN có Flatten tốt hơn MLP trong vùng đã thấy (ít tham số hơn mà đúng hơn) nhưng ngoài vùng vẫn gần ngẫu nhiên: conv equivariant, nhưng `Flatten` + `Linear` gắn từng vị trí của bản đồ đặc trưng 3×3 với một hàng trọng số riêng.
- CNN với global max pooling đạt 0.996 ở cả hai vùng: max trên mọi vị trí làm đầu ra không còn phụ thuộc vị trí. Số tham số khác nhau (19 334, 2 118, 3 670) nhưng lời giải thích là về cấu trúc. Cảnh báo: bài toán tổng hợp rất nhỏ, 2 seed, chỉ minh họa cơ chế; trên ảnh thật, augmentation bằng dịch ngẫu nhiên và ảnh lớn phần nào bù lại.

**Vì sao ML cần cái này.**

- **Đừng nói "CNN bất biến dịch chuyển".** Conv equivariant; invariant xấp xỉ có nhờ pooling và vẫn bị stride và aliasing làm hỏng (Azulay và Weiss, 2019: dự đoán đổi theo phép dịch rất nhỏ); BlurPool (lọc thông thấp trước khi lấy mẫu con, Zhang 2019) cải thiện một phần.
- **Chọn lớp cuối.** Muốn mạng không gắn với tọa độ thì dùng global pooling; muốn biết "cái gì ở đâu" (phát hiện vật thể) thì giữ bản đồ không gian. Vì vậy mạng phân loại hiện đại không còn các lớp FC khổng lồ (mục 7.3).

## 7.3 Kiến trúc kinh điển: vì sao mỗi bước ra đời

Mỗi kiến trúc dạy một ý, và ý đó được chứng minh bằng con số đếm được. Bạn không cần thuộc từng mạng; cần hiểu ba ý: 3×3 xếp chồng, 1×1 nén kênh, và residual.

**LeNet-5** (LeCun, Bottou, Bengio, Haffner, 1998): conv 5×5, pooling, conv 5×5, pooling, rồi các lớp kết nối đầy đủ, đọc chữ số viết tay; ý chính là cục bộ, chia sẻ trọng số, giảm dần độ phân giải. Bản "hiện đại hóa" trong giáo trình (ảnh 28×28 có padding 2, Sigmoid, average pooling, nối đủ) có 61 706 tham số (156 + 2 416 + 48 120 + 10 164 + 850) và 416 520 MAC, do chúng tôi tự đếm. Bản gốc nối thưa ở lớp C3 và có khoảng 60 000 tham số theo paper; LeNet-5 không có trên arXiv, chúng tôi chưa mở được PDF nên số của bản gốc chưa kiểm toàn văn.

**AlexNet** (Krizhevsky, Sutskever, Hinton, 2012): ReLU thay sigmoid/tanh, dropout, tăng cường dữ liệu, huấn luyện trên GPU, thắng ImageNet 2012 cách biệt lớn; abstract ghi khoảng 60 triệu tham số. Bản `torchvision` (một nhánh, khác bản hai GPU trong paper): 61 100 840 tham số, 0.714 GMAC, trong đó 58 631 144 tham số (khoảng 96%) nằm trong ba lớp FC, chỉ 2 469 696 trong các lớp conv.

**VGG** (Simonyan và Zisserman, 2014): chỉ conv 3×3 và pool 2×2, xếp sâu 16 hoặc 19 lớp; ý chính là thiết kế đồng nhất và lập luận xếp chồng 3×3 ở ví dụ 7.5. VGG-16 có 138 357 544 tham số và 15.47 GMAC. Lớp FC đầu (25 088 → 4 096) một mình có 102 764 544 tham số (74.3%), các lớp FC chiếm 123 642 856 (89.4%), trong khi các lớp conv chiếm 99.2% số phép tính: tham số nằm ở FC, tính toán nằm ở conv.

**ResNet** (He, Zhang, Ren, Sun, 2015). Vấn đề là *degradation* (suy giảm): mạng sâu hơn có **train error cao hơn** mạng nông hơn (Hình 4 của paper: đường mảnh là train error). Đó không phải overfitting, vì overfitting làm train error thấp. Bảng 2 của paper (lỗi top-1, %, 10-crop, tập validation ImageNet) cho thấy hệ quả: plain-18 27.94, ResNet-18 27.88; plain-34 28.54, ResNet-34 25.03. Plain sâu hơn thì kém hơn, residual sâu hơn thì tốt hơn. Ý tưởng: thay vì ép các lớp học thẳng ánh xạ H(x), cho chúng học phần dư F(x) = H(x) − x và cộng đường tắt:

$$
y = x + F(x), \qquad\qquad \frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}
$$

Nếu hàm đồng nhất là tối ưu thì chỉ cần đẩy F về 0, dễ hơn ép một chồng lớp phi tuyến xấp xỉ đồng nhất; và đạo hàm luôn có số hạng I, một đường gradient đi thẳng qua khối. Khối cơ bản (basic block) gồm hai conv 3×3; khối **bottleneck** dùng 1×1 (nén kênh), 3×3, 1×1 (mở kênh). Phép cộng cần x và F(x) cùng kích thước; khi đổi số kênh hoặc stride, đường tắt dùng một conv 1×1 để khớp.

**Conv tách theo chiều sâu** (depthwise separable; MobileNet V1, Howard và cộng sự, 2017): depthwise 3×3 (mỗi kênh một kernel, `groups = Cᵢₙ`, chỉ lọc không gian) rồi pointwise 1×1 (chỉ trộn kênh). **MobileNetV2** (Sandler và cộng sự, 2018) là paper khác: giữ conv tách nhưng thêm khối inverted residual và linear bottleneck. Số liệu của chương là của V2 (`torchvision`): 3 504 872 tham số, 0.301 GMAC, so với ResNet-50: 25 557 032 tham số, 4.089 GMAC.

**Ví dụ giải mẫu 7.9 — đọc bảng đếm các mạng** (`torchvision`, `weights=None`, ảnh 224×224; MAC chỉ tính conv và Linear, bỏ BN, bias, pooling; tự đếm bằng code, không lấy từ paper). Tham số và GMAC: LeNet-5 (bản nối đủ, ảnh 28×28) 61 706 và 0.0004; AlexNet 61 100 840 và 0.714; VGG-16 138 357 544 và 15.470; ResNet-18 11 689 512 và 1.814; ResNet-50 25 557 032 và 4.089; MobileNetV2 3 504 872 và 0.301.

- VGG-16 tốn 15.47 GMAC, trong khi ResNet-50 sâu hơn nhiều chỉ cần 4.09 GMAC: nhờ stride lớn ngay từ đầu (ví dụ 7.5), bottleneck, và bỏ các lớp FC lớn bằng global average pooling (lớp FC cuối của ResNet-50 chỉ có 2 049 000 tham số). Bảng để tra cứu: cần nhớ thứ tự độ lớn và nguyên nhân, không cần thuộc.

**Ví dụ giải mẫu 7.10 — bottleneck và conv tách theo chiều sâu.**

- Bottleneck 256 → 64 → 64 → 256 (1×1, 3×3, 1×1), bỏ bias: 256·64 + 9·64·64 + 64·256 = 16 384 + 36 864 + 16 384 = 69 632 tham số. Hai conv 3×3 ở 256 kênh: 2·9·256·256 = 1 179 648, gấp 16.9 lần. Ở 56×56: 218 365 952 MAC so với 3 699 376 128. Tiết kiệm có được vì conv 3×3 đắt chạy ở số kênh nhỏ (64), còn conv 1×1 rẻ lo việc đổi số kênh.
- Conv tách 3×3, 128 → 128: conv thường 9·128·128 = 147 456 tham số; depthwise 9·128 = 1 152 cộng pointwise 128·128 = 16 384 là 17 536 tham số (giảm 8.41 lần). Công thức tỉ lệ: (1/Cₒᵤₜ + 1/k²) = 1/128 + 1/9 = 0.119, nghịch đảo 8.41. Với ảnh 56×56: MAC từ 462 422 016 xuống 54 992 896, cùng tỉ lệ.

% manim: KhoiResidual

**Ví dụ giải mẫu 7.11 — residual giúp gì, và không giúp gì (mô hình đồ chơi và số đo).**

- Đồ chơi vô hướng: mỗi lớp nhân gradient với a; không đường tắt được aᴸ, có đường tắt được (1 + a)ᴸ. Với a = 0.9: L = 10 cho 0.349 và 613.1; L = 50 cho 5.15 × 10⁻³ và 8.66 × 10¹³. Đủ cho trực giác "luôn có đường gradient", nhưng sai ở chỗ không có chuẩn hóa nên (1 + a)ᴸ nổ; thực tế khối residual được chuẩn hóa hoặc khởi tạo sao cho F nhỏ lúc đầu.
- Số đo trên mạng thật (bảng 12 ô ở ô "phân tích thêm" của lab; He init, batch 32 ảnh ngẫu nhiên 3×16×16, L khối conv 3×3 16 kênh, tại khởi tạo, một seed; "g đầu / g cuối" là chuẩn gradient lớp conv đầu chia lớp conv cuối). Ở L = 60: plain không BN có 0.414, tức **không** biến mất gradient ở khởi tạo (degradation là vấn đề tối ưu khi huấn luyện); plain + BN có 3 750, gradient lớp đầu phình lên. Hiện tượng này có giải thích lý thuyết: Yang và cộng sự (ICLR 2019, mean field theory of batch normalization) cho rằng chính BN gây gradient explosion ở mạng sâu khởi tạo ngẫu nhiên (đã đọc abstract, chưa đọc thân bài). Số đo của chúng tôi cũng lệch với một câu trong paper ResNet về gradient của mạng plain có BN [cần kiểm chứng nguyên văn mục 4.1: cách hiểu của chúng tôi là paper nói gradient có chuẩn bình thường]. Cấu hình ở đây khác paper (16 kênh, khởi tạo, một batch ngẫu nhiên), nên chưa kết luận paper sai; residual không BN có loss khởi tạo 1.45 × 10¹⁰ vì mỗi khối cộng thêm phương sai; residual + BN ổn định (loss 5.29, tỉ lệ 6.06).
- Bài học: residual cần chuẩn hóa hoặc khởi tạo cẩn thận (kỹ thuật "zero-init residual": đặt γ của BN cuối khối bằng 0). Đây là cấu hình nhỏ, một seed; chỉ đọc kết luận định tính.

**Vì sao ML cần cái này.**

- **Đếm trước khi huấn luyện.** Biết tham số nằm ở đâu và MAC nằm ở đâu: giới hạn bộ nhớ (6 GB của RTX 4050) bị tham số, activation và batch quyết định, thời gian bị MAC quyết định.
- **Ba ý tưởng quay lại khắp nơi.** 1×1 nén kênh thành lớp chiếu Q, K, V và MLP trong Transformer; residual thành `x + Sublayer(x)`; conv tách theo chiều sâu cho mô hình nhỏ trên thiết bị di động.

## 7.4 Huấn luyện CNN trong thực tế: norm, augmentation, transfer learning

Với CNN, ba quyết định thực tế quyết định chất lượng không kém kiến trúc: chuẩn hóa theo trục nào, tăng cường dữ liệu nào là an toàn, và dùng lại backbone pretrained thế nào cho đúng. Cả ba đều có bẫy chạy không báo lỗi.

**BatchNorm cho conv (`BatchNorm2d`).** Công thức giống chương 6: x̂ = (x − μ)/√(σ² + ε), y = γx̂ + β; train dùng thống kê batch và cập nhật running stats với momentum 0.1, eval dùng running stats. Điểm mới là trục lấy thống kê: với tensor N × C × H × W, `BatchNorm2d` tính μ và σ² cho **từng kênh**, trên mọi mẫu và mọi vị trí (N·H·W giá trị), nên mỗi kênh có một cặp γ, β và một cặp running stats: 2C tham số học được và 2C giá trị running. Lý do trung bình cả H·W: chia sẻ trọng số của conv nghĩa là mọi vị trí của một kênh là "cùng một đặc trưng".

| Chuẩn hóa | Lấy thống kê trên | Phụ thuộc batch | Dùng nhiều ở |
| --- | --- | --- | --- |
| BatchNorm2d | N, H, W cho từng kênh | Có (train khác eval) | CNN thị giác |
| LayerNorm trên (C, H, W) | C, H, W cho từng mẫu | Không | Hiếm trong CNN cổ điển |
| GroupNorm | Nhóm kênh, H, W cho từng mẫu | Không | Batch nhỏ: phát hiện, phân đoạn |
| LayerNorm trên chiều đặc trưng | d của từng token | Không | Transformer (chương 8) |

Wu và He (2018, Group Normalization) báo cáo lỗi của BN tăng nhanh khi batch nhỏ vì thống kê batch ước lượng kém: trên ResNet-50, ImageNet, batch 2, GN có lỗi thấp hơn BN 10.6 điểm phần trăm (24.1% so với 34.7%; mục 4.1 của paper, abstract chỉ nêu con số 10.6). GN chia kênh thành nhóm (mặc định G = 32) và chuẩn hóa trong từng mẫu. Khi batch nhỏ (ảnh lớn trong phát hiện và phân đoạn) dùng GroupNorm hoặc SyncBN (gộp thống kê qua nhiều GPU).

**Ví dụ giải mẫu 7.12 — BatchNorm2d trên tensor N = 2, C = 2, 2×2, tính tay.** Kênh 0: mẫu 0 có các giá trị 1, 2, 3, 4; mẫu 1 có 5, 6, 7, 8. Kênh 1: mẫu 0 có 0, 0, 2, 2; mẫu 1 có 1, 1, 3, 3.

- Kênh 0: 8 giá trị, trung bình 4.5, phương sai lệch (chia 8) là 5.25, phương sai không lệch (chia 7) là 6.0. Chuẩn hóa mẫu 0: (1 − 4.5)/√5.25 = −1.5275, rồi −1.0911, −0.6547, −0.2182; mẫu 1: 0.2182, 0.6547, 1.0911, 1.5275.
- Kênh 1: trung bình 1.5, phương sai lệch 1.25, không lệch 10/7 = 1.4286.
- Running stats sau một bước train (momentum 0.1, khởi đầu 0 và 1): `running_mean` = 0.1·(4.5, 1.5) = (0.45, 0.15); `running_var` = (0.9 + 0.1·6.0, 0.9 + 0.1·1.4286) = (1.5, 1.0429), nhớ rằng `running_var` dùng phương sai không lệch. PyTorch cho đúng các số này.
- LayerNorm trên (C, H, W) của mẫu 0: lấy thống kê trên cả 8 giá trị của hai kênh (trung bình 1.75, phương sai lệch 1.6875), ra −0.577, 0.192, 0.962, 1.732 cho kênh 0 và −1.347, −1.347, 0.192, 0.192 cho kênh 1: khác hẳn BN, vì trục chuẩn hóa khác.

% manim: BN2dKenh

**Gộp BN vào conv khi suy luận (conv-BN fusion).** Ở chế độ eval, BN là một phép affine cố định theo từng kênh, và conv là tuyến tính, nên gộp được thành một conv duy nhất, bớt một lượt đọc ghi bộ nhớ:

$$
w' = w \cdot \frac{\gamma}{\sqrt{\sigma^{2} + \epsilon}}, \qquad\qquad b' = (b - \mu)\cdot \frac{\gamma}{\sqrt{\sigma^{2} + \epsilon}} + \beta
$$

Ở đây σ², μ là running stats, phép nhân áp dụng theo từng kênh ra. Phép gộp **chỉ hợp lệ ở chế độ eval**: lúc train BN dùng thống kê của batch, đổi theo từng batch, nên không gộp vào trọng số cố định được (lab 5: nếu BN còn ở train mode, sai khác lên tới 2.55).

**Ví dụ giải mẫu 7.13 — gộp BN bằng tay.** Conv 1×1 một kênh vào, một kênh ra, w = 2, b = 1. BN có running mean μ = 3, running var σ² = 4, γ = 1.5, β = 0.5, bỏ ε. Hệ số √σ² = 2.

- w′ = 2·1.5/2 = 1.5; b′ = (1 − 3)·1.5/2 + 0.5 = −1.5 + 0.5 = −1.0.
- Kiểm với x = 4: conv cho 2·4 + 1 = 9; BN cho (9 − 3)/2·1.5 + 0.5 = 5; conv đã gộp: 1.5·4 − 1.0 = 5. Khớp. Trên conv 3 → 8 kênh với BN ngẫu nhiên ở eval, sai khác lớn nhất trong lab là 1.9 × 10⁻⁶ (float32).

**Tăng cường dữ liệu (augmentation).** Lật ngang, cắt ngẫu nhiên, đổi màu, dịch ảnh: tạo mẫu mới từ mẫu cũ, rẻ và hiệu quả, miễn là phép biến đổi **bảo toàn nhãn**. Lật ngang đúng với ảnh con mèo, sai với chữ viết (b thành d), ký hiệu toán, ảnh y khoa có hướng, biển báo rẽ trái hay phải, và sai nếu có hai lớp là ảnh gương của nhau. Trong bài săn bug, hai lớp "đường chéo" và "phản chéo" là ảnh gương của nhau: bật lật ngang ngẫu nhiên làm acc val tụt từ 0.981 xuống 0.792, ma trận nhầm lẫn lộ ra hai lớp đó lẫn vào nhau. Lỗi nằm ở dữ liệu, không ở mô hình.

**Transfer learning.** Lấy một backbone đã huấn luyện trước (thường trên ImageNet), thay đầu phân loại, rồi hoặc **đóng băng** backbone và chỉ huấn luyện đầu mới (ít dữ liệu), hoặc **fine-tune** toàn bộ với learning rate nhỏ hơn cho backbone (nhiều dữ liệu hơn, miền gần). Ba yêu cầu để không âm thầm hỏng:

1. Đầu vào khớp lúc pretrain: thứ tự kênh RGB (không phải BGR như OpenCV), trục NCHW, chuẩn hóa bằng mean và std của ImageNet (`torchvision`: mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225), sau khi chia 255).
2. Ảnh từ PIL, scikit-image, OpenCV có dạng H × W × C, PyTorch cần C × H × W: phải dùng `permute`. `view` hay `reshape` **không báo lỗi** (số phần tử khớp) nhưng trộn điểm ảnh và kênh. Trong bài săn bug: acc val 0.607 thay vì 0.981 chỉ vì một dòng đó.
3. Với BN, `requires_grad_(False)` **không** tắt cập nhật running stats. Quên `backbone.eval()` thì mỗi lần forward ở chế độ train ghi đè `running_mean` và `running_var` bằng thống kê của dữ liệu mới, dù trọng số không đổi. Đo trong lab (ô "phân tích thêm"; backbone hai conv + BN đóng băng, miền mới có cường độ ×3 và nhiễu mạnh hơn): trọng số không đổi (kiểm bằng `torch.equal`) nhưng `running_var` của lớp BN đầu **nhân 42.3 lần**, và acc miền mới của bản quên `eval()` là 0.456, **cao hơn** bản đúng 0.411, vì BN ở chế độ train tự thích nghi thống kê với miền mới (kiểu AdaBN). Vẫn là lỗi quy trình: kết quả phụ thuộc thứ tự và kích thước batch, không tái lập, không an toàn khi triển khai.

**Vì sao ML cần cái này.**

- **Đa số lỗi thực tế của CNN nằm ở quy trình, không ở kiến trúc:** sai thứ tự trục, quên `eval()`, augmentation sai nhãn. Bộ nhớ khi huấn luyện chủ yếu là activation (tăng theo batch và độ phân giải), không phải tham số.
- **BN ở NLP.** Transformer không dùng BN; so bảng trên để thấy vì sao LayerNorm (độc lập batch, từng token) được chọn.

## 7.5 Cầu nối sang NLP: conv 1D, residual và chuẩn hóa, vì sao chuyển sang attention

Conv 1D là cùng ý tưởng với 7.1 nhưng trượt trên chiều độ dài của chuỗi. Qua nó bạn thấy vì sao Transformer (chương 8) thay phần lớn conv trong NLP: vùng nhìn của conv tăng chậm và trọng số trộn cố định, còn attention nhìn mọi cặp token ngay trong một lớp và để dữ liệu quyết định trọng số trộn, với cái giá chi phí tăng theo bình phương độ dài.

**Conv 1D trên chuỗi token.** Mỗi token có embedding d chiều; chuỗi n token trong một batch B mẫu là tensor B × d × n (embedding là kênh), và một bộ lọc cỡ k là một "bộ phát hiện k-gram" học được. **TextCNN** (Kim, EMNLP 2014): bộ lọc cỡ 3, 4, 5, mỗi cỡ 100 bộ lọc, dropout 0.5 (mục 3.1 của paper), embedding word2vec 300 chiều (mục 3.2); sau conv và ReLU, **max-over-time pooling** lấy giá trị lớn nhất của mỗi bộ lọc trên toàn chuỗi, rồi một lớp Linear. Đây là global max pooling của mục 7.2 theo chiều độ dài: cho phép chuỗi có độ dài bất kỳ (miễn dài hơn kernel lớn nhất) và cho invariance theo vị trí của k-gram.

**Ví dụ giải mẫu 7.14 — đếm TextCNN và thí nghiệm "thứ tự".**

- Đếm: d = 300, nhóm (3, 100), (4, 100), (5, 100): 3·300·100 + 100 + 4·300·100 + 100 + 5·300·100 + 100 = 360 300 tham số conv; Linear 300 → 2 có 602; tổng 360 902. Bảng embedding cho từ vựng 20 000 từ: 6 000 000, gấp hơn 16 lần phần conv: trong NLP, phần lớn tham số thường là embedding.
- Thí nghiệm (lab 5, dữ liệu tổng hợp): từ vựng 50 token, chuỗi dài 20; nhãn 1 khi cặp token (3, 7) xuất hiện **liền kề đúng thứ tự**; một nửa mẫu nhãn 0 là "mồi nhử" có cả 3 lẫn 7 nhưng không liền kề. Embedding 16 chiều, 32 bộ lọc, 15 epoch Adam lr 3 × 10⁻³, 2 400 mẫu train, 1 000 val, một seed. Acc val: túi từ (trung bình embedding) 0.696; conv k = 1 0.666; **conv k = 2: 1.000**; k = 3: 0.996. Một mô hình chỉ biết "có 3 và có 7 hay không" (đoán nhãn 1 khi có cả hai) đạt tối đa 0.725 trên tập val này: BoW và k = 1 nằm dưới trần đó, k = 2 vượt hẳn vì thấy thứ tự liền kề. Dữ liệu đã được kiểm đúng luật nhãn (nhãn 1 ⇔ có bigram (3, 7) liền kề, cả train lẫn val).
- Giải thích: túi từ và k = 1 chỉ biết có token 3 và 7 trong chuỗi hay không, nên không phân biệt được mẫu nhãn 1 với mồi nhử; kernel k = 2 nhìn được cửa sổ **có thứ tự** gồm hai vị trí liền kề, tiền thân của "ngữ cảnh cục bộ" trong NLP. Bài toán tổng hợp, chỉ minh họa cơ chế.

**Conv nhân quả và dilated.** Với mô hình ngôn ngữ, đầu ra tại t không được thấy tương lai: **conv nhân quả** (causal) đệm d(k − 1) số 0 về phía trái (k − 1 nếu không giãn; đã kiểm ở ô "Kiểm nhanh" của lab: đổi một đầu vào ở t = 5 chỉ đổi đầu ra từ t = 5 trở đi). Để vùng nhìn lớn mà không chồng quá nhiều lớp, dùng **dilation** tăng theo cấp số nhân (mỗi lớp gấp đôi lớp trước): ý của WaveNet và TCN (Bai, Kolter, Koltun, 2018).

**Ví dụ giải mẫu 7.15 — RF của dilated conv.** Kernel k = 2, dilation 1, 2, 4, …, 512 (10 lớp): RF = 1 + (1 + 2 + … + 512) = 1 024; lặp ba khối: 1 + 3·1 023 = 3 070. Với k = 3 và dilation 1, 2, 4: RF = 1 + 2·(1 + 2 + 4) = 15, phủ **liền mạch** từ −7 đến 7 vì {−1, 0, 1} + {−2, 0, 2} + {−4, 0, 4} cho mọi số nguyên trong khoảng (nhưng số đường đi tới mỗi điểm vào không đều: 1, 1, 2, 1, 3, 2, 3, 1, 3, …); ba lớp 3×3 không giãn chỉ có 7. RF tăng theo cấp số nhân với độ sâu (nhớ ERF ở ví dụ 7.6). Bai và cộng sự (2018) nêu trong abstract rằng một kiến trúc tích chập đơn giản vượt LSTM thông thường trên nhiều bài, với "longer effective memory"; chúng tôi chưa đọc thân bài nên không nêu con số.

**Vì sao Transformer thay phần lớn conv trong NLP.**

1. **Vùng nhìn.** Một lớp attention nhìn mọi cặp token; conv cần chồng lớp (tuyến tính theo số lớp, hoặc cấp số nhân với dilation) mới thấy xa.
2. **Trọng số trộn phụ thuộc dữ liệu.** Trong conv, trọng số trộn hàng xóm cố định theo vị trí tương đối; trong attention, nội dung hai token quyết định trọng số giữa chúng.
3. **Cái giá: chi phí n².** Với d = 768, MAC một lớp conv k = 3 là n·3·d² (tuyến tính theo n); phần điểm số và trộn của attention (bỏ các phép chiếu Q, K, V, O) là 2·n²·d.

**Ví dụ giải mẫu 7.16 — chi phí conv so với attention (d = 768, chỉ tính hai thành phần trên).**

| Độ dài n | Conv k = 3: n·3·d² | Attention (điểm số + trộn): 2·n²·d | Tỉ lệ attention / conv |
| --- | --- | --- | --- |
| 128 | 2.26 × 10⁸ | 2.52 × 10⁷ | 0.11 |
| 512 | 9.06 × 10⁸ | 4.03 × 10⁸ | 0.44 |
| 2 048 | 3.62 × 10⁹ | 6.44 × 10⁹ | 1.78 |
| 8 192 | 1.45 × 10¹⁰ | 1.03 × 10¹¹ | 7.11 |

Hai thành phần bằng nhau khi 2n²d = 3nd², tức n = 1.5·d = 1 152. Đừng đọc bảng như "conv rẻ hơn attention": một lớp conv k = 3 chỉ nhìn 3 token, còn attention nhìn cả chuỗi (bài tập 13 so hai thứ ở cùng vùng nhìn).

**ViT biến ảnh thành chuỗi token bằng một conv.** Vision Transformer (Dosovitskiy và cộng sự, 2020) cắt ảnh thành các patch P × P (thường 16×16), duỗi mỗi patch thành vector P²·3 số và nhân với một ma trận để ra vector D chiều: token. Phép đó chính là một conv với kernel P × P, stride P.

**Ví dụ giải mẫu 7.17 — patch embedding.** Conv 16×16 stride 16, 3 → 768 kênh: 3·16·16·768 + 768 = 590 592 tham số. Ảnh 224×224 cho (224/16)² = 14·14 = 196 token, tensor 1 × 196 × 768. Tương đương với `unfold` rồi `Linear`, đã kiểm ở ô "Kiểm nhanh" của lab (khớp tới 10⁻⁴ ở float32, cấu hình nhỏ hơn: 64×64, 32 kênh). Ảnh 384×384 có 576 token, gấp 2.94 lần, nên chi phí attention theo n² gấp khoảng 8.6 lần.

% manim: PatchLaConv

**Residual và chuẩn hóa là nền của Transformer.** Mỗi khối Transformer gồm hai nhánh con (attention và MLP), mỗi nhánh bọc bằng đường tắt cộng và LayerNorm: đúng cấu trúc của ResNet, với LayerNorm thay BN và attention hoặc MLP thay conv. Post-LN, pre-LN và hệ quả với warm-up nằm ở chương 8; ở đây chỉ cần nhớ pre-LN giữ đường tắt là hàm đồng nhất thuần (như pre-activation của He và cộng sự, 2016, "Identity Mappings in Deep Residual Networks"), và ví dụ 7.11 cho thấy residual không chuẩn hóa thì nổ.

**Thiên kiến cấu trúc và dữ liệu.** Conv có thiên kiến mạnh (cục bộ, chia sẻ) nên học được từ ít dữ liệu; Dosovitskiy và cộng sự viết rằng ViT có "much less image-specific inductive bias than CNNs" (mục 3.1 của paper), nên khi huấn luyện ở cỡ ImageNet mà **không có regularization mạnh**, ViT kém ResNet cùng cỡ vài điểm phần trăm, và chỉ vượt khi pretrain trên tập rất lớn (JFT-300M). Điều kiện "không có regularization mạnh" quan trọng: DeiT (Touvron và cộng sự, 2020) đạt 83.1% top-1 trên ImageNet chỉ bằng huấn luyện trên ImageNet-1k (đã đọc abstract), nhờ augmentation, regularization mạnh và distillation. Vậy nhận định đúng là "ViT nhạy với công thức huấn luyện và cần nhiều dữ liệu hoặc regularization mạnh để bù thiên kiến cấu trúc", không phải "ViT luôn cần tập khổng lồ". Chi tiết ở chương 8. Với NLP, lý do thường được nêu cho việc Transformer thắng sớm là lượng văn bản chưa gán nhãn rất lớn để pretrain và nhu cầu phụ thuộc xa, không phải vì văn bản thiếu ngữ cảnh cục bộ (đó là lý do TextCNN hoạt động); cách giải thích phổ biến, chưa kiểm nguồn.

**Vì sao ML cần cái này.**

- **Phỏng vấn NLP vẫn hỏi về CNN** như một nhánh so sánh ("vì sao attention thay conv", "TextCNN làm gì"); bạn trả lời được bằng số: RF, chi phí n² so với n, thiên kiến cấu trúc.
- **Conv 1D vẫn sống:** trích đặc trưng giọng nói, character CNN, và các khối "conv cục bộ + attention" (Conformer: chỉ nêu tên như từ khóa, chưa kiểm nguồn).

## Bài tập giấy bút

14 bài, làm theo thứ tự. Hết ★ trong 1 buổi, hết ★★ trong 2 buổi, ★★★ dành cho Thứ Bảy. Bài nào sai ghi vào sổ lỗi kèm lý do sai.

**★ Hiểu định nghĩa**

1. `nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)` có bias, nhận ảnh 3×32×32. Cho kích thước đầu ra, số tham số và số MAC.
2. Tính kích thước đầu ra theo một chiều: (a) n = 100, k = 5, p = 0, s = 2; (b) n = 64, k = 3, p = 1, s = 2; (c) n = 15, k = 3, d = 2, p = 0, s = 1. (d) Với k = 3, s = 2, p = 0, hai kích thước vào n = 7 và n = 8 cho cùng kích thước ra. Giá trị đó là bao nhiêu, và với n = 8 điểm nào của đầu vào bị bỏ qua?
3. Chồng lớp gồm conv 5×5 (stride 1), max-pool 2×2 (stride 2), conv 3×3 (stride 1), conv 3×3 (stride 1). Tính receptive field và jump của một điểm đầu ra.
4. x = (2, 0, 1, 3, 1), w = (1, 2, 3). Tính cross-correlation (chế độ valid, không đệm) và convolution toán học của x với w. Hai kết quả khác nhau thế nào, và `nn.Conv1d` tính cái nào?

**★★ Mức đề thi đầu vào**

5. (a) Chứng minh: conv 1D với circular padding, stride 1 là equivariant với phép dịch vòng: nếu x′[i] = x[i − a] thì y′[i] = y[i − a]. (b) Với zero padding, phát biểu này sai ở đâu? (c) Nếu thêm stride 2, với những giá trị a nào của phép dịch thì đầu ra còn dịch đúng?
6. L lớp conv 3×3 stride 1, mỗi lớp C kênh vào và ra, bỏ bias, cho receptive field 2L + 1. (a) Viết tỉ lệ tham số so với một lớp conv phẳng (2L + 1)×(2L + 1) cùng C kênh. (b) Tính cho L = 4 với C = 128. (c) L nhỏ nhất để tỉ lệ dưới 0.4 là bao nhiêu? (d) Nêu hai lý do khác (ngoài số tham số) để xếp chồng 3×3.
7. Conv tách theo chiều sâu 3×3, 256 → 256 trên bản đồ 14×14 (bỏ bias): đếm tham số và MAC, so với conv 3×3 thường. Tỉ lệ này phụ thuộc vào những đại lượng nào?
8. (a) Dẫn xuất công thức gộp BN vào conv: w′ và b′ theo w, b, γ, β, running mean μ, running var σ², ε. (b) Vì sao phép gộp sai ở chế độ train? (c) Nếu conv không có bias thì sao?
9. Conv 1D stride 2: x = (1, 2, 3, 4, 5), w = (2, −1), y[i] = 2x[2i] − x[2i + 1] (hai đầu ra), gradient từ phía trên g = (1, 3). Tính y, dw và dx; điểm nào của x nhận gradient 0, vì sao? Kiểm dw[0] bằng cách tăng w[0] thêm 1 và tính lại L = ⟨g, y⟩.
10. TextCNN: từ vựng 30 000, embedding d = 200, các nhóm bộ lọc (cỡ 2, 128 bộ), (cỡ 3, 128 bộ), (cỡ 4, 128 bộ), có bias, rồi Linear sang 2 lớp. (a) Đếm tham số conv, Linear, tổng, và phần của bảng embedding. (b) Câu "TextCNN nhỏ vì chỉ có 3 kích thước kernel" sai ở đâu?

**★★★ Mức phỏng vấn nghiên cứu**

11. Receptive field hiệu dụng. (a) Kernel 1D không âm, chuẩn hóa, có trọng số đều trên k vị trí liên tiếp (k lẻ): chứng minh phương sai là (k² − 1)/12 và suy ra độ lệch chuẩn ERF của L lớp. Kiểm với k = 3. (b) Vì sao ERF/RF giảm khi L tăng? (c) Với k = 3 và dilation 1, 2, 4, 8 (L = 4), tính độ lệch chuẩn ở cùng giả định và so với RF 31. Nhận xét về hình dạng phân bố.
12. Residual. (a) Với y = x + F(x), khai triển tích Jacobian của L khối (I + Jₗ)…(I + J₁) và đếm số đường gradient. (b) Mô hình vô hướng: tính aᴸ và (1 + a)ᴸ với a = 0.9, L = 50, và nói vì sao (1 + a)ᴸ không phải kết luận thực tế. (c) Nếu đặt γ = 0 (và β = 0) cho BN cuối mỗi khối thì mỗi khối giữ nguyên kích thước làm gì lúc khởi tạo, và cả một ResNet thật có là hàm đồng nhất không? (d) Dùng số đo ở ví dụ 7.11 để phản bác câu "residual sửa gradient biến mất", và nêu cách đọc lại câu đó để có phần đúng (Veit và cộng sự).
13. Conv và attention cùng vùng nhìn. d = 768, n = 8 192. (a) Để vùng nhìn của conv k = 2 dilated phủ cả n token cần bao nhiêu lớp (dilation 1, 2, 4, …), và tổng MAC (mỗi lớp n·2·d²)? (b) Một lớp attention đầy đủ (4 phép chiếu n·d² mỗi phép, cộng 2n²d cho điểm số và trộn): tổng MAC? So hai con số. (c) Vì sao MAC chưa nói hết: tính số phần tử ma trận điểm số n × n và bộ nhớ (float32, một đầu, một mẫu). (d) Hai khác biệt về chất lượng thông tin giữa hai cách (đường đi giữa hai token, trọng số phụ thuộc dữ liệu).
14. ViT. (a) Vì sao ViT dùng patch 16×16 chứ không coi mỗi điểm ảnh (patch 1×1) là một token? Dùng số token và chi phí n² của attention cho ảnh 224×224. (b) Vì sao phải nội suy position embedding khi fine-tune ở 384×384 với patch 16? (c) Nêu hai thiên kiến cấu trúc của CNN mà ViT không có sẵn, và một kết quả của paper ViT nói về hệ quả.

## Bài tính nhanh

15 bài, mỗi bài tối đa 2 phút, làm bằng giấy bút và bấm giờ. Mục tiêu: xong trong 25 phút, sai không quá 1 bài. Làm lại bộ này sau 3 ngày và sau 1 tuần.

1. Kích thước ra của conv 3×3, p = 1, s = 1 trên ảnh 32; và của conv 3×3, p = 0, s = 2 trên ảnh 32.
2. Số tham số của `nn.Conv2d(3, 16, 3)` có bias.
3. Số MAC của lớp trên với ảnh 32×32 và padding 1 (đầu ra 32×32).
4. Số tham số của conv 1×1 từ 256 xuống 64 kênh, không bias.
5. Receptive field và jump của hai lớp conv 3×3 stride 2 liên tiếp.
6. Số tham số của conv depthwise 3×3 trên 64 kênh, không bias.
7. Số tham số của lớp Linear từ 7·7·512 chiều xuống 4 096 chiều (có bias).
8. Global average pooling trên tensor 512 × 7 × 7 cho vector bao nhiêu chiều, có bao nhiêu tham số?
9. `nn.BatchNorm2d(64)`: số tham số học được và số giá trị running stats.
10. BN với N = 8, bản đồ 14×14: mỗi kênh lấy thống kê trên bao nhiêu giá trị?
11. ViT với patch 16: số token cho ảnh 224×224 và cho ảnh 384×384.
12. Cross-correlation (valid) của (1, 2, 3) với (1, −1).
13. Receptive field của một lớp conv k = 3 với dilation 4.
14. Max-pool 2 (stride 2) của (3, 1, 4, 1, 5, 9, 2, 6).
15. Gộp BN: conv có w = 4; BN có γ = 2, σ² = 16, bỏ ε. Tính w′.

## Lab code và bài săn bug

Năm lab và ô "Kiểm nhanh" đã chạy ngày 04/11/2026 với Python 3, NumPy 2.5, SciPy 1.18.1, PyTorch 2.14.1 (bản CPU là đủ; bản cài ở đây là 2.14.1+cu130) và torchvision 0.29.1 (chỉ ví dụ 7.9 dùng); toàn bộ notebook lời giải chạy khoảng 1 phút khi máy rảnh (2 luồng CPU), chậm gấp vài lần khi máy bị chia sẻ. Số liệu có thể lệch nhẹ theo phiên bản và số luồng. Dữ liệu sinh tại chỗ, không tải gì: sáu mẫu hình 5×5 trong ảnh 12×12 có nhiễu Gauss 0.3, và một bài văn bản tổng hợp cho lab 5.

| Lab | Việc phải làm | Kết quả mong đợi |
| --- | --- | --- |
| 1. Conv từ đầu | `out_size`, `conv2d_naive` (vòng lặp), `conv2d_im2col` (ma trận cột + một phép nhân ma trận) | Ví dụ 7.1: (−1, −3, −1), (2, −4, 0), (5, −3, −3), `convolve2d` cho đúng âm; khớp `F.conv2d` trên 7 cấu hình; `out_size(224, 11, 0, 4) = 54`, `out_size(227, 11, 0, 4) = 55` |
| 2. Backward của conv | `conv_backward` chỉ bằng `correlate2d` và kernel lật; gradient check float64 | Ví dụ 7.4: dx = (1, 2, 2, −2, −3), dw = (14, 20, 26); sai số tương đối 4.3 × 10⁻¹⁰ (dx), 3.0 × 10⁻¹⁰ (dw); khớp autograd |
| 3. RF và ERF | `rf(layers)` trả (RF, jump); `erf_std(L)` đo bằng autograd | (7, 1), (24, 4), (11, 4) cho ba chồng của ví dụ 7.5; ERF std 0.816, 1.633, 3.266, 6.532 với L = 1, 4, 16, 64 (chia √L luôn 0.8165) |
| 4. Vị trí lạ | MLP, CNN + Flatten, CNN + global max pooling; huấn luyện ở offset 2–4, kiểm ở 0, 1, 6, 7 | 19 334 / 2 118 / 3 670 tham số; acc vùng đã thấy 0.858 / 0.952 / 0.996; vị trí lạ 0.183 / 0.242 / 0.996 (2 seed, 12 epoch) |
| 5. Gộp BN và TextCNN | `fold_bn`; `BoW` và `TextCNN(k)` trên dữ liệu "bigram (3, 7) liền kề" | w′ = 1.5, b′ = −1.0 (ví dụ 7.13); sai khác 1.9 × 10⁻⁶; BN ở train mode sai khác 2.55; trần của mô hình "có 3 và có 7" 0.725; acc val: BoW 0.696, k = 1 0.666, k = 2 1.000, k = 3 0.996 |

Code nằm trong `labs/chuong07/lab.ipynb` (có chỗ TODO để bạn tự viết, kèm ô kiểm tra tự động) và `labs/chuong07/loigiai.ipynb` (lời giải đã chạy kiểm); mở từ trang **Lab** của app. Phần "Phân tích thêm" ở cuối notebook là **tùy chọn**: kết luận chính đã có trong sách, ô đó cho bảng đầy đủ của ví dụ 7.11 và lỗi quên `eval()` ở mục 7.4.

**Câu hỏi sau lab.**

- Lab 3: ERF std / √L không đổi ở 0.8165; trong mạng đã huấn luyện điều này còn đúng không, điều gì đã thay đổi?
- Lab 4: CNN + Flatten ít tham số hơn MLP mà vẫn kém ở vị trí lạ: số tham số có phải biến giải thích không? Đề xuất thí nghiệm kiểm.
- Lab 5: vì sao conv k = 1 không hơn BoW và k = 2 đã đủ? Nếu nhãn phụ thuộc bộ ba (3, 7, 5) liền kề thì sao?

**Bài săn bug.** Hàm `train_buggy` dưới đây chạy không báo lỗi. Ảnh lưu dạng NHWC như PIL hoặc scikit-image; mô hình `Conv2d(3, 8, 3, p=1) → BatchNorm2d → ReLU → MaxPool2d(2) → Conv2d(8, 16, 3, p=1) → BatchNorm2d → ReLU → AdaptiveMaxPool2d(1) → Flatten → Linear(16, 6)`, Adam lr 1e-3, 12 epoch, 2 400 mẫu train, 1 000 val. Triệu chứng đo được: loss val 1.613, acc val 0.346; hai lần đánh giá trên cùng tập val cho acc 0.346 và 0.356; ma trận nhầm lẫn rải khắp các lớp. Bản đúng đạt loss val 0.146, acc val 0.981. Tìm cả **bốn** lỗi trước khi xem lời giải.

```python
def train_buggy(epochs=12, seed=0):
    torch.manual_seed(seed)
    Xtr_hwc, ytr = make_rgb(2400, seed=1)                 # (N, 12, 12, 3)
    Xva_hwc, yva = make_rgb(1000, seed=2)
    def prep(a):
        t = torch.tensor(a)
        return t.view(len(t), 3, 12, 12)                  # đưa về dạng PyTorch cần
    X, y, Xv, yv = prep(Xtr_hwc), torch.tensor(ytr), prep(Xva_hwc), torch.tensor(yva)
    def augment(xb):                                      # tăng cường dữ liệu: lật ngang ngẫu nhiên
        flip = torch.rand(len(xb)) < 0.5
        return torch.where(flip[:, None, None, None], xb.flip(3), xb)
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]
            xb, yb = augment(X[idx]), y[idx]
            if torch.rand(1).item() < 0.5:
                xb = torch.roll(xb, 2, dims=0)            # "xáo trộn" batch cho đa dạng
            opt.zero_grad()
            F.cross_entropy(net(xb), yb).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        xv = augment(Xv)
        out1 = net(xv)
        out2 = net(augment(Xv))                           # gọi lần hai trên cùng tập val
        val_loss = F.cross_entropy(out1, yv).item()
        acc1 = (out1.argmax(1) == yv).float().mean().item()
        acc2 = (out2.argmax(1) == yv).float().mean().item()
    return net, val_loss, acc1, acc2, out1.argmax(1), yv
```

## Bài dựng cảnh Manim

Mỗi chương có một bài tự dựng cảnh: muốn vẽ được một khái niệm, bạn phải hiểu từng bước của nó. Giới hạn **1 giờ** (tính vào luồng C), không chỉnh màu mè.

**Đề bài.** Dùng `manim/chuong07_scenes.py` làm mẫu (cảnh `ChongBaLop` gần nhất), viết cảnh `DilatedRF` vào thư mục `manim/cua-toi/`:

1. Vẽ bốn hàng điểm (đầu vào và đầu ra của ba lớp conv, mỗi hàng 17 điểm).
2. Ba lớp conv k = 3 với dilation 1, 2, 4 (từ đầu vào ra đầu ra). Chọn điểm giữa của hàng đầu ra, rồi lần lượt kẻ các cạnh về lớp trước: lớp cuối (dilation 4) nối tới các điểm cách ±4, lớp giữa ±2, lớp đầu ±1.
3. Tô màu các điểm đầu vào ảnh hưởng đến điểm đầu ra và hiện số điểm: receptive field.

**Tiêu chí tự chấm.** (1) Đúng toán: RF = 1 + 2·(1 + 2 + 4) = 15 điểm đầu vào, phủ liền mạch từ −7 đến 7 (độ lệch {−1, 0, 1} + {−2, 0, 2} + {−4, 0, 4} cho mọi số nguyên); k = 2 với dilation 1, 2, 4 cũng phủ liền 0…7 (thiết kế của WaveNet và TCN, ví dụ 7.15); lỗ chỉ xuất hiện nếu bỏ lớp dilation 1 (k = 3 với dilation 2, 4 chỉ phủ số chẵn) hoặc dilation tăng nhanh hơn k lần mỗi lớp (k = 3 với dilation 1, 4, 16); ba lớp không giãn chỉ cho 7. Ô "Kiểm nhanh" của lab in các trường hợp này. (2) Có tiêu đề và phụ đề nói người xem phải thấy điều gì (vì sao dilation tăng gấp đôi cho RF tăng theo cấp số nhân). (3) Dài không quá 30 giây, chữ không chồng chữ.

Render bằng nút **Render cảnh của tôi** ở trang Manim. Lời giải tham khảo nằm ở `manim/loigiai/chuong07_dilated_rf.py` — chỉ mở sau khi đã tự làm.

## Phỏng vấn, bẫy, thực tế khác lý thuyết

**Cách luyện.** Mỗi chuỗi có 4 tầng câu hỏi đào sâu dần. Trả lời to bằng tiếng Việt trước, rồi nói lại bằng tiếng Anh, 2 phút mỗi tầng, ghi âm. Câu trả lời mẫu tiếng Anh chỉ phủ tầng 1–2; tầng 3–4 bạn tự soạn rồi đối chiếu với lời giải bài tập liên quan. Không có nguồn đề thi công bố cho Qualcomm hay VinAI; các câu dưới đây là dạng thường gặp trong phỏng vấn thị giác máy tính và học sâu, không gán cho công ty nào.

**Chuỗi 1 — Conv, thiên kiến cấu trúc và dịch chuyển**

1. Tích chập khác lớp kết nối đầy đủ ở đâu? Vì sao nó hợp với ảnh?
2. Tính tham số và MAC của một lớp conv 3×3, 64 → 128 kênh trên bản đồ 56×56. Vì sao MAC lớn hơn tham số hàng nghìn lần?
3. CNN có bất biến với phép dịch không? Giải thích bằng equivariance, pooling, stride, và nêu ví dụ `Flatten` làm hỏng. (Ví dụ 7.7 và 7.8.)
4. Receptive field lý thuyết khác receptive field hiệu dụng thế nào, và hệ quả cho thiết kế mạng? (Ví dụ 7.6.)

> *Model answer:* A fully connected layer gives every output its own weight on every input, while a convolution assumes locality and weight sharing: each output sees a small neighborhood and one kernel is reused at every position. That cuts the parameter count by orders of magnitude and makes the layer equivariant to translation. A 3×3 layer from 64 to 128 channels on a 56×56 map has about 74 thousand parameters but about 231 million multiply-adds, since each weight is reused at 3,136 positions. Convolutions are equivariant, not invariant: global pooling gives only approximate invariance, because strided layers alias.

**Chuỗi 2 — Residual và độ sâu**

1. Vấn đề degradation là gì, và vì sao nó không phải overfitting?
2. Khối residual làm gì? Viết đạo hàm ∂y/∂x.
3. Bottleneck 256 → 64 → 64 → 256 tiết kiệm bao nhiêu so với hai conv 3×3 ở 256 kênh, và vì sao? (Ví dụ 7.10.)
4. "Residual sửa gradient biến mất": đúng hay sai? Bạn có số đo gì, và vì sao plain + BN lại có gradient lớp đầu phình lên (Yang và cộng sự, 2019)? Rồi so pre-activation của ResNet với pre-LN của Transformer.

> *Model answer:* Degradation means a deeper plain network has higher training error than a shallower one, so it cannot be overfitting; the optimizer fails to find a solution at least as good. A residual block outputs x plus F of x, so its Jacobian is the identity plus the Jacobian of F, and the block only learns a correction; if the identity is nearly optimal, pushing F to zero is easy.

**Chuỗi 3 — BatchNorm trong CNN và transfer learning**

1. BatchNorm2d tính thống kê trên trục nào? Có bao nhiêu tham số và buffer?
2. Gộp BN vào conv thế nào, và vì sao chỉ ở eval? (Ví dụ 7.13.)
3. Batch nhỏ thì BN hỏng ra sao, GroupNorm là gì? Bạn đóng băng backbone có BN để fine-tune: những việc nào phải làm đúng?
4. Vì sao Transformer dùng LayerNorm chứ không phải BN?

> *Model answer:* BatchNorm2d computes the mean and variance per channel over the batch and both spatial dimensions, so a 64-channel layer has 128 learnable parameters plus 128 running statistics. At inference it is a fixed per-channel affine map, so it folds into the preceding convolution: scale the kernel by gamma over the square root of the running variance plus epsilon and shift the bias. This is valid only in eval mode; and when freezing a pretrained backbone, requires grad false is not enough, because without eval the running statistics keep drifting even though the weights do not change.

**Chuỗi 4 — Từ conv 1D tới attention**

1. TextCNN hoạt động thế nào, và nó phát hiện gì? Vì sao max-over-time?
2. Causal và dilated conv là gì? Tính receptive field của k = 2, dilation 1, 2, …, 512. (Ví dụ 7.15.)
3. So conv, RNN và attention: đường đi thông tin giữa hai token xa nhau, chi phí theo độ dài, tính song song. Dùng số của ví dụ 7.16.
4. ViT chuyển ảnh thành chuỗi token thế nào, và vì sao ViT thiếu thiên kiến cấu trúc thì cần nhiều dữ liệu hoặc regularization mạnh hơn CNN? DeiT thay đổi bức tranh đó thế nào (augmentation, regularization mạnh, distillation)? (Ví dụ 7.17.)

> *Model answer:* A TextCNN slides filters of width three, four and five over the embeddings; each filter is a learned n-gram detector, and max-over-time pooling keeps its strongest activation, so the output does not depend on position or length. A causal convolution pads on the left so the output at time t never sees the future; with kernel size two and dilations one to 512 the receptive field is 1,024. In a convolution the path between two tokens grows with their distance (linearly in depth, or logarithmically with dilation), while attention connects any pair in one layer and computes the mixing weights from the content, at quadratic cost: at width 768 it exceeds a width-three convolution beyond about 1,150 tokens.

**Từ vựng nên nói trôi chảy:** convolution, cross-correlation, dilation, receptive field, effective receptive field, weight sharing, translation equivariance, invariance, aliasing, global average pooling, bottleneck, depthwise separable convolution, skip connection, degradation problem, GroupNorm, frozen backbone, max-over-time pooling, causal convolution, dilated convolution, patch embedding, inductive bias.

**Bẫy thường gặp.**

- Tin `nn.Conv2d` là convolution toán học: nó là cross-correlation (ví dụ 7.1: lệch đúng một dấu âm với Sobel).
- Nói "CNN bất biến dịch chuyển": conv là equivariant; invariant xấp xỉ nhờ global pooling và bị stride phá (ví dụ 7.7); `Flatten` + `Linear` mất hẳn tổng quát hóa vị trí (0.242 so với 0.996, ví dụ 7.8).
- Dùng `view` hay `reshape` thay cho `permute` khi chuyển NHWC sang NCHW: không báo lỗi nhưng trộn điểm ảnh và kênh (bài săn bug: acc 0.981 xuống 0.607).
- Quên `backbone.eval()` khi đóng băng BN: running stats vẫn trôi (`running_var` nhân 42.3 lần, trọng số không đổi); bản quên `eval()` còn cho acc cao hơn (0.456 so với 0.411) nhưng không tái lập.
- Đếm sai: 224 với kernel 11, stride 4, p = 0 cho 54 chứ không phải 55; bỏ lớp FC khi đếm tham số AlexNet và VGG (96% và 89% ở FC); lẫn MAC với FLOP (1 MAC ≈ 2 FLOP); và `padding='same'` với stride > 1: kích thước ra là ⌈n/s⌉, còn PyTorch từ chối tổ hợp này (`ValueError`), nên khi đổi framework hãy tự tính lại.

**Thực tế khác lý thuyết.**

- **Residual và "gradient biến mất".** Câu quen thuộc là residual sửa gradient biến mất. Paper ResNet viết rằng khó khăn tối ưu của plain có BN "unlikely to be caused by vanishing gradients" (mục 4.1); Veit và cộng sự giải thích bằng các đường ngắn mang gradient. Số đo của chúng tôi (ví dụ 7.11, cấu hình nhỏ): plain 60 lớp He-init có tỉ lệ gradient đầu/cuối cỡ 0.4, residual không chuẩn hóa nổ (loss khởi tạo 1.45 × 10¹⁰). Plain + BN còn có gradient lớp đầu phình lên (3 750 ở 60 lớp), điều mà Yang và cộng sự (2019) giải thích bằng lý thuyết mean field: BN ở khởi tạo ngẫu nhiên làm gradient nổ khi sâu. Chắc chắn: tối ưu khó hơn khi sâu và đường tắt cộng chuẩn hóa chữa được; cơ chế còn tranh luận.
- **CNN không thật sự bất biến dịch chuyển.** Zhang (2019) và Azulay và Weiss (2019) đo trên mạng thật: stride gây aliasing, BlurPool và augmentation chỉ cải thiện một phần. Số đo của chúng tôi: một lớp `Flatten` + `Linear` đủ để mất tổng quát hóa vị trí.
- **Số FLOP không bằng tốc độ.** Conv tách theo chiều sâu giảm 8.4 lần tham số và MAC (ví dụ 7.10) nhưng thường không nhanh tương ứng vì cường độ tính toán thấp, bị giới hạn bởi băng thông bộ nhớ GPU. Hiểu biết phổ biến, chưa đo ở đây: khi tối ưu suy luận, hãy đo thời gian thật.

## Mini-test 30 phút

Làm sau khi xong bài tập và lab, bấm giờ đúng 30 phút, không xem sách, không dùng máy tính (câu 5 chỉ đọc code). Tổng 30 điểm. **Từ 24 điểm:** sang chương 8. **18 đến dưới 24:** ôn lại mục ứng với câu sai rồi làm lại sau 3 ngày. **Dưới 18:** học lại chương.

1. **(4 điểm)** Conv 3×3, stride 2, padding 1, không bias, 3 → 32 kênh, ảnh vào 3×64×64. Tính kích thước đầu ra, số tham số và số MAC.
2. **(4 điểm)** Chồng lớp: conv 5×5 (s1), max-pool 3×3 (s2), conv 3×3 (s1), conv 3×3 với dilation 2 (s1). Tính receptive field và jump của một điểm đầu ra.
3. **(5 điểm)** Dẫn xuất công thức gộp BatchNorm vào conv (w′, b′ theo w, b, γ, β, μ, σ², ε) và nêu vì sao phép gộp chỉ hợp lệ ở chế độ eval.
4. **(5 điểm)** So sánh MLP, CNN + Flatten + Linear, và CNN + global max pooling khi mẫu hình ở vị trí chưa từng thấy lúc huấn luyện. Dự đoán kết quả từng mô hình, giải thích cơ chế, và nói một lý do vì sao stride vẫn có thể phá bất biến ngay cả với global pooling ở cuối.
5. **(6 điểm)** Đọc code và trả lời không chạy máy:

```python
img = skimage.io.imread("x.jpg")                      # mảng uint8 cỡ (H, W, 3)
x = torch.from_numpy(img).float().view(1, 3, H, W) / 255
net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
for p in net.parameters():
    p.requires_grad_(False)
net.fc = nn.Linear(512, 10)                            # (nhãn y có sẵn, không viết ra ở đây)
opt = torch.optim.SGD(net.fc.parameters(), lr=0.01)
for step in range(100):
    net.train()
    loss = F.cross_entropy(net(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
```

(a) Dòng chuẩn bị `x` có lỗi gì, vì sao không báo lỗi? (b) Thiếu bước nào trong tiền xử lý để khớp mô hình pretrained (nêu ý bắt buộc và ý nên có)? (c) Dòng `net.train()` gây vấn đề gì với các lớp BN đã đóng băng, và sửa thế nào?

6. **(6 điểm)** Tối đa 5 câu: vì sao Transformer thay phần lớn conv 1D trong NLP, dù attention đắt hơn theo độ dài? Dùng số: d = 768, n = 2 048.

## Flashcard

20 thẻ của chương nằm trong `flashcards/chuong07.csv`. Ôn ở trang **Ôn tập** của app: app tự lên lịch theo hộp Leitner (thẻ nhớ được sẽ giãn lịch ôn 1 → 3 → 7 → 14 ngày, thẻ quên quay lại hộp 1). File CSV này cũng nhập thẳng được vào Anki nếu bạn thích dùng Anki.

## Lời giải

Chỉ đọc sau khi đã tự thử ít nhất 20 phút. Với bài chứng minh, so ý chính chứ không so từng chữ. Đáp số bằng số đã kiểm lại bằng Python; số từ thí nghiệm là kết quả một lần chạy của chúng tôi (ít seed), có thể lệch nhẹ khi bạn chạy lại.

**Kiểm tra nhanh**

1. Đầu ra 16×32×32; tham số 3·16·9 + 16 = 448.
2. (a) 7×7 (1 + 3·2). (b) Không: conv equivariant (dịch đầu vào thì đầu ra dịch theo); bất biến chỉ xấp xỉ, nhờ pooling toàn cục, và stride làm nó mất chính xác.
3. Không phải overfitting, vì train error cũng tăng (degradation: vấn đề tối ưu). ∂y/∂x = I + ∂F/∂x.
4. Trên các trục N, H, W cho từng kênh; 2·64 = 128 tham số học được (γ và β), cộng 128 giá trị running stats (cộng bộ đếm `num_batches_tracked`, 1 giá trị).
5. 100·(3·300 + 1) = 90 100 tham số. Max-over-time lấy kích hoạt mạnh nhất của mỗi bộ lọc trên cả chuỗi: đầu ra có số chiều cố định dù chuỗi dài bao nhiêu, và không phụ thuộc vị trí của n-gram.

**Bài tập giấy bút**

1. Đầu ra 64×32×32. Tham số: 64·3·25 + 64 = 4 864. MAC: 32·32·64·(3·25) = 4 915 200 (bias không tính).
2. (a) ⌊(100 − 5)/2⌋ + 1 = 48. (b) ⌊(64 + 2 − 3)/2⌋ + 1 = 32. (c) kernel hiệu dụng 2·2 + 1 = 5: 15 − 5 + 1 = 11. (d) Cả hai cho 3: n = 7 có ba cửa sổ phủ 0–2, 2–4, 4–6; n = 8 cũng ba cửa sổ đó nên điểm cuối (chỉ số 7) bị bỏ qua, vì phép chia lấy sàn bỏ phần dư.
3. conv 5×5: RF 5, jump 1. Pool 2×2 stride 2: RF = 5 + 1·1 = 6, jump 2. Conv 3×3: 6 + 2·2 = 10. Conv 3×3: 10 + 2·2 = 14. Kết quả: RF 14, jump 2.
4. Cross-correlation: (2, 0, 1) → 2 + 0 + 3 = 5; (0, 1, 3) → 0 + 2 + 9 = 11; (1, 3, 1) → 1 + 6 + 3 = 10: kết quả (5, 11, 10). Convolution lật w thành (3, 2, 1): 7, 5, 10: kết quả (7, 5, 10). `nn.Conv1d` tính cross-correlation, không lật.
5. (a) Với circular padding, y[i] = ∑ⱼ w[j]·x[(i + j) mod n]. Nếu x′[i] = x[i − a] thì y′[i] = ∑ⱼ w[j]·x[(i + j − a) mod n] = y[i − a]. (b) Với zero padding, phần nội dung bị đẩy ra ngoài biên bị mất và số 0 chen vào từ phía kia; đẳng thức chỉ đúng ở các vị trí cách biên xa hơn bán kính kernel. (c) Với stride 2, đầu ra là y[2m]: dịch a chẵn cho đầu ra dịch a/2 (đúng); a lẻ thì các mẫu lấy ra thuộc tính chẵn lẻ khác, đầu ra không phải bản dịch của đầu ra cũ: aliasing.
6. (a) Chồng: L·9·C². Phẳng: (2L + 1)²·C². Tỉ lệ 9L/(2L + 1)². (b) L = 4: 4·9·128² = 589 824 so với 81·128² = 1 327 104, tỉ lệ 0.444. (c) L = 4 cho 0.444, L = 5 cho 45/121 = 0.372: L nhỏ nhất là 5. (d) L lần phi tuyến thay vì một lần (biểu diễn phong phú hơn), MAC giảm theo cùng tỉ lệ vì mỗi tham số dùng lại ở mọi vị trí, và ít tham số hơn nên ít nguy cơ overfit.
7. Conv thường 9·256² = 589 824 tham số; tách theo chiều sâu: 9·256 + 256² = 2 304 + 65 536 = 67 840. Nhân với 14·14 = 196 vị trí: 115 605 504 MAC so với 13 296 640. Tỉ lệ 8.69 lần cho cả tham số và MAC, đúng bằng 1/(1/256 + 1/9): chỉ phụ thuộc Cₒᵤₜ và k, không phụ thuộc kích thước bản đồ.
8. (a) y = γ(w x + b − μ)/√(σ² + ε) + β. Đặt s = γ/√(σ² + ε): w′ = s·w, b′ = s·(b − μ) + β. (b) Ở train mode BN dùng μ, σ² của batch hiện tại, đổi theo từng batch, nên không phải ánh xạ affine cố định. (c) Không có bias thì coi b = 0: b′ = β − s·μ, nên lớp gộp phải có bias (`bias=True`).
9. y = (2·1 − 2, 2·3 − 4) = (0, 2) (hai cửa sổ bắt đầu ở 0 và 2; x₄ không được dùng). dw[0] = ∑ᵢ gᵢ·x[2i] = 1·1 + 3·3 = 10; dw[1] = ∑ᵢ gᵢ·x[2i + 1] = 1·2 + 3·4 = 14. Mỗi gᵢ trả về hai điểm của cửa sổ i: dx = (w₀g₀, w₁g₀, w₀g₁, w₁g₁, 0) = (2, −1, 6, −3, 0); x₄ bị bỏ qua nên không nhận gradient (cùng gốc với bài 2d). Kiểm dw[0]: L = ⟨g, y⟩ = 0 + 6 = 6; với w[0] = 3, y = (1, 5), L = 1 + 15 = 16; thay đổi 10 = dw[0] (L tuyến tính theo w nên chính xác).
10. (a) Conv: mỗi nhóm cỡ k có 128·(k·200 + 1) tham số: 51 328; 76 928; 102 528; tổng 230 784. Linear: 384·2 + 2 = 770. Phần không phải embedding: 231 554. Bảng embedding: 30 000·200 = 6 000 000; tổng 6 231 554, embedding chiếm 96.3%. (b) Sai: tham số bị chi phối bởi bảng embedding (96%), không phải số kích thước kernel; phần conv và Linear chỉ 231 554.
11. (a) Giả thiết: kernel không âm, tổng bằng 1, đọc như phân phối của một độ dịch X; gradient lan ngược qua L lớp là chập lặp L lần, tức phân phối của tổng L độ dịch độc lập nên phương sai cộng (bỏ qua ReLU và hiệu ứng biên). Với trọng số đều trên các độ lệch −m, …, m (k = 2m + 1): 𝔼[X] = 0 và 𝔼[X²] = (2/k)·(1² + … + m²) = (2/k)·m(m + 1)(2m + 1)/6 = m(m + 1)/3 = (k² − 1)/12. Vậy σ = √(L(k² − 1)/12); k = 3: σ = 0.8165·√L, khớp 0.816, 1.633, 3.266, 6.532 của lab 3. (b) RF = 2L + 1 tăng tuyến tính còn σ tăng như √L, nên σ/RF giảm như 1/√L. (c) Dilation d nhân độ dịch với d nên phương sai nhân d²: (2/3)(1 + 4 + 16 + 64) = 56.7, σ = 7.53 (bốn lớp không giãn: 1.633); RF = 31, nửa độ rộng 15 ≈ 2σ. Support liền mạch từ −15 đến 15 nhưng trọng số gồ ghề, nên phân bố rộng hơn nhiều với nhiều đỉnh nhỏ.
12. (a) (I + Jₗ)⋯(I + J₁) = tổng trên mọi tập con các khối: 2ᴸ số hạng, tương ứng 2ᴸ đường từ đầu ra về đầu vào; số hạng toàn I là đường tắt thuần, gradient truyền nguyên vẹn. (b) 0.9⁵⁰ ≈ 0.00515, 1.9⁵⁰ ≈ 8.66 × 10¹³. Mô hình không chuẩn hóa nên cũng cho tín hiệu thuận bùng nổ (loss khởi tạo cỡ 10¹⁰ ở ví dụ 7.11): 1.9⁵⁰ là lý do phải chuẩn hóa hoặc khởi tạo nhỏ nhánh F, không phải dự đoán về gradient thật. (c) Cần γ = 0 và β = 0 ở BN cuối khối (β = 0 là mặc định của PyTorch) để F(x) = 0, và đường tắt là đồng nhất; khi đó mọi khối giữ nguyên kích thước là x ↦ x (ReLU sau phép cộng giữ nguyên x vì x ≥ 0 do ReLU của khối trước). Nhưng cả mạng **không** là hàm đồng nhất: stem, các khối chiếu 1×1 có stride (đổi số kênh hoặc độ phân giải) và lớp fc vẫn còn, nên lúc khởi tạo mạng tương đương một mạng nông gồm stem, các khối chiếu và fc. (d) Ở ví dụ 7.11, plain 60 lớp He-init có tỉ lệ gradient đầu/cuối cỡ 0.4 (plain + BN còn phình lên): không có gradient biến mất ở khởi tạo, nên câu đó không giải thích degradation ở cấu hình này (He và cộng sự cũng viết vậy). Cách đọc lại cho có phần đúng: theo Veit và cộng sự, mạng residual như tập các đường nông, gradient chủ yếu đi qua đường ngắn.
13. (a) Receptive field của k = 2, dilation 1, 2, 4, … sau L lớp là 2ᴸ; 2¹³ = 8 192 nên cần 13 lớp, MAC: 13·8 192·2·768² = 1.256 × 10¹¹. (b) Bốn phép chiếu: 4·8 192·768² = 1.93 × 10¹⁰; điểm số và trộn: 2·8 192²·768 = 1.03 × 10¹¹; tổng 1.224 × 10¹¹, gần bằng chuỗi conv: MAC không phân biệt hai cách ở n này. (c) Ma trận điểm số có 8 192² = 67 108 864 phần tử, 256 MiB (float32) cho mỗi đầu và mỗi mẫu; conv không có ma trận này: bộ nhớ mới là giới hạn thực tế của attention đầy đủ khi n lớn. (d) Attention nối hai token bất kỳ trong một lớp, còn conv dilated cần tới 13 lớp; trọng số attention tính từ nội dung token, còn trọng số conv cố định theo vị trí tương đối.
14. (a) Với patch 1×1, ảnh 224×224 có 50 176 token và ma trận điểm số 50 176² ≈ 2.52 × 10⁹ phần tử (khoảng 9.4 GiB float32 cho một đầu, một mẫu), gấp 256² = 65 536 lần so với patch 16 (196 token). Conv nhìn từng điểm ảnh với chi phí tuyến tính, còn attention là n², nên phải gộp điểm ảnh thành patch trước khi vào attention; patch 16 nén 16·16·3 = 768 số thành một token, patch 32 chỉ còn 49 token nhưng mất chi tiết. (b) Position embedding học theo lưới 14×14; ở 384×384 lưới là 24×24 (576 token), nên nội suy hai chiều (thường bicubic) lên 24×24. (c) Tính cục bộ và equivariance dịch chuyển (do chia sẻ trọng số). Paper ViT: ở cỡ ImageNet và không có regularization mạnh thì kém ResNet cùng cỡ vài điểm, chỉ vượt khi pretrain trên JFT-300M. Điều kiện này quan trọng: DeiT đạt 83.1% top-1 chỉ với ImageNet-1k nhờ augmentation, regularization mạnh và distillation, nên thiếu thiên kiến cấu trúc được bù bằng dữ liệu hoặc công thức huấn luyện, không nhất thiết bằng tập lớn hơn.

**Bài tính nhanh**

1. 32; ⌊(32 − 3)/2⌋ + 1 = 15.
2. 3·16·9 + 16 = 448.
3. 32·32·16·27 = 442 368.
4. 256·64 = 16 384.
5. RF 3, jump 2 sau lớp đầu; lớp hai: 3 + 2·2 = 7, jump 4. Vậy RF 7, jump 4.
6. 64·9 = 576.
7. 25 088·4 096 + 4 096 = 102 764 544.
8. 512 chiều; 0 tham số.
9. 128 tham số học được; 128 giá trị running stats (cộng `num_batches_tracked`).
10. 8·14·14 = 1 568.
11. 196 và 576.
12. (1 − 2, 2 − 3) = (−1, −1).
13. 1 + 4·2 = 9.
14. (3, 4, 9, 6).
15. s = γ/σ = 2/4 = 0.5; w′ = 0.5·4 = 2.

**Câu hỏi sau lab**

- Lab 3: 0.8165 chỉ đúng với kernel đều, tuyến tính. Mạng đã huấn luyện có ReLU, kernel không đều và kết nối tắt nên ERF đổi; theo Luo và cộng sự ERF vẫn gần Gauss. Chúng tôi chưa đo mạng đã huấn luyện.
- Lab 4: số tham số không phải biến giải thích: CNN + Flatten (2 118) ít hơn MLP (19 334) nhưng cả hai hỏng ở vị trí lạ (0.242 và 0.183), còn CNN + GMP (3 670) thì không; biến quyết định là đầu ra có dùng chung trọng số qua các vị trí hay không. Thí nghiệm kiểm: huấn luyện MLP với dịch ngẫu nhiên phủ mọi offset; nếu MLP khá lên thì thiên kiến cấu trúc có thể bù bằng dữ liệu.
- Lab 5: k = 1 chỉ thấy từng token nên chỉ biết "có 3 và có 7 ở đâu đó", không hơn BoW; k = 2 thấy đúng cặp liền kề. Với bộ ba (3, 7, 5) liền kề, k = 2 chỉ thấy (3, 7) và (7, 5) riêng rẽ nên bị nhầm; cần k ≥ 3 hoặc hai lớp k = 2 xếp chồng (RF 3).

**Mini-test**

1. (4 điểm) Đầu ra 32×32 ((64 + 2 − 3)/2 → 31 + 1 = 32): 1 điểm. Tham số 3·32·9 = 864: 1 điểm. MAC 32·32·32·27 = 884 736: 2 điểm (1 điểm nếu quên số vị trí đầu ra).
2. (4 điểm) Conv 5×5: RF 5, jump 1. Pool 3×3 stride 2: 5 + 2·1 = 7, jump 2. Conv 3×3: 7 + 2·2 = 11. Conv 3×3 dilation 2: 11 + 2·2·2 = 19. RF 19: 3 điểm; jump 2: 1 điểm.
3. (5 điểm) y = γ(w x + b − μ)/√(σ² + ε) + β; đặt s = γ/√(σ² + ε): w′ = s·w, b′ = s·(b − μ) + β: 3 điểm. Ở train mode BN dùng thống kê của batch, đổi theo từng batch, nên không phải ánh xạ affine cố định: 2 điểm.
4. (5 điểm) MLP kém ngoài vùng đã thấy (khoảng 0.18) vì mỗi điểm ảnh có trọng số riêng: 1 điểm. CNN + Flatten cũng kém (0.24) vì Linear sau Flatten gắn trọng số riêng cho từng vị trí đầu ra của conv: 1.5 điểm. CNN + global max pooling gần như không đổi (0.996) vì conv equivariant và max toàn cục bất biến: 1.5 điểm. Stride phá bất biến vì lấy mẫu con theo lưới: dịch một điểm đổi mẫu được giữ (aliasing): 1 điểm.
5. (6 điểm) (a) `.view(1, 3, H, W)` trên mảng (H, W, 3) chỉ đổi cách đọc bộ nhớ: số phần tử khớp nên không báo lỗi nhưng trộn kênh và điểm ảnh; dùng `permute(2, 0, 1).unsqueeze(0)`: 2 điểm. (b) Thiếu chuẩn hóa mean, std ImageNet theo kênh (ý bắt buộc): 1.5 điểm; resize/crop về cỡ quen thuộc (ResNet có `AdaptiveAvgPool` nên chạy được với mọi cỡ, nhưng khớp phân phối lúc pretrain thì tốt hơn) là ý cộng: 0.5 điểm. (c) `net.train()` bật lại cập nhật running stats của BN. Với batch chỉ một ảnh như ở đây, BN ở train mode còn dùng thống kê của đúng một ảnh. Sửa: bỏ `net.train()` (hoặc gọi `net.eval()`); mạng không có dropout nên eval không mất gì, và `fc` học được nhờ `requires_grad`, không phụ thuộc chế độ train: 2 điểm.
6. (6 điểm) Gợi ý: attention nối hai token bất kỳ trong một lớp với trọng số tính từ nội dung (2 điểm). Với d = 768, n = 2 048: một lớp attention đầy đủ khoảng 4nd² + 2n²d = 1.13 × 10¹⁰ MAC, một lớp conv k = 3 là 3.6 × 10⁹ nhưng chỉ nhìn 3 token; phủ 2 048 token bằng dilated k = 2 cần 11 lớp, khoảng 2.7 × 10¹⁰ MAC (2 điểm). Nhược điểm n² (vượt conv k = 3 khi n > 1.5d ≈ 1 150, ma trận điểm số tốn bộ nhớ) có thật nhưng ở n vừa phải được bù bằng đường đi ngắn và song hóa tốt (2 điểm).

**Bài săn bug.** Bốn lỗi:

1. `t.view(len(t), 3, 12, 12)` trên mảng NHWC: số phần tử khớp nên không báo lỗi, nhưng bộ nhớ vẫn xếp theo (H, W, C), nên điểm ảnh và kênh bị trộn. Sửa bằng `permute(0, 3, 1, 2).contiguous()`. Một mình lỗi này kéo acc từ 0.981 xuống 0.607.
2. Lật ngang ngẫu nhiên khi huấn luyện: hai lớp "đường chéo" và "phản chéo" là ảnh gương của nhau (bốn lớp còn lại đối xứng khi lật ngang), nên lật biến mẫu của lớp này thành mẫu của lớp kia, tức đổi nhãn thật (không label-preserving). Một mình: acc 0.792.
3. `torch.roll(xb, 2, dims=0)` dịch các ảnh trong batch mà không dịch nhãn, nên một nửa số batch có ảnh lệch nhãn hai vị trí. Một mình: acc 0.899.
4. `augment(Xv)` lúc đánh giá: đây là chính hàm `augment` có lật ngẫu nhiên, nên là "lật lúc đánh giá", độc lập với lỗi 2 (lật lúc huấn luyện). Đánh giá phải xác định và không đổi nhãn. Một mình lỗi này cho acc 0.822 (hai lần gọi đều ra 0.822 trong lần chạy này: lật ngẫu nhiên nhưng tình cờ trùng acc; muốn thấy tính ngẫu nhiên hãy so `torch.equal(out1, out2)`, sẽ ra `False`, hoặc đếm số mẫu val bị lật ở mỗi lần gọi); gộp cả bốn lỗi thì ra 0.346 và 0.356.

Bản sửa (hàm đầy đủ `train_fixed` nằm trong notebook lời giải), ba dòng thay đổi so với `train_buggy`:

```python
return torch.tensor(a).permute(0, 3, 1, 2).contiguous()   # sửa lỗi 1: permute, không view
xb, yb = X[idx], y[idx]                                    # sửa lỗi 2 và 3: bỏ augment lật và torch.roll
out1, out2 = net(Xv), net(Xv)                              # sửa lỗi 4: đánh giá không augment, xác định
```

Đã chạy kiểm: bản sửa cho loss val 0.146, acc val 0.981, và hai lần gọi trên cùng tập val cho output giống hệt nhau.

## Tài liệu đào sâu và nguồn

Nội dung chương được viết lại, đối chiếu với các nguồn dưới đây. Mức đã kiểm (29/10/2026): **đã mở và đọc đúng đoạn** ResNet (Bảng 2, Hình 4, mục 4.1), Group Normalization (mục 4.1), TextCNN (mục 3.1, 3.2), ViT (mục 3.1, 4.2–4.3); **đã mở, chỉ đọc abstract** TCN, Zhang (2019), Yang và cộng sự (2019), DeiT; **link lấy từ kết quả tìm kiếm, chưa mở trang** Azulay và Weiss (2019); **ID arXiv ghi từ trí nhớ, chưa mở, [cần kiểm chứng]** VGG, MobileNet V1 và V2, WaveNet; **chưa có link vì không phải arXiv** AlexNet (NeurIPS 2012) và LeNet-5 (1998); **chưa kiểm lại** CS231n, tài liệu PyTorch (quy ước `Conv2d` được đối chiếu bằng cách chạy `F.conv2d` và `convolve2d`), số của LeNet-5 bản gốc và các bài còn lại. Số ở ví dụ 7.9 do chúng tôi tự đếm bằng `torchvision`.

| Tài liệu | Dùng để làm gì | Phần nên đọc |
| --- | --- | --- |
| [CS231n: Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/) | Công thức kích thước, chia sẻ trọng số, 1×1, dilation, case study LeNet, AlexNet, VGG, ResNet | Toàn bộ ghi chú; chưa mở lại được trong lần viết này |
| [He và cộng sự (2015), Deep Residual Learning](https://arxiv.org/abs/1512.03385) | ResNet, degradation, bottleneck | Bảng 2, Hình 4, mục 4.1 |
| [He và cộng sự (2016), Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) | Pre-activation | Phần phân tích đường tắt |
| [Veit, Wilber, Belongie (2016), Residual Networks Behave Like Ensembles](https://arxiv.org/abs/1605.06431) | Cách đọc khác về residual | Toàn bài |
| [Luo và cộng sự (2016), Effective Receptive Field](https://arxiv.org/abs/1701.04128) | ERF gần Gauss | Toàn bài |
| [Zhang (2019), Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486) | Aliasing, BlurPool | Toàn bài |
| [Wu & He (2018), Group Normalization](https://arxiv.org/abs/1803.08494) | GN khi batch nhỏ | Mục 4.1 (batch nhỏ) |
| [Kim (2014), CNN for Sentence Classification](https://arxiv.org/abs/1408.5882) | TextCNN | Mục 3 (ngắn) |
| [Bai, Kolter, Koltun (2018), Generic Convolutional and Recurrent Networks](https://arxiv.org/abs/1803.01271) | TCN, causal và dilated conv | Phần kiến trúc |
| [Dosovitskiy và cộng sự (2020), An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) | ViT, patch embedding, nhu cầu dữ liệu | Mục 3.1, 4.2–4.3 |
| [Touvron và cộng sự (2020), Training data-efficient image transformers (DeiT)](https://arxiv.org/abs/2012.12877) | ViT chỉ huấn luyện trên ImageNet-1k, 83.1% top-1 (abstract) | Abstract; phần augmentation và distillation |
| [Yang và cộng sự (2019), A Mean Field Theory of Batch Normalization](https://arxiv.org/abs/1902.08129) | BN gây gradient explosion ở mạng sâu khởi tạo ngẫu nhiên | Abstract; phần gradient |
| [Azulay & Weiss (2019), Why do deep convolutional networks generalize so poorly to small image transformations?](https://arxiv.org/abs/1805.12177) | Dự đoán đổi theo phép dịch rất nhỏ (JMLR 2019) | Toàn bài; link từ kết quả tìm kiếm, chưa mở trang |
| [Simonyan & Zisserman (2014), Very Deep Convolutional Networks (VGG)](https://arxiv.org/abs/1409.1556) | VGG, xếp chồng 3×3 | Mục 2 [cần kiểm chứng ID, ghi từ trí nhớ] |
| [Howard và cộng sự (2017), MobileNets (V1)](https://arxiv.org/abs/1704.04861) | Conv tách theo chiều sâu | Mục 3 [cần kiểm chứng ID, ghi từ trí nhớ] |
| [Sandler và cộng sự (2018), MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) | Inverted residual, linear bottleneck; số liệu ví dụ 7.9 là của V2 | Mục 3 [cần kiểm chứng ID, ghi từ trí nhớ] |
| [van den Oord và cộng sự (2016), WaveNet](https://arxiv.org/abs/1609.03499) | Dilated causal conv | Mục 2 [cần kiểm chứng ID, ghi từ trí nhớ] |
| Krizhevsky, Sutskever, Hinton (2012), ImageNet Classification with Deep Convolutional Neural Networks (NeurIPS 2012) | AlexNet: ReLU, dropout, GPU | Chưa có link arXiv (không có trên arXiv); chưa mở được trang trong lần viết này |

**Từ khóa để tự tìm thêm:** BlurPool, zero-init residual, AdaBN, RepVGG, SE block, ResNeXt, Conformer, Swin Transformer, deformable convolution.

**Trước khi sang chương 8:** đạt từ 24/30 điểm mini-test, làm lại các bài ★★★ sai, tự viết lại được `conv2d_im2col` và `conv_backward` (lab 1 và 2) trong 30 phút không nhìn code mẫu, và tìm được cả bốn lỗi của bài săn bug.

Nguồn tra cứu ngày 29/10/2026; bổ sung nguồn Yang và cộng sự, DeiT, Azulay và Weiss ngày 04/11/2026.
