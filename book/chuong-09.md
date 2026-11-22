# Chương 9 — Mô hình ngôn ngữ lớn

## Mục tiêu và kiểm tra nhanh

Chương này đi theo vòng đời của một mô hình ngôn ngữ lớn (LLM): văn bản thành token, token thành phân phối xác suất, huấn luyện trước ở quy mô lớn, căn chỉnh theo ý người dùng, thích nghi và nén để chạy trên phần cứng nhỏ, cuối cùng là đo xem nó có tốt không. Chương 8 đã dạy kiến trúc (attention, KV cache, RoPE, đếm tham số, FLOPs ≈ 6N mỗi token); chương này dùng lại những thứ đó mà không dạy lại. Người phỏng vấn NLP gần như chắc chắn hỏi ở đây. Thời lượng danh nghĩa: 2 tuần, 8 giờ mỗi tuần (tuần 1: mục 9.1–9.2 và Lab 1–3; tuần 2: mục 9.3–9.5, Lab 4–5 và bài săn bug). Làm hết thì nặng hơn 16 giờ; để về gần mức đó, ví dụ 9.2 và 9.13, phần beam và PhoGPT chỉ cần đọc lướt, phần RAG và agent của mục 9.5 chỉ đọc, và bài Manim làm sau cùng, bỏ được nếu chậm tiến độ.

**Cách đọc chương này.** Phương pháp bền, tên mô hình thì không. Tên mô hình, thứ hạng và điểm benchmark năm 2026 nằm gọn trong mục "Có thể lỗi thời" (kiểm ngày 30/10/2026); phần lý thuyết không phụ thuộc vào chúng. Số liệu từ paper phần lớn đã qua tóm tắt công cụ, không phải đọc nguyên văn; riêng Chinchilla (Eq. 10, Table 2–3, Figure 4) và hình minh họa BPE của Sennrich (Figure 1) đã được đối chiếu với PDF gốc ngày 04/11/2026. Chương ghi rõ chỗ nào chưa đối chiếu.

**Sau chương này, bạn phải làm được:**

- ☐ Chạy tay BPE trên một bộ từ nhỏ, giải thích byte-level BPE và vì sao tiếng Việt tốn token hơn tiếng Anh; tính perplexity và nói vì sao không so được giữa hai tokenizer.
- ☐ Tính phân phối sau nhiệt độ, top-k, top-p; chỉ ra vì sao văn bản "xác suất cao nhất" không phải văn bản hay.
- ☐ Dùng 6ND để tính ngân sách huấn luyện, dẫn xuất tỉ lệ tối ưu của Chinchilla từ fit tham số, và nêu ba lý do người ta huấn luyện vượt Chinchilla.
- ☐ Viết loss reward model và loss DPO, dẫn xuất DPO từ bài toán có KL; nói GRPO khác PPO ở đâu và giới hạn của RLVR.
- ☐ Đếm tham số LoRA, tính ngân sách bộ nhớ của một mô hình 7B, tính pass@k không chệch và nói benchmark nhỏ có sai số bao nhiêu.

**Kiểm tra nhanh (5 phút, không dùng máy tính; câu 4 cho sẵn e⁻⁰·⁴ ≈ 0.670, chấp nhận sai số ±0.01).** Câu i kiểm tra mục 9.i.

1. Chạy hai phép gộp BPE đầu tiên trên "aaab aaab" (xem mỗi dấu cách là ranh giới từ, bỏ qua ký hiệu hết từ). Sau mỗi phép, tổng số token của cả hai từ là bao nhiêu?
2. Mô hình gán xác suất đều cho 8 token ở mỗi bước. Perplexity bằng bao nhiêu, bao nhiêu bit mỗi token?
3. Ngân sách 6ND = 1.2·10²³ FLOPs. Mô hình có 10¹⁰ tham số thì huấn luyện bao nhiêu token? Tỉ lệ token mỗi tham số là bao nhiêu?
4. Hai đáp án có reward 1.0 và 0.6. Xác suất reward model (Bradley-Terry) ưa đáp án đầu là bao nhiêu?
5. Ma trận 1000 × 1000, LoRA hạng r = 4. Số tham số học được so với tinh chỉnh toàn phần?

Đáp án ở mục Lời giải cuối chương.

## 9.1 Từ văn bản đến phân phối và trở lại: BPE, perplexity, giải mã

Một mô hình ngôn ngữ là một phân phối xác suất trên chuỗi token, được phân rã theo quy tắc chuỗi xác suất. Huấn luyện là cực tiểu cross-entropy của token kế tiếp (loss ban đầu ≈ ln V, như chương 6); sinh văn bản là lấy mẫu từng token từ phân phối đó.

$$
p(x_1,\dots,x_n)=\prod_{t=1}^{n} p(x_t \mid x_{<t}),\qquad \mathcal{L}=-\frac{1}{n}\sum_{t=1}^{n}\ln p_\theta(x_t \mid x_{<t})
$$

**Vì sao token con (subword).** Từ nguyên vẹn thì từ vựng khổng lồ và gặp từ lạ là bó tay. Ký tự thì chuỗi quá dài, mà attention tốn n² (chương 8). Subword nằm giữa: từ hay gặp là một token, từ hiếm vỡ thành mảnh. Từ vựng V còn quyết định ma trận embedding V·d tham số (GPT-2 small: 38 597 376 trên 124 439 808, chương 8).

**BPE** (Sennrich, Haddow, Birch, 2016, cho dịch máy) lặp: đếm các cặp ký hiệu liền kề (có trọng số tần số của từ), gộp cặp nhiều nhất thành một ký hiệu mới. Từ vựng cuối là các ký tự ban đầu cộng số phép gộp, và số phép gộp là siêu tham số chính (còn bước pre-tokenization bằng regex, chuẩn hóa văn bản và token đặc biệt). Encode một từ mới là áp các phép gộp **theo đúng thứ tự đã học**. **Byte-level BPE** (kiểu GPT-2) bắt đầu từ 256 byte UTF-8 nên không bao giờ có từ ngoài từ vựng; id mới bắt đầu từ 256.

% manim: BPEGhepCap

**Ví dụ giải mẫu 9.1 — BPE tay.** Bốn từ với tần số, ký hiệu cách nhau bằng dấu cách, `</w>` là hết từ: `l o w </w>` 5, `l o w e r </w>` 2, `n e w e s t </w>` 6, `w i d e s t </w>` 3. Chạy 10 phép gộp, mỗi lần lấy cặp có tần số cao nhất (hòa thì lấy cặp xuất hiện trước trong dict):

1. (e, s) tần số 9 → `es`. 2. (es, t) 9 → `est`. 3. (est, `</w>`) 9 → `est</w>`. 4. (l, o) 7 → `lo`. 5. (lo, w) 7 → `low`. 6. (n, e) 6. 7. (ne, w) 6. 8. (new, est`</w>`) 6. 9. (low, `</w>`) 5. 10. (w, i) 3.

- Ba phép đầu **hòa nhau ở 9** (các cặp (e, s), (s, t), (t, `</w>`) đều 9). Tie-break là lựa chọn thiết kế; cài đặt khác có thể ra từ điển khác.
- Encode từ lạ bằng các phép gộp theo thứ tự: `lowest` thành `low`, `est</w>` (2 token); `newer` thành `new`, `e`, `r`, `</w>` (4 token); `slowest` thành `s`, `low`, `est</w>`.
- Kiểm đơn vị byte-level (ví dụ trong README của minbpe): chuỗi `aaabdaaabac` với 3 phép gộp cho các id `[258, 100, 258, 97, 99]`; các phép gộp là (97, 97) → 256, (256, 97) → 257, (257, 98) → 258.
- **Bẫy:** chuỗi phép gộp trên là của bộ bốn từ ở trên. Chú thích Figure 1 của Sennrich cũng liệt kê các phép gộp (r · → r·, l o → lo, lo w → low, e r· → er·), nhưng học từ một từ điển khác, {low, lowest, newer, wider} (đã đối chiếu PDF ngày 04/11/2026). Hai chuỗi khác nhau vì khác dữ liệu, không có gì để "tái hiện" hay so khớp; đừng đặt chúng cạnh nhau. Tự chạy bộ từ trên (Lab 1).

**Byte và chuẩn hóa Unicode của tiếng Việt.** UTF-8 mã hóa a bằng 1 byte, các chữ như à, á, ă, â, ư, ơ, đ bằng 2 byte, ạ hoặc ệ bằng 3 byte. Cùng một chữ có hai cách viết: NFC (ký tự dựng sẵn) và NFD (chữ cái cộng dấu rời). Dữ liệu từ nhiều nguồn trộn hai dạng này là bẫy thật.

**Ví dụ giải mẫu 9.2 — byte, Unicode và công bằng token.** "Hà Nội": NFC 6 ký tự, 9 byte; NFD 9 ký tự, 12 byte. "tiếng Việt": 10 ký tự, 14 byte (NFC) so với 14 ký tự, 18 byte (NFD). "Hello world": 11 byte ở cả hai dạng.

- Số byte gần nhau không có nghĩa số token gần nhau; điều quyết định là những phép gộp nào tồn tại. Lab 1 dùng hai kho ngữ liệu nhỏ (khoảng 1 KB mỗi thứ tiếng, 16 câu song song) và BPE 150 phép gộp. Câu Anh 127 byte, câu Việt 178 byte. Tokenizer học trên tiếng Anh: câu Anh 58 token, câu Việt **167** token (1.07 byte mỗi token). Tokenizer học trên tiếng Việt: câu Anh 107 token, câu Việt **76** token (2.34 byte mỗi token). Tokenizer "sai ngôn ngữ" tốn gấp 2.2 lần cho câu Việt. Kho chỉ vài KB, một lần chạy: số này minh họa **hướng**, không phải số của GPT-2 hay Llama.
- Quy mô thật: Petrov và cộng sự (NeurIPS 2023) đo "tokenization premium": cùng nội dung, một số ngôn ngữ dài gấp nhiều lần tiếng Anh (có ngôn ngữ dài gấp cả chục lần tiếng Anh; số bảng chưa đối chiếu bản gốc). Hệ quả: chi phí, độ trễ và cửa sổ ngữ cảnh hữu hiệu đều kém hơn. PhoGPT dùng từ vựng 20 480 byte-level BPE huấn luyện riêng cho tiếng Việt.

**Perplexity.** Perplexity là lũy thừa của NLL trung bình theo token: exp của loss tính bằng nat, hay 2 mũ số bit trung bình. Đọc là "số lựa chọn đều hiệu dụng ở mỗi bước": phân phối đều trên V token cho đúng V. Cận dưới là exp của entropy rate của nguồn: loss kỳ vọng của mọi mô hình không thấp hơn entropy rate (trên một tập đánh giá hữu hạn có thể thấp hơn chút do nhiễu mẫu).

**Ví dụ giải mẫu 9.3 — perplexity và bẫy tokenizer.** Mô hình gán xác suất 0.5, 0.25, 0.125, 0.5 cho bốn token đúng.

- NLL từng token: 0.6931, 1.3863, 2.0794, 0.6931; trung bình 1.2130; perplexity exp(1.2130) = **3.3636** (bằng 2 mũ 1.75, tức 1.75 bit mỗi token). Phân phối đều trên 10 token cho 10; trên 50 257 token cho 50 257 (loss ban đầu ln 50 257 = 10.8249).
- Bẫy: một đoạn 400 byte, tổng NLL 69.3147 nat (= 100·ln 2). Nếu tokenizer chia thành 100 token, perplexity là 2.0000; 60 token: **3.1748**; 400 token (mức byte): 1.1892. Bits-per-byte như nhau ở cả ba: **0.25**. Perplexity theo token không so được giữa các tokenizer; dùng bits-per-byte.
- Cận dưới thật: chuỗi Markov bậc 1 (V = 24, seed 0) có entropy rate 2.1025 nat, nên perplexity không thể dưới xấp xỉ **8.186** (chuỗi bắt đầu từ trạng thái đều thì cận chính xác hơn là 8.216; không ảnh hưởng kết luận). Lab 2 huấn luyện TinyGPT 300 bước đạt loss 2.1557 (perplexity 8.634), cách entropy rate 0.053 nat.
- Cẩn thận token đệm (pad): tính cả pad làm perplexity "thấp hơn cận dưới".

**Giải mã.** Từ logit z, softmax có nhiệt độ T cho p(i) tỉ lệ với exp(zᵢ/T). T < 1 làm nhọn, T > 1 làm phẳng, T → 0 là greedy (argmax). **Top-k** giữ k token xác suất cao nhất. **Top-p** (nucleus; Holtzman và cộng sự, 2020) giữ tập nhỏ nhất có tổng xác suất ≥ p rồi chuẩn hóa lại. Thứ tự đúng: chia T trên logit, softmax, sắp giảm dần, cắt, chuẩn hóa. **Beam search** giữ b chuỗi tốt nhất; nó tìm chuỗi xác suất cao hơn greedy, nhưng Holtzman cho thấy văn bản người có xác suất trung bình trên token **thấp hơn nhiều** văn bản beam (perplexity 12.38 so với 1.48 ở beam b = 16, Table 1, theo tóm tắt công cụ): "tối đa hóa xác suất" không bằng "văn bản hay"; beam lặp và nhạt.

% manim: GiaiMaNhietDo

**Ví dụ giải mẫu 9.4 — nhiệt độ và top-p.** Logit z = (3, 2, 1, 0, −1).

- T = 1: p = (0.6364, 0.2341, 0.0861, 0.0317, 0.0117), entropy 1.000 nat. T = 0.5: (0.8647, 0.1170, 0.0158, 0.0021, 0.0003), entropy 0.458. T = 2: (0.4287, 0.2600, 0.1577, 0.0956, 0.0580), entropy 1.394.
- Top-p 0.9 ở T = 1: tích lũy 0.6364, 0.8705, 0.9567. Token thứ ba làm tích lũy **vượt** 0.9 và **phải được giữ**. Giữ 3 token, chuẩn hóa lại: (0.6652, 0.2447, 0.0900, 0, 0). Top-p 0.7 giữ 2 token (0.7311, 0.2689), cùng kết quả top-k = 2.
- Cùng p = 0.9: T = 0.5 giữ **2** token (tích lũy 0.8647, 0.9817); T = 2 giữ **4** token (0.4287, 0.6886, 0.8463, 0.9420). Top-p thích nghi theo độ nhọn của phân phối.
- Beam hơn greedy (đọc lướt): chuỗi A rồi x có xác suất 0.5·0.4 = 0.20 (greedy), còn B rồi x có 0.4·0.9 = **0.36**; beam b = 2 tìm ra nó.

**Vì sao ML cần cái này.**

- Mọi LLM đi qua đường ống token, logit, phân phối, mẫu; lỗi ở đây không báo lỗi, chỉ làm số sai hoặc văn bản kỳ lạ (bài săn bug cuối chương có năm lỗi như vậy).
- Từ vựng và chuẩn hóa quyết định chi phí và công bằng giữa ngôn ngữ; đây là chuyện của người làm tiếng Việt, không phải chi tiết kỹ thuật.
- Perplexity là chỉ số huấn luyện chính; hiểu cận dưới và bits-per-byte giúp bạn bắt được số liệu vô lý.

**Thực tế tính toán.** T = 0 không tự động đảm bảo kết quả tất định trong hệ thống thật (khác biệt phần cứng và batch có thể làm đổi argmax khi hai logit sát nhau). Tôi nhớ, chưa có nguồn đã mở.

## 9.2 Tiền huấn luyện và quy luật quy mô

Tiền huấn luyện là dự đoán token kế tiếp trên khối văn bản khổng lồ; chi phí ≈ 6ND FLOPs với N là số tham số (hoạt động) và D là số token (chương 8, dùng không dẫn xuất lại). Câu hỏi của mục này: có ngân sách tính toán C cố định thì nên chọn N và D thế nào?

**Luật lũy thừa của Kaplan** (2020): loss giảm theo luật lũy thừa khi tăng N, D hoặc C. Số mũ qua tóm tắt công cụ: αN ≈ 0.076 (N đếm **không gồm embedding**), αD ≈ 0.095. Nhân đôi N giảm loss một hệ số 2⁻⁰·⁰⁷⁶ = 0.9487; nhân 10 lần giảm 16.1%. Khi ngân sách tăng, Kaplan khuyên tăng N nhanh hơn nhiều so với D (N tăng theo C mũ 0.73, D theo C mũ 0.27): "mô hình rất lớn, dữ liệu vừa phải".

**Chinchilla** (Hoffmann và cộng sự, 2022; hơn 400 mô hình từ 70M tới hơn 16B tham số): kết luận ngược lại, N và D nên tăng **cùng tỉ lệ**. Ba cách ước lượng cho số mũ N tối ưu tăng theo C: 0.50, 0.49, 0.46 (Kaplan: 0.73; Table 2, đã đối chiếu PDF ngày 04/11/2026). Cách 3 fit hàm sau:

$$
L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}
$$

Hằng số công bố (Eq. 10, đã đối chiếu PDF ngày 04/11/2026): E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28. Theo bài Epoch (qua tóm tắt công cụ), số mũ chưa làm tròn là α = 0.339, β = 0.285; ví dụ 9.6 cho thấy kết quả nhạy với phép làm tròn này, và cho một phép kiểm chéo với chính paper. Phép thử: Chinchilla 70B trên 1.4T token thắng Gopher 280B, GPT-3 và vài mô hình khác trên nhiều tác vụ. "Khoảng 20 token mỗi tham số" là **cách người ta tóm tắt** (1.4T chia 70B), không phải hằng số được chứng minh.

**Dẫn xuất tỉ lệ tối ưu.** Cực tiểu A·N⁻ᵅ + B·D⁻ᵝ với ràng buộc ND = C/6. Thế D = C/(6N), đạo hàm theo N cho αA·N⁻ᵅ = βB·D⁻ᵝ, nên N⁻ᵅ ∝ D⁻ᵝ, tức N tối ưu tỉ lệ với C mũ β/(α + β). Với (0.34, 0.28): β/(α + β) = 0.4516, gần 0.46 của paper (bài tập ★★★ 11).

**Ví dụ giải mẫu 9.5 — 6ND và tỉ lệ token mỗi tham số.**

| Mô hình | N | D | D/N | 6ND (FLOPs) |
| --- | --- | --- | --- | --- |
| GPT-3 | 175B | 300B | 1.71 | 3.15·10²³ |
| Gopher | 280B | 300B | 1.07 | 5.04·10²³ |
| Chinchilla | 70B | 1.4T | 20.0 | 5.88·10²³ |
| Llama 3 405B | 405B | 15.6T | 38.5 | 3.79·10²⁵ |

Llama 3 báo ngân sách 3.8·10²⁵ FLOPs, khớp 6ND tới 0.2% (số theo tóm tắt công cụ). Paper Chinchilla nói đếm FLOPs chi tiết lệch 1 đến 10% so với 6ND.

% manim: TokenMoiThamSo

**Ví dụ giải mẫu 9.6 — fit tham số của Chinchilla cho ra gì?** Tại ngân sách của Chinchilla C = 5.88·10²³, cực tiểu hóa L(N, C/6N) theo N bằng tìm số học:

- **Cách 3 của paper tự dự đoán mô hình nhỏ hơn cách 1 và 2.** Chú thích Figure 4 của Chinchilla ghi: ở ngân sách của Gopher, cách 3 cho cỡ tối ưu khoảng 40B tham số, còn cách 1 cho 67B (Table 3: 5.76·10²³ FLOPs, 1.5T token) và cách 2 cho 63B. Tính lại từ Eq. 10 với số mũ đủ chữ số (0.339, 0.285): C = 5.76·10²³ cho N* = 40.8B, D* = 2.36T, tỉ lệ **57.8**, khớp "khoảng 40B" của paper; tại C = 5.88·10²³ của Chinchilla: N* = 41.1B, D* = 2.38T, tỉ lệ **57.9**. Với số mũ làm tròn (0.34, 0.28) ra N* = 32.5B, D* = 3.0T, tỉ lệ **92.8**: con số 93 chỉ là hệ quả của việc làm tròn số mũ, không phải số paper in. Vậy fit công bố **có** tái hiện kết luận của chính nó (cách 3 dự đoán ≈ 40B, lệch cách 1 và 2 mà paper đã nêu); nó chỉ không trùng 70B và 1.4T của mô hình được huấn luyện thật.
- **Fit lại của Epoch AI** (Besiroglu và cộng sự, 2024; E 1.8172, A 482.01, B 2085.43, α 0.3478, β 0.3658, qua tóm tắt): N* = **73.0B**, D* = **1.34T**, tỉ lệ **18.4**, gần trùng điểm 70B/1.4T.
- Số mũ phân bổ β/(α + β): 0.4516 (công bố), 0.5126 (Epoch).
- Trung thực: dữ liệu Epoch tái tạo từ hình nên có sai số; chương không kết luận ai "đúng". Epoch phê bình cách 3 là fit thiếu khớp, với khoảng tin cậy hẹp bất thường (theo tóm tắt công cụ [cần kiểm chứng]). Quy tắc 20 token mỗi tham số đứng vững nhờ cách 1, cách 2 và fit lại của Epoch, còn cách 3 lệch về phía mô hình nhỏ, nhiều token hơn. Epoch nói fit công bố hàm ý cỡ 70 token mỗi tham số. Tôi tính được 57.9 (số mũ đủ chữ số) đến 92.8 (làm tròn) tại C = 5.88·10²³; với số mũ đủ chữ số, tỉ lệ là 49.7 ở C = 10²³, 60.6 ở 10²⁴, 74.0 ở 10²⁵, nên con số 70 nằm trong dải đó tùy ngân sách.

**Vì sao Kaplan và Chinchilla khác nhau.** Pearce và Song (2024; chỉ đọc abstract) cho rằng phần lớn do Kaplan đếm tham số không gồm embedding và phân tích ở quy mô nhỏ; mô phỏng lại Chinchilla trong điều kiện ấy cho số mũ gần Kaplan. Vì sao embedding quan trọng ở quy mô nhỏ: với số tham số 12·L·d² + V·d (V = 50 257, cấu hình họ GPT-3 theo trí nhớ, chưa kiểm), tỉ trọng embedding là 31.2% ở cỡ 125M, 14.6% ở 350M, 7.9% ở 1.3B, 3.1% ở 6.7B, 0.4% ở 175B. Độ dốc của log N tổng theo log N không-embedding là 0.89 ở dải nhỏ và 0.99 ở hai cỡ lớn nhất, nên đường cong cong ở quy mô nhỏ; tại 125M, 6ND với tham số tổng đắt gấp 1.45 lần.

**Vượt Chinchilla.** (1) Tính cả chi phí suy luận: Sardana và cộng sự (2024): nếu kỳ vọng cỡ một tỉ yêu cầu phục vụ trở lên thì nên huấn luyện mô hình nhỏ hơn, lâu hơn (chất lượng còn cải thiện tới 10 000 token mỗi tham số). Llama 3 405B (38.5) là một ví dụ. (2) Dữ liệu hữu hạn: Muennighoff và cộng sự (2023): lặp dữ liệu tới khoảng 4 epoch gần như không hại, sau đó lợi ích giảm về 0. (3) MoE (mixture of experts): N trong 6ND phải là tham số **hoạt động**; dùng tham số tổng thì quá lớn.

**Giới hạn.** Ngoại suy từ mô hình nhỏ nhạy cảm (Lab 3: fit chỉ trên vùng nhỏ cho tỉ lệ 13.1 thay vì 18.4). Loss không bằng năng lực tác vụ. Hằng số phụ thuộc tokenizer và dữ liệu. Luật không dự đoán "năng lực nổi lên"; test-time compute (cho mô hình suy nghĩ lâu hơn) là trục quy mô thứ hai (chỉ nêu từ khóa).

**Vì sao ML cần cái này.**

- Mọi quyết định "huấn luyện mô hình bao lớn, bao lâu" nằm ở đây; trả lời được định lượng là dấu hiệu bạn hiểu chi phí, không chỉ kiến trúc.
- Phê bình của Epoch dạy thói quen đúng: tự tính lại số công bố trước khi tin.
- "20 token mỗi tham số" mà không biết nguồn gốc và giới hạn là câu trả lời yếu.

**Thực tế tính toán.** Lab 3 fit trên dữ liệu tổng hợp có nhiễu: khi tôi thử lưới 4 cỡ mô hình × 4 cỡ dữ liệu trên CPU với số bước cố định (400 bước mỗi ô), mọi ô thấy cùng một số token, nên "cỡ dữ liệu" không đổi D thực tế và loss chỉ phụ thuộc cỡ mô hình. Muốn đổi D phải đổi số bước (hoặc cỡ batch); và scaling law cần mỗi ô huấn luyện đủ lâu để so sánh có nghĩa.

## 9.3 Hậu huấn luyện: SFT, RLHF, DPO, RLVR

Mô hình sau tiền huấn luyện chỉ biết tiếp văn bản. Hậu huấn luyện dạy nó làm theo chỉ dẫn và hợp ý người. Đường ống InstructGPT (Ouyang và cộng sự, 2022): (1) SFT (supervised fine-tuning) trên minh họa của người; (2) huấn luyện reward model (RM) từ các cặp so sánh; (3) tối ưu chính sách chống lại RM bằng PPO. Kết quả đầu bảng: đầu ra của InstructGPT 1.3B được người ưa hơn GPT-3 175B. PPO làm tụt một số bộ đánh giá NLP công khai ("alignment tax"); trộn thêm cập nhật tiền huấn luyện (PPO-ptx) giảm phần lớn.

**SFT** là cùng loss token kế tiếp, nhưng có bốn chi tiết dễ sai: (a) chỉ tính loss trên token **đáp án**, che prompt (nhãn −100); nhưng trên bài đồ chơi (tôi thử ngoài lab, không có trong notebook) che hay không che đều cho exact-match 1.000 vì bài quá dễ, nên đừng coi che prompt là yếu tố quyết định nếu chưa đo trên dữ liệu thật; (b) thêm token kết thúc (EOS) vào mục tiêu, nếu không mô hình không bao giờ dừng; (c) khuôn hội thoại (chat template) lúc huấn luyện và lúc chạy phải giống nhau; (d) trung bình loss theo token hay theo chuỗi. Với chuỗi X 10 token đáp án loss 2.0 và chuỗi Y 100 token loss 1.0: theo token là (10·2 + 100·1)/110 = 1.0909, theo chuỗi là (2 + 1)/2 = 1.5. Theo token để chuỗi dài chi phối gradient. LIMA (Zhou và cộng sự, 2023) tinh chỉnh trên 1 000 cặp và nêu **giả thuyết** căn chỉnh bề mặt: kiến thức học ở tiền huấn luyện, căn chỉnh chủ yếu dạy định dạng.

**Reward model và bài toán có KL.** RM theo Bradley-Terry: xác suất đáp án thắng (ký hiệu yw) hơn đáp án thua (yl) là σ(rw − rl), với rw, rl là phần thưởng của chúng, loss là −ln σ(rw − rl). RLHF cực đại phần thưởng kỳ vọng trừ phạt lệch khỏi chính sách tham chiếu πref:

$$
\max_{\pi}\ \mathbb{E}_{y\sim\pi}[r(x,y)]-\beta\,\mathrm{KL}\big(\pi\,\|\,\pi_{\mathrm{ref}}\big)\quad\Longrightarrow\quad \pi^{*}(y\mid x)=\frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y\mid x)\,e^{r(x,y)/\beta}
$$

β nhỏ thì bám reward, đi xa tham chiếu. **Over-optimization** (Gao và cộng sự, 2022; abstract): tối ưu quá mức chống reward proxy làm điểm "reward vàng" giảm; đây là Goodhart.

**DPO** (Rafailov và cộng sự, 2023): từ nghiệm đóng suy ra r = β·ln(π/πref) + β·ln Z(x). Z chỉ phụ thuộc x nên triệt tiêu trong hiệu rw − rl, và không cần RM riêng:

$$
\mathcal{L}_{\mathrm{DPO}}=-\ln\sigma\Big(\beta\big[\ln\tfrac{\pi_\theta(y_w)}{\pi_{\mathrm{ref}}(y_w)}-\ln\tfrac{\pi_\theta(y_l)}{\pi_{\mathrm{ref}}(y_l)}\big]\Big),\qquad \nabla\mathcal{L}=-\beta\,\sigma(\hat r_l-\hat r_w)\big(\nabla\ln\pi_\theta(y_w)-\nabla\ln\pi_\theta(y_l)\big)
$$

Ở đây r̂ = β·ln(πθ/πref) là phần thưởng ngầm. Trọng số σ(r̂l − r̂w) lớn khi mô hình ngầm xếp **sai** thứ tự. Llama 3 chọn DPO thay PPO ở hậu huấn luyện (lr 10⁻⁵, β = 0.1, thêm số hạng NLL hệ số 0.2; theo tóm tắt công cụ).

**Ví dụ giải mẫu 9.7 — reward model, DPO tay, likelihood displacement.**

(a) RM: rw = 1.2, rl = 0.4, hiệu 0.8, P(thắng) = σ(0.8) = 0.6900, loss = −ln 0.6900 = **0.3711**. Hiệu 0: 0.6931 (đoán mò). Hiệu 2: 0.1269. Hiệu −1 (sai thứ tự): 1.3133. Xếp hạng K = 4 đáp án cho C(4, 2) = 6 cặp.

(b) DPO, β = 0.1: ln πθ(yw) = −12, ln πref(yw) = −14, ln πθ(yl) = −15, ln πref(yl) = −13. Phần thưởng ngầm r̂w = 0.1·(−12 + 14) = 0.2, r̂l = 0.1·(−15 + 13) = −0.2, z = 0.4. Loss = −ln σ(0.4) = **0.5130**; trọng số gradient σ(−0.4) = **0.4013**, hệ số trước (∇ln πθ(yw) − ∇ln πθ(yl)) là −β·0.4013 = −0.0401. Khi πθ = πref thì z = 0, loss = ln 2 = **0.6931** (phép kiểm khởi đầu, như ln V ở chương 6). Autograd khớp công thức gradient tới sai số 0 (Lab 4).

(c) **Likelihood displacement.** ln πθ(yw) giảm từ −14 xuống −16, ln πθ(yl) giảm từ −13 xuống −20. Khi đó z = 0.1·[(−16 + 14) − (−20 + 13)] = 0.5 > 0, loss 0.4741, thấp hơn loss lúc khởi đầu 0.6931 (πθ = πref, z = 0) **dù xác suất của đáp án được chọn đã giảm**. (Đừng so với 0.5130 của phần (b): đó là kịch bản khác.) Loss DPO chỉ thấy hiệu. Razin và cộng sự (2024; abstract) báo hiện tượng này trên mô hình thật (ví dụ trong abstract: tỉ lệ từ chối của một mô hình Llama tụt từ 74.4% xuống 33.4%).

(d) Nghiệm đóng với πref = (0.5, 0.3, 0.2), r = (0, 1, 2): β = 10 cho π* = (0.4648, 0.3082, 0.2271), E[r] = 0.76, KL 0.003; β = 1: (0.1790, 0.2919, 0.5291), E[r] = 1.35, KL 0.32; β = 0.5: (0.0367, 0.1626, 0.8008), KL 0.92; β = 0.1: (0, 0.0001, 0.9999), KL 1.61. Lab 4: DPO trên 20 000 cặp ưa thích sinh từ reward vàng hội tụ về π* (TV 0.030 trong lần chạy của notebook).

% manim: DPOHaiLogTiSo

**RLVR và GRPO.** Với bài toán kiểm chứng được (đáp án toán đúng, test mã qua, ràng buộc định dạng), thay RM bằng **bộ kiểm chứng**: RLVR (Tülu 3 đặt tên, thứ tự SFT, DPO, RLVR). **PPO** cắt tỉ số π/πcũ trong [1 − ε, 1 + ε] để bước cập nhật không quá xa, và cần một critic (mô hình giá trị). **GRPO** (DeepSeekMath) bỏ critic: với mỗi câu hỏi lấy G đầu ra, advantage là phần thưởng chuẩn hóa trong nhóm, và KL vào thẳng loss bằng ước lượng k3 = u − ln u − 1 với u = πref/π (luôn ≥ 0).

**Mục tiêu PPO và GRPO.** Tôi tự viết lại theo ký hiệu của chương; **chưa đối chiếu Eq. 3 của DeepSeekMath** (công cụ tóm tắt báo bị vỡ định dạng), nên đừng chép số hiệu phương trình khi chưa mở bản gốc. Prompt q lấy từ dữ liệu, các đầu ra o lấy từ πcũ, ρₜ là tỉ số xác suất token t giữa πθ và πcũ (chỉ số old trong công thức), Âₜ là advantage, ε là ngưỡng cắt:

$$
\mathcal{J}_{\mathrm{PPO}}=\mathbb{E}\Big[\tfrac{1}{|o|}\textstyle\sum_{t}\min\big(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\big)\Big],\qquad \rho_t=\frac{\pi_\theta(o_t\mid q,o_{<t})}{\pi_{\mathrm{old}}(o_t\mid q,o_{<t})}
$$

$$
\mathcal{J}_{\mathrm{GRPO}}=\mathbb{E}\Big[\tfrac{1}{G}\textstyle\sum_{i=1}^{G}\tfrac{1}{|o_i|}\sum_{t}\Big(\min\big(\rho_{i,t}\hat A_i,\ \mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i\big)-\beta\,\mathrm{k3}_{i,t}\Big)\Big],\quad \hat A_i=\frac{r_i-\mathrm{mean}(r_{1:G})}{\mathrm{std}(r_{1:G})},\quad \mathrm{k3}=u-\ln u-1,\ u=\frac{\pi_{\mathrm{ref}}}{\pi_\theta}
$$

Khác biệt: Âₜ của PPO theo từng token, tính bằng GAE từ critic, còn Âᵢ của GRPO là một số cho cả đầu ra, lấy từ nhóm G đầu ra của cùng prompt; PPO thường đưa KL vào phần thưởng (như InstructGPT), GRPO đưa k3 thẳng vào mục tiêu. k3 là ước lượng không chệch của KL(π ‖ πref) khi mẫu lấy từ π **và π phủ πref** (mọi đầu ra có πref > 0 cũng có π > 0; luôn đúng với softmax logit hữu hạn): u = πref/π có 𝔼[u] = ∑ π·(πref/π) = ∑ πref = 1, tổng lấy trên các đầu ra có π > 0 nên chỉ bằng 1 khi π phủ πref (nếu không thì 𝔼[u] < 1 và ước lượng chệch). Khi đó 𝔼[u − 1 − ln u] = −𝔼[ln u] = 𝔼[ln(π/πref)] = KL(π ‖ πref). (Điều kiện ngược lại, πref phủ π, là điều kiện để KL hữu hạn, không phải điều kiện của bước này.) **Lưu ý cho GRPO:** các đầu ra lấy từ πcũ chứ không từ πθ, nên k3 chỉ không chệch với KL(πθ ‖ πref) khi πcũ = πθ (bước đầu tiên của mỗi vòng cập nhật). Ngoài ra, đạo hàm của k3 theo θ với mẫu cố định không bằng đạo hàm của KL(πθ ‖ πref); có phân tích gần đây bàn việc này [cần kiểm chứng tên paper], nên đừng coi k3 trong loss là "gradient của KL".

**Ví dụ giải mẫu 9.8 — PPO cắt, k3, advantage nhóm.**

- PPO cắt, ε = 0.2, advantage A = +0.9, tỉ số 1.3: min(1.3·0.9, 1.2·0.9) = min(1.17, 1.08) = **1.08** (không thưởng thêm cho việc đã tăng quá 20%). A = −0.9, tỉ số 1.3: −1.17 (không cắt; với A âm, min chọn giá trị tồi hơn). A = −0.9, tỉ số 0.7: min(−0.63, −0.72) = **−0.72**.
- k3: u = 1 cho 0; u = 0.8 cho **0.0231**; u = 0.5 cho 0.1931; u = 2 cho 0.3069.
- Advantage nhóm: G = 8, phần thưởng (1, 0, 0, 1, 0, 0, 0, 1): trung bình 0.375, độ lệch chuẩn (chia G) 0.4841. Đáp án đúng nhận **+1.2910**, sai nhận **−0.7746**; tổng bằng 0. Chia G − 1 thì +1.2076 và −0.7246: chi tiết này khác nhau giữa các cài đặt.
- Nhóm toàn đúng hoặc toàn sai: độ lệch chuẩn 0 nên mọi advantage bằng 0, **không có gradient**. Xác suất cả nhóm cùng kết quả khi tỉ lệ đúng p, G = 8: p = 0.2 cho 0.1678 (0.8⁸); p = 0.9 cho 0.4305; p = 0.99 cho 0.9227. Với G = 64, p = 0.99: 0.5256. DAPO bỏ các nhóm này (Dynamic Sampling).
- Bandit đồ chơi (Lab 4): 6 đáp án, chỉ đáp án 3 đúng, xác suất ban đầu 0.0264; sau 50 bước p(đúng) = 0.968, và khoảng 260 trên 300 nhóm không có tín hiệu vì gần như luôn đúng hết.

% manim: GRPONhom

**Biến thể và phản biện.** DeepSeek-R1 (R1-Zero) huấn luyện RL thuần với phần thưởng luật (đúng đáp án và định dạng); paper ghi AIME 2024 pass@1 từ 15.6% lên 77.9% (bản đầu ghi 71.0%: luôn nói theo bản nào). DAPO bổ sung bốn kỹ thuật (Clip-Higher, Dynamic Sampling, loss theo token, phạt mềm câu quá dài); GSPO dùng tỉ số mức chuỗi. Cả hai theo tóm tắt. **Phản biện quan trọng:** Yue và cộng sự (2025; abstract) thấy mô hình RLVR hơn mô hình nền ở k nhỏ nhưng **kém hơn ở k lớn**, và kết luận RLVR chủ yếu làm việc lấy mẫu ra lời giải mà mô hình nền vốn có đáng tin hơn, chứ chưa thêm năng lực suy luận mới; chưng cất thì có thể. (pass@k ở mục 9.5.)

**Vì sao ML cần cái này.**

- Mọi mô hình trò chuyện đều đi qua SFT và một dạng căn chỉnh; loss DPO và advantage nhóm là câu hỏi phỏng vấn gần như chắc chắn.
- Hiểu nghiệm đóng KL cho bạn thấy ba thuật toán (PPO, DPO, GRPO) cùng giải một bài toán; khác nhau ở cách ước lượng và chi phí.
- Likelihood displacement và over-optimization dạy rằng loss giảm không có nghĩa là hành vi tốt hơn.

**Thực tế tính toán.** Thuật toán RL cụ thể là phần dễ lỗi thời nhất; bền hơn: KL chính quy hóa, advantage theo nhóm, phần thưởng kiểm chứng được.

## 9.4 Tinh chỉnh hiệu quả tham số và lượng tử hóa

Tinh chỉnh toàn phần mô hình 7B không vừa laptop RTX 4050 6 GB; LoRA và lượng tử hóa khiến việc đó khả thi. Ngân sách bộ nhớ trước: Adam với độ chính xác hỗn hợp cần 2Ψ byte (trọng số fp16) + 2Ψ (gradient fp16) + 12Ψ (bản fp32 của trọng số, động lượng m, phương sai v) = **16Ψ byte** (ZeRO, Rajbhandari và cộng sự, 2019), chưa kể activation. Suy luận chỉ cần trọng số: 2Ψ (fp16), Ψ (int8), khoảng 0.516Ψ (4 bit kèm hằng số khối).

**LoRA** (Hu và cộng sự, 2021) đóng băng W₀ ∈ ℝᵈˣᵏ và học cập nhật hạng thấp ΔW = BA, B ∈ ℝᵈˣʳ, A ∈ ℝʳˣᵏ, r nhỏ hơn nhiều min(d, k):

$$
h=W_0x+\frac{\alpha}{r}\,BAx,\qquad A\sim\mathcal{N}(0,\sigma^2),\quad B=0
$$

B = 0 nên ΔW = 0 lúc đầu: mô hình ban đầu không đổi (Lab 5 kiểm). Sau huấn luyện gộp W₀ + (α/r)BA, **không thêm độ trễ suy luận**. Paper áp lên các ma trận chiếu query và value của GPT-3 175B và báo giảm tham số huấn luyện khoảng 10 000 lần, bộ nhớ GPU khoảng 3 lần (theo tóm tắt công cụ). Hiện nay thường áp lên mọi lớp tuyến tính; r và α là siêu tham số.

**Ví dụ giải mẫu 9.9 — đếm tham số LoRA.** Ma trận 4096 × 4096 có 16 777 216 tham số; LoRA r = 8 cần 8·(4096 + 4096) = **65 536**, bằng 2r/d = 0.39%. Llama-2-7B theo công thức (d = 4096, 32 khối, ẩn FFN 11 008, từ vựng 32 000, không dùng chung embedding) có **6 738 415 616** tham số (khớp chương 8; con số gốc tôi nhớ, chưa đối chiếu nguồn).

- LoRA r = 8 trên {q, v}: **4 194 304** (0.062%). Trên {q, k, v, o}: 8 388 608 (0.124%). Trên mọi lớp tuyến tính (q, k, v, o, gate, up, down): **19 988 480** (0.297%). Với r = 16 mọi lớp: 39 976 960.

**Ví dụ giải mẫu 9.10 — ngân sách bộ nhớ cho 7B và laptop 6 GB.** Ψ = 6.738·10⁹ (tính từ công thức, **chưa đo**).

- Tinh chỉnh toàn phần: 16Ψ byte = **100.4 GiB**, chưa activation.
- Chỉ trọng số suy luận: bf16 **12.55 GiB**; int8 6.28 GiB; NF4 kèm double quantization (4.127 bit mỗi tham số) **3.24 GiB**.
- LoRA r = 16 mọi lớp: 39 976 960 tham số có gradient; adapter, gradient, m, v ở fp32 là 16 byte mỗi tham số, tức 0.60 GiB; cộng trọng số nền đóng băng: khoảng 13.1 GiB (bf16) hoặc khoảng 3.8 GiB (NF4), chưa tính activation và KV cache.
- Thẻ 6 GiB, chỉ tính trọng số: chứa được 3.2 tỉ tham số ở 16 bit, 6.4 tỉ ở 8 bit, 12.5 tỉ ở 4.127 bit. Vì vậy 7B ở 4 bit có thể vừa trọng số, nhưng activation và optimizer **quyết định** có chạy được hay không; tôi chưa đo trên GPU, nên chương không hứa.

**Lượng tử hóa.** Absmax int8: mã nguyên = làm tròn(127·x/max|x|). Một giá trị ngoại lệ làm bước lượng tử to ra và nghiền phần còn lại. Cách chữa: lượng tử **theo khối** (mỗi khối 64 hay 256 phần tử có hằng số riêng). LLM.int8() (Dettmers và cộng sự, 2022) còn tách chiều đặc trưng ngoại lệ (độ lớn ≥ 6.0) tính bằng 16 bit, phần còn lại 8 bit; ngoại lệ xuất hiện đột ngột giữa cỡ 6B và 6.7B. **QLoRA** (Dettmers và cộng sự, 2023) ghép: (1) **NF4**, kiểu 4 bit có 16 mức là lượng tử của phân phối chuẩn N(0, 1) (có mức 0 chính xác); (2) **double quantization**: lượng tử lại các hằng số khối, đưa chi phí hằng số từ 32/64 = 0.5 bit xuống 8/64 + 32/(64·256) = 0.127 bit mỗi tham số; (3) paged optimizers (chuyển trạng thái optimizer sang CPU khi hết VRAM). Trọng số nền đóng băng ở NF4, adapter LoRA nhận gradient ở độ chính xác cao hơn. GPTQ, AWQ, QAT chỉ nêu tên.

% manim: LoRAHaiMaTran

**Ví dụ giải mẫu 9.11 — absmax, ngoại lệ, NF4.** x = (0.1, −0.5, 0.3, −0.2, 0.05, 0.4, −0.3, 0.2), int8 absmax (hệ số 127/0.5 = 254): mã (25, −127, 76, −51, 13, 102, −76, 51), sai số lớn nhất 0.0016.

- Thay −0.2 bằng ngoại lệ **12.0**: hệ số 127/12 = 10.58, bước lượng tử 12/127 = **0.0945**; các giá trị còn lại thành (1, −5, 3, 127, 1, 4, −3, 2); 0.05 và 0.1 cùng thành 0.0945; sai số lớn nhất của phần tử không ngoại lệ **0.0445** (gấp 28 lần). Lượng tử theo khối 4 phần tử: ngoại lệ chỉ ảnh hưởng khối chứa nó: khối chứa ngoại lệ vẫn sai tới 0.0276 (với −0.5), còn khối kia (0.05, 0.4, −0.3, 0.2) có mã (16, 127, −95, 64) và sai tối đa 0.0016 như khi không có ngoại lệ.
- 16 mức NF4: −1, −0.6962, −0.5251, −0.3949, −0.2844, −0.1848, −0.0911, 0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1 (đọc từ mã nguồn bitsandbytes).
- Trên 1 048 576 trọng số Gauss N(0, 0.02²) giả lập, lượng tử theo khối: khối 64 có RMSE NF4 0.001839 so với int4 đối xứng 0.002154 (giảm 14.6%); khối 256: 0.001976 so với 0.002529; khối 4096: 0.002203 so với 0.003150 (giảm 30.1%).
- **Trên mô hình đồ chơi** (Lab 5), loss kiểm sau lượng tử hóa khối 16 (nền 2.1557): int8 2.1557, int4 theo tensor 2.1638, int4 khối 16 2.1577, NF4 khối 16 2.1582, int2 khối 16 2.2967. NF4 **không** hơn int4 ở đây dù hơn rõ trên Gauss lý tưởng: chương không hứa bạn thấy NF4 thắng.

% manim: NgoaiLeLuongTu

**Phản biện của LoRA.** Biderman và cộng sự (2024; abstract) thấy trên mã và toán, LoRA hạng thấp thường **kém hơn** tinh chỉnh toàn phần nhưng **quên ít hơn**. Lab 5 **không** tái hiện phần "quên ít hơn" ở cùng mức học.

**Vì sao ML cần cái này.**

- Nó quyết định bạn chạy được thí nghiệm nào trên phần cứng của mình.
- LoRA là ví dụ trực tiếp của xấp xỉ hạng thấp (chương 1); QLoRA là ví dụ của lượng tử hóa có cấu trúc.
- Lượng tử hóa là chỗ lỗi im lặng: một ngoại lệ làm hỏng cả tensor mà không có cảnh báo.

**Thực tế tính toán.** Máy build không có GPU: mọi con số bộ nhớ ở mục này tính từ công thức, chưa đo.

## 9.5 Đánh giá LLM và hệ thống quanh mô hình

Mục này chỉ dạy phần riêng của LLM; phần RAG và agent chỉ cần đọc. Khoảng tin cậy, kiểm định và rò rỉ dữ liệu nói chung ở chương 11 (mục 11.3, 11.4); phát hiện nhiễm bằng n-gram ở ví dụ 11.9.

**Họ phương pháp và cách chúng hỏng.** (a) Nội tại: perplexity, bits-per-byte (hỏng khi so giữa tokenizer). (b) Trắc nghiệm: MMLU (57 tác vụ; GPT-3 175B few-shot đạt 43.9%, ngẫu nhiên 25%). (c) Mã: HumanEval (164 bài) và pass@k. (d) Toán và suy luận. (e) Ưa thích của người (Chatbot Arena; tôi chỉ khớp tiêu đề, chưa đọc). (f) LLM làm giám khảo: sai lệch vị trí và độ dài; tôi nhớ nhưng chưa đo và chưa mở nguồn, nên chỉ coi là câu hỏi mở.

**pass@k không chệch** (Chen và cộng sự, 2021). Sinh n ≥ k mẫu mỗi bài, đếm c mẫu đúng; ước lượng
$$
\widehat{\text{pass@}k}=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}
$$
Công thức ngây thơ 1 − (1 − c/n)ᵏ chệch xuống. Trường hợp n − c < k thì kết quả bằng 1. Hệ số nhị thức tràn số khi n lớn; Chen và cộng sự dùng dạng tích ổn định số, 1 − ∏ᵢ(1 − k/i) với i chạy từ n − c + 1 tới n.

**Ví dụ giải mẫu 9.12 — pass@k.** n = 20, c = 3: pass@1 = 0.15; pass@2 = 0.2842; pass@5 = 1 − C(17, 5)/C(20, 5) = 1 − 6188/15504 = **0.6009**; pass@10 = 0.8947; pass@20 = 1. Ngây thơ: pass@5 = 0.5563; pass@10 = 0.8031; pass@20 = 0.9612. Biên: n = 5, c = 1, k = 5 cho pass@5 = 1 (vì n − c < k), ngây thơ cho 0.6723. Liên hệ RLVR: pass@1 và pass@k trả lời hai câu hỏi khác nhau; mô hình có thể hơn ở k = 1 mà kém ở k lớn (Yue và cộng sự).

**Sai số của benchmark nhỏ.** Nửa độ rộng khoảng 95% là 1.96·√(p(1 − p)/N).

**Ví dụ giải mẫu 9.13 — sai số và độ phân giải.** MMLU, p = 0.88, N ≈ 14 000 (paper v1 ghi 14 080 câu; bộ phổ biến khác chút, nên nói "khoảng"): ±0.54 điểm. HumanEval, p = 0.80, N = 164: **±6.12 điểm**. GPQA Diamond, p = 0.65, N = 198: ±6.64. Để phát hiện chênh 2 điểm ở p ≈ 0.88 giữa hai mô hình độc lập (mức 5%, công suất 80%) cần khoảng 4 100 câu; chênh 1 điểm cần khoảng 16 600. MMLU có khoảng 14 000 câu, nhiều hơn 4 100, nên về sai số mẫu thì chênh 2 điểm trên MMLU phát hiện được; trên HumanEval (±6 điểm) thì không. Dao động do chọn prompt, few-shot, cách tách đáp án là một nguồn ồn khác, cần đo riêng [cần nguồn]. Kiểm định cặp đôi và bootstrap ở chương 11.

**Nhiễm benchmark (riêng LLM).** GPT-3 (Brown và cộng sự, 2020) lập tập "sạch" bằng cách bỏ mẫu có 13-gram trùng dữ liệu huấn luyện, và paper tự thừa nhận một lỗi lọc làm sót chồng lấn, không đủ ngân sách huấn luyện lại. Khảo sát (arXiv 2406.04244, đọc tóm tắt) nêu rằng không loại bỏ hoàn toàn được; đối phó gồm benchmark riêng tư, động (LiveCodeBench, chỉ khớp tiêu đề) và làm lại dữ liệu. **Ca điển hình 2026:** OpenAI đăng ngày 23/2/2026 ngừng báo cáo SWE-bench Verified vì mọi mô hình tiền tuyến họ thử đều tái hiện được bản vá gốc hoặc chi tiết đề bài, và kiểm 138 bài khó thấy 59.4% có test hoặc mô tả hỏng (theo tóm tắt công cụ); họ khuyên dùng SWE-bench Pro. Bài học: benchmark có vòng đời. Đừng viết "benchmark X còn dùng được", hãy viết cách kiểm.

**RAG** (retrieval-augmented generation; Lewis và cộng sự, 2020): mô hình sinh được bổ sung bộ truy hồi lấy đoạn văn từ kho ngoài. Chỉ dạy ý bền: chia đoạn (chunk) và ngân sách ngữ cảnh; đo truy hồi riêng (recall@k, MRR) trước khi đo sinh; hai kiểu lỗi "truy hồi đúng, sinh sai" và "truy hồi sai, bịa"; RAG khi kiến thức đổi nhanh hoặc cần trích nguồn, tinh chỉnh khi cần đổi hành vi hay định dạng. Nối trực tiếp với dự án Graph-RAG của học phần NLP.

**Agent** (kiểu ReAct, tôi chỉ khớp tiêu đề): vòng lặp suy nghĩ, hành động (gọi công cụ), quan sát. Độ tin cậy **nhân** theo số bước.

**Số cho RAG và agent (chỉ đọc, không cần làm lại).** Tám truy vấn, vị trí tài liệu đúng trong danh sách truy hồi: 1, 3, 2, 1, 7, 1, 2, 12. recall@1 = **0.375**; @3 = 0.75; @5 = 0.75; @10 = 0.875. MRR = (1 + 1/3 + 1/2 + 1 + 1/7 + 1 + 1/2 + 1/12)/8 = **0.5699**. Ngân sách ngữ cảnh: chunk 512 token × 5 đoạn = 2 560 token thêm vào prompt; 512 × 10 = 5 120. Agent: mỗi bước thành công 0.95 thì 10 bước còn 0.95¹⁰ = **0.5987**, 20 bước 0.3585, 50 bước 0.0769; mỗi bước 0.99 thì 50 bước 0.6050.

**Vì sao ML cần cái này.**

- Biết khi nào con số benchmark vô nghĩa (sai số, nhiễm, hết hạn) là kỹ năng nghiên cứu cốt lõi.
- pass@k và RLVR gắn với nhau: không phân biệt được pass@1 và pass@k thì không hiểu phản biện của Yue.
- Đo từng khâu (truy hồi trước, sinh sau) là kỹ thuật debug hệ thống RAG của chính bạn.

## Có thể lỗi thời (kiểm ngày 30/10/2026, rà lại 04/11/2026)

Phần này ghi cảnh quan tính đến cuối tháng 10/2026, **theo các trang tổng hợp bên thứ ba**, nên coi là thông tin có thể sai và hãy tự kiểm chứng trước khi dẫn. Lần rà ngày 04/11/2026 không kiểm lại được tên mô hình sau tháng 06/2026.

- Các trang tổng hợp liệt kê nhiều mô hình tiền tuyến của năm 2026 (họ GPT-5.x, Claude, DeepSeek V4, Qwen3.5 và các mô hình khác) **[cần kiểm chứng]**; tên, ngày và thứ hạng đổi mỗi vài tuần. Chương không đưa bảng xếp hạng.
- Xu hướng bền hơn tên: mô hình mở trọng số tiến gần mô hình đóng; MoE phổ biến (N trong 6ND là tham số hoạt động); ngữ cảnh dài; RLVR và biến thể GRPO thành thường lệ; từ vựng lớn và đa ngữ hơn.
- Trang chính thức đã đọc: bài OpenAI về SWE-bench Verified (23/2/2026); còn lại chưa có nguồn chính thức.

## Bài tập giấy bút

14 bài, làm theo thứ tự. Mục tiêu: hết ★ trong 1 buổi, hết ★★ trong 2 buổi, ★★★ dành cho Thứ Bảy. Bài nào sai ghi vào sổ lỗi kèm lý do sai.

**★ Hiểu định nghĩa**

1. Mô hình gán xác suất 0.5, 0.5, 0.25, 0.25 cho bốn token đúng. Tính perplexity và số bit trung bình mỗi token.
2. Chạy phép gộp BPE đầu tiên trên chuỗi byte "ababc" (các cặp tính trên cả chuỗi, không có ranh giới từ). Cặp nào được gộp, sau đó chuỗi dài bao nhiêu, và vì sao kết quả của phép gộp thứ hai không duy nhất?
3. Logit (0, ln 2). Tính phân phối ở T = 1 và T = 0.5.
4. Chứng minh khi πθ = πref thì loss DPO bằng ln 2 và trọng số gradient bằng 1/2.

**★★ Mức đề thi đầu vào**

5. Ngân sách 10²⁴ FLOPs, giữ 20 token mỗi tham số. Tìm N và D (dùng 6ND).
6. Theo Kaplan, loss giảm theo N mũ −0.076. Cần nhân N lên bao nhiêu lần để loss giảm 20%?
7. W ∈ ℝ⁴⁰⁹⁶ˣ¹¹ ⁰⁰⁸ (một lớp FFN). LoRA r = 16: số tham số học được, tỉ lệ so với tinh chỉnh toàn phần, hạng tối đa của ΔW.
8. Lượng tử hóa int4 absmax đối xứng (mức −7 đến 7) vector (0.2, −1.0, 0.4, 0.1): tìm mã và sai số lớn nhất.
9. n = 10 mẫu, c = 2 đúng, k = 3: tính pass@3 không chệch và công thức ngây thơ.
10. Mỗi bước của một agent thành công với xác suất 0.98. Cần ít nhất bao nhiêu bước để xác suất thành công cả chuỗi dưới 0.5?

**★★★ Mức phỏng vấn nghiên cứu**

11. Dẫn xuất N tối ưu tỉ lệ với C mũ β/(α + β) từ L = E + A/N mũ α + B/D mũ β với ràng buộc 6ND = C. Tính với (α, β) = (0.34, 0.28) và giải thích vì sao fit công bố vẫn có thể cho tỉ lệ token mỗi tham số rất lớn (cỡ 93 với số mũ làm tròn, cỡ 58 với 0.339 và 0.285) tại ngân sách của Chinchilla, và nó so với con số 40B mà chính paper nêu ở ngân sách Gopher thế nào.
12. Dẫn xuất nghiệm đóng π* tỉ lệ với πref·exp(r/β) của bài toán có KL, rồi suy ra loss DPO từ Bradley-Terry và chỉ ra Z(x) triệt tiêu ở đâu.
13. Thiết kế số liệu để loss DPO giảm trong khi log-xác suất của đáp án được chọn cũng giảm (β = 0.1). Tính loss. Nêu một cách vá (Llama 3 làm gì?) và lý do loss DPO không thấy hiện tượng này.
14. GRPO với G = 8. (a) Tính xác suất nhóm có phần thưởng đồng nhất khi p = 0.9. (b) Một mô hình RLVR có pass@1 cao hơn mô hình nền nhưng pass@256 thấp hơn: diễn giải theo Yue và cộng sự, và nêu một thí nghiệm kiểm chứng.

## Bài tính nhanh

15 bài, mỗi bài tối đa 2 phút, làm bằng giấy bút và bấm giờ cả bộ (bài 1, 2, 10, 12 và 15 cần ln, e mũ hoặc lũy thừa không làm tay được trong 2 phút: cho phép máy tính bỏ túi, hoặc chấp nhận sai số ±0.01 nếu bạn dùng e⁻⁰·⁴ ≈ 0.670, 2⁰·⁷⁵ ≈ 1.682). Mục tiêu: xong cả 15 bài trong 25 phút, sai không quá 1 bài. Làm lại bộ này sau 3 ngày và sau 1 tuần.

1. σ(0.4) bằng bao nhiêu?
2. 2 mũ 1.75 bằng bao nhiêu (perplexity của 1.75 bit mỗi token)?
3. 6ND với N = 7·10⁹ và D = 2·10¹² token.
4. N = 70·10⁹ ở 20 token mỗi tham số: D bằng bao nhiêu?
5. LoRA r = 16 trên ma trận 4096 × 4096: số tham số.
6. 16Ψ byte với Ψ = 7·10⁹: bao nhiêu GB?
7. 7·10⁹ tham số ở NF4 kèm double quantization (4.127 bit): bao nhiêu GB?
8. pass@1 từ n = 20, c = 3.
9. P(nhóm 8 mẫu toàn sai) khi tỉ lệ đúng là 0.2.
10. ln 50 257 (loss ban đầu, từ vựng GPT-2).
11. Entropy của phân phối đều trên 8 token, tính bằng nat và bằng bit.
12. 0.95 mũ 10.
13. Top-p = 0.75 trên p = (0.5, 0.3, 0.15, 0.05): giữ mấy token, phân phối sau chuẩn hóa?
14. Perplexity 16 theo token, mỗi token trung bình 4 byte: bits-per-byte?
15. Loss reward model −ln σ(0.8).

## Lab code và bài săn bug

Năm lab đã chạy với Python 3.13, NumPy 2.5, SciPy, PyTorch 2.14.1 bản CPU (chỉ cần bản CPU, cài theo chương 6; `environment.yml` chưa liệt kê PyTorch, và tôi chưa thử Python 3.11). Notebook lời giải chạy khoảng 30 giây trên máy rảnh và tới hơn 2 phút khi máy bận (đo ngày 04/11/2026, 2 lõi: 27.4 giây theo ô cuối, 34 giây tính cả khởi động kernel, lúc máy rảnh; 134.8 giây lúc máy bận, tải trên 9; lần đo trước: 118 giây lúc máy bận; thời gian tự đo, in ở ô cuối); số liệu lệch nhẹ theo phiên bản và số luồng. **Ô đầu gọi `torch.set_num_threads(1)`**: mô hình tí hon không hưởng lợi từ nhiều luồng, và trên máy ít lõi, nhiều luồng tranh nhau làm chậm gấp vài lần. Dữ liệu tự sinh, seed cố định, không cần mạng. Tự viết từng lab trước; chỉ mở lời giải khi kết quả lệch cột "kết quả mong đợi".

| Lab | Việc phải làm | Kết quả mong đợi |
| --- | --- | --- |
| 1. BPE byte-level | `get_stats`, `merge`, `train_bpe`, `encode`, `decode`; chạy `aaabdaaabac`; hai tokenizer 150 phép gộp trên hai kho mini Anh và Việt; NFC và NFD | `[258, 100, 258, 97, 99]`; khứ hồi đúng; câu Việt 167 token (tokenizer Anh) so với 76 (tokenizer Việt), câu Anh 58 so với 107; "tiếng Việt" 14 byte (NFC), 18 byte (NFD); bộ từ Sennrich cho ba phép đầu cùng tần số 9 |
| 2. Tiền huấn luyện, perplexity, giải mã | entropy rate của nguồn Markov (V = 24); huấn luyện TinyGPT 300 bước; `filter_probs` (nhiệt độ, top-k, top-p) | H = 2.1025 nat, perplexity sàn 8.186; loss đầu 3.356 (ln V = 3.178), loss val 2.1557 (perplexity 8.634); top-p 0.9 giữ 3, 2, 4 token ở T = 1, 0.5, 2; tần số 200 000 mẫu khớp trong ±0.005 |
| 3. Quy luật quy mô | sinh 64 điểm từ fit Epoch cộng nhiễu; fit 5 tham số bằng `least_squares`; tối ưu ở C = 5.88·10²³; fit chỉ trên vùng nhỏ | fit E 1.8265, A 514.3, B 2238.5, α 0.3517, β 0.3694; D/N tối ưu 18.49 (thật 18.38); hằng số công bố số mũ 0.34/0.28 cho 32.5B, 3.02T, D/N 92.8; số mũ 0.339/0.285 cho 41.1B, 2.38T, D/N 57.9; tại 5.76·10²³ (ngân sách Gopher) số mũ 0.339/0.285 cho 40.8B, D/N 57.8, khớp "khoảng 40B" của paper; vùng nhỏ (25 điểm) ngoại suy D/N 13.13 |
| 4. DPO và GRPO đồ chơi | `dpo_loss`, `closed_form`, `group_advantages`, `grpo_loss`; kiểm gradient; DPO trên 20 000 cặp; bandit | loss ln 2 khi πθ = πref; ví dụ 9.7: 0.5130; autograd khớp công thức; TV tới π* 0.030 (assert dưới 0.05); advantage (+1.2910, −0.7746); p(đúng) 0.0264 lên 0.968 ở bước 50, 260 trên 300 nhóm không tín hiệu |
| 5. LoRA và lượng tử hóa | `LoRALinear`, `add_lora`, `merge`; tinh chỉnh sang chuỗi B; `quant_absmax`, `quant_nf4` | tham số 60 096 / 1 608 / 6 432 / 25 728; loss B: toàn phần 2.1497, r = 1 2.1830, r = 4 2.1503, r = 16 2.1462 (sàn 2.1261); gộp adapter lệch 1.2·10⁻⁶; loss A: int8 2.1557, int4 khối 16 2.1577, NF4 2.1582, int2 2.2967; NF4 giảm RMSE 14.6% đến 30.1% trên Gauss |

Code nằm trong `labs/chuong09/lab.ipynb` (có chỗ TODO và ô kiểm tra tự động) và `labs/chuong09/loigiai.ipynb` (lời giải đã chạy kiểm); mở từ trang **Lab** của app.

**Câu hỏi sau lab.**

- Lab 1: vì sao encode phải áp phép gộp đúng thứ tự học, và vì sao byte-level không bao giờ có token ngoài từ vựng?
- Lab 2: vì sao loss val không thể thấp hơn H, và vì sao cùng entropy mà perplexity theo token đổi khi đổi tokenizer? Vì sao top-p giữ số token khác nhau ở T khác nhau?
- Lab 3: vì sao fit trên vùng nhỏ ngoại suy sai tỉ lệ tối ưu, và vì sao cùng một bộ hằng số cho 32.5B hay 41.1B tùy cách làm tròn số mũ? Con số 40B ở chú thích Figure 4 cho bạn phép kiểm chéo nào?
- Lab 4: vì sao DPO hội tụ về π* mà không cần reward model, và vì sao nhóm toàn đúng không có gradient?
- Lab 5: vì sao B = 0 khi khởi tạo, và vì sao tăng r từ 4 lên 16 thêm ít? Vì sao NF4 hơn int4 trên Gauss lý tưởng mà không hơn trên mô hình đồ chơi? LoRA "quên ít hơn" có thấy ở đây không, và vì sao không kết luận được?

**Bài săn bug.** Bốn đường ống dưới đây chạy không báo lỗi. Ô chẩn đoán trong notebook in năm triệu chứng: (1) SFT cho exact-match 1.000 nhưng tỉ lệ tự dừng 0.000, luôn sinh đủ 12 token mới; (2) perplexity báo cáo 4.359, **thấp hơn** cận dưới lý thuyết 8.186; (3) entropy của 200 000 mẫu ở T = 0.5, 1, 2 là 0.9992, 0.9997, 1.0000 (giải tích 0.4579, 1.0000, 1.3943), đổi T không đổi độ đa dạng; (4) top-p 0.9 trên logit (1, 3, 0, 2, −1) cho tần số (0.0867, 0.6441, 0.0318, 0.2373, 0), tức thỉnh thoảng giữ token vô lý; (5) loss lúc khởi tạo của mô hình LoRA là 5.6345 trong khi mô hình nền là 2.1557. Tìm cả **năm** lỗi, không biết ống nào chứa mấy lỗi, rồi viết bản sửa. Mã nằm trong `labs/chuong09/lab.ipynb`, các ô "Đường ống 1" đến "Đường ống 4"; hãy đọc code, đừng chỉ nhìn triệu chứng. Ô kiểm bản sửa chỉ kiểm hành vi mong đợi, không nói nguyên nhân. **Bài săn bug và Lab 5 dùng `entropy_rate` và mô hình `base` của Lab 2, nên phải làm xong Lab 2 (và chạy ô kiểm của nó) trước**; nếu chưa, ô chẩn đoán dừng ngay với thông báo nhắc việc này.

## Bài dựng cảnh Manim

Muốn vẽ được một khái niệm, bạn phải hiểu từng bước của nó. Giới hạn **1 giờ** (tùy chọn nếu chậm tiến độ), không chỉnh màu mè.

**Đề bài.** Dùng `manim/chuong09_scenes.py` làm mẫu, viết cảnh `PassAtK` vào thư mục `manim/cua-toi/`:

1. Lấy n = 20 mẫu, c = 3 đúng (ví dụ 9.12). Vẽ trục k = 1, 2, 5, 10, 20.
2. Vẽ hai đường: pass@k không chệch (0.15, 0.2842, 0.6009, 0.8947, 1.0) và công thức ngây thơ (0.15, 0.2775, 0.5563, 0.8031, 0.9612), mỗi đường có nhãn số.
3. Phụ đề nói đường nào chệch và theo hướng nào.

**Tiêu chí tự chấm.** (1) Đúng toán: mười số khớp ví dụ 9.12, ngây thơ luôn thấp hơn hoặc bằng không chệch. (2) Hai đường phân biệt rõ, có tiêu đề và phụ đề. (3) Dài không quá 30 giây, chữ không chồng chữ.

Render bằng nút **Render cảnh của tôi** ở trang Manim. Lời giải tham khảo: `manim/loigiai/chuong09_pass_at_k.py`, chỉ mở sau khi đã tự làm.

## Phỏng vấn, bẫy, thực tế khác lý thuyết

**Cách luyện.** Mỗi chuỗi có 4 tầng câu hỏi, đào sâu dần như người phỏng vấn thật. Trả lời to bằng tiếng Việt trước, sau đó nói lại bằng tiếng Anh, 2 phút mỗi tầng, ghi âm và tự nghe lại. Câu mẫu tiếng Anh chỉ phủ tầng 1–2; tầng 3–4 tự soạn rồi đối chiếu lời giải bài tập.

**Chuỗi 1 — Token hóa, perplexity, giải mã**

1. Vì sao dùng subword thay vì từ hoặc ký tự?
2. BPE chạy thế nào? Byte-level giải quyết gì? Tie-break có quan trọng không?
3. Perplexity là gì, vì sao không so giữa hai tokenizer, và chỉ số nào thay thế? Perplexity thấp hơn exp(entropy) nghĩa là gì?
4. Greedy, beam, nhiệt độ, top-p khác nhau thế nào? Vì sao văn bản người có xác suất thấp hơn văn bản beam? Tiếng Việt chịu thiệt gì ở tokenizer tiếng Anh?

> *Model answer:* Words give a huge vocabulary and fail on rare words, characters give very long sequences, so subwords sit in between. BPE starts from single symbols, repeatedly merges the most frequent adjacent pair, and encodes new text by replaying the merges in the order they were learned; the byte-level version starts from 256 bytes so nothing is ever out of vocabulary. Perplexity is the exponential of the average per-token negative log-likelihood, so it depends on how the text is split into tokens. Two tokenizers cannot be compared by perplexity; bits-per-byte can.

**Chuỗi 2 — Quy luật quy mô**

1. Luật lũy thừa là gì và đọc trên đồ thị log-log thế nào?
2. Kaplan và Chinchilla khác nhau ở đâu? 6ND đến từ đâu? "20 token mỗi tham số" đến từ đâu?
3. Dẫn xuất N tối ưu tỉ lệ với C mũ β/(α + β). Vì sao hai paper cho kết quả khác (đếm embedding, quy mô nhỏ, chất lượng fit)? Ngoại suy tin được đến đâu?
4. Vì sao các hãng huấn luyện vượt Chinchilla? Vai trò của MoE, dữ liệu hữu hạn, test-time compute.

> *Model answer:* A scaling law says the loss falls as a power law in parameters, data or compute, which is a straight line on a log-log plot. Kaplan suggested putting most of a larger budget into model size, while Chinchilla found parameters and tokens should grow together, which is roughly 20 tokens per parameter. That figure is a summary of the result, not a constant of nature, and the paper's own third approach predicts about 40B parameters at Gopher's budget, well below the 67B and 63B from the other two approaches. Frontier labs train well past it because inference cost favors smaller models trained longer.

**Chuỗi 3 — Hậu huấn luyện**

1. Vì sao SFT trước RLHF? Vì sao cần che prompt và thêm EOS?
2. Reward model huấn luyện thế nào (Bradley-Terry)? Vì sao cần KL và β làm gì? Nghiệm đóng của bài toán là gì?
3. Dẫn xuất DPO. DPO khác PPO thế nào? Likelihood displacement và over-optimization là gì?
4. GRPO khác PPO ở đâu, vì sao không cần critic, nhóm đồng nhất xử lý thế nào? RLVR dùng được ở đâu? RL có thật sự tạo năng lực mới không (pass@k)?

> *Model answer:* RLHF maximizes expected reward minus a KL penalty to a reference policy, whose optimum is the reference policy reweighted by the exponential of reward over beta. DPO substitutes the implicit reward, beta times the log ratio to the reference, into the Bradley-Terry loss, so the partition function cancels in the difference and no separate reward model is needed. Reward models are trained on pairwise comparisons with a logistic loss. SFT comes first because the policy must already produce reasonable answers to compare, and the loss is computed only on answer tokens, including the end-of-sequence token.

**Chuỗi 4 — LoRA và lượng tử hóa**

1. Bộ nhớ khi huấn luyện 7B gồm những gì? Con số 16 byte mỗi tham số từ đâu?
2. LoRA là gì, vì sao B = 0, α/r làm gì, đếm tham số, gộp thế nào?
3. Vì sao lượng tử hóa gặp khó với ngoại lệ? Theo khối, NF4, double quantization, QLoRA ghép gì với gì?
4. LoRA có ngang tinh chỉnh toàn phần không (Biderman)? Chỗ nào đã đo, chỗ nào chỉ tính từ công thức? Lượng tử hóa sau huấn luyện so với QAT?

> *Model answer:* LoRA freezes the pretrained weight and learns a low-rank update, the product of a d by r matrix and an r by k matrix. B starts at zero so the model is unchanged at step zero, and after training the update can be merged into the weight, so inference costs nothing extra. For a 4096 by 4096 matrix with rank 8 that is 65 536 trainable parameters, about 0.4 percent. QLoRA stores the frozen weights in 4-bit NF4 with double quantization of the block constants and trains LoRA adapters on top, which cuts memory a lot, though activation memory still matters.

**Từ vựng nên nói trôi chảy:** byte-level BPE, tokenizer fertility, perplexity, bits-per-byte, nucleus sampling, temperature, compute-optimal, power law, active parameters, supervised fine-tuning, reward model, Bradley-Terry, KL-regularized objective, likelihood displacement, verifiable rewards, advantage, pass@k, low-rank adaptation, block-wise quantization, benchmark contamination.

**Bẫy thường gặp.** Đọc sau khi đã làm bài săn bug: vài bẫy dưới đây trùng với lỗi trong bài.

- So perplexity giữa hai tokenizer (ví dụ 9.3: 2.00, 3.17, 1.19 cho cùng văn bản); dùng bits-per-byte.
- Tính perplexity có cả token đệm: báo 4.36 trong khi cận dưới xấp xỉ 8.2.
- Áp nhiệt độ lên xác suất thay vì logit (chia rồi chuẩn hóa lại ra đúng phân phối cũ); top-p không sắp xếp trước khi cộng dồn; bỏ token làm tích lũy vượt p.
- Quên EOS trong mục tiêu SFT: exact-match vẫn 1.000 nhưng mô hình không dừng.
- 6ND dùng sai N (tổng, không-embedding hay hoạt động): lệch 1.45 lần ở cỡ 125M.
- DPO: tưởng loss giảm là xác suất đáp án tốt tăng; dùng trung bình thay vì tổng log-xác suất trên token (SimPO cố ý dùng trung bình; lỗi là lẫn lộn hai cách); quên cố định log-xác suất tham chiếu.
- LoRA: khởi tạo B khác 0 (loss đầu 5.63 thay vì 2.16); quên đóng băng mô hình nền; quên hệ số α/r khi gộp; gộp vào trọng số đã lượng tử hóa.
- pass@k ngây thơ thay vì không chệch.
- NFC và NFD lẫn nhau; token đặc biệt do người dùng gài.
- Dẫn số theo phiên bản khác nhau (R1-Zero 71.0 hay 77.9; MMLU 14 080 hay 14 042 câu). Luôn nói "theo bản nào".

**Thực tế khác lý thuyết.**

- **Cách 3 của Chinchilla lệch cách 1 và 2, và số 93 là hệ quả làm tròn.** Paper tự nêu cách 3 dự đoán khoảng 40B ở ngân sách Gopher (67B và 63B từ cách 1, 2). Hằng số cách 3 cho 41.1B và 2.38T (57.9) với số mũ đủ chữ số, hoặc 32.5B và 3.0T (92.8) với số mũ làm tròn, ở ngân sách của Chinchilla; fit lại của Epoch cho 73.0B và 1.34T. Quy tắc 20 token mỗi tham số đứng vững nhờ cách 1, cách 2 và fit lại.
- **Tối ưu cho huấn luyện khác tối ưu cho triển khai.** Mô hình thực tế được huấn luyện xa hơn Chinchilla nhiều vì chi phí suy luận.
- **Trên đồ chơi, LoRA không "quên ít hơn" ở cùng mức học.** Loss A sau tinh chỉnh sang B: toàn phần 2.8665, LoRA r = 4 2.9467, r = 1 2.7985; đó là đánh đổi theo mức học, nên không khẳng định hay phản bác Biderman. Tương tự, NF4 không hơn int4 theo khối nhỏ trên mô hình đồ chơi, và che prompt trong SFT hầu như vô hại ở đây (thử ngoài lab).

## Mini-test 30 phút

Làm sau bài tập và lab, bấm giờ 30 phút, không xem sách, không dùng máy tính (đáp số số học chấp nhận sai số ±0.02 hoặc 2 chữ số có nghĩa). Tổng 30 điểm. **Từ 24 điểm:** sang chương 10. **18 đến dưới 24:** ôn lại mục ứng với câu sai rồi làm lại sau 3 ngày. **Dưới 18:** học lại chương.

1. **(4 điểm)** Cùng một đoạn 400 byte, hai tokenizer chia thành 100 và 50 token; tổng NLL của mô hình trên đoạn đó là 100·ln 2 nat với cả hai. Tính perplexity theo token và bits-per-byte của từng tokenizer. Mô hình nào tốt hơn?
2. **(5 điểm)** Logit (2, 1, 0), T = 1, top-p = 0.8. Tính phân phối sau khi cắt và chuẩn hóa. Nếu T = 0.5 thì số token giữ lại tăng, giảm hay không đổi, và vì sao?
3. **(5 điểm)** Ngân sách 2·10²⁴ FLOPs, 20 token mỗi tham số: tìm N và D. Nêu hai lý do thực tế có thể khiến bạn chọn mô hình nhỏ hơn và huấn luyện nhiều token hơn.
4. **(5 điểm)** DPO với β = 0.2: ln πθ(yw) = −10, ln πref(yw) = −11, ln πθ(yl) = −12, ln πref(yl) = −9. Tính z, loss, trọng số gradient. Xác suất của yw đã tăng hay giảm? Loss có cho bạn biết điều đó không?
5. **(5 điểm)** Llama-2-7B có d = 4096, 32 khối. LoRA r = 8 trên {q, v}: bao nhiêu tham số? Trọng số 7B (6.74·10⁹) ở NF4 kèm double quantization (4.127 bit) chiếm bao nhiêu GiB? Vì sao con số đó chưa đủ để kết luận nó chạy được trên thẻ 6 GiB?
6. **(6 điểm)** Đọc code và tìm **ba** lỗi (không chạy):

```python
p = torch.softmax(logits, -1) / T
cs = torch.cumsum(p, -1)
keep = cs < top_p
p = p * keep
return torch.multinomial(p / p.sum(), 1)
```

## Flashcard

18 thẻ của chương nằm trong `flashcards/chuong09.csv`. Ôn ở trang **Ôn tập** của app: app tự lên lịch theo hộp Leitner (thẻ nhớ được sẽ giãn lịch ôn 1 → 3 → 7 → 14 ngày, thẻ quên quay lại hộp 1). File CSV này cũng nhập thẳng được vào Anki nếu bạn thích dùng Anki.

## Lời giải

Chỉ đọc sau khi đã tự thử ít nhất 20 phút. Với bài chứng minh, so ý chính chứ không so từng chữ. Mọi đáp số bằng số trong chương đã được kiểm lại bằng code.

**Kiểm tra nhanh**

1. "aaab aaab": cặp (a, a) xuất hiện 2 lần mỗi từ, tổng 4; (a, b) 2. Gộp (a, a) thành X: mỗi từ thành X a b, tức 3 token (cả hai từ: 6). Phép thứ hai: (X, a) 2 và (a, b) 2 hòa; gộp (X, a) thành Y thì mỗi từ còn Y b, 2 token; gộp (a, b) thì X và ab, 2 token. Cả hai đều còn 2 token mỗi từ, tổng 4 sau phép hai.
2. Perplexity = 8, tức 3 bit mỗi token (ln 8 = 2.079 nat).
3. D = C/(6N) = 1.2·10²³/(6·10¹⁰) = 2·10¹² token; D/N = 200.
4. σ(1.0 − 0.6) = σ(0.4) = 0.5987.
5. 4·(1000 + 1000) = 8 000 so với 10⁶: 0.8%.

**Bài tập giấy bút**

1. NLL trung bình = (2·ln 2 + 2·ln 4)/4 = 1.5 ln 2, perplexity = 2 mũ 1.5 = 2.828; 1.5 bit mỗi token.
2. Cặp (a, b) xuất hiện 2 lần, các cặp khác 1 lần: gộp (a, b) thành X, chuỗi "XXc" dài 3. Ở phép thứ hai, (X, X) và (X, c) cùng tần số 1, hòa nhau; hai cài đặt tie-break khác nhau cho từ điển khác.
3. T = 1: (1/3, 2/3). T = 0.5: logit (0, 2 ln 2) cho (1, 4)/5 = (0.2, 0.8).
4. Khi πθ = πref, hai số hạng log-tỉ số đều 0, nên z = 0 và loss = −ln σ(0) = ln 2. Trọng số σ(r̂l − r̂w) = σ(0) = 1/2.
5. N = √(C/(6·20)) = √(8.33·10²¹) = 9.13·10¹⁰ (khoảng 91B), D = 20N = 1.83·10¹² (khoảng 1.8T).
6. 0.8 = x mũ −0.076 suy ra x = 0.8 mũ (−1/0.076) = exp(0.2231/0.076) = exp(2.936) khoảng 18.8 lần (bỏ qua số hạng loss không giảm được, chỉ xét phần giảm theo luật lũy thừa).
7. 16·(4096 + 11 008) = 241 664; toàn phần 4096·11 008 = 45 088 768; tỉ lệ 0.54%; hạng tối đa 16.
8. Hệ số 7/1.0 = 7: mã (1, −7, 3, 1) (1.4 làm tròn 1; 2.8 làm tròn 3; 0.7 làm tròn 1). Khôi phục (0.1429, −1, 0.4286, 0.1429); sai số lớn nhất 0.0571 tại phần tử đầu.
9. pass@3 = 1 − C(8, 3)/C(10, 3) = 1 − 56/120 = 0.5333; ngây thơ 1 − 0.8³ = 0.488, chệch xuống.
10. 0.98 mũ n < 0.5 suy ra n > ln 0.5/ln 0.98 = 34.3, nên n = 35.
11. Thế D = C/(6N): L = E + A·N⁻ᵅ + B·(C/6)⁻ᵝ·Nᵝ. Đạo hàm theo N bằng 0: −αA·N⁻ᵅ⁻¹ + βB·(C/6)⁻ᵝ·Nᵝ⁻¹ = 0, suy ra N mũ (α + β) tỉ lệ với (C/6) mũ β, tức N tối ưu tỉ lệ với C mũ β/(α + β). Với (0.34, 0.28): 0.28/0.62 = 0.4516. Với N tối ưu tỉ lệ với C mũ β/(α + β) và D = C/(6N), D tối ưu tỉ lệ với C mũ α/(α + β), nên D/N tỉ lệ với C mũ (α − β)/(α + β), nhân một hệ số phụ thuộc A, B, α, β. Số mũ cho hướng, hệ số cho điểm. Với α > β (cả hai cách ghi số mũ công bố), D/N tăng theo C: số mũ làm tròn (0.34, 0.28) cho 50 ở C = 10²¹, 93 ở 5.88·10²³, 153 ở 10²⁶; số mũ đủ chữ số (0.339, 0.285) cho 33, 58, 90. Fit của Epoch có α < β nên D/N giảm chậm: 21.6, 18.4, 16.1. Vậy "cỡ 93" chỉ là con số của số mũ làm tròn, nhạy với làm tròn; "cỡ 58" ứng với số mũ đủ chữ số và khớp con số **khoảng 40B** mà chính paper nêu cho cách 3 ở ngân sách Gopher (chú thích Figure 4): 5.76·10²³ cho 40.8B, D/N 57.8. Cách 3 lệch cách 1 (67B) và cách 2 (63B), paper đã nêu điều đó; cả hai xa 20. (Tìm số học, Lab 3.)
12. Với Z(x) chuẩn hóa, KL(π ‖ πref) = E[ln π/πref], nên mục tiêu E[r] − β·E[ln π/πref] = −β·KL(π ‖ π*) + hằng, với π* = πref·exp(r/β)/Z. KL ≥ 0, bằng 0 khi π = π*. Giải ngược r = β ln(π*/πref) + β ln Z(x). Thế vào Bradley-Terry: rw − rl = β[ln π*(yw)/πref(yw) − ln π*(yl)/πref(yl)], vì β ln Z(x) giống nhau ở hai đáp án (cùng prompt x) nên triệt tiêu trong hiệu. Bước cuối: thay π* bằng chính sách tham số πθ rồi cực đại likelihood Bradley-Terry của dữ liệu ưa thích, tức cực tiểu −ln σ(β[ln πθ(yw)/πref(yw) − ln πθ(yl)/πref(yl)]) chính là loss DPO.
13. Dùng ln πθ(yw) từ −14 xuống −16, ln πθ(yl) từ −13 xuống −20, β = 0.1: z = 0.1·[(−2) − (−7)] = 0.5, loss = −ln σ(0.5) = 0.4741 < 0.6931 (loss lúc khởi đầu, khi πθ = πref nên z = 0). Loss chỉ phụ thuộc hiệu hai tỉ số, nên đáp án thua giảm nhiều hơn đáp án thắng vẫn làm loss giảm. Cách vá: thêm số hạng NLL trên đáp án được chọn (Llama 3 dùng hệ số 0.2 theo tóm tắt công cụ), hoặc theo dõi log-xác suất của đáp án thắng riêng.
14. (a) 0.9⁸ + 0.1⁸ = 0.4305 + 10⁻⁸ ≈ 0.4305: gần nửa số nhóm không có gradient. (b) RL tập trung xác suất vào những lời giải mô hình nền đã sinh được nên pass@1 tăng, nhưng khả năng phủ (pass@k lớn) thu hẹp; kết luận của Yue và cộng sự là RLVR chủ yếu cải thiện độ tin cậy lấy mẫu. Thí nghiệm: vẽ pass@k với k từ 1 tới vài trăm cho mô hình nền và mô hình RL trên cùng bộ bài, cùng nhiệt độ lấy mẫu, dùng ước lượng không chệch; điểm cắt là dấu hiệu.

**Bài tính nhanh**

1. 0.5987.
2. 3.364.
3. 6·7·10⁹·2·10¹² = 8.4·10²² FLOPs.
4. 1.4·10¹² token.
5. 16·(4096 + 4096) = 131 072.
6. 112 GB (104.3 GiB).
7. 7·10⁹·4.127/8 = 3.61 GB.
8. 0.15.
9. 0.8⁸ = 0.1678.
10. 10.82.
11. ln 8 = 2.079 nat = 3 bit.
12. 0.5987.
13. Tích lũy 0.5, 0.8: giữ 2 token (token thứ hai làm vượt 0.75); chuẩn hóa (0.625, 0.375).
14. log₂ 16 = 4 bit mỗi token, chia 4 byte mỗi token: 1 bit mỗi byte.
15. 0.3711.

**Câu hỏi sau lab**

- Lab 1: phép gộp sau có thể dùng kết quả phép gộp trước (id lớn hơn), nên phải áp theo thứ tự học, nếu không các token ghép không xuất hiện. Byte-level bắt đầu từ đủ 256 byte nên mọi chuỗi UTF-8 đều biểu diễn được.
- Lab 2: loss kỳ vọng của mô hình ≥ entropy rate (cross-entropy lớn hơn entropy), chỉ có nhiễu mẫu mới cho phép hơi thấp hơn. Đổi tokenizer đổi số token nhưng không đổi tổng NLL nên perplexity theo token đổi. Nhiệt độ làm phẳng hay nhọn phân phối nên số token cần để đạt tổng p thay đổi.
- Lab 3: vùng nhỏ không chứa thông tin về đường cong ở xa; đánh đổi giữa αA·N⁻ᵅ và βB·D⁻ᵝ bị đoán sai. Cùng một bộ hằng số cho 32.5B (số mũ làm tròn) hay 41.1B (số mũ đủ chữ số): phải tự tính lại và kiểm chéo với con số paper tự nêu (khoảng 40B ở ngân sách Gopher, khớp 40.8B với số mũ đủ chữ số).
- Lab 4: dữ liệu ưa thích sinh từ Bradley-Terry với reward vàng r*; với cùng β, chính sách tối ưu của loss DPO trên lớp mọi chính sách là π* ∝ πref·exp(r*/β), và reward ngầm β·ln(π/πref) hội tụ về r* (sai khác một hằng số theo prompt). Nhóm toàn đúng có độ lệch chuẩn 0, mọi advantage 0.
- Lab 5: B = 0 để mô hình ban đầu không đổi. Tăng r từ 4 lên 16 chỉ cải thiện 0.004 vì loss đã sát sàn 2.1261. NF4 tối ưu cho phân phối chuẩn, còn trọng số mô hình tí hon chưa chuẩn, và khối nhỏ làm hằng số khối hấp thụ gần hết sai số. "Quên ít hơn" phụ thuộc mức học: so sánh phải ở cùng mức học.

**Mini-test (thang 30 điểm)**

1. Tokenizer 100 token: exp(69.31/100) = 2.00; tokenizer 50 token: exp(69.31/50) = 4.00 (2 điểm). Bits-per-byte cả hai: 100 bit/400 byte = 0.25 (1 điểm). Tốt như nhau; perplexity khác nhau chỉ vì đơn vị token khác (1 điểm).
2. p = (0.6652, 0.2447, 0.0900) (1 điểm). Tích lũy 0.6652 < 0.8 nên giữ token 2 (tích lũy 0.9099 ≥ 0.8), loại token 3 (1 điểm); chuẩn hóa (0.7311, 0.2689) (1 điểm). T = 0.5: logit/T = (4, 2, 0) cho (0.8668, 0.1173, 0.0159), token đầu đã ≥ 0.8 nên chỉ giữ 1 token: giảm (2 điểm), vì T nhỏ làm phân phối nhọn.
3. N = √(2·10²⁴/120) = √(1.667·10²²) = 1.29·10¹¹ (129B) (2 điểm), D = 20N = 2.58·10¹² (2.6T) (1 điểm). Lý do: chi phí suy luận lớn khi phục vụ nhiều (mô hình nhỏ rẻ hơn mỗi token); thẻ nhớ hoặc độ trễ giới hạn kích thước (2 điểm).
4. z = 0.2·[(−10 + 11) − (−12 + 9)] = 0.2·[1 + 3] = 0.8 (1.5 điểm); loss = −ln σ(0.8) = 0.3711 (1 điểm); σ(−0.8) = 0.3100 (1 điểm). ln πθ(yw) tăng từ −11 lên −10: đã tăng (0.5 điểm). Loss không cho biết: nó chỉ thấy hiệu, nên có thể giảm cả khi xác suất đáp án thắng giảm (1 điểm).
5. 32·8·(4096 + 4096)·2 = 4 194 304 (2 điểm). 6.738·10⁹·4.127/8 byte = 3.476·10⁹ byte = 3.24 GiB (1.5 điểm). Chưa tính activation, KV cache, trạng thái optimizer và adapter; chương này chưa đo trên GPU (1.5 điểm).
6. Ba lỗi (2 điểm mỗi lỗi): chia T trên xác suất thay vì logit (và đây là lỗi vô tác dụng sau chuẩn hóa); cumsum trên xác suất chưa sắp giảm dần; điều kiện `cs < top_p` loại token làm tích lũy vượt p (và có thể loại mọi token nếu token đầu đã ≥ p); đúng ra là `cs - p < top_p`. Chấp nhận đủ 3 lỗi trong số này.

**Săn bug — năm lỗi**

1. **Che EOS khỏi mục tiêu** (dòng `y[y >= NC] = -100` xóa EOS; SEP đã bị che sẵn bởi dòng `y[:, :LP] = -100` ngay trên, và PAD không có ở đây): mô hình không học dừng. Sửa: chỉ che prompt. Bản sửa cho tỉ lệ dừng 1.000, 5.00 token mới, exact-match 1.000.
2. **Pad nằm trong loss và trong phép đo perplexity**: pad dễ đoán nên kéo perplexity xuống 4.359, dưới cận dưới 8.186. Sửa: che pad cả lúc huấn luyện (`ignore_index`) và lúc đo; bản sửa cho 8.730 (trên cận dưới 8.186, dưới 1.1 lần cận dưới = 9.00). Ô kiểm đặt ngưỡng ≤ 1.1·exp(H) vì bản chỉ che pad lúc đo mà vẫn để pad trong loss huấn luyện cho 9.365 và phải trượt.
3. **Nhiệt độ áp lên xác suất rồi chuẩn hóa lại**: p/T chuẩn hóa lại ra đúng p. Sửa: chia logit.
4. **Top-p không sắp xếp trước khi cộng dồn**: giữ nhầm token xác suất thấp (0.0318). Sửa: sắp giảm dần rồi mới cộng dồn; bản sửa chỉ giữ ba token (tần số của 200 000 mẫu 0.0896, 0.6654, 0.2449; giải tích 0.0900, 0.6652, 0.2447).
5. **LoRA khởi tạo B ngẫu nhiên** thay vì 0: loss lúc đầu 5.6345 thay vì 2.1557. Sửa: B bằng 0.

```python
y[:, :LP] = -100                                    # chỉ che prompt; EOS nằm trong mục tiêu
y[y == PADT] = -100                                 # và che pad ở cả huấn luyện lẫn đo
p = torch.softmax(logits / T, -1)                   # chia T trên logit
sp, si = torch.sort(p, descending=True)             # sắp trước khi cộng dồn
keep = (torch.cumsum(sp, -1) - sp) < top_p
self.B = nn.Parameter(torch.zeros(base.out_features, r))
```

Ba phép thử đa năng lộ cả năm lỗi mà không cần biết đáp án: chỉ số hoàn hảo mà hành vi sai; con số tốt hơn giới hạn lý thuyết là con số sai; tại bước 0 của LoRA, đầu ra phải bằng đầu ra nền. Phần lấy mẫu kiểm bằng cách so tần số với softmax(z/T) đã biết.

## Tài liệu đào sâu và nguồn

Viết lại bằng lời của người viết. **Mức xác minh (ngày 30/10/2026):** các paper dưới đây đã được mở qua công cụ trả về **bản tóm tắt**, không phải nguyên văn; ngoại trừ hai nguồn đã đối chiếu PDF ngày 04/11/2026 (Chinchilla: Eq. 10, Table 2, Table 3, Figure 4 và bảng A4 về 6ND; Sennrich: chú thích Figure 1), các số bảng, số phương trình trong chương đều **chưa đối chiếu bản gốc**, gồm Eq. 3 của GRPO (và dạng PPO, GRPO tôi viết lại ở mục 9.3), số mũ 0.339 và 0.285 (chỉ có phép kiểm chéo với con số 40B của Figure 4), hàng nghi lỗi in ở bảng phụ lục Chinchilla, số premium của Petrov, Table 4 của Tülu 3. Nguồn chỉ khớp tiêu đề: LiveCodeBench, Chatbot Arena, ReAct. Mở bản gốc trước khi dẫn số.

| Tài liệu | Dùng để làm gì | Phần nên đọc |
| --- | --- | --- |
| [Sennrich, Haddow, Birch (2016)](https://arxiv.org/abs/1508.07909); [minbpe của Karpathy](https://github.com/karpathy/minbpe) | BPE gốc; cài đặt byte-level tham chiếu | Đoạn mã `get_stats` và `merge`; `basic.py`, `regex.py` |
| [Petrov và cộng sự (2023)](https://arxiv.org/abs/2305.15425); [PhoGPT](https://arxiv.org/abs/2311.02945) | Công bằng token giữa các ngôn ngữ; mô hình tiếng Việt | Kết quả premium; mục tokenizer |
| [Holtzman và cộng sự (2020)](https://arxiv.org/abs/1904.09751) | Nucleus sampling, vì sao beam nhạt | Mục giải mã và Table 1 |
| [Kaplan và cộng sự](https://arxiv.org/abs/2001.08361); [Chinchilla](https://arxiv.org/abs/2203.15556); [Epoch AI](https://arxiv.org/abs/2404.10102); [Pearce và Song](https://arxiv.org/abs/2406.12907) | Scaling law, phê bình và dung hòa | Ba cách ước lượng của Chinchilla; phần fit lại của Epoch |
| [Sardana và cộng sự](https://arxiv.org/abs/2401.00448); [Muennighoff và cộng sự](https://arxiv.org/abs/2305.16264); [Llama 3](https://arxiv.org/abs/2407.21783) | Vượt Chinchilla; dữ liệu hữu hạn; ví dụ 405B | Abstract; mục tiền huấn luyện của Llama 3 |
| [InstructGPT](https://arxiv.org/abs/2203.02155); [LIMA](https://arxiv.org/abs/2305.11206); [DPO](https://arxiv.org/abs/2305.18290); [Razin và cộng sự](https://arxiv.org/abs/2410.08847); [Gao và cộng sự](https://arxiv.org/abs/2210.10760) | Đường ống RLHF; DPO và lỗi của nó; over-optimization | Eq. 1 InstructGPT; dẫn xuất DPO |
| [Tülu 3](https://arxiv.org/abs/2411.15124); [DeepSeekMath](https://arxiv.org/abs/2402.03300); [DeepSeek-R1](https://arxiv.org/abs/2501.12948); [DAPO](https://arxiv.org/abs/2503.14476); [GSPO](https://arxiv.org/abs/2507.18071); [Yue và cộng sự](https://arxiv.org/abs/2504.13837) | RLVR, GRPO và biến thể; phản biện về năng lực mới | Mục GRPO; kết quả pass@k của Yue |
| [LoRA](https://arxiv.org/abs/2106.09685); [Biderman và cộng sự](https://arxiv.org/abs/2405.09673); [LLM.int8()](https://arxiv.org/abs/2208.07339); [QLoRA](https://arxiv.org/abs/2305.14314); [ZeRO](https://arxiv.org/abs/1910.02054) | LoRA và phản biện; lượng tử hóa; ngân sách bộ nhớ | Phần phương pháp; phần NF4 và double quantization |
| [Codex và HumanEval](https://arxiv.org/abs/2107.03374); [MMLU](https://arxiv.org/abs/2009.03300); [RAG gốc](https://arxiv.org/abs/2005.11401); [OpenAI về SWE-bench Verified](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) | pass@k không chệch; benchmark; RAG; nhiễm và vòng đời benchmark | Mục đánh giá pass@k; bài viết của OpenAI |
| [Sách RLHF của Lambert](https://rlhfbook.com/) | Đào sâu hậu huấn luyện (chỉ mở trang chủ, chưa đọc từng chương) | Các chương về reward model và over-optimization |

**Từ khóa để tự tìm thêm:** SentencePiece, glitch tokens, μP, IsoFLOP, emergent abilities, test-time compute, loss masking, reward hacking, length bias, IPO, KTO, SimPO, process reward model, Dr. GRPO, DoRA, GPTQ, AWQ, speculative decoding, LLM-as-a-judge, SWE-bench Pro, reranker. Chương 11 (VD 11.9): phát hiện nhiễm bằng n-gram.

**Trước khi sang chương 10:** đạt từ 24/30 điểm mini-test, làm lại các bài ★★★ sai, tự viết được BPE (Lab 1) và `filter_probs` (Lab 2) không nhìn code mẫu, và tìm được cả năm lỗi của bài săn bug.

Nguồn tra cứu ngày 30/10/2026.
