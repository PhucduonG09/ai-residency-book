# Chương 11 — Phương pháp thực nghiệm và đọc paper

## Mục tiêu và kiểm tra nhanh

Chương này dạy bạn hai việc đối xứng nhau: nhìn ra chỗ một paper nói quá bằng chứng của nó, và tự làm thí nghiệm sao cho chính bạn không mắc lỗi đó. Chương 3 đã cho bạn công cụ thống kê (CI, McNemar, bootstrap, đa kiểm định); ở đây ta hỏi so sánh nào đáng làm, cần bao nhiêu seed, dữ liệu rò ở đâu, và viết thế nào cho đúng điều đã đo. Mọi con số trong chương lấy từ một bài toán đồ chơi (10 lớp Gauss trong ℝ²⁰, tập train 300 mẫu, MLP 20–64–10): nó dạy **cấu trúc** của sai lầm, không dạy kích cỡ hiệu ứng trong NLP thật. Thời lượng: khoảng 2 tuần, 8 giờ mỗi tuần.

**Quy ước nguồn.** Dấu † sau tên một paper nghĩa là tôi chưa mở lại được bản gốc khi viết sách (proxy chặn arXiv và ACL Anthology): chỉ có tên, năm và ý chung từ trí nhớ, không có con số "trong paper" nào. Nguồn đã đọc thật nằm ở bảng cuối chương.

**Sau chương này, bạn phải làm được:**

- ☐ Đọc một paper theo ba lượt, điền bản đồ khẳng định – bằng chứng, và kiểm tính tự nhất quán của số liệu (đếm chiến thắng, suy cỡ test set).
- ☐ Thiết kế thí nghiệm: giả thuyết có thể sai, phân loại siêu tham số, baseline mạnh, ngân sách tune công bằng, ablation hai chiều.
- ☐ Báo cáo kết quả nhiều seed: mean ± std, CI bằng phân vị t, số seed theo độ mạnh, và nói được vì sao CI theo seed chưa phải CI của độ chính xác thật.
- ☐ Nhận ra và đo được rò rỉ: chọn đặc trưng trước khi chia, trùng lặp, dùng lại test set, nhiễm benchmark (kể cả giới hạn của kiểm n-gram).
- ☐ Viết bảng, hình, limitations và tài liệu tái lập mà mọi khẳng định đều trỏ được tới bằng chứng.

**Kiểm tra nhanh (5 phút, không dùng máy tính).** Câu i kiểm tra mục 11.i.

1. Một bảng ghi "phương pháp thắng 11 trên 18 cột". Số thắng kỳ vọng nếu mỗi cột là tung đồng xu là 9, độ lệch chuẩn √(18 · 0.25) ≈ 2.1. Đây có phải bằng chứng không?
2. Bạn so phương pháp mới (đã tune learning rate) với baseline lấy từ bảng của paper khác, hơn 5 điểm. Câu hỏi đầu tiên cần đặt là gì?
3. 5 seed cho std mẫu 0.5. SE bằng bao nhiêu, và nửa độ rộng của CI 95% bằng bao nhiêu (hệ số t với 4 bậc tự do là 2.776)?
4. Chọn 20 đặc trưng có tương quan lớn nhất với nhãn trên toàn bộ dữ liệu, rồi mới kiểm định chéo 5 fold. Vì sao độ chính xác CV bị lạc quan ngay cả khi nhãn là ngẫu nhiên?
5. 10 seed cho 74.02 ± 0.27 (std mẫu). SE bằng bao nhiêu, và nên in mấy chữ số thập phân?

Đáp án ở mục Lời giải cuối chương.

## 11.1 Đọc paper theo ba lượt và phê bình có cấu trúc

Đọc một paper thực nghiệm là kiểm toán: với mỗi khẳng định, tìm bằng chứng và hỏi bằng chứng đó có đủ không. Bạn sẽ đọc hàng trăm paper trong năm tới; không có quy trình thì vừa mất thời gian vừa tin nhầm.

**Ba lượt** (cấu trúc theo Keshav, "How to Read a Paper", 2007†; thời gian là điều chỉnh của tôi cho mục đích thi residency, không phải số của tác giả):

| Lượt | Thời gian | Làm gì | Ra gì |
| --- | --- | --- | --- |
| 1 | 10 phút | Tiêu đề, abstract, mở bài, tiêu đề mục, kết luận, lướt hình và bảng | Vấn đề gì, đóng góp gì, có đáng đọc tiếp không |
| 2 | khoảng 1 giờ | Đọc kỹ, bỏ qua chứng minh, ghi chú lề, đọc từng bảng và hình | **Bản đồ khẳng định – bằng chứng** (dưới đây) |
| 3 | vài giờ | Tự dựng lại thí nghiệm từ giả định của tác giả rồi so với bản của họ | Hiểu đủ để tái hiện hoặc phản biện; chỉ làm với 2–3 paper nền mỗi tháng |

**Bản đồ khẳng định – bằng chứng.** Mỗi khẳng định trong abstract là một dòng. Cột cần điền:

| Khẳng định | Bảng/hình | Chỉ số | Baseline | Số seed | Cỡ test | Siêu tham số chọn trên tập nào | Hiệu số so với phương sai |
| --- | --- | --- | --- | --- | --- | --- | --- |

Ô nào không điền được là một nghi vấn. Với dòng nào cũng hỏi sáu câu: (1) khẳng định có đúng là điều thí nghiệm kiểm không (khẳng định "cơ chế X giúp" khác với "mô hình có X tốt hơn"); (2) baseline có mạnh và được tune công bằng không (mục 11.2); (3) hiệu số có lớn hơn nhiễu không (11.3); (4) có đường rò rỉ nào từ train sang test không (11.4); (5) số liệu có tự nhất quán không; (6) thí nghiệm nào sẽ làm tôi đổi ý.

Câu (5) là phần mới so với chương 3. Hai phép kiểm rẻ nhất:

**Ví dụ giải mẫu 11.1 — đếm chiến thắng.** Một paper có 6 dataset × 3 chỉ số = 18 cột và tuyên bố "thắng 11/18". Nếu mỗi cột là tung đồng xu công bằng thì P(thắng ≥ 11 trên 18) = 0.2403 (hai phía 0.4807): chẳng có gì lạ. Thắng 14/18 mới cho 0.0154. Tệ hơn: ba chỉ số của cùng một dataset thường cùng dấu, nên số cột độc lập hiệu dụng gần 6. Khi đó thắng 5/6 cho P = 0.1094 và chỉ thắng 6/6 mới cho 0.0156. Bài học: "thắng nhiều cột" không phải bằng chứng nếu không biết các cột độc lập tới đâu. (Kiểm định dấu là phép nhị thức của chương 3; ví dụ này chỉ dạy cách đọc.)

% manim: DemChienThang

**Ví dụ giải mẫu 11.2 — suy ra cỡ test set từ phần trăm báo cáo.** Một độ chính xác báo cáo 85.37%. Trên một lần chạy và một tập test n câu, độ chính xác chỉ có thể là k/n. Dò n ≤ 3000: n nhỏ nhất có k/n làm tròn thành 85.37 là n = 41 (35/41); kế tiếp 82, 123, 164, 205, 246, 287, 294; tổng cộng 451 giá trị n. Với n = 1000 chỉ có 85.3 và 85.4 (bội của 0.1), nên "85.37% trên 1000 câu" **không tự nhất quán** nếu đó là một lần chạy trên một tập. Ngoại lệ hợp lệ: trung bình nhiều seed hoặc nhiều fold (5 seed trên 1000 câu cho bội của 0.02: 85.36 và 85.38, vẫn không ra 85.37; 10 seed thì có). Vì thế phép kiểm phải nói rõ giả định ("một lần chạy, một tập") và chỉ dùng để đặt câu hỏi cho tác giả.
Tương tự, tính lại trung bình: các cột 88.1, 91.4, 76.3, 84.0 có trung bình 84.95; bảng ghi 85.2 thì lệch 0.25. Có thể là trung bình có trọng số theo cỡ tập, cũng có thể là lỗi.

**Đọc bảng.** Đếm cột độc lập chứ không đếm cột; đặt cạnh mỗi hiệu số cỡ test và phương sai (hiệu 0.3 điểm trên vài nghìn câu thường nằm trong nhiễu); kiểm số tham số và ngân sách của các hàng có công bằng không.

**Bối cảnh của lĩnh vực** (chỉ làm từ khóa, †): Lipton và Steinhardt (2018) về các xu hướng đáng lo trong học thuật ML; Melis và cộng sự (2017) về LSTM cũ được tune kỹ; Henderson và cộng sự (2018, RL) về phương sai giữa seed. Không dẫn con số của chúng.

**Thói quen nghề.** Mỗi paper một trang ghi chú theo mẫu cố định: claim, evidence, gaps, questions, what I would run. Năm paper nền của đề tài nên có đủ trang này.

**Vì sao ML cần cái này.**

- Nhiều "cải tiến" công bố nhỏ hơn nhiễu hoặc đến từ baseline yếu; người đọc giỏi phân biệt được.
- Phỏng vấn research hay đưa một abstract cùng một bảng và hỏi "bạn phản biện thế nào"; bản đồ khẳng định – bằng chứng là câu trả lời có cấu trúc.
- Lỗi bạn nhìn ra ở người khác là lỗi bạn tránh ở mình.

## 11.2 Thiết kế thí nghiệm: giả thuyết, baseline mạnh, ngân sách, ablation

Thí nghiệm tốt bắt đầu từ một giả thuyết có thể sai, và so sánh công bằng nghĩa là mọi biến khác đã được tối ưu hoặc cố định như nhau. Phần lớn kết quả yếu hỏng ở chỗ này, không phải ở thống kê.

**Giả thuyết có thể sai.** "Thành phần X làm tăng chỉ số M trên phân phối D, ít nhất δ điểm; nếu hiệu số theo kiểm định đã chọn nhỏ hơn δ thì giả thuyết bị bác." Viết **kế hoạch phân tích trước khi chạy**: chỉ số chính, tập dùng để chọn, số seed, phép so sánh chính, ngưỡng. Mọi so sánh thêm sau đó gắn nhãn "khám phá" (đa kiểm định, chương 3). Thí nghiệm khám phá và thí nghiệm kiểm chứng không trộn trong một bảng.

**Ba loại siêu tham số** (Google Tuning Playbook, đã đọc): *scientific* là cái ta đang đo tác dụng; *nuisance* là cái phải tối ưu riêng cho từng giá trị scientific thì so sánh mới công bằng; *fixed* là cái ta cố định và nó trở thành điều kiện hạn chế của kết luận. Ví dụ trong playbook: muốn biết thêm tầng ẩn có giảm lỗi không thì số tầng là scientific, learning rate là nuisance (tối ưu lại cho từng độ sâu), hàm kích hoạt có thể là fixed. Đây là ngôn ngữ chuẩn để nói "công bằng" nghĩa là gì: **nuisance phải được tune cho từng nhánh**.

**Baseline mạnh gồm ba tầng:** (a) baseline đơn giản (lớp đa số, bag-of-words; trong NLI thì mô hình chỉ nhìn giả thuyết, Gururangan và cộng sự 2018†, cho thấy dữ liệu có tạo tác annotate); (b) baseline mạnh nhất đã công bố, **chạy lại cùng code, cùng dữ liệu, cùng ngân sách tune** thay vì chép số từ paper khác; (c) chính phương pháp của bạn với thành phần mới tắt đi. Bằng chứng ngoài đời, chỉ nêu tên (†): Melis 2017 (LSTM tune kỹ), Yang và Lin 2019 (baseline yếu), Narang và cộng sự 2021 (sửa đổi Transformer không chuyển được qua cài đặt khác), Choi và cộng sự 2019 (so sánh optimizer phụ thuộc việc tune).

**Ví dụ giải mẫu 11.3 — baseline chưa tune làm hiệu ứng phình ra.** "Phương pháp" = dropout 0.5 + weight decay 1.0 (AdamW); baseline = cùng MLP không có hai thứ này. Lưới learning rate ∈ {10⁻⁴, 3·10⁻⁴, 10⁻³, 3·10⁻³, 10⁻², 3·10⁻²} × số bước ∈ {40, 150, 400}, chọn theo độ chính xác validation (4000 mẫu, 3 seed), báo cáo test (4000 mẫu, 10 seed, mean ± std, đơn vị điểm phần trăm).

| Kịch bản | Baseline | Phương pháp | Chênh |
| --- | --- | --- | --- |
| i. baseline dùng lr 3·10⁻² ("chép từ nơi khác"), phương pháp lr 3·10⁻³ | 68.85 ± 0.67 | 74.02 ± 0.27 | **+5.17** |
| ii. tune lr cho cả hai (150 bước) | 73.73 ± 0.93 (lr 10⁻³) | 74.02 ± 0.27 | **+0.29** |
| iii. tune lr × số bước cho cả hai | 73.73 ± 0.93 | 74.25 ± 0.40 | **+0.52** |

Hiệu ứng thật có (nhỏ), nhưng khoảng 90% của "5 điểm" nằm ở baseline yếu (kịch bản iii: 89.9%). Chú ý baseline ở kịch bản ii có std 0.93, lớn gấp ba std của phương pháp: tôi chưa kiểm vì sao (chưa đo std theo seed ở từng lr).

% manim: BaselineBiDoiXu

**Tune công bằng ngân sách.** Cùng số lần thử, cùng không gian tìm kiếm theo từng siêu tham số, cùng tập chọn, cùng số seed. Khi không gian lớn, dùng tìm kiếm ngẫu nhiên hoặc quasi-random thay lưới (Bergstra và Bengio 2012†; playbook khuyên quasi-random). Playbook (đã đọc) cho thấy trên một ví dụ ResNet-50 rằng khoảng tứ phân vị của kết quả tốt nhất rất rộng với 6 lần thử và hẹp hơn nhiều với 20 lần, và chênh lệch giữa các "study may rủi" có thể lớn hơn chênh lệch giữa các seed. Lấy cấu trúc, đừng lấy con số ImageNet cho NLP.

Cách báo cáo đúng là **hiệu năng kỳ vọng theo ngân sách** (expected validation performance, ý của Dodge và cộng sự 2019†): vẽ "tốt nhất sau k lần thử" theo k thay vì một điểm. Với điểm validation sắp tăng v₍₁₎ ≤ … ≤ v₍ₙ₎ và k lần rút có hoàn lại, max rơi vào thứ hạng i với xác suất (i/n)ᵏ − ((i−1)/n)ᵏ (vì max ≤ v₍ᵢ₎ khi cả k lần rút đều nằm trong i giá trị đầu). Vậy:

$$
\mathrm{EVP}(k) = \sum_{i=1}^{n} v_{(i)} \left[ \left(\tfrac{i}{n}\right)^{k} - \left(\tfrac{i-1}{n}\right)^{k} \right]
$$

Công thức này do tôi tự dẫn và kiểm bằng Monte Carlo (20 000 lần rút, khớp tới chữ số thứ tư), không chép từ paper. Về bản chất nó là kỳ vọng của max theo phân phối thực nghiệm của các điểm validation, cùng dạng với đại lượng expected validation performance của Dodge và cộng sự 2019† [cần kiểm chứng khi mở được paper]; đừng coi đây là một công thức khác.

**Ví dụ giải mẫu 11.4 — cùng ngân sách hay không.** Hai không gian tìm kiếm ngẫu nhiên, mỗi bên 60 lần thử (một seed mỗi lần, validation 4000 mẫu). A (baseline): lr log-đều trong [10⁻⁴, 10⁻¹·⁵], số bước ∈ {40, 150, 400}. B (phương pháp): thêm dropout ∈ [0, 0.6] và weight decay ∈ {0, 0.1, 1.0}.

| k | 1 | 2 | 4 | 8 | 16 | 32 | 60 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EVP của A | 0.6287 | 0.6942 | 0.7166 | 0.7267 | 0.7329 | 0.7368 | 0.7389 |
| EVP của B | 0.6155 | 0.6990 | 0.7267 | 0.7339 | 0.7375 | 0.7396 | 0.7408 |

So A với k = 4 (0.7166) và B với k = 60 (0.7408): chênh **2.4 điểm**. Cùng k = 60: chênh **0.2 điểm**. Cùng k = 4: chênh 1.0. Ở k = 1 chênh −1.3 điểm (0.6155 so với 0.6287) là nhiễu: mỗi lần thử có std khoảng 0.15, nên sai số chuẩn của hiệu hai trung bình 60 lần thử khoảng 3 điểm, và dựng lại B bằng dòng ngẫu nhiên khác thì dấu đảo (B 0.6335, A 0.6287). Cái ổn định là hình dạng đường cong và chênh 2.4 so với 0.2–0.4 điểm khi cùng k (cùng k = 60 là 0.4 với dòng ngẫu nhiên khác, nên đừng đọc 0.2 quá chính xác). Độ chính xác test của cấu hình chọn theo validation trong k lần đầu (5000 lần bootstrap): A 0.7242 (k = 4) và 0.7426 (k = 32); B 0.7347 và 0.7477. Giới hạn: một seed mỗi lần thử, không gian tự chọn; ví dụ chỉ cách đọc đường cong, không kết luận về dropout.

% manim: NganSachCongBang

**Ablation.** *Remove-one* (bỏ từng thành phần khỏi mô hình đủ) và *add-one* (thêm từng thành phần vào baseline) trả lời hai câu hỏi khác nhau. Khi các thành phần tương tác, chúng cho số khác nhau. Thiết kế giai thừa 2³ với nhiều seed rẻ hơn lập luận bằng lời. Ablation phải tune lại nuisance sau khi bỏ thành phần; ghi rõ ablation nào đã tune lại.

**Ví dụ giải mẫu 11.5 — ablation giai thừa 2³.** Ba thành phần: dropout 0.5 (D), weight decay 1.0 (W), LayerNorm giữa Linear và ReLU (L). lr 3·10⁻³, 150 bước, 10 seed, test 4000 mẫu. Độ chính xác (%), thứ tự (D, W, L): (0,0,0) 70.46 ± 0.42; (0,0,1) 70.40 ± 0.39; (0,1,0) 71.42 ± 0.33; (0,1,1) 71.14 ± 0.47; (1,0,0) 73.52 ± 0.30; (1,0,1) 72.30 ± 0.54; (1,1,0) 74.02 ± 0.27; (1,1,1) 72.66 ± 0.36.

| Thành phần | Remove-one (từ mô hình đủ 111) | Add-one (vào baseline 000) |
| --- | --- | --- |
| Dropout | +1.52 | +3.06 |
| Weight decay | +0.36 | +0.96 |
| LayerNorm | **−1.35** | −0.06 |

Tổng ba hiệu ứng add-one là 3.96, nhưng mô hình đủ chỉ hơn baseline 2.20. LayerNorm "vô hại khi đứng một mình" (−0.06) nhưng bỏ nó khỏi mô hình đủ thì tăng 1.35 điểm: tương tác âm với dropout. Tương tác D × W (ở L = 0) là −0.47. Một ablation chỉ báo một chiều che mất tất cả những điều này. Và "LayerNorm tệ" chỉ đúng trong thiết lập này (chưa tune lại lr, mô hình nhỏ, tập nhỏ). Vị trí cũng đổi số: đặt LayerNorm sau ReLU thì cả tám ô đổi (mô hình đủ 71.53 thay vì 72.66; add-one −0.74 thay vì −0.06; remove-one −2.49 thay vì −1.35).

% manim: AblationTuongTac

**Kiểm soát biến.** Cùng dữ liệu, tokenizer, code đánh giá, tập seed; nói rõ cùng ngân sách tính toán hay cùng số tham số (hai lựa chọn cho kết luận khác nhau); ghi mọi thay đổi ngoài biến scientific.

**Vì sao ML cần cái này.**

- Một "kết quả" chỉ có nghĩa nếu mọi nuisance được xử lý như nhau cho hai nhánh; nếu không, nó đo độ may rủi của việc tune.
- Báo cáo EVP(k) thay cho một điểm cho phép người đọc đặt hai phương pháp ở cùng ngân sách, đúng điều người dùng thật quan tâm (tốn bao nhiêu thử nghiệm để đạt mức này).
- Ablation hai chiều là cách rẻ nhất để biết một thành phần có thật sự cần hay chỉ đang đi cùng thành phần khác.

## 11.3 Độ lặp lại: seed, phương sai, mean ± std, khoảng tin cậy

Một "seed" chỉ là một nguồn trong nhiều nguồn phương sai, và thường không phải nguồn lớn nhất. Hiểu phương sai đến từ đâu thì mới biết cần chạy bao nhiêu lần và con số nào được phép báo cáo.

**Ba nguồn** (Tuning Playbook, đã đọc): *trial variance* giữa các lần chạy cùng siêu tham số khác seed (khởi tạo, thứ tự dữ liệu, mặt nạ dropout, cả thứ tự phép tính song song); *study variance* do quy trình chọn siêu tham số; *data sampling variance* do chia dữ liệu. Bouthillier và cộng sự (2021†) khuyên ngẫu nhiên hóa nhiều nguồn cùng lúc; Dodge và cộng sự (2020†) là ví dụ NLP khi fine-tune BERT (khởi tạo đầu phân loại, thứ tự dữ liệu). Nguồn thứ tư mà tôi đo được ở dưới và hay bị quên: phương sai do **lấy mẫu tập test**.

**Ví dụ giải mẫu 11.6 — phương sai đến từ đâu; mean ± std và CI khi n = 5.** Cấu hình dropout + weight decay, lr 3·10⁻³, test 1000 mẫu (một tập cố định), 30 lần lặp mỗi nguồn.

| Chỉ đổi | Trung bình | Std |
| --- | --- | --- |
| seed (khởi tạo + thứ tự dữ liệu), cùng train set | 76.64 | **0.54** |
| tập train (mẫu mới), cùng seed | 75.92 | **1.39** |
| cả hai | 75.53 | **1.38** |
| tập test (1000 mẫu mới), cùng mô hình | 73.22 | **1.31** |

SE nhị thức của n = 1000 tính ở độ chính xác thật p ≈ 0.737 là 1.39, khớp 1.31 (không tính ở 0.766 của tập cố định: SE phải dựa vào giá trị thật). Mỗi nguồn lấy mẫu dữ liệu lớn gấp khoảng 2 đến 2.5 lần nguồn seed (std từ 30 mẫu sai khoảng 13%; dòng ngẫu nhiên khác cho 1.07, 1.25, 1.52). Trên 40 000 mẫu mới, cùng 30 mô hình cho độ chính xác "thật" 73.70 (std theo seed 0.30), trong khi test 1000 mẫu cố định cho 76.64: lệch **+2.94 điểm, bằng 2.1 SE nhị thức**. Đó là tập test đầu tiên của luồng seed 1, tôi không chọn, nhưng là một lần rút ở đuôi: đo mô hình seed 0 trên 300 tập test mới 1000 mẫu, |lệch| điển hình khoảng 1.1 điểm, đúng như lý thuyết 𝔼|Z| = √(2/π) ≈ 0.8 SE ≈ 1.12 (tái lập: 1.26 với 30 tập, 1.09 với 300, 1.13 với 3000), và khoảng 3.5% tập lệch từ 2.1 SE. Độ lệch điển hình cỡ 1 SE; ví dụ minh họa chiều của vấn đề, không phải cỡ điển hình.

Báo cáo 5 seed đầu: 76.4, 76.6, 77.4, 76.8, 76.0. Trung bình 76.64; std mẫu (n − 1) **0.518**, std tổng thể (n) 0.463, chênh khoảng 11.8% nên phải nói dùng cái nào; SE = 0.518/√5 = 0.232. CI 95% dạng t (hệ số 2.776): [76.00, 77.28]; dạng 1.96: [76.19, 77.09]. Độ phủ thật, mô phỏng 200 000 mẫu chuẩn: n = 3 cho 80.9% (1.96) so với 94.9% (t); n = 5: 87.8% so với 95.0%; n = 10: 91.7% so với 95.0%; n = 30: 94.1% so với 95.1%.

Ba bài học. (1) Với n nhỏ phải dùng phân vị t với n − 1 bậc tự do; bootstrap với 5 điểm chỉ tái dùng chính 5 điểm đó, nên cũng hẹp (bootstrap phân vị 95% phủ khoảng 75% khi n = 3, 83% khi n = 5, 90% khi n = 10). (2) CI theo seed rất hẹp quanh 76.64 nhưng **không bao phủ** 73.70 (đa số lần cũng vậy: nửa độ rộng CI chỉ 0.62 so với độ lệch điển hình 1.09). Seed chỉ gom phương sai huấn luyện; phương sai lấy mẫu test vẫn còn. (3) So sánh hai mô hình trên **cùng test set** (cặp) vẫn hợp lệ vì độ lệch chung triệt tiêu; số tuyệt đối thì không.

% manim: HaiKhoangTinCay

**Best-of-k là thiên lệch.** Kỳ vọng max của k biến chuẩn là 0.846σ (k = 3), 1.163σ (k = 5), 1.539σ (k = 10), 1.867σ (k = 20) (tích phân số; Monte Carlo 200 000 lần cho 0.846, 1.162, 1.538, 1.868, lệch ở chữ số thứ ba). Với σ = 0.52, "best of 5" cộng khoảng 0.6 điểm không có thật. Trong 5 seed ở trên: max 77.4 so với mean 76.64, cộng 0.76.

**Ví dụ giải mẫu 11.7 — bao nhiêu seed là đủ, và ghép cặp có luôn giúp không.** Để phát hiện chênh lệch δ khi σ = 0.54 (chỉ phần seed), mức 5% hai phía, công thức xuất phát là n ≈ 2(z₍α/2₎ + z₍β₎)²σ²/δ², với z₍α/2₎ = 1.96, z₍β₎ = 0.84 (độ mạnh 0.8). Công thức dùng z nên hơi thấp khi n nhỏ. Độ mạnh mô phỏng (Welch, 20 000 lần, dao động ±0.003):

| δ | n theo công thức | Độ mạnh khi n = 3 / 5 / 10 / 20 | Độ mạnh tại n công thức |
| --- | --- | --- | --- |
| 1.0 | 5 | 0.32 / 0.70 / 0.97 / 1.00 | 0.70 |
| 0.5 | 19 | 0.10 / 0.23 / 0.49 / 0.81 | 0.79 |
| 0.3 | 51 | 0.06 / 0.11 / 0.21 / 0.40 | 0.79 |

Với độ mạnh 0.8, 5 seed chỉ phát hiện được chênh lệch từ khoảng 2σ và 3 seed từ khoảng 3σ (t gộp phương sai: 2.0σ và 3.1σ; Welch như bảng: 2.1σ và 3.5σ); với δ = 0.5 độ mạnh chỉ 0.23. Và σ = 0.54 là **chỉ phần seed**: nếu cũng ngẫu nhiên hóa tập train (σ = 1.38) thì số seed gấp (1.38/0.54)² ≈ 6.5 lần. Card và cộng sự (EMNLP 2020, "With Little Power Comes Great Responsibility"; repo đi kèm đã đọc README) phân tích độ mạnh thống kê trong NLP và cung cấp notebook tính power cho độ chính xác và BLEU.

Ghép cặp theo seed **có lúc giúp rất nhiều, có lúc không**. D+W so với D riêng chỉ khác weight decay: cùng kiến trúc, cùng khởi tạo, cùng mặt nạ dropout cho cùng seed. Tương quan giữa 10 cặp là **0.923**; std của hiệu theo cặp là **0.117** so với 0.404 nếu coi độc lập (√(0.27² + 0.30²); 0.409 với std chưa làm tròn); hiệu 0.49 (số thật 0.4925) có CI theo cặp [0.41, 0.58] (t, 9 bậc tự do) thay vì [0.22, 0.76] của Welch. Trái lại, "đủ bộ 111" so với D riêng (thêm LayerNorm, đổi số tham số nên dòng số ngẫu nhiên khác): hiệu theo cặp −0.86 với std 0.41, gần bằng giá trị độc lập 0.47; ghép cặp gần như không giảm gì. Đừng hứa "ghép cặp luôn giảm phương sai", cũng đừng bỏ qua nó: kiểm tương quan trước.

**Vì sao ML cần cái này.**

- Số seed không phải thói quen mà là phép tính độ mạnh: 3–5 seed (chuẩn thường gặp) chỉ phát hiện nổi hiệu số lớn hơn khoảng 2–3.5σ (5 seed: 2.0σ t gộp, 2.1σ Welch; 3 seed: 3.1σ và 3.5σ), nên "không thấy khác biệt" với 3 seed thường không chứng minh gì.
- Báo cáo mean ± std phải nói ± là std hay SE, mẫu số n − 1, và n. Hiệu số kèm CI thông tin hơn hai mean ± std đứng cạnh nhau.
- Fine-tune LLM thường chỉ chạy được vài seed; khi đó càng phải dùng t, ghép cặp có kiểm tương quan, và nói thẳng độ mạnh thấp.

## 11.4 Rò rỉ dữ liệu và đánh giá đúng

Rò rỉ là mọi đường thông tin từ test (hoặc từ tương lai) chảy vào huấn luyện hoặc chọn mô hình; nó làm số đo lạc quan mà không để lại dấu vết nào trong train loss. Lỗi nằm ở thứ tự quy trình, không ở mô hình.

**Sáu loại rò rỉ** (phân loại của riêng tôi; Kapoor và Narayanan 2022† có khảo sát khoa học ứng dụng ML, tôi chưa mở lại):

1. **Tiền xử lý hoặc chọn đặc trưng trước khi chia**: chuẩn hóa, PCA, chọn đặc trưng, vocab hoặc tokenizer huấn luyện trên toàn bộ (VD 11.8).
2. **Trùng lặp và gần trùng** giữa train và test (câu giống, tài liệu cùng nguồn). Lee và cộng sự 2021† (khử trùng lặp dữ liệu huấn luyện LM), Lewis và cộng sự 2020† (trùng lặp câu hỏi–đáp trong các bộ QA mở), Elangovan và cộng sự 2021† (nhớ so với tổng quát hóa).
3. **Nhóm và thời gian**: nhiều mẫu của cùng người hoặc cùng tài liệu chia vào hai phía; dữ liệu tương lai nằm trong train. Chia theo nhóm hoặc theo thời gian.
4. **Nhãn hoặc đặc trưng chứa đáp án**, tạo tác annotate (hypothesis-only trong NLI, Gururangan 2018†).
5. **Dùng lại test set** (adaptive overfitting): chọn mô hình, siêu tham số, thậm chí ý tưởng theo test nhiều lần (VD 11.10). Nền lý thuyết: Dwork và cộng sự 2015† (reusable holdout); thực nghiệm ngược: Recht và cộng sự 2018† dựng lại bộ test CIFAR-10 mới. Tôi không dẫn mức tụt điểm của họ.
6. **Nhiễm benchmark với LLM**: dữ liệu test nằm trong dữ liệu tiền huấn luyện. GPT-3 (Brown 2020†) có một phân tích chồng lấn; Magar và Schwartz 2022†, Sainz và cộng sự 2023† bàn về đo nhiễm cho từng benchmark (VD 11.9).

**Ví dụ giải mẫu 11.8 — chọn đặc trưng trước khi chia: 0.867 trên nhãn ngẫu nhiên.** n = 100 mẫu, p = 5000 đặc trưng Gauss độc lập, nhãn nhị phân **ngẫu nhiên** (độ chính xác thật phải là 0.5). Chọn 20 đặc trưng có |tương quan| lớn nhất với nhãn, rồi hồi quy bình phương tối thiểu có L2 (λ = 1) và kiểm định chéo 5 fold; 30 lần lặp.

| Quy trình | Độ chính xác CV | Ghi chú |
| --- | --- | --- |
| Chọn đặc trưng trên toàn bộ rồi mới CV | **0.867** (std 0.043) | cả 30 lần đều > 0.65 |
| Chọn đặc trưng trong từng fold, chỉ trên train của fold | **0.491** (std 0.066) | không lần nào > 0.65 |
| Mô hình rò rỉ chạy trên 2000 mẫu hoàn toàn mới | **0.499** | |

Quét (p, k) với 15 lần lặp, độ chính xác "rò rỉ" so với "đúng": p = 50, k = 5: 0.632 so với 0.502; p = 500, k = 20: 0.766 so với 0.485; p = 5000, k = 5: 0.773 so với 0.493; p = 5000, k = 20: 0.875 so với 0.495. Càng nhiều đặc trưng ứng viên và càng ít mẫu, rò rỉ càng nặng. Không có dấu "overfit" nào lộ ra trong tập train; chỉ nhãn ngẫu nhiên vạch ra. Nối chương 6 (chuẩn hóa val bằng thống kê của chính nó): cùng một phép thử khói "nhãn ngẫu nhiên phải cho 0.5".

% manim: RoRiChonLocTruocKhiChia

**Ví dụ giải mẫu 11.9 — nhiễm benchmark: ảnh hưởng và giới hạn của phát hiện n-gram.** (a) *Ảnh hưởng.* Mô hình ghi nhớ train (lr 3·10⁻³, 600 bước, không dropout, không weight decay; độ chính xác train 100%). Test set 600 mẫu từ luồng riêng, thay một tỉ lệ c bằng bản sao của mẫu train. Độ chính xác test (5 seed): c = 0: 68.53 ± 0.96; 1%: 68.77 ± 1.00; 5%: 70.00 ± 1.05; 20%: 74.13 ± 0.74; 50%: 83.73 ± 0.49. Nhiễm 5% đã cộng khoảng 1.5 điểm, cỡ nhiều "cải tiến" công bố. (Mô hình rất nhỏ dùng ghi nhớ; với LLM mức ghi nhớ khác và tôi chưa đo.)

(b) *Phát hiện bằng chồng lấn n-gram.* Token ngẫu nhiên, từ vựng 5000, tài liệu dài L = 60; bản sao bị đổi mỗi token với xác suất q. Bản sao còn bị bắt nếu tồn tại một đoạn n token liền nhau không bị đổi. Xác suất đó tính chính xác bằng quy hoạch động trên độ dài đoạn không đổi hiện tại (kiểm bằng mô phỏng):

| n \ q | 0.02 | 0.05 | 0.1 | 0.2 | 0.3 |
| --- | --- | --- | --- | --- | --- |
| 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.988 |
| 8 | 1.000 | 1.000 | 0.999 | 0.944 | 0.675 |
| 13 | 1.000 | 0.995 | 0.923 | **0.487** | 0.141 |

Xấp xỉ "các cửa sổ độc lập" là **sai** (các cửa sổ chồng lấn nên tương quan): tài liệu 200 token, n = 13, q = 0.2 cho 0.918, xấp xỉ cho 0.934. (c) *Dương tính giả.* Văn bản kiểu Zipf (số mũ 1.1, từ vựng 5000), 2000 tài liệu train và 100 tài liệu test **độc lập**: tỉ lệ bị gắn cờ là 1.00 (n = 2), 0.13 (n = 5), 0.00 (n = 13). Chọn n là đánh đổi: n dài bỏ sót bản sửa, n ngắn gắn cờ nhầm cụm từ phổ biến.

**Ví dụ giải mẫu 11.10 — dùng lại test set.** (a) K mô hình có độ chính xác thật đều 0.80, test 1000 mẫu (SE 0.0126). Chọn mô hình tốt nhất trên test rồi báo cáo con số đó: độ chính xác quan sát lớn nhất (4000 lần mô phỏng) cao hơn con số trên tập mới là K = 1: +0.00 điểm; K = 5: **+1.44**; K = 20: +2.33; K = 100: **+3.12**; K = 1000: +4.06. Khớp xấp xỉ SE × E[max của K biến chuẩn] = 1.47, 2.36, 3.17, 4.10 điểm (E[max] = 1.163, 1.867, 2.508, 3.241 cho K = 5, 20, 100, 1000). Thiên lệch tăng chậm theo K nhưng không bao giờ về 0.

(b) *Tấn công thích nghi* (kiểu đồng thuận). Nhãn test hoàn toàn ngẫu nhiên (n = 1000); sinh m bộ phân loại ngẫu nhiên, giữ những bộ có độ chính xác test > 0.5, biểu quyết đa số. Độ chính xác trên test: m = 10: 0.520; m = 100: 0.571; m = 1000: **0.714**; m = 10 000: **0.964**; trên dữ liệu mới luôn khoảng 0.50. Đây là cận trên có chủ đích (không tác giả nào cố ý làm thế), nhưng cho thấy nhiều vòng "chọn theo test" đẩy được số liệu cao tùy ý mà không có tín hiệu thật.

% manim: BestOfK

**Vệ sinh đánh giá.** (a) Chia trước, xử lý sau: mọi bước có học (chuẩn hóa, vocab, chọn đặc trưng) fit trên train rồi áp lên val và test. (b) Chọn trên validation, chạm test **một lần** ở cuối. (c) Kiểm trùng lặp train–test bằng băm và n-gram. (d) Báo cáo khoảng cách validation–test. (e) Với benchmark công khai lâu năm, giả định đã bị tune quá mức; dùng bộ giữ kín hoặc bộ mới. (f) Metric phải tái lập (BLEU: sacreBLEU in signature, mục 11.5). (g) Phép thử khói: nhãn ngẫu nhiên phải cho 0.5, loss không được dưới sàn ln C. Bowman và Dahl (2021†) bàn về thiết kế benchmark (đọc thêm).

**Vì sao ML cần cái này.**

- Với LLM, "test" thường đã nằm đâu đó trong dữ liệu tiền huấn luyện; con số trên benchmark công khai có thể đo trí nhớ nhiều hơn khả năng.
- Val 0.95 mà test 0.60 là câu hỏi phỏng vấn kinh điển; danh sách sáu loại trên là thứ tự bạn kiểm.
- Kiểm n-gram là công cụ thực dụng nhưng có giới hạn đo được: nó bỏ sót bản chép đã sửa nhiều, và gắn cờ nhầm nếu n quá ngắn.

## 11.5 Viết, trình bày kết quả và tái lập

Viết kết quả là bước cuối của thí nghiệm: mỗi câu phải nói đúng điều đã đo, không hơn. Lỗi ở các mục trên sẽ lộ ra, hoặc bị che đi, ở chính cách viết bảng và câu chữ.

**Cấu trúc bài.** Abstract (khẳng định có số, một câu giới hạn); mở bài (vấn đề, khoảng trống, đóng góp liệt kê kiểm được); phương pháp; thí nghiệm (câu hỏi nghiên cứu → thiết lập → kết quả → phân tích); related work; limitations; kết luận. Mỗi đóng góp trong mở bài phải trỏ được tới một bảng hoặc hình.

**Bảng.** Nêu số seed và ý nghĩa của ±. Làm tròn theo SE: báo cáo tới chữ số mà SE còn phân biệt được. In đậm chỉ khi hơn có kiểm chứng (quy tắc chọn trước khi nhìn số), không theo max. Cùng số chữ số, ghi đơn vị và hướng tốt (↑ hay ↓), thêm cột "ngân sách tune" và "số tham số" nếu so sánh liên quan. Hiệu số kèm CI ("Δ (CI)") thông tin hơn hai mean ± std đứng cạnh nhau.

**Ví dụ giải mẫu 11.11 — làm tròn theo SE, in đậm có kiểm chứng.** Từ 10 seed của VD 11.5 (test 4000 mẫu; SE = std/√10; Welch so với mô hình tốt nhất):

| Cấu hình | Mean | Std | SE | Trình bày | Chênh so với tốt nhất, p (Welch) |
| --- | --- | --- | --- | --- | --- |
| cơ sở | 70.46 | 0.42 | 0.13 | 70.5 ± 0.4 | 3.56, < 10⁻⁴ |
| + wd | 71.42 | 0.33 | 0.10 | 71.4 ± 0.3 | 2.59, < 10⁻⁴ |
| + dropout | 73.52 | 0.30 | 0.10 | 73.5 ± 0.3 | 0.49, 0.0013 |
| + dropout + wd | 74.02 | 0.27 | 0.09 | **74.0 ± 0.3** | — |
| + dropout + wd + LN | 72.66 | 0.36 | 0.12 | 72.7 ± 0.4 | 1.35, < 10⁻⁴ |

Hai chữ số thập phân (74.02) cho ấn tượng chính xác không có: SE 0.09 nghĩa là chữ số thứ hai là nhiễu. Hiệu số quan trọng nhất là "+ dropout + wd" hơn "+ dropout" 0.49, CI 95% [0.22, 0.76] khi không ghép cặp và [0.41, 0.58] khi ghép cặp theo seed (cùng kiến trúc, tương quan 0.923, VD 11.7). Có 4 phép so với "tốt nhất"; p rất nhỏ nên Bonferroni (α/4 = 0.0125) không đổi kết luận (chương 3).

**Hình.** Trục có nhãn và đơn vị; vạch lỗi nói rõ là std, SE hay CI; không cắt trục y mà không báo; đọc được ở thang xám; đường cong nhiều seed vẽ dải (min–max hoặc ± std) cùng đường trung bình; chú thích tự đủ.

**Related work** định vị chứ không liệt kê: nhóm theo ý tưởng, nói điểm khác biệt có kiểm chứng, không bỏ paper đối đầu trực tiếp, trích bản hội nghị chứ không chỉ preprint. **Limitations** cụ thể và kiểm được ("chỉ tiếng Anh, 3 dataset, mô hình ≤ 1B tham số, 5 seed") thay vì "cần nghiên cứu thêm"; nhiều hội nghị có xu hướng đòi mục này và checklist tái lập (chỉ là xu hướng, tôi chưa kiểm từng hội nghị). **Ngôn ngữ hiệu chỉnh theo bằng chứng:** "cho thấy" (đã kiểm có đối chứng), "gợi ý" (tương quan), "chúng tôi giả thuyết" (chưa kiểm).

**Kết quả âm** vẫn là thông tin nếu thiết kế đúng: ghi vào phần phân tích. Cân bằng: đừng dùng "kết quả âm" để che thiết kế yếu; khi độ mạnh thấp (VD 11.7), "không thấy khác biệt" không chứng minh gì.

**Tái lập.** Mỗi lần chạy lưu: cấu hình đầy đủ (YAML hoặc JSON), seed, băm commit, phiên bản thư viện (`pip freeze`), phần cứng, thời gian chạy, lệnh tái lập, phiên bản và băm của dữ liệu, checkpoint tốt nhất theo validation. Tuning Playbook (đã đọc) có mục theo dõi thí nghiệm (tên study, liên kết cấu hình, ghi chú, số lần thử, hiệu năng validation tốt nhất, lệnh tái lập); câu "Untracked experiments might as well not exist" tóm đúng lý do. Template README của Papers with Code (đã đọc) gồm Requirements, Training, Evaluation, Pre-trained Models, Results. sacreBLEU (README đã đọc) in chuỗi signature kiểu `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0` và khuyên công bố signature trong paper, vì BLEU khác tokenization hay phiên bản thì không so sánh được. Pineau và cộng sự (2021†) là nguồn cho checklist tái lập; Gebru và cộng sự (2018†, datasheet) và Mitchell và cộng sự (2018†, model card) là khuôn tài liệu hóa dữ liệu và mô hình.

**Ví dụ giải mẫu 11.12 — cùng seed, bit giống nhau; nhiễu nhỏ bị khuếch đại.** Trên CPU, MLP nhỏ này, cùng seed: sai khác tham số và logit bằng **0** giữa hai lần chạy một luồng, và giữa 1 luồng với 2 luồng (cả 150 và 1500 bước, không khác nhãn nào trên 4000 mẫu); đánh giá một lần 4000 mẫu so với chia 7 mẫu: sai khác logit 0. Đừng suy ra "đa luồng luôn deterministic": mô hình nhỏ nên phép nhân ma trận không đổi thứ tự cộng, tôi chưa đo mô hình lớn hay GPU. Còn khi có nhiễu số, kết quả trượt và trượt khuếch đại: float32 so với float64 cùng seed (dropout 0.5, lr 3·10⁻³, wd 1.0) lệch logit lớn nhất 1.88·10⁻⁵ ở 150 bước, không đổi nhãn nào; ở 1500 bước lệch 0.44 và đổi 33/4000 nhãn (độ chính xác 71.73 so với 71.95: đổi 0.2 điểm, nhỏ hơn std theo seed). Cả hai số này, và 1.88·10⁻⁵, đo trên cùng một tập 4000 mẫu lấy ngay sau tập train; tập 4000 mẫu kế tiếp cho 0.37 và 41 nhãn, cùng cấu hình, chỉ khác tập đo. Nếu tự tái lập, đừng đặt `torch.set_default_dtype(torch.float64)` trước khi khởi tạo mô hình: khi đó trọng số ban đầu đã khác (lệch logit cỡ 3.6 và khoảng 420 nhãn đổi ngay ở 150 bước), tức bạn đang so hai seed chứ không còn so độ chính xác số. Phải khởi tạo ở float32 rồi mới ép sang float64. Cộng nhiễu Gauss ε vào khởi tạo (mini-batch 32, dropout 0.5, lr 3·10⁻³, wd 1.0), sai khác tham số lớn nhất sau 50 / 300 / 1500 bước:

| ε | 50 bước | 300 bước | 1500 bước |
| --- | --- | --- | --- |
| 10⁻⁷ | 9.2·10⁻⁷ | 2.1·10⁻⁶ | **0.0515** |
| 10⁻⁵ | 0.0045 | 0.0123 | 0.0559 |
| 10⁻³ | 0.0124 | 0.0431 | 0.183 |

Từ 10⁻⁷ lên 0.0515 là khuếch đại khoảng 5·10⁵ lần: huấn luyện là hệ nhạy, và bất kỳ nhiễu số nào (cộng không giao hoán, phiên bản thư viện) hoạt động như một seed mới. Độ chính xác test chỉ đổi 72.45 → 72.65 (ε = 10⁻⁷) và 72.45 → 72.05 (ε = 10⁻³), tức nằm trong khoảng giữa các seed. Phân rã "seed" (12 giá trị mỗi nhóm, mini-batch 32, 300 bước): chỉ đổi khởi tạo std 0.32; chỉ đổi thứ tự batch + mặt nạ dropout 0.57; đổi cả hai 0.73 (gần √(0.32² + 0.57²) = 0.65). Một "seed" thật ra là nhiều nguồn. Trên GPU, các phép gom (atomic add) có thể gây hiệu ứng này một cách tự nhiên (tôi nhớ, chưa đo vì máy không có GPU).

**Vì sao ML cần cái này.**

- "Tôi đã đặt seed" không thay cho "tôi báo cáo nhiều seed": đặt seed chỉ cho lặp lại trong đúng môi trường, còn nhiễu nhỏ nào cũng thành một seed mới.
- Paper không có signature, seed, cỡ test thì người khác không thể kiểm.
- Câu chữ hiệu chỉnh là phần của độ tin cậy: cùng một số liệu, "cho thấy" và "gợi ý" là hai tuyên bố khác nhau.

**Thực tế tính toán.** Một lần huấn luyện MLP nhỏ này mất khoảng 0.15 giây; ở quy mô LLM các bài học vẫn đúng nhưng ngân sách để theo chúng thì không, khi đó hãy nói rõ giới hạn.

## Bài tập giấy bút

14 bài. Mục tiêu: hết ★ trong 1 buổi, ★★ trong 2 buổi, ★★★ cho Thứ Bảy. Bài nào sai, ghi vào sổ lỗi kèm lý do.

**★ Hiểu định nghĩa**

1. Giả thuyết: "pre-LN huấn luyện ổn định hơn post-LN". Phân loại thành scientific, nuisance hay fixed: vị trí LayerNorm, learning rate, số bước warmup, số tầng, seed, tokenizer. Với mỗi cái fixed, nói kết luận bị hạn chế thế nào.
2. Paper tuyên bố "tăng 1.2 F1 nhờ cơ chế X". Bảng chỉ có một lần chạy trên test 2000 câu, baseline lấy từ paper khác. Nêu ba ô trống của bản đồ khẳng định – bằng chứng và ô nào dễ đổi kết luận nhất.
3. Năm seed cho 71.2, 70.8, 71.9, 71.5, 70.6. Tính mean, std mẫu, std tổng thể, SE và CI 95% bằng t (hệ số 2.776). So với CI dùng 1.96.
4. Phân loại vào sáu loại rò rỉ của mục 11.4: (a) chuẩn hóa tính trên cả train + test; (b) huấn luyện BPE vocab trên toàn bộ corpus gồm cả test; (c) nhiều câu của cùng một người nói nằm cả hai phía; (d) chọn epoch theo độ chính xác test; (e) 3% câu test có trong corpus tiền huấn luyện.

**★★ Mức đề thi đầu vào**

5. Tính P(thắng ≥ 14 trên 18) và giá trị hai phía. Nếu số cột độc lập hiệu dụng chỉ là 6, cần thắng ít nhất mấy trên 6 để P một phía < 0.05? Thắng 5/6 thì sao?
6. Báo cáo "71.37% trên 500 câu, một lần chạy". Có tự nhất quán không? n nhỏ nhất cho phép 71.37%? Nếu là trung bình 5 seed trên 500 câu thì sao?
7. Năm điểm validation của năm lần thử: 0.60, 0.64, 0.70, 0.71, 0.74. Tính EVP(1) và EVP(2) bằng công thức mục 11.2. Diễn giải EVP(2).
8. σ = 0.8, cần phát hiện δ = 1.0 ở mức 5% hai phía, độ mạnh 0.8. Tính n theo công thức. Nếu chỉ có 5 seed, δ nhỏ nhất phát hiện được theo cùng công thức bằng bao nhiêu, và công thức này thiên về phía nào?
9. Hai phương pháp thật sự bằng nhau, σ giữa các seed là 0.5. Bên A báo cáo mean của 5 seed, bên B báo cáo best-of-5. Kỳ vọng chênh báo cáo? Nếu cả hai cùng best-of-5 thì sao, và như vậy đã công bằng chưa?
10. Một tài liệu L = 6 token bị chép, mỗi token giữ nguyên với xác suất 1/2 (q = 1/2). Tính tay xác suất còn ít nhất một đoạn 3 token liền nhau không đổi. Gợi ý: đếm chuỗi nhị phân độ dài 6 không có ba số 1 liền nhau.

**★★★ Mức phỏng vấn nghiên cứu**

11. Chứng minh công thức EVP(k) cho n giá trị phân biệt. Chứng minh EVP(1) là trung bình, EVP không giảm theo k, và EVP(k) tiến về max khi k → ∞.
12. K mô hình cùng độ chính xác thật p, test n mẫu, số đo xấp xỉ chuẩn với SE = √(p(1 − p)/n). Tính chính xác thiên lệch của "best-of-2" (gợi ý: E[max của hai biến N(0,1)] = 1/√π). Với n = 1000, p = 0.8 ra bao nhiêu điểm? Giải thích vì sao thiên lệch tăng theo K nhưng chậm, và √(2 ln K) có phải ước lượng tốt không.
13. Hiệu hai mô hình cùng seed: std lần lượt σA = 0.27, σB = 0.30, tương quan ρ = 0.923, n = 10 seed, hiệu trung bình 0.49. Tính std của hiệu theo cặp, CI 95% theo cặp (t với 9 bậc tự do là 2.262), và hệ số tiết kiệm số seed so với coi độc lập. Khi nào ρ ≈ 0, và hậu quả?
14. Viết kế hoạch phân tích trước khi chạy cho giả thuyết "loss mới L giúp tiếng ít tài nguyên": scientific, nuisance, fixed; baseline; ngân sách tune; chỉ số chính và phép kiểm; số seed theo độ mạnh khi σ seed là 0.54 và khi cũng ngẫu nhiên hóa tập train (σ = 1.38), δ = 0.5; điều kiện bác bỏ.

## Bài tính nhanh

15 bài, mỗi bài tối đa 2 phút, giấy bút, bấm giờ cả bộ. Mục tiêu: xong trong 25 phút, sai không quá 1 bài. Làm lại sau 3 ngày và sau 1 tuần.

1. P(thắng ≥ 4 trên 5 lần | đồng xu công bằng) bằng bao nhiêu?
2. Std mẫu và std tổng thể của (2, 4, 6).
3. Hệ số t hai phía 95% khi n = 3 (2 bậc tự do) là 4.303. Gấp mấy lần 1.96?
4. SE của độ chính xác 0.8 trên 400 câu, tính bằng điểm phần trăm.
5. Độ chính xác 62.5% trên một tập test, một lần chạy: n nhỏ nhất?
6. σ = 0.5. Best-of-5 cộng khoảng bao nhiêu điểm (E[max của 5 biến chuẩn] ≈ 1.163)?
7. n seed mỗi bên khi σ = 1, δ = 1, 5% hai phía, độ mạnh 0.8 (tổng z bằng 2.80)?
8. Macro average của 82.0, 76.0, 90.0, 84.0.
9. "85.37% trên 1000 câu, một lần chạy" có hợp lệ không?
10. EVP(2) của ba điểm validation 0.6, 0.7, 0.8.
11. K = 100 mô hình bằng nhau, p = 0.8, test 1000 mẫu (SE 1.26 điểm, E[max] ≈ 2.5): thiên lệch best-of-K?
12. Tổng ba hiệu ứng add-one là 4.0, cả bộ hơn baseline 2.5. Tương tác tổng cộng bằng bao nhiêu?
13. 10 seed, mean 74.02, std mẫu 0.27: nên in thế nào?
14. Tài liệu bị chép, mỗi token bị đổi với q = 0.2. Xác suất 13 token liền nhau không đổi?
15. Nếu cũng ngẫu nhiên hóa tập train, σ tăng từ 0.54 lên 1.38. Số seed cần tăng gấp mấy lần?

## Lab code và bài săn bug

Năm lab dưới đây đã chạy với Python 3, NumPy, SciPy 1.18 và PyTorch 2.14 CPU (1 luồng); số liệu có thể lệch nhẹ theo phiên bản, nên các assert dùng khoảng. Cách học đúng: tự viết từng lab trước, chỉ đối chiếu code mẫu khi kết quả lệch với cột "kết quả mong đợi".

| Lab | Việc phải làm | Kết quả mong đợi |
| --- | --- | --- |
| 1. Đọc số liệu | `sign_test_p(k, n)` và `consistent_sizes(pct)` | P(≥ 11/18) = 0.2403; P(≥ 14/18) = 0.0154; 85.37% có 451 giá trị n ≤ 3000 phù hợp, nhỏ nhất 41, 1000 không thuộc danh sách; 92.31% → 13 |
| 2. Hiệu năng kỳ vọng theo ngân sách | `evp(scores, k)` bằng thống kê thứ tự; kiểm Monte Carlo; random search 30 lần thử trên hai không gian | `evp` khớp Monte Carlo trong 2·10⁻³; EVP không giảm theo k; k = 4 / 30: A 0.717 / 0.733, B 0.725 / 0.739; B với 30 lần thử hơn A với 4 lần thử **+2.19 điểm**, nhưng cùng 30 lần thử chỉ **+0.54**; ở k = 1 chênh 0.2 điểm, nằm trong nhiễu (sai số chuẩn khoảng 4 điểm) nên không đọc dấu |
| 3. Phương sai và CI | `mean_ci`, `coverage`; 20 seed, test 1000 mẫu so với 20 000 mẫu mới | 76.62 ± 0.49 trên test 1000; độ phủ n = 5: t 0.951, 1.96 chỉ 0.879; độ chính xác mới 73.11, test 1000 lệch +3.52 điểm (2.5 SE) và giá trị mới nằm ngoài CI 5 seed đầu; đây là một lần rút ở đuôi (điển hình cỡ 1 SE) |
| 4. Rò rỉ | `cv_accuracy(X, y, k, select_inside)` trên nhãn ngẫu nhiên, n = 100, p = 5000, k = 20, 15 lần lặp | chọn trước khi chia: 0.862; chọn trong fold: 0.511; mô hình rò rỉ trên 2000 mẫu mới: 0.498 |
| 5. Dùng lại test set và nhiễm | `best_of_k_bias`, `p_detect` (DP), `ngram_flag` | thiên lệch K = 5 / 20 / 100 / 1000: +1.45 / +2.33 / +3.11 / +4.00 điểm; P(n = 13, q = 0.2, L = 60) = 0.487, mô phỏng 300 tài liệu 0.530; tài liệu Zipf độc lập bị gắn cờ: n = 2 là 1.00, n = 5 là 0.11, n = 13 là 0.00 |

Code nằm trong `labs/chuong11/lab.ipynb` (bản có chỗ TODO, kèm ô kiểm tra tự động) và `labs/chuong11/loigiai.ipynb` (bản lời giải đã chạy kiểm; toàn notebook mất 57 giây khi máy rảnh và khoảng 100 giây khi máy 2 lõi bị chen). Mở từ trang **Lab** của app. VD 11.3, 11.5, 11.6, 11.7, 11.9(a) và 11.12 là số liệu của người viết, không có lab tái tạo từng ô (Lab 2 và 3 chỉ dựng lại bản nhỏ của VD 11.4 và 11.6).

**Câu hỏi sau lab.**

- Lab 3: CI theo seed hẹp mà vẫn không chứa độ chính xác mới. Lỗi nằm ở đâu, và so sánh hai mô hình trên cùng test set có bị ảnh hưởng không?
- Lab 4: vì sao chọn đặc trưng trong fold cho 0.5 còn ngoài fold cho 0.86?
- Lab 5: vì sao thiên lệch best-of-K tăng chậm theo K? Vì sao n-gram dài bỏ sót nhiều hơn khi bản sao bị sửa nhiều?
- Lab 2: phân biệt ba trường hợp khi thêm một siêu tham số vào không gian random search: vô dụng, có vùng giá trị phần lớn có hại, hữu ích. EVP(k) đổi thế nào ở k nhỏ và k lớn? Với lưới thì sao?

**Bài săn bug.** Hàm dưới đây "chứng minh" dropout + weight decay hơn MLP thường (dữ liệu chung, train 300, validation 1000, test 1000). Nó chạy không báo lỗi. Triệu chứng: bài tuyên bố hơn khoảng 7.6 điểm; chạy lại trên dữ liệu mới thì chênh chưa tới 0.5 điểm; baseline trong bài kém bất thường. Tìm cả **bốn** lỗi quy trình trước khi xem lời giải, rồi viết `run_experiment_fixed` trong notebook (ô kiểm tra theo dõi cách bạn chạm vào tập test).

```python
lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
step_list = [40, 150, 400]
M = dict(p=0.5, wd=1.0)

def run_experiment_buggy():
    base_runs = [train_eval(Xtr, ytr, Xte, yte, s, lr=3e-2, steps=150)[0] for s in range(5)]
    cfg = max(itertools.product(lrs, step_list),
              key=lambda c: train_eval(Xtr, ytr, Xte, yte, 0, lr=c[0], steps=c[1], **M)[0])
    meth_runs = [train_eval(Xtr, ytr, Xte, yte, s, lr=cfg[0], steps=cfg[1], **M)[0] for s in range(5)]
    return dict(cfg=cfg, base=np.mean(base_runs), base_sd=np.std(base_runs),
                meth=np.max(meth_runs), meth_sd=np.std(meth_runs))
```

Hai phép thử dùng được cho mọi pipeline kiểu này: "ước lượng không được dùng dữ liệu đã dùng để chọn", và "tăng gấp đôi ngân sách tune của baseline, hiệu ứng có co không?".

## Bài dựng cảnh Manim

Giới hạn **1 giờ** (tính vào luồng C), không chỉnh màu mè.

**Đề bài.** Dùng `manim/chuong11_scenes.py` làm mẫu, viết cảnh `NGramBatTrung` vào thư mục `manim/cua-toi/`:

1. Vẽ trục q từ 0 đến 0.4 (xác suất mỗi token bị đổi) và trục xác suất bắt được bản sao từ 0 đến 1; tài liệu dài L = 60.
2. Vẽ ba đường cho n = 5, 8, 13, tính chính xác bằng quy hoạch động trên độ dài đoạn không đổi (mục 11.4, VD 11.9).
3. Đánh dấu điểm n = 13, q = 0.2 bằng số 0.487, và thêm một dòng nói vì sao n = 5 bền hơn nhưng sinh dương tính giả (0.13 trên văn bản Zipf độc lập).

**Tiêu chí tự chấm.** (1) Đúng số: 0.487 tại n = 13, q = 0.2 và 0.988 tại n = 5, q = 0.3. (2) Hiện rõ lý do chọn n là một đánh đổi, có phụ đề nói người xem phải thấy gì. (3) Không chữ chồng lên nhau, dài không quá 30 giây.

Render bằng nút **Render cảnh của tôi** ở trang Manim. Lời giải tham khảo nằm ở `manim/loigiai/chuong11_ngram_bat_trung.py`, chỉ mở sau khi đã tự làm.

## Phỏng vấn, bẫy, thực tế khác lý thuyết

**Cách luyện.** Mỗi chuỗi có 4 tầng câu hỏi, đào sâu dần như người phỏng vấn thật. Trả lời to bằng tiếng Việt trước, sau đó nói lại bằng tiếng Anh, 2 phút mỗi tầng, ghi âm và tự nghe lại. Câu trả lời mẫu tiếng Anh chỉ phủ tầng 1–2; tầng 3–4 bạn tự soạn, rồi đối chiếu với lời giải các bài tập liên quan. Tôi không có nguồn đề phỏng vấn cụ thể cho chương này; đây là các câu điển hình, diễn đạt lại.

**Chuỗi 1 — Phê bình một paper**

1. Bạn đọc paper này thế nào trong 15 phút?
2. Bản đồ khẳng định – bằng chứng gồm những cột nào, và bạn điền khi nào?
3. Ba điểm yếu hay gặp trong bảng kết quả (baseline, seed, cỡ test): bạn kiểm thế nào, bằng phép tính gì? (VD 11.1–11.2)
4. Bạn sẽ chạy thí nghiệm nào để bác bỏ khẳng định chính, và kết quả nào làm bạn đổi ý?

> *Model answer:* In the first pass I read the title, abstract, introduction, section headings and conclusion, and glance at the tables, to decide what the paper claims and whether it is worth more time. In the second pass I build a claim-evidence map: for each claim in the abstract I write down which table supports it, the metric, the baseline, the number of seeds, the test set size, and how hyperparameters were chosen and on which split. Then I look for gaps: a single run, a baseline copied from another paper, or a gain smaller than the run-to-run noise.

**Chuỗi 2 — Thiết kế thí nghiệm**

1. Thiết kế thí nghiệm để kiểm giả thuyết "prompt có ví dụ cải thiện NER khi ít dữ liệu".
2. Biến nào là scientific, nuisance, fixed?
3. Baseline mạnh và ngân sách công bằng nghĩa là gì? EVP là gì? (VD 11.3–11.4)
4. Ablation 2³ cho bạn thấy gì mà ablation một chiều không thấy? Kế hoạch phân tích viết trước gồm gì, và cần bao nhiêu seed? (VD 11.5, 11.7)

> *Model answer:* I would first write the hypothesis so it can fail: few-shot examples improve entity F1 by at least some margin on low-resource splits. The number of examples is the scientific variable; prompt wording and example selection are nuisance variables that I tune separately for each setting, with the same budget on the same validation set; model, tokenizer and decoding are fixed, and I would say that limits the conclusion. The baseline is the zero-shot prompt tuned with the same budget, plus a fine-tuned small model, all rerun by me. I would also write the analysis plan before running: primary metric, number of seeds from a power calculation, and the comparison I will test.

**Chuỗi 3 — Phương sai và độ lặp lại**

1. Hai lần chạy cho hai kết quả khác nhau. Vì sao, và bạn làm gì?
2. Nguồn phương sai gồm những gì? (VD 11.6, 11.12)
3. Mean ± std hay ± SE? CI với 5 seed dùng hệ số nào? Vì sao không báo best-of-k? (VD 11.6–11.7)
4. Vì sao CI theo seed chưa đủ, cần bao nhiêu seed, và ghép cặp theo seed có luôn giúp không?

> *Model answer:* Two runs with the same configuration can differ because of initialization, data order, dropout masks, and even numerical noise, so a single number is one draw from a distribution. I would run several seeds and report the mean with the sample standard deviation, using n minus one, and say whether the plus-minus is a standard deviation or a standard error. With five seeds I would use a t quantile with four degrees of freedom, about 2.78, because 1.96 covers only about 88 percent in simulation. I would not report the best seed, since the expected maximum of five normal draws sits about 1.2 standard deviations above the mean.

**Chuỗi 4 — Rò rỉ và đánh giá**

1. Validation 0.95 nhưng test 0.60. Bạn kiểm những gì, theo thứ tự nào?
2. Cho một ví dụ rò rỉ cụ thể trong NLP. (VD 11.8)
3. Nhiễm benchmark với LLM là gì, phát hiện bằng n-gram và giới hạn của nó? (VD 11.9)
4. Vì sao dùng lại test set nhiều lần là xấu ngay cả khi chỉ "xem" kết quả, và quy trình nào giữ test sạch? (VD 11.10)

> *Model answer:* A gap that large points first to a split problem, so I would check for duplicates and near-duplicates between train and validation versus test, and for group or time leakage, such as the same author on both sides. Next I would check preprocessing: was the vocabulary, the scaler or feature selection fit on all the data before splitting? Then I would ask whether the validation set was used for model selection so often that it is optimistic, and whether the test distribution simply differs. A cheap smoke test is to shuffle the labels: a leak-free pipeline must drop to chance, and if it does not, there is leakage.

**Từ vựng nên nói trôi chảy:** claim-evidence map, strong baseline, nuisance hyperparameter, equal tuning budget, expected validation performance, ablation study, factorial design, statistical power, confidence interval, data leakage, held-out test set, benchmark contamination, reproducibility.

**Bẫy thường gặp.**

- **Baseline chưa tune hoặc chép số từ paper khác:** hiệu ứng 5.17 điểm co về 0.29–0.52 khi tune công bằng (VD 11.3).
- **Ngân sách tune lệch:** 4 lần thử cho baseline, 60 cho phương pháp mới: chênh 2.4 điểm so với 0.2 khi cùng ngân sách (VD 11.4).
- **Báo best-of-k seed, hoặc mean của một bên so với best-of-k của bên kia;** chọn mô hình, siêu tham số, điểm dừng theo test (VD 11.7, 11.10, bài săn bug).
- **Chọn đặc trưng, chuẩn hóa, xây vocab trên toàn bộ dữ liệu trước khi chia** (VD 11.8).
- **Dùng 1.96 với 3–5 seed:** phủ 81–88% thay vì 95%; hoặc đọc CI theo seed như CI của độ chính xác thật (VD 11.6).
- **Kiểm nhiễm bằng n-gram quá dài hoặc quá ngắn; in số ở độ chính xác giả, in đậm theo max; không ghi signature metric** (VD 11.9, 11.11).

**Thực tế khác lý thuyết.**

- **Cố định seed không đảm bảo lặp lại khác môi trường, nhưng cũng không luôn phá lặp lại.** Trên CPU với mô hình nhỏ này, 1 luồng hay 2 luồng cho kết quả giống từng bit; chỉ khi có nhiễu số (float32 so với float64, nhiễu 10⁻⁷ ở khởi tạo) kết quả mới trượt, và trượt khuếch đại (VD 11.12).
- **Seed chỉ là một phần phương sai:** lấy mẫu train hoặc test đóng góp gấp khoảng 2 đến 2.5 lần nguồn seed (1.07–1.52 so với 0.54), nên CI 5 seed rất hẹp quanh một giá trị có thể lệch 2 SE so với giá trị thật (VD 11.6).
- **Độ mạnh nhỏ hơn trực giác, và ablation phụ thuộc hướng:** với σ = 0.54, 5 seed chỉ cho độ mạnh 0.23 khi chênh thật 0.5 điểm; LayerNorm "vô hại" khi thêm vào baseline nhưng bỏ nó khỏi mô hình đủ lại tăng 1.35 điểm (VD 11.5, 11.7).

## Mini-test 30 phút

Làm sau khi xong bài tập và lab, bấm giờ 30 phút, không xem sách, được dùng máy tính bỏ túi cho câu 1, 3, 5. Tổng 30 điểm. **Từ 24 điểm:** sang chương sau. **18 đến dưới 24:** ôn lại mục ứng với câu sai rồi làm lại sau 3 ngày. **Dưới 18:** học lại chương.

1. **(4 điểm)** Bốn seed cho 76.1, 75.4, 76.8, 75.9. Tính mean, std mẫu, SE và CI 95% (hệ số t với 3 bậc tự do là 3.182). Vì sao CI này vẫn chưa nói được gì về độ chính xác của mô hình trên dữ liệu mới?
2. **(5 điểm)** Một bảng ghi: "Ours 84.1, baseline 82.9, test 500 câu, 1 seed, baseline dùng lr mặc định chép từ paper gốc, siêu tham số của Ours chọn theo test". Nêu ít nhất ba vấn đề, xếp theo mức nghiêm trọng, và một thí nghiệm bổ sung cho mỗi vấn đề.
3. **(5 điểm)** Bốn điểm validation 0.5, 0.6, 0.7, 0.9 từ bốn lần thử. Tính EVP(2). Nếu phương pháp B có không gian lớn hơn A và được thử 40 lần trong khi A chỉ được thử 4, bạn kết luận gì từ việc B thắng A?
4. **(5 điểm)** Đọc code và chỉ ra hai rò rỉ, nói cách sửa:
   ```python
   X = TfidfVectorizer().fit_transform(all_texts)
   Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
   best_C = max(Cs, key=lambda C: LogisticRegression(C=C).fit(Xtr, ytr).score(Xte, yte))
   ```
5. **(5 điểm)** Cần phát hiện chênh lệch 0.3 điểm khi σ seed là 0.54 (5% hai phía, độ mạnh 0.8, tổng z bằng 2.80). Tính n. Nếu chỉ chạy 5 seed và không thấy khác biệt, ghi kết luận thế nào cho đúng?
6. **(6 điểm)** Viết lại ba câu sau cho đúng mức bằng chứng: (a) "Cơ chế attention mới là nguyên nhân của cải thiện 0.6 F1." (kết quả: 3 seed, một dataset, hơn baseline đã tune). (b) "Phương pháp của chúng tôi thắng trên 11/18 cột nên vượt trội." (c) Mục Limitations ghi "cần nghiên cứu thêm". Tối đa 5 câu.

## Flashcard

18 thẻ của chương nằm trong `flashcards/chuong11.csv`. Ôn ở trang **Ôn tập** của app: app tự lên lịch theo hộp Leitner (thẻ nhớ được sẽ giãn lịch ôn 1 → 3 → 7 → 14 ngày, thẻ quên quay lại hộp 1). File CSV này cũng nhập thẳng được vào Anki nếu bạn thích dùng Anki.

## Lời giải

Chỉ đọc sau khi đã tự thử ít nhất 20 phút. Với bài chứng minh, so ý chính chứ không so từng chữ. Đáp số tính tay đã được kiểm lại bằng code; số huấn luyện lấy từ lần chạy thật (seed cố định), có thể lệch nhẹ khi đổi phiên bản thư viện.

**Kiểm tra nhanh**

1. Không. 11 chỉ cách kỳ vọng 9 đúng 2 đơn vị, chưa tới một độ lệch chuẩn (2.1); chính xác P(≥ 11) = 0.2403. Thắng 14/18 mới cho 0.0154.
2. Baseline có được tune cùng ngân sách, cùng code và dữ liệu không. Hiệu 5 điểm khi baseline chưa tune thường co về dưới 1 điểm khi tune công bằng (VD 11.3).
3. SE = 0.5/√5 ≈ 0.224. Nửa độ rộng = 2.776 × 0.224 ≈ 0.62 (dùng 1.96 sẽ chỉ được 0.44, quá hẹp).
4. Bước chọn đã nhìn nhãn của mọi mẫu, kể cả những mẫu sau đó đóng vai fold kiểm tra. Trong 5000 đặc trưng ngẫu nhiên luôn có 20 đặc trưng tình cờ tương quan với nhãn; chọn chúng trước khi chia làm mô hình "biết" nhãn của fold kiểm tra. VD 11.8: 0.867 so với 0.491.
5. SE = 0.27/√10 ≈ 0.085. Một chữ số thập phân là đủ: 74.0 ± 0.3; chữ số thứ hai là nhiễu.

**Bài tập giấy bút**

1. Vị trí LayerNorm: scientific. Learning rate và số bước warmup: nuisance, tune lại cho từng nhánh vì giá trị tối ưu phụ thuộc vị trí LN. Số tầng và tokenizer: fixed, kết luận chỉ chắc cho độ sâu và tokenizer đã chọn (ví dụ "pre-LN ổn định hơn ở 12 tầng" không suy ra cho 96 tầng). Seed không thuộc ba loại: là nguồn ngẫu nhiên cần lặp lại.
2. Ba ô trống: số seed, siêu tham số của baseline chọn trên tập nào bằng ngân sách nào, và hiệu số so với phương sai (một lần chạy không có phương sai). Dễ đổi kết luận nhất là baseline chép từ paper khác: nó là một lần rút từ quy trình tune của người khác. Khẳng định "nhờ cơ chế X" còn cần ablation.
3. Mean 71.2; độ lệch bình phương 0, 0.16, 0.49, 0.09, 0.36 cộng 1.10. Std mẫu √(1.10/4) = 0.524; std tổng thể √(1.10/5) = 0.469; SE = 0.524/√5 = 0.234. CI t: 71.2 ± 2.776 × 0.234 = 71.2 ± 0.65 = [70.55, 71.85]; CI 1.96: 71.2 ± 0.46 = [70.74, 71.66], hẹp hơn khoảng 30% và chỉ phủ khoảng 88%.
4. (a) loại 1 (tiền xử lý trước khi chia). (b) loại 1 (vocab/tokenizer trên toàn bộ, test lọt vào). (c) loại 3 (nhóm). (d) loại 5 (dùng test chọn mô hình). (e) loại 6 (nhiễm benchmark).
5. P(≥ 14/18) = 0.0154, hai phía 0.0309. Với 6 cột độc lập: 6/6 cho 0.0156 < 0.05 và 5/6 cho 0.1094 > 0.05, nên cần thắng cả 6/6; 5/6 chưa đủ.
6. Một lần chạy, tập 500 câu: giá trị hợp lệ là bội của 0.2, nên 71.2 hoặc 71.4; 71.37 không hợp lệ. n nhỏ nhất là 227 (162/227 = 71.37%). Trung bình 5 seed trên 500 câu là bội của 0.04 (71.36, 71.40), vẫn không ra 71.37. Chỉ ngoại lệ như trung bình có trọng số hoặc làm tròn trung gian mới giải thích được.
7. Trọng số (2i − 1)/25 cho k = 2: (0.60 · 1 + 0.64 · 3 + 0.70 · 5 + 0.71 · 7 + 0.74 · 9)/25 = 17.65/25 = 0.706. EVP(1) = trung bình = 0.678. EVP(2) = 0.706: nếu chỉ có ngân sách hai lần thử ngẫu nhiên và chọn tốt hơn, kỳ vọng đạt 0.706 trên validation.
8. n = 2 × 2.80² × 0.8²/1.0² = 10.05, làm tròn lên 11 mỗi bên (độ mạnh thật khoảng 0.80). Với n = 5: δ = 2.80 × 0.8 × √(2/5) ≈ 1.42. Công thức dùng z nên **thiên thấp** ở n nhỏ (đúng là dùng t): độ mạnh thật ở δ = 1.42 (σ = 0.8, n = 5) chỉ khoảng 0.68, cần δ khoảng 1.6 mới tới 0.8.
9. Best-of-5 cộng 1.163σ ≈ 0.58 điểm không có thật, nên kỳ vọng chênh báo cáo là +0.58 cho B. Nếu cả hai best-of-5 thì độ lệch triệt tiêu trong kỳ vọng (chênh 0), nhưng vẫn không nên: max của 5 nhiễu hơn mean, không có CI chuẩn, và chọn seed là một nhánh quyết định dùng thông tin test. Công bằng hơn là báo mean ± CI. Triệt tiêu chỉ đúng khi σ hai bên bằng nhau. Nếu σ khác nhau thì best-of-5 ưu ái bên phương sai lớn hơn: chênh báo cáo kỳ vọng là 1.163 × (σ của bên báo best-of-5 trừ σ bên kia). Ví dụ lấy σ như VD 11.3 (baseline 0.93, phương pháp 0.27): nếu cả hai cùng best-of-5, baseline được lợi 1.163 × (0.93 − 0.27) ≈ 0.77 điểm, dù hai bên có cùng mean thật.
10. Có 64 chuỗi nhị phân độ dài 6 (1 là giữ nguyên). Số chuỗi không có ba số 1 liền nhau theo truy hồi a(n) = a(n−1) + a(n−2) + a(n−3): 1, 2, 4, 7, 13, 24, 44. Vậy có 64 − 44 = 20 chuỗi có đoạn ≥ 3, xác suất 20/64 = 0.3125.
11. Giả sử n giá trị phân biệt, sắp xếp v₍₁₎ < … < v₍ₙ₎. Max của k lần rút có hoàn lại ≤ v₍ᵢ₎ khi và chỉ khi cả k lần rút thuộc i giá trị đầu: xác suất (i/n)ᵏ. Vậy P(max = v₍ᵢ₎) = (i/n)ᵏ − ((i−1)/n)ᵏ và kỳ vọng là tổng của v₍ᵢ₎ nhân xác suất đó. k = 1: các trọng số là 1/n, nên EVP(1) là trung bình. EVP(k) = E[max của k lần rút] không giảm theo k vì max của k + 1 lần rút ≥ max của k lần rút đầu, nên kỳ vọng cũng ≥. Khi k → ∞, (i/n)ᵏ → 0 với i < n và = 1 với i = n, nên mọi trọng số dồn vào v₍ₙ₎ = max. Nếu có giá trị trùng, kết quả vẫn đúng bằng cách cộng trọng số các hạng bằng nhau.
12. Mô hình xấp xỉ chuẩn: số đo = p + SE · Z. E[max của hai biến chuẩn chuẩn độc lập] = 1/√π ≈ 0.564 (vì max(Z₁, Z₂) = (Z₁ + Z₂)/2 + |Z₁ − Z₂|/2; kỳ vọng nửa đầu 0 và E|Z₁ − Z₂| = √2 · √(2/π) = 2/√π). Thiên lệch best-of-2 = SE/√π; với SE = 1.26 điểm ra 0.71 điểm. Tăng chậm vì E[max của K] cỡ √(2 ln K): K từ 100 lên 1000 (gấp 10) chỉ cộng thêm khoảng 0.73 SE. √(2 ln K) là cận trên tiệm cận: ở K = 1000 cho 3.72 so với giá trị đúng 3.24; K = 100 cho 3.03 so với 2.51. Đủ cho trực giác, không đủ để tính số.
13. Var(hiệu) = σA² + σB² − 2ρσAσB = 0.0729 + 0.09 − 2 × 0.923 × 0.27 × 0.30 = 0.0134, nên std = 0.116 (bảng dùng 0.117 từ số liệu thật). SE = 0.117/√10 = 0.037; CI = 0.49 ± 2.262 × 0.037 = [0.41, 0.57] (số thật 0.4925, std 0.1173 cho [0.41, 0.58]). So với độc lập √(σA² + σB²) = 0.404: tỉ lệ phương sai (0.404/0.117)² ≈ 12, tức cần ít hơn khoảng 12 lần số seed cho cùng độ chính xác. ρ ≈ 0 khi đổi thành phần làm dòng số ngẫu nhiên lệch (đổi kiến trúc, số tham số, số lần gọi RNG): ví dụ "đủ bộ 111 so với D riêng" ở VD 11.7 có hiệu theo cặp std 0.41 so với 0.47 độc lập. Khi đó ghép cặp không giúp; ρ âm (hiếm) còn làm phương sai cặp lớn hơn.
14. Mẫu đáp án: *Giả thuyết:* loss L tăng F1 trung bình macro trên các ngôn ngữ ít tài nguyên ít nhất 0.5 điểm so với cross-entropy; bác bỏ nếu CI 95% của hiệu theo cặp seed nằm dưới 0.5 hoặc chứa 0. *Scientific:* loss (L so với CE). *Nuisance:* learning rate, số bước, hệ số trọng số của L; tune từng nhánh cùng ngân sách (ví dụ 30 lần random search, cùng không gian cho lr và số bước) trên validation. *Fixed:* mô hình nền, tokenizer, kích thước batch, dữ liệu; nêu rõ vì sao kết luận chỉ cho thiết lập này. *Baseline:* CE đã tune, chạy lại cùng code; ngoài ra baseline đơn giản. *Seed:* σ seed 0.54, δ = 0.5: n = 2 × 2.80² × 0.54²/0.25 = 18.3, tức 19 seed mỗi bên; nếu cũng ngẫu nhiên hóa tập train (σ = 1.38): 2 × 2.80² × 1.38²/0.25 = 119.4, tức 120 seed, gấp khoảng 6.5 lần. *Chỉ số và kiểm:* F1 trên test giữ kín, chạm một lần; hiệu theo cặp seed (kiểm tương quan trước) với CI t; so sánh thêm gắn nhãn khám phá. *Đe dọa hợp lệ:* rò rỉ train–test, nhiễm benchmark, cỡ test nhỏ cho ngôn ngữ ít dữ liệu.

**Bài tính nhanh**

1. P(≥ 4 trên 5) = (5 + 1)/32 = 0.1875.
2. Độ lệch bình phương 4, 0, 4: tổng 8. Std mẫu √(8/2) = 2; std tổng thể √(8/3) = 1.63.
3. 4.303/1.96 ≈ 2.2 lần.
4. SE = √(0.8 · 0.2/400) = √0.0004 = 0.02, tức 2 điểm.
5. 62.5% = 5/8, nên n = 8 (một lần chạy, một tập).
6. 1.163 × 0.5 ≈ 0.58 điểm.
7. n = 2 × 2.80² × 1/1 = 15.68, tức 16 seed mỗi bên.
8. (82 + 76 + 90 + 84)/4 = 83.0.
9. Không: n = 1000 chỉ cho bội của 0.1 (85.3, 85.4) nếu là một lần chạy trên một tập.
10. Trọng số 1/9, 3/9, 5/9: (0.6 + 2.1 + 4.0)/9 = 6.7/9 ≈ 0.744.
11. SE × E[max] ≈ 1.26 × 2.5 ≈ 3.2 điểm (đo 3.12).
12. 2.5 − 4.0 = −1.5: tương tác âm tổng cộng 1.5 điểm.
13. SE = 0.27/√10 = 0.085, nên in 74.0 ± 0.3 (std) hoặc 74.0 ± 0.1 (SE), ghi rõ ± là gì.
14. 0.8 mũ 13 ≈ 0.055; nên đoạn 13 token nguyên vẹn khá hiếm trong một bản sao đã sửa 20%.
15. (1.38/0.54)² ≈ 6.5 lần.

**Câu hỏi sau lab**

- Lab 3 (CI hẹp mà lệch; khoảng 1.96 chỉ phủ 0.88 vì std mẫu của 5 điểm hay nhỏ hơn std thật, điều mà phân vị t 2.776 bù lại): seed chỉ gom phương sai huấn luyện; test 1000 mẫu cố định có SE nhị thức 1.4 điểm, gấp 6 lần SE theo seed, và nó là một lần rút duy nhất. So sánh hai mô hình trên cùng test set thì độ lệch chung triệt tiêu nên vẫn hợp lệ.
- Lab 4: chọn trong fold chỉ dùng nhãn của phần train, nên fold kiểm tra thật sự là dữ liệu chưa thấy; chọn ngoài fold dùng nhãn của cả fold kiểm tra, và với 5000 đặc trưng ngẫu nhiên luôn có 20 đặc trưng tình cờ khớp nhãn.
- Lab 5: thiên lệch là SE nhân E[max của K biến chuẩn], cỡ √(2 ln K), tăng theo log K. N-gram dài cần một đoạn dài không bị đổi; q lớn thì đoạn dài nguyên vẹn hiếm, nên bỏ sót nhiều (n = 13, q = 0.3 chỉ còn 0.141).
- Lab 2: (i) siêu tham số vô dụng với random search không đổi phân phối điểm của mỗi lần thử, nên EVP(k) không đổi ở mọi k; chỉ lưới bị thiệt vì số điểm lưới nhân lên, tốn ngân sách. (ii) Có vùng giá trị phần lớn có hại (dropout 0.6, wd 1.0 ở lr không hợp): EVP ở k nhỏ giảm, k lớn bù lại vì max vẫn tìm được vùng tốt. (iii) Hữu ích (dropout, wd của không gian B): nâng EVP ở k lớn. Chênh ở k = 1 giữa A và B trong VD 11.4 là nhiễu, không minh họa (ii).

**Mini-test (thang 30 điểm)**

1. Mean 76.05; độ lệch bình phương 0.0025, 0.4225, 0.5625, 0.0225 cộng 1.01, std mẫu √(1.01/3) = 0.580, SE = 0.290 (1 điểm cho mean, 1 cho std, 1 cho SE); CI = 76.05 ± 3.182 × 0.290 = [75.13, 76.97] (0.5 điểm). CI chỉ gom phương sai theo seed; phương sai lấy mẫu test và train vẫn còn, nên không nói được gì về độ chính xác thật (0.5 điểm).
2. Thang chấm: (2 điểm) siêu tham số chọn theo test: tách validation, chạm test một lần. (1.5 điểm) baseline lr mặc định chép từ paper gốc: tune cùng ngân sách. (1.5 điểm) hiệu 1.2 điểm trên 500 câu với một seed: SE nhị thức của mỗi số cỡ 1.6 điểm, nên hiệu nhiều khả năng nằm trong nhiễu; thêm seed, CI theo cặp, McNemar trên cùng test (chương 3).
3. EVP(2) = (0.5 · 1 + 0.6 · 3 + 0.7 · 5 + 0.9 · 7)/16 = 12.1/16 = 0.756 (3 điểm). Không kết luận được: B có không gian lớn hơn và được thử nhiều hơn 10 lần, nên max của B thiên lên (VD 11.4) ngay cả khi B không tốt hơn. Phải so ở cùng k (EVP); B vẫn thắng ở cùng ngân sách mới là bằng chứng (2 điểm). Nếu đề đảo lại (B thử 4, A thử 40 mà B vẫn thắng) thì chênh ngân sách bất lợi cho B, nên B thắng là bằng chứng nghiêng về B; lúc đó chỉ còn phải kiểm nhiễu bằng nhiều seed.
4. Hai rò rỉ: (1) TF-IDF fit trên toàn bộ văn bản trước khi chia, nên IDF và vocab chứa thông tin test; sửa bằng chia trước, fit vectorizer trên train, rồi transform test (2.5 điểm). (2) Chọn C theo điểm trên test; sửa bằng tách validation hoặc CV trên train, chạm test một lần ở cuối (2.5 điểm).
5. n = 2 × 2.80² × 0.54²/0.3² = 50.8, tức 51 seed mỗi bên (3 điểm). Với 5 seed độ mạnh chỉ cỡ 0.11 nên kết luận đúng là "không đủ bằng chứng để phân biệt; thí nghiệm chưa đủ độ mạnh", không phải "hai phương pháp bằng nhau" (2 điểm).
6. Thang chấm, mỗi câu 2 điểm. (a) "Thí nghiệm với 3 seed trên một dataset cho thấy mô hình dùng attention mới hơn baseline đã tune 0.6 F1; chúng tôi chưa có ablation tách riêng cơ chế, nên nguyên nhân vẫn là giả thuyết." (b) "Phương pháp thắng 11/18 cột, mức này tương thích với ngẫu nhiên (P = 0.24); chúng tôi không khẳng định vượt trội." (c) "Chỉ đánh giá tiếng Anh, 3 dataset, mô hình dưới 1B tham số, 3 seed; chưa kiểm trên ngôn ngữ ít tài nguyên hay mô hình lớn hơn."

**Săn bug — bốn lỗi**

1. **Baseline không được tune.** Baseline dùng lr 3·10⁻² mặc định, còn phương pháp được chọn cấu hình trên lưới lr × số bước: không công bằng ngân sách. Trên dữ liệu mới baseline này chỉ đạt 68.19 trong khi baseline đã tune đạt 73.13 (đo 5 seed).
2. **Chọn cấu hình theo test.** Cấu hình của phương pháp lấy từ độ chính xác test của một seed (lr 0.03, 40 bước); đây là dùng lại test (VD 11.10). Trên dữ liệu mới nó chỉ đạt 72.59, thấp hơn baseline đã tune (73.13, 5 seed).
3. **Best-of-5 so với mean-of-5.** Phương pháp báo max của 5 seed (73.10) còn baseline báo mean của 5 seed (65.54); mean của phương pháp là 71.32. Riêng lỗi này cộng 1.78 điểm.
4. **Báo cáo thiếu:** chỉ 5 seed, không CI, không kiểm định, std dùng mẫu số n (np.std mặc định). Không có triệu chứng số, nhưng người đọc không thể phân biệt hiệu số với nhiễu. Bộ kiểm trong `_kiemtra.py` kiểm hành vi của hàm (phần nào dùng validation, phần nào chạm test, báo cáo có khớp các lần chạy và có khoảng tin cậy không), không kiểm cách bạn viết.

Lỗi phụ thứ năm, ngoài bốn lỗi: cấu hình của phương pháp chọn bằng một seed duy nhất. Phân rã chênh tuyên bố từ lần chạy của notebook lời giải: +7.56 (tuyên bố) → +5.78 (thay best-of-5 bằng mean 5 seed) → +4.40 (đo trên 40 000 mẫu mới thay vì test đã dùng để chọn, cả hai bên) → −0.55 (baseline đã tune thay cho baseline chưa tune) → +0.51 (cấu hình chọn theo validation; ba bước cuối đo trên 40 000 mẫu mới với 5 seed; hai bước đầu đo trên tập test đã dùng để chọn). Bản sửa (cả hai bên tune lưới như nhau trên validation, 3 seed; test một lần với 10 seed): baseline (10⁻³, 150) **72.13 ± 0.77**; phương pháp (10⁻², 40) **72.56 ± 0.47**; chênh **+0.43**; trên 40 000 mẫu mới (10 seed) 73.40 so với 73.79, chênh +0.38. Hai bộ số khác nhau ở số seed: baseline (10⁻³, 150) là 73.13 với 5 seed và 73.40 với 10 seed, chênh +0.51 (5 seed) so với +0.38 (10 seed); cả hai đều nhỏ hơn nhiễu theo seed.

```python
def tune_config(Xa, ya, Xv, yv, kw, n_seeds=3):      # không nhận Xte, yte
    score = {c: np.mean([train_eval(Xa, ya, Xv, yv, s, lr=c[0], steps=c[1], **kw)[0] for s in range(n_seeds)])
             for c in itertools.product(lrs, step_list)}
    return max(score, key=score.get)

def run_experiment_fixed():
    cfg_b = tune_config(Xtr, ytr, Xva, yva, {})       # cùng lưới, cùng validation, cùng số seed
    cfg_m = tune_config(Xtr, ytr, Xva, yva, M)
    tb = np.array([train_eval(Xtr, ytr, Xte, yte, s, lr=cfg_b[0], steps=cfg_b[1])[0] for s in range(10)]) * 100
    tm = np.array([train_eval(Xtr, ytr, Xte, yte, s, lr=cfg_m[0], steps=cfg_m[1], **M)[0] for s in range(10)]) * 100
    d = tm.mean() - tb.mean(); se = np.sqrt(tm.var(ddof=1) / 10 + tb.var(ddof=1) / 10); c = stats.t.ppf(0.975, 18)
    ci = (d - c * se, d + c * se)
    return dict(cfg_b=cfg_b, cfg_m=cfg_m, base=tb.mean(), base_sd=tb.std(ddof=1),
                meth=tm.mean(), meth_sd=tm.std(ddof=1), diff=tm.mean() - tb.mean(), diff_ci=ci)
```

## Tài liệu đào sâu và nguồn

Chỉ các nguồn trong bảng dưới đã được mở và đọc (qua bản raw trên GitHub, ngày 30/10/2026). Các paper ở phần "chưa mở lại được" không kèm link vì proxy của phiên viết sách chặn arXiv và ACL Anthology; hãy tự mở và đọc trước khi dẫn.

| Tài liệu | Dùng để làm gì | Phần nên đọc |
| --- | --- | --- |
| [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) (Google Research) | Ngôn ngữ chuẩn của thiết kế thí nghiệm tune: scientific, nuisance, fixed; trial, study, data variance; ngân sách tune; theo dõi thí nghiệm | "Designing the next round of experiments", "Determining whether to adopt a training pipeline change or hyperparameter configuration", "How many trials are needed to get good results with quasi-random search?", "Setting up experiment tracking" |
| [NLP-power-analysis](https://github.com/dallascard/NLP-power-analysis) (Card và cộng sự, repo đi kèm paper EMNLP 2020) | Notebook tính độ mạnh cho so sánh độ chính xác và BLEU; dữ liệu GLUE và SQuAD 2.0 | README và `notebooks_for_power_calculations` |
| [sacreBLEU](https://github.com/mjpost/sacrebleu) (Post) | Chuỗi signature để BLEU tái lập; kiểm định bootstrap ghép cặp | Mục về signature và mục về kiểm định ý nghĩa thống kê |
| [releasing-research-code](https://github.com/paperswithcode/releasing-research-code) (Papers with Code) | Khung README cho code kèm paper | `templates/README.md` |

**Chưa mở lại được (tên và năm từ trí nhớ hoặc kết quả tìm kiếm; chỉ để tìm đọc; nhiều năm dưới đây là năm bản arXiv, không phải năm hội nghị: ví dụ Melis và cộng sự là ICLR 2018, Lewis và cộng sự là EACL 2021, Lee và cộng sự là ACL 2022 [cần kiểm chứng]):** Keshav (2007), "How to Read a Paper"; Lipton và Steinhardt (2018), "Troubling Trends in Machine Learning Scholarship"; Henderson và cộng sự (2018), "Deep Reinforcement Learning That Matters"; Melis, Dyer, Blunsom (2017), "On the State of the Art of Evaluation in Neural Language Models"; Yang và Lin (2019), "Critically Examining the Neural Hype"; Narang và cộng sự (2021), "Do Transformer Modifications Transfer Across Implementations and Applications?"; Choi và cộng sự (2019), so sánh optimizer; Dodge và cộng sự (2019), "Show Your Work", và (2020), fine-tuning BERT; Bouthillier và cộng sự (2021), "Accounting for Variance in Machine Learning Benchmarks"; Card và cộng sự (2020), "With Little Power Comes Great Responsibility"; Kapoor và Narayanan (2022), rò rỉ trong khoa học dùng ML; Elangovan (2021), Lewis (2020), Lee (2021), Gururangan (2018), Dwork (2015), Recht (2018), Magar và Schwartz (2022), Sainz (2023), Brown (2020), Bowman và Dahl (2021), Pineau (2021), Gebru (2018), Mitchell (2018). Chúng được nhắc ở các mục 11.1 đến 11.5.

**Từ khóa để tự tìm thêm:** expected validation performance, hyperparameter budget, random search vs grid search, ablation study, factorial design, nuisance parameter, winner's curse, reproducibility checklist, pre-registration, power analysis, GRIM test, adaptive data analysis, holdout reuse, benchmark contamination, n-gram decontamination, hypothesis-only baseline, model card, datasheet, seed variance, `torch.use_deterministic_algorithms`, experiment tracking.

**Trước khi sang chương sau:** đạt từ 24/30 điểm mini-test, làm lại các bài ★★★ sai, và tự viết được lab 3 và lab 4 mà không nhìn code mẫu.

Nguồn tra cứu ngày 30/10/2026.
