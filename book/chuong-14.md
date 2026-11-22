# Chương 14 — Technical interview

## Mục tiêu và kiểm tra nhanh

Chương này luyện vòng cuối của hồ sơ: buổi technical interview, nơi người phỏng vấn đào sâu một câu hỏi cho tới khi bạn hết biết. Chương không dạy kiến thức mới như chương 1–12. Nó dạy cách **dùng lại** kiến thức đó dưới áp lực: suy dẫn trên bảng mà không nhìn sách, nói một con số có đơn vị, nối ba chương trong một câu trả lời, nói "tôi không biết" mà vẫn có ích, và kể một câu chuyện hành vi có số. Phần lớn chương là **ngân hàng chuỗi câu hỏi tổng hợp**: 18 chuỗi chia theo ba nhóm chương, cộng 4 chuỗi capstone ở phần Phỏng vấn. Mỗi nhóm đi kèm một phần suy dẫn chung mà nhiều chuỗi dùng lại. Thời lượng: **4 tuần** (tuần 35–38 của lộ trình), không phải 2 tuần như chương thường, vì đây là bài tổng hợp của cả sách. Lộ trình cho luồng chương **6.5 giờ mỗi tuần, tổng 26 giờ**. Bảng lịch dưới đây xếp mọi việc bắt buộc của chương vào đúng các dòng giờ đó. Việc vượt giờ được ghi **tùy chọn** (khoảng 3.25 giờ nếu làm hết). Luồng A (dự án) và luồng C (nói tiếng Anh) có giờ riêng; chương chỉ ghi việc nào của chúng dùng nội dung ở đây.

**Ba quy ước đọc chương này** (giữ như chương 13).

- **Mức xác minh**, kiểm ngày 16/11/2026: **A** là trang chính thức hoặc trang của tác giả (gồm trang biểu mục bài báo trên PubMed, ar5iv, trang hội nghị), đã mở và đọc được; **B** là bản sao tin tuyển dụng trên trang tổng hợp việc làm, chưa đối chiếu cổng gốc; **C** là bài viết bên thứ ba, bản tóm tắt thứ cấp hoặc ý kiến cá nhân; **D** là không tìm thấy hoặc không công khai.
- **Định dạng technical interview của Qualcomm và VinAI không công khai (D).** Mọi câu hỏi trong chương là câu điển hình do sách soạn từ nội dung chương 1–13. Không câu nào là câu của chương trình nào. Khung 60 phút, 12 phút mỗi chuỗi, thang tự chấm 4 tầng, thang 30 điểm cho phỏng vấn thử và quy tắc 6 câu chuyện đều là **quy tắc của sách**.
- Câu chuyện hành vi mẫu và số liệu dự án mẫu là **tự soạn, số giả**, mở đầu bằng "For illustration only." Chúng dùng đúng **hồ sơ giả của chương 13**: dự án hỏi đáp trên abstract arXiv (03/2026–09/2026) của hai người, một teammate làm data pipeline, còn "bạn" viết phần truy hồi (gồm bước citation-graph) và phần đánh giá; recall@5 từ 0.61 lên 0.72, 200 câu, 3 seed, kappa 0.78 trên 50 câu. Đừng trình bày chúng như kết quả thật.

**Lịch 4 tuần.** Mỗi tuần làm 4 trong 6 chuỗi của nhóm (Lab 1 chọn theo sổ lỗi). Mỗi chuỗi khoảng 37 phút: chuẩn bị từ Ý chính 15 phút, nói 12 phút (4 tầng), tự chấm và ghi sổ lỗi 10 phút. Hai chuỗi còn lại để dự phòng. Cột giữa là các dòng giờ luồng chương của lộ trình; giờ trong ngoặc là ước lượng của sách.

| Tuần | Dòng giờ của lộ trình → việc của chương | Tùy chọn (ngoài giờ) |
| --- | --- | --- |
| 35 | w35-doc 2 giờ: đọc 14.1–14.2 (1.5), bài 1–3 (0.5). w35-bank 3 giờ: Lab 1 (0.5), 4 chuỗi C01–C06 (2.5). w35-on 1.5 giờ: Lab 4 (0.75), bài 5 (0.25), bài tính nhanh lần 1 (0.5) | Lab 2 (0.5) |
| 36 | w36-doc 1.5 giờ: đọc 14.3. w36-bank 3.5 giờ: 4 chuỗi C07–C12 (2.5), capstone P1 (0.25), Lab 5 (0.75). w36-on 1.5 giờ: bài 6–9 (1), bài săn bug (0.5) | P3 (0.25) |
| 37 | w37-doc 1.5 giờ: đọc 14.4. w37-bank 3.5 giờ: 4 chuỗi C13–C18 (2.5), P2 và P4 (0.5), bài 10 (0.25), đệm 0.25. w37-on 1.5 giờ: bài 11–13 | Bài 14 (0.5); bài dựng cảnh Manim (1) |
| 38 | w38-doc 1.5 giờ: đọc 14.5 (1.25), bài 4 (0.25). w38-bank 2 giờ: 6 câu chuyện STAR. w38-mock 2 giờ: phỏng vấn thử 60 phút, chép lời, đếm bằng Lab 3, chấm thang 30 điểm. w38-on 1 giờ: bài tính nhanh lần 2 (0.5), mini-test (0.5) | Bài tính nhanh lần 3, 4 (1) |

Tổng bắt buộc: 6.5 + 6.5 + 6.5 + 6.5 = 26 giờ, đúng giờ luồng chương của lộ trình. Việc của luồng A và C dùng chương này: w35-a (5 câu hỏi sâu về dự án) đọc trước đoạn **Đào sâu dự án** ở 14.4; w35-c và w36-c dùng bảng **Đọc công thức thành tiếng** ở 14.1; w36-a dùng **khung trình bày 5 phút** ở 14.4; w37-a dùng C15 và P4; w38-a dùng tài liệu một trang ở 14.4.

**Sau chương này, bạn phải làm được:**

- ☐ Trả lời một chuỗi 4 tầng bằng tiếng Anh trong 12 phút và tự chấm được ít nhất 8/12 theo thang của sách.
- ☐ Suy dẫn trên bảng, không nhìn sách: gradient p − y và Hessian theo logit, bổ đề phương sai và ba chỗ dùng của nó, kích thước KV cache.
- ☐ Tính nhẩm 15 bài (FLOPs, GiB, KV cache, sai số chuẩn, perplexity, MRR) trong 25 phút, sai tối đa 1.
- ☐ Trả lời một câu hỏi chưa học theo 4 bước "I don't know, but here is how I would find out", không bịa tên paper hay con số.
- ☐ Kể 6 câu chuyện STAR, mỗi câu khoảng 2 phút, có số, vai trò rõ và bài học; chạy một buổi phỏng vấn thử 60 phút và tự chấm bằng thang 30 điểm.

**Kiểm tra nhanh (5 phút, không dùng máy tính).** Câu i kiểm tra mục 14.i: đúng thì đọc lướt mục đó, sai thì đọc kỹ.

1. Một chuỗi 4 tầng trong 12 phút: mấy phút mỗi tầng? 2 phút nói ở 140 từ/phút là bao nhiêu từ?
2. Logit (ln 4, 0), nhãn là lớp thứ hai. Tính p − y.
3. Mô hình hình dạng Llama-2-7B (32 lớp, 32 đầu K/V, dₕ = 128) ở fp16: KV cache mỗi token bao nhiêu byte?
4. Accuracy 0.8 trên 1600 câu: sai số chuẩn bằng bao nhiêu?
5. Theo MIT CAPD, phần nào của câu chuyện STAR nên chiếm nhiều thời gian nhất, khoảng bao nhiêu phần trăm?

Đáp án ở mục Lời giải cuối chương.

## 14.1 Buổi technical interview: cái gì công khai, khung mô phỏng, cách trả lời 4 tầng

Không có nguồn công khai nào mô tả nội dung vòng technical interview của Qualcomm RRP hay VinAI, nên chương này luyện bằng một buổi phỏng vấn **mô phỏng**: dựng theo mẫu chung của phỏng vấn nghiên cứu ML mà các nguồn công khai mô tả, rồi đo tiến bộ bằng một thang tự chấm.

**Cái gì công khai, cái gì không** (đã soát ở chương 13, mở lại ngày 16/11/2026). Tin Batch 17 của Qualcomm RRP (đăng 13/10/2026, mức B) chỉ ghi technical interview là **vòng 5, vòng cuối**, sau Application, Entrance test, Coding interview và Video presentation. Thời lượng, số người phỏng vấn, nội dung, ngôn ngữ, tiêu chí chấm và bộ câu hỏi đều không công khai (D); trang chương trình chính thức chỉ có metadata (A); quy trình VinAI hiện hành không xác minh được (D). Kết quả Glassdoor và LeetCode về Qualcomm là vị trí khác ở Mỹ và Ấn Độ, nên sách không dùng. Cách viết an toàn cho cả chương: *sách không biết người phỏng vấn sẽ hỏi gì; chương dựng một buổi mô phỏng theo mẫu chung có nguồn và luyện bằng chuỗi câu hỏi tổng hợp từ chương 1–13.*

**Mẫu chung có nguồn** (không nguồn nào nói về Qualcomm hay VinAI):

- **Google DeepMind**, bài "Applying for technical roles" (mức A, không ghi ngày; dạng hỏi đáp với ba nhân viên có tên): với vị trí research scientist, bài mô tả một buổi trò chuyện ban đầu, một **technical quiz** trải rộng khoa học máy tính, thống kê, toán và ML, một buổi coding, vài buổi phỏng vấn nghiên cứu khoảng 30 phút kèm bài nói về nghiên cứu của chính ứng viên, và một buổi về văn hóa làm việc; bài cũng nói quy trình khác nhau theo vị trí. Các lời khuyên (ôn rộng, giải thích lời giải thành tiếng, luyện code không có IDE) là **lời kể của nhân viên trong bài**, không phải quy định của quy trình.
- **Chip Huyen**, *Machine Learning Interviews Book* (mức A, trang tác giả, ©2021): mục 2.2 **khuyên** bên tuyển dụng nên giao cho mỗi người phỏng vấn một nhóm kỹ năng (lý thuyết, coding, thiết kế hệ thống ML). Một câu đáng nhớ: "No company expects you to know everything". Với câu hỏi mơ hồ, sách khuyên đừng chê câu hỏi; hỏi lại để làm rõ, nói điều đang chặn mình, và hỏi người phỏng vấn muốn đánh giá điều gì.
- **Google re:Work**, hướng dẫn structured interviewing (mức A, tài liệu viết cho **người phỏng vấn**): câu hỏi behavioral ("Tell me about a time…") và hypothetical ("Imagine if…"), mỗi câu có các câu hỏi nối tiếp định sẵn, chấm theo thang có câu trả lời mẫu ở nhiều mức. Đây là cơ sở để sách dạy "chuỗi 4 tầng" và "thang tự chấm có mốc". Thang của sách **không phải** thang của Google.

**Khung mô phỏng 60 phút (quy tắc của sách).** Sách chia buổi thử thành sáu phần. Không chương trình nào công bố khung này; nó là khung luyện, đủ gần với các mẫu chung ở trên để bạn tập chuyển phần và giữ nhịp.

| Phần | Phút |
| --- | --- |
| Mở đầu, giới thiệu bản thân 60 giây | 5 |
| 2 chuỗi kỹ thuật × 12 phút | 24 |
| Đào sâu dự án | 12 |
| 2 câu chuyện hành vi × 6 phút (kể 2 phút, nối tiếp 4 phút) | 12 |
| Câu hỏi của ứng viên | 5 |
| Đệm, chuyển phần | 2 |

**Ví dụ giải mẫu 14.2 — ngân sách thời gian.** Tổng 5 + 24 + 12 + 12 + 5 + 2 = 60 phút. Một chuỗi 12 phút = 4 tầng × 3 phút; mỗi tầng 2 phút trả lời và 1 phút cho câu hỏi nối tiếp. 2 phút ở 140 từ/phút (chuẩn tính của chương 13) là 280 từ: đủ cho một câu trả lời có cấu trúc, không đủ cho một bài giảng. Một câu chuyện STAR 2 phút theo tỉ lệ gợi ý của MIT CAPD (20/10/60/10, xem 14.5) chia thành Situation 24 giây (56 từ), Task 12 giây (28 từ), Action 72 giây (168 từ), Result 12 giây (28 từ). Mỗi tuần 4 chuỗi × 12 phút = 48 phút nói.

**Bốn tầng của một chuỗi** (mô hình sư phạm của sách, không phải phân loại chính thức):

- **Tầng 1 — định nghĩa hoặc phát biểu.** "What is X?", "State X."
- **Tầng 2 — cơ chế hoặc suy dẫn.** "Why does it work?", "Show that…"
- **Tầng 3 — con số, điều kiện, so sánh.** "How big?", "When does it hold?", "Compare with…"
- **Tầng 4 — giới hạn, thực tế khác lý thuyết, hướng mở.** "When does it fail?", "How would you test that claim?"

Người phỏng vấn đào cho tới khi bạn hết biết. Mục tiêu không phải trả lời trọn 4 tầng mà là biết mình dừng ở tầng nào và dừng cho đúng cách. Lộ trình giao 12 phút cho một chuỗi: 3 phút mỗi tầng. Trong ngân hàng chuỗi, số tầng là **thứ tự đào sâu**: với kiến thức nền, câu tầng 1 có thể đã là một suy dẫn ngắn (C01 mở bằng "Derive…"). Bài tập 1 xếp câu hỏi theo định nghĩa ở trên.

% manim: DongHo12Phut

**Khung trả lời một tầng (quy tắc của sách).** Bốn phần, theo thứ tự:

1. Câu trả lời trong **một câu**.
2. Cơ chế hoặc suy dẫn, nói thành tiếng từng bước, viết lên bảng.
3. **Một con số có đơn vị**.
4. **Một điều kiện hoặc giới hạn**.

Ví dụ cho câu "How big is the KV cache per token?": (1) "Two tensors, keys and values, for every layer, every KV head and every head dimension." (2) Viết 2 × số lớp × số đầu K/V × dₕ × số byte. (3) "For a Llama-2-7B shape in fp16 that is 524 288 bytes, half a mebibyte." (4) "With 8 KV heads instead of 32 it is a quarter of that, and it grows linearly with batch size." Sau khoảng 2 phút, dừng và hỏi: "Should I go deeper into the derivation, or move on?" Câu hỏi mơ hồ thì hỏi lại **trước khi** trả lời, theo gợi ý của Chip Huyen ở trên.

**"I don't know, but here is how I would find out" (bốn bước, quy tắc của sách).** Tuần 37 có việc riêng cho kỹ năng này, nhưng công thức dạy ở đây:

1. Nói thẳng phần không biết: "I have not studied X."
2. Nói phần liên quan mà mình biết chắc.
3. Đề xuất một cơ chế dưới giả định nói rõ.
4. Nói cách kiểm: đọc paper nào, mô phỏng gì, đo gì.

Không đoán bừa tên paper hay con số. Một con số bịa bị hỏi lại là mất điểm cả chuỗi; "I would check this against the original paper" thì không. Chuỗi C18 (mục 14.4) là ví dụ đủ bốn bước.

**Thang tự chấm 4 tầng (quy tắc của sách).** Mỗi tầng 0–3 điểm, một chuỗi tối đa 12:

| Điểm | Mốc |
| --- | --- |
| 0 | Trống, hoặc sai |
| 1 | Đúng từ khóa hoặc định nghĩa, chưa có cơ chế |
| 2 | Đúng cơ chế hoặc suy dẫn, có các bước |
| 3 | Mốc 2, cộng một con số đúng có đơn vị **và** một điều kiện hoặc giới hạn |

**Đạt: tổng ≥ 8/12 và tầng 1, tầng 2 mỗi tầng ≥ 2.** Điều kiện thứ hai chặn trường hợp "nói hay ở tầng 4 nhưng không suy dẫn được tầng 2". Ghi điểm mỗi chuỗi vào sổ lỗi dạng CSV ba cột `chuoi,diem,ngay`; Lab 1 đọc sổ này để chọn chuỗi tuần sau.

**Ví dụ giải mẫu 14.1 — chấm một câu trả lời.** Câu trả lời giả định cho chuỗi C01 (mục 14.2):

- Tầng 1 suy dẫn đủ "gradient là p − y", kèm số của ví dụ 14.3 và điều kiện "nhãn one-hot": **3**.
- Tầng 2 nói "loss lồi" nhưng không phân biệt lồi theo logit với lồi theo trọng số của MLP: **1**.
- Tầng 3 nói cực tiểu khi có label smoothing bằng 0: sai, **0**.
- Tầng 4 nêu weight decay nhưng không nói tác động lên calibration: **1**.

Tổng 5/12, **chưa đạt** (cần ≥ 8 và tầng 2 ≥ 2). Hai dòng ghi vào sổ lỗi: "lồi theo logit ≠ lồi theo tham số" và "cực tiểu với nhãn mềm là H(q) > 0". Tuần sau, Lab 1 cho C01 trọng số cao.

**Vì sao sách bắt trả lời miệng, giãn lịch và ghi điểm** (chi tiết và mức nguồn ở bảng cuối chương):

- **Tự kiểm tra thắng đọc lại** ở bài kiểm tra trễ (Roediger và Karpicke 2006; Rowland 2014), và **giãn lịch** có lợi, khoảng giữ càng dài thì khoảng cách tối ưu càng dài (Cepeda và cộng sự 2006). Các mốc Leitner 1 → 3 → 7 → 14 ngày là lựa chọn của sách.
- Dunlosky và cộng sự (2013) xếp practice testing và distributed practice **cao**, self-explanation chỉ **trung bình**: sách dựa vào tự kiểm tra cộng giãn lịch, không nói "giải thích thành tiếng là cách tốt nhất".
- Luyện có chủ đích giải thích chỉ 4% phương sai thành tích ở giáo dục và dưới 1% ở nghề nghiệp (Macnamara và cộng sự 2014): cần, nhưng không bảo đảm kết quả một buổi phỏng vấn.

**Đọc công thức thành tiếng** (cho w35-c và w36-c; cách đọc thông dụng, không phải chuẩn duy nhất):

| Viết | Đọc |
| --- | --- |
| p − y | "p minus y" |
| xᵀAx | "x transpose A x" |
| ∂L/∂zₖ | "the partial derivative of L with respect to z k" |
| σᵢ²/(σᵢ² + λ) | "sigma i squared over sigma i squared plus lambda" |
| √dₖ | "the square root of d k" |
| 𝔼[x²] | "the expected value of x squared" |
| log₂ 20 | "log base two of twenty" |
| (1 − αᵏ⁺¹)/(1 − α) | "one minus alpha to the k plus one, over one minus alpha" |
| 2 × 10⁻⁴ | "two times ten to the minus four" |
| 12.55 GiB | "twelve point five five gibibytes" |

% manim: LichLeitner

**Người phỏng vấn nghe gì** (mô hình của sách; dùng cho cả ba nhóm chuỗi).

- Câu trả lời nằm ở câu đầu, không ở phút thứ ba.
- Con số đi kèm đơn vị, quy ước và giả định (GB hay GiB, chia ε cho K hay K − 1, "assuming 2 TB/s, fp16, batch 1"), và bạn tự nói ước lượng bỏ qua cái gì trước khi bị hỏi.
- Bạn suy dẫn lại được khi đổi giả định, và nhận ra cùng một công cụ (bổ đề phương sai, KL, nhiệt độ) ở nhiều chương.
- Bạn tách "con số" khỏi "khẳng định": con số chỉ có nghĩa khi kèm baseline, số seed, độ dao động và kiểm định đúng thiết kế.
- Bạn biết ranh giới hiểu biết của mình, và khi chạm ranh giới thì đi từ điều chắc chắn sang giả định nói rõ, rồi nói sẽ kiểm ở đâu.

## 14.2 Chuỗi tổng hợp tuần 35: toán và ML nền (chương 1–6)

Nhóm chuỗi đầu tiên kiểm xem bạn còn suy dẫn được các kết quả nền của chương 1–6 trên bảng không, và nối được chúng sang deep learning. Ba kết quả dùng lại nhiều nhất là p − y, bổ đề phương sai và ổn định số; mục này suy dẫn chúng một lần, rồi 6 chuỗi C01–C06 dùng lại.

**Cách dùng ngân hàng.** Mỗi chuỗi có 4 câu hỏi tiếng Anh theo 4 tầng, **Ý chính** (số đã kiểm bằng script), **Lỗi hay gặp**, **Giao** (tầng nào trùng chuỗi của chương trước: quay về đó thay vì học lại) và một câu trả lời mẫu tiếng Anh cho tầng 1–2. Tầng 3–4 bạn tự soạn từ Ý chính rồi nói thành tiếng. Câu hỏi viết bằng tiếng Anh vì sách **giả định** người phỏng vấn nói tiếng Anh; ngôn ngữ thật không công khai (D).

**Phần suy dẫn chung 1: softmax cross-entropy.** Với logit z ∈ ℝᴷ, p = softmax(z) và y one-hot của lớp đúng c:

$$
L = -z_c + \log \sum_{j} e^{z_j}, \qquad \frac{\partial L}{\partial z_k} = p_k - y_k, \qquad \nabla_z^2 L = \operatorname{diag}(p) - p p^{\top}
$$

Đạo hàm của log-sum-exp theo zₖ chính là pₖ, nên gradient là p − y. Hessian nửa xác định dương vì với mọi v:

$$
v^{\top}\left(\operatorname{diag}(p) - p p^{\top}\right) v = \sum_k p_k v_k^2 - \Big(\sum_k p_k v_k\Big)^2 = \operatorname{Var}_{k \sim p}(v_k) \ge 0
$$

Dấu bằng xảy ra khi mọi vₖ bằng nhau, tức theo hướng vector toàn 1: cộng cùng một hằng số vào mọi logit không đổi softmax. Vậy loss **lồi theo logit nhưng không lồi chặt**. Với classifier tuyến tính z = Wx, loss lồi theo W vì hợp của hàm lồi với ánh xạ tuyến tính vẫn lồi; với logistic regression Hessian là XᵀSX, S chéo với sᵢ = σᵢ(1 − σᵢ) > 0. Với MLP, z phụ thuộc phi tuyến vào trọng số nên tính lồi mất (C04).

**Ví dụ giải mẫu 14.3 — cross-entropy trọn vẹn.**

- Logit (1, 2, 0), nhãn là lớp thứ hai: p = (0.2447, 0.6652, 0.0900), p − y = (0.2447, −0.3348, 0.0900), CE = −ln 0.6652 = 0.4076. Gradient số khớp p − y với sai số tương đối 9.8 × 10⁻¹¹.
- Hessian diag(p) − ppᵀ khớp sai phân (sai số 9.9 × 10⁻¹¹); trị riêng nhỏ nhất ≈ 0 (máy in −2.4 × 10⁻¹⁷) theo hướng vector toàn 1. Một ví dụ logistic regression có Hessian XᵀSX với trị riêng nhỏ nhất 1.9557, dương.
- Label smoothing K = 10, ε = 0.1, quy ước PyTorch: mục tiêu q gán 1 − ε + ε/K = 0.91 cho lớp đúng và ε/K = 0.01 cho mỗi lớp khác. Cross-entropy H(q, p) nhỏ nhất khi p = q, nên khe logit tối ưu giữa lớp đúng và lớp khác là ln(0.91/0.01) = ln 91 = 4.5109 (gradient descent hội tụ đúng 4.5109). **Loss nhỏ nhất là H(q) = 0.5003, không phải 0.** Quy ước chia ε cho K − 1 lớp sai cho khe ln 81 = 4.3944. Bẫy: có hai quy ước, phải nói mình dùng quy ước nào.

**Phần suy dẫn chung 2: một bổ đề phương sai.** Nếu w₁, …, wₙ độc lập với x₁, …, xₙ, các wᵢ cùng phân phối, trung bình 0, thì các số hạng chéo có kỳ vọng 0 và

$$
\operatorname{Var}\Big(\sum_{i=1}^{n} w_i x_i\Big) = \sum_{i=1}^{n} \mathbb{E}[w_i^2]\,\mathbb{E}[x_i^2] = n \operatorname{Var}(w)\, \mathbb{E}[x^2]
$$

Chú ý 𝔼[x²], không phải Var(x): sau ReLU, x không có trung bình 0. Nếu tiền kích hoạt z đối xứng quanh 0 thì 𝔼[ReLU(z)²] = Var(z)/2, nên muốn giữ phương sai qua mỗi lớp cần n·Var(w)/2 = 1, tức Var(w) = 2/n (He). Ba chỗ dùng: khởi tạo; q·k trong attention (dₖ số hạng, mỗi số hạng phương sai 1, nên Var(q·k) = dₖ và phải chia √dₖ); dòng residual, nơi mỗi nhánh cộng thêm phương sai của nó.

**Ví dụ giải mẫu 14.4 — một bổ đề, ba chỗ dùng** (mô phỏng một seed; qua 20 seed, giá trị ở lớp 10 với He nằm trong 1.35–3.21, độ lệch chuẩn khoảng 0.49, tức cỡ ±25% quanh 2, nên chỉ kết luận "giữ cỡ hằng số").

- 10 lớp ReLU, rộng 512, đầu vào phương sai 1. Với Var(w) = 2/n: 𝔼[z²] ở lớp 1 là 1.995, ở lớp 10 là 1.808, giữ cỡ hằng số. Với 1/n: lớp 1 là 1.001, lớp 10 còn 0.0017; lý thuyết 2⁻⁹ ≈ 0.0020 vì mỗi lớp ReLU giữ một nửa mô-men bậc hai. Độ lệch chuẩn He ở n = 512 là √(2/512) = 0.0625.
- q·k với dₖ = 64: phương sai đo được 63.9, độ lệch chuẩn 7.99; chia √dₖ: 0.999. Với dₖ = 128: 128.3 và 11.33. Không chia thì logit cỡ ±8 đến ±11, softmax gần one-hot và gradient qua nó gần 0.
- Dòng residual 12 lớp, 24 nhánh (attention và MLP mỗi lớp) phương sai 1, độc lập: phương sai 25.13, lý thuyết 1 + 24 = 25. Nhân mỗi nhánh với 1/√(2L): 2.000, lý thuyết 2. Cách khởi tạo kiểu GPT-2 trong nanoGPT dùng độ lệch chuẩn 0.02/√(2L) = 0.02/√24 = 0.00408 cho phép chiếu cuối mỗi nhánh (chương 8 đã nêu; ở đây chỉ dùng lại).

% manim: BoDePhuongSai

**Phần suy dẫn chung 3: ổn định số.** Softmax không đổi khi trừ cùng hằng số m khỏi mọi logit; chọn m = max zⱼ thì số mũ lớn nhất là e⁰ = 1:

$$
\log \sum_j e^{z_j} = m + \log \sum_j e^{z_j - m}, \qquad m = \max_j z_j
$$

**Ví dụ giải mẫu 14.5 — số học dấu phẩy động cụ thể.**

- float32: giá trị lớn nhất 3.4028 × 10³⁸, ln của nó 88.72, nên exp(89) = inf. fp16: lớn nhất 65 504, ln 65 504 = 11.09, nên exp(12) = inf còn fp16(e¹¹) = 59 872 (e¹¹ = 59 874.14; trong [32 768, 65 536) các số fp16 cách nhau 32).
- fp16: machine epsilon 9.766 × 10⁻⁴, số chuẩn dương nhỏ nhất 6.104 × 10⁻⁵, số dưới chuẩn nhỏ nhất 5.960 × 10⁻⁸; fp16(10⁻⁸) = 0. Trong fp16, 1 + 10⁻⁴ = 1.0 (cập nhật bị mất hẳn) còn 1 + 10⁻³ = 1.001.
- bf16: lớn nhất 3.3895 × 10³⁸ (dải như fp32), epsilon 2⁻⁷ = 7.8125 × 10⁻³; bf16(257) = 256 và bf16(1 + 2⁻⁸) = 1.
- Softmax ngây thơ trên logit (1000, 999, 998) cho (nan, nan, nan); trừ max cho (0.6652, 0.2447, 0.0900). Log-sum-exp = 1000.4076; nếu lớp đúng là 998 thì CE = 1000.4076 − 998 = 2.4076.

**Thực tế tính toán.** Loss scaling tồn tại vì trong fp16, gradient cỡ 10⁻⁸ bị làm tròn về 0 (số dưới chuẩn nhỏ nhất là 5.96 × 10⁻⁸) và gradient dưới 6.1 × 10⁻⁵ mất dần độ chính xác; nhân loss với một hệ số lớn đẩy chúng về vùng biểu diễn được. bf16 có dải như fp32 nên thường không cần loss scaling, nhưng độ chính xác kém, nên phép cộng dồn và bản trọng số chính vẫn giữ fp32. Paper gốc của công thức mixed precision (Micikevicius và cộng sự, 2017) chưa mở được trong lần soát này [cần kiểm chứng trước khi dẫn chi tiết].

**C01 — Cross-entropy, seen whole** (chương 2, 3, 5)

- *Tầng 1.* Derive the gradient of softmax cross-entropy with respect to the logits.
- *Tầng 2.* Is this loss convex in the logits? In the weights of a linear classifier? In the weights of an MLP?
- *Tầng 3.* With label smoothing ε over K classes, what is the optimal predicted distribution, the optimal logit gap, and the minimum loss?
- *Tầng 4.* On separable data the logits grow without bound. Compare label smoothing, weight decay and early stopping as fixes, and what each does to calibration.

**Ý chính.** Tầng 1–3: phần suy dẫn chung 1 và ví dụ 14.3. Tầng 4: label smoothing cho nghiệm hữu hạn; weight decay phạt chuẩn trọng số; early stopping là điều chuẩn ngầm. Calibration đo bằng ECE (chương 12, ví dụ 12.9) và sửa sau huấn luyện bằng temperature scaling (C10).

**Lỗi hay gặp.** Viết y − p (sai dấu); quên số hạng chuẩn hóa của softmax; nói cực tiểu bằng 0 khi có smoothing; trộn hai quy ước ε.

**Giao.** Chương 2 chuỗi 2; chương 5 chuỗi 1 tầng 3–4. Mới: lồi theo logit so với theo tham số; nghiệm khi có smoothing.

> **Sample answer (tiers 1–2):** I write the loss as minus the logit of the true class plus the log-sum-exp of all logits. The derivative of the log-sum-exp with respect to each logit is the softmax probability, so the gradient is p − y, where y is the one-hot label; for logits (1, 2, 0) and the second class it is (0.24, −0.33, 0.09). The Hessian with respect to the logits is diag(p) − ppᵀ, which is positive semi-definite, so the loss is convex in the logits but not strictly convex, because adding the same constant to every logit changes nothing. For a linear classifier the logits are linear in the weights, so convexity carries over; for an MLP the logits are a non-linear function of the weights, and convexity is lost.

**C02 — Ridge seen four ways** (chương 1, 3, 4, 5)

- *Tầng 1.* Write the ridge solution and explain what it does using the SVD.
- *Tầng 2.* Show ridge is the MAP estimate under a Gaussian prior. What is λ in terms of the noise and prior variances?
- *Tầng 3.* How does λ change the condition number and the speed of gradient descent?
- *Tầng 4.* When would you prefer PCA regression or early stopping, and how are they related to ridge?

**Ý chính.** Nghiệm (XᵀX + λI)⁻¹Xᵀy; theo SVD, thành phần dọc mỗi vector kỳ dị phải bị nhân với hệ số co σᵢ²/(σᵢ² + λ). Với trị riêng của XᵀX là 100 và 0.01, λ = 1 cho 0.9901 và 0.0099. MAP: λ = σ²/τ² khi loss là **tổng** bình phương (với loss trung bình thì lệch hệ số n); script xác nhận nghiệm ridge trùng trung bình hậu nghiệm Gauss. Số điều kiện từ 10 000 xuống (100 + 1)/(0.01 + 1) = 100; tốc độ hội tụ của gradient descent với bước tối ưu (κ − 1)/(κ + 1) từ 0.99980 xuống 0.98020, số bước để giảm sai số 10³ lần từ 34 539 xuống 345. Tầng 4: PCA regression cắt cứng các hướng nhỏ, ridge co mềm; early stopping tác động như một bộ lọc phổ và chỉ **gần đúng** ridge (nói là xấp xỉ, không phải tương đương).

**Lỗi hay gặp.** λ lệch hệ số n giữa loss trung bình và loss tổng; phạt cả bias; nói ridge "loại bỏ đặc trưng" (đó là L1).

**Giao.** Chương 1 chuỗi 2 tầng 4; chương 4 chuỗi 1; chương 5 chuỗi 2 tầng 3. Mới: bốn góc nhìn, có κ và số bước.

> **Sample answer (tiers 1–2):** The ridge solution is w = (XᵀX + λI)⁻¹Xᵀy. With the SVD X = UΣVᵀ, it multiplies the least-squares component along each right singular vector by σᵢ²/(σᵢ² + λ), so directions with large singular values are almost untouched and directions with small ones are shrunk towards zero; with eigenvalues of XᵀX equal to 100 and 0.01 and λ = 1, the factors are 0.99 and 0.0099. Ridge is also the MAP estimate when the noise is Gaussian with variance σ² and the prior on the weights is Gaussian with variance τ²: the negative log-posterior is the sum of squared errors over 2σ² plus the squared norm over 2τ², so λ = σ²/τ² when the loss is a sum rather than a mean.

**C03 — One variance lemma, three places** (chương 3, 6, 8)

- *Tầng 1.* For independent zero-mean weights and inputs, what is Var(∑ wᵢxᵢ)? Use it to derive an initialization.
- *Tầng 2.* Why does ReLU need the factor 2? What happens over 10 layers if you forget it?
- *Tầng 3.* Use the same lemma to justify dividing attention scores by √dₖ.
- *Tầng 4.* A pre-LN Transformer adds 2L branches to the residual stream. How does its variance grow, and how do GPT-2-style initializations handle it?

**Ý chính.** Phần suy dẫn chung 2 và ví dụ 14.4 (cả bốn tầng). Nhấn: đầu ra ReLU không có trung bình 0 nên dùng 𝔼[x²].

**Lỗi hay gặp.** Dùng Var(x) thay 𝔼[x²] cho đầu ra ReLU; nhầm fan-in với fan-out; nói "có LayerNorm thì khởi tạo không quan trọng" (chương 6 chuỗi 2 tầng 4 đã phân biệt).

**Giao.** Chương 6 chuỗi 2; chương 8 chuỗi 2 tầng 2. Mới: một bổ đề cho ba chỗ.

> **Sample answer (tiers 1–2):** If the weights and inputs are independent and the weights have mean zero, the cross terms vanish and Var(∑ wᵢxᵢ) = n·Var(w)·𝔼[x²]. To keep the scale constant from layer to layer, I need n·Var(w) times the second moment of the input to equal the variance of the previous pre-activation. After a ReLU, the second moment is half the variance of the symmetric pre-activation, so Var(w) = 2/n, which is He initialization. With 1/n instead, the second moment halves at every layer; in a simulation of 10 layers of width 512 it fell from about 1.0 to 0.0017, close to the predicted 2⁻⁹.

**C04 — Why does a non-convex network train?** (chương 1, 4, 5, 6)

- *Tầng 1.* Prove that the logistic regression loss is convex.
- *Tầng 2.* Show that a one-hidden-layer MLP loss is not convex, using a symmetry argument.
- *Tầng 3.* If the loss is non-convex, why does gradient descent usually find good solutions? What does the evidence say about saddle points versus bad local minima?
- *Tầng 4.* Two independently trained networks reach the same loss. Is the straight line between them low-loss? What do mode connectivity results say, and why does it matter for averaging weights?

**Ý chính.** Hessian XᵀSX với sᵢ = σ(1 − σ) > 0 là nửa xác định dương. Đối xứng hoán vị: mạng tanh 4 nơ-ron ẩn đã huấn luyện có loss 0.0260; hoán vị nơ-ron ẩn: vẫn 0.0260; **trung điểm của hai vector trọng số: 6.4072**, vi phạm bất đẳng thức lồi (lồi thì trung điểm ≤ 0.0260). Số bản sao tương đương: m! = 24 với m = 4; với tanh còn đổi dấu được, 2ᵐ·m! = 384. Tầng 3: điểm yên ngựa (Dauphin, chương 4 chuỗi 4). Tầng 4: Garipov và cộng sự (2018, arXiv 1802.10026) và Draxler và cộng sự (2018, arXiv 1803.00885) tìm được đường cong loss thấp nối hai cực tiểu, còn đường thẳng thường có "rào" [hai paper mới thấy qua kết quả tìm kiếm, chưa mở; cần kiểm chứng trước khi dẫn chi tiết]. Trung bình trọng số hai mạng khác khởi tạo không có nghĩa nếu chưa căn hoán vị (từ khóa: permutation alignment, model merging) [cần kiểm chứng nguồn].

**Lỗi hay gặp.** "Không lồi ⇒ nhiều cực tiểu địa phương tệ"; quên rằng đối xứng tạo ra các nghiệm **cùng loss chính xác**; ở tầng 4 khẳng định quá điều paper nói.

**Giao.** Chương 4 chuỗi 4 tầng 2–3. Mới: không lồi bằng đối xứng, có số.

> **Sample answer (tiers 1–2):** The logistic loss has Hessian XᵀSX, where S is diagonal with entries σ(1 − σ), which are positive, so the Hessian is positive semi-definite and the loss is convex in the weights. For a one-hidden-layer MLP, permuting the hidden units gives a different weight vector with exactly the same loss. If the loss were convex, the midpoint between those two weight vectors could not have a higher loss than the endpoints. In a small tanh network with four hidden units, both endpoints had loss 0.026 and the midpoint had 6.41, so the loss is not convex; every minimum also has 4! = 24 permuted copies, and 384 if we count the sign flips that tanh allows.

**C05 — Generative versus discriminative classifiers** (chương 3, 5)

- *Tầng 1.* When is P(y = 1 ∣ x) exactly a logistic function of x? Derive it from Gaussian class-conditionals.
- *Tầng 2.* Compute the posterior at a point, then change the class prior. Which parameter of the logistic model changes?
- *Tầng 3.* Naive Bayes and logistic regression can have the same functional form. Which one wins with little data and which with a lot, and why?
- *Tầng 4.* You trained on balanced data and deploy where positives are 10%. Fix the model without retraining, and say how you would check calibration.

**Ý chính.** Hiệp phương sai chung ⇒ log-odds tuyến tính, w = Σ⁻¹(μ₁ − μ₀). Ví dụ 1D: μ₀ = 0, μ₁ = 2, σ = 1, prior 0.5: w = 2, b = −2; P(y = 1 ∣ x = 1) = 0.5000; tại x = 2: 0.8808. Prior 0.1: b = −4.1972, P(y = 1 ∣ x = 2) = 0.4509, ngưỡng 0.5 dời sang x = 2.0986 (tính Bayes trực tiếp ra cùng số). Phương sai hai lớp khác nhau ⇒ log-odds có số hạng x² (QDA). Ng và Jordan (NIPS 2001, mức A): mô hình phân biệt có sai số tiệm cận thấp hơn; mô hình sinh tiến về sai số tiệm cận (cao hơn) của nó nhanh hơn, nên có thể thắng khi ít dữ liệu. Tầng 4: chỉ bias đổi, cộng ln(π′/(1 − π′)) − ln(π/(1 − π)) = ln(0.1/0.9) = −2.197; kiểm bằng reliability diagram và ECE (chương 12).

**Lỗi hay gặp.** Quên số hạng prior; nói logistic "luôn tốt hơn"; tưởng Naive Bayes luôn cùng ranh giới với LDA (chỉ khi các giả định tương ứng).

**Giao.** Chương 3 chuỗi 1 tầng 3–4. Mới: LDA → logistic, Ng và Jordan.

> **Sample answer (tiers 1–2):** If the two class-conditional densities are Gaussian with a shared covariance Σ, the quadratic terms in x cancel in the log-odds, so the log-odds is linear in x and the posterior is a logistic function with w = Σ⁻¹(μ₁ − μ₀). In one dimension, with means 0 and 2, unit variance and equal priors, w = 2 and b = −2, so the posterior is 0.5 at x = 1 and 0.88 at x = 2. If the positive prior drops to 0.1, only the bias changes, by ln(0.1/0.9) ≈ −2.2, and the posterior at x = 2 falls to 0.45.

**C06 — Numerical stability from softmax to mixed precision** (chương 2, 4, 6, 9)

- *Tầng 1.* Your softmax returns NaN for logits around 1000. Why, and how do you fix it?
- *Tầng 2.* What range of logits can fp16 exponentiate safely? Compare the range and precision of fp16 and bf16.
- *Tầng 3.* In fp16, what happens to the update w ← w − ηg when w = 1 and ηg = 10⁻⁴? What do mixed-precision recipes do about it?
- *Tầng 4.* Loss scaling, bf16, fp32 master weights, fp32 accumulation: which problem does each solve, and which problems remain?

**Ý chính.** Tầng 1–3: phần suy dẫn chung 3 và ví dụ 14.5. Tầng 4: đoạn Thực tế tính toán sau ví dụ 14.5 (loss scaling cho fp16; bf16 thường không cần nhưng cộng dồn và bản trọng số chính vẫn fp32, vì cập nhật 10⁻⁴ vào trọng số 1 mất hẳn).

**Lỗi hay gặp.** Nhầm epsilon với số dương nhỏ nhất; nói bf16 "chính xác hơn" fp16 (ngược lại: dải rộng hơn, độ chính xác kém hơn); trừ max nhưng quên cộng lại khi tính log-sum-exp.

**Giao.** Chương 2 chuỗi 2 tầng 3; chương 9 chuỗi 4. Mới: số học fp16 và bf16 cụ thể.

> **Sample answer (tiers 1–2):** Exponentiating logits near 1000 overflows to infinity, and infinity divided by infinity is NaN. Softmax does not change if I subtract the same constant from every logit, so I subtract the maximum first; the largest exponential is then exactly 1, and the log-sum-exp is the maximum plus the log of the shifted sum. In fp16 the largest finite value is 65 504, so exp overflows for inputs above ln 65 504 ≈ 11.09. bf16 has roughly the range of fp32, but only 8 bits of precision, so its machine epsilon is about 0.0078, against about 0.001 for fp16: wider range, less precision.

## 14.3 Chuỗi tổng hợp tuần 36: deep learning hiện đại và LLM (chương 7–10)

Nhóm chuỗi thứ hai kiểm xem bạn có biến một kiến trúc thành **chi phí** được không: bao nhiêu byte, bao nhiêu FLOPs, bao nhiêu mili giây, và cùng một phân kỳ KL xuất hiện ở đâu trong một pipeline LLM. Chuỗi "attention → KV cache → chi phí" mà lộ trình tuần 36 nhắc chính là C07 nối với C08, và capstone P1 ở phần Phỏng vấn.

**Phần suy dẫn chung 1: ngân sách bộ nhớ.** Trọng số chiếm N × số byte mỗi tham số. Mỗi lớp attention lưu cho mỗi token một vector key và một vector value cho **mỗi đầu K/V**, nên với số lớp L, số đầu K/V hₖᵥ, chiều mỗi đầu dₕ, b byte mỗi số, độ dài T và batch B:

$$
\text{KV bytes} = 2 \cdot L \cdot h_{kv} \cdot d_h \cdot b \cdot T \cdot B
$$

Hệ số 2 là K và V. Số đầu K/V (không phải số đầu query) là thứ grouped-query attention (GQA) giảm đi. 1 GiB = 2³⁰ byte, lớn hơn 1 GB = 10⁹ byte khoảng 7.4%.

**Ví dụ giải mẫu 14.6 — ngân sách bộ nhớ khi phục vụ mô hình 7B.** Llama-2-7B có 6 738 415 616 tham số (khớp chương 8; mỗi lớp 202 383 360, tính ở đây).

- Trọng số: fp32 25.10 GiB; fp16 13.48 GB = 12.55 GiB; int8 6.28 GiB; 4-bit 3.14 GiB; 4-bit cộng một scale fp16 cho mỗi khối 64 trọng số (0.5 + 2/64 = 0.53125 byte mỗi trọng số) 3.33 GiB.
- KV cache: 2 · 32 · 32 · 128 · 2 = 524 288 byte = 0.5 MiB mỗi token; 4096 token 2.00 GiB; batch 8 là 16.00 GiB. Nếu chỉ có 8 đầu K/V (giả định để so, không phải Llama-2-7B): 0.50 GiB mỗi chuỗi 4096 token.
- GPU **giả định** 80 GiB: còn 80 − 12.55 = 67.45 GiB, chứa 33 chuỗi 4096 token với MHA hoặc 134 với 8 đầu K/V, chưa trừ activation và phân mảnh bộ nhớ. Con số 134 giữ nguyên 12.55 GiB trọng số cho dễ so; một mô hình 8 đầu K/V thật có hai ma trận chiếu K và V nhỏ đi 4 lần, nhẹ hơn 1.50 GiB, nên chứa được 137 chuỗi.
- Laptop 6 GiB của người đọc: fp16 (12.55 GiB) không vừa. 4-bit cộng scale 3.33 GiB, cộng KV 4096 token 2.00 GiB, tổng 5.33 GiB; ngữ cảnh 2048 còn 4.33 GiB. **Chưa tính activation và overhead của framework** [cần đo trên máy của bạn].
- Huấn luyện đầy đủ bằng Adam fp32 (16 byte mỗi tham số, chương 9): 100.4 GiB.

% manim: NganSachBoNho

**Phần suy dẫn chung 2: FLOPs, giờ, mili giây.** Một phép nhân (m × k)(k × n) cần k phép nhân và k phép cộng cho mỗi phần tử kết quả, tổng 2mkn FLOPs. Forward của mạng dày tốn khoảng 2N FLOPs mỗi token; backward tốn khoảng 4N, gồm 2N cho gradient theo activation và 2N cho gradient theo trọng số. Thời gian huấn luyện và cận dưới thời gian giải mã batch 1:

$$
C \approx 6ND, \qquad t_{\text{train}} = \frac{C}{\text{FLOP/s} \times \text{MFU}}, \qquad t_{\text{token}} \ge \frac{N \cdot b + \text{KV bytes}}{\text{bandwidth}}
$$

Ở batch 1, mỗi token phải đọc toàn bộ trọng số từ bộ nhớ nhưng chỉ làm khoảng 1 FLOP trên mỗi byte fp16 đọc vào (2N FLOPs cho 2N byte), nên giải mã bị chặn bởi băng thông (memory-bound), không bởi FLOPs.

**Ví dụ giải mẫu 14.7 — FLOPs → giờ → ms/token.**

- (512 × 1024)(1024 × 256): 2 · 512 · 1024 · 256 = 268 435 456 FLOPs.
- GPT-2 small theo quy ước Lab 5 chương 8 (6N + 12·L·d·T, N là tổng tham số trừ bảng tra embedding vị trí, vẫn gồm LayerNorm và bias): 855 166 464 FLOPs mỗi token ở T = 1024; phần attention chiếm 13.2%. Với 10¹⁰ token: 8.552 × 10¹⁸ FLOPs (6ND thuần với 124 439 808 tham số cho 7.466 × 10¹⁸).
- Ở 100 TFLOP/s **giả định** và MFU (model FLOPs utilization, tỉ lệ FLOP/s thực đạt so với đỉnh) 40%: 8.552 × 10¹⁸ / (4 × 10¹³) = 213 792 giây = 59.4 giờ.
- Giải mã batch 1 mô hình 7B fp16: 13.48 GB trọng số chia 2 TB/s **giả định** = 6.74 ms/token, tối đa khoảng 148 token/s; cộng đọc KV 4096 token: 7.81 ms. Ở 1 TB/s: 13.48 ms và 15.62 ms.

**Phần suy dẫn chung 3: bài toán KL-regularized.** Cực đại hóa phần thưởng kỳ vọng trừ β lần KL tới chính sách tham chiếu πref:

$$
\max_{\pi}\; \mathbb{E}_{y \sim \pi}[r(y)] - \beta\, \mathrm{KL}(\pi \,\|\, \pi_{\text{ref}}) \quad\Rightarrow\quad \pi^*(y) = \frac{\pi_{\text{ref}}(y)\, e^{r(y)/\beta}}{Z}, \qquad \text{giá trị tối ưu} = \beta \ln Z
$$

Suy dẫn: viết mục tiêu thành −β·KL(π ‖ π*) + β ln Z; KL không âm và bằng 0 khi π = π*. Lấy log hai vế: r(y) = β ln(π*(y)/πref(y)) + β ln Z, đó là "phần thưởng ngầm" của DPO; Z triệt tiêu khi lấy hiệu giữa hai câu trả lời.

**Ví dụ giải mẫu 14.8 — đồ chơi hai đáp án.** πref = (0.5, 0.5), r = (1, 0).

- β = 0.5: π* = (0.8808, 0.1192), 𝔼[r] = 0.8808, KL = 0.3278, giá trị mục tiêu 0.7169 = β ln Z (khớp; quét lưới cũng cho p tối ưu 0.8808).
- β = 1: (0.7311, 0.2689). β = 5: (0.5498, 0.4502): β lớn kéo về πref.
- Phần thưởng ngầm β·ln(π*/πref) với β = 0.5 là (0.2831, −0.7169): hiệu đúng bằng hiệu phần thưởng 1.0.
- Phân rã cross-entropy: p = (0.7, 0.2, 0.1), q = (0.5, 0.3, 0.2): H(p, q) = 0.8869 = H(p) 0.8018 + KL(p ‖ q) 0.0851.

% manim: CayChuoiNoiChuong

**Thực tế tính toán.** Mọi băng thông, TFLOP/s và dung lượng GPU trong các chuỗi tuần này (80, 40, 24 GiB; 1, 2 TB/s; 100 TFLOP/s) là **số giả định do người hỏi đưa ra**. Sách không gán chúng cho GPU cụ thể nào; chương 8 đã ghi rằng con số cụ thể phải kiểm theo datasheet. Khi được hỏi, nói rõ: "Assuming 2 TB/s of memory bandwidth…".

**C07 — From weights to a served request: memory** (chương 8, 9)

- *Tầng 1.* How much memory do the weights of a 7B model take in fp16, int8 and 4-bit? Answer in GB and GiB.
- *Tầng 2.* Derive the KV cache size per token, and compute it for a Llama-2-7B-shaped model at 4096 tokens.
- *Tầng 3.* On a hypothetical 80 GiB GPU, how many 4096-token sequences can you serve with full multi-head attention versus 8 KV heads? What did your estimate leave out?
- *Tầng 4.* Could you run it on a 6 GiB laptop GPU? What would you change first, and what does each change cost?

**Ý chính.** Tầng 1–3: ví dụ 14.6. Tầng 4: lượng tử 4-bit trước, cắt ngữ cảnh, rồi lượng tử KV cache hoặc offload; mỗi thay đổi phải đo lại chất lượng (perplexity, tác vụ). Chuyển một checkpoint MHA sang ít đầu K/V không chỉ là sửa cấu hình: paper GQA dùng uptraining (chương 8 đã dẫn paper; chi tiết uptraining [cần kiểm chứng trong paper]).

**Lỗi hay gặp.** Lẫn GB với GiB (chênh 7.4%); quên hệ số 2 của K và V; dùng số đầu query thay số đầu K/V; quên nhân batch; quên activation.

**Giao.** Chương 8 chuỗi 4 tầng 4; chương 9 chuỗi 4. Mới: ngân sách phục vụ trọn vẹn và bài toán laptop của bạn.

> **Sample answer (tiers 1–2):** In fp16 each parameter takes two bytes, so 6.74 billion parameters take 13.48 GB, which is 12.55 GiB; int8 halves that to 6.28 GiB, and 4-bit gives 3.14 GiB, or about 3.33 GiB once I add one fp16 scale per block of 64 weights. The KV cache stores one key vector and one value vector per layer, per KV head and per token, so the bytes per token are 2 × layers × KV heads × head dimension × bytes per value. For a Llama-2-7B-shaped model in fp16 that is 2 × 32 × 32 × 128 × 2 = 524 288 bytes, half a mebibyte, so a 4096-token sequence needs 2 GiB.

**C08 — FLOPs to wall-clock** (chương 8, 9)

- *Tầng 1.* How many FLOPs does an (m × k)(k × n) matrix multiplication take, and why the factor 2?
- *Tầng 2.* Why is training about 6N FLOPs per token? Where does the 4N of the backward pass come from?
- *Tầng 3.* Estimate the cost and time of training GPT-2 small on 10 billion tokens at a hypothetical 100 TFLOP/s with 40% MFU. What does 6ND leave out?
- *Tầng 4.* Decoding is memory-bound. Give a lower bound on milliseconds per token at batch 1 from bandwidth, and explain what batching changes and what it does not.

**Ý chính.** Ví dụ 14.7. 6ND bỏ qua số hạng attention theo T, chi phí tính lại nếu dùng activation checkpointing, và hiệu suất thực (MFU). Tầng 4: batching chia đều chi phí đọc trọng số cho nhiều chuỗi, nhưng không chia được chi phí đọc KV cache riêng của từng chuỗi (chương 8).

**Lỗi hay gặp.** Quên hệ số 2; đếm phép tra embedding là FLOPs; dùng FLOP/s đỉnh thay vì thực đạt; trộn TB với TiB.

**Giao.** Chương 8 chuỗi 4 tầng 3–4; chương 9 chuỗi 2 tầng 2. Mới: đổi ra giờ và ms/token.

> **Sample answer (tiers 1–2):** Each output element of an (m × k)(k × n) product needs k multiplications and k additions, so the product costs 2mkn FLOPs. In a dense network every parameter takes part in one multiply and one add per token, so the forward pass costs about 2N FLOPs per token. The backward pass computes two quantities of the same size, the gradient with respect to the activations, which is passed to the previous layer, and the gradient with respect to the weights, each about 2N, so training costs about 6N per token. For GPT-2 small at a context of 1024, attention accounts for about 13% of the FLOPs per token, which the 6N rule ignores.

**C09 — KL everywhere in an LLM pipeline** (chương 3, 9, 10)

- *Tầng 1.* Show that cross-entropy equals entropy plus KL. What does minimizing it do during pretraining?
- *Tầng 2.* In knowledge distillation with temperature T, why is the soft-target loss multiplied by T²?
- *Tầng 3.* Solve max over π of 𝔼π[r] − β·KL(π ‖ πref) on a two-answer toy problem. What does β control?
- *Tầng 4.* Where else does a KL appear (VAE, DPO's implicit reward), in which direction, and what goes wrong when β is too small?

**Ý chính.** Ví dụ 14.8. Chưng cất: Hinton, Vinyals và Dean (2015, arXiv 1503.02531, mục 2, mức A qua ar5iv) nói độ lớn gradient của soft target tỉ lệ với 1/T², nên nhân T² khi dùng cả nhãn cứng và nhãn mềm. Script: chuẩn gradient nhân T² là 0.3530, 0.4105, 0.3974 ở T = 1, 2, 4, tức gần hằng số khi T lớn nhưng không đúng tuyệt đối khi T nhỏ. β nhỏ: chính sách rời xa πref, dễ reward hacking (chương 9).

**Lỗi hay gặp.** Nhầm chiều của KL; nói T² "để loss to hơn" mà không nêu lý do gradient; quên Z triệt tiêu trong DPO.

**Giao.** Chương 3 chuỗi 3; chương 9 chuỗi 3 tầng 2–3; chương 10 chuỗi A. Mới: một KL ở bốn chỗ.

> **Sample answer (tiers 1–2):** Cross-entropy H(p, q) = −∑ p log q splits into the entropy H(p), which does not depend on the model, plus KL(p ‖ q), so minimizing cross-entropy over q is the same as minimizing the KL divergence from the data distribution to the model. In pretraining, p is the empirical next-token distribution, so this is maximum likelihood. In distillation with temperature T, the gradient of the soft-target term scales roughly as 1/T², so Hinton and colleagues multiply that term by T² to keep its weight comparable to the hard-label term when T changes; in a small check, the gradient norm times T² was 0.35, 0.41 and 0.40 at T = 1, 2 and 4.

**C10 — Temperature, entropy and calibration** (chương 3, 9, 12)

- *Tầng 1.* What does dividing the logits by T do to the distribution and its entropy? Compute it for logits (2, 1, 0).
- *Tầng 2.* The same knob appears in decoding, distillation and contrastive losses. What does it do in each?
- *Tầng 3.* What is calibration, how do you measure it, and why does temperature scaling fix it without changing accuracy?
- *Tầng 4.* A colleague claims post-training made the chat model less calibrated than the base model. How would you test the claim?

**Ý chính.** Logit (2, 1, 0): T = 0.5 cho (0.8668, 0.1173, 0.0159), entropy 0.4411 nats; T = 1 cho (0.6652, 0.2447, 0.0900), 0.8324; T = 2 cho (0.5065, 0.3072, 0.1863), 1.0202; trần ln 3 = 1.0986. Temperature scaling chia mọi logit cho cùng một T học trên tập validation, giữ argmax nên accuracy không đổi; Guo và cộng sự (ICML 2017, PMLR 70, mức A) cho thấy mạng hiện đại hiệu chỉnh kém và phương pháp đơn giản này "surprisingly effective". ECE ví dụ ở chương 12 (ví dụ 12.9). Tầng 4: đây là giả thuyết cần đo (cùng tập, cùng định dạng câu hỏi, reliability diagram, ECE với nhiều cách chia bin), đừng khẳng định như sự thật.

**Lỗi hay gặp.** Tưởng T đổi argmax; quên ECE phụ thuộc cách chia bin; coi entropy cao là "mô hình tốt hơn".

**Giao.** Chương 9 chuỗi 1 tầng 4; chương 12 chuỗi 4 tầng 2. Mới: temperature scaling.

> **Sample answer (tiers 1–2):** Dividing the logits by T > 1 flattens the distribution and raises its entropy, and T < 1 sharpens it. For logits (2, 1, 0) the entropy is 0.44 nats at T = 0.5, 0.83 at T = 1 and 1.02 at T = 2, against a maximum of ln 3 ≈ 1.10. The same knob appears in sampling, where it trades diversity for quality; in distillation, where a high T exposes the teacher's small probabilities; and in contrastive losses, where a small temperature concentrates the gradient on the hardest negatives. In all three it never changes the ordering of the logits, only how peaked the distribution is.

**C11 — From a contrastive loss to a dense retriever** (chương 10, 12)

- *Tầng 1.* Write InfoNCE with in-batch negatives, and show it is a softmax cross-entropy over N candidates.
- *Tầng 2.* What is the loss at initialization, and what does ln N minus the loss tell you?
- *Tầng 3.* How would you train a bi-encoder retriever with it? Discuss temperature, batch size and hard negatives.
- *Tầng 4.* Hard-negative mining can produce false negatives. How do they hurt, and how would you detect them?

**Ý chính.** N = 1024: ln N = 6.9315, đo với điểm số gần 0 được 6.9319. Cận dưới thông tin tương hỗ I ≥ ln N − L: với L = 2.0 là 4.9315 nats; cận không bao giờ vượt ln N. N = 256: ln N = 5.5452. Tầng 3–4: τ nhỏ dồn trọng số vào âm khó nhất; âm giả (đoạn thật ra liên quan nhưng bị coi là âm) kéo các cặp liên quan ra xa nhau; phát hiện bằng cách đọc các mẫu âm có điểm cao, so với nhãn của người, hoặc lọc bằng một reranker.

**Lỗi hay gặp.** Coi cận là ước lượng chính xác của thông tin tương hỗ; quên rằng tăng batch đổi cả số lượng lẫn độ khó của mẫu âm.

**Giao.** Chương 10 chuỗi D tầng 1–2; chương 12 chuỗi 1. Mới: InfoNCE → retriever, âm giả.

> **Sample answer (tiers 1–2):** With a batch of N query–passage pairs, InfoNCE scores each query against all N passages, divides the scores by a temperature and applies softmax cross-entropy with the matching passage as the correct class, so it is an N-way classification loss whose negatives are the other passages in the batch. At initialization, when all scores are about equal, the loss is ln N, which is 6.93 for N = 1024. The loss also gives a lower bound on mutual information, I ≥ ln N − L, so a loss of 2.0 at N = 1024 certifies at least 4.93 nats. The bound can never exceed ln N, so it is a lower bound, not an estimate.

**C12 — Comparing generative models honestly** (chương 3, 9, 10)

- *Tầng 1.* Convert a negative log-likelihood of 3000 nats on a 32 × 32 × 3 image into bits per dimension, and a cross-entropy of 3.0 nats per token into perplexity and bits per token.
- *Tầng 2.* Why can you not compare perplexity across tokenizers, or an ELBO-based bits/dim with an exact autoregressive likelihood, without care?
- *Tầng 3.* Compare the sampling cost of an autoregressive LM, a diffusion model and a VAE.
- *Tầng 4.* Likelihood and sample quality can disagree. When, and what would you report?

**Ý chính.** 3000 / (3072 · ln 2) = 1.4089 bits/dim. CE 3.0 nats ⇒ perplexity e³ = 20.086, 3.0/ln 2 = 4.3281 bits/token; CE 2.0 ⇒ 7.389 và 2.8854; perplexity 20 ⇒ log₂ 20 = 4.3219 bits. So giữa tokenizer: quy về bits per byte hoặc per character (chương 9). ELBO là cận dưới của log-likelihood nên bits/dim tính từ ELBO là cận **trên** của bits/dim thật. Chi phí lấy mẫu: mô hình tự hồi quy T lượt (có KV cache), diffusion S lượt của cả mạng, VAE một lượt decoder.

**Lỗi hay gặp.** Lẫn ln với log₂; lẫn per token với per character; so ELBO với NLL chính xác như cùng loại.

**Giao.** Chương 3 chuỗi 3 tầng 4; chương 9 chuỗi 1 tầng 3; chương 10 chuỗi C tầng 4. Mới: đổi đơn vị trọn vẹn, so ba họ mô hình.

> **Sample answer (tiers 1–2):** Bits per dimension is the negative log-likelihood in nats divided by the number of dimensions and by ln 2, so 3000 nats on a 32 × 32 × 3 image is 3000 / (3072 × ln 2) ≈ 1.41 bits per dimension. For text, perplexity is the exponential of the cross-entropy in nats, so 3.0 nats per token is a perplexity of about 20.1, or 4.33 bits per token. Perplexity depends on what one token is, so I would compare models with different tokenizers only after converting to bits per byte or per character. An ELBO gives only an upper bound on bits per dimension, so it is not directly comparable with an exact likelihood either.

## 14.4 Chuỗi tổng hợp tuần 37: thực nghiệm, NLP, dự án, paper, và khi không biết (chương 11–12)

Nhóm chuỗi thứ ba chuyển từ "bạn biết gì" sang "bạn chứng minh một khẳng định thế nào": cải thiện có thật không, đo truy hồi bằng tay, bảo vệ dự án của chính mình, đọc một paper ngay tại chỗ, và trả lời một câu hỏi mình chưa học.

**Phần suy dẫn chung 1: sai số chuẩn, McNemar, cỡ mẫu.** Accuracy p trên n câu độc lập có sai số chuẩn √(p(1 − p)/n). Khi hai hệ thống trả lời **cùng** n câu, hai accuracy tương quan, nên kiểm định hai mẫu độc lập sai. McNemar chỉ dùng b (A đúng, B sai) và c (A sai, B đúng):

$$
\chi^2 = \frac{(b - c)^2}{b + c}, \qquad \chi^2_{\text{cc}} = \frac{(|b - c| - 1)^2}{b + c}
$$

so với phân phối χ² một bậc tự do. Cỡ mẫu để phát hiện mức tăng δ (α = 0.05 hai phía, power 0.8, z₀.₉₇₅ = 1.96, z₀.₈ = 0.8416), không ghép cặp (mỗi hệ thống) và ghép cặp với tỉ lệ câu bất đồng ψ:

$$
n_{\text{unpaired}} = \frac{(z_{0.975} + z_{0.8})^2 \left(p_1 q_1 + p_2 q_2\right)}{\delta^2}, \qquad n_{\text{paired}} = \frac{\left(z_{0.975}\sqrt{\psi} + z_{0.8}\sqrt{\psi - \delta^2}\right)^2}{\delta^2}
$$

với q = 1 − p.

**Ví dụ giải mẫu 14.9 — cải thiện có thật không.**

- Accuracy 0.85 trên 1000 câu: sai số chuẩn 0.0113, nửa khoảng tin cậy 95% là 0.0221.
- So 0.85 với 0.84 như hai mẫu độc lập: sai số chuẩn của hiệu 0.0162, z = 0.618, p = 0.537.
- Ghép cặp (McNemar) với b = 30, c = 20: χ² = 100/50 = 2.000, p = 0.157; có hiệu chỉnh liên tục 1.620, p = 0.203; kiểm định chính xác (nhị thức) p = 0.203. Gấp đôi dữ liệu với cùng tỉ lệ (b = 60, c = 40): χ² = 4.000, p = 0.0455. Hiệu không đổi, kết luận đổi.
- Cỡ mẫu phát hiện mức tăng 1 điểm quanh 0.85: không ghép cặp 20 556.2, làm tròn lên 20 557 câu mỗi hệ thống; ghép cặp với 5% câu bất đồng 3922.1, làm tròn lên 3923 (mô phỏng power ở n = 3922 được 0.787, sát 0.8). Ghép cặp cần ít dữ liệu hơn khoảng 5 lần **khi hai hệ thống bất đồng ít**.

**Phần suy dẫn chung 2: metric truy hồi tính tay.** Recall@k là tỉ lệ tài liệu liên quan nằm trong top k. MRR là trung bình của 1/hạng của tài liệu liên quan đầu tiên (0 nếu không có). nDCG@k chiết khấu mỗi tài liệu liên quan theo log₂ của vị trí rồi chia cho giá trị tốt nhất có thể:

$$
\mathrm{DCG@}k = \sum_{i=1}^{k} \frac{\mathrm{rel}_i}{\log_2(i + 1)}, \qquad \mathrm{nDCG@}k = \frac{\mathrm{DCG@}k}{\mathrm{IDCG@}k}
$$

Số tính tay nằm trong chuỗi C14.

**Phần suy dẫn chung 3: speculative decoding từ nguyên lý đầu**, như một bài mẫu cho "I don't know, but…". Giả định mỗi token nháp được chấp nhận độc lập với xác suất α, mô hình nháp đề xuất k token, và lượt kiểm luôn sinh thêm một token của mô hình đích (thay token bị từ chối đầu tiên, hoặc nối sau k token nếu tất cả được nhận). Số token có được sau i token nháp đầu tiên đều được nhận với xác suất αⁱ, nên:

$$
\mathbb{E}[\text{tokens per target pass}] = \sum_{i=0}^{k} \alpha^{i} = \frac{1 - \alpha^{k+1}}{1 - \alpha}, \qquad \text{speedup} = \frac{1 - \alpha^{k+1}}{(1 - \alpha)(k c + 1)}
$$

với c là chi phí một lượt mô hình nháp so với một lượt mô hình đích.

**Ví dụ giải mẫu 14.10 — "I don't know, but…": speculative decoding.** α = 0.8, k = 4: 1 + 0.8 + 0.64 + 0.512 + 0.4096 = 3.3616 token mỗi lượt (mô phỏng 3.3511); α = 0.6, k = 4: 2.3056; α = 0.5, k = 3: 1.8750. Tăng tốc với α = 0.8, k = 4: c = 0.05 cho 2.801, c = 0.2 cho 1.868. Với α = 0.8, c = 0.05, k tối ưu là 8 (3.0921; k = 7 cho 3.0823, k = 9 cho 3.0780). **Đối chiếu nguồn:** công thức đúng là Eq. (1) và Theorem 3.8 của Leviathan, Kalman và Matias (ICML 2023, PMLR 202; arXiv 2211.17192, đọc qua ar5iv ngày 16/11/2026, trang PMLR mở ngày 17/11/2026, mức A); abstract báo tăng tốc 2X–3X trên T5-XXL "with identical outputs". Bài học: bạn suy ra được công thức từ giả định độc lập **trước khi** biết paper, rồi nói "I would check this against the paper". Giả định i.i.d. là giả định (paper cũng nói rõ); thực tế α thay đổi theo ngữ cảnh.

% manim: TokenNhapHinhHoc

**Đào sâu dự án: bộ câu hỏi và tài liệu một trang.** Lộ trình giao ba việc cho luồng A quanh mục này. Tuần 35: soạn câu trả lời tiếng Anh cho 5 câu hỏi sâu về dự án, ví dụ "Why this baseline?", "What would you change?", "How many seeds, and what was the spread?", "What exactly did you do yourself?", "What result would make you drop the claim?". Tuần 37: phỏng vấn thử 30 phút về dự án với bạn hoặc giảng viên, ghi lại câu không trả lời được; chuỗi C15 là khung. Tuần 38: một trang tóm tắt gồm khẳng định trung tâm một câu, 5 con số cần nhớ (mỗi số kèm baseline, số seed, độ dao động) và 3 hạn chế bạn tự nói trước khi bị hỏi. Chương 13 đã dạy chuỗi "một con số" từ góc hồ sơ và video; ở đây bạn đi tiếp sang thiết kế thí nghiệm và ngân sách.

**Khung trình bày kỹ thuật 5 phút** (w36-a; quy tắc của sách): vấn đề 30 giây; phương pháp 90 giây, trong đó nói rõ phần nào bạn tự làm; bằng chứng 90 giây (khẳng định trung tâm, baseline, số seed, độ dao động, một ablation); hạn chế 60 giây; câu hỏi mở 30 giây. Ở 140 từ/phút là khoảng 700 từ. Chấm bằng hàng "Đào sâu dự án" và "Cách nói" của thang 30 điểm ở 14.5.

**Đọc paper ngay tại chỗ: ngân sách 15 phút (quy tắc của sách).** 3 phút cho tiêu đề, abstract, hình 1; 7 phút cho phương pháp và bảng kết quả chính; 5 phút điền bản đồ khẳng định – bằng chứng của chương 11. Chọn paper từ phiếu 10 paper của chương 12.

**C13 — Is the improvement real?** (chương 3, 11)

- *Tầng 1.* Model A scores 0.85 and the baseline 0.84 on the same 1000 test examples. What is the standard error of each accuracy?
- *Tầng 2.* Why is an unpaired test the wrong tool here, and what does McNemar's test compute? Use b = 30 and c = 20.
- *Tầng 3.* How many test examples would you need to detect a one-point gain? Compare the unpaired and paired designs.
- *Tầng 4.* You also have three seeds per system. How do you combine seed variance and test-set variance, and what do you report?

**Ý chính.** Tầng 1–3: ví dụ 14.9. Tầng 4: chương 11 (khoảng tin cậy theo seed với hệ số t, ghép cặp theo seed, không báo best-of-k); phương sai tổng có hai nguồn, báo cả hai.

**Lỗi hay gặp.** Tính sai số chuẩn của hiệu như hai mẫu độc lập khi dùng cùng tập test; đọc p = 0.157 như "84% xác suất A tốt hơn"; quên rằng gấp đôi n đổi kết luận (p = 0.0455) dù hiệu không đổi.

**Giao.** Chương 3 chuỗi 4; chương 11 chuỗi 3. Mới: McNemar bằng số, cỡ mẫu ghép cặp.

> **Sample answer (tiers 1–2):** The standard error of an accuracy p on n independent examples is √(p(1 − p)/n), about 0.0113 for 0.85 on 1000 examples, so the 95% interval is roughly ±0.022. Treating the two systems as independent gives a standard error of the difference of 0.016 and z = 0.62, but that test is wrong here, because both systems answered the same questions and their errors are correlated. McNemar's test uses only the questions where they disagree: with 30 where A is right and the baseline wrong, and 20 the other way, χ² = (30 − 20)²/50 = 2.0 and p ≈ 0.16, so a one-point gain on 1000 questions is not convincing.

**C14 — Retrieval metrics by hand** (chương 3, 11, 12)

- *Tầng 1.* Define recall@k, MRR and nDCG@k. Compute MRR when the first relevant document is at ranks 1, 3, not retrieved, and 2.
- *Tầng 2.* Compute nDCG@3 for relevance lists (0, 1, 1) and (1, 0, 1). Why is the second one higher?
- *Tầng 3.* Your recall@5 is 0.72 on 200 questions. What is its standard error, and is a 0.03 gain meaningful?
- *Tầng 4.* For a RAG system whose generator reads the top 5 passages, which metric would you choose, and when can MRR mislead?

**Ý chính.** MRR = (1 + 1/3 + 0 + 1/2)/4 = 0.4583. nDCG@3 với (0, 1, 1): DCG = 1/log₂3 + 1/log₂4 = 1.1309, IDCG = 1 + 1/log₂3 = 1.6309, nDCG = 0.6934; với (1, 0, 1): 0.9197, vì vị trí đầu bị chiết khấu ít nhất. Sai số chuẩn recall@5 = √(0.72 · 0.28/200) = 0.0317 nếu recall mỗi câu là 0 hoặc 1 (mỗi câu một tài liệu liên quan), nên 0.03 chỉ khoảng 1 sai số chuẩn nếu coi độc lập; dùng phân tích ghép cặp (C13). Khi một câu có nhiều tài liệu liên quan, recall mỗi câu là phân số (1/3, 1/2, …), không còn là biến Bernoulli: dùng độ lệch chuẩn mẫu của recall từng câu chia √n (√(p(1 − p)/n) khi đó chỉ là cận trên), và ghép cặp như C13. Tầng 4: generator đọc cả top 5 nên recall@5 sát mục tiêu hơn MRR; MRR chỉ nhìn tài liệu liên quan đầu tiên, bỏ qua câu hỏi cần nhiều đoạn.

**Lỗi hay gặp.** Dùng log tự nhiên trong DCG; tính IDCG từ danh sách đã truy hồi thay vì từ mọi tài liệu liên quan (ví dụ ở đây giả định các tài liệu liên quan đều nằm trong danh sách).

**Giao.** Chương 12 chuỗi 2. Mới: tính tay, nối với sai số chuẩn.

> **Sample answer (tiers 1–2):** Recall at k is the fraction of the relevant documents that appear in the top k, MRR averages one over the rank of the first relevant document, with zero when none is retrieved, and nDCG discounts each relevant document by the base-2 log of its position plus one and divides by the best possible ordering. With first relevant documents at ranks 1, 3, none and 2, MRR is (1 + 1/3 + 0 + 1/2)/4 ≈ 0.46. For relevance lists (0, 1, 1) and (1, 0, 1), nDCG at 3 is 0.69 and 0.92: the second list puts a relevant document in the first position, which has the smallest discount.

**C15 — Project deep dive** (chương 11, 12, 13; dự án luồng A)

- *Tầng 1.* State your project's main claim in one sentence, with the number and the baseline.
- *Tầng 2.* What experiment would falsify it, and what result would change your mind?
- *Tầng 3.* A skeptical reviewer asks for a stronger baseline and an interaction analysis. Plan the runs within 20 GPU-hours.
- *Tầng 4.* If you restarted, what would you do differently, and what is the biggest remaining threat to validity?

**Ý chính.** Dùng số giả của chương 13 làm ví dụ (recall@5 từ 0.61 lên 0.72, 200 câu, 3 seed). Ngân sách, giả định mỗi lần chạy 0.5 giờ GPU: ba thành phần bật/tắt, 2³ cấu hình × 3 seed = 24 lần chạy = 12 giờ, còn 8 giờ để tinh chỉnh baseline mạnh hơn với ngân sách tương đương; nếu chỉ cần ablation một chiều (đầy đủ và bỏ từng thành phần): 4 × 3 = 12 lần = 6 giờ. Mối đe dọa lớn nhất của dự án mẫu: tập test do một người gán nhãn chính, một lĩnh vực.

**Lỗi hay gặp.** Baseline được tinh chỉnh ít hơn phương pháp của mình; chạy ablation trên tập test; nêu nhiều con số mà không có một khẳng định trung tâm.

**Giao.** Chương 13 chuỗi 1 và 3; chương 11 chuỗi 2 tầng 4. Mới: kế hoạch thí nghiệm có ngân sách giờ.

> **Sample answer (tiers 1–2):** For illustration only: my main claim is that adding a citation-graph step to dense retrieval raised recall at five from 0.61 to 0.72 on 200 hand-labeled questions, averaged over three seeds, against the same pipeline without the graph step. The experiment that could falsify it is the same comparison on a second subject area, or against a stronger baseline, for example a tuned hybrid of keyword and dense retrieval with the same compute. If the gain on the new field fell inside the spread I saw across seeds, I would stop claiming that the method generalizes and describe it as specific to this citation network.

**C16 — Reading a paper live** (chương 11, 12)

- *Tầng 1.* Summarize a paper from your reading list in three sentences: problem, method, evidence.
- *Tầng 2.* What is its key assumption, and where in the paper is it tested, or not?
- *Tầng 3.* Name the one experiment most likely to break the main claim, and estimate its cost.
- *Tầng 4.* How would you extend it in two weeks on one 6 GB GPU, and what result would make it worth a workshop paper?

**Ý chính.** Chọn từ 10 paper của chương 12, theo ngân sách 15 phút ở trên. Tầng 3–4 dùng C07, C08 để ước lượng bộ nhớ và FLOPs của thí nghiệm đề xuất, và nói rõ nó có chạy được trên phần cứng mình nêu không.

**Lỗi hay gặp.** Đọc lại abstract thay vì tóm tắt; chê mà không đề xuất phép thử; đề xuất thí nghiệm không chạy được trên phần cứng vừa nói.

**Giao.** Chương 11 chuỗi 1. Mới: "mở rộng trong 2 tuần" kèm chi phí.

> **Sample answer (tiers 1–2):** For illustration, take the speculative decoding paper from chain C18. The problem is that decoding a large model one token at a time is slow, because each step is limited by reading the weights from memory. The method lets a small draft model propose several tokens and the large model verify them in one parallel pass, with an accept–reject rule that keeps the output distribution identical to the large model's; the abstract reports a 2X–3X speedup on T5-XXL with identical outputs. The key assumption in the analysis is that acceptances are independent with a fixed rate, which the paper states as a simplification, so the first thing I would measure is how the acceptance rate varies across positions and prompts.

**C17 — Tokenization cost for Vietnamese** (chương 8, 9, 12)

- *Tầng 1.* Why does the same Vietnamese text cost more tokens than English under an English-trained BPE?
- *Tầng 2.* If a Vietnamese text takes about 2.9 times as many tokens as its English translation, what happens to the effective context, the KV cache and the attention cost?
- *Tầng 3.* How would you measure this "tokenization premium" fairly, and how much of it would a Vietnamese-trained tokenizer win back?
- *Tầng 4.* Should you extend an existing LLM's vocabulary for Vietnamese? What must be retrained, and how would you evaluate it?

**Ý chính.** Hai con số của Lab 1 chương 9 (kho đồ chơi khoảng 1 KB mỗi thứ tiếng, BPE 150 phép gộp; chỉ minh họa hướng) đo **hai thứ khác nhau**. (1) Dưới tokenizer học trên tiếng Anh, câu Việt 167 token, câu Anh song song 58 token: **167/58 ≈ 2.88 lần**; một phần vì câu Việt cũng dài hơn tính theo byte (178 so với 127). (2) Con số 2.2 là **cùng câu Việt** qua tokenizer học trên tiếng Anh (167 token) so với tokenizer học trên tiếng Việt (76 token): cái giá của tokenizer sai ngôn ngữ, không phải tỉ lệ Việt/Anh. Với 2.88 lần: ngữ cảnh 4096 token chứa lượng nội dung bằng khoảng 4096/2.88 ≈ 1423 token tiếng Anh; KV cache và chi phí tuyến tính mỗi văn bản tăng 2.88 lần; số hạng attention, tỉ lệ với bình phương độ dài, tăng khoảng 2.88² ≈ 8.3 lần. Tầng 3: kho song song, so số token trên byte hoặc trên ký tự, chuẩn hóa Unicode NFC trước (chương 9); Petrov và cộng sự (NeurIPS 2023) là nguồn chương 9 đã dẫn. Tokenizer học trên tiếng Việt đưa câu Việt từ 167 về 76 token (2.2 lần), trên kho đồ chơi này. Tầng 4: embedding mới cần khởi tạo và huấn luyện tiếp, đánh giá bằng bits per byte và tác vụ; mọi đề xuất ở tầng này là giả thuyết.

**Lỗi hay gặp.** So perplexity giữa hai tokenizer; quên NFC và NFD; khẳng định tỉ lệ cho tokenizer của GPT hay Llama khi chưa đo.

**Giao.** Chương 9 chuỗi 1 tầng 4 và Lab 1. Mới: hệ quả chi phí định lượng.

> **Sample answer (tiers 1–2):** BPE merges are learned from the training text, so if that text is mostly English, frequent English words become single tokens while Vietnamese syllables, with their diacritics, are split into several byte-level pieces. If a Vietnamese sentence takes about 2.9 times as many tokens as its English translation, a 4096-token context holds about as much Vietnamese content as 1400 tokens of English, and the KV cache and the per-token costs of a document grow about 2.9 times. The attention term grows with the square of the length, so it grows about 8 times. The 2.9 ratio comes from a toy corpus in my notes, so I would measure the real ratio on a parallel corpus before relying on it.

**C18 — "I don't know, but here is how I would find out": speculative decoding** (chương 8, 9; chủ đề sách chưa dạy)

- *Tầng 1.* Have you heard of speculative decoding? If not, given that decoding is memory-bound, how might a small draft model help?
- *Tầng 2.* Derive the expected number of tokens per target-model pass if each draft token is accepted independently with probability α and the draft proposes k tokens.
- *Tầng 3.* Add the draft cost c relative to one target pass. What is the speedup, and which k is best for α = 0.8 and c = 0.05?
- *Tầng 4.* When does it stop helping, and why can the output distribution stay exactly that of the target model?

**Ý chính.** Phần suy dẫn chung 3 và ví dụ 14.10. Tầng 4: tăng tốc giảm khi α thấp (bản nháp lệch phân phối), khi c lớn, hoặc khi batch lớn làm mô hình đích chuyển sang bị chặn bởi FLOPs thay vì băng thông (suy luận từ chương 8). Phân phối đầu ra giữ nguyên nhờ bước chấp nhận – từ chối của paper; chi tiết bước lấy mẫu lại cần đọc mục 2 của paper.

**Lỗi hay gặp.** Nói số token kỳ vọng là kα; quên token "thưởng" của mô hình đích (số hạng α⁰ = 1); bịa con số tăng tốc.

**Giao.** Không có (chương 9 chỉ ghi từ khóa). Đây là chuỗi mẫu cho việc luyện "I don't know…" của tuần 37.

> **Sample answer (tiers 1–2):** I have not studied speculative decoding in detail. What I know is that decoding at batch one is limited by reading the weights from memory, so checking several tokens in one forward pass costs about the same as generating one. So I would guess that a cheap model drafts a few tokens and the large model checks them in parallel, keeping the ones it agrees with and adding one token of its own. If each draft token is accepted independently with probability α and the draft proposes k tokens, the expected number of tokens per large-model pass is 1 + α + … + αᵏ = (1 − αᵏ⁺¹)/(1 − α), which is 3.36 for α = 0.8 and k = 4; I would check this model and the acceptance rule against the original paper.

## 14.5 Câu hỏi hành vi và phỏng vấn thử 60 phút (tuần 38)

Câu hỏi hành vi ("Tell me about a time…") kiểm cách bạn làm việc thật, và câu trả lời tốt có cùng một cấu trúc: STAR, tức Situation, Task, Action, Result, với phần Action dài nhất và nói bằng "I". Tuần 38 dùng mục này để soạn 6 câu chuyện và chạy một buổi phỏng vấn thử trọn vẹn.

**STAR theo nguồn chính thức của nhà tuyển dụng và của trường** (không phải của Qualcomm hay VinAI):

- **Amazon**, "Amazon interview guide" (08/04/2025, mức A) và "Amazon job interview questions, tips, preparation" (12/06/2024, mức A): STAR là nền tảng của phỏng vấn hành vi; nêu số liệu cho tác động ("measurable results"); dùng "I" để nói rõ đóng góp của chính mình.
- **MIT CAPD**, "Using the STAR method" (đăng 13/06/2022, sửa 30/08/2023, mức A): tỉ trọng gợi ý Situation khoảng 20%, Task 10%, Action 60%, Result 10%; chuẩn bị **3–5 câu chuyện**, nhớ dàn ý chứ không học thuộc; dùng "I"; có kết quả đo được và bài học; trả lời trung thực, vì mâu thuẫn làm mất tin cậy.
- Các blog "STAR cho Google/Meta" là mức C; sách không dùng làm căn cứ.

**Sách yêu cầu 6 câu chuyện** (lộ trình tuần 38; nhiều hơn mức 3–5 của MIT, lựa chọn của sách) để phủ 6 loại câu hỏi dưới đây, mỗi câu 2 phút, khoảng 280 từ; một câu chuyện có thể phục vụ hai loại nếu có hai cách kết. Câu hỏi là câu điển hình do sách diễn đạt; không chương trình nào công khai bộ câu hỏi hành vi (D).

| Loại | Câu hỏi | Nối tiếp hay gặp | Người phỏng vấn nghe gì (mô hình của sách) |
| --- | --- | --- | --- |
| 1. Làm việc nhóm, vai trò | "Tell me about a team project. What exactly was your part?" / "How did you split the work?" | "What did your teammates do that you could not?" | Vai trò cụ thể, câu "I", ghi công người khác |
| 2. Thất bại | "Tell me about an experiment or project that failed." / "What is a result you expected that did not hold?" | "What would you do differently?" | Phân tích nguyên nhân, không đổ lỗi, đổi cách làm |
| 3. Bất đồng | "Tell me about a disagreement with a teammate or advisor." | "What did the other person think, in their words?" | Lắng nghe, dùng bằng chứng, kết cục rõ |
| 4. Học nhanh | "Tell me about something hard you had to learn quickly." | "How did you know you had learned it?" | Cách học có kiểm tra (lab, gradient check), thời gian cụ thể |
| 5. Tự sửa sai | "Tell me about a mistake you found in your own work." | "How did you make sure it would not happen again?" | Trung thực, quy trình phòng lỗi (chương 11) |
| 6. Ưu tiên khi mơ hồ | "Tell me about a time the goal was unclear or the deadline was too short." | "What did you drop, and why?" | Đánh đổi có lý do, báo sớm |

Chương 13 (chuỗi 4 "Con người") đã hỏi một câu cho các loại 2, 3, 5 và có câu trả lời mẫu về một lỗi tokenizer; chương này không dùng lại câu chuyện đó. Ở đây bạn học câu chuyện đủ 4 phần, có ngân sách giây và có câu nối tiếp.

**Câu chuyện mẫu (loại 2 và 3 gộp).** Văn bản nằm trong `work/ch14/scripts/star_story.py`; đếm bằng script đó.

> *For illustration only.* **(S)** In 2026, a teammate and I built a question answering system over arXiv abstracts. He built the data pipeline; I wrote the retrieval, including the citation-graph step, and the evaluation. **(T)** Five weeks before the final presentation, we still had no test set. My job was to test whether my graph step really helped. **(A)** I wrote 200 questions from papers neither of us had used during development. I asked a classmate to label 50 of them again, and the agreement was a kappa of 0.78, so I kept the labels but noted the limit. I ran every system with three seeds. Then I found the problem: with the reranker removed from both systems, my graph step added only 0.05 instead of 0.11. I told my teammate two weeks before the deadline, and I proposed that we report both numbers rather than only the better one. He disagreed, because the smaller number looked weak. I showed him the per-question results, and we agreed to present the gain as conditional on reranking. **(R)** The final report kept both numbers, and the claim we made was one we could defend. I learned to build the test set before the model, and I now write the analysis plan before the first run.

**Ví dụ giải mẫu 14.11 — đếm một câu chuyện STAR.** Nhãn "For illustration only." là của sách, người kể không nói, nên không đếm. 206 từ, 88 giây ở 140 từ/phút. S 30 từ (15%), T 23 (11%), A 116 (56%), R 37 (18%). "I" 12 lần, "we" 5, "my" 4, "our" 0. Chữ số: 2026, 200, 50, 0.78, 0.05, 0.11, khớp hồ sơ giả của chương 13: hai người, teammate làm data pipeline, người kể làm phần truy hồi (gồm bước đồ thị trích dẫn) và phần đánh giá, nên "my graph step" khớp với CV "Built hybrid dense + citation-graph retrieval". So với gợi ý MIT (20/10/60/10): Result dài hơn gợi ý, chấp nhận được vì nó chứa bài học; cả câu chuyện ngắn hơn ngân sách 2 phút khoảng 32 giây, đủ chỗ cho một câu hỏi nối tiếp. Câu nối tiếp nên chuẩn bị: "What did your teammate think, in his words?" ("He thought the smaller number would make the project look weak, and he was right that it looked weaker.") và "What would you do differently?" ("Build the test set in the first week, before any model.").

% manim: NganSachSTAR

**Phỏng vấn thử 60 phút (việc của tuần 38).** Một người khác đóng vai người phỏng vấn (bạn học hoặc giảng viên). Trước buổi, đưa họ: 2 chuỗi từ ngân hàng mà bạn **không** luyện trong 7 ngày qua, chuỗi C15 hoặc capstone P4 cho phần dự án, và bảng 6 loại câu hỏi hành vi để họ chọn 2. Theo khung 60 phút ở mục 14.1. Ghi hình cả buổi; sau đó chép lời (Whisper hoặc tự gõ) và đếm bằng Lab 3.

**Thang 30 điểm cho phỏng vấn thử** (**thang của chương 14**, không phải tiêu chí chính thức của Qualcomm, VinAI hay Google):

| Phần | Tiêu chí đếm được | Điểm |
| --- | --- | --- |
| Kỹ thuật (2 chuỗi) | Tổng điểm 8 tầng theo thang 0–3 (tối đa 24), chia 2 | 12 |
| Con số | Số con số có đơn vị nói đúng trong phần kỹ thuật: ≥ 6 → 3; 4–5 → 2; 2–3 → 1 | 3 |
| Làm rõ và "không biết" | ≥ 1 câu hỏi làm rõ (1); khi bí, nói phần đã biết và cách kiểm, không bịa (2) | 3 |
| Đào sâu dự án | Khẳng định trong ≤ 2 câu (1); baseline, số seed, độ dao động (1); vai trò của mình (1); tự nêu một giới hạn (1) | 4 |
| Hành vi (2 câu chuyện) | Mỗi câu: có số (0.5), "I" ≥ "we" (0.5), có bài học (0.5), ≤ 2.5 phút (0.5) | 4 |
| Cách nói | Từ/phút 120–160 (1); từ đệm ≤ 3 trên 100 từ (1); không câu trả lời nào quá 3 phút mà không dừng kiểm tra người nghe (1); hỏi ≥ 2 câu thực chất ở cuối (1) | 4 |

Ngưỡng như mini-test: **≥ 24** sẵn sàng; **18 đến dưới 24** luyện lại phần yếu nhất rồi thử lại sau 1 tuần; **dưới 18** quay về ngân hàng chuỗi. Dòng Kỹ thuật chia 2 nên tổng có thể lẻ 0.5 (ví dụ 23.5); ba vùng phủ liên tục, nên 23.5 thuộc vùng giữa. Các ngưỡng từ/phút và từ đệm là quy tắc của sách (chương 13 dùng 140 từ/phút làm chuẩn tính; bản ghi giả lập của chương 13 có 4.5 từ đệm trên 100 từ).

**Ví dụ giải mẫu 14.12 — chấm một buổi thử** (số giả). Hai chuỗi được 9/12 và 6/12: (9 + 6)/2 = 7.5. Nói đúng 5 con số có đơn vị: 2. Có hỏi lại một câu và xử lý tốt một câu không biết: 3. Dự án: khẳng định gọn, có baseline và seed, nói vai trò, nhưng không tự nêu giới hạn: 3. Câu chuyện 1 đủ bốn tiêu chí (2.0); câu chuyện 2 không có số và dài 3 phút (1.0): 3. Cách nói: từ/phút và độ dài ổn, từ đệm 4.1 trên 100 từ: 3. Tổng 7.5 + 2 + 3 + 3 + 3 + 3 = 21.5, nằm trong vùng 18 đến dưới 24. Việc tuần sau: chuỗi được 6/12 vào sổ lỗi với trọng số cao, thêm một con số vào câu chuyện 2 và cắt nó xuống 2 phút, luyện câu "One limitation is…" cho phần dự án.

**Người phỏng vấn nghe gì ở đây** (mô hình của sách).

- Việc **bạn** làm, bằng "I", có số, và việc người khác làm được ghi công.
- Bạn thay đổi cách làm sau một thất bại, chứ không chỉ kể thất bại.
- Câu chuyện khớp với hồ sơ: con số, vai trò và ngày tháng giống CV, statement và video (chương 13).

## Bài tập giấy bút

14 bài, xếp theo bảng lịch đầu chương: bài 1–3 và 5 ở tuần 35, 6–9 ở tuần 36, 10–13 ở tuần 37, bài 4 ở tuần 38; bài 14 tùy chọn. Bài nào sai ghi vào sổ lỗi kèm lý do sai. Mọi mẫu dùng số giả.

**★ Hiểu định nghĩa**

1. Xếp 8 câu hỏi sau vào tầng 1–4 theo định nghĩa ở mục 14.1: (a) "What is a KV cache?" (b) "Derive the gradient of softmax cross-entropy with respect to the logits." (c) "How many bytes per token does the KV cache of a Llama-2-7B-shaped model take in fp16?" (d) "When would speculative decoding stop giving a speedup?" (e) "Why does dividing by √dₖ keep the softmax away from saturation?" (f) "What is McNemar's test?" (g) "With b = 30 and c = 20, is the difference significant at the 0.05 level?" (h) "A colleague says post-training hurt calibration. How would you test that claim?"
2. Logit (1, 2, 0), nhãn là lớp thứ hai: tính p, p − y và cross-entropy.
3. Đổi đơn vị: 6.74 · 10⁹ tham số fp16 là bao nhiêu GB và bao nhiêu GiB? 2 GiB là bao nhiêu byte?
4. Câu chuyện 81 từ dưới đây là bản nháp của **một bạn khác, về một dự án khác** (nhóm bốn người; số giả, không phải hồ sơ mẫu của sách): gán S, T, A hoặc R cho từng câu, đếm số lần "I" và "we", và nêu một chỗ yếu theo tiêu chí của 14.5. *In my third year our team of four had two weeks to submit a retrieval project. My task was the evaluation. We had no labeled test set. I wrote 150 questions, I asked a classmate to label 30 of them again, and I computed kappa. I also ran three seeds for every system. We found that the gain dropped from 0.08 to 0.03 without the reranker, and I reported both numbers. I learned to build the test set before the model.*

**★★ Mức đề thi đầu vào**

5. Label smoothing K = 5, ε = 0.1, quy ước PyTorch: phân phối mục tiêu q, khe logit tối ưu giữa lớp đúng và lớp khác, và loss nhỏ nhất.
6. Một mô hình có 28 lớp, 4 đầu K/V, dₕ = 128, fp16. Tính KV cache mỗi token và ở T = 32 768 token. Nếu mô hình dùng 28 đầu K/V thì bao nhiêu?
7. Hai hệ thống trên cùng 1000 câu: b = 45 câu A đúng B sai, c = 25 câu ngược lại. Tính hiệu accuracy, χ² McNemar và p, có và không hiệu chỉnh liên tục.
8. Bài toán KL-regularized với πref = (0.6, 0.4), r = (0, 1), β = 1: tìm π* và KL(π* ‖ πref).
9. Logit (3, 1, 0) ở T = 1 và T = 3: tính phân phối và entropy (nats).
10. Speculative decoding với α = 0.7: số token kỳ vọng mỗi lượt khi k = 3 và k = 6; tăng tốc khi c = 0.1; k nào tốt hơn, và k tối ưu là bao nhiêu?

**★★★ Mức phỏng vấn nghiên cứu**

11. Suy dẫn Jacobian của LayerNorm không affine, x̂ = (x − μ)/σ với σ tính bằng cách chia n: J = (I − 11ᵀ/n − x̂x̂ᵀ/n)/σ. Chứng minh J·1 = 0 và J·x̂ = 0 và giải thích bằng tính bất biến. Hạng của J bằng bao nhiêu?
12. GPU giả định 24 GiB, mô hình hình dạng Llama-2-7B, chuỗi 4096 token. Bao nhiêu chuỗi với MHA và với 8 đầu K/V, ở fp16 và ở 4-bit cộng scale? Nêu ba thứ ước lượng của bạn bỏ qua.
13. Thiết kế một đánh giá đủ để phát hiện mức tăng 2 điểm quanh accuracy 0.70 (α = 0.05 hai phía, power 0.8): cỡ mẫu ghép cặp với 10% câu bất đồng, và không ghép cặp. Giải thích vì sao tỉ lệ bất đồng quyết định.
14. Với α = 0.8, c = 0.05, tìm k tối ưu của tăng tốc (1 − αᵏ⁺¹)/((1 − α)(kc + 1)) và giải thích vì sao tăng k quá mức làm chậm lại. Giả định nào của mô hình này dễ sai nhất trong thực tế?

## Bài tính nhanh

15 bài, mỗi bài tối đa 2 phút, làm bằng giấy bút và bấm giờ cả bộ. Mục tiêu: xong 15 bài trong 25 phút, sai không quá 1. Đây là loại tính nhẩm người phỏng vấn hay chen vào giữa chuỗi. Làm hai lần, tuần 35 và tuần 38 (lần 3–4 tùy chọn). Mọi băng thông là giả định của đề.

1. FLOPs của phép nhân (512 × 1024)(1024 × 256).
2. Số tham số một khối GPT có bias với d = 1024, theo công thức 12d² + 13d.
3. 13 · 10⁹ tham số ở bf16 chiếm bao nhiêu GiB?
4. KV cache mỗi token với 40 lớp, 8 đầu K/V, dₕ = 128, fp16; ở 8192 token thì bao nhiêu GiB?
5. 6ND với N = 3 · 10⁹ và D = 6 · 10¹⁰.
6. Mô hình 6.74 · 10⁹ tham số fp16, băng thông giả định 1 TB/s: cận dưới ms/token ở batch 1 (chỉ tính trọng số).
7. Sai số chuẩn của accuracy 0.9 trên 400 câu.
8. softmax(ln 3, 0); cross-entropy và p − y khi nhãn là lớp thứ hai.
9. Var(q·k) với dₖ = 128, mỗi phần tử độc lập, trung bình 0, phương sai 1; độ lệch chuẩn?
10. Độ lệch chuẩn khởi tạo He với n = 512 đầu vào.
11. Cross-entropy 2.0 nats/token: perplexity và bits/token.
12. MRR khi tài liệu liên quan đầu tiên ở hạng 2, 1, 4 và không có.
13. LoRA hạng r = 16 trên ma trận 4096 × 4096: số tham số và phần trăm so với ma trận gốc.
14. Speculative decoding α = 0.5, k = 3: số token kỳ vọng mỗi lượt.
15. Logit lớn nhất mà exp() không tràn ở fp16.

## Lab code và bài săn bug

Năm lab là **công cụ luyện tập** cho bốn tuần: chọn chuỗi, lên lịch ôn, đếm bản chép lời, kiểm suy dẫn bảng trắng, tính chi phí. Lab 1, 4, 5 và bài săn bug nằm trong giờ của lộ trình; Lab 3 nằm trong giờ phỏng vấn thử tuần 38; Lab 2 tùy chọn. Notebook lời giải đã chạy với Python 3.13.17, NumPy 2.5.4 và pandas 3.1.0, hết dưới 30 giây trên CPU (kể cả khởi động kernel); chỉ dùng numpy và pandas. Lab có ngẫu nhiên thì tạo Generator riêng với seed cố định (Lab 1 seed 0 và seed 0–3999 cho phép kiểm tần suất, Lab 2 seed 2, Lab 4 seed 1). Sổ lỗi và bản chép lời là mẫu tự soạn. Tự viết từng lab trước, chỉ mở lời giải khi kết quả lệch cột "kết quả mong đợi". Bước cuối (không bắt buộc): thay dữ liệu mẫu bằng dữ liệu thật của bạn.

| Lab | Việc phải làm | Kết quả mong đợi |
| --- | --- | --- |
| 1. Chọn chuỗi theo sổ lỗi | `read_log` đọc CSV `chuoi,diem,ngay`; `weights` = (12 − điểm + 1)·(1 + ngày/7); `pick` rút 4 chuỗi không lặp theo trọng số | Trọng số lớn nhất C11 = 27.143 (điểm 3, 12 ngày), nhỏ nhất C06 = 1.000; xác suất rút C11 là 0.1737, gấp 27.1 lần C06; seed 0 chọn C11, C05, C02, C01; `pick(k=1)` qua 4000 seed rút C11 với tần suất 0.1742 |
| 2. Lịch ôn Leitner (tùy chọn) | `schedule`: hộp 1-3-7-14 ngày, mỗi ngày thêm một chuỗi mới, quên thì về hộp 1; quy tắc qua/trượt xác định (qua 80%) | 28 ngày, 18 chuỗi: tải mỗi ngày 1, 2, 1, 1, 3, 2, 3, 2, 2, 3, 2, 6, 3, 3, 3, 2, 6, 3, 3, 2, 1, 3, 1, 2, 0, 2, 4, 3; tổng 69 lượt; ngày 12 và 17 nặng nhất với 6 chuỗi = 72 phút nói. Biến thể qua ngẫu nhiên 80% (seed 2): tổng 78 lượt, nặng nhất 6 |
| 3. Phân tích bản chép lời (tuần 38) | `basic_stats` (từ, từ/phút, từ đệm kể cả cụm nhiều chữ) **đưa sẵn**, cùng việc với Lab 4 chương 13; bạn viết phần mới `voice_and_digits`: số "I", "we", cụm chữ số | Bản mẫu (C01 tầng 1, 50 giây): 116 từ, 139.2 từ/phút, 8 từ đệm = 6.9 trên 100 từ (vượt ngưỡng 3), I = 3, we = 1, chỉ 1 cụm chữ số ("2 000"); bỏ hết từ đệm còn 104 từ |
| 4. Kiểm suy dẫn bảng trắng | Bước 1: `num_grad` sai phân trung tâm float64, `rel_err`, `check` (ngưỡng 10⁻⁶) cho 4 khẳng định. Bước 2: tìm hai khẳng định sai, suy dẫn lại và điền gradient đúng vào `FIXES` | Hai khẳng định đúng cho sai số tương đối cỡ 10⁻¹⁰; hai khẳng định sai cho cỡ 0.25; hai bản sửa cho cỡ 10⁻¹⁰. Ô kiểm còn thử `num_grad` trên một hàm đã biết đạo hàm (sai số dưới 10⁻⁸) và kiểm `rel_err` không đổi khi nhân hai vế với 10⁻⁶ hoặc 10⁶. Khẳng định nào sai: xem Lời giải |
| 5. Máy tính chi phí | `weight_gib`, `kv_bytes` (có batch B), `train_flops` (quy ước chương 8: bỏ bảng embedding đầu vào), `decode_ms` cho cấu hình Llama-2-7B | 6 738 415 616 tham số; fp16 12.55 GiB; 4-bit cộng scale 3.33 GiB; KV 524 288 byte mỗi token, 4096 token 2.00 GiB (MHA), batch 8 là 16.00 GiB, 8 đầu K/V 0.50 GiB; 6·N·D với D = 2 · 10¹²: 7.929 × 10²² (không bỏ embedding: 8.086 × 10²²); 6.74 ms/token ở 2 TB/s giả định, 7.81 ms khi cộng KV 4096; GPU 80 GiB giả định: 33 và 134 chuỗi; laptop: 5.33 GiB |

Code nằm trong notebook `labs/chuong14/lab.ipynb` (bản có chỗ TODO để bạn tự viết, kèm ô kiểm tra tự động) và `labs/chuong14/loigiai.ipynb` (bản lời giải đã chạy kiểm, có phụ lục kiểm các số mới của chương). Mở từ trang **Lab** của app.

**Câu hỏi sau lab.**

- Lab 1: vì sao trọng số nhân với (1 + ngày/7) chứ không chỉ dựa vào điểm? Chuỗi C06 đạt 12/12 hôm nay có nên biến mất khỏi lịch không?
- Lab 2: vì sao ngày 12 và ngày 17 nặng nhất? 6 chuỗi × 12 phút = 72 phút có vừa một buổi tối không, và bạn xử lý thế nào?
- Lab 3: công cụ bỏ sót những gì? Nghĩ tới số đọc bằng chữ và từ đệm tiếng Việt.
- Lab 4: vì sao dùng float64 và sai số **tương đối** với ngưỡng 10⁻⁶?
- Lab 5: chênh 7.4% giữa GB và GiB có đổi kết luận "vừa / không vừa" ở bài laptop 6 GiB không?

**Bài săn bug.** Trong notebook có hàm `cost_report` do "một bạn cùng nhóm" viết: nhận cấu hình mô hình, trả về trọng số (GiB), KV cache mỗi chuỗi (GiB), số chuỗi 4096 token vừa GPU giả định 80 GiB, FLOPs huấn luyện và ms/token. Hàm chạy không báo lỗi và mọi số trông "hợp lý". Triệu chứng (số thật của bản lỗi): trọng số báo 13.48 "GiB", lớn hơn số chương 8 khoảng 7%; KV cache 4096 token là 1.00 GiB cho **cả** cấu hình MHA lẫn cấu hình 8 đầu K/V; GPU 80 GiB báo chứa được 66 chuỗi MHA; FLOPs huấn luyện 2.643 × 10²², nhỏ hơn phép tính tay của bạn. Có **bốn** lỗi. Sửa tại chỗ rồi chạy ô kiểm tra; ô kiểm (7 phép kiểm, có một phép kiểm batch 8) chỉ in ĐẠT hoặc số phép kiểm chưa đạt, không nói lỗi nằm ở đâu.

## Bài dựng cảnh Manim

Mỗi chương có một bài tự dựng cảnh: muốn vẽ được một chuỗi, bạn phải biết từng tầng của nó. Giới hạn **1 giờ**, **tùy chọn** (tuần 37; lộ trình không có dòng giờ cho bài này), không chỉnh màu mè.

**Đề bài.** Dùng `manim/chuong14_scenes.py` (cảnh `CayChuoiNoiChuong`) làm mẫu, viết cảnh `CayChuoiCuaToi` vào thư mục `manim/cua-toi/`: một cây chuỗi 4 tầng cho **dự án của chính bạn**.

1. Bốn nút xếp dọc, nối bằng mũi tên, mỗi nút là một tầng (khẳng định, có thật không, ablation, mối đe dọa lớn nhất, hoặc 4 tầng bạn tự chọn).
2. Mỗi nút ghi chương sách mà tầng đó dùng; cả cây nối ít nhất 3 chương khác nhau.
3. Bên phải mỗi nút là một con số có đơn vị hoặc một phép kiểm cụ thể; chỉ dùng `Text`, không dùng LaTeX.

**Tiêu chí tự chấm.** (1) Đúng 4 tầng, mỗi tầng một nút có số chương. (2) Không chữ chồng lên nhau và không chữ tràn khung, kiểm bằng cách xem khung hình cuối. (3) Mỗi nút có một con số có đơn vị hoặc một phép kiểm cụ thể.

Render bằng nút **Render cảnh của tôi** ở trang Manim. Lời giải tham khảo (dự án mẫu, số giả của chương 13) nằm ở `manim/loigiai/chuong14_cay_chuoi.py`; chỉ mở sau khi đã tự làm.

## Phỏng vấn, bẫy, thực tế khác lý thuyết

**Cách luyện.** Mỗi chuỗi capstone dưới đây nối ít nhất hai nhóm chương (1–6, 7–10, 11–13): P1 và P3 nối nhóm 1 với nhóm 2, P2 đi qua cả ba nhóm, P4 nối nhóm 2 với nhóm 3. Chúng dùng lại các tầng của ngân hàng, nhưng giữa hai tầng liền nhau bạn phải chuyển nhóm kiến thức. Không bộ câu hỏi chính thức nào của Qualcomm hay VinAI được công khai (D): đây là chuỗi do sách soạn. Trả lời to bằng tiếng Việt trước rồi bằng tiếng Anh, 12 phút một chuỗi (3 phút mỗi tầng), ghi âm, tự chấm bằng thang 4 tầng ở mục 14.1 và ghi vào sổ lỗi. Câu trả lời mẫu tiếng Anh phủ tầng 1–2 trong 3–5 câu và chỉ dùng số đã kiểm; tầng 3–4 bạn tự soạn từ các ví dụ giải mẫu và chuỗi được trỏ tới. Làm P1 ở tuần 36, P2 và P4 ở tuần 37; P3 tùy chọn, dành cho tuần bạn còn yếu ở nhóm chương 1–6.

**Chuỗi P1 — From a logit to a served token** (chương 2, 8, 9)

1. During training, what is the gradient of the loss with respect to the next-token logits, and what does it tell you?
2. At inference, how does temperature change the next-token distribution? Give numbers for logits (2, 1, 0).
3. Each generated token adds to the KV cache. How big is it per token for a Llama-2-7B-shaped model in fp16, and how many 4096-token sequences fit on a hypothetical 80 GiB GPU? (C07)
4. Decoding is memory-bound. How could a draft model speed it up, and under what assumption is your estimate valid? (C18)

> *Model answer:* During training, the gradient of the cross-entropy with respect to the logits is p − y, so each logit is pushed by the gap between the predicted and the target probability, and the push vanishes only when the prediction matches the label. At inference I divide the same logits by a temperature before the softmax: for logits (2, 1, 0) the top probability is 0.87 at T = 0.5, 0.67 at T = 1 and 0.51 at T = 2, so a higher temperature flattens the distribution without changing the ranking. Every generated token then stores a key and a value per layer and per KV head, which for a Llama-2-7B-shaped model in fp16 is half a mebibyte per token, or 2 GiB for a 4096-token sequence.

**Chuỗi P2 — From a variance lemma to a decision** (chương 3, 8, 11)

1. Accuracy is a mean of per-question outcomes. Derive its standard error from the variance of a sum of independent terms. (C03)
2. Two models answer the same 1000 questions; A wins 30 and B wins 20 of the disagreements. Derive the standard error of the paired difference and compare it with the unpaired one. How does it relate to McNemar's test? (C13)
3. How many questions do you need to detect a one-point gain with 5% disagreement?
4. Each answer is 100 generated tokens from a 7B-shaped model in fp16 at batch 1, on a hypothetical 2 TB/s GPU. What is the lower bound on the time to evaluate both systems, and what would batching change? (C08)

> *Model answer:* Each question contributes an outcome that is 1 with probability p and 0 otherwise, with variance p(1 − p); the questions are independent, so the cross terms vanish and the variance of the sum is n·p(1 − p), which makes the standard error of the mean √(p(1 − p)/n), about 0.011 for 0.85 on 1000 questions. For two models on the same questions I work with the per-question difference, which is +1, −1 or 0: its mean is δ = (30 − 20)/1000 = 0.01 and its variance is ψ − δ², where ψ = 0.05 is the disagreement rate, so the paired standard error is √(0.0499/1000) ≈ 0.0071. The unpaired formula gives 0.016, more than twice as large, because it ignores that most questions are answered the same way by both models. The paired z is about 1.42 and z² ≈ 2.0, which is McNemar's statistic, so a one-point gain on 1000 questions is still not convincing.

**Chuỗi P3 — Why training works at all** (chương 1, 4, 6, 8)

1. How does the condition number of the Hessian control gradient descent on a quadratic, and what does ridge do to it? (C02)
2. Why do we initialize with variance 2/n in ReLU networks? (C03)
3. In a 12-layer pre-LN Transformer, what happens to the variance of the residual stream, and how is it controlled?
4. The loss is non-convex. Why does gradient descent still find good solutions, and what does a permutation argument tell you? (C04)

> *Model answer:* On a quadratic, gradient descent with the best fixed step contracts the error by (κ − 1)/(κ + 1) per step, so with a condition number of 10 000 it needs about 34 500 steps to reduce the error a thousandfold. Ridge adds λ to every eigenvalue: with eigenvalues 100 and 0.01 and λ = 1 the condition number drops to 100 and about 345 steps suffice, at the price of a biased solution. The same reasoning about scale explains initialization: the variance of a pre-activation is n·Var(w)·𝔼[x²], and since a ReLU keeps half of the second moment, Var(w) = 2/n keeps the signal at a constant scale through depth.

**Chuỗi P4 — Your project under a budget** (chương 8, 11, 13)

1. In two sentences: what did you build yourself, and which single number would you defend?
2. Could you rerun your main comparison on your 6 GiB laptop GPU, with a 7B-shaped generator reading five retrieved passages? Estimate the memory. (C07)
3. What is the standard error of your recall@5, and is the 0.05 gain without the reranker distinguishable from noise? (C14)
4. You have 20 GPU-hours. Which runs close the biggest threat to validity, and what result would make you drop the claim? (C15)

> *Model answer:* For illustration only: a teammate built the data pipeline, and I wrote the retrieval, including the citation-graph step, and the evaluation. The number I defend is recall at five rising from 0.61 to 0.72 on 200 hand-labeled questions over three seeds, against the same pipeline without the graph step. To rerun it on a 6 GiB laptop GPU, a 7B-shaped generator in 4-bit with one fp16 scale per 64 weights takes about 3.33 GiB, and the fp16 KV cache for a 2048-token prompt with five passages adds 1 GiB, so about 4.33 GiB on paper. That leaves about 1.7 GiB for activations and framework overhead, which I would measure before trusting the plan.

**Từ vựng nên nói trôi chảy:** logits, one-hot, positive semi-definite, label smoothing, fan-in, residual stream, log-sum-exp, underflow, mixed precision, KV cache, grouped-query attention, memory-bound, model FLOPs utilization, KL-regularized objective, partition function, temperature scaling, expected calibration error, in-batch negatives, bits per dimension, standard error, McNemar's test, paired design, statistical power, nDCG, ablation budget, threat to validity, acceptance rate, draft model.

**Bẫy thường gặp.**

- Trả lời tầng 3 bằng một con số không đơn vị hoặc không nói quy ước: GB hay GiB, chia ε cho K hay K − 1, FLOPs có tính embedding không.
- Học thuộc công thức mà không suy dẫn lại được khi đổi giả định, ví dụ KV cache khi chuyển từ MHA sang 8 đầu K/V.
- Bịa tên paper hay con số khi không biết, thay vì dùng bốn bước "I don't know, but…".
- Nói liền 4 phút mà không dừng kiểm tra người phỏng vấn có theo kịp và có muốn đi sâu hướng đó không.
- Câu chuyện hành vi dùng "we" suốt, không có số, không có bài học (Amazon và MIT CAPD đều yêu cầu ngược lại).
- Gán định dạng hay câu hỏi cho Qualcomm hoặc VinAI: định dạng không công khai (D).

**Thực tế khác lý thuyết.**

- Định dạng thật không công khai. Khung 60 phút là của sách, và chính bài của DeepMind nói quy trình khác nhau theo vị trí. Chuẩn bị rộng để không phụ thuộc vào một khung.
- Công thức số token kỳ vọng của speculative decoding dựa trên giả định chấp nhận độc lập với tỉ lệ cố định, một giả định mà paper nói rõ; tăng tốc thật phụ thuộc phần cứng, batch và độ khớp giữa mô hình nháp với mô hình đích. Mọi công thức "trên bảng" trong chương này có giả định tương tự, và tầng 4 thường hỏi đúng giả định đó.
- Kết quả một buổi phỏng vấn có nhiễu lớn: luyện có chủ đích chỉ giải thích phần nhỏ phương sai thành tích (mục 14.1), và Chip Huyen nói kết quả phụ thuộc cả người phỏng vấn lẫn các ứng viên khác. Một buổi thử dưới 24/30 là tín hiệu để luyện tiếp, không phải dự đoán kết quả thật.

## Mini-test 30 phút

Làm cuối tuần 38, sau buổi phỏng vấn thử, bấm giờ đúng 30 phút, không xem sách, không dùng máy tính. Tổng 30 điểm. **Từ 24 điểm:** chương 14 xong; sang tuần 39 (chẩn đoán điểm yếu). **18 đến dưới 24:** ôn lại mục ứng với câu sai rồi làm lại sau 3 ngày. **Dưới 18:** quay về ngân hàng chuỗi của nhóm yếu nhất trước khi sang tuần 39.

1. **(4 điểm)** Một mô hình có 32 lớp, 8 đầu K/V, dₕ = 128, fp16. Tính KV cache mỗi token, mỗi chuỗi 8192 token, và cho batch 4 chuỗi như vậy. Nếu mô hình dùng 32 đầu K/V thì batch đó cần bao nhiêu?
2. **(4 điểm)** Suy dẫn gradient của softmax cross-entropy theo logit và Hessian theo logit. Với hai lớp và p = (0.5, 0.5), viết Hessian, tìm trị riêng, và nói hướng ứng với trị riêng 0 có nghĩa gì.
3. **(5 điểm)** Hai hệ thống trên cùng 2000 câu: b = 40 câu A đúng B sai, c = 22 câu ngược lại. Tính hiệu accuracy, χ² McNemar (không hiệu chỉnh), kết luận ở mức 0.05, và giải thích vì sao không dùng kiểm định hai mẫu độc lập. (Ngưỡng χ² một bậc tự do ở 0.05 là 3.84.)
4. **(5 điểm)** Speculative decoding với α = 0.75, k = 4, c = 0.1: số token kỳ vọng mỗi lượt và tăng tốc. Tăng k lên 5 có lợi không?
5. **(6 điểm)** Viết câu trả lời tầng 3 theo khung 4 phần của mục 14.1 cho câu: "On a hypothetical 40 GiB GPU, how many 4096-token sequences can a Llama-2-7B-shaped model in fp16 serve, with full multi-head attention and with 8 KV heads?" Chấm bằng thang 0–3 nhân 2.
6. **(6 điểm)** Viết dàn ý STAR tiếng Anh cho câu "Tell me about an experiment that failed", tối đa 200 từ. Chấm theo bốn tiêu chí đếm được, mỗi tiêu chí 1.5 điểm: ≤ 200 từ; có ít nhất một con số; số lần "I" nhiều hơn "we"; có bài học hoặc điều sẽ làm khác.

## Flashcard

18 thẻ của chương nằm trong `flashcards/chuong14.csv`. Ôn ở trang **Ôn tập** của app: app tự lên lịch theo hộp Leitner (thẻ nhớ được sẽ giãn lịch ôn 1 → 3 → 7 → 14 ngày, thẻ quên quay lại hộp 1). File CSV này cũng nhập thẳng được vào Anki nếu bạn thích dùng Anki.

## Lời giải

Chỉ đọc sau khi đã tự thử ít nhất 20 phút. Mọi con số tính toán trong chương đã được kiểm bằng script trong `work/ch14/scripts/` và bằng phụ lục của `loigiai.ipynb`. Với 18 chuỗi trong ngân hàng, ý chính và lỗi hay gặp nằm ngay dưới từng chuỗi; tầng 3–4 bạn tự soạn và tự chấm. Các bài mở (mini-test câu 5 và 6) có nhiều đáp án đúng: so với ý chính và thang điểm, không so từng chữ.

**Kiểm tra nhanh**

1. 3 phút mỗi tầng (2 phút trả lời, 1 phút hỏi lại); 2 × 140 = 280 từ.
2. p = (4, 1)/5 = (0.8, 0.2), nên p − y = (0.8, −0.8). Cross-entropy là ln 5 ≈ 1.609.
3. 2 · 32 · 32 · 128 · 2 = 524 288 byte = 0.5 MiB.
4. √(0.8 · 0.2/1600) = √(10⁻⁴) = 0.01.
5. Action, khoảng 60%.

**Bài tập giấy bút**

1. (a) 1; (b) 2; (c) 3; (d) 4; (e) 2; (f) 1; (g) 3; (h) 4. Câu (b) là suy dẫn nên thuộc tầng 2 theo định nghĩa, dù chuỗi C01 đặt nó ở vị trí đầu.
2. e¹ = 2.718, e² = 7.389, e⁰ = 1, tổng 11.107. p = (0.2447, 0.6652, 0.0900); p − y = (0.2447, −0.3348, 0.0900); CE = −ln 0.6652 = 0.4076.
3. 6.74 · 10⁹ × 2 byte = 13.48 · 10⁹ byte = 13.48 GB; chia 2³⁰ = 1 073 741 824 được 12.55 GiB. 2 GiB = 2 147 483 648 byte.
4. S: câu 1. T: câu 2 và 3. A: câu 4, 5 và 6. R: câu 7. "I" 6 lần, "we" 2 lần. Chỗ yếu: (1) R chỉ có bài học, không có kết cục của dự án; (2) "We found" che mất ai phát hiện, trong khi đó là việc của người kể; (3) "without the reranker" không nói bỏ reranker ở hệ thống nào, đúng lỗi chương 13 đã sửa; (4) không có câu nối tiếp nào được chuẩn bị (ví dụ "what did the team think?").
5. q = 0.92 cho lớp đúng, 0.02 cho mỗi lớp trong 4 lớp còn lại. Cross-entropy H(q, p) nhỏ nhất khi p = q (bất đẳng thức Gibbs), nên khe logit tối ưu là ln(0.92/0.02) = ln 46 = 3.8286 và loss nhỏ nhất là H(q) = −0.92 ln 0.92 − 4 · 0.02 ln 0.02 = 0.0767 + 0.3130 = 0.3897.
6. 2 · 28 · 4 · 128 · 2 = 57 344 byte (56 KiB) mỗi token. Ở 32 768 token: 57 344 × 32 768 = 1 879 048 192 byte = 1.750 GiB. Với 28 đầu K/V: gấp 7 lần, 12.25 GiB.
7. Hiệu accuracy (45 − 25)/1000 = 0.02. χ² = 20²/70 = 5.714, p = 0.0168. Có hiệu chỉnh liên tục: 19²/70 = 5.157, p = 0.0232; kiểm định chính xác cho p = 0.0225. Cả ba đều dưới 0.05.
8. π* ∝ πref · eʳ = (0.6, 0.4e) = (0.6, 1.0873), Z = 1.6873, π* = (0.3556, 0.6444). KL = 0.3556 ln(0.3556/0.6) + 0.6444 ln(0.6444/0.4) = −0.1860 + 0.3073 = 0.1213.
9. T = 1: e³ = 20.09, e¹ = 2.718, e⁰ = 1, tổng 23.80, p = (0.8438, 0.1142, 0.0420), entropy 0.5243 nats. T = 3: logit (1, 1/3, 0), p = (0.5315, 0.2729, 0.1955), entropy 1.0094 nats (trần ln 3 = 1.0986).
10. k = 3: (1 − 0.7⁴)/0.3 = 0.7599/0.3 = 2.5330. k = 6: (1 − 0.7⁷)/0.3 = 3.0588. Tăng tốc với c = 0.1: 2.5330/1.3 = 1.9485 và 3.0588/1.6 = 1.9118, nên k = 3 tốt hơn k = 6. Quét k cho tối ưu k = 4 với 1.9808.
11. Đặt d = x − μ1, σ² = ‖d‖²/n. Từ ∂μ/∂x = 1ᵀ/n và ∂σ/∂x = dᵀ/(nσ) = x̂ᵀ/n, quy tắc thương cho J = (I − 11ᵀ/n)/σ − d x̂ᵀ/(nσ²) = (I − 11ᵀ/n − x̂x̂ᵀ/n)/σ. Vì 1ᵀx̂ = 0: J·1 = (1 − 1 − 0)/σ = 0. Vì ‖x̂‖² = n: J·x̂ = (x̂ − 0 − x̂)/σ = 0. Ý nghĩa: LN(x + c1) = LN(x) (bất biến với dịch) và LN(x + t·d) = LN(x) với 1 + t > 0, vì đó chỉ là co giãn quanh trung bình. J đối xứng, không gian null gồm 1 và x̂ (trực giao nhau), trên phần bù trực giao J tác động như I/σ, nên hạng J = n − 2. Kiểm số với n = 6: ‖J·1‖ = 1.7 × 10⁻¹⁶, ‖J·x̂‖ = 1.3 × 10⁻¹⁶, hạng 4. Bỏ số hạng x̂x̂ᵀ/n (Jacobian sai J′ = (I − 11ᵀ/n)/σ): theo chuẩn Frobenius, phần bị bỏ x̂x̂ᵀ/(nσ) có chuẩn 1/σ (vì x̂x̂ᵀ/n là phép chiếu hạng 1), ‖J‖ = √(n − 2)/σ, ‖J′‖ = √(n − 1)/σ. Vậy sai số tương đối theo nghĩa thông thường ‖J′ − J‖/‖J‖ = 1/√(n − 2) = **0.5** với n = 6; theo định nghĩa đối xứng của `rel_err` ở Lab 4, ‖J′ − J‖/(‖J′‖ + ‖J‖) = 1/(√4 + √5) = **0.236**. Cả hai **không phụ thuộc x**: σ triệt tiêu. Khi nêu một sai số tương đối, nói rõ định nghĩa.
12. fp16: 24 − 12.55 = 11.45 GiB còn lại, chia 2 GiB được 5 chuỗi (MHA), chia 0.5 GiB được 22 chuỗi (8 đầu K/V). 4-bit cộng scale: 24 − 3.33 = 20.67 GiB, được 10 và 41. Ba thứ bỏ qua: activation và bộ đệm tạm lúc prefill; overhead của CUDA context, framework và phân mảnh bộ nhớ; mô hình 8 đầu K/V thật nhẹ hơn 1.50 GiB trọng số (fp16 chứa được 25 chuỗi chứ không phải 22). Thêm: KV cache vẫn giả định fp16 dù trọng số 4-bit.
13. Không ghép cặp: (1.96 + 0.8416)² (0.70 · 0.30 + 0.72 · 0.28)/0.02² = 7.849 × 0.4116/0.0004 = 8076.5, làm tròn lên 8077 câu mỗi hệ thống. Ghép cặp với ψ = 0.10: (1.96√0.1 + 0.8416√0.0996)²/0.0004 = (0.6198 + 0.2656)²/0.0004 = 1959.9, làm tròn lên 1960. Ít hơn khoảng 4 lần. Tỉ lệ bất đồng quyết định vì chỉ câu bất đồng mang thông tin về hiệu: phương sai của hiệu ghép cặp xấp xỉ (ψ − δ²)/n. Hai hệ thống càng giống nhau (ψ nhỏ) thì thiết kế ghép cặp càng lợi; nếu chúng bất đồng trên 40% câu, lợi thế gần như mất.
14. Quét k = 1…: 1.7143, 2.2182, 2.5670, 2.8013, 2.9514, 3.0396, 3.0823, **3.0921** (k = 8), rồi 3.0780 (k = 9). Tăng k quá mức làm chậm vì tử số bão hòa ở 1/(1 − α) = 5 (token thứ i chỉ được nhận với xác suất αⁱ, giảm dần), còn mẫu số kc + 1 tăng tuyến tính. Giả định dễ sai nhất: các lần chấp nhận độc lập với cùng một α; thực tế α phụ thuộc ngữ cảnh và vị trí, và c phụ thuộc phần cứng và batch.

**Bài tính nhanh**

1. 2 · 512 · 1024 · 256 = 268 435 456 ≈ 2.68 × 10⁸.
2. 12 · 1 048 576 + 13 · 1024 = 12 582 912 + 13 312 = 12 596 224.
3. 26 · 10⁹ byte / 2³⁰ = 24.2 GiB.
4. 2 · 40 · 8 · 128 · 2 = 163 840 byte = 160 KiB; × 8192 = 1.25 GiB.
5. 6 · 3 · 10⁹ · 6 · 10¹⁰ = 1.08 × 10²¹.
6. 13.48 · 10⁹ byte / 10¹² byte/s = 13.48 ms.
7. √(0.9 · 0.1/400) = 0.3/20 = 0.015.
8. p = (0.75, 0.25); CE = −ln 0.25 = ln 4 = 1.3863; p − y = (0.75, −0.75).
9. Var = 128; độ lệch chuẩn √128 = 11.31.
10. √(2/512) = √(1/256) = 0.0625.
11. Perplexity e² = 7.389; 2/ln 2 = 2.885 bits/token.
12. (1/2 + 1 + 1/4 + 0)/4 = 0.4375.
13. 16 · 4096 · 2 = 131 072; 131 072 / 16 777 216 = 0.78%.
14. 1 + 0.5 + 0.25 + 0.125 = 1.875.
15. ln 65 504 ≈ 11.09.

**Câu hỏi sau lab**

- Lab 1: đây là spacing (mục 14.1): một chuỗi từng được điểm cao vẫn bị quên nếu lâu không luyện. Hệ số (1 + ngày/7) làm trọng số gấp đôi sau 7 ngày, nên C06 (12/12, 0 ngày, trọng số 1.000, xác suất 0.0064) không biến mất mà quay lại dần khi số ngày tăng.
- Lab 2: các chuỗi được thêm đều mỗi ngày và cùng qua hộp theo các khoảng 1, 3, 7, nên nhiều chuỗi đến hạn trùng ngày; chuỗi quên quay về hộp 1 và chồng thêm. 72 phút nói thường quá dài cho một buổi tối; quy tắc gợi ý của sách: tối đa 4 chuỗi một ngày, phần dư dời sang ngày sau. Lệch một ngày làm khoảng ôn dài hơn một chút, ít hại hơn bỏ buổi.
- Lab 3: công cụ không đếm số đọc bằng chữ ("ten to the minus ten" không được tính là con số, giống giới hạn của công cụ chương 13), không biết từ đệm tiếng Việt ("ờ", "kiểu là", "thì là"), không thấy quãng ngắt khi không có mốc thời gian, và không biết nội dung đúng hay sai. Dùng nó để soát nhịp, không để chấm nội dung.
- Lab 4, đáp án bước 2: (b) sai vì A không đối xứng; ∇(xᵀAx) = (A + Aᵀ)x, sai số 0.261 → 7.47 × 10⁻¹¹ sau khi sửa. (d) sai vì thiếu số hạng −x̂x̂ᵀ/n (bài 11): 0.251 → 8.82 × 10⁻¹¹; Jacobian đủ có hạng 4 = n − 2. (a) và (c) đúng: 6.74 × 10⁻¹¹ và 8.67 × 10⁻¹¹.
- Lab 4, câu hỏi: sai phân trung tâm có sai số cắt cụt cỡ h² và sai số làm tròn cỡ ε/h. Với float64 (ε ≈ 2.2 × 10⁻¹⁶) và h = 10⁻⁶, cả hai nhỏ hơn 10⁻⁹ nhiều, nên khẳng định đúng cho sai số tương đối cỡ 10⁻¹⁰ và khẳng định sai cho cỡ 0.25: hai nhóm cách nhau 9 bậc. Với float32 (ε ≈ 1.2 × 10⁻⁷), sai số làm tròn cỡ 0.1 và không phân biệt được. Dùng sai số tương đối vì độ lớn gradient khác nhau giữa các bài (nối chương 2, chuỗi 3).
- Lab 5: không đổi. 5.33 GiB vẫn nhỏ hơn 6 GiB; tính bằng GB thì 5.73 GB, cũng nhỏ hơn 6 GiB = 6.44 GB. Nhưng khoảng dư 0.67 GiB nhỏ hơn mức overhead thường gặp của framework và CUDA context [cần đo trên máy của bạn], nên câu trả lời trung thực là "vừa trên giấy, phải đo".

**Mini-test (thang 30 điểm)**

1. 2 · 32 · 8 · 128 · 2 = 131 072 byte = 128 KiB mỗi token (1 điểm). Mỗi chuỗi 8192 token: 1 073 741 824 byte = 1 GiB (1 điểm). Batch 4: 4 GiB (1 điểm). Với 32 đầu K/V: gấp 4, 16 GiB (1 điểm).
2. Viết đúng L = −(logit của lớp đúng) + log-sum-exp và đạo hàm ra p − y (1.5 điểm). Hessian diag(p) − ppᵀ (1 điểm). Với p = (0.5, 0.5): Hessian có hàng (0.25, −0.25) và (−0.25, 0.25), trị riêng 0.5 và 0 (1 điểm). Hướng (1, 1): cộng cùng hằng số vào hai logit không đổi p, nên loss phẳng theo hướng đó; loss lồi nhưng không lồi chặt (0.5 điểm).
3. Hiệu (40 − 22)/2000 = 0.009 (1 điểm). χ² = 18²/62 = 324/62 = 5.23 (1.5 điểm). 5.23 > 3.84 nên bác bỏ giả thuyết không có khác biệt ở mức 0.05; p ≈ 0.022 (có hiệu chỉnh liên tục 4.66, p ≈ 0.031; kiểm định chính xác p ≈ 0.030; vẫn dưới 0.05) (1 điểm). Hai hệ thống trả lời cùng 2000 câu nên hai accuracy tương quan; kiểm định hai mẫu độc lập dùng sai phương sai của hiệu, và chỉ các câu bất đồng mang thông tin về hiệu (1.5 điểm).
4. (1 − 0.75⁵)/0.25 = (1 − 0.2373)/0.25 = 3.0508 token mỗi lượt (2 điểm). Tăng tốc 3.0508/(4 · 0.1 + 1) = 2.179 (2 điểm). k = 5: (1 − 0.75⁶)/0.25 = 3.2881, chia 1.5 được 2.192, nhỉnh hơn; k = 5 là tối ưu ở cấu hình này (1 điểm).
5. Câu trả lời đủ điểm, theo khung 4 phần: (1) "About 13 sequences with full multi-head attention, and about 54 with 8 KV heads." (2) Trọng số fp16 là 12.55 GiB, nên còn 40 − 12.55 = 27.45 GiB; mỗi chuỗi 4096 token cần 2 GiB KV với MHA và 0.5 GiB với 8 đầu K/V; lấy phần nguyên 27.45/2 = 13 và 27.45/0.5 = 54. (3) Các con số có đơn vị như trên. (4) "This ignores activations, framework overhead and fragmentation, and a real 8-KV-head model has 1.5 GiB less weight, which would allow 57." Thang: 3 (6 điểm) đủ bốn phần; 2 (4 điểm) đúng cơ chế và số nhưng thiếu điều kiện, hoặc ngược lại; 1 (2 điểm) chỉ có công thức; 0 sai hoặc trống.
6. Mỗi tiêu chí 1.5 điểm: ≤ 200 từ (đếm bằng Lab 3); có ít nhất một con số; "I" nhiều hơn "we"; có bài học hoặc "what I would do differently". Lỗi hay gặp làm mất điểm: kể theo thứ tự thời gian và hết chỗ cho Action; Result chỉ là "we learned a lot".

**Capstone (số cho tầng 3–4)**

- P2 tầng 3: 3922.1, làm tròn lên 3923 câu (ví dụ 14.9). Tầng 4: 3923 × 100 token × 6.74 ms = 2643 giây ≈ 44 phút mỗi hệ thống, 88 phút cả hai; chỉ tính đọc trọng số, bỏ prefill và KV. Batching chia chi phí đọc trọng số cho nhiều câu nên rút thời gian mạnh, nhưng không chia được phần đọc KV cache riêng của từng câu (C08).
- P4 tầng 3: √(0.72 · 0.28/200) = 0.0317 (cận Bernoulli, C14); 0.05 chỉ khoảng 1.6 sai số chuẩn nếu coi độc lập, nên cần ghép cặp theo câu và theo seed (C13). Tầng 4: ablation một chiều 4 cấu hình × 3 seed × 0.5 giờ = 6 giờ (C15); phần còn lại cho cùng phép so sánh trên một lĩnh vực thứ hai, vì mối đe dọa lớn nhất là tập test một lĩnh vực, một người gán nhãn chính. Bỏ khẳng định nếu mức tăng ở lĩnh vực mới nằm trong độ dao động giữa các seed.

**Săn bug — bốn lỗi**

1. **Thiếu hệ số 2 trong KV cache.** Mỗi token lưu cả key lẫn value; bản lỗi chỉ tính một tensor nên KV và mọi thứ suy ra từ nó (số chuỗi) lệch gấp 2.
2. **Số đầu query thay số đầu K/V.** Với MHA hai số bằng nhau nên lỗi bị che; chỉ cấu hình 8 đầu K/V mới lộ, và triệu chứng là KV của hai cấu hình **bằng hệt** nhau. Lỗi 1 và 2 nằm cùng một dòng.
3. **GB thay GiB.** Chia 10⁹ thay vì 2³⁰ làm số "GiB" của trọng số lớn hơn 7.4% (13.48 thay vì 12.55).
4. **2ND thay 6ND.** 2N mỗi token chỉ là forward; huấn luyện cần thêm 4N cho backward, nên FLOPs bị thấp 3 lần (2.643 × 10²² thay vì 7.929 × 10²²).

```python
def cost_report(c, T=4096, B=1, nbytes=2, D=2e12, bw=BW, gpu_gib=80):
    n = count_params(c)
    dh = c["d"] // c["n_h"]
    weights = n * nbytes / GiB                          # sửa lỗi 3
    kv_tok = 2 * c["L"] * c["n_kv"] * dh * nbytes       # sửa lỗi 1 và 2
    kv_seq = kv_tok * T * B / GiB
    n_mm = n - c["V"] * c["d"]                          # quy ước chương 8
    return dict(weights_gib=round(weights, 2),
                kv_gib=round(kv_seq, 2),
                max_seqs=int((gpu_gib - weights) // kv_seq),
                train_flops=6 * n_mm * D,                # sửa lỗi 4
                decode_ms=round(n * nbytes / bw * 1e3, 2))
```

Đã chạy kiểm: bản sửa cho 12.55 GiB, 2.00 GiB (MHA) và 0.50 GiB (8 đầu K/V), 16.00 GiB cho batch 8, 33 chuỗi trên GPU 80 GiB giả định, 7.929 × 10²² FLOPs, 6.74 ms/token; 7/7 phép kiểm đạt (bản lỗi hỏng 6/7). Kiểm đột biến trong phụ lục của `loigiai.ipynb`: gài lại từng lỗi riêng lẻ vào bản sửa thì ô kiểm đều bắt được (lỗi 1 làm hỏng 4 phép kiểm, lỗi 2, 3, 4 mỗi lỗi 1 phép kiểm); riêng lỗi 2, nếu chỉ kiểm cấu hình MHA thì lọt (vẫn ra 2.00 GiB), nên ô kiểm phải có cấu hình 8 đầu K/V.

## Tài liệu đào sâu và nguồn

Cột "Mức" là mức xác minh định nghĩa ở đầu chương. Các nguồn về cách phỏng vấn là của công ty khác hoặc của trường; không nguồn nào mô tả vòng technical interview của Qualcomm hay VinAI.

| Tài liệu | Dùng để làm gì | Phần nên đọc | Mức |
| --- | --- | --- | --- |
| [Qualcomm, AI Residency Program](https://www.qualcomm.com/research/artificial-intelligence/ai-residency-program) | Trang chương trình chính thức | Chỉ đọc được metadata; không có thông tin vòng tuyển | A |
| [Qualcomm RRP Batch 17 (bản trên iagora, đăng 13/10/2026)](https://www.iagora.com/work/it/offerta/stage-vietnam-ricerca/6488078) | 5 vòng; technical interview là vòng cuối, không chi tiết | Phần quy trình tuyển | B |
| [VinAI Research, Insights from AI Residency Program (webinar 2021)](https://research.vinai.io/?p=2617) | Xác nhận không có quy trình tuyển công khai | Toàn bài | A |
| [Google DeepMind, Applying for technical roles](https://deepmind.google/discover/blog/applying-for-technical-roles) | Các vòng của research scientist, technical quiz rộng, research interview khoảng 30 phút; lời khuyên là lời kể của nhân viên | Phần research scientist và lời khuyên | A |
| [Chip Huyen, Machine Learning Interviews Book](https://huyenchip.com/ml-interviews-book/) | Lời khuyên giao mỗi người phỏng vấn một nhóm kỹ năng (2.2), kết quả phụ thuộc người phỏng vấn và ứng viên khác (2.1), xử lý câu hỏi mơ hồ | Chương 2 (2.1–2.3) và Appendix A; mục 4.4.1 không đọc được | A |
| [Google re:Work, A guide to structured interviewing](https://rework.withgoogle.com/intl/en/guides/a-guide-to-structured-interviewing-for-better-hiring-practices) | Câu hỏi behavioral và hypothetical, câu nối tiếp định sẵn, thang có mốc (góc người phỏng vấn) | Phần câu hỏi và thang chấm | A |
| [Amazon, Amazon interview guide (08/04/2025)](https://aboutamazon.com/news/workplace/amazon-interview-guide) | STAR, số liệu, câu "I" | Phần STAR | A |
| [Amazon, Amazon job interview questions, tips, preparation (12/06/2024)](https://aboutamazon.com/news/workplace/amazon-job-interview-questions-tips-preparation) | STAR để đánh giá kết quả đo được | Phần STAR | A |
| [MIT CAPD, Using the STAR method (13/06/2022, sửa 30/08/2023)](https://capd.mit.edu/resources/the-star-method-for-behavioral-interviews) | Tỉ lệ 20/10/60/10, 3–5 câu chuyện, "I", trung thực | Toàn bài | A |
| [Roediger và Karpicke (2006), Psychological Science 17(3):249–255](https://pubmed.ncbi.nlm.nih.gov/16507066/) | Testing effect: thua học lại sau 5 phút, thắng ở bài kiểm tra trễ 2 ngày và 1 tuần | Abstract | A |
| [Rowland (2014), Psychological Bulletin 140(6):1432–1463](https://pubmed.ncbi.nlm.nih.gov/25150680/) | Recall có lợi hơn recognition | Abstract | A |
| [Cepeda và cộng sự (2006), Psychological Bulletin 132(3):354–380](https://pubmed.ncbi.nlm.nih.gov/16719566/) | Spacing (839 phép đo, 317 thí nghiệm, 184 bài); khoảng tối ưu tăng theo khoảng giữ | Abstract | A |
| [Dunlosky và cộng sự (2013), PSPI 14(1):4–58](https://scholars.duke.edu/publication/954654) | Xếp hạng 10 kỹ thuật học: cao (practice testing, distributed practice), trung bình (self-explanation, elaborative interrogation, interleaving), thấp (đọc lại, tô đậm, tóm tắt) | Abstract và bảng tổng kết | A |
| [Chi và cộng sự (1994), Cognitive Science 18(3):439–477](https://asu.elsevierpure.com/en/publications/eliciting-self-explanations-improves-understanding/) | Self-explanation: 14 học sinh lớp 8 được nhắc tự giải thích so với 10 đọc hai lần; mẫu nhỏ | Abstract | A |
| [Bisra và cộng sự (2018), tóm tắt của BPS Research Digest](https://www.bps.org.uk/research-digest/self-explanation-powerful-learning-technique-according-meta-analysis-64-studies) | Meta-analysis self-explanation: 64 nghiên cứu, gần 6 000 người, g = 0.55 [cần kiểm chứng volume, trang; trang Springer bị chặn] | Toàn bài | C |
| [Macnamara, Hambrick, Oswald (2014), Psychological Science 25(8):1608–1618](https://pubmed.ncbi.nlm.nih.gov/24986855/) | Luyện có chủ đích giải thích 26% phương sai ở trò chơi, 21% âm nhạc, 18% thể thao, 4% giáo dục, dưới 1% nghề nghiệp | Abstract | A |
| [Leviathan, Kalman, Matias, Fast Inference from Transformers via Speculative Decoding (ICML 2023, PMLR 202:19274–19286; bản ar5iv của arXiv 2211.17192)](https://ar5iv.labs.arxiv.org/html/2211.17192) | C18, ví dụ 14.10: Eq. (1), Theorem 3.8, 2X–3X trên T5-XXL; trang hội nghị: proceedings.mlr.press/v202/leviathan23a | Mục 2 và 3 | A |
| [Guo, Pleiss, Sun, Weinberger, On Calibration of Modern Neural Networks (ICML 2017)](https://proceedings.mlr.press/v70/guo17a.html) | C10: hiệu chỉnh kém, temperature scaling | Abstract, mục temperature scaling | A |
| [Hinton, Vinyals, Dean, Distilling the Knowledge in a Neural Network (arXiv 1503.02531, bản ar5iv)](https://ar5iv.labs.arxiv.org/html/1503.02531) | C09: vì sao nhân T² | Mục 2 | A |
| [Ng và Jordan, On Discriminative vs. Generative Classifiers (NIPS 2001)](https://papers.nips.cc/paper/2001/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html) | C05: hai chế độ dữ liệu ít và nhiều | Abstract và kết luận | A |

**Điều chưa biết (đừng viết như đã biết).**

- Định dạng technical interview của Qualcomm RRP và VinAI (thời lượng, người phỏng vấn, nội dung, ngôn ngữ, hình thức, tiêu chí chấm) và bộ câu hỏi của hai chương trình (D); VinAI còn tuyển hay không (D), như chương 13.
- Trang "how we hire" chính thức của Google Careers và amazon.jobs bị chặn trong môi trường kiểm nguồn; trang In-person interview của amazon.jobs mở được nhưng phần STAR không tải; sách không dẫn nội dung các trang đó. Mục "Do's" (4.4.1) của sách Chip Huyen cũng không đọc được.
- Chưa mở được: paper Mixed Precision Training (Micikevicius và cộng sự, 2017); hai paper mode connectivity (Garipov và cộng sự, arXiv 1802.10026; Draxler và cộng sự, arXiv 1803.00885; mới thấy qua kết quả tìm kiếm, mức C); trang Springer của Bisra 2018; chi tiết uptraining trong paper GQA.
- Tỉ lệ token tiếng Việt so với tiếng Anh của tokenizer thật (GPT, Llama): sách chỉ có số đồ chơi của chương 9 (Việt/Anh 167/58 ≈ 2.88 dưới tokenizer học trên tiếng Anh; 2.2 là mức tokenizer học trên tiếng Việt thu lại cho cùng câu Việt).
- Bộ nhớ thật khi chạy mô hình 7B 4-bit trên laptop 6 GiB của bạn: ước lượng 5.33 GiB chưa tính overhead; phải đo.

**Từ khóa để tự tìm thêm:** ML research interview, technical quiz machine learning, structured interview rubric, STAR method, retrieval practice, spacing effect, self-explanation, speculative decoding acceptance rate, temperature scaling calibration, McNemar sample size, mode connectivity, permutation alignment, mixed precision loss scaling.

**Trước khi sang tuần 39:** đạt từ 24/30 điểm mini-test; sổ lỗi có ít nhất một điểm cho 12 chuỗi đã làm (4 mỗi tuần, tuần 35–37) và P1, P2, P4, và mọi chuỗi dưới 8/12 đã được luyện lại một lần; 6 câu chuyện STAR đã ghi âm, mỗi câu dưới 2.5 phút; đã chạy một buổi phỏng vấn thử 60 phút và chấm bằng thang 30 điểm của chương 14. Tuần 39–40 dùng kết quả này cùng bài chẩn đoán chương 0 để xếp hạng điểm yếu.

Nguồn tra cứu ngày 16/11/2026 (tin Batch 17 mở lại cùng ngày; trang ghi "verificato il 12 Nov 2026"; Batch 17 đã đóng nhận hồ sơ ngày 12/11/2026, nên khi đợt 2027 mở, đọc lại danh sách vòng như chương 13 đã dạy).
