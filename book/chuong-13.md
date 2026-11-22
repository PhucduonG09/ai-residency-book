# Chương 13 — Hồ sơ và video presentation

## Mục tiêu và kiểm tra nhanh

Chương này dạy bạn làm ra các sản phẩm của một hồ sơ AI Residency: CV một trang, research statement, gói tài liệu cho người viết thư giới thiệu, kịch bản video bản 3 phút (giả định của sách: độ dài và ngôn ngữ thật của vòng video chưa công khai) và lịch làm ngược từ hạn nộp. Điểm khó không nằm ở toán. Nó nằm ở chất lượng bằng chứng: phần lớn lời khuyên về hồ sơ là kinh nghiệm của người đi trước, và phần quan trọng nhất (vòng video) lại không có thông tin công khai. Vì vậy chương này làm hai việc: dạy bạn sản phẩm kiểm được bằng số, và ghi nhãn trung thực cho từng điều khẳng định. Thời lượng: khoảng 2 tuần, 8 giờ mỗi tuần.

**Ba quy ước đọc chương này.**

- Mọi mẫu CV, statement, kịch bản và bản ghi trong chương là **mẫu minh họa tự soạn, số liệu là số giả**. Đừng chép chúng, đừng coi các con số 0.61 hay 0.72 là kết quả thật của ai.
- Mỗi quy tắc mang nhãn nguồn: "theo hướng dẫn của ..." hoặc "**quy tắc của sách**" (không phải của chương trình tuyển).
- Mức xác minh của thông tin về chương trình tuyển, kiểm ngày 03/11/2026: **A** là trang chính thức hoặc trang của tác giả đã mở và đọc được; **B** là bản sao tin tuyển dụng trên trang tổng hợp việc làm, chưa đối chiếu với cổng gốc; **C** là bài viết bên thứ ba hoặc ý kiến cá nhân; **D** là không tìm thấy hoặc không công khai.

**Sau chương này, bạn phải làm được:**

- ☐ Vẽ bản đồ hồ sơ: nói được vòng nào công khai, vòng nào không, và biết phải đọc danh sách tài liệu của đợt tuyển hiện tại thay vì thuộc lòng.
- ☐ Viết CV một trang mà linter bullet không gắn cờ, và giải thích vì sao linter vẫn không thay được người đọc.
- ☐ Viết research statement 250–550 từ nêu việc đã làm, vai trò của mình, giới hạn và bước tiếp theo.
- ☐ Lập lịch ngược cho thư giới thiệu và gửi gói tài liệu cho người viết đúng mốc.
- ☐ Soạn kịch bản video bản 3 phút làm mặc định chuẩn bị (khoảng 420 từ), kiểm bằng công cụ, và chéo kiểm các con số với CV.

**Kiểm tra nhanh (5 phút, không dùng máy tính).** Câu i kiểm tra mục 13.i: đúng thì đọc lướt mục đó, sai thì đọc kỹ.

1. Trong quy trình công khai của Qualcomm RRP Batch 17, video là vòng thứ mấy, và định dạng của nó đã công khai chưa?
2. Linter bullet gắn cờ gì cho bullet "Implemented BM25 baseline."?
3. Một statement 300 từ có 18 lần xuất hiện của I, my, me. Đó là bao nhiêu lần trên 100 từ?
4. Giả sử hạn nộp là 12/11/2027. Theo quy tắc của sách, bạn hỏi người viết thư vào ngày nào (120 ngày trước hạn)?
5. Kịch bản 420 từ nói ở 150 từ/phút hết bao nhiêu giây?

Đáp án ở mục Lời giải cuối chương.

## 13.1 Bản đồ hồ sơ: quy trình thật, cái gì công khai, cái gì không

Quy trình tuyển công khai của nhánh Research (RRP) thuộc Qualcomm AI Residency Vietnam, Batch 17, có 5 vòng, và vòng video là vòng thứ tư, **sau** entrance test và coding interview. Hệ quả thực tế: video không phải tài liệu nộp kèm hồ sơ ban đầu, nhưng bạn phải có sẵn bản chất lượng trước khi biết hạn nộp nó.

Bảng dưới là những gì **công khai** về Research Residency Program (RRP), Batch 17 (tin đăng 13/10/2026, mức B). Cột phải ghi những gì không công khai, mức D.

| Vòng | Công khai | Không công khai (D) |
| --- | --- | --- |
| 1. Application | Có; cửa sổ nộp 12/10–12/11/2026 | Danh sách tài liệu (tin Batch 17 không liệt kê; danh sách ở mục dưới là của Batch 14) |
| 2. Entrance test | Có | Nội dung đề |
| 3. Coding interview | Có | Nội dung đề |
| 4. Video presentation | Có | Độ dài, chủ đề, ngôn ngữ, hạn nộp, nền tảng, người chấm, tiêu chí chấm |
| 5. Technical interview | Có | Bộ câu hỏi |

% manim: PheuTuyenRRP

Vài sự kiện khác, mỗi cái kèm mức xác minh:

- **Tài liệu** (Batch 14, mức B): resume, bảng điểm, bằng tốt nghiệp (sinh viên năm cuối nộp giấy chứng nhận tốt nghiệp tạm thời), thư giới thiệu tùy chọn nhưng "strongly encouraged". **Không có research statement trong danh sách.** Vì thế statement ở mục 13.3 là công cụ tư duy và chất liệu cho ô tự luận (nếu cổng có), kịch bản video và câu trả lời phỏng vấn, chứ không phải tài liệu bắt buộc.
- **Điều kiện** (mức B): sinh viên năm cuối hoặc mới tốt nghiệp ngành CS hay gần, GPA tối thiểu 3.2/4.0 hoặc 8.0/10 hoặc tương đương, toán và lập trình mạnh, tiếng Anh đọc, viết, nói, ưu tiên bằng chứng cụ thể (nghiên cứu, cuộc thi, GitHub, môn AI). Chương trình RRP kéo dài 24 tháng; nhánh Engineering (ERP) 12 tháng theo tin Batch 1 (tin cũ, chưa rõ ngày đăng; kết quả tìm kiếm còn các tin ERP khác mà sách chưa đối chiếu), có thêm "Portfolio review" và gộp tài liệu thành một file PDF.
- **Danh sách vòng thay đổi theo batch**: Batch 14 có thêm vòng "Screening" giữa Application và Entrance test, Batch 17 không có. Bài học: đọc danh sách của đợt hiện tại, đừng thuộc danh sách này.
- **Cửa sổ nộp có thể thay đổi.** Trang chương trình (mức A) chỉ ghi "Applications should open in November", còn Batch 17 thực tế mở từ 12/10. Với đợt của bạn, sách **giả định** cửa sổ nằm trong tháng 10–11/2027. Đây là suy luận từ một mẫu duy nhất, không phải thông tin đã xác minh.
- **Fast-track** cho người đạt giải cao Olympic quốc tế Toán hoặc Tin, hoặc ACM-ICPC (Batch 17, mức B): nên biết, không nên dựa vào.

**VinAI Residency.** Quy trình hiện hành **không xác minh được** (D): các trang tuyển sinh batch 10–11 trên `news.vinai.io` bị chuyển hướng về trang chủ, trang careers không nhắc residency (mở ngày 03/11/2026). Bản đầy đủ nhất mở được là batch 9 (đăng ký 01/10–15/11/2022, mức A), không nêu các vòng tuyển. Qualcomm đã mua mảng generative AI của VinAI (nay là MovianAI, tin khoảng 01/04/2025, mức A), nhưng bài tin đó không nói gì về residency. Vì vậy **không được viết** rằng chương trình Qualcomm là phiên bản kế thừa của VinAI Residency. Cách viết an toàn: hai chương trình khác nhau về tổ chức; quy trình VinAI hiện hành chưa xác minh; sách lấy quy trình Qualcomm làm khung chính.

**So với chương trình khác.** Meta AI Residency: theo bài đăng 06/10/2022 cho khóa 2023 (mức A), chương trình nhận chủ yếu cử nhân hoặc thạc sĩ mới tốt nghiệp, 1 năm toàn thời gian, kinh nghiệm ML trước đó là điểm mạnh nhưng không bắt buộc; chưa rõ chương trình còn tuyển [chưa công bố trong nguồn đã mở; soát lại 05/11/2026]. Apple AIML Residency 2026 (mức A) hướng tới người có bằng cao học, 1 năm, trang thông báo không liệt kê tài liệu. Mẫu chung trích được từ nguồn công khai khá nghèo; điểm chung duy nhất đủ chắc là các chương trình đánh giá tiềm năng nghiên cứu hơn thành tích đã công bố.

**Mô hình "mỗi tài liệu trả lời một câu hỏi của người chấm"** — đây là mô hình sư phạm của sách, không phải tiêu chí chính thức của chương trình nào:

| Tài liệu | Câu hỏi nó phải trả lời |
| --- | --- |
| CV | Bạn có bằng chứng đã làm việc không? |
| Bảng điểm | Nền tảng toán và lập trình có vững không? |
| Thư giới thiệu | Người khác có tin điều đó không? |
| Video | Bạn giải thích rõ và trung thực không? |
| Technical interview | Bạn hiểu sâu đến đâu? |

**Ví dụ giải mẫu 13.1 — thang GPA và đếm vòng.**

- Ngưỡng 3.2/4.0 bằng 3.2 / 4 = 0.8, và 8.0/10 cũng bằng 0.8. Sách đổi thang bằng tỉ lệ cho đơn giản; thật ra chương trình ghi "hoặc tương đương" và có thể có cách đổi riêng. Nhiều trường Việt Nam có bảng quy đổi chính thức, không nhất thiết tuyến tính (có trường quy 8.0/10 về khoảng 3.5/4 chứ không phải 3.2; **chưa xác minh**, tùy trường). Cách an toàn: ghi GPA theo thang của trường, kèm bảng quy đổi chính thức của trường.
- GPA 3.4/4.0 cho tỉ lệ 0.85, đạt. Điểm 7.9/10 cho 0.79, chưa đạt. Cả hai số đều phải ghi kèm thang: "3.4" trơ trọi không cho người đọc biết gì.
- Đếm vòng: Batch 17 có 5 vòng, Batch 14 có 6 (thêm Screening). Một danh sách "5 vòng" học thuộc từ năm trước có thể đã sai.

**Người chấm đọc gì ở đây.**

- Hồ sơ là một chuỗi tài liệu, mỗi tài liệu có một nhiệm vụ riêng; tài liệu thừa không thay được tài liệu thiếu.
- Thông tin trung thực về mức chắc chắn là một phần của sự chuẩn bị: nói "chưa biết" tốt hơn bịa quy trình.

**Việc nên làm ngay (gợi ý của sách, không phải thông tin xác minh).** Theo dõi trang tuyển dụng Qualcomm mỗi tháng 10; khi cổng mở, đọc kỹ danh sách tài liệu thật; dự các buổi giới thiệu chương trình ở trường (ví dụ UIT có buổi giới thiệu Qualcomm AI Residency ngày 21/11/2025, mức A).

## 13.2 CV nghiên cứu cho sinh viên

CV tốt cho người chấm nghiên cứu là CV **một trang**, mỗi gạch đầu dòng có công thức **động từ hành động + bối cảnh + kết quả có số**, tối đa hai dòng. Các quy tắc dưới đây theo hướng dẫn CMU SCS (mức A): sinh viên dưới 10 năm kinh nghiệm dùng một trang, lề tối thiểu 0.5 inch, font 10–12 pt.

Những lỗi CMU liệt kê và sách làm theo: GPA không có thang (viết 3.4/4.0, không viết 3.4); liệt kê mọi môn học; ghi soft skill như "teamwork" vào mục kỹ năng; bullet mơ hồ không có kết quả. Thứ tự các mục theo độ liên quan: mục Research đặt trên mục Courses.

Ba điểm riêng cho dự án nhóm và cho người Việt:

- **Vai trò rõ ràng.** Viết "tôi viết, bạn tôi làm" (Tellex, mục statement): người chấm sẽ hỏi lại, và câu trả lời phải khớp.
- **GPA theo thang của trường.** Ghi đúng thang trường dùng (ví dụ 8.2/10 hoặc 3.4/4.0 nếu trường có bảng chính thức), không tự đổi sang thang khác bằng tỉ lệ.
- **Ảnh thẻ và thông tin cá nhân dài** là thói quen phổ biến ở CV Việt Nam. Hướng dẫn CMU khuyên không dùng. Qualcomm không nêu quy định; sách ghi: *chưa có quy định; khuyến nghị bỏ ảnh* (đây là quan điểm của sách).

Mỗi dự án trong mục Research nên trả lời bốn câu: baseline là gì, đo bằng chỉ số nào, chạy mấy seed, kết quả có khoảng dao động không (nối chương 11).

% manim: BulletCV

**Ví dụ giải mẫu 13.2 — linter bullet (Lab 2).** Linter của sách kiểm: bắt đầu bằng động từ hành động (không phải "responsible for", "worked on", "helped", "participated"), có chữ số, không dài quá 2 dòng, không tự khen. Ngưỡng 95 ký tự mỗi dòng là xấp xỉ cho font 10–11 pt, **không phải quy định**.

| Bullet xấu | Cờ |
| --- | --- |
| Responsible for working on a graph-based question answering project for a course. | weak-verb, no-number |
| Passionate and hard-working student with teamwork skills who participated in NLP research. | no-action-verb, no-number, soft-skill-claim |
| Helped my lab with various experiments on language models and many other tasks in the group including data cleaning and writing code. | weak-verb, no-number |
| Implemented BM25 baseline. | no-number |

Cả bốn bị gắn cờ (2, 3, 2, 1 cờ). Bốn bullet tốt cùng chủ đề đều sạch, ví dụ: "Built hybrid dense + citation-graph retrieval over 100 000 arXiv abstracts; raised recall@5 from 0.61 to 0.72 (3 seeds)" (119 ký tự, hai dòng).

Hai điều về chính linter, đều rút ra từ lần chạy thật:

1. **BM25 không phải số.** Bản đầu của tôi dùng regex bắt mọi chữ số nên "BM25" lọt qua. Regex đúng đòi chữ số đứng đầu một từ (ranh giới từ trước chữ số), nên "BM25" vẫn bị cờ no-number.
2. **"Có số" khác "có kết quả".** Bullet "Built ... over 100 000 arXiv abstracts" qua được cờ no-number chỉ vì có kích thước dữ liệu, dù chưa có kết quả nào. Linter chỉ là rào chắn thô; người đọc mới phán xét con số có ý nghĩa không.

**Người chấm đọc gì ở đây.**

- Con số đi kèm động từ: người chấm tìm bằng chứng đã làm, không tìm tính từ.
- Vai trò và baseline: hai chỗ người phỏng vấn sẽ đào tiếp.
- Trang giấy gọn là tín hiệu biết chọn lọc.

## 13.3 Research statement và câu trả lời tự luận

Hãy coi statement là **một đề xuất nhỏ** (Tellex, mức A): người đọc cần biết bạn sẽ làm gì, vì sao quan trọng, và bằng chứng nào cho thấy bạn làm được. Cấu trúc: đoạn mở nêu một vấn đề cụ thể; các đoạn giữa kể công việc đã làm như tóm tắt một paper nhỏ, nói rõ vai trò của mình và ghi công người khác; đoạn cuối nêu bước tiếp theo và vì sao khớp với chương trình.

Thêm từ bài của krrish94 (mức A; tác giả tự nhận là research scientist đã đọc hơn 50 bản, bối cảnh tiến sĩ Bắc Mỹ): đặt công việc tác động mạnh nhất trước thay vì kể theo thời gian; không mở bằng giai thoại thời thơ ấu; không biến statement thành CV dài; không chỉ kể quá khứ; nhất quán với CV và thư; người ngoài lĩnh vực đọc vẫn hiểu.

Từ Raffel (mức A) về cover letter của chương trình Google AI Residency cũ, **không áp trực tiếp cho Qualcomm**: 1–2 trang, 500–1000 từ; ba câu hỏi (mối quan tâm nghiên cứu và vì sao; chương trình giúp gì; một dự án đã làm và thách thức); người xét không cần thành tích lớn mà tìm sự thoải mái với bất định nghiên cứu, khả năng giải thích rõ và học nhanh. Câu "reviewers look for reasons to reject" là phát biểu của chính tác giả đó.

Quy tắc độ dài **của sách** (không phải của chương trình): bản ngắn 250–550 từ, bản dài tới 1000 từ khi chương trình yêu cầu.

**Cảnh báo trung thực.** Qualcomm không liệt kê statement. Nếu cổng nộp có ô tự luận thì dùng bản này; nếu không có, chất liệu chuyển thành kịch bản video (mục 13.5) và câu trả lời phỏng vấn.

% manim: MuiTenStatement

**Ví dụ giải mẫu 13.3 — statement yếu và mạnh (Lab 3).** Công cụ đếm từ, đại từ ngôi thứ nhất trên 100 từ, cụm sáo, "việc đã làm" (i built, i wrote, i ran, ...), "việc sẽ làm" (i plan to, i want to, ...) và số con số.

| | Bản yếu | Bản mạnh |
| --- | --- | --- |
| Số từ | 61 | 306 (6 đoạn) |
| I, my, me trên 100 từ | 13.1 | 5.9 |
| Cụm sáo | 4 | 0 |
| Việc đã làm / sẽ làm | 0 / 0 | 5 / 4 |
| Số con số | 0 | 5 |
| Đoạn mở | 61 từ (cả bài) | 34 từ |

Bản yếu mở bằng bốn cụm sáo ("since i was a child", "in today's world", "fascinated by the power of ai", "artificial intelligence is changing the world"). Bài học: hai bản khác nhau không ở văn phong mà ở **bằng chứng đo được**: con số, vai trò, giới hạn, bước tiếp theo.

Bản mạnh còn một đoạn lạc đề (đoạn 5, 59 từ, kể lại môn NLP của trường). Các đoạn có 34, 75, 48, 34, 59, 56 từ. Bỏ đoạn 5 thì số việc đã làm và sẽ làm không đổi (5 và 4), nhưng bài tụt còn 247 từ, dưới sàn 250 của sách. Cách chữa đúng là thêm bằng chứng hoặc giới hạn thật, không độn lại bằng giai thoại.

**Thực tế tính toán.** Công cụ đếm cụm chữ chứ không hiểu câu: một câu có hai cụm "việc sẽ làm" vẫn được đếm hai lần, và câu sáo viết khác đi thì lọt lưới. Dùng nó để soát, không để chấm.

**Người chấm đọc gì ở đây.**

- Một vấn đề cụ thể trong 1–2 câu, không phải "giải quyết AI".
- Phần của bạn trong dự án nhóm, và chỗ bạn thừa nhận giới hạn.
- Bước tiếp theo nối với một việc cụ thể của chương trình.

## 13.4 Thư giới thiệu

Thư giới thiệu là tài liệu duy nhất để **người khác xác nhận** bằng chứng của bạn, nên đáng làm dù chương trình ghi "tùy chọn". Qualcomm ghi "optional but strongly encouraged" (Batch 14, mức B). Chênh lệch giữa "tùy chọn" và "nên nộp" là phán đoán của sách, không phải thông tin chính thức. Chưa rõ thư nộp qua cổng hay do người viết gửi trực tiếp (D): kiểm tra khi cổng mở.

**Chọn người viết.** Tutorial của Mogu Lab (mức A, bối cảnh Mỹ) nói người hướng dẫn nghiên cứu thường viết mạnh hơn giảng viên chỉ dạy một môn. Áp vào Việt Nam là suy luận. Hỏi thẳng: "thầy/cô có thể viết một thư **mạnh** cho em không?" để người viết có đường lui; một thư yếu có thể hại.

**Gửi gói tài liệu sớm** (Mogu Lab): CV, statement, bảng điểm, mô tả chương trình, và **brag sheet**: một đoạn mô tả việc cụ thể bạn đã làm cùng người viết, điều vượt yêu cầu (sáng kiến, kiên trì), cùng mọi thứ cần giải thích (ví dụ một điểm thấp). Theo dõi bằng bảng tính. Brag sheet chống ba rủi ro: thư trùng ý với CV, thư nói điều CV không có, người viết quên chi tiết.

**Hai việc riêng cho sinh viên Việt Nam** (gợi ý của sách từ kinh nghiệm phổ biến, **chưa xác minh** cho từng trường hay từng đợt tuyển):

- **Bảng điểm và chứng chỉ tiếng Anh.** Tin tuyển yêu cầu bảng điểm (mức B) nhưng chưa rõ có nhận bản tiếng Việt không (D). Nhiều trường cấp bản tiếng Anh hoặc bản dịch công chứng qua phòng đào tạo và có thể mất vài tuần: hỏi sớm, hỏi rõ cần mấy bản. Chương trình có đòi chứng chỉ tiếng Anh không cũng chưa biết (D): nếu bạn có, chuẩn bị bản scan; nếu chưa có, đừng đoán mà theo danh sách tài liệu khi cổng mở.
- **Thầy cô chưa quen viết thư giới thiệu bằng tiếng Anh.** Gửi kèm gợi ý hoặc khung thư (quen biết bạn trong hoàn cảnh nào, việc cụ thể, so với bạn cùng lứa, điểm mạnh, giới hạn trung thực), hạn nộp rõ và cách nộp (qua cổng hay gửi trực tiếp: chưa rõ). Khung chỉ để dễ bắt đầu; nội dung và nhận xét là của thầy cô, bạn không viết thay.

**Lịch ngược, quy tắc của sách.** Mogu Lab chỉ nói "hỏi sớm, gửi vài tuần trước hạn"; các con số dưới đây do sách đặt ra. Gọi H là hạn nộp (chữ H để không nhầm với mức D, không công khai), mỗi mốc tính ngược:

$$
\text{mốc} = H - k \text{ ngày}, \qquad k \in \{120,\ 56,\ 21,\ 7\} \quad (\text{hỏi, gửi gói, nhắc, nhắc lần cuối})
$$

% manim: LichNguocThu

**Ví dụ giải mẫu 13.4 — lịch ngược (Lab 5).** Giả định hạn nộp 12/11/2027 (**chưa xác minh**, suy từ Batch 17). Thứ tự theo ngày:

| Ngày | Thứ | Trước hạn | Việc |
| --- | --- | --- | --- |
| 15/07/2027 | T5 | 120 ngày | Chốt 2 người viết, hỏi trực tiếp |
| 20/08 | T6 | 84 | Xin bảng điểm tiếng Anh hoặc bản dịch công chứng (gợi ý của sách) |
| 03/09 | T6 | 70 | CV bản 1, nhờ 2 người đọc |
| 10/09 | T6 | 63 | Statement bản 1 |
| 17/09 | T6 | 56 | Gửi gói tài liệu cho người viết |
| 24/09 | T6 | 49 | Quay video lần 1, tự chấm |
| 15/10 | T6 | 28 | Quay video lần 2 sau phản hồi |
| 22/10 | T6 | 21 | Nhắc thư lần 1 |
| 29/10 | T6 | 14 | Hồ sơ đủ, tự nộp thử trên cổng |
| 05/11 | T6 | 7 | Nhắc thư lần cuối |
| 09/11 | T3 | 3 | Nộp thật |

Từ lúc hỏi đến hạn: 120 ngày, khoảng 17.1 tuần. Vì video là vòng 4 và hạn nộp video không công khai, bản video cuối nên xong **trước** hạn nộp hồ sơ. Toàn bộ lịch này là suy luận, không phải thông tin đã xác minh.

**Người chấm đọc gì ở đây.**

- Thư và CV nói cùng một câu chuyện, bằng hai giọng khác nhau.
- Chi tiết cụ thể từ người đã làm việc với bạn, không phải tính từ.

## 13.5 Video presentation: kịch bản bản thủ 3 phút (giả định của sách)

Sách chọn tiếng Anh vì tin tuyển Batch 14 có đào tạo tiếng Anh kỹ thuật, tin Batch 17 ghi "specialized technical English training may be provided", và chương trình yêu cầu tiếng Anh (mức B), chứ không vì có quy định nào nói video phải bằng tiếng Anh. Nếu cổng nộp cho phép tiếng Việt hoặc đặt yêu cầu khác thì theo cổng; cấu trúc sáu đoạn và bảng số dưới đây vẫn dùng được.

Vì định dạng vòng video **không công khai** (D: không tìm được độ dài, chủ đề, ngôn ngữ, hạn nộp, nền tảng, người chấm, tiêu chí chấm; cũng không tìm được bài kể lại đáng tin của người đã qua vòng này), sách dạy mô hình bản thủ: một kịch bản **3 phút** có thể cắt từ bản 5 phút; khi cổng nộp ghi giới hạn khác, theo cổng.

**Tốc độ nói.** Baruch TFCS (mức A): trung bình khoảng 150 từ/phút, nên chậm lại khoảng 140 nếu độ rõ là vấn đề; nói nhanh không làm rõ hơn, nói quá chậm cũng giảm độ hiểu. Sách dùng 140 từ/phút làm chuẩn tính:

$$
\text{số từ} = \frac{\text{giây} \times \text{từ/phút}}{60}, \qquad 180 \text{ giây} \to 420 \text{ từ}, \quad 270 \text{ giây} \to 630 \text{ từ}, \quad 300 \text{ giây} \to 700 \text{ từ}
$$

**Cấu trúc sáu đoạn, 180 giây (quy tắc của sách).** Hook và kết luận một câu 20 giây; vấn đề và vì sao khó 25; cách làm 45; bằng chứng 45; vai trò của tôi và giới hạn 25; kết, vì sao chương trình này, 20. Ngân sách từ ở 140 từ/phút: 47, 58, 105, 105, 58, 47, tổng 420.

% manim: BaPhutPresentation

Tham chiếu tương tự, **không phải của Qualcomm**: cuộc thi Three Minute Thesis (Kent State, mức A) giới hạn 3 phút, một slide tĩnh, chấm theo Content, Comprehension, Engagement, Communication; tránh biệt ngữ, giải thích thuật ngữ.

**Nội dung.** Nêu rõ baseline, số lần chạy, ablation, giới hạn (chương 11), vai trò cá nhân và một con số trung tâm. Trung thực về giới hạn là điểm cộng trong mọi nguồn về chấm nghiên cứu (Raffel: thoải mái với bất định; Tellex: nói rõ vai trò). **Ngôn ngữ:** câu ngắn, định nghĩa thuật ngữ lần đầu, không viết tắt chưa định nghĩa.

**Kỹ thuật quay.** Không có yêu cầu công khai, nên sách chỉ khuyên các việc không rủi ro: nhìn vào ống kính, nền yên tĩnh, âm thanh rõ hơn hình ảnh, quay nhiều lần, rồi kiểm lại bằng công cụ (nhịp, ngắt, từ đệm). Sách không nêu thông số video nào như một yêu cầu.

**Ví dụ giải mẫu 13.5 — kịch bản tốt và xấu (Lab 1).** Độ dễ đọc Flesch Reading Ease: 206.835 − 1.015 × (từ/câu) − 84.6 × (âm tiết/từ).

| | Tốt (6 đoạn) | Xấu (một câu) |
| --- | --- | --- |
| Số từ, số câu | 425, 39 | 108, 1 |
| Thời lượng ở 140 từ/phút | 182.1 giây | 46.3 giây |
| Từ trên câu (trung bình, dài nhất) | 10.9, 18 | 108, 108 |
| Flesch, lớp Flesch–Kincaid | 73.2, 5.8 | −34.4, 44.9 |
| Viết tắt chưa định nghĩa | 0 | 10 (AI, FAISS, GNN, KG, LLM, MRR, NDCG, RAG, SOTA, XYZ) |

Lệch từng đoạn so với ngân sách giây: Hook +24.3%, Vấn đề +11.4%, Cách làm −4.8%, Bằng chứng −11.4%, Vai trò +2.9%, Kết +5.0%. Quy tắc kiểm của sách: tổng 160–190 giây, mỗi đoạn lệch không quá 30%. Kiểm tay công thức: "The cat sat on the mat." có 6 từ, 1 câu, 6 âm tiết, nên 206.835 − 6.09 − 84.6 = 116.1. Bộ đếm âm tiết là heuristic, sai số vài phần trăm và chỉ đúng với tiếng Anh; Flesch dùng để **so hai bản**, không phải điểm tuyệt đối. Kịch bản mẫu tốt có một câu nói thẳng "for illustration only"; trong video của bạn, con số phải là số thật.

% manim: NhipNoiWPM

**Ví dụ giải mẫu 13.6 — thống kê một bản ghi (Lab 4).** Dữ liệu **giả lập** (seed 0): nói nhanh lúc đầu, một quãng ngắt, nhiều "um". Kết quả: 440 từ trong 203.0 giây, trung bình 130.0 từ/phút; tốc độ theo cửa sổ 30 giây là 172, 140, 136, 118, 122, 106, 112 từ/phút (hệ số biến thiên 0.16); một quãng ngắt 3.2 giây; 20 từ đệm (4.5 trên 100 từ). Trung bình 130 trông ổn nhưng **che mất hai lỗi**: mở đầu 172, cuối 106–112. Cửa sổ cuối chỉ dài 23 giây: nếu chia nhầm cho 30 thì ra 86 từ/phút, một lỗi giả mà bản công cụ đầu của tôi đã mắc. Với bản ghi thật, nhập tay hơn 400 mốc thời gian là không thực tế. Cách làm: chạy Whisper hoặc faster-whisper với `word_timestamps=True` (thư viện này không có trong môi trường của sách, bạn tự cài; model cỡ nhỏ đến vừa nhiều khả năng chạy được trên GPU 6 GB, [cần kiểm chứng] trên máy của bạn), xuất CSV ba cột `start,end,word`, rồi đọc bằng hàm `read_words_csv` ở ô cuối Lab 4. Cách dự phòng không cần công cụ: bấm giờ từng đoạn và chia số từ của đoạn cho thời gian của nó, được tốc độ theo đoạn thay vì theo cửa sổ 30 giây.

**Ví dụ giải mẫu 13.7 — chéo kiểm nhất quán (Lab 5).** Con số nào có ở statement hoặc kịch bản mà CV không có? GPA có thang chưa? Dự án nào chồng ngày mà chưa chú thích? Thử trên **mẫu B** (không phải hồ sơ của bài săn bug): CV ghi accuracy 0.84 (5 seeds), statement ghi 0.86, GPA 3.5/4.0, hai dự án chồng ngày. Công cụ trả về: statement có số {0.86} mà CV không có; GPA đã có thang; một cặp dự án chồng ngày (Course project và Hackathon). Giới hạn cần nhớ: công cụ chỉ bắt **chữ số**. Statement mẫu B viết "five seeds" bằng chữ nên số seed không được so; và công cụ không bắt được lỗi ngữ nghĩa.

**Ví dụ giải mẫu 13.8 — chuỗi nhất quán của một con số.** "Recall tại 5 tăng từ 0.61 lên 0.72": tăng tuyệt đối 0.11, tăng tương đối (0.72 − 0.61) / 0.61 = 18.0%. Ba seed cho mức tăng trong [0.09, 0.13] (giả định của mẫu). Người phỏng vấn sẽ hỏi: baseline là gì, bộ dữ liệu bao nhiêu câu, ai gán nhãn, dao động bao nhiêu, ablation bỏ gì. Về ablation, phải nói rõ bỏ ở hệ thống nào: bỏ reranker ở cả hai hệ thống (baseline và hệ thống có bước đồ thị) thì mức tăng của bước đồ thị còn khoảng 0.05 (giả định của mẫu). So hệ thống không reranker với baseline có reranker là đổi hai thứ một lúc. Mỗi con số trong video phải giải thích được trong 60 giây; nếu không, bỏ nó ra.

**Người chấm đọc gì ở đây.**

- Một con số trung tâm với baseline, số lần chạy và giới hạn, không phải danh sách chỉ số.
- Bạn nói rõ, chậm vừa phải, không sa đà biệt ngữ (tiêu chí tham chiếu từ 3MT, không phải của Qualcomm).
- Sau video là technical interview: người xem video sẽ hỏi về chính nội dung video.

## Bài tập giấy bút

14 bài, làm theo thứ tự. Mục tiêu: hết ★ trong 1 buổi, hết ★★ trong 2 buổi, ★★★ dành cho Thứ Bảy. Mọi mẫu dùng số giả. Bài nào sai ghi vào sổ lỗi kèm lý do sai.

**★ Hiểu định nghĩa**

1. Gán mức xác minh A, B, C hay D: (a) trang chương trình Qualcomm ghi "Applications should open in November"; (b) bản sao tin tuyển Batch 17 trên trang tổng hợp việc làm liệt kê 5 vòng; (c) một bài trên diễn đàn nói "phải có paper NeurIPS"; (d) độ dài video của Qualcomm.
2. Ghi các cờ linter (weak-verb, no-action-verb, no-number, too-long, soft-skill-claim) cho từng bullet: (a) "Worked on a chatbot for the university website." (b) "Developed a dense retriever; recall@5 rose from 0.52 to 0.58 on 150 questions (2 seeds)" (c) "Hard-working team player with strong teamwork skills." (d) "Won 2nd prize at the faculty AI contest; trained a 3-model ensemble in 2 weeks"
3. Với đoạn mở "Ever since I was young, I have had a passion for AI. In the era of big data, I want to contribute. I built a classifier on 800 reviews.", đếm: cụm sáo, con số, việc đã làm, việc sẽ làm (theo danh sách của Lab 3).
4. Nối mỗi tài liệu với câu hỏi nó trả lời theo mô hình của sách: CV, bảng điểm, thư giới thiệu, video, technical interview; câu hỏi: nền tảng toán vững không; hiểu sâu đến đâu; giải thích rõ và trung thực không; có bằng chứng đã làm không; người khác có tin không.

**★★ Mức đề thi đầu vào**

5. Viết lại bullet "Wrote the evaluation scripts for a class project and ran experiments on many different datasets with various settings and models and reported what happened in the final report, which the whole group then read." (209 ký tự) thành bullet sạch cờ, tối đa 2 dòng. Bạn được thêm số giả.
6. Chạy bằng tay (rồi kiểm bằng `check` của Lab 3): "I am interested in language models. My project improved a retriever and reached recall of 0.58 on 150 questions over 2 seeds. I wrote the evaluation scripts; a teammate built the index. The main limit is that only one person labeled the questions. Next, I plan to add a second labeler. I want to join a team that studies retrieval." Đếm từ, con số, việc đã làm, việc sẽ làm, rồi nêu một thứ statement này còn thiếu so với bản mạnh ở ví dụ 13.3.
7. Hạn nộp giả định 08/11/2027 (thứ Hai). Tính ngày hỏi người viết (120 ngày trước), gửi gói (56), nhắc (21 và 7). Mốc nào rơi cuối tuần và bạn dời thế nào?
8. Video 4 phút ở 140 từ/phút: tổng bao nhiêu từ? Giữ tỉ lệ 20:25:45:45:25:20 của sáu đoạn, mỗi đoạn bao nhiêu giây và bao nhiêu từ?
9. Sáu cửa sổ 30 giây của một bản ghi có tốc độ 150, 148, 152, 110, 95, 90 từ/phút. Tính trung bình và nhận xét: trung bình có cho bạn biết lỗi nhịp không? Nên sửa gì?
10. Mẫu C. CV: "GPA 3.6/4.0" và "Fine-tuned a BERT classifier on 12 000 tickets; F1 rose from 0.71 to 0.78 (4 seeds)". Statement: "F1 rose from 0.71 to 0.79 over four seeds, on 12 000 tickets. I won second place at a contest with 40 teams." Kịch bản: "F1 rose from 0.71 to 0.78 on 12 000 tickets." Công cụ chéo kiểm của Lab 5 báo gì? Nó bỏ sót gì?

**★★★ Mức phỏng vấn nghiên cứu**

11. Thiết kế video 5 phút (700 từ ở 140 từ/phút) theo sáu đoạn. Rút bản đó xuống 3 phút khi cổng chỉ cho 3 phút: vẫn giữ đủ sáu đoạn, nêu số từ mỗi đoạn của bản 3 phút, đoạn nào bạn rút chi tiết nhiều nhất và vì sao, và cách ghi "bản cắt" để các con số vẫn nhất quán với bản dài.
12. Một statement dài khoảng 870 từ có bảy đoạn (số từ, chức năng): 60 vấn đề; 140 giai thoại tuổi thơ; 180 dự án chính; 120 dự án thứ hai; 150 hoạt động ngoại khóa; 130 giới hạn; 90 bước tiếp theo và khớp chương trình. Rút xuống trong khoảng 250–550 từ mà vẫn giữ đủ vai trò, bằng chứng, giới hạn, bước tiếp theo. Nêu đoạn nào bỏ, đoạn nào cắt, tổng số từ.
13. Lập kế hoạch xin thư cho hai người: người hướng dẫn nghiên cứu và giảng viên một môn. Viết nội dung năm mục của brag sheet cho mỗi người, câu hỏi đầu tiên bạn hỏi họ, và việc làm nếu một người từ chối hoặc chưa gửi thư ở mốc 21 ngày trước hạn.
14. Bạn báo "recall@5 tăng từ 0.61 lên 0.72 trên 200 câu". (a) Mức tăng tương ứng bao nhiêu câu? (b) Sai số chuẩn thô của mỗi tỉ lệ, √(p(1 − p)/n), của hiệu hai tỉ lệ (coi hai hệ thống độc lập), √(SE₁² + SE₂²), và hiệu bằng bao nhiêu lần sai số đó? (c) Vì sao ước lượng này thô, và chương 11 cho cách tốt hơn?

## Bài tính nhanh

15 bài, mỗi bài tối đa 2 phút, làm bằng giấy bút và bấm giờ cả bộ. Mục tiêu: xong cả 15 bài trong 25 phút, sai không quá 1 bài. Làm lại bộ này sau 3 ngày và sau 1 tuần.

1. Video 4.5 phút ở 140 từ/phút có bao nhiêu từ?
2. Kịch bản 504 từ, nói 140 từ/phút: bao nhiêu giây?
3. Cùng 504 từ ở 150 từ/phút: bao nhiêu giây, ngắn hơn bao nhiêu giây?
4. GPA 3.1/4.0 và 7.9/10: cái nào đạt ngưỡng tỉ lệ 0.8?
5. Hạn nộp 20/11/2027: gói tài liệu cho người viết gửi ngày nào (56 ngày trước)?
6. Linter, bullet "Helped with various experiments.": bao nhiêu cờ?
7. Linter, bullet "Trained a 6-layer transformer on 50 000 sentences; BLEU 21.3 (2 seeds)": bao nhiêu cờ?
8. Một câu 56 từ. Tách tối thiểu thành mấy câu để mỗi câu không quá 25 từ?
9. Statement 360 từ có 27 lần I, my, me: bao nhiêu trên 100 từ?
10. 77 từ trong một cửa sổ 30 giây: bao nhiêu từ/phút?
11. Bốn cửa sổ 120, 150, 90, 140 từ/phút: trung bình, và cửa sổ nào ngoài vùng 130–150?
12. Recall tăng từ 0.61 lên 0.72: tăng tuyệt đối và tương đối?
13. Bộ test 200 câu, mức tăng 0.11: bao nhiêu câu?
14. CV: "0.58 to 0.66 on 150 questions". Kịch bản: "0.58 to 0.66 on 120 questions". Con số nào lệch?
15. Trang khổ Letter cao 11 inch, lề 0.5 inch mỗi phía, dòng cách nhau 12 pt (1 inch = 72 pt): tối đa bao nhiêu dòng?

## Lab code và bài săn bug

Năm lab dưới đây đã chạy với Python 3.13, NumPy 2.5 và pandas 3.1; toàn bộ notebook lời giải chạy dưới 10 giây. Mọi mẫu là minh họa với số giả. Cách học đúng: tự viết từng lab trước, chỉ đối chiếu code mẫu khi kết quả lệch với cột "kết quả mong đợi". Bước cuối mỗi lab (không bắt buộc): dán dữ liệu thật của bạn vào và đọc kết quả. Lưu ý khi dán bullet thật vào Lab 2: danh sách động từ trong notebook chỉ có khoảng 50 từ, nên động từ tốt mà ngoài danh sách (ví dụ "curated") sẽ bị cờ no-action-verb sai; hãy tự mở rộng danh sách.

| Lab | Việc phải làm | Kết quả mong đợi |
| --- | --- | --- |
| 1. Kiểm kịch bản video | `analyse`: từ, câu, giây, Flesch, câu dài, viết tắt chưa định nghĩa; `section_budget` so sáu đoạn với ngân sách giây | Mẫu tốt 425 từ, 39 câu, 182.1 giây, Flesch 73.2, 0 viết tắt lạ. Mẫu xấu 108 từ một câu, 46.3 giây, Flesch −34.4, 10 viết tắt. "The cat sat on the mat." cho 116.1 |
| 2. Linter bullet CV | `lint_bullet`: động từ hành động, có số, tối đa 2 dòng, không soft skill | Bốn bullet xấu có 2, 3, 2, 1 cờ; bốn bullet tốt 0 cờ; "BM25" bị cờ no-number; bullet 266 ký tự bị too-long(3 lines) |
| 3. Phân tích statement | `check`: từ, ngôi thứ nhất, cụm sáo, đã làm/sẽ làm, con số, đoạn mở | Yếu: 61 từ, 4 cụm sáo, 0 số. Mạnh: 306 từ, 0 cụm sáo, 5 đã làm, 4 sẽ làm, 5 số, đoạn mở 34 từ, 5.9 trên 100 từ. Bỏ đoạn lạc đề: 247 từ |
| 4. Thống kê bản ghi | `talk_stats`: từ/phút, cửa sổ 30 giây, quãng ngắt, từ đệm; `read_words_csv` đọc bản ghi thật dạng CSV | 130.0 từ/phút; cửa sổ 172, 140, 136, 118, 122, 106, 112; một quãng ngắt 3.2 giây; 20 từ đệm; chia nhầm cửa sổ cuối cho 30 ra 86; ô cuối đọc lại CSV `start,end,word` ra cùng kết quả |
| 5. Nhất quán và lịch ngược (mẫu B) | `nums`, `overlaps`, `consistency_report`, `timeline` | Statement-không-CV chỉ có {0.86}; một cặp dự án chồng ngày; hạn 12/11/2027: gửi gói 17/09/2027, hỏi người viết 15/07/2027 (17.1 tuần) |

Code nằm trong `labs/chuong13/lab.ipynb` (bản có chỗ TODO để bạn tự viết, kèm ô kiểm tra tự động) và `labs/chuong13/loigiai.ipynb` (bản lời giải đã chạy kiểm). Mở từ trang **Lab** của app.

**Câu hỏi sau lab.**

- Lab 4: vì sao trung bình 130 từ/phút không đủ để kết luận bản ghi ổn?
- Lab 1: vì sao Flesch dùng để so hai bản hơn là chấm điểm tuyệt đối?
- Lab 2 và 3: vì sao linter không thay được người đọc? Cho một bullet qua linter mà vẫn là bullet tệ. Với bullet thật của bạn, cờ no-action-verb nào là do danh sách động từ quá ngắn?
- Lab 5: vì sao chéo kiểm bằng số bắt được ít lỗi hơn người đọc?
- Một con số "đẹp" trong CV có nên giữ nếu bạn không giải thích được trong 60 giây?

**Bài săn bug.** Trong notebook có một gói hồ sơ giả (CV, statement, kịch bản, danh sách dự án). Triệu chứng: *mỗi file đọc xuôi, nhưng người chấm thứ hai báo hồ sơ có chỗ không khớp*. Không có dòng nào báo lỗi, vì mỗi tài liệu đúng riêng lẻ; chỉ so chéo mới thấy. Có **bốn** lỗi gài sẵn. Công cụ của Lab 5 chỉ thấy một phần, và có thể báo cả cờ mà thực ra chỉ cần giải thích: bạn tự phán xét, rồi đọc chéo bằng mắt. Sửa vào các biến `FIXED_*` bằng cách đưa dữ liệu đúng vào, không xóa câu. Ô kiểm tra chỉ in ĐẠT khi hồ sơ nhất quán và không nói lỗi nằm ở đâu.

**Bài săn bug code (tùy chọn).** Hai hàm nhỏ trong notebook chạy không báo lỗi nhưng sai. Triệu chứng: thời lượng của kịch bản mẫu in ra chỉ vài giây thay vì ba phút, và một chữ viết tắt đã được định nghĩa vẫn bị đòi định nghĩa. Sửa tại chỗ rồi chạy ô kiểm tra. Đáp án ở mục Lời giải.

## Bài dựng cảnh Manim

Mỗi chương có một bài tự dựng cảnh: muốn vẽ được một khái niệm, bạn phải hiểu từng bước của nó. Giới hạn **1 giờ** (tính vào luồng C), không chỉnh màu mè.

**Đề bài.** Dùng `manim/chuong13_scenes.py` làm mẫu, viết cảnh `ThanhThoiGianDuAn` vào thư mục `manim/cua-toi/`: dựng một thanh thời gian 3 phút cho **chính dự án của bạn**.

1. Sáu khối theo thứ tự Hook, Vấn đề, Cách làm, Bằng chứng, Vai trò, Kết, rộng tỉ lệ với số giây (bạn tự chọn, tổng 180).
2. Dưới thanh, mỗi khối một dòng gồm tên đoạn và **một câu** của dự án bạn.
3. Mỗi câu có một con số hoặc một việc làm cụ thể; chỉ dùng `Text`, không dùng LaTeX.

**Tiêu chí tự chấm.** (1) Tổng đúng 180 giây. (2) Không có chữ chồng lên nhau, kiểm bằng cách xem khung hình cuối. (3) Mỗi khối có một con số hoặc một việc làm cụ thể.

Render bằng nút **Render cảnh của tôi** ở trang Manim. Lời giải tham khảo nằm ở `manim/loigiai/chuong13_thanh_thoi_gian.py` — chỉ mở sau khi đã tự làm.

## Phỏng vấn, bẫy, thực tế khác lý thuyết

**Cách luyện.** Mỗi chuỗi có 4 tầng câu hỏi, đào sâu dần như người phỏng vấn thật. Các chuỗi này **bắt nguồn từ hồ sơ và video của bạn**; không có bộ câu hỏi chính thức nào của Qualcomm hay VinAI được công khai (D), nên đừng gán câu nào cho họ. Trả lời to bằng tiếng Việt trước, rồi nói lại bằng tiếng Anh, 2 phút mỗi tầng, ghi âm và tự nghe lại. Câu trả lời mẫu tiếng Anh dài 3–5 câu, gói gọn ý chính của chuỗi và dùng số giả của chương; các tầng sâu hơn bạn tự soạn bằng dự án thật của mình. Câu hỏi kỹ thuật thuần thuộc chương 14.

**Chuỗi 1 — Dự án**

1. Tóm tắt dự án trong 30 giây.
2. Vai trò của bạn là gì, đồng đội làm gì?
3. Giới hạn lớn nhất của kết quả là gì?
4. Có thêm hai tuần thì bạn làm gì?

> *Model answer:* I built a question answering system over arXiv abstracts that retrieves with text and with the citation graph. On 200 questions I labeled by hand, recall at five rose from 0.61 to 0.72, and my teammate built the data pipeline while I wrote the retrieval and evaluation code. The main limit is the test set: one main labeler and one subject area. A second labeler checked 50 of the questions and kappa was 0.78, which is acceptable but not strong, so I cannot yet claim the gain holds in other fields.

**Chuỗi 2 — Động cơ**

1. Vì sao research thay vì engineering?
2. Vì sao NLP, và hướng cụ thể nào bạn muốn làm?
3. Bạn muốn học gì ở chương trình, nêu việc cụ thể? Sau 24 tháng bạn định làm gì, và bạn có cam kết hết chương trình không?
4. Nếu bị từ chối, bạn làm gì tiếp?

> *Model answer:* I like research because the hardest part is deciding what to measure and whether a result is real, not only building the system. Retrieval for language models is where my project gave me evidence that I can contribute, and I want to learn how to design experiments with mentors who review them critically. After 24 months I would like to be ready to apply for a PhD, and I plan to stay for the whole program, because the learning only pays off if I finish projects with my mentors. If I am not accepted, I will finish the cross-field test and submit the work to a workshop.

Lời khuyên cho câu "sau 24 tháng": nói thật. Học tiếp tiến sĩ là một hướng hợp lý (webinar VinAI Research ngày 15/05/2021 trên trang FIT-HCMUS, mức A, giới thiệu cựu cư dân sang học tiến sĩ ở UT Austin và UCLA; đây là VinAI, không phải Qualcomm, và webinar không nêu tiêu chí tuyển). Đừng hứa điều bạn không định giữ, ví dụ cam kết ở lại làm việc nếu bạn chưa chắc.

**Chuỗi 3 — Một con số**

1. Giải thích con số trung tâm trong video.
2. Baseline là gì, vì sao công bằng?
3. Mấy seed, dao động bao nhiêu?
4. Ablation nào, nó nói gì?

> *Model answer:* The central number is recall at five, which went from 0.61 to 0.72 when I added the citation-graph step. The baseline is the same pipeline without that step, with the language model and the reranker held fixed, so the only change is the graph step. I ran three seeds and the gain stayed between 0.09 and 0.13. When I removed the reranker from both systems, the graph step added only about 0.05, so the expanded pool needs reranking.

**Chuỗi 4 — Con người**

1. Người viết thư sẽ nói gì về bạn?
2. Kể một thí nghiệm thất bại.
3. Kể một lần bất đồng với đồng đội hoặc người hướng dẫn.
4. Kể một điều bạn đã sai trong CV hoặc báo cáo và cách bạn sửa.

> *Model answer:* My advisor would say that I check baselines before claiming an improvement, because I once reproduced a reranker and found a four point gap to the reported numbers. I traced it to a tokenizer mismatch, which cost me a week but changed how I start every project. She would also say that I sometimes take on too much, and that I am learning to ask for help earlier.

**Từ vựng nên nói trôi chảy:** baseline, ablation, random seed, test set, annotator agreement, recall at five, held fixed, reranker, citation graph, brag sheet, recommendation letter, research statement, limitation.

**Bẫy thường gặp.**

- Mở bằng giai thoại cá nhân hoặc câu sáo "AI đang thay đổi thế giới" (krrish94; bản xấu của chương có 4 cụm sáo).
- Kể dự án theo thứ tự thời gian thay vì mạnh nhất trước; biến statement thành CV dài (krrish94).
- Không nói rõ vai trò cá nhân trong dự án nhóm, hoặc mục tiêu quá chung kiểu "giải quyết AI" (Tellex).
- Lỗi chính tả, quá kỹ thuật, nhắc tới chương trình khác (Raffel, chương trình Google cũ).
- CV: GPA không có thang, liệt kê mọi môn, bullet không kết quả (CMU). Video: viết tắt không định nghĩa, câu dài, nói nhanh lúc đầu (172 từ/phút ở mẫu giả lập). Chia tốc độ của cửa sổ cuối cho 30 giây cố định thay vì độ dài thật.
- Con số trong CV, statement và video không khớp nhau: mất tin cậy lớn hơn từng lỗi nhỏ.
- Thư giới thiệu gửi muộn hoặc không có brag sheet (Mogu Lab).

**Thực tế khác lý thuyết.**

- Quy trình công khai thiếu thông tin đúng ở vòng quan trọng nhất của chương này: vòng video. Nhiều lời khuyên "đã đạt" trên mạng là bản nhớ lại cá nhân, không kiểm được; sách không dẫn bản nào.
- Danh sách vòng và tài liệu **thay đổi giữa các batch** (Screening, Portfolio review). Hãy tập đọc danh sách của đợt hiện tại.
- Bằng chứng nghiên cứu, cuộc thi, GitHub được **ưu tiên** chứ không bắt buộc (Meta, theo bài đăng 06/10/2022 cho khóa 2023: kinh nghiệm ML không bắt buộc; Qualcomm: preferred). Ý kiến "phải có paper NeurIPS" trên diễn đàn là của một người (Blind 2020, mức C) và không nguồn chính thức nào xác nhận.

## Mini-test 30 phút

Làm sau khi xong bài tập và lab, bấm giờ đúng 30 phút, không xem sách, không dùng máy tính. Tổng 30 điểm. **Từ 24 điểm:** sang chương 14. **18 đến dưới 24:** ôn lại mục ứng với câu sai rồi làm lại sau 3 ngày. **Dưới 18:** học lại chương.

1. **(4 điểm)** Video 4.5 phút ở 140 từ/phút, chia theo tỉ lệ 20:25:45:45:25:20 cho sáu đoạn. Tổng số từ, và số từ mỗi đoạn.
2. **(4 điểm)** Ghi các cờ linter cho từng bullet: (a) "Assisted with data labeling for a research group." (b) "Reproduced the BM25 baseline; recall@5 = 0.52 vs 0.50 reported (3 seeds)" (c) "Passionate about NLP." (d) "Implemented BM25 baseline"
3. **(4 điểm)** Hạn nộp giả định 15/11/2027 (thứ Hai). Tính bốn mốc 120, 56, 21, 7 ngày trước hạn kèm thứ trong tuần; nêu mốc nào rơi cuối tuần và quy tắc dời bạn chọn.
4. **(4 điểm)** CV: "GPA 3.5" và "Raised recall@5 from 0.58 to 0.66 on 150 questions (3 seeds)". Statement: "recall rose from 0.58 to 0.68". Kịch bản: "from 0.58 to 0.66 on 120 questions". Nêu mọi chỗ không nhất quán và cách sửa.
5. **(8 điểm)** Đoạn dưới (135 từ, tám câu) có nhiều lỗi. Chỉ ra ít nhất sáu lỗi thuộc các loại khác nhau và viết lại câu đầu tiên thành một câu mở cụ thể. *Ever since I was a child, I have been fascinated by computers. In today's world, AI is everywhere, and I am passionate about it. I am a hard-working student, and my team and I built a RAG system with LLM, GNN and FAISS that reached SOTA results on our dataset, which is very novel, and because it is so good the performance improved significantly in terms of NDCG and MRR. We did many experiments and we think the results are promising. The model is much better than before. My teammate did most of the evaluation and I helped. I learned a lot from this project and from many courses at my university. I hope the residency will give me a chance to become a good researcher and to work on many exciting topics in AI.*
6. **(6 điểm)** Tối đa 5 câu: định dạng vòng video của Qualcomm không công khai. Bạn chuẩn bị thế nào, và bạn sẽ không khẳng định điều gì?

## Flashcard

18 thẻ của chương nằm trong `flashcards/chuong13.csv`. Ôn ở trang **Ôn tập** của app: app tự lên lịch theo hộp Leitner (thẻ nhớ được sẽ giãn lịch ôn 1 → 3 → 7 → 14 ngày, thẻ quên quay lại hộp 1). File CSV này cũng nhập thẳng được vào Anki nếu bạn thích dùng Anki.

## Lời giải

Chỉ đọc sau khi đã tự thử ít nhất 20 phút. Mọi con số tính toán trong chương đã được kiểm lại bằng code (phụ lục của `loigiai.ipynb`). Các bài mở (11, 13, mini-test câu 5 và 6) có nhiều đáp án đúng: so với ý chính và thang điểm, không so từng chữ.

**Kiểm tra nhanh**

1. Vòng 4, sau entrance test và coding interview. Định dạng (độ dài, hạn nộp, tiêu chí chấm) chưa công khai (mức D).
2. no-number. "BM25" có chữ số nhưng chữ số đó không đứng đầu một từ (không có ranh giới từ ngay trước nó) nên không tính là số, và tên phương pháp cũng không phải kết quả. Cùng quy tắc với bài 2(d): "2nd" có chữ số đứng đầu từ nên tính.
3. 18 / 300 × 100 = 6.0 trên 100 từ.
4. 12/11/2027 trừ 120 ngày là 15/07/2027 (thứ Năm).
5. 420 / 150 × 60 = 168 giây, ngắn hơn 12 giây so với ở 140 từ/phút.

**Bài tập giấy bút**

1. (a) A: trang chính thức, nhưng chỉ có thông tin chung, không nêu năm. (b) B: bản sao tin tuyển, chưa đối chiếu cổng gốc. (c) C: ý kiến cá nhân. (d) D: không công khai.
2. (a) weak-verb, no-number. (b) sạch. (c) no-action-verb, no-number, soft-skill-claim. (d) sạch: "Won" là động từ trong danh sách, "2nd" có chữ số đứng đầu từ.
3. Ba cụm sáo (ever since i was young, passion for ai, in the era of); một con số (800); một việc đã làm (i built); một việc sẽ làm (i want to).
4. CV: có bằng chứng đã làm không. Bảng điểm: nền tảng toán vững không. Thư giới thiệu: người khác có tin không. Video: giải thích rõ và trung thực không. Technical interview: hiểu sâu đến đâu.
5. Ví dụ: "Wrote evaluation scripts and ran 12 experiments on 3 datasets for a class project; reported mean and spread over 3 seeds" (động từ, bối cảnh, kết quả có số, 2 dòng, sạch cờ). Kiểm bằng linter.
6. 61 từ, 3 con số (0.58, 150, 2), 1 việc đã làm (i wrote), 2 việc sẽ làm (i plan to, i want to). Thiếu baseline: 0.58 so với cái gì? Chưa nêu ablation hay khoảng dao động, và đoạn chỉ 61 từ nên còn xa sàn 250 của sách.
7. 120 ngày: 11/07/2027 (Chủ nhật). 56 ngày: 13/09 (T2). 21 ngày: 18/10 (T2). 7 ngày: 01/11 (T2). Chỉ mốc 11/07 rơi cuối tuần; quy tắc hợp lý là dời **sớm** về thứ Sáu 09/07, vì dời muộn ăn vào thời gian của người viết.
8. 560 từ. Giây (tổng 240): 26.7, 33.3, 60, 60, 33.3, 26.7. Từ: 62, 78, 140, 140, 78, 62 (tổng 560).
9. Trung bình 745 / 6 = 124.2. Ba cửa sổ đầu (148–152) nằm trong hoặc sát mép trên vùng 130–150, ba cửa sổ cuối 110, 95, 90 chậm dần: nhịp tụt về cuối. Trung bình không nói gì về việc đó (hệ số biến thiên 0.21). Sửa: tập phần kết, và giữ nhịp đều khoảng 140.
10. Công cụ báo statement có {0.79, 40} mà CV không có; GPA đã có thang. Bỏ sót: "four seeds" viết bằng chữ nên không so được với "4 seeds" của CV; tuyên bố "second place" không có trong CV, và công cụ chỉ so số chứ không biết có thành tích đó hay không. Sửa: 0.79 thành 0.78, thêm giải thưởng (và số đội) vào CV hoặc bỏ khỏi statement.
11. 700 từ trên 300 giây: giây 33.3, 41.7, 75, 75, 41.7, 33.3; từ 78, 97, 175, 175, 97, 78. Bản 3 phút (420 từ) là bản 5 phút nhân 0.6 ở mọi đoạn: 47, 58, 105, 105, 58, 47 từ (từ 46.8, 58.2, 105, 105, 58.2, 46.8). Vẫn đủ sáu đoạn, để bản ngắn còn phương pháp (baseline, bạn đổi gì) và còn bằng chứng. Rút nhiều chi tiết nhất ở Cách làm và Vấn đề (mỗi đoạn bớt 70 và 39 từ): bỏ tên thành phần, siêu tham số, ví dụ minh họa; giữ một câu nêu baseline, một câu nêu điều bạn thay đổi, con số trung tâm, số seed, vai trò và giới hạn. Ghi một bảng số duy nhất cho cả hai bản: mọi số trong bản 3 phút phải có nguyên văn trong bản 5 phút, chỉ bỏ câu, không đổi số, và chéo kiểm cả hai bản với CV.
12. Bỏ đoạn giai thoại (140) và đoạn ngoại khóa (150): còn 580, vẫn vượt trần 550. Cắt dự án chính 180 thành 140, dự án thứ hai 120 thành 90: 60 + 140 + 90 + 130 + 90 = 510 từ, nằm trong 250–550 và đủ vai trò, bằng chứng, giới hạn, bước tiếp theo.
13. Ý chính cần có: hỏi thẳng "thầy/cô có thể viết một thư mạnh cho em không"; brag sheet năm mục (việc cụ thể làm cùng người viết, kết quả có số, điều vượt yêu cầu, khó khăn và cách vượt, điều cần giải thích hoặc cần nhấn mạnh); người hướng dẫn nghiên cứu nói về nghiên cứu, giảng viên môn nói về năng lực học và độ tin cậy, tránh trùng ý. Nếu từ chối: cảm ơn và chuyển sang người dự phòng (hỏi 120 ngày trước hạn nên còn thời gian). Nếu chưa gửi ở mốc 21 ngày: nhắc lịch sự, hỏi cần thêm gì; mốc 7 ngày nhắc lần cuối; kiểm tra cổng xem thư có bắt buộc không, và đừng để thư làm trễ cả hồ sơ.
14. (a) 0.11 × 200 = 22 câu. (b) Sai số chuẩn thô √(p(1 − p)/n): 0.0345 với 0.61 và 0.0317 với 0.72; sai số của hiệu (hai hệ thống độc lập) √(0.0345² + 0.0317²) là 0.0469; hiệu 0.11 bằng khoảng 2.35 lần sai số đó. (c) Thô vì hai hệ thống chạy trên cùng 200 câu nên kết quả tương quan; kiểm định ghép cặp (chương 11) dùng chênh lệch trên từng câu và thường nhạy hơn. Thêm nữa, bộ test chỉ có một người gán nhãn.

**Bài tính nhanh**

1. 630 từ.
2. 504 / 140 × 60 = 216 giây (3 phút 36 giây).
3. 504 / 150 × 60 = 201.6 giây, ngắn hơn 14.4 giây.
4. Cả hai chưa đạt: 3.1 / 4 = 0.775 và 7.9 / 10 = 0.79, đều dưới 0.8.
5. 25/09/2027 là thứ Bảy; theo quy tắc dời sớm của bài 7, dời về thứ Sáu 24/09/2027.
6. 2 cờ: weak-verb, no-number.
7. 0 cờ.
8. 3 câu (56 / 25 = 2.24, làm tròn lên).
9. 27 / 360 × 100 = 7.5.
10. 77 / 30 × 60 = 154 từ/phút.
11. Trung bình 500 / 4 = 125; ngoài vùng 130–150 là 120 và 90.
12. Tăng tuyệt đối 0.11, tương đối 0.11 / 0.61 = 18.0%.
13. 0.11 × 200 = 22 câu.
14. Số câu: CV ghi 150, kịch bản ghi 120.
15. (11 − 1) × 72 / 12 = 60 dòng (từ các giả định cho trước, không phải quy định).

**Câu hỏi sau lab**

- Lab 4: trung bình chỉ là một con số; nhiều bản ghi khác nhau có thể cùng 130.0 từ/phút. Ở đây mở đầu 172 và cuối 106–112 (hệ số biến thiên 0.16) bị trung bình che mất. Cửa sổ cuối phải chia cho độ dài thật của nó.
- Lab 1: bộ đếm âm tiết là heuristic với sai số vài phần trăm và công thức Flesch dựng cho văn bản viết, không phải lời nói. Khi so hai bản của cùng một kịch bản, sai số hệ thống như nhau nên phần lớn triệt tiêu.
- Lab 2 và 3: danh sách động từ của notebook ngắn nên có thể gắn cờ sai bullet tốt; mở rộng danh sách trước khi tin cờ. Linter chỉ kiểm hình thức. Bullet "Built a system over 100 000 abstracts; raised accuracy by 12 points (1 seed)" qua linter dù con số vô nghĩa: một seed, không baseline.
- Lab 5: công cụ chỉ so chữ số. Nó không bắt số viết bằng chữ, cách diễn đạt khác nhau, hay một tuyên bố có trong statement mà CV hoàn toàn không nhắc.
- Một con số mà bạn không giải thích được trong 60 giây là rủi ro hơn là lợi thế: người phỏng vấn sẽ hỏi baseline, số seed, độ dao động và ablation của chính nó.

**Mini-test (thang 30 điểm)**

1. Tổng 4.5 × 60 × 140 / 60 = 630 từ (1 điểm). Từ mỗi đoạn: 70, 87.5, 157.5, 157.5, 87.5, 70 (3 điểm; chấp nhận làm tròn, tổng 630).
2. (a) weak-verb, no-number. (b) sạch. (c) no-action-verb, no-number, soft-skill-claim. (d) no-number. (1 điểm mỗi câu.)
3. 18/07/2027 (Chủ nhật), 20/09 (T2), 25/10 (T2), 08/11 (T2) (2 điểm). Chỉ mốc 18/07 rơi cuối tuần (1 điểm); dời sớm về thứ Sáu 16/07 (1 điểm).
4. GPA không có thang, sửa thành 3.5/4.0 (1 điểm). Statement ghi 0.68, CV và kịch bản ghi 0.66: sửa theo log thí nghiệm thật (1 điểm). Kịch bản ghi 120 câu, CV ghi 150: chọn số đúng và sửa nơi sai (1 điểm). Chỉ có một cách đúng là kiểm lại với số liệu gốc, không sửa cho khớp (1 điểm).
5. Mỗi loại lỗi 1 điểm, tối đa 6, cộng 2 điểm cho câu mở viết lại. Các lỗi: (i) mở bằng giai thoại thời thơ ấu và câu sáo "in today's world"; (ii) tự khen "hard-working", "passionate" thay vì bằng chứng; (iii) viết tắt chưa định nghĩa (AI, RAG, LLM, GNN, FAISS, SOTA, NDCG, MRR); (iv) không có con số, baseline hay số lần chạy ("much better than before"); (v) vai trò mập mờ ("my team and I built", "I helped"); (vi) câu 46 từ, khó nghe; (vii) tuyên bố không có bằng chứng ("SOTA", "novel", "significantly"); (viii) không nêu bước tiếp theo hay vì sao chương trình này. Ví dụ câu mở: "Scientific question answering often returns one relevant paper and misses the papers that cite it."
6. (2 điểm) Nói thẳng rằng định dạng, hạn nộp, tiêu chí chưa công khai (mức D). (2 điểm) Kế hoạch bản thủ: kịch bản 3 phút cắt được từ bản 5 phút, tuân theo mọi giới hạn mà cổng nộp ghi khi đến lúc, chuẩn bị bản cuối trước hạn nộp hồ sơ. (2 điểm) Không khẳng định điều chưa xác minh: không nói "Qualcomm yêu cầu X", không gán thông số video hay câu hỏi cho chương trình, đánh dấu quy tắc của sách là của sách.

**Săn bug — bốn lỗi và một cờ**

1. **Con số lệch.** Statement ghi 0.75 còn CV và kịch bản ghi 0.72. Sửa statement thành 0.72, sau khi đối chiếu log thí nghiệm thật.
2. **Số chỉ có một nơi, và CV dễ bị hiểu sai.** Con số 50 (số câu dùng đo độ đồng thuận) chỉ có ở statement. CV viết "Labeled 200 ...; measured agreement ... (kappa 0.78)", nên người đọc hiểu kappa đo trên cả 200 câu, trong khi statement nói 50. Statement được phép chi tiết hơn CV; lỗi ở đây là hai tài liệu cho hai phạm vi đo khác nhau. Thêm "on 50 questions" vào CV.
3. **GPA không có thang.** "GPA 3.4" thành "GPA 3.4/4.0" (hoặc theo thang thật của trường).
4. **Lỗi ngữ nghĩa mà công cụ không bắt.** Kịch bản nói tôi tự gán nhãn và không có người gán nhãn thứ hai, còn CV và statement nói có người gán nhãn thứ hai (kappa 0.78). Không con số nào lệch nên công cụ im lặng. Hỏi lại log xem ai thật sự gán nhãn, rồi sửa nơi sai.

**Cờ cần giải thích (không tính là lỗi).** Công cụ báo dự án truy hồi (03/2026 đến 09/2026) chồng ngày với hackathon (3 ngày, 15–17/09/2026) và với dự án tái hiện reranker. Làm song song là bình thường; chú thích "concurrent" hoặc sửa ngày nếu ngày sai, để người chấm không hiểu lầm. Đổi tên chỉ làm công cụ im lặng chứ không chứng minh ngày đúng: việc cần làm là kiểm ngày.

```python
FIXED_CV = (CV.replace("GPA 3.4 (", "GPA 3.4/4.0 (")
              .replace("second labeler (Cohen", "second labeler on 50 questions (Cohen"))
FIXED_STATEMENT = STATEMENT.replace("0.75", "0.72")
FIXED_SCRIPT = SCRIPT.replace("that I labeled alone, so there was no second labeler",
                              "that I labeled myself; a second labeler checked a sample")
FIXED_PROJECTS = [("Retrieval project", date(2026, 3, 1), date(2026, 9, 30)),
                  ("Hackathon (concurrent)", date(2026, 9, 15), date(2026, 9, 17)),
                  ("Reranker reproduction (concurrent)", date(2026, 8, 1), date(2026, 8, 20))]
```

Đã chạy kiểm: bản sửa qua ô kiểm tra; đưa riêng từng lỗi (và cờ chồng ngày) vào bản sửa thì ô kiểm tra đều báo không nhất quán, kể cả hai cách viết khác của lỗi 4 ("only one person labeled it ...", "agreement is not measured yet"). Phụ lục của `loigiai.ipynb` còn kiểm rằng bộ mẫu "tốt" của chương (bullet CV, statement mạnh, kịch bản tốt) không tự mâu thuẫn về người gán nhãn thứ hai: cả ba cùng nói một câu chuyện (một người gán nhãn chính cho 200 câu, người thứ hai kiểm 50 câu, kappa 0.78). Công cụ của Lab 5 chỉ bắt được ba trong bốn lỗi, nên vẫn cần một người đọc.

**Săn bug code (tùy chọn).** (1) `duration_s` chia số từ cho wpm nên ra phút, quên nhân 60: 3.0 thay vì khoảng 180 giây. (2) `acronyms_to_define` đếm cả chữ đã định nghĩa dạng "Long Name (ACR)"; thêm bước bỏ chữ có dạng `(ACR)` hoặc "ACR, meaning ...", như `undefined_acronyms` của Lab 1.

## Tài liệu đào sâu và nguồn

Nội dung chương được viết lại và đối chiếu với các nguồn dưới đây. Cột "Mức" là mức xác minh đã định nghĩa ở đầu chương. Lời khuyên là hướng dẫn của tác giả, không phải quy định của Qualcomm hay VinAI.

| Tài liệu | Dùng để làm gì | Phần nên đọc | Mức |
| --- | --- | --- | --- |
| [Qualcomm, AI Residency Program](https://www.qualcomm.com/research/artificial-intelligence/ai-residency-program) | Trang chương trình chính thức | Chỉ metadata: "Applications should open in November"; không có vòng hay tài liệu | A |
| [Qualcomm RRP Batch 17 (bản trên iagora, đăng 13/10/2026)](https://www.iagora.com/work/it/offerta/stage-vietnam-ricerca/6488078) | 5 vòng, cửa sổ nộp, 24 tháng, fast-track | Phần quy trình tuyển | B |
| [Qualcomm RRP Batch 14 (anitab, đăng 30/05/2025, đã đóng)](https://jobs.anitab.org/companies/qualcomm/jobs/51775609-ai-resident-research) | Danh sách tài liệu, vòng Screening | Phần hồ sơ | B |
| [Qualcomm ERP Batch 1 (outscal)](https://outscal.com/job/ai-resident-engineering-at-qualcomm-in-hanoi-hanoi-vietnam-1) | Nhánh Engineering, Portfolio review, một file PDF | Phần quy trình | B |
| [UIT, giới thiệu Qualcomm AI Residency (21/11/2025)](https://www.uit.edu.vn/bai-viet/gioi-thieu-chuong-trinh-ai-residency-va-seminar-unified-one-step-diffusion-lightning-fast-image-generation-and-image-editing) | Ví dụ buổi giới thiệu ở trường | Thông báo seminar | A |
| [Maginative, Qualcomm acquires VinAI's generative AI division](https://www.maginative.com/article/qualcomm-acquires-vinais-generative-ai-division/) | Quan hệ Qualcomm–MovianAI; không nhắc residency | Toàn bài | A |
| [VinAI Residency Batch 9 (bản đăng lại của UET)](https://uet.edu.vn/vinai-residency-program-2023-batch-9-applications-now-open/) | Điều kiện và quyền lợi 2022; không có các vòng | Toàn bài | A |
| [CMU SCS Resume Guide (2020)](https://www.cmu.edu/career/documents/resources-by-college/scs-resume-guide-2020.pdf) | Một trang, công thức bullet, lỗi thường gặp | Phần bullet và lỗi thường gặp | A |
| [Tellex, Writing a Research Statement](https://h2r.cs.brown.edu/writing-a-research-statement-for-graduate-school-and-fellowships/) | Statement như đề xuất, vai trò, lỗi thường gặp | Toàn bài | A |
| [krrish94, Grad school SOP](https://krrish94.github.io/blog/2020/gradschool-sop) | Mạnh nhất trước, tránh giai thoại, nhất quán | Toàn bài (bối cảnh tiến sĩ Bắc Mỹ) | A |
| [Raffel, Writing a Google AI Residency Cover Letter](https://colinraffel.com/blog/writing-a-google-ai-residency-cover-letter.html) | Ba câu hỏi, 500–1000 từ, người xét tìm gì | Toàn bài (chương trình Google cũ) | A |
| [Mogu Lab, How to Get a Good Recommendation Letter](https://mogu-lab.github.io/tutorials/recommendation-letters) | Chọn người, hỏi, brag sheet, nhắc hạn | Toàn bài (bối cảnh Mỹ) | A |
| [Kent State, 3MT Rules and Judging Criteria](https://www.kent.edu/graduatestudies/3mt-rules-judging) | Tiêu chí tham chiếu cho bài nói ngắn | Luật và tiêu chí chấm | A |
| [Baruch TFCS, speaking rate](https://tfcs.baruch.cuny.edu/speaking-rate/) | Khoảng 150 từ/phút, khoảng 140 khi cần rõ | Toàn bài | A |
| [Meta AI Residency Profiles (đăng 06/10/2022, khóa 2023)](https://ai.meta.com/blog/meta-ai-residency-profiles/) | So sánh: đối tượng, 1 năm; chưa rõ còn tuyển | Phần giới thiệu chương trình | A |
| [FIT-HCMUS, VinAI Research Webinar: Insights from AI Residency Program (15/05/2021)](https://www.fit.hcmus.edu.vn/tin-tuc/d/thong-bao-hoi-thao-online-vinai-research-webinar-insights-from-ai-residency-program) | VinAI có track Research và Engineering; cựu cư dân sang học tiến sĩ ở UT Austin và UCLA; không có tiêu chí tuyển | Thông báo hội thảo | A |
| [Apple, 2026 AIML Residency Program Application](https://machinelearning.apple.com/updates/aiml-residency-program-application-2026) | So sánh: bằng cao học, 1 năm | Toàn bài | A |
| [Blind, "AI residency how to get in" (2020)](https://www.teamblind.com/post/ai-residency-how-to-get-in-0xj0uw6d) | Chỉ để thấy giới hạn của lời đồn | Không dùng làm căn cứ | C |

**Điều chưa biết (đừng viết như đã biết).**

- Định dạng, độ dài, chủ đề, ngôn ngữ, hạn nộp và tiêu chí chấm của vòng video Qualcomm và VinAI (D).
- Nội dung entrance test và coding interview của Qualcomm và VinAI (D).
- Tài liệu hồ sơ chính xác của đợt cuối 2027: có cần statement không, thư nộp qua cổng hay do người viết gửi.
- Quy trình VinAI Residency hiện hành, và việc chương trình còn tuyển hay không sau thương vụ Qualcomm–MovianAI (D).
- Trang chính thức của Google về residency chuyển hướng sang trang khác, nên sách không khẳng định chương trình Google AI Residency còn tuyển hay không.

**Từ khóa để tự tìm thêm:** research statement for ML residency, statement of purpose advice undergraduate ML, brag sheet recommender CS, 3MT script structure, speaking rate for non-native speakers, Whisper word timestamps, AAAI Undergraduate Consortium.

**Trước khi sang chương 14:** đạt từ 24/30 điểm mini-test, có một CV một trang không cờ, một statement đã qua công cụ và một kịch bản 3 phút đã quay ít nhất một lần, và đã chép lịch ngược vào lịch của bạn.

Nguồn tra cứu ngày 03/11/2026; Qualcomm Batch 17 và 14, webinar VinAI và Meta được mở lại và đối chiếu ngày 04/11/2026. Các điều ghi "chưa rõ" hoặc [chưa công bố] (ví dụ Meta còn tuyển hay không) là tình trạng tính đến ngày soát lại 05/11/2026.
