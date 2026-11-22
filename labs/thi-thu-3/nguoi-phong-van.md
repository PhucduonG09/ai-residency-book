# Đề thi thử 3 — Tài liệu cho người phỏng vấn (Vòng 4)

*File này đi kèm `book/thi-thu-3.md`. Người làm đề in hoặc gửi file cho người phỏng vấn một ngày trước buổi 2 và **không tự mở** nó trước khi buổi 2 kết thúc. Sau buổi 2, người làm đề mới mở file để đối chiếu phiếu chấm, đọc ý đáp án và câu trả lời mẫu, rồi ghi sổ lỗi.*

Thứ tự trong file: Hướng dẫn → Phiếu câu hỏi (mỗi chuỗi một tiêu đề riêng, in tách được) → Ý đáp án và lỗi hay gặp → Phiếu chấm → Câu trả lời mẫu tầng 1–2 (cho người làm đề, đọc sau buổi) → Số liệu đã kiểm của bốn chuỗi.

## Hướng dẫn cho người phỏng vấn

Cảm ơn bạn đã giúp. Bạn đóng vai người phỏng vấn kỹ thuật trong 45 phút, bằng tiếng Anh. Bạn không cần là chuyên gia: mỗi câu hỏi có sẵn ý đáp án và lỗi hay gặp ở mục sau. Việc của bạn là đọc câu hỏi, giữ giờ, ghi chép và chấm theo phiếu. Đừng gợi ý đáp án. Ứng viên bí quá một phút thì bạn có thể hỏi "What would you check first?" rồi chờ thêm.

**Trước buổi.** Bốc 2 trong 4 chuỗi N1–N4 (tung xúc xắc, hoặc rút thăm). Đừng cho ứng viên biết trước. Đọc ý đáp án của 2 chuỗi đó, của câu dự án và của câu hành vi. Nhờ ứng viên bật ghi hình cả buổi.

**Khung 45 phút.** Mở đầu 3 phút ("Please introduce yourself in about a minute."); 2 chuỗi × 12 phút; đào sâu dự án 8 phút; 1 câu hành vi 6 phút; ứng viên hỏi lại 3 phút; đệm 1 phút. Mỗi chuỗi có 4 tầng, khoảng 3 phút mỗi tầng. Hết 3 phút thì chuyển tầng, dù câu trả lời chưa xong.

**Ghi chép trong lúc hỏi** (để chấm, xem Phiếu chấm):

- Điểm 0–3 của từng tầng.
- Số con số có đơn vị mà ứng viên nói **đúng** trong phần kỹ thuật (ví dụ "24.6 million parameters", "1.4 nats per token").
- Ứng viên có hỏi lại để làm rõ câu hỏi không. Khi bí, ứng viên nói phần mình biết và cách kiểm, hay bịa một con số.
- Có câu trả lời nào dài quá 3 phút mà ứng viên không dừng lại hỏi bạn có muốn đi sâu hơn không.

**Chế độ tự ghi hình** (không có ai hỏi). Người làm đề nhờ một người bất kỳ in hoặc chép câu hỏi của 2 chuỗi bốc ngẫu nhiên lên thẻ, mỗi thẻ một tầng, rồi úp xuống. Nếu không nhờ được ai, người làm đề chạy `python -c "import random; print(random.sample([1, 2, 3, 4], 2))"`, chỉ mở mục **Phiếu câu hỏi** tới đúng hai chuỗi đã bốc và lật từng tầng một; không cuộn xuống mục Ý đáp án. Chế độ này yếu hơn có người hỏi, vì người làm đề nhìn thấy tên cả bốn chuỗi.

## Phiếu câu hỏi

### N1 — Weight tying: one matrix at both ends (chương 2, 8, 9)

- *Tầng 1.* What is weight tying between the input embedding and the output layer of a language model? How many parameters does it save for a 32 000-token vocabulary and d = 768?
- *Tầng 2.* Differentiate the loss through both uses of the tied matrix. Which rows receive gradient from the input side, and which from the output side?
- *Tầng 3.* For a model with 85 million non-embedding parameters, what fraction of the untied model does tying save? Why does the same choice matter much less for a model shaped like Llama-2-7B, with a 32 000-token vocabulary and d = 4096?
- *Tầng 4.* What can go wrong for rare tokens when the matrix is tied, and how would you check whether it is happening in your model?

### N2 — Gradient accumulation and the token-count trap (chương 4, 6, 9)

- *Tầng 1.* What is gradient accumulation, and why can it give the same gradient as one large batch?
- *Tầng 2.* You accumulate over k micro-batches that contain different numbers of target tokens. Inside each micro-batch you average the loss over its own tokens, and then you average the k micro-batch losses. Show that the result is not the token-level mean over the whole batch.
- *Tầng 3.* Micro-batch 1 has 100 target tokens with summed loss 300; micro-batch 2 has 400 target tokens with summed loss 400. Compute both averages and the weight that each token receives. What changes with four GPUs in data-parallel training?
- *Tầng 4.* How would you test that your accumulation loop matches a single large batch, and which parts of a training loop can still differ even when the loss is normalized correctly?

### N3 — Self-consistency: voting over sampled answers (chương 3, 9, 12)

- *Tầng 1.* What is self-consistency decoding, and how is it different from greedy chain-of-thought decoding?
- *Tầng 2.* Each sample is correct with probability p, independently, and every wrong sample gives the same wrong answer. What is the probability that a majority vote over 5 samples is correct? Compute it for p = 0.6 and p = 0.4.
- *Tầng 3.* Keep p = 0.4, but now wrong samples are spread evenly over three different wrong answers. What changes? What does correlation between samples do to the gain, and how does the method compare with pass@k in cost and in what it needs?
- *Tầng 4.* How would you turn the agreement between samples into a confidence score that decides when a RAG system should abstain, and how would you check that this confidence is calibrated?

### N4 — Token classification after subword tokenization (chương 5, 9, 12)

- *Tầng 1.* In named entity recognition with BIO tags, how do you assign word-level labels to subword tokens for training and read them back at prediction time?
- *Tầng 2.* A 10-word sentence contains a two-word person name and a one-word organization. Model 1 predicts O everywhere. Model 2 finds the organization and tags only the first word of the name. Compute token accuracy and strict entity-level F1 for both.
- *Tầng 3.* Why should you evaluate at the word or span level rather than the subword level? What goes wrong if you compare two models with different tokenizers at the subword level?
- *Tầng 4.* In Vietnamese, one word can be several syllables separated by spaces. How does that interact with BIO tagging and with an English-trained tokenizer, and how would you design a fair evaluation?

### Đào sâu dự án (8 phút; dự án luồng A của ứng viên)

- First, state your project's main result in one or two sentences, with the number and the baseline.
- Walk me through one specific failure of your system: the question, the stage that failed, and how you found it.
- How many questions in your test set fail in the same way, and how do you know? What did you change, and which number shows it helped rather than moved the error elsewhere? Over how many runs?
- What is the biggest limitation of your evaluation, and could it change your main claim?

### Hành vi (6 phút: ứng viên kể khoảng 2 phút, bạn hỏi nối tiếp 4 phút)

- Tell me about a time when a result that you or your team liked turned out to be wrong, or weaker than you thought. What did you do?
- Nối tiếp: "Who did you tell, and when?" và "What do you do differently now?"

## Ý đáp án và lỗi hay gặp

**N1.** *Ý chính.* Tầng 1: buộc trọng số là dùng cùng một ma trận E ∈ ℝᵛˣᵈ cho cả tra embedding đầu vào lẫn lớp ra (logit z = Eh). Tiết kiệm đúng một ma trận: 32 000 × 768 = 24 576 000 tham số. Press và Wolf (EACL 2017, arXiv 1608.05859; theo abstract, đọc ngày 17/11/2026) đề xuất cách này và báo perplexity giảm trên nhiều mô hình ngôn ngữ. Tầng 2: gradient theo E là tổng của hai đường dùng E, đúng quy tắc fan-out của chương 2. Phía ra là ∑ₜ (pₜ − yₜ)hₜᵀ, chạm **mọi** hàng (mọi token có xác suất dương). Phía vào chỉ cộng gradient vào hàng của những token có mặt trong đầu vào. Tầng 3: không buộc 85 000 000 + 2·24 576 000 = 134 152 000 tham số, buộc còn 109 576 000, tiết kiệm 18.3%. Với hình dạng Llama-2-7B, một ma trận 32 000 × 4096 có 131 072 000 tham số, chỉ khoảng 1.9% của 6 738 415 616 (cách đếm của ví dụ 9.10 ra 6.738·10⁹, tức có `lm_head` riêng). Lợi về tham số nhỏ, nên mô hình lớn có thể để riêng hai ma trận. Đừng khẳng định mô hình nào buộc hay không khi chưa xem cấu hình của nó. Tầng 4: token hiếm gần như không xuất hiện ở đầu vào, nên hàng của nó gần như chỉ nhận gradient phía ra. Ở mọi vị trí mà nó không phải mục tiêu, gradient đó đẩy hàng theo hướng −pᵥ·h, tức ra xa trạng thái ẩn. Gao và cộng sự (ICLR 2019, arXiv 1907.12009; theo abstract) gọi hiện tượng embedding co vào một hình nón hẹp khi huấn luyện likelihood có buộc trọng số là "representation degeneration". Cách kiểm: cosin trung bình giữa các cặp embedding và chuẩn của từng hàng, chia theo nhóm tần suất token; so mô hình buộc với mô hình không buộc.

*Lỗi hay gặp.* Nói buộc trọng số giảm FLOPs: không, phép nhân của lớp ra vẫn tốn cỡ V·d phép nhân mỗi token. Quên rằng hàng không có trong đầu vào vẫn nhận gradient. Khi đã buộc mà vẫn cộng `lm_head` vào tổng tham số (chương 8 đã nhắc lỗi này).

**N2.** *Ý chính.* Tầng 1: chạy k micro-batch, cộng gradient của chúng rồi mới gọi một lần `step()`. Gradient tuyến tính theo loss, nên tổng gradient của các phần bằng gradient của tổng loss; chia đúng hệ số thì bằng gradient của batch lớn (mục 6.3, ví dụ giải mẫu 6.6). Tầng 2: loss chuẩn là ∑ mọi ℓᵢ chia cho T, với T là tổng số token mục tiêu của cả batch. Lấy trung bình trong từng micro-batch rồi trung bình k kết quả cho (1/k)·∑ⱼ Sⱼ/Tⱼ, khác ∑ⱼ Sⱼ / ∑ⱼ Tⱼ trừ khi mọi Tⱼ bằng nhau. Mỗi token ở micro-batch j nhận trọng số 1/(k·Tⱼ) thay vì 1/T. Tầng 3: trung bình đúng (300 + 400)/500 = 1.4; trung bình hai trung bình (3.0 + 1.0)/2 = 2.0. Token ở micro-batch ngắn nhận trọng số 1/200 thay vì 1/500 (gấp 2.5 lần); token ở micro-batch dài nhận 1/800 (bằng 0.625 lần). Với bốn GPU data parallel, mỗi GPU lấy trung bình cục bộ rồi all-reduce lấy trung bình, nên mắc cùng lỗi. Sửa: cộng loss, all-reduce cả tổng loss lẫn tổng số token, rồi chia. Tầng 4: chạy cùng dữ liệu một lần như một batch lớn và một lần chia micro-batch, ở float64, rồi so gradient từng tham số (phải trùng tới cỡ 10⁻¹²). Những thứ vẫn có thể khác: BatchNorm thống kê theo micro-batch (bài 13 chương 6), mặt nạ dropout khác nhau, lịch learning rate phải đếm theo bước tối ưu chứ không theo micro-batch, và gradient clipping phải làm sau khi đã cộng đủ.

*Lỗi hay gặp.* Chia loss cho k mà quên các micro-batch có số token khác nhau. Gọi `zero_grad()` sau mỗi micro-batch. Gọi scheduler sau mỗi micro-batch. Cắt gradient trước khi cộng đủ. Ghi chú: chương 9 (SFT) đã nói chọn trung bình theo token hay theo chuỗi là một lựa chọn; ở đây vấn đề là cộng dồn phải tái tạo **đúng** lựa chọn đó.

**N3.** *Ý chính.* Tầng 1: lấy mẫu nhiều chuỗi suy luận ở nhiệt độ lớn hơn 0, rút đáp án cuối của mỗi chuỗi, rồi chọn đáp án xuất hiện nhiều nhất. Greedy chain-of-thought chỉ có một chuỗi. Wang và cộng sự (ICLR 2023, arXiv 2203.11171; theo abstract, đọc ngày 17/11/2026) báo mức tăng trên nhiều benchmark suy luận, abstract ghi GSM8K "+17.9%". Tầng 2: P(Bin(5, p) ≥ 3) bằng 0.683 khi p = 0.6 và 0.317 khi p = 0.4. Bỏ phiếu khuếch đại phía đang chiếm đa số: khi p < 0.5 và mọi đáp án sai giống nhau, bỏ phiếu tệ hơn một mẫu. Tầng 3: nếu đáp án sai chia đều cho ba giá trị (mỗi giá trị 0.2), xác suất đa số tương đối của 5 mẫu chọn đúng **phụ thuộc cách xử lý hòa** (liệt kê đủ 4⁵ = 1 024 trường hợp): hòa tính là sai cho 0.394, phá hòa ngẫu nhiên cho 0.509, hòa tính là đúng cho 0.625. Câu hỏi không nói quy ước: nhận đáp số nào trong ba nếu ứng viên nói rõ quy ước mình dùng, và ghi một câu hỏi làm rõ (dòng "Làm rõ" của phiếu chấm) nếu ứng viên hỏi lại quy ước. Ý chính: chia nhỏ phía sai giúp đáp án đúng thắng đa số tương đối dù p < 0.5 (với phá hòa ngẫu nhiên, 0.509 > 0.4); riêng khi hòa tính là sai thì 0.394 còn thấp hơn một mẫu một chút. Các mẫu của cùng một mô hình tương quan với nhau, nên số phiếu độc lập hiệu dụng ít hơn 5; tương quan hoàn toàn thì bỏ phiếu không thêm gì. Chi phí gấp k lần số token sinh. pass@k (mục 9.5) cần một bộ kiểm đáp án và là cận trên kiểu "có ít nhất một mẫu đúng"; bỏ phiếu không cần bộ kiểm, nhưng không vượt được pass@k. Tầng 4: độ tin = tỉ lệ mẫu đồng ý với đáp án được chọn; từ chối khi độ tin dưới ngưỡng; vẽ đường rủi ro–độ phủ (ví dụ giải mẫu 12.9 (c)) và đo ECE theo bin của độ tin (ví dụ giải mẫu 12.9 (b)) trên một tập giữ riêng. Cảnh báo: mô hình sai một cách nhất quán thì vẫn có độ đồng ý cao.

*Lỗi hay gặp.* Cho rằng bỏ phiếu luôn có lợi. Coi các mẫu là độc lập. So với greedy mà không cân bằng ngân sách tính toán. So đáp án bằng chuỗi ký tự mà không chuẩn hóa ("0.5" với "1/2").

**N4.** *Ý chính.* Tầng 1: gán nhãn cho subword **đầu** của mỗi từ; các subword sau gán −100 để loss bỏ qua (hoặc gán I- cùng loại nếu chọn huấn luyện trên mọi subword); lúc dự đoán đọc nhãn ở subword đầu. Cần bảng ánh xạ subword sang từ (`word_ids` trong thư viện tokenizer). Tầng 2: nhãn vàng là B-PER I-PER O O O B-ORG O O O O. Mô hình 1 (toàn O): độ chính xác theo token 7/10 = 0.7, F1 thực thể 0. Mô hình 2: độ chính xác 0.9; tập dự đoán {PER chỉ ở từ 1, ORG ở từ 6}, tập vàng {PER ở từ 1–2, ORG ở từ 6}; TP = 1, P = R = 0.5, F1 = 0.5. Tên người bị cắt biên vừa là FP vừa là FN. Tầng 3: số subword của một từ phụ thuộc tokenizer (từ tiếng Việt có dấu bị cắt nhiều mảnh, ví dụ giải mẫu 9.2), nên chấm theo subword cho mẫu số khác nhau với cùng một dự đoán và thưởng phạt khác nhau giữa hai tokenizer. Lớp O chiếm đa số nên độ chính xác bị thổi phồng (chương 5, dữ liệu lệch). Tầng 4: trong tiếng Việt, dấu cách ngăn âm tiết chứ không ngăn từ ("Hà Nội" là một từ hai âm tiết). Có hai lựa chọn: tách từ trước (công cụ tách từ có lỗi riêng, lỗi đó đi vào biên thực thể), hoặc gắn BIO theo âm tiết. Dù chọn cách nào, mọi hệ phải được chấm trên **cùng một đơn vị** (âm tiết hoặc ký tự), sau chuẩn hóa NFC (chương 9). Báo cả khớp chặt và khớp lỏng (có chồng lấn), giống mục 12.3 báo F1 chặt và F1 sau chuẩn hóa.

*Lỗi hay gặp.* Tính loss trên mọi subword với nhãn B- lặp lại, sinh ra nhiều B- liên tiếp trong một từ. Báo độ chính xác theo token thay cho F1 thực thể. So hai mô hình ở mức subword.

**Đào sâu dự án.** Câu trả lời tốt có: khẳng định chính trong tối đa 2 câu, có con số và baseline; một lỗi cụ thể với tên khâu hỏng (truy hồi, đồ thị, bộ đọc); cách tìm ra lỗi (đọc từng câu, bảng phân rã R, a₁, a₀ của mục 12.2); số câu lỗi cùng kiểu; thay đổi đã làm và con số cho thấy nó có ích, kèm số lần chạy và độ dao động; một giới hạn tự nêu ra (ví dụ một người gán nhãn chính, một lĩnh vực). Dấu hiệu yếu: không nêu được số câu lỗi; con số chỉ có một lần chạy; nói "we" suốt mà không tách được phần của mình.

**Hành vi.** Câu trả lời tốt theo STAR (mục 14.5), phần Action dài nhất, nói bằng "I", có một con số, nói rõ báo cho ai và khi nào, và kết bằng một thay đổi cách làm cụ thể. Dấu hiệu yếu: đổ lỗi cho người khác; kết thúc ở "we fixed it" mà không có bài học.

## Phiếu chấm

**Thang tự chấm 4 tầng** (chép từ mục 14.1, quy tắc của sách): mỗi tầng 0–3 điểm, một chuỗi tối đa 12. 0: trống hoặc sai. 1: đúng từ khóa hoặc định nghĩa, chưa có cơ chế. 2: đúng cơ chế hoặc suy dẫn, có các bước. 3: mốc 2, cộng một con số đúng có đơn vị **và** một điều kiện hoặc giới hạn.

**Thang 30 điểm cho phỏng vấn thử** (chép từ mục 14.5, thang của chương 14, không phải tiêu chí chính thức của chương trình nào). Năm dòng giữ nguyên chương 14. Riêng dòng Hành vi là **điều chỉnh của đề này**, vì đề có một câu chuyện thay vì hai; tối đa vẫn 4 điểm.

| Phần | Tiêu chí đếm được | Điểm |
| --- | --- | --- |
| Kỹ thuật (2 chuỗi) | Tổng điểm 8 tầng theo thang 0–3 (tối đa 24), chia 2 | 12 |
| Con số | Số con số có đơn vị nói đúng trong phần kỹ thuật: ≥ 6 → 3; 4–5 → 2; 2–3 → 1 | 3 |
| Làm rõ và "không biết" | ≥ 1 câu hỏi làm rõ (1); khi bí, nói phần đã biết và cách kiểm, không bịa (2) | 3 |
| Đào sâu dự án | Khẳng định trong ≤ 2 câu (1); baseline, số seed, độ dao động (1); vai trò của mình (1); tự nêu một giới hạn (1) | 4 |
| Hành vi (1 câu chuyện; điều chỉnh của đề này) | Có số (1), "I" ≥ "we" (1), có bài học (1), ≤ 2.5 phút (1); tương đương chấm câu chuyện đó hai lần theo mức 0.5 của chương 14 | 4 |
| Cách nói | Từ/phút 120–160 (1); từ đệm ≤ 3 trên 100 từ (1); không câu trả lời nào quá 3 phút mà không dừng kiểm tra người nghe (1); hỏi ≥ 2 câu thực chất ở cuối (1) | 4 |

Ngưỡng của mục 14.5: **≥ 24** sẵn sàng; **18 đến dưới 24** luyện lại phần yếu nhất; **dưới 18** quay về ngân hàng chuỗi. Dòng Kỹ thuật chia 2 nên tổng có thể lẻ 0.5; ba vùng phủ liên tục, nên 23.5 thuộc vùng giữa. Hai tiêu chí đầu của dòng Cách nói do ứng viên đếm sau buổi, bằng Lab 3 chương 14; bạn chỉ cần điền bốn dòng còn lại và tiêu chí 3, 4 của dòng Cách nói. Sau buổi, đưa lại phiếu cho ứng viên kèm ba nhận xét viết.

## Câu trả lời mẫu tầng 1–2 (cho người làm đề, đọc sau buổi 2)

Theo quy ước của chương 14: 3–5 câu, chỉ dùng số đã kiểm. Tầng 3–4 bạn tự soạn từ mục Ý đáp án, rồi luyện lại hai chuỗi người phỏng vấn chưa bốc.

**N1.**

> **Sample answer (tiers 1–2):** Weight tying means the model uses one matrix at both ends: the input embedding looks up a row of it, and the output layer multiplies the hidden state by the same matrix to get the logits. With a 32 000-token vocabulary and d = 768, that saves one 32 000 by 768 matrix, about 24.6 million parameters. The gradient of the tied matrix is the sum over its two uses. The output side adds the outer product of p minus y with the hidden state at every position, which touches every row, while the input side only adds gradient to the rows of tokens that actually appear in the input. So a rare token's row is trained almost only through the softmax.

**N2.**

> **Sample answer (tiers 1–2):** Gradient accumulation runs several micro-batches, adds up their gradients and then takes one optimizer step, which lets a small GPU imitate a larger batch. Because the gradient is linear in the loss, the summed gradients equal the gradient of the summed loss, as long as each micro-batch loss is scaled correctly. The trap is the scaling: if each micro-batch averages over its own tokens and I then average the averages, a token in a short micro-batch gets a larger weight than a token in a long one. With 100 and 400 target tokens and summed losses of 300 and 400, the correct mean is 1.4 nats per token, but the average of the two averages is 2.0.

**N3.**

> **Sample answer (tiers 1–2):** Self-consistency samples several reasoning paths at a temperature above zero, takes the final answer of each, and returns the most common answer, instead of trusting a single greedy chain of thought. If each sample is right with probability p, independently, and all wrong samples agree on the same wrong answer, a majority of five is right with the probability that a binomial with five trials has at least three successes. That is about 0.68 for p = 0.6 but only about 0.32 for p = 0.4, so voting amplifies whichever side already has the majority. In practice wrong answers are often spread out, which helps the right one win a plurality, but samples from one model are correlated, so the gain is smaller than this independent calculation suggests.

**N4.**

> **Sample answer (tiers 1–2):** With BIO tags, the first word of an entity gets B, the following words get I, and everything else gets O. After subword tokenization I keep the label on the first subword of each word and set the other subwords to minus 100, so the loss ignores them, and at prediction time I read the label from the first subword. Token accuracy is misleading because most tokens are O: in a ten-word sentence with a two-word name and a one-word organization, predicting O everywhere already gives 0.7 accuracy and an entity F1 of zero. A model that finds the organization but tags only the first word of the name gets 0.9 accuracy and an entity F1 of 0.5, because the cut name counts as both a false positive and a false negative.

## Số liệu đã kiểm (bốn chuỗi)

Các số dưới đây được `work/mock3/check_numbers.py` tính lại (kiểm ngày 2026-11-19, Python 3.13, PyTorch 2.14 CPU float64) và so với đúng chuỗi in trong file này.

| Chuỗi | Giá trị | Cách kiểm | Kết quả |
| --- | --- | --- | --- |
| N1 | 24 576 000; 134 152 000 → 109 576 000 (18.3%); 6 738 415 616, 1.9% | đếm; autograd: gradient của ma trận buộc bằng tổng hai phía, token không có ở đầu vào vẫn nhận gradient | khớp |
| N2 | 1.4 và 2.0; trọng số 2.5 và 0.625 lần | autograd: tổng rồi chia tổng token trùng batch đầy đủ, trung bình các trung bình thì không | khớp |
| N3 | 0.683, 0.317; đa số tương đối 0.394 (hòa tính sai), 0.509 (phá hòa ngẫu nhiên), 0.625 (hòa tính đúng) | công thức nhị thức; liệt kê 4⁵ kết quả với ba quy ước hòa; mô phỏng 200 000 lần (seed 5) cho quy ước ngẫu nhiên | khớp |
| N4 | độ chính xác 0.7 và 0.9; F1 thực thể 0 và 0.5 | tách span BIO theo định nghĩa | khớp |

**Giao với sách** (khái niệm đã xuất hiện, chuỗi thì mới): N1 với chương 2 (fan-out) và ví dụ 8.9 (đếm GPT-2 có buộc trọng số); N2 với ví dụ 6.6 và bài ★★★13 chương 6 (cộng dồn gradient, BatchNorm) và đoạn SFT của chương 9 (trung bình theo token hay theo chuỗi); N3 với mục 9.5 (pass@k) và ví dụ 12.9, còn lời giải câu 24 chương 0 chỉ nhắc tên self-consistency như một baseline cùng ngân sách token; N4 với mục 12.3 (NER, F1 chặt và sau chuẩn hóa) và ví dụ 9.2. Câu hỏi tiếng Anh của bốn chuỗi không trùng cụm 6 từ nào với chương 0–14 (`work/mock3/check_exam.py`).
