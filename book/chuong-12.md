# Chương 12 — Chuyên sâu NLP và đạo đức AI

## Mục tiêu và kiểm tra nhanh

Chương này dạy cách xây một hệ thống hỏi đáp trên kho văn bản học thuật (hướng Hybrid Graph-RAG trên ArXiv của dự án luồng A) và cách **đo** nó cho trung thực: truy hồi, ghép ngữ cảnh, đồ thị tri thức, đánh giá câu trả lời, hallucination, rồi công bằng, riêng tư, bản quyền, rủi ro. Thứ tự là cố ý: tìm đúng đoạn (12.1), đưa vào ngữ cảnh và đo cả đường ống (12.2), thêm cấu trúc đồ thị khi một đoạn là chưa đủ (12.3), đo chất lượng câu trả lời (12.4), đặt cả hệ thống vào bối cảnh rủi ro (12.5). Thời lượng: khoảng 2 tuần, 8 giờ mỗi tuần. Chương dùng lại power iteration (chương 1), precision và tỉ lệ nền (chương 3), P/R/F1 (chương 5), 2N FLOPs mỗi token (chương 8), InfoNCE (chương 10), khung đọc paper và lỗi đánh giá (chương 11), recall@k (mục 9.5). Phần **mới**: định nghĩa tf-idf (lần đầu trong sách), BM25, hybrid và RRF, reranking, nDCG, chunking, phân rã lỗi RAG, PPR, Cohen κ, ECE, DP, bất khả thi của công bằng.

**Hạn dùng của chương.** Bài học nằm ở phương pháp và cách đo. Tên công cụ, vị trí mô hình trên bảng xếp hạng, điều khoản pháp lý và vụ kiện là **ví dụ có ngày tháng, tính đến cuối tháng 10/2026** (các mục điều khoản arXiv, EU AI Act và vụ *Bartz v. Anthropic* đã kiểm lại một phần ngày 05/11/2026), nên chương không nêu tên mô hình embedding hay reranker nào đang dẫn đầu. Mọi thí nghiệm trong chương chạy trên dữ liệu **tự dựng**, là đồ chơi để lộ cơ chế: chúng cho biết *vì sao* một phương pháp thắng hoặc thua trong thiết lập đó, không cho biết cỡ hiệu ứng trên dữ liệu thật và không xếp hạng phương pháp. Về nguồn: **không paper nào trong chương được đọc toàn văn**; mọi số "trong paper" là số trong abstract hoặc phần tóm tắt, ghi "chưa đối chiếu bản gốc" (xem bảng nguồn cuối chương). Phần luật và bản quyền chỉ nêu sự việc kèm mức xác minh, không phải tư vấn pháp lý: hãy tự kiểm tra nguồn gốc.

**Sau chương này, bạn phải làm được:**

- ☐ Tính BM25 và nDCG bằng tay, nói được khi nào BM25 thắng dense và ngược lại, và gộp hai bộ truy hồi bằng RRF mà không rơi vào bẫy danh sách rác.
- ☐ Chọn kích thước và độ chồng lấn chunk bằng phép đo, và phân rã lỗi RAG thành truy hồi (R), đọc đúng (a₁) và trả lời khi thiếu bằng chứng (a₀).
- ☐ Chạy Personalized PageRank bằng power iteration có khởi động lại, và đặt truy hồi lặp làm đối chứng bắt buộc khi so Graph-RAG với RAG thường.
- ☐ Đọc một con số "đồng thuận" của giám khảo LLM bằng κ, khoảng tin cậy và thiên lệch vị trí; đo hallucination ở mức tuyên bố, ECE và đường rủi ro–độ phủ.
- ☐ Nói được bằng một đoạn tiếng Anh, kèm một con số, về công bằng, riêng tư (ε), bản quyền và prompt injection gián tiếp, rồi viết Phiếu rủi ro một trang cho hệ thống của bạn.

**Kiểm tra nhanh (5 phút, không dùng máy tính).** Câu i kiểm tra mục 12.i: đúng thì đọc lướt mục đó, sai thì đọc kỹ.

1. Với k₁ = 1.2 và đoạn có độ dài trung bình, từ xuất hiện 100 lần được điểm tf gấp xấp xỉ bao nhiêu lần từ xuất hiện 10 lần (1.1, 10 hay 100)?
2. Chia tài liệu thành chunk 4 câu, không chồng lấn. Một sự thật gồm 2 câu liền nhau bị cắt ngang biên chunk với xác suất xấp xỉ bao nhiêu?
3. Trong đồ thị 7 nút của VD 12.7 (hạt giống X), đoạn P₂ cách hai bước nằm trước hay sau các đoạn một bước P₁, P₃? Điều đó có đúng trong mọi đồ thị không?
4. Hai người chấm đồng thuận 90%, nhưng cả hai đều chọn "A" 90% số lần. Cohen κ gần 0.9, gần 0.4 hay gần 0?
5. Hai nhóm có tỉ lệ nền khác nhau, cùng PPV và cùng tỉ lệ bỏ sót (FNR). Hai nhóm có thể có cùng tỉ lệ báo nhầm (FPR) không?

Đáp án ở mục Lời giải cuối chương.

## 12.1 Truy hồi: BM25, dense, hybrid và reranking

Cho truy vấn q và kho N đoạn, bộ truy hồi trả k đoạn. Muốn dựng một bộ truy hồi tốt phải trả lời bốn câu: điểm từ vựng tính thế nào (BM25), điểm ngữ nghĩa tính thế nào (dense), gộp hai thứ ra sao (RRF), và đọc kỹ hơn bằng gì (reranking). Chỉ số: recall@k và MRR (mục 9.5) cộng **nDCG@k** ở VD 12.5.

**tf-idf** cho từ t trong đoạn d điểm tf(t, d) · idf(t). Chương 5 chỉ nhắc TF-IDF một lần như baseline văn bản ("TF-IDF + logistic regression") mà không định nghĩa; chương 11 chỉ dùng `TfidfVectorizer` và nhắc lỗi rò rỉ IDF khi fit trên toàn bộ văn bản (không định nghĩa tf-idf); vì vậy **chương này định nghĩa nó lần đầu**. tf(t, d) là số lần t xuất hiện trong d. idf(t) = ln(N/n) ở dạng đơn giản nhất, với N là số đoạn và n là số đoạn chứa t: từ có mặt ở hầu hết đoạn ("the", "model") có idf gần 0 và bị phạt, từ hiếm có idf lớn, nên nó phân biệt đoạn tốt hơn. **BM25** sửa hai điểm yếu của tf-idf thô: tf tăng tuyến tính (lặp một từ 100 lần không nên đáng giá gấp 100 lần) và không xét độ dài đoạn. Điểm của đoạn d với truy vấn q là tổng trên các từ t của q:

$$
\mathrm{BM25}(q, d) = \sum_{t \in q} \mathrm{idf}(t)\cdot \frac{f\,(k_1 + 1)}{f + k_1\,(1 - b + b\,\cdot\,dl/avgdl)}, \qquad \mathrm{idf}(t) = \ln\!\Big(1 + \frac{N - n + 0.5}{n + 0.5}\Big)
$$

Ở đây f là số lần t xuất hiện trong d, dl là độ dài d, avgdl là độ dài trung bình, n là số đoạn chứa t. Đây là bản idf kiểu Lucene (luôn dương). Ba ý cần tự tính được. **Bão hòa tf:** phần tf tiến về k₁ + 1 khi f lớn (k₁ = 1.2: f = 1, 2, 5, 10, 50 cho 1.000, 1.375, 1.774, 1.964, 2.148; giới hạn 2.2), nên nhồi từ khóa không có lợi. **Chuẩn hóa độ dài** bằng b: b = 0 bỏ qua độ dài, b = 1 chuẩn hóa hoàn toàn. **idf** phạt từ phổ biến. Hai tham số k₁ ≈ 1.2 và b ≈ 0.75 là mặc định hay gặp (tôi nhớ chứ chưa mở tài liệu; trong lab dùng đúng hai số này và coi là quy ước). BM25 là bộ xếp hạng **từ vựng**: không có từ chung thì điểm bằng 0, nó không bao giờ "hiểu" đồng nghĩa.

% manim: BM25BaoHoa

**Ví dụ giải mẫu 12.1 — BM25 tay trên 4 đoạn.** d₁ "graph neural networks for citation graph classification" (7 từ), d₂ "retrieval augmented generation with dense retrieval" (6), d₃ "dense retrieval and sparse retrieval for question answering" (8), d₄ "graph retrieval augmented generation builds a knowledge graph" (8). N = 4, avgdl = 7.25, truy vấn "graph retrieval".

- df(graph) = 2 nên idf = ln(1 + 2.5/2.5) = ln 2 = **0.6931**; df(retrieval) = 3 nên idf = ln(1 + 1.5/3.5) = **0.3567**.
- d₁: f(graph) = 2, dl/avgdl = 7/7.25 = 0.9655, mẫu số hệ số độ dài 0.25 + 0.75·0.9655 = 0.9741, phần tf = 2·2.2/(2 + 1.2·0.9741) = 1.3885, điểm = 0.6931·1.3885 = **0.9624**. d₂ (retrieval hai lần): **0.5154**. d₃: **0.4766**. d₄ (graph hai lần 0.9261, retrieval một lần 0.3422): **1.2683**.
- Thứ hạng d₄ > d₁ > d₂ > d₃. d₁ **không có** từ "retrieval" mà vẫn thắng d₂ và d₃, hoàn toàn nhờ **idf**: idf(graph) = 0.6931 gấp 1.94 lần idf(retrieval) = 0.3567. Phần tf thì không phân biệt: d₁ có "graph" f = 2, d₂ và d₃ có "retrieval" f = 2, và phần tf sau chuẩn hóa độ dài là 1.389 (d₁), 1.445 (d₂), 1.336 (d₃), cùng cỡ (f = 2 còn xa mức bão hòa: 1.375 trên giới hạn 2.2 khi đoạn có độ dài trung bình). Chuẩn hóa độ dài chỉ xê dịch nhẹ (d₂ ngắn nhất nên phần tf cao nhất, d₃ dài nhất nên thấp nhất).

**Dense retrieval (bi-encoder).** Mã hóa q và từng đoạn thành vector, điểm là cosine. Huấn luyện bằng InfoNCE với âm tính trong batch (chương 10, VD 10.9). Ưu: bắt được đồng nghĩa và diễn đạt lại. Nhược: **định danh hiếm** (tên riêng, mã, số liệu: embedding không nhớ chuỗi hiếm), phụ thuộc phân phối huấn luyện, cần chỉ mục ANN khi N lớn (chỉ nêu tên). DPR (Karpukhin và cộng sự, 2020; abstract nói top-20 hơn BM25 9–19 điểm trên các bộ hỏi đáp; theo bảng do công cụ trích, **SQuAD là bộ BM25 thắng**, 68.8 so với 63.2; chưa đối chiếu bản gốc). BEIR (Thakur và cộng sự, 2021) gọi BM25 là baseline "robust" ngoài miền huấn luyện, còn re-ranking và late-interaction cho trung bình zero-shot tốt nhất nhưng đắt (theo abstract).

**Ví dụ giải mẫu 12.2 — BM25, bi-encoder và hybrid: mỗi bên thắng ở đâu (thí nghiệm đồ chơi).** Thế giới tự dựng: 60 chủ đề, mỗi chủ đề có 8 "từ chuẩn" và 8 "từ đồng nghĩa"; mỗi tài liệu có 5 từ chủ đề, 9 từ nhiễu và một **id hiếm** của riêng nó. Tập train 600 tài liệu (30% từ chủ đề thay bằng đồng nghĩa để mô hình học chúng gần nhau); tập test 400 tài liệu chỉ dùng từ chuẩn, id mới chưa từng thấy. Bi-encoder là `EmbeddingBag` trung bình 32 chiều, InfoNCE nhiệt độ 0.05, 400 bước. Trung bình 3 seed, mỗi loại 300 truy vấn (hit@1 / hit@10 / MRR@10):

| Loại truy vấn | BM25 | Bi-encoder | Hybrid (RRF, k = 60) |
| --- | --- | --- | --- |
| id hiếm + 1 từ chủ đề | 1.000 / 1.000 / 1.000 | 0.373 / 0.902 / 0.553 | 0.697 / 1.000 / 0.816 |
| paraphrase: 5 từ đồng nghĩa | 0.000 / 0.000 / 0.000 | 0.311 / 0.971 / 0.521 | 0.311 / 0.971 / 0.521 |
| từ khóa: 3 từ chuẩn | 0.589 / 1.000 / 0.771 | 0.456 / 0.998 / 0.660 | 0.551 / 1.000 / 0.741 |
| hỗn hợp 50% id + 50% paraphrase | 0.500 / 0.500 / 0.500 | 0.342 / 0.937 / 0.537 | 0.504 / **0.986** / 0.669 |

Cách đọc trung thực. (1) BM25 thắng tuyệt đối truy vấn định danh và không trả gì khi truy vấn chỉ có từ đồng nghĩa. (2) Trên hỗn hợp, hybrid hit@10 là 0.986, vượt cả hai bộ đơn, nhưng hit@1 chỉ 0.504, ngang BM25. (3) Trên truy vấn từ khóa hybrid **không thắng** BM25 (hit@1 0.551 so với 0.589). (4) **Bẫy cài đặt:** nếu bộ từ vựng trả **cả kho** theo thứ tự tùy ý khi không khớp từ nào, RRF trộn nhiễu vào: hit@10 của hybrid trên paraphrase tụt về 0.399 trong khi dense một mình là 0.971. Bộ từ vựng chỉ nên trả những đoạn có điểm lớn hơn 0. (5) **Chỗ hybrid thua nặng nhất là truy vấn id**: hit@1 0.697 so với 1.000 của BM25 (hụt 0.303, lớn hơn nhiều mức hụt 0.038 ở truy vấn từ khóa nêu ở (3)); hit@10 vẫn 1.000. Cơ chế hợp lý: dense chưa từng thấy id nên chỉ xếp đúng đoạn ở hạng 1 trong 37.3% số truy vấn, và RRF cho các đoạn mà dense xếp cao một điểm ngang đoạn BM25 xếp cao, nên đoạn đúng của BM25 bị đẩy xuống (tôi chưa tách riêng cơ chế này bằng thí nghiệm). Giới hạn: bi-encoder này là túi embedding đồ chơi; nó thua ở định danh chỉ vì id chưa từng được học, còn mô hình thật có token con nên đừng suy rộng số 0.373. Lab 2 chạy lại ở seed 0.

**Hybrid và RRF.** Gộp hai danh sách theo **hạng**, không theo điểm. Điểm BM25 không bị chặn, cosine nằm trong [−1, 1]; cộng điểm cần chuẩn hóa, và min–max rất nhạy với ngoại lệ. Reciprocal Rank Fusion (Cormack, Clarke, Büttcher, SIGIR 2009; đã đọc PDF) cho mỗi tài liệu điểm

$$
\mathrm{RRF}(d) = \sum_{r \in \text{bộ truy hồi}} \frac{1}{k + \mathrm{hạng}_r(d)}
$$

với k = 60, giá trị mà bài gốc ghi là cố định trong thử nghiệm sơ bộ. k lớn làm phẳng đóng góp của hạng đầu, k nhỏ để hạng 1 áp đảo.

% manim: RRFHopNhat

**Ví dụ giải mẫu 12.3 — RRF tay và vì sao dùng hạng.** BM25 trả (d3, d1, d7, d2, d9), dense trả (d1, d5, d3, d8, d2).

- k = 60: d1 = 1/62 + 1/61 = **0.03252**; d3 = 1/61 + 1/63 = **0.03227**; d2 = 1/64 + 1/65 = **0.03101**; d5 0.01613; d7 0.01587; d8 0.01562; d9 0.01538. Thứ tự d1, d3, d2, d5, d7, d8, d9.
- k = 10: d1 0.17424, d3 0.16783. k = 1: d1 0.8333, d3 0.7500, d2 0.3667: hạng đầu áp đảo.
- Cộng điểm thô thì sao? Điểm BM25 (d3 12.4, d1 8.1, d7 7.9, d2 3.0, d9 2.5) và cosine (d1 0.71, d5 0.69, d3 0.66, d8 0.52, d2 0.20); chuẩn hóa min–max rồi lấy trung bình cho d3 0.951, d1 0.783, d5 0.480. Thay điểm BM25 của d3 bằng ngoại lệ 40.0: d1 tụt từ 0.783 xuống 0.575 (d3 vẫn 0.951), tức thứ hạng của các tài liệu khác bị kéo theo một con số bất thường.

**Reranking.** Truy hồi rẻ lấy khoảng 100 ứng viên, rồi **cross-encoder** đọc cả cặp (q, đoạn) cùng nhau, chính xác hơn nhưng không tính trước được. Phép tính chi phí (mô hình 110·10⁶ tham số, forward ≈ 2N FLOPs mỗi token như chương 8, bỏ qua attention): bi-encoder mã hóa mỗi đoạn 256 token hết 5.63·10¹⁰ FLOPs **một lần** lúc lập chỉ mục; cross-encoder tốn 6.34·10¹⁰ cho **mỗi cặp** (truy vấn 32 + đoạn 256 token), nên chấm cả kho 10⁶ đoạn cho một truy vấn là 6.34·10¹⁶ còn chấm 100 ứng viên là 6.34·10¹². Đường ống hai tầng là hệ quả của phép tính, không phải thói quen. Một ví dụ chuỗi cải tiến có số, **của chính một hãng, trên dữ liệu của hãng**: Anthropic, "Contextual Retrieval" (blog 9/2024, đã đọc) báo tỉ lệ truy hồi hỏng ở top-20 giảm từ 5.7% xuống 3.7% (thêm ngữ cảnh vào chunk), 2.9% (cộng BM25 có ngữ cảnh), 1.9% (cộng reranking). Dùng nó làm mẫu quy trình (cải tiến từng bước và đo từng bước), không như hằng số.

**Vì sao ML cần cái này.**

- BM25 là baseline bắt buộc: bài nói "dense thắng" mà không có BM25 cùng ngân sách là thiếu đối chứng (chương 11).
- Truy vấn thật là hỗn hợp (tên phương pháp, mã số cho BM25; câu diễn đạt lại cho dense), nên hybrid là mặc định thực tế; phải báo **theo loại truy vấn**, vì VD 12.2 cho thấy hybrid thắng, hòa và thua tùy loại (thua nặng nhất ở truy vấn id: hit@1 0.697 so với 1.000 của BM25).

**Nối với dự án (luồng A).** Tuần này: dựng BM25 trên tiêu đề + abstract (một bản ghi một đoạn), đo recall@k và nDCG@k trên 50 câu hỏi **viết tay** (đừng để LLM sinh câu hỏi từ chính đoạn: từ vựng rò rỉ làm BM25 đẹp giả, xem mục 12.2); thêm một bi-encoder nhỏ; báo bảng BM25 / dense / hybrid theo loại câu hỏi (định danh, diễn đạt lại), kèm khoảng tin cậy (chương 11). Trước khi mã hóa toàn kho, hãy đo thông lượng thật trên RTX 4050 (ước lượng FLOP ở bài tính nhanh 13 chưa thay được phép đo).

## 12.2 Đường ống RAG: chunking, đánh giá và các kiểu lỗi

RAG gồm hai nửa: (tài liệu → chunk → chỉ mục) và (truy vấn → truy hồi → [rerank] → dựng prompt → sinh). Lỗi có thể nằm ở bất kỳ khâu nào, và đo mỗi khâu riêng mới chỉ ra được khâu hỏng. RAG gốc là Lewis và cộng sự (2020, mục 9.5); bản đồ thuật ngữ Naive, Advanced, Modular RAG ở khảo sát Gao và cộng sự (arXiv 2312.10997, chưa đối chiếu bản gốc).

**Chunking.** Chia tài liệu thành đoạn c câu, bước nhảy s câu, chồng lấn o = c − s. Đánh đổi: chunk nhỏ thì chính xác nhưng **cắt đứt sự thật trải qua nhiều câu**; chunk lớn thì đủ ngữ cảnh nhưng tốn token và dễ pha loãng. Với sự thật gồm m câu liền nhau và chồng lấn o, xác suất bị cắt ngang là max(0, m − 1 − o)/(c − o): chồng lấn từ m − 1 câu trở lên loại bỏ hẳn lỗi cắt.

% manim: ChunkBienGioi

**Ví dụ giải mẫu 12.4 — sự thật hai câu bị cắt ngang.** Tài liệu 64 câu × 12 token (từ nhiễu theo Zipf); mỗi tài liệu có một sự thật hai câu liền nhau: câu 1 chứa tên thực thể và từ thuộc tính, câu 2 (không nhắc lại tên) chứa đáp án. Truy vấn = [tên, thuộc tính], BM25 trên chunk, 100 tài liệu × 3 seed; "ans@top-k" là đáp án nằm trong k chunk đầu. Chunk c câu không chồng lấn; xác suất cắt đếm chính xác trên 63 vị trí bắt đầu:

| c | số chunk | xác suất cắt | cận trên kỳ vọng ans@top-1 (1 − cắt) | đo | token vào prompt (top-3) |
| --- | --- | --- | --- | --- | --- |
| 2 | 3200 | 0.492 | 0.508 | 0.470 | 72 |
| 4 | 1600 | 0.238 | 0.762 | 0.743 | 144 |
| 8 | 800 | 0.111 | 0.889 | 0.893 | 288 |
| 16 | 400 | 0.048 | 0.952 | 0.967 | 576 |

Cột "cận trên kỳ vọng" là giá trị trung bình khi vị trí sự thật được chọn ngẫu nhiên; số đo có thể vượt nó nhẹ (c = 8: 0.893 so với 0.889; c = 16: 0.967 so với 0.952) vì chỉ có 300 tài liệu (100 × 3 seed), nên vị trí thật rơi vào chỗ bị cắt ít hơn kỳ vọng: sai số chuẩn xấp xỉ √(p(1 − p)/300), cỡ 0.012 (c = 16) đến 0.018 (c = 8). Chênh dưới một sai số chuẩn không phải hiệu ứng.

Cùng thí nghiệm **có chồng lấn** (3 seed, hòa điểm phá ngẫu nhiên):

| c, bước nhảy | số chunk | ans@top-1 | ans@top-3 |
| --- | --- | --- | --- |
| 2, 1 (chồng lấn 1 câu) | 6300 | 0.483 | 1.000 |
| 4, 3 (chồng lấn 1 câu) | 2100 | 0.833 | 1.000 |
| 4, 2 (chồng lấn 50%) | 3100 | 0.763 | 1.000 |
| 8, 7 (chồng lấn 1 câu) | 900 | 0.927 | 1.000 |
| 8, 4 (chồng lấn 50%) | 1500 | 0.897 | 1.000 |
| 16, 15 (chồng lấn 1 câu) | 500 | 0.967 | 1.000 |

Chồng lấn từ 1 câu trở lên cho **ans@top-3 = 1.000 với mọi c ≥ 2**, đúng như công thức (xác suất cắt bằng 0), và **không làm ans@top-1 tệ hơn** không chồng lấn (c = 4: 0.833 so với 0.743; c = 8: 0.927 so với 0.893; c = 16 ngang nhau 0.967). Ở c = 2 bước 1, ans@top-1 chỉ 0.483, ngang không chồng lấn (0.470), vì đây là tình huống **hòa điểm**: chunk (câu pos − 1, pos) và chunk (câu pos, pos + 1) đều chứa tên và thuộc tính, cùng độ dài, nên BM25 cho điểm bằng nhau tuyệt đối, và chỉ chunk sau chứa đáp án. Ai đứng đầu do **quy tắc phá hòa** quyết định, nên xấp xỉ tung đồng xu. **Cảnh báo về phép đo:** nếu phá hòa theo chỉ số nhỏ (cách mặc định của nhiều hàm sắp xếp), chunk đứng trước luôn là chunk chưa chứa đáp án, và bạn đo được ans@top-1 = 0.013 (c = 2) và 0.677 (c = 4, bước 3) rồi kết luận nhầm rằng chồng lấn gây hại; đó là hiện vật của quy tắc phá hòa, không phải hiệu ứng của chunking (phá hòa theo chỉ số lớn sẽ cho số cao bất thường). Khi đầu danh sách có hòa điểm, hãy phá hòa ngẫu nhiên, chạy nhiều seed, hoặc bỏ chỉ số @1. Cái giá thật của chồng lấn là chi phí: số chunk gấp c/bước lần (c = 2: 6300 so với 3200, gần gấp đôi; c = 4 bước 3: 1.31 lần), chunk kề nhau chia sẻ câu (c = 2: một nửa số câu) nên top-k chứa token trùng; cần khử trùng lặp (MMR, chỉ nêu tên). Số của lab (seed 0, 100 tài liệu, chồng lấn 1 câu) nằm ở bảng Lab: ví dụ c = 4, bước 3 cho ans@top-1 = 0.830, ans@top-3 = 1.000. Giới hạn: thí nghiệm không mô phỏng hiệu ứng pha loãng của chunk lớn; chỉ đo cắt ngang và chi phí token. Chunk theo cấu trúc (mục, đoạn, câu) thường tốt hơn chia cứng theo token, chưa đo ở đây.

**Vị trí trong ngữ cảnh.** Liu và cộng sự (2023, "Lost in the Middle", arXiv 2307.03172; chưa đối chiếu bản gốc): theo abstract, hiệu năng tốt nhất khi thông tin liên quan ở đầu hoặc cuối ngữ cảnh, giảm khi ở giữa, kể cả mô hình ngữ cảnh dài. Các mô hình trong paper là của 2023, nên hiện tượng phải **đo lại** với mô hình bạn dùng: hoán vị vị trí đoạn đúng trong ngữ cảnh và vẽ độ chính xác theo vị trí (chưa chạy trong chương vì lab CPU không có mô hình sinh thật). Hệ quả thiết kế: giảm k, rerank, đặt đoạn điểm cao ở đầu và cuối.

**Đánh giá RAG gồm ba tầng, phải đo riêng.** (1) *Truy hồi:* recall@k, MRR, nDCG@k theo nhãn đoạn vàng. (2) *Trung thành* (faithfulness): mỗi tuyên bố trong câu trả lời có được đoạn truy hồi ủng hộ không. (3) *Đúng* (correctness) so với đáp án vàng. Ba thứ này khác nhau: câu trả lời có thể đúng mà không trung thành (mô hình dùng trí nhớ tham số), hoặc trung thành mà sai (đoạn sai được truy hồi). RAGAS (Es và cộng sự, arXiv 2309.15217) là bộ chỉ số tự động không cần nhãn người; các chỉ số do LLM chấm kế thừa thiên lệch của giám khảo (mục 12.4). Tên cụ thể ba chỉ số của RAGAS tôi chỉ nhớ, chưa kiểm.

**Ví dụ giải mẫu 12.5 — nDCG tay và ba chỉ số trả lời ba câu hỏi khác nhau.** Ba đoạn liên quan với độ liên quan có mức: A = 3, B = 2, C = 1, mọi đoạn khác 0. Hệ 1 trả (D, A, E, B, C), hệ 2 trả (A, B, D, E, C). Độ lợi 2ᵍ − 1, chiết khấu 1/log₂(hạng + 1) = 1.000, 0.631, 0.500, 0.431, 0.387.

- Hệ 1: DCG = 7/1.585 + 3/2.322 + 1/2.585 = 4.4165 + 1.2920 + 0.3869 = **6.0954**. Lý tưởng (A, B, C): 7 + 1.8928 + 0.5 = **9.3928**. nDCG@5 = **0.6489**.
- Hệ 2: DCG = 7 + 1.8928 + 0.3869 = 9.2796, nDCG@5 = **0.9880**.
- recall@3: hệ 1 là 1/3 (chỉ A), hệ 2 là 2/3 (A, B). MRR theo đoạn liên quan đầu tiên: 0.5 và 1.0.
- Ba chỉ số, ba câu hỏi: "có lấy được không" (recall), "đoạn đúng đầu tiên ở đâu" (MRR), "cả thứ hạng tốt không, theo độ quan trọng" (nDCG). Dùng độ lợi tuyến tính g thay 2ᵍ − 1 cho 0.6596 và 0.9762: khác nhau ít ở ví dụ này, nhiều hơn khi mức liên quan lớn.

**Phân rã lỗi.** Gọi R là xác suất đoạn vàng nằm trong ngữ cảnh, a₁ độ chính xác khi có, a₀ khi không có (từ trí nhớ tham số hoặc đoán):

$$
\mathrm{Acc} = R\,a_1 + (1 - R)\,a_0, \qquad \frac{\partial\, \mathrm{Acc}}{\partial R} = a_1 - a_0, \qquad \frac{\partial\, \mathrm{Acc}}{\partial a_1} = R
$$

% manim: PhanRaLoiRAG

**Ví dụ giải mẫu 12.6 — phân rã lỗi RAG.** 200 câu hỏi; đoạn vàng nằm trong ngữ cảnh ở 140 câu (R = 0.70); trong 140 câu đó đúng 119 (a₁ = 0.85); trong 60 câu còn lại đúng 12 (a₀ = 0.20). Acc = 0.70·0.85 + 0.30·0.20 = 0.595 + 0.060 = **0.655** (131/200).

- Kịch bản: R lên 0.85 (a₁, a₀ giữ nguyên) cho **0.7525**; a₁ lên 0.92 cho **0.704**; cả hai cho 0.812.
- Độ nhạy: ∂Acc/∂R = a₁ − a₀ = 0.65 và ∂Acc/∂a₁ = R = 0.70: cùng cỡ, nên chọn tầng nào để cải thiện phụ thuộc **chi phí và độ khó**, không phụ thuộc công thức.
- 12 trong 131 câu đúng là **đúng mà không có đoạn vàng làm cơ sở** (đoạn vàng không nằm trong ngữ cảnh). Giả định để gọi chúng là "đúng không trung thành": đáp án đến từ bộ nhớ tham số. Thực tế đây chỉ là cận trên, vì câu cũng có thể đúng nhờ một đoạn khác được truy hồi mà chứa đáp án, nên phải kiểm từng câu. Chỉ đo accuracy cuối thì không thấy nhóm này. Đây là số tay minh họa; bạn làm thật trên 30 câu của dự án.

**Các lỗi im lặng thường gặp** (mỗi lỗi có triệu chứng đo được; bài săn bug của chương gài năm lỗi loại này): chuẩn hóa văn bản khác nhau giữa lúc lập chỉ mục và lúc truy vấn; cắt ngữ cảnh khi dựng prompt; vòng lặp chunk bỏ sót phần cuối; nhãn vàng hoặc đoạn vàng lọt vào ngữ cảnh; thứ hạng bị đảo ở bước hợp nhất; metric tính ở mức tài liệu thay vì mức đoạn chứa đáp án. Hai phép thử dùng được mọi nơi: **tắt thành phần thì chỉ số phải tụt** (ablation phải đau), và **đo ở nơi dữ liệu thực sự đi vào mô hình**.

**Rò rỉ riêng của RAG.** Các lỗi đánh giá chung đã dạy ở chương 11 (VD 11.8–11.10). Riêng với RAG có hai loại: câu hỏi do LLM sinh từ chính đoạn thì trùng từ vựng với đoạn, nên BM25 đẹp giả; và kho chứa tài liệu có câu trả lời của bộ test mà LLM đã thấy lúc tiền huấn luyện (nhiễm kép).

**Vì sao ML cần cái này.**

- "RAG trả lời sai" là ba câu hỏi (truy hồi có lấy được không, ngữ cảnh có đủ không, mô hình có đọc đúng không), mỗi câu cần số riêng.
- Công thức phân rã cho **đòn bẩy** (cải thiện R hay a₁ có giá bao nhiêu điểm) nhưng không nói cái nào rẻ hơn; chọn chunk cũng là quyết định đo được bằng thí nghiệm nhỏ.

**Nối với dự án.** Với 30 câu hỏi, ghi bảng (đoạn vàng có trong top-k không; trả lời đúng không; trả lời có trung thành không) rồi tính R, a₁, a₀ và báo phân rã; chọn **một** tầng để cải thiện theo đòn bẩy. Thử chunk theo câu với c = 2, 4, 8 và chồng lấn 1 câu; đo ngân sách token; khi báo ans@top-1, phá hòa điểm ngẫu nhiên và nói rõ quy tắc.

## 12.3 Trích xuất tri thức, đồ thị tri thức và Graph-RAG

Đồ thị chỉ giúp khi câu hỏi cần **đi qua nhiều đoạn**, và nó có giá: chi phí lập chỉ mục và rủi ro trích xuất sai. Vì vậy mục này dạy ba thứ theo thứ tự: trích xuất và cách đo nó, PPR như cơ chế lan truyền, rồi bằng chứng về khi nào đồ thị không thắng.

**Trích xuất bộ ba (chủ thể, quan hệ, đối tượng)** gồm ba bước: nhận diện thực thể (NER), **chuẩn hóa/liên kết thực thể** ("BERT", "Bert", "BERT-base" là một hay ba nút?), trích quan hệ. Hiện nay bước 1 và 3 thường do LLM làm bằng prompt có schema đầu ra JSON; bước 2 vẫn là điểm yếu. Đánh giá bằng cách so tập bộ ba dự đoán với tập vàng: precision, recall, F1 (chương 5). Khớp **chặt** (từng ký tự) và khớp **sau chuẩn hóa** cho số rất khác nhau: với tập vàng 8 bộ ba mà hệ thống trả 7, trong đó 5 bộ đúng nội dung nhưng 3 bộ sai chữ hoa/thường, F1 chặt là 0.267 (P = 2/7, R = 2/8) còn F1 sau chuẩn hóa chữ thường là 0.667 (P = 5/7, R = 5/8): chuẩn hóa thay F1 gấp 2.5 lần mà không đụng tới mô hình. Chữ thường là entity resolution **tầm thường**; viết tắt và biệt danh khó hơn nhiều. Luôn báo cả hai và nói quy ước. Lỗi cũng **nhân** theo số bước: câu hỏi k bước cần k cạnh đúng, xác suất ≈ pᵏ nếu các cạnh độc lập (0.9³ = 0.729), nhưng lỗi thực thể thường tương quan dương (một thực thể bị gọi sai tên làm hỏng nhiều cạnh cùng lúc), và khi đó pᵏ là ước lượng **bi quan**: xác suất thật nằm trong [0.729, 0.9]; chỉ tương quan âm mới kéo xuống dưới 0.729, tới cận 0.7 (bài tập 14).

**Đồ thị cho ArXiv và giới hạn dữ liệu.** Metadata ArXiv có tác giả, thể loại (categories), journal-ref, doi, abstract, phiên bản. Trên bản mirror Hugging Face của metadata (thẻ dữ liệu `librarian-bots/arxiv-metadata-snapshot`, 3 224 635 bản ghi ở lần đọc đầu, khoảng 3.16 triệu hàng ở lần kiểm lại ngày 4/11/2026; số này đổi theo thời điểm nên chỉ lấy cỡ 3 triệu) **không có trường trích dẫn**, nên **đồ thị trích dẫn của dự án phải lấy từ nguồn khác** (từ toàn văn hoặc một kho khác; tôi chưa xác minh nguồn nào, việc đó bạn phải kiểm). Cạnh tác giả–bài và thể loại–bài có sẵn; cạnh phương pháp, dataset, nhiệm vụ phải trích từ abstract. Về điều khoản: theo trang "Bulk Data Access" của arXiv (đã đọc), metadata lấy qua OAI-PMH, toàn văn qua Kaggle hoặc Amazon S3; **phần lớn bài dùng giấy phép mặc định của arXiv, nên arXiv "không thể cấp quyền phân phối lại" toàn văn**, một phần nhỏ bài có giấy phép khác ghi trong metadata; đừng tải cả kho bằng chương trình qua API (dùng S3). Về tần suất có **hai trang khác nhau, đừng gộp**. (1) Trang "Terms of Use for arXiv APIs" (đọc ngày 05/11/2026) quy định cho các API cũ, gồm OAI-PMH, RSS và arXiv API: **tối đa 1 yêu cầu mỗi 3 giây và 1 kết nối tại một thời điểm**, tính chung cho mọi máy bạn điều khiển; trang ghi các giới hạn này có thể đổi. Cùng trang: metadata mô tả của bài (không phải toàn văn) dùng được theo CC0 1.0. Dự án lấy metadata qua OAI-PMH hoặc arXiv API thì theo giới hạn này. (2) Trang "Bulk Data Access" có gợi ý riêng về cụm 4 yêu cầu mỗi giây rồi nghỉ 1 giây cho tải hàng loạt; đó không phải mức được phép cho API cũ ở (1). Tôi có thấy kết quả tìm kiếm nhắc chính sách giới hạn tần suất "cập nhật 10/2026" từ một trang không chính thức mà tôi không mở được [cần kiểm chứng]. Hãy tự đọc lại cả hai trang trước khi tải: điều khoản có thể đổi. Cách làm an toàn: lưu id, metadata và abstract, link về arXiv, không đẩy toàn văn lên repo công khai.

**Personalized PageRank (PPR).** Bước đi ngẫu nhiên trên đồ thị, mỗi bước nhảy về nút hạt giống s với xác suất α. Phân phối dừng x thỏa

$$
x \leftarrow \alpha\, s + (1 - \alpha)\, T x \qquad\Longrightarrow\qquad x = \alpha\,\big(I - (1 - \alpha) T\big)^{-1} s
$$

với T là ma trận chuyển chuẩn hóa theo cột. Đây chính là power iteration của chương 1 cộng số hạng khởi động lại. Điểm của đoạn văn là khối lượng x trên nút đoạn. HippoRAG (Gutiérrez và cộng sự, arXiv 2405.14831) dùng LLM trích đồ thị tri thức rồi chạy PPR từ các thực thể trong truy vấn; theo abstract (mở lại ngày 05/11/2026: "rẻ hơn 10–30 lần và nhanh hơn 6–13 lần"), truy hồi một bước ngang hoặc hơn IRCoT với chi phí thấp hơn rất nhiều (chưa đối chiếu bản gốc; PPR ở đây không nhất thiết giống HippoRAG).

% manim: PPRLanToa

**Ví dụ giải mẫu 12.7 — PPR tay, rồi thí nghiệm hai bước.**

Quy ước đếm "bước": đoạn **một bước** là đoạn nối thẳng với thực thể hạt giống; đoạn **hai bước** là đoạn phải đi thêm qua một thực thể trung gian (đoạn, thực thể, đoạn), tức cách hạt giống 3 cạnh trên đồ thị thực thể–đoạn. Bài tập 9 thì đếm khoảng cách theo cạnh.

(a) *Tay.* Thực thể X, Y, Z, D; đoạn P₁ nhắc X, Y ("X mở rộng Y"), P₂ nhắc Y, D ("Y được đánh giá trên D"), P₃ nhắc X, Z. Cạnh vô hướng thực thể–đoạn: X–P₁, Y–P₁, Y–P₂, D–P₂, X–P₃, Z–P₃. Hạt giống X, α = 0.5.

- Lặp 1: X 0.5, P₁ 0.25, P₃ 0.25. Lặp 2: X 0.625, Y 0.0625, Z 0.0625, P₁ 0.125, P₃ 0.125, P₂ và D vẫn 0. Đồ thị hai phía nên giá trị dao động trước khi hội tụ.
- Hội tụ (nghiệm đóng): X **0.5803**, Y 0.0419, Z 0.0415, D 0.0030, P₁ **0.1556**, P₂ **0.0120**, P₃ **0.1658**, tổng 1.
- Xếp hạng đoạn P₃ > P₁ > P₂. **Đoạn đích hai bước (P₂) xếp cuối**, sau cả đoạn nhiễu một bước P₃, nhưng có khối lượng dương (0.012) trong khi đoạn không nối thông có đúng 0. α nhỏ cho khối lượng đi xa hơn (α = 0.15: P₃ 0.2098, P₁ 0.1791, P₂ 0.0706).
- **Đây là tính chất của đồ thị này, không phải luật.** Đoạn xa nhận khối lượng qua nhiều đường ngắn có thể thắng đoạn lá một bước. Phản ví dụ: hạt giống X nối 10 đoạn lá P₁…P₁₀; ba thực thể Y₁, Y₂, Y₃ mỗi cái nối một đoạn trong P₁…P₃ và cùng nối đoạn Q (Q cách X hai bước, qua ba đường). Với α = 0.5, P₄ (lá, một bước) 0.0319 hơn Q 0.0074; với α = 0.15, Q = 0.0439 vượt P₄ = 0.0372. Lab 4 kiểm cả hai. Nói cho đúng: PPR **thường** xếp đoạn xa sau đoạn gần, thứ hạng phụ thuộc α và hình dạng đồ thị.

(b) *Thí nghiệm đồ chơi* (3 seed × 200 câu). 80 phương pháp (10 gốc, mỗi phương pháp còn lại mở rộng một phương pháp trước đó), 12 dataset; mỗi phương pháp có đoạn giới thiệu ("paper m giới thiệu m, mở rộng m cha, dùng kỹ thuật t") và đoạn đánh giá ("m được đánh giá trên dataset d"). Câu hỏi: "dataset nào được dùng để đánh giá phương pháp mà m x mở rộng?". Bằng chứng vàng là đoạn giới thiệu của x và **đoạn đánh giá của cha y** (không chứa x). Đồ thị thực thể–đoạn; hòa điểm phá **ngẫu nhiên**; trung bình 3 seed:

| Phương pháp | đích trong top-5 | recall bằng chứng@5 | đích trong top-10 | recall bằng chứng@10 |
| --- | --- | --- | --- | --- |
| BM25 | 0.000 | 0.497 | 0.000 | 0.500 |
| Truy hồi lặp 2 vòng (thực thể mới từ top-2, rồi truy vấn lại) | 0.685 | 0.704 | 0.737 | 0.730 |
| PPR (hạt giống: thực thể trong truy vấn), α = 0.5 | 0.477 | 0.738 | **0.892** | 0.946 |
| RRF(BM25, PPR) | 0.218 | 0.608 | 0.595 | 0.797 |

Giữa các seed, ở k = 5 truy hồi lặp dao động 0.650–0.715 và PPR 0.375–0.530; ở k = 10 lặp 0.720–0.760 và PPR 0.840–0.950. Ở k = 20 PPR đạt 1.000 còn lặp dừng ở 0.740 (lặp chỉ mở rộng từ hai đoạn đầu của vòng 1, nên nếu vòng 1 xếp sai thì tăng k không cứu được).

- **Chiều của kết quả đảo theo k.** k = 5: lặp hơn PPR (0.685 so với 0.477), nhưng lặp **không gần 1**. k = 10: PPR hơn lặp (0.892 so với 0.737).
- **Mất cạnh do trích xuất sót** (giữ 100%, 90%, 80%, 70% liên kết thực thể–đoạn): PPR@5 là 0.477, 0.398, 0.340, 0.258; PPR@10 là 0.892, 0.685, 0.507, 0.337; truy hồi lặp 0.685 (k = 5) và 0.737 (k = 10) ở mọi mức vì không dùng đồ thị. Tức là lợi thế của PPR ở k = 10 chỉ có khi đồ thị đủ sạch: mất 10% liên kết (0.685) là đã dưới lặp (0.737). Lưu ý cách mô phỏng: cạnh bị mất được lấy mẫu lại cho **từng câu hỏi**, trong khi lỗi trích xuất thật cố định theo kho (cùng một cạnh sót cho mọi câu). Cách của Lab 4 làm phương sai giữa các câu mượt hơn thực tế; dựng đồ thị mất cạnh một lần cho mỗi seed sẽ sát thực hơn.
- **Kết quả nhạy với quy tắc phá hòa, và cơ chế thật của nó.** Ở vòng 1, top-1 **luôn** là đoạn đánh giá của chính x (nó chứa mx, evaluated, dataset, for; không hòa với ai; 200/200 câu ở seed 0). Hòa xảy ra ở **hạng 2**, giữa đoạn giới thiệu của x ("paper x introduces mx which extends my ...") và đoạn giới thiệu của các **con** của x ("paper c introduces mc which extends mx ..."): cả hai chứa đúng một lần "mx" và một lần "extends" và dài bằng nhau, nên điểm BM25 bằng nhau tuyệt đối. Ở seed 0, 99 trong 200 câu có hòa như vậy (x có từ 1 đến 3 con); 101 câu còn lại x không có con nên đoạn giới thiệu của x vào hạng 2 mà không phải may rủi. Phá hòa ngẫu nhiên đưa đoạn giới thiệu của x vào top-2 ở 71% số câu (seed 0; 71%, 73%, 81% ở ba seed, trung bình khoảng 75%); phá hòa theo chỉ số nhỏ cho 200/200. Lý do: dữ liệu được sinh **cha trước con** (phương pháp con mở rộng một phương pháp có chỉ số nhỏ hơn), nên chỉ số nhỏ luôn chọn x thay vì con của x. Thứ tự chỉ mục đã mang theo **hướng của quan hệ "mở rộng"**, thứ mà nội dung văn bản không chứa: BM25 túi từ không phân biệt "x mở rộng y" với "c mở rộng x", chỉ thứ tự trong kho phân biệt được. Kết quả: lặp "đạt" 0.963 (k = 5) và 1.000 (k = 10), RRF 0.350 và 0.880. PPR không đổi theo quy tắc phá hòa trong thí nghiệm này (đã đo; còn đầu danh sách PPR có hòa điểm hay không thì tôi chưa kiểm). Đây là rò rỉ thông tin qua cách đánh số chỉ mục (cùng họ với rò rỉ dữ liệu ở chương 11). Phép thử đa năng: **đổi quy tắc phá hòa (ngẫu nhiên, đảo ngược thứ tự chỉ mục) và chạy nhiều seed**; nếu số đổi nhiều thì so sánh chưa đáng tin.

Kết luận **đúng cho thí nghiệm này, không suy rộng**: bài toán chỉ có một kiểu câu hỏi, đồ thị sạch, thực thể khớp chuỗi. Điều nên mang đi là phương pháp: đồ thị cho đường đi nhiều bước nhưng trả bằng k, bằng chất lượng trích xuất, và **truy hồi lặp là đối chứng bắt buộc, rẻ**; không bên nào "thắng áp đảo", nên luôn báo kèm k, nhiều seed và quy tắc phá hòa. Lab 4 chạy lại ở seed 0 và in cả hai quy tắc phá hòa.

**GraphRAG của Microsoft** (Edge và cộng sự, arXiv 2404.16130; chưa đối chiếu bản gốc): LLM trích đồ thị thực thể từ cả kho, phân cụm cộng đồng, **tóm tắt trước** từng cộng đồng; lúc truy vấn mỗi tóm tắt cho câu trả lời từng phần rồi gộp lại. Mục tiêu là câu hỏi **toàn cục** ("các chủ đề chính của kho là gì"), nơi RAG thường không đủ; abstract nói cải thiện tính toàn diện và đa dạng ở cỡ 1 triệu token. Chi phí chính nằm ở lập chỉ mục (nhiều lần gọi LLM); blog "Moving to GraphRAG 1.0" (12/2024, đã đọc) mô tả LazyGraphRAG là hướng hoãn dùng LLM tới lúc truy vấn.

**Khi nào đồ thị không thắng (bằng chứng ngược, theo abstract, chưa đối chiếu bản gốc).** Han và cộng sự (arXiv 2502.11371): RAG thường và GraphRAG mỗi bên mạnh ở tác vụ khác nhau, chiến lược chọn hoặc hợp nhất cho cải thiện ổn định. Xiang và cộng sự (GraphRAG-Bench, arXiv 2506.05690): GraphRAG thường kém RAG thường trên nhiều tác vụ thực tế. HippoRAG 2 (arXiv 2502.14802): các biến thể RAG có cấu trúc kém RAG chuẩn rõ rệt trên tác vụ ghi nhớ sự thật cơ bản, và đề xuất cách khắc phục. Cũng có bằng chứng thuận (GraphRAG cho câu hỏi toàn cục, HippoRAG cho đa bước). Chương giữ cả hai phía: "graph tốt hơn" là giả thuyết phải kiểm theo từng loại câu hỏi, không phải mặc định.

**Vì sao ML cần cái này.**

- Lỗi trích xuất nhân lên theo số bước, nên đánh giá trích xuất là điều kiện cần trước khi đánh giá Graph-RAG.
- PPR là power iteration của chương 1 áp lên truy hồi; so Graph-RAG với RAG thường mà thiếu đối chứng cùng ngân sách (k, token, số lần gọi LLM) là lỗi chương 11 áp lên một mảng đang đổi nhanh.

**Nối với dự án (luồng A).** Giai đoạn 1: chọn schema nhỏ (Paper, Author, Category, Method, Dataset, Task; cạnh authored_by, in_category, introduces, extends, evaluated_on), trích bằng quy tắc + LLM trên 200 abstract, đánh giá bằng 50 bộ ba gắn nhãn tay, báo P/R/F1 chặt và sau chuẩn hóa. Giai đoạn 2: so bốn cấu hình trên cùng bộ câu hỏi chia theo loại (một bước, nhiều bước, toàn cục): BM25, dense, hybrid + rerank, hybrid + đồ thị (PPR hoặc mở rộng láng giềng) **và** truy hồi lặp làm đối chứng; báo chi phí (số lần gọi LLM lúc lập chỉ mục, token, thời gian). Ngân sách số học thuần: trích 10 000 abstract (250 token) bằng LLM với prompt 600 token là 8.5 triệu token vào và khoảng 1.5 triệu token ra nếu mỗi lần ra 150 token; giá và tốc độ tùy dịch vụ, bạn tự đo. Chưa biết ngân sách API thì dùng một tập con (một thể loại, vài nghìn bài) cho mọi thí nghiệm có LLM.

## 12.4 Đánh giá NLP/LLM và hallucination

Với câu trả lời mở, mỗi cách đo có một điểm mù: khớp chuỗi (EM, F1 mức token) phạt diễn đạt khác; chồng n-gram (BLEU, ROUGE) không đo sự thật; giám khảo LLM rẻ nhưng có thiên lệch; người đắt và cũng không đồng nhất. Mục này dạy cách **đọc** các con số đó cho trung thực.

**Độ đồng thuận giữa người chấm.** Tỉ lệ đồng thuận thô p₀ bị thổi phồng bởi chuyện ngẫu nhiên. Cohen κ trừ phần đó đi, với pₑ tính từ phân phối biên của hai người chấm:

$$
\kappa = \frac{p_0 - p_e}{1 - p_e}, \qquad p_e = \sum_{c} P_1(c)\, P_2(c)
$$

Zheng và cộng sự (2023, arXiv 2306.05685; chưa đối chiếu bản gốc) báo rằng giám khảo mạnh đạt hơn 80% đồng thuận với người, ngang đồng thuận giữa người với người, và nêu thiên lệch vị trí, độ dài, tự ưu ái. Số 80% đo trên MT-bench và Chatbot Arena năm 2023 với mô hình 2023; VD 12.8 dạy cách đọc một con số như vậy, không để bác nó.

**Ví dụ giải mẫu 12.8 — đồng thuận người–LLM: κ, khoảng tin cậy, thiên lệch vị trí.** 100 cặp câu trả lời (A hay B tốt hơn): cùng chọn A 45, người A giám khảo B 10, người B giám khảo A 12, cùng chọn B 33.

- p₀ = (45 + 33)/100 = 0.78. Người chọn A 0.55, giám khảo chọn A 0.57, nên pₑ = 0.55·0.57 + 0.45·0.43 = 0.3135 + 0.1935 = **0.507** và κ = (0.78 − 0.507)/(1 − 0.507) = **0.554**. Khoảng Wald 95% cho p₀ (chương 3): ±1.96·√(0.78·0.22/100) = **±0.081**, tức 0.70 đến 0.86; với n = 100, "đồng thuận 80%" nằm lẫn trong khoảng đó.
- Đồng thuận cao mà thông tin thấp: p₀ = 0.90 nhưng cả hai chọn A 90% thì pₑ = 0.82 và κ = **0.444**.
- Thiên lệch vị trí: chấm 100 cặp hai lần (đổi chỗ). Nhất quán 70, không nhất quán 30, trong đó 24 lần chọn vị trí đầu: 24/30 = 0.80 trong các cặp không nhất quán, 24% toàn bộ cặp. Biện pháp: hoán vị, chỉ tính khi nhất quán, chuẩn hóa độ dài. Dữ liệu là ví dụ tay, không đo mô hình thật.

**Hallucination.** Phân biệt (a) **không trung thành với ngữ cảnh** (RAG: nói điều đoạn truy hồi không có hoặc mâu thuẫn) và (b) **sai sự thật** so với thế giới. Đo ở mức tuyên bố: tách câu trả lời thành các tuyên bố nguyên tử, gắn nhãn ủng hộ, mâu thuẫn, hoặc không kiểm chứng được. Phát hiện không cần kho tham chiếu: SelfCheckGPT (Manakul và cộng sự, arXiv 2303.08896) lấy nhiều mẫu và coi mâu thuẫn giữa các mẫu là tín hiệu bịa (chỉ nêu ý, không hứa con số). Vì sao mô hình bịa: Kalai, Nachum, Vempala, Zhang (arXiv 2509.04664, 9/2025) lập luận rằng huấn luyện và đánh giá nhị phân đúng/sai thưởng cho việc đoán thay vì thừa nhận không chắc, và đề xuất đổi cách chấm. Đây là **lập luận lý thuyết kèm đề xuất**, chưa phải kết quả thực nghiệm đã tái lập, và tôi chưa kiểm phản hồi của cộng đồng.

**Ví dụ giải mẫu 12.9 — trung thành, ECE, rủi ro–độ phủ và thang chấm thưởng đoán.**

(a) Câu trả lời có 5 tuyên bố: 3 được ủng hộ, 1 mâu thuẫn, 1 không kiểm chứng được, nên trung thành = 3/5 = **0.6**; tỉ lệ bịa là 0.4 nếu coi "không kiểm chứng" là bịa (nói rõ quy ước).

(b) ECE (expected calibration error) là trung bình có trọng số của chênh giữa độ chính xác và độ tự tin trong từng bin. 10 dự đoán, độ tự tin (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.55, 0.50, 0.30), đúng/sai (1, 1, 1, 0, 1, 0, 1, 0, 0, 0). Ba bin: [0.75, 1] có 5 mẫu, accuracy 0.80, tự tin 0.85; [0.5, 0.75) có 4 mẫu, accuracy 0.25, tự tin 0.5875; [0, 0.5) có 1 mẫu, accuracy 0, tự tin 0.30. ECE = 0.5·0.05 + 0.4·0.3375 + 0.1·0.30 = 0.025 + 0.135 + 0.03 = **0.19**. Mẫu nhỏ, và ECE phụ thuộc cách chia bin.

(c) Rủi ro–độ phủ: xếp theo độ tự tin, chỉ trả lời phần đầu. Độ phủ 0.3 cho accuracy 1.000; 0.5 cho 0.800; 0.7 cho 0.714; 1.0 cho 0.500. Đây là cách trả lời "RAG của bạn nên từ chối khi nào" bằng số.

(d) Chấm +1 nếu đúng, 0 nếu bỏ qua, −λ nếu sai. Kỳ vọng khi đoán với xác suất đúng p là p − λ(1 − p), dương khi p > λ/(1 + λ). λ = 0 (chấm chỉ đúng/sai): ngưỡng 0, **luôn nên đoán**. λ = 1: 0.5. λ = 3: 0.75. Với p = 0.3, kỳ vọng khi đoán là 0.3, −0.4, −1.8 cho λ = 0, 1, 3. Phép tính này là của chúng ta, không khẳng định nó nằm trong paper: nó chỉ minh họa ý "hệ thống chỉ biết bỏ qua nếu hàm chấm điểm cho phép".

**Vì sao ML cần cái này.**

- Một con số đồng thuận phải đi kèm cỡ mẫu, phân phối biên và κ. Giám khảo LLM là công cụ đo có sai số hệ thống: hoán vị vị trí, không dùng cùng một họ mô hình để sinh nhãn và để chấm.
- Hallucination vừa là vấn đề của mô hình vừa của thang đo: cho phép "không biết" và đo bằng đường rủi ro–độ phủ.

**Nối với dự án.** Trên 30 câu: tách tuyên bố, tính trung thành; cho RAG từ chối khi điểm truy hồi thấp và đo "tỉ lệ trả lời khi không có đoạn vàng" (a₀ ở VD 12.6: bịa hay từ chối?); luôn trả trích dẫn kèm câu trả lời. Lab CPU của chương không có mô hình sinh thật, nên VD 12.9 dùng dữ liệu tay và bạn làm thật ở luồng A. Pass@k và nhiễm benchmark: xem mục 9.5 và chương 11.

## 12.5 An toàn, thiên lệch, riêng tư, bản quyền và tác động xã hội

Mục này là phần "bias, quyền riêng tư và đánh giá rủi ro" của đề cương, ở mức một người nghiên cứu phải nói được: một đoạn tiếng Anh chắc chắn, có một con số hoặc một định lý đi kèm, không nói sáo, không có kết luận đạo đức một chiều, và **không phải tư vấn pháp lý**.

**Thiên lệch và công bằng.** Khảo sát của Gallegos và cộng sự (arXiv 2309.00770; chưa đối chiếu bản gốc) phân loại theo chỉ số (biểu diễn, xác suất token, văn bản sinh), bộ dữ liệu, và giảm thiểu (trước, trong, sau huấn luyện hoặc suy luận). Điều cần tự tính được là các định nghĩa công bằng **xung đột**. Gọi p là tỉ lệ nền. Từ định nghĩa PPV = TP/(TP + FP) và FNR = FN/(FN + TP), với p cố định và PPV, FNR cho trước:

$$
\mathrm{FPR} = \frac{p}{1 - p}\cdot\frac{1 - \mathrm{PPV}}{\mathrm{PPV}}\cdot(1 - \mathrm{FNR})
$$

Nên khi hai nhóm có p khác nhau, không thể đồng thời bằng PPV, bằng FNR và bằng FPR (trừ trường hợp suy biến). Đây là phép tính đơn giản của loại định lý bất khả; công thức này cùng họ với kết quả của Chouldechova (2017, "Fair prediction with disparate impact") và của Kleinberg, Mullainathan, Raghavan (2016) [cần kiểm chứng tên, năm và nội dung: tôi chưa mở nguồn].

% manim: TamGiacBatKhaThi

**Ví dụ giải mẫu 12.10 — công bằng: không thể có tất cả.** Hai nhóm A, B, tỉ lệ nền 0.5 và 0.2, mỗi nhóm 1000 người; yêu cầu cùng PPV = 0.8 và cùng FNR = 0.2.

- FPR = p/(1 − p) · (1 − PPV)/PPV · (1 − FNR): nhóm A 1·0.25·0.8 = **0.200**, nhóm B 0.25·0.25·0.8 = **0.050**. Tỉ lệ được chọn p(1 − FNR)/PPV: **0.500** và **0.200**.
- Bảng nhầm lẫn: A có TP 400, FP 100, FN 100, TN 400; B có TP 160, FP 40, FN 40, TN 760. PPV cả hai bằng 0.8.
- Chênh lệch tỉ lệ được chọn (demographic parity) là 0.30; chênh FPR là 0.15 (equalized odds không thỏa), dù PPV và FNR bằng nhau. Muốn FPR bằng nhau thì PPV hoặc FNR phải khác nhau. Chương 3 đã dạy precision phụ thuộc tỉ lệ nền; đây là hệ quả cho công bằng: **chọn chỉ số công bằng là một quyết định có giá trị**, phải nói rõ và có lý do.

Với RAG, thiên lệch vào từ kho, từ bộ truy hồi và từ bộ sinh; đo bằng chênh recall@k hoặc tỉ lệ hiển thị giữa các nhóm. Chương chưa đo (không có dữ liệu có nhãn nhóm hợp lệ), nên đây là thiết kế đánh giá, không phải kết quả.

**Riêng tư.** (1) *Ghi nhớ và trích xuất:* Nasr, Carlini và cộng sự (arXiv 2311.17035; theo abstract) trích được hàng GB dữ liệu huấn luyện từ nhiều mô hình, và một tấn công "divergence" làm ChatGPT phát dữ liệu huấn luyện với tần suất gấp 150 lần hành vi bình thường; lỗ hổng cụ thể có thể đã được vá, tôi không kiểm, bài học bền là bộ nhớ của mô hình trích ra được. (2) *Suy luận thành viên* (membership inference): chỉ nêu tên. (3) *Biện pháp có bảo đảm:* quyền riêng tư vi phân (DP). Cơ chế M là **ε-DP** nếu với hai bộ dữ liệu khác nhau một bản ghi, tỉ số xác suất đầu ra không vượt eᵋ. ε **không** phải xác suất: ε = 1 không có nghĩa "99% an toàn", mà là cận trên của tỉ số xác suất e = 2.718.

**Ví dụ giải mẫu 12.11 — DP: cơ chế Laplace và randomized response.**

(a) Truy vấn đếm có độ nhạy 1; thêm nhiễu Laplace thang 1/ε. Khi đó P(|nhiễu| > t) = exp(−ε·t) và độ lệch chuẩn là √2/ε. ε = 0.1: P(> 5) = 0.607, sd 14.1. ε = 0.5: 0.082, sd 2.83. ε = 1: **0.0067**, sd 1.41. ε = 2: 0.00005, sd 0.71. Mô phỏng 200 000 mẫu ở ε = 1 cho 0.0068 (khớp). ε nhỏ thì riêng tư mạnh nhưng nhiễu lớn; số đếm cỡ vài chục không dùng được ở ε = 0.1.

(b) Randomized response: mỗi người nói thật với xác suất 0.75, nói ngược với 0.25; ε = ln(0.75/0.25) = ln 3 = **1.0986**. Quan sát tỉ lệ "có" 0.55 thì ước lượng π̂ = (0.55 − 0.25)/(0.75 − 0.25) = **0.60**. Sai số chuẩn với n = 1000: √(0.55·0.45/1000)/0.5 = **0.0315**, so với 0.0155 nếu không có nhiễu riêng tư; tỉ số 2.03, tức cần **gấp 4.1 lần mẫu** để cùng độ chính xác. Riêng tư có giá bằng cỡ mẫu. Không dạy DP-SGD ở đây; mục đích là để bạn thấy một ε ứng với con số cụ thể.

**Bản quyền và nguồn gốc dữ liệu.** Hai lớp. (a) Dữ liệu của dự án: điều khoản arXiv ở mục 12.3. (b) Dữ liệu huấn luyện của mô hình nền: vụ *Bartz v. Anthropic*. Theo trang tin của Authors Guild (kiểm ngày 05/11/2026) và một bài của Jurist (7/2026; hai nguồn tin, không phải văn bản tòa), thẩm phán Araceli Martínez-Olguín của tòa liên bang quận Bắc California phê duyệt cuối cùng thỏa thuận 1.5 tỉ USD ngày 20/7/2026, khoảng 3 000 USD mỗi tác phẩm. Số tác phẩm chính xác tôi chưa xác minh [cần kiểm chứng]: Jurist ghi "hơn 400 000 cuốn" trong khi 1.5 tỉ chia 3 000 ra khoảng 500 000, và trang Authors Guild tôi đọc không nêu tổng số. Phán quyết trước đó của thẩm phán Alsup phân biệt huấn luyện trên sách mua hợp pháp với dùng bản lậu dựng thư viện trung tâm; năm phán quyết (2025 theo trí nhớ của tôi) [cần kiểm chứng]. Chỉ lấy ra nguyên tắc "nguồn gốc dữ liệu quan trọng": án lệ đổi nhanh, chưa thống nhất, và các vụ khác (như *NYT v. OpenAI*) tôi chưa xác minh trạng thái tới 10/2026. Biện pháp kỹ thuật cho vấn đề này là ghi **provenance** và viết datasheet và model card (chương 11 đã dẫn Gebru và Mitchell).

**Prompt injection gián tiếp.** Greshake và cộng sự (arXiv 2302.12173, 2023; theo abstract) mô tả kẻ tấn công cài lệnh vào văn bản mà ứng dụng sẽ truy hồi; mô hình không phân biệt dữ liệu với chỉ thị, và các tác giả viết rằng biện pháp giảm thiểu hiệu quả hiện chưa có (mức này có thể đã đổi tới 2026, tôi không kiểm). Với kho ArXiv công khai rủi ro thấp nhưng **không bằng 0**: ai cũng nộp được bài, comments và abstract là văn bản tự do. Biện pháp: coi văn bản truy hồi là đầu vào không tin cậy, tách chỉ thị khỏi dữ liệu, không cho mô hình gọi công cụ có tác dụng phụ từ nội dung truy hồi.

**Tác động xã hội và đánh giá rủi ro.** Khung bốn câu hỏi (diễn đạt của tôi): Ai dùng, dùng vào việc gì? Ai chịu hậu quả khi sai, và có đảo ngược được không? Sai kiểu nào (bịa, thiên lệch, lộ thông tin, bị thao túng) và đo từng kiểu bằng gì? Giới hạn nào ghi vào limitations (mục 11.5)? Về luật, chỉ để thấy khung pháp lý đang dịch chuyển, theo các nguồn tổng hợp đã đọc (thông báo hãng luật 5/2026 và 7/2026, bài tổng hợp 8/2026; không phải văn bản luật): nghĩa vụ cho mô hình đa dụng của EU AI Act áp dụng từ 8/2025; quy định sửa đổi Digital Omnibus on AI đăng Công báo ngày 24/7/2026 và có hiệu lực ngày 27/7/2026 (theo Lewis Silkin, kiểm ngày 05/11/2026) dời mốc cho hệ thống rủi ro cao sang 2/12/2027 (Annex III) và 2/8/2028 (Annex I). Riêng nghĩa vụ minh bạch của Điều 50, trong đó có việc cho người dùng biết họ đang tương tác với một hệ thống AI, vẫn áp dụng từ 2/8/2026 theo cùng nguồn; một hệ hỏi đáp có giao diện trò chuyện là loại hệ thống mà nghĩa vụ ấy nhắm tới, còn hệ của bạn có thuộc diện hay không là câu hỏi pháp lý mà chương không trả lời. Chương không dạy điều luật; hãy tự kiểm tra văn bản hiện hành.

**Vì sao ML cần cái này.**

- Mỗi biện pháp đạo đức cần một **phép đo** (FPR theo nhóm, ε, tỉ lệ lộ chuỗi huấn luyện, tỉ lệ bị lệnh giả làm đổi hành vi), và người nghiên cứu phải nêu đánh đổi mình chọn.
- Với dự án: không đưa dữ liệu cá nhân vào kho; abstract ArXiv công khai nhưng có tên tác giả nên chỉ dùng cho mục đích học tập, và không suy luận thuộc tính cá nhân của tác giả.

**Nối với dự án.** Bài tập tuần: viết "Phiếu rủi ro" một trang cho hệ thống hỏi đáp ArXiv theo bốn câu hỏi trên, cộng một đoạn Limitations cụ thể và kiểm được; thêm một test tự động: đoạn truy hồi chứa lệnh giả ("bỏ qua mọi chỉ thị") không được làm đổi hành vi, chạy trên một đoạn tự dựng.

## Bài tập giấy bút

14 bài, làm theo thứ tự. Mục tiêu: hết ★ trong 1 buổi, hết ★★ trong 2 buổi, ★★★ dành cho Thứ Bảy. Bài nào sai ghi vào sổ lỗi kèm lý do sai.

**★ Hiểu định nghĩa**

1. (a) idf Lucene khi N = 100 và từ có mặt ở n = 9 đoạn. (b) Phần tf của BM25 khi f = 3, k₁ = 2, đoạn có độ dài trung bình. (c) Phần tf tiến về đâu khi f → ∞ với k₁ = 2?
2. Hai bộ truy hồi trả (a, b, c) và (c, a, b). Tính RRF với k = 60 và cho thứ tự gộp.
3. Mức liên quan A = 3, B = 2, C = 1, còn lại 0. Hệ trả (B, A, X). Tính nDCG@3 và recall@3.
4. R = 0.6, a₁ = 0.9, a₀ = 0.1. Tính độ chính xác cuối. Bao nhiêu phần của toàn bộ câu hỏi là câu đúng mà không có đoạn vàng làm cơ sở (coi như đúng không trung thành)?

**★★ Mức đề thi đầu vào**

5. (a) Tài liệu 64 câu, chunk 6 câu không chồng lấn, sự thật hai câu liền nhau bắt đầu ở vị trí đều trong 63 vị trí. Tính chính xác xác suất bị cắt và so với 1/6. (b) Sự thật 3 câu, chunk 4 câu, chồng lấn 1 câu: xác suất cắt theo công thức mục 12.2?
6. Bộ từ vựng trả cả kho theo thứ tự tùy ý khi không khớp từ nào. Tài liệu đúng U có hạng dense 1 và hạng "rác" 200; tài liệu nhiễu V có hạng dense 5 và hạng rác 3. Tính RRF (k = 60) của hai tài liệu và giải thích hiện tượng.
7. Bảng người so với giám khảo LLM: cùng A 40, người A và giám khảo B 10, người B và giám khảo A 5, cùng B 45. Tính p₀, pₑ, κ và nửa độ rộng Wald 95% của p₀.
8. Chấm +1 đúng, 0 bỏ qua, −2 sai. Ngưỡng xác suất để nên đoán là bao nhiêu? Với p = 0.6 nên đoán hay bỏ qua? Hàm chấm cần λ bằng bao nhiêu để ngưỡng là 0.9?
9. Đồ thị đường v₁–v₂–v₃–v₄, hạt giống v₁, α = 0.5. Tính x sau lặp 1 và lặp 2 từ x = s. Nghiệm hội tụ (26, 14, 4, 1)/45: kiểm tổng và cho biết nút nào xếp thứ hai.
10. Nhiễu Laplace cho truy vấn đếm, ε = 0.5. Tìm t để P(|nhiễu| > t) = 0.05 và độ lệch chuẩn. Số đếm thật cỡ 30 thì dùng được không?

**★★★ Mức phỏng vấn nghiên cứu**

11. Từ định nghĩa PPV = TP/(TP + FP), FNR = FN/(FN + TP) và tỉ lệ nền p, chứng minh FPR = p/(1 − p) · (1 − PPV)/PPV · (1 − FNR). Suy ra điều kiện để hai nhóm có cùng PPV, FNR và FPR.
12. Hai người chấm độc lập, người 1 chọn A với xác suất a₁, người 2 với a₂. Chứng minh kỳ vọng p₀ bằng pₑ, nên κ "kỳ vọng" bằng 0. Giải thích vì sao p₀ = 0.90 vẫn cho κ = 0.444 khi cả hai chọn A 90% số lần.
13. **Thiết kế thí nghiệm.** Bạn có 30 câu hỏi hai bước viết tay trên kho ArXiv và ba hệ: BM25, truy hồi lặp 2 vòng, BM25 + PPR trên đồ thị thực thể–đoạn. Cần trả lời: "Graph-RAG có hơn truy hồi lặp không?". (a) Chọn ngân sách chung (k, token vào prompt, số lần gọi mô hình, chi phí lập chỉ mục) và nêu vì sao. (b) Liệt kê ít nhất ba nguồn có thể làm kết quả đổi chiều (gợi ý từ VD 12.7) và cách kiểm từng nguồn. (c) Nếu 30% số câu là cặp bất đồng (đúng ở hệ này, sai ở hệ kia), với 30 câu thì chênh tối thiểu phát hiện được xấp xỉ bao nhiêu ở α = 0.05 và độ mạnh 0.8 (mục 11.3; Wald ghép cặp, sai số chuẩn của chênh ≈ √(0.3/n))? Cần bao nhiêu câu để phát hiện chênh 0.1 với cùng α và độ mạnh? Ngưỡng 1.96·√(0.3/n) một mình cho độ mạnh bao nhiêu?
14. Câu hỏi 3 bước cần 3 cạnh đúng, mỗi cạnh đúng với xác suất 0.9. (a) Xác suất cả ba đúng nếu độc lập. (b) Nếu lỗi tương quan hoàn toàn (một thực thể bị gọi sai tên làm hỏng cả ba hoặc không cạnh nào). (c) Cận dưới Fréchet cho xác suất cả ba đúng. Kết luận gì về giả định độc lập?

## Bài tính nhanh

15 bài, mỗi bài tối đa 2 phút, giấy bút, bấm giờ cả bộ. Mục tiêu: xong trong 25 phút, sai không quá 1 bài. Làm lại sau 3 ngày và sau 1 tuần.

1. idf Lucene với N = 1000, n = 10.
2. Giới hạn của phần tf BM25 khi f → ∞, k₁ = 1.2.
3. Hệ số độ dài 1 − b + b·dl/avgdl khi dl/avgdl = 2, b = 0.75.
4. Đóng góp RRF (k = 60) của hạng 1 và hạng 10.
5. Sự thật hai câu, chunk 8 câu không chồng lấn: xác suất cắt xấp xỉ bao nhiêu?
6. Chiết khấu nDCG ở hạng 4.
7. Vị trí đoạn vàng của 4 truy vấn là 1, 4, 2, không có. Tính recall@3 và MRR (không cắt @k; truy vấn không có đoạn vàng tính 0).
8. F1 với P = 0.75, R = 0.60.
9. RRF: tài liệu A hạng 1 và hạng 20, tài liệu B hạng 6 và hạng 6. Ai thắng với k = 60? Với k = 1?
10. κ với p₀ = 0.90, pₑ = 0.50.
11. Ngưỡng nên đoán khi sai bị phạt λ = 4.
12. Nửa độ rộng Wald 95% của p₀ = 0.78, n = 100.
13. FLOPs một cặp cross-encoder 288 token, 110·10⁶ tham số (2N mỗi token).
14. Nhiễu Laplace ε = 0.5: P(|nhiễu| > 5)?
15. nDCG@3 với mức liên quan A = 2, B = 1 (độ lợi 2ᵍ − 1), hệ trả (B, A, X).

## Lab code và bài săn bug

Năm lab và một bài săn bug, chạy trên CPU với NumPy và PyTorch CPU 1 luồng; dữ liệu **tự dựng, không tải mạng**, seed 0. Số dưới đây lấy từ lần chạy thật của `loigiai.ipynb`; assert dùng khoảng vì kết quả huấn luyện phụ thuộc phiên bản thư viện. Cách học đúng: tự viết từng lab trước, chỉ đối chiếu lời giải khi lệch cột "kết quả mong đợi". Các thí nghiệm là đồ chơi: đừng suy rộng ra dữ liệu thật.

| Lab | Việc phải làm | Kết quả mong đợi (seed 0) |
| --- | --- | --- |
| 1. BM25 và chỉ số xếp hạng | `bm25_idf`, `bm25_scores`, recall@k, MRR, nDCG@k, RRF; tái hiện VD 12.1, 12.3, 12.5 | Điểm 4 đoạn 0.9624 / 0.5154 / 0.4766 / 1.2683; RRF d1 0.03252, d3 0.03227; nDCG@5 0.6489 và 0.9880; khớp lớp `BM25` của ô đầu |
| 2. BM25, bi-encoder, hybrid | Viết hàm gộp RRF; đo ba loại truy vấn, hybrid đúng và hybrid bẫy | id: BM25 hit@1 1.000, dense hit@10 0.900, hybrid hit@10 1.000; paraphrase: BM25 hit@10 0.000, dense 0.970, hybrid đúng bằng dense, hybrid bẫy 0.403; hỗn hợp hit@10: BM25 0.500, dense 0.935, hybrid 0.985; từ khóa hit@1: BM25 0.600, hybrid 0.553 |
| 3. Chunking | `chunk_docs` và hàm đếm chính xác xác suất cắt; so với đo, có và không chồng lấn | Xác suất cắt c = 4: 0.238, c = 8: 0.111; hòa điểm phá ngẫu nhiên: ans@top-1 không chồng lấn c = 2, 4, 8, 16: 0.460, 0.700, 0.880, 0.950; chồng lấn 1 câu: 0.500, 0.830, 0.900, 0.970 với ans@top-3 = 1.000; phá hòa theo chỉ số nhỏ (c = 2, chồng lấn 1 câu): 0.010, hiện vật |
| 4. PPR và hai bước | `ppr_power`; đồ thị 7 nút; BM25, truy hồi lặp, PPR; mất cạnh | P₁ 0.1556, P₂ 0.0120, P₃ 0.1658 (tổng 1, khớp nghiệm đóng); phản ví dụ α = 0.15: Q 0.0439 > P₄ 0.0372; phá hòa ngẫu nhiên, k = 5: BM25 0.000, lặp 0.650, PPR 0.375, RRF 0.210; giữ 70% cạnh: PPR 0.275, RRF 0.145, lặp 0.650; k = 10: lặp 0.730, PPR 0.840; phá hòa theo chỉ số nhỏ: lặp 0.935 (k = 5), 1.000 (k = 10), PPR không đổi |
| 5. Đánh giá và rủi ro | κ, Wald, ECE, rủi ro–độ phủ, FPR bất khả thi, Laplace, randomized response | κ 0.5538, Wald ±0.0812; ECE 0.19; accuracy 1.000 / 0.800 / 0.714 / 0.500 ở độ phủ 0.3 / 0.5 / 0.7 / 1.0; FPR 0.200 và 0.050; Laplace ε = 1: 0.0068; π̂ 0.60, SE 0.0315, tỉ số 2.03 |

Số trong ví dụ 12.2, 12.4 và 12.7 là trung bình 3 seed nên khác số seed 0 ở bảng này; nhiễu giữa các seed cỡ vài điểm phần trăm. Bản 3 seed của cả ba ví dụ tái lập được bằng ô tùy chọn ở cuối `loigiai.ipynb` (đặt `RUN_3_SEED = True`; thêm khoảng 1 phút, nên tắt mặc định để cả notebook dưới 2 phút): VD 12.2 gọi `run_world(s, hybrid_ranking)`; VD 12.4 gọi `trial(*build_docs(s), c, bước, rng=np.random.default_rng(1000 + s))`; VD 12.7 gọi `run_arxiv(s, k=..., keep=..., tie=...)`, với s = 0, 1, 2. **Quy ước seed phá hòa:** VD 12.4 dùng `default_rng(1000 + s)` (s = 0 trùng `default_rng(1000)` của Lab 3); VD 12.7 dùng `default_rng(s + 99)` bên trong `run_arxiv`. Dùng seed phá hòa khác thì số ở các hàng hòa điểm (như c = 2 chồng lấn 1 câu) lệch vài điểm phần trăm. **Thời gian:** `nbconvert` toàn bộ `loigiai.ipynb` mất 16 giây (12 giây bên trong notebook) khi máy rảnh và khoảng 62 giây (47 giây bên trong) khi máy đang bận 6 tiến trình; ở cả hai trường hợp dưới 2 phút.

Code nằm trong `labs/chuong12/lab.ipynb` (có chỗ TODO và ô kiểm tra tự động) và `labs/chuong12/loigiai.ipynb` (lời giải đã chạy kiểm). Mở từ trang **Lab** của app. Các lab nối tiếp nhau (Lab 2 và 4 dùng `rrf` của Lab 1).

**Câu hỏi sau lab.**

- Lab 2: vì sao BM25 không bao giờ trả kết quả cho truy vấn chỉ có từ đồng nghĩa? RRF không cần chuẩn hóa điểm, vậy vì sao vẫn phải "chỉ lấy đoạn có điểm lớn hơn 0"?
- Lab 3: vì sao chồng lấn 1 câu đưa ans@top-3 lên 1.000 mà ans@top-1 ở c = 2 chỉ khoảng 0.5 (và 0.010 nếu phá hòa theo chỉ số nhỏ)? Đó là hiệu ứng của chunking hay của phép đo?
- Lab 4: vì sao PPR thường xếp đoạn đích hai bước sau đoạn nhiễu một bước, và khi nào không? Tăng k làm gì? Vì sao truy hồi lặp không phụ thuộc chất lượng trích xuất, nhưng lại nhạy với quy tắc phá hòa (hòa xảy ra giữa những đoạn nào, và vì sao chỉ số nhỏ luôn chọn đúng một bên?), còn số của PPR thì không đổi?
- Lab 5: κ khác p₀ ở chỗ nào? ε nhỏ nghĩa là gì với kích thước nhiễu?

**Bài săn bug.** Một đường ống RAG nhỏ (60 tài liệu; chunk 4 câu; truy hồi BM25 cộng tri-gram ký tự, gộp RRF; lấy top-3; "đọc" đúng nếu token đáp án nằm trong ngữ cảnh) chạy không báo lỗi. Triệu chứng đo ở seed 0: **recall@3 của truy hồi chỉ 0.017 nhưng chỉ số cuối (e2e) là 0.917, và khi thay truy hồi bằng chọn ngẫu nhiên e2e vẫn 0.917**. Ngoài ra khi chạy riêng từng thành phần, người ta thấy: truy vấn viết hoa tên riêng khó tìm; hybrid thua BM25 đơn lẻ; vài sự thật cuối tài liệu không bao giờ được tìm; có cấu hình recall 1.000 mà e2e chỉ 0.700. Tìm cả **năm** lỗi rồi viết `rag_fixed`. Code lỗi nằm trong `lab.ipynb`, ô `rag_buggy`; sách cố ý không chú thích nguyên nhân. Hai phép thử đa năng nên áp dụng: tắt thành phần thì chỉ số phải tụt, và đo ở nơi dữ liệu thực sự đi vào mô hình.

## Bài dựng cảnh Manim

Giới hạn **1 giờ**, không chỉnh màu mè. Dùng `manim/chuong12_scenes.py` làm mẫu, viết cảnh `RuiRoDoPhu` vào `manim/cua-toi/`:

1. 10 ô xếp ngang, mỗi ô là một câu trả lời với độ tự tin (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.55, 0.50, 0.30), xanh nếu đúng, đỏ nếu sai (đúng/sai theo ví dụ 12.9: 1, 1, 1, 0, 1, 0, 1, 0, 0, 0).
2. Hạ "ngưỡng" từng bước: phần bị từ chối làm mờ; sau mỗi bước hiện độ phủ và accuracy trên phần được trả lời.
3. Dừng ở độ phủ 0.3, 0.5, 0.7, 1.0.

**Tiêu chí tự chấm.** (1) Đúng số: accuracy 1.000, 0.800, 0.714, 0.500. (2) Phần bị từ chối nhìn thấy được là bị làm mờ, không biến mất. (3) Có tiêu đề và phụ đề, không chữ chồng nhau, dài không quá 30 giây. Lời giải tham khảo: `manim/loigiai/chuong12_rui_ro_do_phu.py` (chỉ mở sau khi tự làm).

## Phỏng vấn, bẫy, thực tế khác lý thuyết

**Cách luyện.** Mỗi chuỗi có 4 tầng câu hỏi đào sâu dần. Trả lời to bằng tiếng Việt trước, rồi nói lại bằng tiếng Anh, 2 phút mỗi tầng, ghi âm và nghe lại. Câu mẫu tiếng Anh chỉ phủ tầng 1–2; tầng 3–4 bạn tự soạn rồi đối chiếu với lời giải các bài liên quan.

**Chuỗi 1 — Truy hồi**

1. Thiết kế truy hồi cho hỏi đáp trên kho ArXiv.
2. BM25 và dense: mỗi bên thắng ở loại truy vấn nào? (Định danh, tên riêng so với diễn đạt lại; VD 12.2.)
3. Hybrid và RRF: vì sao dùng hạng, k = 60 làm gì, bẫy danh sách rác, khi nào hybrid không thắng (id hit@1 0.697 so với 1.000 của BM25)?
4. Reranking: chi phí bi-encoder so với cross-encoder, cắt ở top-100, và vì sao không tin một mô hình chỉ vì nó thắng ở miền huấn luyện (BEIR)?

> *Model answer:* I would start with BM25 on titles and abstracts as a strong baseline, because it is hard to beat on identifier-like queries such as method or dataset names, and then add a dense bi-encoder for paraphrased questions. I would fuse the two with reciprocal rank fusion, which uses ranks instead of raw scores and so needs no score normalization, taking care that the lexical list holds only documents with a positive score, because otherwise the fusion mixes in noise and can end up worse than dense alone. Then I would rerank the top hundred with a cross-encoder, since scoring the whole corpus with it is far too expensive. I would report results per query type with confidence intervals, because a hybrid is not guaranteed to beat the best single retriever: in my toy experiment it lost to BM25 at top-1 on identifier queries (0.70 against 1.00).

**Chuỗi 2 — Đánh giá và lỗi RAG**

1. RAG của bạn trả lời sai, bạn làm gì?
2. Phân rã R, a₁, a₀ và đòn bẩy (VD 12.6).
3. Chunking: cắt ngang, chồng lấn, trùng lặp, ngân sách token; Lost in the Middle và cách đo.
4. Đánh giá bằng LLM: faithfulness, κ, hoán vị vị trí, tự ưu ái; rò rỉ do câu hỏi sinh từ chính đoạn và phép thử "tắt truy hồi".

> *Model answer:* I would diagnose the pipeline layer by layer instead of looking at final accuracy. First I measure whether the gold passage is in the context, which gives R, then the accuracy when it is there, a₁, and when it is not, a₀. The overall accuracy is R·a₁ + (1 − R)·a₀, so the two slopes tell me which layer has more leverage, though cost and difficulty decide what I actually fix. I also check faithfulness separately, because an answer can be correct but come from the model's parametric memory instead of the retrieved text. Finally I run an ablation with random retrieval; if accuracy barely drops, my evaluation leaks the answer.

**Chuỗi 3 — Tri thức và Graph-RAG**

1. Vì sao thêm đồ thị tri thức vào RAG ArXiv?
2. Trích xuất: schema, P/R/F1 chặt và sau chuẩn hóa, lỗi nhân pᵏ.
3. PPR: định nghĩa, power iteration có khởi động lại, vì sao đoạn hai bước thường nhận ít khối lượng hơn đoạn một bước và khi nào không (phản ví dụ ở VD 12.7).
4. Khi nào không thêm: đối chứng truy hồi lặp, chi phí lập chỉ mục, bằng chứng ngược 2025, câu hỏi toàn cục so với cục bộ.

> *Model answer:* A graph helps when a question needs evidence spread over several passages, because mass can flow along entity edges that plain similarity search cannot follow. Personalized PageRank does this: it is power iteration with a restart term toward the entities in the query, and a passage is scored by the mass it receives; two-hop passages usually get less mass than one-hop ones, but that is not a law, since several short paths into one passage can outweigh a leaf. In my toy experiment a simple two-round iterative retrieval baseline beat it at k = 5 (0.69 against 0.48), PPR won at k = 10 (0.89 against 0.74) only while the graph was clean, and the numbers moved a lot with the tie-breaking rule. So I compare the graph against iterative retrieval under the same budget, report several values of k and several seeds, measure extraction quality first, and treat any "graph is better" claim as a hypothesis per question type.

**Chuỗi 4 — Hallucination, an toàn, đạo đức**

1. Vì sao mô hình bịa và bạn giảm thế nào?
2. Trung thành mức tuyên bố, hiệu chỉnh, rủi ro–độ phủ, thang chấm thưởng đoán, ngưỡng λ/(1 + λ) (VD 12.9).
3. Riêng tư: ghi nhớ và trích xuất, ε trong DP và cái giá cỡ mẫu (VD 12.11); công bằng: bất khả thi khi tỉ lệ nền khác nhau (VD 12.10).
4. "Viết Phiếu rủi ro cho hệ thống của bạn": ai chịu hậu quả, kiểu sai, prompt injection gián tiếp, limitations kiểm được; bản quyền và nguồn gốc dữ liệu.

> *Model answer:* One argument is that binary right-or-wrong grading rewards guessing: if wrong answers are not penalized, the best policy is to always answer. With a penalty λ for a wrong answer, guessing only pays off when the probability of being right exceeds λ divided by one plus λ, so the scoring rule must allow abstention. I would measure this with a risk-coverage curve, answering only the most confident fraction, and add citations so claims can be checked against passages. This is a theoretical argument I have not verified experimentally. On fairness, if two groups have different base rates, equal precision and equal miss rate force different false positive rates, so choosing a fairness metric is a value judgment I would state explicitly.

**Từ vựng nên nói trôi chảy:** BM25, saturation, length normalization, dense retrieval, bi-encoder, cross-encoder, reciprocal rank fusion, nDCG, chunk overlap, faithfulness, hallucination, calibration, risk-coverage, Personalized PageRank, entity resolution, inter-annotator agreement, differential privacy, base rate, indirect prompt injection, provenance.

**Bẫy thường gặp** (đọc sau khi làm bài săn bug).

- Nhét danh sách "rác" vào RRF (VD 12.2: hit@10 0.399 so với 0.971 của dense), và giả định hybrid luôn không kém bộ đơn tốt nhất (hit@1 0.551 so với 0.589 ở truy vấn từ khóa; 0.697 so với 1.000 của BM25 ở truy vấn id).
- Cộng điểm thô BM25 với cosine, hoặc min–max có ngoại lệ (VD 12.3: một điểm 40.0 đẩy d₁ từ 0.783 xuống 0.575).
- Chỉ báo accuracy cuối; không thấy tầng nào hỏng, không thấy 12/131 câu đúng mà không có đoạn vàng làm cơ sở (VD 12.6); rò rỉ đoạn vàng làm accuracy 1.000 ngay cả khi truy hồi ngẫu nhiên.
- So Graph-RAG với RAG thường mà thiếu truy hồi lặp làm đối chứng hoặc không cùng ngân sách; báo F1 trích xuất mà bỏ qua chuẩn hóa thực thể (0.267 so với 0.667).
- Báo chỉ số @1 hoặc so sánh hai phương pháp khi đầu danh sách có hòa điểm mà để quy tắc phá hòa mặc định (chỉ số nhỏ): ans@top-1 của chồng lấn 0.013 thay vì 0.483, truy hồi lặp 0.963 thay vì 0.685 (VD 12.4, 12.7). Phá hòa ngẫu nhiên, nhiều seed. Gốc rễ: thứ tự chỉ mục có thể chứa cấu trúc của quá trình sinh dữ liệu (cha được tạo trước con), còn BM25 túi từ thì không biết hướng quan hệ.
- Dùng đồng thuận thô làm thước đo giám khảo (p₀ = 0.78 nhưng κ = 0.554, ±0.081) và đọc ε như xác suất; dẫn con số của paper theo trí nhớ hoặc theo abstract như thể đã đọc bảng; tải cả ArXiv qua API (hoặc gọi API nhanh hơn 1 yêu cầu mỗi 3 giây) rồi đẩy toàn văn lên repo công khai.

**Thực tế khác lý thuyết.**

- **Hybrid là mặc định thực tế nhưng không đảm bảo thắng**: ở VD 12.2 nó thắng hit@10 trên hỗn hợp (0.986 so với 0.937), ngang BM25 ở hit@1, thua BM25 ở truy vấn từ khóa (0.551 so với 0.589) và thua nặng nhất ở truy vấn id (hit@1 0.697 so với 1.000). Con số "đồng thuận trên 80%" của giám khảo LLM cũng là con số có ngữ cảnh (MT-bench và Arena 2023, mô hình 2023) và p₀ không trừ ngẫu nhiên.
- **Graph-RAG không mặc định tốt hơn**: có bằng chứng ngược trong abstract của ba paper 2025 và trong VD 12.7 (k = 5: truy hồi lặp 0.685 so với PPR 0.477), cũng có bằng chứng thuận (câu hỏi toàn cục, đa bước; ở k = 10 PPR 0.892 hơn lặp 0.737 trong chính VD đó). Kết quả nhạy với k và với chất lượng trích xuất (giữ 70% liên kết làm PPR@5 tụt 0.477 → 0.258, PPR@10 tụt 0.892 → 0.337) và với quy tắc phá hòa (phá hòa theo chỉ số nhỏ đẩy lặp lên 0.963 ở k = 5). Con số của hãng (Contextual Retrieval 5.7% → 1.9%) là thí nghiệm của hãng trên dữ liệu của hãng, dùng như mẫu quy trình.
- **Hallucination không biến mất khi có RAG** (a₀ = 0.20 ở VD 12.6: khi truy hồi trượt, hệ thống vẫn trả lời và bịa phần còn lại nếu không được phép từ chối), và **số liệu pháp lý, công nghệ ở chương này có hạn dùng ngắn** (điều luật dời ngày, vụ kiện đổi trạng thái, lỗ hổng được vá).

## Mini-test 30 phút

Làm sau khi xong bài tập và lab, bấm giờ đúng 30 phút, không xem sách, không dùng máy tính (trừ câu 5 chỉ đọc code, không chạy). Tổng 30 điểm. **Từ 24 điểm:** sang chương sau. **18 đến dưới 24:** ôn lại mục ứng với câu sai rồi làm lại sau 3 ngày. **Dưới 18:** học lại chương.

1. **(4 điểm)** Kho N = 4 đoạn, một từ t của truy vấn chỉ có ở 1 đoạn. Tính idf Lucene. Đoạn đó chứa t hai lần, k₁ = 1.2, b = 0.75: tính điểm khi độ dài bằng độ dài trung bình, và khi gấp đôi độ dài trung bình.
2. **(5 điểm)** Mức liên quan A = 3, B = 2, C = 1. Hệ 1 trả (A, X, B, X, C), hệ 2 trả (X, A, B, C, X). Tính recall@3 và nDCG@5 của từng hệ, rồi nói hai chỉ số cho thấy gì.
3. **(5 điểm)** 100 câu hỏi: đoạn vàng nằm trong ngữ cảnh ở 60 câu, trong đó đúng 51; ở 40 câu còn lại đúng 8. Tính R, a₁, a₀, độ chính xác cuối. Tăng R lên 0.75 hoặc tăng a₁ lên 0.95: cái nào cho độ chính xác cao hơn? Bao nhiêu phần trăm câu đúng là câu đúng mà không có đoạn vàng làm cơ sở?
4. **(5 điểm)** Người và giám khảo LLM chấm 100 cặp: cùng A 60, người A và giám khảo B 5, người B và giám khảo A 15, cùng B 20. Tính p₀, pₑ, κ và nửa độ rộng Wald 95% của p₀.
5. **(6 điểm)** Đọc code, không chạy:

```python
questions = [llm("Hãy đặt một câu hỏi mà đoạn sau trả lời được", p) for p in passages]
correct = 0
for q, p in zip(questions, passages):
    top = retriever.top_k(q, k=5)
    ctx = top + [p]
    correct += gold_answer(p) in llm(make_prompt(q, ctx))
print(correct / len(questions))
```

(a) Chỉ ra hai vấn đề đánh giá. (b) Triệu chứng nào của con số cuối sẽ lộ ra khi thay `retriever` bằng truy hồi ngẫu nhiên? (c) Sửa thế nào?

6. **(5 điểm)** Tối đa 5 câu: hai nhóm có tỉ lệ nền 0.5 và 0.2 muốn cùng PPV = 0.8 và cùng FNR = 0.2. Vì sao không thể có cùng FPR? Nêu hai số FPR và kết luận về việc chọn chỉ số công bằng.

## Flashcard

18 thẻ của chương nằm trong `flashcards/chuong12.csv`. Ôn ở trang **Ôn tập** của app: app tự lên lịch theo hộp Leitner (thẻ nhớ được sẽ giãn lịch ôn 1 → 3 → 7 → 14 ngày, thẻ quên quay lại hộp 1). File CSV này cũng nhập thẳng được vào Anki nếu bạn thích dùng Anki.

## Lời giải

**Kiểm tra nhanh**

1. Khoảng 1.1 lần: f = 100 cho 100·2.2/101.2 = 2.174, f = 10 cho 1.964, tỉ số 1.107. Lặp từ không cộng thẳng.
2. Xấp xỉ 1/4 (đếm chính xác trên tài liệu 64 câu: 15/63 = 0.238).
3. Sau: P₂ nhận 0.0120 so với P₁ 0.1556 và P₃ 0.1658, nhưng vẫn dương. Không đúng trong mọi đồ thị: nhiều đường ngắn hội tụ về một đoạn xa có thể thắng đoạn lá một bước (phản ví dụ ở VD 12.7(a): α = 0.15, Q = 0.0439 > P₄ = 0.0372). "Thường sau", không phải luật.
4. Gần 0.4: pₑ = 0.9·0.9 + 0.1·0.1 = 0.82, κ = (0.90 − 0.82)/0.18 = 0.444.
5. Không, trừ trường hợp suy biến (PPV = 1 hoặc FNR = 1, khi đó FPR = 0 ở cả hai nhóm): FPR = p/(1 − p) · (1 − PPV)/PPV · (1 − FNR) vẫn phụ thuộc p dù PPV và FNR đã bằng nhau, nên muốn cùng FPR thì phải để PPV hoặc FNR lệch nhau.

**Bài tập giấy bút**

1. (a) ln(1 + 91.5/9.5) = ln 10.632 = 2.364. (b) 3·3/(3 + 2) = 1.8. (c) k₁ + 1 = 3.
2. a: 1/61 + 1/62 = 0.03252; c: 1/63 + 1/61 = 0.03227; b: 1/62 + 1/63 = 0.03200. Thứ tự a, c, b.
3. DCG = 3/1 + 7/1.585 = 3 + 4.4165 = 7.4165; lý tưởng (A, B, C) = 9.3928; nDCG@3 = 0.7896. recall@3 = 2/3 (có A, B; thiếu C).
4. Acc = 0.6·0.9 + 0.4·0.1 = 0.58. Câu đúng mà không có đoạn vàng = (1 − R)·a₀ = 0.04, tức 4% toàn bộ câu hỏi (khoảng 6.9% số câu đúng); là "đúng không trung thành" nếu giả định chúng đến từ bộ nhớ tham số, còn thực tế đây chỉ là cận trên.
5. (a) Cắt khi (pos + 1) chia hết cho 6: pos = 5, 11, …, 59, tức 10 vị trí, xác suất 10/63 = 0.159, gần 1/6 = 0.167. (b) max(0, 3 − 1 − 1)/(4 − 1) = 1/3.
6. U: 1/61 + 1/260 = 0.0164 + 0.0038 = 0.0202. V: 1/65 + 1/63 = 0.0154 + 0.0159 = 0.0313. V thắng dù không liên quan gì: hạng "rác" tùy ý vẫn được cộng điểm như một bằng chứng thật, nên danh sách từ vựng chỉ được chứa đoạn có điểm lớn hơn 0.
7. p₀ = 0.85; người chọn A 0.50, giám khảo chọn A 0.45; pₑ = 0.5·0.45 + 0.5·0.55 = 0.50; κ = 0.35/0.50 = 0.70; Wald ±1.96·√(0.85·0.15/100) = ±0.070.
8. Ngưỡng λ/(1 + λ) = 2/3. Với p = 0.6 kỳ vọng khi đoán = 0.6 − 2·0.4 = −0.2 < 0: bỏ qua. λ/(1 + λ) = 0.9 cho λ = 9.
9. Lặp 1: x = (0.5, 0.5, 0, 0). Lặp 2: x = (0.625, 0.25, 0.125, 0). Hội tụ (26, 14, 4, 1)/45 = (0.578, 0.311, 0.089, 0.022), tổng 45/45 = 1; nút xếp thứ hai là v₂ (láng giềng của hạt giống), khối lượng giảm theo khoảng cách.
10. exp(−0.5t) = 0.05 cho t = 2 ln 20 = 5.99; độ lệch chuẩn √2/0.5 = 2.83. Số đếm cỡ 30 có sai số tương đối cỡ 10%, dùng được một cách thận trọng; ở ε = 0.1 (sd 14) thì không.
11. TP = (1 − FNR)·pN, FN = FNR·pN; PPV = TP/(TP + FP) cho FP = TP·(1 − PPV)/PPV. FPR = FP/((1 − p)N) = p/(1 − p) · (1 − FNR) · (1 − PPV)/PPV. Hai nhóm có cùng PPV và FNR thì FPR chỉ phụ thuộc p/(1 − p), nên chỉ bằng nhau khi p bằng nhau (hoặc PPV = 1 hoặc FNR = 1, các trường hợp suy biến).
12. Nếu độc lập, P(đồng thuận) = a₁a₂ + (1 − a₁)(1 − a₂) = pₑ nên 𝔼[p₀ − pₑ] = 0 và κ kỳ vọng bằng 0 (xấp xỉ, vì κ là tỉ số). Với a₁ = a₂ = 0.9: pₑ = 0.82, p₀ = 0.90 cho κ = 0.08/0.18 = 0.444: 0.90 đồng thuận chỉ hơn mức ngẫu nhiên 0.08 trong thang tối đa 0.18.
13. (a) Cùng k và cùng số token vào prompt cho cả ba hệ (báo ít nhất hai giá trị k, ví dụ 5 và 10, vì chiều kết quả đảo theo k); tính riêng chi phí lập chỉ mục đồ thị (gọi LLM trích thực thể) và chi phí lúc truy vấn (lặp tốn hai lần truy vấn BM25 và nối thực thể). (b) Ba nguồn: **quy tắc phá hòa** (chạy phá hòa ngẫu nhiên nhiều seed và so với phá hòa theo chỉ số; nếu lặp đổi nhiều, nghi rò rỉ qua thứ tự chỉ mục: BM25 túi từ không phân biệt "x mở rộng y" với "c mở rộng x", còn thứ tự chỉ mục cha trước con thì phân biệt được), **k** (quét k = 3, 5, 10, 20, vẽ đường cong), **chất lượng đồ thị** (xóa ngẫu nhiên 10%, 20%, 30% cạnh và báo độ nhạy). Thêm: kích thước mẫu nhỏ (bootstrap ghép cặp hoặc McNemar), thực thể khớp chuỗi quá sạch so với thật. (c) Sai số chuẩn của chênh với 30 câu: √(0.3/30) = 0.10. Độ mạnh 0.8 cần chênh thật ≥ (1.96 + 0.84)·0.10 ≈ **0.28**. Để phát hiện chênh 0.1 ở độ mạnh 0.8: n ≥ 0.3·((1.96 + 0.84)/0.1)² = 0.3·784 = 235.2, tức ≥ **236 câu**. Ngưỡng ý nghĩa đơn thuần 1.96·√(0.3/30) = 0.196 (≈ 0.20) và n ≥ 0.3·(1.96/0.1)² = 115.2 (116 câu) chỉ là chênh quan sát đủ để p < 0.05; nếu chênh thật đúng bằng 0.1 thì với 116 câu phép thử chỉ có ý nghĩa khoảng 50% số lần (độ mạnh ≈ 0.50). Kết luận: 30 câu chỉ thấy chênh lớn (cỡ 0.28 trở lên) hoặc tìm chế độ lỗi, không đủ để xếp hạng hai hệ gần nhau; muốn phát hiện chênh 0.1 một cách đáng tin cần cỡ 236 câu.
14. (a) 0.9³ = 0.729. (b) 0.9 nếu lỗi tương quan hoàn toàn (mọi cạnh đúng hoặc sai cùng nhau). (c) max(0, 1 − 3·0.1) = 0.7. Xác suất thật nằm trong [0.7, 0.9] tùy cấu trúc phụ thuộc. 0.729 nằm **bên trong** khoảng, không là đầu mút nào, nên trên mọi cấu trúc phụ thuộc nó không là cận bảo thủ hay lạc quan. Nếu lỗi tương quan dương (các cạnh cùng đúng hoặc cùng sai, trường hợp hay gặp) thì xác suất thật nằm trong [0.729, 0.9], tức độc lập là cận dưới (bi quan); chỉ tương quan âm mới đẩy xuống tới 0.7. Vì vậy phải đo trên câu hỏi nhiều bước thật.

**Bài tính nhanh**

1. ln(1 + 990.5/10.5) = 4.557. 2. 2.2. 3. 0.25 + 0.75·2 = 1.75. 4. 1/61 = 0.01639; 1/70 = 0.01429. 5. 1/8 = 0.125 (đếm chính xác 64 câu: 7/63 = 0.111). 6. 1/log₂5 = 0.4307. 7. recall@3 = 2/4 = 0.5 (vị trí 1 và 2); MRR = (1 + 0.25 + 0.5 + 0)/4 = 0.4375. 8. 2·0.75·0.60/1.35 = 0.667. 9. k = 60: A = 1/61 + 1/80 = 0.0289, B = 2/66 = 0.0303, B thắng; k = 1: A = 1/2 + 1/21 = 0.548, B = 2/7 = 0.286, A thắng (k nhỏ thưởng hạng 1, k lớn thưởng sự nhất quán). 10. 0.40/0.50 = 0.80. 11. 4/5 = 0.8. 12. 0.0812. 13. 2·110·10⁶·288 = 6.336·10¹⁰. 14. exp(−2.5) = 0.0821. 15. DCG = 1 + 3/1.585 = 2.893; lý tưởng (A, B) = 3 + 1/1.585 = 3.631; nDCG@3 = 0.797.

**Câu hỏi sau lab**

- Lab 2: BM25 chỉ cộng điểm cho từ có chung với đoạn; đồng nghĩa không có chung chuỗi nên mọi điểm bằng 0 và danh sách rỗng. RRF dùng hạng nên không cần chuẩn hóa điểm, nhưng một danh sách chứa cả kho cho mọi tài liệu một hạng tùy ý và được cộng điểm như bằng chứng thật, nên tài liệu nhiễu có thể vượt tài liệu đúng (bài tập 6).
- Lab 3: chồng lấn 1 câu làm mọi sự thật hai câu nằm trọn trong một chunk, nên đáp án luôn nằm trong top-3. Ở c = 2, bước 1, hai chunk kề nhau cùng chứa tên và thuộc tính, cùng độ dài, nên **hòa điểm tuyệt đối**, và chỉ chunk sau chứa đáp án: ans@top-1 xấp xỉ tung đồng xu (0.500 ở seed 0, 0.483 trung bình 3 seed, ngang không chồng lấn). Con số 0.010 khi phá hòa theo chỉ số nhỏ là hiện vật của phép đo (chunk chỉ số nhỏ luôn là chunk chưa có đáp án; phá hòa theo chỉ số lớn sẽ đảo ngược), không phải hiệu ứng của chunking. Cái giá thật của chồng lấn là số chunk gần gấp đôi ở c = 2 (6300 so với 3200) và token trùng.
- Lab 4: mỗi bước đi nhân với (1 − α) và chia theo bậc nút, nên khối lượng thường giảm theo khoảng cách: đoạn lá của thực thể truy vấn nhận trước, đoạn đích hai bước nhận sau nhưng dương. Đó không phải luật: nhiều đường ngắn hội tụ về một đoạn xa có thể vượt đoạn lá (phản ví dụ α = 0.15: Q 0.0439 > P₄ 0.0372). Tăng k lên 10 thì PPR lấy được đoạn đích (0.840 ở seed 0, hơn lặp 0.730), nhưng chỉ khi đồ thị còn đủ cạnh. Truy hồi lặp chỉ dùng BM25 trên văn bản, không có cạnh nào để mất, nhưng nhạy với quy tắc phá hòa. Ở vòng 1, top-1 luôn là đoạn đánh giá của x; hòa xảy ra ở hạng 2, giữa đoạn giới thiệu của x và đoạn giới thiệu của các con của x (cùng chứa một lần "mx" và một lần "extends", cùng độ dài, nên BM25 túi từ không phân biệt "x mở rộng y" với "c mở rộng x"). Phá hòa theo chỉ số nhỏ luôn chọn x, vì dữ liệu được sinh cha trước con nên thứ tự chỉ mục mang theo hướng của quan hệ "mở rộng": 0.935 (k = 5) và 1.000 (k = 10). Phá hòa ngẫu nhiên đưa đoạn của x vào top-2 ở khoảng 71% số câu (seed 0) và cho 0.650. Số của PPR không đổi theo quy tắc phá hòa trong thí nghiệm này (đã đo bằng assert).
- Lab 5: p₀ không trừ phần đồng thuận do ngẫu nhiên; κ trừ nó theo phân phối biên. ε nhỏ là riêng tư mạnh hơn nhưng nhiễu lớn hơn (thang 1/ε, độ lệch chuẩn √2/ε).

**Mini-test**

1. idf = ln(1 + 3.5/1.5) = 1.204 (1 điểm). dl = avgdl: phần tf = 2·2.2/(2 + 1.2) = 1.375, điểm 1.655 (1.5 điểm). dl = 2·avgdl: hệ số 0.25 + 0.75·2 = 1.75, phần tf = 4.4/(2 + 2.1) = 1.073, điểm 1.292 (1.5 điểm).
2. recall@3: hệ 1 (A, X, B) là 2/3; hệ 2 (X, A, B) là 2/3 (1 điểm). nDCG@5 hệ 1: (7 + 3/2 + 1/2.585)/9.3928 = 8.8869/9.3928 = 0.946; hệ 2: (7/1.585 + 3/2 + 1/2.322)/9.3928 = 6.347/9.3928 = 0.676 (2 điểm). Cùng recall@3 nhưng nDCG khác nhiều vì A (độ quan trọng cao nhất) ở hạng 1 hay hạng 2 (2 điểm).
3. R = 0.6, a₁ = 51/60 = 0.85, a₀ = 8/40 = 0.20, Acc = 0.51 + 0.08 = 0.59 (2 điểm). R lên 0.75: 0.75·0.85 + 0.25·0.2 = 0.6875; a₁ lên 0.95: 0.6·0.95 + 0.4·0.2 = 0.65; tăng R cho nhiều hơn ở số liệu này (1.5 điểm). Câu đúng mà không có đoạn vàng làm cơ sở: 8/59 = 13.6% (1.5 điểm; ghi chú đây là cận trên của "không trung thành").
4. p₀ = 0.80 (1 điểm). Người chọn A 0.65, giám khảo chọn A 0.75; pₑ = 0.65·0.75 + 0.35·0.25 = 0.575 (1.5 điểm). κ = 0.225/0.425 = 0.529 (1 điểm). Wald ±1.96·√(0.8·0.2/100) = ±0.078 (1.5 điểm).
5. (a) Câu hỏi sinh từ chính đoạn nên trùng từ vựng với đoạn, BM25 đẹp giả (1 điểm); đoạn vàng `p` bị đưa thẳng vào ngữ cảnh: rò rỉ nhãn (1 điểm). (b) Accuracy cuối gần như không giảm, vì đáp án luôn nằm trong ngữ cảnh (2 điểm). (c) Bỏ `+ [p]`; đo recall của truy hồi riêng; câu hỏi viết tay hoặc sinh từ đoạn khác và lọc từ vựng trùng; thêm phép thử truy hồi ngẫu nhiên (2 điểm).
6. FPR = p/(1 − p) · (1 − PPV)/PPV · (1 − FNR): nhóm p = 0.5 cho 0.200, nhóm p = 0.2 cho 0.050 (3 điểm cho công thức và hai số). Muốn FPR bằng nhau thì phải để PPV hoặc FNR khác nhau, nên chọn chỉ số công bằng là quyết định có giá trị phải nêu và biện minh (2 điểm).

**Săn bug.** Năm lỗi trong `rag_buggy`, mỗi lỗi một dòng; triệu chứng đo ở seed 0 (recall@3 / e2e / e2e với truy hồi ngẫu nhiên):

1. Truy vấn đưa vào BM25 không hạ chữ thường trong khi chỉ mục đã hạ: 0.150 / 0.150 / 0.000.
2. Cắt mỗi chunk ở 40 token khi dựng ngữ cảnh (`ch[i][2][:40]`): 1.000 / 0.700 / 0.000 (truy hồi hoàn hảo, trả lời vẫn sai 30%).
3. Thêm đoạn chứa đáp án vào ngữ cảnh (biến `extra`): 1.000 / 1.000 / **1.000** (không giảm khi truy hồi ngẫu nhiên; chỉ phép thử tắt truy hồi bắt được).
4. Sắp tri-gram theo `np.argsort(sf)` tăng dần trong khi sf là cosine (cần giảm dần): 0.817 / 0.817 / 0.000 (hybrid thua BM25 một mình).
5. Vòng lặp chunk `range(0, len(sents) - 4, 3)` bỏ chunk cuối: 0.917 / 0.917 / 0.000 (recall không bao giờ chạm 1.000).

Cả năm lỗi cùng lúc: 0.017 / **0.917** / **0.917**: số cuối trông ổn trong khi truy hồi đã chết. Các dòng sửa:

```python
for s in range(0, len(sents), 3):               # phủ đến câu cuối
    ch.append((d, s, [t for x in sents[s:s + 4] for t in x]))
    if s + 4 >= len(sents):
        break
sb = bm.score([t.lower() for t in q])           # cùng chuẩn hóa với chỉ mục
rf = list(np.argsort(-sf))                      # cosine giảm dần
ctx = [ch[i][2] for i in top]                   # không cắt, không thêm đoạn nào ngoài top-k
```

Bản sửa cho 1.000 / 1.000 / 0.000. Bài học: ablation phải đau (tắt truy hồi thì chỉ số phải tụt), và đo ở nơi dữ liệu thực sự đi vào mô hình.

## Tài liệu đào sâu và nguồn

**Bảng 10 paper theo phiếu đọc của chương 11.** Mỗi paper một trang ghi chú (claim, evidence, gaps, questions, what I would run). **Lượt 1 (10 phút) cho cả 10 paper** (khoảng 1.7 giờ); **lượt 2 (60 phút) cho 4 paper** gắn với dự án: #5 GraphRAG, #6 HippoRAG, #7 RAG vs. GraphRAG, #2 Lost in the Middle; **lượt 3 chỉ cho một** (gợi ý #6 hoặc #7, vì dự án phải quyết định có dùng đồ thị không). Cột "mức xác minh" cho biết tôi đã làm gì: **ABS** là mở được trang arXiv và đọc phần tóm tắt do công cụ cho, **không có paper nào được đọc toàn văn**, nên mọi số dưới đây là mức abstract, **chưa đối chiếu bản gốc**; các phiên bản sau có thể đổi số. Cột "tìm trong thân bài" là câu hỏi bạn phải tự trả lời khi đọc, không phải kết quả tôi đã đọc.

| # | Paper | Khẳng định theo abstract | Tìm trong thân bài | Mức xác minh |
| --- | --- | --- | --- | --- |
| 1 | Thakur và cộng sự 2021, [BEIR](https://arxiv.org/abs/2104.08663) | 18 bộ dữ liệu, 10 hệ thống; BM25 là baseline "robust"; re-ranking và late-interaction tốt nhất trung bình nhưng đắt | BM25 thắng ở bộ nào; chỉ số và ngân sách tính toán; ai chọn siêu tham số | ABS; venue theo trang arXiv; mô hình 2021 đã cũ |
| 2 | Liu và cộng sự 2023, [Lost in the Middle](https://arxiv.org/abs/2307.03172) | Hiệu năng tốt nhất khi thông tin ở đầu hoặc cuối ngữ cảnh | Hai tác vụ nào; số tài liệu; đường cong theo vị trí; số seed | ABS; venue (TACL 2024) theo kết quả tìm kiếm; mô hình 2023, phải đo lại |
| 3 | Es và cộng sự 2023, [RAGAS](https://arxiv.org/abs/2309.15217) | Bộ chỉ số không cần nhãn người cho truy hồi, trung thành, chất lượng sinh | Từng chỉ số tính bằng prompt nào; tương quan với nhãn người; chỉ số nào dễ bị đánh lừa | ABS; tên ba chỉ số là NHỚ; dùng v2, API thư viện đổi nhanh |
| 4 | Zheng và cộng sự 2023, [LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) | Giám khảo mạnh đạt hơn 80% đồng thuận với người; nêu thiên lệch vị trí, độ dài, tự ưu ái | Đồng thuận tính thế nào (có trừ ngẫu nhiên không); cỡ mẫu; hiệu quả cách giảm thiên lệch | ABS; giám khảo là mô hình 2023 |
| 5 | Edge và cộng sự 2024, [GraphRAG](https://arxiv.org/abs/2404.16130) | Đồ thị thực thể và tóm tắt cộng đồng cải thiện tính toàn diện và đa dạng cho câu hỏi toàn cục | Định nghĩa câu hỏi toàn cục; ai chấm; baseline RAG được tune ra sao; chi phí lập chỉ mục | ABS; trang arXiv không ghi venue; kết luận có thể đã đổi |
| 6 | Gutiérrez và cộng sự 2024, [HippoRAG](https://arxiv.org/abs/2405.14831) | LLM, đồ thị tri thức và PPR; hơn SOTA đa bước "tới 20%"; một bước ngang IRCoT, rẻ hơn 10–30 lần | Bộ dữ liệu và metric; "tới 20%" ở bộ nào; cấu hình IRCoT; cách tính chi phí | ABS; venue là NHỚ; có HippoRAG 2, đọc cả hai |
| 7 | Han và cộng sự 2025, [RAG vs. GraphRAG](https://arxiv.org/abs/2502.11371) | Mỗi hướng mạnh ở tác vụ khác nhau; chọn hoặc hợp nhất cho cải thiện ổn định | Giao thức thống nhất gồm gì; hướng nào thắng ở tác vụ nào; so với VD 12.7 | ABS; ba phiên bản, dùng v3 |
| 8 | Kalai và cộng sự 2025, [Why Language Models Hallucinate](https://arxiv.org/abs/2509.04664) | Huấn luyện và đánh giá thưởng cho đoán; hallucination như lỗi phân loại nhị phân | Phát biểu định lý nào; giả định; có thí nghiệm không; so với VD 12.9(d) | ABS; lập luận lý thuyết, chưa kiểm phản hồi |
| 9 | Nasr, Carlini và cộng sự 2023, [Extraction of Training Data](https://arxiv.org/abs/2311.17035) | Trích hàng GB dữ liệu huấn luyện; tấn công divergence làm ChatGPT phát dữ liệu huấn luyện gấp 150 lần bình thường | Cách xác minh một chuỗi là dữ liệu huấn luyện; chi phí; phản ứng của nhà cung cấp | ABS; lỗ hổng cụ thể có thể đã vá |
| 10 | Gallegos và cộng sự 2023, [Bias and Fairness in LLMs: A Survey](https://arxiv.org/abs/2309.00770) | Khảo sát; ba phân loại: chỉ số, bộ dữ liệu, giảm thiểu | Chỉ số nào ở tầng nào và hạn chế; so với VD 12.10 | ABS; venue là NHỚ; chỉ số cụ thể có thể cũ |

**Nguồn khác** (trạng thái: **ĐÃ ĐỌC** là mở được trang và đọc nội dung; **ABS** như trên).

| Tài liệu | Dùng để làm gì | Phần nên đọc |
| --- | --- | --- |
| [Cormack, Clarke, Büttcher 2009, RRF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) (ĐÃ ĐỌC) | Công thức RRF và k = 60 | Mục phương pháp, vài trang đầu |
| [Karpukhin và cộng sự 2020, DPR](https://arxiv.org/abs/2004.04906) (ABS) | Bi-encoder, so với BM25; SQuAD là bộ BM25 thắng | Bảng kết quả (tự đọc, chưa đối chiếu) |
| [Anthropic, Contextual Retrieval, 9/2024](https://www.anthropic.com/engineering/contextual-retrieval) (ĐÃ ĐỌC, blog hãng) | Chuỗi cải tiến có đo; số 5.7% → 1.9% là của hãng | Phần kết quả |
| [Microsoft Research, Moving to GraphRAG 1.0, 12/2024](https://www.microsoft.com/en-us/research/blog/moving-to-graphrag-1-0-streamlining-ergonomics-for-developers-and-users/) (ĐÃ ĐỌC) | LazyGraphRAG, lệnh `update` | Toàn bài ngắn |
| [Gao và cộng sự 2023, RAG survey](https://arxiv.org/abs/2312.10997); [Greshake và cộng sự 2023, indirect prompt injection](https://arxiv.org/abs/2302.12173); [Manakul và cộng sự 2023, SelfCheckGPT](https://arxiv.org/abs/2303.08896) (ABS) | Bản đồ thuật ngữ RAG; prompt injection gián tiếp; phát hiện hallucination | Abstract, rồi phần phương pháp |
| [arXiv, Bulk Data Access](https://info.arxiv.org/help/bulk_data.html) (ĐÃ ĐỌC, kiểm lại 05/11/2026) | Cách tải dữ liệu ArXiv: OAI-PMH, S3, giấy phép; gợi ý cụm 4 yêu cầu mỗi giây cho tải hàng loạt | Toàn trang; đọc lại trước khi tải |
| [arXiv, Terms of Use for arXiv APIs](https://info.arxiv.org/help/api/tou.html) (ĐÃ ĐỌC 05/11/2026) | Giới hạn API cũ (OAI-PMH, RSS, arXiv API): 1 yêu cầu mỗi 3 giây, 1 kết nối; metadata CC0 1.0 | Mục giới hạn tần suất và mục giấy phép metadata |
| [Thẻ dữ liệu metadata ArXiv trên Hugging Face](https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot) (ĐÃ ĐỌC, bản mirror, không phải bản gốc) | Danh sách trường; **không có cạnh trích dẫn**; giấy phép ghi trên thẻ | Phần mô tả trường |
| [Gibson Dunn, 27/5/2026](https://www.gibsondunn.com/eu-ai-act-omnibus-agreement-postponed-high-risk-deadlines-and-other-key-changes/); [Mondaq, 3/8/2026](https://www.mondaq.com/eu-digital-omnibus-on-ai-enters-into-force/1825722) (ĐÃ ĐỌC; thông báo hãng luật và bài tổng hợp, không phải văn bản luật) | Các mốc EU AI Act; chỉ để thấy khung luật dịch chuyển | Đối chiếu với văn bản chính thức |
| [Lewis Silkin, 27/7/2026, Digital Omnibus on AI](https://www.lewissilkin.com/insights/2026/07/27/the-digital-omnibus-on-ai-enters-into-force-today-102nedo) (ĐÃ ĐỌC 05/11/2026; thông báo hãng luật) | Ngày Công báo 24/7/2026, hiệu lực 27/7/2026; mốc rủi ro cao; Điều 50 từ 2/8/2026 | Đối chiếu với văn bản chính thức; không phải tư vấn pháp lý |
| [Authors Guild, phê duyệt cuối thỏa thuận Anthropic](https://authorsguild.org/news/court-grants-final-approval-anthropic-copyright-settlement/) (ĐÃ ĐỌC 05/11/2026) và [Jurist, 7/2026](https://www.jurist.org/news/2026/07/judge-approves-record-1-5-billion-settlement-involving-anthropic/) (ĐÃ ĐỌC; báo chí) | Vụ *Bartz v. Anthropic*: ngày 20/7/2026, thẩm phán Martínez-Olguín, 1.5 tỉ USD, khoảng 3 000 USD mỗi tác phẩm | Chỉ nêu sự việc; kiểm lại nguồn pháp lý |

Không mở được hoặc chưa xác minh: toàn văn mọi paper ở trên; ColBERT và MTEB; Pan và cộng sự (khảo sát LLM và đồ thị tri thức), Gebru (Datasheets), Mitchell (Model Cards); tên mô hình embedding hoặc reranker đang phổ biến; trạng thái mã GraphRAG, HippoRAG, LightRAG; trạng thái vụ *NYT v. OpenAI*; tên, năm và nội dung của Chouldechova (2017) và Kleinberg và cộng sự (2016) (theo trí nhớ, chưa mở); số tác phẩm và năm phán quyết Alsup trong vụ *Bartz*; tình hình "RAG so với ngữ cảnh dài" tới 10/2026 (câu hỏi mở).

**Từ khóa để tự tìm thêm:** BM25, Okapi, inverted index, dense passage retrieval, hard negative mining, late interaction, cross-encoder, reciprocal rank fusion, query rewriting, contextual retrieval, MMR, lost in the middle, faithfulness, attribution, nDCG, knowledge graph construction, entity linking, entity resolution, Personalized PageRank, community detection, multi-hop QA, IRCoT, LLM-as-a-judge, Cohen's kappa, calibration, selective prediction, membership inference, differential privacy, Laplace mechanism, randomized response, demographic parity, equalized odds, model card, datasheet, provenance, Chouldechova fair prediction disparate impact [cần kiểm chứng], Kleinberg inherent trade-offs fair determination of risk scores [cần kiểm chứng].

**Trước khi sang chương sau:** làm xong Phiếu rủi ro một trang cho hệ thống hỏi đáp ArXiv, bảng phân rã R, a₁, a₀ trên 30 câu, bảng bốn cấu hình truy hồi (có truy hồi lặp làm đối chứng) chia theo loại câu hỏi, và đạt từ 24 điểm mini-test.

Nguồn tra cứu ngày 2026-11-03; điều khoản arXiv, EU AI Act và vụ *Bartz* kiểm lại ngày 05/11/2026. Số liệu pháp lý và công nghệ trong chương có hạn dùng ngắn: hãy kiểm lại trước khi dựa vào chúng.
