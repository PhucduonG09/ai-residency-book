"""Sinh hai notebook cho lab chương 13: lab.ipynb (có TODO) và loigiai.ipynb (lời giải).

Chạy lại script này nếu muốn tạo lại notebook gốc:  python labs/build_chuong13.py

Mọi mẫu CV, statement, kịch bản, bản ghi trong notebook là MẪU MINH HỌA tự soạn; số liệu là số giả.
"""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "chuong13"
OUT.mkdir(parents=True, exist_ok=True)

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

SETUP = r"""import re, math
import statistics as st
from datetime import date, timedelta
import numpy as np
import pandas as pd
pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 20)"""

# =====================================================================================
# LAB 1 — kiểm kịch bản video
# =====================================================================================
SCRIPT_DATA = r"""# Dữ liệu mẫu (số giả, chỉ để minh họa). Quy tắc của SÁCH: 140 từ/phút; tổng 160–190 giây; mỗi đoạn lệch <= 30%.
WPM = 140

def words(t):
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-\.]*", t)

def sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]

def syllables(w):                      # heuristic đếm âm tiết, sai số vài phần trăm; chỉ đúng cho tiếng Anh
    w = re.sub(r"[^a-z]", "", w.lower())
    if not w:
        return 0
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(1, n)

TARGETS = [("Hook + one-line takeaway", 20), ("Problem and why it is hard", 25), ("Approach", 45),
           ("Evidence", 45), ("My role and limits", 25), ("Close: why this residency", 20)]

GOOD = {
 "Hook + one-line takeaway": "Imagine asking about a new paper and getting an answer that cites the wrong one. I built a question answering system over arXiv abstracts. It retrieves with text and with the citation graph. It makes fewer wrong citations than text search alone. In the next three minutes, I will show how it works and how I tested it.",
 "Problem and why it is hard": "Papers are linked by citations, but most retrieval systems treat each abstract as a separate document. Take the question: which methods improve on this baseline? It needs two steps. First, find the baseline paper. Second, follow the links to the papers that cite it. Plain vector search often finds the first and misses the second. A wrong citation is costly, because readers trust the source.",
 "Approach": "My system has two stages. In the first stage, dense embeddings retrieve fifty candidate abstracts. In the second stage, I expand these candidates along citation edges, one hop away. Then a small cross encoder reranks the whole pool and keeps the top five. I kept the language model fixed in every run. So any change in the answers comes from retrieval only. This was a deliberate choice, because it lets me isolate one variable. The baseline is the same pipeline without the graph step. I also compared against a keyword method, to check that the baseline itself is not weak.",
 "Evidence": "I tested on two hundred questions that I labeled by hand. The metric is recall at five: does the right paper appear in the top five? The graph step raised recall at five from 0.61 to 0.72. I ran three random seeds, and the gain stayed between 0.09 and 0.13. I also removed the reranker from both systems. The graph step then added only about 0.05, so the expanded pool needs reranking. These numbers are for illustration only. In your own video, report your real results and say how many runs you made.",
 "My role and limits": "I wrote the retrieval code and the evaluation. My teammate built the data pipeline. The main limit is the test set: one main labeler and one subject area. A second labeler checked 50 questions, and kappa was 0.78. So I cannot yet claim the gain holds for other fields. My next step is to test on a different subject area.",
 "Close: why this residency": "I want to learn how to run research at a larger scale, with strong mentors and careful review. Retrieval for language models is where I can contribute most today. I also want to work on evaluation, because good tests matter as much as good models. Thank you for watching."}

BAD = ("Hello, my name is Nguyen Van A and I am a third year student at XYZ University and today I would like to present about my project which is a Hybrid Graph-RAG system leveraging LLM and KG with GNN-based reranking on the ArXiv dataset using a state-of-the-art SOTA pipeline that we implemented with FAISS and Neo4j and BM25 and also a cross-encoder, and because the problem of hallucination in LLM is very important in the era of AI, our system which is novel can improve the performance significantly in terms of NDCG and MRR and also recall, and I think that it is very promising for future work.")"""

UNDEF = ("""def undefined_acronyms(text):
    \"\"\"Trả về danh sách (đã sắp xếp) các từ viết hoa toàn bộ >= 2 chữ cái chưa được định nghĩa.
    Một viết tắt coi là đã định nghĩa nếu xuất hiện dạng 'Long Name (ACR)' hoặc 'ACR, meaning/which means/short for'.\"\"\"
    # TODO
    raise NotImplementedError""",
r"""def undefined_acronyms(text):
    out = []
    for m in sorted(set(re.findall(r"\b[A-Z]{2,}[a-z]?\b", text))):
        if re.search(r"\(" + m + r"\)", text) or re.search(m + r",? (meaning|which means|short for)", text):
            continue
        out.append(m)
    return out""")

ANALYSE = ("""def analyse(text, wpm=WPM):
    \"\"\"Trả về dict: words, sentences, seconds (làm tròn 1 chữ số), mean_sent, max_sent, flesch, fk_grade,
    long_sents (số câu > 25 từ), undefined (danh sách viết tắt chưa định nghĩa).
    Flesch = 206.835 - 1.015*(từ/câu) - 84.6*(âm tiết/từ);  FK = 0.39*(từ/câu) + 11.8*(âm tiết/từ) - 15.59.
    seconds là thời lượng ước tính khi nói với tốc độ wpm.\"\"\"
    # TODO
    raise NotImplementedError""",
r"""def analyse(text, wpm=WPM):
    w, s = words(text), sentences(text)
    nw, ns = len(w), len(s)
    syl = sum(syllables(x) for x in w)
    lens = [len(words(x)) for x in s]
    return dict(words=nw, sentences=ns, seconds=round(nw / wpm * 60, 1),
                mean_sent=round(nw / ns, 1), max_sent=max(lens),
                flesch=round(206.835 - 1.015 * nw / ns - 84.6 * syl / nw, 1),
                fk_grade=round(0.39 * nw / ns + 11.8 * syl / nw - 15.59, 1),
                long_sents=sum(l > 25 for l in lens), undefined=undefined_acronyms(text))""")

BUDGET = ("""def section_budget(sections, targets, wpm=WPM):
    \"\"\"Trả về DataFrame, mỗi dòng một đoạn: section, words, seconds, target_s, dev_pct = (seconds - target)/target * 100.\"\"\"
    # TODO
    raise NotImplementedError""",
r"""def section_budget(sections, targets, wpm=WPM):
    rows = []
    for name, target in targets:
        n = len(words(sections[name]))
        sec = n / wpm * 60
        rows.append(dict(section=name, words=n, seconds=round(sec, 1), target_s=target,
                         dev_pct=round(100 * (sec - target) / target, 1)))
    return pd.DataFrame(rows)""")

SCRIPT_CHECK = r"""good_text = " ".join(GOOD.values())
g, b = analyse(good_text), analyse(BAD)
print("GOOD:", g)
print("BAD :", b)
tab = section_budget(GOOD, TARGETS)
print(tab.to_string(index=False))
print("tổng giây:", round(tab.seconds.sum(), 1), "| lệch lớn nhất:", tab.dev_pct.abs().max(), "%")
print("ngân sách từ mỗi đoạn ở 140 từ/phút:", [round(t * WPM / 60) for _, t in TARGETS], "tổng", sum(round(t * WPM / 60) for _, t in TARGETS))
print("bản ghi một lần: 150 từ/phút ->", analyse(good_text, 150)["seconds"], "giây")
cat = analyse("The cat sat on the mat.")
print("Flesch của 'The cat sat on the mat.':", cat["flesch"])
assert abs(g["words"] - 421) <= 5 and abs(g["seconds"] - 180) <= 5, "đếm từ hoặc thời lượng chưa khớp mẫu"
assert g["undefined"] == [], "mẫu tốt không được có viết tắt chưa định nghĩa"
assert len(b["undefined"]) >= 8 and b["flesch"] < 0 and b["max_sent"] > 100
assert abs(cat["flesch"] - 116.1) < 0.5
assert 160 <= tab.seconds.sum() <= 190 and tab.dev_pct.abs().max() <= 30
print("Lab 1 ĐẠT ✔")"""

# =====================================================================================
# LAB 2 — linter bullet CV
# =====================================================================================
CV_DATA = r"""# Quy tắc theo hướng dẫn CMU SCS: động từ hành động + bối cảnh + kết quả có số; tối đa 2 dòng. Ngưỡng 95 ký tự/dòng chỉ là ước lượng.
VERBS = set('''built designed implemented trained evaluated reduced improved raised developed analyzed
proposed wrote led reproduced benchmarked fine-tuned deployed optimized compared tested ran labeled
collected measured replicated scaled authored presented achieved won ranked
created conducted annotated organized contributed constructed extended generated identified
maintained managed mentored refactored studied validated visualized'''.split())
WEAK_START = ("responsible for", "worked on", "helped", "participated", "involved in", "assisted")
SOFT = r"\b(teamwork|hard-working|passionate|fast learner)\b"
LINE_CHARS = 95

BAD_BULLETS = [
    "Responsible for working on a graph-based question answering project for a course.",
    "Passionate and hard-working student with teamwork skills who participated in NLP research.",
    "Helped my lab with various experiments on language models and many other tasks in the group including data cleaning and writing code.",
    "Implemented BM25 baseline.",
]
GOOD_BULLETS = [
    "Built hybrid dense + citation-graph retrieval over 100 000 arXiv abstracts; raised recall@5 from 0.61 to 0.72 (3 seeds)",
    "Labeled 200 evaluation questions; measured inter-annotator agreement (Cohen kappa 0.78) with a second labeler on 50 of them",
    "Reproduced a published reranker and found a 4-point gap to reported results; traced it to a tokenizer mismatch",
    "Ranked 12th of 150 teams at a national AI hackathon; led a team of 4 and shipped a demo in 36 hours",
]"""

LINT = ("""def lint_bullet(b, line_chars=LINE_CHARS):
    \"\"\"Trả về danh sách cờ (str). Các cờ: 'weak-verb', 'no-action-verb', 'no-number',
    'too-long(N lines)' khi N = ceil(len/line_chars) > 2, 'soft-skill-claim', và cờ phong cách 'style-trailing-period'.
    Gợi ý: 'BM25' có chữ số nhưng KHÔNG phải kết quả, nên chỉ tính chữ số đứng ĐẦU một từ (regex với ranh giới từ).\"\"\"
    # TODO
    raise NotImplementedError""",
r"""def lint_bullet(b, line_chars=LINE_CHARS):
    flags = []
    first = b.strip().split()[0].lower().strip(",")
    if b.lower().startswith(WEAK_START):
        flags.append("weak-verb")
    elif first not in VERBS:
        flags.append("no-action-verb")
    if not re.search(r"\b\d", b):                    # 'BM25': chữ số dính vào chữ, không có ranh giới từ -> không tính
        flags.append("no-number")
    lines = -(-len(b) // line_chars)
    if lines > 2:
        flags.append(f"too-long({lines} lines)")
    if re.search(SOFT, b, re.I):
        flags.append("soft-skill-claim")
    if b.strip().endswith("."):
        flags.append("style-trailing-period")        # cờ phong cách, không tính là lỗi nội dung
    return flags""")

CV_CHECK = r"""def real_flags(b):
    return [f for f in lint_bullet(b) if not f.startswith("style-")]

rows = [dict(nhom=n, ky_tu=len(b), co=", ".join(real_flags(b)) or "-", dau=b[:48]) for n, L in (("xấu", BAD_BULLETS), ("tốt", GOOD_BULLETS)) for b in L]
print(pd.DataFrame(rows).to_string(index=False))
long_b = BAD_BULLETS[2] + " " + BAD_BULLETS[2]
print("bullet 266 ký tự ->", real_flags(long_b))
assert all(len(real_flags(b)) >= 1 for b in BAD_BULLETS), "mỗi bullet xấu phải có >= 1 cờ"
assert all(real_flags(b) == [] for b in GOOD_BULLETS), "bullet tốt phải sạch"
assert "no-number" in real_flags("Implemented BM25 baseline."), "BM25 không phải kết quả"
assert any(f.startswith("too-long") for f in real_flags(long_b))
print("Lab 2 ĐẠT ✔")"""

# =====================================================================================
# LAB 3 — statement
# =====================================================================================
STMT_DATA = r"""# Quy tắc của SÁCH (không phải của chương trình): bản ngắn 250–550 từ.
CLICHES = ["since i was a child", "ever since i was young", "passion for ai", "in today's world", "in the era of",
           "eureka", "fascinated by the power of ai", "artificial intelligence is changing the world"]
FUTURE = re.compile(r"\b(i will|i plan to|i want to|my next step|i would like to|i aim to|i propose)\b", re.I)
PAST = re.compile(r"\b(i built|i wrote|i ran|i tested|i found|i labeled|i measured|i reproduced|i trained|i implemented|i compared|i designed)\b", re.I)

WEAK = '''Ever since I was a child, I have been fascinated by the power of AI. In today's world, artificial intelligence is changing the world and I am passionate about it. I am a hard-working student with many skills. I did some projects about NLP and I learned a lot. I hope the residency can help me to become a good researcher.'''

STRONG = '''Retrieval systems for scientific questions often return one relevant paper and miss the papers connected to it. I want to study whether citation structure can fix this, and how to measure the fix honestly.

My strongest work so far is a question answering system over arXiv abstracts. I built a two-stage retriever: dense search, then expansion along citation edges and reranking. On 200 questions that I labeled by hand, recall at five rose from 0.61 to 0.72 over three seeds. Without the reranker in both systems, the graph step added about half as much. I wrote the retrieval and evaluation code; a teammate built the data pipeline.

The main weakness is the test set: one main labeler, one subject area. I measured agreement with a second labeler on 50 questions and found kappa of 0.78, which is acceptable but not strong. This taught me that an evaluation is a research result of its own.

Earlier, I reproduced a published reranker and found a four point gap to the reported numbers. I traced it to a tokenizer mismatch. This is why I check baselines before I claim an improvement.

Before this project, I took the course on natural language processing at my university and wrote a small sentiment classifier. It was a normal course project, and I mention it only because it is where I first met the evaluation problems described above: a test set that is too small, and a score that moves with the random seed.

Next, I plan to test the method on a second field, and to study when graph expansion hurts, for example when citations are noisy. I want to work on retrieval for language models, where Qualcomm research teams publish on efficient LLMs. I would like to learn how to design experiments with mentors who review them critically.'''"""

CHECK = ("""def check(t, lo=250, hi=550):
    \"\"\"Trả về dict: words, in_range, paragraphs, first_person_per_100w (I/my/me trên 100 từ, 1 chữ số),
    cliches (list), past_claims, future_claims (đếm theo PAST và FUTURE), numbers (số con số),
    first_para_words, names_specific_fit (có nhắc Qualcomm, VinAI, 'research area(s)' hoặc 'team').\"\"\"
    # TODO
    raise NotImplementedError""",
r"""def check(t, lo=250, hi=550):
    ws = re.findall(r"[A-Za-z0-9'\-]+", t)
    paras = [p for p in t.split("\n\n") if p.strip()]
    low = t.lower()
    fp = len(re.findall(r"\b(i|my|me)\b", low))
    return dict(words=len(ws), in_range=lo <= len(ws) <= hi, paragraphs=len(paras),
                first_person_per_100w=round(100 * fp / len(ws), 1),
                cliches=[c for c in CLICHES if c in low],
                past_claims=len(PAST.findall(t)), future_claims=len(FUTURE.findall(t)),
                numbers=len(re.findall(r"\d+(?:\.\d+)?", t)),
                first_para_words=len(re.findall(r"[A-Za-z0-9'\-]+", paras[0])),
                names_specific_fit=bool(re.search(r"\b(Qualcomm|VinAI|research areas?|team)\b", t)))""")

STMT_CHECK = r"""w, s = check(WEAK), check(STRONG)
print("WEAK  :", w)
print("STRONG:", s)
paras = STRONG.split("\n\n")
print("từ mỗi đoạn của STRONG:", [len(re.findall(r"[A-Za-z0-9'\-]+", p)) for p in paras])
cut = "\n\n".join(p for i, p in enumerate(paras) if i != 4)       # bỏ đoạn lạc đề (đoạn 5)
c = check(cut)
print("STRONG bỏ đoạn 5:", {k: c[k] for k in ("words", "in_range", "paragraphs", "first_person_per_100w", "past_claims", "future_claims", "numbers")})
assert len(w["cliches"]) == 4 and w["numbers"] == 0 and w["past_claims"] == 0 and not w["in_range"]
assert s["cliches"] == [] and s["past_claims"] == 5 and s["future_claims"] == 4 and s["numbers"] == 5
assert s["in_range"] and s["first_para_words"] == 34 and abs(s["first_person_per_100w"] - 6.0) < 0.15
assert c["words"] < s["words"] and c["past_claims"] == s["past_claims"] and c["future_claims"] == s["future_claims"]
assert not c["in_range"], "bỏ đoạn lạc đề thì statement tụt dưới sàn 250 từ của sách: phải thêm bằng chứng, không độn giai thoại"
print("Lab 3 ĐẠT ✔")"""

# =====================================================================================
# LAB 4 — thống kê bản ghi
# =====================================================================================
TALK_DATA = r"""# BẢN GHI GIẢ LẬP (seed 0), không phải giọng người thật: nói nhanh lúc đầu, một quãng ngắt dài, nhiều 'um'.
# Định dạng (bắt đầu_s, kết_thúc_s, từ) giống đầu ra của công cụ nhận dạng giọng nói có mốc thời gian từng từ.
FILLERS = {"um", "uh", "like"}

def synth(seed=0, n=440):
    rng = np.random.default_rng(seed)
    out, t = [], 0.0
    for i in range(n):
        rate = 190 if i < 110 else (140 if i < 330 else 110)
        dur = 60 / rate * rng.uniform(0.7, 1.3)
        w = "um" if rng.random() < 0.045 else "word"
        out.append((t, t + dur, w))
        t += dur
        gap = rng.exponential(0.03)
        if i == 250:
            gap += 3.2
        t += gap
    return out

TALK = synth()
print(len(TALK), "từ; kết thúc ở giây", round(TALK[-1][1], 1))"""

TALK_FN = ("""def talk_stats(words_ts, win=30.0):
    \"\"\"words_ts: list (start, end, word). Trả về dict: total_s, words, wpm, window_wpm (list, làm tròn),
    wpm_cv (độ lệch chuẩn tổng thể / trung bình của các cửa sổ, 3 chữ số), long_pauses (list các quãng ngắt >= 1 s, 1 chữ số),
    fillers, filler_per_100.
    Lưu ý: cửa sổ cuối thường NGẮN hơn win; tốc độ phải chia cho độ dài thật của nó, và bỏ cửa sổ cuối nếu ngắn hơn win/3.\"\"\"
    # TODO
    raise NotImplementedError""",
r"""def talk_stats(words_ts, win=30.0):
    t_start, t_end = words_ts[0][0], words_ts[-1][1]
    total, n = t_end - t_start, len(words_ts)
    gaps = [words_ts[i + 1][0] - words_ts[i][1] for i in range(n - 1)]
    fill = sum(w[2] in FILLERS for w in words_ts)
    rates, t0 = [], t_start
    while t0 < t_end:
        length = min(win, t_end - t0)                      # cửa sổ cuối ngắn: chia cho độ dài thật
        if length < win / 3 and rates:
            break
        rates.append(sum(t0 <= w[0] < t0 + win for w in words_ts) / length * 60)
        t0 += win
    return dict(total_s=round(total, 1), words=n, wpm=round(n / total * 60, 1),
                window_wpm=[round(x) for x in rates], wpm_cv=round(st.pstdev(rates) / st.mean(rates), 3),
                long_pauses=[round(g, 1) for g in gaps if g >= 1.0], fillers=fill, filler_per_100=round(100 * fill / n, 1))""")

TALK_CHECK = r"""r = talk_stats(TALK)
print(r)
naive = len([w for w in TALK if w[0] >= 180]) / 30 * 60        # nếu chia nhầm cho 30 s cố định
print("cửa sổ cuối nếu chia nhầm cho 30 s:", round(naive), "từ/phút (sai); chia đúng:", r["window_wpm"][-1])
assert abs(r["wpm"] - 130.0) <= 0.5 and r["words"] == 440
assert len(r["long_pauses"]) == 1 and abs(r["long_pauses"][0] - 3.2) <= 0.1
assert r["fillers"] == 20
assert len(r["window_wpm"]) == 7 and r["window_wpm"][0] > 160 and r["window_wpm"][-1] > naive + 15
assert r["window_wpm"][0] - r["window_wpm"][-1] > 40
print("Lab 4 ĐẠT ✔")"""

TALK_CSV_CELL = r"""import io
def read_words_csv(path_or_buffer):
    '''Đọc CSV có cột start,end,word (giây, giây, từ): đúng dạng bạn xuất ra từ Whisper hoặc faster-whisper
    chạy với word_timestamps=True. Trả về list (start, end, word) cho talk_stats; chữ thường, bỏ dấu câu.'''
    df = pd.read_csv(path_or_buffer)
    return [(float(a), float(b), re.sub(r"[^\w']", "", str(c).strip().lower())) for a, b, c in zip(df["start"], df["end"], df["word"])]

buf = io.StringIO()                                    # thử khứ hồi: ghi TALK ra CSV trong bộ nhớ rồi đọc lại
pd.DataFrame(TALK, columns=["start", "end", "word"]).to_csv(buf, index=False)
buf.seek(0)
r2 = talk_stats(read_words_csv(buf))
print(r2["words"], "từ;", r2["wpm"], "từ/phút;", r2["window_wpm"])
assert r2["words"] == 440 and abs(r2["wpm"] - 130.0) <= 0.5 and r2["window_wpm"] == talk_stats(TALK)["window_wpm"]
# Bước cuối (không bắt buộc): talk_stats(read_words_csv("ban_ghi_cua_toi.csv"))
print("Lab 4 (đọc CSV) ĐẠT ✔")"""

# =====================================================================================
# LAB 5 — nhất quán + lịch ngược (dữ liệu mẫu RIÊNG, khác hồ sơ săn bug)
# =====================================================================================
CONS_DATA = r"""# MẪU B (số giả). Hồ sơ nhỏ để thử công cụ. Hồ sơ dùng cho bài săn bug nằm ở phần cuối notebook và KHÁC mẫu này.
CV_B = '''Education: B.Sc. Computer Science, GPA 3.5/4.0 (expected 06/2028)
Trained a sentiment classifier on 8 000 product reviews; reached accuracy 0.84 (5 seeds)
Led a team of 3 at a 24-hour hackathon; shipped a demo'''
STATEMENT_B = '''I trained a sentiment classifier on product reviews. Accuracy reached 0.86 over five seeds. I also led a team of 3 at a hackathon.'''
SCRIPT_B = '''My classifier reached accuracy 0.84 on 8 000 reviews, over 5 seeds.'''
PROJECTS_B = [("Course project", date(2026, 2, 1), date(2026, 5, 30)), ("Hackathon", date(2026, 5, 10), date(2026, 5, 11)),
              ("Reading group", date(2026, 6, 1), date(2026, 8, 31))]

DEADLINE = date(2027, 11, 12)      # GIẢ ĐỊNH, chưa xác minh: suy từ cửa sổ Batch 17 (12/10–12/11/2026)
STEPS = [("Chốt 2 người viết thư, hỏi trực tiếp", 120), ("Xin bảng điểm tiếng Anh (gợi ý của sách)", 84), ("CV bản 1 xong, nhờ 2 người đọc", 70),
         ("Statement bản 1 xong", 63), ("Gửi họ gói tài liệu (CV, statement, brag sheet)", 56),
         ("Quay video lần 1 (tự chấm)", 49), ("Quay video lần 2 sau phản hồi", 28), ("Nhắc thư lần 1", 21),
         ("Hồ sơ đủ, tự nộp thử trên cổng", 14), ("Nhắc thư lần cuối", 7), ("Nộp thật", 3)]
WD = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]"""

CONS_FN = ("""def nums(text):
    \"\"\"Tập các con số (str, bỏ khoảng trắng nghìn) xuất hiện trong text: '100 000' -> '100000', '0.61', '2.9'.
    Chữ số dính vào chữ ('BM25', '12th') không tính. Số viết bằng chữ ('five') KHÔNG bắt được — đó là giới hạn của công cụ.\"\"\"
    # TODO
    raise NotImplementedError


def overlaps(projects):
    \"\"\"projects: list (tên, ngày_bắt_đầu, ngày_kết_thúc). Trả về list cặp (tên_a, tên_b) chồng ngày,
    trừ khi một trong hai tên có chữ 'concurrent' (đã chú thích).\"\"\"
    # TODO
    raise NotImplementedError


def consistency_report(cv, statement, script, projects):
    \"\"\"Trả về dict: stmt_not_cv (set), script_not_cv (set), cv_not_script (set, chỉ để tham khảo),
    gpa_without_scale (bool: có kiểu 'GPA 3.5' mà không có '/4.0' hoặc '/10' ngay sau), overlaps (list).\"\"\"
    # TODO
    raise NotImplementedError


def timeline(deadline, steps):
    \"\"\"Trả về DataFrame cột date, thu (T2..CN), days_before, step; sắp xếp theo date tăng dần.\"\"\"
    # TODO
    raise NotImplementedError""",
r"""NUM = re.compile(r"(?<![\w.])\d[\d ]*(?:\.\d+)?(?![\w])")

def nums(text):
    return {m.group(0).strip().replace(" ", "") for m in NUM.finditer(text)}


def overlaps(projects):
    out = []
    for i, a in enumerate(projects):
        for b in projects[i + 1:]:
            if "concurrent" in (a[0] + b[0]).lower():
                continue
            if a[1] <= b[2] and b[1] <= a[2]:
                out.append((a[0], b[0]))
    return out


def consistency_report(cv, statement, script, projects):
    c, s, v = nums(cv), nums(statement), nums(script)
    return dict(stmt_not_cv=s - c, script_not_cv=v - c, cv_not_script=c - v,
                gpa_without_scale=bool(re.search(r"GPA\s*:?\s*\d+(?:\.\d+)?(?![\d.]|\s*/)", cv)),
                overlaps=overlaps(projects))


def timeline(deadline, steps):
    rows = [dict(date=deadline - timedelta(days=d), thu=WD[(deadline - timedelta(days=d)).weekday()],
                 days_before=d, step=name) for name, d in steps]
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)""")

CONS_CHECK = r"""rep = consistency_report(CV_B, STATEMENT_B, SCRIPT_B, PROJECTS_B)
for k, v in rep.items():
    print(f"{k:18s}", v)
tl = timeline(DEADLINE, STEPS)
print(tl.to_string(index=False))
print("từ lúc hỏi người viết đến hạn:", (DEADLINE - tl.date.iloc[0]).days, "ngày =", round((DEADLINE - tl.date.iloc[0]).days / 7, 1), "tuần")
print("nums('100 000 abstracts, BM25, 12th, 0.61'):", sorted(nums("100 000 abstracts, BM25, 12th, 0.61")))
assert rep["stmt_not_cv"] == {"0.86"} and rep["script_not_cv"] == set()
assert rep["gpa_without_scale"] is False and rep["overlaps"] == [("Course project", "Hackathon")]
assert consistency_report("GPA 3.5", "", "", [])["gpa_without_scale"] is True
assert consistency_report("GPA 3.45/4.0", "", "", [])["gpa_without_scale"] is False
assert nums("100 000 abstracts, BM25, 12th, 0.61") == {"100000", "0.61"}
assert overlaps([("A", date(2026, 1, 1), date(2026, 3, 1)), ("B (concurrent)", date(2026, 2, 1), date(2026, 2, 5))]) == []
row = tl[tl.days_before == 56].iloc[0]
assert row.date == date(2027, 9, 17) and tl[tl.days_before == 120].iloc[0].date == date(2027, 7, 15)
assert tl.date.is_monotonic_increasing
print("Lab 5 ĐẠT ✔")"""

# =====================================================================================
# SĂN BUG — hồ sơ giả (KHÔNG chú thích chỗ lỗi trong bản lab)
# =====================================================================================
BUG_DATA = r"""# GÓI HỒ SƠ GIẢ (số giả). Triệu chứng: mỗi file đọc xuôi, nhưng người chấm thứ hai báo hồ sơ có chỗ không khớp.
CV = '''Education: B.Sc. Artificial Intelligence, GPA 3.4 (expected 06/2028)
Built hybrid dense + citation-graph retrieval over 100 000 arXiv abstracts; raised recall@5 from 0.61 to 0.72 (3 seeds)
Labeled 200 evaluation questions; measured agreement with a second labeler (Cohen kappa 0.78)
Reproduced a published reranker and found a 4-point gap to reported results'''
STATEMENT = '''On 200 questions labeled by hand, recall at 5 rose from 0.61 to 0.75 over three seeds. I measured agreement with a second labeler on 50 questions and found kappa of 0.78. I reproduced a reranker and found a 4 point gap.'''
SCRIPT = '''The graph step raised recall at 5 from 0.61 to 0.72. I tested on 200 questions that I labeled alone, so there was no second labeler.'''
PROJECTS = [("Retrieval project", date(2026, 3, 1), date(2026, 9, 30)),
            ("Hackathon", date(2026, 9, 15), date(2026, 9, 17)),
            ("Reranker reproduction", date(2026, 8, 1), date(2026, 8, 20))]"""

BUG_FIX = ("""# TODO: đọc chéo ba tài liệu, dùng công cụ của Lab 5 để hỗ trợ, tìm các chỗ không khớp, rồi sửa vào các biến FIXED_*.
# Quy tắc: sửa bằng cách đưa dữ liệu đúng vào; không xóa câu của statement.
FIXED_CV, FIXED_STATEMENT, FIXED_SCRIPT, FIXED_PROJECTS = CV, STATEMENT, SCRIPT, list(PROJECTS)""",
r"""print(consistency_report(CV, STATEMENT, SCRIPT, PROJECTS))      # công cụ chỉ thấy một phần; phần còn lại phải đọc bằng mắt
FIXED_CV = (CV.replace("GPA 3.4 (", "GPA 3.4/4.0 (")                                          # lỗi: GPA thiếu thang
              .replace("second labeler (Cohen", "second labeler on 50 questions (Cohen"))     # lỗi: số 50 chỉ có ở statement
FIXED_STATEMENT = STATEMENT.replace("0.75", "0.72")                                          # lỗi: CV, kịch bản đều ghi 0.72
FIXED_SCRIPT = SCRIPT.replace("that I labeled alone, so there was no second labeler",
                              "that I labeled myself; a second labeler checked a sample")    # lỗi ngữ nghĩa: công cụ không bắt được
FIXED_PROJECTS = [("Retrieval project", date(2026, 3, 1), date(2026, 9, 30)),
                  ("Hackathon (concurrent)", date(2026, 9, 15), date(2026, 9, 17)),          # cờ cần giải thích, không phải lỗi chắc chắn
                  ("Reranker reproduction (concurrent)", date(2026, 8, 1), date(2026, 8, 20))]""")

BUG_CHECK = r"""PAIRS = [(r"second (labeler|annotator)", r"labeled (it |them )?(alone|by myself)|\balone\b.{0,30}label|only (one )?(labeler|annotator|person)|single (labeler|annotator)|no second|not (yet )?(measured|checked)|next step is to measure agreement"),      # các cặp tuyên bố loại trừ nhau
         (r"three seeds|3 seeds", r"single seed|one seed|1 seed"),
         (r"reproduced", r"did not reproduce|could not reproduce")]

def ho_so_nhat_quan(cv, statement, script, projects):
    rep = consistency_report(cv, statement, script, projects)
    if rep["stmt_not_cv"] or rep["script_not_cv"] or rep["gpa_without_scale"] or rep["overlaps"]:
        return False
    corpus = "\n".join((cv, statement, script)).lower()
    return not any(re.search(a, corpus) and re.search(b, corpus) for a, b in PAIRS)

ok = ho_so_nhat_quan(FIXED_CV, FIXED_STATEMENT, FIXED_SCRIPT, FIXED_PROJECTS)
print("Hồ sơ:", "nhất quán" if ok else "không nhất quán")
assert ok, "Hồ sơ chưa nhất quán."
assert len(words(FIXED_STATEMENT)) >= 0.95 * len(words(STATEMENT)), "Nội dung bị rút ngắn quá nhiều."
assert len(words(FIXED_CV)) >= 0.95 * len(words(CV)), "Nội dung bị rút ngắn quá nhiều."
print("Săn bug ĐẠT ✔")"""

# Bài "bug code" tùy chọn: hai hàm chạy không báo lỗi nhưng kết quả sai
BUGCODE = ("""# (Tùy chọn) Hai hàm dưới chạy không báo lỗi nhưng cho kết quả sai. Tìm lỗi, sửa tại chỗ, rồi chạy ô kiểm tra bên dưới.
def duration_s(text, wpm=WPM):
    return round(len(words(text)) / wpm, 1)

def acronyms_to_define(text):
    return sorted(set(re.findall(r"\\b[A-Z]{2,}[a-z]?\\b", text)))""",
r"""# (Tùy chọn) Bản đã sửa của hai hàm.
def duration_s(text, wpm=WPM):
    return round(len(words(text)) / wpm * 60, 1)                    # lỗi 1: thiếu * 60 nên ra phút, không phải giây

def acronyms_to_define(text):
    out = []
    for m in sorted(set(re.findall(r"\b[A-Z]{2,}[a-z]?\b", text))):
        if re.search(r"\(" + m + r"\)", text) or re.search(m + r",? (meaning|which means|short for)", text):
            continue                                                 # lỗi 2: chữ đã định nghĩa trong ngoặc vẫn bị đếm
        out.append(m)
    return out""")

BUGCODE_CHECK = r"""demo = "We use Retrieval-Augmented Generation (RAG) with BM25. RAG then feeds a large language model (LLM) and a GNN."
good_text = " ".join(GOOD.values())
assert abs(duration_s(good_text) - 180.4) <= 3, "Thời lượng chưa khớp mẫu."
assert acronyms_to_define(demo) == ["GNN"], "Danh sách viết tắt chưa khớp mẫu."
assert acronyms_to_define(good_text) == [], "Danh sách viết tắt chưa khớp mẫu."
print("Săn bug code ĐẠT ✔")"""

# =====================================================================================
# Phụ lục kiểm số của sách (chỉ trong loigiai)
# =====================================================================================
APPENDIX = r"""# PHỤ LỤC (chỉ có trong bản lời giải): kiểm các số được dùng trong phần Lời giải và Mini-test của sách.
print("Đổi từ/giây:", {m: round(m * 140) for m in (3, 4.5, 5)}, "| 420 từ ở 150 từ/phút:", round(420 / 150 * 60), "giây")
print("tích lũy ngân sách chia 630 từ theo tỉ lệ 20:25:45:45:25:20 =", [round(630 * x / 180, 1) for x in (20, 25, 45, 45, 25, 20)])
print("tăng tương đối 0.61 -> 0.72:", round((0.72 - 0.61) / 0.61, 4), "| tuyệt đối:", round(0.72 - 0.61, 2))
for dl, label in ((date(2027, 11, 15), "mini-test câu 3"), (date(2027, 11, 20), "bài tính nhanh 5")):
    print(label, dl, WD[dl.weekday()], {d: str(dl - timedelta(days=d)) + " " + WD[(dl - timedelta(days=d)).weekday()] for d in (120, 56, 21, 7)})
print("GPA/thang:", {x: round(x[0] / x[1], 3) for x in ((3.4, 4.0), (3.1, 4.0), (7.9, 10), (3.2, 4.0), (8.0, 10))})
print("trang 11 in, lề 0.5 in, dòng 12 pt:", (11 - 1.0) * 72 / 12, "dòng")
print("trang 11 in, lề 0.5 in, dòng 13 pt:", round((11 - 1.0) * 72 / 13, 1), "dòng")
print("từ ngoài khối Lời giải: 504 từ ở 140 -> ", round(504 / 140 * 60, 1), "giây; ở 150 ->", round(504 / 150 * 60, 1), "giây")
print("câu 56 từ chia ra tối thiểu", math.ceil(56 / 25), "câu <= 25 từ")
print("77 từ trong 30 giây =", round(77 / 30 * 60), "từ/phút; 360 từ có 27 I/my/me =", round(100 * 27 / 360, 1), "trên 100 từ")
print("tốc độ trung bình của 7 cửa sổ trong bài tính nhanh:", round(sum((172, 140, 136, 118, 122, 106, 112)) / 7, 1))
print("hệ số biến thiên của 4 cửa sổ (120, 150, 90, 140):", round(st.pstdev([120, 150, 90, 140]) / st.mean([120, 150, 90, 140]), 3))
print("Flesch của câu 56 từ 80 âm tiết 1 câu?  ->", round(206.835 - 1.015 * 56 - 84.6 * 80 / 56, 1))

MINI_SAMPLE = '''Ever since I was a child, I have been fascinated by computers. In today's world, AI is everywhere, and I am passionate about it. I am a hard-working student, and my team and I built a RAG system with LLM, GNN and FAISS that reached SOTA results on our dataset, which is very novel, and because it is so good the performance improved significantly in terms of NDCG and MRR. We did many experiments and we think the results are promising. The model is much better than before. My teammate did most of the evaluation and I helped. I learned a lot from this project and from many courses at my university. I hope the residency will give me a chance to become a good researcher and to work on many exciting topics in AI.'''
print("mini-test câu 5: số từ của mẫu =", len(words(MINI_SAMPLE)), "| câu =", len(sentences(MINI_SAMPLE)), "| viết tắt chưa định nghĩa:", undefined_acronyms(MINI_SAMPLE))
sm = check(MINI_SAMPLE)
print("   cliches:", sm["cliches"], "| con số:", sm["numbers"], "| ngôi thứ nhất/100 từ:", sm["first_person_per_100w"], "| lint 'Helped' :", [f for f in lint_bullet('Helped with the evaluation of our system') if not f.startswith('style-')])
# Mỗi lỗi gài, đưa RIÊNG vào bản đã sửa, phải làm ô kiểm tra của săn bug báo "không nhất quán".
cases = {
    "GPA thiếu thang": (FIXED_CV.replace("GPA 3.4/4.0 (", "GPA 3.4 ("), FIXED_STATEMENT, FIXED_SCRIPT, FIXED_PROJECTS),
    "số 50 chỉ có ở statement": (FIXED_CV.replace(" on 50 questions", ""), FIXED_STATEMENT, FIXED_SCRIPT, FIXED_PROJECTS),
    "0.75 lệch 0.72": (FIXED_CV, FIXED_STATEMENT.replace("0.72", "0.75"), FIXED_SCRIPT, FIXED_PROJECTS),
    "tuyên bố gán nhãn một mình": (FIXED_CV, FIXED_STATEMENT, SCRIPT, FIXED_PROJECTS),
    "kịch bản nói chưa đo (cách viết khác)": (FIXED_CV, FIXED_STATEMENT, FIXED_SCRIPT + " Only one person labeled it, and my next step is to measure agreement with a second labeler.", FIXED_PROJECTS),
    "kịch bản nói 'chưa đo' (không có chữ second)": (FIXED_CV, FIXED_STATEMENT, FIXED_SCRIPT + " Agreement is not measured yet.", FIXED_PROJECTS),
    "dự án chồng ngày chưa giải thích": (FIXED_CV, FIXED_STATEMENT, FIXED_SCRIPT, PROJECTS),
}
# Bộ hồ sơ MẪU TỐT của chính sách (bullet CV tốt + statement mạnh + kịch bản tốt) không được tự mâu thuẫn về người gán nhãn thứ hai.
good_corpus = "\n".join(GOOD_BULLETS + [STRONG, " ".join(GOOD.values())]).lower()
assert not any(re.search(a_, good_corpus) and re.search(b_, good_corpus) for a_, b_ in PAIRS), "bộ mẫu tốt tự mâu thuẫn"
assert all("0.78" in t for t in (GOOD_BULLETS[1], STRONG, GOOD["My role and limits"])) and all("50" in t for t in (GOOD_BULLETS[1], STRONG, GOOD["My role and limits"]))
print("Bộ mẫu tốt: nhất quán về người gán nhãn thứ hai (kappa 0.78, 50 câu ở cả CV, statement, kịch bản)")
assert ho_so_nhat_quan(FIXED_CV, FIXED_STATEMENT, FIXED_SCRIPT, FIXED_PROJECTS)
for name, args in cases.items():
    caught = not ho_so_nhat_quan(*args)
    print("  đưa lại lỗi", name.ljust(34), "->", "bị bắt" if caught else "LỌT")
    assert caught, name
print("Bản gốc (chưa sửa) được xếp là", "nhất quán" if ho_so_nhat_quan(CV, STATEMENT, SCRIPT, PROJECTS) else "không nhất quán")
print("Phụ lục ĐẠT ✔")"""


def build(solution: bool):
    pick = (lambda pair: pair[1]) if solution else (lambda pair: pair[0])
    nb = nbf.v4.new_notebook()
    head = "Lời giải" if solution else "Bài lab"
    cells = [
        md(f"# Chương 13 — {head}: Hồ sơ và video presentation\n\n"
           + ("Bản lời giải đã chạy kiểm. Chỉ mở sau khi đã tự làm `lab.ipynb`."
              if solution else
              "Viết code vào các ô có `# TODO`, rồi chạy ô kiểm tra ngay bên dưới. Ô kiểm tra in **ĐẠT ✔** là xong. "
              "Bí quá 20 phút mới mở `loigiai.ipynb`.")
           + "\n\nMọi mẫu CV, statement, kịch bản và bản ghi trong notebook này là **mẫu minh họa tự soạn, số liệu là số giả**. "
             "Bước cuối mỗi lab (không bắt buộc): dán dữ liệu thật của bạn vào và đọc kết quả."),
        code(SETUP),
        md("## Lab 1 — Kiểm kịch bản video\nĐếm từ, đổi ra giây, độ dễ đọc (Flesch), câu dài, viết tắt chưa định nghĩa, và so từng đoạn với ngân sách giây. "
           "Câu hỏi: vì sao Flesch dùng để so hai bản hơn là chấm điểm tuyệt đối?"),
        code(SCRIPT_DATA), code(pick(UNDEF)), code(pick(ANALYSE)), code(pick(BUDGET)), code(SCRIPT_CHECK),
        md("## Lab 2 — Linter bullet CV\nĐộng từ hành động, có số, tối đa 2 dòng, không tự khen. Chú ý: `BM25` có chữ số nhưng không phải kết quả."),
        code(CV_DATA), code(pick(LINT)), code(CV_CHECK),
        md("## Lab 3 — Phân tích research statement\nĐếm từ, ngôi thứ nhất, câu sáo, việc đã làm so với việc sẽ làm, số lượng con số. "
           "Sau đó thử bỏ một đoạn và xem số liệu thay đổi thế nào."),
        code(STMT_DATA), code(pick(CHECK)), code(STMT_CHECK),
        md("## Lab 4 — Thống kê bản ghi giọng nói\nTừ/phút trung bình, tốc độ theo cửa sổ 30 giây, quãng ngắt, từ đệm. "
           "Câu hỏi: vì sao trung bình 130 từ/phút chưa đủ để kết luận bản ghi ổn?"),
        code(TALK_DATA), code(pick(TALK_FN)), code(TALK_CHECK),
        md("**Lab 4, bước cuối: bản ghi thật.** Chạy Whisper hoặc faster-whisper với `word_timestamps=True` trên bản quay của bạn, xuất CSV ba cột `start,end,word`, rồi đọc bằng hàm dưới (ô này chạy được khi chưa có CSV nhờ phép thử khứ hồi)."), code(TALK_CSV_CELL),
        md("## Lab 5 — Chéo kiểm nhất quán và lịch ngược\nCon số nào có ở statement hoặc kịch bản mà CV không có; GPA có thang chưa; dự án nào chồng ngày. "
           "Rồi lập lịch ngược từ hạn nộp giả định. Hạn 12/11/2027 là **giả định**, không phải thông tin đã xác minh."),
        code(CONS_DATA), code(pick(CONS_FN)), code(CONS_CHECK),
        md("## Săn bug — gói hồ sơ \"đọc xuôi nhưng không khớp\"\nDùng công cụ của Lab 5 và đọc chéo gói hồ sơ giả dưới đây. Có **bốn** lỗi gài sẵn. "
           "Công cụ chỉ thấy một phần, và có thể báo cả cờ mà thực ra chỉ cần giải thích; bạn tự phán xét. "
           "Sửa vào các biến `FIXED_*`; ô kiểm tra bên dưới chỉ in ĐẠT khi hồ sơ nhất quán."),
        code(BUG_DATA), code(pick(BUG_FIX)), code(BUG_CHECK),
        md("## Săn bug code (tùy chọn)\nHai hàm nhỏ dưới đây chạy không báo lỗi nhưng cho kết quả sai. Sửa tại chỗ rồi chạy ô kiểm tra."),
        code(pick(BUGCODE)), code(BUGCODE_CHECK),
    ]
    if solution:
        cells += [md("## Phụ lục — kiểm số của sách\nChỉ có trong bản lời giải: tính lại các số dùng ở phần Lời giải và Mini-test."), code(APPENDIX)]
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    name = "loigiai.ipynb" if solution else "lab.ipynb"
    nbf.write(nb, OUT / name)
    return OUT / name


if __name__ == "__main__":
    print(build(False), build(True))
