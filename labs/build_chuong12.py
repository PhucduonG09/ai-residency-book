"""Sinh hai notebook cho lab chương 12: lab.ipynb (có TODO) và loigiai.ipynb (lời giải).

Chạy lại:  python labs/build_chuong12.py
Dữ liệu hoàn toàn tự dựng (không tải mạng), seed cố định, PyTorch CPU 1 luồng; loigiai.ipynb chạy dưới 2 phút.
Các thí nghiệm là đồ chơi để lộ cơ chế, không xếp hạng phương pháp. Assert dùng khoảng vì kết quả phụ thuộc phiên bản thư viện.
"""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "chuong12"
OUT.mkdir(parents=True, exist_ok=True)
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = r'''import time, math, re
from collections import Counter
import numpy as np
import torch, torch.nn.functional as F
torch.set_num_threads(1)
np.set_printoptions(precision=4, suppress=True)
T0 = time.time()

def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())

def ranking(scores, rng=None):
    """Xếp giảm dần theo điểm. Hòa điểm: theo chỉ số nhỏ (rng=None, ổn định nhưng THIÊN VỊ vì chỉ số là thông tin có thật)
    hoặc theo thứ tự ngẫu nhiên (truyền rng). Điểm được làm tròn 12 chữ số để hai điểm bằng nhau về toán học mà lệch sai số float vẫn coi là hòa."""
    s = np.asarray(scores, float)
    if rng is None:
        return list(np.lexsort((np.arange(len(s)), -s)))
    return list(np.lexsort((rng.permutation(len(s)), -np.round(s, 12))))

def hit_at_k(ranked, rel, k):
    return float(len(set(ranked[:k]) & set(rel)) > 0)

def rr(ranked, rel, k=None):
    for i, d in enumerate(ranked[:k] if k else ranked):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0

class BM25:
    """BM25 bản Lucene, bản nhanh dùng cho các lab 2-5 (Lab 1 bạn tự viết lại và đối chiếu với lớp này)."""
    def __init__(self, docs_tokens, k1=1.2, b=0.75):
        self.k1, self.b, self.N = k1, b, len(docs_tokens)
        self.tf = [Counter(d) for d in docs_tokens]
        self.dl = np.array([len(d) for d in docs_tokens], float)
        self.avgdl = self.dl.mean()
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        self.df = df
    def idf(self, t):
        n = self.df.get(t, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))
    def score(self, q_tokens):
        s = np.zeros(self.N)
        for t in sorted(set(q_tokens)):  # sorted: cộng số thực theo thứ tự cố định, kết quả tái lập được
            if t not in self.df:
                continue
            idf = self.idf(t)
            for i, c in enumerate(self.tf):
                f = c.get(t, 0)
                if f:
                    s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return s'''

# ------------------------------------------------------------------ Lab 1
L1 = (r'''def bm25_idf(N, n):
    """idf bản Lucene cho từ có mặt trong n trong N đoạn."""
    # TODO
    raise NotImplementedError("Viết hàm này (Lab 1)")

def bm25_tf_part(f, dl, avgdl, k1=1.2, b=0.75):
    """Phần tf đã bão hòa và chuẩn hóa độ dài của BM25 (chưa nhân idf)."""
    # TODO
    raise NotImplementedError("Viết hàm này (Lab 1)")

def bm25_scores(q, docs, k1=1.2, b=0.75):
    """Điểm BM25 của truy vấn q (list từ) với từng đoạn trong docs (list các list từ)."""
    # TODO
    raise NotImplementedError("Viết hàm này (Lab 1)")

def recall_at_k(ranked, rel, k):
    """Tỉ lệ đoạn liên quan nằm trong k đầu."""
    # TODO
    raise NotImplementedError("Viết hàm này (Lab 1)")

def mrr(ranked, rel):
    """Nghịch đảo hạng của đoạn liên quan đầu tiên (0 nếu không có)."""
    # TODO
    raise NotImplementedError("Viết hàm này (Lab 1)")

def ndcg_at_k(ranked, gains, k):
    """gains: dict đoạn -> mức liên quan (số nguyên, thiếu = 0). Độ lợi 2^g - 1, chiết khấu theo log2(hạng + 1)."""
    # TODO
    raise NotImplementedError("Viết hàm này (Lab 1)")

def rrf(rankings, k=60):
    """Gộp nhiều danh sách xếp hạng (mỗi danh sách là list id, phần tử đầu = hạng 1). Trả dict id -> điểm."""
    # TODO
    raise NotImplementedError("Viết hàm này (Lab 1)")''',
r'''def bm25_idf(N, n):
    return math.log(1 + (N - n + 0.5) / (n + 0.5))

def bm25_tf_part(f, dl, avgdl, k1=1.2, b=0.75):
    return f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))

def bm25_scores(q, docs, k1=1.2, b=0.75):
    N = len(docs); avgdl = sum(len(d) for d in docs) / N
    out = []
    for d in docs:
        c = Counter(d); s = 0.0
        for t in sorted(set(q)):
            if c.get(t, 0):
                n = sum(1 for x in docs if t in x)
                s += bm25_idf(N, n) * bm25_tf_part(c[t], len(d), avgdl, k1, b)
        out.append(s)
    return np.array(out)

def recall_at_k(ranked, rel, k):
    return len(set(ranked[:k]) & set(rel)) / len(rel)

def mrr(ranked, rel):
    for i, d in enumerate(ranked):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(ranked, gains, k):
    dcg = sum((2 ** gains.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def rrf(rankings, k=60):
    sc = {}
    for r in rankings:
        for pos, d in enumerate(r):
            sc[d] = sc.get(d, 0) + 1.0 / (k + pos + 1)
    return sc''')

L1_CHECK = r'''docs = [tok(s) for s in ["graph neural networks for citation graph classification",
                         "retrieval augmented generation with dense retrieval",
                         "dense retrieval and sparse retrieval for question answering",
                         "graph retrieval augmented generation builds a knowledge graph"]]
q = ["graph", "retrieval"]
assert abs(bm25_idf(4, 2) - 0.6931) < 1e-4 and abs(bm25_idf(4, 3) - 0.3567) < 1e-4, "idf sai"
sc = bm25_scores(q, docs)
print("điểm 4 đoạn:", sc)
assert np.allclose(sc, [0.9624, 0.5154, 0.4766, 1.2683], atol=2e-4), "điểm BM25 không khớp ví dụ tay"
assert ranking(sc) == [3, 0, 1, 2], "thứ hạng phải là d4 > d1 > d2 > d3"
bm = BM25(docs)
assert np.allclose(bm.score(q), sc), "bản của bạn phải khớp lớp BM25 trong ô đầu"
tf = [bm25_tf_part(f, 7.25, 7.25) for f in (1, 2, 5, 10, 50)]
print("phần tf:", np.round(tf, 3))
assert np.allclose(tf, [1.0, 1.375, 1.774, 1.964, 2.148], atol=1e-3)
assert [round(bm25_tf_part(1, r, 1.0, b=bb), 3) for bb in (0.0, 0.75, 1.0) for r in (0.5, 1, 2)] == \
       [1.0, 1.0, 1.0, 1.257, 1.0, 0.71, 1.375, 1.0, 0.647]
r = rrf([["d3", "d1", "d7", "d2", "d9"], ["d1", "d5", "d3", "d8", "d2"]])
print("RRF k = 60:", {d: round(r[d], 5) for d in ("d1", "d3", "d2")})
assert abs(r["d1"] - 0.03252) < 1e-5 and abs(r["d3"] - 0.03227) < 1e-5 and abs(r["d2"] - 0.03101) < 1e-5
assert max(r, key=r.get) == "d1"
g = {"A": 3, "B": 2, "C": 1}
n1, n2 = ndcg_at_k(list("DAEBC"), g, 5), ndcg_at_k(list("ABDEC"), g, 5)
print(f"nDCG@5: hệ 1 {n1:.4f}, hệ 2 {n2:.4f}")
assert abs(n1 - 0.6489) < 5e-4 and abs(n2 - 0.9880) < 5e-4
assert abs(recall_at_k(list("DAEBC"), "ABC", 3) - 1 / 3) < 1e-9 and abs(recall_at_k(list("ABDEC"), "ABC", 3) - 2 / 3) < 1e-9
assert mrr(list("DAEBC"), set("ABC")) == 0.5 and mrr(list("ABDEC"), set("ABC")) == 1.0
print("Lab 1 ĐẠT ✔")'''

# ------------------------------------------------------------------ Lab 2
L2_WORLD = r'''def make_world(seed, n_topics=60, n_gen=200, n_train=600, n_test=400):
    """60 chủ đề; mỗi chủ đề 8 'từ chuẩn' a và 8 'từ đồng nghĩa' b. Mỗi tài liệu: 5 từ chủ đề + 9 từ nhiễu + 1 id hiếm.
    Train: 30% từ chủ đề đã bị thay bằng từ đồng nghĩa. Test: chỉ từ chuẩn, id chưa từng thấy lúc train."""
    rng = np.random.default_rng(seed)
    G = [f"g{i}" for i in range(n_gen)]
    A = [[f"a{t}x{j}" for j in range(8)] for t in range(n_topics)]
    B = [[f"b{t}x{j}" for j in range(8)] for t in range(n_topics)]
    def mkdoc(uid, t, mix):
        words = []
        for j in rng.choice(8, 5, replace=False):
            words.append(B[t][j] if (mix and rng.random() < 0.3) else A[t][j])
        toks = words + list(rng.choice(G, 9))
        rng.shuffle(toks)
        return dict(uid=uid, topic=t, toks=toks + [uid], words=words)
    train = [mkdoc(f"u{i}", int(rng.integers(n_topics)), True) for i in range(n_train)]
    test = [mkdoc(f"v{i}", int(rng.integers(n_topics)), False) for i in range(n_test)]
    return rng, G, A, B, train, test

def to_b(w):
    return "b" + w[1:]

def run_world(seed, hybrid_fn, steps=400, dim=32, nq=300):
    """Huấn luyện bi-encoder (EmbeddingBag trung bình + InfoNCE, nhiệt độ 0.05, batch 64) rồi đo 3 loại truy vấn.
    hybrid_fn(sb, sd, positive_only) -> danh sách chỉ số đã xếp hạng. Trả về {loại: {phương pháp: (hit@1, hit@10, MRR@10)}}."""
    rng, G, A, B, train, test = make_world(seed)
    torch.manual_seed(seed)
    vocab = {}
    for d in train + test:
        for t in d["toks"]:
            vocab.setdefault(t, len(vocab))
    for t in range(60):
        for w in A[t] + B[t]:
            vocab.setdefault(w, len(vocab))
    for g in G:
        vocab.setdefault(g, len(vocab))
    emb = torch.nn.EmbeddingBag(len(vocab), dim, mode="mean")
    torch.nn.init.normal_(emb.weight, std=0.3)
    opt = torch.optim.Adam(emb.parameters(), lr=0.03)
    def enc(lists):
        idx = torch.tensor([vocab[t] for toks in lists for t in toks])
        offs = torch.tensor(np.cumsum([0] + [len(t) for t in lists[:-1]]))
        return F.normalize(emb(idx, offs), dim=-1)
    def train_query(d):
        ws = [w if rng.random() < .5 else to_b(w) for w in d["words"]]
        return list(rng.choice(ws, 3, replace=False))
    for s in range(steps):
        batch = [train[i] for i in rng.choice(len(train), 64, replace=False)]
        q = enc([train_query(d) for d in batch]); p = enc([d["toks"][:-1] for d in batch])
        loss = F.cross_entropy(q @ p.T / 0.05, torch.arange(64))
        opt.zero_grad(); loss.backward(); opt.step()
    perm = rng.permutation(len(test)); corpus = [test[i] for i in perm]
    bm = BM25([d["toks"] for d in corpus])
    with torch.no_grad():
        P = enc([d["toks"] for d in corpus])
    out = {}
    for kind in ("id", "para", "words"):
        res = {"bm25": [], "dense": [], "hybrid": [], "hybrid_trap": []}
        for _ in range(nq):
            gi = int(rng.integers(len(corpus))); d = corpus[gi]
            if kind == "id":
                q = [d["uid"], str(rng.choice(d["words"]))]
            elif kind == "para":
                q = [to_b(w) for w in d["words"]]
            else:
                q = list(rng.choice(d["words"], 3, replace=False))
            sb = bm.score(q)
            with torch.no_grad():
                sd = (enc([q]) @ P.T)[0].numpy()
            lists = {"bm25": [i for i in ranking(sb) if sb[i] > 0], "dense": ranking(sd),
                     "hybrid": hybrid_fn(sb, sd, True), "hybrid_trap": hybrid_fn(sb, sd, False)}
            for name, r in lists.items():
                res[name].append((hit_at_k(r, [gi], 1), hit_at_k(r, [gi], 10), rr(r, {gi}, 10)))
        out[kind] = {k: tuple(np.mean(v, axis=0)) for k, v in res.items()}
    return out'''

L2 = (r'''def hybrid_ranking(sb, sd, positive_only=True):
    """Gộp điểm từ vựng sb (BM25) và điểm dense sd (cosine) bằng RRF; trả danh sách chỉ số đoạn đã xếp hạng (đủ mọi đoạn).
    positive_only=True: danh sách từ vựng chỉ gồm các đoạn có điểm > 0.
    positive_only=False: danh sách từ vựng gồm cả kho."""
    # TODO
    raise NotImplementedError("Viết hàm này (Lab 2)")''',
r'''def hybrid_ranking(sb, sd, positive_only=True):
    rb = [i for i in ranking(sb) if (sb[i] > 0 or not positive_only)]
    rd = ranking(sd)
    f = rrf([rb, rd])
    full = np.full(len(sb), -1.0)
    for d, v in f.items():
        full[d] = v
    return ranking(full)''')

L2_RUN = r'''t = time.time(); W = run_world(0, hybrid_ranking); print(f"huấn luyện + đo: {time.time() - t:.1f} s")
for kind in ("id", "para", "words"):
    print(kind.ljust(6), {m: tuple(round(float(x), 3) for x in v) for m, v in W[kind].items()})
mix = {m: np.mean([W["id"][m], W["para"][m]], axis=0) for m in ("bm25", "dense", "hybrid", "hybrid_trap")}
print("hỗn hợp 50% id + 50% paraphrase (hit@1, hit@10, MRR@10):", {m: tuple(round(float(x), 3) for x in v) for m, v in mix.items()})
assert W["id"]["bm25"][0] == 1.0, "BM25 phải tìm đúng mọi truy vấn định danh ở hạng 1"
assert 0.85 <= W["id"]["dense"][1] <= 0.95 and W["id"]["hybrid"][1] >= 0.99
assert W["para"]["bm25"][1] <= 0.03, "truy vấn chỉ có từ đồng nghĩa: BM25 gần như không trả được gì"
assert W["para"]["dense"][1] >= 0.95
assert abs(W["para"]["hybrid"][1] - W["para"]["dense"][1]) < 1e-9, "hybrid đúng phải trùng dense khi BM25 không khớp từ nào"
assert W["para"]["hybrid_trap"][1] < 0.6, "danh sách từ vựng 'rác' phải kéo hybrid xuống thấp hơn dense nhiều"
assert mix["hybrid"][1] - max(mix["bm25"][1], mix["dense"][1]) >= 0.04, "trên truy vấn hỗn hợp hybrid phải vượt cả hai bộ đơn ở hit@10"
assert W["words"]["bm25"][0] > W["words"]["hybrid"][0], "trên truy vấn từ khóa, hybrid không thắng BM25 ở hit@1"
print("Lab 2 ĐẠT ✔")'''

# ------------------------------------------------------------------ Lab 3
L3_DATA = r'''def build_docs(seed, n_docs=100, S=64, sl=12, nG=400):
    """Mỗi tài liệu 64 câu x 12 token (Zipf). Một sự thật hai câu liền nhau: câu 1 có tên thực thể và từ thuộc tính,
    câu 2 (không nhắc lại tên) có đáp án. Truy vấn = [tên, thuộc tính]."""
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, nG + 1); p /= p.sum()
    G = np.array([f"w{i}" for i in range(nG)])
    docs, facts = [], []
    for d in range(n_docs):
        sents = [list(rng.choice(G, sl, p=p)) for _ in range(S)]
        pos = int(rng.integers(0, S - 1))
        ent, attr, ans = f"ent{d}", f"attr{d}", f"ans{d}"
        sents[pos][0] = ent; sents[pos][1] = attr; sents[pos + 1][0] = ans
        docs.append(sents); facts.append((pos, ent, attr, ans))
    return docs, facts

def trial(docs, facts, c, stride, k=3, rng=None):
    """rng=None: hòa điểm phá theo chỉ số nhỏ (thiên vị, chỉ để minh họa hiện vật); truyền rng để phá hòa ngẫu nhiên (trung lập)."""
    ch = chunk_docs(docs, c, stride)
    bm = BM25([x[2] for x in ch])
    h1 = hk = 0; ctx = []
    for pos, ent, attr, ans in facts:
        r = ranking(bm.score([ent, attr]), rng)[:k]
        h1 += ans in ch[r[0]][2]
        hk += any(ans in ch[i][2] for i in r)
        ctx.append(sum(len(ch[i][2]) for i in r))
    n = len(facts)
    return h1 / n, hk / n, float(np.mean(ctx)), len(ch)'''

L3 = (r'''def chunk_docs(docs, c, stride):
    """Chia mỗi tài liệu (list câu, mỗi câu là list token) thành các chunk c câu, bước nhảy stride câu.
    Trả list (chỉ số tài liệu, câu bắt đầu, list token). Phải phủ đến câu cuối của tài liệu."""
    # TODO
    raise NotImplementedError("Viết hàm này (Lab 3)")

def p_cut(S, c, stride):
    """Xác suất (đếm chính xác trên mọi vị trí) để sự thật hai câu liền nhau không nằm trọn trong chunk nào.
    Sự thật bắt đầu ở câu pos, mỗi pos từ 0 đến S - 2 đều như nhau. Tài liệu có S câu."""
    # TODO
    raise NotImplementedError("Viết hàm này (Lab 3)")''',
r'''def chunk_docs(docs, c, stride):
    out = []
    for d, sents in enumerate(docs):
        for s in range(0, len(sents), stride):
            out.append((d, s, [t for x in sents[s:s + c] for t in x]))
            if s + c >= len(sents):
                break
    return out

def p_cut(S, c, stride):
    starts = list(range(0, S, stride))
    bad = 0
    for pos in range(S - 1):
        ok = any(s <= pos and pos + 1 <= min(s + c, S) - 1 for s in starts)
        bad += not ok
    return bad / (S - 1)''')

L3_RUN = r'''assert abs(p_cut(64, 4, 4) - 15 / 63) < 1e-9 and abs(p_cut(64, 8, 8) - 7 / 63) < 1e-9
print("xác suất cắt, không chồng lấn c = 2, 4, 8, 16:", [round(p_cut(64, c, c), 3) for c in (2, 4, 8, 16)])
assert p_cut(64, 4, 3) == 0.0 and p_cut(64, 8, 7) == 0.0, "chồng lấn 1 câu phải loại bỏ hoàn toàn lỗi cắt sự thật hai câu"
assert chunk_docs([[["x"]] * 10], 4, 3)[-1][1] + 4 >= 10, "chunk cuối phải chạm câu cuối tài liệu"
docs, facts = build_docs(0)
rows = {}
for c in (2, 4, 8, 16):
    for name, stride in (("không chồng lấn", c), ("chồng lấn 1 câu", max(1, c - 1))):
        r = trial(docs, facts, c, stride, rng=np.random.default_rng(1000)); rows[(c, name)] = r   # hòa điểm: phá NGẪU NHIÊN
        print(f"c = {c:2d}  {name:16s} ans@top-1 = {r[0]:.3f}  ans@top-3 = {r[1]:.3f}  token vào prompt (top-3) = {r[2]:.0f}  số chunk = {r[3]}")
# Cùng cấu hình c = 2, chồng lấn 1 câu, nhưng hòa điểm phá theo chỉ số nhỏ: kết quả đổi hẳn, vì hai chunk kề nhau HÒA điểm tuyệt đối.
art = trial(docs, facts, 2, 1)
print(f"c = 2  chồng lấn 1 câu, phá hòa theo chỉ số nhỏ: ans@top-1 = {art[0]:.3f}  (so với {rows[(2, 'chồng lấn 1 câu')][0]:.3f} khi phá hòa ngẫu nhiên)")
for c in (2, 4, 8, 16):
    assert abs(rows[(c, "không chồng lấn")][0] - (1 - p_cut(64, c, c))) < 0.07, "đo phải nằm trong 0.07 của 1 - xác suất cắt"
    assert rows[(c, "chồng lấn 1 câu")][1] >= 0.999, "chồng lấn 1 câu: đáp án luôn nằm trong top-3"
    assert rows[(c, "không chồng lấn")][2] == 3 * c * 12
    assert rows[(c, "chồng lấn 1 câu")][0] >= rows[(c, "không chồng lấn")][0] - 0.05, "chồng lấn không được làm ans@top-1 tệ hơn không chồng lấn"
assert 0.35 < rows[(2, "chồng lấn 1 câu")][0] < 0.65, "c = 2: hai chunk hòa điểm, chỉ một chứa đáp án, nên top-1 xấp xỉ tung đồng xu"
assert art[0] < 0.05, "phá hòa theo chỉ số nhỏ cho 0.010: đó là hiện vật của quy tắc phá hòa, không phải hiệu ứng của chunking"
print("Lab 3 ĐẠT ✔")'''

# ------------------------------------------------------------------ Lab 4
L4 = (r'''def ppr_power(T, s, alpha=0.5, iters=50):
    """Personalized PageRank bằng power iteration có khởi động lại: x <- alpha*s + (1 - alpha)*T x.
    T: ma trận chuyển chuẩn hóa theo cột, s: phân phối hạt giống (tổng 1). Trả về x."""
    # TODO
    raise NotImplementedError("Viết hàm này (Lab 4)")''',
r'''def ppr_power(T, s, alpha=0.5, iters=50):
    x = s.copy()
    for _ in range(iters):
        x = alpha * s + (1 - alpha) * T @ x
    return x''')

L4_DATA = r'''def adj_to_T(A):
    deg = A.sum(0); deg[deg == 0] = 1
    return A / deg

def build_arxiv(seed, nm=80, nd=12, roots=10):
    """80 phương pháp, 10 gốc, mỗi phương pháp còn lại mở rộng một phương pháp trước đó; 12 dataset.
    Mỗi phương pháp có đoạn giới thiệu và đoạn đánh giá."""
    rng = np.random.default_rng(seed)
    parent = {m: (None if m < roots else int(rng.integers(0, m))) for m in range(nm)}
    ds = {m: int(rng.integers(nd)) for m in range(nm)}
    techs = [f"tech{i}" for i in range(15)]; tasks = [f"task{i}" for i in range(8)]
    P = []
    for m in range(nm):
        if parent[m] is None:
            P.append(dict(kind="intro", m=m, text=f"paper {m} introduces m{m} a new method using {rng.choice(techs)}", ents=[f"m{m}"]))
        else:
            P.append(dict(kind="intro", m=m, text=f"paper {m} introduces m{m} which extends m{parent[m]} and uses {rng.choice(techs)}",
                          ents=[f"m{m}", f"m{parent[m]}"]))
    for m in range(nm):
        P.append(dict(kind="eval", m=m, text=f"m{m} is evaluated on dataset d{ds[m]} for {rng.choice(tasks)}", ents=[f"m{m}", f"d{ds[m]}"]))
    return rng, parent, ds, P

def ppr_passages(P, seeds, alpha=0.5, keep=1.0, rng=None):
    ents = sorted({e for p in P for e in p["ents"]}); eid = {e: i for i, e in enumerate(ents)}
    n = len(ents) + len(P); A = np.zeros((n, n))
    for pi, p in enumerate(P):
        for e in p["ents"]:
            if keep >= 1.0 or rng.random() < keep:
                a, b = eid[e], len(ents) + pi
                A[a, b] = A[b, a] = 1
    s = np.zeros(n)
    for e in seeds:
        if e in eid:
            s[eid[e]] = 1
    if s.sum() == 0:
        return np.zeros(len(P))
    x = ppr_power(adj_to_T(A), s / s.sum(), alpha)
    return x[len(ents):]

def run_arxiv(seed, nq=200, keep=1.0, k=5, tie="random"):
    """Câu hỏi hai bước: 'which dataset was evaluated for the method that m_x extends'.
    Vàng: đoạn giới thiệu của x và đoạn đánh giá của cha y (không chứa x). Trả {phương pháp: (đoạn đích có trong top-k, recall bằng chứng)}.
    tie = "random": hòa điểm phá ngẫu nhiên (rng riêng, seed + 99); tie = "index": hòa điểm theo chỉ số nhỏ (thiên vị, chỉ để so sánh)."""
    rng, parent, ds, P = build_arxiv(seed)
    trng = np.random.default_rng(seed + 99) if tie == "random" else None
    bm = BM25([tok(p["text"]) for p in P])
    nonroot = [m for m in parent if parent[m] is not None]
    pid = {(p["kind"], p["m"]): i for i, p in enumerate(P)}
    res = {"bm25": [], "iter": [], "ppr": [], "bm25+ppr": []}
    for x in rng.choice(nonroot, nq):
        y = parent[x]; gold = {pid[("intro", x)], pid[("eval", y)]}
        s1 = bm.score(tok(f"which dataset was evaluated for the method that m{x} extends")); r1 = ranking(s1, trng)
        out = {"bm25": r1[:k]}
        ents = sorted({e for i in r1[:2] for e in P[i]["ents"]} - {f"m{x}"})
        r2 = ranking(bm.score(tok("which dataset was evaluated for the method") + ents), trng)
        out["iter"] = r1[:2] + [i for i in r2 if i not in r1[:2]][:k - 2]
        sc = ppr_passages(P, [f"m{x}"], keep=keep, rng=rng)
        rp = ranking(sc, trng); out["ppr"] = rp[:k]
        rb = [i for i in r1 if s1[i] > 0]; rpp = [i for i in rp if sc[i] > 1e-9]
        f = rrf([rb, rpp]); full = np.full(len(P), -1.0)
        for d, v in f.items():
            full[d] = v
        out["bm25+ppr"] = ranking(full, trng)[:k]
        for name, r in out.items():
            res[name].append((float(pid[("eval", y)] in r), len(gold & set(r)) / 2))
    return {m: tuple(np.mean(v, axis=0)) for m, v in res.items()}'''

L4_RUN = r'''# Đồ thị 7 nút: thực thể X, Y, Z, D; đoạn P1 (X, Y), P2 (Y, D), P3 (X, Z). Hạt giống X, alpha = 0.5.
names = ["X", "Y", "Z", "D", "P1", "P2", "P3"]
edges = [("X", "P1"), ("Y", "P1"), ("Y", "P2"), ("D", "P2"), ("X", "P3"), ("Z", "P3")]
A7 = np.zeros((7, 7))
for a, b in edges:
    A7[names.index(a), names.index(b)] = A7[names.index(b), names.index(a)] = 1
T7 = adj_to_T(A7); s7 = np.zeros(7); s7[0] = 1
x = ppr_power(T7, s7, 0.5, iters=200)
exact = 0.5 * np.linalg.solve(np.eye(7) - 0.5 * T7, s7)
print(dict(zip(names, np.round(x, 4))))
assert np.allclose(x, exact, atol=1e-9) and abs(x.sum() - 1) < 1e-9
assert abs(x[0] - 0.5803) < 5e-4 and abs(x[4] - 0.1556) < 5e-4 and abs(x[5] - 0.0120) < 5e-4 and abs(x[6] - 0.1658) < 5e-4
x1 = ppr_power(T7, s7, 0.5, iters=1)
assert np.allclose(x1, [0.5, 0, 0, 0, 0.25, 0, 0.25]), "sau một lần lặp: X 0.5, P1 0.25, P3 0.25"
assert x[6] > x[4] > x[5] > 0, "đoạn đích hai bước (P2) xếp cuối nhưng khối lượng dương"

# Phản ví dụ: "đoạn hai bước luôn xếp sau đoạn một bước" KHÔNG phải luật. Hạt giống X nối 10 đoạn lá P1..P10;
# Y1, Y2, Y3 mỗi cái nối một đoạn P1..P3 và cùng nối đoạn Q (Q cách X hai bước, nhận khối lượng qua ba đường).
nm = ["X", "Y1", "Y2", "Y3"] + [f"P{i}" for i in range(1, 11)] + ["Q"]
Ac = np.zeros((len(nm), len(nm)))
def link(a, b):
    Ac[nm.index(a), nm.index(b)] = Ac[nm.index(b), nm.index(a)] = 1
for i in range(1, 11):
    link("X", f"P{i}")
for i in (1, 2, 3):
    link(f"Y{i}", f"P{i}"); link(f"Y{i}", "Q")
sc_ = np.zeros(len(nm)); sc_[0] = 1
for al in (0.5, 0.15):
    xc = ppr_power(adj_to_T(Ac), sc_, al, iters=400)
    print(f"phản ví dụ, alpha = {al}: P4 (lá, một bước) = {xc[nm.index('P4')]:.4f}, Q (hai bước, ba đường) = {xc[nm.index('Q')]:.4f}")
    if al == 0.5:
        assert xc[nm.index("P4")] > xc[nm.index("Q")]
    else:
        assert xc[nm.index("Q")] > xc[nm.index("P4")], "alpha nhỏ: Q hai bước vượt đoạn lá một bước"

t = time.time()
R = run_arxiv(0); R7 = run_arxiv(0, keep=0.7); R10 = run_arxiv(0, k=10)
Ri = run_arxiv(0, tie="index"); Ri10 = run_arxiv(0, k=10, tie="index")
print(f"(đo {time.time() - t:.1f} s)  [đoạn đích có trong top-k, recall bằng chứng]; hòa điểm phá NGẪU NHIÊN")
fmt = lambda r: {m: tuple(round(float(v), 3) for v in w) for m, w in r.items()}
print("k = 5, giữ 100% cạnh :", fmt(R))
print("k = 5, giữ 70% cạnh  :", fmt(R7))
print("k = 10, giữ 100% cạnh:", fmt(R10))
print("ĐỐI CHIẾU phá hòa theo chỉ số nhỏ, k = 5 :", fmt(Ri))
print("ĐỐI CHIẾU phá hòa theo chỉ số nhỏ, k = 10:", fmt(Ri10))
assert R["bm25"][0] == 0.0, "BM25 thuần không bao giờ lấy được đoạn đánh giá của cha"
assert 0.55 < R["iter"][0] < 0.8 and 0.3 < R["ppr"][0] < 0.45 and R["bm25+ppr"][0] < R["ppr"][0]
assert R["iter"][0] > R["ppr"][0] + 0.1, "k = 5: truy hồi lặp hơn PPR"
assert R10["ppr"][0] > R10["iter"][0] + 0.05, "k = 10: chiều đảo ngược, PPR hơn truy hồi lặp"
assert R7["ppr"][0] < R["ppr"][0] - 0.08 and abs(R7["iter"][0] - R["iter"][0]) < 1e-9, "mất cạnh làm PPR tụt, truy hồi lặp không đổi"
assert R10["ppr"][0] > R["ppr"][0] + 0.3, "tăng k thì PPR lấy được đoạn hai bước"
assert abs(Ri["ppr"][0] - R["ppr"][0]) < 1e-9, "đổi quy tắc phá hòa không đổi số của PPR trong thí nghiệm này"
assert Ri["iter"][0] > R["iter"][0] + 0.2, "quy tắc phá hòa theo chỉ số làm truy hồi lặp đẹp lên một cách giả tạo"
print("Lab 4 ĐẠT ✔")'''

# ------------------------------------------------------------------ Lab 5
L5 = (r'''def cohen_kappa(n_aa, n_ab, n_ba, n_bb):
    """Bảng 2x2 (hàng = người chấm 1 chọn A/B, cột = người chấm 2 chọn A/B). Trả (p0, pe, kappa)."""
    # TODO
    raise NotImplementedError("Viết hàm này (Lab 5)")

def wald_halfwidth(p, n, z=1.96):
    """Nửa độ rộng khoảng Wald cho một tỉ lệ."""
    # TODO
    raise NotImplementedError("Viết hàm này (Lab 5)")

def ece(conf, correct, edges):
    """ECE với các bin [edges[i], edges[i+1]) (bin cuối đóng ở 1.0)."""
    # TODO
    raise NotImplementedError("Viết hàm này (Lab 5)")

def risk_coverage(conf, correct, coverages):
    """Xếp theo độ tự tin giảm dần, chỉ trả lời phần đầu. Trả list accuracy trên phần được trả lời ứng với từng độ phủ."""
    # TODO
    raise NotImplementedError("Viết hàm này (Lab 5)")

def fpr_needed(base_rate, ppv, fnr):
    """FPR bắt buộc để nhóm có tỉ lệ nền base_rate đạt đúng ppv và fnr đã cho."""
    # TODO
    raise NotImplementedError("Viết hàm này (Lab 5)")

def laplace_tail(eps, t, n=200_000, seed=0):
    """Ước lượng P(|nhiễu Laplace thang 1/eps| > t) bằng mô phỏng n mẫu."""
    # TODO
    raise NotImplementedError("Viết hàm này (Lab 5)")

def rr_estimate(frac_yes_observed, p_truth=0.75):
    """Randomized response: mỗi người nói thật với xác suất p_truth, nói ngược với 1 - p_truth.
    Trả (epsilon, ước lượng tỉ lệ thật)."""
    # TODO
    raise NotImplementedError("Viết hàm này (Lab 5)")''',
r'''def cohen_kappa(n_aa, n_ab, n_ba, n_bb):
    n = n_aa + n_ab + n_ba + n_bb
    p0 = (n_aa + n_bb) / n
    a1, a2 = (n_aa + n_ab) / n, (n_aa + n_ba) / n
    pe = a1 * a2 + (1 - a1) * (1 - a2)
    return p0, pe, (p0 - pe) / (1 - pe)

def wald_halfwidth(p, n, z=1.96):
    return z * math.sqrt(p * (1 - p) / n)

def ece(conf, correct, edges):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    tot = 0.0
    for i in range(len(edges) - 1):
        m = (conf >= edges[i]) & ((conf < edges[i + 1]) if i < len(edges) - 2 else (conf <= edges[i + 1]))
        if m.any():
            tot += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return tot

def risk_coverage(conf, correct, coverages):
    order = np.argsort(-np.asarray(conf, float), kind="stable")
    c = np.asarray(correct, float)[order]
    return [c[:int(round(cov * len(c)))].mean() for cov in coverages]

def fpr_needed(base_rate, ppv, fnr):
    return base_rate / (1 - base_rate) * (1 - ppv) / ppv * (1 - fnr)

def laplace_tail(eps, t, n=200_000, seed=0):
    noise = np.random.default_rng(seed).laplace(0.0, 1.0 / eps, n)
    return float((np.abs(noise) > t).mean())

def rr_estimate(frac_yes_observed, p_truth=0.75):
    return math.log(p_truth / (1 - p_truth)), (frac_yes_observed - (1 - p_truth)) / (2 * p_truth - 1)''')

L5_CHECK = r'''p0, pe, kap = cohen_kappa(45, 10, 12, 33)
print(f"p0 = {p0:.3f}, pe = {pe:.3f}, kappa = {kap:.4f}, Wald ± {wald_halfwidth(p0, 100):.4f}")
assert abs(p0 - 0.78) < 1e-9 and abs(pe - 0.507) < 1e-9 and abs(kap - 0.5538) < 5e-4 and abs(wald_halfwidth(0.78, 100) - 0.0812) < 5e-4
k2 = cohen_kappa(81, 9, 9, 1)
assert abs(k2[0] - 0.82) < 1e-9 and abs(k2[2] - 0.0) < 0.06, "đồng thuận thô cao nhưng kappa gần 0 khi cả hai gần như luôn chọn A"
conf = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.55, 0.50, 0.30]
corr = [1, 1, 1, 0, 1, 0, 1, 0, 0, 0]
e = ece(conf, corr, [0, 0.5, 0.75, 1.0])
print("ECE =", round(e, 4), "| rủi ro-độ phủ:", np.round(risk_coverage(conf, corr, [0.3, 0.5, 0.7, 1.0]), 3))
assert abs(e - 0.19) < 1e-9
assert np.allclose(risk_coverage(conf, corr, [0.3, 0.5, 0.7, 1.0]), [1.0, 0.8, 5 / 7, 0.5])
fa, fb = fpr_needed(0.5, 0.8, 0.2), fpr_needed(0.2, 0.8, 0.2)
print(f"FPR cần: nhóm A {fa:.3f}, nhóm B {fb:.3f}")
assert abs(fa - 0.2) < 1e-9 and abs(fb - 0.05) < 1e-9
assert fa != fb
for base, want in ((0.3, 60 / 700), (0.2, 40 / 800)):   # kiểm bằng bảng nhầm lẫn 1000 người: TP = (1 - FNR)·base·1000, FP = TP·(1 - PPV)/PPV
    tp = 0.8 * base * 1000; fp = tp * 0.2 / 0.8
    assert abs(fp / ((1 - base) * 1000) - want) < 1e-12 and abs(fpr_needed(base, 0.8, 0.2) - want) < 1e-12
for eps in (0.5, 1.0, 2.0):
    sim, th = laplace_tail(eps, 5), math.exp(-5 * eps)
    print(f"eps = {eps}: P(|nhiễu| > 5) mô phỏng {sim:.4f}, lý thuyết {th:.4f}")
    assert abs(sim - th) < 0.004
eps_rr, pi_hat = rr_estimate(0.55)
print(f"randomized response: eps = {eps_rr:.4f} (= ln 3), ước lượng = {pi_hat:.2f}")
assert abs(eps_rr - math.log(3)) < 1e-12 and abs(pi_hat - 0.60) < 1e-12
se_priv = math.sqrt(0.55 * 0.45 / 1000) / 0.5; se_plain = math.sqrt(0.6 * 0.4 / 1000)
print(f"SE có nhiễu riêng tư {se_priv:.4f}, SE không nhiễu {se_plain:.4f}, tỉ lệ {se_priv / se_plain:.2f}, cỡ mẫu gấp {(se_priv / se_plain) ** 2:.1f} lần")
assert abs(se_priv / se_plain - 2.03) < 0.01
print("Lab 5 ĐẠT ✔")'''

# ------------------------------------------------------------------ Bug
BUG_DATA = r'''def make_docs(seed, n_docs=60):
    rng = np.random.default_rng(seed)
    nG = 300; p = 1 / np.arange(1, nG + 1); p /= p.sum(); G = np.array([f"w{i}" for i in range(nG)])
    docs, facts = [], []
    for d in range(n_docs):
        S = int(rng.integers(20, 101))
        sents = [list(rng.choice(G, int(rng.integers(8, 25)), p=p)) for _ in range(S)]
        pos = int(rng.integers(0, S - 1))
        sents[pos][0], sents[pos][1] = f"ent{d}", "w40"; sents[pos + 1][0] = f"ans{d}"
        docs.append(sents); facts.append((pos, f"ent{d}", "w40", f"ans{d}"))
    for d in range(n_docs):
        for o in rng.choice([x for x in range(n_docs) if x != d], 1, replace=False):
            j = int(rng.integers(len(docs[o])))
            if f"ans{o}" in docs[o][j] or f"ent{o}" in docs[o][j]:
                continue
            docs[o][j][2] = f"ent{d}"
    return rng, docs, facts

def trigram_matrix(texts):
    grams, rows = {}, []
    for t in texts:
        s = " ".join(t); cnt = Counter(s[i:i + 3] for i in range(len(s) - 2))
        for g in cnt:
            grams.setdefault(g, len(grams))
        rows.append(cnt)
    M = np.zeros((len(texts), len(grams)), np.float32)
    for i, cnt in enumerate(rows):
        for g, v in cnt.items():
            M[i, grams[g]] = 1 + math.log(v)
    df = (M > 0).sum(0); idf = np.log(len(texts) / (1 + df)) + 1
    M *= idf; M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    return grams, idf, M

def trigram_query(q, grams, idf):
    s = " ".join(t.lower() for t in q); cnt = Counter(s[i:i + 3] for i in range(len(s) - 2))
    v = np.zeros(len(grams), np.float32)
    for g, c_ in cnt.items():
        if g in grams:
            v[grams[g]] = (1 + math.log(c_)) * idf[grams[g]]
    return v / (np.linalg.norm(v) + 1e-9)

def rag_buggy(seed, shuffle_retrieval=False, k=3):
    """Đường ống RAG nhỏ: chunk 4 câu, bước 3 -> truy hồi BM25 + tri-gram ký tự, gộp RRF -> lấy top-3 -> 'đọc'
    (đúng nếu token đáp án nằm trong ngữ cảnh). Trả dict: recall (đáp án nằm trong top-k chunk), e2e (đáp án nằm trong
    ngữ cảnh thực sự đưa vào mô hình), chunks, docs. shuffle_retrieval=True thay truy hồi bằng chọn ngẫu nhiên."""
    rng, docs, facts = make_docs(seed)
    ch = []
    for d, sents in enumerate(docs):
        for s in range(0, len(sents) - 4, 3):
            ch.append((d, s, [t for x in sents[s:s + 4] for t in x]))
    texts = [x[2] for x in ch]
    bm = BM25(texts)
    grams, idf, M = trigram_matrix(texts)
    ok_top = ok_e2e = 0
    for d, (pos, ent, attr, ans) in enumerate(facts):
        q = ["w0", "w1", "w2", ent.capitalize(), attr]
        sb = bm.score(q)
        rb = [i for i in ranking(sb) if sb[i] > 0]
        sf = M @ trigram_query(q, grams, idf)
        rf = list(np.argsort(sf))
        fused = rrf([rb, rf]); full = np.full(len(ch), -1.0)
        for i_, v_ in fused.items():
            full[i_] = v_
        top = ranking(full)[:k]
        if shuffle_retrieval:
            top = list(rng.choice(len(ch), k, replace=False))
        ctx = [ch[i][2][:40] for i in top]
        extra = [x[2] for x in ch if x[0] == d and ans in x[2]][:1]
        ctx = ctx + extra
        ok_e2e += any(ans in c_ for c_ in ctx)
        ok_top += any(ans in ch[i][2] for i in top)
    return dict(recall=ok_top / len(facts), e2e=ok_e2e / len(facts), chunks=ch, docs=docs)'''

BUG_RUN = r'''t = time.time()
b = rag_buggy(0); br = rag_buggy(0, shuffle_retrieval=True)
print(f"BẢN LỖI (seed 0): recall@3 = {b['recall']:.3f} | e2e = {b['e2e']:.3f} | e2e khi truy hồi ngẫu nhiên = {br['e2e']:.3f}   ({time.time() - t:.1f} s)")'''

FIX = (r'''def rag_fixed(seed, shuffle_retrieval=False, k=3):
    """Viết lại rag_buggy cho đúng. Cùng chữ ký, cùng kiểu trả về."""
    # TODO
    raise NotImplementedError("Viết hàm này (Săn bug; ô này gọi rrf của Lab 1, nên làm Lab 1 trước)")''',
r'''def rag_fixed(seed, shuffle_retrieval=False, k=3):
    rng, docs, facts = make_docs(seed)
    ch = []
    for d, sents in enumerate(docs):
        for s in range(0, len(sents), 3):
            ch.append((d, s, [t for x in sents[s:s + 4] for t in x]))
            if s + 4 >= len(sents):
                break
    texts = [x[2] for x in ch]
    bm = BM25(texts)
    grams, idf, M = trigram_matrix(texts)
    ok_top = ok_e2e = 0
    for d, (pos, ent, attr, ans) in enumerate(facts):
        q = ["w0", "w1", "w2", ent.capitalize(), attr]
        sb = bm.score([t.lower() for t in q])
        rb = [i for i in ranking(sb) if sb[i] > 0]
        sf = M @ trigram_query(q, grams, idf)
        rf = list(np.argsort(-sf))
        fused = rrf([rb, rf]); full = np.full(len(ch), -1.0)
        for i_, v_ in fused.items():
            full[i_] = v_
        top = ranking(full)[:k]
        if shuffle_retrieval:
            top = list(rng.choice(len(ch), k, replace=False))
        ctx = [ch[i][2] for i in top]
        ok_e2e += any(ans in c_ for c_ in ctx)
        ok_top += any(ans in ch[i][2] for i in top)
    return dict(recall=ok_top / len(facts), e2e=ok_e2e / len(facts), chunks=ch, docs=docs)''')

BUG_CHECK = r'''f = rag_fixed(0); fr = rag_fixed(0, shuffle_retrieval=True)
print(f"BẢN SỬA (seed 0): recall@3 = {f['recall']:.3f} | e2e = {f['e2e']:.3f} | e2e khi truy hồi ngẫu nhiên = {fr['e2e']:.3f}")
assert f["recall"] >= 0.98, "Kiểm tra A không đạt: truy hồi vẫn bỏ sót đáp án"
assert f["e2e"] >= 0.98, "Kiểm tra B không đạt: đáp án không tới được mô hình"
assert fr["e2e"] <= 0.05, "Kiểm tra C không đạt: tắt truy hồi mà kết quả vẫn cao"
last = {}
for d, s, _ in f["chunks"]:
    last[d] = max(last.get(d, -1), s)
assert all(last[d] + 4 >= len(f["docs"][d]) for d in last) and len(last) == len(f["docs"]), "Kiểm tra D không đạt: có câu không thuộc chunk nào"
assert f["recall"] >= b["recall"] + 0.9
print("Săn bug ĐẠT ✔")'''

BUG_PHAN_TICH = r'''# Phân tích (chỉ có trong lời giải): bật từng lỗi riêng lẻ trên bản đúng để thấy triệu chứng từng lỗi.
def rag_with(seed, bugs, shuffle_retrieval=False, k=3):
    rng, docs, facts = make_docs(seed)
    ch = []
    for d, sents in enumerate(docs):
        end = len(sents) - 4 if 5 in bugs else len(sents)
        for s in range(0, max(end, 1), 3):
            ch.append((d, s, [t for x in sents[s:s + 4] for t in x]))
            if s + 4 >= len(sents):
                break
    bm = BM25([x[2] for x in ch]); grams, idf, M = trigram_matrix([x[2] for x in ch])
    ok_top = ok_e2e = 0
    for d, (pos, ent, attr, ans) in enumerate(facts):
        q = ["w0", "w1", "w2", ent.capitalize(), attr]
        sb = bm.score(q if 1 in bugs else [t.lower() for t in q])
        rb = [i for i in ranking(sb) if sb[i] > 0]
        sf = M @ trigram_query(q, grams, idf)
        rf = list(np.argsort(sf)) if 4 in bugs else list(np.argsort(-sf))
        fused = rrf([rb, rf]); full = np.full(len(ch), -1.0)
        for i_, v_ in fused.items():
            full[i_] = v_
        top = ranking(full)[:k]
        if shuffle_retrieval:
            top = list(rng.choice(len(ch), k, replace=False))
        ctx = [ch[i][2][:40] if 2 in bugs else ch[i][2] for i in top]
        if 3 in bugs:
            ctx = ctx + [x[2] for x in ch if x[0] == d and ans in x[2]][:1]
        ok_e2e += any(ans in c_ for c_ in ctx)
        ok_top += any(ans in ch[i][2] for i in top)
    return ok_top / len(facts), ok_e2e / len(facts)

names = {1: "không hạ chữ thường truy vấn", 2: "cắt mỗi chunk ở 40 token", 3: "thêm đoạn chứa đáp án vào ngữ cảnh",
         4: "đảo thứ tự truy hồi mềm", 5: "bỏ chunk cuối", (1, 2, 3, 4, 5): "cả năm lỗi"}
res = {}
for key, name in [((), "bản đúng")] + [((k,) if isinstance(k, int) else k, v) for k, v in names.items()]:
    r = rag_with(0, key); rs = rag_with(0, key, shuffle_retrieval=True)
    res[key] = (r, rs)
    print(f"  {name:36s} recall@3 {r[0]:.3f} | e2e {r[1]:.3f} | e2e truy hồi ngẫu nhiên {rs[1]:.3f}")
assert res[(1, 2, 3, 4, 5)][0][0] < 0.05 and res[(1, 2, 3, 4, 5)][0][1] > 0.85 and res[(1, 2, 3, 4, 5)][1][1] > 0.85
assert res[(3,)][1][1] > 0.95 and res[(2,)][0][0] > 0.99 and res[(2,)][0][1] < 0.8 and res[(5,)][0][0] < 0.97
print(f"Tổng thời gian notebook: {time.time() - T0:.1f} s")
print("Phân tích lỗi ĐẠT ✔")'''


TIE_MECH = r'''# Cơ chế của nhạy cảm với quy tắc phá hòa (VD 12.7, seed 0, 200 câu): hòa xảy ra ở HẠNG 2, giữa đoạn giới thiệu của x
# và đoạn giới thiệu của các con của x; chỉ số nhỏ luôn chọn x vì dữ liệu sinh cha trước con.
rng_, parent_, ds_, P_ = build_arxiv(0)
bm_ = BM25([tok(p["text"]) for p in P_])
pid_ = {(p["kind"], p["m"]): i for i, p in enumerate(P_)}
nonroot_ = [m for m in parent_ if parent_[m] is not None]
trng_ = np.random.default_rng(0 + 99)
top1_eval = n_tie = rand_in = idx_in = 0
for x in rng_.choice(nonroot_, 200):
    s1 = bm_.score(tok(f"which dataset was evaluated for the method that m{x} extends"))
    r_rand, r_idx = ranking(s1, trng_), ranking(s1)
    intro_x, eval_x = pid_[("intro", x)], pid_[("eval", x)]
    top1_eval += (r_rand[0] == eval_x)
    tie_grp = [i for i in range(len(s1)) if round(s1[i], 12) == round(s1[r_rand[1]], 12)]
    if len(tie_grp) > 1:
        n_tie += 1
        kids = {m for m in parent_ if parent_[m] == x}
        assert intro_x in tie_grp and all(P_[i]["kind"] == "intro" and (P_[i]["m"] == x or P_[i]["m"] in kids) for i in tie_grp)
    rand_in += intro_x in r_rand[:2]; idx_in += intro_x in r_idx[:2]
print(f"top-1 = đoạn đánh giá của x: {top1_eval}/200; câu có hòa ở hạng 2: {n_tie}/200; "
      f"đoạn giới thiệu của x trong top-2: phá hòa ngẫu nhiên {rand_in}/200, theo chỉ số nhỏ {idx_in}/200")
assert top1_eval == 200 and n_tie == 99 and idx_in == 200 and rand_in == 142
assert all(parent_[m] < m for m in nonroot_), "cha luôn có chỉ số nhỏ hơn con: đó là lý do chỉ số nhỏ chọn đúng x"
print("Cơ chế phá hòa ĐẠT ✔")'''

THREE_SEED = r'''RUN_3_SEED = False   # đặt True để chạy (thêm khoảng 1 phút); tắt mặc định để cả notebook dưới 2 phút
if not RUN_3_SEED:
    print("Bỏ qua bản 3 seed (RUN_3_SEED = False).")
else:
    t0 = time.time()
    seeds = (0, 1, 2)
    def mean3(f):
        rs = [f(s) for s in seeds]
        return {m: tuple(np.mean([r[m] for r in rs], axis=0)) for m in rs[0]}
    # VD 12.2: run_world(s) cho s = 0, 1, 2
    Ws = [run_world(s, hybrid_ranking) for s in seeds]
    for kind in ("id", "para", "words"):
        print("VD 12.2", kind.ljust(6), {m: tuple(round(float(x), 3) for x in np.mean([w[kind][m] for w in Ws], axis=0)) for m in Ws[0][kind]})
    # VD 12.4: hòa điểm phá ngẫu nhiên bằng default_rng(1000 + s) (seed 0 chính là default_rng(1000) của Lab 3)
    cfgs = [(c, c) for c in (2, 4, 8, 16)] + [(2, 1), (4, 3), (4, 2), (8, 7), (8, 4), (16, 15)]
    D = [build_docs(s) for s in seeds]
    for c, st in cfgs:
        r = np.mean([trial(*D[s], c, st, rng=np.random.default_rng(1000 + s)) for s in seeds], axis=0)
        print(f"VD 12.4  c = {c:2d}, bước {st:2d}: ans@top-1 = {r[0]:.3f}, ans@top-3 = {r[1]:.3f}, token = {r[2]:.0f}, số chunk = {r[3]:.0f}")
    ai = np.mean([trial(*D[s], 2, 1)[0] for s in seeds])
    assert abs(np.mean([w["id"]["hybrid"][0] for w in Ws]) - 0.697) < 6e-4 and abs(ai - 0.013) < 6e-4
    print(f"VD 12.4  c = 2, bước 1, phá hòa theo chỉ số nhỏ: ans@top-1 = {ai:.3f}")
    # VD 12.7: hòa điểm phá ngẫu nhiên bằng default_rng(seed + 99) (đã nằm trong run_arxiv)
    res7 = {}
    for label, kw in (("k = 5", {}), ("k = 10", dict(k=10)), ("k = 20", dict(k=20)),
                      ("k = 5, giữ 90%", dict(keep=0.9)), ("k = 5, giữ 80%", dict(keep=0.8)), ("k = 5, giữ 70%", dict(keep=0.7)),
                      ("k = 10, giữ 90%", dict(k=10, keep=0.9)), ("k = 10, giữ 80%", dict(k=10, keep=0.8)), ("k = 10, giữ 70%", dict(k=10, keep=0.7)),
                      ("k = 5, phá hòa theo chỉ số", dict(tie="index")), ("k = 10, phá hòa theo chỉ số", dict(k=10, tie="index"))):
        res7[label] = mean3(lambda s: run_arxiv(s, **kw))
        print("VD 12.7", label.ljust(28), {m: tuple(round(float(x), 3) for x in v) for m, v in res7[label].items()})
    assert abs(res7["k = 5"]["iter"][0] - 0.685) < 6e-4 and abs(res7["k = 5"]["ppr"][0] - 0.477) < 6e-4
    assert abs(res7["k = 10"]["ppr"][0] - 0.892) < 6e-4 and abs(res7["k = 5, phá hòa theo chỉ số"]["iter"][0] - 0.963) < 6e-4
    print("Phụ lục 3 seed ĐẠT ✔")
    print(f"(chạy 3 seed mất {time.time() - t0:.0f} s)")'''


def build(solution: bool):
    pick = (lambda pair: pair[1]) if solution else (lambda pair: pair[0])
    head = "Lời giải" if solution else "Bài lab"
    intro = ("Bản lời giải đã chạy kiểm (NumPy, PyTorch CPU, dưới 2 phút). Chỉ mở sau khi đã tự làm `lab.ipynb`."
             if solution else
             "Viết code vào các ô có `# TODO`, rồi chạy ô kiểm tra ngay bên dưới. Ô kiểm tra in **ĐẠT ✔** là xong. "
             "Bí quá 20 phút mới mở `loigiai.ipynb`. Dữ liệu tự dựng, không tải mạng, chỉ cần PyTorch bản CPU. "
             "Các lab nối tiếp nhau (Lab 2 và 4 dùng `rrf` của Lab 1), nên làm theo thứ tự.")
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(f"# Chương 12 — {head}: Truy hồi, RAG, Graph-RAG, đánh giá và rủi ro\n\n{intro}"),
        md("## Ô chuẩn bị\nBM25 bản nhanh, hàm xếp hạng, hàm đo. Mọi thí nghiệm dưới đây là **đồ chơi dựng để lộ cơ chế**, "
           "không xếp hạng phương pháp thật và không cho biết cỡ hiệu ứng trên dữ liệu thật."),
        code(SETUP),
        md("## Lab 1 — BM25 và các chỉ số xếp hạng từ đầu\nTự viết BM25 bản Lucene, recall@k, MRR, nDCG@k và RRF; "
           "tái hiện các ví dụ tay của mục 12.1."),
        code(pick(L1)), code(L1_CHECK),
        md("## Lab 2 — BM25, bi-encoder và hybrid trên cùng một thế giới tổng hợp\nBạn chỉ viết hàm gộp RRF; "
           "phần huấn luyện bi-encoder (InfoNCE, chương 10) đã cho sẵn. Đo ba loại truy vấn và hai cách cài hybrid."),
        code(L2_WORLD), code(pick(L2)), code(L2_RUN),
        md("## Lab 3 — Chunking: sự thật hai câu bị cắt ngang\nViết hàm chia chunk và hàm đếm chính xác xác suất cắt, "
           "rồi so với đo."),
        code(L3_DATA), code(pick(L3)), code(L3_RUN),
        md("## Lab 4 — PPR và câu hỏi hai bước\nPersonalized PageRank là power iteration (chương 1) cộng số hạng khởi động lại. "
           "So với BM25 và với truy hồi lặp làm đối chứng."),
        code(pick(L4)), code(L4_DATA), code(L4_RUN),
        md("## Lab 5 — Đánh giá và rủi ro bằng số\nCohen κ, khoảng Wald, ECE, rủi ro-độ phủ, công thức công bằng không thể thỏa đồng thời, "
           "nhiễu Laplace và randomized response."),
        code(pick(L5)), code(L5_CHECK),
        md("## Săn bug\nMột đường ống RAG nhỏ chạy không báo lỗi. Triệu chứng: báo cáo cuối (e2e) trông ổn, nhưng recall@3 của truy hồi gần 0; "
           "tắt truy hồi (chọn ngẫu nhiên) mà e2e gần như không giảm; truy vấn viết hoa tên riêng khó tìm; hybrid thua BM25 đơn lẻ; "
           "vài sự thật cuối tài liệu không bao giờ được tìm thấy; có cấu hình recall 1.000 mà e2e thấp. "
           "Tìm cả **năm** lỗi rồi viết `rag_fixed`."),
        code(BUG_DATA), code(BUG_RUN), code(pick(FIX)), code(BUG_CHECK),
    ]
    if solution:
        nb.cells += [md("## Phân tích từng lỗi\nBật riêng từng lỗi trên bản đúng để thấy triệu chứng của nó."), code(BUG_PHAN_TICH),
                     md("## Cơ chế phá hòa của Lab 4\nVì sao truy hồi lặp đẹp lên khi phá hòa theo chỉ số nhỏ: hòa nằm ở đâu và chỉ số mang thông tin gì."), code(TIE_MECH),
                     md("## Phụ lục tùy chọn: tái lập số 3 seed của VD 12.2, 12.4, 12.7\nTắt mặc định (`RUN_3_SEED = False`). Quy ước seed phá hòa: VD 12.4 dùng `default_rng(1000 + s)`, VD 12.7 dùng `default_rng(s + 99)` bên trong `run_arxiv`."), code(THREE_SEED)]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    name = "loigiai.ipynb" if solution else "lab.ipynb"
    nbf.write(nb, OUT / name)
    return OUT / name


if __name__ == "__main__":
    print(build(False), build(True))
