"""Sinh hai notebook cho lab chương 9: lab.ipynb (có TODO) và loigiai.ipynb (lời giải).

Chạy lại script này nếu muốn tạo lại notebook gốc:  python labs/build_chuong09.py
Chỉ cần PyTorch bản CPU. Ô đầu gọi torch.set_num_threads(1): mô hình tí hon không hưởng lợi gì từ nhiều luồng,
và trên máy ít lõi, nhiều luồng tranh nhau còn làm chậm gấp vài lần. Toàn bộ loigiai.ipynb chạy khoảng 30 giây trên máy rảnh (hơn 2 phút nếu máy bận)
(thời gian tự đo, in ở ô cuối). Dữ liệu tự sinh, seed cố định, không cần mạng.
"""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "chuong09"
OUT.mkdir(parents=True, exist_ok=True)

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

# ---------------------------------------------------------------- SETUP
SETUP = r'''import time, math, copy, unicodedata, itertools
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import least_squares, minimize_scalar
torch.set_num_threads(1)   # mô hình tí hon: nhiều luồng chỉ tranh nhau và làm chậm
np.set_printoptions(precision=4, suppress=True)
T0 = time.time()

# ---- Transformer tí hon dùng chung (đã học ở chương 8): 2 khối pre-LN, d = 48, 2 đầu, ngữ cảnh 24
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.h = h
        self.ln1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d); self.fc1 = nn.Linear(d, 4 * d); self.fc2 = nn.Linear(4 * d, d)
    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(a)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))

class TinyGPT(nn.Module):
    def __init__(self, V, d=48, L=2, h=2, T=24):
        super().__init__()
        self.tok = nn.Embedding(V, d); self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, V, bias=False)
    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        for b in self.blocks: x = b(x)
        return self.head(self.lnf(x))

def get_batch(data, T, B, g):
    ix = torch.randint(0, len(data) - T - 1, (B,), generator=g)
    return torch.stack([data[i:i + T] for i in ix]), torch.stack([data[i + 1:i + T + 1] for i in ix])

@torch.no_grad()
def eval_loss(model, data, T=24, nb=20, B=32, seed=123):
    model.eval(); g = torch.Generator().manual_seed(seed); tot = 0.0
    for _ in range(nb):
        x, y = get_batch(data, T, B, g)
        tot += F.cross_entropy(model(x).reshape(-1, model.head.out_features), y.reshape(-1)).item()
    model.train(); return tot / nb

def train(model, data, steps, lr=3e-3, B=32, T=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    for _ in range(steps):
        x, y = get_batch(data, T, B, g)
        loss = F.cross_entropy(model(x).reshape(-1, model.head.out_features), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model

# ---- "Ngôn ngữ Markov": nguồn có entropy biết trước, nên biết cả cận dưới của perplexity
def markov_matrix(V=24, conc=0.25, seed=0):
    return np.random.default_rng(seed).dirichlet(np.full(V, conc), size=V)

def sample_markov(P, n, seed):
    rng = np.random.default_rng(seed); cum = np.cumsum(P, 1); V = len(P)
    out = np.empty(n, dtype=np.int64); out[0] = 0; u = rng.random(n)
    for t in range(1, n): out[t] = min(np.searchsorted(cum[out[t - 1]], u[t]), V - 1)
    return torch.tensor(out)

V = 24
print("sẵn sàng; torch", torch.__version__)'''

# ---------------------------------------------------------------- Lab 1
BPE_CORPUS = r'''EN = """The cat sat on the mat and looked at the window.
She walks to the market every morning to buy fresh bread.
We are learning how language models read and write text.
The students asked many questions about the new project.
It was raining, so they decided to stay at home and read.
He wrote a short letter to his friend in the city.
The teacher explained the lesson and gave us some homework.
They were walking along the river when the sun went down.
My brother works at a small company that builds software.
There is a good restaurant near the station and the park.
She thought about the problem for a long time before answering.
The children played in the garden until it was dark.
I would like to learn more about machine learning and language.
The book on the table was written by a famous author.
We will meet at the library after the class is over.
Many people think that the weather is better in the spring.""".split("\n")
EN_TEST = """The teacher walked to the library and read a book about the river.
My friend thinks the students will learn more in the spring."""
VI = """Con mèo ngồi trên chiếc chiếu và nhìn ra cửa sổ.
Sáng nào bà cũng đi bộ ra chợ để mua bánh mì tươi.
Chúng tôi đang học cách các mô hình ngôn ngữ đọc và viết văn bản.
Các học sinh hỏi rất nhiều câu hỏi về dự án mới.
Trời đang mưa nên họ quyết định ở nhà và đọc sách.
Anh ấy viết một lá thư ngắn cho người bạn ở thành phố.
Giáo viên giải thích bài học và giao cho chúng tôi một ít bài tập về nhà.
Họ đang đi dọc bờ sông thì mặt trời lặn xuống.
Anh trai tôi làm việc tại một công ty nhỏ chuyên xây dựng phần mềm.
Có một nhà hàng ngon gần nhà ga và công viên.
Cô ấy suy nghĩ về vấn đề rất lâu trước khi trả lời.
Bọn trẻ chơi trong vườn cho đến khi trời tối.
Tôi muốn tìm hiểu thêm về học máy và ngôn ngữ.
Quyển sách trên bàn được viết bởi một tác giả nổi tiếng.
Chúng ta sẽ gặp nhau ở thư viện sau khi tan học.
Nhiều người nghĩ rằng thời tiết vào mùa xuân đẹp hơn.""".split("\n")
VI_TEST = """Giáo viên đi bộ đến thư viện và đọc một quyển sách về dòng sông.
Bạn tôi nghĩ rằng các học sinh sẽ học được nhiều hơn vào mùa xuân."""
print("kho Anh", len("\n".join(EN).encode()), "byte; kho Việt", len("\n".join(VI).encode()), "byte")'''

BPE = (
r'''def get_stats(ids):
    """Đếm các cặp id liền kề: {(a, b): số lần}."""
    # TODO
    raise NotImplementedError

def merge(ids, pair, new_id):
    """Thay mọi cặp `pair` liền kề trong danh sách ids bằng new_id (quét một lượt từ trái sang)."""
    # TODO
    raise NotImplementedError

def train_bpe(text, n_merges):
    """Byte-level BPE. Bắt đầu từ các byte UTF-8 (id 0..255); mỗi vòng gộp cặp nhiều nhất thành id mới
    (256, 257, ...). Dừng sớm nếu cặp nhiều nhất chỉ xuất hiện 1 lần. Trả về dict {(a, b): id}."""
    # TODO
    raise NotImplementedError

def encode(text, merges):
    """Áp các phép gộp THEO THỨ TỰ ĐÃ HỌC (id nhỏ trước): lặp, chọn cặp có id gộp nhỏ nhất đang có mặt."""
    # TODO
    raise NotImplementedError

def decode(ids, merges):
    """Ghép byte của từng id rồi giải mã UTF-8 (errors='replace')."""
    # TODO
    raise NotImplementedError''',
r'''def get_stats(ids):
    counts = {}
    for p in zip(ids, ids[1:]):
        counts[p] = counts.get(p, 0) + 1
    return counts

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

def train_bpe(text, n_merges):
    ids, merges = list(text.encode("utf-8")), {}
    for _ in range(n_merges):
        st = get_stats(ids)
        if not st: break
        pair = max(st, key=st.get)                    # hòa thì lấy cặp xuất hiện sớm nhất trong dict
        if st[pair] < 2: break
        new_id = 256 + len(merges)
        ids = merge(ids, pair, new_id); merges[pair] = new_id
    return merges

def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        st = get_stats(ids)
        pair = min(st, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges: break
        ids = merge(ids, pair, merges[pair])
    return ids

def decode(ids, merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), idx in merges.items():                # merges được thêm đúng thứ tự id tăng dần
        vocab[idx] = vocab[a] + vocab[b]
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")''')

BPE_CHECK = r'''# (a) ví dụ trong README của minbpe: 3 phép gộp trên "aaabdaaabac"
m3 = train_bpe("aaabdaaabac", 3)
print("merges:", m3, "| encode:", encode("aaabdaaabac", m3))
assert encode("aaabdaaabac", m3) == [258, 100, 258, 97, 99]
assert decode(encode("aaabdaaabac", m3), m3) == "aaabdaaabac"

# (b) hai tokenizer 150 phép gộp, mỗi cái học trên một ngôn ngữ; đo hai câu kiểm tra
M_en = train_bpe("\n".join(EN), 150); M_vi = train_bpe("\n".join(VI), 150)
tok = {}
for mn, M in (("tokenizer Anh", M_en), ("tokenizer Việt", M_vi)):
    for tn, tt in (("câu Anh", EN_TEST), ("câu Việt", VI_TEST)):
        ids = encode(tt, M); assert decode(ids, M) == tt          # khứ hồi
        tok[(mn, tn)] = len(ids)
        print(f"{mn:15s} | {tn}: {len(tt.encode()):3d} byte -> {len(ids):3d} token ({len(tt.encode()) / len(ids):.2f} byte/token)")
ratio_vi = tok[("tokenizer Anh", "câu Việt")] / tok[("tokenizer Việt", "câu Việt")]
ratio_en = tok[("tokenizer Việt", "câu Anh")] / tok[("tokenizer Anh", "câu Anh")]
print(f"tokenizer 'sai ngôn ngữ' tốn gấp {ratio_vi:.2f} lần (câu Việt) và {ratio_en:.2f} lần (câu Anh)")
assert ratio_vi > 1.8 and ratio_en > 1.5

# (c) Unicode: cùng một chữ, hai cách mã hóa
for s in ("Hà Nội", "tiếng Việt", "Hello world"):
    nfc, nfd = unicodedata.normalize("NFC", s), unicodedata.normalize("NFD", s)
    print(f"{s!r}: NFC {len(nfc)} ký tự / {len(nfc.encode())} byte; NFD {len(nfd)} ký tự / {len(nfd.encode())} byte")
assert len(unicodedata.normalize("NFC", "tiếng Việt").encode()) == 14 and len(unicodedata.normalize("NFD", "tiếng Việt").encode()) == 18

# (d) BPE mức từ của Sennrich (có </w>), để đối chiếu ví dụ giải mẫu 9.1
def word_bpe(vocab, n):
    merges = []
    for _ in range(n):
        pairs = {}
        for w, f in vocab.items():
            s = w.split()
            for a, b in zip(s, s[1:]): pairs[(a, b)] = pairs.get((a, b), 0) + f
        best = max(pairs, key=pairs.get); merges.append((best, pairs[best])); new = {}
        for w, f in vocab.items():
            s, o, i = w.split(), [], 0
            while i < len(s):
                if i < len(s) - 1 and (s[i], s[i + 1]) == best: o.append(s[i] + s[i + 1]); i += 2
                else: o.append(s[i]); i += 1
            new[" ".join(o)] = f
        vocab = new
    return merges
sen = word_bpe({"l o w </w>": 5, "l o w e r </w>": 2, "n e w e s t </w>": 6, "w i d e s t </w>": 3}, 10)
print([f"{a}+{b}:{c}" for (a, b), c in sen])
assert [c for _, c in sen[:3]] == [9, 9, 9]                   # ba phép đầu cùng tần số 9
print("Lab 1 ĐẠT ✔")'''

# ---------------------------------------------------------------- Lab 2
PRE_DATA = r'''PA = markov_matrix(V, 0.25, 0)
dataA = sample_markov(PA, 60000, 1)
trA, vaA = dataA[:50000], dataA[50000:]
print("train", len(trA), "val", len(vaA))'''

PRE = (
r'''def entropy_rate(P):
    """Entropy rate của chuỗi Markov (nat/token) = -sum_i pi_i sum_j P_ij ln P_ij, pi là phân phối dừng.
    Tính pi bằng cách nhân lặp pi <- pi P từ phân phối đều."""
    # TODO
    raise NotImplementedError

def filter_probs(logits, T=1.0, top_k=None, top_p=1.0):
    """Trả về phân phối (vector xác suất, tổng 1) sau khi áp nhiệt độ T, top-k, top-p.
    Thứ tự: chia logit cho T -> softmax -> sắp giảm dần -> cắt -> chuẩn hóa lại.
    Token làm tích lũy VƯỢT p vẫn phải được giữ."""
    # TODO
    raise NotImplementedError''',
r'''def entropy_rate(P):
    pi = np.full(len(P), 1 / len(P))
    for _ in range(2000):
        pi = pi @ P
    return float(-(pi[:, None] * P * np.log(P)).sum())

def filter_probs(logits, T=1.0, top_k=None, top_p=1.0):
    p = torch.softmax(logits / T, dim=-1)                  # T chia trên LOGIT, không phải trên xác suất
    sp, si = torch.sort(p, descending=True)
    keep = (torch.cumsum(sp, -1) - sp) < top_p             # giữ tới khi tích lũy TRƯỚC token này đã đạt p
    if top_k is not None:
        keep &= torch.arange(len(sp)) < top_k
    out = torch.zeros_like(p); out[si[keep]] = sp[keep]
    return out / out.sum()''')

PRE_CHECK = r'''# (a) tiền huấn luyện: dự đoán token kế tiếp, 300 bước
H = entropy_rate(PA); floor = math.exp(H)
torch.manual_seed(0); base = TinyGPT(V)
l0 = eval_loss(base, vaA); train(base, trA, 300); l1 = eval_loss(base, vaA)
print(f"H = {H:.4f} nat -> perplexity thấp nhất có thể {floor:.3f}")
print(f"loss ban đầu {l0:.3f} (ln V = {math.log(V):.3f}); sau 300 bước {l1:.4f} -> perplexity {math.exp(l1):.3f}; cách H {l1 - H:.3f} nat")
assert abs(H - 2.1025) < 1e-3
assert l0 < math.log(V) + 0.3
assert H - 0.02 <= l1 <= H + 0.15, "loss val không thể thấp hơn entropy rate (trừ nhiễu mẫu)"

# (b) perplexity không so được giữa các tokenizer: cùng 400 byte, cùng tổng NLL = 100 ln 2
total_nll, n_byte = 100 * math.log(2), 400
for n_tok in (100, 60, 400):
    print(f"{n_tok:3d} token: perplexity {math.exp(total_nll / n_tok):.4f} | bit/byte {total_nll / math.log(2) / n_byte:.2f}")
assert abs(math.exp(total_nll / 60) - 3.1748) < 1e-3

# (c) giải mã trên logit (3, 2, 1, 0, -1) của ví dụ 9.4
z = torch.tensor([3., 2., 1., 0., -1.])
for T, kept in ((1.0, 3), (0.5, 2), (2.0, 4)):
    q = filter_probs(z, T=T, top_p=0.9); print(f"T = {T}, p = 0.9 -> giữ {int((q > 0).sum())} token:", q.numpy().round(4))
    assert int((q > 0).sum()) == kept and abs(q.sum().item() - 1) < 1e-6
assert np.allclose(filter_probs(z, top_p=0.9).numpy(), [0.6652, 0.2447, 0.0900, 0, 0], atol=1e-4)
assert np.allclose(filter_probs(z, top_k=2).numpy(), filter_probs(z, top_p=0.7).numpy())
H_T = lambda T: float(-(lambda p: (p * p.log()).sum())(torch.softmax(z / T, 0)))
assert abs(H_T(0.5) - 0.4579) < 1e-3 and abs(H_T(1.0) - 1.0) < 1e-3 and abs(H_T(2.0) - 1.3943) < 1e-3

# (d) tần số 200 000 mẫu khớp phân phối
g = torch.Generator().manual_seed(0)
for T in (0.5, 1.0, 2.0):
    q = filter_probs(z, T=T)
    f = torch.bincount(torch.multinomial(q, 200000, replacement=True, generator=g), minlength=5).double() / 200000
    assert (f - torch.softmax(z / T, 0).double()).abs().max() < 0.005
print("Lab 2 ĐẠT ✔")'''

# ---------------------------------------------------------------- Lab 3
SCALE_DATA = r'''TRUE = dict(E=1.8172, A=482.01, B=2085.43, a=0.3478, b=0.3658)     # fit lại của Epoch AI (đã công bố)
PUB = dict(E=1.69, A=406.4, B=410.7, a=0.34, b=0.28)                # hằng số Chinchilla, cách 3, số mũ làm tròn như đã in
PUB2 = dict(PUB, a=0.339, b=0.285)                                   # cùng E, A, B nhưng số mũ đủ chữ số (theo tóm tắt bài Epoch)
Ns = np.array([5e7, 1e8, 2e8, 5e8, 1e9, 2e9, 5e9, 1e10]); Ds = np.array([2e9, 5e9, 1e10, 2e10, 5e10, 1e11, 2e11, 5e11])
NN, DD = np.meshgrid(Ns, Ds); N_ = NN.ravel(); D_ = DD.ravel()
rng = np.random.default_rng(0)
true_loss = lambda N, D: TRUE["E"] + TRUE["A"] / N ** TRUE["a"] + TRUE["B"] / D ** TRUE["b"]
y_ = true_loss(N_, D_) + rng.normal(0, 0.005, N_.size)               # 64 điểm có nhiễu
C_CHIN = 5.88e23                                                       # ngân sách của Chinchilla, FLOPs
print(len(y_), "điểm; loss từ", y_.min().round(3), "đến", y_.max().round(3))'''

SCALE = (
r'''def loss_surface(N, D, E, A, B, a, b):
    """L(N, D) = E + A / N^a + B / D^b (chạy được với mảng numpy)."""
    # TODO
    raise NotImplementedError

def fit_scaling(N, D, y, n_starts=5):
    """Fit 5 tham số (E, A, B, a, b) bằng scipy.optimize.least_squares. Tham số hóa A, B qua log để luôn dương;
    thử n_starts điểm khởi tạo ngẫu nhiên (default_rng(s) cho s = 0..n_starts-1), giữ nghiệm có cost nhỏ nhất.
    Trả về (E, A, B, a, b)."""
    # TODO
    raise NotImplementedError

def optimal_allocation(C, E, A, B, a, b):
    """Cực tiểu hóa L(N, C / (6N)) theo N (dùng minimize_scalar trên ln N, khoảng 1e6..1e14).
    Trả về (N*, D*)."""
    # TODO
    raise NotImplementedError''',
r'''def loss_surface(N, D, E, A, B, a, b):
    return E + A / N ** a + B / D ** b

def fit_scaling(N, D, y, n_starts=5):
    def resid(th):
        E, lA, lB, a, b = th
        return loss_surface(N, D, E, math.exp(lA), math.exp(lB), a, b) - y
    best = None
    for s in range(n_starts):
        r = np.random.default_rng(s)
        x0 = [r.uniform(1, 2.5), math.log(r.uniform(100, 1000)), math.log(r.uniform(100, 3000)), r.uniform(.2, .5), r.uniform(.2, .5)]
        sol = least_squares(resid, x0)
        if best is None or sol.cost < best.cost: best = sol
    E, lA, lB, a, b = best.x
    return E, math.exp(lA), math.exp(lB), a, b

def optimal_allocation(C, E, A, B, a, b):
    f = lambda ln: loss_surface(math.exp(ln), C / (6 * math.exp(ln)), E, A, B, a, b)
    r = minimize_scalar(f, bounds=(math.log(1e6), math.log(1e14)), method="bounded", options={"xatol": 1e-10})
    n = math.exp(r.x); return n, C / (6 * n)''')

SCALE_CHECK = r'''# (a) fit đầy đủ 64 điểm, rồi tìm điểm tối ưu ở ngân sách của Chinchilla
par = fit_scaling(N_, D_, y_)
rmse = math.sqrt(np.mean((loss_surface(N_, D_, *par) - y_) ** 2))
n1, d1 = optimal_allocation(C_CHIN, *par)
nt, dt = optimal_allocation(C_CHIN, *TRUE.values())
print("fit: E %.4f A %.1f B %.1f a %.4f b %.4f | RMSE dư %.4f" % (*par, rmse))
print(f"tối ưu từ fit : N* = {n1 / 1e9:.1f}B, D* = {d1 / 1e12:.2f}T, D/N = {d1 / n1:.2f}")
print(f"tối ưu thật   : N* = {nt / 1e9:.1f}B, D* = {dt / 1e12:.2f}T, D/N = {dt / nt:.2f}")
assert 16 < d1 / n1 < 21 and rmse < 0.01

# (b) hằng số công bố của cách 3 có cho ra 70B / 1.4T không? Thử cả hai cách ghi số mũ
np_, dp_ = optimal_allocation(C_CHIN, *PUB.values())
print(f"hằng số công bố, số mũ 0.34/0.28   : N* = {np_ / 1e9:.1f}B, D* = {dp_ / 1e12:.2f}T, D/N = {dp_ / np_:.1f}")
assert 80 < dp_ / np_ < 105
np2, dp2 = optimal_allocation(C_CHIN, *PUB2.values())
print(f"hằng số công bố, số mũ 0.339/0.285 : N* = {np2 / 1e9:.1f}B, D* = {dp2 / 1e12:.2f}T, D/N = {dp2 / np2:.1f}")
assert 50 < dp2 / np2 < 65 and dp_ / np_ > 1.4 * dp2 / np2      # nhạy với làm tròn số mũ, nhưng cả hai đều xa 20

# (b') kiểm chéo với chính paper: chú thích Figure 4 ghi cách 3 cho "khoảng 40B" ở ngân sách của Gopher (5.76e23, Table 3)
C_GOPHER = 5.76e23
ng, dg = optimal_allocation(C_GOPHER, *PUB2.values())
print(f"ngân sách Gopher 5.76e23, số mũ 0.339/0.285: N* = {ng / 1e9:.1f}B, D* = {dg / 1e12:.2f}T, D/N = {dg / ng:.1f} (paper: khoảng 40B)")
assert 39e9 < ng < 42e9, "với số mũ đủ chữ số, cách 3 phải cho khoảng 40B như chú thích Figure 4 của paper"

# (c) ngoại suy: chỉ fit trên vùng nhỏ (N <= 1e9, D <= 5e10)
m = (N_ <= 1e9) & (D_ <= 5e10)
par_s = fit_scaling(N_[m], D_[m], y_[m], n_starts=10)
ns, ds = optimal_allocation(C_CHIN, *par_s)
print(f"fit trên {m.sum()} điểm vùng nhỏ: D/N ngoại suy = {ds / ns:.2f} (so với {dt / nt:.2f} của fit đầy đủ)")
assert abs(ds / ns - dt / nt) / (dt / nt) > 0.15, "fit trên vùng nhỏ phải ngoại suy lệch tỉ lệ tối ưu"
print("Lab 3 ĐẠT ✔")'''

# ---------------------------------------------------------------- Lab 4
POST_SETUP = r'''DT = torch.float64
K = 6
r_star = torch.tensor([0., .5, 1., 1.5, 2., 2.5], dtype=DT)                    # reward "vàng" của 6 đáp án
ref_lp = torch.log_softmax(torch.tensor([.5, 0., -.3, .2, -.5, .1], dtype=DT), 0)   # log pi_ref
beta_dpo = 0.5'''

POST = (
r'''def dpo_loss(lp_w, lp_l, ref_w, ref_l, beta):
    """Loss DPO trung bình: -log sigmoid(beta * [(lp_w - ref_w) - (lp_l - ref_l)]).
    Các đối số là log-xác suất của đáp án thắng (w) và thua (l) dưới pi_theta và pi_ref."""
    # TODO
    raise NotImplementedError

def closed_form(ref_lp, r, beta):
    """Nghiệm của max E[r] - beta KL(pi || pi_ref): pi* tỉ lệ với pi_ref * exp(r / beta). Trả về xác suất."""
    # TODO
    raise NotImplementedError

def group_advantages(r, eps=1e-8):
    """Advantage theo nhóm của GRPO: (r - trung bình) / (độ lệch chuẩn quần thể + eps).
    Nhóm có độ lệch chuẩn bằng 0 thì trả về toàn 0."""
    # TODO
    raise NotImplementedError

def grpo_loss(lp_new, lp_old, lp_ref, adv, clip=0.2, kl_beta=0.01):
    """-(trung bình của min(ratio * A, clip(ratio) * A) - kl_beta * trung bình KL k3),
    ratio = exp(lp_new - lp_old), k3 = u - ln u - 1 với u = exp(lp_ref - lp_new)."""
    # TODO
    raise NotImplementedError''',
r'''def dpo_loss(lp_w, lp_l, ref_w, ref_l, beta):
    z = beta * ((lp_w - ref_w) - (lp_l - ref_l))
    return -F.logsigmoid(z).mean()

def closed_form(ref_lp, r, beta):
    return torch.softmax(ref_lp + r / beta, 0)

def group_advantages(r, eps=1e-8):
    s = r.std(unbiased=False)
    return torch.zeros_like(r) if s == 0 else (r - r.mean()) / (s + eps)

def grpo_loss(lp_new, lp_old, lp_ref, adv, clip=0.2, kl_beta=0.01):
    ratio = torch.exp(lp_new - lp_old)
    obj = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
    u = torch.exp(lp_ref - lp_new)
    k3 = (u - torch.log(u) - 1).mean()
    return -(obj - kl_beta * k3)''')

POST_CHECK = r'''# (a) kiểm tra khởi đầu và ví dụ 9.7: loss DPO = ln 2 khi pi_theta = pi_ref; z = 0.4 ra 0.5130
t = lambda *v: torch.tensor(v, dtype=DT)
assert abs(dpo_loss(t(-3.), t(-5.), t(-3.), t(-5.), 0.1).item() - math.log(2)) < 1e-12
L7 = dpo_loss(t(-12.), t(-15.), t(-14.), t(-13.), 0.1).item()
print("loss DPO ví dụ 9.7:", round(L7, 4)); assert abs(L7 - 0.5130) < 1e-4

# (b) gradient autograd khớp công thức  -beta * sigmoid(-z) * (grad lp_w - grad lp_l)
th = torch.tensor([0.3, -0.2, 0.5], dtype=DT, requires_grad=True); lp = torch.log_softmax(th, 0)
ref3 = torch.log(torch.full((3,), 1 / 3, dtype=DT))
loss = dpo_loss(lp[0:1], lp[2:3], ref3[0:1], ref3[2:3], 0.1); loss.backward()
with torch.no_grad():
    zz = 0.1 * ((lp[0] - ref3[0]) - (lp[2] - ref3[2]))
    gw = torch.autograd.grad(lp[0], th, retain_graph=True)[0] if th.grad is None else None
th2 = th.detach().clone().requires_grad_(True); l2 = torch.log_softmax(th2, 0)
gw = torch.autograd.grad(l2[0], th2, retain_graph=True)[0]; gl = torch.autograd.grad(l2[2], th2)[0]
formula = -0.1 * torch.sigmoid(-zz) * (gw - gl)
print("autograd", th.grad.numpy().round(4), "| công thức", formula.numpy().round(4))
assert (th.grad - formula).abs().max() < 1e-12

# (c) DPO trên 20 000 cặp ưa thích kiểu Bradley-Terry sinh từ reward vàng -> hội tụ về nghiệm đóng
pistar = closed_form(ref_lp, r_star, beta_dpo)
g = torch.Generator().manual_seed(0)
a = torch.randint(0, K, (20000,), generator=g); b = torch.randint(0, K, (20000,), generator=g)
mk = a != b; a, b = a[mk], b[mk]
win_first = torch.rand(len(a), generator=g) < torch.sigmoid(r_star[a] - r_star[b])
win, lose = torch.where(win_first, a, b), torch.where(win_first, b, a)
theta = torch.zeros(K, dtype=DT, requires_grad=True); opt = torch.optim.Adam([theta], lr=0.05)
for i in range(600):
    lpt = torch.log_softmax(theta, 0)
    loss = dpo_loss(lpt[win], lpt[lose], ref_lp[win], ref_lp[lose], beta_dpo)
    opt.zero_grad(); loss.backward(); opt.step()
pi = torch.softmax(theta.detach(), 0)
tv = 0.5 * (pi - pistar).abs().sum().item(); imp = beta_dpo * (torch.log(pi) - ref_lp)
print("pi* (nghiệm đóng):", pistar.numpy().round(4)); print("pi  (DPO)        :", pi.numpy().round(4))
print(f"TV(pi, pi*) = {tv:.3f}; TV(pi_ref, pi*) = {0.5 * (ref_lp.exp() - pistar).abs().sum().item():.3f}")
print("reward ngầm (trừ hằng số):", (imp - imp[0]).numpy().round(3), "so với", r_star.numpy())
assert tv < 0.05

# (d) GRPO trên bandit: chỉ đáp án 3 đúng, ban đầu xác suất 0.0264
def run_grpo(steps=300, G=8, correct=3, seed=0):
    torch.manual_seed(seed)
    th0 = torch.zeros(K, dtype=DT); th0[correct] = -2.0
    lp_ref = torch.log_softmax(th0, 0); th = th0.clone().requires_grad_(True)
    opt = torch.optim.Adam([th], lr=0.1); hist, zero = [], 0
    for _ in range(steps):
        ys = torch.multinomial(torch.softmax(th.detach(), 0), G, replacement=True)
        r = (ys == correct).to(DT); zero += int(r.std(unbiased=False) == 0)
        lp = torch.log_softmax(th, 0)[ys]
        loss = grpo_loss(lp, lp.detach(), lp_ref[ys], group_advantages(r))
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(torch.softmax(th.detach(), 0)[correct].item())
    return hist, zero
adv = group_advantages(t(1., 0., 0., 1., 0., 0., 0., 1.))
print("advantage ví dụ 9.8:", adv.numpy().round(4)); assert abs(adv[0].item() - 1.2910) < 1e-4 and abs(adv.sum().item()) < 1e-9
assert group_advantages(t(0., 0., 0., 0.)).abs().sum() == 0
hist, zero = run_grpo()
print("p(đúng) tại bước 0/50/100/299:", [round(hist[i], 3) for i in (0, 50, 100, 299)], "| nhóm không có tín hiệu:", zero, "/ 300")
assert hist[100] > 0.9
print("Lab 4 ĐẠT ✔")'''

# ---------------------------------------------------------------- Lab 5
LORA_SETUP = r'''perm = np.arange(V)
for i, j in [(0, 1), (2, 3), (4, 5), (6, 7)]:
    perm[i], perm[j] = perm[j], perm[i]
PB = PA[:, perm]                                   # chuỗi B: hoán 4 cặp token ĐẦU RA của chuỗi A
dataB = sample_markov(PB, 60000, 2); trB, vaB = dataB[:50000], dataB[50000:]
HB = entropy_rate(PB)
n_train = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
ev = lambda m, d: eval_loss(m, d)
print("nền (đã học chuỗi A): loss A", round(ev(base, vaA), 4), "| loss B", round(ev(base, vaB), 4), "| sàn B", round(HB, 4))'''

LORA = (
r'''class LoRALinear(nn.Module):
    """h = W0 x + b + (alpha / r) * B A x. Đóng băng lớp nền; A ~ Gauss(0, 1/in_features); B khởi tạo sao cho lúc đầu mô hình không đổi."""
    def __init__(self, base, r, alpha):
        super().__init__()
        # TODO: lưu base, scale = alpha / r, in_features, out_features (train() đọc head.out_features);
        # đóng băng base; tạo A (r, in) và B (out, r)
        raise NotImplementedError
    def forward(self, x):
        # TODO
        raise NotImplementedError
    @torch.no_grad()
    def merge(self):
        """Gộp BA vào trọng số nền (nhớ hệ số alpha / r) và trả về lớp nn.Linear đã gộp."""
        # TODO
        raise NotImplementedError

def add_lora(model, r, alpha, targets=("qkv", "proj", "fc1", "fc2", "head")):
    for p in model.parameters(): p.requires_grad_(False)
    def rec(mod):
        for n, c in list(mod.named_children()):
            if isinstance(c, nn.Linear) and n in targets: setattr(mod, n, LoRALinear(c, r, alpha))
            else: rec(c)
    rec(model); return model

NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                    0.7229568362236023, 1.0])      # 16 mức NF4 (lượng tử của N(0, 1), chuẩn hóa về [-1, 1])

def quant_absmax(W, bits, block=None):
    """Lượng tử hóa giả (quantize rồi dequantize) đối xứng theo absmax: mức nguyên -qmax..qmax, qmax = 2^(bits-1) - 1.
    block=None: một hằng số cho cả tensor; block=b: mỗi khối b phần tử liên tiếp có hằng số riêng."""
    # TODO
    raise NotImplementedError

def quant_nf4(W, block=16):
    """Chia khối, chia mỗi khối cho absmax của nó, rồi thay mỗi giá trị bằng mức NF4 gần nhất."""
    # TODO
    raise NotImplementedError''',
r'''class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.in_features, self.out_features = base.in_features, base.out_features   # lớp head cần out_features
        for p in self.base.parameters(): p.requires_grad_(False)
        g = torch.Generator().manual_seed(0)
        self.A = nn.Parameter(torch.randn(r, base.in_features, generator=g) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))          # B = 0  =>  Delta W = 0 lúc đầu
    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale
    @torch.no_grad()
    def merge(self):
        self.base.weight += (self.B @ self.A) * self.scale
        return self.base

def add_lora(model, r, alpha, targets=("qkv", "proj", "fc1", "fc2", "head")):
    for p in model.parameters(): p.requires_grad_(False)
    def rec(mod):
        for n, c in list(mod.named_children()):
            if isinstance(c, nn.Linear) and n in targets: setattr(mod, n, LoRALinear(c, r, alpha))
            else: rec(c)
    rec(model); return model

NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                    0.7229568362236023, 1.0])      # 16 mức NF4 (lượng tử của N(0, 1), chuẩn hóa về [-1, 1])

def quant_absmax(W, bits, block=None):
    qmax = 2 ** (bits - 1) - 1
    Wb = W.reshape(1, -1) if block is None else W.reshape(-1, block)
    s = Wb.abs().max(1, keepdim=True).values / qmax
    return (torch.round(Wb / s).clamp(-qmax, qmax) * s).reshape(W.shape)

def quant_nf4(W, block=16):
    Wb = W.reshape(-1, block); s = Wb.abs().max(1, keepdim=True).values
    idx = ((Wb / s).unsqueeze(-1) - NF4).abs().argmin(-1)
    return (NF4[idx] * s).reshape(W.shape)''')

LORA_CHECK = r'''# (a) kiểm khởi tạo: lúc đầu LoRA không đổi gì
x = trB[:24].unsqueeze(0).repeat(2, 1)
m0 = add_lora(copy.deepcopy(base), 4, 8).eval(); b0 = copy.deepcopy(base).eval()
assert torch.allclose(m0(x), b0(x), atol=1e-6); print("đầu ra LoRA lúc đầu = đầu ra nền")

# (b) đếm tham số huấn luyện, rồi tinh chỉnh sang chuỗi B (200 bước)
full = copy.deepcopy(base); n_full = n_train(full); train(full, trB, 200, lr=3e-3)
rows = [("toàn phần", n_full, ev(full, vaB), ev(full, vaA))]
for r in (1, 4, 16):
    m = add_lora(copy.deepcopy(base), r, 2 * r); n = n_train(m); train(m, trB, 200, lr=1e-2)
    rows.append((f"LoRA r = {r}", n, ev(m, vaB), ev(m, vaA)))
    if r == 4: m4 = m
print(f"{'cách':14s} {'tham số':>8s} {'loss B':>8s} {'loss A (quên)':>14s}")
for nm, n, lb, la in rows: print(f"{nm:14s} {n:8d} {lb:8.4f} {la:14.4f}")
assert [r_[1] for r_ in rows] == [60096, 1608, 6432, 25728]
assert rows[2][2] < ev(base, vaB) - 0.5 and abs(rows[2][2] - rows[0][2]) < 0.05

# (c) gộp adapter: không thêm độ trễ suy luận
m4.eval(); o1 = m4(x)
def merge_all(mod):
    for n, c in list(mod.named_children()):
        if isinstance(c, LoRALinear): setattr(mod, n, c.merge())
        else: merge_all(c)
merge_all(m4); dif = (o1 - m4(x)).abs().max().item(); print("sai lệch sau khi gộp:", f"{dif:.1e}"); assert dif < 1e-4

# (d) lượng tử hóa trọng số mô hình nền, đo loss A
def quant_model(model, fn):
    mm = copy.deepcopy(model)
    with torch.no_grad():
        for mod in mm.modules():
            if isinstance(mod, nn.Linear) and mod.weight.numel() % 16 == 0: mod.weight.copy_(fn(mod.weight))
    return mm
res = {"fp32": ev(base, vaA)}
for nm, fn in (("int8 khối 16", lambda W: quant_absmax(W, 8, 16)), ("int4 theo tensor", lambda W: quant_absmax(W, 4)),
               ("int4 khối 16", lambda W: quant_absmax(W, 4, 16)), ("NF4 khối 16", lambda W: quant_nf4(W, 16)),
               ("int2 khối 16", lambda W: quant_absmax(W, 2, 16))):
    res[nm] = ev(quant_model(base, fn), vaA)
for k_, v_ in res.items(): print(f"{k_:18s} loss A = {v_:.4f}")
assert abs(res["int8 khối 16"] - res["fp32"]) < 0.001 and res["int2 khối 16"] > 2.25

# (e) trên trọng số ĐÚNG Gauss, NF4 có sai số thấp hơn int4 không?
Wg = torch.randn(1 << 20, generator=torch.Generator().manual_seed(0)) * 0.02
for bs in (64, 256, 4096):
    e4 = (quant_absmax(Wg, 4, bs) - Wg).pow(2).mean().sqrt().item(); en = (quant_nf4(Wg, bs) - Wg).pow(2).mean().sqrt().item()
    print(f"khối {bs:4d}: RMSE int4 {e4:.6f} | NF4 {en:.6f} | NF4 giảm {100 * (1 - en / e4):.1f}%")
    assert en < e4
print("Lab 5 ĐẠT ✔")'''

# ---------------------------------------------------------------- Bug hunt
BUG_CODE = r'''# ======== Đường ống 1: SFT (prompt 4 số, đáp án a_i = (p_i + 3) mod 16) + sinh tham lam ========
NC, SEP, EOS, PAD, VS, LP = 16, 16, 17, 18, 19, 4
TT = 2 * LP + 2
def make_sft(n, seed):
    g = np.random.default_rng(seed); p = g.integers(0, NC, (n, LP)); a = (p + 3) % NC
    return torch.tensor(np.concatenate([p, np.full((n, 1), SEP), a, np.full((n, 1), EOS)], 1))

def sft_buggy(seq, steps=400, seed=0, lr=3e-3, B=64):
    torch.manual_seed(seed); m = TinyGPT(VS, T=TT)
    opt = torch.optim.AdamW(m.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        ix = torch.randint(0, len(seq), (B,), generator=g)
        x, y = seq[ix, :-1], seq[ix, 1:].clone()
        y[:, :LP] = -100
        y[y >= NC] = -100
        loss = F.cross_entropy(m(x).reshape(-1, VS), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def generate(m, prompts, max_new=12):
    m.eval(); x = torch.cat([prompts, torch.full((len(prompts), 1), SEP)], 1)
    done = torch.zeros(len(x), dtype=torch.bool); n_new = torch.zeros(len(x))
    for _ in range(max_new):
        nxt = m(x[:, -TT:])[:, -1].argmax(-1); nxt[done] = PAD
        x = torch.cat([x, nxt[:, None]], 1); n_new += (~done).float(); done |= (nxt == EOS)
    m.train(); return x, done, n_new

def eval_sft(m, te):
    out, done, n_new = generate(m, te[:, :LP])
    em = (out[:, LP + 1:LP + 1 + LP] == te[:, LP + 1:LP + 1 + LP]).all(1).float().mean().item()
    return em, done.float().mean().item(), n_new.mean().item()

# ======== Đường ống 2: mô hình ngôn ngữ trên chuỗi có độ dài khác nhau, đệm bằng PAD ========
PADT, VP = V, V + 1
def make_padded(P, n, seed, Tm=24):
    rng = np.random.default_rng(seed); cum = np.cumsum(P, 1); out = np.full((n, Tm), PADT)
    for i in range(n):
        l = rng.integers(6, Tm + 1); x = rng.integers(V)
        for t in range(l):
            out[i, t] = x; x = min(np.searchsorted(cum[x], rng.random()), V - 1)
    return torch.tensor(out)

def lm_buggy(trn, val, steps=300):
    torch.manual_seed(0); m = TinyGPT(VP); opt = torch.optim.AdamW(m.parameters(), lr=3e-3); g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        ix = torch.randint(0, len(trn), (32,), generator=g); x, y = trn[ix, :-1], trn[ix, 1:]
        loss = F.cross_entropy(m(x).reshape(-1, VP), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        nll = F.cross_entropy(m(val[:, :-1]).reshape(-1, VP), val[:, 1:].reshape(-1), reduction="none")
    return nll.mean().exp().item()

# ======== Đường ống 3: bộ lấy mẫu ========
def sample_buggy(logits, T=1.0, top_p=1.0, n=1, gen=None):
    p = torch.softmax(logits, -1) / T
    p = p / p.sum()
    cs = torch.cumsum(p, -1); keep = (cs - p) < top_p
    q = p * keep; q = q / q.sum()
    return torch.multinomial(q, n, replacement=True, generator=gen)

# ======== Đường ống 4: LoRA ========
class LoRABuggy(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__(); self.base, self.scale = base, alpha / r
        self.in_features, self.out_features = base.in_features, base.out_features
        for p in self.base.parameters(): p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.randn(base.out_features, r) * 0.5)
    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

def wrap_buggy(mod, r=4, alpha=8):
    for n, c in list(mod.named_children()):
        if isinstance(c, nn.Linear) and n in ("qkv", "proj", "fc1", "fc2", "head"): setattr(mod, n, LoRABuggy(c, r, alpha))
        else: wrap_buggy(c, r, alpha)
    return mod'''

BUG_DIAG = r'''# ---- Chạy cả bốn đường ống và in triệu chứng (mỗi dòng là một bất thường bạn phải giải thích)
try:
    entropy_rate(markov_matrix(V, 0.25, 0)); base
except (NotImplementedError, NameError):
    raise RuntimeError("Bài săn bug dùng entropy_rate và mô hình `base` của Lab 2: làm xong Lab 2 và chạy ô kiểm của nó trước.")
tr_s, te_s = make_sft(5000, 0), make_sft(500, 1)
em, stop, nn_ = eval_sft(sft_buggy(tr_s), te_s)
print(f"[1] SFT : exact-match {em:.3f} | tỉ lệ tự dừng {stop:.3f} | token mới trung bình {nn_:.2f} (giới hạn 12)")

PH = entropy_rate(PA); trn_p, val_p = make_padded(PA, 3000, 5), make_padded(PA, 500, 6)
print(f"[2] LM  : perplexity báo cáo {lm_buggy(trn_p, val_p):.3f} | cận dưới lý thuyết exp(H) = {math.exp(PH):.3f}")

lg = torch.tensor([1., 3., 0., 2., -1.]); gen = torch.Generator().manual_seed(0)
freq = lambda T, tp: torch.bincount(sample_buggy(lg, T, tp, 200000, gen), minlength=5).double() / 200000
ent = lambda c: float(-(c[c > 0] * c[c > 0].log()).sum())
print("[3] lấy mẫu: entropy của 200 000 mẫu ở T = 0.5 / 1 / 2:", [round(ent(freq(T, 1.0)), 4) for T in (0.5, 1.0, 2.0)],
      "| giải tích:", [round(ent(torch.softmax(lg / T, 0).double()), 4) for T in (0.5, 1.0, 2.0)])
print("[4] top-p = 0.9, tần số 5 token (logit 1, 3, 0, 2, -1):", freq(1.0, 0.9).numpy().round(4))

mb = wrap_buggy(copy.deepcopy(base)); print(f"[5] LoRA : loss lúc khởi tạo {ev(mb, vaA):.4f} | loss mô hình nền {ev(base, vaA):.4f}")'''

BUG_FIX_TODO = r'''# TODO: viết bản sửa cho từng đường ống. Giữ nguyên chữ ký hàm; đừng sửa thêm gì ngoài lỗi.
def sft_fixed(seq, steps=400, seed=0, lr=3e-3, B=64):
    raise NotImplementedError

def lm_fixed(trn, val, steps=300):
    """Trả về perplexity trên val."""
    raise NotImplementedError

def sample_fixed(logits, T=1.0, top_p=1.0, n=1, gen=None):
    raise NotImplementedError

class LoRAFixed(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        raise NotImplementedError
    def forward(self, x):
        raise NotImplementedError'''

BUG_FIX = r'''def sft_fixed(seq, steps=400, seed=0, lr=3e-3, B=64):
    torch.manual_seed(seed); m = TinyGPT(VS, T=TT)
    opt = torch.optim.AdamW(m.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        ix = torch.randint(0, len(seq), (B,), generator=g)
        x, y = seq[ix, :-1], seq[ix, 1:].clone()
        y[:, :LP] = -100                                  # chỉ che prompt; EOS PHẢI nằm trong mục tiêu
        loss = F.cross_entropy(m(x).reshape(-1, VS), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
    return m

def lm_fixed(trn, val, steps=300):
    torch.manual_seed(0); m = TinyGPT(VP); opt = torch.optim.AdamW(m.parameters(), lr=3e-3); g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        ix = torch.randint(0, len(trn), (32,), generator=g); x, y = trn[ix, :-1], trn[ix, 1:].clone()
        y[y == PADT] = -100                               # che PAD ở huấn luyện
        loss = F.cross_entropy(m(x).reshape(-1, VP), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        y = val[:, 1:].reshape(-1)
        nll = F.cross_entropy(m(val[:, :-1]).reshape(-1, VP), y, reduction="none")
    return nll[y != PADT].mean().exp().item()             # che PAD ở đo

def sample_fixed(logits, T=1.0, top_p=1.0, n=1, gen=None):
    p = torch.softmax(logits / T, -1)                     # chia T trên logit
    sp, si = torch.sort(p, descending=True)               # sắp giảm dần rồi mới cộng dồn
    keep = (torch.cumsum(sp, -1) - sp) < top_p
    q = sp * keep; q = q / q.sum()
    return si[torch.multinomial(q, n, replacement=True, generator=gen)]

class LoRAFixed(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__(); self.base, self.scale = base, alpha / r
        self.in_features, self.out_features = base.in_features, base.out_features
        for p in self.base.parameters(): p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))                   # B = 0
    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale'''

BUG_CHECK = r'''em, stop, nn_ = eval_sft(sft_fixed(tr_s), te_s)
print(f"[1] sửa: exact-match {em:.3f} | dừng {stop:.3f} | token mới {nn_:.2f}"); assert stop == 1.0 and em > 0.99
ppl = lm_fixed(trn_p, val_p)
print(f"[2] sửa: perplexity {ppl:.3f} (cận dưới {math.exp(PH):.3f})"); assert math.exp(PH) <= ppl <= 1.1 * math.exp(PH)
e_fix = [ent(torch.bincount(sample_fixed(lg, T, 1.0, 200000, gen), minlength=5).double() / 200000) for T in (0.5, 1.0, 2.0)]
print("[3] sửa: entropy mẫu", np.round(e_fix, 4)); assert e_fix[0] < e_fix[1] < e_fix[2]
assert all(abs(e - ent(torch.softmax(lg / T, 0).double())) < 0.02 for e, T in zip(e_fix, (0.5, 1.0, 2.0)))
fq = torch.bincount(sample_fixed(lg, 1.0, 0.9, 200000, gen), minlength=5).double() / 200000
print("[4] sửa: tần số top-p 0.9", fq.numpy().round(4))
pr = torch.softmax(lg, 0).double(); ok_sets = [S for S in (set(c) for k in range(1, 6) for c in itertools.combinations(range(5), k))
                                               if sum(pr[i].item() for i in S) >= 0.9]
nucleus = min(ok_sets, key=lambda S: (len(S), -sum(pr[i].item() for i in S)))   # định nghĩa top-p: tập nhỏ nhất đủ khối lượng p
assert set(torch.nonzero(fq).flatten().tolist()) == nucleus and abs(fq.sum().item() - 1) < 1e-9
mfix = copy.deepcopy(base)
for p in mfix.parameters(): p.requires_grad_(False)
def wrap_fixed(mod):
    for n, c in list(mod.named_children()):
        if isinstance(c, nn.Linear) and n in ("qkv", "proj", "fc1", "fc2", "head"): setattr(mod, n, LoRAFixed(c, 4, 8))
        else: wrap_fixed(c)
wrap_fixed(mfix); l_fix = ev(mfix, vaA)
print(f"[5] sửa: loss lúc khởi tạo {l_fix:.4f} (nền {ev(base, vaA):.4f})"); assert abs(l_fix - ev(base, vaA)) < 1e-6
print("Săn bug ĐẠT ✔")'''

BUG_MUT = r'''# Kiểm đột biến: ô kiểm của lỗi 2 phải TRƯỢT với các bản sửa một nửa (chỉ che pad ở một phía)
def lm_variant(trn, val, mask_train, mask_eval, steps=300):
    torch.manual_seed(0); m = TinyGPT(VP); opt = torch.optim.AdamW(m.parameters(), lr=3e-3); g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        ix = torch.randint(0, len(trn), (32,), generator=g); x, y = trn[ix, :-1], trn[ix, 1:].clone()
        if mask_train: y[y == PADT] = -100
        loss = F.cross_entropy(m(x).reshape(-1, VP), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        y = val[:, 1:].reshape(-1); nll = F.cross_entropy(m(val[:, :-1]).reshape(-1, VP), y, reduction="none")
    return (nll[y != PADT] if mask_eval else nll).mean().exp().item()
ceil = 1.1 * math.exp(PH)
for name, mt, me in (("che pad chỉ lúc đo", False, True), ("che pad chỉ lúc huấn luyện", True, False), ("không che", False, False)):
    v = lm_variant(trn_p, val_p, mt, me)
    print(f"đột biến '{name}': perplexity {v:.3f} -> {'LỌT' if math.exp(PH) <= v <= ceil else 'trượt (đúng)'}")
    assert not (math.exp(PH) <= v <= ceil), "ô kiểm để lọt bản sửa một nửa"
assert math.exp(PH) <= lm_variant(trn_p, val_p, True, True) <= ceil
print("Kiểm đột biến ĐẠT ✔")'''

BUG_ABL = r'''# Ba phép thử "đa năng" lộ cả năm lỗi mà không cần biết đáp án:
#  (1) chỉ số hoàn hảo nhưng hành vi sai          -> exact-match 1.000 mà tỉ lệ dừng 0.000
#  (2) con số tốt hơn giới hạn lý thuyết là số sai -> perplexity < exp(H)
#  (3) tại bước 0, LoRA phải trả đúng đầu ra nền   -> so loss khởi tạo với loss nền
# và một phép thử thống kê cho bộ lấy mẫu: so tần số 200 000 mẫu với softmax(z / T) đã biết.
print(f"Tổng thời gian notebook: {time.time() - T0:.1f} s")'''


def build(solution: bool):
    pick = (lambda pair: pair[1]) if solution else (lambda pair: pair[0])
    nb = nbf.v4.new_notebook()
    head = "Lời giải" if solution else "Bài lab"
    cells = [
        md(f"# Chương 9 — {head}: Mô hình ngôn ngữ lớn\n\n"
           + ("Bản lời giải đã chạy kiểm (NumPy, SciPy, PyTorch CPU; toàn notebook khoảng 30 giây trên máy rảnh, tới hơn 2 phút khi máy bận; thời gian tự đo ở ô cuối). "
              "Chỉ mở sau khi đã tự làm `lab.ipynb`."
              if solution else
              "Viết code vào các ô có `# TODO`, rồi chạy ô kiểm tra ngay bên dưới. Ô kiểm tra in **ĐẠT ✔** là xong. "
              "Bí quá 20 phút mới mở `loigiai.ipynb`. Chỉ cần PyTorch bản CPU; toàn notebook chạy khoảng 30 giây trên máy rảnh (hơn 2 phút nếu máy bận), "
              "dữ liệu tự sinh, không cần mạng.")),
        md("## Chuẩn bị\nTransformer tí hon `TinyGPT` (chương 8), vòng huấn luyện, và nguồn Markov có entropy biết trước. "
           "Ô này gọi `torch.set_num_threads(1)`: mô hình bé không hưởng lợi từ nhiều luồng."),
        code(SETUP),
        md("## Lab 1 — BPE byte-level từ con số 0\nGhép cặp byte phổ biến nhất, mã hóa theo đúng thứ tự đã học, giải mã. "
           "Rồi đo: một tokenizer học trên tiếng Anh tốn bao nhiêu token cho câu tiếng Việt (và ngược lại)? "
           "Hai kho ngữ liệu dưới đây chỉ vài KB: số token minh họa **hướng** của hiện tượng, không phải số của GPT-2 hay Llama."),
        code(BPE_CORPUS), code(pick(BPE)), code(BPE_CHECK),
        md("## Lab 2 — Tiền huấn luyện, perplexity và giải mã\nNguồn Markov bậc 1, V = 24, nên entropy rate H tính chính xác được: "
           "không mô hình nào có loss thấp hơn H. Bạn tính H, kiểm tra loss của TinyGPT, rồi viết bộ lọc nhiệt độ, top-k, top-p."),
        code(PRE_DATA), code(pick(PRE)), code(PRE_CHECK),
        md("## Lab 3 — Quy luật quy mô: fit rồi ngoại suy\n64 điểm (N, D, loss) sinh từ hằng số fit lại của Epoch AI cộng nhiễu 0.005. "
           "Fit lại 5 tham số, tìm điểm tối ưu ở ngân sách của Chinchilla, và xem điều gì xảy ra khi chỉ fit trên vùng nhỏ."),
        code(SCALE_DATA), code(pick(SCALE)), code(SCALE_CHECK),
        md("## Lab 4 — DPO và GRPO trên bài toán đồ chơi\nChính sách là phân phối trên 6 đáp án. (a) loss DPO, nghiệm đóng của bài toán KL; "
           "(b) advantage theo nhóm và loss GRPO có cắt tỉ số và phạt KL k3. Kiểm gradient bằng autograd."),
        code(POST_SETUP), code(pick(POST)), code(POST_CHECK),
        md("## Lab 5 — LoRA và lượng tử hóa\nMô hình nền `base` (Lab 2) đã học chuỗi A. Tinh chỉnh sang chuỗi B (hoán 4 cặp token đầu ra) "
           "bằng tinh chỉnh toàn phần và LoRA, rồi lượng tử hóa trọng số nền. **Cần làm xong Lab 2 trước** (dùng `base` và `entropy_rate`). Lab này **không** hứa NF4 thắng int4 hay LoRA quên ít hơn: bạn đo và đọc số."),
        code(LORA_SETUP), code(pick(LORA)), code(LORA_CHECK),
        md("## Săn bug\nBốn đường ống dưới đây chạy không báo lỗi nhưng cho kết quả vô lý. Ô chẩn đoán in năm triệu chứng. "
           "Tìm cả **năm** lỗi (không biết ống nào chứa mấy lỗi), rồi viết bản sửa. "
           "**Cần làm xong Lab 2 trước** (ô chẩn đoán dùng `entropy_rate` và mô hình `base`)."),
        code(BUG_CODE), code(BUG_DIAG),
    ]
    if solution:
        cells += [md("### Bản sửa và kiểm"), code(BUG_FIX), code(BUG_CHECK), md("### Kiểm ô kiểm (đột biến)"), code(BUG_MUT), md("### Vì sao ba phép thử là đủ"), code(BUG_ABL)]
    else:
        cells += [md("### Viết bản sửa"), code(BUG_FIX_TODO),
                  md("### Kiểm bản sửa"),
                  code(BUG_CHECK),
                  md("Khi đã xong, in thời gian chạy: `print(f'{time.time() - T0:.1f} s')`.")]
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    name = "loigiai.ipynb" if solution else "lab.ipynb"
    nbf.write(nb, OUT / name)
    return OUT / name


if __name__ == "__main__":
    print(build(False), build(True))
