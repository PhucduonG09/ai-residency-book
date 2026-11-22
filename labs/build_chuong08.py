"""Sinh hai notebook cho lab chương 8: lab.ipynb (có TODO) và loigiai.ipynb (lời giải).

Chạy lại script này nếu muốn tạo lại notebook gốc:  python labs/build_chuong08.py
Toàn bộ loigiai.ipynb chạy khoảng 1.5 đến 3 phút trên CPU 2 lõi tùy tải máy (NumPy + PyTorch CPU, seed cố định, dữ liệu tự sinh, không cần mạng).
Mọi con số của bài săn bug (kể cả khi gài nhiều lỗi cùng lúc) được tính lại ngay trong notebook.
"""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "chuong08"
OUT.mkdir(parents=True, exist_ok=True)

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

SETUP = """import time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(2)
np.set_printoptions(precision=4, suppress=True)
T0 = time.time()
print("torch", torch.__version__, "| numpy", np.__version__)"""

# ---------------------------------------------------------------- Lab 1
ATT_NP = ("""def softmax(Z):
    \"\"\"Softmax theo trục cuối, ổn định số: trừ max từng hàng trước khi lấy exp. Hàng toàn −inf (bị che hết) trả về toàn 0, không NaN.\"\"\"
    # TODO
    raise NotImplementedError

def attention(Q, K, V, causal=False, pad=None):
    \"\"\"Scaled dot-product attention một đầu. Q, K: (n, dk); V: (n, dv).
    causal=True: đặt −∞ vào các ô j > i TRƯỚC softmax.
    pad: None hoặc mảng bool độ dài n, pad[j] = True nếu token j là token đệm (che cột j). Mặt nạ kết hợp: ô bị che nếu bất kỳ mặt nạ nào che.
    Trả về (O, A) với A = softmax(QKᵀ/√dk) có shape (n, n).\"\"\"
    # TODO
    raise NotImplementedError""",
"""def softmax(Z):
    m = Z.max(axis=-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)             # hàng toàn −inf: tránh −inf − (−inf) = NaN
    E = np.exp(Z - m)                                # ổn định số: phần tử lớn nhất thành exp(0) = 1
    s = E.sum(axis=-1, keepdims=True)
    return E / np.where(s == 0, 1.0, s)              # hàng bị che hết cho toàn 0 (một quy ước, không phải NaN)

def attention(Q, K, V, causal=False, pad=None):
    S = Q @ K.T / np.sqrt(Q.shape[-1])
    mask = np.zeros(S.shape, dtype=bool)
    if causal:
        mask |= np.triu(np.ones(S.shape, dtype=bool), 1)
    if pad is not None:
        mask |= np.asarray(pad, dtype=bool)[None, :]
    S = np.where(mask, -np.inf, S)                   # che TRƯỚC softmax
    A = softmax(S)
    return A @ V, A""")

ATT_CHECK = """# (a) Ví dụ giải mẫu 8.3: ba token, dk = 2
Q = K = np.array([[1., 0], [0, 1], [1, 1]])
V = np.array([[1., 2], [3, 4], [5, 6]])
O, A = attention(Q, K, V)
Oc, Ac = attention(Q, K, V, causal=True)
print("A =\\n", A, "\\nO =\\n", O, "\\nA (causal) =\\n", Ac, "\\nO (causal) =\\n", Oc)
assert np.allclose(O, [[3.0, 4.0], [3.4067, 4.4067], [3.5105, 4.5105]], atol=1e-4)
assert np.allclose(Oc[0], [1, 2]) and np.allclose(Oc[1], [2.3395, 3.3395], atol=1e-4) and np.allclose(Oc[2], O[2])
assert np.allclose(A.sum(1), 1) and (A >= 0).all()
# softmax ổn định: bản exp(Z)/sum ngây thơ ra NaN ở đây
z = softmax(np.array([[1000., 1001.]]))
print("softmax(1000, 1001) =", z)
assert np.isfinite(z).all() and np.allclose(z, [[0.2689, 0.7311]], atol=1e-4)
assert np.allclose(softmax(np.array([[1., 2., 3.]])), softmax(np.array([[101., 102., 103.]])))     # softmax(z) = softmax(z − c)
lo, hi = V.min(0), V.max(0)
assert ((O >= lo - 1e-12) & (O <= hi + 1e-12)).all()      # tổ hợp lồi của các hàng V

# (a2) Mặt nạ padding và mặt nạ kết hợp: token 4 là đệm (giá trị lớn để lộ lỗi nếu nó lọt vào)
Qp = Kp = np.array([[1., 0], [0, 1], [1, 1], [2, 2]])
Vp = np.array([[1., 2], [3, 4], [5, 6], [100, 100]])
padm = [False, False, False, True]
Op, Ap = attention(Qp, Kp, Vp, pad=padm)
print("A có mặt nạ padding (hàng 1) =", Ap[0], "| cột đệm =", Ap[:, 3])
assert (Ap[:, 3] == 0).all() and np.allclose(Ap.sum(1), 1)
assert np.allclose(Op[:3], attention(Qp[:3], Kp[:3], Vp[:3])[0])    # token thật không bị ảnh hưởng bởi token đệm
left = attention(Qp, Kp, Vp, causal=True, pad=[True, False, False, False])    # đệm TRÁI + causal: hàng của token đệm bị che hết
print("đệm trái + causal, hàng 1 của A =", left[1][0])
assert np.isfinite(left[0]).all() and (left[1][0] == 0).all()

# (b) Khớp F.scaled_dot_product_attention (float64)
tq, tk, tv = (torch.tensor(a)[None, None] for a in (Q, K, V))
ref = F.scaled_dot_product_attention(tq, tk, tv, is_causal=True)[0, 0].numpy()
print("max |tay − PyTorch| =", np.abs(ref - Oc).max())
assert np.allclose(ref, Oc, atol=1e-10)

# (c) Hoán vị-đẳng biến: Attn(PX) = P·Attn(X) khi không có thông tin vị trí
r = np.random.default_rng(0)
X = r.standard_normal((6, 8)); Wq, Wk, Wv = (r.standard_normal((8, 8)) / np.sqrt(8) for _ in range(3))
perm = r.permutation(6)
O1 = attention(X @ Wq, X @ Wk, X @ Wv)[0]
O2 = attention(X[perm] @ Wq, X[perm] @ Wk, X[perm] @ Wv)[0]
print("hoán vị-đẳng biến: sai khác lớn nhất =", np.abs(O1[perm] - O2).max())
assert np.allclose(O1[perm], O2, atol=1e-12)

# (d) Vì sao chia √dk: std(q·k) = √dk, và hệ quả với softmax
def std_dot(dk, n=20000, seed=1):
    g = np.random.default_rng(seed)
    return (g.standard_normal((n, dk)) * g.standard_normal((n, dk))).sum(1).std()

def saturation(dk, scale, trials=4000, nk=10, seed=2):
    \"\"\"1 query với nk key ngẫu nhiên chuẩn và một gradient ngược g ngẫu nhiên theo p. Trả về: trung bình max p; trung bình và trung vị của
    ‖J‖F (J = Jacobian softmax theo logit đưa vào softmax); trung bình và trung vị của ‖∂L/∂q‖ (L = g·p, q là query gốc, trước khi chia).\"\"\"
    g = np.random.default_rng(seed)
    q, K = g.standard_normal((trials, 1, dk)), g.standard_normal((trials, nk, dk))
    gout = g.standard_normal((trials, nk))
    c = 1 / np.sqrt(dk) if scale else 1.0
    p = softmax((q * K).sum(-1) * c)
    J = np.einsum("ti,ij->tij", p, np.eye(nk)) - p[:, :, None] * p[:, None, :]
    jf = np.sqrt((J ** 2).sum((1, 2)))
    gq = np.linalg.norm(c * np.einsum("tjd,tj->td", K[:, :, :], np.einsum("tij,tj->ti", J, gout)), axis=1)
    return p.max(1).mean(), jf.mean(), np.median(jf), gq.mean(), np.median(gq)

for dk in (4, 16, 64, 256):
    print(f"dk = {dk:3d}: std(q·k) = {std_dot(dk):6.2f}   √dk = {np.sqrt(dk):5.2f}")
assert abs(std_dot(64) / 8 - 1) < 0.05
print("   dk | cách | max p | ‖J‖F trung bình | ‖J‖F trung vị | ‖∂L/∂q‖ trung bình | ‖∂L/∂q‖ trung vị")
sat = {}
for dk in (4, 64, 512):
    for name, sc in (("không chia", False), ("có chia", True)):
        r = sat[(dk, sc)] = saturation(dk, sc)
        print(f"  {dk:4d} | {name:10s} | {r[0]:.3f} | {r[1]:.3f} | {r[2]:.5f} | {r[3]:.3f} | {r[4]:.4f}")
assert sat[(64, False)][0] > 0.8 and sat[(64, True)][0] < 0.4 and sat[(512, False)][0] > sat[(64, False)][0]
# Không chia: trung bình gradient KHÔNG nhỏ (vài mẫu có gradient lớn), nhưng trung vị gần 0: đa số query hầu như không nhận gradient.
assert sat[(512, False)][4] < 0.01 and sat[(512, True)][4] > 0.2 and sat[(512, False)][3] > sat[(512, True)][3]

# (e) Các bẫy của mặt nạ: mask SAU softmax làm hàng không còn tổng 1; fp16 tràn −1e9 thành −inf
S = Q @ K.T / np.sqrt(2)
bad = softmax(S) * np.tril(np.ones((3, 3)))
print("mask sau softmax, tổng từng hàng:", bad.sum(1))
assert np.allclose(bad.sum(1), [0.4011, 0.5989, 1.0], atol=1e-4)
h16 = torch.tensor(-1e9, dtype=torch.float16)
print("−1e9 ở fp16 =", h16.item(), "| hàng toàn −inf qua softmax:", torch.softmax(torch.full((3,), -float('inf')), 0).tolist())
assert torch.isinf(h16)
try:      # đường thứ hai: điền thẳng −1e9 vào tensor fp16 (PyTorch 2.14.1: báo lỗi, không tràn im lặng)
    torch.zeros(3, dtype=torch.float16).masked_fill(torch.tensor([True, False, False]), -1e9); fill_msg = "không báo lỗi"
except RuntimeError as e:
    fill_msg = "báo lỗi: " + str(e)[:62]
print("masked_fill(-1e9) trên tensor fp16:", fill_msg, "| giá trị an toàn torch.finfo(torch.float16).min =", torch.finfo(torch.float16).min)
print("Lab 1 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 2
MHA_CODE = ("""class MHA(nn.Module):
    \"\"\"Multi-head self-attention. Thuộc tính BẮT BUỘC (các lab sau dùng): self.h, self.qkv = nn.Linear(d, 3d), self.o = nn.Linear(d, d).
    forward(x): x (B, T, d) → (B, T, d). Tách đầu bằng view(B, T, h, dk).transpose(1, 2); gộp lại bằng transpose rồi reshape.
    causal=True: che các ô j > i bằng −∞ TRƯỚC softmax.\"\"\"
    def __init__(self, d, h, causal=True):
        super().__init__()
        # TODO
        raise NotImplementedError
    def forward(self, x):
        # TODO
        raise NotImplementedError

class Block(nn.Module):
    \"\"\"Khối pre-LN: x ← x + MHA(LN(x)); x ← x + FFN(LN(x)); FFN = Linear(d, 4d) → GELU → Linear(4d, d).
    Thuộc tính BẮT BUỘC: ln1, att, ln2, ffn (và dr = nn.Dropout(p) dùng cho cả hai nhánh).\"\"\"
    def __init__(self, d, h, causal=True, p=0.0):
        super().__init__()
        # TODO
        raise NotImplementedError
    def forward(self, x):
        # TODO
        raise NotImplementedError""",
"""class MHA(nn.Module):
    def __init__(self, d, h, causal=True):
        super().__init__()
        assert d % h == 0
        self.d, self.h, self.causal = d, h, causal
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x):
        B, T, d = x.shape
        h, dk = self.h, d // self.h
        q, k, v = self.qkv(x).split(d, dim=2)
        q, k, v = [t.view(B, T, h, dk).transpose(1, 2) for t in (q, k, v)]      # (B, h, T, dk)
        S = q @ k.transpose(-2, -1) / math.sqrt(dk)                               # (B, h, T, T)
        if self.causal:
            S = S.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float('-inf'))
        y = S.softmax(-1) @ v                                                     # (B, h, T, dk)
        return self.o(y.transpose(1, 2).reshape(B, T, d))                         # gộp đầu

class Block(nn.Module):
    def __init__(self, d, h, causal=True, p=0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = MHA(d, h, causal)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.dr = nn.Dropout(p)

    def forward(self, x):
        x = x + self.dr(self.att(self.ln1(x)))
        return x + self.dr(self.ffn(self.ln2(x)))""")

LM_CODE = """class LM(nn.Module):
    \"\"\"Mô hình ngôn ngữ nhỏ kiểu nanoGPT: embedding token + vị trí học được → L khối pre-LN → LN cuối → lm_head.
    (Cho sẵn; dùng MHA và Block của bạn.) causal=False cho bộ mã hóa hai chiều; pos=False bỏ mã hóa vị trí.\"\"\"
    def __init__(self, V=50, T=32, d=64, h=4, L=2, causal=True, pos=True, p=0.0):
        super().__init__()
        self.pos = pos
        self.te, self.pe = nn.Embedding(V, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block(d, h, causal, p) for _ in range(L)])
        self.lnf, self.head = nn.LayerNorm(d), nn.Linear(d, V)

    def forward(self, idx):
        x = self.te(idx) + (self.pe(torch.arange(idx.shape[1])) if self.pos else 0)
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))

def n_params(m):
    return sum(p.numel() for p in m.parameters())"""

MHA_CHECK = """# (a) Ví dụ giải mẫu 8.8: khối tí hon d = 4, d_ff = 16 (2 đầu)
blk = Block(4, 2)
parts = (n_params(blk.att), n_params(blk.ffn), n_params(blk.ln1))
print("attention / FFN / mỗi LN =", parts, "| tổng khối =", n_params(blk))
assert parts == (80, 148, 8) and n_params(blk) == 244

# (b) Số tham số của MHA KHÔNG đổi theo số đầu
for h in (1, 8, 16):
    print(f"d = 512, h = {h:2d}, dk = {512 // h:3d}: {n_params(MHA(512, h)):,}".replace(",", " "))
    assert n_params(MHA(512, h)) == 4 * 512 ** 2 + 4 * 512 == 1050624

# (c) Khớp nn.MultiheadAttention khi chép trọng số (cùng bố cục in_proj = [Wq; Wk; Wv])
torch.manual_seed(0)
d, h, B, T = 32, 4, 3, 10
mine = MHA(d, h, causal=True)
ref = nn.MultiheadAttention(d, h, batch_first=True)
with torch.no_grad():
    ref.in_proj_weight.copy_(mine.qkv.weight); ref.in_proj_bias.copy_(mine.qkv.bias)
    ref.out_proj.weight.copy_(mine.o.weight); ref.out_proj.bias.copy_(mine.o.bias)
x = torch.randn(B, T, d)
cm = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
with torch.no_grad():
    y_ref = ref(x, x, x, attn_mask=cm, need_weights=False)[0]
    y_me = mine(x)
err = (y_ref - y_me).abs().max().item()
print("max |MHA của bạn − nn.MultiheadAttention| =", f"{err:.2e}")
assert err < 1e-5

# (d) Mô hình tí hon chạy được, shape đúng, loss ban đầu ≈ ln V
V = 50
lm = LM(V=V, T=32, d=64, h=4, L=3)
idx = torch.randint(0, V, (4, 32))
logits = lm(idx)
assert logits.shape == (4, 32, V)
l0 = F.cross_entropy(logits.reshape(-1, V), torch.randint(0, V, (4 * 32,))).item()
print(f"loss ban đầu {l0:.3f} so với ln 50 = {math.log(50):.3f}; tham số mô hình d = 64, L = 3: {n_params(lm):,}".replace(",", " "))
assert abs(l0 - math.log(V)) < 0.5

# (e) Pre-LN: dòng residual không được chuẩn hóa nên chuẩn của nó tăng theo độ sâu; post-LN bị LN kéo về mỗi khối
class PostBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        b = Block(d, h, causal=False)
        self.att, self.ffn, self.ln1, self.ln2 = b.att, b.ffn, b.ln1, b.ln2
    def forward(self, x):
        x = self.ln1(x + self.att(x))
        return self.ln2(x + self.ffn(x))

torch.manual_seed(0)
dd = 256
pre_stack = nn.ModuleList([Block(dd, 4, causal=False) for _ in range(24)])
post_stack = nn.ModuleList([PostBlock(dd, 4) for _ in range(24)])
x_pre = x_post = torch.randn(8, 32, dd)
s_pre, s_post = {}, {}
with torch.no_grad():
    for i in range(1, 25):
        x_pre, x_post = pre_stack[i - 1](x_pre), post_stack[i - 1](x_post)
        if i in (1, 6, 12, 24):
            s_pre[i], s_post[i] = x_pre.std().item(), x_post.std().item()
print("std dòng residual sau 1/6/12/24 khối (d = 256, đầu vào N(0, 1), khởi tạo mặc định PyTorch):")
print("  pre-LN :", [round(s_pre[i], 3) for i in (1, 6, 12, 24)])
print("  post-LN:", [round(s_post[i], 3) for i in (1, 6, 12, 24)])
assert s_pre[24] > s_pre[1] and all(abs(s_post[i] - 1) < 0.05 for i in s_post)
print("Lab 2 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 3
POS_CODE = ("""def sinusoidal(L, d):
    \"\"\"Bảng PE của Vaswani: PE[pos, 2i] = sin(pos / 10000^(2i/d)), PE[pos, 2i+1] = cos(·). Shape (L, d), d chẵn.\"\"\"
    # TODO
    raise NotImplementedError

def rope_thetas(d, base=10000.0):
    \"\"\"d/2 tần số θⱼ = base^(−2j/d), j = 0..d/2−1.\"\"\"
    # TODO
    raise NotImplementedError

def rope(x, pos, thetas):
    \"\"\"Xoay từng cặp (x[..., 2j], x[..., 2j+1]) một góc pos·θⱼ. x: (..., d). Trả về cùng shape.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def sinusoidal(L, d):
    pos = np.arange(L)[:, None]
    i = np.arange(0, d, 2)[None, :]
    ang = pos / 10000 ** (i / d)
    P = np.zeros((L, d))
    P[:, 0::2], P[:, 1::2] = np.sin(ang), np.cos(ang)
    return P

def rope_thetas(d, base=10000.0):
    return base ** (-2.0 * np.arange(d // 2) / d)

def rope(x, pos, thetas):
    x = np.asarray(x, dtype=float)
    c, s = np.cos(pos * thetas), np.sin(pos * thetas)
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2], out[..., 1::2] = a * c - b * s, a * s + b * c
    return out""")

ENC_CODE = ("""class Enc(nn.Module):
    \"\"\"Bộ mã hóa hai chiều (không causal), mean-pool, 2 lớp: dùng cho bài 'token đầu > token cuối'. (Cho sẵn.)\"\"\"
    def __init__(self, V=10, T=8, d=32, h=2, L=2, pos=True):
        super().__init__()
        self.pos = pos
        self.te, self.pe = nn.Embedding(V, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block(d, h, causal=False) for _ in range(L)])
        self.lnf, self.head = nn.LayerNorm(d), nn.Linear(d, 2)

    def forward(self, idx):
        x = self.te(idx) + (self.pe(torch.arange(idx.shape[1])) if self.pos else 0)
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x).mean(1))

def first_gt_last(B, gen):
    x = torch.randint(0, 10, (B, 8), generator=gen)
    return x, (x[:, 0] > x[:, -1]).long()

def train_enc(pos, steps=300, B=64, lr=3e-3, seed=0):
    \"\"\"Huấn luyện Enc bằng Adam; trả về accuracy trên 4000 chuỗi mới (seed khác).\"\"\"
    # TODO: vòng lặp chuẩn zero_grad → forward + cross_entropy → backward → step; nhớ model.eval() + no_grad khi đo
    raise NotImplementedError""",
"""class Enc(nn.Module):
    def __init__(self, V=10, T=8, d=32, h=2, L=2, pos=True):
        super().__init__()
        self.pos = pos
        self.te, self.pe = nn.Embedding(V, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block(d, h, causal=False) for _ in range(L)])
        self.lnf, self.head = nn.LayerNorm(d), nn.Linear(d, 2)

    def forward(self, idx):
        x = self.te(idx) + (self.pe(torch.arange(idx.shape[1])) if self.pos else 0)
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x).mean(1))

def first_gt_last(B, gen):
    x = torch.randint(0, 10, (B, 8), generator=gen)
    return x, (x[:, 0] > x[:, -1]).long()

def train_enc(pos, steps=300, B=64, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = Enc(pos=pos)
    opt = torch.optim.Adam(model.parameters(), lr)
    for _ in range(steps):
        x, y = first_gt_last(B, gen)
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        x, y = first_gt_last(4000, torch.Generator().manual_seed(7))
        return (model(x).argmax(1) == y).float().mean().item(), y.float().mean().item()""")

POS_CHECK = """# (a) Ví dụ giải mẫu 8.5: d = 4, vị trí 0..3
P = sinusoidal(4, 4)
print(P)
assert np.allclose(P[0], [0, 1, 0, 1]) and np.allclose(P[1], [0.8415, 0.5403, 0.0100, 1.0], atol=1e-4)
assert np.allclose(P[3], [0.1411, -0.9900, 0.0300, 0.9996], atol=1e-4)

# (b) PE(p)·PE(p + k) chỉ phụ thuộc k (d = 64); không đơn điệu theo k
P = sinusoidal(400, 64)
dots5 = [P[p] @ P[p + 5] for p in (0, 10, 100)]
print("k = 5, p = 0/10/100:", np.round(dots5, 4), "| |PE|² =", round(P[7] @ P[7], 4))
assert np.allclose(dots5, dots5[0]) and abs(dots5[0] - 23.504) < 1e-3
ks = (0, 1, 2, 5, 10, 20, 50, 100)
dk_tab = [P[0] @ P[k] for k in ks]
print("k:", ks, "\\nPE(0)·PE(k):", np.round(dk_tab, 3))
assert dk_tab[-1] > dk_tab[-2]                       # k = 100 lớn hơn k = 50

# (c) Dịch vị trí = phép quay từng cặp: với thứ tự (sin, cos), PE(p + k) = R(−ωk)·PE(p), R(φ) là quay ngược chiều kim đồng hồ
d, k, p = 8, 7, 13
P8 = sinusoidal(40, d); w = 1 / 10000 ** (np.arange(0, d, 2) / d)
rot = np.empty(d)
for j in range(d // 2):
    c, s = np.cos(w[j] * k), np.sin(w[j] * k)
    a, b = P8[p, 2 * j], P8[p, 2 * j + 1]            # (sin, cos) tại p
    rot[2 * j], rot[2 * j + 1] = a * c + b * s, b * c - a * s
assert np.allclose(rot, P8[p + k])
print("bước sóng cặp đầu 2π =", round(2 * np.pi, 2), "| cặp cuối (d = 512):", round(2 * np.pi * 10000 ** (510 / 512)))

# (d) RoPE: ví dụ 8.6 và tính bất biến tương đối
q, kk = np.array([1., 2]), np.array([3., 1])
th = np.array([np.pi / 6])
sc = {mn: rope(q, mn[0], th) @ rope(kk, mn[1], th) for mn in [(2, 0), (5, 3), (12, 10)]}
print("điểm RoPE (m, n):", {k_: round(v, 4) for k_, v in sc.items()}, "| không quay:", q @ kk)
assert all(abs(v + 1.8301) < 1e-4 for v in sc.values())
g = np.random.default_rng(0); q8, k8 = g.standard_normal(8), g.standard_normal(8); t8 = rope_thetas(8)
sc8 = [rope(q8, m, t8) @ rope(k8, n, t8) for m, n in [(5, 2), (15, 12), (105, 102), (1005, 1002)]]
print("d = 8, n − m = −3:", np.round(sc8, 6), "| không quay:", round(q8 @ k8, 6),
      "| ‖q‖ trước/sau:", round(np.linalg.norm(q8), 4), round(np.linalg.norm(rope(q8, 1005, t8)), 4))
assert np.allclose(sc8, sc8[0], atol=1e-9) and abs(np.linalg.norm(q8) - np.linalg.norm(rope(q8, 77, t8))) < 1e-12
print("bước sóng chậm nhất, d = 128: base 1e4 →", round(2 * np.pi * 10000 ** (126 / 128)),
      "| base 5e5 →", round(2 * np.pi * 500000 ** (126 / 128)))

# (e) Không có vị trí: bộ mã hóa hai chiều đẳng biến theo hoán vị, mô hình causal thì không
torch.manual_seed(0)
idx = torch.randint(0, 50, (1, 16)); pm = torch.randperm(16)
with torch.no_grad():
    nc = LM(V=50, T=16, causal=False, pos=False).eval()
    ca = LM(V=50, T=16, causal=True, pos=False).eval()
    d_nc = (nc(idx)[:, pm] - nc(idx[:, pm])).abs().max().item()
    d_ca = (ca(idx)[:, pm] - ca(idx[:, pm])).abs().max().item()
print(f"không PE, không causal: sai khác sau hoán vị = {d_nc:.1e} | không PE, causal: {d_ca:.3f}")
assert d_nc < 1e-5 and d_ca > 0.05

# (f) Bài 'token đầu > token cuối' (chuỗi 8 token, từ vựng 10): có và không có mã hóa vị trí
acc_pos, base = train_enc(True)
acc_nopos, _ = train_enc(False)
print(f"acc có PE = {acc_pos:.3f} | không PE = {acc_nopos:.3f} | tỉ lệ lớp 1 = {base:.3f} (đoán lớp đa số được {1 - base:.3f})")
assert acc_pos >= 0.95 and acc_nopos <= 0.65
print("Lab 3 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 4
CACHE_CODE = ("""def decode_step(lm, tok, pos, cache):
    \"\"\"Một bước giải mã với KV cache. tok: (B,) token mới; pos: int, vị trí THẬT của nó; cache: list độ dài L,
    phần tử i là None (chưa có) hoặc cặp (K, V) shape (B, h, t, dk) của khối i. Cập nhật cache tại chỗ.
    Trả về logit (B, V) của vị trí pos. Gợi ý: không cần mặt nạ — cache chỉ chứa quá khứ và hiện tại.\"\"\"
    # TODO
    raise NotImplementedError

def cached_logits(lm, idx):
    \"\"\"Chạy decode_step lần lượt cho t = 0..T−1 và xếp logit thành (B, T, V). Trả về (logits, cache).\"\"\"
    # TODO
    raise NotImplementedError

def leak_test(lm, idx, t0, seed=5):
    \"\"\"Thay mọi token ở vị trí ≥ t0 bằng token ngẫu nhiên khác; trả về thay đổi lớn nhất của logit ở các vị trí < t0.
    Mô hình không nhìn thấy tương lai thì phải trả về đúng 0.0. (Nhớ lm.eval() để tắt dropout.)\"\"\"
    # TODO
    raise NotImplementedError""",
"""def decode_step(lm, tok, pos, cache):
    x = (lm.te(tok) + lm.pe(torch.tensor(pos)))[:, None, :]                  # (B, 1, d)
    for i, blk in enumerate(lm.blocks):
        a = blk.att
        B, _, d = x.shape
        h, dk = a.h, d // a.h
        q, k, v = a.qkv(blk.ln1(x)).split(d, dim=2)
        q, k, v = [t.view(B, 1, h, dk).transpose(1, 2) for t in (q, k, v)]   # (B, h, 1, dk)
        cache[i] = (k, v) if cache[i] is None else (torch.cat([cache[i][0], k], 2), torch.cat([cache[i][1], v], 2))
        K, V = cache[i]                                                       # (B, h, t+1, dk)
        y = ((q @ K.transpose(-2, -1)) / math.sqrt(dk)).softmax(-1) @ V      # không cần mask
        x = x + a.o(y.transpose(1, 2).reshape(B, 1, d))
        x = x + blk.ffn(blk.ln2(x))
    return lm.head(lm.lnf(x))[:, 0]

def cached_logits(lm, idx):
    cache = [None] * len(lm.blocks)
    out = [decode_step(lm, idx[:, t], t, cache) for t in range(idx.shape[1])]
    return torch.stack(out, 1), cache

def leak_test(lm, idx, t0, seed=5):
    lm.eval()
    g = torch.Generator().manual_seed(seed)
    idx2 = idx.clone()
    idx2[:, t0:] = torch.randint(0, lm.te.num_embeddings, idx[:, t0:].shape, generator=g)
    with torch.no_grad():
        return (lm(idx)[:, :t0] - lm(idx2)[:, :t0]).abs().max().item()""")

CACHE_CHECK = """torch.manual_seed(0)
V, T = 50, 32
lm = LM(V=V, T=T, d=64, h=4, L=3, causal=True).eval()
idx = torch.randint(0, V, (1, T))
with torch.no_grad():
    full = lm(idx)
    cl, cache = cached_logits(lm, idx)
err = (full - cl).abs().max().item()
n_cache = sum(k.numel() + v.numel() for k, v in cache)
print(f"cache so với forward đầy đủ: sai lệch lớn nhất {err:.1e} | số phần tử cache {n_cache} = 2·L·h·T·dk = {2 * 3 * 4 * T * 16}")
assert err < 1e-5 and n_cache == 12288

# Thử rò rỉ tương lai: mô hình đúng cho 0.0, mô hình hai chiều (quên mặt nạ) thì khác 0
leak_ok = leak_test(lm, idx, 20)
leak_bad = leak_test(LM(V=V, T=T, d=64, h=4, L=3, causal=False), idx, 20)
print(f"rò rỉ: causal {leak_ok} | quên mặt nạ {leak_bad:.4f}")
assert leak_ok == 0.0 and leak_bad > 1e-3

# Đo thời gian sinh 31 token: tính lại toàn bộ so với cache. Lặp 7 lần, báo khoảng giá trị (dao động theo máy và lần chạy).
def gen_recompute():
    seq = idx[:, :1]
    for t in range(1, T):
        seq = torch.cat([seq, lm(seq)[:, -1].argmax(-1, keepdim=True)], 1)

def gen_cached():
    c, tok = [None] * 3, idx[:, 0]
    for t in range(1, T):
        tok = decode_step(lm, tok, t - 1, c).argmax(-1)

def timeit(f, reps=7):
    out = []
    for _ in range(reps):
        t1 = time.perf_counter(); f(); out.append((time.perf_counter() - t1) * 1000)
    return out

with torch.no_grad():
    gen_recompute(); gen_cached()                    # khởi động
    t_re, t_ca = timeit(gen_recompute), timeit(gen_cached)
ratio = [a / b for a, b in zip(t_re, t_ca)]
print(f"sinh {T - 1} token, 7 lần: tính lại {min(t_re):.0f}–{max(t_re):.0f} ms, cache {min(t_ca):.0f}–{max(t_ca):.0f} ms, "
      f"tỉ lệ {min(ratio):.2f}–{max(ratio):.2f}× (mô hình bé, overhead Python lấn át; đừng suy ra tỉ lệ lý thuyết)")
print("số lượt xử lý một token qua cả mạng: tính lại n(n+1)/2 =", T * (T - 1) // 2, "so với cache n =", T - 1)
print("Lab 4 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 5
COST_CODE = ("""GPT2_SMALL = dict(V=50257, T=1024, d=768, L=12, h=12, n_kv=12, d_ff=3072, ffn='gelu', bias=True, norm='ln', pos='learned', tie=True)
LLAMA2_7B = dict(V=32000, T=4096, d=4096, L=32, h=32, n_kv=32, d_ff=11008, ffn='swiglu', bias=False, norm='rms', pos='rope', tie=False)

def params(c):
    \"\"\"Tổng số tham số theo cấu hình c. ffn='gelu' có 2 ma trận (có bias nếu c['bias']), 'swiglu' có 3 ma trận; norm 'ln' có 2d tham số,
    'rms' có d; pos='learned' thêm T·d; tie=False thì lm_head thêm V·d. Số đầu K/V là n_kv (GQA); độ rộng một đầu dh = d/h.\"\"\"
    # TODO
    raise NotImplementedError

def flops_per_token(c, T):
    \"\"\"FLOPs huấn luyện mỗi token theo nanoGPT: 6N + 12·L·d·T. N = số tham số THAM GIA NHÂN MA TRẬN = params trừ các bảng chỉ để tra
    (bảng vị trí học được T·d; bảng embedding đầu vào V·d nếu không buộc trọng số với lm_head).\"\"\"
    # TODO
    raise NotImplementedError

def kv_bytes(c, T, B=1, nbytes=2):
    \"\"\"Số byte KV cache: 2 (K và V) · L · n_kv · dh · nbytes · T · B.\"\"\"
    # TODO
    raise NotImplementedError

def params_vaswani(d, d_ff, n_layers, V):
    \"\"\"Số tham số Transformer gốc: n_layers lớp encoder + n_layers lớp decoder (post-LN, có bias), embedding dùng chung V·d,
    không có LN cuối, mã hóa vị trí sinusoidal không có tham số. Lớp decoder có 2 khối attention và 3 LN.\"\"\"
    # TODO
    raise NotImplementedError""",
"""GPT2_SMALL = dict(V=50257, T=1024, d=768, L=12, h=12, n_kv=12, d_ff=3072, ffn='gelu', bias=True, norm='ln', pos='learned', tie=True)
LLAMA2_7B = dict(V=32000, T=4096, d=4096, L=32, h=32, n_kv=32, d_ff=11008, ffn='swiglu', bias=False, norm='rms', pos='rope', tie=False)

def params(c):
    d, L, b = c['d'], c['L'], int(c['bias'])
    kv = c['n_kv'] * (d // c['h'])
    attn = d * d + 2 * d * kv + d * d + b * (d + 2 * kv + d)
    if c['ffn'] == 'swiglu':
        ffn = 3 * d * c['d_ff']
    else:
        ffn = 2 * d * c['d_ff'] + b * (c['d_ff'] + d)
    norm = d * (2 if c['norm'] == 'ln' else 1)
    total = c['V'] * d + L * (attn + ffn + 2 * norm) + norm
    if c['pos'] == 'learned':
        total += c['T'] * d
    if not c['tie']:
        total += c['V'] * d
    return total

def flops_per_token(c, T):
    N = params(c) - (c['T'] * c['d'] if c['pos'] == 'learned' else 0) - (0 if c['tie'] else c['V'] * c['d'])
    return 6 * N + 12 * c['L'] * c['d'] * T

def kv_bytes(c, T, B=1, nbytes=2):
    return 2 * c['L'] * c['n_kv'] * (c['d'] // c['h']) * nbytes * T * B

def params_vaswani(d, d_ff, n_layers, V):
    mha = 4 * d * d + 4 * d
    ffn = 2 * d * d_ff + d_ff + d
    enc = mha + ffn + 2 * (2 * d)
    dec = 2 * mha + ffn + 3 * (2 * d)
    return n_layers * (enc + dec) + V * d""")

COST_CHECK = """g = params(GPT2_SMALL)
print(f"GPT-2 small: {g:,} tham số; trừ wpe: {g - 1024 * 768:,}".replace(",", " "))
assert g == 124_439_808 and g - 1024 * 768 == 123_653_376
fl = flops_per_token(GPT2_SMALL, 1024)
att_part = 12 * 12 * 768 * 1024
print(f"FLOPs huấn luyện mỗi token (T = 1024): {fl:,}; phần attention {att_part / fl:.1%}; 6ND với D = 300e9: {6 * 123_653_376 * 300e9:.3e}".replace(",", " "))
assert fl == 855_166_464
fl7 = flops_per_token(LLAMA2_7B, 4096)       # không buộc trọng số: bảng embedding đầu vào chỉ là phép tra, không tính FLOPs
print(f"Llama-2-7B: N tham gia nhân ma trận = {params(LLAMA2_7B) - 32000 * 4096:,} (đã trừ bảng embedding đầu vào)".replace(",", " "))
assert fl7 == 6 * (6_738_415_616 - 32000 * 4096) + 12 * 32 * 4096 * 4096
kv_small = kv_bytes(GPT2_SMALL, 1)
print(f"KV cache GPT-2 small fp16: {kv_small:,} byte mỗi token = {kv_small * 1024 / 2**20:.0f} MiB ở T = 1024".replace(",", " "))
assert kv_small == 36_864
l7 = params(LLAMA2_7B)
print(f"Llama-2-7B theo công thức: {l7:,}".replace(",", " "))
assert l7 == 6_738_415_616
GiB = 2 ** 30
mha = kv_bytes(LLAMA2_7B, 4096) / GiB
gqa = kv_bytes({**LLAMA2_7B, 'n_kv': 8}, 4096) / GiB
mqa = kv_bytes({**LLAMA2_7B, 'n_kv': 1}, 4096) / GiB
print(f"Llama-2-7B, T = 4096, batch 1, fp16: MHA {mha:.4f} GiB | GQA 8 đầu K/V {gqa:.4f} | MQA {mqa:.4f} | MHA batch 8: {8 * mha:.1f} GiB")
assert (mha, gqa, mqa) == (2.0, 0.5, 0.0625)
print(f"Trọng số Llama-2-7B fp16: {2 * l7 / GiB:.2f} GiB | GPT-2 small huấn luyện Adam fp32 (trọng số + gradient + m + v): {4 * 4 * g / GiB:.2f} GiB")
# các cỡ GPT-2 khác theo công thức
for name, (d, L, h) in {"medium": (1024, 24, 16), "large": (1280, 36, 20), "xl": (1600, 48, 25)}.items():
    print(f"  GPT-2 {name}: {params(dict(GPT2_SMALL, d=d, L=L, h=h, n_kv=h, d_ff=4 * d)):,}".replace(",", " "))
# tỉ số chi phí attention / (phép chiếu + FFN) theo n, d = 768 (mỗi lớp, mỗi token, forward): 4nd / 24d² = n/(6d)
print("n:", [512, 1024, 4096, 8192], "tỉ số:", [round(n / (6 * 768), 3) for n in (512, 1024, 4096, 8192)], "| điểm giao n = 6d =", 6 * 768)
# Vaswani
vb, vg = params_vaswani(512, 2048, 6, 37000), params_vaswani(1024, 4096, 6, 37000)
print(f"Transformer base theo cách đếm này: {vb:,} | big: {vg:,}".replace(",", " "))
assert vb == 63_082_496 and vg == 214_245_376
print("Lab 5 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 6
S2S_CODE = ("""def jac_norm(rho, T, n=64, seed=0):
    \"\"\"RNN tanh: h ← tanh(W h + nhiễu 0.3·N(0,1)), W ngẫu nhiên rescale về bán kính phổ rho. Trả về ‖∂h_T/∂h₀‖₂ (chuẩn phổ)
    bằng cách nhân các Jacobian diag(1 − h²)·W.\"\"\"
    # TODO
    raise NotImplementedError

class S2S(nn.Module):
    \"\"\"Seq2seq GRU cho bài chép chuỗi. forward(src, tin): mã hóa src, giải mã với teacher forcing tin.
    att=True: tại mỗi bước giải mã tính điểm = dec_out @ enc_outᵀ, softmax theo nguồn, ngữ cảnh = A @ enc_out, nối vào dec_out trước lớp ra.\"\"\"
    def __init__(self, V=12, h=32, att=False):
        super().__init__()
        self.emb = nn.Embedding(V, 16)
        self.enc, self.dec = nn.GRU(16, h, batch_first=True), nn.GRU(16, h, batch_first=True)
        self.att = att
        self.out = nn.Linear(2 * h if att else h, V)
    def forward(self, src, tin):
        # TODO
        raise NotImplementedError""",
"""def jac_norm(rho, T, n=64, seed=0):
    r = np.random.default_rng(seed)
    W = r.standard_normal((n, n)) / math.sqrt(n)
    W *= rho / np.max(np.abs(np.linalg.eigvals(W)))
    h = r.standard_normal(n) * 0.5
    J = np.eye(n)
    for _ in range(T):
        h = np.tanh(W @ h + 0.3 * r.standard_normal(n))
        J = np.diag(1 - h ** 2) @ W @ J
    return np.linalg.norm(J, 2)

class S2S(nn.Module):
    def __init__(self, V=12, h=32, att=False):
        super().__init__()
        self.emb = nn.Embedding(V, 16)
        self.enc, self.dec = nn.GRU(16, h, batch_first=True), nn.GRU(16, h, batch_first=True)
        self.att = att
        self.out = nn.Linear(2 * h if att else h, V)
    def forward(self, src, tin):
        enc_out, hn = self.enc(self.emb(src))            # hn: nút thắt, một vector cho cả câu
        dec_out, _ = self.dec(self.emb(tin), hn)
        if self.att:
            A = (dec_out @ enc_out.transpose(1, 2)).softmax(-1)      # (B, T_dec, T_src)
            dec_out = torch.cat([dec_out, A @ enc_out], -1)
        return self.out(dec_out)""")

S2S_CHECK = """# (a) Gradient qua thời gian trong RNN tanh 64 chiều (một seed, một lần chạy)
rows = {}
for rho in (0.9, 1.0):
    rows[rho] = [jac_norm(rho, T) for T in (5, 20, 50, 100)]
    print(f"bán kính phổ {rho}: T = 5/20/50/100 → " + " / ".join(f"{v:.2e}" for v in rows[rho]))
assert rows[0.9][3] < 1e-6 and rows[0.9][0] > rows[0.9][1] > rows[0.9][2] > rows[0.9][3] and rows[1.0][3] < 1e-3

# (a2) Vì sao ρ = 1 vẫn biến mất? Tách từng ứng viên (T = 100, cùng cách đo như jac_norm).
def jac_parts(rho, T, n=64, seed=0, orth=False, tanh=True):
    r = np.random.default_rng(seed)
    if orth:
        W = np.linalg.qr(r.standard_normal((n, n)))[0]                 # trực giao: hoàn toàn chuẩn tắc, ρ = 1
    else:
        W = r.standard_normal((n, n)) / math.sqrt(n); W *= rho / np.max(np.abs(np.linalg.eigvals(W)))
    h, J, gain = r.standard_normal(n) * 0.5, np.eye(n), []
    for _ in range(T):
        h = np.tanh(W @ h + 0.3 * r.standard_normal(n))
        D = 1 - h ** 2 if tanh else np.ones(n)                          # tanh=False: bỏ đạo hàm của tanh
        gain.append(D.mean()); J = np.diag(D) @ W @ J
    return np.linalg.norm(J, 2), np.mean(gain), np.linalg.norm(W, 2)

n_full, g_full, w2 = jac_parts(1.0, 100)
n_notanh = jac_parts(1.0, 100, tanh=False)[0]
n_orth, g_orth, _ = jac_parts(1.0, 100, orth=True)
Wt = np.random.default_rng(0).standard_normal((64, 64)) / 8; Wt *= 1.0 / np.max(np.abs(np.linalg.eigvals(Wt)))
print(f"ρ = 1, T = 100: ‖J‖ đủ = {n_full:.2e} | bỏ đạo hàm tanh (chỉ còn ‖W¹⁰⁰‖₂) = {n_notanh:.2f} | W trực giao = {n_orth:.2e}")
print(f"  ‖W‖₂ = {w2:.2f} (> 1: chuẩn từng bước lớn hơn ρ), trung bình 1 − h² = {g_full:.2f}, 0.82¹⁰⁰ = {g_full ** 100:.1e} (ước lượng thô, chỉ để so bậc độ lớn)")
assert n_notanh > 1 and n_full < 1e-6 and n_orth < 1e-3        # không có tanh thì không co; có tanh thì co, kể cả khi W trực giao

# (b) Bài chép chuỗi: từ vựng 10 ký tự (id 2..11), teacher forcing, Adam 3e-3, 300 bước
VOC = 12
def copy_batch(B, L, gen):
    x = torch.randint(2, VOC, (B, L), generator=gen)
    return x, torch.cat([torch.zeros(B, 1, dtype=torch.long), x[:, :-1]], 1)

def run_copy(L, att, steps=300, seed=0, h=32):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    m = S2S(VOC, h, att); opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(steps):
        x, tin = copy_batch(64, L, gen)
        loss = F.cross_entropy(m(x, tin).reshape(-1, VOC), x.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    with torch.no_grad():
        x, tin = copy_batch(1000, L, torch.Generator().manual_seed(123))
        return (m(x, tin).argmax(-1) == x).float().mean().item()

copy_res = {}
for L in (8, 16):
    t1 = time.time()
    copy_res[L] = (run_copy(L, False), run_copy(L, True))
    print(f"độ dài {L:2d}: acc theo token không attention {copy_res[L][0]:.3f} | có attention {copy_res[L][1]:.3f}   ({time.time() - t1:.0f} s)")
assert copy_res[16][1] >= 0.95 and copy_res[16][0] <= 0.65
# Nút thắt dung lượng hay chưa hội tụ? Cho mô hình không attention rộng hơn (h = 64) và huấn luyện lâu hơn (600 bước), độ dài 16
big = run_copy(16, False, steps=600, h=64)
print(f"độ dài 16, không attention, h = 64, 600 bước: {big:.3f} (so với {copy_res[16][0]:.3f} ở h = 32, 300 bước)")
print("Lab 6 ĐẠT ✔")"""

# ---------------------------------------------------------------- Săn bug
BUG_DATA = """V, T = 50, 16
LAG_NOTE = "lag1: mục tiêu tại vị trí t là token NGAY TRƯỚC đó x[t−1] (cần chú ý đúng vị trí tương đối)"

def task_rand(B, gen):
    \"\"\"Dự đoán token kế tiếp trên chuỗi NGẪU NHIÊN: không học được gì thật; sàn loss = ln 50 = 3.912.\"\"\"
    x = torch.randint(0, V, (B, T + 1), generator=gen)
    return x[:, :-1], x[:, 1:]

def task_lag1(B, gen):
    \"\"\"Mục tiêu tại vị trí t là x[t−1]. Học được nhanh nếu có thông tin vị trí.\"\"\"
    x = torch.randint(0, V, (B, T), generator=gen)
    return x, torch.cat([x[:, :1], x[:, :-1]], 1)

class BuggyAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x):
        B, T, d = x.shape
        dk = d // self.h
        q, k, v = self.qkv(x).split(d, dim=2)
        q = q.reshape(B, self.h, T, dk)
        k = k.reshape(B, self.h, T, dk)
        v = v.reshape(B, self.h, T, dk)
        A = (q @ k.transpose(-2, -1) / math.sqrt(dk)).softmax(-1)
        A = A * torch.tril(torch.ones(T, T))
        return self.o((A @ v).reshape(B, T, d))

class BuggyBlock(nn.Module):
    def __init__(self, d, h, p=0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = BuggyAttention(d, h)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.dr = nn.Dropout(p)

    def forward(self, x):
        x = x + self.dr(self.att(self.ln1(x)))
        return x + self.dr(self.ffn(self.ln2(x)))

class BuggyLM(nn.Module):
    def __init__(self, V=50, T=16, d=64, h=4, L=2):
        super().__init__()
        self.te, self.pe = nn.Embedding(V, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList([BuggyBlock(d, h) for _ in range(L)])
        self.lnf, self.head = nn.LayerNorm(d), nn.Linear(d, V)

    def forward(self, idx):
        x = self.te(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))

def train_buggy(task, steps=300):
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    model = BuggyLM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = []
    for _ in range(steps):
        x, y = task(32, gen)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(loss.item())
    xe, ye = task(512, torch.Generator().manual_seed(9))
    with torch.no_grad():
        o1, o2 = model(xe), model(xe)
    return dict(model=model, train=sum(hist[-20:]) / 20, fresh=F.cross_entropy(o1.reshape(-1, V), ye.reshape(-1)).item(),
                repeat_diff=(o1 - o2).abs().max().item())

def show(tag, r, model_leak):
    print(f"{tag:34s} loss train {r['train']:.3f} | loss batch mới {r['fresh']:.3f} | hai lần gọi chênh {r['repeat_diff']:.3f} | rò rỉ {model_leak:.4f}")"""

BUG_RUN = """buggy = {name: train_buggy(task) for name, task in (("rand", task_rand), ("lag1", task_lag1))}
idx_leak = task_rand(8, torch.Generator().manual_seed(3))[0]
bl = {k: leak_test(r["model"], idx_leak, 10) for k, r in buggy.items()}
print(f"sàn của nhãn ngẫu nhiên: ln 50 = {math.log(50):.3f}")
show("BuggyLM, nhãn ngẫu nhiên", buggy["rand"], bl["rand"])
show("BuggyLM, bài lag1", buggy["lag1"], bl["lag1"])"""

BUGFIX = ("""def train_fixed(task, steps=300):
    \"\"\"Sửa MỌI lỗi của BuggyAttention / BuggyLM / train_buggy rồi huấn luyện. Dùng lại MHA, Block, LM của Lab 2 nếu muốn
    (nhớ truyền p=0.1 cho Dropout nếu bạn muốn giữ dropout). Trả về dict cùng khóa với train_buggy: model, train, fresh, repeat_diff.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def train_fixed(task, steps=300):
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    model = LM(V=V, T=T, d=64, h=4, L=2, causal=True, pos=True, p=0.1)     # MHA đúng (view + transpose, mask TRƯỚC softmax), có PE
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = []
    model.train()
    for _ in range(steps):
        x, y = task(32, gen)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(loss.item())
    model.eval()                                                            # tắt dropout trước khi đo
    xe, ye = task(512, torch.Generator().manual_seed(9))
    with torch.no_grad():
        o1, o2 = model(xe), model(xe)
    return dict(model=model, train=sum(hist[-20:]) / 20, fresh=F.cross_entropy(o1.reshape(-1, V), ye.reshape(-1)).item(),
                repeat_diff=(o1 - o2).abs().max().item())""")

BUG_CHECK = """fixed = {name: train_fixed(task) for name, task in (("rand", task_rand), ("lag1", task_lag1))}
fl = {k: leak_test(r["model"], idx_leak, 10) for k, r in fixed.items()}
show("bản sửa, nhãn ngẫu nhiên", fixed["rand"], fl["rand"])
show("bản sửa, bài lag1", fixed["lag1"], fl["lag1"])
assert fl["rand"] == 0.0 and fl["lag1"] == 0.0, "mô hình vẫn nhìn thấy tương lai"
assert fixed["rand"]["fresh"] >= 3.85, "nhãn ngẫu nhiên mà loss xuống dưới ln 50 thì còn rò rỉ"
assert fixed["lag1"]["fresh"] < 0.1, "bài lag1 phải giải được khi có vị trí"
assert fixed["rand"]["repeat_diff"] == 0.0 and fixed["lag1"]["repeat_diff"] == 0.0, "quên model.eval()"
print("Săn bug ĐẠT ✔")"""

ABLATION = """# Tách từng lỗi: bật riêng mỗi lỗi trên bản đúng (cùng seed, cùng dữ liệu).
class FAtt(nn.Module):
    def __init__(self, d, h, mask_after=False, bad_split=False):
        super().__init__()
        self.h, self.ma, self.bs = h, mask_after, bad_split
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
    def forward(self, x):
        B, T, d = x.shape; h, dk = self.h, d // self.h
        q, k, v = self.qkv(x).split(d, dim=2)
        q, k, v = [t.reshape(B, h, T, dk) if self.bs else t.view(B, T, h, dk).transpose(1, 2) for t in (q, k, v)]
        S = q @ k.transpose(-2, -1) / math.sqrt(dk)
        m = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        A = S.softmax(-1).masked_fill(m, 0.) if self.ma else S.masked_fill(m, float('-inf')).softmax(-1)
        y = A @ v
        return self.o(y.reshape(B, T, d) if self.bs else y.transpose(1, 2).reshape(B, T, d))

def ablate(task, mask_after=False, bad_split=False, pos=True, steps=300):
    torch.manual_seed(0); gen = torch.Generator().manual_seed(0)
    lm = LM(V=V, T=T, d=64, h=4, L=2)
    for b in lm.blocks:
        b.att = FAtt(64, 4, mask_after, bad_split)
    lm.pos = pos
    opt = torch.optim.Adam(lm.parameters(), 1e-3)
    for _ in range(steps):
        x, y = task(32, gen)
        loss = F.cross_entropy(lm(x).reshape(-1, V), y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    lm.eval(); xe, ye = task(512, torch.Generator().manual_seed(9))
    with torch.no_grad():
        fresh = F.cross_entropy(lm(xe).reshape(-1, V), ye.reshape(-1)).item()
    return fresh, leak_test(lm, idx_leak, 10)

print(f"{'lỗi bật riêng':38s} | {'loss batch mới':>14s} | rò rỉ")
# Mọi hàng cùng điều kiện: dropout 0 (LM mặc định p = 0), cùng seed, cùng dữ liệu; chỉ khác lỗi được bật.
res = {}
for name, task, kw in [("không lỗi, nhãn ngẫu nhiên", task_rand, dict()),
                       ("không lỗi, bài lag1", task_lag1, dict()),
                       ("mask sau softmax, nhãn ngẫu nhiên", task_rand, dict(mask_after=True)),
                       ("tách đầu sai, nhãn ngẫu nhiên", task_rand, dict(bad_split=True)),
                       ("không có PE, bài lag1", task_lag1, dict(pos=False)),
                       ("tách đầu sai, bài lag1", task_lag1, dict(bad_split=True))]:
    res[name] = ablate(task, **kw)
for name in res:
    print(f"{name:38s} | {res[name][0]:14.3f} | {res[name][1]:.4f}")
# Lỗi tách đầu im lặng hơn cả: không rò rỉ, loss nhãn ngẫu nhiên bình thường. Chỉ so với bản chuẩn mới thấy.
torch.manual_seed(1)
good, bad = FAtt(64, 4), FAtt(64, 4, bad_split=True)
bad.load_state_dict(good.state_dict())
xs = torch.randn(2, 16, 64)
dsplit = (good(xs) - bad(xs)).abs().max().item()
print(f"cùng trọng số, đầu ra tách đầu sai so với đúng: chênh lệch lớn nhất {dsplit:.3f}")
assert res["mask sau softmax, nhãn ngẫu nhiên"][1] > 0 and res["tách đầu sai, nhãn ngẫu nhiên"][1] == 0.0 and dsplit > 0.05
assert res["không có PE, bài lag1"][0] > 5 * res["không lỗi, bài lag1"][0]
print("Phân tích lỗi ĐẠT ✔")"""


def build(solution: bool):
    pick = (lambda pair: pair[1]) if solution else (lambda pair: pair[0])
    same = lambda s: s
    nb = nbf.v4.new_notebook()
    head = "Lời giải" if solution else "Bài lab"
    nb.cells = [
        md(f"# Chương 8 — {head}: Attention và Transformer\n\n"
           + ("Bản lời giải đã chạy kiểm (NumPy + PyTorch CPU, khoảng 1.5 đến 3 phút). Chỉ mở sau khi đã tự làm `lab.ipynb`."
              if solution else
              "Viết code vào các ô có `# TODO`, rồi chạy ô kiểm tra ngay bên dưới. "
              "Ô kiểm tra in **ĐẠT ✔** là xong. Bí quá 20 phút mới mở `loigiai.ipynb`. "
              "Chỉ cần PyTorch bản CPU; toàn notebook (bản lời giải) chạy khoảng 1.5 đến 3 phút. Làm theo thứ tự: lab sau dùng code của lab trước.")),
        code(SETUP),
        md("## Lab 1 — Attention từ đầu bằng NumPy\n"
           "Viết `softmax` ổn định và `attention(Q, K, V, causal)`. Đối chiếu ví dụ giải mẫu 8.3, so với `F.scaled_dot_product_attention`, "
           "rồi tự đo ba điều: hoán vị-đẳng biến, std(q·k) = √dₖ, và hai bẫy của mặt nạ."),
        code(pick(ATT_NP)), code(ATT_CHECK),
        md("## Lab 2 — Multi-head attention và khối Transformer pre-LN\n"
           "Viết `MHA` và `Block`. Mô hình `LM` cho sẵn (dùng code của bạn). Đếm tham số bằng tay rồi đối chiếu, "
           "và so với `nn.MultiheadAttention` bằng cách chép trọng số."),
        code(pick(MHA_CODE)), code(LM_CODE), code(MHA_CHECK),
        md("## Lab 3 — Mã hóa vị trí: sinusoidal, RoPE, và vì sao thiếu nó thì mù thứ tự\n"
           "Viết bảng sinusoidal và RoPE, kiểm hai tính chất (chỉ phụ thuộc khoảng cách). Rồi đo: bộ mã hóa không có vị trí không giải được bài "
           "'token đầu > token cuối'."),
        code(pick(POS_CODE)), code(pick(ENC_CODE)), code(POS_CHECK),
        md("## Lab 4 — KV cache và phép thử rò rỉ tương lai\n"
           "Viết một bước giải mã có cache, kiểm nó khớp forward đầy đủ tới sai số máy, đếm kích thước cache, và viết phép thử "
           "'đổi tương lai, quá khứ không được đổi'."),
        code(pick(CACHE_CODE)), code(CACHE_CHECK),
        md("## Lab 5 — Máy tính chi phí: tham số, FLOPs, KV cache\n"
           "Viết ba hàm đếm cho một cấu hình bất kỳ, đối chiếu GPT-2 small và Llama-2-7B, rồi tự đếm Transformer gốc của Vaswani."),
        code(pick(COST_CODE)), code(COST_CHECK),
        md("## Săn bug\n`BuggyLM` dưới đây chạy không báo lỗi. Hai bài kiểm: (1) dự đoán token kế tiếp trên chuỗi NGẪU NHIÊN (không học được, sàn loss "
           "ln 50 = 3.912); (2) bài `lag1` (mục tiêu là token ngay trước, học được nếu có vị trí). Chạy ô đầu để xem triệu chứng, "
           "tìm cả **bốn** lỗi rồi viết `train_fixed`."),
        code(BUG_DATA), code(BUG_RUN), code(pick(BUGFIX)), code(BUG_CHECK),
    ]
    if solution:       # bảng tách lỗi chứa đáp án của bài săn bug nên chỉ nằm trong lời giải
        nb.cells += [md("## Phân tích từng lỗi\nBật riêng từng lỗi trên bản đúng (cùng điều kiện) để thấy lỗi nào gây triệu chứng nào."),
                     code(ABLATION)]
    nb.cells += [   # Lab 6 tùy chọn nằm cuối để người bỏ qua nó vẫn chạy được phần trên bằng Run All
        md("## Lab 6 (tùy chọn) — Vì sao attention ra đời: RNN so với attention\n"
           "Đo gradient qua thời gian của RNN, rồi so seq2seq GRU có và không có attention trên bài chép chuỗi. "
           "Đây là bài đồ chơi, một seed: chỉ cho thấy nút thắt, không chứng minh RNN 'không nhớ được dài'."),
        code(pick(S2S_CODE)), code(S2S_CHECK),
        code('print(f"Tổng thời gian notebook: {time.time() - T0:.1f} s")') if solution else md("Hết. Mở `loigiai.ipynb` để đối chiếu."),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    name = "loigiai.ipynb" if solution else "lab.ipynb"
    nbf.write(nb, OUT / name)
    return OUT / name


if __name__ == "__main__":
    print(build(False), build(True))
