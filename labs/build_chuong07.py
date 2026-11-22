"""Sinh hai notebook cho lab chương 7: lab.ipynb (có TODO) và loigiai.ipynb (lời giải).

Chạy lại script này nếu muốn tạo lại notebook gốc:  python labs/build_chuong07.py
Toàn bộ loigiai.ipynb chạy dưới 2 phút trên CPU (NumPy + SciPy + PyTorch CPU, seed cố định, dữ liệu sinh tại chỗ).
Mọi con số của bài săn bug và các thí nghiệm đo được tính lại ngay trong notebook.
"""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "chuong07"
OUT.mkdir(parents=True, exist_ok=True)

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

SETUP = """import time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.signal import correlate2d, convolve2d
torch.set_num_threads(2)
np.set_printoptions(precision=4, suppress=True)
T0 = time.time()

# ---- dữ liệu mẫu hình dùng chung (sinh tại chỗ, không tải mạng) ----
# 6 lớp: thanh ngang, thanh dọc, dấu cộng, khung vuông, đường chéo, phản chéo; mẫu 5×5 đặt vào ảnh 12×12 + nhiễu
GLYPH = {}
GLYPH[0] = np.zeros((5, 5)); GLYPH[0][2, :] = 1
GLYPH[1] = np.zeros((5, 5)); GLYPH[1][:, 2] = 1
GLYPH[2] = np.zeros((5, 5)); GLYPH[2][2, :] = 1; GLYPH[2][:, 2] = 1
GLYPH[3] = np.zeros((5, 5)); GLYPH[3][[0, -1], :] = 1; GLYPH[3][:, [0, -1]] = 1
GLYPH[4] = np.eye(5); GLYPH[5] = np.fliplr(np.eye(5)).copy()

def make_glyphs(n, offs_r, offs_c, seed, noise=0.3, size=12):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 1, size, size), np.float32); y = rng.integers(0, 6, n)
    for i in range(n):
        r, c = rng.choice(offs_r), rng.choice(offs_c)
        X[i, 0, r:r + 5, c:c + 5] = GLYPH[y[i]]
    X += rng.normal(0, noise, X.shape).astype(np.float32)
    return X, y"""

# ---------------------------------------------------------------- Lab 1
CONV1 = ("""def out_size(n, k, p=0, s=1, d=1):
    \"\"\"Kích thước đầu ra theo một chiều: ⌊(n + 2p − d(k − 1) − 1)/s⌋ + 1.\"\"\"
    # TODO
    raise NotImplementedError

def conv2d_naive(x, w, b=None, s=1, p=0, d=1):
    \"\"\"x: (N, Cin, H, W); w: (Cout, Cin, k, k). Cross-correlation (KHÔNG lật kernel), vòng lặp trên vị trí ra.
    Trả về (N, Cout, Ho, Wo).\"\"\"
    # TODO
    raise NotImplementedError

def conv2d_im2col(x, w, b=None, s=1, p=0, d=1):
    \"\"\"Cùng kết quả nhưng bằng một phép nhân ma trận: dựng ma trận cột (im2col) rồi nhân với w.reshape(Cout, -1).
    Gợi ý: chỉ cần vòng lặp trên k·k độ lệch (u, v), mỗi lần cắt một lát có bước s.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def out_size(n, k, p=0, s=1, d=1):
    return (n + 2 * p - d * (k - 1) - 1) // s + 1

def _pad(x, p):
    return np.pad(x, ((0, 0), (0, 0), (p, p), (p, p))) if p else x

def conv2d_naive(x, w, b=None, s=1, p=0, d=1):
    N, Cin, H, W = x.shape
    Co, _, k, _ = w.shape
    Ho, Wo = out_size(H, k, p, s, d), out_size(W, k, p, s, d)
    xp = _pad(x, p)
    y = np.zeros((N, Co, Ho, Wo))
    ext = d * (k - 1) + 1                                   # bề rộng thật của vùng nhìn
    for n in range(N):
        for o in range(Co):
            for i in range(Ho):
                for j in range(Wo):
                    patch = xp[n, :, i * s:i * s + ext:d, j * s:j * s + ext:d]   # (Cin, k, k)
                    y[n, o, i, j] = (patch * w[o]).sum() + (0 if b is None else b[o])
    return y

def conv2d_im2col(x, w, b=None, s=1, p=0, d=1):
    N, Cin, H, W = x.shape
    Co, _, k, _ = w.shape
    Ho, Wo = out_size(H, k, p, s, d), out_size(W, k, p, s, d)
    xp = _pad(x, p)
    cols = np.empty((N, Cin, k, k, Ho, Wo))
    for u in range(k):
        for v in range(k):
            cols[:, :, u, v] = xp[:, :, u * d:u * d + s * (Ho - 1) + 1:s, v * d:v * d + s * (Wo - 1) + 1:s]
    cols = cols.reshape(N, Cin * k * k, Ho * Wo)             # mỗi cột = một vùng nhìn duỗi thẳng
    y = w.reshape(Co, -1) @ cols                             # (N, Co, Ho·Wo): MỘT phép nhân ma trận
    if b is not None:
        y = y + b[None, :, None]
    return y.reshape(N, Co, Ho, Wo)""")

CONV1_CHECK = """# (a) Ví dụ giải mẫu 7.1: Sobel dọc trên ảnh 5×5
X = np.array([[1, 2, 0, 1, 3], [0, 1, 2, 3, 1], [2, 1, 0, 1, 0], [1, 0, 1, 2, 2], [3, 1, 0, 0, 1]], float)
K = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], float)
Y = conv2d_naive(X[None, None], K[None, None])[0, 0]
print(Y)
assert np.allclose(Y, [[-1, -3, -1], [2, -4, 0], [5, -3, -3]])
assert np.allclose(Y, correlate2d(X, K, mode="valid"))
assert np.allclose(convolve2d(X, K, mode="valid"), -Y), "convolution thật (lật kernel) cho đúng ÂM của kết quả này"
assert np.allclose(conv2d_im2col(X[None, None], K[None, None])[0, 0], Y)

# (b) So với F.conv2d trên nhiều cấu hình (n, k, p, s, d)
rng = np.random.default_rng(0)
cases = [(9, 3, 0, 1, 1), (9, 3, 1, 1, 1), (10, 3, 1, 2, 1), (11, 5, 2, 2, 1), (12, 3, 0, 1, 2), (13, 3, 2, 3, 2), (8, 1, 0, 1, 1)]
for n, k, p, s, d in cases:
    x = rng.standard_normal((2, 3, n, n)); w = rng.standard_normal((4, 3, k, k)); b = rng.standard_normal(4)
    ref = F.conv2d(torch.tensor(x), torch.tensor(w), torch.tensor(b), stride=s, padding=p, dilation=d).numpy()
    assert ref.shape[-1] == out_size(n, k, p, s, d), (n, k, p, s, d)
    assert np.allclose(conv2d_naive(x, w, b, s, p, d), ref), ("naive", n, k, p, s, d)
    assert np.allclose(conv2d_im2col(x, w, b, s, p, d), ref), ("im2col", n, k, p, s, d)

# (c) Kích thước của AlexNet và của stem ResNet
assert out_size(224, 11, 0, 4) == 54 and out_size(227, 11, 0, 4) == 55 and out_size(224, 11, 2, 4) == 55
assert out_size(224, 7, 3, 2) == 112 and out_size(112, 3, 1, 2) == 56 and out_size(32, 3, 0, 1, 2) == 28
print("Lab 1 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 2
CONV2 = ("""def conv_backward(x, w, gy):
    \"\"\"Một ảnh, stride 1, không padding. x: (Cin, H, W); w: (Cout, Cin, k, k); gy: (Cout, Ho, Wo) = ∂L/∂y.
    Trả về (dx, dw). Chỉ dùng correlate2d (mode='valid' hoặc 'full') và lật kernel bằng [::-1, ::-1];
    KHÔNG dùng autograd. Gợi ý: dw là correlation của x với gy; dx là correlation 'full' của gy với kernel đã lật.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def conv_backward(x, w, gy):
    Co, Cin, k, _ = w.shape
    dx, dw = np.zeros_like(x), np.zeros_like(w)
    for o in range(Co):
        for c in range(Cin):
            dw[o, c] = correlate2d(x[c], gy[o], mode="valid")                 # x ⋆ gy
            dx[c] += correlate2d(gy[o], w[o, c][::-1, ::-1], mode="full")     # gy ⋆ (kernel lật), đệm đủ
    return dx, dw""")

CONV2_CHECK = """# (a) Ví dụ 7.4 (1D, nhúng vào ảnh 1 hàng): x = (1..5), w = (1, 0, −1), gy = (1, 2, 3)
x1 = np.arange(1, 6, dtype=float).reshape(1, 1, 5)            # (Cin, H, W) với H = 1
w1 = np.array([1, 0, -1.0]).reshape(1, 1, 1, 3)               # (Cout, Cin, 1, 3)
gy1 = np.array([1, 2, 3.0]).reshape(1, 1, 3)                  # (Cout, Ho, Wo)
dx1, dw1 = conv_backward(x1, w1, gy1)
print("dx =", dx1.ravel(), " dw =", dw1.ravel())
assert np.allclose(dx1.ravel(), [1, 2, 2, -2, -3]) and np.allclose(dw1.ravel(), [14, 20, 26])

# (b) Gradient check sai phân trung tâm (float64) và đối chiếu autograd
r = np.random.default_rng(1)
x = r.standard_normal((2, 6, 6)); w = r.standard_normal((3, 2, 3, 3)); gy = r.standard_normal((3, 4, 4))
dx, dw = conv_backward(x, w, gy)
loss = lambda x_, w_: (gy * conv2d_naive(x_[None], w_)[0]).sum()          # L = ⟨gy, y⟩ nên ∂L/∂y = gy
def numgrad(f, a, h=1e-6):
    g = np.zeros_like(a)
    for idx in np.ndindex(a.shape):
        old = a[idx]; a[idx] = old + h; fp = f(); a[idx] = old - h; fm = f(); a[idx] = old
        g[idx] = (fp - fm) / (2 * h)
    return g
ndx, ndw = numgrad(lambda: loss(x, w), x), numgrad(lambda: loss(x, w), w)
rel = lambda a, b: np.abs(a - b).max() / max(np.abs(a).max(), np.abs(b).max())
print(f"sai số tương đối: dx {rel(dx, ndx):.1e}, dw {rel(dw, ndw):.1e}")
assert rel(dx, ndx) < 1e-6 and rel(dw, ndw) < 1e-6

tx, tw = torch.tensor(x, requires_grad=True), torch.tensor(w, requires_grad=True)
(torch.tensor(gy) * F.conv2d(tx[None], tw)[0]).sum().backward()
assert np.allclose(tx.grad.numpy(), dx) and np.allclose(tw.grad.numpy(), dw)
print("Lab 2 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 3
RF = ("""def rf(layers):
    \"\"\"layers: danh sách (k, s) hoặc (k, s, d) từ đầu vào ra đầu ra (pooling cũng là một lớp (k, s)).
    Trả về (RF, jump) của một điểm ở đầu ra. Công thức đệ quy: r ← r + d(k − 1)·j rồi j ← j·s, khởi đầu r = j = 1.\"\"\"
    # TODO
    raise NotImplementedError

def erf_std(L):
    \"\"\"Chồng L lớp conv 3×3, 1 kênh, không bias, mọi trọng số bằng 1/9, ảnh đủ lớn.
    Đặt gradient 1 tại điểm giữa của đầu ra, lan ngược bằng autograd, rồi lấy ĐỘ LỆCH CHUẨN (tính theo điểm ảnh)
    của |gradient| dọc theo hàng giữa của đầu vào, coi nó là một phân bố (chia cho tổng).\"\"\"
    # TODO
    raise NotImplementedError""",
"""def rf(layers):
    r, j = 1, 1
    for layer in layers:
        k, s = layer[0], layer[1]
        d = layer[2] if len(layer) > 2 else 1
        r += d * (k - 1) * j
        j *= s
    return r, j

def erf_std(L):
    n = 2 * L + 9
    net = nn.Sequential(*[nn.Conv2d(1, 1, 3, padding=1, bias=False) for _ in range(L)])
    for m in net:
        nn.init.constant_(m.weight, 1 / 9)
    x = torch.zeros(1, 1, n, n, requires_grad=True)
    c = n // 2
    net(x)[0, 0, c, c].backward()
    row = x.grad[0, 0, c].abs()
    pr = row / row.sum()
    idx = torch.arange(n) - c
    return torch.sqrt((pr * idx.float() ** 2).sum()).item()""")

RF_CHECK = """assert rf([(3, 1)] * 3) == (7, 1) and rf([(3, 1)] * 2) == (5, 1)
vgg = [(3, 1), (3, 1), (2, 2), (3, 1), (3, 1), (2, 2), (3, 1)]
assert rf(vgg) == (24, 4), rf(vgg)
assert rf([(7, 2), (3, 2)]) == (11, 4)                       # stem ResNet: conv 7×7 s2, pool 3×3 s2
assert rf([(2, 1, 1), (2, 1, 2), (2, 1, 4), (2, 1, 8)]) == (16, 1)   # k = 2, dilation 1, 2, 4, 8: 1 + (1 + 2 + 4 + 8)
print("RF của chồng VGG:", rf(vgg), "| stem ResNet:", rf([(7, 2), (3, 2)]))

print("L   RF lý thuyết   ERF std   ERF std / √L")
for L in [1, 4, 16, 64]:
    sd = erf_std(L)
    print(f"{L:2d}  {2 * L + 1:6d}        {sd:7.3f}   {sd / np.sqrt(L):.4f}")
    assert 0.80 < sd / np.sqrt(L) < 0.83, "lý thuyết: √(2/3) = 0.8165"
print("Lab 3 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 4
SHIFT = ("""def build_model(kind):
    \"\"\"Ba mô hình cho ảnh 1×12×12, 6 lớp:
    'mlp'      : Flatten → Linear(144, 128) → ReLU → Linear(128, 6)
    'cnn_flat' : Conv(1→8, 3, p=1) → ReLU → MaxPool2 → Conv(8→16, 3, p=1) → ReLU → MaxPool2 → Flatten → Linear(144, 6)
    'cnn_gmp'  : Conv(1→8) → ReLU → Conv(8→16) → ReLU → Conv(16→16) → ReLU (đều 3×3, p=1, không pooling)
                 → AdaptiveMaxPool2d(1) → Flatten → Linear(16, 6)\"\"\"
    # TODO
    raise NotImplementedError

def train_eval(kind, seed, ntrain=2000, epochs=12):
    \"\"\"Huấn luyện chỉ trên mẫu hình đặt ở vị trí (hàng, cột) ∈ {2, 3, 4}², Adam 1e-3, batch 64.
    Trả về (số tham số, acc trong phân phối, acc ở vị trí lạ {0, 1, 6, 7}²). Nhớ net.eval().\"\"\"
    # TODO
    raise NotImplementedError""",
"""def build_model(kind):
    if kind == "mlp":
        return nn.Sequential(nn.Flatten(), nn.Linear(144, 128), nn.ReLU(), nn.Linear(128, 6))
    if kind == "cnn_flat":
        return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Flatten(), nn.Linear(16 * 3 * 3, 6))
    if kind == "cnn_gmp":
        return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
                             nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
                             nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                             nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(16, 6))
    raise ValueError(kind)

def accuracy(net, X, y):
    with torch.no_grad():
        return (net(torch.tensor(X)).argmax(1).numpy() == y).mean()

def train_eval(kind, seed, ntrain=2000, epochs=12):
    torch.manual_seed(seed)
    Xtr, ytr = make_glyphs(ntrain, [2, 3, 4], [2, 3, 4], seed=1)
    X, y = torch.tensor(Xtr), torch.tensor(ytr)
    net = build_model(kind)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]
            opt.zero_grad(); F.cross_entropy(net(X[idx]), y[idx]).backward(); opt.step()
    net.eval()
    return (n_params, accuracy(net, *make_glyphs(1000, [2, 3, 4], [2, 3, 4], seed=2)),
            accuracy(net, *make_glyphs(1000, [0, 1, 6, 7], [0, 1, 6, 7], seed=3)))""")

SHIFT_CHECK = """res = {}
for kind in ["mlp", "cnn_flat", "cnn_gmp"]:
    runs = [train_eval(kind, s) for s in range(2)]
    res[kind] = (runs[0][0], np.mean([r[1] for r in runs]), np.mean([r[2] for r in runs]), [round(float(r[2]), 3) for r in runs])
    print(f"{kind:9s} {res[kind][0]:6d} tham số | trong phân phối {res[kind][1]:.3f} | vị trí lạ {res[kind][2]:.3f}  (từng seed: {res[kind][3]})")
assert all(v[1] > 0.8 for v in res.values())
assert res["mlp"][2] < 0.35 and res["cnn_flat"][2] < 0.45 and res["cnn_gmp"][2] > 0.9
print(f"thời gian từ đầu notebook: {time.time() - T0:.1f} s")
print("Lab 4 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 5
FOLD = ("""def fold_bn(conv, bn):
    \"\"\"Gộp BatchNorm2d (chế độ eval) vào Conv2d đứng ngay trước nó. Trả về nn.Conv2d MỚI có bias,
    cùng cấu hình (in/out_channels, kernel_size, stride, padding, dilation, groups).
    w' = w·γ/√(σ² + ε),  b' = (b − μ)·γ/√(σ² + ε) + β  (σ², μ là running stats). conv.bias có thể là None.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def fold_bn(conv, bn):
    scale = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
    b0 = conv.bias.detach() if conv.bias is not None else torch.zeros(conv.out_channels)
    new = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                    conv.dilation, conv.groups, bias=True)
    with torch.no_grad():
        new.weight.copy_(conv.weight * scale[:, None, None, None])
        new.bias.copy_((b0 - bn.running_mean) * scale + bn.bias)
    return new""")

TEXTDATA = """# ---- dữ liệu văn bản tổng hợp: nhãn 1 ⇔ bigram (3, 7) xuất hiện LIỀN KỀ đúng thứ tự ----
V, LEN = 50, 20
def make_text(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, V, (n, LEN))
    y = rng.integers(0, 2, n)
    for i in range(n):
        if y[i] == 1:
            j = rng.integers(0, LEN - 1)
            X[i, j], X[i, j + 1] = 3, 7
        elif rng.random() < 0.5:                         # mồi nhử: có cả 3 lẫn 7 nhưng không liền kề đúng thứ tự
            a = rng.integers(0, LEN)
            b = rng.choice([q for q in range(LEN) if q != a and q != a + 1])
            X[i, a], X[i, b] = 3, 7
            if a + 1 < LEN and a + 1 != b and X[i, a + 1] == 7:
                X[i, a + 1] = 8                          # token ngẫu nhiên bên cạnh vô tình tạo (3, 7)
            if b - 1 >= 0 and b - 1 != a and X[i, b - 1] == 3:
                X[i, b - 1] = 9
        if y[i] == 0:                                    # mẫu nhãn 0 không được chứa (3, 7) liền kề
            for j in range(LEN - 1):
                if X[i, j] == 3 and X[i, j + 1] == 7:
                    X[i, j + 1] = 8
        else:                                            # mẫu nhãn 1: xóa bigram ngẫu nhiên khác, chỉ giữ cái đã chèn
            for q in range(LEN - 1):
                if X[i, q] == 3 and X[i, q + 1] == 7 and q != j:
                    X[i, q + 1] = 8
    return torch.tensor(X), torch.tensor(y)

def has_bigram(X):
    return ((X[:, :-1] == 3) & (X[:, 1:] == 7)).any(1)

def train_text(model, epochs=15, seed=0):
    torch.manual_seed(seed)
    Xt, yt = make_text(2400, 1); Xv, yv = make_text(1000, 2)
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            idx = perm[i:i + 64]
            opt.zero_grad(); F.cross_entropy(model(Xt[idx]), yt[idx]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return (model(Xv).argmax(1) == yv).float().mean().item()"""

TEXTCNN = ("""class BoW(nn.Module):
    \"\"\"Embedding(V, 16) → TRUNG BÌNH theo vị trí → Linear(16, 32) → ReLU → Linear(32, 2). Đầu vào (N, LEN) số nguyên.\"\"\"
    def __init__(self):
        super().__init__()
        # TODO
        raise NotImplementedError
    def forward(self, x):
        # TODO
        raise NotImplementedError

class TextCNN(nn.Module):
    \"\"\"Embedding(V, 16) → Conv1d(16, 32, k) → ReLU → MAX theo thời gian (dim cuối) → Linear(32, 2).
    Chú ý: Conv1d cần (N, kênh, độ dài), embedding trả (N, độ dài, kênh).\"\"\"
    def __init__(self, k):
        super().__init__()
        # TODO
        raise NotImplementedError
    def forward(self, x):
        # TODO
        raise NotImplementedError""",
"""class BoW(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, 16)
        self.head = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, x):
        return self.head(self.emb(x).mean(1))

class TextCNN(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.emb = nn.Embedding(V, 16)
        self.conv = nn.Conv1d(16, 32, k)
        self.fc = nn.Linear(32, 2)
    def forward(self, x):
        h = F.relu(self.conv(self.emb(x).transpose(1, 2)))     # (N, 32, LEN − k + 1)
        return self.fc(h.max(2).values)                         # max-over-time: độ dài bất kỳ""")

FOLD_CHECK = """# (a) Ví dụ giải mẫu 7.13: conv 1×1 với w = 2, b = 1; BN với μ = 3, σ² = 4, γ = 1.5, β = 0.5, ε = 0
c1 = nn.Conv2d(1, 1, 1); b1 = nn.BatchNorm2d(1, eps=0.0)
with torch.no_grad():
    c1.weight.fill_(2.0); c1.bias.fill_(1.0)
    b1.running_mean.fill_(3.0); b1.running_var.fill_(4.0); b1.weight.fill_(1.5); b1.bias.fill_(0.5)
c1.eval(); b1.eval()
f1 = fold_bn(c1, b1)
print("w' =", f1.weight.item(), " b' =", f1.bias.item())
assert abs(f1.weight.item() - 1.5) < 1e-6 and abs(f1.bias.item() + 1.0) < 1e-6
xx = torch.tensor([[[[4.0]]]])
assert abs(b1(c1(xx)).item() - 5.0) < 1e-6 and abs(f1(xx).item() - 5.0) < 1e-6

# (b) Ngẫu nhiên: conv có và không bias, stride, padding, dilation, groups
torch.manual_seed(0)
worst = 0.0
for kw in [dict(bias=True), dict(bias=False), dict(bias=True, stride=2, padding=1), dict(bias=False, dilation=2, padding=2)]:
    conv = nn.Conv2d(3, 8, 3, **kw); bn = nn.BatchNorm2d(8)
    with torch.no_grad():
        bn.running_mean.normal_(); bn.running_var.uniform_(0.5, 2); bn.weight.normal_(); bn.bias.normal_()
    conv.eval(); bn.eval()
    x = torch.randn(4, 3, 10, 10)
    with torch.no_grad():
        worst = max(worst, (bn(conv(x)) - fold_bn(conv, bn)(x)).abs().max().item())
print(f"sai khác lớn nhất giữa 'conv rồi BN' và conv đã gộp: {worst:.2e}")
assert worst < 1e-5

# (c) Chỉ đúng ở eval: ở chế độ train, BN dùng thống kê của batch nên phép gộp không còn khớp
conv = nn.Conv2d(3, 8, 3); bn = nn.BatchNorm2d(8)
with torch.no_grad():
    bn.running_mean.normal_(); bn.running_var.uniform_(0.5, 2)
folded = fold_bn(conv, bn)
bn.train()
x = torch.randn(4, 3, 10, 10)
with torch.no_grad():
    gap_train = (bn(conv(x)) - folded(x)).abs().max().item()
print(f"nếu BN ở chế độ train, sai khác tới {gap_train:.2f}")
assert gap_train > 0.1

# (d) Kiểm luật nhãn của dữ liệu, trần của mô hình chỉ biết "có 3 và có 7", rồi TextCNN và BoW
for sd in (1, 2):
    Xc, yc = make_text(2400 if sd == 1 else 1000, sd)
    assert bool((has_bigram(Xc) == yc.bool()).all()), "nhãn 1 ⇔ có bigram (3, 7) liền kề"
Xv_, yv_ = make_text(1000, 2)
both = ((Xv_ == 3).any(1)) & ((Xv_ == 7).any(1))
ceil = ((~both & (yv_ == 0)).sum() + (both & (yv_ == 1)).sum()) / len(yv_)   # đoán 1 khi có cả 3 lẫn 7
print(f"trần acc val của mô hình chỉ biết 'có 3 và có 7' (đoán 1 nếu có cả hai): {ceil:.3f}")
print(f"số tham số: BoW {sum(p.numel() for p in BoW().parameters())}, "
      f"k=1 {sum(p.numel() for p in TextCNN(1).parameters())}, k=2 {sum(p.numel() for p in TextCNN(2).parameters())}")
accs = {"BoW": train_text(BoW())}
for k in (1, 2, 3):
    accs[f"conv k={k}"] = train_text(TextCNN(k))
for name, a in accs.items():
    print(f"  {name:9s} acc val {a:.3f}")
assert accs["conv k=2"] > 0.95 and accs["conv k=3"] > 0.95
assert accs["BoW"] < 0.80 and accs["conv k=1"] < 0.80
print(f"thời gian từ đầu notebook: {time.time() - T0:.1f} s")
print("Lab 5 ĐẠT ✔")"""


# ---------------------------------------------------------------- Kiểm nhanh các khẳng định "đã kiểm" trong sách
QUICK = """# Mỗi khối dưới đây tái lập một câu "đã kiểm" trong sách (mục 7.1, 7.2, 7.5).
torch.manual_seed(0)

# (a) Mục 7.1: conv 1×1 = Linear áp riêng tại từng điểm ảnh (hoán trục kênh về cuối)
c11 = nn.Conv2d(5, 7, 1); lin = nn.Linear(5, 7)
with torch.no_grad():
    lin.weight.copy_(c11.weight[:, :, 0, 0]); lin.bias.copy_(c11.bias)
    xx = torch.randn(2, 5, 4, 4)
    d11 = (c11(xx) - lin(xx.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)).abs().max().item()
print(f"(a) conv 1×1 so với Linear: sai khác {d11:.1e}")
assert d11 < 1e-5

# (b) Mục 7.2 (cuối ví dụ 7.7): conv 1D + ReLU + max-pool 2, circular padding: dịch 2 điểm thì đầu ra dịch đúng 1 ô
cc = nn.Conv1d(1, 4, 3, padding=1, padding_mode="circular")
f = lambda z: F.max_pool1d(F.relu(cc(z)), 2)
with torch.no_grad():
    z = torch.randn(1, 1, 32)
    o = f(z)
    ok2 = torch.allclose(f(torch.roll(z, 2, -1)), torch.roll(o, 1, -1), atol=1e-6)
    ok1 = any(torch.allclose(f(torch.roll(z, 1, -1)), torch.roll(o, r, -1), atol=1e-6) for r in range(16))
print(f"(b) dịch 2 điểm -> đầu ra dịch 1 ô: {ok2}; dịch 1 điểm khớp một phép dịch nào đó: {ok1}")
assert ok2 and not ok1

# (c) Mục 7.5: conv nhân quả (đệm trái d(k − 1)): đổi đầu vào ở t = 5 chỉ đổi đầu ra từ t = 5 trở đi
k_, d_ = 3, 2
cz = nn.Conv1d(1, 1, k_, dilation=d_)
causal = lambda z: cz(F.pad(z, (d_ * (k_ - 1), 0)))
with torch.no_grad():
    z = torch.randn(1, 1, 16); z2 = z.clone(); z2[..., 5] += 1.0
    dy = (causal(z2) - causal(z)).abs()[0, 0]
print("(c) vị trí đầu ra bị đổi:", torch.nonzero(dy > 1e-7).flatten().tolist())
assert causal(z).shape[-1] == 16 and torch.nonzero(dy > 1e-7).flatten().tolist() == [5, 5 + d_, 5 + 2 * d_]   # k = 3, d = 2: chỉ t = 5, 7, 9

# (d) Mục 7.5, ví dụ 7.17: patch embedding = conv P×P stride P = unfold rồi Linear
P_, D_ = 16, 32
pe = nn.Conv2d(3, D_, P_, stride=P_)
with torch.no_grad():
    img = torch.randn(1, 3, 64, 64)
    tok_conv = pe(img).flatten(2).transpose(1, 2)                              # (1, 16, D)
    lin2 = nn.Linear(3 * P_ * P_, D_)
    lin2.weight.copy_(pe.weight.reshape(D_, -1)); lin2.bias.copy_(pe.bias)
    tok_lin = lin2(F.unfold(img, P_, stride=P_).transpose(1, 2))
    dpe = (tok_conv - tok_lin).abs().max().item()
print(f"(d) patch embedding: conv so với unfold + Linear, sai khác {dpe:.1e}")
assert dpe < 1e-4

# (e) Mục 7.1: "same" với dilation cần p = d(k − 1)/2; mục 7.5 / bài dựng Manim: phủ liền hay có lỗ
assert out_size(32, 3, 2, 1, 2) == 32 and out_size(32, 3, 1, 1, 2) == 30
def coverage(k, dils):
    offs = {0}
    for d in dils:
        offs = {o + d * u for o in offs for u in range(k)}
    return sorted(offs)
for k, dils in [(2, (1, 2, 4)), (3, (1, 2, 4)), (3, (1, 3, 9)), (3, (1, 4, 16)), (3, (2, 4))]:
    c = coverage(k, dils)
    holes = [q for q in range(c[0], c[-1] + 1) if q not in c]
    print(f"(e) k = {k}, dilation {dils}: RF {c[-1] + 1}, lỗ: {holes if holes else 'không'}")
assert coverage(2, (1, 2, 4)) == list(range(8)) and coverage(3, (1, 2, 4)) == list(range(15))
assert coverage(3, (1, 4, 16)) != list(range(max(coverage(3, (1, 4, 16))) + 1))
assert coverage(3, (2, 4)) != list(range(max(coverage(3, (2, 4))) + 1))

# (f) Ví dụ 7.16: conv k = 3 ở n = 128, d = 768
print("(f) 128·3·768² =", 128 * 3 * 768 ** 2)
assert 128 * 3 * 768 ** 2 == 226_492_416
print("Kiểm nhanh ĐẠT ✔")"""

# ---------------------------------------------------------------- Bug hunt
BUG_DATA = """def make_rgb(n, seed, offs=range(8)):
    \"\"\"Ảnh 3 kênh (cùng mẫu hình, mỗi kênh một cường độ ngẫu nhiên + nhiễu), lưu dạng NHWC như PIL/skimage.\"\"\"
    X, y = make_glyphs(n, list(offs), list(offs), seed=seed, noise=0.0)
    gain = np.random.default_rng(seed + 100).uniform(0.3, 1.0, (n, 3, 1, 1)).astype(np.float32)
    Xc = X * gain + np.random.default_rng(seed + 200).normal(0, 0.3, (n, 3, 12, 12)).astype(np.float32)
    return np.ascontiguousarray(Xc.transpose(0, 2, 3, 1)), y        # (N, 12, 12, 3)

def make_net():
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                         nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(16, 6))"""

BUGGY_SRC = '''def train_buggy(epochs=12, seed=0):
    torch.manual_seed(seed)
    Xtr_hwc, ytr = make_rgb(2400, seed=1)                 # (N, 12, 12, 3)
    Xva_hwc, yva = make_rgb(1000, seed=2)
    def prep(a):
        t = torch.tensor(a)
        return t.view(len(t), 3, 12, 12)                  # đưa về dạng PyTorch cần
    X, y, Xv, yv = prep(Xtr_hwc), torch.tensor(ytr), prep(Xva_hwc), torch.tensor(yva)
    def augment(xb):                                      # tăng cường dữ liệu: lật ngang ngẫu nhiên
        flip = torch.rand(len(xb)) < 0.5
        return torch.where(flip[:, None, None, None], xb.flip(3), xb)
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]
            xb, yb = augment(X[idx]), y[idx]
            if torch.rand(1).item() < 0.5:
                xb = torch.roll(xb, 2, dims=0)            # "xáo trộn" batch cho đa dạng
            opt.zero_grad()
            F.cross_entropy(net(xb), yb).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        xv = augment(Xv)
        out1 = net(xv)
        out2 = net(augment(Xv))                           # gọi lần hai trên cùng tập val
        val_loss = F.cross_entropy(out1, yv).item()
        acc1 = (out1.argmax(1) == yv).float().mean().item()
        acc2 = (out2.argmax(1) == yv).float().mean().item()
    return net, val_loss, acc1, acc2, out1.argmax(1), yv'''

BUGFIX_STUB = """def train_fixed(epochs=12, seed=0):
    \"\"\"Viết lại train_buggy cho đúng. Trả về cùng 6 giá trị; hai lần gọi trên tập val phải cho
    output GIỐNG HỆT nhau (torch.equal).\"\"\"
    # TODO
    raise NotImplementedError"""

BUGFIX_SOL = '''def train_fixed(epochs=12, seed=0):
    torch.manual_seed(seed)
    Xtr_hwc, ytr = make_rgb(2400, seed=1)
    Xva_hwc, yva = make_rgb(1000, seed=2)
    def prep(a):
        return torch.tensor(a).permute(0, 3, 1, 2).contiguous()   # sửa lỗi 1: permute, không view
    X, y, Xv, yv = prep(Xtr_hwc), torch.tensor(ytr), prep(Xva_hwc), torch.tensor(yva)
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]
            xb, yb = X[idx], y[idx]                               # sửa lỗi 2 và 3: bỏ lật ngang (hai lớp là ảnh gương)
            opt.zero_grad()                                       # và bỏ roll (ảnh lệch nhãn)
            F.cross_entropy(net(xb), yb).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        out1, out2 = net(Xv), net(Xv)                             # sửa lỗi 4: không tăng cường lúc đánh giá
        val_loss = F.cross_entropy(out1, yv).item()
        acc1 = (out1.argmax(1) == yv).float().mean().item()
        acc2 = (out2.argmax(1) == yv).float().mean().item()
    return net, val_loss, acc1, acc2, out1.argmax(1), yv'''

BUG_RUN = """net_b, loss_b, acc1_b, acc2_b, pred_b, yv_b = train_buggy()
print(f"BẢN LỖI: loss val {loss_b:.3f}, acc val {acc1_b:.3f} (lần gọi hai: {acc2_b:.3f})")
cm = np.zeros((6, 6), int)
for p, t in zip(pred_b.numpy(), yv_b.numpy()):
    cm[t, p] += 1
print("ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán):\\n", cm)"""

BUG_CHECK = """net_f, loss_f, acc1_f, acc2_f, pred_f, yv_f = train_fixed()
print(f"BẢN SỬA: loss val {loss_f:.3f}, acc val {acc1_f:.3f} (lần gọi hai: {acc2_f:.3f})")
assert acc1_f > 0.95 and loss_f < 0.3, "bản đúng phải đạt acc > 0.95, loss < 0.3"
assert acc1_f == acc2_f, "hai lần gọi trên cùng tập val phải cho cùng kết quả"
assert acc1_b < 0.6, "bản lỗi phải thấp rõ rệt (chạy ô trước)"
print("Săn bug ĐẠT ✔")"""

ABLATION = """# Bật riêng từng lỗi trên bản đúng (cùng seed) để thấy lỗi nào gây triệu chứng nào.
def run_bug(view=False, flip=False, roll=False, valaug=False, epochs=12, seed=0):
    torch.manual_seed(seed)
    Xtr_hwc, ytr = make_rgb(2400, seed=1); Xva_hwc, yva = make_rgb(1000, seed=2)
    prep = (lambda a: torch.tensor(a).view(len(a), 3, 12, 12)) if view else (lambda a: torch.tensor(a).permute(0, 3, 1, 2).contiguous())
    X, y, Xv, yv = prep(Xtr_hwc), torch.tensor(ytr), prep(Xva_hwc), torch.tensor(yva)
    def aug(xb):
        m = torch.rand(len(xb)) < 0.5
        return torch.where(m[:, None, None, None], xb.flip(3), xb)
    net = make_net(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]; xb, yb = (aug(X[idx]) if flip else X[idx]), y[idx]
            if roll and torch.rand(1).item() < 0.5:
                xb = torch.roll(xb, 2, dims=0)
            opt.zero_grad(); F.cross_entropy(net(xb), yb).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        o1 = net(aug(Xv) if valaug else Xv); o2 = net(aug(Xv) if valaug else Xv)
        return (F.cross_entropy(o1, yv).item(), (o1.argmax(1) == yv).float().mean().item(),
                (o2.argmax(1) == yv).float().mean().item())

rows = [("không lỗi", {}), ("1. view", dict(view=True)), ("2. lật ngang", dict(flip=True)), ("3. roll", dict(roll=True)),
        ("4. aug lúc val (một mình)", dict(valaug=True)),
        ("2 + 3 + 4", dict(flip=True, roll=True, valaug=True)), ("cả bốn", dict(view=True, flip=True, roll=True, valaug=True))]
table = {}
for name, kw in rows:
    table[name] = run_bug(**kw)
    print(f"  {name:26s} loss val {table[name][0]:.3f}  acc val {table[name][1]:.3f} (lần hai {table[name][2]:.3f})")
assert table["không lỗi"][1] > 0.95
assert table["1. view"][1] < 0.7 and table["2. lật ngang"][1] < 0.85 and table["3. roll"][1] < 0.95
assert abs(table["cả bốn"][1] - acc1_b) < 1e-9, "bản ablation 'cả bốn' phải trùng train_buggy"
print(f"thời gian từ đầu notebook: {time.time() - T0:.1f} s")
print("Phân tích lỗi ĐẠT ✔")"""

# ---------------------------------------------------------------- Phân tích thêm
EXTRA = """# (A) Độ sâu, residual, BatchNorm: chuẩn gradient lớp conv đầu so với lớp conv cuối, ngay tại khởi tạo (He init)
class DeepNet(nn.Module):
    def __init__(self, L, res, bn, c=16):
        super().__init__()
        self.stem = nn.Conv2d(3, c, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=not bn), *([nn.BatchNorm2d(c)] if bn else []))
                                     for _ in range(L)])
        self.res, self.head = res, nn.Linear(c, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    def forward(self, x):
        h = F.relu(self.stem(x))
        for b in self.blocks:
            h = F.relu(h + b(h)) if self.res else F.relu(b(h))
        return self.head(h.mean((2, 3)))

xg, yg = torch.randn(32, 3, 16, 16), torch.randint(0, 10, (32,))
print("L   cấu hình          loss khởi tạo   g_đầu/g_cuối")
depth_res = {}
for L in (10, 30, 60):
    for res, bn, name in [(False, False, "plain"), (False, True, "plain + BN"), (True, False, "residual"), (True, True, "residual + BN")]:
        torch.manual_seed(1)
        net = DeepNet(L, res, bn); net.train()
        loss = F.cross_entropy(net(xg), yg); loss.backward()
        ratio = net.blocks[0][0].weight.grad.norm().item() / net.blocks[-1][0].weight.grad.norm().item()
        depth_res[(L, name)] = (loss.item(), ratio)
        print(f"{L:2d}  {name:14s}  {loss.item():14.4g}   {ratio:10.3g}")
assert depth_res[(60, "residual")][0] > 1e6, "residual không chuẩn hóa: tín hiệu bùng nổ"
assert depth_res[(60, "residual + BN")][0] < 10 and depth_res[(60, "plain")][1] < 2
print("không có biến mất gradient ở plain He-init; BN làm gradient lớp đầu phình ra; residual cần chuẩn hóa")

# (B) Quên eval() khi đóng băng backbone có BN: trọng số không đổi nhưng running stats trôi
def backbone():
    return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.Conv2d(8, 16, 3, padding=1),
                         nn.BatchNorm2d(16), nn.ReLU(), nn.AdaptiveMaxPool2d(1), nn.Flatten())
torch.manual_seed(0)
bb, head = backbone(), nn.Linear(16, 6)
Xp, yp = make_glyphs(2000, [2, 3, 4, 5], [2, 3, 4, 5], seed=1); Xp, yp = torch.tensor(Xp), torch.tensor(yp)
opt = torch.optim.Adam(list(bb.parameters()) + list(head.parameters()), 1e-3)
for ep in range(8):
    bb.train(); perm = torch.randperm(2000)
    for i in range(0, 2000, 64):
        idx = perm[i:i + 64]; opt.zero_grad(); F.cross_entropy(head(bb(Xp[idx])), yp[idx]).backward(); opt.step()
bb.eval()
Xn, yn = make_glyphs(2000, [2, 3, 4, 5], [2, 3, 4, 5], seed=5, noise=0.8); Xn, yn = torch.tensor(Xn) * 3, torch.tensor(yn)
Xnv, ynv = make_glyphs(1000, [2, 3, 4, 5], [2, 3, 4, 5], seed=6, noise=0.8); Xnv, ynv = torch.tensor(Xnv) * 3, torch.tensor(ynv)
import copy
frozen = {}
for mode in ("quên eval()", "đúng: eval()"):
    b2, h2 = copy.deepcopy(bb), nn.Linear(16, 6)
    torch.manual_seed(1)
    for p in b2.parameters():
        p.requires_grad_(False)
    v0 = b2[1].running_var.clone()
    o = torch.optim.Adam(h2.parameters(), 1e-2)
    for ep in range(6):
        b2.train() if mode.startswith("quên") else b2.eval()
        perm = torch.randperm(2000)
        for i in range(0, 2000, 64):
            idx = perm[i:i + 64]; o.zero_grad(); F.cross_entropy(h2(b2(Xn[idx])), yn[idx]).backward(); o.step()
    b2.eval()
    with torch.no_grad():
        acc = (h2(b2(Xnv)).argmax(1) == ynv).float().mean().item()
    changed = any(not torch.equal(p, q) for p, q in zip(b2.parameters(), bb.parameters()))
    frozen[mode] = (acc, (b2[1].running_var / v0).mean().item(), changed)
    print(f"{mode:13s} acc miền mới {acc:.3f} | trọng số đổi: {changed} | running_var nhân {frozen[mode][1]:.1f} lần")
assert not frozen["quên eval()"][2] and frozen["quên eval()"][1] > 10 and frozen["đúng: eval()"][1] == 1.0
print(f"Tổng thời gian notebook: {time.time() - T0:.1f} s")
print("Phân tích thêm ĐẠT ✔")"""


def build(solution: bool):
    pick = (lambda pair: pair[1]) if solution else (lambda pair: pair[0])
    nb = nbf.v4.new_notebook()
    head = "Lời giải" if solution else "Bài lab"
    cells = [
        md(f"# Chương 7 — {head}: CNN và thị giác máy tính\n\n"
           + ("Bản lời giải đã chạy kiểm (NumPy + SciPy + PyTorch CPU, dưới 2 phút). Chỉ mở sau khi đã tự làm `lab.ipynb`."
              if solution else
              "Viết code vào các ô có `# TODO`, rồi chạy ô kiểm tra ngay bên dưới. "
              "Ô kiểm tra in **ĐẠT ✔** là xong. Bí quá 20 phút mới mở `loigiai.ipynb`. "
              "Chỉ cần PyTorch bản CPU; toàn notebook chạy dưới 2 phút.")),
        md("## Dữ liệu chung\nSáu mẫu hình 5×5 (thanh ngang, thanh dọc, dấu cộng, khung vuông, đường chéo, phản chéo) đặt vào ảnh 12×12 "
           "ở vị trí ngẫu nhiên, cộng nhiễu Gauss. Sinh tại chỗ, không tải gì từ mạng."),
        code(SETUP),
        md("## Lab 1 — Tích chập từ đầu: vòng lặp, im2col, kích thước đầu ra\n"
           "Nhớ: `nn.Conv2d` tính **cross-correlation** (không lật kernel). Đối chiếu với ví dụ giải mẫu 7.1 rồi với `F.conv2d`."),
        code(pick(CONV1)), code(CONV1_CHECK),
        md("## Lab 2 — Backward của conv và gradient check\n"
           "Gradient theo đầu vào là correlation 'full' của gradient đầu ra với kernel **lật**; gradient theo kernel là "
           "correlation của đầu vào với gradient đầu ra. Kiểm bằng sai phân trung tâm (float64) và autograd."),
        code(pick(CONV2)), code(CONV2_CHECK),
        md("## Lab 3 — Máy tính receptive field và receptive field hiệu dụng\n"
           "Công thức đệ quy cho RF lý thuyết; rồi đo bằng autograd vùng ảnh hưởng thật của chồng conv 3×3 đều."),
        code(pick(RF)), code(RF_CHECK),
        md("## Lab 4 — Vị trí lạ: MLP, CNN + Flatten, CNN + global max pooling\n"
           "Huấn luyện chỉ với mẫu hình ở vùng giữa (offset 2–4), kiểm tra ở các góc (offset 0, 1, 6, 7). "
           "Hai seed, 12 epoch, để notebook chạy nhanh."),
        code(pick(SHIFT)), code(SHIFT_CHECK),
        md("## Lab 5 — Gộp BN vào conv và TextCNN\nPhần đầu: `fold_bn` rồi so sánh. Phần sau: dữ liệu văn bản tổng hợp "
           "mà nhãn phụ thuộc thứ tự hai token liền kề; so BoW với conv 1D cỡ k = 1, 2, 3."),
    ]
    # Lab 5 gồm: ô định nghĩa fold_bn, ô dữ liệu văn bản, ô định nghĩa mô hình, ô kiểm tra
    cells += [code(pick(FOLD)), code(TEXTDATA), code(pick(TEXTCNN)), code(FOLD_CHECK),
              md("## Kiểm nhanh các khẳng định \"đã kiểm\" trong sách\nMột ô, không có TODO: tái lập các câu của mục 7.1, 7.2, 7.5 "
                 "(conv 1×1 = Linear, dịch với max-pool, conv nhân quả, patch = unfold + Linear, dilation phủ liền hay có lỗ)."),
              code(QUICK),
              md("## Săn bug\n`train_buggy` chạy không báo lỗi. Ảnh lưu dạng NHWC như PIL/skimage; bản đúng đạt acc val khoảng 0.98. "
                 "Chạy ô dưới để thấy triệu chứng, đọc code, tìm cả **bốn** lỗi rồi viết `train_fixed`."),
              code(BUG_DATA), code(BUGGY_SRC), code(BUG_RUN), code(pick((BUGFIX_STUB, BUGFIX_SOL))), code(BUG_CHECK),
              md("## Phân tích thêm (tùy chọn)\nPhần này làm nếu còn giờ: kết luận chính đã có trong sách (ví dụ 7.11 và mục 7.4). "
                 "Ô dưới chạy được cả khi chưa viết gì: hai thí nghiệm đo với bảng đầy đủ, độ sâu với residual/BN và quên "
                 "`eval()` khi đóng băng backbone. " + ("" if solution else "Bảng phân tích từng lỗi của bài săn bug chỉ có trong loigiai.ipynb, "
                 "để bạn tự tìm lỗi trước.")),
              *([code(ABLATION)] if solution else []), code(EXTRA)]
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    name = "loigiai.ipynb" if solution else "lab.ipynb"
    nbf.write(nb, OUT / name)
    return OUT / name


if __name__ == "__main__":
    print(build(False), build(True))
