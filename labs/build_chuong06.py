"""Sinh hai notebook cho lab chương 6: lab.ipynb (có TODO) và loigiai.ipynb (lời giải).

Chạy lại script này nếu muốn tạo lại notebook gốc:  python labs/build_chuong06.py
Toàn bộ loigiai.ipynb chạy khoảng 40 giây trên CPU rảnh (có thể 2–3 phút khi máy bận) (NumPy + PyTorch CPU, seed cố định).
Mọi con số của bài săn bug (kể cả khi gài nhiều lỗi cùng lúc) được tính lại ngay trong notebook.
"""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "chuong06"
OUT.mkdir(parents=True, exist_ok=True)

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

SETUP = """import time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
torch.set_num_threads(2)
np.set_printoptions(precision=4, suppress=True)
T0 = time.time()

def make_data(seed=0, N=6000, D=20, C=10):
    \"\"\"10 cụm Gauss chồng lấn trong ℝ²⁰ (đủ khó để lỗi lộ triệu chứng), đặc trưng chưa chuẩn hóa (×5 + 3).\"\"\"
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(C, D, generator=g) * 0.6
    y = torch.randint(0, C, (N,), generator=g)
    X = (centers[y] + torch.randn(N, D, generator=g)) * 5 + 3
    return X[:4000], y[:4000], X[4000:], y[4000:]

Xtr_raw, ytr, Xva_raw, yva = make_data()
mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0)            # thống kê CHỈ lấy từ tập train
Xtr, Xva = (Xtr_raw - mu) / sd, (Xva_raw - mu) / sd
print("train", tuple(Xtr.shape), "val", tuple(Xva.shape))"""

# ---------------------------------------------------------------- Lab 1
MLP_NP = ("""def forward(X, params):
    \"\"\"X: (N, d). params = (W1 (d,H), b1 (H,), W2 (H,C), b2 (C,)).
    Trả về (logits Z2, cache) với cache chứa những gì backward cần (X, Z1, H).\"\"\"
    # TODO
    raise NotImplementedError

def softmax_ce(Z, y):
    \"\"\"Softmax ổn định số (trừ max từng hàng) + cross-entropy trung bình.
    Trả về (loss, P).\"\"\"
    # TODO
    raise NotImplementedError

def backward(P, y, cache, params):
    \"\"\"Trả về dict gradient {'W1','b1','W2','b2'} của loss trung bình.
    Gợi ý: dZ2 = (P − Y)/N; kiểm shape để đặt chuyển vị.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def forward(X, params):
    W1, b1, W2, b2 = params
    Z1 = X @ W1 + b1
    H = np.maximum(0, Z1)
    Z2 = H @ W2 + b2
    return Z2, (X, Z1, H)

def softmax_ce(Z, y):
    Zs = Z - Z.max(axis=1, keepdims=True)           # ổn định số
    P = np.exp(Zs); P /= P.sum(axis=1, keepdims=True)
    loss = -np.log(P[np.arange(len(y)), y]).mean()
    return loss, P

def backward(P, y, cache, params):
    W1, b1, W2, b2 = params
    X, Z1, H = cache
    N = len(y)
    dZ2 = P.copy(); dZ2[np.arange(N), y] -= 1; dZ2 /= N   # (N, C)
    dW2 = H.T @ dZ2                                      # (H, C)
    db2 = dZ2.sum(axis=0)
    dH = dZ2 @ W2.T                                      # (N, H)
    dZ1 = dH * (Z1 > 0)                                  # ReLU chặn gradient khi z ≤ 0
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}""")

MLP_CHECK = """# (a) Ví dụ giải mẫu 6.4 — một mẫu, quy ước batch nên W1 ở đây là chuyển vị của W₁ trong sách
W1 = np.array([[0.5, -1.0], [1.0, 0.5], [-0.5, 0.5]]).T
b1 = np.array([0.0, 0.0, 1.0])
W2 = np.array([[1.0, -1.0, 0.5], [0.5, 1.0, -1.0]]).T
b2 = np.zeros(2)
p6 = (W1, b1, W2, b2)
Z, cache = forward(np.array([[1.0, 2.0]]), p6)
L, P = softmax_ce(Z, np.array([0]))
g = backward(P, np.array([0]), cache, p6)
print("L =", round(L, 4), "| dW₁ (theo sách) =\\n", g["W1"].T)
assert abs(L - 1.9102) < 1e-4
assert np.allclose(g["W1"].T, [[0, 0], [1.7039, 3.4078], [-1.2779, -2.5559]], atol=1e-4)

# (b) Gradient check sai phân trung tâm trên mạng ngẫu nhiên, float64
def rel_err(a, b):
    return np.abs(a - b).max() / max(np.abs(a).max(), np.abs(b).max(), 1e-12)

r = np.random.default_rng(0)
Xg, yg = r.standard_normal((5, 4)), r.integers(0, 3, 5)
pg = [r.standard_normal((4, 6)), r.standard_normal(6) * 0.1, r.standard_normal((6, 3)), r.standard_normal(3) * 0.1]
Pg = softmax_ce(forward(Xg, pg)[0], yg)[1]
ga = backward(Pg, yg, forward(Xg, pg)[1], pg)
worst = 0.0
for k, name in enumerate(["W1", "b1", "W2", "b2"]):
    num = np.zeros_like(pg[k]); delta = 1e-5
    for idx in np.ndindex(pg[k].shape):
        old = pg[k][idx]
        pg[k][idx] = old + delta; lp = softmax_ce(forward(Xg, pg)[0], yg)[0]
        pg[k][idx] = old - delta; lm = softmax_ce(forward(Xg, pg)[0], yg)[0]
        pg[k][idx] = old
        num[idx] = (lp - lm) / (2 * delta)
    e = rel_err(num, ga[name]); worst = max(worst, e)
    print(f"  {name}: sai số tương đối {e:.1e}")
assert worst < 1e-6, "backward chưa đúng"
print("Lab 1 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 2
TRAIN_NP = ("""def train_numpy(Xtr, ytr, H=128, lr=0.1, epochs=8, batch=64, seed=0):
    \"\"\"SGD mini-batch thuần NumPy cho MLP d–H–10, init He cho W1, xáo trộn mỗi epoch.
    Trả về params đã huấn luyện.\"\"\"
    # TODO
    raise NotImplementedError

class MLP(nn.Module):
    \"\"\"Linear(d, H) → ReLU → Linear(H, C).\"\"\"
    def __init__(self, d=20, H=128, C=10):
        super().__init__()
        # TODO
        raise NotImplementedError
    def forward(self, x):
        # TODO
        raise NotImplementedError

def train_torch(model, Xtr, ytr, lr=1e-3, epochs=8, batch=64):
    \"\"\"Vòng lặp chuẩn: train() → zero_grad → forward + loss → backward → step. Dùng Adam và DataLoader(shuffle=True).\"\"\"
    # TODO
    raise NotImplementedError

def evaluate(model, X, y):
    \"\"\"model.eval() + torch.no_grad(); trả về (loss, acc) dạng float.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def train_numpy(Xtr, ytr, H=128, lr=0.1, epochs=8, batch=64, seed=0):
    r = np.random.default_rng(seed)
    d, C = Xtr.shape[1], int(ytr.max()) + 1
    params = [r.standard_normal((d, H)) * np.sqrt(2 / d), np.zeros(H),
              r.standard_normal((H, C)) * np.sqrt(1 / H), np.zeros(C)]
    for ep in range(epochs):
        order = r.permutation(len(Xtr))                  # xáo trộn X và y CÙNG một hoán vị
        for s in range(0, len(Xtr), batch):
            idx = order[s:s + batch]
            Z, cache = forward(Xtr[idx], params)
            _, P = softmax_ce(Z, ytr[idx])
            g = backward(P, ytr[idx], cache, params)
            for k, name in enumerate(["W1", "b1", "W2", "b2"]):
                params[k] -= lr * g[name]
    return params

class MLP(nn.Module):
    def __init__(self, d=20, H=128, C=10):
        super().__init__()
        self.fc1 = nn.Linear(d, H)
        self.fc2 = nn.Linear(H, C)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))             # trả LOGIT thô

def train_torch(model, Xtr, ytr, lr=1e-3, epochs=8, batch=64):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True)
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    return model

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        out = model(X)
        return F.cross_entropy(out, y).item(), (out.argmax(1) == y).float().mean().item()""")

TRAIN_CHECK = """Xtr_np, ytr_np, Xva_np, yva_np = Xtr.double().numpy(), ytr.numpy(), Xva.double().numpy(), yva.numpy()
pn = train_numpy(Xtr_np, ytr_np)
Zv = forward(Xva_np, pn)[0]
acc_np = (Zv.argmax(1) == yva_np).mean()
print(f"NumPy SGD lr 0.1, 8 epoch: loss val {softmax_ce(Zv, yva_np)[0]:.4f}, acc val {acc_np:.4f}")

torch.manual_seed(0)
model = train_torch(MLP(), Xtr, ytr)
lv, av = evaluate(model, Xva, yva)
print(f"PyTorch Adam lr 1e-3, 8 epoch: loss val {lv:.4f}, acc val {av:.4f}")

# Cùng trọng số ⇒ cùng loss, cùng gradient (đối chiếu NumPy với autograd)
m = MLP().double()
with torch.no_grad():
    m.fc1.weight.copy_(torch.tensor(pn[0].T)); m.fc1.bias.copy_(torch.tensor(pn[1]))
    m.fc2.weight.copy_(torch.tensor(pn[2].T)); m.fc2.bias.copy_(torch.tensor(pn[3]))
xb, yb = Xtr[:64].double(), ytr[:64]
lt = F.cross_entropy(m(xb), yb); lt.backward()
Zb, cb = forward(xb.numpy(), pn); ln_, Pb = softmax_ce(Zb, yb.numpy())
gb = backward(Pb, yb.numpy(), cb, pn)
print(f"loss torch {lt.item():.10f} | numpy {ln_:.10f}")
assert acc_np > 0.7 and av > 0.7
assert abs(lt.item() - ln_) < 1e-10 and np.allclose(m.fc1.weight.grad.numpy().T, gb["W1"], atol=1e-10)
print("Lab 2 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 3
INIT = ("""def activation_stds(init, depth=20, width=256, n=512, seed=0):
    \"\"\"Đẩy x ~ N(0, 1) qua `depth` lớp nn.Linear(width, width) + ReLU.
    init ∈ {'std0.01', 'default', 'xavier', 'he'} — 'default' giữ nguyên init của nn.Linear;
    các init khác đặt bias = 0. Trả về list std của activation sau từng lớp.\"\"\"
    # TODO
    raise NotImplementedError

def zero_init_grad():
    \"\"\"Mạng 4–5–3, khởi tạo MỌI trọng số bằng 0.3 và bias 0, một batch ngẫu nhiên.
    Trả về gradient của trọng số lớp đầu (shape 5×4 theo quy ước nn.Linear).\"\"\"
    # TODO
    raise NotImplementedError""",
"""def activation_stds(init, depth=20, width=256, n=512, seed=0):
    torch.manual_seed(seed)
    h = torch.randn(n, width)
    out = []
    with torch.no_grad():
        for _ in range(depth):
            lin = nn.Linear(width, width)
            if init != "default":
                nn.init.zeros_(lin.bias)
                if init == "std0.01":
                    nn.init.normal_(lin.weight, std=0.01)
                elif init == "xavier":
                    nn.init.xavier_normal_(lin.weight)
                elif init == "he":
                    nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            h = F.relu(lin(h))
            out.append(h.std().item())
    return out

def zero_init_grad():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
    with torch.no_grad():
        net[0].weight.fill_(0.3); net[0].bias.zero_()
        net[2].weight.copy_(torch.tensor([[0.1], [0.2], [0.3]]).expand(3, 5)); net[2].bias.zero_()
    x, y = torch.randn(8, 4), torch.randint(0, 3, (8,))
    F.cross_entropy(net(x), y).backward()
    return net[0].weight.grad""")

INIT_CHECK = """res = {k: activation_stds(k) for k in ["std0.01", "default", "xavier", "he"]}
for k, v in res.items():
    print(f"{k:8s} std sau lớp 1 / 10 / 20: {v[0]:.3g} / {v[9]:.3g} / {v[19]:.3g}")
# init mặc định của nn.Linear nhưng đặt bias = 0 (cùng seed, cùng trọng số với 'default')
torch.manual_seed(0); h = torch.randn(512, 256)
with torch.no_grad():
    for _ in range(20):
        lin = nn.Linear(256, 256); nn.init.zeros_(lin.bias); h = F.relu(lin(h))
print(f"default, bỏ bias: std sau lớp 20 = {h.std().item():.2e}")
assert h.std().item() < 1e-6, "bỏ bias thì tín hiệu từ đầu vào phải tắt"
gz = zero_init_grad()
print("gradient lớp đầu khi init hằng số:\\n", gz.numpy())
assert 0.5 < res["he"][-1] < 1.5, "He phải giữ std ổn định"
assert res["std0.01"][-1] < 1e-6, "std 0.01 phải làm tín hiệu tắt"
assert res["xavier"][-1] < 0.1 * res["he"][-1], "Xavier co dần qua ReLU"
assert gz.abs().max() > 1e-4, "gradient phải khác 0"
assert torch.allclose(gz, gz[0].expand_as(gz)), "mọi neuron phải nhận cùng gradient"
print("Lab 3 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 4
NORM = ("""def batchnorm_forward(x, gamma, beta, state, train, momentum=0.1, eps=1e-5):
    \"\"\"x: (N, F). state = {'mean': (F,), 'var': (F,)} được cập nhật tại chỗ khi train.
    Train: chuẩn hóa bằng mean/phương sai LỆCH của batch; running_var cập nhật bằng phương sai KHÔNG LỆCH.
    Eval: dùng running stats.\"\"\"
    # TODO
    raise NotImplementedError

def layernorm_forward(x, gamma, beta, eps=1e-5):
    \"\"\"Chuẩn hóa theo trục đặc trưng của TỪNG mẫu (giống nhau ở train và eval).\"\"\"
    # TODO
    raise NotImplementedError

def dropout_forward(x, p, train, gen=None):
    \"\"\"Inverted dropout: train thì giữ mỗi phần tử với xác suất 1 − p và chia cho (1 − p); eval trả x.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def batchnorm_forward(x, gamma, beta, state, train, momentum=0.1, eps=1e-5):
    if train:
        m = x.mean(0)
        v = x.var(0, unbiased=False)                       # dùng để chuẩn hóa
        n = x.shape[0]
        state["mean"] = (1 - momentum) * state["mean"] + momentum * m
        state["var"] = (1 - momentum) * state["var"] + momentum * v * n / (n - 1)   # bản không lệch
    else:
        m, v = state["mean"], state["var"]
    return gamma * (x - m) / torch.sqrt(v + eps) + beta

def layernorm_forward(x, gamma, beta, eps=1e-5):
    m = x.mean(-1, keepdim=True)
    v = x.var(-1, unbiased=False, keepdim=True)
    return gamma * (x - m) / torch.sqrt(v + eps) + beta

def dropout_forward(x, p, train, gen=None):
    if not train or p == 0:
        return x
    mask = (torch.rand(x.shape, generator=gen) > p).to(x.dtype)
    return x * mask / (1 - p)""")

NORM_CHECK = """# (a) Ví dụ giải mẫu 6.9: một đặc trưng, batch (1, 2, 3, 6)
st = {"mean": torch.zeros(1), "var": torch.ones(1)}
xb = torch.tensor([[1.0], [2.0], [3.0], [6.0]])
yb = batchnorm_forward(xb, 1.0, 0.0, st, train=True)
print("x̂ =", yb.flatten().numpy(), "| running_mean =", round(st["mean"].item(), 4), "| running_var =", round(st["var"].item(), 4))
assert abs(st["mean"].item() - 0.3) < 1e-6 and abs(st["var"].item() - 1.3667) < 1e-4

# (b) So với nn.BatchNorm1d qua 3 bước train rồi eval
torch.manual_seed(0)
bn = nn.BatchNorm1d(6); st = {"mean": torch.zeros(6), "var": torch.ones(6)}
with torch.no_grad():
    bn.weight.uniform_(0.5, 1.5); bn.bias.uniform_(-0.5, 0.5)
    for _ in range(3):
        x = torch.randn(16, 6) * 3 + 1
        assert torch.allclose(bn(x), batchnorm_forward(x, bn.weight, bn.bias, st, True), atol=1e-5)
    bn.eval(); x = torch.randn(5, 6)
    assert torch.allclose(bn(x), batchnorm_forward(x, bn.weight, bn.bias, st, False), atol=1e-5)
    assert torch.allclose(bn.running_var, st["var"], atol=1e-5)

# (c) LayerNorm: ví dụ (2, 4, 6, 8) và so với nn.LayerNorm
print("LN(2, 4, 6, 8) =", layernorm_forward(torch.tensor([2.0, 4, 6, 8]), 1.0, 0.0).numpy())
ln = nn.LayerNorm(6); x = torch.randn(4, 6)
with torch.no_grad():
    assert torch.allclose(ln(x), layernorm_forward(x, ln.weight, ln.bias), atol=1e-5)

# (d) Inverted dropout giữ kỳ vọng; eval là hàm đồng nhất
gen = torch.Generator().manual_seed(0)
x = torch.ones(200_000)
yd = dropout_forward(x, 0.5, True, gen)
print(f"dropout p=0.5: trung bình {yd.mean():.4f}, tỉ lệ bị tắt {(yd == 0).float().mean():.4f}")
assert abs(yd.mean().item() - 1) < 0.01 and torch.equal(dropout_forward(x, 0.5, False), x)
print("Lab 4 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 5
DEBUG = ("""def make_deep(seed=1):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(20, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 10))

def initial_loss(model, X, y):
    \"\"\"Loss trước khi huấn luyện (no_grad). So với ln C.\"\"\"
    # TODO
    raise NotImplementedError

def overfit_one_batch(model, xb, yb, steps=200, lr=1e-2):
    \"\"\"Adam trên đúng một batch, không regularization. Trả về list loss từng bước.\"\"\"
    # TODO
    raise NotImplementedError

def grad_norm(model):
    \"\"\"Chuẩn L2 toàn cục của mọi gradient (gọi sau backward).\"\"\"
    # TODO
    raise NotImplementedError

def lr_sweep(lrs, epochs=3):
    \"\"\"Với mỗi lr: make_deep(), Adam, `epochs` epoch, trả về dict lr → (loss val, acc val).\"\"\"
    # TODO
    raise NotImplementedError""",
"""def make_deep(seed=1):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(20, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 10))

def initial_loss(model, X, y):
    model.eval()
    with torch.no_grad():
        return F.cross_entropy(model(X), y).item()

def overfit_one_batch(model, xb, yb, steps=200, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward(); opt.step()
        hist.append(loss.item())
    return hist

def grad_norm(model):
    return torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None)).item()

def lr_sweep(lrs, epochs=3):
    res = {}
    for lr in lrs:
        torch.manual_seed(0)
        res[lr] = evaluate(train_torch(make_deep(), Xtr, ytr, lr=lr, epochs=epochs), Xva, yva)
    return res""")

DEBUG_CHECK = """L0 = initial_loss(make_deep(), Xtr, ytr)
print(f"loss ban đầu {L0:.4f}  (ln 10 = {np.log(10):.4f})")

torch.manual_seed(2)
hist = overfit_one_batch(nn.Sequential(nn.Linear(20, 128), nn.ReLU(), nn.Linear(128, 10)), Xtr[:32], ytr[:32])
print(f"overfit 32 mẫu: bước 0 {hist[0]:.3f} → bước 50 {hist[50]:.2e} → bước 199 {hist[-1]:.2e}")

m = make_deep(); F.cross_entropy(m(Xtr[:64]), ytr[:64]).backward()
print(f"grad norm lúc khởi tạo {grad_norm(m):.4f}")

sweep = lr_sweep([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
for lr, (l, a) in sweep.items():
    print(f"  lr {lr:g}: loss val {l:.3f}, acc val {a:.3f}")
best = min(sweep, key=lambda k: sweep[k][0])

# tỉ lệ ‖Δw‖/‖w‖ của lớp đầu sau đúng một bước, cùng lr 1e-3 cho hai optimizer
ratios = {}
for name in ["Adam", "SGD"]:
    m = make_deep(); w0 = m[0].weight.detach().clone()
    opt = (torch.optim.Adam if name == "Adam" else torch.optim.SGD)(m.parameters(), lr=1e-3)
    opt.zero_grad(); F.cross_entropy(m(Xtr[:64]), ytr[:64]).backward(); opt.step()
    ratios[name] = ((m[0].weight - w0).norm() / w0.norm()).item()
print("‖Δw‖/‖w‖ sau một bước, lr 1e-3:", {k: f"{v:.1e}" for k, v in ratios.items()})
assert abs(L0 - np.log(10)) < 0.15
assert hist[-1] < 1e-2
assert best in (1e-3, 1e-2)
assert ratios["Adam"] > 50 * ratios["SGD"]
print("Lab 5 ĐẠT ✔")"""

# ---------------------------------------------------------------- Săn bug
BUG_DATA = """def make_bn_model():
    torch.manual_seed(1)
    return nn.Sequential(nn.Linear(20, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 10))

def train_buggy(Xtr_raw, ytr, Xva_raw, yva, epochs=8):
    mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0)
    xtr = (Xtr_raw - mu) / sd
    xva = (Xva_raw - Xva_raw.mean(0)) / Xva_raw.std(0)
    model = make_bn_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    dl = DataLoader(TensorDataset(xtr, ytr), batch_size=64, shuffle=True)
    total_loss = 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            probs = F.softmax(model(xb), dim=1)
            loss = F.cross_entropy(probs, yb)
            loss.backward()
            opt.step()
            total_loss += loss
    with torch.no_grad():
        out = F.softmax(model(xva), dim=1)
        val_loss = F.cross_entropy(out, yva).item()
        val_acc = (out.argmax(1) == yva).float().mean().item()
    return model, xva, val_loss, val_acc, total_loss

torch.manual_seed(0)
mb, xva_b, lb, ab, tl = train_buggy(Xtr_raw, ytr, Xva_raw, yva)
with torch.no_grad():
    d = (mb(xva_b[:8]) - mb(xva_b[:8])).abs().max().item()
print(f"bản lỗi: loss val {lb:.4f}, acc val {ab:.4f}, chênh lệch hai lần gọi cùng input {d:.3f}")"""

BUGFIX = ("""def train_fixed(Xtr_raw, ytr, Xva_raw, yva, epochs=8):
    \"\"\"Sửa MỌI lỗi của train_buggy. Trả về cùng bộ (model, xva, val_loss, val_acc, total_loss).\"\"\"
    # TODO
    raise NotImplementedError""",
"""def train_fixed(Xtr_raw, ytr, Xva_raw, yva, epochs=8):
    mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0)
    xtr = (Xtr_raw - mu) / sd
    xva = (Xva_raw - mu) / sd                            # sửa lỗi 4: dùng thống kê của TRAIN
    model = make_bn_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    dl = DataLoader(TensorDataset(xtr, ytr), batch_size=64, shuffle=True)
    total_loss = 0.0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()                              # sửa lỗi 1
            loss = F.cross_entropy(model(xb), yb)        # sửa lỗi 2: logit thô
            loss.backward()
            opt.step()
            total_loss += loss.item()                    # sửa lỗi 5: số Python, không giữ đồ thị
    model.eval()                                         # sửa lỗi 3
    with torch.no_grad():
        out = model(xva)
        val_loss = F.cross_entropy(out, yva).item()
        val_acc = (out.argmax(1) == yva).float().mean().item()
    return model, xva, val_loss, val_acc, total_loss""")

BUG_CHECK = """epochs_ = 8
torch.manual_seed(0)
mf, xva_f, lf, af, tlf = train_fixed(Xtr_raw, ytr, Xva_raw, yva)
with torch.no_grad():
    d = (mf(xva_f[:8]) - mf(xva_f[:8])).abs().max().item()
print(f"bản sửa: loss val {lf:.4f}, acc val {af:.4f}, chênh lệch hai lần gọi {d:.3f}")
assert lf < 0.8 and af > 0.76 and d == 0.0 and isinstance(tlf, float)
assert xva_f.mean(0).abs().max() > 1e-4, "kiểm tra lại cách tạo xva"
assert tlf / (epochs_ * 63) < 1.4, "loss huấn luyện trung bình còn quá cao"   # 4000 mẫu / batch 64 = 63 bước mỗi epoch
print("Săn bug ĐẠT ✔")"""

ABLATION = """# Tách từng lỗi: bật riêng mỗi lỗi trên bản đúng (cùng seed) để thấy lỗi nào gây triệu chứng nào.
def run(bugs, epochs=8):
    xtr = (Xtr_raw - mu) / sd
    xva = (Xva_raw - Xva_raw.mean(0)) / Xva_raw.std(0) if "valstats" in bugs else (Xva_raw - mu) / sd
    if "nonorm" in bugs:                                   # bỏ hẳn chuẩn hóa đầu vào
        xtr, xva = Xtr_raw, Xva_raw
    torch.manual_seed(0)
    ytr_ = ytr[torch.randperm(len(ytr))] if "labelshuffle" in bugs else ytr   # xáo y độc lập với X
    model = make_bn_model(); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    dl = DataLoader(TensorDataset(xtr, ytr_), batch_size=64, shuffle=True)
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            if "nozero" not in bugs: opt.zero_grad()
            out = model(xb)
            if "softmax" in bugs: out = F.softmax(out, 1)
            F.cross_entropy(out, yb).backward(); opt.step()
    if "noeval" not in bugs: model.eval()
    with torch.no_grad():
        out = model(xva)
        if "softmax" in bugs: out = F.softmax(out, 1)
        return F.cross_entropy(out, yva).item(), (out.argmax(1) == yva).float().mean().item()

table = {}
for bugs in [(), ("nozero",), ("softmax",), ("noeval",), ("valstats",), ("nonorm",), ("labelshuffle",),
             ("nozero", "softmax", "noeval"), ("nozero", "softmax", "noeval", "valstats")]:
    table[bugs] = run(set(bugs))
    print(f"  {'+'.join(bugs) or 'không lỗi':38s} loss val {table[bugs][0]:.4f}  acc val {table[bugs][1]:.4f}")
print("CE nhỏ nhất khi đưa xác suất vào cross_entropy (10 lớp):",
      round(F.cross_entropy(torch.eye(10)[:1], torch.tensor([0])).item(), 4))
# grad norm lúc khởi tạo: logit thô so với softmax trước CE (batch 64 đầu tiên)
gnorms = {}
for sm in (False, True):
    m = make_bn_model(); m.train()
    out = m(Xtr[:64]); out = F.softmax(out, 1) if sm else out
    F.cross_entropy(out, ytr[:64]).backward()
    gnorms[sm] = torch.sqrt(sum((p.grad ** 2).sum() for p in m.parameters())).item()
print(f"grad norm lúc khởi tạo: logit thô {gnorms[False]:.3f}, softmax trước CE {gnorms[True]:.3f}")
assert table[("softmax",)][0] > 1.4612 and table[()][0] < 0.8
assert gnorms[True] < 0.2 * gnorms[False]
assert table[("labelshuffle",)][1] < 0.15
assert tl.requires_grad   # total_loss của bản lỗi còn giữ grad_fn (lỗi 5)
print(f"Tổng thời gian notebook: {time.time() - T0:.1f} s")
print("Phân tích lỗi ĐẠT ✔")"""


def build(solution: bool):
    pick = (lambda pair: pair[1]) if solution else (lambda pair: pair[0])
    nb = nbf.v4.new_notebook()
    head = "Lời giải" if solution else "Bài lab"
    nb.cells = [
        md(f"# Chương 6 — {head}: Mạng nơ-ron từ con số 0\n\n"
           + ("Bản lời giải đã chạy kiểm (NumPy + PyTorch CPU, khoảng 40 giây khi máy rảnh). Chỉ mở sau khi đã tự làm `lab.ipynb`."
              if solution else
              "Viết code vào các ô có `# TODO`, rồi chạy ô kiểm tra ngay bên dưới. "
              "Ô kiểm tra in **ĐẠT ✔** là xong. Bí quá 20 phút mới mở `loigiai.ipynb`. "
              "Chỉ cần PyTorch bản CPU; toàn notebook chạy khoảng 40 giây khi máy rảnh, lâu hơn khi máy bận.")),
        md("## Dữ liệu chung\n10 cụm Gauss chồng lấn trong ℝ²⁰, 4000 mẫu train và 2000 mẫu val. "
           "Đặc trưng bị nhân 5 và cộng 3 để bước chuẩn hóa có ý nghĩa. Thống kê chuẩn hóa chỉ lấy từ train."),
        code(SETUP),
        md("## Lab 1 — MLP bằng NumPy: forward, backward, gradient check\n"
           "Quy ước batch: X ∈ ℝᴺˣᵈ, Z₁ = XW₁ + b₁. Đối chiếu với ví dụ giải mẫu 6.3 rồi kiểm bằng sai phân trung tâm."),
        code(pick(MLP_NP)), code(MLP_CHECK),
        md("## Lab 2 — Huấn luyện bằng NumPy, rồi bằng PyTorch\n"
           "Viết vòng SGD NumPy, sau đó viết lại bằng `nn.Module` + `DataLoader` + Adam. "
           "Cuối cùng chép trọng số NumPy vào PyTorch: loss và gradient phải trùng tới sai số máy."),
        code(pick(TRAIN_NP)), code(TRAIN_CHECK),
        md("## Lab 3 — Khởi tạo trọng số\nĐo std activation qua 20 lớp ReLU rộng 256 với bốn kiểu init, "
           "và xem chuyện gì xảy ra khi mọi trọng số bằng nhau."),
        code(pick(INIT)), code(INIT_CHECK),
        md("## Lab 4 — BatchNorm, LayerNorm, dropout viết tay\nSo với `nn.BatchNorm1d`, `nn.LayerNorm` bằng `allclose`. "
           "Chú ý: chuẩn hóa dùng phương sai lệch, running_var lưu phương sai không lệch."),
        code(pick(NORM)), code(NORM_CHECK),
        md("## Lab 5 — Hộp công cụ debug\nKiểm loss ban đầu, overfit một batch, grad norm, quét learning rate, "
           "và so tỉ lệ ‖Δw‖/‖w‖ của Adam với SGD ở cùng lr."),
        code(pick(DEBUG)), code(DEBUG_CHECK),
        md("## Săn bug\n`train_buggy` chạy không báo lỗi. Triệu chứng: acc val khoảng 0.71, loss val không xuống dưới 1.7, "
           "và gọi model hai lần trên cùng input cho kết quả khác nhau. Tìm cả **năm** lỗi rồi viết `train_fixed`."),
        code(BUG_DATA), code(pick(BUGFIX)), code(BUG_CHECK),
    ]
    if solution:   # bảng tách lỗi nêu tên từng lỗi nên chỉ nằm ở bản lời giải
        nb.cells += [
            md("## Phân tích từng lỗi\nBật riêng từng lỗi để thấy lỗi nào gây triệu chứng nào."),
            code(ABLATION),
        ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    name = "loigiai.ipynb" if solution else "lab.ipynb"
    nbf.write(nb, OUT / name)
    return OUT / name


if __name__ == "__main__":
    print(build(False), build(True))
