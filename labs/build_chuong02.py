"""Sinh hai notebook cho lab chương 2: lab.ipynb (có TODO) và loigiai.ipynb (lời giải).

Chạy lại script này nếu muốn tạo lại notebook gốc:  python labs/build_chuong02.py
"""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "chuong02"
OUT.mkdir(parents=True, exist_ok=True)

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

SETUP = """import math
import numpy as np
np.set_printoptions(precision=6, suppress=True)"""

# ---- Lab 1: gradient số -------------------------------------------------------
NUMGRAD = ("""def numerical_grad(f, x, h=1e-5):
    \"\"\"Gradient của hàm vô hướng f tại mảng x (shape bất kỳ) bằng sai phân trung tâm.
    Trả về mảng cùng shape với x. Gợi ý: np.nditer(x, flags=['multi_index']);
    nhớ trả x[idx] về giá trị cũ sau mỗi lần nhiễu.\"\"\"
    # TODO
    raise NotImplementedError


def rel_error(a, b):
    \"\"\"max|a − b| / max(max|a|, max|b|, 1e-12).\"\"\"
    # TODO
    raise NotImplementedError""",
"""def numerical_grad(f, x, h=1e-5):
    g = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + h; fp = f(x)
        x[i] = old - h; fm = f(x)
        x[i] = old                                   # trả lại giá trị cũ
        g[i] = (fp - fm) / (2 * h)
    return g


def rel_error(a, b):
    return np.max(np.abs(a - b)) / max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-12)""")

NUMGRAD_CHECK = """rng = np.random.default_rng(0)                       # seed riêng: chạy lại ô này số vẫn y nguyên
A = rng.standard_normal((5, 5))                      # KHÔNG đối xứng
x = rng.standard_normal(5)
g_num = numerical_grad(lambda v: v @ A @ v, x.copy())
e_right = rel_error((A + A.T) @ x, g_num)
e_wrong = rel_error(2 * A @ x, g_num)
print(f"(A + Aᵀ)x: {e_right:.1e}   |   2Ax (sai): {e_wrong:.1e}")

# X phải KHÔNG đối xứng, nếu không X⁻ᵀ = X⁻¹ và phép kiểm không phân biệt được lỗi quên chuyển vị
X = rng.standard_normal((4, 4)) + 4 * np.eye(4)
assert np.linalg.det(X) > 0 and not np.allclose(X, X.T), "cần X không đối xứng, det > 0"
g_num = numerical_grad(lambda Z: np.log(np.linalg.det(Z)), X.copy())
e_logdet = rel_error(np.linalg.inv(X).T, g_num)
e_notrans = rel_error(np.linalg.inv(X), g_num)       # đáp án sai: quên chuyển vị
print(f"∂ log det X / ∂X = X⁻ᵀ: {e_logdet:.1e}   |   X⁻¹ (quên chuyển vị, sai): {e_notrans:.1e}")
assert e_right < 1e-7 and e_logdet < 1e-7, "gradient số chưa khớp công thức"
assert e_wrong > 1e-2, "công thức sai 2Ax phải bị phát hiện"
assert e_notrans > 1e-2, "phép kiểm log det phải bắt được lỗi quên chuyển vị"
print("Lab 1 ĐẠT ✔")"""

# ---- Lab 2: quét h --------------------------------------------------------------
SWEEP = ("""def fd_errors(dtype, hs, x0=1.0):
    \"\"\"Với f = sin tại x0, trả về hai list sai số tuyệt đối so với cos(x0):
    (sai phân trung tâm, sai phân tiến), tính toàn bộ trong kiểu dtype.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def fd_errors(dtype, hs, x0=1.0):
    x = dtype(x0)
    exact = math.cos(x0)
    cen, fwd = [], []
    for h in hs:
        h = dtype(h)
        c = (np.sin(x + h) - np.sin(x - h)) / (dtype(2) * h)
        f = (np.sin(x + h) - np.sin(x)) / h
        cen.append(abs(float(c) - exact)); fwd.append(abs(float(f) - exact))
    return cen, fwd""")

SWEEP_RUN = """hs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
res = {}
for dt in (np.float64, np.float32):
    cen, fwd = fd_errors(dt, hs)
    res[dt] = cen
    print(dt.__name__)
    for h, c, f in zip(hs, cen, fwd):
        print(f"  h = {h:.0e}: trung tâm {c:.1e}   tiến {f:.1e}")
best64 = hs[int(np.argmin(res[np.float64]))]
best32 = hs[int(np.argmin(res[np.float32]))]
print("h tốt nhất: float64", best64, "| float32", best32)
print("lý thuyết ε^(1/3):", np.finfo(np.float64).eps ** (1/3), np.finfo(np.float32).eps ** (1/3))
assert 1e-6 <= best64 <= 1e-4 and min(res[np.float64]) < 1e-10
assert res[np.float32][hs.index(1e-7)] > 1e-2, "float32 với h nhỏ phải sai nhiều"
print("Lab 2 ĐẠT ✔")"""

# ---- Lab 3: mini autograd -----------------------------------------------------
VALUE = ("""class Value:
    \"\"\"Một số vô hướng biết gradient của mình. Cần: __add__, __mul__ (nhận cả số thường),
    sigmoid(), backward(). Mỗi phép toán tạo nút mới, lưu các nút cha và một hàm
    _backward cộng dồn (+=) gradient vào cha. backward(): sắp xếp topo rồi đi ngược.\"\"\"
    def __init__(self, data, parents=()):
        self.data, self.grad = float(data), 0.0
        self._parents, self._backward = parents, lambda: None

    # TODO: __add__, __radd__, __mul__, __rmul__, sigmoid, backward
""",
"""class Value:
    def __init__(self, data, parents=()):
        self.data, self.grad = float(data), 0.0
        self._parents, self._backward = parents, lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad                    # cổng cộng: chia đều
            other.grad += out.grad
        out._backward = _backward
        return out
    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad       # cổng nhân: đổi chéo
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    __rmul__ = __mul__

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,))
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        order, seen = [], set()
        def visit(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v._parents:
                    visit(p)
                order.append(v)
        visit(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()""")

VALUE_CHECK = """# Ví dụ 2.7: f = (x + y)·y tại x = 3, y = 2
x, y = Value(3.0), Value(2.0)
f = (x + y) * y
f.backward()
print("f =", f.data, "  ∂f/∂x =", x.grad, "  ∂f/∂y =", y.grad)
assert f.data == 10 and x.grad == 2 and y.grad == 7, "fan-out phải cộng dồn"

# Ví dụ 2.6: một neuron sigmoid
w1, w2, b, x1, x2 = Value(0.5), Value(-1.0), Value(0.2), Value(2.0), Value(1.0)
s = (w1 * x1 + w2 * x2 + b).sigmoid()
s.backward()
print(f"s = {s.data:.6f}  ∂s/∂w₁ = {w1.grad:.6f}  ∂s/∂x₁ = {x1.grad:.6f}")
assert np.isclose(s.data, 0.549834, atol=1e-6) and np.isclose(w1.grad, 0.495033, atol=1e-6)
assert np.isclose(x1.grad, 0.123758, atol=1e-6)

# So với gradient số
def neuron(p):
    return 1 / (1 + np.exp(-(p[0] * p[3] + p[1] * p[4] + p[2])))
g_num = numerical_grad(neuron, np.array([0.5, -1.0, 0.2, 2.0, 1.0]))
g_ad = np.array([w1.grad, w2.grad, b.grad, x1.grad, x2.grad])
print("rel_error so với sai phân:", f"{rel_error(g_ad, g_num):.1e}")
assert rel_error(g_ad, g_num) < 1e-8
print("Lab 3 ĐẠT ✔")"""

# ---- Lab 4: JVP vs VJP --------------------------------------------------------
JV = ("""rng = np.random.default_rng(4)
Ws = [rng.standard_normal((5, 6)) * 0.5,       # ℝ⁶ → ℝ⁵ → ℝ⁴ → ℝ³
      rng.standard_normal((4, 5)) * 0.5,
      rng.standard_normal((3, 4)) * 0.5]
calls = {"jvp": 0, "vjp": 0}

def f(x):
    for W in Ws:
        x = np.tanh(W @ x)
    return x

def jvp(x, v):
    \"\"\"Forward mode: trả về (f(x), J v), đẩy cặp (giá trị, tiếp tuyến) qua từng lớp.
    Đạo hàm của tanh(a) là 1 − tanh(a)².  Nhớ calls['jvp'] += 1.\"\"\"
    # TODO
    raise NotImplementedError

def vjp(x, u):
    \"\"\"Reverse mode: forward lưu các activation, rồi đi ngược để trả về uᵀJ (vector shape (6,)).
    Nhớ calls['vjp'] += 1.\"\"\"
    # TODO
    raise NotImplementedError""",
"""rng = np.random.default_rng(4)
Ws = [rng.standard_normal((5, 6)) * 0.5,       # ℝ⁶ → ℝ⁵ → ℝ⁴ → ℝ³
      rng.standard_normal((4, 5)) * 0.5,
      rng.standard_normal((3, 4)) * 0.5]
calls = {"jvp": 0, "vjp": 0}

def f(x):
    for W in Ws:
        x = np.tanh(W @ x)
    return x

def jvp(x, v):
    calls["jvp"] += 1
    for W in Ws:
        y = np.tanh(W @ x)
        v = (1 - y**2) * (W @ v)                 # tiếp tuyến đi cùng chiều với giá trị
        x = y
    return x, v

def vjp(x, u):
    calls["vjp"] += 1
    acts = []
    for W in Ws:                                 # forward: phải LƯU activation
        x = np.tanh(W @ x)
        acts.append(x)
    for W, y in zip(reversed(Ws), reversed(acts)):
        u = W.T @ ((1 - y**2) * u)               # đi ngược: nhân với Jᵀ từng lớp
    return u""")

JV_CHECK = """rng = np.random.default_rng(40)
calls["jvp"] = calls["vjp"] = 0                      # đặt lại bộ đếm để chạy lại ô này vẫn đúng
x0 = rng.standard_normal(6)
n, m = 6, 3
J_fwd = np.stack([jvp(x0, e)[1] for e in np.eye(n)], axis=1)   # n lượt, mỗi lượt 1 cột
J_rev = np.stack([vjp(x0, e) for e in np.eye(m)], axis=0)      # m lượt, mỗi lượt 1 hàng
J_num = np.stack([numerical_grad(lambda z, i=i: f(z)[i], x0.copy()) for i in range(m)])
print("số lượt:", calls, "| shape J:", J_fwd.shape)
print("rel_error(J_fwd, J_rev) =", f"{rel_error(J_fwd, J_rev):.1e}",
      "| so với sai phân:", f"{rel_error(J_fwd, J_num):.1e}")
u, v = rng.standard_normal(m), rng.standard_normal(n)
lhs, rhs = u @ jvp(x0, v)[1], vjp(x0, u) @ v
print(f"uᵀ(Jv) = {lhs:.12f}   (uᵀJ)v = {rhs:.12f}")
assert calls["jvp"] == n + 1 and calls["vjp"] == m + 1
assert np.allclose(J_fwd, J_rev, atol=1e-12) and rel_error(J_fwd, J_num) < 1e-8
assert abs(lhs - rhs) < 1e-10
print("Lab 4 ĐẠT ✔")"""

# ---- Lab 5: MLP + softmax-CE -------------------------------------------------
MLP = ("""def mlp_loss_grad(params, X, y, reg=0.0):
    \"\"\"MLP hai lớp: S = ReLU(X W1 + b1) W2 + b2, loss = trung bình cross-entropy + ½·reg·(‖W1‖² + ‖W2‖²)
    (chuẩn Frobenius). X ∈ ℝᴺˣᴰ, y là nhãn nguyên. Trả về (loss, grads) với grads cùng key và shape như params.
    Nhớ: softmax trừ max; ∂L/∂S = (P − Y)/N; db = tổng theo trục batch (axis=0).\"\"\"
    # TODO
    raise NotImplementedError""",
"""def mlp_loss_grad(params, X, y, reg=0.0):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    N = X.shape[0]
    A = X @ W1 + b1                              # (N, H)
    Hh = np.maximum(0, A)
    S = Hh @ W2 + b2                             # (N, C)
    S = S - S.max(axis=1, keepdims=True)         # log-sum-exp ổn định
    logp = S - np.log(np.exp(S).sum(axis=1, keepdims=True))
    loss = -logp[np.arange(N), y].mean() + 0.5 * reg * (np.sum(W1**2) + np.sum(W2**2))
    dS = np.exp(logp)
    dS[np.arange(N), y] -= 1
    dS /= N                                      # P − Y, chia N vì loss là trung bình
    g = {"W2": Hh.T @ dS + reg * W2, "b2": dS.sum(axis=0)}
    dH = dS @ W2.T
    dA = dH * (A > 0)                            # ReLU chặn gradient ở chỗ A ≤ 0
    g["W1"] = X.T @ dA + reg * W1
    g["b1"] = dA.sum(axis=0)
    return loss, g""")

MLP_CHECK = """rng = np.random.default_rng(5)
# (a) khớp Ví dụ 2.10 (sách dùng quy ước cột; ở đây dữ liệu theo hàng nên W ở đây = Wᵀ ở sách)
p_ex = {"W1": np.array([[0.1, 0.2], [-0.3, 0.4]]).T, "b1": np.array([0.0, 0.1]),
        "W2": np.array([[1.0, -1.0], [0.5, 0.5]]).T, "b2": np.zeros(2)}
L_ex, g_ex = mlp_loss_grad(p_ex, np.array([[1.0, 2.0]]), np.array([0]))
print(f"Ví dụ 2.10: L = {L_ex:.6f}, ∂L/∂W₁ (theo sách) =\\n{g_ex['W1'].T}")
assert np.isclose(L_ex, 1.070055, atol=1e-6) and np.isclose(g_ex["W1"].T[1, 1], 1.971031, atol=1e-6)

# (b) gradient check trên dữ liệu ngẫu nhiên nhỏ, float64
N, D, H, C = 5, 4, 6, 3
Xs, ys = rng.standard_normal((N, D)), rng.integers(0, C, N)
params = {"W1": rng.standard_normal((D, H)) * 0.5, "b1": rng.standard_normal(H) * 0.1,
          "W2": rng.standard_normal((H, C)) * 0.5, "b2": np.zeros(C)}
_, g = mlp_loss_grad(params, Xs, ys, reg=0.1)
worst = 0.0
for k in params:
    def fk(v, k=k):
        return mlp_loss_grad({**params, k: v}, Xs, ys, reg=0.1)[0]
    e = rel_error(g[k], numerical_grad(fk, params[k].copy()))
    worst = max(worst, e)
    print(f"  {k}: rel_error = {e:.1e}")
assert worst < 1e-7
print("Lab 5a ĐẠT ✔")"""

TRAIN = """def spiral(n_per=100, K=3, seed=0):
    r = np.random.default_rng(seed)
    X, y = np.zeros((n_per * K, 2)), np.zeros(n_per * K, dtype=int)
    for k in range(K):
        t = np.linspace(0, 1, n_per)
        ang = np.linspace(k * 4, (k + 1) * 4, n_per) + r.standard_normal(n_per) * 0.2
        X[k * n_per:(k + 1) * n_per] = np.c_[t * np.sin(ang), t * np.cos(ang)]
        y[k * n_per:(k + 1) * n_per] = k
    return X, y

Xsp, ysp = spiral()
r = np.random.default_rng(1)
P = {"W1": r.standard_normal((2, 100)) * 0.1, "b1": np.zeros(100),
     "W2": r.standard_normal((100, 3)) * 0.1, "b2": np.zeros(3)}
for step in range(3001):
    L, g = mlp_loss_grad(P, Xsp, ysp, reg=1e-3)
    for k in P:
        P[k] -= 1.0 * g[k]
    if step % 1000 == 0:
        print(f"bước {step:4d}: loss = {L:.4f}")
Hh = np.maximum(0, Xsp @ P["W1"] + P["b1"])
acc = np.mean(np.argmax(Hh @ P["W2"] + P["b2"], axis=1) == ysp)
print("accuracy train =", acc)
assert acc > 0.95
# Tách loss cuối thành hai phần: cross-entropy dữ liệu và regularization
L_final = mlp_loss_grad(P, Xsp, ysp, reg=1e-3)[0]
L_data = mlp_loss_grad(P, Xsp, ysp, reg=0.0)[0]
wsq = np.sum(P["W1"]**2) + np.sum(P["W2"]**2)
print(f"loss cuối {L_final:.4f} = dữ liệu {L_data:.4f} + regularization {0.5e-3 * wsq:.4f}  (tổng ‖W‖² = {wsq:.0f})")
assert np.isclose(L_final, L_data + 0.5e-3 * wsq)
print("Lab 5b ĐẠT ✔")"""

BUG_DATA = """def grads_buggy(W1, b1, W2, b2, X, y):
    N = X.shape[0]
    A = X @ W1 + b1
    Hh = np.maximum(0, A)
    S = Hh @ W2 + b2
    P = np.exp(S - S.max(axis=1, keepdims=True)); P /= P.sum(axis=1, keepdims=True)
    dS = P.copy(); dS[np.arange(N), y] -= 1
    dW2 = Hh.T @ dS
    db2 = dS.sum(axis=0)
    dH = dS @ W2
    dA = dH * (A > 0)
    dW1 = X.T @ dA
    db1 = dA.sum(axis=1)
    return dW1, db1, dW2, db2

def loss_fn(W1, b1, W2, b2, X, y):
    S = np.maximum(0, X @ W1 + b1) @ W2 + b2
    S = S - S.max(axis=1, keepdims=True)
    return -(S[np.arange(len(y)), y] - np.log(np.exp(S).sum(axis=1))).mean()

rb = np.random.default_rng(1)
Nb, Db, Hb, Cb = 4, 3, 4, 4                       # N = H = C: mọi ma trận "vừa khít"
Xb, yb = rb.standard_normal((Nb, Db)), np.array([0, 1, 2, 3])
Pb = [rb.standard_normal((Db, Hb)) * 0.5, np.zeros(Hb), rb.standard_normal((Hb, Cb)) * 0.5, np.zeros(Cb)]

def check(grad_fn):
    G = grad_fn(*Pb, Xb, yb)
    errs = []
    for i in range(4):
        def fi(v, i=i):
            ps = list(Pb); ps[i] = v
            return loss_fn(*ps, Xb, yb)
        errs.append(rel_error(G[i], numerical_grad(fi, Pb[i].copy())))
    return errs

print("bản lỗi, rel_error W1, b1, W2, b2:", [f"{e:.2f}" for e in check(grads_buggy)])"""

BUGFIX = ("""def grads_fixed(W1, b1, W2, b2, X, y):
    \"\"\"Sửa cả ba lỗi của grads_buggy. Trả về (dW1, db1, dW2, db2).\"\"\"
    # TODO
    raise NotImplementedError""",
"""def grads_fixed(W1, b1, W2, b2, X, y):
    N = X.shape[0]
    A = X @ W1 + b1
    Hh = np.maximum(0, A)
    S = Hh @ W2 + b2
    P = np.exp(S - S.max(axis=1, keepdims=True)); P /= P.sum(axis=1, keepdims=True)
    dS = P.copy(); dS[np.arange(N), y] -= 1
    dS /= N                                      # sửa lỗi 1: loss là trung bình
    dW2 = Hh.T @ dS
    db2 = dS.sum(axis=0)
    dH = dS @ W2.T                               # sửa lỗi 2: W2ᵀ, không phải W2
    dA = dH * (A > 0)
    dW1 = X.T @ dA
    db1 = dA.sum(axis=0)                         # sửa lỗi 3: cộng theo trục batch
    return dW1, db1, dW2, db2""")

BUG_CHECK = """errs = check(grads_fixed)
print("bản sửa, rel_error:", [f"{e:.1e}" for e in errs])
assert max(errs) < 1e-7
print("Săn bug ĐẠT ✔")"""


def build(solution: bool):
    pick = (lambda pair: pair[1]) if solution else (lambda pair: pair[0])
    nb = nbf.v4.new_notebook()
    head = "Lời giải" if solution else "Bài lab"
    nb.cells = [
        md(f"# Chương 2 — {head}: Giải tích ma trận và backpropagation\n\n"
           + ("Bản lời giải đã chạy kiểm. Chỉ mở sau khi đã tự làm `lab.ipynb`."
              if solution else
              "Viết code vào các ô có `# TODO`, rồi chạy ô kiểm tra ngay bên dưới. "
              "Ô kiểm tra in **ĐẠT ✔** là xong. Bí quá 20 phút mới mở `loigiai.ipynb`. "
              "Chỉ dùng NumPy — không PyTorch, để bạn tự viết mọi gradient.")),
        code(SETUP),
        md("## Lab 1 — Gradient số và sai số tương đối\nCông cụ dùng cho mọi lab sau. Kiểm ∇(xᵀAx) = (A + Aᵀ)x "
           "và ∂ log det X/∂X = X⁻ᵀ; công thức sai 2Ax và đáp án quên chuyển vị (X⁻¹) đều phải bị bắt."),
        code(pick(NUMGRAD)), code(NUMGRAD_CHECK),
        md("## Lab 2 — Chọn bước h: float64 và float32\nTái hiện bảng Ví dụ 2.11. "
           "Câu hỏi: vì sao sai số không giảm mãi khi h nhỏ dần?"),
        code(pick(SWEEP)), code(SWEEP_RUN),
        md("## Lab 3 — Mini autograd vô hướng\nViết lớp `Value` kiểu micrograd. Kiểm trên Ví dụ 2.6 và 2.7."),
        code(pick(VALUE)), code(VALUE_CHECK),
        md("## Lab 4 — JVP và VJP\nDựng Jacobian 3×6 bằng 6 lượt forward mode và bằng 3 lượt reverse mode; "
           "kiểm uᵀ(Jv) = (uᵀJ)v."),
        code(pick(JV)), code(JV_CHECK),
        md("## Lab 5 — MLP hai lớp + softmax-CE theo batch\n(a) Khớp Ví dụ 2.10 và qua gradient check. "
           "(b) Huấn luyện trên dữ liệu xoắn ốc 3 lớp."),
        code(pick(MLP)), code(MLP_CHECK), code(TRAIN),
        md("## Săn bug\n`grads_buggy` chạy không lỗi, shape đều đúng, nhưng gradient sai. "
           "Tìm cả **ba** lỗi rồi viết `grads_fixed`."),
        code(BUG_DATA), code(pick(BUGFIX)), code(BUG_CHECK),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    name = "loigiai.ipynb" if solution else "lab.ipynb"
    nbf.write(nb, OUT / name)
    return OUT / name


if __name__ == "__main__":
    print(build(False), build(True))
