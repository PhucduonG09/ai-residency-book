"""Sinh hai notebook cho lab chương 4: lab.ipynb (có TODO) và loigiai.ipynb (lời giải).

Chạy lại script này nếu muốn tạo lại notebook gốc:  python labs/build_chuong04.py
Chỉ dùng NumPy và SciPy. Toàn bộ notebook lời giải chạy dưới 30 giây trên CPU.
"""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "chuong04"
OUT.mkdir(parents=True, exist_ok=True)

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

SETUP = """import numpy as np
from scipy.optimize import minimize, rosen, rosen_der
np.set_printoptions(precision=4, suppress=True)

# Bài toán chạy xuyên chương: f(x) = ½(λ₁x₁² + λ₂x₂²), λ = (1, 100), κ = 100, điểm đầu (1, 1)
LAM = np.array([1.0, 100.0])
X0 = np.array([1.0, 1.0])
grad_q = lambda x: LAM * x

# Bài toán hồi quy tuyến tính dùng chung cho lab 2 và 4 (κ ≈ 98)
rng = np.random.default_rng(0)
n, d = 500, 10
Xr = rng.normal(size=(n, d)) * np.linspace(1, 10, d)
yr = Xr @ rng.normal(size=d) + 0.1 * rng.normal(size=n)
w_star = np.linalg.lstsq(Xr, yr, rcond=None)[0]
H_r = Xr.T @ Xr / n
L_r = np.linalg.eigvalsh(H_r)[-1]
f_r = lambda w: 0.5 * np.mean((Xr @ w - yr) ** 2)
f_star = f_r(w_star)

def grad_r(w, idx=None):
    Xi, yi = (Xr, yr) if idx is None else (Xr[idx], yr[idx])
    return Xi.T @ (Xi @ w - yi) / len(yi)            # TRUNG BÌNH, không phải tổng

print(f"hồi quy: κ = {np.linalg.cond(H_r):.1f}, L = {L_r:.2f}")"""

# ---------------------------------------------------------------- Lab 1
GD = ("""def gd_steps(grad, x0, eta, tol=1e-6, max_iter=100_000):
    \"\"\"Chạy x ← x − η∇f(x). Trả về số bước để ‖x‖ < tol·‖x0‖ (nghiệm x* = 0),
    hoặc None nếu chưa đạt sau max_iter bước hay ‖x‖ vượt 1e6 (phân kỳ).\"\"\"
    # TODO
    raise NotImplementedError""",
"""def gd_steps(grad, x0, eta, tol=1e-6, max_iter=100_000):
    x = x0.copy()
    for k in range(1, max_iter + 1):
        x = x - eta * grad(x)
        nx = np.linalg.norm(x)
        if nx < tol * np.linalg.norm(x0):
            return k
        if nx > 1e6:
            return None
    return None""")

GD_CHECK = """L, mu = LAM.max(), LAM.min()
k1 = gd_steps(grad_q, X0, 1 / L)
k2 = gd_steps(grad_q, X0, 2 / (L + mu))
k3 = gd_steps(grad_q, X0, 0.0201)
print("η = 1/L:", k1, "| η = 2/(L+μ):", k2, "| η = 0.0201:", k3)
print("lý thuyết:", np.log(1e-6) / np.log(1 - mu / L), np.log(1e-6) / np.log((L - mu) / (L + mu)))
assert 1300 <= k1 <= 1400 and 680 <= k2 <= 700 and k3 is None

# phần b — chuẩn hóa đặc trưng (ví dụ 4.5)
t = np.array([0.0, 1, 2, 10])
A = np.column_stack([np.ones(4), t])
As = np.column_stack([np.ones(4), (t - t.mean()) / t.std()])
kap = lambda A: np.linalg.cond(A.T @ A / len(A))
print(f"κ trước = {kap(A):.2f}, sau chuẩn hóa = {kap(As):.2f}")
assert abs(kap(A) - 45.3) < 0.1 and abs(kap(As) - 1) < 1e-9
print("Lab 1 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 2
SGD = ("""def sgd_minibatch(w0, eta_fn, B, epochs, seed=0):
    \"\"\"SGD minibatch trên bài hồi quy (dùng grad_r(w, idx)).
    Mỗi epoch: xáo trộn lại chỉ số (permutation), cắt thành n // B minibatch rời nhau.
    eta_fn(t) cho learning rate ở bước t (t đếm từ 0, liên tục qua các epoch).
    Trả về w cuối.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def sgd_minibatch(w0, eta_fn, B, epochs, seed=0):
    r = np.random.default_rng(seed)
    w, t = w0.copy(), 0
    for _ in range(epochs):
        perm = r.permutation(n)                       # xáo trộn lại MỖI epoch
        for i in range(n // B):
            idx = perm[i * B:(i + 1) * B]             # minibatch thứ i, không phải perm[:B]
            w = w - eta_fn(t) * grad_r(w, idx)
            t += 1
    return w""")

SGD_CHECK = """# phần a — sàn nhiễu trên f(w) = ½w², g = w + ξ, Var ξ = σ²/B, σ² = 4 (ví dụ 4.6)
def stationary_var(eta, B, sigma2=4.0, chains=4000, steps=3000, seed=1):
    r = np.random.default_rng(seed)
    w = np.zeros(chains); tail = []
    for s in range(steps):
        w = w - eta * (w + r.normal(0, np.sqrt(sigma2 / B), chains))
        if s >= steps - 500:
            tail.append(w.copy())
    return np.var(np.concatenate(tail))

for eta, B in [(0.1, 1), (0.1, 16), (0.01, 1)]:
    sim, th = stationary_var(eta, B), eta * 4.0 / (B * (2 - eta))
    print(f"η = {eta}, B = {B:2d}: mô phỏng {sim:.4f}, lý thuyết {th:.4f}")
    assert abs(sim / th - 1) < 0.05

# phần b — η hằng dừng ở sàn nhiễu; giảm η thì xuống tiếp
E, B = 200, 32
steps = E * (n // B)
eta0 = 0.5 / L_r
w_const = sgd_minibatch(np.zeros(d), lambda t: eta0, B, E)
def eta_decay(t):                                    # giữ η₀ nửa đầu, rồi giảm tuyến tính tới η₀/100
    half = steps // 2
    return eta0 if t < half else eta0 * max(0.01, 1 - (t - half) / (steps - half))
w_decay = sgd_minibatch(np.zeros(d), eta_decay, B, E)
gap_c, gap_d = f_r(w_const) - f_star, f_r(w_decay) - f_star
print(f"f − f*: η hằng {gap_c:.2e} | η giảm dần {gap_d:.2e}")
assert gap_c > 1e-6 and gap_d < gap_c / 10

# đối chứng: GD full-batch cùng η₀, cùng bài (mỗi bước dùng cả n = 500 mẫu)
w_fb = np.zeros(d)
for k in range(1, 3126):
    w_fb = w_fb - eta0 * grad_r(w_fb)
    if k == 200:
        gap_200 = f_r(w_fb) - f_star
print(f"GD full-batch: f − f* = {gap_200:.1e} sau 200 bước, {f_r(w_fb) - f_star:.1e} sau 3 125 bước")
assert gap_200 > 1e-4 and f_r(w_fb) - f_star < 1e-14
print("Lab 2 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 3
MOM = ("""def momentum_steps(grad, x0, eta, beta, nesterov=False, tol=1e-6, max_iter=100_000):
    \"\"\"Heavy ball:  v ← βv − η∇f(x);        x ← x + v
       Nesterov:    v ← βv − η∇f(x + βv);   x ← x + v
    Trả về số bước để ‖x‖ < tol·‖x0‖ (None nếu không đạt).\"\"\"
    # TODO
    raise NotImplementedError""",
"""def momentum_steps(grad, x0, eta, beta, nesterov=False, tol=1e-6, max_iter=100_000):
    x, v = x0.copy(), np.zeros_like(x0)
    for k in range(1, max_iter + 1):
        g = grad(x + beta * v) if nesterov else grad(x)
        v = beta * v - eta * g
        x = x + v
        if np.linalg.norm(x) < tol * np.linalg.norm(x0):
            return k
    return None""")

MOM_CHECK = """L, mu = LAM.max(), LAM.min()
sk = np.sqrt(L / mu)
eta_hb = 4 / (np.sqrt(L) + np.sqrt(mu)) ** 2
beta_hb = ((sk - 1) / (sk + 1)) ** 2
k_hb = momentum_steps(grad_q, X0, eta_hb, beta_hb)
k_gd = gd_steps(grad_q, X0, 2 / (L + mu))
print(f"heavy ball tối ưu (η = {eta_hb:.4f}, β = {beta_hb:.4f}): {k_hb} bước; GD tối ưu: {k_gd}; tỉ lệ {k_gd / k_hb:.1f}")
for eta, beta in [(0.01, 0.9), (0.01, 9 / 11)]:
    print(f"η = {eta}, β = {beta:.3f}: heavy ball {momentum_steps(grad_q, X0, eta, beta)}, "
          f"Nesterov {momentum_steps(grad_q, X0, eta, beta, nesterov=True)}")
assert k_hb <= 100 and k_gd >= 5 * k_hb
print("Lab 3 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 4
ADAM = ("""def adam_step(w, g, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, decoupled=False):
    \"\"\"Một bước Adam (Kingma & Ba, Algorithm 1). state = dict(m, v, t), sửa tại chỗ.
    decoupled=False: wd là L2 — cộng wd·w vào gradient trước khi cập nhật m, v.
    decoupled=True (AdamW, dạng PyTorch): trừ lr·wd·w trực tiếp khỏi w, không đi qua m, v.
    Trả về w mới.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def adam_step(w, g, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, decoupled=False):
    if wd and not decoupled:
        g = g + wd * w                                # L2: decay đi qua bộ chuẩn hóa
    state["t"] += 1
    t = state["t"]
    state["m"] = b1 * state["m"] + (1 - b1) * g
    state["v"] = b2 * state["v"] + (1 - b2) * g * g
    m_hat = state["m"] / (1 - b1 ** t)
    v_hat = state["v"] / (1 - b2 ** t)
    if wd and decoupled:
        w = w - lr * wd * w                           # AdamW: decay tách riêng
    return w - lr * m_hat / (np.sqrt(v_hat) + eps)""")

ADAM_CHECK = """new_state = lambda shape=(): {"m": np.zeros(shape), "v": np.zeros(shape), "t": 0}

# (a) bước đầu ≈ α bất kể độ lớn gradient
for g in [1e-4, 1.0, 1e3]:
    step = 1.0 - adam_step(1.0, g, new_state())
    print(f"g = {g:g}: bước đầu = {step:.7f}")
    assert abs(step - 1e-3) < 2e-7

# (b) bất biến khi nhân gradient từng tọa độ với hằng số dương
def traj(scale, T=50):
    w, st = X0.copy(), new_state(2)
    for _ in range(T):
        w = adam_step(w, scale * LAM * w, st, lr=0.01, eps=1e-12)
    return w
wa, wb = traj(np.ones(2)), traj(np.array([1e4, 1e-2]))
print("quỹ đạo sau 50 bước:", wa, wb)
assert np.allclose(wa, wb, atol=1e-6)

# (c) tham số có gradient loss bằng 0, λ = 0.01, 1000 bước
def shrink(decoupled, T=1000):
    w, st = 1.0, new_state()
    for _ in range(T):
        w = adam_step(w, 0.0, st, wd=0.01, decoupled=decoupled)
    return w
w_l2, w_dw = shrink(False), shrink(True)
print(f"L2 trong Adam: {w_l2:.4f} | AdamW: {w_dw:.4f}")
assert abs(w_dw - 0.9900) < 1e-4 and abs(w_l2 - 0.258) < 1e-3

# (d) hội tụ trên bài hồi quy
w, st = np.zeros(d), new_state(d)
for _ in range(3000):
    w = adam_step(w, grad_r(w), st, lr=0.05)
print(f"Adam 3000 bước: ‖w − w*‖ = {np.linalg.norm(w - w_star):.1e}")
assert np.linalg.norm(w - w_star) < 1e-6
print("Lab 4 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 5
NEWTON = ("""def newton_logistic(X, y, max_iter=50, tol=1e-10):
    \"\"\"Newton cho logistic regression (nhãn 0/1, loss trung bình).
    ∇ = Xᵀ(s − y)/n,  H = Xᵀ diag(s(1 − s)) X / n,  s = sigmoid(Xw).
    Trả về (w, số bước) — dừng khi ‖∇‖ < tol.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def newton_logistic(X, y, max_iter=50, tol=1e-10):
    w = np.zeros(X.shape[1])
    for k in range(1, max_iter + 1):
        s = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (s - y) / len(y)
        H = (X.T * (s * (1 - s))) @ X / len(y)
        w = w - np.linalg.solve(H, g)                 # giải hệ, không nghịch đảo
        if np.linalg.norm(X.T @ (1 / (1 + np.exp(-X @ w)) - y) / len(y)) < tol:
            return w, k
    return w, max_iter""")

NEWTON_CHECK = """r5 = np.random.default_rng(5)
Xl = np.column_stack([np.ones(300), r5.normal(size=(300, 2))])
yl = (r5.random(300) < 1 / (1 + np.exp(-Xl @ np.array([-0.5, 2.0, -1.0])))).astype(float)
w_nt, k_nt = newton_logistic(Xl, yl)

def nll(w):
    z = Xl @ w
    return np.mean(np.logaddexp(0, z) - yl * z)
res = minimize(nll, np.zeros(3), jac=lambda w: Xl.T @ (1 / (1 + np.exp(-Xl @ w)) - yl) / len(yl),
               method="L-BFGS-B", options={"gtol": 1e-10})
print(f"Newton: {k_nt} bước, w = {w_nt} | L-BFGS-B: {res.nit} lần lặp, w = {res.x}")
assert k_nt < 10 and np.allclose(w_nt, res.x, atol=1e-5)

# phần b — Rosenbrock từ (−1.2, 1)
x0 = np.array([-1.2, 1.0])
its = {m: minimize(rosen, x0, jac=rosen_der, method=m).nit for m in ["BFGS", "L-BFGS-B", "CG"]}
x, k_gd = x0.copy(), 0
while np.linalg.norm(x - 1) > 1e-4:
    x -= 1e-3 * rosen_der(x); k_gd += 1
print("Rosenbrock:", its, "| GD η = 1e-3:", k_gd)
assert k_gd > 10_000 and its["L-BFGS-B"] < 100

# phần c — f = x² − y²: Newton về yên ngựa, GD thoát
Hs = np.diag([2.0, -2.0]); gs = lambda p: Hs @ p
p_newton = np.array([1.0, 0.5]) - np.linalg.solve(Hs, gs(np.array([1.0, 0.5])))
p, k_esc = np.array([1.0, 1e-3]), 0
while abs(p[1]) <= 1:
    p = p - 0.1 * gs(p); k_esc += 1
print("Newton một bước:", p_newton, "| GD thoát |y| > 1 sau", k_esc, "bước, x =", p[0])
assert np.allclose(p_newton, 0) and k_esc == 38

# phần d — logistic regression trên dữ liệu tách được: ‖w‖ tăng mãi, không có cực tiểu hữu hạn
# (không có cột bias: đặc trưng thứ hai là +1 ở lớp 1 và −1 ở lớp 0)
Xs = np.array([[1, 1], [2, 1], [-1, -1], [-2, -1.]]); ys = np.array([1, 1, 0, 0.])
ws, norms = np.zeros(2), {}
for k in range(1, 100_001):
    ws = ws - 1.0 * Xs.T @ (1 / (1 + np.exp(-Xs @ ws)) - ys) / 4          # GD, η = 1
    if k in (100, 1_000, 10_000, 100_000):
        norms[k] = np.linalg.norm(ws)
print("‖w‖ ở bước 10², 10³, 10⁴, 10⁵:", {k: round(float(v), 2) for k, v in norms.items()})
vals = list(norms.values())
assert np.allclose(vals, [3.41, 4.95, 6.54, 8.16], atol=0.01)
assert all(1.4 < b - a < 1.7 for a, b in zip(vals, vals[1:]))              # mỗi ×10 bước: cộng ≈ hằng số
print("Lab 5 ĐẠT ✔")"""

# ---------------------------------------------------------------- Săn bug
BUG_DATA = """r7 = np.random.default_rng(0)
nb, db = 1000, 5
Xb = r7.normal(size=(nb, db))
yb = Xb @ np.arange(1, db + 1.0) + 0.5 * r7.normal(size=nb)
wb_star = np.linalg.lstsq(Xb, yb, rcond=None)[0]
fb = lambda w: 0.5 * np.mean((Xb @ w - yb) ** 2)

def train_buggy(epochs=20, B=32, lr0=0.05, beta=0.9):
    w, v = np.zeros(db), np.zeros(db)
    T, t = epochs * (nb // B), 0
    r = np.random.default_rng(1)
    perm = r.permutation(nb)
    for ep in range(epochs):
        for i in range(nb // B):
            idx = perm[:B]
            Xi = Xb[idx]
            g = Xi.T @ (Xi @ w - yb[idx]) / B
            lr = 0.5 * lr0 * (1 + np.cos(np.pi * t))
            v = beta * v + g
            w = w - lr * g
            t += 1
    return w

wbug = train_buggy()
print(f"bản lỗi: ‖w − w*‖ = {np.linalg.norm(wbug - wb_star):.3f}, f − f* = {fb(wbug) - fb(wb_star):.3f}")"""

BUGFIX = ("""def train_fixed(epochs=20, B=32, lr0=0.05, beta=0.9):
    \"\"\"Sửa cả ba lỗi của train_buggy (giữ nguyên seed và siêu tham số).\"\"\"
    # TODO
    raise NotImplementedError""",
"""def train_fixed(epochs=20, B=32, lr0=0.05, beta=0.9):
    w, v = np.zeros(db), np.zeros(db)
    T, t = epochs * (nb // B), 0
    r = np.random.default_rng(1)
    for ep in range(epochs):
        perm = r.permutation(nb)                          # xáo trộn mỗi epoch
        for i in range(nb // B):
            idx = perm[i * B:(i + 1) * B]                 # sửa lỗi 1: chạy theo i
            Xi = Xb[idx]
            g = Xi.T @ (Xi @ w - yb[idx]) / B
            lr = 0.5 * lr0 * (1 + np.cos(np.pi * t / T)) # sửa lỗi 2: chia cho T
            v = beta * v + g
            w = w - lr * v                                # sửa lỗi 3: dùng vận tốc
            t += 1
    return w""")

BUG_CHECK = """wfix = train_fixed()
err, gap = np.linalg.norm(wfix - wb_star), fb(wfix) - fb(wb_star)
print(f"bản sửa: ‖w − w*‖ = {err:.4f}, f − f* = {gap:.1e}")
assert err < 0.01 and gap < 5e-5
print("Săn bug ĐẠT ✔")"""


def build(solution: bool):
    pick = (lambda pair: pair[1]) if solution else (lambda pair: pair[0])
    nb = nbf.v4.new_notebook()
    head = "Lời giải" if solution else "Bài lab"
    nb.cells = [
        md(f"# Chương 4 — {head}: Tối ưu hóa\n\n"
           + ("Bản lời giải đã chạy kiểm. Chỉ mở sau khi đã tự làm `lab.ipynb`."
              if solution else
              "Viết code vào các ô có `# TODO`, rồi chạy ô kiểm tra ngay bên dưới. "
              "Ô kiểm tra in **ĐẠT ✔** là xong. Bí quá 20 phút mới mở `loigiai.ipynb`. "
              "Lab 3 dùng lại `gd_steps` của lab 1.")),
        code(SETUP),
        md("## Lab 1 — GD và số điều kiện\nĐếm số bước trên f = ½(x₁² + 100x₂²) với ba learning rate; "
           "sau đó đo κ trước và sau khi chuẩn hóa đặc trưng (ví dụ 4.4, 4.5)."),
        code(pick(GD)), code(GD_CHECK),
        md("## Lab 2 — SGD, sàn nhiễu, minibatch\n(a) Phương sai dừng của SGD khớp công thức ησ²/(B(2 − η)). "
           "(b) Trên hồi quy, η hằng dừng ở sàn nhiễu; giảm dần η thì xuống tiếp."),
        code(pick(SGD)), code(SGD_CHECK),
        md("## Lab 3 — Momentum và Nesterov\nVới tham số tối ưu, heavy ball cần khoảng √κ·ln(1/ε) bước. "
           "Câu hỏi: với η = 0.01, β = 0.9 thì Nesterov thắng; với β = 9/11 thì sao?"),
        code(pick(MOM)), code(MOM_CHECK),
        md("## Lab 4 — Adam và AdamW từ đầu\nKiểm bốn tính chất: bước đầu ≈ α, bất biến đổi thang theo tọa độ, "
           "L2 khác weight decay tách rời, và hội tụ trên bài hồi quy."),
        code(pick(ADAM)), code(ADAM_CHECK),
        md("## Lab 5 — Newton, L-BFGS và điểm yên ngựa\n(a) Newton cho logistic regression, đối chiếu L-BFGS-B. "
           "(b) Rosenbrock: đếm số lần lặp. (c) Newton bị hút về yên ngựa, GD thì thoát. (d) Logistic trên dữ liệu tách được: ‖w‖ tăng mãi."),
        code(pick(NEWTON)), code(NEWTON_CHECK),
        md("## Săn bug\n`train_buggy` (SGD + momentum + lịch cosine) chạy không lỗi, loss vẫn giảm, "
           "nhưng ‖w − w*‖ lớn gấp khoảng 175 lần bản đúng. Tìm cả **ba** lỗi rồi viết `train_fixed`."),
        code(BUG_DATA), code(pick(BUGFIX)), code(BUG_CHECK),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    name = "loigiai.ipynb" if solution else "lab.ipynb"
    nbf.write(nb, OUT / name)
    return OUT / name


if __name__ == "__main__":
    print(build(False), build(True))
