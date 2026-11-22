"""Sinh hai notebook cho lab chương 5: lab.ipynb (có TODO) và loigiai.ipynb (lời giải).

Chạy lại script này nếu muốn tạo lại notebook gốc:  python labs/build_chuong05.py
Chỉ dùng NumPy (không cần scikit-learn): mọi thuật toán tự cài từ đầu.
"""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "chuong05"
OUT.mkdir(parents=True, exist_ok=True)

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

SETUP = """import numpy as np
rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True)"""

# ---------------------------------------------------------------- Lab 1
SOFTMAX = ("""def softmax(Z):
    \"\"\"Softmax theo hàng của Z ∈ ℝᴺˣᴷ, ổn định số (trừ max từng hàng trước khi lấy exp).\"\"\"
    # TODO
    raise NotImplementedError


def ce_loss_grad(W, b, X, y, lam=0.0):
    \"\"\"Trả về (loss, dW, db) của cross-entropy trung bình + (lam/2)‖W‖².
    W ∈ ℝᵈˣᴷ, b ∈ ℝᴷ, X ∈ ℝᴺˣᵈ, y là nhãn nguyên 0..K−1.
    Gợi ý: gradient theo logit là (P − Y)/N, với Y là one-hot.\"\"\"
    # TODO
    raise NotImplementedError


def fit_softmax(X, y, K, eta=0.5, n_iter=300, lam=0.0):
    \"\"\"Gradient descent từ W = 0, b = 0. Trả về (W, b, lịch sử loss).\"\"\"
    # TODO
    raise NotImplementedError


def logistic_gd(x, y, eta=0.5, n_steps=1000, lam=0.0):
    \"\"\"Logistic regression 1 chiều p = σ(wx + b), GD trên log-loss trung bình (+ lam·w trong gradient của w).
    Khởi tạo w = b = 0. Trả về (w, b).\"\"\"
    # TODO
    raise NotImplementedError""",
"""def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)          # bất biến khi cộng hằng ⇒ trừ max cho an toàn
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def ce_loss_grad(W, b, X, y, lam=0.0):
    N = len(X)
    P = softmax(X @ W + b)
    loss = -np.mean(np.log(P[np.arange(N), y])) + 0.5 * lam * np.sum(W ** 2)
    G = P.copy()
    G[np.arange(N), y] -= 1.0                      # P − Y
    G /= N
    return loss, X.T @ G + lam * W, G.sum(axis=0)


def fit_softmax(X, y, K, eta=0.5, n_iter=300, lam=0.0):
    W, b = np.zeros((X.shape[1], K)), np.zeros(K)
    hist = []
    for _ in range(n_iter):
        loss, dW, db = ce_loss_grad(W, b, X, y, lam)
        hist.append(loss)
        W -= eta * dW
        b -= eta * db
    return W, b, np.array(hist)


def logistic_gd(x, y, eta=0.5, n_steps=1000, lam=0.0):
    w = b = 0.0
    for _ in range(n_steps):
        p = 1 / (1 + np.exp(-(w * x + b)))
        w -= eta * (np.mean((p - y) * x) + lam * w)
        b -= eta * np.mean(p - y)
    return w, b""")

SOFTMAX_CHECK = """# (a) ổn định số
print("softmax([1000, 1001]) =", softmax(np.array([[1000.0, 1001.0]])))
assert np.all(np.isfinite(softmax(np.array([[1000.0, 1001.0]]))))
# (b) gradient đúng: so với sai phân hữu hạn trung tâm
Xg, yg = rng.standard_normal((20, 4)), rng.integers(0, 3, 20)
Wg, bg = rng.standard_normal((4, 3)), rng.standard_normal(3)
_, dW, _ = ce_loss_grad(Wg, bg, Xg, yg, lam=0.1)
num, h = np.zeros_like(Wg), 1e-6
for i in range(4):
    for k in range(3):
        E = np.zeros_like(Wg); E[i, k] = h
        num[i, k] = (ce_loss_grad(Wg + E, bg, Xg, yg, 0.1)[0] - ce_loss_grad(Wg - E, bg, Xg, yg, 0.1)[0]) / (2 * h)
rel = np.linalg.norm(num - dW) / np.linalg.norm(num)
print(f"sai lệch tương đối gradient: {rel:.1e}")
assert rel < 1e-6
# (c) huấn luyện trên 3 cụm Gauss
centers = np.array([[0.0, 2.0], [-2.0, -1.0], [2.0, -1.0]])
y3 = np.repeat(np.arange(3), 200)
X3 = centers[y3] + rng.standard_normal((600, 2))
W3, b3, hist = fit_softmax(X3, y3, 3)
acc = np.mean(np.argmax(X3 @ W3 + b3, axis=1) == y3)
print(f"loss: {hist[0]:.4f} → {hist[-1]:.4f},  accuracy train = {acc:.3f}")
assert np.all(np.diff(hist) <= 1e-12) and acc > 0.85
# (d) dữ liệu tách được (ví dụ 5.2): không L2 thì w tăng mãi, có L2 thì hội tụ
x4, y4 = np.array([1.0, 2, 3, 4]), np.array([0.0, 0, 1, 1])
for steps in (100, 10000, 20000):
    w, b = logistic_gd(x4, y4, n_steps=steps)
    print(f"không L2, {steps:5d} bước: w = {w:.3f}, b = {b:.3f}, ranh giới −b/w = {-b / w:.3f}")
wl, bl = logistic_gd(x4, y4, n_steps=20000, lam=0.1)
print(f"L2 λ = 0.1, 20000 bước: w = {wl:.3f}, b = {bl:.3f}, ranh giới = {-bl / wl:.3f}")
assert logistic_gd(x4, y4, n_steps=20000)[0] > logistic_gd(x4, y4, n_steps=10000)[0] > 8
assert abs(wl - 1.508) < 1e-3 and abs(-bl / wl - 2.5) < 1e-3
print("Lab 1 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 2
REG = ("""def ridge(X, y, lam):
    \"\"\"Nghiệm của min ½‖y − Xw‖² + (λ/2)‖w‖²: giải (XᵀX + λI)w = Xᵀy bằng np.linalg.solve.\"\"\"
    # TODO
    raise NotImplementedError


def soft_threshold(z, t):
    # TODO: sign(z)·max(|z| − t, 0)
    raise NotImplementedError


def lasso_cd(X, y, lam, n_iter=200):
    \"\"\"Coordinate descent cho min ½‖y − Xw‖² + λ‖w‖₁, khởi tạo w = 0.
    Cập nhật wⱼ = S(xⱼᵀrⱼ, λ)/‖xⱼ‖², với rⱼ là phần dư khi bỏ đóng góp của cột j.\"\"\"
    # TODO
    raise NotImplementedError


def kfold_indices(n, k, rng):
    \"\"\"Xáo 0..n−1 một lần, chia thành k phần gần bằng nhau. Trả về list (train_idx, val_idx).\"\"\"
    # TODO
    raise NotImplementedError""",
"""def ridge(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def soft_threshold(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def lasso_cd(X, y, lam, n_iter=200):
    n, d = X.shape
    w = np.zeros(d)
    col_sq = (X ** 2).sum(axis=0)
    r = y - X @ w
    for _ in range(n_iter):
        for j in range(d):
            r += X[:, j] * w[j]                       # bỏ đóng góp của cột j
            w[j] = soft_threshold(X[:, j] @ r, lam) / col_sq[j]
            r -= X[:, j] * w[j]
    return w


def kfold_indices(n, k, rng):
    folds = np.array_split(rng.permutation(n), k)
    return [(np.concatenate([folds[j] for j in range(k) if j != i]), folds[i]) for i in range(k)]""")

REG_CHECK = """# (a) thiết kế trực chuẩn (ví dụ 5.5): Qᵀy = z = (3, 0.8, −2), λ = 1
Q = np.linalg.qr(rng.standard_normal((30, 3)))[0]
yq = Q @ np.array([3.0, 0.8, -2.0])
print("ridge:", ridge(Q, yq, 1.0), " lasso:", lasso_cd(Q, yq, 1.0))
assert np.allclose(ridge(Q, yq, 1.0), [1.5, 0.4, -1.0]) and np.allclose(lasso_cd(Q, yq, 1.0), [2, 0, -1])
# (b) dữ liệu thưa: 5 trên 50 hệ số khác 0
n, d = 100, 50
w_true = np.zeros(d); w_true[:5] = [3.0, -2.0, 1.5, 1.0, -1.0]
Xs = rng.standard_normal((n, d)); ys = Xs @ w_true + rng.standard_normal(n)
Xt = rng.standard_normal((1000, d)); yt = Xt @ w_true + rng.standard_normal(1000)
lams = [1, 3, 10, 30, 100]
nnz = [int(np.sum(lasso_cd(Xs, ys, l) != 0)) for l in lams]
print("số hệ số khác 0 theo λ", dict(zip(lams, nnz)))
assert all(a >= b for a, b in zip(nnz, nnz[1:]))
# (c) chọn λ bằng 5-fold CV
cv = []
for l in lams:
    errs = [np.mean((ys[va] - Xs[va] @ lasso_cd(Xs[tr], ys[tr], l)) ** 2)
            for tr, va in kfold_indices(n, 5, np.random.default_rng(1))]
    cv.append(np.mean(errs))
best = lams[int(np.argmin(cv))]
w_best = lasso_cd(Xs, ys, best)
w_ols = np.linalg.lstsq(Xs, ys, rcond=None)[0]
mse = lambda w: np.mean((yt - Xt @ w) ** 2)
print("MSE CV theo λ:", np.round(cv, 3), "→ chọn λ =", best)
print(f"test MSE: OLS = {mse(w_ols):.3f}, lasso = {mse(w_best):.3f} (nhiễu σ² = 1);",
      "hệ số khác 0:", np.flatnonzero(w_best))
assert set(range(5)) <= set(np.flatnonzero(w_best)) and mse(w_best) < mse(w_ols)
print("Lab 2 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 2 (tiếp): bảng bias–variance của ví dụ 5.4
BV_CHECK = """# Tái hiện bảng bias²–variance của ví dụ 5.4. Dùng MỘT rng chung, bậc theo thứ tự 1 → 3 → 9.
f = lambda t: np.sin(2 * np.pi * t)
xtr, xte = np.linspace(0, 1, 20), np.linspace(0.05, 0.95, 100)
rng_bv = np.random.default_rng(0)
table = {}
for deg in (1, 3, 9):
    P = np.array([np.polyval(np.polyfit(xtr, f(xtr) + 0.3 * rng_bv.standard_normal(20), deg), xte) for _ in range(2000)])
    sim = (np.mean((P.mean(0) - f(xte)) ** 2), np.mean(P.var(0)))
    # giá trị chính xác, không phụ thuộc rng: f̂ là phép chiếu tuyến tính của nhãn, Var = σ²·mean(‖hàng của A‖²)
    A = np.vander(xte, deg + 1) @ np.linalg.pinv(np.vander(xtr, deg + 1))
    exact = (np.mean((A @ f(xtr) - f(xte)) ** 2), 0.09 * np.mean((A ** 2).sum(1)))
    table[deg] = (sim, exact)
    print(f"bậc {deg}: mô phỏng bias² = {sim[0]:.4f}, variance = {sim[1]:.4f} | chính xác {exact[0]:.4f} / {exact[1]:.4f}")
assert [round(table[d][0][0], 4) for d in (1, 3, 9)] == [0.1809, 0.0054, 0.0] and \
       [round(table[d][0][1], 4) for d in (1, 3, 9)] == [0.0078, 0.0136, 0.0380]
assert all(abs(a - b) < 2e-3 for d in table for a, b in zip(*table[d]))      # sai số mô phỏng cỡ 10⁻³
print("Lab 2 (bias–variance) ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 3
MET = ("""def precision_recall_f1(y, yhat):
    \"\"\"y, yhat ∈ {0, 1}. Trả về (precision, recall, f1).\"\"\"
    # TODO
    raise NotImplementedError


def auc_pairs(y, s):
    \"\"\"AUC = tỉ lệ cặp (dương, âm) mà điểm dương > điểm âm; cặp hòa tính 0.5.\"\"\"
    # TODO
    raise NotImplementedError


def roc_curve(y, s):
    \"\"\"Trả về (FPR, TPR) khi hạ ngưỡng qua từng giá trị điểm phân biệt, bắt đầu từ (0, 0).
    Các mẫu có cùng điểm phải được thêm vào CÙNG MỘT bước.\"\"\"
    # TODO
    raise NotImplementedError


def auc_trapz(fpr, tpr):
    # TODO: diện tích dưới đường gấp khúc bằng quy tắc hình thang
    raise NotImplementedError


def average_precision(y, s):
    \"\"\"AP không nội suy = tổng (Rₙ − Rₙ₋₁)·Pₙ qua các ngưỡng phân biệt.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def precision_recall_f1(y, yhat):
    tp = np.sum((y == 1) & (yhat == 1)); fp = np.sum((y == 0) & (yhat == 1)); fn = np.sum((y == 1) & (yhat == 0))
    p, r = tp / (tp + fp), tp / (tp + fn)
    return p, r, 2 * p * r / (p + r)


def auc_pairs(y, s):
    pos, neg = s[y == 1][:, None], s[y == 0][None, :]
    return np.mean((pos > neg) + 0.5 * (pos == neg))


def _cum_counts(y, s):
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]   # chỉ số cuối mỗi nhóm điểm bằng nhau
    tp = np.cumsum(y)[last]
    fp = (last + 1) - tp
    return tp, fp


def roc_curve(y, s):
    tp, fp = _cum_counts(y, s)
    return np.r_[0, fp / np.sum(y == 0)], np.r_[0, tp / np.sum(y == 1)]


def auc_trapz(fpr, tpr):
    return np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)


def average_precision(y, s):
    tp, fp = _cum_counts(y, s)
    recall, precision = tp / np.sum(y == 1), tp / (tp + fp)
    return np.sum(np.diff(np.r_[0, recall]) * precision)""")

MET_CHECK = """# (a) số của ví dụ 5.7
s8 = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.2]); y8 = np.array([1, 1, 0, 1, 0, 0, 1, 0])
print(f"AUC đếm cặp = {auc_pairs(y8, s8):.4f}, hình thang = {auc_trapz(*roc_curve(y8, s8)):.4f}, "
      f"AP = {average_precision(y8, s8):.4f}")
assert np.isclose(auc_pairs(y8, s8), 0.75) and np.isclose(average_precision(y8, s8), 0.830357, atol=1e-6)
# (b) ví dụ 5.6: lớp dương 1%
yb = np.r_[np.ones(100), np.zeros(9900)].astype(int)
yhat = np.r_[np.ones(90), np.zeros(10), np.ones(495), np.zeros(9405)].astype(int)
p, r, f = precision_recall_f1(yb, yhat)
print(f"precision = {p:.4f}, recall = {r:.4f}, F1 = {f:.4f}")
assert np.isclose(p, 0.1538, atol=1e-4) and np.isclose(f, 0.2628, atol=1e-4)
# (c) có điểm hòa: hai cách tính AUC vẫn phải bằng nhau
yr = rng.integers(0, 2, 500); sr = np.round(rng.random(500) + 0.3 * yr, 1)
assert np.isclose(auc_pairs(yr, sr), auc_trapz(*roc_curve(yr, sr)))
# (d) cùng bộ phân loại, đổi tỉ lệ lớp dương: AUC giữ nguyên, AP sụp
for n_pos in (5000, 100):
    yy = np.r_[np.ones(n_pos), np.zeros(10000 - n_pos)].astype(int)
    ss = np.where(yy == 1, 2.0, 0.0) + rng.standard_normal(10000)
    print(f"tỉ lệ dương {n_pos / 100:4.0f}%: ROC-AUC = {auc_pairs(yy, ss):.3f}, AP = {average_precision(yy, ss):.3f}")
print("Lab 3 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 4
GB = ("""def fit_stump(X, r):
    \"\"\"Stump hồi quy: chọn (đặc trưng j, ngưỡng t) cực tiểu tổng bình phương sai số khi dự đoán
    trung bình của r ở mỗi bên. Ngưỡng là trung điểm giữa hai giá trị liên tiếp đã sắp xếp.
    Trả về (j, t, giá trị trái, giá trị phải).\"\"\"
    # TODO
    raise NotImplementedError


def predict_stump(stump, X):
    # TODO
    raise NotImplementedError


def boost(X, y, n_rounds, eta, X_test=None, y_test=None):
    \"\"\"Gradient boosting với MSE: F₀ = trung bình y; mỗi vòng khớp stump với phần dư y − F,
    rồi F ← F + η·stump. Trả về (danh sách MSE train sau mỗi vòng, danh sách MSE test hoặc None).\"\"\"
    # TODO
    raise NotImplementedError""",
"""def fit_stump(X, r):
    best = (np.inf, None)
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j]); xs, rs = X[order, j], r[order]
        cs, cs2, n = np.cumsum(rs), np.cumsum(rs ** 2), len(rs)
        for i in range(1, n):                          # trái = i phần tử đầu
            if xs[i] == xs[i - 1]:
                continue
            sl, sr = cs[i - 1], cs[-1] - cs[i - 1]
            sse = cs2[-1] - sl ** 2 / i - sr ** 2 / (n - i)
            if sse < best[0] - 1e-12:
                best = (sse, (j, (xs[i] + xs[i - 1]) / 2, sl / i, sr / (n - i)))
    return best[1]


def predict_stump(stump, X):
    j, t, left, right = stump
    return np.where(X[:, j] < t, left, right)


def boost(X, y, n_rounds, eta, X_test=None, y_test=None):
    F = np.full(len(y), y.mean())
    Ft = None if X_test is None else np.full(len(X_test), y.mean())
    tr, te = [], []
    for _ in range(n_rounds):
        st = fit_stump(X, y - F)                       # MSE: âm gradient = phần dư
        F += eta * predict_stump(st, X)
        tr.append(np.mean((y - F) ** 2))
        if Ft is not None:
            Ft += eta * predict_stump(st, X_test)
            te.append(np.mean((y_test - Ft) ** 2))
    return tr, (te if Ft is not None else None)""")

GB_CHECK = """# (a) ví dụ 5.12
x6 = np.arange(1.0, 7.0)[:, None]; y6 = np.array([1.0, 1.2, 1.1, 3.0, 3.2, 2.9])
j6, t6, l6, r6 = fit_stump(x6, y6 - y6.mean())
print(f"stump vòng 1: cột {int(j6)}, ngưỡng {float(t6)}, trái {float(l6):.4f}, phải {float(r6):.4f}")
tr6, _ = boost(x6, y6, 3, 0.5)
print("MSE ba vòng đầu:", np.round(tr6, 4))
assert np.allclose(tr6, [0.244722, 0.069514, 0.025712], atol=1e-6)
# (b) dữ liệu sin nhiễu: MSE train giảm mãi, MSE test có đáy
xa = rng.uniform(0, 1, (80, 1)); ya = np.sin(2 * np.pi * xa[:, 0]) + 0.3 * rng.standard_normal(80)
xb = rng.uniform(0, 1, (2000, 1)); yb_ = np.sin(2 * np.pi * xb[:, 0]) + 0.3 * rng.standard_normal(2000)
tr, te = boost(xa, ya, 400, 0.5, xb, yb_)
k = int(np.argmin(te))
print(f"η = 0.5: MSE test nhỏ nhất {te[k]:.4f} ở vòng {k + 1}; vòng 400: train {tr[-1]:.4f}, test {te[-1]:.4f}")
tr2, te2 = boost(xa, ya, 400, 0.05, xb, yb_)
k2 = int(np.argmin(te2))
print(f"η = 0.05: MSE test nhỏ nhất {te2[k2]:.4f} ở vòng {k2 + 1}; vòng 400: test {te2[-1]:.4f}")
assert np.all(np.diff(tr) <= 1e-12) and te[-1] > te[k] + 0.01
print("Lab 4 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 5
CLU = ("""def kmeans(X, C0, n_iter=100):
    \"\"\"X ∈ ℝᴺˣᵈ, C0 ∈ ℝᴷˣᵈ tâm khởi tạo. Lặp gán ↔ cập nhật đến khi phép gán không đổi.
    Trả về (tâm, nhãn, lịch sử J). Cụm rỗng: giữ nguyên tâm cũ.\"\"\"
    # TODO
    raise NotImplementedError


def kmeans_pp_init(X, K, rng):
    \"\"\"k-means++: tâm đầu chọn đều; tâm tiếp theo chọn với xác suất tỉ lệ D(x)²
    (bình phương khoảng cách tới tâm gần nhất đã chọn).\"\"\"
    # TODO
    raise NotImplementedError


def em_gmm_diag(X, mu, var, pi, n_iter=50, reg=1e-6):
    \"\"\"EM cho GMM hiệp phương sai chéo. mu, var ∈ ℝᴷˣᵈ, pi ∈ ℝᴷ.
    E-step tính γ ∈ ℝᴺˣᴷ (làm trong miền log cho ổn định); M-step cập nhật π, μ, σ² (+ reg).
    Trả về (mu, var, pi, gamma, lịch sử log-likelihood tính TRƯỚC mỗi vòng).\"\"\"
    # TODO
    raise NotImplementedError""",
"""def kmeans(X, C0, n_iter=100):
    C = np.array(C0, dtype=float)
    labels, J = None, []
    for _ in range(n_iter):
        D = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        new = D.argmin(1)
        J.append(D[np.arange(len(X)), new].sum())
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for k in range(len(C)):
            if np.any(labels == k):
                C[k] = X[labels == k].mean(0)
    return C, labels, np.array(J)


def kmeans_pp_init(X, K, rng):
    C = [X[rng.integers(len(X))]]
    for _ in range(K - 1):
        d2 = ((X[:, None, :] - np.array(C)[None]) ** 2).sum(-1).min(1)
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.array(C)


def _log_joint(X, mu, var, pi):
    # log πₖ + log 𝒩(xᵢ; μₖ, diag(σ²ₖ)),  kích thước N × K
    return (np.log(pi)[None] - 0.5 * np.sum(np.log(2 * np.pi * var), 1)[None]
            - 0.5 * (((X[:, None, :] - mu[None]) ** 2) / var[None]).sum(-1))


def em_gmm_diag(X, mu, var, pi, n_iter=50, reg=1e-6):
    mu, var, pi = (np.array(a, dtype=float) for a in (mu, var, pi))
    ll = []
    for _ in range(n_iter):
        L = _log_joint(X, mu, var, pi)
        m = L.max(1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(L - m).sum(1))    # log-sum-exp
        ll.append(lse.sum())
        gamma = np.exp(L - lse[:, None])                # E-step
        Nk = gamma.sum(0)                                # M-step
        pi = Nk / len(X)
        mu = (gamma.T @ X) / Nk[:, None]
        var = (gamma.T @ X ** 2) / Nk[:, None] - mu ** 2 + reg
    return mu, var, pi, gamma, np.array(ll)""")

CLU_CHECK = """# (a) ví dụ 5.13: hai khởi tạo, hai cực tiểu địa phương
X7 = np.array([1.0, 2, 3, 8, 9, 10, 25])[:, None]
for C0 in ([[1.0], [10.0]], [[1.0], [25.0]]):
    C, lab, J = kmeans(X7, C0)
    print(f"khởi tạo {np.ravel(C0)} → tâm {C.ravel()}, J = {J[-1]:.1f}")
best = min(kmeans(X7, kmeans_pp_init(X7, 2, np.random.default_rng(s)))[2][-1] for s in range(10))
print("tốt nhất trong 10 lần k-means++:", best)
assert np.isclose(kmeans(X7, [[1.0], [10.0]])[2][-1], 196) and np.isclose(best, 77.5)
# (b) J không tăng trên dữ liệu 2 chiều 3 cụm
Xc = np.concatenate([rng.standard_normal((150, 2)) * 0.6 + c for c in ([0, 0], [4, 1], [1, 4])])
_, _, J = kmeans(Xc, kmeans_pp_init(Xc, 3, rng))
assert np.all(np.diff(J) <= 1e-9)
# (c) ví dụ 5.15: một vòng EM
X5 = np.array([0.0, 1, 2.5, 4, 5])[:, None]
mu1, var1, pi1, g1, ll1 = em_gmm_diag(X5, [[0.0], [5.0]], [[1.0], [1.0]], [0.5, 0.5], n_iter=1, reg=0.0)
print("γ thành phần 1:", g1[:, 0], " μ sau 1 vòng:", mu1.ravel(), " σ²:", var1.ravel(), " ll trước:", ll1)
assert np.allclose(mu1.ravel(), [0.9007, 4.0993], atol=1e-4) and np.isclose(ll1[0], -11.4912, atol=1e-4)
# (d) log-likelihood không giảm trên dữ liệu 2 chiều
mu, var, pi, g, ll = em_gmm_diag(Xc, kmeans_pp_init(Xc, 3, rng), np.ones((3, 2)), np.ones(3) / 3, n_iter=60)
print(f"log-likelihood: {ll[0]:.2f} → {ll[-1]:.2f};  μ tìm được:\\n", mu[np.argsort(mu[:, 0] + 10 * mu[:, 1])])
assert np.all(np.diff(ll) >= -1e-9)
print("Lab 5 ĐẠT ✔")"""

# ---------------------------------------------------------------- Săn bug
BUG_DATA = """def nearest_centroid(Xtr, ytr, Xte):
    m0, m1 = Xtr[ytr == 0].mean(0), Xtr[ytr == 1].mean(0)
    return (((Xte - m1) ** 2).sum(1) < ((Xte - m0) ** 2).sum(1)).astype(int)


def top_k_corr(X, y, k):
    \"\"\"Chỉ số k cột có |tương quan| với y lớn nhất (dùng trên X đã chuẩn hóa: khi đó |hiệp phương sai| tỉ lệ với |tương quan|).\"\"\"
    c = np.abs((X - X.mean(0)).T @ (y - y.mean()))
    return np.argsort(-c)[:k]


def cv_buggy(X, y, ks=(5, 10, 20, 50), n_folds=5, seed=0):
    X = (X - X.mean(0)) / X.std(0)
    folds = np.array_split(np.random.default_rng(seed).permutation(len(y)), n_folds)
    scores = {}
    for k in ks:
        sel = top_k_corr(X, y, k)
        accs = []
        for i in range(n_folds):
            te = folds[i]; tr = np.setdiff1d(np.arange(len(y)), te)
            accs.append(np.mean(nearest_centroid(X[tr][:, sel], y[tr], X[te][:, sel]) == y[te]))
        scores[k] = np.mean(accs)
    best_k = max(scores, key=scores.get)
    return best_k, scores[best_k]


rb = np.random.default_rng(1)
Xb, yb = rb.standard_normal((100, 5000)), rb.integers(0, 2, 100)   # nhãn tung đồng xu
kb, accb = cv_buggy(Xb, yb)
print(f"bản lỗi: k tốt nhất = {kb}, accuracy CV = {float(accb):.3f}")"""

BUGFIX = ("""def cv_fixed(X, y, ks=(5, 10, 20, 50), n_folds=5, seed=0):
    \"\"\"Sửa cả ba lỗi của cv_buggy. Trả về accuracy ước lượng trung thực.
    Gợi ý: mọi thứ 'học từ dữ liệu' phải nằm trong vòng fold; chọn k bằng CV lồng bên trong.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def _fit_predict(Xtr, ytr, Xte, k):
    mu, sd = Xtr.mean(0), Xtr.std(0)                  # sửa lỗi 1: chuẩn hóa bằng thống kê của train
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    sel = top_k_corr(Xtr, ytr, k)                     # sửa lỗi 2: chọn đặc trưng trên train
    return nearest_centroid(Xtr[:, sel], ytr, Xte[:, sel])


def cv_fixed(X, y, ks=(5, 10, 20, 50), n_folds=5, seed=0):
    rngf = np.random.default_rng(seed)
    folds = np.array_split(rngf.permutation(len(y)), n_folds)
    accs = []
    for i in range(n_folds):
        te = folds[i]; tr = np.setdiff1d(np.arange(len(y)), te)
        # sửa lỗi 3: chọn k bằng CV bên trong tập train (nested CV), không nhìn fold test
        inner = np.array_split(rngf.permutation(tr), 4)
        def inner_score(k):
            return np.mean([np.mean(_fit_predict(X[np.setdiff1d(tr, f)], y[np.setdiff1d(tr, f)], X[f], k) == y[f])
                            for f in inner])
        k_best = max(ks, key=inner_score)
        accs.append(np.mean(_fit_predict(X[tr], y[tr], X[te], k_best) == y[te]))
    return np.mean(accs)""")

BUG_CHECK = """acc_fixed = cv_fixed(Xb, yb)
print(f"bản sửa: accuracy CV = {acc_fixed:.3f}")
assert 0.3 < acc_fixed < 0.7, "Nhãn ngẫu nhiên thì accuracy phải quanh 0.5"
print("Săn bug ĐẠT ✔")"""


def build(solution: bool):
    pick = (lambda pair: pair[1]) if solution else (lambda pair: pair[0])
    nb = nbf.v4.new_notebook()
    head = "Lời giải" if solution else "Bài lab"
    nb.cells = [
        md(f"# Chương 5 — {head}: ML cổ điển\n\n"
           + ("Bản lời giải đã chạy kiểm. Chỉ mở sau khi đã tự làm `lab.ipynb`."
              if solution else
              "Viết code vào các ô có `# TODO`, rồi chạy ô kiểm tra ngay bên dưới. "
              "Ô kiểm tra in **ĐẠT ✔** là xong. Bí quá 20 phút mới mở `loigiai.ipynb`.\n\n"
              "Chỉ dùng NumPy: không cần (và không được dùng) scikit-learn.")),
        code(SETUP),
        md("## Lab 1 — Softmax regression từ đầu\nSoftmax ổn định số, cross-entropy và gradient vectorized; "
           "kiểm gradient bằng sai phân hữu hạn; xem logistic không regularize trên dữ liệu tách được."),
        code(pick(SOFTMAX)), code(SOFTMAX_CHECK),
        md("## Lab 2 — Ridge, lasso và k-fold CV\nLasso bằng coordinate descent với soft-thresholding; chọn λ bằng 5-fold CV. "
           "Câu hỏi: vì sao lasso thắng OLS trên test dù OLS khớp train tốt hơn?"),
        code(pick(REG)), code(REG_CHECK),
        md("### Lab 2 (tiếp) — Tái hiện bảng bias–variance (ví dụ 5.4)\nBảng chỉ tái hiện đúng đến chữ số thứ tư nếu dùng một `default_rng(0)` chung cho cả ba bậc theo thứ tự 1 → 3 → 9."),
        code(BV_CHECK),
        md("## Lab 3 — Metric tự viết\nAUC bằng đếm cặp và bằng hình thang (kể cả khi có điểm hòa), average precision. "
           "Câu hỏi: ở phần (d), vì sao AUC không đổi mà AP sụp?"),
        code(pick(MET)), code(MET_CHECK),
        md("## Lab 4 — Gradient boosting với stump\nKhớp lại ba vòng của ví dụ 5.12, rồi xem MSE test theo số vòng với hai learning rate."),
        code(pick(GB)), code(GB_CHECK),
        md("## Lab 5 — k-means, k-means++ và EM cho GMM\nKiểm J không tăng, log-likelihood không giảm, và khớp số ví dụ 5.13, 5.15."),
        code(pick(CLU)), code(CLU_CHECK),
        md("## Săn bug\n`cv_buggy` chạy không lỗi và báo accuracy rất cao trên nhãn **tung đồng xu**. "
           "Tìm cả **ba** lỗi rò rỉ rồi viết `cv_fixed`."),
        code(BUG_DATA), code(pick(BUGFIX)), code(BUG_CHECK),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    name = "loigiai.ipynb" if solution else "lab.ipynb"
    nbf.write(nb, OUT / name)
    return OUT / name


if __name__ == "__main__":
    print(build(False), build(True))
