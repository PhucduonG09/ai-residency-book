"""Sinh hai notebook cho lab chương 11: lab.ipynb (có TODO) và loigiai.ipynb (lời giải).

Chạy lại script này nếu muốn tạo lại notebook gốc:  python labs/build_chuong11.py
Toàn bộ loigiai.ipynb chạy dưới 2 phút trên CPU (NumPy, SciPy, PyTorch CPU, seed cố định, 1 luồng).
Mọi con số trong sách (bảng "Lab code") lấy từ lần chạy của notebook này; kiểm bằng khoảng, không bằng đẳng thức,
vì kết quả huấn luyện phụ thuộc phiên bản thư viện.
"""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "chuong11"
OUT.mkdir(parents=True, exist_ok=True)

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

SETUP = r'''import time, math, itertools, inspect
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy import stats
torch.set_num_threads(1)
np.set_printoptions(precision=4, suppress=True)
T0 = time.time()

def make_world(seed=0, C=10, d=20, scale=0.6):
    """Tâm của 10 lớp Gauss trong ℝ²⁰ (nhiễu chuẩn 1 quanh mỗi tâm)."""
    return np.random.default_rng(seed).standard_normal((C, d)) * scale

def sample(centers, n, rng):
    y = rng.integers(0, len(centers), n)
    X = centers[y] + rng.standard_normal((n, centers.shape[1]))
    return X.astype(np.float32), y

def mlp(d=20, h=64, C=10, p=0.0):
    layers = [nn.Linear(d, h), nn.ReLU()]
    if p > 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(h, C))
    return nn.Sequential(*layers)

def train_eval(Xtr, ytr, Xev, yev, seed, lr=3e-3, steps=150, wd=0.0, p=0.0, extra=None):
    """MLP 20-64-10, AdamW, full-batch. Trả về (độ chính xác, loss) trên tập (Xev, yev);
    nếu có extra = (X2, y2) thì trả thêm độ chính xác trên tập đó (cùng một mô hình)."""
    torch.manual_seed(seed)
    m = mlp(Xtr.shape[1], 64, 10, p)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    for _ in range(steps):
        m.train(); opt.zero_grad()
        F.cross_entropy(m(Xt), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        lo = m(torch.from_numpy(Xev))
        acc2 = None if extra is None else float((m(torch.from_numpy(extra[0])).argmax(1).numpy() == extra[1]).mean())
    out = (float((lo.argmax(1).numpy() == yev).mean()), float(F.cross_entropy(lo, torch.from_numpy(yev))))
    return out if extra is None else out + (acc2,)

centers = make_world(0)
Xtr, ytr = sample(centers, 300, np.random.default_rng(1))   # tập train chỉ 300 mẫu: mô hình overfit, regularization có tác dụng thật

def splits(*sizes):
    """Các tập val/test lấy ngay sau tập train từ cùng một luồng ngẫu nhiên (seed 1), nên cố định và không trùng train."""
    r = np.random.default_rng(1); sample(centers, 300, r)
    return [sample(centers, n, r) for n in sizes]
print("train", Xtr.shape, "| thời gian một lần huấn luyện:", end=" ")
t = time.time(); train_eval(Xtr, ytr, Xtr, ytr, 0); print(f"{time.time() - t:.2f} s")'''

# ---------------------------------------------------------------- Lab 1
READ = (r'''def sign_test_p(k, n):
    """P(X ≥ k) với X ~ Binomial(n, 0.5): xác suất thắng ít nhất k trên n cột nếu mỗi cột là tung đồng xu.
    Tự tính bằng math.comb, không gọi scipy."""
    # TODO
    raise NotImplementedError

def consistent_sizes(pct, decimals=2, nmax=3000):
    """Danh sách các n ≤ nmax sao cho tồn tại số nguyên k (0 ≤ k ≤ n) với round(100·k/n, decimals) == pct.
    Dùng để hỏi: 'độ chính xác báo cáo là pct%, tập test có thể có bao nhiêu câu?'"""
    # TODO
    raise NotImplementedError''',
r'''def sign_test_p(k, n):
    return sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n

def consistent_sizes(pct, decimals=2, nmax=3000):
    out = []
    for n in range(1, nmax + 1):
        k0 = int(math.floor(pct * n / 100))
        if any(0 <= k <= n and abs(round(100 * k / n, decimals) - pct) < 1e-9 for k in (k0, k0 + 1)):
            out.append(n)
    return out''')

READ_CHECK = r'''print("P(≥11/18) =", round(sign_test_p(11, 18), 4), "| P(≥14/18) =", round(sign_test_p(14, 18), 4),
      "| P(≥5/6) =", round(sign_test_p(5, 6), 4))
assert abs(sign_test_p(11, 18) - 0.2403) < 1e-4 and abs(sign_test_p(14, 18) - 0.0154) < 1e-4
assert abs(sign_test_p(5, 6) - 0.1094) < 1e-4 and abs(sign_test_p(10, 10) - 2 ** -10) < 1e-12

s = consistent_sizes(85.37)
print("85.37% → n nhỏ nhất:", s[:8], "| số n ≤ 3000 phù hợp:", len(s), "| 1000 có trong danh sách?", 1000 in s)
assert s[0] == 41 and 294 in s and 1000 not in s and len(s) == 451
assert consistent_sizes(92.31)[0] == 13
assert 1000 in consistent_sizes(85.3, decimals=1)       # một chữ số thập phân thì n = 1000 hợp lệ

macro = np.mean([88.1, 91.4, 76.3, 84.0])
print("trung bình bốn cột =", macro, "(báo cáo 85.2)")
assert abs(macro - 84.95) < 1e-9
print("Lab 1 ĐẠT ✔")'''

# ---------------------------------------------------------------- Lab 2
EVP = (r'''def evp(scores, k):
    """Hiệu năng kỳ vọng theo ngân sách: kỳ vọng của max trong k lần rút CÓ HOÀN LẠI từ các điểm validation `scores`.
    Tự dẫn công thức từ phân phối của max (bài tập ★★★ số 11), rồi đối chiếu với Monte Carlo ở ô dưới."""
    # TODO
    raise NotImplementedError''',
r'''def evp(scores, k):
    v = np.sort(np.asarray(scores, dtype=float))
    n = len(v); i = np.arange(1, n + 1)
    return float(np.sum(v * ((i / n) ** k - ((i - 1) / n) ** k)))''')

EVP_RUN = r'''# Hai không gian tìm kiếm ngẫu nhiên, mỗi bên N lần thử (một seed mỗi lần thử), chọn bằng validation 4000 mẫu.
(Xva4, yva4), = splits(4000)
N_TRIALS = 30

def random_search(full, n_trials, seed):
    """full=False: không gian A (lr, số bước). full=True: không gian B thêm dropout và weight decay."""
    R = np.random.default_rng(seed)
    scores = []
    for i in range(n_trials):
        kw = dict(lr=10 ** R.uniform(-4, -1.5), steps=int(R.choice([40, 150, 400])))
        if full:
            kw.update(p=float(R.uniform(0, 0.6)), wd=float(R.choice([0, 0.1, 1.0])))
        scores.append(train_eval(Xtr, ytr, Xva4, yva4, i, **kw)[0])
    return np.array(scores)

t = time.time()
A = random_search(False, N_TRIALS, 123)
B = random_search(True, N_TRIALS, 123)
print(f"{2 * N_TRIALS} lần huấn luyện mất {time.time() - t:.1f} s")

def evp_mc(scores, k, reps=20000, seed=0):
    return float(np.mean(np.max(np.random.default_rng(seed).choice(scores, (reps, k)), axis=1)))

ks = [1, 2, 4, 8, 16, N_TRIALS]
print("k      :", ks)
print("EVP A  :", [round(evp(A, k), 4) for k in ks])
print("EVP B  :", [round(evp(B, k), 4) for k in ks])
for name, s in (("A", A), ("B", B)):
    assert abs(evp(s, 1) - s.mean()) < 1e-9
    curve = [evp(s, k) for k in range(1, N_TRIALS + 1)]
    assert all(b >= a - 1e-12 for a, b in zip(curve, curve[1:])), "EVP phải không giảm theo k"
    assert evp(s, 10 ** 4) <= s.max() + 1e-12 and evp(s, 10 ** 4) > s.max() - 0.01
    for k in (2, 8, 32):
        assert abs(evp(s, k) - evp_mc(s, k)) < 2e-3, f"EVP lệch Monte Carlo ở k = {k}"
gap_unfair = evp(B, N_TRIALS) - evp(A, 4)
gap_fair = evp(B, N_TRIALS) - evp(A, N_TRIALS)
print(f"B với {N_TRIALS} lần thử so với A với 4 lần thử: {gap_unfair * 100:+.2f} điểm")
print(f"B so với A cùng {N_TRIALS} lần thử:            {gap_fair * 100:+.2f} điểm")
assert gap_unfair > 0.015 and abs(gap_fair) < 0.01
print("Lab 2 ĐẠT ✔")'''

# ---------------------------------------------------------------- Lab 3
VAR = (r'''def mean_ci(x, use_t=True):
    """Trả về (trung bình, cận dưới, cận trên) của khoảng tin cậy 95% cho trung bình, từ các điểm x (một số cho mỗi seed).
    SE = std mẫu (mẫu số n − 1) / √n. Hệ số: phân vị t với n − 1 bậc tự do nếu use_t, ngược lại 1.96."""
    # TODO
    raise NotImplementedError

def coverage(n, use_t, reps=100000, seed=0):
    """Mô phỏng: lấy mẫu chuẩn chuẩn cỡ n (tâm thật = 0), dựng khoảng như trên.
    Trả về tỉ lệ khoảng chứa 0. Viết vectơ hóa, không vòng lặp Python."""
    # TODO
    raise NotImplementedError''',
r'''def mean_ci(x, use_t=True):
    x = np.asarray(x, dtype=float); n = len(x)
    se = x.std(ddof=1) / np.sqrt(n)
    crit = stats.t.ppf(0.975, n - 1) if use_t else 1.96
    return x.mean(), x.mean() - crit * se, x.mean() + crit * se

def coverage(n, use_t, reps=100000, seed=0):
    z = np.random.default_rng(seed).standard_normal((reps, n))
    se = z.std(axis=1, ddof=1) / np.sqrt(n)
    crit = stats.t.ppf(0.975, n - 1) if use_t else 1.96
    return float(np.mean(np.abs(z.mean(axis=1)) <= crit * se))''')

VAR_RUN = r'''(Xte1k, yte1k), = splits(1000)                      # MỘT tập test 1000 mẫu, cố định
Xnew, ynew = sample(centers, 20000, np.random.default_rng(999))   # 20 000 mẫu mới: xấp xỉ độ chính xác "thật"
kw = dict(lr=3e-3, steps=150, p=0.5, wd=1.0)
t = time.time()
res = [train_eval(Xtr, ytr, Xte1k, yte1k, s, extra=(Xnew, ynew), **kw) for s in range(20)]
acc1k = np.array([r_[0] for r_ in res]) * 100            # độ chính xác của từng seed trên test 1000 mẫu
accnew = np.array([r_[2] for r_ in res]) * 100           # cùng các mô hình đó trên 20 000 mẫu mới
print(f"20 seed (mỗi mô hình đo trên cả hai tập) mất {time.time() - t:.1f} s")

print(f"20 seed trên test 1000 mẫu: {acc1k.mean():.2f} ± {acc1k.std(ddof=1):.2f} (std mẫu); std tổng thể {acc1k.std(ddof=0):.2f}")
first5 = acc1k[:5]
for use_t in (True, False):
    m, lo, hi = mean_ci(first5, use_t)
    print(f"5 seed đầu, CI {'t   ' if use_t else '1.96'}: [{lo:.2f}, {hi:.2f}]  (rộng {hi - lo:.2f})")
cov_t, cov_z = coverage(5, True), coverage(5, False)
print(f"độ phủ khi n = 5: khoảng t {cov_t:.3f}, khoảng 1.96 {cov_z:.3f}")

p_hat = accnew.mean() / 100
se_binom = 100 * math.sqrt(p_hat * (1 - p_hat) / 1000)
gap = acc1k.mean() - accnew.mean()
print(f"độ chính xác trên 20 000 mẫu mới: {accnew.mean():.2f} (std theo seed {accnew.std(ddof=1):.2f})")
print(f"test 1000 mẫu lệch {gap:+.2f} điểm so với giá trị mới = {gap / se_binom:+.2f} SE nhị thức ({se_binom:.2f})")
lo5, hi5 = mean_ci(first5, True)[1:]
print("giá trị mới nằm trong CI t của 5 seed đầu?", lo5 <= accnew.mean() <= hi5)

assert acc1k.std(ddof=1) > acc1k.std(ddof=0)
assert abs(mean_ci([1, 2, 3, 4, 5], True)[1] - (3 - 2.776 * np.sqrt(2.5) / np.sqrt(5))) < 1e-3
assert 0.94 < cov_t < 0.96 and 0.85 < cov_z < 0.90
assert abs(gap) < 3 * se_binom
print("Lab 3 ĐẠT ✔")'''

# ---------------------------------------------------------------- Lab 4
LEAK_DATA = r'''def select(X, y, k):
    """Chọn k đặc trưng có |tương quan| lớn nhất với nhãn."""
    Xc = X - X.mean(0); yc = (2 * y - 1).astype(float); yc -= yc.mean()
    sc = np.abs(Xc.T @ yc) / (np.linalg.norm(Xc, axis=0) * np.linalg.norm(yc) + 1e-12)
    return np.argsort(-sc)[:k]

def fit_predict(Xa, ya, Xb):
    """Chuẩn hóa bằng thống kê của Xa, hồi quy bình phương tối thiểu có L2 (λ = 1), dự đoán nhãn cho Xb."""
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
    A = np.c_[(Xa - mu) / sd, np.ones(len(Xa))]
    w = np.linalg.solve(A.T @ A + np.eye(A.shape[1]), A.T @ (2 * ya - 1.0))
    return ((np.c_[(Xb - mu) / sd, np.ones(len(Xb))] @ w) > 0).astype(int)'''

LEAK = (r'''def cv_accuracy(X, y, k, select_inside, folds=5, seed=0):
    """Độ chính xác kiểm định chéo `folds` fold của quy trình 'chọn k đặc trưng → fit_predict'.
    select_inside=False: chọn đặc trưng MỘT lần trên toàn bộ (X, y) rồi mới chia fold.
    select_inside=True : chọn lại đặc trưng trong từng fold. Hoán vị chỉ số bằng np.random.default_rng(seed)."""
    # TODO
    raise NotImplementedError''',
r'''def cv_accuracy(X, y, k, select_inside, folds=5, seed=0):
    n = len(y)
    idx = np.random.default_rng(seed).permutation(n)
    parts = np.array_split(idx, folds)
    S_all = None if select_inside else select(X, y, k)
    acc = []
    for f in parts:
        tr = np.setdiff1d(idx, f)
        S = select(X[tr], y[tr], k) if select_inside else S_all
        acc.append(np.mean(fit_predict(X[tr][:, S], y[tr], X[f][:, S]) == y[f]))
    return float(np.mean(acc))''')

LEAK_RUN = r'''n, p, k, REPS = 100, 5000, 20, 15
leak, right, fresh = [], [], []
t = time.time()
for rep in range(REPS):
    r = np.random.default_rng(rep)
    X = r.standard_normal((n, p)); y = r.integers(0, 2, n)       # nhãn NGẪU NHIÊN: độ chính xác thật phải là 0.5
    leak.append(cv_accuracy(X, y, k, select_inside=False))
    right.append(cv_accuracy(X, y, k, select_inside=True))
    Xn = r.standard_normal((2000, p)); yn = r.integers(0, 2, 2000)
    S = select(X, y, k)                                          # mô hình "rò rỉ" thử trên dữ liệu hoàn toàn mới
    fresh.append(np.mean(fit_predict(X[:, S], y, Xn[:, S]) == yn))
print(f"{REPS} lần lặp mất {time.time() - t:.1f} s")
print(f"chọn trước khi chia : CV {np.mean(leak):.3f} (std {np.std(leak, ddof=1):.3f})")
print(f"chọn trong từng fold: CV {np.mean(right):.3f} (std {np.std(right, ddof=1):.3f})")
print(f"mô hình rò rỉ trên 2000 mẫu mới: {np.mean(fresh):.3f}")
assert np.mean(leak) >= 0.75
assert 0.44 <= np.mean(right) <= 0.56
assert 0.46 <= np.mean(fresh) <= 0.54
print("Lab 4 ĐẠT ✔")'''

# ---------------------------------------------------------------- Lab 5
REUSE = (r'''def best_of_k_bias(K, n=1000, p=0.8, reps=4000, seed=0):
    """K mô hình có độ chính xác thật đều bằng p. Mỗi mô hình được đo trên cùng loại tập test n mẫu
    (số câu đúng ~ Binomial(n, p), độc lập giữa các mô hình). Chọn mô hình có số đo cao nhất rồi báo cáo số đó.
    Trả về thiên lệch trung bình (điểm phần trăm) = 100 · (số đo cao nhất − p), lấy trung bình qua `reps` lần."""
    # TODO
    raise NotImplementedError

def p_detect(L, n, q):
    """Xác suất bắt được bản sao: tài liệu dài L token bị sao chép, mỗi token bị đổi độc lập với xác suất q;
    'bắt được' nghĩa là còn ít nhất một đoạn n token liên tiếp không bị đổi. Tính CHÍNH XÁC bằng quy hoạch động
    trên độ dài đoạn không đổi hiện tại (0 … n−1); đoạn đạt n thì rơi ra khỏi trạng thái."""
    # TODO
    raise NotImplementedError

def ngrams(doc, n):
    return {tuple(doc[i:i + n]) for i in range(len(doc) - n + 1)}

def ngram_flag(doc, train_ngrams, n):
    """True nếu doc có ít nhất một n-gram nằm trong tập train_ngrams (tập các tuple)."""
    # TODO
    raise NotImplementedError''',
r'''def best_of_k_bias(K, n=1000, p=0.8, reps=4000, seed=0):
    R = np.random.default_rng(seed)
    obs = R.binomial(n, p, (reps, K)) / n
    return float(100 * (obs.max(axis=1) - p).mean())

def p_detect(L, n, q):
    s = np.zeros(n); s[0] = 1.0                     # s[j] = P(đoạn không đổi hiện tại dài j, chưa có đoạn ≥ n)
    for _ in range(L):
        ns = np.zeros(n)
        ns[0] = s.sum() * q                         # token này bị đổi: đoạn về 0
        for j in range(n - 1):
            ns[j + 1] = s[j] * (1 - q)              # token giữ nguyên: đoạn dài thêm 1 (j = n−1 → đạt n, rơi ra)
        s = ns
    return float(1 - s.sum())

def ngrams(doc, n):
    return {tuple(doc[i:i + n]) for i in range(len(doc) - n + 1)}

def ngram_flag(doc, train_ngrams, n):
    return len(ngrams(doc, n) & train_ngrams) > 0''')

REUSE_RUN = r'''# (a) Dùng lại test set: chọn tốt nhất trong K mô hình có chất lượng thật như nhau
for K in (1, 5, 20, 100, 1000):
    emax = {1: 0.0, 5: 1.163, 20: 1.867, 100: 2.508, 1000: 3.241}[K]       # E[max của K biến chuẩn chuẩn]
    print(f"K = {K:4d}: thiên lệch {best_of_k_bias(K):+.2f} điểm   (xấp xỉ chuẩn SE·E[max] = {1.2649 * emax:+.2f})")
b100 = best_of_k_bias(100)
assert 2.5 <= b100 <= 3.8 and best_of_k_bias(1) < 0.2 and best_of_k_bias(1000) > best_of_k_bias(100) > best_of_k_bias(5)

# (b) Phát hiện nhiễm bằng n-gram: xác suất chính xác (DP) và mô phỏng
print("DP, L = 60:")
for n_ in (5, 8, 13):
    print(f"  n = {n_:2d}:", [round(p_detect(60, n_, q), 3) for q in (0.02, 0.05, 0.1, 0.2, 0.3)])
assert abs(p_detect(60, 13, 0.2) - 0.487) < 1e-3
assert abs(p_detect(200, 13, 0.2) - 0.918) < 1e-3
assert abs(p_detect(60, 5, 0.0) - 1.0) < 1e-12

R = np.random.default_rng(0)
V, L, DOCS = 5000, 60, 300
orig = R.integers(0, V, (DOCS, L)); mut = orig.copy()
mask = R.random(mut.shape) < 0.2; mut[mask] = R.integers(0, V, mask.sum())
sim = np.mean([ngram_flag(mut[i], ngrams(orig[i], 13), 13) for i in range(DOCS)])
print(f"mô phỏng n = 13, q = 0.2 trên {DOCS} tài liệu: {sim:.3f} (chính xác {p_detect(60, 13, 0.2):.3f})")
assert abs(sim - p_detect(60, 13, 0.2)) < 0.1

# (c) Dương tính giả: văn bản kiểu Zipf, tài liệu test ĐỘC LẬP với train
z = 1.0 / np.arange(1, V + 1) ** 1.1; z /= z.sum()
train_docs = R.choice(V, (2000, L), p=z); test_docs = R.choice(V, (100, L), p=z)
fp = {}
for n_ in (2, 5, 13):
    big = set()
    for d in train_docs: big |= ngrams(d, n_)
    fp[n_] = float(np.mean([ngram_flag(d, big, n_) for d in test_docs]))
print("tỉ lệ tài liệu độc lập bị gắn cờ:", {k_: round(v, 2) for k_, v in fp.items()})
assert fp[5] >= 0.05 and fp[13] <= 0.01 and fp[2] > 0.9
print("Lab 5 ĐẠT ✔")'''

# ---------------------------------------------------------------- Săn bug
BUG_DATA = r'''(Xva, yva), (Xte, yte) = splits(1000, 1000)     # validation 1000 mẫu, test 1000 mẫu
M = dict(p=0.5, wd=1.0)                   # "phương pháp" = dropout 0.5 + weight decay 1.0 (baseline: MLP thường)
lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
step_list = [40, 150, 400]

def run_experiment_buggy():
    base_runs = [train_eval(Xtr, ytr, Xte, yte, s, lr=3e-2, steps=150)[0] for s in range(5)]
    cfg = max(itertools.product(lrs, step_list),
              key=lambda c: train_eval(Xtr, ytr, Xte, yte, 0, lr=c[0], steps=c[1], **M)[0])
    meth_runs = [train_eval(Xtr, ytr, Xte, yte, s, lr=cfg[0], steps=cfg[1], **M)[0] for s in range(5)]
    return dict(cfg=cfg, base_runs=base_runs, meth_runs=meth_runs,
                base=float(np.mean(base_runs)), base_sd=float(np.std(base_runs)),
                meth=float(np.max(meth_runs)), meth_sd=float(np.std(meth_runs)))

t = time.time()
bug = run_experiment_buggy()
print(f"[{time.time() - t:.0f} s] baseline {100 * bug['base']:.2f} | phương pháp {100 * bug['meth']:.2f} "
      f"| chênh tuyên bố {100 * (bug['meth'] - bug['base']):+.2f} điểm")'''

FIX = (r'''def run_experiment_fixed():
    """Viết lại quy trình cho cùng câu hỏi ("dropout + weight decay có hơn MLP thường không?") sao cho kết luận đáng tin.
    Được viết thêm hàm phụ. Trả về dict gồm: cfg_b, cfg_m (cấu hình (lr, steps) cuối cùng của baseline và của phương pháp),
    base, base_sd, meth, meth_sd (điểm phần trăm), diff = meth − base và diff_ci = (cận dưới, cận trên) khoảng tin cậy 95% của diff."""
    # TODO
    raise NotImplementedError''',
r'''def tune_config(Xa, ya, Xv, yv, kw, n_seeds=3):
    score = {c: np.mean([train_eval(Xa, ya, Xv, yv, s, lr=c[0], steps=c[1], **kw)[0] for s in range(n_seeds)])
             for c in itertools.product(lrs, step_list)}
    return max(score, key=score.get)

def run_experiment_fixed():
    cfg_b = tune_config(Xtr, ytr, Xva, yva, {})          # cùng lưới, cùng validation, cùng số seed cho cả hai bên
    cfg_m = tune_config(Xtr, ytr, Xva, yva, M)
    tb = np.array([train_eval(Xtr, ytr, Xte, yte, s, lr=cfg_b[0], steps=cfg_b[1])[0] for s in range(10)]) * 100
    tm = np.array([train_eval(Xtr, ytr, Xte, yte, s, lr=cfg_m[0], steps=cfg_m[1], **M)[0] for s in range(10)]) * 100
    d = tm.mean() - tb.mean(); se = np.sqrt(tm.var(ddof=1) / 10 + tb.var(ddof=1) / 10); c = stats.t.ppf(0.975, 18)
    return dict(cfg_b=cfg_b, cfg_m=cfg_m, base=tb.mean(), base_sd=tb.std(ddof=1),
                meth=tm.mean(), meth_sd=tm.std(ddof=1), diff=d, diff_ci=(d - c * se, d + c * se))''')

KIEMTRA_PY = r'''"""Bộ kiểm cho bài săn bug chương 11: kiểm hành vi của run_experiment_fixed, không kiểm cách viết."""
import time
import numpy as np


def kiemtra_sanbug(ns):
    lrs, step_list, M = ns["lrs"], ns["step_list"], ns["M"]
    Xte, Xva = ns["Xte"], ns["Xva"]
    real = ns["train_eval"]
    calls = []

    def spy(Xa, ya, Xev, yev, seed, **kw):
        out = real(Xa, ya, Xev, yev, seed, **kw)
        cfg = (kw.get("lr", 3e-3), kw.get("steps", 150), kw.get("p", 0.0), kw.get("wd", 0.0))
        calls.append(dict(test=Xev.shape == Xte.shape and bool(np.array_equal(Xev, Xte)),
                          val=Xev.shape == Xva.shape and bool(np.array_equal(Xev, Xva)),
                          cfg=cfg, acc=out[0]))
        return out

    t = time.time()
    ns["train_eval"] = spy
    try:
        fx = ns["run_experiment_fixed"]()
    finally:
        ns["train_eval"] = real
    print(f"[{time.time() - t:.0f} s] baseline {fx['cfg_b']}: {fx['base']:.2f} ± {fx['base_sd']:.2f} | "
          f"phương pháp {fx['cfg_m']}: {fx['meth']:.2f} ± {fx['meth_sd']:.2f} | chênh {fx['diff']:+.2f}")

    def arm(c, name):
        return (c["cfg"][2:] == (0.5, 1.0)) == (name == "phương pháp")

    on_test = [c for c in calls if c["test"]]
    on_val = [c for c in calls if c["val"]]
    grid = len(lrs) * len(step_list)
    assert len({c["cfg"] for c in on_test}) <= 2, "Kiểm tra 1 không đạt: đối chiếu quy trình với đề bài."
    for name in ("baseline", "phương pháp"):
        n_cfg = len({c["cfg"] for c in on_val if arm(c, name)})
        assert n_cfg >= grid - 1, f"Kiểm tra 2 không đạt ({name}): đối chiếu quy trình với đề bài."
    for name, cfg in (("baseline", fx["cfg_b"]), ("phương pháp", fx["cfg_m"])):
        accs = np.array([c["acc"] for c in on_test if c["cfg"][:2] == tuple(cfg) and arm(c, name)]) * 100
        key = "base" if name == "baseline" else "meth"
        assert len(accs) >= 10, f"Kiểm tra 3 không đạt ({name}): đối chiếu quy trình với đề bài."
        assert abs(fx[key] - accs.mean()) < 0.01, f"Kiểm tra 4 không đạt ({name}): số báo cáo không khớp các lần chạy."
        assert abs(fx[key + "_sd"] - accs.std(ddof=1)) < 0.01, f"Kiểm tra 5 không đạt ({name}): đối chiếu cách đo độ phân tán."
    lo, hi = fx["diff_ci"]
    assert lo <= fx["diff"] <= hi and hi > lo, "Kiểm tra 6 không đạt: kết quả trả về chưa đúng hợp đồng trong docstring."
    assert -0.5 <= fx["diff"] <= 1.5, "Kiểm tra 7 không đạt: kết luận chưa hợp lý."
    assert abs(fx["diff"]) < 2 * max(fx["base_sd"], fx["meth_sd"]), "Kiểm tra 7 không đạt: kết luận chưa hợp lý."

    Xbig, ybig = ns["sample"](ns["centers"], 40000, np.random.default_rng(999))
    te = ns["train_eval"]
    pop_b = np.mean([te(ns["Xtr"], ns["ytr"], Xbig, ybig, s, lr=fx["cfg_b"][0], steps=fx["cfg_b"][1])[0] for s in range(10)]) * 100
    pop_m = np.mean([te(ns["Xtr"], ns["ytr"], Xbig, ybig, s, lr=fx["cfg_m"][0], steps=fx["cfg_m"][1], **M)[0] for s in range(10)]) * 100
    print(f"dữ liệu mới: baseline {pop_b:.2f}, phương pháp {pop_m:.2f}, chênh {pop_m - pop_b:+.2f}")
    assert abs((pop_m - pop_b) - fx["diff"]) < 1.0, "Kiểm tra 8 không đạt: so với dữ liệu mới."
    print("Săn bug ĐẠT ✔")
    return fx
'''

BUG_CHECK = r'''# Ô kiểm tra nằm trong file phụ _kiemtra.py cùng thư mục (đừng mở trước khi tự tìm đủ lỗi).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()))
from _kiemtra import kiemtra_sanbug
fx = kiemtra_sanbug(globals())
Xbig, ybig = sample(centers, 40000, np.random.default_rng(999))   # dữ liệu mới, dùng cho phần phân tích bên dưới'''

BUG_PHAN_TICH = r'''# Phân tích: lỗi nào đóng góp bao nhiêu vào "+7 điểm"?  Bật dần từng lỗi trên bản đúng.
def pop_acc(cfg, kw, seeds=5):
    return np.mean([train_eval(Xtr, ytr, Xbig, ybig, s, lr=cfg[0], steps=cfg[1], **kw)[0] for s in range(seeds)]) * 100

base_pop_weak = pop_acc((3e-2, 150), {})
print(f"độ chính xác thật (40 000 mẫu mới, 5 seed): baseline lr 3e-2 {base_pop_weak:.2f} | baseline đã tune {pop_acc(fx['cfg_b'], {}):.2f}"
      f" | phương pháp chọn theo test {pop_acc(bug['cfg'], M):.2f} | phương pháp chọn theo validation {pop_acc(fx['cfg_m'], M):.2f}")
true_meth_bug = pop_acc(bug["cfg"], M)
decomp = [
    ("tuyên bố ban đầu (best-of-5 so với mean baseline chưa tune)", 100 * (bug["meth"] - bug["base"])),
    ("thay best-of-5 bằng mean 5 seed", 100 * (np.mean(bug["meth_runs"]) - bug["base"])),
    ("đo trên 40 000 mẫu mới (cả hai bên) thay vì tập test đã dùng để chọn", true_meth_bug - base_pop_weak),
    ("thay baseline chưa tune bằng baseline đã tune (cùng lưới)", true_meth_bug - pop_acc(fx["cfg_b"], {})),
    ("thay cấu hình chọn theo test bằng cấu hình chọn theo validation", pop_acc(fx["cfg_m"], M) - pop_acc(fx["cfg_b"], {})),
]
for name, v in decomp:
    print(f"  {v:+6.2f}  {name}")
assert decomp[0][1] > 5 and decomp[-1][1] < 1.5
print(f"Tổng thời gian notebook: {time.time() - T0:.1f} s")
print("Phân tích lỗi ĐẠT ✔")'''


def build(solution: bool):
    pick = (lambda pair: pair[1]) if solution else (lambda pair: pair[0])
    nb = nbf.v4.new_notebook()
    head = "Lời giải" if solution else "Bài lab"
    nb.cells = [
        md(f"# Chương 11 — {head}: Phương pháp thực nghiệm và đọc paper\n\n"
           + ("Bản lời giải đã chạy kiểm (NumPy, SciPy, PyTorch CPU, dưới 2 phút). Chỉ mở sau khi đã tự làm `lab.ipynb`."
              if solution else
              "Viết code vào các ô có `# TODO`, rồi chạy ô kiểm tra ngay bên dưới. Ô kiểm tra in **ĐẠT ✔** là xong. "
              "Bí quá 20 phút mới mở `loigiai.ipynb`. Chỉ cần PyTorch bản CPU; toàn notebook chạy dưới 2 phút. "
              "Các assert dùng khoảng, vì kết quả huấn luyện phụ thuộc phiên bản thư viện.")),
        md("## Dữ liệu và mô hình chung\n10 lớp Gauss trong ℝ²⁰, tâm = `randn(10, 20) × 0.6`, nhiễu chuẩn 1. "
           "Tập train chỉ 300 mẫu để mô hình overfit và regularization có tác dụng thật. "
           "MLP 20–64–10, AdamW, full-batch. Đây là bài toán đồ chơi: nó dạy cấu trúc của sai lầm, "
           "không dạy kích cỡ hiệu ứng trong NLP thật."),
        code(SETUP),
        md("## Lab 1 — Đọc số liệu\nKiểm định dấu cho \"thắng k trên n cột\", và suy ra cỡ tập test có thể có "
           "từ một phần trăm được báo cáo."),
        code(pick(READ)), code(READ_CHECK),
        md("## Lab 2 — Hiệu năng kỳ vọng theo ngân sách\nCông thức theo thống kê thứ tự (bạn tự dẫn), đối chiếu bằng Monte Carlo; "
           "rồi chạy tìm kiếm ngẫu nhiên trên hai không gian có kích cỡ khác nhau và đọc đường cong."),
        code(pick(EVP)), code(EVP_RUN),
        md("## Lab 3 — Phương sai theo seed, khoảng tin cậy khi chỉ có vài seed\nDựng CI bằng phân vị t, đo độ phủ bằng mô phỏng, "
           "rồi so khoảng theo seed với độ chính xác đo trên 20 000 mẫu mới."),
        code(pick(VAR)), code(VAR_RUN),
        md("## Lab 4 — Rò rỉ do chọn đặc trưng trước khi chia\nNhãn hoàn toàn ngẫu nhiên: độ chính xác thật phải là 0.5. "
           "Quy trình nào báo cao hơn nhiều là quy trình có rò rỉ."),
        code(LEAK_DATA), code(pick(LEAK)), code(LEAK_RUN),
        md("## Lab 5 — Dùng lại test set và phát hiện nhiễm bằng n-gram\nThiên lệch của \"chọn tốt nhất trong K\", "
           "xác suất bắt được bản sao đã sửa (tính chính xác bằng quy hoạch động), và dương tính giả."),
        code(pick(REUSE)), code(REUSE_RUN),
        md("## Săn bug\nNgười viết một bài báo giả định muốn \"chứng minh\" dropout + weight decay hơn MLP thường. "
           "`run_experiment_buggy` chạy không báo lỗi. Triệu chứng: bài tuyên bố hơn khoảng 7.6 điểm; "
           "chạy lại trên dữ liệu mới thì chênh chưa tới 0.5 điểm, và baseline trong bài kém bất thường. "
           "Tìm cả **bốn** lỗi quy trình rồi viết `run_experiment_fixed`. Ô kiểm tra gọi hàm trong file phụ `_kiemtra.py` cùng thư mục: đừng mở file đó trước khi tự tìm đủ bốn lỗi."),
        code(BUG_DATA), code(pick(FIX)), code(BUG_CHECK),
    ]
    if solution:
        nb.cells += [md("## Phân tích từng lỗi\nChạy sau khi đã sửa xong: lỗi nào đóng góp bao nhiêu điểm vào con số tuyên bố."),
                     code(BUG_PHAN_TICH)]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    name = "loigiai.ipynb" if solution else "lab.ipynb"
    nbf.write(nb, OUT / name)
    (OUT / "_kiemtra.py").write_text(KIEMTRA_PY, encoding="utf-8")
    return OUT / name


if __name__ == "__main__":
    print(build(False), build(True))
