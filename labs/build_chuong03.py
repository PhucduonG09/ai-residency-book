"""Sinh hai notebook cho lab chương 3: lab.ipynb (có TODO) và loigiai.ipynb (lời giải).

Chạy lại script này nếu muốn tạo lại notebook gốc:  python labs/build_chuong03.py
"""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "chuong03"
OUT.mkdir(parents=True, exist_ok=True)

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

SETUP = """import numpy as np
from scipy import stats, special, optimize
from scipy.linalg import solve_triangular
rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True)"""

# ---- Lab 1: Bayes ---------------------------------------------------------------
BAYES = ("""def posterior(prior, tpr, fpr):
    \"\"\"P(dương thật ∣ bị gắn cờ) theo định lý Bayes.
    prior = P(dương), tpr = P(cờ ∣ dương), fpr = P(cờ ∣ âm).\"\"\"
    # TODO
    raise NotImplementedError


def simulate(prior, tpr, fpr, n, rng):
    \"\"\"Mô phỏng n mẫu: sinh nhãn thật, sinh cờ theo tpr/fpr,
    trả về tỉ lệ dương thật trong nhóm bị gắn cờ.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def posterior(prior, tpr, fpr):
    evidence = tpr * prior + fpr * (1 - prior)        # xác suất toàn phần P(cờ)
    return tpr * prior / evidence


def simulate(prior, tpr, fpr, n, rng):
    toxic = rng.random(n) < prior
    flag = np.where(toxic, rng.random(n) < tpr, rng.random(n) < fpr)
    return toxic[flag].mean()""")

BAYES_CHECK = """p1 = posterior(0.02, 0.90, 0.05)
p_sim = simulate(0.02, 0.90, 0.05, 1_000_000, np.random.default_rng(1))
p2 = posterior(p1, 0.90, 0.05)                       # posterior cũ làm prior mới
print(f"công thức: {p1:.4f} | mô phỏng 10⁶ mẫu: {p_sim:.4f} | sau bộ lọc thứ hai: {p2:.4f}")
assert abs(p1 - 0.2687) < 1e-4
assert abs(p_sim - p1) < 0.01
assert abs(p2 - 0.8686) < 1e-4
print("Lab 1 ĐẠT ✔")"""

# ---- Lab 2: Gauss nhiều chiều -------------------------------------------------------
MVN = ("""def mvn_logpdf(x, mu, Sigma):
    \"\"\"log N(x; mu, Sigma) cho một điểm x. Dùng Cholesky Sigma = LLᵀ, KHÔNG dùng inv.
    Gợi ý: giải hệ tam giác dưới Lz = x − mu bằng `solve_triangular(L, x - mu, lower=True)` (thế tiến, chi phí d²), khi đó Mahalanobis² = ‖z‖₂², log det Sigma = 2·∑ log Lᵢᵢ.\"\"\"
    # TODO
    raise NotImplementedError


def sample_mvn(mu, Sigma, n, rng):
    \"\"\"n mẫu từ N(mu, Sigma) theo x = mu + Lz, z ~ N(0, I). Trả về mảng n × d.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def mvn_logpdf(x, mu, Sigma):
    d = len(mu)
    L = np.linalg.cholesky(Sigma)
    z = solve_triangular(L, x - mu, lower=True)       # thế tiến: z = L⁻¹(x − mu), chi phí d²
    maha2 = z @ z
    logdet = 2 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2 * np.pi) + logdet + maha2)


def sample_mvn(mu, Sigma, n, rng):
    L = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((n, len(mu)))
    return mu + Z @ L.T                               # mỗi hàng: mu + L z""")

MVN_CHECK = """mu, S = np.zeros(2), np.array([[2.0, 1.0], [1.0, 2.0]])
ref = stats.multivariate_normal(mu, S)
for x in [np.array([1.0, 1.0]), np.array([1.0, -1.0]), np.array([-3.0, 0.5])]:
    assert abs(mvn_logpdf(x, mu, S) - ref.logpdf(x)) < 1e-10
X = sample_mvn(mu, S, 200_000, np.random.default_rng(2))
S_hat = np.cov(X.T)
print("Σ̂ từ 200 000 mẫu:\\n", S_hat)
maha = [x @ np.linalg.solve(S, x) for x in (np.array([1.0, 1.0]), np.array([1.0, -1.0]))]
print("Mahalanobis² của (1, 1) và (1, −1):", np.round(maha, 4))
assert np.allclose(S_hat, S, atol=0.03)
assert np.allclose(maha, [2 / 3, 2.0])
print("Lab 2 ĐẠT ✔")"""

# ---- Lab 3: log-sum-exp và cross-entropy --------------------------------------------
LSE = ("""def logsumexp(z, axis=-1):
    \"\"\"log ∑ exp(z) ổn định số: trừ max trước.\"\"\"
    # TODO
    raise NotImplementedError


def log_softmax(z, axis=-1):
    # TODO
    raise NotImplementedError


def cross_entropy(logits, y, eps=0.0):
    \"\"\"Cross-entropy trung bình (nat). logits: N × K, y: N nhãn nguyên.
    eps > 0: label smoothing, mục tiêu = (1 − eps)·one-hot + eps/K.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))).squeeze(axis)


def log_softmax(z, axis=-1):
    return z - np.expand_dims(logsumexp(z, axis=axis), axis)


def cross_entropy(logits, y, eps=0.0):
    logits = np.atleast_2d(logits)
    N, K = logits.shape
    target = np.full((N, K), eps / K)
    target[np.arange(N), y] += 1 - eps
    return -(target * log_softmax(logits)).sum(axis=1).mean()""")

LSE_CHECK = """big = np.array([1000.0, 999.0, 998.0])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(big) / np.exp(big).sum()
print("softmax ngây thơ:", naive, "| ổn định:", np.exp(log_softmax(big)))
assert np.all(np.isnan(naive)) and np.all(np.isfinite(log_softmax(big)))
Z = rng.normal(0, 50, (20, 7))
assert np.allclose(logsumexp(Z), special.logsumexp(Z, axis=-1))
logq = np.log(np.array([[0.7, 0.2, 0.1]]))          # logit = log q cho đúng q
ce0, ce_ls = cross_entropy(logq, [0]), cross_entropy(logq, [0], eps=0.1)
t = np.array([[0.9 + 0.1 / 3, 0.1 / 3, 0.1 / 3]])
floor = cross_entropy(np.log(t), [0], eps=0.1)       # dự đoán đúng bằng mục tiêu
print(f"CE = {ce0:.4f} | CE có label smoothing = {ce_ls:.4f} | sàn H(target) = {floor:.4f}")
assert abs(ce0 - 0.3567) < 1e-4 and abs(ce_ls - 0.4633) < 1e-4 and abs(floor - 0.2911) < 1e-4
print("Lab 3 ĐẠT ✔")"""

# ---- Lab 4: MLE vs MAP, ridge -------------------------------------------------------
MAP = ("""def bernoulli_mle_map(heads, n, a=2.0, b=2.0):
    \"\"\"Trả về (MLE, MAP) của p với prior Beta(a, b).\"\"\"
    # TODO
    raise NotImplementedError


def ridge_map(X, y, sigma2, tau2):
    \"\"\"MAP của w cho y = Xw + ε, ε ~ N(0, sigma2·I), prior w ~ N(0, tau2·I).
    Giải hệ (XᵀX + λI)w = Xᵀy với λ = sigma2/tau2 (dùng solve, không dùng inv).\"\"\"
    # TODO
    raise NotImplementedError


def neg_log_posterior(w, X, y, sigma2, tau2):
    \"\"\"−log p(w ∣ D) bỏ hằng số.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def bernoulli_mle_map(heads, n, a=2.0, b=2.0):
    return heads / n, (heads + a - 1) / (n + a + b - 2)


def ridge_map(X, y, sigma2, tau2):
    lam = sigma2 / tau2
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def neg_log_posterior(w, X, y, sigma2, tau2):
    r = y - X @ w
    return r @ r / (2 * sigma2) + w @ w / (2 * tau2)""")

MAP_CHECK = """print("7/10 ngửa:", bernoulli_mle_map(7, 10), "| 3/3 ngửa:", bernoulli_mle_map(3, 3))
assert np.allclose(bernoulli_mle_map(7, 10), (0.7, 2 / 3)) and np.allclose(bernoulli_mle_map(3, 3), (1.0, 0.8))
X1, y1 = np.array([[1.0], [2.0], [3.0]]), np.array([2.0, 4.0, 7.0])
w_mle = np.linalg.lstsq(X1, y1, rcond=None)[0][0]
w_map = ridge_map(X1, y1, 1.0, 0.1)[0]
print(f"ví dụ 3.8: w_MLE = {w_mle:.4f}, w_MAP = {w_map:.4f}")
assert abs(w_mle - 31 / 14) < 1e-10 and abs(w_map - 31 / 24) < 1e-10
# hai cách giải MAP phải khớp; khoảng cách MAP–MLE giảm khi n tăng
rng4 = np.random.default_rng(40)                       # rng cục bộ: số in ra không phụ thuộc thứ tự chạy ô
w_true, gaps = rng4.standard_normal(5), []
for n in [10, 100, 1000, 10_000]:
    X = rng4.standard_normal((n, 5)); y = X @ w_true + rng4.standard_normal(n)
    w_closed = ridge_map(X, y, 1.0, 0.1)
    w_opt = optimize.minimize(neg_log_posterior, np.zeros(5), args=(X, y, 1.0, 0.1), method="BFGS").x
    assert np.allclose(w_closed, w_opt, atol=1e-5)
    gaps.append(np.linalg.norm(w_closed - np.linalg.lstsq(X, y, rcond=None)[0]))
print("‖w_MAP − w_MLE‖₂ theo n = 10, 100, 1000, 10 000:", np.round(gaps, 4))
assert all(g1 > g2 for g1, g2 in zip(gaps, gaps[1:]))
print("Lab 4 ĐẠT ✔")"""

# ---- Lab 5: so sánh hai model -------------------------------------------------------
DATA5 = """# Đúng/sai của hai model trên cùng 1000 câu test (bảng chéo của ví dụ 3.11), xáo thứ tự
a = np.r_[np.ones(820), np.ones(30), np.zeros(15), np.zeros(135)]
b = np.r_[np.ones(820), np.zeros(30), np.ones(15), np.zeros(135)]
perm = np.random.default_rng(3).permutation(1000)
a, b = a[perm], b[perm]
print("acc A =", a.mean(), "| acc B =", b.mean())"""

CMP = ("""def wald_ci(correct, z=1.96):
    \"\"\"Khoảng tin cậy Wald cho độ chính xác: p̂ ± z·√(p̂(1 − p̂)/n).\"\"\"
    # TODO
    raise NotImplementedError


def mcnemar_exact(a, b):
    \"\"\"p-value hai phía của McNemar chính xác: chỉ dùng b10 (A đúng, B sai) và b01.
    Dưới H₀, b10 ~ Binomial(b10 + b01, 0.5).\"\"\"
    # TODO
    raise NotImplementedError


def paired_bootstrap_ci(a, b, n_boot=10_000, seed=0):
    \"\"\"CI 95% (percentile) cho acc(A) − acc(B): lấy mẫu lại CÙNG chỉ số cho A và B.
    Trả về (lo, hi, tỉ lệ lần hiệu ≤ 0).\"\"\"
    # TODO
    raise NotImplementedError


def unpaired_bootstrap_ci(a, b, n_boot=10_000, seed=0):
    \"\"\"CI 95% của acc(A) − acc(B) khi lấy mẫu lại A và B bằng HAI bộ chỉ số độc lập (cách SAI,
    để so với bản có cặp). Trả về (lo, hi).\"\"\"
    # TODO
    raise NotImplementedError""",
"""def wald_ci(correct, z=1.96):
    p, n = np.mean(correct), len(correct)
    se = np.sqrt(p * (1 - p) / n)
    return p - z * se, p + z * se


def mcnemar_exact(a, b):
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    return stats.binomtest(n10, n10 + n01, 0.5).pvalue


def paired_bootstrap_ci(a, b, n_boot=10_000, seed=0):
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(a), (n_boot, len(a)))     # cùng chỉ số cho cả hai
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return lo, hi, np.mean(d <= 0)


def unpaired_bootstrap_ci(a, b, n_boot=10_000, seed=0):
    r = np.random.default_rng(seed)
    i = r.integers(0, len(a), (n_boot, len(a)))        # hai bộ chỉ số ĐỘC LẬP: phá cấu trúc cặp
    j = r.integers(0, len(b), (n_boot, len(b)))
    d = a[i].mean(axis=1) - b[j].mean(axis=1)
    return tuple(np.percentile(d, [2.5, 97.5]))""")

CMP_CHECK = """lo_a, hi_a = wald_ci(a)
p_mc = mcnemar_exact(a, b)
lo, hi, frac = paired_bootstrap_ci(a, b)
se_paired = np.std(a - b, ddof=1) / np.sqrt(len(a))
se_unpaired = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
print(f"Wald CI cho A: [{lo_a:.3f}, {hi_a:.3f}] | McNemar chính xác p = {p_mc:.4f}")
print(f"paired bootstrap CI của hiệu: [{lo:.3f}, {hi:.3f}], tỉ lệ hiệu ≤ 0: {frac:.4f}")
print(f"SE có cặp = {se_paired:.4f} | SE không cặp = {se_unpaired:.4f}")
lo_u, hi_u = unpaired_bootstrap_ci(a, b)
print(f"bootstrap độc lập (sai): [{lo_u:.3f}, {hi_u:.3f}] (đổi nhẹ theo seed và thứ tự dữ liệu)")
# Mô phỏng tỉ lệ phủ của Wald CI khi n = 100, p = 0.85
r = np.random.default_rng(4)
k = r.binomial(100, 0.85, 200_000) / 100
half = 1.96 * np.sqrt(k * (1 - k) / 100)
coverage = np.mean((k - half <= 0.85) & (0.85 <= k + half))
upper = np.mean(k - half > 0.85)      # khoảng nằm trọn phía trên p thật
lower = np.mean(k + half < 0.85)      # khoảng nằm trọn phía dưới p thật
print(f"tỉ lệ phủ thực của Wald CI 95% (n = 100, p = 0.85): {coverage:.3f}")
print(f"lỗi phủ phía trên (lo > p): {upper:.3%} | phía dưới (hi < p): {lower:.3%}")
# Giá trị chính xác: cộng pmf binomial qua mọi k, không nhiễu Monte Carlo
kk = np.arange(101); ph = kk / 100; hw = 1.96 * np.sqrt(ph * (1 - ph) / 100); pm = stats.binom.pmf(kk, 100, 0.85)
cov_ex = pm[(ph - hw <= 0.85) & (0.85 <= ph + hw)].sum()
up_ex, lo_ex = pm[ph - hw > 0.85].sum(), pm[ph + hw < 0.85].sum()
print(f"chính xác (cộng pmf): phủ {cov_ex:.2%} | phía trên {up_ex:.2%} | phía dưới {lo_ex:.2%}")
assert abs(cov_ex - 0.9330) < 5e-4 and abs(up_ex - 0.0551) < 5e-4 and abs(lo_ex - 0.0119) < 5e-4
assert abs(coverage - cov_ex) < 0.005
assert abs(p_mc - stats.binomtest(30, 45, 0.5).pvalue) < 1e-12 and abs(p_mc - 0.0357) < 1e-3
assert lo < 0.015 < hi and lo > 0
assert se_paired < se_unpaired / 2
assert 0.92 <= coverage <= 0.95
assert upper > 3 * lower and abs(coverage + upper + lower - 1) < 1e-12
assert lo_u < 0 < hi_u
print("Lab 5 ĐẠT ✔")"""

# ---- Săn bug ---------------------------------------------------------------------
BUG_DATA = """def make_logits(correct, y, rng, K=3):
    \"\"\"Logit giả: hàng đúng thì lớp y lớn nhất, hàng sai thì một lớp khác lớn nhất.\"\"\"
    n = len(y)
    L = rng.normal(0, 1, (n, K))
    top = np.where(correct == 1, y, (y + 1 + rng.integers(0, K - 1, n)) % K)
    L[np.arange(n), top] = L.max(axis=1) + rng.uniform(0.5, 2.5, n)
    hot = rng.random(n) < 0.05                        # 5% hàng có logit bị dời thêm 1000
    L[hot] += 1000.0                                  # (softmax không đổi khi cộng hằng số)
    return L

rb = np.random.default_rng(5)
y = rb.integers(0, 3, 1000)
logits_A, logits_B = make_logits(a, y, rb), make_logits(b, y, rb)
seeds_A = np.array([84.1, 85.3, 83.7, 84.9, 85.0])   # ví dụ 3.12, cùng seed theo cặp
seeds_B = np.array([83.6, 84.2, 83.9, 84.1, 84.3])


def compare_buggy(logits_A, logits_B, y, seeds_A, seeds_B, n_boot=10_000):
    out = {}
    correct = {}
    for name, L in [("A", logits_A), ("B", logits_B)]:
        with np.errstate(over="ignore", invalid="ignore"):
            p = np.exp(L) / np.exp(L).sum(axis=1, keepdims=True)
            out["loss_" + name] = -np.nanmean(np.log(p.max(axis=1)))
        correct[name] = (L.argmax(axis=1) == y).astype(float)
        out["acc_" + name] = correct[name].mean()
    r = np.random.default_rng(0)
    i = r.integers(0, len(y), (n_boot, len(y)))
    j = r.integers(0, len(y), (n_boot, len(y)))
    d = correct["A"][i].mean(axis=1) - correct["B"][j].mean(axis=1)
    out["ci_diff"] = np.percentile(d, [2.5, 97.5])
    out["p_seeds"] = stats.ttest_ind(seeds_A, seeds_B).pvalue
    return out

for k, v in compare_buggy(logits_A, logits_B, y, seeds_A, seeds_B).items():
    print(f"{k:8s}: {np.round(v, 4)}")"""

BUGFIX = ("""def compare_fixed(logits_A, logits_B, y, seeds_A, seeds_B, n_boot=10_000):
    \"\"\"Sửa cả bốn lỗi của compare_buggy. Trả về dict cùng khóa.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def compare_fixed(logits_A, logits_B, y, seeds_A, seeds_B, n_boot=10_000):
    out = {}
    correct = {}
    n = len(y)
    for name, L in [("A", logits_A), ("B", logits_B)]:
        logp = L - special.logsumexp(L, axis=1, keepdims=True)    # sửa lỗi 1: log-softmax ổn định
        out["loss_" + name] = -logp[np.arange(n), y].mean()       # sửa lỗi 2: chỉ lấy cột nhãn đúng
        correct[name] = (L.argmax(axis=1) == y).astype(float)
        out["acc_" + name] = correct[name].mean()
    r = np.random.default_rng(0)
    i = r.integers(0, n, (n_boot, n))
    d = correct["A"][i].mean(axis=1) - correct["B"][i].mean(axis=1)  # sửa lỗi 3: cùng chỉ số
    out["ci_diff"] = np.percentile(d, [2.5, 97.5])
    out["p_seeds"] = stats.ttest_rel(seeds_A, seeds_B).pvalue     # sửa lỗi 4: kiểm định có cặp
    return out""")

BUG_CHECK = """bad = compare_buggy(logits_A, logits_B, y, seeds_A, seeds_B)
good = compare_fixed(logits_A, logits_B, y, seeds_A, seeds_B)
for k, v in good.items():
    print(f"{k:8s}: {np.round(v, 4)}")
ref_A = -stats.multinomial.logpmf(np.eye(3)[y], 1, special.softmax(logits_A, axis=1)).mean()
assert np.isfinite(good["loss_A"]) and abs(good["loss_A"] - ref_A) < 1e-8
assert good["loss_A"] > 2 * bad["loss_A"]
assert good["ci_diff"][0] > 0 and bad["ci_diff"][0] < 0
assert abs(good["p_seeds"] - 0.0562) < 1e-3
print("Săn bug ĐẠT ✔")"""


def build(solution: bool):
    pick = (lambda pair: pair[1]) if solution else (lambda pair: pair[0])
    nb = nbf.v4.new_notebook()
    head = "Lời giải" if solution else "Bài lab"
    nb.cells = [
        md(f"# Chương 3 — {head}: Xác suất, thống kê, lý thuyết thông tin\n\n"
           + ("Bản lời giải đã chạy kiểm. Chỉ mở sau khi đã tự làm `lab.ipynb`."
              if solution else
              "Viết code vào các ô có `# TODO`, rồi chạy ô kiểm tra ngay bên dưới. "
              "Ô kiểm tra in **ĐẠT ✔** là xong. Bí quá 20 phút mới mở `loigiai.ipynb`.")),
        code(SETUP),
        md("## Lab 1 — Bayes và tỉ lệ nền\nTính posterior bằng công thức, rồi kiểm lại bằng mô phỏng 10⁶ bình luận."),
        code(pick(BAYES)), code(BAYES_CHECK),
        md("## Lab 2 — Gauss nhiều chiều từ đầu\nViết log-mật độ bằng Cholesky và lấy mẫu x = μ + Lz. "
           "Câu hỏi: vì sao không nên tính Σ⁻¹ trực tiếp?"),
        code(pick(MVN)), code(MVN_CHECK),
        md("## Lab 3 — log-sum-exp và cross-entropy ổn định\nCross-entropy tính từ logit, có label smoothing. "
           "Câu hỏi: khi bật label smoothing, loss nhỏ nhất đạt được là bao nhiêu?"),
        code(pick(LSE)), code(LSE_CHECK),
        md("## Lab 4 — MLE, MAP và ridge\nKiểm: ridge = MAP với prior Gauss; prior bị dữ liệu lấn át khi n tăng."),
        code(pick(MAP)), code(MAP_CHECK),
        md("## Lab 5 — So sánh hai model đúng cách\nMcNemar, paired bootstrap, và tỉ lệ phủ thực của khoảng Wald."),
        code(DATA5), code(pick(CMP)), code(CMP_CHECK),
        md("## Săn bug\n`compare_buggy` chạy không lỗi nhưng sai. Tìm cả **bốn** lỗi rồi viết `compare_fixed`."),
        code(BUG_DATA), code(pick(BUGFIX)), code(BUG_CHECK),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    name = "loigiai.ipynb" if solution else "lab.ipynb"
    nbf.write(nb, OUT / name)
    return OUT / name


if __name__ == "__main__":
    print(build(False), build(True))
