"""Sinh hai notebook cho lab chương 1: lab.ipynb (có TODO) và loigiai.ipynb (lời giải).

Chạy lại script này nếu muốn tạo lại notebook gốc:  python labs/build_chuong01.py
"""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "chuong01"
OUT.mkdir(parents=True, exist_ok=True)

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

SETUP = """import numpy as np
np.set_printoptions(precision=4, suppress=True)
# Mỗi lab tự tạo Generator riêng (seed cố định) ngay trong ô dữ liệu của nó,
# nên kết quả lab này không phụ thuộc code bạn viết ở lab khác."""

# ---- (TODO, lời giải) cho từng hàm ---------------------------------------------
POWER = ("""def power_iteration(A, n_iter=1000, tol=1e-10):
    \"\"\"Trả về (λ lớn nhất, vector riêng đơn vị) bằng cách nhân lặp v ← Av/‖Av‖₂.
    Gợi ý: ước lượng λ bằng thương Rayleigh vᵀAv; dừng khi λ thay đổi < tol.
    Nếu cần vector khởi tạo ngẫu nhiên, dùng Generator riêng của hàm, đừng dùng biến toàn cục.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def power_iteration(A, n_iter=1000, tol=1e-10):
    v = np.random.default_rng(12345).standard_normal(A.shape[0])   # Generator riêng của hàm
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(n_iter):
        w = A @ v
        v_new = w / np.linalg.norm(w)
        lam_new = v_new @ A @ v_new          # thương Rayleigh
        if abs(lam_new - lam) < tol:
            break
        v, lam = v_new, lam_new
    return lam_new, v_new""")

POWER_CHECK = """rng = np.random.default_rng(101)
M = rng.standard_normal((50, 50)); S = M @ M.T          # đối xứng, PSD
lam, v = power_iteration(S)
evals, evecs = np.linalg.eigh(S)                         # eigh: thứ tự TĂNG dần
print("power iteration:", lam, "| eigh:", evals[-1])
assert abs(lam - evals[-1]) / evals[-1] < 1e-6, "λ chưa khớp"
assert abs(abs(v @ evecs[:, -1]) - 1) < 1e-6, "vector riêng chưa khớp"
print("Lab 1 ĐẠT ✔")"""

SOLVE = ("""def solve_three_ways(A, b):
    \"\"\"Trả về dict {'normal': x, 'qr': x, 'lstsq': x}.
    normal: giải (AᵀA)x = Aᵀb.  qr: A = QR rồi giải Rx = Qᵀb.  lstsq: np.linalg.lstsq.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def solve_three_ways(A, b):
    x_normal = np.linalg.solve(A.T @ A, A.T @ b)      # phương trình chuẩn — không nên dùng
    Q, R = np.linalg.qr(A)
    x_qr = np.linalg.solve(R, Q.T @ b)                # Rx = Qᵀb
    x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]    # dựa trên SVD
    return {"normal": x_normal, "qr": x_qr, "lstsq": x_lstsq}""")

SOLVE_RUN = """t = np.linspace(0, 1, 50)
results = {}
for deg in [6, 12]:
    rng = np.random.default_rng(200 + deg)               # seed riêng cho từng bậc
    A = np.vander(t, deg + 1, increasing=True)
    x_true = rng.standard_normal(deg + 1)
    b = A @ x_true                                    # (a) hệ nhất quán, không nhiễu
    print(f"bậc {deg}: cond(A) = {np.linalg.cond(A):.1e}")
    for name, x in solve_three_ways(A, b).items():
        fe = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
        results[(deg, name)] = fe
        print(f"  (a) {name:6s}: sai số nghiệm = {fe:.1e}, phần dư = {np.linalg.norm(A @ x - b):.1e}")
    b_noisy = b + 1e-6 * rng.standard_normal(len(b))  # (b) thêm nhiễu cỡ 1e-6 mỗi phần tử (nhiễu tuyệt đối)
    x = np.linalg.lstsq(A, b_noisy, rcond=None)[0]
    print(f"  (b) lstsq có nhiễu: sai số nghiệm = {np.linalg.norm(x - x_true) / np.linalg.norm(x_true):.1e}")

assert results[(12, "qr")] < 1e-6, "QR phải chính xác"
assert results[(12, "normal")] > 100 * results[(12, "qr")], "Phương trình chuẩn phải kém hơn hẳn ở bậc 12"
print("Lab 2 ĐẠT ✔")"""

PCA = ("""def pca_svd(X, k):
    \"\"\"Trả về (components k×d, explained_var, mu, s) — components là các HÀNG của Vᵀ.
    Nhớ trừ trung bình trước khi SVD.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def pca_svd(X, k):
    mu = X.mean(axis=0)
    Xc = X - mu                                       # BẮT BUỘC trừ trung bình
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k], s**2 / (len(X) - 1), mu, s         # thành phần chính = hàng của Vᵀ""")

PCA_CHECK = """rng = np.random.default_rng(301)
N, d, k = 500, 10, 3
Z = rng.standard_normal((N, k)) * np.array([5.0, 2.0, 1.0])
W = np.linalg.qr(rng.standard_normal((d, d)))[0][:, :k]
X = Z @ W.T + 0.1 * rng.standard_normal((N, d)) + 3.0
comps, var, mu, s = pca_svd(X, k)
X_rec = ((X - mu) @ comps.T) @ comps + mu
err, tail = np.linalg.norm(X - X_rec) ** 2, np.sum(s[k:] ** 2)
print("sai số tái tạo =", err, "| tổng σ² bị bỏ =", tail)
assert np.isclose(err, tail), "công thức sai số chưa khớp"
# So với một phép chiếu hạng k ngẫu nhiên: Eckart–Young nói không phép nào thắng được PCA
Qr = np.linalg.qr(rng.standard_normal((d, k)))[0]
err_rand = np.linalg.norm((X - mu) - (X - mu) @ Qr @ Qr.T) ** 2
print("sai số của chiếu hạng k ngẫu nhiên =", err_rand)
assert err_rand > err, "PCA phải thắng chiếu ngẫu nhiên"
print("Lab 3 ĐẠT ✔")"""

LOWRANK = ("""def lowrank_report(img, ranks):
    \"\"\"In sai số tương đối của xấp xỉ hạng r (r trong ranks) và
    số thành phần cần để giữ 99% năng lượng (tổng σᵢ²). Trả về số đó.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def lowrank_report(img, ranks):
    U, s, Vt = np.linalg.svd(img, full_matrices=False)
    for r in ranks:
        img_r = (U[:, :r] * s[:r]) @ Vt[:r]
        print(f"  hạng {r:3d}: sai số tương đối {np.linalg.norm(img - img_r) / np.linalg.norm(img):.4f}")
    energy = np.cumsum(s**2) / np.sum(s**2)
    n99 = int(np.searchsorted(energy, 0.99) + 1)
    print("  số thành phần giữ 99% năng lượng:", n99)
    return n99""")

LOWRANK_RUN = """rng = np.random.default_rng(401)
synth = np.add.outer(np.sin(np.linspace(0, 6, 200)), np.cos(np.linspace(0, 9, 300)))
synth += 0.05 * rng.standard_normal(synth.shape)
print("4a — ma trận tổng hợp:")
n_syn = lowrank_report(synth, [1, 2, 5, 20])
try:
    from skimage import data
except ImportError:
    data = None
    print("Chưa cài scikit-image (pip install scikit-image): bỏ qua 4b, Lab 4 CHƯA ĐẠT.")
if data is not None:
    print("4b — ảnh thật 512×512:")
    n_cam = lowrank_report(data.camera().astype(float), [1, 5, 20, 50, 100])
    assert n_cam > 5 * n_syn
    print("Lab 4 ĐẠT ✔")"""

BUG_DATA = """def pca_buggy(X, k):
    C = X.T @ X / (len(X) - 1)
    evals, evecs = np.linalg.eigh(C)
    comps = evecs[:k]
    explained_ratio = evals[:k] / evals.sum()
    return comps, explained_ratio

r1 = np.random.default_rng(1)
Zb = r1.standard_normal((500, 2)) * np.array([5.0, 2.0])
Wb = np.linalg.qr(r1.standard_normal((10, 10)))[0][:, :2]
Xb = Zb @ Wb.T + 0.1 * r1.standard_normal((500, 10)) + 3.0
print("bản lỗi báo tỉ lệ:", pca_buggy(Xb, 2)[1])"""

BUGFIX = ("""def pca_fixed(X, k):
    \"\"\"Sửa cả ba lỗi của pca_buggy. Trả về (components k×d, explained_ratio).\"\"\"
    # TODO
    raise NotImplementedError""",
"""def pca_fixed(X, k):
    Xc = X - X.mean(axis=0)                          # sửa lỗi 1
    C = Xc.T @ Xc / (len(X) - 1)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]                  # sửa lỗi 2: giảm dần
    evals, evecs = evals[order], evecs[:, order]
    return evecs[:, :k].T, evals[:k] / evals.sum()   # sửa lỗi 3: lấy cột""")

BUG_CHECK = """comps, ratio = pca_fixed(Xb, 2)
print("bản sửa:", ratio)
cos = np.abs(comps @ Wb)
assert ratio[0] > 0.8 and np.allclose(np.sort(cos.max(axis=1)), 1, atol=0.01)
print("Săn bug ĐẠT ✔")"""


def build(solution: bool):
    pick = (lambda pair: pair[1]) if solution else (lambda pair: pair[0])
    nb = nbf.v4.new_notebook()
    head = "Lời giải" if solution else "Bài lab"
    nb.cells = [
        md(f"# Chương 1 — {head}: Đại số tuyến tính\n\n"
           + ("Bản lời giải đã chạy kiểm. Chỉ mở sau khi đã tự làm `lab.ipynb`."
              if solution else
              "Viết code vào các ô có `# TODO`, rồi chạy ô kiểm tra ngay bên dưới. "
              "Ô kiểm tra in **ĐẠT ✔** là xong. Bí quá 20 phút mới mở `loigiai.ipynb`.")),
        code(SETUP),
        md("## Lab 1 — Power iteration\nTìm λₘₐₓ của ma trận đối xứng bằng cách nhân lặp, đối chiếu với `np.linalg.eigh`."),
        code(pick(POWER)), code(POWER_CHECK),
        md("## Lab 2 — Bình phương tối thiểu ba cách\nĐo **sai số nghiệm** ‖x̂ − x‖₂/‖x‖₂, không chỉ phần dư. "
           "Dữ liệu và seed nằm trong ô chạy lab; chỉ viết `solve_three_ways`. "
           "Câu hỏi: ở 2b bậc 12, `lstsq` cũng sai rất nhiều — lỗi của thuật toán hay của bài toán?"),
        code(pick(SOLVE)), code(SOLVE_RUN),
        md("## Lab 3 — PCA bằng SVD\nKiểm công thức sai số của SVD cắt cụt (bình phương sai số tái tạo = tổng σᵢ² bị bỏ) "
           "và so với một phép chiếu hạng k ngẫu nhiên."),
        code(pick(PCA)), code(PCA_CHECK),
        md("## Lab 4 — Xấp xỉ hạng thấp: ma trận tổng hợp và ảnh thật\n"
           "Câu hỏi: vì sao giữ 99% năng lượng lại ứng với sai số tương đối khoảng 10%?"),
        code(pick(LOWRANK)), code(LOWRANK_RUN),
        md("## Săn bug\n`pca_buggy` chạy không lỗi nhưng sai. Tìm cả **ba** lỗi rồi viết `pca_fixed`."),
        code(BUG_DATA), code(pick(BUGFIX)), code(BUG_CHECK),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    name = "loigiai.ipynb" if solution else "lab.ipynb"
    nbf.write(nb, OUT / name)
    return OUT / name


if __name__ == "__main__":
    print(build(False), build(True))
