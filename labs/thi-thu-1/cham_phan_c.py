"""Bộ chấm tự động Phần C của Đề thi thử 1 (chỉ dùng NumPy).

CHỈ CHẠY SAU KHI HẾT GIỜ. File này không chứa lời giải, nhưng có dữ liệu test và vài đáp số
(ví dụ đáp số câu C1(b)), nên đừng mở nó trong lúc làm đề.

Cách dùng, từ thư mục gốc của sách:
    python labs/thi-thu-1/cham_phan_c.py --student bai_lam.py

`bai_lam.py` là bản bạn chép từ `labs/thi-thu-1/bai_lam_mau.py` rồi điền hai hàm
`softmax_cross_entropy` và `logreg_step`. Hàm nào còn `raise NotImplementedError` thì được 0 điểm phần đó.

Thang điểm bám đúng thang trong đề (mục Lời giải, Phần C):
  C1(a) 8 điểm: shape 1, không sửa Z tại chỗ 1, giá trị và gradient 3, ổn định số 3.
  C2(a) 8 điểm: shape 2, giá trị J, dw, db 3, không phạt b 1, ổn định số 2.
Mỗi test chỉ đo đúng một ý: các test ổn định số dùng N = 1 (trung bình bằng tổng) và so gradient
không phụ thuộc hệ số 1/N, nên một lỗi chỉ bị trừ một lần. Vòng lặp Python theo mẫu hay theo lớp,
câu C1(b) và C2(b) chấm tay theo lời giải trong đề.

Tham chiếu tự tính bằng NumPy: loss viết thẳng theo định nghĩa trên dữ liệu có logit nhỏ (không tràn),
gradient tham chiếu lấy bằng sai phân trung tâm; các ca logit cực lớn so với đáp số tính tay.
Kiểm với Python 3.11–3.13, NumPy 2.x, ngày 2026-11-16.
"""
import argparse
import ast
import importlib.util
import inspect
import math
import sys
import textwrap
import warnings

import numpy as np

try:  # để in tiếng Việt trên console Windows khi output bị chuyển hướng
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


# ===================== DỮ LIỆU TEST (tất định, không dùng bộ sinh ngẫu nhiên) =====================
def _c1_data():
    k = np.arange(35, dtype=np.float64)
    Z = 3.0 * np.sin(1.7 * k + 0.3).reshape(7, 5)                 # logit nhỏ, |Z| ≤ 3
    y = np.array([0, 3, 1, 4, 2, 0, 3])
    return Z, y


def _c2_data():
    N, d = 50, 4
    k = np.arange(N * d, dtype=np.float64)
    X = 1.5 * np.sin(0.37 * k + 0.1).reshape(N, d)
    y = (np.cos(1.3 * np.arange(N) + 0.2) + 0.3 * X[:, 0] > 0).astype(np.float64)
    w0 = 0.1 * np.cos(np.arange(d, dtype=np.float64))
    return X, y, w0


# ===================== THAM CHIẾU (định nghĩa thẳng, chỉ dùng cho logit nhỏ) =====================
def _ce_by_definition(Z, y):
    """Cross-entropy trung bình viết thẳng theo định nghĩa. Đúng khi |Z| nhỏ; tràn số khi |Z| lớn."""
    N = Z.shape[0]
    return float(np.mean(np.log(np.exp(Z).sum(axis=1)) - Z[np.arange(N), y]))


def _logreg_J_by_definition(w, b, X, y, lam):
    """J(w, b) viết thẳng theo công thức trong đề. Đúng khi |z| nhỏ; tràn số khi |z| lớn."""
    z = X @ w + b
    return float(np.mean(np.log(1.0 + np.exp(z)) - y * z) + 0.5 * lam * (w @ w))


def _central_diff(f, x, h=1e-5):
    x = np.array(x, dtype=np.float64)
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g


def _unit(a):
    a = np.asarray(a, dtype=np.float64)
    return a / np.abs(a).sum()


# ===================== BỘ TEST =====================
def run_tests(sce=None, lrs=None, verbose=True):
    """Chấm hai hàm. Trả về dict {tên test: (điểm đạt, điểm tối đa, chương)}."""
    score = {}

    def check(name, pts, chapter, fn):
        try:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                fn()
            score[name] = (pts, pts, chapter)
            if verbose:
                print(f"  ĐẠT   {name} (+{pts})")
        except NotImplementedError:
            score[name] = (0, pts, chapter)
            if verbose:
                print(f"  TRƯỢT {name}: hàm chưa viết (NotImplementedError)")
        except Exception as e:  # noqa: BLE001
            score[name] = (0, pts, chapter)
            if verbose:
                print(f"  TRƯỢT {name}: {type(e).__name__}: {e}")

    # ---------------------------------------------------------------- C1
    if sce is not None:
        if verbose:
            print("C1 softmax_cross_entropy")
        Z, y = _c1_data()

        def c1_shape():
            loss, dZ = sce(Z.copy(), y.copy())
            assert np.ndim(loss) == 0, "loss phải là số vô hướng"
            assert np.shape(dZ) == Z.shape, f"dZ có shape {np.shape(dZ)}, phải là {Z.shape}"

        def c1_no_mutation():
            Zin, yin = Z.copy(), y.copy()
            sce(Zin, yin)
            assert np.array_equal(Zin, Z) and np.array_equal(yin, y), "hàm đã sửa Z (hoặc y) tại chỗ"

        def c1_value():
            loss, dZ = sce(Z.copy(), y.copy())
            L_ref = _ce_by_definition(Z, y)
            g_ref = _central_diff(lambda V: _ce_by_definition(V, y), Z)
            assert abs(loss - L_ref) < 1e-10, f"loss = {loss}, tham chiếu {L_ref}"
            assert np.allclose(dZ, g_ref, rtol=0, atol=1e-8), "dZ khác gradient sai phân (nhớ chia N)"
            # ví dụ tay: Z = (0, ln 3), y = 1 -> loss = ln(4/3), dZ = (0.25, −0.25)
            l1, d1 = sce(np.array([[0.0, math.log(3.0)]]), np.array([1]))
            assert abs(l1 - math.log(4 / 3)) < 1e-12, f"ví dụ tay: loss {l1}, phải là ln(4/3)"
            assert np.allclose(d1, [[0.25, -0.25]], rtol=0, atol=1e-12), f"ví dụ tay: dZ {d1}"

        def c1_stable():
            # (1) Từng mẫu một (N = 1) với logit cực lớn; đáp số tính tay.
            cases = [  # (Z một hàng, nhãn, loss đúng, dZ đúng)
                ([1000.0, 1000.0, 1000.0], 2, math.log(3.0), [1 / 3, 1 / 3, -2 / 3]),
                ([-1000.0, 0.0, 1000.0], 0, 2000.0, [-1.0, 0.0, 1.0]),
                ([1e4, -1e4, 0.0], 1, 2e4, [1.0, -1.0, 0.0]),
            ]
            for row, lab, L_true, g_true in cases:
                loss, dZ = sce(np.array([row]), np.array([lab]))
                assert np.isfinite(loss) and np.all(np.isfinite(dZ)), f"tràn số với logit {row}"
                assert abs(loss - L_true) <= 1e-12 * max(1.0, L_true), f"logit {row}: loss {loss}, đúng là {L_true}"
                assert np.allclose(dZ, [g_true], rtol=0, atol=1e-12), f"logit {row}: dZ {dZ}"
            # (2) Nhiều hàng lệch nhau xa (cộng 0, 800, …, 4800 vào từng hàng): softmax từng hàng không đổi,
            #     nên kết quả phải trùng với chính hàm của bạn trên Z gốc. Bắt lỗi trừ max của cả ma trận.
            shift = 800.0 * np.arange(Z.shape[0])[:, None]
            l0, g0 = sce(Z.copy(), y.copy())
            l1, g1 = sce(Z + shift, y.copy())
            assert np.isfinite(l1) and np.all(np.isfinite(g1)), "tràn số khi các hàng lệch nhau 800 trở lên"
            assert abs(l1 - l0) < 1e-9 and np.allclose(g1, g0, rtol=0, atol=1e-10), \
                "cộng hằng vào từng hàng làm đổi kết quả (trừ max theo hàng chưa?)"

        check("C1 shape trả về", 1, 2, c1_shape)
        check("C1 không sửa Z tại chỗ", 1, 2, c1_no_mutation)
        check("C1 đúng giá trị và gradient", 3, 2, c1_value)
        check("C1 ổn định số (logit ±1000, ±1e4; hàng lệch xa)", 3, 3, c1_stable)

    # ---------------------------------------------------------------- C2
    if lrs is not None:
        if verbose:
            print("C2 logreg_step")
        X, y, w0 = _c2_data()
        d = X.shape[1]

        def c2_shape():
            w1, b1, J = lrs(w0.copy(), 0.0, X.copy(), y.copy(), 0.1, 0.01)
            assert np.shape(w1) == (d,), f"w mới có shape {np.shape(w1)}, phải là ({d},)"
            assert np.ndim(b1) == 0 and np.ndim(J) == 0, "b mới và J phải là số vô hướng"

        def c2_value():
            lr, lam, b0 = 0.1, 0.05, 0.0            # b0 = 0: lỗi phạt b chỉ bị bắt ở test riêng
            w1, b1, J = lrs(w0.copy(), b0, X.copy(), y.copy(), lr, lam)
            J_ref = _logreg_J_by_definition(w0, b0, X, y, lam)
            g = _central_diff(lambda v: _logreg_J_by_definition(v[:d], v[d], X, y, lam), np.append(w0, b0))
            assert abs(J - J_ref) < 1e-10, f"J = {J}, tham chiếu {J_ref} (J tính tại (w, b) cũ)"
            assert np.allclose(w1, w0 - lr * g[:d], rtol=0, atol=1e-9), "dw sai (nhớ chia N và cộng λw)"
            assert abs(b1 - (b0 - lr * g[d])) < 1e-9, "db sai"

        def c2_no_bias_penalty():
            # w = 0 nên phạt L2 của w bằng 0; nếu b không bị phạt thì b mới không phụ thuộc λ.
            Xs, ys = np.array([[0.0], [1.0]]), np.array([1.0, 0.0])
            _, b_a, _ = lrs(np.zeros(1), 3.0, Xs, ys, 1.0, 0.0)
            _, b_b, _ = lrs(np.zeros(1), 3.0, Xs, ys, 1.0, 10.0)
            assert np.isfinite(b_a) and abs(b_a - b_b) < 1e-12, f"b mới đổi theo λ ({b_a} và {b_b}): b đang bị phạt"

        def c2_stable():
            # Từng mẫu một (N = 1), |z| = 800, λ = 0; đáp số tính tay.
            cases = [  # (x, y, J đúng, w mới đúng, b mới đúng) với w = 1, b = 0, lr = 0.01
                (800.0, 0.0, 800.0, -7.0, -0.01),
                (-800.0, 1.0, 800.0, -7.0, 0.01),
                (800.0, 1.0, 0.0, 1.0, 0.0),
            ]
            for x, yy, J_true, w_true, b_true in cases:
                Xb, yb = np.array([[x]]), np.array([yy])
                _, _, J = lrs(np.array([1.0]), 0.0, Xb, yb, 0.0, 0.0)     # lr = 0: J chắc chắn tại (w, b) cũ
                assert np.isfinite(J) and abs(J - J_true) <= 1e-12 * max(1.0, J_true), f"x = {x}, y = {yy}: J = {J}, đúng là {J_true}"
                w1, b1, _ = lrs(np.array([1.0]), 0.0, Xb, yb, 0.01, 0.0)
                assert np.all(np.isfinite(w1)) and np.isfinite(b1), f"x = {x}: w hoặc b mới tràn số"
                assert np.allclose(w1, w_true, rtol=0, atol=1e-9) and abs(b1 - b_true) < 1e-12, \
                    f"x = {x}, y = {yy}: (w, b) mới = ({w1}, {b1}), đúng là ({w_true}, {b_true})"

        check("C2 shape (w mới là (d,), b và J là số)", 2, 6, c2_shape)
        check("C2 đúng J, dw, db", 3, 5, c2_value)
        check("C2 không phạt b", 1, 4, c2_no_bias_penalty)
        check("C2 ổn định số (|z| = 800)", 2, 5, c2_stable)
    return score


def find_loops(fn):
    """Trả về danh sách dòng có vòng lặp (for, while, comprehension) trong thân hàm, để chấm tay."""
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError):
        return []
    kinds = (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    first = fn.__code__.co_firstlineno
    return sorted({first + n.lineno - 1 for n in ast.walk(tree) if isinstance(n, kinds)})


def report(score, fns):
    auto = sum(v[0] for v in score.values())
    total = sum(v[1] for v in score.values())
    by_ch = {}
    for got, mx, ch in score.values():
        a, b = by_ch.get(ch, (0, 0))
        by_ch[ch] = (a + got, b + mx)
    print(f"\nĐiểm phần tự động: {auto} / {total}")
    print("Chia về chương (ghi vào bảng chấm theo chương, cộng thêm phần chấm tay):")
    for ch in sorted(by_ch):
        print(f"  chương {ch}: {by_ch[ch][0]} / {by_ch[ch][1]}")
    print("Chấm tay thêm theo lời giải trong đề: C1(b) 2 điểm (chương 3), C2(b) 2 điểm (chương 6).")
    for name, fn in fns.items():
        if fn is None:
            continue
        loops = find_loops(fn)
        if loops:
            print(f"Chú ý: {name} có vòng lặp ở dòng {loops}. Nếu lặp theo mẫu hay theo lớp thì trừ 2 điểm "
                  "(điểm (a) của câu đó không âm).")


def _load(path):
    spec = importlib.util.spec_from_file_location("bai_lam", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chấm tự động Phần C, Đề thi thử 1 (16 điểm).")
    ap.add_argument("--student", required=True, help="file .py định nghĩa softmax_cross_entropy và logreg_step")
    args = ap.parse_args()
    try:
        mod = _load(args.student)
    except Exception as e:  # noqa: BLE001
        print(f"Không nạp được {args.student}: {type(e).__name__}: {e}")
        sys.exit(1)
    fns = {"softmax_cross_entropy": getattr(mod, "softmax_cross_entropy", None),
           "logreg_step": getattr(mod, "logreg_step", None)}
    for name, fn in fns.items():
        if fn is None:
            print(f"Không thấy hàm {name} trong {args.student}: phần đó 0 điểm.")
    def _missing(*_a, **_k):  # hàm thiếu được chấm như hàm chưa viết
        raise NotImplementedError

    sc = run_tests(fns["softmax_cross_entropy"] or _missing, fns["logreg_step"] or _missing)
    report(sc, fns)
