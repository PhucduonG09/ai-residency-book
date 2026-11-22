"""Bộ chấm tự động Phần C của Đề thi thử 2 (chỉ dùng NumPy).

CHỈ CHẠY SAU KHI HẾT GIỜ. File này không chứa lời giải, nhưng có dữ liệu test và vài đáp số
tính tay, nên đừng mở nó trong lúc làm đề.

Cách dùng, từ thư mục gốc của sách:
    python labs/thi-thu-2/cham_phan_c.py --student bai_lam.py

`bai_lam.py` là bản bạn chép từ `labs/thi-thu-2/bai_lam_mau.py` rồi điền hai hàm
`causal_attention` và `paired_bootstrap_ci`. Hàm nào còn `raise NotImplementedError` thì được 0 điểm phần đó.

Thang điểm bám đúng thang trong đề (mục Lời giải, Phần C):
  C1(a) 8 điểm: shape 1, mặt nạ nhân quả 2, giá trị 3, ổn định số 2.
  C2(a) 8 điểm: định dạng 1, seed và lấy mẫu lại thật 1, hiệu đúng 1, ghép cặp 3, mức 95% 2.
Mỗi test cố gắng đo đúng một ý:
  - C1 "giá trị" chỉ so hàng CUỐI của A và O (hàng nhìn mọi key, nên mặt nạ không ảnh hưởng);
    C1 "ổn định số" so hàm của bạn với chính nó khi cộng một hằng lớn vào từng hàng điểm, và chỉ đòi
    hữu hạn ở mọi ô cộng giá trị đúng ở hàng cuối, nên quên mặt nạ không bị trừ ở đây.
  - C2 "ghép cặp" so tỉ số độ rộng CI trên hai bộ dữ liệu cùng tỉ lệ đúng (một bộ hai mô hình tương quan
    mạnh, một bộ độc lập), nên chọn sai mức (90% thay 95%) hay đổi thang (phần trăm) không bị trừ ở đây;
    C2 "mức 95%" đo độ rộng và tâm trên bộ độc lập, chia cho chính hiệu bạn trả về, nên lỗi dấu
    hay lỗi phần trăm chỉ bị trừ ở test "hiệu đúng";
    C2 "seed" còn đòi khoảng có độ rộng dương (bắt lỗi tạo lại bộ sinh ngẫu nhiên trong vòng lặp bootstrap).
Vòng lặp Python (theo mẫu, theo vị trí, theo câu test) và câu C1(b), C2(b) chấm tay theo lời giải trong đề.

Tham chiếu của C1 tự tính bằng vòng lặp theo định nghĩa trên dữ liệu có điểm nhỏ (không tràn); ca điểm
cực lớn so với đáp số tính tay. Tham chiếu của C2 là SE ghép cặp dạng đóng (bootstrap phân vị xấp xỉ
mean ± 1.96·SE khi n lớn), dung sai rộng hơn sai số Monte Carlo của bootstrap.
Kiểm với Python 3.13, NumPy 2.x, ngày 2026-11-17 (không dùng cú pháp mới hơn Python 3.11).
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


# ===================== DỮ LIỆU TEST C1 (tất định, không dùng bộ sinh ngẫu nhiên) =====================
def _c1_data():
    """B = 2, T = 5, d = 6, dk = 3, dv = 4. Điểm q·k/√dk nhỏ (|s| < 3). d ≠ dk để lộ lỗi chia √d."""
    B, T, d, dk, dv = 2, 5, 6, 3, 4
    X = np.sin(0.7 * np.arange(B * T * d, dtype=np.float64) + 0.2).reshape(B, T, d)
    Wq = 0.8 * np.cos(1.3 * np.arange(d * dk, dtype=np.float64)).reshape(d, dk)
    Wk = 0.8 * np.sin(0.9 * np.arange(d * dk, dtype=np.float64) + 1.0).reshape(d, dk)
    Wv = np.cos(0.5 * np.arange(d * dv, dtype=np.float64) + 0.3).reshape(d, dv)
    return X, Wq, Wk, Wv


def _attn_last_row_by_definition(X, Wq, Wk, Wv):
    """Hàng cuối (query T−1, nhìn mọi key) của A và O, viết bằng vòng lặp theo định nghĩa.
    Chỉ đúng khi điểm nhỏ (exp trực tiếp, không trừ max)."""
    B, T, _ = X.shape
    dk = Wq.shape[1]
    A_last = np.zeros((B, T))
    O_last = np.zeros((B, Wv.shape[1]))
    for b in range(B):
        q = X[b, T - 1] @ Wq
        e = np.array([math.exp(float(q @ (X[b, j] @ Wk)) / math.sqrt(dk)) for j in range(T)])
        a = e / e.sum()
        A_last[b] = a
        O_last[b] = sum(a[j] * (X[b, j] @ Wv) for j in range(T))
    return A_last, O_last


def _augment_with_row_shift(X, Wq, Wk, Wv, c):
    """Thêm hai chiều vào X để điểm của query i cộng thêm c[i]/√dk' (dk' = dk + 1) cho MỌI key.
    Softmax theo hàng không đổi khi cộng một hằng vào cả hàng, nên đầu ra phải như khi c = 0."""
    B, T, d = X.shape
    dk, dv = Wq.shape[1], Wv.shape[1]
    Xa = np.concatenate([X, np.broadcast_to(np.asarray(c, float)[None, :, None], (B, T, 1)),
                         np.ones((B, T, 1))], axis=2)
    Wq2 = np.zeros((d + 2, dk + 1)); Wq2[:d, :dk] = Wq; Wq2[d, dk] = 1.0     # q' = (q, c_i)
    Wk2 = np.zeros((d + 2, dk + 1)); Wk2[:d, :dk] = Wk; Wk2[d + 1, dk] = 1.0  # k' = (k, 1)
    Wv2 = np.zeros((d + 2, dv)); Wv2[:d] = Wv
    return Xa, Wq2, Wk2, Wv2


# ===================== DỮ LIỆU TEST C2 (đếm sẵn, xếp tất định) =====================
def _from_counts(n11, n10, n01, n00):
    """Hai mảng 0/1: n11 câu cả hai đúng, n10 chỉ A đúng, n01 chỉ B đúng, n00 cả hai sai.
    Thứ tự các câu được trộn bằng một hoán vị tất định (không dùng bộ sinh ngẫu nhiên)."""
    a = np.array([1] * n11 + [1] * n10 + [0] * n01 + [0] * n00, dtype=np.int64)
    b = np.array([1] * n11 + [0] * n10 + [1] * n01 + [0] * n00, dtype=np.int64)
    order = np.argsort(np.sin(12.9898 * np.arange(a.size) + 0.5))
    return a[order], b[order]


def _paired_se(a, b):
    d = a.astype(float) - b.astype(float)
    return math.sqrt(d.var() / d.size)          # phương sai chia n, đúng như phân phối bootstrap


# Bộ P (tương quan mạnh) và bộ I (độc lập: hiệp phương sai mẫu đúng bằng 0) có cùng tỉ lệ đúng 0.80 và 0.75.
_C2_P = (1450, 150, 50, 350)
_C2_I = (1200, 400, 300, 100)
# Bộ L (độc lập, n = 500): A đúng 400, B đúng 300, cả hai đúng 240 = 400·300/500, hiệu 0.2.
_C2_L = (240, 160, 60, 40)


# ===================== BỘ TEST =====================
def run_tests(attn=None, boot=None, verbose=True):
    """Chấm hai hàm. Trả về dict {tên test: (điểm đạt, điểm tối đa, chương)}.
    Chương ghi là số nguyên (7–11) hoặc "I" (phần nhắc lại chương 1–6)."""
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
    if attn is not None:
        if verbose:
            print("C1 causal_attention")
        X, Wq, Wk, Wv = _c1_data()
        B, T, _ = X.shape
        dv = Wv.shape[1]

        def call(Xi, Wqi, Wki, Wvi):
            O, A = attn(Xi.copy(), Wqi.copy(), Wki.copy(), Wvi.copy())
            return np.asarray(O), np.asarray(A)

        def c1_shape():
            O, A = call(X, Wq, Wk, Wv)
            assert O.shape == (B, T, dv), f"O có shape {O.shape}, phải là {(B, T, dv)}"
            assert A.shape == (B, T, T), f"A có shape {A.shape}, phải là {(B, T, T)}"

        def c1_causal():
            O, A = call(X, Wq, Wk, Wv)
            upper = A[:, np.triu_indices(T, 1)[0], np.triu_indices(T, 1)[1]]
            assert np.all(np.abs(upper) <= 1e-12), "có trọng số khác 0 ở ô j > i (key tương lai)"
            assert np.allclose(A.sum(axis=-1), 1.0, rtol=0, atol=1e-6), \
                f"mỗi hàng của A phải cộng bằng 1, đang là {np.round(A.sum(axis=-1), 4)}"
            # phép thử rò rỉ: đổi X ở các vị trí t ≥ 2, đầu ra ở vị trí < 2 phải giữ nguyên
            X2 = X.copy()
            X2[:, 2:, :] = 3.0 * np.cos(np.arange(X2[:, 2:, :].size, dtype=float)).reshape(X2[:, 2:, :].shape)
            O2, A2 = call(X2, Wq, Wk, Wv)
            assert np.allclose(O2[:, :2], O[:, :2], rtol=0, atol=1e-9), "đổi token tương lai làm đổi đầu ra quá khứ (rò rỉ)"
            assert np.allclose(A2[:, :2, :2], A[:, :2, :2], rtol=0, atol=1e-9), "đổi token tương lai làm đổi A ở các hàng đầu"

        def c1_value():
            O, A = call(X, Wq, Wk, Wv)
            A_ref, O_ref = _attn_last_row_by_definition(X, Wq, Wk, Wv)
            assert np.allclose(A[:, T - 1], A_ref, rtol=0, atol=1e-10), \
                "hàng cuối của A khác tham chiếu (chia √dk với dk = số cột của Wq? softmax theo trục cuối? float64?)"
            assert np.allclose(O[:, T - 1], O_ref, rtol=0, atol=1e-10), "hàng cuối của O khác tham chiếu"
            # ví dụ tay: dk = 4 (√dk = 2), điểm hàng cuối là (0, ln 3), nên A hàng cuối = (1/4, 3/4)
            Xh = np.array([[[1.0, 0.0], [0.0, 1.0]]])
            Wqh = np.zeros((2, 4)); Wqh[1, 0] = 2.0
            Wkh = np.zeros((2, 4)); Wkh[1, 0] = math.log(3.0)
            Oh, Ah = call(Xh, Wqh, Wkh, np.eye(2))
            assert np.allclose(Ah[0, 1], [0.25, 0.75], rtol=0, atol=1e-12), f"ví dụ tay: hàng cuối của A là {Ah[0, 1]}"
            assert np.allclose(Oh[0, 1], [0.25, 0.75], rtol=0, atol=1e-12), f"ví dụ tay: hàng cuối của O là {Oh[0, 1]}"

        def c1_stable():
            # (1) Cộng hằng cực lớn, khác nhau giữa các hàng, vào điểm của từng query: kết quả phải như cũ.
            c = 3000.0 * np.array([1.0, -2.0, 3.0, -1.0, 2.0]) * np.sqrt(Wq.shape[1] + 1)
            Xa, Wqa, Wka, Wva = _augment_with_row_shift(X, Wq, Wk, Wv, c)
            X0, Wq0, Wk0, Wv0 = _augment_with_row_shift(X, Wq, Wk, Wv, np.zeros(T))
            Oa, Aa = call(Xa, Wqa, Wka, Wva)
            O0, A0 = call(X0, Wq0, Wk0, Wv0)
            assert np.all(np.isfinite(Oa)) and np.all(np.isfinite(Aa)), "tràn số (inf hoặc nan) khi điểm cỡ ±10⁴"
            assert np.allclose(Oa, O0, rtol=0, atol=1e-9) and np.allclose(Aa, A0, rtol=0, atol=1e-9), \
                "cộng một hằng vào cả hàng điểm làm đổi kết quả (trừ max theo từng hàng chưa?)"
            # (2) Key tương lai có điểm cực lớn: hàng đầu chỉ được nhìn key 1 (điểm 0) trong khi key 2 có điểm ≈ 1414.
            #     Hàng cuối có điểm (≈707, ≈−707), nên A hàng cuối = (1, 0) và O hàng cuối = v1 = (1, 0).
            Xs = np.array([[[0.0, 2000.0, 1.0, 0.0], [1000.0, -1000.0, 0.0, 1.0]]])
            Wqs = np.zeros((4, 2)); Wqs[0, 0] = Wqs[1, 1] = 1.0
            Wks = np.zeros((4, 2)); Wks[2, 0] = Wks[3, 1] = 1.0
            Os, As = call(Xs, Wqs, Wks, Wks.copy())
            assert np.all(np.isfinite(Os)) and np.all(np.isfinite(As)), \
                "tràn số khi key tương lai có điểm rất lớn (lấy max sau khi che chưa?)"
            assert np.allclose(As[0, 1], [1.0, 0.0], rtol=0, atol=1e-12) and np.allclose(Os[0, 1], [1.0, 0.0], rtol=0, atol=1e-12), \
                f"điểm cỡ ±700: hàng cuối của A là {As[0, 1]}, phải là (1, 0)"

        check("C1 shape trả về", 1, 8, c1_shape)
        check("C1 mặt nạ nhân quả (ô tương lai bằng 0, hàng cộng 1, không rò rỉ)", 2, 8, c1_causal)
        check("C1 đúng giá trị (chia √dk, softmax theo hàng)", 3, 8, c1_value)
        check("C1 ổn định số (điểm cỡ ±10⁴; key tương lai rất lớn)", 2, 8, c1_stable)

    # ---------------------------------------------------------------- C2
    if boot is not None:
        if verbose:
            print("C2 paired_bootstrap_ci")
        aP, bP = _from_counts(*_C2_P)
        aI, bI = _from_counts(*_C2_I)
        aL, bL = _from_counts(*_C2_L)

        def callb(a, b, n_boot, seed):
            out = boot(a.copy(), b.copy(), n_boot, seed)
            assert isinstance(out, (tuple, list)) and len(out) == 3, "phải trả về đúng ba số (diff, lo, hi)"
            diff, lo, hi = out
            assert all(np.ndim(v) == 0 for v in out), "diff, lo, hi phải là số vô hướng"
            return float(diff), float(lo), float(hi)

        def c2_format():
            diff, lo, hi = callb(aP, bP, 1000, 0)
            assert all(math.isfinite(v) for v in (diff, lo, hi)), "có giá trị không hữu hạn"
            assert lo <= hi, f"lo = {lo} lớn hơn hi = {hi}"

        def c2_seed():
            r1 = callb(aP, bP, 1000, 7)
            r2 = callb(aP, bP, 1000, 7)
            assert r1 == r2, "cùng seed nhưng kết quả khác nhau (đã dùng default_rng(seed) chưa?)"
            r3 = callb(aP, bP, 1000, 8)
            assert r1[2] != r1[1], "khoảng có độ rộng 0: mọi lần bootstrap giống nhau (tạo rng trong vòng lặp?)"
            assert (r3[1], r3[2]) != (r1[1], r1[2]), \
                "đổi seed mà khoảng không đổi: chưa lấy mẫu lại thật, hoặc chưa dùng seed"

        def c2_diff():
            diff, _, _ = callb(np.array([1, 1, 1, 0, 1]), np.array([1, 0, 1, 0, 0]), 200, 0)
            assert abs(diff - 0.4) < 1e-12, f"ví dụ tay: diff = {diff}, phải là 0.8 − 0.4 = 0.4 (tỉ lệ, A − B)"
            diff, _, _ = callb(aP, bP, 200, 0)
            assert abs(diff - 0.05) < 1e-12, f"diff = {diff}, phải là 0.80 − 0.75 = 0.05"

        def c2_paired():
            dP, loP, hiP = callb(aP, bP, 4000, 0)
            dI, loI, hiI = callb(aI, bI, 4000, 0)
            assert loP <= dP <= hiP and loI <= dI <= hiI, \
                "khoảng [lo, hi] không chứa chính hiệu diff (phân vị 2.5 và 97.5 của phân phối bootstrap?)"
            wP, wI = hiP - loP, hiI - loI
            assert wI > 0, "khoảng có độ rộng 0 (lấy mẫu có hoàn lại chưa?)"
            ratio = wP / wI
            expect = _paired_se(aP, bP) / _paired_se(aI, bI)          # 0.530
            assert 0.85 * expect <= ratio <= 1.15 * expect, (
                f"tỉ số độ rộng CI (bộ tương quan / bộ độc lập) = {ratio:.3f}, ghép cặp đúng cho ≈ {expect:.3f}; "
                "≈ 1 nghĩa là A và B được lấy mẫu lại bằng hai bộ chỉ số khác nhau")

        def c2_level():
            diff, lo, hi = callb(aL, bL, 10000, 0)
            s = diff / 0.2                                            # 1 nếu đúng; −1 nếu ngược dấu; 100 nếu phần trăm
            assert abs(s) > 1e-9, "diff bằng 0 nên không chấm được mức"
            se = _paired_se(aL, bL)                                   # 0.02828
            width = (hi - lo) / abs(s)
            target = 2 * 1.959964 * se
            assert 0.93 * target <= width <= 1.07 * target, \
                f"độ rộng CI {width:.4f}, khoảng 95% phải ≈ {target:.4f} (phân vị 2.5 và 97.5?)"
            center = 0.5 * (lo + hi) / s
            assert abs(center - 0.2) <= 0.25 * se, f"tâm khoảng {center:.4f} lệch xa hiệu 0.2"

        check("C2 định dạng (ba số hữu hạn, lo ≤ hi)", 1, 11, c2_format)
        check("C2 tất định theo seed, có lấy mẫu lại thật", 1, 11, c2_seed)
        check("C2 hiệu đúng (A − B, dạng tỉ lệ)", 1, "I", c2_diff)
        check("C2 ghép cặp (cùng chỉ số cho A và B)", 3, 11, c2_paired)
        check("C2 mức 95% (phân vị 2.5 và 97.5)", 2, "I", c2_level)
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


def _ch_key(ch):
    return (1, ch) if isinstance(ch, int) else (0, 0)


def report(score, fns):
    auto = sum(v[0] for v in score.values())
    total = sum(v[1] for v in score.values())
    by_ch = {}
    for got, mx, ch in score.values():
        a, b = by_ch.get(ch, (0, 0))
        by_ch[ch] = (a + got, b + mx)
    print(f"\nĐiểm phần tự động: {auto} / {total}")
    print("Chia về chương (ghi vào bảng chấm theo chương, cộng thêm phần chấm tay):")
    for ch in sorted(by_ch, key=_ch_key):
        name = "Phần I (chương 3)" if ch == "I" else f"chương {ch}"
        print(f"  {name}: {by_ch[ch][0]} / {by_ch[ch][1]}")
    print("Chấm tay thêm theo lời giải trong đề: C1(b) 2 điểm (chương 8), C2(b) 2 điểm (Phần I, chương 3).")
    msgs = {"causal_attention": "lặp theo mẫu trong batch hay theo vị trí (token)",
            "paired_bootstrap_ci": "lặp theo câu test (lặp theo lần bootstrap thì được phép)"}
    for name, fn in fns.items():
        if fn is None:
            continue
        loops = find_loops(fn)
        if loops:
            print(f"Chú ý: {name} có vòng lặp ở dòng {loops}. Nếu đó là vòng {msgs[name]} thì trừ 2 điểm "
                  "(điểm (a) của câu đó không âm).")


def _load(path):
    spec = importlib.util.spec_from_file_location("bai_lam", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chấm tự động Phần C, Đề thi thử 2 (16 điểm).")
    ap.add_argument("--student", required=True, help="file .py định nghĩa causal_attention và paired_bootstrap_ci")
    args = ap.parse_args()
    try:
        mod = _load(args.student)
    except Exception as e:  # noqa: BLE001
        print(f"Không nạp được {args.student}: {type(e).__name__}: {e}")
        sys.exit(1)
    fns = {"causal_attention": getattr(mod, "causal_attention", None),
           "paired_bootstrap_ci": getattr(mod, "paired_bootstrap_ci", None)}
    for name, fn in fns.items():
        if fn is None:
            print(f"Không thấy hàm {name} trong {args.student}: phần đó 0 điểm.")

    def _missing(*_a, **_k):  # hàm thiếu được chấm như hàm chưa viết
        raise NotImplementedError

    sc = run_tests(fns["causal_attention"] or _missing, fns["paired_bootstrap_ci"] or _missing)
    report(sc, fns)
