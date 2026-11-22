"""Bộ chấm tự động Vòng 2 (Coding interview mô phỏng) của Đề thi thử 3. Chỉ dùng thư viện chuẩn và NumPy.

CHỈ CHẠY SAU KHI HẾT GIỜ. File này không chứa lời giải, nhưng có dữ liệu test và vài đáp số tính tay,
nên đừng mở nó trong lúc làm đề.

Cách dùng, từ thư mục gốc của sách:
    python labs/thi-thu-3/cham_coding.py --student bai_lam.py

`bai_lam.py` là bản bạn chép từ `labs/thi-thu-3/bai_lam_mau.py` rồi điền lớp `MetricLog` và hàm `corpus_bleu`.
Phần nào còn `raise NotImplementedError` thì được 0 điểm phần đó, không làm hỏng phần kia.

Thang điểm tự động bám đúng thang trong đề (mục Lời giải, Vòng 2); mỗi bài 7 điểm tự động + 3 điểm chấm tay
(phần nói to cách tiếp cận, độ phức tạp, câu hỏi nối tiếp), tổng Vòng 2 là 20.
  Bài 1 (MetricLog), 7 điểm: giữa hai bản ghi và sau bản ghi cuối 2, đúng bằng step đã ghi 1, chưa có bản ghi
        trả None 1, nhiều lần chạy xen kẽ 1, tốc độ 2 (2·10⁵ lời gọi trong giới hạn thời gian, và phải đúng:
        7·10⁴ lần record, rồi 3·10⁴ cặp record/at xen kẽ, rồi khoảng 7·10⁴ lần at).
  Bài 2 (corpus_bleu), 7 điểm: định dạng 1, cắt số đếm (clipping) 2, cộng dồn cấp corpus 1, brevity penalty 2,
        kết hợp thành BLEU (trung bình nhân, trọng số 1/max_n, precision bằng 0, mẫu số 0, c = 0) 1.
Mỗi test cố gắng đo đúng một ý:
  - Bài 1: truy vấn "sau bản ghi cuối" chỉ có ở test "giữa"; các truy vấn của test "nhiều lần chạy" và "tốc độ"
    đều rơi CHẶT giữa hai step đã ghi. Vì vậy lỗi "<" thay vì "≤" chỉ bị trừ ở test "đúng bằng step", và lỗi trả
    None sau bản ghi cuối chỉ bị trừ ở test "giữa". Đáp án của các test được dựng sẵn khi sinh dữ liệu (truy vấn
    sinh từ chỉ số bản ghi), bộ chấm không chứa cách tìm kiếm nào.
  - Bài 2: test "cắt" và "corpus" chỉ so precisions, test "BP" chỉ so bp, còn test "kết hợp" so bleu với chính
    precisions và bp mà hàm của bạn trả về. Một lỗi ở MỘT thành phần vì thế chỉ bị trừ ở test của thành phần đó.
    Lỗi khái niệm làm sai nhiều thành phần cùng lúc bị trừ ở mỗi thành phần nó làm sai: tính mọi thứ theo câu rồi
    lấy trung bình còn 3/7; làm trơn cộng 1 ở mọi n còn 3/7; làm trơn cộng 1 từ n ≥ 2 còn 4/7.
Mỗi test chạy trong một tiến trình Python riêng có giới hạn thời gian; vòng lặp vô hạn hay bài quá chậm chỉ làm
mất điểm test đó. Dữ liệu ngẫu nhiên sinh bằng random.Random với seed cố định, nên chạy lại cho cùng kết quả.
File bài làm import thư viện ngoài bị cấm (nltk, sacrebleu, sortedcontainers) thì CẢ HAI bài 0 điểm tự động
(một file chung cho hai bài). File được đọc bằng UTF-8, chấp nhận BOM (Notepad trên Windows).

Kiểm với Python 3.13 ngày 2026-11-19; không dùng cú pháp mới hơn Python 3.8.
"""
import argparse
import ast
import importlib.util
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

try:  # để in tiếng Việt trên console Windows khi output bị chuyển hướng
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

MARK = "@@KET_QUA@@"
FORBIDDEN = {"nltk", "sacrebleu", "sortedcontainers"}
SPEED_LIMIT_S = 5.0          # giới hạn cho 2·10⁵ lời gọi của test tốc độ
DEFAULT_TIMEOUT_S = 20.0     # giới hạn mỗi tiến trình test

# (tên, bài, điểm, mô tả, giới hạn giây của tiến trình)
TESTS = [
    ("b1_giua", 1, 2, "truy vấn chặt giữa hai bản ghi và sau bản ghi cuối", DEFAULT_TIMEOUT_S),
    ("b1_bang", 1, 1, "truy vấn đúng bằng một step đã ghi (kể cả step 0)", DEFAULT_TIMEOUT_S),
    ("b1_khong_co", 1, 1, "trước bản ghi đầu và lần chạy chưa ghi: trả None", DEFAULT_TIMEOUT_S),
    ("b1_nhieu_run", 1, 1, "ba lần chạy ghi xen kẽ, mỗi lần chạy độc lập", DEFAULT_TIMEOUT_S),
    ("b1_toc_do", 1, 2, f"2·10⁵ lời gọi, có pha record/at xen kẽ, trong {SPEED_LIMIT_S:.0f} giây, đáp án đúng", DEFAULT_TIMEOUT_S),
    ("b2_dinh_dang", 2, 1, "trả (float, list max_n float, float), chạy với max_n = 4 và 3", DEFAULT_TIMEOUT_S),
    ("b2_cat", 2, 2, "precision có cắt số đếm theo câu tham chiếu", DEFAULT_TIMEOUT_S),
    ("b2_corpus", 2, 1, "precision cấp corpus: cắt theo từng câu, cộng tử và mẫu trên cả corpus", DEFAULT_TIMEOUT_S),
    ("b2_bp", 2, 2, "brevity penalty dùng tổng độ dài cả corpus, đúng chiều r/c", DEFAULT_TIMEOUT_S),
    ("b2_ket_hop", 2, 1, "bleu = bp·exp(trung bình ln p), trọng số 1/max_n; p = 0, mẫu số 0, c = 0 cho 0.0; corpus 3 000 câu", DEFAULT_TIMEOUT_S),
]


# ===================================================================== tiện ích
def load_student(path):
    spec = importlib.util.spec_from_file_location("bai_lam_hoc_vien", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def forbidden_imports(path):
    """Tên các thư viện bị cấm mà file import (quét cây cú pháp, không chạy code)."""
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))   # utf-8-sig: bỏ BOM nếu có
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return set()                                                   # file hỏng: các test sẽ tự báo lỗi
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found & FORBIDDEN


def close(x, y, tol=1e-9):
    return abs(float(x) - float(y)) <= tol * max(1.0, abs(float(y)))


# ===================================================================== test bài 1
def _b1_base(mod):
    log = mod.MetricLog()
    for s, v in [(10, 0.9), (20, 0.8), (35, 0.7), (100, 0.5)]:
        log.record("a", s, v)
    return log


def t_b1_giua(mod):
    log = _b1_base(mod)
    for q, want in [(15, 0.9), (19, 0.9), (34, 0.8), (36, 0.7), (99, 0.7), (1000, 0.5)]:
        got = log.at("a", q)
        assert got is not None and close(got, want), f"at('a', {q}) = {got!r}, cần {want}"
    log.record("a", 120, 0.45)
    for q, want in [(119, 0.5), (121, 0.45)]:
        got = log.at("a", q)
        assert got is not None and close(got, want), f"sau khi ghi thêm step 120: at('a', {q}) = {got!r}, cần {want}"


def t_b1_bang(mod):
    log = _b1_base(mod)
    for q, want in [(10, 0.9), (20, 0.8), (35, 0.7), (100, 0.5)]:
        got = log.at("a", q)
        assert got is not None and close(got, want), f"at('a', {q}) = {got!r}, cần {want} (step {q} đã được ghi)"
    log.record("z", 0, 1.25)
    got = log.at("z", 0)
    assert got is not None and close(got, 1.25), f"at('z', 0) = {got!r}, cần 1.25"


def t_b1_khong_co(mod):
    log = _b1_base(mod)
    for run, q in [("a", 9), ("a", 0), ("chua_ghi", 50)]:
        got = log.at(run, q)
        assert got is None, f"at({run!r}, {q}) = {got!r}, cần None"


def t_b1_nhieu_run(mod):
    log = mod.MetricLog()
    for t in range(1, 301):                       # r1 ghi ở 3t, r2 ở 10k, r3 ở 21k; ghi xen kẽ theo thời gian
        log.record("r1", 3 * t, 1000.0 + t)
        if t % 2 == 0:
            log.record("r2", 5 * t, 2000.0 + t // 2)
        if t % 3 == 0:
            log.record("r3", 7 * t, 3000.0 + t // 3)
    # mọi truy vấn chặt giữa hai bản ghi của chính lần chạy đó (không có truy vấn sau bản ghi cuối)
    checks = [("r1", 3 * t + 1, 1000.0 + t) for t in (1, 50, 298, 299)]
    checks += [("r2", 10 * k + 3, 2000.0 + k) for k in (1, 37, 149)]
    checks += [("r3", 21 * k + 5, 3000.0 + k) for k in (1, 64, 99)]
    for run, q, want in checks:
        got = log.at(run, q)
        assert got is not None and close(got, want), f"at({run!r}, {q}) = {got!r}, cần {want}"


def t_b1_toc_do(mod):
    """2·10⁵ lời gọi, đúng giới hạn của docstring. Pha 1: 7·10⁴ lần record (lần chạy "nho" ghi xen kẽ).
    Pha 2: 3·10⁴ cặp record rồi at xen kẽ (bắt bài dựng lại cấu trúc sau mỗi lần ghi). Pha 3: at cho đủ 2·10⁵."""
    rnd = random.Random(2027)
    n, n_pair, total = 100_000, 30_000, 200_000
    n1 = n - n_pair
    gaps = [rnd.randint(2, 10_000) for _ in range(n)]   # khoảng cách ≥ 2 nên luôn có step chặt giữa hai bản ghi
    steps, s = [], 0
    for g in gaps:
        s += g
        steps.append(s)

    def between(k):                               # (step truy vấn, đáp án): chặt giữa bản ghi k và k + 1
        return rnd.randint(steps[k] + 1, steps[k + 1] - 1), float(k)

    pair_q = [between(rnd.randrange(j)) for j in range(n1, n)]        # ở bước j, bản ghi 0..j đã có
    n_small = (n1 + 999) // 1000
    n3 = total - n1 - n_small - 2 * n_pair
    tail_q = [between(rnd.randrange(n - 1)) for _ in range(n3)]
    log = mod.MetricLog()
    got_pair = []
    t0 = time.perf_counter()
    for j in range(n1):
        log.record("lon", steps[j], float(j))
        if j % 1000 == 0:
            log.record("nho", j + 1, -1.0)        # một lần chạy khác ghi xen kẽ
    for j, (q, _) in zip(range(n1, n), pair_q):
        log.record("lon", steps[j], float(j))
        got_pair.append(log.at("lon", q))
    got_tail = [log.at("lon", q) for q, _ in tail_q]
    el = time.perf_counter() - t0
    bad = [(q, g, w) for (q, w), g in zip(pair_q + tail_q, got_pair + got_tail) if g is None or not close(g, w)]
    assert not bad, f"{len(bad)} truy vấn sai, ví dụ at('lon', {bad[0][0]}) = {bad[0][1]!r}, cần {bad[0][2]}"
    assert el <= SPEED_LIMIT_S, f"đúng nhưng chậm: {el:.1f} giây cho 2·10⁵ lời gọi (giới hạn {SPEED_LIMIT_S:.0f} giây)"
    return f"{el:.2f} giây"


# ===================================================================== test bài 2
def _split(s):
    return s.split()


def _call_bleu(mod, hyps, refs, max_n):
    out = mod.corpus_bleu(hyps, refs, max_n=max_n)
    assert isinstance(out, (tuple, list)) and len(out) == 3, f"cần trả về (bleu, precisions, bp), nhận {type(out).__name__}"
    bleu, prec, bp = out
    prec = list(prec)
    assert len(prec) == max_n, f"precisions có {len(prec)} phần tử, cần {max_n}"
    return bleu, prec, bp


def t_b2_dinh_dang(mod):
    hyps = [_split("the cat sat on the mat today"), _split("a dog runs in the park")]
    refs = [_split("the cat sat on the red mat today"), _split("a dog runs in a park")]
    for mn in (4, 3):
        bleu, prec, bp = _call_bleu(mod, hyps, refs, mn)
        for name, v in [("bleu", bleu), ("bp", bp)] + [(f"precisions[{i}]", p) for i, p in enumerate(prec)]:
            assert isinstance(v, (float, int)) and not isinstance(v, bool), f"{name} không phải số thực: {type(v).__name__}"
            assert math.isfinite(float(v)) and float(v) >= 0.0, f"{name} = {v!r} không hữu hạn hoặc âm"


def t_b2_cat(mod):
    cases = [  # (hyp, ref, (p1, p2)) tính tay
        ("the the the the cat", "the cat sat on a mat", (2 / 5, 1 / 4)),
        ("a a b", "a a a", (2 / 3, 1 / 2)),
    ]
    for h, r, want in cases:
        _, prec, _ = _call_bleu(mod, [_split(h)], [_split(r)], 2)
        assert close(prec[0], want[0]) and close(prec[1], want[1]), \
            f"hyp {h!r}, ref {r!r}: precisions {[round(float(p), 4) for p in prec]}, cần {[round(w, 4) for w in want]}"


def t_b2_corpus(mod):
    hyps = [_split("a b c d"), _split("x y"), _split("q q"), _split("r")]
    refs = [_split("a b c d"), _split("y z"), _split("r r"), _split("q q")]
    _, prec, _ = _call_bleu(mod, hyps, refs, 1)
    assert close(prec[0], 5 / 9), f"precision 1-gram của corpus 4 câu = {float(prec[0]):.4f}, cần 5/9 ≈ 0.5556"


def t_b2_bp(mod):
    toks = [f"w{i}" for i in range(12)]
    cases = [  # (hyps, refs, bp đúng)
        ([toks[:9]], [toks[:12]], math.exp(1 - 12 / 9)),                    # c < r
        ([toks[:12]], [toks[:9]], 1.0),                                     # c > r
        ([toks[:9]], [toks[:9]], 1.0),                                      # c = r
        ([toks[:3], toks[:6]], [toks[:6], toks[:3]], 1.0),                  # c = r trên cả corpus, khác nhau theo câu
    ]
    for hyps, refs, want in cases:
        _, _, bp = _call_bleu(mod, hyps, refs, 1)
        c, r = sum(map(len, hyps)), sum(map(len, refs))
        assert close(bp, want), f"độ dài theo câu {[len(h) for h in hyps]} so với {[len(x) for x in refs]} (c = {c}, r = {r}): bp = {float(bp):.4f}, cần {want:.4f}"


def _random_corpus(seed, n_sent):
    rnd = random.Random(seed)
    vocab = [f"t{i}" for i in range(60)]
    hyps, refs = [], []
    for _ in range(n_sent):
        ref = [rnd.choice(vocab) for _ in range(rnd.randint(6, 25))]
        hyp = list(ref)
        for _ in range(rnd.randint(0, 4)):                                   # sửa vài chỗ: thay, xóa, chèn
            op = rnd.random()
            if op < 0.4 and hyp:
                hyp[rnd.randrange(len(hyp))] = rnd.choice(vocab)
            elif op < 0.7 and len(hyp) > 1:
                del hyp[rnd.randrange(len(hyp))]
            else:
                hyp.insert(rnd.randrange(len(hyp) + 1), rnd.choice(vocab))
        hyps.append(hyp)
        refs.append(ref)
    return hyps, refs


def t_b2_ket_hop(mod):
    hyps, refs = _random_corpus(17, 3000)
    t0 = time.perf_counter()
    for mn in (4, 2):
        bleu, prec, bp = _call_bleu(mod, hyps, refs, mn)
        assert all(float(p) > 0 for p in prec), f"max_n = {mn}: precisions của bạn có số 0 trên corpus này: {prec}"
        want = float(bp) * math.exp(sum(math.log(float(p)) for p in prec) / mn)
        assert close(bleu, want), f"max_n = {mn}: bleu = {float(bleu):.6f}, nhưng bp·exp(trung bình ln p) từ chính kết quả của bạn là {want:.6f}"
    el = time.perf_counter() - t0
    bleu, prec, bp = _call_bleu(mod, [_split("a b c")], [_split("a x c")], 2)
    assert isinstance(bleu, (float, int)) and math.isfinite(float(bleu)) and float(bleu) == 0.0, \
        f"có precision bằng 0 (2-gram) thì bleu phải là 0.0, nhận {bleu!r}"
    # hai mệnh đề của docstring: "mẫu số bằng 0 thì precision là 0.0" và "bp = 0.0 nếu c = 0"
    bleu, prec, bp = _call_bleu(mod, [[], ["a"]], [["a", "b"], ["a"]], 2)
    assert float(prec[1]) == 0.0 and float(bleu) == 0.0, \
        f"mẫu số 2-gram bằng 0 (câu dài 0 và 1): precisions[1] và bleu phải là 0.0, nhận {prec!r}, {bleu!r}"
    bleu, prec, bp = _call_bleu(mod, [[]], [["a"]], 2)       # chỉ so bleu: giá trị bp đã được test "BP" chấm riêng
    assert float(bleu) == 0.0, f"c = 0 (câu máy dịch rỗng): bleu phải là 0.0 (không chia cho 0), nhận {bleu!r}"
    return f"{el:.2f} giây cho corpus 3 000 câu"


RUNNERS = {name: globals()["t_" + name] for name, *_ in TESTS}


# ===================================================================== chạy một test trong tiến trình con
def run_one(name, student):
    try:
        mod = load_student(student)
        info = RUNNERS[name](mod)
        res = {"ok": True, "msg": info or ""}
    except NotImplementedError:
        res = {"ok": False, "msg": "chưa làm (NotImplementedError)"}
    except AssertionError as e:
        res = {"ok": False, "msg": str(e)}
    except Exception as e:  # noqa: BLE001
        res = {"ok": False, "msg": f"lỗi {type(e).__name__}: {e}"}
    print(MARK + json.dumps(res, ensure_ascii=False), flush=True)


def run_tests(student, verbose=True):
    """Chấm cả hai bài. Trả về dict {tên test: (điểm đạt, điểm tối đa, bài)}."""
    bad_imports = forbidden_imports(student)
    out = {}
    for name, bai, pts, desc, limit in TESTS:
        try:
            cp = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--_run", name, "--student", str(student)],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=limit)
            line = [ln for ln in cp.stdout.splitlines() if ln.startswith(MARK)]
            res = json.loads(line[-1][len(MARK):]) if line else {"ok": False, "msg": "tiến trình dừng bất thường: " + cp.stderr.strip()[-300:]}
        except subprocess.TimeoutExpired:
            res = {"ok": False, "msg": f"quá {limit:.0f} giây (chậm hoặc lặp vô hạn)"}
        if bad_imports:
            res = {"ok": False, "msg": "import thư viện bị cấm: " + ", ".join(sorted(bad_imports))}
        got = pts if res["ok"] else 0
        out[name] = (got, pts, bai)
        if verbose:
            tag = "ĐẠT " if res["ok"] else "TRƯỢT"
            extra = f" ({res['msg']})" if res["msg"] else ""
            print(f"  [{tag}] Bài {bai} · {name:13s} {got}/{pts}  {desc}{extra}")
    return out


def summarize(score):
    b = {1: [0, 0], 2: [0, 0]}
    for got, pts, bai in score.values():
        b[bai][0] += got
        b[bai][1] += pts
    print()
    print(f"Bài 1 (MetricLog):   {b[1][0]}/{b[1][1]} điểm tự động; cộng tối đa 3 điểm chấm tay (nói to, độ phức tạp, câu nối tiếp)")
    print(f"Bài 2 (corpus_bleu): {b[2][0]}/{b[2][1]} điểm tự động; cộng tối đa 3 điểm chấm tay")
    print(f"Tổng tự động: {b[1][0] + b[2][0]}/14. Chấm tay 6 điểm còn lại theo mục Lời giải, Vòng 2. Vòng 2 tổng 20.")
    return b


def main():
    ap = argparse.ArgumentParser(description="Chấm Vòng 2, Đề thi thử 3")
    ap.add_argument("--student", required=True, help="đường dẫn tới bai_lam.py")
    ap.add_argument("--_run", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args._run:
        run_one(args._run, args.student)
        return
    path = Path(args.student)
    if not path.exists():
        sys.exit(f"Không thấy {path}. Chép labs/thi-thu-3/bai_lam_mau.py thành bai_lam.py rồi điền bài.")
    print(f"Chấm {path} (mỗi test một tiến trình riêng; test tốc độ có thể mất tới {DEFAULT_TIMEOUT_S:.0f} giây)")
    summarize(run_tests(path))


if __name__ == "__main__":
    main()
