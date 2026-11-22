"""Sinh hai notebook cho lab chương 14: lab.ipynb (có TODO) và loigiai.ipynb (lời giải).

Chạy lại script này nếu muốn tạo lại notebook gốc:  python labs/build_chuong14.py

Năm lab là công cụ luyện phỏng vấn kỹ thuật (chọn chuỗi theo sổ lỗi, lịch ôn, phân tích bản chép lời,
kiểm suy dẫn bảng trắng, máy tính chi phí LLM) cộng một bài săn bug. Chỉ dùng numpy và pandas.
Mọi băng thông, dung lượng GPU, TFLOP/s là SỐ GIẢ ĐỊNH của đề, không gán cho GPU cụ thể nào.
"""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "chuong14"
OUT.mkdir(parents=True, exist_ok=True)

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

SETUP = r"""import io, re
import numpy as np
import pandas as pd
np.set_printoptions(precision=4, suppress=True)
GiB = 2**30
# Lab nào có ngẫu nhiên thì tự tạo Generator riêng với seed cố định ngay trong lab đó,
# nên số của lab này không phụ thuộc code bạn viết ở lab khác."""

# =====================================================================================
# LAB 1 — chọn chuỗi theo sổ lỗi
# =====================================================================================
LOG_DATA = r"""# Sổ lỗi MẪU (số giả): điểm tự chấm lần gần nhất trên thang 12 của chương 14, và số ngày từ lần luyện đó.
LOG_CSV = '''chuoi,diem,ngay
C01,10,2
C02,6,5
C03,11,1
C04,4,9
C05,9,3
C06,12,0
C07,7,6
C08,5,8
C09,8,4
C10,12,1
C11,3,12
C12,9,2
C13,10,3
C14,6,7
C15,11,1
C16,8,5
C17,7,6
C18,12,0
'''
MAX_SCORE = 12"""

LAB1_FN = ("""def read_log(text_or_path):
    \"\"\"Đọc sổ lỗi CSV có cột chuoi,diem,ngay. Nhận chuỗi CSV hoặc đường dẫn file. Trả về DataFrame.\"\"\"
    # TODO
    raise NotImplementedError


def weights(score, days, max_score=MAX_SCORE):
    \"\"\"Trọng số luyện lại của mỗi chuỗi (quy tắc của sách): (max_score − điểm + 1) · (1 + ngày/7).
    score, days là mảng numpy cùng độ dài. Trả về mảng trọng số.\"\"\"
    # TODO
    raise NotImplementedError


def pick(log, k=4, seed=0):
    \"\"\"Rút k chuỗi KHÔNG lặp, xác suất tỉ lệ với trọng số. Dùng np.random.default_rng(seed) và rng.choice
    với replace=False, p=trọng số đã chuẩn hóa. Trả về list tên chuỗi theo thứ tự rút.\"\"\"
    # TODO
    raise NotImplementedError""",
r"""def read_log(text_or_path):
    src = io.StringIO(text_or_path) if "\n" in str(text_or_path) else text_or_path
    return pd.read_csv(src)


def weights(score, days, max_score=MAX_SCORE):
    return (max_score - np.asarray(score) + 1) * (1 + np.asarray(days) / 7)


def pick(log, k=4, seed=0):
    w = weights(log["diem"].to_numpy(), log["ngay"].to_numpy())
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(log), size=k, replace=False, p=w / w.sum())
    return [log["chuoi"].iloc[i] for i in idx]""")

LAB1_CHECK = r"""log = read_log(LOG_CSV)
w = weights(log["diem"].to_numpy(), log["ngay"].to_numpy())
p = w / w.sum()
log = log.assign(w=w.round(3), p=p.round(4))
print(log.sort_values("w", ascending=False).head(5).to_string(index=False))
i_max, i_min = int(np.argmax(w)), int(np.argmin(w))
print("lớn nhất:", log.chuoi[i_max], round(w[i_max], 3), "| nhỏ nhất:", log.chuoi[i_min], round(w[i_min], 3),
      "| tỉ số xác suất:", round(p[i_max] / p[i_min], 1))
chosen = pick(log, k=4, seed=0)
print("tuần này luyện (seed 0):", chosen)
# Kiểm chính hàm pick của bạn: 4000 seed khác nhau, mỗi lần rút 1 chuỗi
freq = np.mean([pick(log, k=1, seed=s)[0] == log.chuoi[i_max] for s in range(4000)])
print("tần suất pick rút được", log.chuoi[i_max], "qua 4000 seed:", round(freq, 4), "| xác suất lý thuyết:", round(p[i_max], 4))
assert log.chuoi[i_max] == "C11" and abs(w[i_max] - 27.143) < 1e-3
assert log.chuoi[i_min] == "C06" and abs(w[i_min] - 1.0) < 1e-12
assert len(chosen) == 4 and len(set(chosen)) == 4
assert chosen == pick(log, k=4, seed=0), "cùng seed phải ra cùng lựa chọn"
assert chosen == ["C11", "C05", "C02", "C01"], "seed 0 phải cho đúng lựa chọn đã công bố trong sách"
assert all(len(set(pick(log, k=4, seed=s))) == 4 for s in range(300)), "pick không được rút lặp"
assert abs(freq - p[i_max]) < 0.02, "tần suất rút không khớp xác suất theo trọng số"
print("Lab 1 ĐẠT ✔")"""

# =====================================================================================
# LAB 2 — lịch Leitner
# =====================================================================================
LAB2_DATA = r"""# Hộp Leitner giống trang Ôn tập của app: hộp 1, 2, 3, 4 ôn lại sau 1, 3, 7, 14 ngày; quên thì về hộp 1.
INTERVAL = {1: 1, 2: 3, 3: 7, 4: 14}

def pass_rule(i, d):
    '''Quy tắc qua/trượt XÁC ĐỊNH để số tái lập: chuỗi i ở ngày d trượt khi (7i + d) chia hết cho 5 (qua 80%).'''
    return (i * 7 + d) % 5 != 0"""

LAB2_FN = ("""def schedule(n_items=18, days=28, new_per_day=1, pass_fn=pass_rule, interval=INTERVAL):
    \"\"\"Mô phỏng lịch ôn. Mỗi ngày d (0..days−1): thêm new_per_day chuỗi mới vào hộp 1, đến hạn ngay ngày d
    (cho tới khi đủ n_items). Mọi chuỗi đến hạn ngày d được luyện; qua thì lên một hộp (tối đa hộp 4), trượt thì về hộp 1;
    hạn mới = d + interval[hộp mới]. Trả về mảng int dài `days`: số chuỗi phải luyện mỗi ngày.\"\"\"
    # TODO
    raise NotImplementedError""",
r"""def schedule(n_items=18, days=28, new_per_day=1, pass_fn=pass_rule, interval=INTERVAL):
    box, due, load, added = {}, {}, np.zeros(days, int), 0
    for d in range(days):
        for _ in range(new_per_day):
            if added < n_items:
                box[added], due[added] = 1, d
                added += 1
        today = [i for i in box if due[i] == d]
        load[d] = len(today)
        for i in today:
            box[i] = min(box[i] + 1, 4) if pass_fn(i, d) else 1
            due[i] = d + interval[box[i]]
    return load""")

LAB2_CHECK = r"""load = schedule()
print("tải mỗi ngày:", load.tolist())
heavy = [int(d) + 1 for d in np.flatnonzero(load == load.max())]
print("tổng", int(load.sum()), "lượt; ngày nặng nhất", int(load.max()), "chuỗi, vào ngày", heavy,
      "->", int(load.max()) * 12, "phút nói nếu mỗi chuỗi 12 phút")
rng2 = np.random.default_rng(2)                          # RNG riêng: biến thể qua/trượt ngẫu nhiên 80%
load_r = schedule(pass_fn=lambda i, d: rng2.random() < 0.8)
print("biến thể ngẫu nhiên (seed 2):", load_r.tolist(), "| tổng", int(load_r.sum()), "| nặng nhất", int(load_r.max()))
assert load.tolist() == [1, 2, 1, 1, 3, 2, 3, 2, 2, 3, 2, 6, 3, 3, 3, 2, 6, 3, 3, 2, 1, 3, 1, 2, 0, 2, 4, 3]
assert load.sum() == 69 and load.max() == 6 and heavy == [12, 17]
assert schedule(pass_fn=lambda i, d: True)[:4].tolist() == [1, 1, 1, 2]
print("Lab 2 ĐẠT ✔")"""

# =====================================================================================
# LAB 3 — phân tích bản chép lời
# =====================================================================================
LAB3_DATA = r"""# Bản chép lời MẪU (tự soạn) của một câu trả lời tầng 1 chuỗi C01, nói trong 50 giây.
# Dùng ở tuần 38 cho bản chép lời buổi phỏng vấn thử của bạn.
TRANSCRIPT = ("um so the gradient of cross entropy with respect to the logits is p minus y . "
  "basically you take the softmax output and subtract the one hot label . "
  "uh the reason is that the log of the softmax cancels the exponential , so you know the derivative is linear in p . "
  "we checked it with finite differences and the relative error was about ten to the minus ten . "
  "like I think the key point is that the gradient never saturates when the model is wrong , "
  "so I would say it trains faster than mean squared error with a sigmoid . "
  "um one limitation is that it says nothing about calibration , which I measured separately on 2 000 examples .")
SECONDS = 50
FILLERS = ["um", "uh", "like", "you know", "basically", "kind of", "i think", "i would say"]
# Ngưỡng của SÁCH (không phải của chương trình tuyển nào): 120–160 từ/phút; từ đệm <= 3 trên 100 từ.


def basic_stats(text, seconds, fillers=FILLERS):
    '''ĐƯA SẴN, không phải viết lại: cùng việc với talk_stats ở Lab 4 chương 13 (từ, từ/phút, từ đệm), nhưng nhận
    văn bản và tổng số giây thay cho mốc thời gian từng từ, và khớp cả cụm từ đệm nhiều chữ ("you know").'''
    t = " " + re.sub(r"\s+", " ", text.lower()) + " "
    toks = [w for w in t.split() if re.search(r"[a-z0-9]", w)]
    fill = {f: len(re.findall(r"(?<= )" + re.escape(f) + r"(?= )", t)) for f in fillers}
    n_fill = sum(fill.values())
    return dict(words=len(toks), wpm=round(len(toks) / seconds * 60, 1), fillers=n_fill,
                per100=round(100 * n_fill / len(toks), 1), fill_detail={k: v for k, v in fill.items() if v})"""

LAB3_FN = ("""def voice_and_digits(text):
    \"\"\"Phần MỚI so với chương 13. Trả về dict: I, we (số token đúng bằng 'i', 'we' sau khi chuyển chữ thường,
    token tách theo khoảng trắng), digits (số cụm chữ số trong văn bản gốc; '2 000' tính là một cụm).\"\"\"
    # TODO
    raise NotImplementedError""",
r"""def voice_and_digits(text):
    toks = text.lower().split()
    return dict(I=sum(w == "i" for w in toks), we=sum(w == "we" for w in toks),
                digits=len(re.findall(r"\b\d[\d ]*\b", text)))""")

LAB3_CHECK = r"""r = {**basic_stats(TRANSCRIPT, SECONDS), **voice_and_digits(TRANSCRIPT)}
print(r)
print("tốc độ trong vùng 120–160:", 120 <= r["wpm"] <= 160, "| từ đệm <= 3 trên 100 từ:", r["per100"] <= 3)
clean = re.sub(r"\b(um|uh|basically|you know|like|i think|i would say)\b ?", "", TRANSCRIPT.lower())
print("bỏ hết từ đệm:", basic_stats(clean, SECONDS)["words"], "từ")
assert r["words"] == 116 and abs(r["wpm"] - 139.2) < 0.05 and r["fillers"] == 8
assert r["I"] == 3 and r["we"] == 1 and r["digits"] == 1
assert voice_and_digits("i said it is fine, We ran 3 seeds on 2 000 items and I think we won") == dict(I=2, we=2, digits=2)
print("Lab 3 ĐẠT ✔")"""

# =====================================================================================
# LAB 4 — kiểm suy dẫn bảng trắng
# =====================================================================================
LAB4_DATA = r"""rng4 = np.random.default_rng(1)          # RNG riêng của lab 4
def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()
def layernorm(v):                         # LayerNorm không affine, phương sai chia n (không chia n − 1)
    mu = v.mean(); s = np.sqrt(((v - mu) ** 2).mean())
    return (v - mu) / s

z = rng4.normal(size=5); y = np.eye(5)[1]
A = rng4.normal(size=(4, 4)); x = rng4.normal(size=4)
X = rng4.normal(size=(6, 3)); W = rng4.normal(size=(3, 2)); Y = rng4.normal(size=(6, 2))
v = rng4.normal(size=6); u = rng4.normal(size=6)                 # u: vector chiếu, kiểm Jacobian qua hàm vô hướng uᵀLN(v)
n = 6                                                            # σ của LN(v) là v.std() (chia n)

# Bốn khẳng định "viết trên bảng". Mỗi khẳng định: (hàm vô hướng f, điểm x0, gradient khẳng định tại x0)
CLAIMS = {
  "a) dCE/dz = p − y":                   (lambda z_: -np.log(softmax(z_)[1]), z, softmax(z) - y),
  "b) d(xᵀAx)/dx = 2Ax":                 (lambda x_: x_ @ A @ x_, x, 2 * A @ x),
  "c) d‖XW − Y‖²/dW = 2Xᵀ(XW − Y)":       (lambda W_: np.sum((X @ W_ - Y) ** 2), W, 2 * X.T @ (X @ W - Y)),
  "d) d(uᵀLN(v))/dv = (I − 11ᵀ/n)u/σ":   (lambda v_: u @ layernorm(v_), v, (np.eye(n) - np.ones((n, n)) / n) @ u / v.std()),
}"""

LAB4_FN = ("""def num_grad(f, x0, h=1e-6):
    \"\"\"Gradient số bằng sai phân trung tâm (f(x + h·eᵢ) − f(x − h·eᵢ))/(2h), cho x0 có shape bất kỳ (float64).\"\"\"
    # TODO
    raise NotImplementedError


def rel_err(a, b):
    \"\"\"‖a − b‖ / max(‖a‖ + ‖b‖, 1e-12).\"\"\"
    # TODO
    raise NotImplementedError


def check(f, x0, g_claim, tol=1e-6):
    \"\"\"Trả về (sai số tương đối giữa gradient số và gradient khẳng định, 'ĐÚNG' nếu < tol, ngược lại 'SAI').\"\"\"
    # TODO
    raise NotImplementedError""",
r"""def num_grad(f, x0, h=1e-6):
    x0 = np.asarray(x0, dtype=np.float64)
    g = np.zeros_like(x0)
    for i in np.ndindex(x0.shape):
        e = np.zeros_like(x0); e[i] = h
        g[i] = (f(x0 + e) - f(x0 - e)) / (2 * h)
    return g


def rel_err(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-12)


def check(f, x0, g_claim, tol=1e-6):
    e = rel_err(num_grad(f, x0), g_claim)
    return e, ("ĐÚNG" if e < tol else "SAI")""")

LAB4_FIX = ("""# Bước 2: với MỖI khẳng định mà ô trên báo SAI,
# suy dẫn lại trên giấy và viết gradient đúng tại điểm x0 của khẳng định đó.
# Khóa của FIXES: tên khẳng định, chép đúng như trong CLAIMS. Giá trị: mảng cùng shape với x0, tính bằng CÔNG THỨC
# từ các biến có sẵn (z, y, A, x, X, W, Y, v, u, n, softmax, layernorm). Không dùng num_grad ở đây:
# điền gradient số là chép đáp án, không phải sửa suy dẫn.
FIXES = {}
# TODO
raise NotImplementedError""",
r"""FIXES = {
    "b) d(xᵀAx)/dx = 2Ax": (A + A.T) @ x,          # A không đối xứng: ∇(xᵀAx) = (A + Aᵀ)x; 2Ax chỉ đúng khi A = Aᵀ
}
vh, s_v = layernorm(v), v.std()
J_LN = (np.eye(n) - np.ones((n, n)) / n - np.outer(vh, vh) / n) / s_v   # Jacobian LayerNorm đủ (đối xứng), bài 11
FIXES["d) d(uᵀLN(v))/dv = (I − 11ᵀ/n)u/σ"] = J_LN.T @ u                # khẳng định d thiếu số hạng − x̂x̂ᵀ/n
print("J·1 =", f"{np.linalg.norm(J_LN @ np.ones(n)):.1e}", "| J·x̂ =", f"{np.linalg.norm(J_LN @ vh):.1e}",
      "| hạng J =", np.linalg.matrix_rank(J_LN), "= n − 2")""")

LAB4_RUN = r"""# Bước 1: phán quyết của check() cho bốn khẳng định
res = {name: check(f, x0, g) for name, (f, x0, g) in CLAIMS.items()}
for name, (e, verdict) in res.items():
    print(f"{name:40s} rel err {e:.2e} -> {verdict}")"""

LAB4_CHECK = r"""res = {name: check(f, x0, g) for name, (f, x0, g) in CLAIMS.items()}
# (1) num_grad và rel_err đúng trên hàm đã biết đạo hàm: f(t) = Σ t·sin t, f'(t) = sin t + t·cos t
ts = np.linspace(-1.0, 2.0, 7)
assert rel_err(num_grad(lambda t: np.sum(t * np.sin(t)), ts), np.sin(ts) + ts * np.cos(ts)) < 1e-8, "num_grad chưa đủ chính xác"
assert all(np.isclose(rel_err(c * ts, c * (ts + 0.1)), rel_err(ts, ts + 0.1)) for c in (1e-6, 1e6)), "rel_err phải không đổi khi nhân cả hai vế cùng hệ số"
# (2) phán quyết và bản sửa của bạn
wrong = [k for k, (e, verdict) in res.items() if verdict == "SAI"]
assert len(wrong) == 2, "đề nói đúng hai khẳng định sai"
assert all(res[k][0] < 1e-8 for k in CLAIMS if k not in wrong), "khẳng định đúng phải khớp tới cỡ 1e-10"
assert set(FIXES) == set(wrong), "khóa của FIXES phải đúng là các khẳng định bị báo SAI"
for k in wrong:
    f, x0, g_claim = CLAIMS[k]
    e, verdict = check(f, x0, FIXES[k])
    print(f"bản sửa của {k[:2]} rel err {e:.2e} -> {verdict}")
    assert verdict == "ĐÚNG" and rel_err(FIXES[k], g_claim) > 1e-3
print("Lab 4 ĐẠT ✔")"""

# =====================================================================================
# LAB 5 — máy tính chi phí
# =====================================================================================
LAB5_DATA = r"""# Cấu hình hình dạng Llama-2-7B (như chương 8). n_kv = số đầu K/V; n_kv = n_h là multi-head attention (MHA).
CFG = dict(L=32, d=4096, n_h=32, n_kv=32, V=32000, ffn=11008)
CFG_KV8 = dict(CFG, n_kv=8)            # GIẢ ĐỊNH để so sánh: 8 đầu K/V (không phải Llama-2-7B)
BW = 2e12                              # băng thông GIẢ ĐỊNH 2 TB/s (byte/giây)
Q4 = 0.5 + 2 / 64                      # 4-bit cộng một scale fp16 cho mỗi khối 64 trọng số: byte/trọng số

def count_params(c):
    '''Số tham số (đã kiểm ở chương 8): attention, MLP SwiGLU 3 ma trận, 2 RMSNorm mỗi lớp,
    embedding vào và lm_head KHÔNG buộc trọng số, một RMSNorm cuối.'''
    dh = c["d"] // c["n_h"]
    att = 2 * c["d"] * c["d"] + 2 * c["d"] * c["n_kv"] * dh
    return c["L"] * (att + 3 * c["d"] * c["ffn"] + 2 * c["d"]) + 2 * c["V"] * c["d"] + c["d"]"""

LAB5_FN = ("""def weight_gib(c, nbytes=2):
    \"\"\"Bộ nhớ trọng số, đơn vị GiB (2³⁰ byte).\"\"\"
    # TODO
    raise NotImplementedError


def kv_bytes(c, T, B=1, nbytes=2):
    \"\"\"Số byte KV cache cho B chuỗi, mỗi chuỗi T token. Tự suy công thức từ hình dạng các tensor được lưu ở mỗi lớp.\"\"\"
    # TODO
    raise NotImplementedError


def train_flops(c, D):
    \"\"\"FLOPs huấn luyện ≈ 6·N·D theo QUY ƯỚC CHƯƠNG 8: N là tổng tham số trừ bảng embedding đầu vào (V·d),
    vì tra bảng không phải nhân ma trận (N vẫn gồm các RMSNorm, tuy nhỏ). Không tính số hạng attention theo T.\"\"\"
    # TODO
    raise NotImplementedError


def decode_ms(c, bw=BW, nbytes=2, T=0):
    \"\"\"Cận dưới ms/token khi giải mã batch 1, bị chặn bởi băng thông: đọc toàn bộ trọng số
    cộng KV cache của T token đã có, chia cho bw.\"\"\"
    # TODO
    raise NotImplementedError""",
r"""def weight_gib(c, nbytes=2):
    return count_params(c) * nbytes / GiB


def kv_bytes(c, T, B=1, nbytes=2):
    dh = c["d"] // c["n_h"]
    return 2 * c["L"] * c["n_kv"] * dh * nbytes * T * B          # 2 = K và V; dùng số đầu K/V, không phải số đầu query


def train_flops(c, D):
    n_mm = count_params(c) - c["V"] * c["d"]                      # bỏ embedding đầu vào (tra bảng, không nhân ma trận)
    return 6 * n_mm * D


def decode_ms(c, bw=BW, nbytes=2, T=0):
    return (count_params(c) * nbytes + kv_bytes(c, T, 1, nbytes)) / bw * 1e3""")

LAB5_CHECK = r"""N = count_params(CFG)
rows = [("tham số", N), ("trọng số fp16, GiB", round(weight_gib(CFG), 2)), ("trọng số 4-bit + scale, GiB", round(weight_gib(CFG, Q4), 2)),
        ("KV mỗi token, byte", kv_bytes(CFG, 1)), ("KV 4096 token MHA, GiB", round(kv_bytes(CFG, 4096) / GiB, 2)),
        ("KV 4096 token, 8 đầu K/V, GiB", round(kv_bytes(CFG_KV8, 4096) / GiB, 2)),
        ("FLOPs 6·N·D, D = 2·10¹²", f"{train_flops(CFG, 2e12):.3e}"),
        ("decode ms/token, 2 TB/s, chỉ trọng số", round(decode_ms(CFG), 2)), ("decode ms/token, cộng KV 4096", round(decode_ms(CFG, T=4096), 2))]
for k_, v_ in rows:
    print(f"{k_:40s} {v_}")
free = 80 - weight_gib(CFG)
print("GPU giả định 80 GiB, còn", round(free, 2), "GiB -> MHA", int(free // (kv_bytes(CFG, 4096) / GiB)),
      "chuỗi 4096 token; 8 đầu K/V", int(free // (kv_bytes(CFG_KV8, 4096) / GiB)), "(chưa trừ activation, phân mảnh)")
lap = weight_gib(CFG, Q4) + kv_bytes(CFG, 4096) / GiB
print("laptop 6 GiB: 4-bit + KV 4096 =", round(lap, 2), "GiB; KV 2048:", round(weight_gib(CFG, Q4) + kv_bytes(CFG, 2048) / GiB, 2), "GiB (chưa tính overhead)")
print("nếu KHÔNG bỏ embedding đầu vào: 6ND =", f"{6 * N * 2e12:.3e}", "-> phải nói mình dùng quy ước nào")
assert N == 6738415616
assert abs(weight_gib(CFG) - 12.551) < 1e-3 and abs(weight_gib(CFG, Q4) - 3.334) < 1e-3
assert kv_bytes(CFG, 1) == 524288 and kv_bytes(CFG_KV8, 4096) * 4 == kv_bytes(CFG, 4096)
assert kv_bytes(CFG, 4096, B=8) == 16 * GiB, "KV cache phải tỉ lệ với batch (ví dụ 14.6: batch 8 là 16 GiB)"
assert abs(train_flops(CFG, 2e12) - 7.929e22) / 7.929e22 < 1e-3
assert abs(decode_ms(CFG) - 6.738) < 1e-3 and abs(decode_ms(CFG, T=4096) - 7.812) < 1e-3
print("Lab 5 ĐẠT ✔")"""

# =====================================================================================
# SĂN BUG — báo cáo chi phí của "đồng nghiệp" (KHÔNG chú thích chỗ lỗi trong bản lab)
# =====================================================================================
BUG_CODE = ("""# Báo cáo chi phí phục vụ và huấn luyện do một bạn cùng nhóm viết. Chạy không báo lỗi, số trông "hợp lý".
# Sửa tại chỗ trong hàm cost_report, rồi chạy ô kiểm tra bên dưới.
def cost_report(c, T=4096, B=1, nbytes=2, D=2e12, bw=BW, gpu_gib=80):
    n = count_params(c)
    dh = c["d"] // c["n_h"]
    weights = n * nbytes / 1e9
    kv_tok = c["L"] * c["n_h"] * dh * nbytes
    kv_seq = kv_tok * T * B / GiB
    n_mm = n - c["V"] * c["d"]
    return dict(weights_gib=round(weights, 2),
                kv_gib=round(kv_seq, 2),
                max_seqs=int((gpu_gib - weights) // kv_seq),
                train_flops=2 * n_mm * D,
                decode_ms=round(n * nbytes / bw * 1e3, 2))""",
"""# Bản đã sửa (bốn lỗi, xem chú thích từng dòng).
def cost_report(c, T=4096, B=1, nbytes=2, D=2e12, bw=BW, gpu_gib=80):
    n = count_params(c)
    dh = c["d"] // c["n_h"]
    weights = n * nbytes / GiB                          # lỗi 3: chia 1e9 là GB, không phải GiB
    kv_tok = 2 * c["L"] * c["n_kv"] * dh * nbytes       # lỗi 1: thiếu hệ số 2 (K và V); lỗi 2: n_h thay vì n_kv
    kv_seq = kv_tok * T * B / GiB
    n_mm = n - c["V"] * c["d"]
    return dict(weights_gib=round(weights, 2),
                kv_gib=round(kv_seq, 2),
                max_seqs=int((gpu_gib - weights) // kv_seq),
                train_flops=6 * n_mm * D,                # lỗi 4: 2ND chỉ là forward; huấn luyện là 6ND
                decode_ms=round(n * nbytes / bw * 1e3, 2))""")

BUG_CHECK = r"""def failing_checks(report_fn):
    '''Trả về số phép kiểm chưa đạt (so với các số chuẩn đã có trong sách).'''
    r1, r8, rb = report_fn(CFG), report_fn(CFG_KV8), report_fn(CFG, B=8)
    expect = [(r1["weights_gib"], 12.55), (r1["kv_gib"], 2.00), (r8["kv_gib"], 0.50), (rb["kv_gib"], 16.00),
              (r1["max_seqs"], 33), (r1["train_flops"], 7.929e22), (r1["decode_ms"], 6.74)]
    return sum(abs(got - want) > 0.006 * abs(want) for got, want in expect)

r = cost_report(CFG)
print("cấu hình MHA     :", r)
print("cấu hình 8 đầu K/V:", cost_report(CFG_KV8))
bad = failing_checks(cost_report)
print("Số phép kiểm chưa đạt:", bad, "/ 7")
assert bad == 0, "Còn phép kiểm chưa đạt."
print("Săn bug ĐẠT ✔")"""

# =====================================================================================
# PHỤ LỤC (chỉ trong loigiai): kiểm đột biến + kiểm các số mới của sách
# =====================================================================================
APPENDIX = r"""# PHỤ LỤC (chỉ có trong bản lời giải).
# (1) Kiểm đột biến: gài lại TỪNG lỗi vào bản đã sửa và xác nhận ô kiểm của săn bug bắt được.
import inspect
src_fixed = inspect.getsource(cost_report)
MUTANTS = {
    "lỗi 1: thiếu hệ số 2 của K/V": ("kv_tok = 2 * c[\"L\"]", "kv_tok = c[\"L\"]"),
    "lỗi 2: n_h thay n_kv":          ("c[\"n_kv\"] * dh", "c[\"n_h\"] * dh"),
    "lỗi 3: GB thay GiB":            ("n * nbytes / GiB", "n * nbytes / 1e9"),
    "lỗi 4: 2ND thay 6ND":           ("6 * n_mm * D", "2 * n_mm * D"),
}
for name, (old, new) in MUTANTS.items():
    assert src_fixed.count(old) == 1, name
    ns = dict(globals())
    exec(src_fixed.replace(old, new), ns)
    n_bad = failing_checks(ns["cost_report"])
    print(f"  {name:32s} -> {n_bad} phép kiểm chưa đạt", "(bị bắt)" if n_bad else "(LỌT)")
    assert n_bad > 0, name
ns = dict(globals()); exec(src_fixed, ns); assert failing_checks(ns["cost_report"]) == 0
# lỗi 2 chỉ lộ ra ở cấu hình có n_kv < n_h: thử bỏ cấu hình 8 đầu K/V khỏi phép kiểm
ns = dict(globals()); exec(src_fixed.replace(*MUTANTS["lỗi 2: n_h thay n_kv"]), ns)
print("  lỗi 2 trên cấu hình MHA:", ns["cost_report"](CFG)["kv_gib"], "GiB (giống bản đúng) -> cần cấu hình 8 đầu K/V để lộ")

# Triệu chứng của bản lỗi (để đối chiếu với đoạn mô tả trong sách)
BUGGY = '''def cost_report(c, T=4096, B=1, nbytes=2, D=2e12, bw=BW, gpu_gib=80):
    n = count_params(c)
    dh = c["d"] // c["n_h"]
    weights = n * nbytes / 1e9
    kv_tok = c["L"] * c["n_h"] * dh * nbytes
    kv_seq = kv_tok * T * B / GiB
    n_mm = n - c["V"] * c["d"]
    return dict(weights_gib=round(weights, 2), kv_gib=round(kv_seq, 2), max_seqs=int((gpu_gib - weights) // kv_seq),
                train_flops=2 * n_mm * D, decode_ms=round(n * nbytes / bw * 1e3, 2))'''
ns = dict(globals()); exec(BUGGY, ns)
print("bản lỗi, MHA     :", ns["cost_report"](CFG))
print("bản lỗi, 8 đầu K/V:", ns["cost_report"](CFG_KV8), "| phép kiểm chưa đạt:", failing_checks(ns["cost_report"]))

# (2) Kiểm đột biến cho các ô kiểm lab: một lời giải sai điển hình phải bị bắt.
mut_ok = True
w_bad = (MAX_SCORE - log["diem"].to_numpy() + 1) * (1 + log["ngay"].to_numpy())      # quên chia 7
mut_ok &= int(np.argmax(w_bad)) == i_max and abs(w_bad.max() - 27.143) > 1e-3
mut_ok &= basic_stats(TRANSCRIPT, SECONDS / 60)["wpm"] != 139.2                      # giây <-> phút
mut_ok &= rel_err(num_grad(lambda x_: x_ @ A @ x_, x, h=1e-6), 2 * A @ x) > 1e-6       # 2Ax vẫn SAI với h nhỏ
print("đột biến lab bị bắt:", bool(mut_ok)); assert mut_ok

# (3) Các số MỚI của sách (kiểm tra nhanh, mini-test) — tính lại ở đây.
from math import log as ln, sqrt, erfc
p2 = np.array([4.0, 1.0]) / 5
print("KTN2: softmax(ln4, 0) =", p2, "p − y (nhãn lớp 2) =", p2 - np.array([0, 1]))
print("KTN3: KV/token Llama-2-7B =", kv_bytes(CFG, 1), "byte =", kv_bytes(CFG, 1) / 2**20, "MiB")
print("KTN4: SE(0.8, 1600) =", sqrt(0.8 * 0.2 / 1600))
cfg_mt = dict(L=32, d=4096, n_h=32, n_kv=8, V=128256, ffn=14336)
kv_t = kv_bytes(cfg_mt, 1)
print("MT1: KV/token L=32, 8 đầu K/V, dh=128 =", kv_t, "byte =", kv_t / 1024, "KiB; T=8192:", kv_bytes(cfg_mt, 8192) / GiB,
      "GiB; B=4:", kv_bytes(cfg_mt, 8192, 4) / GiB, "GiB; nếu 32 đầu K/V:", kv_bytes(dict(cfg_mt, n_kv=32), 8192, 4) / GiB, "GiB")
P = np.array([0.5, 0.5]); H = np.diag(P) - np.outer(P, P)
print("MT2: Hessian p=(0.5,0.5):", H.tolist(), "trị riêng", np.linalg.eigvalsh(H).round(4).tolist())
b_, c_ = 40, 22
chi2 = (b_ - c_) ** 2 / (b_ + c_); chi2c = (abs(b_ - c_) - 1) ** 2 / (b_ + c_)
pval = erfc(sqrt(chi2 / 2)); pvalc = erfc(sqrt(chi2c / 2))
print(f"MT3: McNemar b=40 c=22: chi2={chi2:.3f} p={pval:.4f}; hiệu chỉnh {chi2c:.3f} p={pvalc:.4f}; hiệu acc trên 2000 = {(b_ - c_) / 2000}")
a_, k_, cc = 0.75, 4, 0.1
E = (1 - a_ ** (k_ + 1)) / (1 - a_)
print(f"MT4: spec a=0.75 k=4: E={E:.4f}; tăng tốc c=0.1: {E / (k_ * cc + 1):.4f}")
assert abs(chi2 - 5.226) < 1e-3 and abs(E - 3.0508) < 1e-4
print("Phụ lục ĐẠT ✔")"""


def build(solution: bool):
    pick_ = (lambda pair: pair[1]) if solution else (lambda pair: pair[0])
    nb = nbf.v4.new_notebook()
    head = "Lời giải" if solution else "Bài lab"
    cells = [
        md(f"# Chương 14 — {head}: Technical interview\n\n"
           + ("Bản lời giải đã chạy kiểm. Chỉ mở sau khi đã tự làm `lab.ipynb`."
              if solution else
              "Viết code vào các ô có `# TODO`, rồi chạy ô kiểm tra ngay bên dưới. Ô kiểm tra in **ĐẠT ✔** là xong. "
              "Bí quá 20 phút mới mở `loigiai.ipynb`.")
           + "\n\nNăm lab là **công cụ luyện tập**: chọn chuỗi theo sổ lỗi, lịch ôn, phân tích bản chép lời câu trả lời, "
             "kiểm suy dẫn bảng trắng, máy tính chi phí LLM. Sổ lỗi và bản chép lời là **mẫu tự soạn**. "
             "Mọi băng thông, dung lượng GPU là **giả định của đề**. Bước cuối mỗi lab (không bắt buộc): thay dữ liệu mẫu bằng dữ liệu của bạn."),
        code(SETUP),
        md("## Lab 1 — Chọn chuỗi theo sổ lỗi\nĐọc sổ lỗi (`chuoi,diem,ngay`), tính trọng số luyện lại theo quy tắc của sách, "
           "rút 4 chuỗi không lặp cho tuần này. Câu hỏi: vì sao trọng số nhân với số ngày chứ không chỉ theo điểm?"),
        code(LOG_DATA), code(pick_(LAB1_FN)), code(LAB1_CHECK),
        md("**Bước cuối (không bắt buộc):** `pick(read_log('so_loi.csv'))` với sổ lỗi thật của bạn."),
        md("## Lab 2 — Lịch ôn Leitner\nMô phỏng 28 ngày, 18 chuỗi, mỗi ngày thêm một chuỗi mới. "
           "Câu hỏi: ngày nặng nhất có bao nhiêu chuỗi, và có vừa một buổi tối không?"),
        code(LAB2_DATA), code(pick_(LAB2_FN)), code(LAB2_CHECK),
        md("## Lab 3 — Phân tích bản chép lời câu trả lời (tuần 38)\nTừ, từ/phút và từ đệm đã có sẵn trong `basic_stats` "
           "(cùng việc với Lab 4 chương 13). Bạn chỉ viết phần mới: đếm I/we và cụm chữ số. "
           "Câu hỏi: công cụ bỏ sót những gì?"),
        code(LAB3_DATA), code(pick_(LAB3_FN)), code(LAB3_CHECK),
        md("## Lab 4 — Kiểm suy dẫn bảng trắng\nBước 1: gradient số bằng sai phân trung tâm (float64), so với gradient khẳng định. "
           "Bước 2: hai trong bốn khẳng định sai; tìm ra khẳng định nào, suy dẫn lại trên giấy và điền bản sửa vào `FIXES`."),
        code(LAB4_DATA), code(pick_(LAB4_FN)), code(LAB4_RUN), code(pick_(LAB4_FIX)), code(LAB4_CHECK),
        md("## Lab 5 — Máy tính chi phí LLM\nTrọng số, KV cache, FLOPs huấn luyện theo quy ước chương 8, cận dưới ms/token. "
           "Mọi băng thông và dung lượng GPU là giả định."),
        code(LAB5_DATA), code(pick_(LAB5_FN)), code(LAB5_CHECK),
        md("## Săn bug — báo cáo chi phí \"trông hợp lý\"\nHàm `cost_report` dưới đây chạy không báo lỗi. Triệu chứng: với cấu hình "
           "8 đầu K/V, KV cache 4096 token **bằng đúng** cấu hình MHA; số GiB của trọng số lớn hơn số chương 8 khoảng 7%; "
           "FLOPs huấn luyện nhỏ hơn phép tính tay của bạn. Có **bốn** lỗi. Sửa tại chỗ rồi chạy ô kiểm tra."),
        code(pick_(BUG_CODE)), code(BUG_CHECK),
    ]
    if solution:
        cells += [md("## Phụ lục — kiểm đột biến và kiểm số của sách\nChỉ có trong bản lời giải."), code(APPENDIX)]
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    name = "loigiai.ipynb" if solution else "lab.ipynb"
    nbf.write(nb, OUT / name)
    return OUT / name


if __name__ == "__main__":
    print(build(False), build(True))
