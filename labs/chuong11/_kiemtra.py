"""Bộ kiểm cho bài săn bug chương 11: kiểm hành vi của run_experiment_fixed, không kiểm cách viết."""
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
