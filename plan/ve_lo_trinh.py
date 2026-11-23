"""Vẽ lại plan/lo-trinh.png từ plan/weeks.yaml và book/chapters.yaml.

Chạy: python plan/ve_lo_trinh.py   (cần matplotlib, pyyaml)
Mọi mốc lấy từ ngày bắt đầu tuần trong weeks.yaml, nên khi dời lịch chỉ cần chạy lại.
"""
import datetime as dt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
weeks = {w["id"]: w for w in yaml.safe_load((ROOT / "plan/weeks.yaml").read_text(encoding="utf-8"))["weeks"]}
mocks = yaml.safe_load((ROOT / "book/chapters.yaml").read_text(encoding="utf-8"))["mock_tests"]


def d(x):
    return x if isinstance(x, dt.date) else dt.date.fromisoformat(str(x))


def start(wid):
    return d(weeks[wid]["start"])


def end(wid):
    return start(wid) + dt.timedelta(days=7)


# Hạn nộp hồ sơ là GIẢ ĐỊNH (mục 13.4), không dời theo lịch học.
DEADLINE = dt.date(2027, 11, 12)

rows = ["Chương học", "Đề thi thử", "Hồ sơ", "A · Nghiên cứu", "B · Coding", "C · Tiếng Anh"]
y = {r: len(rows) - i for i, r in enumerate(rows)}
bars = {
    "Chương học": [("w00", "w13", "I Nền tảng"), ("w14", "w24", "II Deep learning"),
                   ("w25", "w31", "III NLP"), ("w32", "w48", "IV Hồ sơ, phỏng vấn")],
    "B · Coding": [("w00", "w13", "Mảng, hash, stack"), ("w14", "w24", "Cây, đồ thị, DP"),
                   ("w25", "w48", "Bài bấm giờ như thật")],
    "C · Tiếng Anh": [("w00", "w17", "Trả lời mẫu theo chương"), ("w18", "w48", "Nói 3 phút về 1 paper/tuần")],
}
points = {
    "A · Nghiên cứu": [(start("w09"), "Câu hỏi + baseline"), (start("w18"), "Ablation"),
                       (start("w30"), "Báo cáo 4–6 trang"), (start("w44"), "Báo cáo cuối")],
    "Hồ sơ": [(dt.date(2027, 7, 15), "Hỏi người viết thư"), (start("w39"), "Gửi gói cho người viết"),
              (dt.date(2027, 10, 31), "Nộp (mục tiêu)")],
}

fig, ax = plt.subplots(figsize=(15, 6.2), dpi=110)
for row, segs in bars.items():
    for a, b, label in segs:
        x0, x1 = mdates.date2num(start(a)), mdates.date2num(end(b))
        ax.barh(y[row], x1 - x0, left=x0, height=0.5, color="#e8eef7", edgecolor="#5a6b85")
        ax.text((x0 + x1) / 2, y[row], label, ha="center", va="center", fontsize=10)
for row, pts in points.items():
    for i, (day, label) in enumerate(pts):
        ax.plot(mdates.date2num(day), y[row], "D", color="#5a6b85", ms=10)
        right = i == len(pts) - 1 and row == "Hồ sơ"  # nhãn cuối nằm sát hạn nộp: đặt sang trái
        ax.text(mdates.date2num(day) + (-4 if right else 4), y[row] + (0.3 if i % 2 == 0 else -0.33), label,
                fontsize=9.5, va="center", ha="right" if right else "left")
for i, m in enumerate(mocks):
    last = i == len(mocks) - 1
    ax.plot(mdates.date2num(d(m["date"])), y["Đề thi thử"], "D", color="#2f6fdb" if last else "#5a6b85", ms=11)
    ax.text(mdates.date2num(d(m["date"])) + (-4 if last else 4), y["Đề thi thử"] + (0.3 if last else -0.3),
            f"{m['name']} · {d(m['date']):%d/%m}", fontsize=9.5, ha="right" if last else "left",
            fontweight="bold" if last else "normal", va="center")
ax.axvline(mdates.date2num(DEADLINE), color="#c0392b", ls="--", lw=1.4)
ax.text(mdates.date2num(DEADLINE) - 3, y["Chương học"] + 0.45, "Hạn nộp giả định 12/11/2027",
        color="#c0392b", fontsize=9.5, ha="right", va="center")

ax.set_yticks(list(y.values()), list(y.keys()), fontsize=11, fontweight="bold")
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
ax.set_xlim(mdates.date2num(start("w00")) - 8, mdates.date2num(end("w48")) + 5)
ax.set_ylim(0.3, len(rows) + 0.8)
ax.grid(axis="x", color="#dddddd")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.set_title(f"Lộ trình {start('w00'):%d/%m/%Y} – {end('w48') - dt.timedelta(days=1):%d/%m/%Y}: "
             "ba luồng chạy song song, nộp hồ sơ trước hạn rồi làm đề thi thử 3",
             loc="left", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(ROOT / "plan/lo-trinh.png")
print("Đã ghi", ROOT / "plan/lo-trinh.png")
