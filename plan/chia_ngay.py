"""Chia việc của mỗi tuần trong plan/weeks.yaml ra từng ngày, ghi vào trường `days` của từng việc.

Chạy: python plan/chia_ngay.py          (ghi đè trường days của mọi việc)
      python plan/chia_ngay.py --check  (chỉ in thống kê, không ghi)

Quy ước: days là {thứ: giờ}, 1 = Thứ Hai ... 7 = Chủ nhật. Tổng giờ các ngày bằng `hours` của việc.
Cách chia:
- Thứ Hai đến Thứ Bảy chia đều giờ của tuần; Chủ nhật để nghỉ hoặc làm bù, trừ việc có ngày cố định.
- Việc của chương đi đúng thứ tự trong file (đọc → bài tập → lab → mini-test); việc luồng A, B, C và
  việc chung xen vào giữa, rải đều cả tuần.
- Việc dài (trên 1.5 giờ, không phải bài bấm giờ hay buổi thi) được cắt thành nhiều buổi, mỗi buổi ít nhất 0.5 giờ.
- Việc có ngày cố định (đề thi thử, mốc hồ sơ) nằm trong PINS bên dưới.
Sửa tay trường days trong weeks.yaml cũng được; chạy lại script sẽ ghi đè các sửa đó.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

PLAN = Path(__file__).resolve().parent / "weeks.yaml"
WORKDAYS = [1, 2, 3, 4, 5, 6]
STEP = 0.25   # độ mịn khi cắt giờ
MIN_PIECE = 0.5

# Việc có ngày cố định: id -> {thứ: giờ} hoặc thứ (cả việc nằm trong một ngày).
PINS = {
    "w00-setup": 1,  # cài app trước mọi việc khác
    "w13-mock": 7,   # Đề thi thử 1, Chủ nhật 14/3/2027
    "w24-mock": 7,   # Đề thi thử 2, Chủ nhật 30/5/2027
    "w31-lor": 1,    # hỏi người viết thư trước Thứ Năm 15/7
    "w45-apply": 5,  # nhắc thư lần 1, Thứ Sáu 22/10
    "w46-apply": 1,  # nộp sớm trong tuần, trước 31/10
    "w47-plan": 1,   # hẹn người phỏng vấn sớm
    "w47-apply": 5,  # nhắc thư lần cuối, Thứ Sáu 5/11
    "w48-mock": 4,   # mục tham chiếu 0 giờ của Đề thi thử 3
    "w48-run1": 4,   # buổi 1, Thứ Năm 11/11
    "w48-run2": 6,   # buổi 2, Thứ Bảy 13/11
    "w48-c": 7,      # Chủ nhật 14/11, sau Vòng 3
    "w30-on": {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5},  # "30 phút mỗi ngày"
}
# Hậu tố id của việc không cắt được: bài bấm giờ, buổi thi, việc ngắn có hạn.
WHOLE = {"mt", "mock", "run1", "run2", "kt", "tn", "retest", "full", "talk", "lor", "apply", "plan",
         "video", "manim", "code", "review", "setup"}


def splittable(t):
    suffix = t["id"].split("-", 1)[1]
    return t["hours"] > 1.5 and suffix not in WHOLE and t["stream"] != "C"


def rnd(x):
    return round(round(x / STEP) * STEP, 2)


def order(tasks):
    """Việc chương giữ thứ tự; việc khác xen kẽ giữa các luồng; trộn hai làn theo tỉ lệ giờ đã đi."""
    ch = [t for t in tasks if t["stream"] == "chuong"]
    by = defaultdict(list)
    for t in tasks:
        if t["stream"] != "chuong":
            by[t["stream"]].append(t)
    other, lanes = [], [by[k] for k in ("A", "B", "C", "chung") if by[k]]
    while any(lanes):
        for lane in lanes:
            if lane:
                other.append(lane.pop(0))
    out, i, j = [], 0, 0
    H_ch = sum(t["hours"] for t in ch) or 1
    H_ot = sum(t["hours"] for t in other) or 1
    done_ch = done_ot = 0.0
    while i < len(ch) or j < len(other):
        take_ch = j >= len(other) or (i < len(ch) and done_ch / H_ch <= done_ot / H_ot)
        if take_ch:
            out.append(ch[i]); done_ch += ch[i]["hours"]; i += 1
        else:
            out.append(other[j]); done_ot += other[j]["hours"]; j += 1
    return out


def plan_week(w):
    tasks = w["tasks"]
    alloc = {}
    load = defaultdict(float)
    for t in tasks:
        p = PINS.get(t["id"])
        if p is not None:
            alloc[t["id"]] = {p: t["hours"]} if isinstance(p, int) else dict(p)
            for d, h in alloc[t["id"]].items():
                load[d] += h
    free = order([t for t in tasks if t["id"] not in alloc])
    rem = sum(t["hours"] for t in free)  # giờ của việc tự do chưa xếp
    di = 0
    for t in free:
        left = t["hours"]
        parts = {}
        while left > 1e-9:
            d = WORKDAYS[min(di, len(WORKDAYS) - 1)]
            later = [x for x in WORKDAYS if x >= d]
            cap = (rem + sum(load[x] for x in later)) / len(later)  # chia đều phần còn lại cho các ngày còn lại
            room = cap - load[d]
            last = di >= len(WORKDAYS) - 1
            if last or left <= room + 0.5:
                put = left
            elif splittable(t) and room >= MIN_PIECE:
                put = min(rnd(room), left - MIN_PIECE)
            elif not splittable(t) and room >= cap * 0.5:
                put = left  # việc nguyên khối: nhận hơi quá tải còn hơn để ngày trống một nửa
            else:
                di += 1
                continue
            parts[d] = parts.get(d, 0) + put
            load[d] += put
            rem -= put
            left = round(left - put, 6)
            if load[d] >= cap - 0.25:
                di += 1
        alloc[t["id"]] = parts
    return alloc, load


def fmt_days(days):
    return "{" + ", ".join(f"{d}: {h:g}" for d, h in sorted(days.items())) + "}"


def main():
    text = PLAN.read_text(encoding="utf-8")
    weeks = yaml.safe_load(text)["weeks"]
    plans, worst = {}, []
    for w in weeks:
        alloc, load = plan_week(w)
        for t in w["tasks"]:
            assert abs(sum(alloc[t["id"]].values()) - t["hours"]) < 1e-6, t["id"]
        plans.update(alloc)
        work = [load[d] for d in WORKDAYS]
        worst.append((w["id"], min(work), max(work), load[7]))
    lo = min(x[1] for x in worst)
    hi = max(x[2] for x in worst)
    print(f"{len(weeks)} tuần, {len(plans)} việc. Giờ mỗi ngày Thứ Hai–Thứ Bảy: thấp nhất {lo:g}, cao nhất {hi:g}.")
    for wid, a, b, sun in worst:
        if b - a > 1.5 or b > 3.5:
            print(f"  {wid}: ngày nhẹ nhất {a:g} giờ, nặng nhất {b:g} giờ, Chủ nhật {sun:g} giờ")
    if "--check" in sys.argv:
        return

    def repl(m):
        line = re.sub(r" days: \{[^}]*\},", "", m.group(0))
        tid = m.group("id")
        return re.sub(r"(hours: [0-9.]+,)", lambda h: f"{h.group(1)} days: {fmt_days(plans[tid])},", line, count=1)

    new, n = re.subn(r"^\s+- \{id: (?P<id>[\w-]+),.*\}$", repl, text, flags=re.M)
    assert n == len(plans), (n, len(plans))
    PLAN.write_text(new, encoding="utf-8")
    # kiểm lại sau khi ghi
    again = yaml.safe_load(PLAN.read_text(encoding="utf-8"))["weeks"]
    for w in again:
        for t in w["tasks"]:
            assert abs(sum(t["days"].values()) - t["hours"]) < 1e-6, t["id"]
    print("Đã ghi trường days vào", PLAN)
    write_markdown(again)


NAMES = {1: "Thứ Hai", 2: "Thứ Ba", 3: "Thứ Tư", 4: "Thứ Năm", 5: "Thứ Sáu", 6: "Thứ Bảy", 7: "Chủ nhật"}
STREAM_NAMES = {"chuong": "Chương", "A": "A", "B": "B", "C": "C", "chung": "Chung"}


def write_markdown(weeks):
    """Bản đọc được của lịch theo ngày (plan/lich-theo-ngay.md), sinh lại mỗi lần chạy script."""
    import datetime as dt
    out = ["# Lịch theo ngày", "",
           "File này do `python plan/chia_ngay.py` sinh ra từ `plan/weeks.yaml`; đừng sửa tay. "
           "Thứ Hai đến Thứ Bảy chia đều giờ của tuần, Chủ nhật để nghỉ hoặc làm bù trừ khi có đề thi. "
           "Việc dài ghi \"buổi k/n\". Trong app, trang Tuần này → Xem theo Ngày cho tích từng buổi.", ""]
    for w in weeks:
        start = w["start"] if isinstance(w["start"], dt.date) else dt.date.fromisoformat(str(w["start"]))
        out += [f"## {w['title']}", ""]
        for d, name in NAMES.items():
            items = [t for t in w["tasks"] if d in t["days"]]
            date = start + dt.timedelta(days=d - 1)
            if not items:
                if d == 7:
                    out.append(f"**{name} {date:%d/%m}** · nghỉ hoặc làm bù")
                    out.append("")
                continue
            out.append(f"**{name} {date:%d/%m}** · {sum(t['days'][d] for t in items):g} giờ")
            out.append("")
            for t in items:
                n = len(t["days"])
                part = f", buổi {sorted(t['days']).index(d) + 1}/{n}" if n > 1 else ""
                out.append(f"- [{STREAM_NAMES[t['stream']]}] {t['text']} ({t['days'][d]:g} giờ{part})")
            out.append("")
    path = PLAN.parent / "lich-theo-ngay.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print("Đã ghi", path)


if __name__ == "__main__":
    main()
