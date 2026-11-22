"""Lời giải tham khảo — bài dựng cảnh chương 13: thanh thời gian 3 phút cho một dự án.

Chỉ mở sau khi đã tự làm. Render:  manim -qm chuong13_thanh_thoi_gian.py ThanhThoiGianDuAn
Dự án mẫu (số giả): truy hồi lai văn bản và đồ thị trích dẫn. Bạn thay BLOCKS bằng dự án của chính mình.
"""
import sys
from manim import *

FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"

# (tên đoạn, giây, một câu có một con số hoặc một việc làm cụ thể)
BLOCKS = [
    ("Hook", 20, "kết luận một câu: recall@5 từ 0.61 lên 0.72"),
    ("Vấn đề", 25, "vector search thường bỏ sót bài trích dẫn"),
    ("Cách làm", 45, "hai tầng: 50 ứng viên, mở rộng một bước theo cạnh trích dẫn"),
    ("Bằng chứng", 45, "200 câu tự gán nhãn, 3 seed, mức tăng 0.09 đến 0.13"),
    ("Vai trò", 25, "tôi viết truy hồi và đánh giá; bạn tôi làm dữ liệu"),
    ("Kết", 20, "muốn học cách thiết kế thí nghiệm có người phản biện"),
]
assert sum(b[1] for b in BLOCKS) == 180, "tổng phải đúng 180 giây"


def T(s, size=28, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


class ThanhThoiGianDuAn(Scene):
    def construct(self):
        title = T("Thanh thời gian 3 phút cho dự án của tôi", 32).to_edge(UP, buff=0.3)
        self.play(Write(title))
        colors = [BLUE, TEAL, GREEN, GREEN, ORANGE, PURPLE]
        total_w = 12.0
        scale = total_w / 180
        x = -total_w / 2
        bar = VGroup()
        for (name, s, _), c in zip(BLOCKS, colors):
            w = s * scale
            r = Rectangle(width=w, height=0.9, color=c, fill_opacity=0.4, stroke_width=3).move_to([x + w / 2, 1.9, 0])
            lab = T(f"{s} s", 22).move_to(r)
            bar.add(VGroup(r, lab))
            x += w
        self.play(FadeIn(bar, lag_ratio=0.2), run_time=1.8)
        rows = VGroup()
        for (name, s, sentence), c in zip(BLOCKS, colors):
            rows.add(VGroup(Square(side_length=0.22, color=c, fill_opacity=0.9), T(f"{name}: {sentence}", 22)).arrange(RIGHT, buff=0.2))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([0, -0.5, 0])
        for r in rows:
            self.play(FadeIn(r, shift=RIGHT * 0.2), run_time=0.5)
        total = T("Tổng đúng 180 giây; mỗi đoạn có một con số hoặc một việc làm cụ thể", 24, YELLOW).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(total))
        self.wait(2.5)
