"""Lời giải tham khảo — bài dựng cảnh chương 14: cây chuỗi 4 tầng cho dự án của chính bạn.

Chỉ mở sau khi đã tự làm. Render:  manim -qm chuong14_cay_chuoi.py CayChuoiCuaToi
Dự án mẫu (số giả, khớp chương 13): hỏi đáp trên tóm tắt arXiv, truy hồi lai văn bản và đồ thị trích dẫn.
Bạn thay NODES bằng dự án của mình: đúng 4 tầng, mỗi tầng một chương và một con số có đơn vị hoặc một phép kiểm cụ thể.
"""
import sys
from manim import *

FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"

# (tầng, câu hỏi ngắn, chương, con số hoặc phép kiểm)
NODES = [
    ("Tầng 1", "khẳng định trung tâm", "ch11, ch13", "recall@5: 0.61 → 0.72 trên 200 câu"),
    ("Tầng 2", "có thật không?", "ch3, ch11", "SE mỗi tỉ lệ ≈ 0.03; cần phân tích ghép cặp"),
    ("Tầng 3", "ablation trong 20 giờ GPU", "ch11", "2³ cấu hình × 3 seed = 24 lần = 12 giờ"),
    ("Tầng 4", "mối đe dọa lớn nhất", "ch12", "một người gán nhãn chính; kappa 0.78 trên 50 câu"),
]
assert len(NODES) == 4


def T(s, size=28, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


class CayChuoiCuaToi(Scene):
    def construct(self):
        title = T("Cây chuỗi 4 tầng cho dự án của tôi", 32).to_edge(UP, buff=0.3)
        self.play(Write(title))
        colors = [BLUE, TEAL, GREEN, ORANGE]
        ys = [1.8, 0.5, -0.8, -2.1]
        prev = None
        for (tier, q, ch, num), y, c in zip(NODES, ys, colors):
            box = RoundedRectangle(corner_radius=0.15, width=4.4, height=0.95, color=c, fill_opacity=0.2, stroke_width=3).move_to([-3.2, y, 0])
            txt = VGroup(T(tier, 20, c), T(q, 22)).arrange(DOWN, buff=0.06).move_to(box)
            chl = T(ch, 18, c).next_to(box, UP, buff=0.05).align_to(box, RIGHT)
            nb = T(num, 21, YELLOW).next_to(box, RIGHT, buff=0.35)
            if nb.get_right()[0] > 6.9:
                nb.scale_to_fit_width(6.9 - nb.get_left()[0])
            if prev is not None:
                self.play(GrowArrow(Arrow(prev.get_bottom(), box.get_top(), buff=0.05, color=GREY_B, stroke_width=3,
                                          max_tip_length_to_length_ratio=0.35)), run_time=0.4)
            self.play(FadeIn(VGroup(box, txt, chl, nb), shift=DOWN * 0.1), run_time=0.8)
            prev = box
        note = T("số giả của chương 13; thay bằng dự án thật của bạn", 22, GREY_B).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(note))
        self.wait(2.5)
