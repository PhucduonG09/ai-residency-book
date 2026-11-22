"""Lời giải tham khảo — bài dựng cảnh chương 1: power iteration.

Chỉ mở sau khi đã tự làm. Render:  manim -qm chuong01_power_iteration.py PowerIteration
"""
import sys
import numpy as np
from manim import *

FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"


def T(s, size=28, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


class PowerIteration(Scene):
    def construct(self):
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        q1 = np.array([1.0, 1.0]) / np.sqrt(2)              # vector riêng của λ = 3
        title = T("Power iteration: vₖ₊₁ = Avₖ / ‖Avₖ‖₂", 32).to_edge(UP, buff=0.3)
        plane = NumberPlane(x_range=[-2, 2, 1], y_range=[-2, 2, 1], x_length=6, y_length=6,
                            background_line_style={"stroke_opacity": 0.3}).shift(DOWN * 0.3)
        c2p = plane.c2p
        circle = Circle(radius=plane.x_length / 4, color=GREY_B).move_to(c2p(0, 0))
        eig = DashedLine(c2p(*(-1.9 * q1)), c2p(*(1.9 * q1)), color=YELLOW)
        eig_lab = T("vector riêng λ = 3", 22, YELLOW).next_to(c2p(*(1.9 * q1)), RIGHT, buff=0.1)
        self.play(Write(title), Create(plane), Create(circle), Create(eig), FadeIn(eig_lab))

        v = np.array([np.cos(np.deg2rad(80)), np.sin(np.deg2rad(80))])
        arrow = Arrow(c2p(0, 0), c2p(*v), buff=0, color=GREEN, stroke_width=6)
        info = T("", 24)

        def angle_text(v, k):
            ang = np.degrees(np.arccos(np.clip(abs(v @ q1), -1, 1)))
            return T(f"bước {k}: góc tới vector riêng = {ang:5.2f}°,  tan = {np.tan(np.radians(ang)):.4f}",
                     24).to_edge(DOWN, buff=0.4)

        info = angle_text(v, 0)
        self.play(GrowArrow(arrow), FadeIn(info))
        for k in range(1, 7):
            v = A @ v
            v = v / np.linalg.norm(v)
            new_arrow = Arrow(c2p(0, 0), c2p(*v), buff=0, color=GREEN, stroke_width=6)
            new_info = angle_text(v, k)
            self.play(Transform(arrow, new_arrow), Transform(info, new_info), run_time=1.1)
            self.wait(0.4)
        note = T("tan giảm đúng 3 lần mỗi bước, vì λ₂/λ₁ = 1/3", 26, YELLOW).next_to(info, UP, buff=0.25)
        self.play(FadeIn(note))
        self.wait(2.5)
