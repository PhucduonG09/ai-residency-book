"""Lời giải tham khảo — bài dựng cảnh chương 5: phương sai của trung bình B mô hình tương quan.

Chỉ mở sau khi đã tự làm. Render:  manim -qm chuong05_bagging_phuong_sai.py BaggingPhuongSai
"""
import sys
import numpy as np
from manim import *

FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"


def T(s, size=28, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


class BaggingPhuongSai(Scene):
    def construct(self):
        title = T("Phương sai trung bình B cây: ρσ² + (1 − ρ)σ²/B", 30).to_edge(UP, buff=0.3)
        ax = Axes(x_range=[0, 100, 20], y_range=[0, 1, 0.25], x_length=9, y_length=4.6, tips=False).shift(DOWN * 0.3 + LEFT * 1.2)
        labels = VGroup(*[T(str(v), 18, GREY_B).next_to(ax.c2p(v, 0), DOWN, buff=0.12) for v in range(0, 101, 20)],
                        *[T(f"{v:g}", 18, GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.12) for v in (0.25, 0.5, 0.75, 1)])
        xl = T("B", 22).next_to(ax.x_axis, RIGHT, buff=0.2)
        yl = T("Var / σ²", 20).next_to(ax.y_axis, UP, buff=0.1)
        self.play(Write(title), Create(ax), FadeIn(labels), FadeIn(xl), FadeIn(yl))
        legend = VGroup()
        for rho, col in [(0.0, GREEN), (0.3, YELLOW), (0.7, RED)]:
            curve = ax.plot(lambda B, r=rho: r + (1 - r) / B, x_range=[1, 100, 0.25], color=col, stroke_width=5)
            floor = DashedLine(ax.c2p(0, rho), ax.c2p(100, rho), color=col, stroke_opacity=0.6)
            v10, v100 = rho + (1 - rho) / 10, rho + (1 - rho) / 100
            item = T(f"ρ = {rho}: B = 10 → {v10:.3f}, B = 100 → {v100:.3f}", 20, col)
            legend.add(item)
            legend.arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_corner(UR, buff=0.3).shift(DOWN * 0.9)
            self.play(Create(curve), Create(floor), FadeIn(item), run_time=1.6)
            self.wait(0.4)
        cap = T("Thêm cây chỉ kéo về sàn ρσ²; muốn thấp hơn phải giảm ρ (random forest)", 24)
        cap = VGroup(BackgroundRectangle(cap, fill_opacity=0.85, buff=0.12), cap).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(cap))
        self.wait(3)
