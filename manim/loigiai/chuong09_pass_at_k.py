"""Lời giải tham khảo — bài dựng cảnh chương 9: pass@k không chệch và công thức ngây thơ.

Chỉ mở sau khi đã tự làm. Render:  manim -qm chuong09_pass_at_k.py PassAtK
"""
import sys
from math import comb
from manim import *

FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"


def T(s, size=28, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


class PassAtK(Scene):
    def construct(self):
        n, c = 20, 3
        ks = [1, 2, 5, 10, 20]
        unb = [1 - comb(n - c, k) / comb(n, k) if n - c >= k else 1.0 for k in ks]
        naive = [1 - (1 - c / n) ** k for k in ks]
        title = T("pass@k: n = 20 mẫu, c = 3 đúng", 34).to_edge(UP, buff=0.3)
        self.play(Write(title))
        ax = Axes(x_range=[0, 5, 1], y_range=[0, 1.0, 0.25], x_length=8, y_length=3.8, tips=False).shift(LEFT * 1.2 + DOWN * 0.1)
        xl = VGroup(*[T(f"k={k}", 18).next_to(ax.c2p(i + 0.5, 0), DOWN, buff=0.15) for i, k in enumerate(ks)])
        self.play(Create(ax), FadeIn(xl))
        pu = [ax.c2p(i + 0.5, v) for i, v in enumerate(unb)]
        pn = [ax.c2p(i + 0.5, v) for i, v in enumerate(naive)]
        lu = VMobject(color=GREEN).set_points_as_corners(pu)
        ln = VMobject(color=RED).set_points_as_corners(pn)
        self.play(Create(lu), Create(ln), run_time=2)
        tu = VGroup(*[T(f"{v:.4f}", 16, GREEN).next_to(p, UP, buff=0.12) for v, p in zip(unb, pu)])
        tn = VGroup(*[T(f"{v:.4f}", 16, RED).next_to(p, DOWN, buff=0.12) for v, p in zip(naive, pn)])
        tn[0].shift(RIGHT * 0.5)
        self.play(FadeIn(tu), FadeIn(tn))
        leg = VGroup(T("không chệch", 20, GREEN), T("ngây thơ", 20, RED)).arrange(DOWN, aligned_edge=LEFT).move_to(RIGHT * 5.3 + UP * 1.0)
        self.play(FadeIn(leg))
        t = T("Công thức ngây thơ luôn chệch xuống", 22)
        cap = VGroup(BackgroundRectangle(t, fill_opacity=0.85, buff=0.12), t).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(cap))
        self.wait(3)
