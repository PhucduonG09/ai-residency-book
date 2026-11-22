"""Lời giải tham khảo — bài dựng cảnh chương 3: cập nhật Beta–Bernoulli.

Chỉ mở sau khi đã tự làm. Render:  manim -qm chuong03_beta_cap_nhat.py BetaCapNhat
"""
import sys
import numpy as np
from scipy import stats
from manim import *

FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"


def T(s, size=28, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


class BetaCapNhat(Scene):
    def construct(self):
        title = T("Prior Beta(2, 2) cập nhật sau từng lần tung", 32).to_edge(UP, buff=0.3)
        ax = Axes(x_range=[0, 1, 0.1], y_range=[0, 3.5, 1], x_length=9, y_length=4.2,
                  tips=False, axis_config={"stroke_opacity": 0.6}).shift(DOWN * 0.4)
        ticks = VGroup(*[T(f"{x:g}", 20).next_to(ax.c2p(x, 0), DOWN, buff=0.12) for x in (0, 0.5, 1)])
        pl = T("p", 24).next_to(ax.c2p(1, 0), RIGHT, buff=0.2)
        self.play(Write(title), Create(ax), FadeIn(ticks), FadeIn(pl))

        def curve(a, b):
            return ax.plot(lambda x: stats.beta(a, b).pdf(x), x_range=[0.001, 0.999, 0.005],
                           color=YELLOW, stroke_width=5)

        def info(h, n, a, b):
            mle = "—" if n == 0 else f"{h / n:.3f}"
            return T(f"{h}/{n} ngửa   Beta({a}, {b})   MLE = {mle}   MAP = {(a - 1) / (a + b - 2):.3f}",
                     24).next_to(title, DOWN, buff=0.25)

        a, b, h, n = 2, 2, 0, 0
        c, txt = curve(a, b), info(h, n, a, b)
        self.play(Create(c), FadeIn(txt))
        # ba lần ngửa liên tiếp, rồi thêm 7 lần để thành 7/10 ngửa (4 ngửa, 3 sấp)
        flips = [1, 1, 1] + [1, 0, 1, 0, 1, 0, 1]
        for k, f in enumerate(flips):
            h, n = h + f, n + 1
            a, b = a + f, b + (1 - f)
            self.play(Transform(c, curve(a, b)), Transform(txt, info(h, n, a, b)),
                      run_time=0.9 if k < 3 else 0.5)
            if n == 3:
                mle_line = DashedLine(ax.c2p(1, 0), ax.c2p(1, 3.3), color=RED)
                map_line = DashedLine(ax.c2p(0.8, 0), ax.c2p(0.8, 3.3), color=GREEN)
                note = T("MLE = 1 (gán P(sấp) = 0), MAP = 0.8", 24).to_edge(DOWN, buff=0.35)
                self.play(Create(mle_line), Create(map_line), FadeIn(note))
                self.wait(1.5)
                self.play(FadeOut(mle_line), FadeOut(map_line), FadeOut(note))
        end = T("7/10: MLE 0.7, MAP 8/12 ≈ 0.667, posterior hẹp dần khi có thêm dữ liệu", 24
                ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(end))
        self.wait(2.5)
