"""Lời giải tham khảo — bài dựng cảnh chương 7: receptive field của dilated conv.

Chỉ mở sau khi đã tự làm. Render:  manim -qm chuong07_dilated_rf.py DilatedRF
"""
import sys
import numpy as np
from manim import *

FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"


def T(s, size=28, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


class DilatedRF(Scene):
    def construct(self):
        title = T("Conv k = 3, dilation 1, 2, 4: RF 3, 7, 15", 32).to_edge(UP, buff=0.3)
        self.play(Write(title))
        n, sp, x0 = 17, 0.62, -5.0
        center = 8
        ys = [-2.0, -0.8, 0.4, 1.6]                     # lớp 0 (vào) … lớp 3 (ra)
        dots = [[Dot([x0 + sp * i, ys[l], 0], radius=0.09, color=GREY_B) for i in range(n)] for l in range(4)]
        self.play(*[FadeIn(d) for row in dots for d in row], run_time=1)
        self.play(dots[3][center].animate.set_color(YELLOW).scale(1.5))
        lo = hi = center
        rf_txt = None
        for layer, d in ((2, 4), (1, 2), (0, 1)):          # đi ngược từ đầu ra: lớp có dilation 4 trước
            lines = VGroup()
            idx = set()
            for j in range(lo, hi + 1):
                if layer == 2 and j != center:
                    continue
                if layer < 2 and (j not in prev):
                    continue
                for off in (-d, 0, d):
                    lines.add(Line(dots[layer + 1][j].get_center(), dots[layer][j + off].get_center(),
                                   stroke_width=2, color=YELLOW, stroke_opacity=0.8))
                    idx.add(j + off)
            prev = idx
            lo, hi = min(idx), max(idx)
            self.play(Create(lines), *[dots[layer][j].animate.set_color(YELLOW) for j in idx], run_time=1.0)
            if layer == 0:
                new = T(f"RF = {hi - lo + 1} điểm đầu vào", 24, YELLOW).move_to([0, ys[0] - 0.55, 0])
                self.play(FadeIn(new), run_time=0.4)
        note = T("Không giãn: 3 lớp 3×3 chỉ có RF = 7. Giãn 1, 2, 4: RF = 15", 24, GREEN).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(note))
        self.wait(3)
