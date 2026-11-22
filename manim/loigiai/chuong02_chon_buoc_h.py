"""Lời giải tham khảo — bài dựng cảnh chương 2: chọn bước h khi gradient check.

Chỉ mở sau khi đã tự làm. Render:  manim -qm chuong02_chon_buoc_h.py ChonBuocH
"""
import math
import sys
import numpy as np
from manim import *

FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"


def T(s, size=28, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


def central_error(dtype, h, x0=1.0):
    x, h = dtype(x0), dtype(h)
    c = (np.sin(x + h) - np.sin(x - h)) / (dtype(2) * h)
    return abs(float(c) - math.cos(x0))


class ChonBuocH(Scene):
    def construct(self):
        title = T("Sai số sai phân trung tâm theo h (f = sin, x = 1)", 30).to_edge(UP, buff=0.3)
        # trục dùng log10 dời gốc: X = log10 h + 9 ∈ [0, 8], Y = log10 sai số + 12 ∈ [0, 12]
        ax = Axes(x_range=[0, 8, 1], y_range=[0, 12, 2], x_length=7.5, y_length=4.6, tips=False).shift(LEFT * 1.6 + DOWN * 0.5)
        P = lambda lh, le: ax.c2p(lh + 9, le + 12)
        sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
        labels = VGroup()
        for k in range(-9, 0, 2):
            labels.add(T(f"10{str(k).translate(sup)}", 18).next_to(P(k, -12), DOWN, buff=0.12))
        for k in range(-12, 1, 4):
            labels.add(T(f"10{str(k).translate(sup)}", 18).next_to(P(-9, k), LEFT, buff=0.12))
        xname = T("h", 22).next_to(ax.x_axis, RIGHT, buff=0.2)
        yname = T("sai số", 22).next_to(ax.y_axis, UP, buff=0.15)
        self.play(Write(title), Create(ax), FadeIn(labels), FadeIn(xname), FadeIn(yname))

        hs = np.logspace(-9, -1, 33)
        curves = {}
        for dt, col, name in [(np.float64, GREEN, "float64"), (np.float32, RED, "float32")]:
            pts = [P(np.log10(h), max(np.log10(central_error(dt, h) + 1e-17), -12)) for h in hs]
            line = VMobject(color=col, stroke_width=4).set_points_as_corners(pts)
            curves[name] = (line, pts)
        legend = VGroup(T("float64", 22, GREEN), T("float32", 22, RED)).arrange(DOWN, aligned_edge=LEFT)
        legend.move_to(RIGHT * 4.6 + UP * 1.6)
        self.play(Create(curves["float64"][0]), FadeIn(legend[0]), run_time=2)
        self.play(Create(curves["float32"][0]), FadeIn(legend[1]), run_time=2)
        # đánh dấu điểm tốt nhất
        notes = VGroup()
        for name, dt, col in [("float64", np.float64, GREEN), ("float32", np.float32, RED)]:
            errs = [central_error(dt, h) for h in hs]
            i = int(np.argmin(errs))
            d = Dot(curves[name][1][i], color=YELLOW, radius=0.08)
            notes.add(d)
        opt = T("h tốt nhất ≈ ∛ε: 6 × 10⁻⁶ (float64), 5 × 10⁻³ (float32)", 22, YELLOW)
        opt.next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(notes), FadeIn(opt))
        cap_t = T("Bên phải: sai số cắt cụt ~ h². Bên trái: sai số làm tròn ~ ε/h", 24)
        cap = VGroup(BackgroundRectangle(cap_t, fill_opacity=0.85, buff=0.12), cap_t).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(cap))
        self.wait(3)
