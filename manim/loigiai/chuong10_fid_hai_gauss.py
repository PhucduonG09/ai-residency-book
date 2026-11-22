"""Lời giải tham khảo — bài dựng cảnh chương 10: FID giữa hai Gauss 2D (ví dụ 10.12).

Chỉ mở sau khi đã tự làm. Render:  manim -qm chuong10_fid_hai_gauss.py FIDHaiGauss
"""
import sys
import numpy as np
from scipy.linalg import sqrtm
from manim import *

FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"


def T(s, size=28, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


def caption(s, size=24):
    t = T(s, size)
    return VGroup(BackgroundRectangle(t, fill_opacity=0.85, buff=0.12), t).to_edge(DOWN, buff=0.25)


def ellipse(ax, m, C, color):
    w, V = np.linalg.eigh(C)
    pts = [m + V @ (np.sqrt(w) * np.array([np.cos(a), np.sin(a)])) for a in np.linspace(0, 2 * np.pi, 80)]
    return VMobject(color=color, stroke_width=4).set_points_as_corners([ax.c2p(*p) for p in pts + [pts[0]]])


class FIDHaiGauss(Scene):
    def construct(self):
        m1, m2 = np.zeros(2), np.array([1.0, 2.0])
        C1, C2 = np.array([[1, 0.5], [0.5, 2]]), np.diag([2.0, 1.0])
        mean_part = float(((m1 - m2) ** 2).sum())
        cov_part = float(np.trace(C1 + C2 - 2 * sqrtm(C1 @ C2).real))     # tính bằng code, không gõ tay
        fid = mean_part + cov_part
        title = T("FID giữa hai Gauss 2D", 34).to_edge(UP, buff=0.3)
        ax = Axes(x_range=[-3, 4, 1], y_range=[-3, 4.5, 1], x_length=5.6, y_length=5.0, tips=False).shift(LEFT * 3.2 + DOWN * 0.2)
        e1, e2 = ellipse(ax, m1, C1, BLUE), ellipse(ax, m2, C2, RED)
        self.play(Write(title), Create(ax))
        self.play(Create(e1), Create(e2))
        cap = caption("Xanh: (m₁, C₁). Đỏ: (m₂, C₂). FID đo khoảng cách giữa hai Gauss này")
        self.play(FadeIn(cap))
        arrow = Arrow(ax.c2p(*m1), ax.c2p(*m2), buff=0, color=YELLOW)
        p1 = T(f"phần trung bình = |m₁ − m₂|² = {mean_part:.1f}", 26, YELLOW).move_to([3.2, 1.2, 0])
        self.play(Create(arrow), FadeIn(p1))
        self.wait(1.5)
        p2 = T(f"phần hiệp phương sai = {cov_part:.4f}", 26, GREEN).next_to(p1, DOWN, buff=0.4)
        self.play(FadeIn(p2))
        self.wait(1.5)
        tot = T(f"FID = {mean_part:.1f} + {cov_part:.4f} = {fid:.4f}", 30, WHITE).next_to(p2, DOWN, buff=0.6)
        self.play(FadeIn(tot))
        self.wait(3)
