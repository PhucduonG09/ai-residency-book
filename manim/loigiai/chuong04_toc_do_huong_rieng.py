"""Lời giải tham khảo — bài dựng cảnh chương 4: GD theo từng hướng riêng có thanh trượt η.

Chỉ mở sau khi đã tự làm. Render:  manim -qm chuong04_toc_do_huong_rieng.py ThanhTruotEta
"""
import sys
import numpy as np
from manim import *

FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"
LAM = np.array([1.0, 100.0])


def T(s, size=28, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


def steps_to(eta, tol=1e-6, kmax=5000):
    """Số bước để mọi |1 − ηλᵢ|ᵏ < tol; None nếu một hướng có |1 − ηλ| ≥ 1."""
    r = np.max(np.abs(1 - eta * LAM))
    if r >= 1:
        return None
    if r == 0:
        return 1
    return int(np.ceil(np.log(tol) / np.log(r)))


class ThanhTruotEta(Scene):
    def construct(self):
        title = T("Chọn η cho f = ½(x₁² + 100x₂²)", 32).to_edge(UP, buff=0.3)
        line = NumberLine(x_range=[0, 0.022, 0.002], length=10, include_tip=False).shift(UP * 0.6)
        marks = VGroup(*[T(f"{v:.3f}", 16, GREY_B).next_to(line.n2p(v), DOWN, buff=0.15)
                         for v in (0, 0.01, 0.02)])
        danger = Line(line.n2p(0.02), line.n2p(0.022), color=RED, stroke_width=10)
        dl = T("η > 2/L: phân kỳ", 20, RED).next_to(danger, UP, buff=0.6)
        self.play(Write(title), Create(line), FadeIn(marks), Create(danger), FadeIn(dl))
        eta = ValueTracker(0.002)
        knob = always_redraw(lambda: Triangle(color=YELLOW, fill_opacity=1).scale(0.15).rotate(PI)
                             .next_to(line.n2p(eta.get_value()), UP, buff=0.05))

        def panel():
            e = eta.get_value()
            r1, r2 = abs(1 - e * LAM[0]), abs(1 - e * LAM[1])
            k = steps_to(e)
            g = VGroup(T(f"η = {e:.4f}", 26, YELLOW),
                       T(f"hướng λ = 1:    |1 − η| = {r1:.4f}", 24, GREEN),
                       T(f"hướng λ = 100:  |1 − 100η| = {r2:.4f}", 24, RED),
                       T("số bước tới 10⁻⁶: " + ("phân kỳ" if k is None else f"{k}"), 26))
            return g.arrange(DOWN, aligned_edge=LEFT, buff=0.2).shift(DOWN * 1.7)

        p = always_redraw(panel)
        self.add(knob, p)
        for target in (0.01, 2 / 101, 0.0201):
            self.play(eta.animate.set_value(target), run_time=2.5, rate_func=smooth)
            self.wait(1.5)
        note = T("Tốt nhất tại η = 2/(L + μ): hai hướng cân bằng, tốc độ (κ − 1)/(κ + 1)", 22, YELLOW)
        note.to_edge(DOWN, buff=0.3)
        self.play(eta.animate.set_value(2 / 101), FadeIn(note), run_time=1.5)
        self.wait(2.5)
