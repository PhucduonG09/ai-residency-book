"""Lời giải tham khảo — bài dựng cảnh chương 11: xác suất bắt được bản sao theo n-gram.

Chỉ mở sau khi đã tự làm. Render:  manim -qm chuong11_ngram_bat_trung.py NGramBatTrung
Xác suất tính chính xác bằng quy hoạch động (cùng hàm với lab 5): L = 60, n = 5, 8, 13, q từ 0 đến 0.4.
"""
import sys
import numpy as np
from manim import *

FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"


def T(s, size=28, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


def p_detect(L, n, q):
    """P(còn ít nhất một đoạn n token liền nhau không bị đổi), mỗi token bị đổi độc lập với xác suất q."""
    s = np.zeros(n)
    s[0] = 1.0
    for _ in range(L):
        ns = np.zeros(n)
        ns[0] = s.sum() * q
        for j in range(n - 1):
            ns[j + 1] = s[j] * (1 - q)
        s = ns
    return float(1 - s.sum())


def caption(s, size=24):
    t = T(s, size)
    if t.width > 12.6:
        t.scale(12.6 / t.width)
    return VGroup(BackgroundRectangle(t, fill_opacity=0.85, buff=0.12), t).to_edge(DOWN, buff=0.25)


class NGramBatTrung(Scene):
    def construct(self):
        L = 60
        title = T("Bắt bản sao đã sửa bằng n-gram (tài liệu dài 60 token)", 30).to_edge(UP, buff=0.3)
        self.play(Write(title))
        x0, x1, y0, y1 = -5.0, 4.0, -2.2, 2.3          # q từ 0 đến 0.4, xác suất từ 0 đến 1

        def P(q, p):
            return np.array([x0 + (x1 - x0) * q / 0.4, y0 + (y1 - y0) * p, 0])

        axes = VGroup(Line(P(0, 0), P(0.4, 0), color=GREY_B), Line(P(0, 0), P(0, 1), color=GREY_B))
        xt = VGroup(*[T(f"{v:.1f}", 18).move_to(P(v, 0) + DOWN * 0.3) for v in (0, 0.1, 0.2, 0.3, 0.4)])
        yt = VGroup(*[T(f"{v:g}", 18).move_to(P(0, v) + LEFT * 0.4) for v in (0, 0.5, 1)])
        xl = T("q = xác suất mỗi token bị đổi", 20).move_to(P(0.2, 0) + DOWN * 0.7)
        yl = T("xác suất bắt được", 20).rotate(PI / 2).move_to(P(0, 0.5) + LEFT * 1.0)
        self.play(Create(axes), FadeIn(xt), FadeIn(yt), FadeIn(xl), FadeIn(yl))

        colors = {5: GREEN, 8: YELLOW, 13: RED}
        qs = np.linspace(0, 0.4, 41)
        curves = {}
        for n, col in colors.items():
            pts = [P(q, p_detect(L, n, q)) for q in qs]
            curves[n] = VMobject(color=col, stroke_width=5).set_points_as_corners(pts)
        leg = VGroup(*[T(f"n = {n}", 22, col) for n, col in colors.items()]).arrange(DOWN, buff=0.15, aligned_edge=LEFT).move_to([5.6, 0.6, 0])
        cap = caption("Bản sao còn bị bắt nếu tồn tại một đoạn n token liền nhau không bị đổi")
        self.add(cap)
        self.play(*[Create(c) for c in curves.values()], FadeIn(leg), run_time=3)
        self.wait(1)

        p13 = p_detect(L, 13, 0.2)
        dot = Dot(P(0.2, p13), color=WHITE, radius=0.09)
        lab = T(f"n = 13, q = 0.2:\n{p13:.3f}", 24, RED).move_to(P(0.10, 0.22))
        pointer = Arrow(lab.get_top() + UP * 0.05 + RIGHT * 0.3, dot.get_center() + DOWN * 0.1, buff=0.1, color=WHITE, stroke_width=2, max_tip_length_to_length_ratio=0.15)
        new_cap = caption("n = 13 bỏ sót hơn một nửa bản sao khi 20% token bị đổi")
        self.play(FadeOut(cap), FadeIn(new_cap), FadeIn(dot), FadeIn(lab), Create(pointer))
        self.wait(2)
        note = T("n = 5 bền hơn, nhưng gắn cờ nhầm 13% tài liệu độc lập (văn bản Zipf)", 24, GREEN).to_edge(UP, buff=1.1)
        cap3 = caption("Chọn n là một đánh đổi: bỏ sót bản sửa hay gắn cờ nhầm cụm từ phổ biến")
        self.play(FadeOut(new_cap), FadeIn(cap3), FadeIn(note))
        self.wait(3)
