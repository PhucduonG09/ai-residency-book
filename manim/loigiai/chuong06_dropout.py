"""Lời giải tham khảo — bài dựng cảnh chương 6: inverted dropout.

Chỉ mở sau khi đã tự làm. Render:  manim -qm chuong06_dropout.py InvertedDropout

Lưu ý: mỗi lần chạy ở đây tắt đúng 4/8 neuron chỉ để minh họa. Dropout thật tắt từng neuron
độc lập, nên số neuron bị tắt là ngẫu nhiên (Binomial(8, 0.5)), không phải luôn bằng p·n.
"""
import sys
import numpy as np
from manim import *

FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"


def T(s, size=28, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


class InvertedDropout(Scene):
    def construct(self):
        p = 0.5
        h = np.array([0.8, 1.2, 0.4, 1.0, 0.6, 1.4, 0.2, 0.4])     # activation của 8 neuron, tổng 6.0
        title = T("Inverted dropout, p = 0.5", 34).to_edge(UP, buff=0.3)
        self.play(Write(title))
        circles = VGroup(*[Circle(0.42, color=WHITE).move_to([-5.25 + 1.5 * i, 0.6, 0]) for i in range(8)])
        vals = VGroup(*[T(f"{v:g}", 22).move_to(c) for v, c in zip(h, circles)])
        self.play(Create(circles), FadeIn(vals))
        total = T(f"tổng đầu ra = {h.sum():.1f}", 26).shift(DOWN * 1.0)
        mode = T("chế độ eval: không tắt, không nhân", 26, GREEN).shift(DOWN * 1.7)
        self.play(FadeIn(total), FadeIn(mode))
        self.wait(1)

        masks = [[1, 0, 1, 1, 0, 0, 1, 0], [0, 1, 0, 1, 1, 0, 0, 1], [1, 1, 0, 0, 0, 1, 1, 0]]   # mỗi lần giữ 4/8 (chỉ để minh họa, thực tế là ngẫu nhiên)
        sums = []
        for run in range(3):
            keep = np.array(masks[run], dtype=bool)
            out = h * keep / (1 - p)
            sums.append(out.sum())
            new_vals = VGroup(*[T(f"{v:g}", 22, YELLOW if k else GREY).move_to(c)
                                for v, k, c in zip(out, keep, circles)])
            new_total = T(f"lần {run + 1}: tổng = {out.sum():.1f}   (giữ {keep.sum()}/8, nhân 2)", 26).move_to(total)
            new_mode = T("chế độ train: tắt ngẫu nhiên, phần còn lại chia cho (1 − p)", 26, YELLOW).move_to(mode)
            self.play(*[c.animate.set_fill(YELLOW if k else GREY_D, opacity=0.35 if k else 0.9) for c, k in zip(circles, keep)],
                      Transform(vals, new_vals), Transform(total, new_total), Transform(mode, new_mode), run_time=1.2)
            self.wait(1.2)

        # Lưu ý: tắt đúng 4/8 neuron chỉ để minh họa; dropout thật tắt mỗi neuron độc lập, số bị tắt là ngẫu nhiên.
        note = T("Tắt đúng 4 chỉ để minh họa: thực tế số neuron bị tắt là ngẫu nhiên", 22, GREY_B).shift(DOWN * 2.5)
        mean_line = T(f"Mỗi lần tổng dao động, nhưng 𝔼[tổng] = {h.sum():.1f} đúng bằng lúc eval", 26).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(mean_line), FadeIn(note))
        self.wait(1)
        back = VGroup(*[T(f"{v:g}", 22).move_to(c) for v, c in zip(h, circles)])
        self.play(*[c.animate.set_fill(BLACK, opacity=0) for c in circles], Transform(vals, back),
                  Transform(total, T(f"tổng đầu ra = {h.sum():.1f}", 26).move_to(total)),
                  Transform(mode, T("model.eval(): mọi neuron sáng, test không cần nhân gì", 26, GREEN).move_to(mode)))
        self.wait(2.5)
