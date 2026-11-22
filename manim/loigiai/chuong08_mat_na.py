"""Lời giải tham khảo — bài dựng cảnh chương 8: mặt nạ trước và sau softmax.

Chỉ mở sau khi đã tự làm. Render:  manim -qm chuong08_mat_na.py MatNaNhanQua
"""
import sys
import numpy as np
from manim import *

FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"


def T(s, size=28, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


def grid(rows, color, center):
    nr, nc = len(rows), len(rows[0])
    return VGroup(*[T(v, 22, color).move_to(center + np.array([(j - (nc - 1) / 2) * 1.2, -(i - (nr - 1) / 2) * 0.6, 0]))
                    for i, r in enumerate(rows) for j, v in enumerate(r)])


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class MatNaNhanQua(Scene):
    def construct(self):
        S = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 2]]) / np.sqrt(2)
        tri = np.tril(np.ones((3, 3)))
        before = softmax(np.where(tri == 1, S, -np.inf))     # mask TRƯỚC softmax
        after = softmax(S) * tri                              # mask SAU softmax
        fmt = lambda M: [[f"{v:.4f}".rstrip("0").rstrip(".") if v else "0" for v in r] for r in M]
        title = T("Mặt nạ nhân quả: trước hay sau softmax?", 34).to_edge(UP, buff=0.3)
        self.play(Write(title))
        cl, cr = LEFT * 5.0 + UP * 0.3, RIGHT * 1.6 + UP * 0.3
        hl = T("mask TRƯỚC softmax (đúng)", 20, GREEN).move_to(cl + UP * 1.6 + RIGHT * 1.0)
        hr = T("mask SAU softmax (sai)", 22, RED).move_to(cr + UP * 1.6)
        gl, gr = grid(fmt(before), WHITE, cl), grid(fmt(after), WHITE, cr)
        self.play(FadeIn(hl), FadeIn(hr), FadeIn(gl), FadeIn(gr))
        sl = VGroup(*[T(f"tổng = {before[i].sum():.4f}", 20, GREEN).move_to(cl + np.array([3.0, -(i - 1) * 0.6, 0])) for i in range(3)])
        sr = VGroup(*[T(f"tổng = {after[i].sum():.4f}", 20, RED if abs(after[i].sum() - 1) > 1e-9 else GREEN).move_to(cr + np.array([3.0, -(i - 1) * 0.6, 0])) for i in range(3)])
        self.wait(1)
        self.play(FadeIn(sl), FadeIn(sr))
        cap = T("Che trước: mọi hàng vẫn là phân phối (tổng 1). Che sau: hàng 1 và 2 mất khối lượng", 22)
        cap = VGroup(BackgroundRectangle(cap, fill_opacity=0.85, buff=0.12), cap).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(cap))
        self.wait(2.5)
        cap2 = T("Mẫu số của softmax vẫn gồm cả các ô tương lai, nên chúng đã nhận khối lượng trước khi bị xóa", 20)
        cap2 = VGroup(BackgroundRectangle(cap2, fill_opacity=0.85, buff=0.12), cap2).to_edge(DOWN, buff=0.25)
        self.play(FadeOut(cap), FadeIn(cap2))
        self.wait(3)
