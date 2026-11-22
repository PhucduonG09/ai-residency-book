"""Lời giải tham khảo — bài dựng cảnh chương 12: đường rủi ro–độ phủ.

Chỉ mở sau khi đã tự làm. Render:  manim -qm chuong12_rui_ro_do_phu.py RuiRoDoPhu
Dữ liệu: 10 dự đoán xếp theo độ tự tin giảm dần (đúng/sai như ví dụ 12.11). Hạ ngưỡng từng bước,
phần bị từ chối làm mờ; hiện độ phủ 0.3 / 0.5 / 0.7 / 1.0 và độ chính xác 1.000 / 0.800 / 0.714 / 0.500.
"""
import sys
from manim import *

FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"
CONF = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.55, 0.50, 0.30]
OK = [1, 1, 1, 0, 1, 0, 1, 0, 0, 0]


def T(s, size=28, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


def caption(s, size=24):
    t = T(s, size)
    if t.width > 12.6:
        t.scale(12.6 / t.width)
    return VGroup(BackgroundRectangle(t, fill_opacity=0.85, buff=0.12), t).to_edge(DOWN, buff=0.25)


class RuiRoDoPhu(Scene):
    def construct(self):
        self.add(T("Rủi ro – độ phủ: chỉ trả lời khi đủ tự tin", 32).to_edge(UP, buff=0.3))
        boxes = VGroup()
        for i in range(10):
            col = GREEN if OK[i] else RED
            r = Rectangle(width=1.0, height=1.0, color=col, fill_color=col, fill_opacity=0.7).move_to([-5.4 + 1.2 * i, 1.2, 0])
            boxes.add(VGroup(r, T(f"{CONF[i]:.2f}", 20).move_to(r)))
        self.play(FadeIn(boxes))
        cap = caption("Xếp theo độ tự tin giảm dần; xanh = đúng, đỏ = sai")
        self.play(FadeIn(cap))
        self.wait(1)
        result = None
        for n, cov in ((3, 0.3), (5, 0.5), (7, 0.7), (10, 1.0)):
            acc = sum(OK[:n]) / n
            line = Line([-5.4 + 1.2 * (n - 1) + 0.6, 0.4, 0], [-5.4 + 1.2 * (n - 1) + 0.6, 2.0, 0], color=YELLOW, stroke_width=5)
            new = T(f"độ phủ {cov:.1f}  ⇒  accuracy trên phần trả lời = {acc:.3f}", 30, YELLOW).move_to([0, -1.2, 0])
            anims = [boxes[i].animate.set_opacity(1.0 if i < n else 0.25) for i in range(10)]
            self.play(*anims, FadeIn(line))
            self.play(FadeIn(new) if result is None else Transform(result, new))
            result = result or new
            self.wait(1.5)
            self.play(FadeOut(line))
        self.play(FadeOut(cap))
        cap = caption("Từ chối nhiều hơn thì phần còn lại chính xác hơn, nếu độ tự tin có ý nghĩa; phải được phép nói \"không biết\"")
        self.play(FadeIn(cap))
        self.wait(2)
