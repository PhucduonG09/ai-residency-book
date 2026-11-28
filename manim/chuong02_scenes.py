"""Các cảnh Manim cho Chương 2 — Giải tích ma trận và backpropagation.

Render một cảnh (chất lượng 720p):
    manim -qm chuong02_scenes.py GradientDongMuc

Chỉ dùng Text (không cần LaTeX), nên chỉ cần Manim Community + ffmpeg.
Mỗi cảnh dùng đúng số liệu của ví dụ giải mẫu trong sách.
"""
import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parent))
from giong_doc import noi  # giọng đọc Google (gTTS); MANIM_KHONG_GIONG=1 để render câm
import nen

nen.bat()  # nền lưới mờ thay cho nền đen trơn

FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"
C1, C2, C_ACC = GREEN, RED, YELLOW


def T(s, size=28, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


def caption(s):
    t = T(s, 26)
    return VGroup(BackgroundRectangle(t, fill_opacity=0.85, buff=0.12), t).to_edge(DOWN, buff=0.25)


def swap_caption(scene, old, text, run_time=0.5):
    new = caption(text)
    if old is None:
        scene.play(FadeIn(new), run_time=run_time)
    else:
        # Tuần tự (mờ dần rồi mới hiện) để phụ đề cũ và mới không bao giờ chồng nhau.
        scene.play(FadeOut(old), run_time=run_time / 2)
        scene.play(FadeIn(new), run_time=run_time / 2)
    return new


def tick_labels(ax, xs, ys, size=18, fmt="{:g}"):
    """Nhãn số trên trục bằng Text (Axes.include_numbers cần LaTeX)."""
    x0 = 0 if ax.x_range[0] <= 0 <= ax.x_range[1] else ax.x_range[0]
    y0 = 0 if ax.y_range[0] <= 0 <= ax.y_range[1] else ax.y_range[0]
    g = VGroup()
    for v in xs:
        g.add(T(fmt.format(v), size, GREY_A).next_to(ax.c2p(v, y0), DOWN, buff=0.12))
    for v in ys:
        g.add(T(fmt.format(v), size, GREY_A).next_to(ax.c2p(x0, v), LEFT, buff=0.12))
    return g


def f21(x1, x2):
    return x1 ** 2 * x2 + 3 * x2


# ---------------------------------------------------------------- 2.1
class GradientDongMuc(Scene):
    """∇f vuông góc đường đồng mức và chỉ hướng tăng nhanh nhất (Ví dụ 2.1)."""

    def construct(self):
        title = T("Gradient vuông góc với đường đồng mức", 32).to_edge(UP, buff=0.3)
        ax = Axes(x_range=[-1, 3, 1], y_range=[0, 4, 1], x_length=5.0, y_length=5.0, tips=False,
                  axis_config={"include_numbers": False}).shift(LEFT * 3.0 + DOWN * 0.05)
        levels = VGroup()
        for c, col in [(4, BLUE_E), (6, BLUE_D), (8, C_ACC), (10, BLUE_D), (12, BLUE_E)]:
            levels.add(ax.plot(lambda t, c=c: c / (t ** 2 + 3), x_range=[-1, 3], color=col,
                               stroke_width=4 if c == 8 else 2))
        lab8 = T("f = 8", 22, C_ACC).next_to(ax.c2p(3, 8 / (3 ** 2 + 3)), RIGHT, buff=0.15)
        p0 = ax.c2p(1, 2)
        dot = Dot(p0, color=WHITE)
        unit = ax.c2p(1, 0)[0] - ax.c2p(0, 0)[0]
        g = np.array([4.0, 4.0])
        gdir = g / np.linalg.norm(g)
        garrow = Arrow(p0, p0 + 1.3 * unit * np.array([*gdir, 0]), buff=0, color=C1, stroke_width=6)
        glab = T("∇f = (4, 4)", 24, C1).next_to(garrow.get_end(), UP, buff=0.1)
        tang = DashedLine(p0 - unit * 0.9 * np.array([1, -1, 0]) / np.sqrt(2),
                          p0 + unit * 0.9 * np.array([1, -1, 0]) / np.sqrt(2), color=GREY_B)

        nums = tick_labels(ax, [0, 1, 2, 3], [1, 2, 3, 4])
        with noi(self, "Xét hàm f bằng x một bình phương nhân x hai, cộng ba x hai. "
                       "Mỗi đường cong là một đường đồng mức, nơi giá trị của f không đổi."):
            self.play(Write(title), Create(ax), FadeIn(nums))
            self.play(*[Create(l) for l in levels], FadeIn(lab8), FadeIn(dot), run_time=2)
        with noi(self, "Tại điểm một, hai, f bằng tám, và gradient bằng bốn, bốn."):
            cap = swap_caption(self, None, "Tại x₀ = (1, 2): f = 8, ∇f = (2x₁x₂, x₁² + 3) = (4, 4)")
            self.play(GrowArrow(garrow), FadeIn(glab), Create(tang))
            self.wait(1)

        ang = ValueTracker(np.deg2rad(160))

        def u():
            a = ang.get_value()
            return np.array([np.cos(a), np.sin(a)])

        uarrow = always_redraw(lambda: Arrow(p0, p0 + unit * np.array([*u(), 0]), buff=0,
                                             color=C2, stroke_width=6))
        panel_title = T("Đạo hàm theo hướng u", 26).move_to(RIGHT * 3.6 + UP * 2.2)
        val = always_redraw(lambda: T(f"∇fᵀu = {g @ u():+.3f}", 30, C2).next_to(panel_title, DOWN, buff=0.4))
        ang_txt = always_redraw(lambda: T(
            f"u = ({u()[0]:+.2f}, {u()[1]:+.2f})", 24).next_to(panel_title, DOWN, buff=1.1))
        with noi(self, "Giờ quay một vector đơn vị u quanh điểm này. Tích vô hướng giữa gradient và u "
                       "cho biết f tăng nhanh cỡ nào khi đi theo hướng u."):
            self.play(GrowArrow(uarrow), FadeIn(panel_title), FadeIn(val), FadeIn(ang_txt))
            cap = swap_caption(self, cap, "Quay vector đơn vị u quanh x₀ và đọc tốc độ thay đổi của f")
            self.play(ang.animate.set_value(np.arctan2(4, 3)), run_time=2.5)
        note1 = T("u = (0.6, 0.8): 5.6", 22, GREY_A).next_to(panel_title, DOWN, buff=1.8)
        with noi(self, "Theo hướng không phẩy sáu, không phẩy tám, tốc độ là năm phẩy sáu."):
            self.play(FadeIn(note1))
        note2 = T("trùng ∇f: lớn nhất = ‖∇f‖ = 4√2 ≈ 5.657", 22, C1).next_to(note1, DOWN, buff=0.25)
        with noi(self, "Khi u trùng hướng gradient, tốc độ đạt lớn nhất, "
                       "bằng độ dài của gradient, bốn căn hai, xấp xỉ năm phẩy sáu năm bảy."):
            self.play(ang.animate.set_value(PI / 4), run_time=1.2)
            self.play(FadeIn(note2))
            self.wait(0.8)
        note3 = T("tiếp tuyến đồng mức: bằng 0", 22, GREY_B).next_to(note2, DOWN, buff=0.25)
        with noi(self, "Còn khi u đi dọc tiếp tuyến của đường đồng mức, tốc độ bằng không."):
            self.play(ang.animate.set_value(-PI / 4), run_time=2)
            self.play(FadeIn(note3))
        with noi(self, "Vậy gradient chỉ hướng f tăng nhanh nhất, và luôn vuông góc với đường đồng mức."):
            cap = swap_caption(self, cap, "∇f chỉ hướng tăng nhanh nhất và vuông góc với đường f = 8")
        self.wait(1.8)


class TaylorXapXi(Scene):
    """Taylor bậc 1 và bậc 2 dọc hướng d = (0.5, 0.5) từ x₀ = (1, 2) (Ví dụ 2.1)."""

    def construct(self):
        title = T("Taylor bậc 1 và bậc 2 chỉ đúng cục bộ", 32).to_edge(UP, buff=0.3)
        sub = T("Lát cắt g(t) = f(x₀ + t·d), d = (0.5, 0.5)", 24, GREY_A).next_to(title, DOWN, buff=0.15)
        ax = Axes(x_range=[0, 1.1, 0.2], y_range=[7.5, 14, 1], x_length=6.6, y_length=4.4, tips=False,
                  axis_config={"include_numbers": False}).shift(LEFT * 2.6 + DOWN * 0.45)
        true_g = ax.plot(lambda t: f21(1 + 0.5 * t, 2 + 0.5 * t), x_range=[0, 1.1], color=WHITE, stroke_width=5)
        lin = ax.plot(lambda t: 8 + 4 * t, x_range=[0, 1.1], color=C2, stroke_width=3)
        quad = ax.plot(lambda t: 8 + 4 * t + t ** 2, x_range=[0, 1.1], color=C1, stroke_width=3)
        legend = VGroup(T("f thật", 22), T("bậc 1: 8 + 4t", 22, C2),
                        T("bậc 2: 8 + 4t + t²", 22, C1)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        legend.move_to(RIGHT * 3.9 + UP * 1.5)
        nums = tick_labels(ax, [0, 0.2, 0.4, 0.6, 0.8, 1.0], [8, 9, 10, 11, 12, 13])
        self.play(Write(title), FadeIn(sub), Create(ax), FadeIn(nums))
        self.play(Create(true_g), FadeIn(legend[0]))
        self.play(Create(lin), FadeIn(legend[1]))
        self.play(Create(quad), FadeIn(legend[2]))

        def gap_marks(t, vals):
            return VGroup(*[Dot(ax.c2p(t, v), radius=0.06, color=c) for v, c in vals])

        info = VGroup()
        for t, d_txt in [(0.2, "d = (0.1, 0.1)"), (1.0, "d = (0.5, 0.5)")]:
            tv, t1, t2 = f21(1 + 0.5 * t, 2 + 0.5 * t), 8 + 4 * t, 8 + 4 * t + t ** 2
            marks = gap_marks(t, [(tv, WHITE), (t1, C2), (t2, C1)])
            vline = DashedLine(ax.c2p(t, 7.5), ax.c2p(t, 14), color=GREY_B, stroke_width=2)
            block = VGroup(T(f"t = {t:.1f}: {d_txt}", 22, C_ACC),
                           T(f"thật {tv:.3f}", 22),
                           T(f"sai bậc 1: {tv - t1:.3f}", 22, C2),
                           T(f"sai bậc 2: {tv - t2:.3f}", 22, C1)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            if len(info) == 0:
                block.next_to(legend, DOWN, buff=0.4).align_to(legend, LEFT)
            else:
                block.next_to(info[-1], DOWN, buff=0.3).align_to(legend, LEFT)
            self.play(Create(vline), FadeIn(marks))
            self.play(FadeIn(block))
            info.add(block)
            self.wait(0.8)
        self.play(FadeIn(caption("Bước ×5 → sai bậc 1 ×27, sai bậc 2 ×125: tỉ lệ ‖d‖² và ‖d‖³")))
        self.wait(3)


class JacobianOVuong(Scene):
    """Jacobian của phép đổi tọa độ cực tại (r, θ) = (2, π/6): ô nhỏ → hình bình hành, diện tích ×2 (Ví dụ 2.2)."""

    def construct(self):
        title = T("Jacobian = xấp xỉ tuyến tính tốt nhất tại một điểm", 30).to_edge(UP, buff=0.3)
        r0, th0 = 2.0, np.pi / 6
        # Cả hai panel dùng CÙNG thang: 2.0 đơn vị màn hình cho mỗi đơn vị dữ liệu, trên cả hai trục,
        # nên diện tích nhìn thấy đúng bằng diện tích thật (ảnh của ô rộng gấp 2 ô gốc).
        U = 2.0
        left = Axes(x_range=[1.5, 2.5, 0.5], y_range=[0.2, 1.2, 0.2], x_length=1.0 * U, y_length=1.0 * U, tips=False,
                    axis_config={"include_numbers": False}).shift(LEFT * 5.0 + DOWN * 0.45)
        right = Axes(x_range=[0.5, 2.5, 0.5], y_range=[0, 2, 0.5], x_length=2.0 * U, y_length=2.0 * U, tips=False,
                     axis_config={"include_numbers": False}).shift(RIGHT * 2.6 + DOWN * 0.45)
        ll = T("(r, θ)", 24).next_to(left, UP, buff=0.1)
        rl = T("(x, y) = (r cos θ, r sin θ)", 24).next_to(right, UP, buff=0.1)
        arrow = Arrow(LEFT * 3.2 + DOWN * 0.45, LEFT * 1.2 + DOWN * 0.45, color=GREY_A)

        rs = np.linspace(1.6, 2.4, 5)
        ths = np.linspace(th0 - 0.3, th0 + 0.3, 5)
        grid_l, grid_r = VGroup(), VGroup()
        for r in rs:
            grid_l.add(Line(left.c2p(r, ths[0]), left.c2p(r, ths[-1]), color=BLUE_D, stroke_width=2))
            grid_r.add(ParametricFunction(lambda t, r=r: right.c2p(r * np.cos(t), r * np.sin(t)),
                                          t_range=[ths[0], ths[-1]], color=BLUE_D, stroke_width=2))
        for th in ths:
            grid_l.add(Line(left.c2p(rs[0], th), left.c2p(rs[-1], th), color=BLUE_D, stroke_width=2))
            grid_r.add(ParametricFunction(lambda t, th=th: right.c2p(t * np.cos(th), t * np.sin(th)),
                                          t_range=[rs[0], rs[-1]], color=BLUE_D, stroke_width=2))
        dr = dth = 0.2
        cell = Polygon(left.c2p(r0, th0), left.c2p(r0 + dr, th0), left.c2p(r0 + dr, th0 + dth),
                       left.c2p(r0, th0 + dth), color=C_ACC, fill_opacity=0.5, stroke_width=2)
        pol = lambda r, t: right.c2p(r * np.cos(t), r * np.sin(t))
        img = Polygon(*[pol(r0 + dr * a, th0 + dth * b) for a, b in [(0, 0), (1, 0), (1, 1), (0, 1)]],
                      color=C_ACC, fill_opacity=0.5, stroke_width=2)
        J = np.array([[np.cos(th0), -r0 * np.sin(th0)], [np.sin(th0), r0 * np.cos(th0)]])
        o = np.array([r0 * np.cos(th0), r0 * np.sin(th0)])
        par = Polygon(*[right.c2p(*(o + J @ np.array([dr * a, dth * b])))
                        for a, b in [(0, 0), (1, 0), (1, 1), (0, 1)]],
                      color=C1, stroke_width=4)
        nums = VGroup(tick_labels(left, [1.5, 2.0, 2.5], [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]),
                      tick_labels(right, [0.5, 1.0, 1.5, 2.0, 2.5], [0, 0.5, 1.0, 1.5, 2.0]))
        self.play(Write(title), Create(left), Create(right), FadeIn(nums), FadeIn(ll), FadeIn(rl), GrowArrow(arrow))
        self.play(Create(grid_l), run_time=1.2)
        self.play(TransformFromCopy(grid_l, grid_r), run_time=2)
        cap = swap_caption(self, None, "Ảnh của lưới thẳng là lưới cong: ánh xạ phi tuyến")
        self.wait(0.8)
        self.play(FadeIn(cell))
        self.play(TransformFromCopy(cell, img), run_time=1.5)
        cap = swap_caption(self, cap, "Ô nhỏ quanh (2, π/6) → gần như hình bình hành có cạnh là hai cột của J")
        self.play(Create(par))
        jtxt = VGroup(T("J ≈ [[0.866, −1.000], [0.500, 1.732]]", 24),
                      T("det J = r = 2", 24, C1)).arrange(RIGHT, buff=0.6)
        jtxt.next_to(title, DOWN, buff=0.25)
        bg = BackgroundRectangle(jtxt, fill_opacity=0.85, buff=0.1)
        self.play(FadeIn(bg), FadeIn(jtxt))
        cap = swap_caption(self, cap, "Hai khung cùng thang trục: diện tích ô nhân |det J| = 2")
        self.wait(3)


# ---------------------------------------------------------------- 2.3
def node(label, pos, color=WHITE, r=0.42):
    c = Circle(radius=r, color=color, stroke_width=3).move_to(pos)
    return VGroup(c, T(label, 24, color).move_to(pos))


def edge(a, b):
    return Arrow(a.get_right(), b.get_left(), buff=0.05, color=GREY_B, stroke_width=3,
                 max_tip_length_to_length_ratio=0.12)


class DoThiNeuron(Scene):
    """Forward rồi backward trên đồ thị của một neuron sigmoid (Ví dụ 2.6)."""

    def construct(self):
        title = T("Đồ thị tính toán: forward lưu giá trị, backward nhân gradient", 28).to_edge(UP, buff=0.3)
        ys = [2.0, 1.0, 0.0, -1.0, -2.5]
        names = ["w₁", "x₁", "w₂", "x₂", "b"]
        vals = ["0.5", "2", "−1", "1", "0.2"]
        grads = ["0.4950", "0.1238", "0.2475", "−0.2475", "0.2475"]
        X0 = -4.8
        leaves = [T(n, 28).move_to([X0, y, 0]) for n, y in zip(names, ys)]
        m1 = node("×", [-2.6, 1.5, 0])
        m2 = node("×", [-2.6, -0.5, 0])
        add = node("+", [-0.3, 0.0, 0])
        sg = node("σ", [2.0, 0.0, 0])
        out = T("s", 30).move_to([4.0, 0.0, 0])
        edges = VGroup(
            Arrow(leaves[0].get_right(), m1.get_left(), buff=0.1, color=GREY_B, stroke_width=3),
            Arrow(leaves[1].get_right(), m1.get_left(), buff=0.1, color=GREY_B, stroke_width=3),
            Arrow(leaves[2].get_right(), m2.get_left(), buff=0.1, color=GREY_B, stroke_width=3),
            Arrow(leaves[3].get_right(), m2.get_left(), buff=0.1, color=GREY_B, stroke_width=3),
            Arrow(leaves[4].get_right(), add.get_bottom(), buff=0.1, color=GREY_B, stroke_width=3),
            edge(m1, add), edge(m2, add), edge(add, sg),
            Arrow(sg.get_right(), out.get_left(), buff=0.1, color=GREY_B, stroke_width=3))
        self.play(Write(title))
        self.play(*[FadeIn(l) for l in leaves], FadeIn(m1), FadeIn(m2), FadeIn(add), FadeIn(sg), FadeIn(out),
                  Create(edges), run_time=1.5)
        cap = swap_caption(self, None, "Forward (xanh): tính và LƯU mọi giá trị trung gian")
        fwd = VGroup(*[T(v, 22, C1).next_to(l, LEFT, buff=0.2) for v, l in zip(vals, leaves)])
        self.play(FadeIn(fwd))
        f_mid = [T("1.0", 22, C1).next_to(m1, UP, buff=0.1), T("−1.0", 22, C1).next_to(m2, UP, buff=0.1),
                 T("z = 0.2", 22, C1).next_to(add, UP, buff=0.1), T("0.5498", 22, C1).next_to(sg, UP, buff=0.1),
                 T("0.5498", 22, C1).next_to(out, UP, buff=0.15)]
        for t in f_mid:
            self.play(FadeIn(t, shift=RIGHT * 0.2), run_time=0.45)
        cap = swap_caption(self, cap, "Backward (đỏ): gradient từ trên × gradient cục bộ, đi ngược")
        b_out = T("1", 22, C2).next_to(out, DOWN, buff=0.15)
        b_sg = T("0.2475", 22, C2).next_to(sg, DOWN, buff=0.1)
        self.play(FadeIn(b_out))
        loc = T("σ′(z) = s(1 − s) = 0.2475", 22, C2).move_to([2.4, -1.3, 0])
        self.play(FadeIn(b_sg), FadeIn(loc))
        self.wait(0.5)
        b_add = T("0.2475", 22, C2).next_to(add, DR, buff=0.02)
        b_m1 = T("0.2475", 22, C2).next_to(m1, DOWN, buff=0.1)
        b_m2 = T("0.2475", 22, C2).next_to(m2, DOWN, buff=0.1)
        self.play(FadeIn(b_add))
        cap = swap_caption(self, cap, "Cổng cộng: chuyển nguyên gradient cho mọi đầu vào")
        self.play(FadeIn(b_m1), FadeIn(b_m2))
        self.wait(0.6)
        cap = swap_caption(self, cap, "Cổng nhân đổi chéo: ∂s/∂w₁ = 0.2475 × x₁,  ∂s/∂x₁ = 0.2475 × w₁")
        # Cột gradient nằm bên trái cột giá trị forward, cùng hàng với tên biến.
        bl = VGroup(*[T(gv, 20, C2).next_to(v, LEFT, buff=0.25) for gv, v in zip(grads, fwd)])
        hl = SurroundingRectangle(VGroup(leaves[0], leaves[1], fwd[0], fwd[1], bl[0], bl[1]), color=C_ACC, buff=0.12)
        self.play(FadeIn(bl), Create(hl), run_time=1.2)
        self.wait(1)
        cap = swap_caption(self, cap, "x lớn → gradient của w lớn: lý do phải chuẩn hóa đầu vào")
        self.wait(3)


class FanOutCongDon(Scene):
    """Biến dùng hai lần thì gradient phải cộng dồn: f = (x + y)·y tại x = 3, y = 2 (Ví dụ 2.7)."""

    def construct(self):
        title = T("Fan-out: gradient từ nhiều đường phải cộng lại", 30).to_edge(UP, buff=0.3)
        x = node("x", [-4.5, 1.2, 0])
        y = node("y", [-4.5, -1.2, 0])
        q = node("+", [-1.2, 1.2, 0])
        f = node("×", [2.2, 0.0, 0])
        out = T("f", 30).move_to([4.6, 0, 0])
        e_xq = Arrow(x.get_right(), q.get_left(), buff=0.05, color=GREY_B, stroke_width=3)
        e_yq = Arrow(y.get_right(), q.get_left() + DOWN * 0.15, buff=0.05, color=GREY_B, stroke_width=3)
        e_qf = Arrow(q.get_right(), f.get_left() + UP * 0.15, buff=0.05, color=GREY_B, stroke_width=3)
        e_yf = Arrow(y.get_right(), f.get_left() + DOWN * 0.15, buff=0.05, color=GREY_B, stroke_width=3)
        e_fo = Arrow(f.get_right(), out.get_left(), buff=0.1, color=GREY_B, stroke_width=3)
        with noi(self, "Khi một biến được dùng ở nhiều chỗ, gradient từ mọi nhánh phải được cộng dồn lại. "
                       "Xét f bằng x cộng y, rồi nhân với y."):
            self.play(Write(title))
            self.play(FadeIn(VGroup(x, y, q, f, out)), Create(VGroup(e_xq, e_yq, e_qf, e_yf, e_fo)))
        fv = VGroup(T("3", 22, C1).next_to(x, LEFT, buff=0.15), T("2", 22, C1).next_to(y, LEFT, buff=0.15),
                    T("q = 5", 22, C1).next_to(q, UP, buff=0.1), T("10", 22, C1).next_to(out, UP, buff=0.15))
        with noi(self, "Chạy forward tại x bằng ba, y bằng hai: q bằng năm, f bằng mười. "
                       "Để ý rằng y được dùng hai lần."):
            cap = swap_caption(self, None, "Forward: q = x + y = 5, f = q·y = 10. Biến y được dùng HAI lần")
            self.play(FadeIn(fv))
            self.wait(0.8)
        p1 = T("2", 24, C2).next_to(e_qf.get_center(), UP, buff=0.12)
        p2 = T("5", 24, C2).next_to(e_yf.get_center(), DOWN, buff=0.12)
        with noi(self, "Đi backward qua cổng nhân: đạo hàm theo q bằng y, tức là hai. "
                       "Còn nhánh nối thẳng về y nhận giá trị q, tức là năm."):
            cap = swap_caption(self, cap, "Cổng nhân: ∂f/∂q = y = 2, còn nhánh trực tiếp ∂f/∂y = q = 5")
            self.play(FadeIn(p1), FadeIn(p2))
        p3 = T("2", 24, C2).next_to(e_yq.get_center(), LEFT, buff=0.2)
        p4 = T("2", 24, C2).next_to(e_xq.get_center(), UP, buff=0.12)
        with noi(self, "Cổng cộng chuyển nguyên gradient hai về cả x và y."):
            cap = swap_caption(self, cap, "Cổng cộng chuyển nguyên 2 về cả x và y")
            self.play(FadeIn(p3), FadeIn(p4))
        total = T("∂f/∂y = 5 + 2 = 7", 28, C_ACC).next_to(y, DOWN, buff=0.35).shift(RIGHT * 1.2)
        gx = T("∂f/∂x = 2", 28, C_ACC).next_to(x, UP, buff=0.3).shift(RIGHT * 0.8)
        with noi(self, "Vậy y nhận hai phần đóng góp, năm và hai, tổng là bảy. Còn đạo hàm theo x bằng hai."):
            self.play(Indicate(p2, color=C_ACC), Indicate(p3, color=C_ACC))
            self.play(FadeIn(total), FadeIn(gx))
        wrong = T("y.grad = 2  (ghi đè)", 24, C2).next_to(total, RIGHT, buff=0.6)
        cross = Line(wrong.get_left(), wrong.get_right(), color=C2, stroke_width=4)
        with noi(self, "Nếu code ghi đè gradient thay vì cộng dồn, ta chỉ được hai: sai mà không hề báo lỗi. "
                       "Autograd của PyTorch tự cộng dồn các nhánh; lỗi này hay gặp khi tự viết backprop bằng tay."):
            cap = swap_caption(self, cap, "Viết y.grad = ... (ghi đè) thay vì += sẽ ra 2: sai mà không báo lỗi")
            self.play(FadeIn(wrong))
            self.play(Create(cross))
        self.wait(3)


# ---------------------------------------------------------------- 2.4
class ForwardVsReverse(Scene):
    """J 2×3 dựng bằng 3 JVP (cột) hoặc 2 VJP (hàng); rồi đếm chi phí chuỗi 10 lớp (Ví dụ 2.8)."""

    def construct(self):
        title = T("Forward mode lấy cột, reverse mode lấy hàng", 32).to_edge(UP, buff=0.3)
        sub = T("f(x₁, x₂, x₃) = (x₁x₂, x₂x₃) tại x = (1, 2, 3)", 24, GREY_A).next_to(title, DOWN, buff=0.15)
        Jv = [[2, 1, 0], [0, 3, 2]]
        cells = VGroup()
        for i in range(2):
            for j in range(3):
                sq = Square(side_length=0.9, stroke_width=2, color=GREY_B)
                sq.move_to([(j - 1) * 0.95, (0.5 - i) * 0.95, 0])
                cells.add(VGroup(sq, T(str(Jv[i][j]), 30).move_to(sq)))
        mat = cells.copy().shift(UP * 0.6)
        Jlab = T("J (2 × 3)", 24).next_to(mat, LEFT, buff=0.4)
        self.play(Write(title), FadeIn(sub))
        self.play(FadeIn(mat), FadeIn(Jlab))

        def cell(i, j):
            return mat[i * 3 + j]

        cap = swap_caption(self, None, "Forward mode (JVP): mỗi lượt đẩy một hướng v, nhận một cột Jv")
        cnt = None
        for j in range(3):
            box = SurroundingRectangle(VGroup(cell(0, j), cell(1, j)), color=C1, buff=0.05)
            new = T(f"JVP lượt {j + 1}/3: v = e{'₁₂₃'[j]} → cột ({Jv[0][j]}, {Jv[1][j]})", 26, C1)
            new.next_to(mat, DOWN, buff=0.4)
            if cnt is not None:
                self.play(FadeOut(cnt), run_time=0.2)
            self.play(Create(box), FadeIn(new), run_time=0.6)
            cnt = new
            self.wait(0.35)
            self.play(FadeOut(box), run_time=0.3)
        cap = swap_caption(self, cap, "Reverse mode (VJP): mỗi lượt đi ngược một vector u, nhận một hàng uᵀJ")
        for i in range(2):
            box = SurroundingRectangle(VGroup(*[cell(i, j) for j in range(3)]), color=C2, buff=0.05)
            new = T(f"VJP lượt {i + 1}/2: u = e{'₁₂'[i]} → hàng {tuple(Jv[i])}", 26, C2).next_to(mat, DOWN, buff=0.4)
            self.play(FadeOut(cnt), run_time=0.2)
            self.play(Create(box), FadeIn(new), run_time=0.6)
            cnt = new
            self.wait(0.35)
            self.play(FadeOut(box), run_time=0.3)
        cap = swap_caption(self, cap, "Cần n lượt forward hoặc m lượt reverse. Loss vô hướng: m = 1")
        self.wait(1)
        self.play(FadeOut(mat), FadeOut(Jlab), FadeOut(cnt), FadeOut(sub))
        head = T("Chuỗi 10 lớp 1000 × 1000 → loss vô hướng", 28).next_to(title, DOWN, buff=0.4)
        rev = T("Reverse: 10 tích vector × ma trận = 10 × 10⁶ = 10⁷ phép nhân", 26, C2)
        fwd = T("Dựng Jacobian từ đầu vào: 10 × 10⁹ = 10¹⁰ phép nhân", 26, C1)
        grp = VGroup(rev, fwd).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(head, DOWN, buff=0.6)
        bar_r = Rectangle(width=0.012, height=0.35, color=C2, fill_opacity=1, stroke_width=0)
        bar_f = Rectangle(width=12 * 0.9, height=0.35, color=C1, fill_opacity=1, stroke_width=0)
        bars = VGroup(bar_r, bar_f).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(grp, DOWN, buff=0.5)
        bars.to_edge(LEFT, buff=0.8)
        tag = T("← vạch đỏ gần như không thấy: rẻ hơn 1000 lần", 22, C2).next_to(bar_r, RIGHT, buff=0.2)
        self.play(FadeIn(head))
        self.play(FadeIn(rev), FadeIn(fwd))
        self.play(GrowFromEdge(bar_r, LEFT), GrowFromEdge(bar_f, LEFT), FadeIn(tag), run_time=1.5)
        cap = swap_caption(self, cap, "Backprop = nhân vector từ loss ngược về, không bao giờ dựng Jacobian")
        self.wait(2.2)


# ---------------------------------------------------------------- 2.5
class SoftmaxGradient(Scene):
    """Softmax + cross-entropy: ∂L/∂z = p − y; và vì sao phải trừ max (Ví dụ 2.9)."""

    def construct(self):
        title = T("Softmax + cross-entropy: gradient = p − y", 32).to_edge(UP, buff=0.3)
        self.play(Write(title))
        z = np.array([2.0, 1.0, 0.0])
        p = np.exp(z - z.max()); p /= p.sum()
        y = np.array([0.0, 1.0, 0.0])
        g = p - y
        base_y = -0.6
        groups = []
        specs = [("logit z", z, 0.6, BLUE, "{:.0f}"), ("p = softmax(z)", p, 2.4, C1, "{:.3f}"),
                 ("p − y", g, 2.4, C2, "{:+.3f}")]
        xs = [-4.4, 0.0, 4.4]
        for (name, vals, scale, col, fmt), cx in zip(specs, xs):
            grp = VGroup()
            baseline = Line([cx - 1.4, base_y, 0], [cx + 1.4, base_y, 0], color=GREY_B)
            grp.add(baseline)
            for k, v in enumerate(vals):
                bx = cx + (k - 1) * 0.9
                h = abs(v) * scale
                rect = Rectangle(width=0.6, height=max(h, 0.01), color=col, fill_opacity=0.8, stroke_width=0)
                if v >= 0:
                    rect.move_to([bx, base_y + h / 2, 0])
                    num = T(fmt.format(v), 20, col).next_to(rect, UP, buff=0.08)
                else:
                    rect.move_to([bx, base_y - h / 2, 0])
                    num = T(fmt.format(v), 20, col).next_to(rect, DOWN, buff=0.08)
                lab = T(f"lớp {k + 1}", 18, GREY_A).move_to([bx, base_y - 2.35, 0])
                grp.add(rect, num, lab)
            grp.add(T(name, 24, col).move_to([cx, 2.3, 0]))
            groups.append(grp)
        self.play(FadeIn(groups[0]))
        cap = swap_caption(self, None, "Logit (2, 1, 0); nhãn đúng là lớp 2")
        self.play(FadeIn(groups[1]))
        cap = swap_caption(self, cap, "p ≈ (0.665, 0.245, 0.090); loss = −log 0.245 ≈ 1.408")
        self.wait(0.8)
        self.play(FadeIn(groups[2]))
        cap = swap_caption(self, cap, "Âm ở lớp đúng (đẩy logit lên), dương ở lớp sai (kéo xuống)")
        self.wait(1.5)
        self.play(*[FadeOut(gp) for gp in groups], FadeOut(cap))
        l1 = T("z = (1000, 999, 998)", 28).shift(UP * 1.4)
        l2 = T("np.exp(z) → inf,  inf / inf → nan", 28, C2).next_to(l1, DOWN, buff=0.5)
        l3 = T("trừ max: z − 1000 = (0, −1, −2)", 28, C1).next_to(l2, DOWN, buff=0.5)
        l4 = T("p = (0.665, 0.245, 0.090) — đúng như trên", 28, C1).next_to(l3, DOWN, buff=0.4)
        self.play(FadeIn(l1))
        self.play(FadeIn(l2))
        self.wait(0.6)
        self.play(FadeIn(l3))
        self.play(FadeIn(l4))
        self.play(FadeIn(caption("Cộng cùng một hằng số vào mọi logit không đổi p: trừ max là miễn phí")))
        self.wait(3)
