"""Các cảnh Manim cho Chương 4 — Tối ưu hóa.

Render một cảnh (chất lượng 720p):
    manim -qm chuong04_scenes.py TocDoTungHuongRieng

Chỉ dùng Text (không cần LaTeX), nên chỉ cần Manim Community + ffmpeg.
Mỗi cảnh dùng đúng số liệu của ví dụ giải mẫu trong sách (bài toán chạy xuyên chương
f = ½(x₁² + 100x₂²), điểm đầu (1, 1)).
"""
import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parent))
from giong_doc import noi  # giọng đọc tiếng Việt; MANIM_KHONG_GIONG=1 để render câm
import nen

nen.bat()  # nền lưới mờ thay cho nền đen trơn

FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"
C1, C2, C_ACC = GREEN, RED, YELLOW


def T(s, size=28, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


def caption(s, size=26):
    t = T(s, size)
    return VGroup(BackgroundRectangle(t, fill_opacity=0.85, buff=0.12), t).to_edge(DOWN, buff=0.25)


def tick_labels(ax, xs=(), ys=(), size=18, fmt="{:g}"):
    """Nhãn số trên trục bằng Text (không cần LaTeX như include_numbers)."""
    g = VGroup()
    for x in xs:
        g.add(T(fmt.format(x), size, GREY_B).next_to(ax.x_axis.n2p(x), DOWN, buff=0.12))
    for y in ys:
        g.add(T(fmt.format(y), size, GREY_B).next_to(ax.y_axis.n2p(y), LEFT, buff=0.12))
    return g


def swap_caption(scene, old, text, size=26):
    new = caption(text, size)
    if old is None:
        scene.play(FadeIn(new))
    else:
        scene.play(FadeOut(old), FadeIn(new), run_time=0.6)
    return new


# ---------------------------------------------------------------- 4.1
class DayCungHamLoi(Scene):
    """Hàm lồi: dây cung nằm trên đồ thị, tiếp tuyến nằm dưới. Hàm hai giếng x⁴/4 − x² thì không."""

    def construct(self):
        title = T("Hàm lồi: dây cung luôn nằm trên đồ thị", 32).to_edge(UP, buff=0.3)
        with noi(self, "Hàm lồi có một tính chất hình học đơn giản: dây cung nối hai điểm bất kỳ luôn nằm trên đồ thị. So sánh x bình phương chia hai, là hàm lồi, với hàm hai giếng bên phải."):
            self.play(Write(title))
            f = lambda x: 0.5 * x ** 2
            g = lambda x: x ** 4 / 4 - x ** 2
            axL = Axes(x_range=[-2.2, 2.2, 1], y_range=[-1.5, 2.5, 1], x_length=5.6, y_length=4.4,
                       tips=False, axis_config={"stroke_opacity": 0.5}).move_to(LEFT * 3.4 + DOWN * 0.35)
            axR = axL.copy().move_to(RIGHT * 3.4 + DOWN * 0.35)
            cf = axL.plot(f, x_range=[-2.2, 2.2], color=BLUE)
            cg = axR.plot(g, x_range=[-2.2, 2.2], color=BLUE)
            labL = T("f(x) = x²/2  (lồi)", 24).next_to(axL, UP, buff=0.1)
            labR = T("g(x) = x⁴/4 − x²  (không lồi)", 24).next_to(axR, UP, buff=0.1)
            self.play(Create(axL), Create(axR), Create(cf), Create(cg), FadeIn(labL), FadeIn(labR))

        with noi(self, "Kéo hai đầu dây cung ra xa. Bên trái, dây cung luôn nằm trên đồ thị."):
            a, b = ValueTracker(-1.8), ValueTracker(-0.6)

            def chord(ax, h, col):
                return Line(ax.c2p(a.get_value(), h(a.get_value())), ax.c2p(b.get_value(), h(b.get_value())),
                            color=col, stroke_width=4)

            def above(h):
                xs = np.linspace(a.get_value(), b.get_value(), 40)[1:-1]
                ya, yb = h(a.get_value()), h(b.get_value())
                line = ya + (xs - a.get_value()) / (b.get_value() - a.get_value()) * (yb - ya)
                return np.all(h(xs) <= line + 1e-9)

            chL = always_redraw(lambda: chord(axL, f, C_ACC))
            chR = always_redraw(lambda: chord(axR, g, C_ACC if above(g) else C2))
            dots = always_redraw(lambda: VGroup(*[Dot(ax.c2p(t.get_value(), h(t.get_value())), color=WHITE,
                                                      radius=0.06)
                                                  for ax, h in ((axL, f), (axR, g)) for t in (a, b)]))
            self.play(Create(chL), Create(chR), FadeIn(dots))
            cap = swap_caption(self, None, "Kéo hai đầu dây cung: bên trái luôn nằm trên đồ thị")
            self.play(a.animate.set_value(-1.6), b.animate.set_value(1.6), run_time=3)
        with noi(self, "Bên phải thì không: tại x bằng không, đồ thị nằm trên dây cung, nên hàm này không lồi."):
            cap = swap_caption(self, cap, "Bên phải: g(0) = 0 nằm TRÊN dây cung có độ cao −0.92 (đỏ)")
            self.wait(1.5)
        with noi(self, "Một cách nhìn tương đương: với hàm lồi, tiếp tuyến tại bất kỳ điểm nào cũng nằm dưới đồ thị. Nên điểm có gradient bằng không chắc chắn là cực tiểu toàn cục."):
            tan = axL.plot(lambda x: 0.8 * x - 0.32, x_range=[-1.2, 2.2], color=C1)
            tdot = Dot(axL.c2p(0.8, f(0.8)), color=C1)
            cap = swap_caption(self, cap, "Tiếp tuyến tại 0.8 nằm dưới đồ thị: f(y) ≥ f(x) + f′(x)(y − x)")
            self.play(Create(tan), FadeIn(tdot))
        self.wait(2.5)


class KKTTiepXuc(Scene):
    """KKT (ví dụ 4.2): min (x − 2)² + (y − 1)² với x + y ≤ 2 → (1.5, 0.5), μ = 1; nới thành ≤ 4 → μ = 0."""

    def construct(self):
        title = T("KKT: đường mức chạm biên ràng buộc", 32).to_edge(UP, buff=0.3)
        ax = Axes(x_range=[-0.5, 4.5, 1], y_range=[-1, 3, 1], x_length=6.25, y_length=5,
                  tips=False, axis_config={"stroke_opacity": 0.6}).move_to(LEFT * 3.3 + DOWN * 0.3)
        with noi(self, "Bài toán: tìm điểm gần điểm hai, một nhất, với ràng buộc x cộng y nhỏ hơn hoặc bằng hai. Cực tiểu tự do hai, một vi phạm ràng buộc."):
            self.play(Write(title), Create(ax), FadeIn(tick_labels(ax, xs=(0, 1, 2, 3, 4), ys=(-1, 1, 2, 3))))
            rhs = ValueTracker(2.0)

            def region():
                r = rhs.get_value()
                box = [(-0.5, -1), (4.5, -1), (4.5, 3), (-0.5, 3)]
                pts = []                                   # cắt hình chữ nhật bởi nửa mặt phẳng x + y ≤ r
                for i in range(4):
                    P, Q = np.array(box[i]), np.array(box[(i + 1) % 4])
                    inP, inQ = P.sum() <= r, Q.sum() <= r
                    if inP:
                        pts.append(tuple(P))
                    if inP != inQ:
                        tt = (r - P.sum()) / (Q.sum() - P.sum())
                        pts.append(tuple(P + tt * (Q - P)))
                return Polygon(*[ax.c2p(*p) for p in pts], stroke_width=0, fill_color=BLUE, fill_opacity=0.18)

            def boundary():
                r = rhs.get_value()
                x0, x1 = max(-0.5, r - 3), min(4.5, r + 1)
                return Line(ax.c2p(x0, r - x0), ax.c2p(x1, r - x1), color=BLUE, stroke_width=4)

            reg, bnd = always_redraw(region), always_redraw(boundary)
            info = VGroup(T("min (x − 2)² + (y − 1)²", 24), T("với x + y ≤ 2", 24, BLUE)).arrange(DOWN, aligned_edge=LEFT)
            info.move_to([1.0, 2.2, 0], aligned_edge=LEFT)
            center = Dot(ax.c2p(2, 1), color=C_ACC)
            _t = T("(2, 1)", 22, C_ACC).next_to(center, UP, buff=0.15)
            clab = VGroup(BackgroundRectangle(_t, fill_opacity=0.9, buff=0.06), _t)   # nền để đường mức không cắt chữ
            self.play(FadeIn(reg), Create(bnd), FadeIn(info), FadeIn(center), FadeIn(clab))
            cap = swap_caption(self, None, "Cực tiểu tự do (2, 1) vi phạm ràng buộc: 2 + 1 = 3 > 2")
            self.wait(1)
        with noi(self, "Co dần đường mức quanh điểm hai, một, cho tới khi nó chỉ còn chạm miền khả thi."):
            unit = ax.x_length / 5
            rad = ValueTracker(1.6)
            circ = always_redraw(lambda: Circle(radius=rad.get_value() * unit, color=C_ACC, stroke_width=3)
                                 .move_to(ax.c2p(2, 1)))
            self.play(Create(circ))
            cap = swap_caption(self, cap, "Co đường mức cho tới khi chỉ còn chạm miền khả thi")
            self.play(rad.animate.set_value(1 / np.sqrt(2)), run_time=2.5)
        with noi(self, "Điểm chạm là một phẩy năm, không phẩy năm. Tại đó gradient của f và gradient của ràng buộc ngược chiều nhau, với nhân tử mu bằng một, dương: ràng buộc đang có hiệu lực."):
            p = ax.c2p(1.5, 0.5)
            sol = Dot(p, color=WHITE)
            slab = T("(1.5, 0.5)", 22).next_to(sol, LEFT, buff=0.15).shift(UP * 0.15)
            gf = Arrow(p, ax.c2p(1.0, 0.0), buff=0, color=C2, stroke_width=5)
            gg = Arrow(p, ax.c2p(1.85, 0.85), buff=0, color=C1, stroke_width=5)
            res = VGroup(T("∇f = (−1, −1)", 24, C2), T("∇g = (1, 1)", 24, C1),
                         T("∇f + μ∇g = 0 ⇒ μ = 1 > 0", 24, C_ACC), T("(g = x + y − 2)", 20, GREY_B)).arrange(DOWN, aligned_edge=LEFT)
            res.next_to(info, DOWN, buff=0.5, aligned_edge=LEFT)
            self.play(FadeIn(sol), FadeIn(slab), GrowArrow(gf), GrowArrow(gg), FadeIn(res))
            cap = swap_caption(self, cap, "Tại nghiệm, ∇f và ∇g ngược chiều: ràng buộc đang chạm")
            self.wait(2)
        with noi(self, "Nếu nới ràng buộc thành x cộng y nhỏ hơn bốn, điểm hai, một trở nên khả thi. Ràng buộc không chạm, nên điều kiện bù cho mu bằng không."):
            info2 = VGroup(T("min (x − 2)² + (y − 1)²", 24), T("với x + y ≤ 4", 24, BLUE)).arrange(DOWN, aligned_edge=LEFT)
            info2.move_to(info, aligned_edge=LEFT)
            res2 = T("ràng buộc không chạm ⇒ μ = 0", 24, C_ACC).next_to(info2, DOWN, buff=0.5, aligned_edge=LEFT)
            self.play(FadeOut(gf), FadeOut(gg), FadeOut(sol), FadeOut(slab), FadeOut(res), FadeOut(circ),
                      Transform(info, info2))
            self.play(rhs.animate.set_value(4.0), run_time=2)
            self.play(FadeIn(res2), center.animate.scale(1.6))
            cap = swap_caption(self, cap, "Nới ràng buộc: (2, 1) khả thi, điều kiện bù μ·g = 0 cho μ = 0")
        self.wait(2.5)


# ---------------------------------------------------------------- 4.2
class TocDoTungHuongRieng(Scene):
    """GD trên f = ½(x₁² + 100x₂²): mỗi hướng riêng nhân với (1 − ηλ) mỗi bước (ví dụ 4.4)."""

    def construct(self):
        title = T("GD nhìn theo từng hướng riêng: sai số × (1 − ηλ) mỗi bước", 28).to_edge(UP, buff=0.3)
        # trục tung vẽ log₁₀|sai số| + 6 để trục hoành nằm ở đáy (−6)
        ax = Axes(x_range=[0, 1400, 200], y_range=[0, 7, 1], x_length=7.6, y_length=4.6, tips=False,
                  axis_config={"stroke_opacity": 0.6}).move_to(LEFT * 2.4 + DOWN * 0.3)
        ticks = VGroup(*[T(f"{x}", 16, GREY_B).next_to(ax.c2p(x, 0), DOWN, buff=0.1) for x in range(0, 1401, 400)],
                       *[T(f"{y - 6}", 16, GREY_B).next_to(ax.c2p(0, y), LEFT, buff=0.1) for y in (0, 2, 4, 6)])
        xl = T("số bước k", 20).next_to(ax.c2p(1400, 0), RIGHT, buff=0.2)
        yl = T("log₁₀ |sai số|", 20).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.45)
        with noi(self, "Gradient descent trên hàm bậc hai có thể tách theo từng hướng riêng. Mỗi bước, sai số theo hướng có trị riêng lambda được nhân với một trừ eta lambda. Ở đây hai trị riêng là một và một trăm."):
            self.play(Write(title), Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(ticks))

            def curve(r, col):
                if r == 0:
                    return VGroup(ax.plot_line_graph([0, 1], [6, 0], add_vertex_dots=False,
                                                     line_color=col, stroke_width=5),
                                  Dot(ax.c2p(1, 0), color=col, radius=0.08))
                kmax = min(1400, 6 / -np.log10(r)) if r < 1 else min(1400, 1 / np.log10(r))   # dừng vẽ khi chạm ngưỡng 10⁻⁶
                ks = np.linspace(0, kmax, 300)
                ys = np.clip(ks * np.log10(r), -6, 1) + 6
                return ax.plot_line_graph(ks, ys, add_vertex_dots=False, line_color=col, stroke_width=5)

            legend = VGroup(T("hướng λ = 1", 22, C1), T("hướng λ = 100", 22, C2)).arrange(DOWN, aligned_edge=LEFT)
            legend.move_to([2.4, 2.2, 0], aligned_edge=LEFT)
            self.play(FadeIn(legend))
        cases = [(0.01, "η = 1/L = 0.01", "1 341 bước",
                  "Hướng dốc xong sau 1 bước; hướng thoải nhân 0.99 mỗi bước"),
                 (2 / 101, "η = 2/(L + μ) ≈ 0.0198", "691 bước",
                  "Cân bằng hai hướng: cả hai cùng nhân 0.980 — chỉ nhanh gấp 2"),
                 (0.0201, "η = 0.0201 > 2/L", "phân kỳ",
                  "|1 − 0.0201·100| = 1.01 > 1: hướng dốc lớn dần, đổi dấu mỗi bước")]
        cur, info, cap = None, None, None
        loi = ["Với learning rate một chia L, bằng không phẩy không một, hướng dốc hội tụ chỉ sau một bước. "
               "Nhưng hướng thoải mỗi bước chỉ nhân không phẩy chín chín, mất một nghìn ba trăm bốn mốt bước.",
               "Learning rate tối ưu, hai chia L cộng mu, cân bằng hai hướng: cả hai cùng nhân không phẩy chín tám. "
               "Cũng chỉ nhanh gấp đôi, sáu trăm chín mốt bước.",
               "Vượt quá hai chia L một chút thôi, hệ số của hướng dốc lớn hơn một. "
               "Sai số đổi dấu và lớn dần mỗi bước: thuật toán phân kỳ."]
        for (eta, name, steps, msg), loi_doc in zip(cases, loi):
            r1, r2 = abs(1 - eta), abs(1 - 100 * eta)
            if r2 < 1e-12:
                r2 = 0
            g = VGroup(curve(r1, C1), curve(r2, C2))
            new_info = VGroup(T(name, 22, C_ACC),
                              T(f"|1 − η·1| = {r1:.4f}", 20, C1),
                              T(f"|1 − η·100| = {r2:.4f}", 20, C2),
                              T(f"tới 10⁻⁶: {steps}", 22)).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
            new_info.next_to(legend, DOWN, buff=0.5, aligned_edge=LEFT)
            if cur is not None:
                self.play(FadeOut(cur), FadeOut(info), run_time=0.4)
            with noi(self, loi_doc):
                self.play(Create(g), FadeIn(new_info), run_time=2)
                cur, info = g, new_info
                cap = swap_caption(self, cap, msg, 24)
                self.wait(2.2)


# ---------------------------------------------------------------- 4.3
class SanNhieuSGD(Scene):
    """Sàn nhiễu của SGD (ví dụ 4.6): phương sai dừng ησ²/(B(2 − η)) với σ² = 4."""

    def construct(self):
        title = T("SGD với η hằng: dừng trong một vùng nhiễu, không về đúng 0", 28).to_edge(UP, buff=0.3)
        ax = Axes(x_range=[-1.6, 1.6, 0.5], y_range=[0, 1.4, 0.5], x_length=7.6, y_length=3.6, tips=False,
                  axis_config={"stroke_opacity": 0.6}).shift(DOWN * 0.1 + LEFT * 2.2)
        para = ax.plot(lambda w: 0.5 * w * w, x_range=[-1.6, 1.6], color=BLUE)
        wl = T("w", 22).next_to(ax.x_axis, RIGHT, buff=0.1)
        with noi(self, "SGD với learning rate cố định không hội tụ về đúng cực tiểu, mà dừng lại trong một vùng nhiễu."):
            self.play(Write(title), Create(ax), Create(para), FadeIn(wl))
        rng = np.random.default_rng(0)
        N = 400          # chấm vẽ; phương sai đo trên 4 000 chuỗi như lab 2a
        jitter = rng.uniform(0.04, 0.32, N)

        def samples(eta, B):
            """Đúng quy trình lab 2a: 4 000 chuỗi, 3 000 bước, seed 1, phương sai gộp 500 bước cuối."""
            r = np.random.default_rng(1)
            w, tail = np.zeros(4000), []
            for t in range(3000):
                w = w - eta * (w + r.normal(0, np.sqrt(4.0 / B), 4000))
                if t >= 3000 - 500:
                    tail.append(w.copy())
            return np.clip(w[:N], -1.58, 1.58), float(np.var(np.concatenate(tail)))

        cases = [(0.1, 1, "η = 0.1, B = 1", "0.2105"), (0.1, 16, "η = 0.1, B = 16", "0.0132"),
                 (0.01, 1, "η = 0.01, B = 1", "0.0201")]
        dots, band, info, cap = None, None, None, None
        msgs = ["4 000 chuỗi độc lập sau 3 000 bước (vẽ 400): vẫn rải quanh cực tiểu",
                "Tăng B lên 16: phương sai giảm đúng 16 lần",
                "Hoặc giảm η 10 lần: phương sai giảm ≈ 10 lần"]
        loi = ["Với learning rate không phẩy một và batch một, bốn nghìn chuỗi SGD sau ba nghìn bước vẫn rải quanh cực tiểu, "
               "phương sai khoảng không phẩy hai mốt.",
               "Tăng batch lên mười sáu, phương sai giảm đúng mười sáu lần.",
               "Hoặc giảm learning rate mười lần, phương sai cũng giảm khoảng mười lần."]
        for (eta, B, name, var), msg, loi_doc in zip(cases, msgs, loi):
            s, meas = samples(eta, B)
            th = eta * 4 / (B * (2 - eta))
            new_dots = VGroup(*[Dot(ax.c2p(x, y), radius=0.035, color=C_ACC) for x, y in zip(s, jitter)])
            sd = np.sqrt(th)
            new_band = Rectangle(width=2 * sd * ax.x_length / 3.2, height=0.36 * ax.y_length / 1.4,
                                 stroke_width=0, fill_color=C2, fill_opacity=0.18).move_to(ax.c2p(0, 0.18))
            new_info = VGroup(T(name, 26, C_ACC), T(f"phương sai lý thuyết {var}", 24),
                              T(f"đo được {meas:.4f}", 24)).arrange(DOWN, aligned_edge=LEFT)
            new_info.to_corner(UR, buff=0.4).shift(DOWN * 0.8)
            with noi(self, loi_doc):
                if dots is None:
                    self.play(FadeIn(new_band), FadeIn(new_dots), FadeIn(new_info))
                else:
                    self.play(Transform(dots, new_dots), Transform(band, new_band), Transform(info, new_info),
                              run_time=1.5)
                    new_dots, new_band, new_info = dots, band, info
                dots, band, info = new_dots, new_band, new_info
                cap = swap_caption(self, cap, msg, 24)
                self.wait(2)
        with noi(self, "Phương sai tỉ lệ với eta chia B. Đó là lý do cuối quá trình huấn luyện ta phải giảm learning rate."):
            cap = swap_caption(self, cap, "Phương sai ∝ η/B, bán kính ∝ √(η/B): vì vậy cuối huấn luyện phải giảm learning rate", 22)
        self.wait(2.5)


# ---------------------------------------------------------------- 4.4
class MomentumBongLan(Scene):
    """GD tối ưu (691 bước) và heavy ball tối ưu (93 bước) trên cùng bài κ = 100 (ví dụ 4.8).

    Vẽ từng tọa độ x₁(k), x₂(k) theo số bước thay vì quỹ đạo 2D: heavy ball làm x₂ vọt tới ≈ 3.7
    ở vài bước đầu, khung 2D cùng tỉ lệ sẽ không chứa nổi.
    """

    def construct(self):
        title = T("Bài bậc hai κ = 100: GD tối ưu và momentum tối ưu", 30).to_edge(UP, buff=0.3)
        with noi(self, "Cùng bài toán bậc hai có số điều kiện một trăm, so sánh gradient descent tối ưu với momentum tối ưu."):
            self.play(Write(title))
        lam = np.array([1.0, 100.0])

        def run(eta, beta, K):
            x, v, xs = np.array([1.0, 1.0]), np.zeros(2), []
            xs.append(x.copy())
            for _ in range(K):
                v = beta * v - eta * lam * x
                x = x + v
                xs.append(x.copy())
            return np.array(xs)

        K = 100
        paths = [run(2 / 101, 0.0, K), run(4 / 121, (9 / 11) ** 2, K)]
        names = ["GD: η = 2/101", "Heavy ball: η = 4/121, β = (9/11)²"]
        totals = ["691", "93"]
        axes, groups = [], []
        for i, c in enumerate([LEFT * 3.45, RIGHT * 3.45]):
            ax = Axes(x_range=[0, 100, 20], y_range=[-4, 4, 1], x_length=5.6, y_length=3.9, tips=False,
                      axis_config={"stroke_opacity": 0.5}).move_to(c + DOWN * 0.35)
            ticks = VGroup(*[T(f"{x}", 16, GREY_B).next_to(ax.c2p(x, -4), DOWN, buff=0.08) for x in (0, 50, 100)],
                           *[T(f"{y}", 16, GREY_B).next_to(ax.c2p(0, y), LEFT, buff=0.08) for y in (-4, 0, 4)])
            lab = T(names[i], 22).next_to(ax, UP, buff=0.15)
            axes.append(ax)
            groups.append(VGroup(ax, ticks, lab))
        legend = VGroup(T("x₁ (λ = 1)", 22, C1), T("x₂ (λ = 100)", 22, C2)).arrange(RIGHT, buff=0.6)
        legend.next_to(title, DOWN, buff=0.2)
        self.play(*[FadeIn(g) for g in groups], FadeIn(legend))
        k = ValueTracker(1)

        def path_mob(i):
            n = max(1, int(k.get_value()))
            ks = np.arange(n + 1)
            return VGroup(*[axes[i].plot_line_graph(ks, paths[i][: n + 1, j], add_vertex_dots=False,
                                                    line_color=col, stroke_width=3)
                            for j, col in ((1, C2), (0, C1))])

        def counter(i):
            n = int(k.get_value())
            nrm = np.linalg.norm(paths[i][n])
            return T(f"k = {n:3d}    ‖x‖ = {nrm:.2g}", 22, C_ACC).next_to(axes[i], DOWN, buff=0.35)

        p0, p1 = always_redraw(lambda: path_mob(0)), always_redraw(lambda: path_mob(1))
        c0, c1 = always_redraw(lambda: counter(0)), always_redraw(lambda: counter(1))
        self.add(p0, p1, c0, c1)
        with noi(self, "Với gradient descent, tọa độ x hai đổi dấu mỗi bước, còn x một mỗi bước chỉ nhân không phẩy chín tám, giảm rất chậm."):
            cap = swap_caption(self, None, "GD: x₂ đổi dấu mỗi bước, x₁ chỉ nhân 0.98 mỗi bước", 24)
            self.play(k.animate.set_value(30), run_time=4, rate_func=linear)
        with noi(self, "Với momentum, x hai vọt lên tới ba phẩy bảy rồi tắt nhanh, còn x một được vận tốc kéo về không. Momentum cần chín mươi ba bước, gradient descent cần sáu trăm chín mốt bước."):
            cap = swap_caption(self, cap, "Momentum: x₂ vọt tới 3.7 rồi tắt nhanh; x₁ được vận tốc kéo về 0", 24)
            self.play(k.animate.set_value(K), run_time=4, rate_func=linear)
            res = VGroup(T(f"tới 10⁻⁶: {totals[0]} bước", 24, C_ACC), T(f"tới 10⁻⁶: {totals[1]} bước", 24, C_ACC))
            for r, ax in zip(res, axes):
                r.move_to(ax.c2p(62, 2.6))
            self.play(FadeIn(res))
        with noi(self, "Về lý thuyết, hệ số hội tụ giảm từ không phẩy chín tám xuống không phẩy tám một tám, nhờ căn kappa thay cho kappa."):
            cap = swap_caption(self, cap, "Tốc độ (κ − 1)/(κ + 1) = 0.980 so với (√κ − 1)/(√κ + 1) = 0.818", 24)
            self.wait(2.5)
        with noi(self, "Lý thuyết dự đoán sáu mươi chín bước, đo được chín mươi ba, vì trị riêng kép làm sai số giảm chậm hơn một chút."):
            cap = swap_caption(self, cap, "Lý thuyết 69 bước, đo được 93: trị riêng kép, sai số cỡ k·0.818ᵏ", 22)
            self.wait(2.5)


class AdamChuanHoaBuoc(Scene):
    """Adam: bước đầu ≈ α bất kể độ lớn gradient (ví dụ 4.9); AdamW và L2 khác nhau (ví dụ 4.10)."""

    def construct(self):
        title = T("Adam chuẩn hóa độ lớn bước theo từng tọa độ", 30).to_edge(UP, buff=0.3)
        with noi(self, "Adam chuẩn hóa độ lớn bước theo từng tọa độ. Xét ba gradient chênh nhau rất xa, từ mười mũ trừ bốn tới một nghìn."):
            self.play(Write(title))
            rows = [("g = 10⁻⁴", "10⁻⁷", 0.0009999), ("g = 1", "10⁻³", 0.0010000), ("g = 10³", "1", 0.0010000)]
            head = VGroup(T("gradient", 24, GREY_B), T("bước SGD, η = 10⁻³", 22, GREY_B),
                          T("bước đầu của Adam, α = 10⁻³", 22, GREY_B))
            xs = [-5.2, -2.0, 3.0]
            for h, x in zip(head, xs):
                h.move_to([x, 2.2, 0])
            self.play(FadeIn(head))
            table = VGroup()
            for i, (g, sgd, adam) in enumerate(rows):
                y = 1.2 - 1.0 * i
                a = T(g, 26).move_to([xs[0], y, 0])
                b = T(sgd, 26, C2).move_to([xs[1], y, 0])
                bar = Rectangle(width=2.6, height=0.32, fill_color=C1, fill_opacity=0.8, stroke_width=0)
                bar.move_to([xs[2] - 0.7, y, 0])
                c = T(f"{adam:.7f}", 22, C1).next_to(bar, RIGHT, buff=0.2)
                table.add(VGroup(a, b, bar, c))
            for r in table:
                self.play(FadeIn(r[0]), FadeIn(r[1]), run_time=0.5)
        with noi(self, "Với SGD, bước đi tỉ lệ với gradient, nên chênh nhau tới mười triệu lần."):
            cap = swap_caption(self, None, "SGD: bước chênh nhau 10⁷ lần, theo đúng độ lớn gradient", 24)
            self.wait(1)
        with noi(self, "Với Adam, ở bước đầu m mũ chia căn v mũ xấp xỉ dấu của gradient, nên mọi bước đều xấp xỉ alpha, không phụ thuộc độ lớn gradient."):
            self.play(*[GrowFromEdge(r[2], LEFT) for r in table], *[FadeIn(r[3]) for r in table])
            cap = swap_caption(self, cap, "Adam: m̂/√v̂ ≈ g/|g| ở bước đầu, nên mọi bước đều ≈ α", 24)
            self.wait(2)
        self.play(FadeOut(table), FadeOut(head), FadeOut(cap))
        # --- phần 2: AdamW và L2
        with noi(self, "Hệ quả: weight decay trong Adam phải tách riêng. Xét một tham số có gradient loss bằng không. Với AdamW, w giảm rất nhẹ, còn khoảng không phẩy chín chín. Với L hai cộng vào gradient, w tụt xuống khoảng không phẩy hai sáu."):
            t2 = T("Tham số có gradient loss = 0, λ = 0.01, α = 10⁻³", 26).next_to(title, DOWN, buff=0.25)
            ax = Axes(x_range=[0, 1000, 200], y_range=[0, 1.05, 0.25], x_length=8.5, y_length=4.0, tips=False,
                      axis_config={"stroke_opacity": 0.6}).shift(DOWN * 0.75 + LEFT * 1.3)
            wl = T("w", 22).next_to(ax.y_axis, UP, buff=0.1)
            ticks = tick_labels(ax, xs=(0, 200, 400, 600, 800, 1000), ys=(0, 0.25, 0.5, 0.75, 1.0))
            self.play(FadeIn(t2), Create(ax), FadeIn(wl), FadeIn(ticks))

            def run(decoupled):
                w, m, v, out = 1.0, 0.0, 0.0, [1.0]
                for t in range(1, 1001):
                    g = 0.0 if decoupled else 0.01 * w
                    m = 0.9 * m + 0.1 * g
                    v = 0.999 * v + 0.001 * g * g
                    if decoupled:
                        w -= 1e-3 * 0.01 * w
                    w -= 1e-3 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
                    out.append(w)
                return np.array(out)

            ks = np.arange(1001)
            wW, wL = run(True), run(False)
            gW = ax.plot_line_graph(ks[::10], wW[::10], add_vertex_dots=False, line_color=C1, stroke_width=5)
            gL = ax.plot_line_graph(ks[::10], wL[::10], add_vertex_dots=False, line_color=C2, stroke_width=5)
            lW = T(f"AdamW: {wW[-1]:.4f}", 24, C1).next_to(ax.c2p(1000, wW[-1]), RIGHT, buff=0.15)
            lL = T(f"L2 trong Adam: {wL[-1]:.3f}", 24, C2).next_to(ax.c2p(1000, wL[-1]), RIGHT, buff=0.15)
            self.play(Create(gW), Create(gL), run_time=3)
            self.play(FadeIn(lW), FadeIn(lL))
        with noi(self, "Vì số hạng L hai bị chia cho căn v mũ của chính nó, nên decay mạnh hơn khoảng bảy mươi tư lần và không còn do lambda quyết định. Đó là lý do nên dùng AdamW."):
            cap = swap_caption(self, None, "L2 bị chia cho √v̂ của chính nó: decay mạnh gấp ~74 lần, không còn do λ quyết định", 22)
        self.wait(3)


# ---------------------------------------------------------------- 4.5
class NewtonYenNgua(Scene):
    """f = x² − y² (ví dụ 4.13): Newton nhảy thẳng vào yên ngựa, GD thoát theo hướng y."""

    def construct(self):
        title = T("Newton tìm điểm dừng, không phân biệt yên ngựa", 30).to_edge(UP, buff=0.3)
        ax = Axes(x_range=[-1.5, 1.5, 0.5], y_range=[-1.5, 1.5, 0.5], x_length=5.2, y_length=5.2, tips=False,
                  axis_config={"stroke_opacity": 0.4}).move_to(LEFT * 3.4 + DOWN * 0.4)
        levels = VGroup()
        for c, col in [(0.25, BLUE_D), (0.75, BLUE_D), (1.4, BLUE_D)]:
            for sgn in (1, -1):
                # x² − y² = c: x = ±√(c + y²);  x² − y² = −c: y = ±√(c + x²)
                levels.add(ParametricFunction(lambda t, c=c, s=sgn: ax.c2p(s * np.sqrt(c + t * t), t),
                                              t_range=[-np.sqrt(2.25 - c), np.sqrt(2.25 - c)], color=col, stroke_width=2))
                levels.add(ParametricFunction(lambda t, c=c, s=sgn: ax.c2p(t, s * np.sqrt(c + t * t)),
                                              t_range=[-np.sqrt(2.25 - c), np.sqrt(2.25 - c)], color=C2, stroke_width=1.5,
                                              stroke_opacity=0.6))
        with noi(self, "Hàm x bình phương trừ y bình phương có một điểm yên ngựa ở gốc. Hessian có hai trị riêng trái dấu, hai và trừ hai."):
            self.play(Write(title), Create(ax), Create(levels), run_time=2)
            saddle = Dot(ax.c2p(0, 0), color=WHITE)
            info = VGroup(T("f = x² − y²", 26), T("H = diag(2, −2)", 24),
                          T("trị riêng trái dấu ⇒ yên ngựa", 24)).arrange(DOWN, aligned_edge=LEFT)
            info.move_to([0.6, 2.0, 0], aligned_edge=LEFT)
            self.play(FadeIn(saddle), FadeIn(info))
        with noi(self, "Phương pháp Newton từ điểm một, không phẩy năm nhảy đúng vào gốc chỉ sau một bước. Newton tìm điểm có gradient bằng không, dù đó không phải cực tiểu."):
            start = Dot(ax.c2p(1, 0.5), color=C_ACC)
            jump = Arrow(ax.c2p(1, 0.5), ax.c2p(0, 0), buff=0.05, color=C_ACC, stroke_width=5)
            n1 = T("Newton từ (1, 0.5): 1 bước tới (0, 0)", 21, C_ACC).next_to(info, DOWN, buff=0.5, aligned_edge=LEFT)
            self.play(FadeIn(start), FadeIn(n1))
            self.play(GrowArrow(jump))
            cap = swap_caption(self, None, "Bước −H⁻¹∇f nhảy đúng vào điểm có ∇f = 0, dù đó không phải cực tiểu", 22)
            self.wait(1.5)
        with noi(self, "Còn gradient descent, xuất phát gần trục x, thì tọa độ y nhân một phẩy hai mỗi bước, và thoát ra ngoài sau ba mươi tám bước."):
            p = np.array([1.0, 1e-3])
            pts = [p.copy()]
            while abs(p[1]) <= 1:
                p = p - 0.1 * np.array([2 * p[0], -2 * p[1]])
                pts.append(p.copy())
            path = VMobject(color=C1, stroke_width=3).set_points_as_corners([ax.c2p(*q) for q in pts])
            dots = VGroup(*[Dot(ax.c2p(*q), radius=0.035, color=C1) for q in pts])
            n2 = T("GD η = 0.1 từ (1, 0.001):", 21, C1).next_to(n1, DOWN, buff=0.4, aligned_edge=LEFT)
            n3 = T(f"y × 1.2 mỗi bước, |y| > 1 sau {len(pts) - 1} bước", 21, C1).next_to(n2, DOWN, buff=0.15,
                                                                                        aligned_edge=LEFT)
            self.play(FadeIn(n2), FadeIn(n3))
            self.play(Create(path), FadeIn(dots, lag_ratio=0.05), run_time=3.5)
        with noi(self, "Gradient descent tự đi theo hướng có trị riêng âm và thoát khỏi yên ngựa. Newton thuần thì bị hút vào."):
            cap = swap_caption(self, cap, "GD đi theo hướng trị riêng âm và thoát khỏi yên ngựa", 24)
        self.wait(3)
