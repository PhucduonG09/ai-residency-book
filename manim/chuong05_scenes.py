"""Các cảnh Manim cho Chương 5 — ML cổ điển.

Render một cảnh (chất lượng 720p):
    manim -qm chuong05_scenes.py HamMatMatSoSanh

Chỉ dùng Text (không cần LaTeX), nên chỉ cần Manim Community + ffmpeg.
Mỗi cảnh dùng đúng số liệu của ví dụ giải mẫu trong sách.
"""
import re
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
    s = re.sub(r"(?<![\w])-(?=\d)", "−", s)          # dấu trừ Unicode cho số âm
    return Text(s, font=FONT, font_size=size, color=color)


def caption(s, size=26):
    t = T(s, size)
    return VGroup(BackgroundRectangle(t, fill_opacity=0.85, buff=0.12), t).to_edge(DOWN, buff=0.25)


def ticks(ax, xs=(), ys=(), size=20):
    """Nhãn trục bằng Text (include_numbers của Manim cần LaTeX)."""
    g = VGroup()
    for v in xs:
        g.add(T(f"{v:g}", size, GREY_B).next_to(ax.c2p(v, ax.y_range[0]), DOWN, buff=0.12))
    for v in ys:
        g.add(T(f"{v:g}", size, GREY_B).next_to(ax.c2p(ax.x_range[0], v), LEFT, buff=0.12))
    return g


def swap_caption(scene, old, text, **kw):
    new = caption(text, **kw)
    if old is None:
        scene.play(FadeIn(new), run_time=0.5)
    else:
        scene.play(FadeOut(old), run_time=0.2)           # mờ hẳn rồi mới hiện phụ đề mới: không chữ chồng chữ
        scene.play(FadeIn(new), run_time=0.3)
    return new


# ---------------------------------------------------------------- 5.1
class LogisticTachDuoc(Scene):
    """Ví dụ 5.2: x = 1..4, y = (0, 0, 1, 1). Không L2: w tăng mãi; L2 λ = 0.1: dừng ở w = 1.508."""

    def construct(self):
        title = T("Dữ liệu tách được: logistic không regularize thì ‖w‖ → ∞", 28).to_edge(UP, buff=0.3)
        ax = Axes(x_range=[0, 5, 1], y_range=[0, 1, 0.5], x_length=9, y_length=4.2,
                  tips=False).shift(DOWN * 0.2 + LEFT * 0.6)
        xs, ys = [1, 2, 3, 4], [0, 0, 1, 1]
        dots = VGroup(*[Dot(ax.c2p(x, y), radius=0.09, color=C1 if y else C2) for x, y in zip(xs, ys)])
        bound = DashedLine(ax.c2p(2.5, 0), ax.c2p(2.5, 1), color=GREY_B)
        blab = T("x = 2.5", 20, GREY_B).next_to(ax.c2p(2.5, 1), UP, buff=0.08)
        with noi(self, "Bốn điểm dữ liệu tách được hoàn toàn: hai điểm nhãn không bên trái, hai điểm nhãn một bên phải, ranh giới ở hai phẩy năm."):
            self.play(Write(title), Create(ax), FadeIn(ticks(ax, [0, 1, 2, 3, 4, 5], [0, 0.5, 1])), FadeIn(dots), Create(bound), FadeIn(blab))

        params = [(0.25, 0.0, "1 bước"), (1.716, -3.959, "100 bước"), (4.583, -11.263, "1 000 bước"),
                  (8.925, -22.135, "10 000 bước"), (10.312, -25.606, "20 000 bước")]

        def curve(w, b, color=C_ACC):
            return ax.plot(lambda x: 1 / (1 + np.exp(-(w * x + b))), x_range=[0, 5, 0.01], color=color, stroke_width=5)

        def info(w, b, name, color=WHITE):
            g = VGroup(T(name, 24, color), T(f"w = {w:.3f}", 24, color), T(f"b = {b:.3f}", 24, color),
                       T(f"−b/w = {(-b / w) + 0.0:.3f}" if w else "", 24, color))
            return g.arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_edge(RIGHT, buff=0.4).shift(UP * 0.6)

        cur, box = curve(*params[0][:2]), info(*params[0])
        with noi(self, "Chạy gradient descent cho logistic regression không có regularization. Sau một trăm, một nghìn, rồi hai vạn bước, w cứ tăng mãi."):
            self.play(Create(cur), FadeIn(box))
            cap = swap_caption(self, None, "Gradient descent, η = 0.5, không regularization")
            for w, b, name in params[1:]:
                self.play(Transform(cur, curve(w, b)), Transform(box, info(w, b, name)), run_time=1.3)
                self.wait(0.3)
        with noi(self, "Sigmoid dốc dần thành bậc thang, chuẩn của w tiến ra vô cùng. Ranh giới vẫn đúng, nhưng xác suất dự đoán ngày càng tự tin quá mức."):
            cap = swap_caption(self, cap, "Sigmoid thành bậc thang; ‖w‖ → ∞, −b/w vẫn tăng chậm về 2.5")
            self.wait(1.2)
        with noi(self, "Thêm L hai với lambda bằng không phẩy một, thuật toán hội tụ ở w bằng một phẩy năm không tám, ranh giới đúng hai phẩy năm."):
            w_l2, b_l2 = 1.507673670141066, -3.7691841753526623     # đúng giá trị logistic_gd(lam=0.1) sau 20 000 bước
            reg = curve(w_l2, b_l2, BLUE)
            rbox = info(w_l2, b_l2, "L2 λ = 0.1 (hội tụ)", BLUE)
            self.play(Transform(cur, reg), Transform(box, rbox), run_time=1.5)
            cap = swap_caption(self, cap, "Thêm L2: dừng ở w = 1.508, ranh giới đúng 2.5")
        self.wait(2.5)


# ---------------------------------------------------------------- 5.2
class BiasVarianceDaThuc(Scene):
    """Bảng bias–variance (x cố định, 20 điểm, σ = 0.3): bậc 1, 3, 9."""

    def construct(self):
        title = T("Bias và variance: khớp lại với nhiều bộ nhãn nhiễu", 30).to_edge(UP, buff=0.3)
        with noi(self, "Bias và variance: khớp đa thức với nhiều bộ nhãn nhiễu khác nhau, rồi xem các đường khớp phân tán thế nào."):
            self.play(Write(title))
        rng = np.random.default_rng(0)
        xs = np.linspace(0, 1, 20)
        f = lambda x: np.sin(2 * np.pi * x)
        stats = {1: ("0.1809", "0.0078"), 3: ("0.0054", "0.0136"), 9: ("0.0000", "0.0380")}
        panels, fits = [], []
        for i, deg in enumerate([1, 3, 9]):
            ax = Axes(x_range=[0, 1, 0.5], y_range=[-1.8, 1.8, 1], x_length=3.9, y_length=3.4,
                      tips=False, axis_config={"stroke_opacity": 0.5}).move_to(LEFT * 4.4 + RIGHT * 4.4 * i + UP * 0.35)
            truth = ax.plot(f, x_range=[0, 1, 0.01], color=WHITE, stroke_width=4)
            lab = T(f"bậc {deg}", 26).next_to(ax, UP, buff=0.1)
            b2, var = stats[deg]
            st = VGroup(T(f"bias² = {b2}", 22, C_ACC), T(f"variance = {var}", 22, C1)).arrange(DOWN, buff=0.1)
            st.next_to(ax, DOWN, buff=0.15)
            lines = VGroup()
            for _ in range(20):
                ys = f(xs) + 0.3 * rng.standard_normal(20)
                c = np.polyfit(xs, ys, deg)
                lines.add(ax.plot(lambda x, c=c: np.clip(np.polyval(c, x), -1.8, 1.8), x_range=[0.05, 0.95, 0.01],
                                  color=C1, stroke_width=1.5, stroke_opacity=0.45))
            panels.append(VGroup(ax, truth, lab))
            fits.append((lines, st))
        with noi(self, "Đường trắng là hàm thật, sin hai pi x. Mỗi đường xanh là một lần khớp với một bộ nhãn nhiễu khác, dùng đa thức bậc một, bậc ba và bậc chín."):
            self.play(*[FadeIn(p) for p in panels])
            cap = swap_caption(self, None, "Đường trắng: hàm thật sin(2πx). Mỗi đường xanh: một bộ nhãn nhiễu khác")
            self.play(*[Create(l) for l, _ in fits], run_time=3)
            self.play(*[FadeIn(s) for _, s in fits])
        with noi(self, "Bậc một lệch đều khỏi hàm thật: bias lớn. Bậc chín đúng trung bình nhưng dao động mạnh giữa các lần khớp: variance lớn."):
            cap = swap_caption(self, cap, "Bậc 1: lệch đều (bias). Bậc 9: đúng trung bình nhưng dao động (variance)")
            self.wait(2)
        with noi(self, "Cộng cả nhiễu, tổng sai số kỳ vọng là không phẩy hai bảy tám, không phẩy một không chín và không phẩy một hai tám. Bậc ba cân bằng tốt nhất."):
            cap = swap_caption(self, cap, "Tổng sai số kỳ vọng (+ σ² = 0.09): 0.2787, 0.1090, 0.1280. Bậc 3 thắng")
        self.wait(2.5)


class L1L2HinhHoc(Scene):
    """Đường đồng mức chạm quả cầu L2 (cả hai tọa độ ≠ 0) và quả cầu L1 (ở đỉnh, một tọa độ = 0)."""

    def construct(self):
        title = T("Vì sao L1 cho nghiệm thưa còn L2 thì không", 30).to_edge(UP, buff=0.3)
        with noi(self, "Vì sao L một cho nghiệm thưa còn L hai thì không? Câu trả lời nằm ở hình dạng của vùng ràng buộc."):
            self.play(Write(title))
        c = np.array([2.0, 0.9])                              # cực tiểu của loss không phạt
        H = np.array([[1.0, 0.35], [0.35, 0.6]])              # loss = (w − c)ᵀH(w − c)
        q = lambda p: (p - c) @ H @ (p - c)
        t = np.linspace(0, 2 * np.pi, 4001)
        circ = np.c_[np.cos(t), np.sin(t)]
        dia = circ / np.abs(circ).sum(1, keepdims=True)
        p2 = circ[np.argmin([q(p) for p in circ])]
        p1 = dia[np.argmin([q(p) for p in dia])]
        L = np.linalg.cholesky(np.linalg.inv(H))

        def ellipse(ax, level, color=BLUE):
            pts = (np.sqrt(level) * (L @ circ[::40].T)).T + c
            return VMobject(color=color, stroke_width=3).set_points_smoothly([ax.c2p(*p) for p in pts])

        groups, touch = [], []
        for k, (name, ball, pt) in enumerate([("L2: hình tròn", "l2", p2), ("L1: hình thoi", "l1", p1)]):
            ax = Axes(x_range=[-1.5, 3.2, 1], y_range=[-1.5, 2.2, 1], x_length=5.4, y_length=4.3,
                      tips=False, axis_config={"stroke_opacity": 0.6}).move_to(LEFT * 3.4 + RIGHT * 6.8 * k + DOWN * 0.15)
            shape = (Circle(radius=ax.x_length / 4.7, color=C_ACC, fill_opacity=0.25).move_to(ax.c2p(0, 0))
                     if ball == "l2" else
                     Polygon(ax.c2p(1, 0), ax.c2p(0, 1), ax.c2p(-1, 0), ax.c2p(0, -1), color=C_ACC, fill_opacity=0.25))
            star = Dot(ax.c2p(*c), color=WHITE)
            lab = T(name, 24).next_to(ax, DOWN, buff=0.05)
            groups.append(VGroup(ax, shape, star, lab))
            touch.append((ax, pt))
        with noi(self, "Chấm trắng là cực tiểu khi không phạt. Nới rộng dần đường đồng mức của loss cho tới khi chạm hình tròn L hai: điểm chạm có cả hai tọa độ khác không."):
            self.play(*[FadeIn(g) for g in groups])
            cap = swap_caption(self, None, "Chấm trắng: cực tiểu khi không phạt. Đường xanh: đồng mức của loss")
            lev = ValueTracker(0.02)
            ells = [always_redraw(lambda ax=ax: ellipse(ax, lev.get_value())) for ax, _ in touch]
            self.add(*ells)
            self.play(lev.animate.set_value(q(p2)), run_time=2.5)
            a0, pt0 = touch[0]
            lab0 = T(f"({pt0[0]:.2f}, {pt0[1]:.2f})", 22, WHITE).move_to(a0.c2p(-1.0, 1.65))   # nhãn ở ngoài hình, nối bằng đoạn mảnh
            self.play(FadeIn(Dot(a0.c2p(*pt0), color=C2, radius=0.1)), FadeIn(lab0),
                      FadeIn(Line(lab0.get_bottom() + DOWN * 0.05, a0.c2p(*pt0), color=C2, stroke_width=2)))
        with noi(self, "Với hình thoi L một, đường đồng mức chạm đúng vào đỉnh, nơi w hai bằng không. Các đỉnh nhọn nằm trên trục nên rất dễ bị chạm trước."):
            self.play(lev.animate.set_value(q(p1)), run_time=1.5)
            a1, pt1 = touch[1]
            self.play(FadeIn(Dot(a1.c2p(*pt1), color=C2, radius=0.1)),
                      FadeIn(T(f"({pt1[0]:.2f}, {abs(pt1[1]):.2f})", 22, WHITE).next_to(a1.c2p(*pt1), DR, buff=0.3).shift(DOWN * 0.15 + RIGHT * 0.45)))
            cap = swap_caption(self, cap, "L2 chạm ở điểm có hai tọa độ khác 0; L1 chạm ở đỉnh: w₂ = 0")
        self.wait(2)
        # phần 2: soft-threshold trên thiết kế trực chuẩn (ví dụ 5.5)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not title and not nen.la_luoi(m)], run_time=0.8)
        with noi(self, "Với thiết kế trực chuẩn, có thể tính trực tiếp. Lấy z bằng ba, không phẩy tám, trừ hai, và lambda bằng một."):
            z, rid, las = [3, 0.8, -2], [1.5, 0.4, -1.0], [2, 0, -1]
            ax = Axes(x_range=[0, 3.6, 1], y_range=[-2.5, 3.5, 1], x_length=8.5, y_length=4.6, tips=False,
                      x_axis_config={"stroke_opacity": 0.0}).shift(DOWN * 0.3)
            bars = VGroup()
            for j in range(3):
                for k, (vals, col) in enumerate([(z, GREY_B), (rid, BLUE), (las, C_ACC)]):
                    x0 = 0.35 + j * 1.1 + k * 0.28
                    v = vals[j]
                    if v != 0:
                        bars.add(Polygon(ax.c2p(x0, 0), ax.c2p(x0 + 0.24, 0), ax.c2p(x0 + 0.24, v), ax.c2p(x0, v),
                                         color=col, fill_opacity=0.85, stroke_width=1))
                    else:
                        bars.add(T("0", 22, col).move_to(ax.c2p(x0 + 0.12, 0.25)))
                bars.add(T(f"z{'₁₂₃'[j]} = {z[j]}", 22).move_to(ax.c2p(0.63 + j * 1.1, -2.3)))
            legend = VGroup(T("OLS z", 22, GREY_B), T("ridge z/(1+λ)", 22, BLUE), T("lasso: trừ λ, cắt về 0", 22, C_ACC))
            legend.arrange(RIGHT, buff=0.5).next_to(title, DOWN, buff=0.2)
            self.play(Create(ax), FadeIn(ticks(ax, [], [-2, -1, 0, 1, 2, 3])), FadeIn(legend))
            self.play(FadeIn(bars), run_time=1.5)
        with noi(self, "Ridge co đều mọi hệ số, chia cho hai. Lasso trừ đi lambda rồi cắt về không: hệ số nhỏ không phẩy tám thành đúng không."):
            cap = swap_caption(self, None, "λ = 1: ridge (1.5, 0.4, −1) co đều; lasso (2, 0, −1) đưa số nhỏ về đúng 0")
        self.wait(3)


class NguongVaPrecision(Scene):
    """Ví dụ 5.6: tại TPR 0.9, FPR 0.05, precision phụ thuộc tỉ lệ lớp dương (0.947, 0.667, 0.154)."""

    def construct(self):
        from scipy.stats import norm
        title = T("Cùng một bộ phân loại, đổi tỉ lệ lớp dương", 30).to_edge(UP, buff=0.3)
        with noi(self, "Cùng một bộ phân loại, nhưng đổi tỉ lệ lớp dương. Bên trái là phân phối điểm số của lớp âm và lớp dương, bên phải là đường ROC."):
            self.play(Write(title))
            mu = norm.ppf(0.95) + norm.ppf(0.9)                     # để ngưỡng 1.645 cho TPR 0.9, FPR 0.05
            ax = Axes(x_range=[-3, 6, 1], y_range=[0, 0.45, 0.2], x_length=6.4, y_length=3.4, tips=False,
                      axis_config={"stroke_opacity": 0.6}).move_to(LEFT * 3.3 + UP * 0.25)
            neg = ax.plot(lambda x: norm.pdf(x), x_range=[-3, 6, 0.02], color=C2)
            pos = ax.plot(lambda x: norm.pdf(x - mu), x_range=[-3, 6, 0.02], color=C1)
            lg = VGroup(T("âm", 22, C2), T("dương", 22, C1)).arrange(RIGHT, buff=0.4).next_to(ax, UP, buff=0.05)
            xl = T("điểm số của mô hình", 20, GREY_B).next_to(ax, DOWN, buff=0.1)
            roc = Axes(x_range=[0, 1, 0.5], y_range=[0, 1, 0.5], x_length=3.2, y_length=3.2, tips=False,
                       ).move_to(RIGHT * 1.6 + UP * 0.35)
            rlab = T("ROC: TPR theo FPR", 20).next_to(roc, UP, buff=0.1)
            curve = roc.plot_parametric_curve(lambda s: np.array([norm.sf(s), norm.sf(s - mu), 0]), t_range=[-3, 7], color=BLUE)
            self.play(Create(ax), Create(neg), Create(pos), FadeIn(lg), FadeIn(xl), Create(roc), FadeIn(ticks(roc, [0, 0.5, 1], [0.5, 1], 18)), FadeIn(rlab), Create(curve))
        th = ValueTracker(3.6)
        line = always_redraw(lambda: DashedLine(ax.c2p(th.get_value(), 0), ax.c2p(th.get_value(), 0.43), color=C_ACC))
        dot = always_redraw(lambda: Dot(roc.c2p(norm.sf(th.get_value()), norm.sf(th.get_value() - mu)), color=C_ACC))

        def prec(t, p):
            a, b = norm.sf(t - mu) * p, norm.sf(t) * (1 - p)
            return a / (a + b)

        def panel():
            t = th.get_value()
            rows = [f"ngưỡng = {t:.2f}", f"TPR = {norm.sf(t - mu):.3f}", f"FPR = {norm.sf(t):.3f}", "precision khi dương chiếm:",
                    f"  50%: {prec(t, 0.5):.3f}", f"  10%: {prec(t, 0.1):.3f}", f"   1%: {prec(t, 0.01):.3f}"]
            g = VGroup(*[T(r, 21, C_ACC if i >= 4 else WHITE) for i, r in enumerate(rows)])
            return g.arrange(DOWN, aligned_edge=LEFT, buff=0.1).to_edge(RIGHT, buff=0.65).shift(UP * 0.35)

        pan = always_redraw(panel)
        with noi(self, "Hạ dần ngưỡng: điểm trên đường ROC chạy lên, TPR và FPR cùng tăng."):
            self.play(Create(line), FadeIn(dot), FadeIn(pan))
            cap = swap_caption(self, None, "Hạ ngưỡng: điểm trên đường ROC chạy lên, TPR và FPR cùng tăng")
            self.play(th.animate.set_value(0.2), run_time=3)
            self.play(th.animate.set_value(norm.ppf(0.95)), run_time=2)
        with noi(self, "Tại TPR không phẩy chín, FPR không phẩy không năm. Đường ROC không đổi theo tỉ lệ lớp, nhưng precision thì thay đổi rất mạnh."):
            cap = swap_caption(self, cap, "TPR 0.9, FPR 0.05: đường ROC không đổi theo tỉ lệ lớp, precision thì sụp")
            self.wait(1.5)
        with noi(self, "Khi lớp dương chỉ chiếm một phần trăm, precision chỉ còn không phẩy một năm bốn: khoảng tám lăm phần trăm cảnh báo là báo nhầm."):
            cap = swap_caption(self, cap, "Lớp dương 1%: precision 0.154, tức 85% cảnh báo là báo nhầm")
        self.wait(2.5)


# ---------------------------------------------------------------- 5.3
class HamMatMatSoSanh(Scene):
    """0-1, hinge, log-loss (chia ln 2) theo margin m = y·f(x); đánh dấu m = 0.5 và m = −1 (ví dụ 5.8)."""

    def construct(self):
        title = T("Ba hàm mất mát trên cùng một trục margin m = y·f(x)", 30).to_edge(UP, buff=0.3)
        ax = Axes(x_range=[-2, 3, 1], y_range=[0, 3.2, 1], x_length=8.2, y_length=4.6, tips=False).shift(DOWN * 0.3 + LEFT * 1.4)
        xl = T("m", 24).next_to(ax.x_axis, RIGHT, buff=0.15)
        with noi(self, "So sánh ba hàm mất mát trên cùng trục margin m, bằng y nhân f x: zero-one loss, hinge loss, và log loss."):
            self.play(Write(title), Create(ax), FadeIn(ticks(ax, [-2, -1, 0, 1, 2, 3], [1, 2, 3])), FadeIn(xl))
            zero_one = VGroup(Line(ax.c2p(-2, 1), ax.c2p(0, 1), color=GREY_B, stroke_width=5),
                              Line(ax.c2p(0, 0), ax.c2p(3, 0), color=GREY_B, stroke_width=5))
            hinge = ax.plot(lambda m: max(0.0, 1 - m), x_range=[-2, 3, 0.01], color=C_ACC, stroke_width=5)
            logl = ax.plot(lambda m: np.log1p(np.exp(-m)) / np.log(2), x_range=[-2, 3, 0.01], color=BLUE, stroke_width=5)
            names = [(zero_one, "0-1: sai hay đúng", GREY_B), (hinge, "hinge: max(0, 1 − m)", C_ACC),
                     (logl, "log-loss: ln(1 + e⁻ᵐ)/ln 2", BLUE)]
            legend = VGroup()
            for i, (mob, txt, col) in enumerate(names):
                legend.add(T(txt, 22, col))
            legend.arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(RIGHT, buff=0.25).shift(UP * 0.9)
            for i, (mob, _, _) in enumerate(names):
                self.play(Create(mob), FadeIn(legend[i]), run_time=1.2)
        with noi(self, "Hinge và log loss đều lồi và nằm trên zero-one loss, nên tối ưu chúng thay cho zero-one là hợp lý."):
            cap = swap_caption(self, None, "Cả hinge và log-loss đều lồi và nằm trên 0-1 loss")
            self.wait(1)
        with noi(self, "Một điểm có margin không phẩy năm, dù phân loại đúng, vẫn chịu hinge loss không phẩy năm. Điểm sai với margin trừ một chịu hinge loss bằng hai."):
            marks = VGroup()
            for m, h, txt in [(0.5, 0.5, "điểm (2.5, 1): m = 0.5, hinge 0.5"), (-1, 2, "điểm (2, 2): m = −1, hinge 2")]:
                marks.add(Dot(ax.c2p(m, h), color=C2, radius=0.1))
                marks.add(DashedLine(ax.c2p(m, 0), ax.c2p(m, h), color=C2))
            notes = VGroup(T("điểm (2.5, 1): m = 0.5", 20, C2), T("  → hinge 0.5", 20, C2), T("điểm (2, 2): m = −1", 20, C2), T("  → hinge 2", 20, C2))
            notes.arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(legend, DOWN, buff=0.5, aligned_edge=LEFT)
            self.play(FadeIn(marks), FadeIn(notes))
        with noi(self, "Khi margin từ một trở lên, hinge bằng đúng không, còn log loss chỉ nhỏ dần chứ không bao giờ bằng không."):
            cap = swap_caption(self, cap, "Trong lề vẫn bị phạt; m ≥ 1: hinge = 0, log-loss chỉ nhỏ dần")
        self.wait(3)


# ---------------------------------------------------------------- 5.4
class BoostingPhanDu(Scene):
    """Ví dụ 5.12: 6 điểm, stump chia ở 3.5, η = 0.5; MSE 0.2447 → 0.0695 → 0.0257."""

    def construct(self):
        x = np.arange(1, 7, dtype=float)
        y = np.array([1.0, 1.2, 1.1, 3.0, 3.2, 2.9])
        title = T("Gradient boosting: mỗi vòng khớp một stump với phần dư", 30).to_edge(UP, buff=0.3)
        ax = Axes(x_range=[0, 7, 1], y_range=[0, 3.6, 1], x_length=8.4, y_length=4.6, tips=False).shift(DOWN * 0.3 + LEFT * 1.5)
        dots = VGroup(*[Dot(ax.c2p(a, b), color=WHITE, radius=0.08) for a, b in zip(x, y)])
        split = DashedLine(ax.c2p(3.5, 0), ax.c2p(3.5, 3.6), color=GREY_B)
        slab = T("chia tại x < 3.5", 20, GREY_B).next_to(ax.c2p(3.5, 3.6), UP, buff=0.05)
        with noi(self, "Gradient boosting với sáu điểm. Mô hình ban đầu là trung bình, hai phẩy không sáu bảy. Các đoạn đỏ là phần dư, cũng chính là âm gradient của MSE."):
            self.play(Write(title), Create(ax), FadeIn(ticks(ax, range(1, 7), [1, 2, 3])), FadeIn(dots))

            def model(F):
                return VGroup(Line(ax.c2p(0.5, F[0]), ax.c2p(3.5, F[0]), color=C_ACC, stroke_width=6),
                              Line(ax.c2p(3.5, F[3]), ax.c2p(6.5, F[3]), color=C_ACC, stroke_width=6))

            def resid(F):
                return VGroup(*[Line(ax.c2p(a, f), ax.c2p(a, b), color=C2, stroke_width=4) for a, b, f in zip(x, y, F)])

            def info(k, F):
                mse = np.mean((y - F) ** 2)
                g = VGroup(T(f"vòng {k}", 26), T(f"trái = {F[0]:.4f}", 24, C_ACC), T(f"phải = {F[3]:.4f}", 24, C_ACC),
                           T(f"MSE = {mse:.4f}", 26, C2))
                return g.arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(RIGHT, buff=0.5).shift(UP * 0.4)

            F = np.full(6, y.mean())
            m, r, box = model(F), resid(F), info(0, F)
            self.play(Create(m), Create(r), FadeIn(box))
            cap = swap_caption(self, None, "F₀ = trung bình = 2.0667. Đoạn đỏ: phần dư = âm gradient của MSE")
            self.wait(1)
        with noi(self, "Mỗi vòng khớp một cây một tầng, chia tại x bằng ba phẩy năm, với phần dư hiện tại."):
            self.play(Create(split), FadeIn(slab))
        for k, loi_doc in zip(range(1, 4), ["Vòng một: khớp stump với phần dư rồi chỉ cộng một nửa. MSE giảm còn không phẩy hai bốn.",
                                             "Vòng hai: phần dư nhỏ đi, MSE còn không phẩy không bảy.",
                                             "Vòng ba: MSE còn không phẩy không hai sáu."]):
            rr = y - F
            L = x < 3.5
            F = F + 0.5 * np.where(L, rr[L].mean(), rr[~L].mean())
            with noi(self, loi_doc):
                self.play(Transform(m, model(F)), Transform(r, resid(F)), Transform(box, info(k, F)), run_time=1.5)
                self.wait(0.6)
        with noi(self, "Với learning rate không phẩy năm, mỗi vòng chỉ đi nửa đường, phần dư co lại dần mà không khớp quá đà."):
            cap = swap_caption(self, cap, "η = 0.5: mỗi vòng chỉ đi nửa đường, phần dư co lại dần")
        self.wait(3)


# ---------------------------------------------------------------- 5.5
class KMeansVaEM(Scene):
    """Ví dụ 5.13 (k-means kẹt: J = 196 vs 77.5) và ví dụ 5.15 (một vòng EM)."""

    def construct(self):
        title = T("k-means kẹt ở cực tiểu địa phương", 30).to_edge(UP, buff=0.3)
        with noi(self, "K-means có thể bị kẹt ở cực tiểu địa phương. Cùng bảy điểm, chạy với hai cách khởi tạo tâm khác nhau."):
            self.play(Write(title), run_time=0.8)
            X = np.array([1.0, 2, 3, 8, 9, 10, 25])
            rows = []
            for k, (init, yoff) in enumerate([((1.0, 10.0), 1.3), ((1.0, 25.0), -1.2)]):
                nl = NumberLine(x_range=[0, 26, 5], length=11).shift(UP * (yoff - 0.3))
                lab = T(f"khởi tạo tâm ({init[0]:g}, {init[1]:g})", 22, GREY_A).next_to(nl, UP, buff=0.55).align_to(nl, LEFT)
                nums = VGroup(*[T(f"{v}", 18, GREY_B).next_to(nl.n2p(v), DOWN, buff=0.15) for v in range(0, 26, 5)])
                pts = VGroup(*[Dot(nl.n2p(v), radius=0.1, color=GREY_B) for v in X])
                cen = VGroup(*[Triangle(color=C_ACC, fill_opacity=1).scale(0.14).move_to(nl.n2p(c) + UP * 0.35) for c in init])
                rows.append((nl, lab, pts, cen, np.array(init)))
                self.play(Create(nl), FadeIn(nums), FadeIn(lab), FadeIn(pts), FadeIn(cen), run_time=0.6)
        with noi(self, "Mỗi vòng gán điểm cho tâm gần nhất, rồi dời tâm về trung bình. Khởi tạo thứ nhất dừng ở J bằng một trăm chín mươi sáu, khởi tạo thứ hai dừng ở bảy mươi bảy phẩy năm."):
            cols = [C1, BLUE]
            for nl, lab, pts, cen, c in rows:
                for _ in range(2):
                    a = np.argmin(np.abs(X[:, None] - c[None]), 1)
                    self.play(*[pts[i].animate.set_color(cols[a[i]]) for i in range(7)], run_time=0.35)
                    c = np.array([X[a == k].mean() for k in range(2)])
                    self.play(*[cen[k].animate.move_to(nl.n2p(c[k]) + UP * 0.35) for k in range(2)], run_time=0.45)
                J = np.sum((X - c[a]) ** 2)
                res = T(f"tâm ({c[0]:g}, {c[1]:g}),  J = {J:g}", 22, C_ACC).next_to(nl, DOWN, buff=0.45)
                self.play(FadeIn(res), run_time=0.4)
        with noi(self, "Cả hai đều là điểm dừng. Trong thực tế ta chạy nhiều khởi tạo và giữ kết quả có J nhỏ nhất."):
            cap = swap_caption(self, None, "Cả hai đều là điểm dừng. Chạy nhiều khởi tạo, giữ J nhỏ nhất (77.5)")
            self.wait(1)
        self.play(*[FadeOut(m) for m in self.mobjects if not nen.la_luoi(m)], run_time=0.6)

        # --- phần 2: một vòng EM
        with noi(self, "Thuật toán EM cho mô hình hỗn hợp Gauss thì gán mềm. Khởi tạo hai thành phần tâm không và năm, phương sai một."):
            title2 = T("EM cho GMM: gán mềm, rồi cập nhật có trọng số", 30).to_edge(UP, buff=0.3)
            Xe = np.array([0.0, 1, 2.5, 4, 5])
            ax = Axes(x_range=[-2, 7, 1], y_range=[0, 0.5, 0.25], x_length=10, y_length=3.6, tips=False).shift(DOWN * 0.2)
            self.play(Write(title2), Create(ax), FadeIn(ticks(ax, range(-2, 8), [])), run_time=1.0)

            def bells(mu, v):
                return VGroup(*[ax.plot(lambda t, m=m, s=s: 0.5 * np.exp(-(t - m) ** 2 / (2 * s)) / np.sqrt(2 * np.pi * s),
                                        x_range=[-2, 7, 0.02], color=col, stroke_width=4)
                                for m, s, col in zip(mu, v, [C1, BLUE])])

            def ll_text(val):
                return T(f"log-likelihood = {val:.3f}", 24, C_ACC).to_corner(UR, buff=0.4).shift(DOWN * 0.8)

            b = bells([0, 5], [1, 1])
            pts = VGroup(*[Dot(ax.c2p(v, 0), radius=0.11, color=GREY_B) for v in Xe])
            ll = ll_text(-11.491)
            self.play(Create(b), FadeIn(pts), FadeIn(ll), run_time=0.8)
        with noi(self, "Bước E tính xác suất mỗi điểm thuộc thành phần xanh lá. Điểm hai phẩy năm nằm giữa nên được chia đôi, mỗi bên không phẩy năm."):
            cap = swap_caption(self, None, "Khởi tạo μ = (0, 5), σ² = (1, 1), π = (0.5, 0.5)")
            g = np.array([1.0, 0.9994, 0.5, 0.0006, 0.0])
            cols = [interpolate_color(BLUE, C1, gi) for gi in g]
            gl = VGroup(*[T(f"{gi:.2f}", 18).next_to(ax.c2p(v, 0), DOWN, buff=0.45) for v, gi in zip(Xe, g)])
            gl_name = T("γ (xanh lá):", 18, C1).next_to(gl, LEFT, buff=0.3)
            self.play(*[pts[i].animate.set_color(cols[i]) for i in range(5)], FadeIn(gl), FadeIn(gl_name), run_time=0.8)
            cap = swap_caption(self, cap, "E-step: γ = xác suất thuộc thành phần xanh lá; điểm 2.5 chia đôi 0.5")
            self.wait(0.8)
        with noi(self, "Bước M cập nhật tâm và phương sai theo trọng số đó. Hai tâm dịch vào không phẩy chín và bốn phẩy một, và log likelihood tăng lên."):
            self.play(Transform(b, bells([0.9007, 4.0993], [0.8421, 0.8421])), Transform(ll, ll_text(-9.425)), run_time=1.5)
            cap = swap_caption(self, cap, "M-step: μ ≈ (0.901, 4.099), σ² ≈ 0.842; log-likelihood tăng lên")
            self.wait(1.8)
