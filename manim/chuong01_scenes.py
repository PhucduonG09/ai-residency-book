"""Các cảnh Manim cho Chương 1 — Đại số tuyến tính cho ML.

Render một cảnh (chất lượng 720p):
    manim -qm chuong01_scenes.py BienDangLuoi

Chỉ dùng Text (không cần LaTeX), nên chỉ cần Manim Community + ffmpeg.
Mỗi cảnh dùng đúng ma trận / số liệu của ví dụ giải mẫu trong sách.
"""
import sys
import numpy as np
from manim import *

FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"
C1, C2, C_ACC = GREEN, RED, YELLOW


def T(s, size=28, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


def caption(s):
    t = T(s, 26)
    return VGroup(BackgroundRectangle(t, fill_opacity=0.85, buff=0.12), t).to_edge(DOWN, buff=0.25)


def apply_m(mob, M, about=ORIGIN):
    """Áp ma trận 2×2 lên mobject (tuyến tính quanh điểm about)."""
    M3 = np.eye(3)
    M3[:2, :2] = np.array(M, dtype=float)
    return mob.apply_matrix(M3, about_point=about)


# ---------------------------------------------------------------- 1.1
class BienDangLuoi(Scene):
    """Ma trận là phép biến đổi: cột j của A là ảnh của eⱼ (ví dụ 1.6, A = [[2,1],[1,2]])."""

    def construct(self):
        A = [[2, 1], [1, 2]]
        title = T("Ma trận là một phép biến đổi", 34).to_edge(UP, buff=0.3)
        plane = NumberPlane(x_range=[-7, 7, 1], y_range=[-4, 4, 1],
                            background_line_style={"stroke_opacity": 0.5})
        ghost = plane.copy().set_opacity(0.15)
        e1 = Arrow(ORIGIN, RIGHT, buff=0, color=C1, stroke_width=6)
        e2 = Arrow(ORIGIN, UP, buff=0, color=C2, stroke_width=6)
        sq = Polygon(ORIGIN, RIGHT, RIGHT + UP, UP, color=C_ACC, fill_opacity=0.35, stroke_width=2)
        l1 = T("e₁", 26, C1).next_to(e1, DOWN, buff=0.1)
        l2 = T("e₂", 26, C2).next_to(e2, LEFT, buff=0.1)
        self.add(ghost)
        self.play(Write(title), Create(plane), run_time=1.5)
        self.play(FadeIn(sq), GrowArrow(e1), GrowArrow(e2), FadeIn(l1), FadeIn(l2))
        cap0 = caption("Hình vuông đơn vị có diện tích 1")
        self.play(FadeIn(cap0))
        self.wait(0.5)
        self.play(FadeOut(l1), FadeOut(l2), FadeOut(cap0))
        group = VGroup(plane, sq, e1, e2)
        self.play(ApplyMatrix(np.array(A, dtype=float), group), run_time=2.5)
        a1 = T("Ae₁ = cột 1 = (2, 1)", 26, C1).next_to(e1.get_end(), RIGHT, buff=0.15)
        a2 = T("Ae₂ = cột 2 = (1, 2)", 26, C2).next_to(e2.get_end(), UP, buff=0.15)
        self.play(FadeIn(a1), FadeIn(a2))
        cap = caption("Diện tích hình bình hành = |det A| = 2·2 − 1·1 = 3")
        self.play(FadeIn(cap))
        self.wait(2.5)


class ABKhacBA(Scene):
    """AB ≠ BA: S = diag(2,1), R = quay 90° (ví dụ 1.2)."""

    def construct(self):
        S = [[2, 0], [0, 1]]
        R = [[0, -1], [1, 0]]
        title = T("Thứ tự phép nhân quan trọng: SR ≠ RS", 32).to_edge(UP, buff=0.3)
        self.play(Write(title))
        centers = [LEFT * 3.3 + DOWN * 0.45, RIGHT * 3.3 + DOWN * 0.45]
        planes, shapes, labels = [], [], []
        pts = [[0, 0], [0, 1.6], [1.0, 1.6], [1.0, 1.3], [0.3, 1.3], [0.3, 0.9],
               [0.8, 0.9], [0.8, 0.6], [0.3, 0.6], [0.3, 0]]       # chữ F, bất đối xứng
        names = ["SR: quay trước, kéo giãn sau", "RS: kéo giãn trước, quay sau"]
        for c, name in zip(centers, names):
            p = NumberPlane(x_range=[-3, 3, 1], y_range=[-3, 3, 1], x_length=4.4, y_length=4.4,
                            background_line_style={"stroke_opacity": 0.35}).move_to(c + UP * 0.15)
            unit = p.x_length / 6
            f = Polygon(*[p.get_center() + unit * np.array([x, y, 0]) for x, y in pts],
                        color=C_ACC, fill_opacity=0.6, stroke_width=2)
            planes.append(p)
            shapes.append(f)
            labels.append(T(name, 22).next_to(p, UP, buff=0.12))
        self.play(*[Create(p) for p in planes], *[FadeIn(f) for f in shapes], *[FadeIn(l) for l in labels])
        self.wait(0.5)
        # trái: SR = áp R trước rồi S;  phải: RS = áp S trước rồi R
        o0, o1 = planes[0].get_center(), planes[1].get_center()
        self.play(apply_m(shapes[0].animate, R, o0), apply_m(shapes[1].animate, S, o1), run_time=1.5)
        self.play(apply_m(shapes[0].animate, S, o0), apply_m(shapes[1].animate, R, o1), run_time=1.5)
        self.play(FadeIn(caption("Cùng hai lớp, đổi thứ tự → kết quả khác hẳn")))
        self.wait(2.5)


# ---------------------------------------------------------------- 1.3
class PhepChieu(Scene):
    """Bình phương tối thiểu là phép chiếu: chiếu b = (2,3) lên hướng a = (1,1)."""

    def construct(self):
        title = T("Bình phương tối thiểu = phép chiếu", 34).to_edge(UP, buff=0.3)
        plane = NumberPlane(x_range=[-1, 5, 1], y_range=[-1, 4, 1], x_length=7.2, y_length=6,
                            background_line_style={"stroke_opacity": 0.35}).shift(DOWN * 0.3)
        c2p = plane.c2p
        line = Line(c2p(-0.8, -0.8), c2p(3.8, 3.8), color=BLUE, stroke_width=4)
        lab_line = T("C(A): mọi bội của a = (1, 1)", 22, BLUE).next_to(c2p(3.8, 3.8), RIGHT, buff=0.1).shift(DOWN * 0.9)
        b = Arrow(c2p(0, 0), c2p(2, 3), buff=0, color=C_ACC, stroke_width=6)
        lb = T("b = (2, 3)", 24, C_ACC).next_to(c2p(2, 3), UP, buff=0.1)
        p = Arrow(c2p(0, 0), c2p(2.5, 2.5), buff=0, color=C1, stroke_width=6)
        lp = T("Ax̂ = (2.5, 2.5)", 24, C1).next_to(c2p(2.5, 2.5), DR, buff=0.12)
        r = DashedLine(c2p(2, 3), c2p(2.5, 2.5), color=C2, stroke_width=4)
        lr = T("phần dư (−0.5, 0.5)", 22, C2).next_to(c2p(2.25, 2.75), RIGHT, buff=0.35)
        corner = RightAngle(Line(c2p(2.5, 2.5), c2p(0, 0)), Line(c2p(2.5, 2.5), c2p(2, 3)),
                            length=0.25, color=WHITE)
        self.play(Write(title), Create(plane))
        self.play(Create(line), FadeIn(lab_line))
        self.play(GrowArrow(b), FadeIn(lb))
        cap0 = caption("b không nằm trên đường thẳng: Ax = b vô nghiệm")
        self.play(FadeIn(cap0))
        self.wait(1)
        self.play(FadeOut(cap0))
        # một điểm chạy trên đường thẳng; khoảng cách tới b nhỏ nhất ở chân đường vuông góc
        t = ValueTracker(0.3)
        dot = always_redraw(lambda: Dot(c2p(t.get_value(), t.get_value()), color=WHITE))
        seg = always_redraw(lambda: Line(c2p(t.get_value(), t.get_value()), c2p(2, 3),
                                         color=GREY_B, stroke_width=3))
        dist = always_redraw(lambda: T(
            f"khoảng cách = {np.hypot(2 - t.get_value(), 3 - t.get_value()):.3f}", 24
        ).to_corner(UL, buff=0.5).shift(DOWN * 0.8))
        self.play(FadeIn(dot), Create(seg), FadeIn(dist))
        self.play(t.animate.set_value(3.6), run_time=2)
        self.play(t.animate.set_value(2.5), run_time=1.5)
        self.play(FadeOut(seg), FadeOut(dot))
        self.play(GrowArrow(p), FadeIn(lp), Create(r), FadeIn(lr), Create(corner))
        self.play(FadeIn(caption("Gần nhất khi phần dư vuông góc với C(A): Aᵀ(b − Ax̂) = 0")))
        self.wait(3)


# ---------------------------------------------------------------- 1.4
class VectorRieng(Scene):
    """Vector riêng: chỉ ở hướng riêng thì Av cùng phương với v (A = [[2,1],[1,2]])."""

    def construct(self):
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        title = T("Vector riêng: hướng mà A chỉ kéo giãn, không xoay", 30).to_edge(UP, buff=0.3)
        plane = NumberPlane(x_range=[-4, 4, 1], y_range=[-4, 4, 1], x_length=6.4, y_length=6.4,
                            background_line_style={"stroke_opacity": 0.3}).shift(DOWN * 0.35)
        c2p = plane.c2p
        circle = Circle(radius=plane.x_length / 8, color=GREY_B).move_to(c2p(0, 0))
        ang = ValueTracker(0.0)

        def v():
            a = ang.get_value()
            return np.array([np.cos(a), np.sin(a)])

        def is_eig():
            w = A @ v()
            return abs(v()[0] * w[1] - v()[1] * w[0]) < 0.06

        va = always_redraw(lambda: Arrow(c2p(0, 0), c2p(*v()), buff=0, color=C1, stroke_width=6))
        wa = always_redraw(lambda: Arrow(c2p(0, 0), c2p(*(A @ v())), buff=0,
                                         color=C_ACC if is_eig() else C2, stroke_width=6))
        legend = VGroup(T("v (độ dài 1)", 24, C1), T("Av", 24, C2)).arrange(DOWN, aligned_edge=LEFT)
        legend.to_corner(UL, buff=0.5).shift(DOWN * 0.8)
        self.play(Write(title), Create(plane), Create(circle))
        self.play(FadeIn(va), FadeIn(wa), FadeIn(legend))
        stops = [(PI / 4, "v = (1, 1)/√2 → Av = 3v,  λ = 3"),
                 (3 * PI / 4, "v = (−1, 1)/√2 → Av = 1·v,  λ = 1"),
                 (5 * PI / 4, "v = −(1, 1)/√2 → vẫn λ = 3"),
                 (7 * PI / 4, "v = (1, −1)/√2 → λ = 1")]
        for a, msg in stops:
            self.play(ang.animate.set_value(a), run_time=1.6, rate_func=linear)
            cap = caption(msg)
            self.play(FadeIn(cap), run_time=0.4)
            self.wait(0.8)
            self.play(FadeOut(cap), run_time=0.3)
        self.play(ang.animate.set_value(2 * PI), run_time=0.8, rate_func=linear)
        self.play(FadeIn(caption("Ở mọi hướng khác, Av bị xoay đi so với v")))
        self.wait(2)


# ---------------------------------------------------------------- 1.5
class SVDXoayKeoXoay(Scene):
    """SVD của A = [[3,0],[4,5]] (ví dụ 1.7): Vᵀ xoay → Σ kéo giãn → U xoay.

    Chọn dấu v₂ = (−1, 1)/√2, u₂ = (−3, 1)/√10 để V và U đều là phép quay
    (SVD chỉ xác định tới dấu của từng cặp uᵢ, vᵢ).
    Đường tròn và hai mũi tên được vẽ lại ở mỗi khung hình từ ma trận nội suy,
    nên đầu mũi tên không bị méo như khi dùng apply_matrix lên cả nhóm.
    """

    def construct(self):
        A = np.array([[3.0, 0.0], [4.0, 5.0]])
        s1, s2 = 3 * np.sqrt(5), np.sqrt(5)
        V = np.array([[1, -1], [1, 1]]) / np.sqrt(2)          # cột v₁, v₂
        U = np.array([[1, -3], [3, 1]]) / np.sqrt(10)          # cột u₁, u₂
        Sig = np.diag([s1, s2])
        assert np.allclose(U @ Sig @ V.T, A)
        k = 0.32                                                # tỉ lệ vẽ: 1 đơn vị = 0.32 (elip cao ±6.4 vẫn nằm trong lưới)
        title = T("SVD: xoay → kéo giãn → xoay", 34).to_edge(UP, buff=0.3)
        axes = NumberPlane(x_range=[-8, 8, 1], y_range=[-7, 7, 1], x_length=16 * k, y_length=14 * k,
                           background_line_style={"stroke_opacity": 0.25}).shift(DOWN * 0.2)
        o = axes.c2p(0, 0)
        info = T("A = [[3, 0], [4, 5]]   σ₁ = 3√5 ≈ 6.71,  σ₂ = √5 ≈ 2.24", 24).next_to(title, DOWN, buff=0.15)

        state = {"M0": np.eye(2), "M1": np.eye(2)}
        s = ValueTracker(0.0)

        def M():
            return state["M0"] + s.get_value() * (state["M1"] - state["M0"])

        th = np.linspace(0, TAU, 241)
        unit = np.stack([np.cos(th), np.sin(th)])                # 2 × 241

        def pt(v):
            return o + k * np.array([v[0], v[1], 0.0])

        def circle_now():
            P = M() @ unit
            return VMobject(color=WHITE, stroke_width=3).set_points_as_corners(
                [pt(P[:, i]) for i in range(P.shape[1])])

        # Mũi tên là ảnh của v₁, v₂ dưới ma trận đang nội suy
        def arrow_img(j, color):
            return Arrow(o, pt(M() @ V[:, j]), buff=0, color=color, stroke_width=5,
                         max_tip_length_to_length_ratio=0.35)

        circ = always_redraw(circle_now)
        a1 = always_redraw(lambda: arrow_img(0, C1))
        a2 = always_redraw(lambda: arrow_img(1, C2))
        self.play(Write(title), FadeIn(axes), FadeIn(info))
        self.play(Create(circ), GrowArrow(a1), GrowArrow(a2))
        cap = caption("Đường tròn đơn vị, cùng hai hướng v₁ (xanh) và v₂ (đỏ)")
        self.play(FadeIn(cap)); self.wait(1)
        steps = [(V.T, "Bước 1 — Vᵀ: xoay để v₁, v₂ nằm trên hai trục"),
                 (Sig @ V.T, "Bước 2 — Σ: kéo giãn trục 1 gấp σ₁, trục 2 gấp σ₂"),
                 (U @ Sig @ V.T, "Bước 3 — U: xoay elip về hướng u₁, u₂")]
        for Mnext, msg in steps:
            new = caption(msg)
            self.play(FadeOut(cap), FadeIn(new), run_time=0.5)
            cap = new
            state["M0"], state["M1"] = M().copy(), Mnext
            s.set_value(0.0)
            self.play(s.animate.set_value(1.0), run_time=2, rate_func=smooth)
            self.wait(0.6)
        P = A @ unit
        ghost = VMobject(color=C_ACC, stroke_width=6).set_points_as_corners(
            [pt(P[:, i]) for i in range(P.shape[1])])
        new = caption("Khớp với áp thẳng A lên đường tròn: A = UΣVᵀ")
        self.play(FadeOut(cap), FadeIn(new), Create(ghost), run_time=1.5)
        self.wait(3)


class PCAXoayTruc(Scene):
    """PCA = hướng có phương sai chiếu lớn nhất (thương Rayleigh wᵀCw)."""

    def construct(self):
        rng = np.random.default_rng(3)
        th = np.deg2rad(30)
        Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        X = (rng.standard_normal((160, 2)) * np.array([1.6, 0.45])) @ Rm.T
        X -= X.mean(axis=0)
        C = X.T @ X / (len(X) - 1)
        evals, evecs = np.linalg.eigh(C)
        best = np.arctan2(evecs[1, -1], evecs[0, -1]) % PI
        title = T("PCA: tìm hướng có phương sai chiếu lớn nhất", 32).to_edge(UP, buff=0.3)
        axes = NumberPlane(x_range=[-5, 5, 1], y_range=[-3, 3, 1], x_length=9, y_length=5.4,
                           background_line_style={"stroke_opacity": 0.2}).shift(DOWN * 0.4)
        dots = VGroup(*[Dot(axes.c2p(*p), radius=0.035, color=BLUE_B) for p in X])
        ang = ValueTracker(best + 1.2)

        def w():
            a = ang.get_value()
            return np.array([np.cos(a), np.sin(a)])

        line = always_redraw(lambda: Line(axes.c2p(*(-5 * w())), axes.c2p(*(5 * w())),
                                          color=C_ACC, stroke_width=4))
        proj = always_redraw(lambda: VGroup(*[
            Dot(axes.c2p(*((p @ w()) * w())), radius=0.03, color=C_ACC) for p in X[::2]]))
        var = always_redraw(lambda: T(
            f"phương sai chiếu wᵀCw = {w() @ C @ w():.3f}", 26).to_corner(UL, buff=0.5).shift(DOWN * 0.75))
        self.play(Write(title), FadeIn(axes), FadeIn(dots))
        self.play(Create(line), FadeIn(proj), FadeIn(var))
        self.play(ang.animate.set_value(best - 0.9), run_time=3, rate_func=there_and_back_with_pause)
        self.play(ang.animate.set_value(best), run_time=2.5)
        cap = caption(f"Lớn nhất = λₘₐₓ = {evals[-1]:.3f}, tại vector riêng lớn nhất của C")
        self.play(FadeIn(cap))
        self.wait(3)


# ---------------------------------------------------------------- Bài tập 14
def clipped_curve(points, xmax, ymax, to_screen, **style):
    """Nối các điểm (toạ độ dữ liệu) thành các đoạn, chỉ giữ phần trong hộp |x| ≤ xmax, |y| ≤ ymax."""
    runs, cur = [], []
    for p in points:
        if abs(p[0]) <= xmax and abs(p[1]) <= ymax:
            cur.append(to_screen(*p))
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    g = VGroup()
    for r in runs:
        if len(r) > 1:
            g.add(VMobject(**style).set_points_as_corners(r))
    return g


class GDSoDieuKien(Scene):
    """Gradient descent trên f = ½(λ₁x² + λ₂y²), cùng λₘₐₓ = 10 và cùng η = 0.18.

    Trái: λ = (10, 10), κ = 1.  Phải: λ = (10, 0.5), κ = 20.
    Hệ số co mỗi bước theo trục i là |1 − ηλᵢ|: 0.8 (λ = 10) và 0.91 (λ = 0.5),
    nên bên phải chậm hơn vì λₘᵢₙ nhỏ, không phải vì learning rate khác.
    """

    def construct(self):
        title = T("Số điều kiện κ làm gradient descent chậm thế nào", 30).to_edge(UP, buff=0.3)
        self.play(Write(title))
        eta, n_steps = 0.18, 25                                # η < 2/λₘₐₓ = 0.2: vẫn ổn định
        xmax, ymax = 1.5, 2.5
        configs = [((10.0, 10.0), LEFT * 3.5, "κ = 1   (λ = 10, 10)"),
                   ((10.0, 0.5), RIGHT * 3.5, "κ = 20   (λ = 10, 0.5)")]
        groups, paths, finals = [], [], []
        for (l1, l2), c, name in configs:
            ax = Axes(x_range=[-xmax, xmax], y_range=[-ymax, ymax], x_length=3.0, y_length=5.0,
                      tips=False, axis_config={"stroke_opacity": 0.3}).move_to(c + DOWN * 0.35)
            th = np.linspace(0, TAU, 400)
            levels = VGroup()
            for f in (0.4, 2.0, 6.0, 14.0, 25.0):             # mức f = ½(λ₁x² + λ₂y²)
                ring = [(np.sqrt(2 * f / l1) * np.cos(t), np.sqrt(2 * f / l2) * np.sin(t)) for t in th]
                levels.add(clipped_curve(ring, xmax, ymax, ax.c2p, color=BLUE_D, stroke_width=2))
            lab = T(name, 24).next_to(ax, UP, buff=0.08)
            x = np.array([1.0, 2.0])
            pts = [x.copy()]
            for _ in range(n_steps):
                x = x - eta * np.array([l1 * x[0], l2 * x[1]])
                pts.append(x.copy())
            finals.append(float(np.linalg.norm(pts[-1])))
            path = VMobject(color=C_ACC, stroke_width=3).set_points_as_corners([ax.c2p(*p) for p in pts])
            groups.append(VGroup(ax, levels, lab))
            paths.append((path, VGroup(*[Dot(ax.c2p(*p), radius=0.04, color=C_ACC) for p in pts])))
        self.play(*[FadeIn(g) for g in groups])
        self.play(*[Create(p[0]) for p in paths], *[FadeIn(p[1]) for p in paths], run_time=4)
        self.play(FadeIn(caption(
            f"Cùng η = 0.18, 25 bước: κ = 1 còn {finals[0]:.3f}, κ = 20 còn {finals[1]:.2f}")))
        self.wait(1.2)
        self.play(FadeIn(T("Hệ số co mỗi bước: 0.8 theo λ = 10, nhưng 0.91 theo λ = 0.5",
                           24, C_ACC).next_to(title, DOWN, buff=0.12)))
        self.wait(2.5)
