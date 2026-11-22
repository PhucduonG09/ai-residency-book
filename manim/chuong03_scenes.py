"""Các cảnh Manim cho Chương 3 — Xác suất, thống kê, lý thuyết thông tin.

Render một cảnh (chất lượng 720p):
    manim -qm chuong03_scenes.py LuoiBayes

Chỉ dùng Text (không cần LaTeX), nên chỉ cần Manim Community + ffmpeg.
Không dùng include_numbers của Axes (nó gọi LaTeX); nhãn trục viết bằng Text.
Mỗi cảnh dùng đúng số liệu của ví dụ giải mẫu trong sách.
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


def swap_cap(old, new):
    """Phụ đề cũ mờ hẳn rồi phụ đề mới mới hiện: không bao giờ chồng chữ."""
    return Succession(FadeOut(old, run_time=0.3), FadeIn(new, run_time=0.3))


def npdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def tick_labels(ax, xs=(), ys=(), size=20, fmt="{:g}"):
    """Nhãn số cho trục, dùng Text thay cho include_numbers."""
    g = VGroup()
    for x in xs:
        g.add(T(fmt.format(x), size).next_to(ax.c2p(x, ax.y_range[0]), DOWN, buff=0.12))
    for y in ys:
        g.add(T(fmt.format(y), size).next_to(ax.c2p(ax.x_range[0], y), LEFT, buff=0.12))
    return g


# ---------------------------------------------------------------- 3.1
class LuoiBayes(Scene):
    """Ví dụ 3.1: 1000 bình luận, tỉ lệ nền 2%, recall 90%, FPR 5% ⇒ chỉ 18/67 cờ là đúng."""

    def construct(self):
        rng = np.random.default_rng(7)
        n_col, n_row, side = 50, 20, 0.19
        idx = rng.permutation(1000)
        toxic = set(idx[:20])
        toxic_flag = set(idx[:18])                       # 18 trong 20 bình luận độc bị gắn cờ
        clean_flag = set(idx[20:69])                     # 49 trong 980 bình luận sạch bị gắn cờ
        title = T("Tỉ lệ nền: vì sao recall 90% vẫn cho precision 27%", 30).to_edge(UP, buff=0.3)
        cells = VGroup(*[Square(side, stroke_width=0.6, stroke_color=GREY_D,
                                fill_color=GREY_E, fill_opacity=1) for _ in range(1000)])
        cells.arrange_in_grid(n_row, n_col, buff=0.04).move_to(UP * 0.15)
        self.play(Write(title), FadeIn(cells))
        cap = caption("1000 bình luận. Đỏ = độc hại (2% ⇒ 20 bình luận)")
        self.play(FadeIn(cap), *[cells[i].animate.set_fill(C2) for i in toxic], run_time=1.2)
        self.wait(1)
        new = caption("Bộ lọc gắn cờ (viền vàng): 18 bình luận độc + 49 bình luận sạch")
        flagged = list(toxic_flag | clean_flag)
        self.play(swap_cap(cap, new),
                  *[cells[i].animate.set_stroke(C_ACC, width=3) for i in flagged], run_time=1.5)
        cap = new
        self.wait(1.2)
        dim = [cells[i].animate.set_opacity(0.15) for i in range(1000) if i not in toxic_flag | clean_flag]
        self.play(*dim, run_time=1)
        tally = VGroup(T("bị gắn cờ: 67", 26, C_ACC), T("thật sự độc: 18", 26, C2),
                       T("P(độc ∣ cờ) = 18/67 ≈ 0.27", 28)).arrange(RIGHT, buff=0.6)
        tally.next_to(cells, DOWN, buff=0.3)
        self.play(FadeIn(tally))
        new = caption("Thêm bộ lọc thứ hai độc lập cũng gắn cờ ⇒ posterior ≈ 0.87")
        self.play(swap_cap(cap, new))
        self.wait(3)


# ---------------------------------------------------------------- 3.2
class PhuongSaiEnsemble(Scene):
    """Ví dụ 3.3: Var(trung bình K model) = ρσ² + (1 − ρ)σ²/K, σ² = 1."""

    def construct(self):
        title = T("Trung bình K model: phương sai dừng ở ρσ²", 32).to_edge(UP, buff=0.3)
        ax = Axes(x_range=[1, 20, 1], y_range=[0, 1.05, 0.25], x_length=9, y_length=4.4,
                  tips=False, axis_config={"stroke_opacity": 0.6}).shift(UP * 0.05 + LEFT * 0.6)
        labels = tick_labels(ax, xs=(1, 5, 10, 15, 20), ys=(0, 0.5, 1))
        xl = T("số model K", 22).next_to(ax.c2p(20, 0), RIGHT, buff=0.3).shift(DOWN * 0.35)
        yl = T("phương sai", 22).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.55)
        self.play(Write(title), Create(ax), FadeIn(labels), FadeIn(xl), FadeIn(yl))
        cols = {0.0: BLUE, 0.5: C1, 0.9: C_ACC}
        for rho, col in cols.items():
            f = lambda k, r=rho: r + (1 - r) / k
            curve = ax.plot(f, x_range=[1, 20], color=col, stroke_width=4)
            floor = DashedLine(ax.c2p(1, rho), ax.c2p(20, rho), color=col, stroke_opacity=0.6)
            dot = Dot(ax.c2p(5, f(5)), color=col)
            val = T(f"{f(5):.2f}", 20, col).next_to(dot, UR, buff=0.08)
            name = T(f"ρ = {rho:g}", 22, col).next_to(ax.c2p(20, f(20)), RIGHT, buff=0.15)
            self.play(Create(curve), FadeIn(floor), run_time=1.3)
            self.play(FadeIn(dot), FadeIn(val), FadeIn(name), run_time=0.5)
        cap = caption("Tại K = 5: 0.20, 0.60, 0.92. Với ρ = 0.5, cả 1000 model cũng chỉ còn 0.5005")
        self.play(FadeIn(cap))
        self.wait(3)


# ---------------------------------------------------------------- 3.3
class ElipGauss(Scene):
    """Ví dụ 3.6: N(0, Σ), Σ = [[2, 1], [1, 2]]; (1, 1) và (1, −1) cùng xa gốc theo Euclid."""

    def construct(self):
        S = np.array([[2.0, 1.0], [1.0, 2.0]])
        L = np.linalg.cholesky(S)
        evals, evecs = np.linalg.eigh(S)                 # 1 theo (1, −1), 3 theo (1, 1)
        title = T("Gauss 2 chiều: elip đồng mức theo vector riêng của Σ", 30).to_edge(UP, buff=0.3)
        plane = NumberPlane(x_range=[-4, 4, 1], y_range=[-3, 3, 1], x_length=8, y_length=6,
                            background_line_style={"stroke_opacity": 0.25}).shift(DOWN * 0.35 + LEFT * 2.2)
        c2p = plane.c2p
        Z = np.random.default_rng(0).standard_normal((350, 2)) @ L.T   # x = Lz
        dots = VGroup(*[Dot(c2p(*p), radius=0.025, color=BLUE_B) for p in Z if np.abs(p).max() < 2.9])
        info = VGroup(T("Σ = [[2, 1], [1, 2]]", 24),
                      T("λ = 3 theo (1, 1)", 22, C1),
                      T("λ = 1 theo (1, −1)", 22, C2),
                      T("mẫu: x = Lz, Σ = LLᵀ", 22)).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        info.to_edge(RIGHT, buff=0.35).align_to(UP * 2.6, UP)
        self.play(Write(title), FadeIn(plane), FadeIn(info))
        self.play(FadeIn(dots, lag_ratio=0.01), run_time=1.5)

        def ellipse(c, col):
            pts = [c2p(*(np.sqrt(c) * evecs @ (np.sqrt(evals) * np.array([np.cos(t), np.sin(t)]))))
                   for t in np.linspace(0, TAU, 120)]
            return VMobject(color=col, stroke_width=4).set_points_smoothly(pts)

        ax1 = Arrow(c2p(0, 0), c2p(*(np.sqrt(3) * evecs[:, 1])), buff=0, color=C1, stroke_width=5)
        ax2 = Arrow(c2p(0, 0), c2p(*(1.0 * evecs[:, 0])), buff=0, color=C2, stroke_width=5)
        self.play(GrowArrow(ax1), GrowArrow(ax2))
        e1, e2 = ellipse(2 / 3, C_ACC), ellipse(2.0, ORANGE)
        p1, p2 = Dot(c2p(1, 1), color=C_ACC, radius=0.09), Dot(c2p(1, -1), color=ORANGE, radius=0.09)
        circ = DashedVMobject(Circle(radius=np.sqrt(2) * plane.x_length / 8, color=WHITE).move_to(c2p(0, 0)))
        cap = caption("Hai điểm (1, 1) và (1, −1): cùng khoảng cách Euclid √2 (đường tròn nét đứt)")
        self.play(FadeIn(p1), FadeIn(p2), Create(circ), FadeIn(cap))
        self.wait(1.2)
        self.play(Create(e1), Create(e2), run_time=1.5)
        res = VGroup(T("Mahalanobis², mật độ:", 20),
                     T("(1, 1):  0.667,  0.066", 20, C_ACC),
                     T("(1, −1): 2.0,  0.034", 20, ORANGE)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        res.next_to(info, DOWN, buff=0.45, aligned_edge=LEFT)
        if res.get_right()[0] > 6.9:
            res.shift(LEFT * (res.get_right()[0] - 6.9))
        new = caption("Nhưng nằm trên hai elip khác nhau: đi ngược hướng tương quan thì “lạ” hơn")
        self.play(FadeIn(res), swap_cap(cap, new))
        self.wait(3)


# ---------------------------------------------------------------- 3.4
class PriorKeoVe0(Scene):
    """Ví dụ 3.8: likelihood ∝ N(w; 31/14, 1/14), prior N(0, τ²), posterior đỉnh 31/(14 + 1/τ²)."""

    def construct(self):
        title = T("MAP: prior Gauss kéo w về 0 (ridge)", 32).to_edge(UP, buff=0.3)
        ax = Axes(x_range=[-1, 3.5, 0.5], y_range=[0, 1.1, 0.5], x_length=10, y_length=4.2,
                  tips=False, axis_config={"stroke_opacity": 0.6}).shift(DOWN * 0.45)
        labels = tick_labels(ax, xs=(-1, 0, 1, 2, 3))
        xl = T("w", 24).next_to(ax.c2p(3.5, 0), RIGHT, buff=0.15)
        self.play(Write(title), Create(ax), FadeIn(labels), FadeIn(xl))
        w_mle, s_lik = 31 / 14, np.sqrt(1 / 14)
        tau2 = ValueTracker(0.1)

        def bump(m, s, col, width=4):
            return ax.plot(lambda w: np.exp(-0.5 * ((w - m) / s) ** 2), x_range=[-1, 3.5, 0.01],
                           color=col, stroke_width=width)

        lik = bump(w_mle, s_lik, BLUE)
        lik_l = T("likelihood: đỉnh 2.214 (MLE)", 22, BLUE).next_to(ax.c2p(w_mle, 1.0), UP, buff=0.12)
        self.play(Create(lik), FadeIn(lik_l))
        prior = always_redraw(lambda: bump(0, np.sqrt(tau2.get_value()), C1))
        post = always_redraw(lambda: bump(31 / (14 + 1 / tau2.get_value()),
                                          np.sqrt(1 / (14 + 1 / tau2.get_value())), C_ACC, 6))
        prior_l = T("prior N(0, τ²)", 22, C1).to_corner(UL, buff=0.4).shift(DOWN * 0.9)
        post_l = T("posterior: đỉnh = MAP", 22, C_ACC).next_to(prior_l, DOWN, aligned_edge=LEFT)
        scale_note = T("(mọi đường vẽ với đỉnh = 1)", 18, GREY_B).next_to(post_l, DOWN, aligned_edge=LEFT)
        readout = always_redraw(lambda: T(
            f"τ² = {tau2.get_value():.3f}   λ = σ²/τ² = {1 / tau2.get_value():.1f}   "
            f"ŵ MAP = {31 / (14 + 1 / tau2.get_value()):.3f}", 24).next_to(title, DOWN, buff=0.2))
        self.play(Create(prior), FadeIn(prior_l))
        self.play(Create(post), FadeIn(post_l), FadeIn(scale_note), FadeIn(readout))
        cap = caption("σ² = 1, τ² = 0.1 ⇒ λ = 10, ŵ MAP = 31/24 ≈ 1.292")
        self.play(FadeIn(cap))
        self.wait(1.5)
        new = caption("τ² tăng dần (prior rộng dần) ⇒ λ nhỏ dần")
        self.play(swap_cap(cap, new), tau2.animate.set_value(1.0), run_time=2.5)
        cap = new
        new = caption("τ² = 1: prior rộng ⇒ λ = 1 nhỏ ⇒ MAP gần MLE")
        self.play(swap_cap(cap, new))
        self.wait(1.2)
        cap = new
        new = caption("τ² giảm dần (prior hẹp dần) ⇒ λ lớn dần")
        self.play(swap_cap(cap, new), tau2.animate.set_value(0.01), run_time=2.5)
        cap = new
        new = caption("τ² = 0.01: prior hẹp ⇒ λ = 100 lớn ⇒ w bị ép mạnh về 0")
        self.play(swap_cap(cap, new))
        self.wait(1.2)
        cap = new
        new = caption("Quay lại τ² = 0.1: λ = 10, ŵ MAP = 31/24 ≈ 1.292")
        self.play(swap_cap(cap, new), tau2.animate.set_value(0.1), run_time=2)
        self.wait(2)


# ---------------------------------------------------------------- 3.5A
class KLKhongDoiXung(Scene):
    """Fit một Gauss q vào hỗn hợp p = ½N(−2, 0.6²) + ½N(2, 0.6²) theo hai chiều KL."""

    def construct(self):
        title = T("KL không đối xứng: phủ mọi đỉnh hay bám một đỉnh", 30).to_edge(UP, buff=0.3)
        ax = Axes(x_range=[-5, 5, 1], y_range=[0, 1.0, 0.2], x_length=10.5, y_length=4.0,
                  tips=False, axis_config={"stroke_opacity": 0.6}).shift(DOWN * 0.7)
        labels = tick_labels(ax, xs=(-4, -2, 0, 2, 4))
        p = lambda x: 0.5 * npdf(x, -2, 0.6) + 0.5 * npdf(x, 2, 0.6)
        pc = ax.plot(p, x_range=[-5, 5, 0.02], color=WHITE, stroke_width=4)
        pa = ax.get_area(pc, x_range=[-5, 5], color=GREY_B, opacity=0.25)
        pl = T("p: hai đỉnh tại −2 và 2", 22).to_corner(UL, buff=0.4).shift(DOWN * 0.75)
        self.play(Write(title), Create(ax), FadeIn(labels))
        self.play(Create(pc), FadeIn(pa), FadeIn(pl))
        # nghiệm số (tính bằng scipy.optimize trên lưới, xem sách mục 3.5)
        qf = ax.plot(lambda x: npdf(x, 0.0, 2.088), x_range=[-5, 5, 0.02], color=C1, stroke_width=5)
        lf = VGroup(T("q = argmin KL(p‖q): N(0, 2.09²)", 22, C1),
                    T("mode-covering: trải rộng, phủ cả hai đỉnh", 20, C1)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(pl, DOWN, aligned_edge=LEFT)
        cap = caption("Forward KL phạt nặng khi q ≈ 0 ở nơi p > 0 ⇒ q phải trải ra phủ cả hai đỉnh")
        self.play(Create(qf), FadeIn(lf), FadeIn(cap), run_time=1.5)
        self.wait(2.2)
        qr = ax.plot(lambda x: npdf(x, 1.997, 0.605), x_range=[-5, 5, 0.02], color=C_ACC, stroke_width=5)
        lr = VGroup(T("q = argmin KL(q‖p): N(2.0, 0.60²)", 22, C_ACC),
                    T("mode-seeking: bám chặt một đỉnh", 20, C_ACC)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).to_corner(UR, buff=0.4).shift(DOWN * 1.2)
        new = caption("Reverse KL phạt nặng khi q > 0 ở nơi p ≈ 0 ⇒ q chọn một đỉnh, bỏ đỉnh kia")
        self.play(Create(qr), FadeIn(lr), swap_cap(cap, new), run_time=1.5)
        cap = new
        self.wait(2.2)
        new = caption("Variational inference và VAE tối thiểu KL(q‖p): xấp xỉ thường hẹp hơn p")
        self.play(swap_cap(cap, new))
        cap = new
        self.wait(1.8)
        nl = VGroup(T("KL(p‖q) = 0.555 nat, KL(q‖p) = 1.273 nat", 18, C1),
                    T("(q reverse cao hơn p: mật độ phải có diện tích 1)", 16, GREY_B)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(lf, DOWN, aligned_edge=LEFT, buff=0.2)
        nr = T("KL(q‖p) = 0.692 nat (≈ ln 2), KL(p‖q) ≈ 10 nat", 18, C_ACC)
        nr.next_to(lr, DOWN, aligned_edge=RIGHT, buff=0.2).align_to(lr, RIGHT)
        nums = VGroup(nl, nr)
        self.play(FadeIn(nums))
        self.wait(3)


# ---------------------------------------------------------------- 3.5B
class PairedBootstrap(Scene):
    """Ví dụ 3.11: bảng chéo 820/30/15/135; bootstrap có cặp vs không cặp (seed 0, 10 000 lần)."""

    def construct(self):
        a = np.r_[np.ones(820), np.ones(30), np.zeros(15), np.zeros(135)]
        b = np.r_[np.ones(820), np.zeros(30), np.ones(15), np.zeros(135)]
        rng = np.random.default_rng(0)
        i = rng.integers(0, 1000, (10000, 1000))
        j = rng.integers(0, 1000, (10000, 1000))
        d_pair = a[i].mean(1) - b[i].mean(1)
        d_unp = a[i].mean(1) - b[j].mean(1)
        title = T("So sánh hai model trên cùng 1000 câu: có cặp vs không cặp", 28).to_edge(UP, buff=0.3)
        self.play(Write(title))
        # bảng chéo 2 × 2
        vals = [["", "B đúng", "B sai"], ["A đúng", "820", "30"], ["A sai", "15", "135"]]
        cells = VGroup()
        for r, row in enumerate(vals):
            for c, v in enumerate(row):
                col = C_ACC if (r, c) in [(1, 2), (2, 1)] else WHITE
                box = Rectangle(width=1.35, height=0.6, stroke_width=1.5 if r and c else 0)
                cells.add(VGroup(box, T(v, 22, col).move_to(box)))
        cells.arrange_in_grid(3, 3, buff=0).to_corner(UL, buff=0.4).shift(DOWN * 0.9)
        note = VGroup(T("Chỉ 45 câu hai model", 20, C_ACC), T("bất đồng mới mang", 20, C_ACC),
                      T("thông tin về hiệu số", 20, C_ACC),
                      T("McNemar: p ≈ 0.036", 22)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        note.next_to(cells, DOWN, buff=0.35, aligned_edge=LEFT)
        self.play(FadeIn(cells), FadeIn(note))
        ax = Axes(x_range=[-0.04, 0.07, 0.01], y_range=[0, 1.05, 0.5], x_length=7.6, y_length=4.4,
                  tips=False, axis_config={"stroke_opacity": 0.6}).to_edge(RIGHT, buff=0.45).shift(DOWN * 0.3)
        xl = VGroup(*[T(f"{x:g}", 18).next_to(ax.c2p(x, 0), DOWN, buff=0.1) for x in (-0.02, 0, 0.02, 0.04, 0.06)])
        cap_x = T("acc(A) − acc(B)", 20).next_to(ax.c2p(0.07, 0.75), UP + LEFT, buff=0.1)
        zero = DashedLine(ax.c2p(0, 0), ax.c2p(0, 1.05), color=WHITE)
        self.play(Create(ax), FadeIn(xl), Create(zero), FadeIn(cap_x))
        edges = np.arange(-0.04, 0.0701, 0.002)

        h_ref = np.histogram(d_pair, bins=edges)[0].max()   # cùng thang cho cả hai histogram

        def hist(d, col):
            h, _ = np.histogram(d, bins=edges)
            h = h / h_ref
            return VGroup(*[Rectangle(width=ax.x_length * 0.002 / 0.11 * 0.95, height=max(v, 1e-3) * ax.y_length / 1.05,
                                      fill_color=col, fill_opacity=0.55, stroke_width=0)
                            .move_to(ax.c2p((lo + 0.001), 0), aligned_edge=DOWN)
                            for lo, v in zip(edges[:-1], h)])

        hp, hu = hist(d_pair, C1), hist(d_unp, C2)
        lo_p, hi_p = np.percentile(d_pair, [2.5, 97.5])
        lo_u, hi_u = np.percentile(d_unp, [2.5, 97.5])
        cap = caption(f"Có cặp (cùng chỉ số): CI 95% = [{lo_p:.3f}, {hi_p:.3f}], không chứa 0")
        lp = T("có cặp: SE ≈ 0.0067", 22, C1).next_to(note, DOWN, buff=0.4, aligned_edge=LEFT)
        self.play(LaggedStart(*[GrowFromEdge(r, DOWN) for r in hp], lag_ratio=0.03), FadeIn(lp), FadeIn(cap),
                  run_time=2.5)
        br_p = Line(ax.c2p(lo_p, 1.02), ax.c2p(hi_p, 1.02), color=C1, stroke_width=6)
        self.play(Create(br_p))
        self.wait(1.2)
        new = caption(f"Không cặp (chỉ số độc lập): CI = [{lo_u:.3f}, {hi_u:.3f}], vắt qua 0")
        lu = T("không cặp: SE ≈ 0.016", 22, C2).next_to(lp, DOWN, aligned_edge=LEFT)
        self.play(swap_cap(cap, new), FadeIn(lu),
                  LaggedStart(*[GrowFromEdge(r, DOWN) for r in hu], lag_ratio=0.03), run_time=2.5)
        br_u = Line(ax.c2p(lo_u, 0.5), ax.c2p(hi_u, 0.5), color=C2, stroke_width=6)
        self.play(Create(br_u))
        self.wait(3)
