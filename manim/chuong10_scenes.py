"""Các cảnh Manim cho Chương 10 — Mô hình sinh và học tự giám sát.

Render một cảnh (chất lượng 720p):
    manim -qm chuong10_scenes.py ELBOvaKhoangHo

Chỉ dùng Text (không cần LaTeX). Mỗi cảnh dùng đúng số liệu của ví dụ giải mẫu trong sách.
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


def caption(s, size=24):
    t = T(s, size)
    return VGroup(BackgroundRectangle(t, fill_opacity=0.85, buff=0.12), t).to_edge(DOWN, buff=0.25)


def swap_caption(scene, old, text, size=24):
    new = caption(text, size)
    if old is None:
        scene.play(FadeIn(new), run_time=0.5)
    else:
        scene.play(FadeOut(old), run_time=0.25)
        scene.play(FadeIn(new), run_time=0.25)
    return new


def kl1(mu, s):
    return 0.5 * (s ** 2 + mu ** 2 - 1 - np.log(s ** 2))


# ---------------------------------------------------------------- 10.1
class ELBOvaKhoangHo(Scene):
    """Ví dụ 10.2: ELBO = −1.5321, log p(x) = −1.5155, khe hở 0.0166 = KL(q‖posterior)."""

    def construct(self):
        elbo = -0.5 * np.log(2 * np.pi) - 0.5 * (0.25 + 0.64) - kl1(0.5, 0.8)
        logpx = -0.5 * np.log(4 * np.pi) - 0.25
        assert abs(elbo + 1.5321) < 1e-4 and abs(logpx + 1.5155) < 1e-4
        title = T("ELBO và khe hở tới log p(x)", 34).to_edge(UP, buff=0.3)
        with noi(self, "ELBO và khe hở của nó tới log p x. Trên trục số đã được phóng to, log p x bằng trừ một phẩy năm một năm năm."):
            self.play(Write(title))
            pos = lambda v: RIGHT * ((v + 1.56) / 0.06 * 10 - 5) + DOWN * 0.3
            line = Line(pos(-1.56), pos(-1.50), color=GREY)
            ticks = VGroup(*[Line(UP * 0.12, DOWN * 0.12).move_to(pos(v)) for v in (-1.56, -1.54, -1.52, -1.50)])
            tl = VGroup(*[T(f"{v:.2f}", 20, GREY).next_to(pos(v), DOWN, buff=0.2) for v in (-1.56, -1.54, -1.52, -1.50)])
            note = T("trục đã cắt: chỉ vẽ đoạn từ −1.56 đến −1.50", 20, GREY).next_to(tl, DOWN, buff=0.3)
            self.play(Create(line), FadeIn(ticks), FadeIn(tl), FadeIn(note))
            d_log = Dot(pos(logpx), color=C1, radius=0.12)
            l_log = T(f"log p(x) = {logpx:.4f}", 26, C1).move_to([pos(logpx)[0] + 1.0, 2.1, 0])
            d_el = Dot(pos(elbo), color=C2, radius=0.12)
            l_el = T(f"ELBO = {elbo:.4f}", 26, C2).move_to([pos(elbo)[0] - 1.0, 2.1, 0])
            self.play(FadeIn(d_log), FadeIn(l_log))
        with noi(self, "Với phân phối xấp xỉ q tâm không phẩy năm, độ lệch chuẩn không phẩy tám, ELBO bằng trừ một phẩy năm ba hai mốt, thấp hơn log p x. ELBO luôn là cận dưới."):
            cap = swap_caption(self, None, "q = N(0.5, 0.8²) cho ELBO thấp hơn log p(x): ELBO luôn là cận dưới")
            self.play(FadeIn(d_el), FadeIn(l_el))
        with noi(self, "Khe hở bằng không phẩy không một sáu sáu, đúng bằng KL từ q tới posterior thật."):
            gap = BraceBetweenPoints(pos(elbo) + UP * 0.25, pos(logpx) + UP * 0.25, direction=UP, color=C_ACC)
            gl = T(f"khe hở = {logpx - elbo:.4f} = KL(q‖posterior)", 24, C_ACC).next_to(gap, UP, buff=0.1).shift(UP * 0.0)
            gl.move_to([0, 1.0, 0])
            self.play(GrowFromCenter(gap), FadeIn(gl))
            cap = swap_caption(self, cap, "Khe hở đúng bằng KL từ q tới posterior thật N(0.5, 0.5)")
            self.wait(2)
        with noi(self, "Nếu q chính là posterior thật, KL bằng không và ELBO bằng đúng log p x. Tối đa ELBO vừa nâng likelihood vừa kéo q về gần posterior."):
            self.play(FadeOut(gap), FadeOut(gl))
            cap = swap_caption(self, cap, "Đổi q thành posterior thật: KL = 0, ELBO = log p(x)")
            l_el2 = T(f"ELBO = {logpx:.4f}", 26, C2).move_to([-1.2, 2.1, 0])
            self.play(d_el.animate.move_to(pos(logpx) + DOWN * 0.0), Transform(l_el, l_el2), run_time=2)
        self.wait(2.5)


class TaiThamSoHoa(Scene):
    """Ví dụ 10.3: z = μ + σε; cùng kỳ vọng 1.0, độ lệch chuẩn 1.60 (reparam) so với 3.64 (score function)."""

    def construct(self):
        title = T("Reparameterization: cùng kỳ vọng, khác phương sai", 32).to_edge(UP, buff=0.3)
        with noi(self, "Thủ thuật tái tham số hóa viết z bằng mu cộng sigma nhân epsilon. Phần ngẫu nhiên nằm ở epsilon, nên z khả vi theo mu và sigma, và gradient đi qua được."):
            self.play(Write(title))
            f1 = T("z = μ + σ·ε,   ε ~ N(0, 1)", 30).shift(UP * 1.7)
            f2 = T("μ = 0.5, σ = 0.8;  nếu ε = 1.0 thì z = 0.5 + 0.8 = 1.3", 26, C_ACC).next_to(f1, DOWN, buff=0.3)
            self.play(FadeIn(f1), FadeIn(f2))
            cap = swap_caption(self, None, "Ngẫu nhiên nằm ở ε, nên z khả vi theo μ và σ: gradient đi qua")
            self.wait(2)
        with noi(self, "So sánh độ lệch chuẩn của gradient theo mu cho từng mẫu. Hai cách đều có kỳ vọng khoảng một, nhưng tái tham số hóa cho độ lệch chuẩn một phẩy sáu, còn score function là ba phẩy sáu bốn."):
            self.play(FadeOut(f1), FadeOut(f2))
            base = DOWN * 1.9
            sc = 0.8
            data = [("reparam", 1.600, C1, -2.5), ("score function", 3.639, C2, 2.5)]
            bars = VGroup()
            for name, sd, col, x in data:
                r = Rectangle(width=1.6, height=sd * sc, fill_color=col, fill_opacity=0.8, stroke_width=0)
                r.move_to([x, base[1] + sd * sc / 2, 0])
                lab = T(f"{sd:.2f}", 28, col).next_to(r, UP, buff=0.15)
                nm = T(name, 24).next_to(r, DOWN, buff=0.2)
                bars.add(VGroup(r, lab, nm))
            mean_line = DashedLine([-4.5, base[1] + 1.0 * sc, 0], [4.5, base[1] + 1.0 * sc, 0], color=C_ACC)
            ml = T("trung bình ≈ 1.0", 22, C_ACC).move_to([0, base[1] + 1.0 * sc + 0.3, 0])
            sub = T("độ lệch chuẩn của gradient theo μ, mỗi mẫu", 24).shift(UP * 1.7)
            self.play(FadeIn(sub), *[FadeIn(b) for b in bars])
            self.play(Create(mean_line), FadeIn(ml))
        with noi(self, "Score function nhiễu gấp hai phẩy ba lần, nên cần khoảng năm lần số mẫu để đạt cùng độ chính xác."):
            cap = swap_caption(self, cap, "Score function nhiễu gấp 2.3 lần, cần khoảng 5 lần số mẫu cho cùng độ chính xác")
        self.wait(3.5)


# ---------------------------------------------------------------- 10.2
class DiscriminatorToiUu(Scene):
    """Ví dụ 10.4–10.5: D*(x) = σ(2 − 2x), JSD = 0.3368; dịch xa thì JSD → ln 2 còn KL tăng."""

    def construct(self):
        title = T("Discriminator tối ưu giữa N(0, 1) và N(2, 1)", 32).to_edge(UP, buff=0.3)
        ax = Axes(x_range=[-4, 6, 1], y_range=[0, 1.05, 0.5], x_length=9, y_length=3.2, tips=False).shift(DOWN * 0.1)
        pd = lambda x, m: np.exp(-0.5 * (x - m) ** 2) / np.sqrt(2 * np.pi)
        cp = ax.plot(lambda x: pd(x, 0), x_range=[-4, 6], color=BLUE, stroke_width=4)
        cg = ax.plot(lambda x: pd(x, 2), x_range=[-4, 6], color=C2, stroke_width=4)
        ds = ax.plot(lambda x: 1 / (1 + np.exp(2 * x - 2)), x_range=[-4, 6], color=C_ACC, stroke_width=5)
        leg = VGroup(T("pdata", 22, BLUE), T("pg", 22, C2), T("D*(x)", 22, C_ACC)).arrange(RIGHT, buff=0.5).next_to(title, DOWN, buff=0.2)
        xl = VGroup(*[T(str(v).replace("-", "−"), 18, GREY).next_to(ax.c2p(v, 0), DOWN, buff=0.12) for v in range(-4, 7)])
        yl = VGroup(*[T(str(v), 18, GREY).next_to(ax.c2p(-4, v), LEFT, buff=0.12) for v in (0, 0.5, 1)])
        with noi(self, "Phân phối dữ liệu là Gauss tâm không, phân phối sinh là Gauss tâm hai."):
            self.play(Write(title), Create(ax), FadeIn(xl), FadeIn(yl))
            self.play(Create(cp), Create(cg), FadeIn(leg))
        with noi(self, "Discriminator tối ưu bằng p data chia tổng hai mật độ, ở đây là một hàm sigmoid. Nó gần một ở bên trái, gần không ở bên phải."):
            cap = swap_caption(self, None, "D*(x) = pdata / (pdata + pg) = 1 / (1 + exp(2x − 2))")
            self.play(Create(ds), run_time=2)
            xs = [-1, 0, 1, 2, 3]
            vals = [1 / (1 + np.exp(2 * x - 2)) for x in xs]
            assert abs(vals[1] - 0.8808) < 1e-4
            dots = VGroup(*[Dot(ax.c2p(x, v), color=WHITE, radius=0.07) for x, v in zip(xs, vals)])
            tab = T("x = −1, 0, 1, 2, 3  →  D* = " + ", ".join(f"{v:.3f}" for v in vals), 22).next_to(xl, DOWN, buff=0.2)
            self.play(FadeIn(dots), FadeIn(tab))
        with noi(self, "Tại x bằng một, hai mật độ bằng nhau nên discriminator trả về không phẩy năm. Với discriminator này, loss của GAN liên hệ với Jensen Shannon divergence, bằng không phẩy ba ba sáu tám."):
            cap = swap_caption(self, cap, "Tại x = 1 hai mật độ bằng nhau nên D* = 0.5; JSD = 0.3368")
            self.wait(2.5)
        with noi(self, "Dịch phân phối sinh ra xa dần: KL tăng không giới hạn, nhưng Jensen Shannon bão hòa ở ln hai. Khi hai phân phối không chồng nhau, JSD không còn thấy khoảng cách, và gradient biến mất."):
            self.play(FadeOut(VGroup(ax, xl, yl, cp, cg, ds, leg, dots, tab, cap)))
            cap = caption("JSD không còn thấy khoảng cách: nguồn của gradient biến mất")
            head = T("Dịch pg ra xa: JSD bão hòa, KL không", 30).shift(UP * 1.5)
            rows = VGroup(*[T(t, 26, c) for t, c in [
                ("d = 0.5:   JSD = 0.0303   KL = 0.125", WHITE),
                ("d = 2:      JSD = 0.3368   KL = 2", WHITE),
                ("d = 5:      JSD = 0.6759   KL = 12.5", WHITE),
                ("d = 10:    JSD = 0.6931 = ln 2   KL = 50", C_ACC)]]).arrange(DOWN, buff=0.3, aligned_edge=LEFT).shift(DOWN * 0.3)
            self.play(FadeIn(head), FadeIn(rows), FadeIn(cap))
        self.wait(3.5)


# ---------------------------------------------------------------- 10.3
class LichNhieuKhuechTan(Scene):
    """Ví dụ 10.6: ᾱ < 0.5 tại t = 260 (tuyến tính) và t = 497 (cosine), T = 1000."""

    def construct(self):
        Tn = 1000
        beta = np.linspace(1e-4, 0.02, Tn)
        ab_lin = np.cumprod(1 - beta)
        tt = np.arange(Tn + 1) / Tn
        f = np.cos(((tt + 0.008) / 1.008) * np.pi / 2) ** 2
        ab_cos = (f / f[0])[1:]
        t_lin = int(np.argmax(ab_lin < 0.5)) + 1
        t_cos = int(np.argmax(ab_cos < 0.5)) + 1
        assert (t_lin, t_cos) == (260, 497), (t_lin, t_cos)
        title = T("Lịch nhiễu: ᾱ theo t, T = 1000", 34).to_edge(UP, buff=0.3)
        ax = Axes(x_range=[0, 1000, 250], y_range=[0, 1, 0.5], x_length=9, y_length=3.6, tips=False,
                  x_axis_config={"include_numbers": False}).shift(DOWN * 0.1)
        xl = VGroup(*[T(str(v), 18, GREY).next_to(ax.c2p(v, 0), DOWN, buff=0.15) for v in (0, 250, 500, 750, 1000)])
        yl = VGroup(*[T(str(v), 18, GREY).next_to(ax.c2p(0, v), LEFT, buff=0.15) for v in (0, 0.5, 1)])
        with noi(self, "Lịch nhiễu trong mô hình khuếch tán: alpha gạch thể hiện bao nhiêu tín hiệu còn lại ở bước t, với tổng một nghìn bước. So sánh lịch tuyến tính và lịch cosine."):
            self.play(Write(title), Create(ax), FadeIn(xl), FadeIn(yl))
            ts = np.arange(1, Tn + 1)
            g_lin = ax.plot_line_graph(ts[::10], ab_lin[::10], add_vertex_dots=False, line_color=C2, stroke_width=4)
            g_cos = ax.plot_line_graph(ts[::10], ab_cos[::10], add_vertex_dots=False, line_color=C1, stroke_width=4)
            leg = VGroup(T("tuyến tính", 22, C2), T("cosine", 22, C1)).arrange(RIGHT, buff=0.6).next_to(title, DOWN, buff=0.2)
            self.play(Create(g_lin), FadeIn(leg), run_time=2)
            self.play(Create(g_cos), run_time=2)
        with noi(self, "Lịch tuyến tính mất một nửa tín hiệu ngay ở bước hai trăm sáu mươi, còn lịch cosine giữ tới bước bốn trăm chín mươi bảy."):
            half = DashedLine(ax.c2p(0, 0.5), ax.c2p(1000, 0.5), color=GREY)
            d1, d2 = Dot(ax.c2p(t_lin, 0.5), color=C2), Dot(ax.c2p(t_cos, 0.5), color=C1)
            a1 = T("t = 260", 22, C2).next_to(d1, DOWN, buff=0.3).shift(RIGHT * 1.4)
            a2 = T("t = 497", 22, C1).next_to(d2, UP, buff=0.35).shift(RIGHT * 0.4)
            self.play(Create(half), FadeIn(d1), FadeIn(d2), FadeIn(a1), FadeIn(a2))
            cap = swap_caption(self, None, "ᾱ < 0.5 lần đầu: bước 260 (tuyến tính) so với bước 497 (cosine)")
            self.wait(2.5)
        with noi(self, "Với lịch tuyến tính, ba trăm hai mươi sáu bước gần như chỉ còn nhiễu, lãng phí. Lịch cosine chỉ có sáu mươi tư bước như vậy."):
            cap = swap_caption(self, cap, "Số bước có SNR < 0.01: 326 (tuyến tính) so với 64 (cosine)")
        self.wait(3)


class HuongDanKhongPhanLoai(Scene):
    """Ví dụ 10.8 và Lab 3: ε̃ = (1 + w)·0.50 − w·0.20; std lớp 0 co từ 1.338 xuống 0.550."""

    def construct(self):
        title = T("Classifier-free guidance", 34).to_edge(UP, buff=0.3)
        with noi(self, "Classifier free guidance. Mô hình dự đoán nhiễu hai lần: một lần không có điều kiện, được không phẩy hai, và một lần có điều kiện c, được không phẩy năm."):
            self.play(Write(title))
            nl = NumberLine(x_range=[0, 3, 0.5], length=10, include_numbers=False).shift(UP * 0.6)
            nums = VGroup(*[T(f"{v:g}", 20, GREY).next_to(nl.n2p(v), DOWN, buff=0.15) for v in (0, 1, 2, 3)])
            du = Dot(nl.n2p(0.2), color=BLUE, radius=0.12)
            dc = Dot(nl.n2p(0.5), color=C1, radius=0.12)
            lu = T("ε(x, ∅) = 0.20", 22, BLUE).next_to(du, UP, buff=0.5).shift(LEFT * 0.9)
            lc = T("ε(x, c) = 0.50", 22, C1).next_to(dc, UP, buff=0.5).shift(RIGHT * 0.9)
            self.play(Create(nl), FadeIn(nums), FadeIn(du), FadeIn(dc), FadeIn(lu), FadeIn(lc))
        with noi(self, "Kết quả guidance đi từ dự đoán không điều kiện về phía có điều kiện, rồi ngoại suy thêm theo hệ số w. Với w bằng một, ba, bảy phẩy năm, giá trị đẩy ra xa dần."):
            cap = swap_caption(self, None, "ε̃ = (1 + w)·ε(x, c) − w·ε(x, ∅): đi từ ∅ về phía c, rồi ngoại suy thêm")
            res = [(0, 0.50), (1, 0.80), (3, 1.40), (7.5, 2.75)]
            for (w, v) in res:
                assert abs((1 + w) * 0.5 - w * 0.2 - v) < 1e-9
            dot = Dot(nl.n2p(0.5), color=C_ACC, radius=0.14)
            val = T("w = 0:  ε̃ = 0.50", 28, C_ACC).shift(DOWN * 0.6)
            self.play(FadeIn(dot), FadeIn(val))
            for w, v in res[1:]:
                nv = T(f"w = {w:g}:  ε̃ = {v:.2f}", 28, C_ACC).move_to(val)
                self.play(dot.animate.move_to(nl.n2p(v)), Transform(val, nv), run_time=1.4)
                self.wait(0.6)
            self.wait(1)
        with noi(self, "Trong lab, w càng lớn thì mẫu càng bám sát lớp, nhưng độ lệch chuẩn co lại từ một phẩy ba bốn xuống không phẩy năm lăm, trong khi dữ liệu thật là một phẩy bốn hai. Guidance đổi độ đa dạng lấy độ chính xác."):
            self.play(FadeOut(VGroup(nl, nums, du, dc, lu, lc, dot, val)))
            cap = swap_caption(self, cap, "Lab 3: w tăng thì mẫu sát vành hơn nhưng độ lệch chuẩn của lớp 0 co lại")
            sds = [(0, 1.338), (1, 1.115), (3, 0.853), (7, 0.550)]
            bars = VGroup()
            for i, (w, s) in enumerate(sds):
                r = Rectangle(width=s * 3.0, height=0.5, fill_color=C_ACC, fill_opacity=0.8, stroke_width=0)
                r.move_to([-4.2 + s * 1.5, 1.2 - 0.9 * i, 0])
                lab = T(f"w = {w}:  std x = {s:.3f}", 24).move_to([-4.5 + 1.2, 1.2 - 0.9 * i, 0])
                bars.add(r)
            labs = VGroup(*[T(f"w = {w}: std x = {s:.3f}", 24).move_to([4.4, 1.2 - 0.9 * i, 0]) for i, (w, s) in enumerate(sds)])
            ref = T("dữ liệu thật: std x = 1.42", 22, GREY).move_to([0, -2.4, 0])
            self.play(*[GrowFromEdge(b, LEFT) for b in bars], FadeIn(labs), FadeIn(ref))
        self.wait(3.5)


# ---------------------------------------------------------------- 10.4
class MaTranInfoNCE(Scene):
    """Ví dụ 10.9: ma trận tương đồng 4×4, loss trung bình 0.9898 / 0.6816 / 0.0322 cho τ = 1 / 0.5 / 0.1."""

    def construct(self):
        S = np.array([[0.9, 0.1, 0.0, 0.2], [0.2, 0.8, 0.3, 0.1], [0.0, 0.2, 0.7, 0.4], [0.1, 0.1, 0.3, 0.6]])

        def rows(tau):
            L = S / tau
            return -(np.diag(L) - np.log(np.exp(L).sum(1)))
        for tau, m in ((1.0, 0.9898), (0.5, 0.6816), (0.1, 0.0322)):
            assert abs(rows(tau).mean() - m) < 1e-4
        title = T("InfoNCE trên ma trận tương đồng 4×4", 32).to_edge(UP, buff=0.3)
        with noi(self, "InfoNCE trên ma trận tương đồng bốn nhân bốn. Mỗi hàng là một bài phân loại bốn lớp, và đáp án đúng nằm trên đường chéo."):
            self.play(Write(title))
            cells = VGroup()
            for i in range(4):
                for j in range(4):
                    sq = Square(0.8, stroke_width=2, color=C_ACC if i == j else GREY)
                    if i == j:
                        sq.set_fill(C_ACC, opacity=0.18)
                    sq.move_to([-4.5 + 0.85 * j, 1.5 - 0.85 * i, 0])
                    cells.add(VGroup(sq, T(f"{S[i, j]:.1f}", 24).move_to(sq)))
            hl = T("khóa →", 20, GREY).move_to([-4.5 + 0.85 * 1.5, 2.4, 0])
            vl = T("truy vấn", 20, GREY).rotate(PI / 2).move_to([-6.45, 0.4, 0])
            self.play(FadeIn(cells), FadeIn(hl), FadeIn(vl))
            cap = swap_caption(self, None, "Mỗi hàng là một bài phân loại 4 lớp; đáp án đúng nằm trên đường chéo")
            self.wait(2)
        with noi(self, "Với nhiệt độ tau bằng một, loss trung bình là không phẩy chín chín. Giảm tau xuống không phẩy năm, rồi không phẩy một, softmax nhọn dần và loss gần như về không."):
            col_title = T("loss từng hàng", 24, C_ACC).move_to([1.0, 2.4, 0])
            self.play(FadeIn(col_title))
            shown = None
            for tau in (1.0, 0.5, 0.1):
                r = rows(tau)
                txt = VGroup(T(f"τ = {tau:g}", 26, C_ACC),
                             *[T(f"hàng {i + 1}: {v:.4f}", 24) for i, v in enumerate(r)],
                             T(f"trung bình = {r.mean():.4f}", 26, C1)).arrange(DOWN, buff=0.18, aligned_edge=LEFT).move_to([2.3, 0.1, 0])
                if shown is None:
                    self.play(FadeIn(txt))
                else:
                    self.play(FadeOut(shown), FadeIn(txt), run_time=0.8)
                shown = txt
                if tau == 0.1:
                    cap = swap_caption(self, cap, "τ nhỏ: softmax nhọn, loss gần 0. τ lớn: loss tiến về ln 4")
                self.wait(2)
        with noi(self, "Nếu mọi tương đồng bằng nhau, mô hình không phân biệt được gì, và loss bằng ln bốn."):
            last = T("Nếu mọi tương đồng bằng nhau: loss = ln 4 = 1.3863", 24, C2).move_to([1.2, -2.0, 0])
            self.play(FadeIn(last))
        self.wait(3)


class MatNaMaskedVsCausal(Scene):
    """Ví dụ 10.10: palindrome 8 token; causal LM kẹt ở sàn (3/7)·ln 10 = 0.9868, masked LM đạt 0.0023."""

    def construct(self):
        toks = list("abcddcba")
        title = T("Palindrome: cùng Transformer, khác mặt nạ", 32).to_edge(UP, buff=0.3)
        with noi(self, "Chuỗi palindrome tám token: nửa sau là đảo ngược của nửa đầu. So sánh hai cách dùng mặt nạ trên cùng Transformer."):
            self.play(Write(title))

            def row(y, label):
                g = VGroup()
                for i, t in enumerate(toks):
                    sq = Square(0.8, stroke_width=2, color=WHITE).move_to([-3.5 + 1.0 * i, y, 0])
                    g.add(VGroup(sq, T(t, 28).move_to(sq)))
                lab = T(label, 24, C_ACC).move_to([0, y + 0.85, 0])
                return g, lab
            r1, l1 = row(1.5, "Causal LM: mỗi ô chỉ nhìn bên trái, đoán ô tiếp theo")
            self.play(FadeIn(r1), FadeIn(l1))
        with noi(self, "Causal LM chỉ nhìn bên trái. Nửa sau đoán được từ nửa đầu, nhưng ba token ở nửa đầu thì hoàn toàn ngẫu nhiên. Loss kẹt ở sàn khoảng không phẩy chín tám sáu tám, và đo được không phẩy chín chín sáu sáu."):
            cap = swap_caption(self, None, "Nửa sau là đảo của nửa đầu, nên token 5–8 suy ra được từ token 4–1")
            self.play(*[r1[i][0].animate.set_fill(C2, opacity=0.6) for i in (1, 2, 3)])
            n1 = T("3 ô đỏ không đoán được: ln 10 = 2.3026 mỗi ô", 22, C2).move_to([0, 0.6, 0])
            n1b = T("sàn = (3/7)·ln 10 = 0.9868;  đo được 0.9966", 24, C_ACC).move_to([0, 0.15, 0])
            assert abs(3 / 7 * np.log(10) - 0.9868) < 1e-4
            self.play(FadeIn(n1), FadeIn(n1b))
            self.wait(2)
        with noi(self, "Masked LM che một ô và nhìn cả hai phía. Che ô thứ ba thì ô đối xứng thứ sáu vẫn thấy, nên đoán gần như chắc chắn, loss chỉ còn không phẩy không không hai."):
            r2, l2 = row(-1.6, "Masked LM: che một ô, nhìn cả hai phía")
            self.play(FadeIn(r2), FadeIn(l2))
            cap = swap_caption(self, cap, "Che ô thứ 3: ô đối xứng (thứ 6) vẫn thấy, nên đoán chắc chắn")
            q = T("?", 30, C2).move_to(r2[2][1])
            self.play(Transform(r2[2][1], q), r2[2][0].animate.set_fill(C2, opacity=0.6))
            arrow = CurvedArrow(r2[5][0].get_bottom() + DOWN * 0.05, r2[2][0].get_bottom() + DOWN * 0.05, angle=-1.0, color=C1)
            self.play(Create(arrow))
            n2 = T("loss = 0.0023", 28, C1).move_to([0, -2.75, 0])
            self.play(FadeIn(n2))
        self.wait(3.5)
