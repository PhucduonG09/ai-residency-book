"""Các cảnh Manim cho Chương 6 — Mạng nơ-ron từ con số 0.

Render một cảnh (chất lượng 720p):
    manim -qm chuong06_scenes.py LanNguocMLP

Chỉ dùng Text (không cần LaTeX), nên chỉ cần Manim Community + ffmpeg.
Mỗi cảnh dùng đúng số liệu của ví dụ giải mẫu trong sách.
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
        # mờ hẳn phụ đề cũ rồi mới hiện phụ đề mới, tránh hai phụ đề chồng nhau
        scene.play(FadeOut(old), run_time=0.2)
        scene.play(FadeIn(new), run_time=0.3)
    return new


# ---------------------------------------------------------------- 6.1
class SupDoTuyenTinh(Scene):
    """Ví dụ 6.1: W₂W₁ = (−1, 4); chèn ReLU vào thì đường thẳng gập thành gấp khúc."""

    def construct(self):
        title = T("Hai lớp tuyến tính chỉ là một lớp", 34).to_edge(UP, buff=0.3)
        with noi(self, "Xét một mạng hai lớp tuyến tính, không có hàm kích hoạt: y bằng W hai nhân W một nhân x."):
            self.play(Write(title))

            def mat(rows, color=WHITE):
                m = VGroup(*[T(v, 30, color).move_to([0.75 * j, -0.6 * i, 0])
                             for i, r in enumerate(rows) for j, v in enumerate(r)])
                h = m.height + 0.35
                lb = Line(UP * h / 2, DOWN * h / 2, color=color).next_to(m, LEFT, buff=0.18)
                rb = Line(UP * h / 2, DOWN * h / 2, color=color).next_to(m, RIGHT, buff=0.18)
                ticks = VGroup(*[Line(ORIGIN, RIGHT * 0.12 * s, color=color).move_to(b.get_end() if e else b.get_start(),
                                                                                    aligned_edge=LEFT if s > 0 else RIGHT)
                                 for b, s in ((lb, 1), (rb, -1)) for e in (0, 1)])
                return VGroup(lb, rb, ticks, m)

            W2 = mat([["1", "0", "2"]], C2)
            W1 = mat([["1", "2"], ["0", "1"], ["−1", "1"]], C1)
            eq = T("=", 36)
            R = mat([["−1", "4"]], C_ACC)
            row = VGroup(W2, W1, eq, R).arrange(RIGHT, buff=0.4).shift(UP * 0.9)
            labs = VGroup(T("W₂ (1×3)", 22, C2).next_to(W2, UP, buff=0.25),
                          T("W₁ (3×2)", 22, C1).next_to(W1, UP, buff=0.25),
                          T("W₂W₁ (1×2)", 22, C_ACC).next_to(R, UP, buff=0.25))
            self.play(FadeIn(W2), FadeIn(W1), FadeIn(labs[0]), FadeIn(labs[1]))
            cap = swap_caption(self, None, "Mạng 2→3→1 không có kích hoạt: y = W₂(W₁x)")
            self.wait(0.8)
        with noi(self, "Nhưng tích W hai W một chỉ là một ma trận một nhân hai, bằng trừ một, bốn. Cả mạng sụp về đúng một neuron tuyến tính."):
            self.play(FadeIn(eq), FadeIn(R), FadeIn(labs[2]))
            cap = swap_caption(self, cap, "= (W₂W₁)x: đúng bằng một neuron tuyến tính có trọng số (−1, 4)")
            self.wait(1.5)
        self.play(FadeOut(VGroup(row, labs)))

        # hai đồ thị: không kích hoạt vs có ReLU (h = ReLU(W₁x), x = (t, 0), y = W₂h)
        with noi(self, "Cho đầu vào chạy dọc một trục và vẽ đầu ra. Không có kích hoạt thì luôn là đường thẳng."):
            axl = Axes(x_range=[-2, 2, 1], y_range=[-3, 3, 1], x_length=5, y_length=3.6, tips=False).shift(LEFT * 3.3 + DOWN * 0.3)
            axr = axl.copy().shift(RIGHT * 6.6)
            tl = T("không kích hoạt: y = −t", 22).next_to(axl, UP, buff=0.15)
            tr = T("có ReLU: y = ReLU(t) + 2·ReLU(−t)", 22).next_to(axr, UP, buff=0.15)
            lin = axl.plot(lambda t: -t, x_range=[-2, 2], color=C_ACC, stroke_width=5)
            relu = axr.plot(lambda t: max(t, 0) + 2 * max(-t, 0), x_range=[-1.5, 2], color=C_ACC, stroke_width=5)
            self.play(Create(axl), Create(axr), FadeIn(tl), FadeIn(tr))
            cap = swap_caption(self, cap, "Cho x = (t, 0) chạy trên một trục, vẽ đầu ra y theo t")
            self.play(Create(lin), Create(relu), run_time=2)
        with noi(self, "Chèn ReLU vào giữa, đường thẳng gập thành đường gấp khúc. Chính hàm phi tuyến cho mạng sức biểu diễn."):
            cap = swap_caption(self, cap, "Không có phi tuyến: luôn là đường thẳng. Có ReLU: đường gấp khúc")
        self.wait(3)


class XORHaiNeuron(Scene):
    """Ví dụ 6.2: h₁ = ReLU(x₁ + x₂), h₂ = ReLU(x₁ + x₂ − 1), y = h₁ − 2h₂."""

    def construct(self):
        title = T("XOR với hai neuron ReLU", 34).to_edge(UP, buff=0.3)
        with noi(self, "Bài toán XOR: hai điểm xanh có nhãn không, hai điểm đỏ có nhãn một. Không có đường thẳng nào tách được chúng."):
            self.play(Write(title))
            axl = Axes(x_range=[-0.5, 1.5, 1], y_range=[-0.5, 1.5, 1], x_length=4.2, y_length=4.2,
                       tips=False).shift(LEFT * 3.4 + DOWN * 0.35)
            axr = Axes(x_range=[-0.5, 2.5, 1], y_range=[-0.5, 1.5, 1], x_length=5.2, y_length=3.4,
                       tips=False).shift(RIGHT * 3.0 + DOWN * 0.35)
            ll = T("đầu vào (x₁, x₂)", 22).next_to(axl, UP, buff=0.15)
            lr = T("không gian ẩn (h₁, h₂)", 22).next_to(axr, UP, buff=0.15)
            pts = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]
            col = {0: BLUE, 1: C2}

            def hid(p):
                s = p[0] + p[1]
                return (max(s, 0), max(s - 1, 0))

            dl = VGroup(*[Dot(axl.c2p(*p), radius=0.12, color=col[c]) for p, c in pts])
            names = VGroup(*[T(f"({p[0]},{p[1]})", 18).next_to(axl.c2p(*p), DL if p[0] == 0 else DR, buff=0.08)
                             for p, _ in pts])
            self.play(Create(axl), FadeIn(ll), FadeIn(dl), FadeIn(names))
            cap = swap_caption(self, None, "Xanh: XOR = 0, đỏ: XOR = 1. Không đường thẳng nào tách được")
            self.wait(1.5)
        with noi(self, "Dùng hai neuron ReLU để biến đổi đầu vào sang không gian ẩn. Hai điểm đỏ dồn về cùng một chỗ, còn điểm một, một bị đẩy ra xa."):
            self.play(Create(axr), FadeIn(lr))
            cap = swap_caption(self, cap, "h₁ = ReLU(x₁ + x₂),  h₂ = ReLU(x₁ + x₂ − 1)")
            moved = VGroup(*[Dot(axl.c2p(*p), radius=0.12, color=col[c]) for p, c in pts])
            self.add(moved)
            self.play(*[moved[i].animate.move_to(axr.c2p(*hid(p))) for i, (p, _) in enumerate(pts)], run_time=2)
            hn = VGroup(T("(0,0)", 18).next_to(axr.c2p(0, 0), DL, buff=0.08),
                        T("(0,1),(1,0) → (1,0)", 18).next_to(axr.c2p(1, 0), DOWN, buff=0.2),
                        T("(1,1) → (2,1)", 18).next_to(axr.c2p(2, 1), UP, buff=0.15))
            self.play(FadeIn(hn))
            cap = swap_caption(self, cap, "Hai điểm đỏ trùng nhau tại (1, 0); điểm (1,1) bị đẩy tới (2, 1)")
            self.wait(1.5)
        # đường y = h₁ − 2h₂ = 0.5
        with noi(self, "Trong không gian ẩn, một đường thẳng đã tách được đỏ khỏi xanh. Lớp cuối y bằng h một trừ hai h hai cho đúng bảng XOR."):
            sep = axr.plot(lambda h1: (h1 - 0.5) / 2, x_range=[0.2, 2.5], color=C_ACC, stroke_width=4)
            self.play(Create(sep))
            cap = swap_caption(self, cap, "Lớp cuối y = h₁ − 2h₂: đường y = 0.5 tách đỏ khỏi xanh")
            self.wait(1)
            table = T("y: (0,0)→0  (0,1)→1  (1,0)→1  (1,1)→2−2=0", 22, C_ACC).next_to(title, DOWN, buff=0.2)
            self.play(FadeIn(table))
            self.wait(2.5)


class XapXiTungKhuc(Scene):
    """Mục 6.1: nội suy tuyến tính sin x trên [0, π] với K khúc; sai số lớn nhất giảm ~4 lần khi K gấp đôi."""

    def construct(self):
        title = T("Mạng ReLU = hàm tuyến tính từng khúc", 34).to_edge(UP, buff=0.3)
        ax = Axes(x_range=[0, PI, 1], y_range=[0, 1.1, 0.5], x_length=9, y_length=4.4, tips=False).shift(DOWN * 0.3)
        sine = ax.plot(np.sin, x_range=[0, PI], color=BLUE, stroke_width=4)
        with noi(self, "Mạng ReLU biểu diễn hàm tuyến tính từng khúc. Thử xấp xỉ sin x trên đoạn không tới pi."):
            self.play(Write(title), Create(ax), Create(sine))
        errs = {2: 0.2105, 4: 0.0704, 8: 0.0188, 16: 0.0048}
        xs = np.linspace(0, PI, 2001)
        poly, lab, cap = None, None, None
        loi = {2: "Với hai khúc, sai số lớn nhất là không phẩy hai một. Mỗi điểm gãy ứng với khoảng một neuron ReLU trong lớp ẩn.",
               4: "Bốn khúc: sai số còn không phẩy không bảy.",
               8: "Tám khúc: không phẩy không một chín. Mỗi lần gấp đôi số khúc, sai số giảm khoảng bốn lần.",
               16: "Mười sáu khúc: chỉ còn không phẩy không không năm."}
        for K, e in errs.items():
            kn = np.linspace(0, PI, K + 1)
            assert abs(np.abs(np.interp(xs, kn, np.sin(kn)) - np.sin(xs)).max() - e) < 1e-4
            new = VMobject(color=C_ACC, stroke_width=4).set_points_as_corners([ax.c2p(k, np.sin(k)) for k in kn])
            dots = VGroup(*[Dot(ax.c2p(k, np.sin(k)), radius=0.05, color=C_ACC) for k in kn])
            g = VGroup(new, dots)
            nl = T(f"K = {K} khúc:  sai số lớn nhất = {e}", 28, C_ACC).next_to(title, DOWN, buff=0.25)
            if poly is not None:
                # nhãn cũ mờ hẳn rồi nhãn mới mới hiện, tránh hai nhãn chồng nhau
                self.play(FadeOut(lab), run_time=0.4)
            with noi(self, loi[K]):
                if poly is None:
                    self.play(Create(g), FadeIn(nl), run_time=1.2)
                else:
                    self.play(ReplacementTransform(poly, g), FadeIn(nl), run_time=1.2)
                poly, lab = g, nl
                if K == 2:
                    cap = swap_caption(self, cap, "Mỗi điểm gãy ứng với khoảng một neuron ReLU trong lớp ẩn")
                if K == 8:
                    cap = swap_caption(self, cap, "Gấp đôi số khúc → sai số giảm khoảng 4 lần (bậc 1/K²)")
                self.wait(1.3)
        with noi(self, "Định lý xấp xỉ phổ quát chỉ nói mạng như vậy tồn tại, chứ không nói gradient descent sẽ tìm được nó."):
            cap = swap_caption(self, cap, "Định lý xấp xỉ nói mạng như vậy TỒN TẠI, không nói GD tìm được")
        self.wait(3)


# ---------------------------------------------------------------- 6.2
class LanNguocMLP(Scene):
    """Ví dụ 6.4: forward rồi backward của mạng 2–3–2, x = (1, 2), nhãn 0."""

    def construct(self):
        title = T("Forward rồi backward trên mạng 2–3–2", 32).to_edge(UP, buff=0.3)
        with noi(self, "Chạy forward rồi backward trên một mạng nhỏ hai, ba, hai, với đầu vào một, hai và nhãn là lớp không."):
            self.play(Write(title))
            xs = [-5.2, -1.9, 1.5, 4.6]
            ys2, ys3 = [0.9, -0.9], [1.5, 0.0, -1.5]
            x_n = [Circle(0.38, color=WHITE).move_to([xs[0], y + 0.2, 0]) for y in ys2]
            h_n = [Circle(0.38, color=WHITE).move_to([xs[1], y + 0.2, 0]) for y in ys3]
            o_n = [Circle(0.38, color=WHITE).move_to([xs[2], y + 0.2, 0]) for y in ys2]
            edges = VGroup(*[Line(a.get_right(), b.get_left(), stroke_opacity=0.35, stroke_width=2)
                             for a in x_n for b in h_n],
                           *[Line(a.get_right(), b.get_left(), stroke_opacity=0.35, stroke_width=2)
                             for a in h_n for b in o_n])
            heads = VGroup(T("x", 24).next_to(x_n[0], UP, buff=0.25), T("z₁ → h", 24).next_to(h_n[0], UP, buff=0.25),
                           T("z₂ → p", 24).next_to(o_n[0], UP, buff=0.25))
            self.play(*[Create(c) for c in x_n + h_n + o_n], Create(edges), FadeIn(heads))

            def put(vals, nodes, color, size=20, side=None):
                g = VGroup()
                for v, n in zip(vals, nodes):
                    t = T(v, size, color)
                    t.move_to(n) if side is None else t.next_to(n, side, buff=0.12)
                    g.add(t)
                return g

            fx = put(["1", "2"], x_n, WHITE, 24)
            self.play(FadeIn(fx))
        with noi(self, "Lớp ẩn có z một bằng trừ một phẩy năm, hai, một phẩy năm. Sau ReLU, neuron đầu tiên tắt."):
            cap = swap_caption(self, None, "Forward: z₁ = W₁x + b₁ = (−1.5, 2, 1.5),  h = ReLU(z₁) = (0, 2, 1.5)")
            fz1 = put(["−1.5", "2", "1.5"], h_n, WHITE, 18, DOWN)
            fh = put(["0", "2", "1.5"], h_n, WHITE, 22)
            self.play(FadeIn(fz1), FadeIn(fh))
            self.play(h_n[0].animate.set_fill(GREY_D, opacity=0.9).set_stroke(GREY))
            self.wait(1)
        with noi(self, "Lớp ra qua softmax cho xác suất lớp đúng chỉ không phẩy một bốn tám, nên loss là một phẩy chín một."):
            cap = swap_caption(self, cap, "z₂ = W₂h = (−1.25, 0.5),  p = softmax(z₂) = (0.148, 0.852)")
            fz2 = put(["−1.25", "0.5"], o_n, WHITE, 18, DOWN)
            fp = put(["0.148", "0.852"], o_n, WHITE, 17)
            self.play(FadeIn(fz2), FadeIn(fp))
            loss = T("L = −ln 0.148 = 1.9102", 24, C_ACC).move_to([xs[3] - 0.5, 0.2, 0])
            self.play(FadeIn(loss))
            self.wait(1.2)
        # backward
        with noi(self, "Lượt backward bắt đầu từ p trừ y, bằng trừ không phẩy tám năm hai và không phẩy tám năm hai."):
            cap = swap_caption(self, cap, "Backward: dz₂ = p − y = (−0.852, 0.852)")
            bz2 = put(["−0.852", "0.852"], o_n, C2, 18, DOWN)
            self.play(FadeOut(fz2), *[n.animate.set_stroke(C2) for n in o_n])
            self.play(FadeIn(bz2))
            self.wait(1)
        with noi(self, "Nhân với W hai chuyển vị để đưa gradient về lớp ẩn."):
            cap = swap_caption(self, cap, "dh = W₂ᵀdz₂ = (−0.426, 1.704, −1.278)")
            bh = put(["−0.426", "1.704", "−1.278"], h_n, C2, 18, DOWN)
            self.play(FadeOut(fz1), FadeIn(bh))
            self.wait(1)
        with noi(self, "Qua ReLU, neuron có z âm chặn hoàn toàn gradient: gradient của nó bằng không."):
            cap = swap_caption(self, cap, "dz₁ = dh ⊙ 1[z₁ > 0]: neuron 1 có z = −1.5 nên chặn gradient")
            # nhãn gradient của nút bị chặn đổi tại chỗ từ −0.426 thành "dz = 0" (không vẽ đè lên số hay cạnh)
            zero = T("dz = 0", 18, C2).move_to(bh[0])
            self.play(ReplacementTransform(bh[0], zero), run_time=0.8)
            self.wait(1)
        with noi(self, "Gradient của W một là dz một nhân x chuyển vị, nên hàng ứng với neuron tắt bằng không. Kiểm tra bằng sai phân cho sai số chỉ cỡ mười mũ trừ mười một."):
            dw = T("dW₁: hàng (0, 0), (1.704, 3.408), (−1.278, −2.556)", 22, C2).next_to(title, DOWN, buff=0.15)
            cap = swap_caption(self, cap, "dW₁ = dz₁ xᵀ: hàng của neuron tắt bằng 0. Gradient check: sai số 1.8e-11")
            self.play(FadeIn(dw))
        self.wait(3)


# ---------------------------------------------------------------- 6.4
class TinHieuQuaLop(Scene):
    """Ví dụ 6.8: std activation qua 10 lớp ReLU rộng 512 (thang log)."""

    def construct(self):
        title = T("Độ lệch chuẩn activation qua 10 lớp ReLU", 32).to_edge(UP, buff=0.3)
        with noi(self, "Theo dõi độ lệch chuẩn của activation qua mười lớp ReLU, mỗi lớp rộng năm trăm mười hai, với ba cách khởi tạo. Trục đứng là thang log."):
            self.play(Write(title))
            data = [("std w = 0.01", [0.132, 8.32e-5, 7.97e-9], C2),
                    ("Xavier: Var = 1/512", [0.585, 0.153, 0.0276], C_ACC),
                    ("He: Var = 2/512", [0.827, 0.743, 0.758], C1)]
            ax = Axes(x_range=[0, 3, 1], y_range=[-9, 0.5, 1], x_length=9.5, y_length=4.3, tips=False,
                      x_axis_config={"include_ticks": False}).shift(DOWN * 0.25 + RIGHT * 0.5)
            ylab = VGroup(*[T(f"1e{k}" if k else "1", 16).next_to(ax.c2p(0, k), LEFT, buff=0.12) for k in range(-9, 1, 3)])
            self.play(Create(ax), FadeIn(ylab))
            groups = ["lớp 1", "lớp 5", "lớp 10"]
            gl = VGroup(*[T(g, 20).next_to(ax.c2p(i + 0.5, -9), DOWN, buff=0.12) for i, g in enumerate(groups)])
            self.play(FadeIn(gl))
            legend = VGroup(*[VGroup(Square(0.22, fill_color=c, fill_opacity=1, stroke_width=0), T(n, 20)).arrange(RIGHT, buff=0.12)
                              for n, _, c in data]).arrange(RIGHT, buff=0.5).next_to(title, DOWN, buff=0.2)
            self.play(FadeIn(legend))
        cap = None
        msgs = ["std 0.01: mỗi lớp nhân std với khoảng 0.16 → sau 10 lớp còn 8e-9",
                "Xavier: ReLU bỏ một nửa năng lượng → mỗi lớp nhân khoảng 1/√2",
                "He: Var(w) = 2/n bù đúng phần ReLU bỏ đi → std giữ quanh 0.75"]
        loi = ["Khởi tạo với độ lệch chuẩn không phẩy không một: mỗi lớp nhân độ lệch chuẩn với khoảng không phẩy một sáu, "
               "sau mười lớp tín hiệu gần như tắt hẳn.",
               "Khởi tạo Xavier tốt hơn, nhưng ReLU bỏ đi một nửa năng lượng, nên mỗi lớp vẫn nhân khoảng một chia căn hai.",
               "Khởi tạo He dùng phương sai hai chia n, bù đúng phần ReLU bỏ đi, nên độ lệch chuẩn giữ quanh không phẩy bảy lăm."]
        for k, ((name, vals, c), msg, loi_doc) in enumerate(zip(data, msgs, loi)):
            bars = VGroup()
            for i, v in enumerate(vals):
                x0 = i + 0.22 + 0.28 * k
                top = np.log10(v)
                b = Rectangle(width=ax.x_length / 3 * 0.2, height=ax.c2p(0, top)[1] - ax.c2p(0, -9)[1],
                              fill_color=c, fill_opacity=0.9, stroke_width=0)
                b.move_to(ax.c2p(x0, -9), aligned_edge=DOWN)
                bars.add(b)
            vl = VGroup(*[T(f"{v:.3g}", 13, c).next_to(b, UP, buff=0.06) for v, b in zip(vals, bars)])
            with noi(self, loi_doc):
                cap = swap_caption(self, cap, msg)
                self.play(*[GrowFromEdge(b, DOWN) for b in bars], FadeIn(vl), run_time=1.3)
                self.wait(1.6)
        with noi(self, "Thanh càng ngắn, tín hiệu càng tắt. Chỉ khởi tạo He giữ được tín hiệu ổn định qua các lớp."):
            cap = swap_caption(self, cap, "Thang log: thanh càng ngắn, tín hiệu càng tắt. Chỉ He giữ được std")
        self.wait(3)


class BatchVaLayerNorm(Scene):
    """Ví dụ 6.9: BN chuẩn hóa theo cột (batch), LN chuẩn hóa theo hàng (mẫu)."""

    def construct(self):
        title = T("BatchNorm theo cột, LayerNorm theo hàng", 32).to_edge(UP, buff=0.3)
        with noi(self, "Một batch gồm bốn mẫu là bốn hàng, mỗi mẫu bốn đặc trưng là bốn cột."):
            self.play(Write(title))
            M = np.array([[1, 2, 5, 0], [2, 4, 6, 8], [3, 1, 4, 2], [6, 3, 1, 2]], dtype=float)
            cells = VGroup()
            for i in range(4):
                for j in range(4):
                    sq = Square(0.85, stroke_width=2, color=GREY_B)
                    sq.move_to([-3.2 + j * 0.85, 1.2 - i * 0.85, 0])
                    cells.add(VGroup(sq, T(f"{M[i, j]:g}", 24).move_to(sq)))
            rlab = VGroup(*[T(f"mẫu {i + 1}", 18).next_to(cells[4 * i], LEFT, buff=0.15) for i in range(4)])
            clab = VGroup(*[T(f"{j + 1}", 18).next_to(cells[j], UP, buff=0.1) for j in range(4)])
            clab.add(T("đặc trưng", 20).next_to(clab, UP, buff=0.12))
            self.play(FadeIn(cells), FadeIn(rlab), FadeIn(clab))
            cap = swap_caption(self, None, "Một batch: 4 mẫu (hàng) × 4 đặc trưng (cột)")
            self.wait(1)
        with noi(self, "BatchNorm chuẩn hóa theo cột: cùng một đặc trưng, lấy trung bình và phương sai trên cả batch. Cột một có trung bình ba, phương sai ba phẩy năm."):
            col = SurroundingRectangle(VGroup(*[cells[4 * i] for i in range(4)]), color=C1, buff=0.04, stroke_width=5)
            self.play(Create(col))
            cap = swap_caption(self, cap, "BatchNorm: cùng một đặc trưng, lấy thống kê trên cả batch")
            info = VGroup(T("cột 1 = (1, 2, 3, 6)", 24, C1),
                          T("mean 3, phương sai lệch 3.5", 22),
                          T("x̂ = (−1.07, −0.53, 0, 1.60)", 24, C1),
                          T("running_mean = 0.3", 20),
                          T("running_var = 0.9 + 0.1·(14/3) = 1.367", 20)).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
            info.move_to([3.4, 0.0, 0])
            self.play(FadeIn(info[:3], shift=LEFT * 0.2))
            self.wait(1.2)
        with noi(self, "Khi huấn luyện, BatchNorm cập nhật thống kê chạy để dùng lúc suy luận. Lưu ý running var lưu phương sai không lệch, chia cho N trừ một."):
            self.play(FadeIn(info[3:]))
            cap = swap_caption(self, cap, "running_var lưu phương sai KHÔNG lệch (chia N − 1 = 3)")
            self.wait(1.8)
        with noi(self, "LayerNorm thì chuẩn hóa theo hàng: thống kê trên mọi đặc trưng của một mẫu. Không cần batch, nên huấn luyện và suy luận giống hệt nhau. Đó là lý do Transformer dùng LayerNorm."):
            row = SurroundingRectangle(cells[4:8], color=C_ACC, buff=0.04, stroke_width=5)
            info2 = VGroup(T("hàng 2 = (2, 4, 6, 8)", 24, C_ACC),
                           T("mean 5, phương sai 5", 22),
                           T("x̂ = (−1.34, −0.45, 0.45, 1.34)", 24, C_ACC),
                           T("không cần batch, train = eval", 20)).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to(info)
            self.play(FadeOut(info), run_time=0.5)
            cap = swap_caption(self, cap, "LayerNorm: một mẫu, lấy thống kê trên mọi đặc trưng của nó")
            self.play(ReplacementTransform(col, row), FadeIn(info2))
        self.wait(3)


# ---------------------------------------------------------------- 6.5
class QuetLearningRate(Scene):
    """Ví dụ 6.12 / Lab 5: loss val sau 3 epoch Adam theo lr (thang log)."""

    def construct(self):
        title = T("Quét learning rate trên thang log", 34).to_edge(UP, buff=0.3)
        ax0 = Axes(x_range=[0, 6, 1], y_range=[0, 2.8, 0.5], x_length=9.5, y_length=4.6, tips=False,
                   x_axis_config={"include_ticks": False}).shift(DOWN * 0.35)

        class _Ax:                                           # trục x = log10(lr), dịch +5.5 để trục y nằm bên trái
            def c2p(self, k, v):
                return ax0.c2p(k + 5.5, v)
            y_length = ax0.y_length
        ax = _Ax()
        xl = VGroup(*[T(f"1e{k}" if k else "1", 18).next_to(ax.c2p(k, 0), DOWN, buff=0.15) for k in range(-5, 1)])
        xt = VGroup(*[Line(ax.c2p(k, 0) + UP * 0.08, ax.c2p(k, 0) + DOWN * 0.08) for k in range(-5, 1)])
        yl = VGroup(*[T(f"{v:.1f}", 16).next_to(ax.c2p(-5.5, v), LEFT, buff=0.15) for v in (0.5, 1.0, 1.5, 2.0, 2.5)])
        axlab = T("loss val (Adam, 3 epoch)", 20).next_to(ax.c2p(-5.5, 2.8), RIGHT, buff=0.2)
        xl.add(xt)
        with noi(self, "Quét learning rate trên thang log, từ mười mũ trừ năm tới một. Đường nét đứt là loss khi đoán mò mười lớp, bằng ln mười."):
            self.play(Write(title), Create(ax0), FadeIn(xl), FadeIn(yl), FadeIn(axlab))
            ln10 = DashedLine(ax.c2p(-5.5, 2.3026), ax.c2p(0.5, 2.3026), color=GREY_B)
            ln10l = T("ln 10 = 2.30 (đoán mò)", 18, GREY_B).next_to(ax.c2p(-2.5, 2.3026), UP, buff=0.08)
            self.play(Create(ln10), FadeIn(ln10l))
        vals = [(-5, 2.289, 0.120), (-4, 1.956, 0.573), (-3, 0.646, 0.789),
                (-2, 0.767, 0.759), (-1, 1.945, 0.319), (0, 2.315, 0.105)]   # số của Lab 5
        dots, labels = VGroup(), VGroup()
        with noi(self, "Mỗi chấm là một lần huấn luyện đầy đủ với một learning rate, đo loss validation sau ba epoch."):
            cap = swap_caption(self, None, "Mỗi chấm là một lần huấn luyện đầy đủ với một lr")
            for k, l, a in vals:
                d = Dot(ax.c2p(k, l), radius=0.09, color=C_ACC)
                t = T(f"{l:.3f}", 16, C_ACC).next_to(d, DOWN if l > 2 else (UR if k < -2 else UL if k > -2 else DOWN), buff=0.12)
                dots.add(d); labels.add(t)
                self.play(FadeIn(d, scale=0.5), FadeIn(t), run_time=0.5)
            curve = VMobject(color=C_ACC, stroke_width=3).set_points_as_corners([d.get_center() for d in dots])
            self.play(Create(curve))
        with noi(self, "Đường cong có hình chữ U: learning rate quá nhỏ thì chưa kịp học, quá lớn thì nhảy loạn."):
            band = Rectangle(width=ax.c2p(-2, 0)[0] - ax.c2p(-3, 0)[0], height=ax.y_length, fill_color=C1,
                             fill_opacity=0.18, stroke_width=0).move_to(ax.c2p(-2.5, 1.4))
            self.play(FadeIn(band))
            cap = swap_caption(self, cap, "Hình chữ U: lr quá nhỏ chưa kịp học, quá lớn thì nhảy loạn")
            self.wait(1.5)
        with noi(self, "Vùng tốt nằm giữa mười mũ trừ ba và mười mũ trừ hai. Hai đầu đều gần mức đoán mò."):
            cap = swap_caption(self, cap, "Vùng tốt 1e-3 đến 1e-2; hai đầu đều ở gần mức đoán mò ln 10")
        self.wait(3)
