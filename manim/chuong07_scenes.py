"""Các cảnh Manim cho Chương 7 — CNN và thị giác máy tính.

Render một cảnh (chất lượng 720p):
    manim -qm chuong07_scenes.py TruotKernel

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
    if t.width > 12.4:                       # tự thu nhỏ để không tràn khung
        t.scale_to_fit_width(12.4)
    return VGroup(BackgroundRectangle(t, fill_opacity=0.85, buff=0.12), t).to_edge(DOWN, buff=0.25)


def swap_caption(scene, old, text, size=24):
    new = caption(text, size)
    if old is None:
        scene.play(FadeIn(new), run_time=0.5)
    else:
        scene.play(FadeOut(old), FadeIn(new), run_time=0.5)
    return new


def grid(values, cell, center, size=24, color=WHITE, stroke=GREY_B):
    """Lưới số: trả về VGroup các ô theo thứ tự hàng trước; ô (i, j) là g[i * ncols + j]."""
    nr, nc = len(values), len(values[0])
    g = VGroup()
    for i in range(nr):
        for j in range(nc):
            sq = Square(cell, stroke_width=2, color=stroke)
            sq.move_to(np.array(center) + np.array([(j - (nc - 1) / 2) * cell, ((nr - 1) / 2 - i) * cell, 0]))
            v = values[i][j]
            txt = T(f"{v:g}" if isinstance(v, (int, float, np.integer, np.floating)) else str(v), size, color).move_to(sq)
            g.add(VGroup(sq, txt))
    return g


def fmt(v):
    return f"{v:g}".replace("-", "−")


# ---------------------------------------------------------------- 7.1
class TruotKernel(Scene):
    """Ví dụ 7.1: cross-correlation Sobel dọc trên ảnh 5×5, đầu ra 3×3."""

    def construct(self):
        title = T("Tích chập 2D: kernel trượt, nhân rồi cộng", 32).to_edge(UP, buff=0.3)
        with noi(self, "Tích chập hai chiều: một kernel nhỏ trượt trên ảnh, tại mỗi vị trí nhân từng cặp số rồi cộng lại."):
            self.play(Write(title))
            X = [[1, 2, 0, 1, 3], [0, 1, 2, 3, 1], [2, 1, 0, 1, 0], [1, 0, 1, 2, 2], [3, 1, 0, 0, 1]]
            K = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
            Y = [[-1, -3, -1], [2, -4, 0], [5, -3, -3]]
            cx, ck, cy = np.array([-4.6, -0.1, 0]), np.array([0.2, 1.35, 0]), np.array([4.7, -0.1, 0])
            gx = grid([[fmt(v) for v in r] for r in X], 0.72, cx, 26)
            gk = grid([[fmt(v) for v in r] for r in K], 0.62, ck, 22, C_ACC, C_ACC)
            gy = grid([["" for _ in range(3)] for _ in range(3)], 0.8, cy, 26, C1, GREY_D)
            labs = VGroup(T("ảnh X (5×5)", 22).next_to(gx, UP, buff=0.2), T("kernel K (3×3)", 22, C_ACC).next_to(gk, UP, buff=0.15),
                          T("đầu ra Y (3×3)", 22, C1).next_to(gy, UP, buff=0.2))
            self.play(FadeIn(gx), FadeIn(gk), FadeIn(gy), FadeIn(labs))
        with noi(self, "Kernel ở đây là bộ lọc Sobel dọc. Với stride một và không padding, ảnh năm nhân năm cho đầu ra ba nhân ba."):
            cap = swap_caption(self, None, "K là bộ lọc Sobel dọc. Stride 1, không padding: ảnh 5×5 cho đầu ra 3×3")
        with noi(self, "Ô đầu tiên: nhân mỗi số trong cửa sổ với số cùng vị trí của kernel rồi cộng lại, được trừ một. Sau đó cửa sổ trượt dần qua mọi vị trí."):
            win = Square(0.72 * 3, color=C_ACC, stroke_width=6)
            formula = T("", 24, C_ACC)
            outs = VGroup()
            for p in range(9):
                i, j = divmod(p, 3)
                pos = gx[i * 5 + j + 0][0].get_center() + np.array([0.72, -0.72, 0])      # tâm cửa sổ 3×3
                if p == 0:
                    win.move_to(pos)
                    self.play(Create(win), run_time=0.8)
                    nf = T("1·1 + 2·(−2) + 2·1 = −1   (các tích còn lại bằng 0)", 22, C_ACC).move_to([0.0, -2.6, 0])
                    self.play(FadeIn(nf))
                    cap = swap_caption(self, cap, "Ô đầu: mỗi số trong cửa sổ nhân với số cùng vị trí của K, rồi cộng lại")
                    self.wait(0.8)
                    self.play(FadeOut(nf), run_time=0.4)
                    rt = 0.6
                else:
                    self.play(win.animate.move_to(pos), run_time=0.45)
                    rt = 0.3
                val = T(fmt(Y[i][j]), 26, C1).move_to(gy[p][0])
                outs.add(val)
                self.play(FadeIn(val), run_time=rt)
        with noi(self, "Cùng một kernel được dùng ở mọi vị trí. Đó là chia sẻ trọng số, giúp conv có rất ít tham số."):
            cap = swap_caption(self, cap, "Cùng một K được dùng ở mọi vị trí: đó là chia sẻ trọng số")
            self.wait(0.8)
        with noi(self, "Lưu ý: tích chập trong toán học lật kernel trước khi trượt. Conv2d của PyTorch thực ra tính cross-correlation, không lật, nên lệch dấu so với giáo trình xử lý tín hiệu."):
            note = T("convolution toán học lật K trước: cho đúng âm của bảng này", 24, C2).move_to([0.0, -2.6, 0])
            self.play(FadeIn(note))
            cap = swap_caption(self, cap, "nn.Conv2d tính cross-correlation (không lật), nên lệch dấu so với giáo trình tín hiệu")
            self.wait(2)


class ConvLaMaTran(Scene):
    """Ví dụ 7.3: conv 1D = nhân ma trận Toeplitz."""

    def construct(self):
        title = T("Conv 1D là một phép nhân ma trận", 34).to_edge(UP, buff=0.3)
        with noi(self, "Tích chập một chiều thực chất là một phép nhân ma trận. Mỗi hàng của ma trận T là kernel một, không, trừ một, dịch sang phải một cột, còn lại là số không."):
            self.play(Write(title))
            Tm = [[1, 0, -1, 0, 0], [0, 1, 0, -1, 0], [0, 0, 1, 0, -1]]
            xv = [1, 2, 3, 4, 5]
            cm, cv, cr = np.array([-4.3, 0.5, 0]), np.array([-0.6, 0.5, 0]), np.array([2.0, 0.5, 0])
            gm = grid([[fmt(v) for v in r] for r in Tm], 0.62, cm, 24, WHITE, GREY_B)
            gv = grid([[str(v)] for v in xv], 0.62, cv, 24, WHITE, GREY_B)
            gr = grid([["" ] for _ in range(3)], 0.62, cr, 24, C1, GREY_D)
            gr.shift(np.array([0, 0.0, 0]))
            times, eq = T("×", 32).move_to([-2.2, 0.5, 0]), T("=", 32).move_to([0.7, 0.5, 0])
            labs = VGroup(T("T (3×5)", 22).next_to(gm, UP, buff=0.2), T("x (5)", 22).next_to(gv, UP, buff=0.2),
                          T("y (3)", 22, C1).next_to(gr, UP, buff=0.2))
            self.play(FadeIn(gm), FadeIn(gv), FadeIn(gr), FadeIn(times), FadeIn(eq), FadeIn(labs))
            info = VGroup(T("kernel w = (1, 0, −1)", 24, C_ACC), T("hàng i của T = w", 22), T("đặt từ cột i", 22)
                          ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([5.0, 0.5, 0])
            self.play(FadeIn(info))
            cap = swap_caption(self, None, "Mỗi hàng của T là kernel (1, 0, −1) dịch đi một cột; chỗ còn lại là 0")
            self.wait(1.2)
        with noi(self, "Nhân từng hàng của T với x đúng bằng trượt kernel qua x: hàng một cho một trừ ba, bằng trừ hai, và tương tự cho các hàng sau."):
            res = [-2, -2, -2]
            rowbox, vbox = None, None
            outs = VGroup()
            for i in range(3):
                rb = SurroundingRectangle(VGroup(*[gm[i * 5 + c] for c in range(5)]), color=C_ACC, buff=0.03, stroke_width=4)
                vb = SurroundingRectangle(VGroup(*[gv[i + c] for c in range(3)]), color=C_ACC, buff=0.03, stroke_width=4)
                if rowbox is None:
                    self.play(Create(rb), Create(vb), run_time=0.7)
                else:
                    self.play(ReplacementTransform(rowbox, rb), ReplacementTransform(vbox, vb), run_time=0.7)
                rowbox, vbox = rb, vb
                vals = xv[i:i + 3]
                f = T(f"hàng {i + 1}: {vals[0]} − {vals[2]} = {fmt(res[i])}", 24, C_ACC).move_to([5.0, -1.2 - 0.0, 0])
                o = T(fmt(res[i]), 24, C1).move_to(gr[i][0])
                outs.add(o)
                self.play(FadeIn(f), FadeIn(o), run_time=0.6)
                self.wait(0.9)
                self.play(FadeOut(f), run_time=0.3)
            self.play(FadeOut(rowbox), FadeOut(vbox))
        with noi(self, "Kết quả trùng với np.correlate. Trong hai chiều, thư viện dùng im2col để dựng ma trận rồi nhân một lần, còn backward chỉ là nhân với ma trận chuyển vị."):
            cap = swap_caption(self, cap, "y = Tx = (−2, −2, −2): đúng bằng np.correlate. Trượt kernel = nhân ma trận có cấu trúc")
            self.wait(1.2)
            end = T("2D: im2col dựng ma trận, thư viện nhân một lần. Backward: nhân với chuyển vị Tᵀ", 22, C_ACC).move_to([0, -2.1, 0])
            self.play(FadeIn(end))
            self.wait(3)


# ---------------------------------------------------------------- 7.2
class ChongBaLop(Scene):
    """Ví dụ 7.5: receptive field của ba lớp 3×3 xếp chồng, và số tham số so với một kernel lớn."""

    def construct(self):
        title = T("Ba lớp 3×3 nhìn được vùng 7×7", 34).to_edge(UP, buff=0.3)
        with noi(self, "Receptive field là vùng đầu vào mà một điểm nhìn thấy. Một điểm ở lớp ba phụ thuộc ba điểm của lớp hai."):
            self.play(Write(title))
            xs0, n = -6.6, 9
            sp = 0.5
            ys = [-1.9, -0.7, 0.5, 1.7]                       # lớp 0 (đầu vào) … lớp 3 (đầu ra)
            names = ["đầu vào", "sau lớp 1", "sau lớp 2", "sau lớp 3"]
            dots = [[Dot([xs0 + sp * i, ys[l], 0], radius=0.1, color=GREY_B) for i in range(n)] for l in range(4)]
            self.play(*[FadeIn(d) for row in dots for d in row], run_time=1)
            for l in range(4):
                self.add(T(names[l], 18, GREY_B).move_to([xs0 + sp * (n - 1) + 1.3, ys[l], 0]))
            center = 4
            cap = swap_caption(self, None, "Một điểm ở lớp 3 phụ thuộc 3 điểm của lớp 2 (cửa sổ 3×3)")
            self.play(dots[3][center].animate.set_color(C_ACC).scale(1.5))
        with noi(self, "Mỗi điểm đó lại phụ thuộc ba điểm của lớp một, tổng cộng năm. Lùi thêm một lớp nữa là bảy điểm của ảnh đầu vào."):
            lo = hi = center
            for l, msg in zip((2, 1, 0), [None, "Mỗi điểm đó lại phụ thuộc 3 điểm của lớp 1: tổng cộng 5",
                                          "Lùi thêm một lớp: 7 điểm của ảnh đầu vào"]):
                lo, hi = lo - 1, hi + 1
                parents = [center] if l == 2 else list(range(lo + 1, hi))
                lines = VGroup(*[Line(dots[l + 1][j].get_center(), dots[l][j + d].get_center(), stroke_width=2, color=C_ACC, stroke_opacity=0.8)
                                 for j in parents for d in (-1, 0, 1)])
                self.play(Create(lines), *[dots[l][j].animate.set_color(C_ACC) for j in range(lo, hi + 1)], run_time=1.2)
                rf = T(f"RF = {hi - lo + 1}", 22, C_ACC).move_to([xs0 + sp * (n - 1) + 2.75, ys[l], 0])
                self.play(FadeIn(rf), run_time=0.4)
                if msg:
                    cap = swap_caption(self, cap, msg)
                self.wait(0.7)
        with noi(self, "Ba lớp ba nhân ba cũng nhìn được vùng bảy nhân bảy như một kernel lớn, nhưng chỉ tốn hai mươi bảy C bình phương tham số thay vì bốn mươi chín, và có thêm phi tuyến xen giữa. Đó là lập luận của VGG."):
            table = VGroup(T("C kênh vào và ra, bỏ bias", 20, GREY_B),
                           T("3 lớp 3×3: 27C² tham số", 24, C1),
                           T("1 lớp 7×7: 49C² tham số", 24, C2),
                           T("tỉ lệ 27/49 = 0.551", 22, WHITE),
                           T("2 lớp 3×3 so với 5×5:", 22, WHITE),
                           T("18C² so với 25C² = 0.72", 22, WHITE),
                           T("cộng thêm phi tuyến xen giữa", 20, GREY_B)).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
            table.to_edge(RIGHT, buff=0.4).shift(DOWN * 0.1)
            self.play(FadeIn(table))
            cap = swap_caption(self, cap, "Cùng vùng nhìn 7×7, nhưng ít tham số hơn và có thêm hai lần phi tuyến (lập luận của VGG)")
            self.wait(3)


class AliasingStride(Scene):
    """Ví dụ 7.7: stride 2 phá tính bất biến dịch chuyển (1D)."""

    def construct(self):
        title = T("Stride 2: dịch một bước, đầu ra đảo hoàn toàn", 32).to_edge(UP, buff=0.3)
        with noi(self, "Tín hiệu x xen kẽ một và không. Lấy mẫu với stride hai, giữ các vị trí chẵn, ta được toàn số một."):
            self.play(Write(title))
            a = [1, 0, 1, 0, 1, 0, 1, 0]
            b = [0, 1, 0, 1, 0, 1, 0, 1]
            cell = 0.8
            xr = -4.2
            ga = grid([[str(v) for v in a]], cell, [0, 1.6, 0], 26)
            la = T("x", 26).next_to(ga, LEFT, buff=0.4)
            self.play(FadeIn(ga), FadeIn(la))
            cap = swap_caption(self, None, "Tín hiệu x = (1, 0, 1, 0, 1, 0, 1, 0), lấy mẫu với stride 2 (giữ vị trí 0, 2, 4, 6)")
            boxes = VGroup(*[SurroundingRectangle(ga[i], color=C_ACC, buff=0.02, stroke_width=5) for i in (0, 2, 4, 6)])
            self.play(Create(boxes))
            oa = grid([["1"] * 4], cell, [0, 0.2, 0], 26, C1, C1)
            self.play(FadeIn(oa))
            oal = T("x[::2] = (1, 1, 1, 1)", 24, C1).next_to(oa, RIGHT, buff=0.5)
            self.play(FadeIn(oal))
            self.wait(0.3)
        with noi(self, "Dịch x sang phải đúng một bước. Cùng một tín hiệu, nhưng đầu ra giờ toàn số không."):
            cap = swap_caption(self, cap, "Dịch x sang phải một bước: x′ = (0, 1, 0, 1, 0, 1, 0, 1), cùng tín hiệu, chỉ lệch một điểm")
            gb = grid([[str(v) for v in b]], cell, [0, -1.4, 0], 26)
            lb = T("x′", 26).next_to(gb, LEFT, buff=0.4)
            self.play(FadeIn(gb), FadeIn(lb))
            boxes2 = VGroup(*[SurroundingRectangle(gb[i], color=C_ACC, buff=0.02, stroke_width=5) for i in (0, 2, 4, 6)])
            self.play(Create(boxes2))
            ob = grid([["0"] * 4], cell, [0, -2.6, 0], 26, C2, C2)
            self.play(FadeIn(ob))
            obl = T("x′[::2] = (0, 0, 0, 0)", 24, C2).next_to(ob, RIGHT, buff=0.5)
            self.play(FadeIn(obl))
        with noi(self, "Một bước dịch làm đầu ra đảo hoàn toàn. Đây là hiện tượng aliasing: conv kèm stride không bất biến với phép dịch."):
            cap = swap_caption(self, cap, "Một bước dịch làm đầu ra đảo hoàn toàn: đây là aliasing, conv + stride không bất biến dịch chuyển")
            self.wait(0.8)
        with noi(self, "Max pooling cũng vậy. Với xung đơn ở vị trí một, đầu ra là một, không, không, không. Dịch một bước, đầu ra dịch đúng một ô. Nhưng dịch thêm bước nữa, đầu ra lại không đổi. Chỉ dịch theo bội của stride mới cho đầu ra dịch đều."):
            title_b = T("Max-pool 2: dịch một bước, đầu ra có khi không đổi", 30).to_edge(UP, buff=0.3)
            self.play(FadeOut(VGroup(ga, la, boxes, oa, oal, gb, lb, boxes2, ob, obl, title)), run_time=0.5)
            self.play(FadeIn(title_b), run_time=0.5)
            # nhánh max-pool: xung đơn ở vị trí 1, 2, 3
            sub = T("Max-pool 2 (stride 2) trên xung đơn, độ dài 8", 28, C_ACC).move_to([0, 2.1, 0])
            self.play(FadeIn(sub))
            rows = []
            for r, p in enumerate((1, 2, 3)):
                y = 0.8 - r * 1.3
                inp = [1 if i == p else 0 for i in range(8)]
                out = [max(inp[2 * k], inp[2 * k + 1]) for k in range(4)]
                gi = grid([[str(v) for v in inp]], 0.55, [-3.0, y, 0], 22)
                go = grid([[str(v) for v in out]], 0.55, [3.2, y, 0], 22, C1, C1)
                arr = Arrow(gi.get_right(), go.get_left(), buff=0.25, stroke_width=3, color=GREY_B)
                lab = T(f"xung ở vị trí {p}", 20, GREY_B).next_to(gi, UP, buff=0.08)
                rows.append(VGroup(gi, go, arr, lab))
            for i, r in enumerate(rows):
                self.play(FadeIn(r), run_time=0.5)
                if i == 0:
                    cap = swap_caption(self, cap, "Xung ở vị trí 1 cho (1, 0, 0, 0)")
                elif i == 1:
                    cap = swap_caption(self, cap, "Dịch một bước: đầu ra (0, 1, 0, 0), dịch đúng một ô")
                else:
                    cap = swap_caption(self, cap, "Dịch thêm một bước: đầu ra vẫn (0, 1, 0, 0). Chỉ dịch theo bội của stride mới cho đầu ra dịch đều")
                self.wait(0.5)
        self.wait(1.0)


# ---------------------------------------------------------------- 7.3
class KhoiResidual(Scene):
    """Ví dụ 7.10: bottleneck 256 → 64 → 64 → 256 với đường tắt, và số tham số."""

    def construct(self):
        title = T("Khối residual bottleneck", 34).to_edge(UP, buff=0.3)
        with noi(self, "Khối residual bottleneck: conv một nhân một nén hai trăm năm mươi sáu kênh xuống sáu mươi tư, conv ba nhân ba xử lý ở sáu mươi tư kênh, rồi một nhân một mở lại thành hai trăm năm mươi sáu."):
            self.play(Write(title))
            def box(txt, sub, x, col):
                r = Rectangle(width=2.1, height=1.0, color=col, fill_color=col, fill_opacity=0.15, stroke_width=3).move_to([x, 0, 0])
                return VGroup(r, T(txt, 24, WHITE).move_to(r.get_center() + UP * 0.15), T(sub, 18, GREY_B).move_to(r.get_center() + DOWN * 0.22))
            b1 = box("1×1", "256 → 64", -3.6, BLUE)
            b2 = box("3×3", "64 → 64", -1.2, BLUE)
            b3 = box("1×1", "64 → 256", 1.2, BLUE)
            xin = T("x", 30).move_to([-6.3, 0, 0])
            plus = VGroup(Circle(0.3, color=WHITE), T("+", 30)).move_to([3.5, 0, 0])
            yout = T("y", 30).move_to([5.1, 0, 0])
            a0 = Arrow(xin.get_right(), b1.get_left(), buff=0.1, stroke_width=3)
            a1 = Arrow(b1.get_right(), b2.get_left(), buff=0.05, stroke_width=3)
            a2 = Arrow(b2.get_right(), b3.get_left(), buff=0.05, stroke_width=3)
            a3 = Arrow(b3.get_right(), plus.get_left(), buff=0.05, stroke_width=3)
            a4 = Arrow(plus.get_right(), yout.get_left(), buff=0.1, stroke_width=3)
            self.play(FadeIn(xin), FadeIn(b1), FadeIn(b2), FadeIn(b3), FadeIn(plus), FadeIn(yout),
                      *[Create(a) for a in (a0, a1, a2, a3, a4)])
            cap = swap_caption(self, None, "1×1 nén 256 kênh xuống 64, 3×3 xử lý ở 64 kênh, 1×1 mở lại thành 256")
            self.wait(1.2)
        with noi(self, "Đường tắt cộng thẳng x vào đầu ra: y bằng x cộng F x. Khối chỉ phải học phần dư, và gradient luôn có một đường đi thẳng về các lớp trước."):
            skip = VMobject(color=C1, stroke_width=5)
            skip.set_points_as_corners([[-5.7, 0, 0], [-5.7, 1.5, 0], [3.5, 1.5, 0], [3.5, 0.32, 0]])
            sk = T("đường tắt: cộng thẳng x vào đầu ra", 22, C1).move_to([-1.2, 1.9, 0])
            self.play(Create(skip), FadeIn(sk), run_time=1.5)
            cap = swap_caption(self, cap, "y = x + F(x): khối chỉ phải học phần dư F; ∂y/∂x = I + ∂F/∂x có sẵn đường gradient trực tiếp")
            self.wait(1.5)
        with noi(self, "Đếm tham số, bỏ bias: tổng cộng sáu mươi chín nghìn sáu trăm ba mươi hai."):
            cnt = VGroup(T("16 384", 20, WHITE).next_to(b1, DOWN, buff=0.25), T("36 864", 20, WHITE).next_to(b2, DOWN, buff=0.25),
                         T("16 384", 20, WHITE).next_to(b3, DOWN, buff=0.25))
            self.play(FadeIn(cnt))
            cap = swap_caption(self, cap, "Tham số (bỏ bias): 256·64 + 9·64·64 + 64·256")
            self.wait(1.2)
            tot = T("tổng = 69 632", 28, C_ACC).move_to([-3.6, -2.0, 0])
            cmp_ = T("hai conv 3×3 ở 256 kênh: 2·9·256·256 = 1 179 648 (gấp 16.9 lần)", 24, C2).move_to([0.9, -2.0, 0])
            self.play(FadeIn(tot))
            self.wait(0.8)
        with noi(self, "So với hai conv ba nhân ba ở hai trăm năm mươi sáu kênh, cần hơn một triệu một trăm nghìn tham số, gấp gần mười bảy lần."):
            cap = swap_caption(self, cap, "So với hai conv 3×3 ở 256 kênh:")
            self.play(FadeIn(cmp_.move_to([0.5, -2.7, 0])))
            self.wait(1.2)
        with noi(self, "Số phép tính cũng giảm tương ứng. Nhưng tốc độ thật còn tùy phần cứng, không chỉ tùy số phép nhân."):
            self.play(tot.animate.move_to([-3.6, -2.0, 0]))
            cap = swap_caption(self, cap, "Ở 56×56: 218 365 952 MAC so với 3 699 376 128. Số đếm đúng, nhưng tốc độ thật còn tùy phần cứng")
            self.wait(3)


# ---------------------------------------------------------------- 7.4
class BN2dKenh(Scene):
    """Ví dụ 7.12: BatchNorm2d lấy thống kê trên (N, H, W) của từng kênh; LayerNorm lấy trên (C, H, W) của từng mẫu."""

    def construct(self):
        title = T("BatchNorm2d: thống kê theo từng kênh", 34).to_edge(UP, buff=0.3)
        with noi(self, "Một tensor có hai mẫu, hai kênh, mỗi kênh là một bản đồ hai nhân hai."):
            self.play(Write(title))
            s0c0, s0c1 = [[1, 2], [3, 4]], [[0, 0], [2, 2]]
            s1c0, s1c1 = [[5, 6], [7, 8]], [[1, 1], [3, 3]]
            cell = 0.62
            pos = {("0", 0): [-4.3, 1.1, 0], ("0", 1): [-2.4, 1.1, 0], ("1", 0): [-4.3, -1.1, 0], ("1", 1): [-2.4, -1.1, 0]}
            data = {("0", 0): s0c0, ("0", 1): s0c1, ("1", 0): s1c0, ("1", 1): s1c1}
            gs = {k: grid([[fmt(v) for v in r] for r in data[k]], cell, pos[k], 24) for k in data}
            heads = VGroup(T("kênh 0", 22).move_to([-4.3, 2.2, 0]), T("kênh 1", 22).move_to([-2.4, 2.2, 0]),
                           T("mẫu 0", 22).move_to([-6.0, 1.1, 0]), T("mẫu 1", 22).move_to([-6.0, -1.1, 0]))
            self.play(*[FadeIn(g) for g in gs.values()], FadeIn(heads))
            cap = swap_caption(self, None, "Tensor N = 2 mẫu, C = 2 kênh, mỗi bản đồ 2×2")
            self.wait(0.5)
        with noi(self, "BatchNorm hai chiều lấy thống kê theo từng kênh: kênh không gồm tám giá trị của cả hai mẫu, gộp chung lại. Trung bình bốn phẩy năm, phương sai năm phẩy hai lăm."):
            bx = VGroup(SurroundingRectangle(VGroup(gs[("0", 0)], gs[("1", 0)]), color=C1, buff=0.12, stroke_width=5))
            self.play(Create(bx))
            cap = swap_caption(self, cap, "BN: kênh 0 gồm 8 giá trị (N·H·W = 2·2·2) của cả hai mẫu, gộp chung")
            info = VGroup(T("kênh 0: 1…8", 24, C1), T("mean = 4.5, phương sai lệch = 5.25", 22),
                          T("x̂ mẫu 0 = (−1.53, −1.09, −0.65, −0.22)", 22, C1),
                          T("kênh 1: mean 1.5, phương sai lệch 1.25", 22)).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.2, 0.9, 0])
            self.play(FadeIn(info[:3], shift=LEFT * 0.2))
            self.wait(1.2)
            self.play(FadeIn(info[3]))
        with noi(self, "Mỗi kênh có một cặp gamma, beta và một cặp thống kê chạy riêng."):
            extra = T("mỗi kênh một cặp γ, β và một cặp running stats", 22, GREY_B).move_to([3.2, -0.7, 0])
            self.play(FadeIn(extra))
            cap = swap_caption(self, cap, "running_mean = (0.45, 0.15), running_var = (1.5, 1.043) sau một bước (momentum 0.1)")
            self.wait(1.4)
        self.play(FadeOut(info), FadeOut(extra), FadeOut(bx))
        with noi(self, "LayerNorm thì lấy thống kê theo từng mẫu: mẫu không gồm tám giá trị của cả hai kênh. Kết quả khác hẳn BatchNorm, và không cần batch."):
            title2 = T("LayerNorm trên (C, H, W): thống kê theo từng mẫu", 34).to_edge(UP, buff=0.3)
            self.play(FadeOut(title), run_time=0.4)
            self.play(FadeIn(title2), run_time=0.5)
            bx2 = SurroundingRectangle(VGroup(gs[("0", 0)], gs[("0", 1)]), color=C_ACC, buff=0.12, stroke_width=5)
            self.play(Create(bx2))
            cap = swap_caption(self, cap, "LN: mẫu 0 gồm 8 giá trị của cả hai kênh, thống kê lấy trên chúng")
            info2 = VGroup(T("mean = 1.75, phương sai lệch = 1.6875", 22, C_ACC),
                           T("kênh 0 → (−0.58, 0.19, 0.96, 1.73)", 22, C_ACC),
                           T("kênh 1 → (−1.35, −1.35, 0.19, 0.19)", 22, C_ACC),
                           T("khác hẳn BN, và không cần batch", 22, GREY_B)).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.2, 0.5, 0])
            self.play(FadeIn(info2))
            self.wait(2.5)


# ---------------------------------------------------------------- 7.5
class PatchLaConv(Scene):
    """Ví dụ 7.17: ViT cắt ảnh thành patch 16×16 và nhúng bằng một conv 16×16 stride 16."""

    def construct(self):
        title = T("ViT: biến ảnh thành chuỗi token bằng một conv", 32).to_edge(UP, buff=0.3)
        with noi(self, "Vision Transformer biến ảnh thành chuỗi token. Ảnh hai trăm hai mươi tư nhân hai trăm hai mươi tư được chia thành lưới mười bốn nhân mười bốn patch, mỗi patch mười sáu nhân mười sáu điểm ảnh."):
            self.play(Write(title))
            S = 3.5
            img = Square(S, color=WHITE, stroke_width=3, fill_color=BLUE_E, fill_opacity=0.35).move_to([-4.6, 0.3, 0])
            lines = VGroup(*[Line(img.get_corner(UL) + RIGHT * S * k / 14, img.get_corner(DL) + RIGHT * S * k / 14, stroke_width=1, color=GREY_B)
                             for k in range(1, 14)],
                           *[Line(img.get_corner(UL) + DOWN * S * k / 14, img.get_corner(UR) + DOWN * S * k / 14, stroke_width=1, color=GREY_B)
                             for k in range(1, 14)])
            lab = T("ảnh 224×224×3", 22).next_to(img, DOWN, buff=0.2)
            self.play(FadeIn(img), FadeIn(lab))
            cap = swap_caption(self, None, "Chia ảnh thành lưới 14×14 patch, mỗi patch 16×16 điểm ảnh")
            self.play(Create(lines), run_time=1.5)
            self.wait(0.8)
        with noi(self, "Mỗi patch có bảy trăm sáu mươi tám số. Duỗi thẳng rồi nhân với một ma trận, ta được một token. Cả ảnh thành một trăm chín mươi sáu token."):
            patch = Square(S / 14, color=C_ACC, stroke_width=4, fill_color=C_ACC, fill_opacity=0.7).move_to(img.get_corner(UL) + np.array([S * 3.5 / 14, -S * 2.5 / 14, 0]))
            self.play(FadeIn(patch))
            cap = swap_caption(self, cap, "Mỗi patch có 16·16·3 = 768 số; duỗi thẳng thành một vector, rồi nhân với ma trận 768 × D (D = 768 ở ViT-B)")
            toks = VGroup()
            for k in range(8):
                r = Rectangle(width=0.42, height=1.3, color=C1, fill_color=C1, fill_opacity=0.25, stroke_width=2)
                toks.add(r)
            toks.arrange(RIGHT, buff=0.14).move_to([2.2, 0.8, 0])
            dots_ = T("…", 30).next_to(toks, RIGHT, buff=0.15)
            arrow = Arrow(img.get_right(), toks.get_left(), buff=0.25, stroke_width=3, color=GREY_B)
            self.play(Create(arrow), FadeIn(toks), FadeIn(dots_), run_time=1.2)
            tl = T("196 token, mỗi token là vector 768 số", 24, C1).next_to(toks, DOWN, buff=0.35)
            self.play(FadeIn(tl))
            self.wait(1)
        with noi(self, "Phép nhúng này chính là một Conv2d có kernel mười sáu, stride mười sáu. Tổng số tham số khoảng năm trăm chín mươi nghìn."):
            cap = swap_caption(self, cap, "Phép nhúng này chính là Conv2d(3, 768, kernel 16, stride 16): đã kiểm bằng unfold + Linear")
            cnt = VGroup(T("tham số: 3·16·16·768 + 768 = 590 592", 24, C_ACC),
                         T("đầu ra: 768 kênh × 14 × 14 = chuỗi 196 token", 24, WHITE)).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([2.2, -1.4, 0])
            self.play(FadeIn(cnt))
            self.wait(1.5)
        with noi(self, "Từ đây ảnh đã thành chuỗi token, và attention ở chương tám sẽ thay cho conv."):
            cap = swap_caption(self, cap, "Từ đây là chuỗi token: attention (chương 8) thay conv. Ảnh 384×384 có 576 token")
        self.wait(3)
