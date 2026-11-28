"""Các cảnh Manim cho Chương 11 — Phương pháp thực nghiệm và đọc paper.

Render một cảnh (chất lượng 720p):
    manim -qm chuong11_scenes.py BestOfK

Chỉ dùng Text (không cần LaTeX), nên chỉ cần Manim Community + ffmpeg.
Mỗi cảnh dùng đúng số liệu của ví dụ giải mẫu trong sách (bài toán đồ chơi: 10 lớp Gauss, train 300 mẫu, MLP 20–64–10).
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
C_A, C_B = BLUE_C, ORANGE


def T(s, size=28, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


def caption(s, size=24):
    t = T(s, size)
    if t.width > 12.6:                       # phụ đề dài thì thu nhỏ cho vừa khung hình
        t.scale(12.6 / t.width)
    return VGroup(BackgroundRectangle(t, fill_opacity=0.85, buff=0.12), t).to_edge(DOWN, buff=0.25)


def swap_caption(scene, old, text, size=24):
    new = caption(text, size)
    if old is None:
        scene.play(FadeIn(new), run_time=0.5)
    else:
        scene.play(FadeOut(old), run_time=0.25)
        scene.play(FadeIn(new), run_time=0.25)
    return new


# ---------------------------------------------------------------- 11.1
class DemChienThang(Scene):
    """Ví dụ 11.1: thắng 11/18 cột là chuyện thường; chỉ 6 phép so sánh độc lập thì còn yếu hơn."""

    def construct(self):
        title = T("Đếm chiến thắng: 6 dataset × 3 chỉ số = 18 cột", 32).to_edge(UP, buff=0.3)
        with noi(self, "Một paper tuyên bố phương pháp của họ thắng mười một trên mười tám cột: sáu dataset, mỗi dataset ba chỉ số."):
            self.play(Write(title))
            wins = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 1), (2, 2), (3, 0), (3, 2), (4, 0), (4, 1)}   # 11 ô; dataset 1–5 thắng đa số cột, dataset 6 không thắng
            cells = {}
            grid = VGroup()
            for r in range(6):
                for c in range(3):
                    sq = Square(0.62, color=GREY_B, fill_color=GREY_D, fill_opacity=0.9).move_to([-4.2 + 0.72 * c, 2.0 - 0.72 * r, 0])
                    cells[(r, c)] = sq
                    grid.add(sq)
            rl = VGroup(*[T(f"dataset {r + 1}", 20).next_to(cells[(r, 0)], LEFT, buff=0.25) for r in range(6)])
            cl = VGroup(*[T(f"{c + 1}", 20).next_to(cells[(0, c)], UP, buff=0.12) for c in range(3)],
                        T("chỉ số", 18, GREY_B).next_to(cells[(0, 1)], UP, buff=0.5))
            self.play(Create(grid), FadeIn(rl), FadeIn(cl), run_time=1.5)
            cap = swap_caption(self, None, "Tuyên bố: \"phương pháp của chúng tôi thắng 11/18 cột\"")
            self.play(*[cells[k].animate.set_fill(C1, opacity=0.9) for k in sorted(wins)], run_time=1.5)
            self.wait(0.5)
        with noi(self, "Nếu mỗi cột chỉ như tung đồng xu, xác suất thắng từ mười một cột trở lên là khoảng hai mươi bốn phần trăm. Không có gì lạ cả."):
            r1 = VGroup(T("Nếu mỗi cột là tung đồng xu:", 26), T("P(thắng ≥ 11 / 18) = 0.2403", 30, C_ACC),
                        T("không có gì lạ", 26), T("(thắng 14/18 mới ra 0.0154)", 24, GREY_B)).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to([3.1, 0.9, 0])
            self.play(FadeIn(r1))
            cap = swap_caption(self, cap, "Mỗi cột có thắng hay không đều tính như một phép thử độc lập")
            self.wait(2)

        with noi(self, "Hơn nữa, ba chỉ số của cùng một dataset thường cùng thắng hoặc cùng thua. Thực chất chỉ có khoảng sáu phép so sánh độc lập."):
            cap = swap_caption(self, cap, "Nhưng ba chỉ số của cùng một dataset thường cùng dấu")
            self.play(FadeOut(r1))
            boxes = VGroup()
            for r in range(6):
                b = RoundedRectangle(corner_radius=0.08, width=2.1, height=0.62, color=GREY_B, fill_color=GREY_D, fill_opacity=0.9)
                b.move_to([1.2, 2.0 - 0.72 * r, 0])
                boxes.add(b)
            bl = VGroup(*[T(f"dataset {r + 1}", 20).move_to(boxes[r]) for r in range(6)])
            self.play(FadeIn(boxes), FadeIn(bl))
            self.play(*[boxes[r].animate.set_fill(C1, opacity=0.9) for r in range(5)], run_time=1.2)
        with noi(self, "Thắng năm trên sáu có xác suất khoảng mười một phần trăm. Thắng nhiều cột chưa phải bằng chứng, nếu không biết các cột độc lập tới đâu."):
            r2 = VGroup(T("Chỉ còn khoảng 6 phép", 24), T("so sánh độc lập:", 24), T("thắng 5/6: P = 0.1094", 26, C_ACC),
                        T("thắng 6/6: P = 0.0156", 26, C_ACC)).arrange(DOWN, buff=0.22, aligned_edge=LEFT).next_to(boxes, RIGHT, buff=0.45)
            self.play(FadeIn(r2))
            cap = swap_caption(self, cap, "Thắng nhiều cột chưa phải bằng chứng nếu không biết các cột độc lập tới đâu")
            self.wait(3)


# ---------------------------------------------------------------- 11.2
class BaselineBiDoiXu(Scene):
    """Ví dụ 11.3: cùng một phương pháp, ba cách tune baseline cho ba mức chênh."""

    def construct(self):
        title = T("Baseline chưa tune làm hiệu ứng phình ra", 34).to_edge(UP, buff=0.3)
        with noi(self, "Cùng một phương pháp, nhưng ba cách tune baseline cho ba mức chênh lệch rất khác nhau."):
            self.play(Write(title))
            base_y = -1.6
            scale = 0.58
            data = [("i. baseline lr = 3e-2\n(chép từ nơi khác)", 5.17, "68.85 ± 0.67", "74.02 ± 0.27", C2),
                    ("ii. tune lr cho cả hai", 0.29, "73.73 ± 0.93", "74.02 ± 0.27", C_ACC),
                    ("iii. tune lr × số bước\ncho cả hai", 0.52, "73.73 ± 0.93", "74.25 ± 0.40", C1)]
            xs = [-4.4, 0.0, 4.4]
            line = Line([-6.4, base_y, 0], [6.4, base_y, 0], color=GREY_B)
            self.play(Create(line))
        with noi(self, "Nếu chép learning rate của baseline từ nơi khác, baseline chưa hội tụ tốt, và phương pháp mới hơn hẳn năm điểm. Tune learning rate cho cả hai bên, chênh lệch co lại còn không phẩy hai chín, nhỏ hơn cả độ lệch chuẩn. Tune cả learning rate và số bước, chênh không phẩy năm hai. Khoảng chín mươi phần trăm của năm điểm kia là do baseline yếu."):
            cap = None
            groups = VGroup()
            for (name, gap, b, m, col), x in zip(data, xs):
                bar = Rectangle(width=1.5, height=gap * scale, color=col, fill_color=col, fill_opacity=0.85)
                bar.move_to([x, base_y + gap * scale / 2, 0])
                lbl = T(f"+{gap:.2f}", 30, col).next_to(bar, UP, buff=0.12)
                nm = T(name, 20).move_to([x, base_y - 0.65, 0])
                vals = VGroup(T(f"baseline {b}", 20, GREY_A), T(f"phương pháp {m}", 20, GREY_A)).arrange(DOWN, buff=0.1).move_to([x, 2.75, 0])
                groups.add(VGroup(bar, lbl, nm, vals))
            for i, g in enumerate(groups):
                self.play(FadeIn(g[2]), FadeIn(g[3]), GrowFromEdge(g[0], DOWN), FadeIn(g[1]), run_time=1.3)
                if i == 0:
                    cap = swap_caption(self, cap, "Baseline lr 3e-2 chưa hội tụ tốt: chênh +5.17 điểm")
                elif i == 1:
                    cap = swap_caption(self, cap, "Tune lr cho cả hai bên: chênh co còn +0.29, nhỏ hơn std 0.93 của baseline")
                else:
                    cap = swap_caption(self, cap, "Tune cả lr và số bước: +0.52. Khoảng 90% của \"5 điểm\" là do baseline yếu")
                self.wait(1.2)
            self.wait(2)


class NganSachCongBang(Scene):
    """Ví dụ 11.4: hiệu năng kỳ vọng theo ngân sách EVP(k) của hai không gian tìm kiếm."""

    def construct(self):
        title = T("Cùng ngân sách tune hay không?", 34).to_edge(UP, buff=0.3)
        with noi(self, "Hiệu năng kỳ vọng theo ngân sách tune: với k lần thử ngẫu nhiên, độ chính xác validation tốt nhất kỳ vọng là bao nhiêu. Baseline A có hai siêu tham số, phương pháp B có bốn."):
            self.play(Write(title))
            ks = [1, 2, 4, 8, 16, 32, 60]
            A = [0.6287, 0.6942, 0.7166, 0.7267, 0.7329, 0.7368, 0.7389]
            B = [0.6155, 0.6990, 0.7267, 0.7339, 0.7375, 0.7396, 0.7408]
            x0, dx, y0, ysc = -5.2, 1.3, -2.1, 22.0      # y = y0 + (v − 0.60) · ysc

            def P(i, v):
                return np.array([x0 + dx * i, y0 + (v - 0.60) * ysc, 0])

            axes = VGroup(Line([x0 - 0.3, y0, 0], [x0 + dx * 6 + 0.4, y0, 0], color=GREY_B),
                          Line([x0 - 0.3, y0, 0], [x0 - 0.3, y0 + 0.16 * ysc, 0], color=GREY_B))
            xt = VGroup(*[T(str(k), 18).move_to([x0 + dx * i, y0 - 0.28, 0]) for i, k in enumerate(ks)])
            xl = T("k = số lần thử (random search; trục không theo tỉ lệ)", 20).move_to([x0 + dx * 3, y0 - 0.62, 0])
            yt = VGroup(*[T(f"{v:.2f}", 18).move_to([x0 - 0.85, y0 + (v - 0.60) * ysc, 0]) for v in (0.60, 0.65, 0.70, 0.75)])
            yl = T("EVP(k): độ chính xác validation kỳ vọng", 18).rotate(PI / 2).move_to([x0 - 1.5, y0 + 1.6, 0])
            self.play(Create(axes), FadeIn(xt), FadeIn(xl), FadeIn(yt), FadeIn(yl))
            la = VMobject(color=C_A, stroke_width=5).set_points_as_corners([P(i, v) for i, v in enumerate(A)])
            lb = VMobject(color=C_B, stroke_width=5).set_points_as_corners([P(i, v) for i, v in enumerate(B)])
            da = VGroup(*[Dot(P(i, v), color=C_A, radius=0.07) for i, v in enumerate(A)])
            db = VGroup(*[Dot(P(i, v), color=C_B, radius=0.07) for i, v in enumerate(B)])
            leg = VGroup(T("A: baseline (2 siêu tham số)", 20, C_A), T("B: phương pháp (4 siêu tham số)", 20, C_B)).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
            leg.move_to([4.0, -0.7, 0])
            cap = swap_caption(self, None, "Hai không gian tìm kiếm ngẫu nhiên, mỗi bên 60 lần thử (VD 11.4)")
            self.play(Create(la), Create(lb), FadeIn(da), FadeIn(db), FadeIn(leg), run_time=2)
            self.wait(0.8)

        # so sánh không công bằng: A với k = 4 và B với k = 60
        with noi(self, "Nếu chỉ thử baseline bốn lần, nhưng tune phương pháp mới hết sáu mươi lần, ta thấy chênh hai phẩy bốn điểm."):
            pa, pb = P(2, A[2]), P(6, B[6])
            ha = DashedLine([pa[0], pa[1], 0], [x0 + dx * 6 + 0.4, pa[1], 0], color=C_A)
            hb = DashedLine([pb[0], pb[1], 0], [pb[0] + 0.4, pb[1], 0], color=C_B)
            ring_a = Circle(0.2, color=C_ACC).move_to(pa)
            ring_b = Circle(0.2, color=C_ACC).move_to(pb)
            cap = swap_caption(self, cap, "Chép A ở k = 4 (0.7166), B tune hết 60 lần (0.7408)")
            self.play(Create(ring_a), Create(ring_b), Create(ha), Create(hb))
            arrow = DoubleArrow([x0 + dx * 6 + 0.55, pa[1], 0], [x0 + dx * 6 + 0.55, pb[1], 0], color=C_ACC, buff=0, stroke_width=4)
            gap1 = T("chênh 2.4", 26, C_ACC).next_to(arrow, RIGHT, buff=0.12)
            self.play(Create(arrow), FadeIn(gap1))
            self.wait(2)

        # so sánh công bằng: cùng k = 60
        with noi(self, "Nhưng với cùng ngân sách sáu mươi lần thử, chênh lệch chỉ còn không phẩy hai điểm."):
            pa60 = P(6, A[6])
            cap = swap_caption(self, cap, "Cùng ngân sách k = 60: A = 0.7389, B = 0.7408")
            self.play(FadeOut(ring_a), FadeOut(ha), FadeOut(arrow), FadeOut(gap1))
            ring_a2 = Circle(0.2, color=C_ACC).move_to(pa60)
            arrow2 = DoubleArrow([pb[0] + 0.55, pa60[1], 0], [pb[0] + 0.55, pb[1], 0], color=C_ACC, buff=0, stroke_width=4)
            gap2 = T("chênh 0.2", 26, C_ACC).next_to(arrow2, RIGHT, buff=0.12)
            self.play(Create(ring_a2), Create(arrow2), FadeIn(gap2))
            self.wait(1)
        with noi(self, "Gần hết cải tiến hai phẩy bốn điểm thực ra là chênh lệch ngân sách tune."):
            cap = swap_caption(self, cap, "Gần hết \"cải tiến\" 2.4 điểm là chênh ngân sách tune; cùng k thì chỉ 0.2–0.4 điểm")
            self.wait(3)


class AblationTuongTac(Scene):
    """Ví dụ 11.5: ablation giai thừa 2³; remove-one và add-one trả lời hai câu hỏi khác nhau."""

    def construct(self):
        title = T("Ablation: bỏ một thành phần khác thêm một thành phần", 32).to_edge(UP, buff=0.3)
        with noi(self, "Ablation có hai cách: bỏ một thành phần khỏi mô hình đầy đủ, hoặc thêm một thành phần vào baseline. Hai cách trả lời hai câu hỏi khác nhau."):
            self.play(Write(title))
            zero_y, sc = 1.0, 0.5
            names = ["Dropout", "Weight decay", "LayerNorm"]
            rem = [1.52, 0.36, -1.35]
            add = [3.06, 0.96, -0.06]
            xs = [-4.2, 0.0, 4.2]
            line = Line([-6.4, zero_y, 0], [6.4, zero_y, 0], color=GREY_B)
            zl = T("0", 18).next_to(line, LEFT, buff=0.05).shift(LEFT * 0.0)
            leg = VGroup(T("remove-one: bỏ khỏi mô hình đủ", 20, C_A), T("add-one: thêm vào baseline", 20, C_B)).arrange(RIGHT, buff=0.7).move_to([0, -1.75, 0])
            self.play(Create(line), FadeIn(leg))
        with noi(self, "Đo trên mười seed với ba thành phần: dropout, weight decay và LayerNorm."):
            cap = swap_caption(self, None, "10 seed, test 4000 mẫu; điểm phần trăm so với mô hình đối chứng (VD 11.5)")

            def bar(x, v, col):
                h = max(abs(v) * sc, 0.03)
                r = Rectangle(width=1.0, height=h, color=col, fill_color=col, fill_opacity=0.85)
                r.move_to([x, zero_y + (h / 2 if v >= 0 else -h / 2), 0])
                lab = T(f"{v:+.2f}", 24, col)
                lab.next_to(r, UP if v >= 0 else DOWN, buff=0.1)
                return VGroup(r, lab)

            for nm, r_, a_, x in zip(names, rem, add, xs):
                g = VGroup(bar(x - 0.65, r_, C_A), bar(x + 0.65, a_, C_B))
                n = T(nm, 24).move_to([x, -1.1, 0])
                self.play(FadeIn(n), *[GrowFromEdge(b[0], DOWN if b[0].get_center()[1] > zero_y else UP) for b in g],
                          *[FadeIn(b[1]) for b in g], run_time=1.2)
                self.wait(0.4)
        with noi(self, "LayerNorm thêm vào riêng lẻ gần như vô hại, nhưng bỏ nó khỏi bộ đầy đủ lại làm tăng một phẩy ba lăm điểm."):
            cap = swap_caption(self, cap, "LayerNorm đứng một mình gần như vô hại (−0.06), nhưng bỏ nó khỏi bộ đủ lại tăng 1.35 điểm")
            self.wait(1.5)
        with noi(self, "Tổng ba hiệu ứng thêm vào là ba phẩy chín sáu, nhưng cả bộ chỉ hơn baseline hai phẩy hai. Các thành phần tương tác với nhau, và ablation một chiều che mất điều đó."):
            s1 = T("Tổng ba hiệu ứng add-one = 3.96, nhưng cả bộ chỉ hơn baseline 2.20", 26, C_ACC).move_to([0, -2.5, 0])
            self.play(FadeIn(s1))
            cap = swap_caption(self, cap, "Các thành phần tương tác: ablation một chiều che mất điều đó")
            self.wait(3)


# ---------------------------------------------------------------- 11.3
class HaiKhoangTinCay(Scene):
    """Ví dụ 11.6: CI theo 5 seed rất hẹp, nhưng test set 1000 mẫu cố định lệch khỏi giá trị thật."""

    def construct(self):
        title = T("Khoảng tin cậy theo seed không phải khoảng của độ chính xác thật", 28).to_edge(UP, buff=0.3)
        with noi(self, "Chạy năm seed trên cùng một test set một nghìn mẫu, trung bình được bảy mươi sáu phẩy sáu bốn phần trăm."):
            self.play(Write(title))
            lo_, hi_, x0, x1 = 72.0, 80.0, -5.8, 5.8
            sx = (x1 - x0) / (hi_ - lo_)

            def X(v):
                return x0 + (v - lo_) * sx

            yax = -1.9
            axis = Line([x0, yax, 0], [x1, yax, 0], color=GREY_B)
            ticks = VGroup(*[VGroup(Line([X(v), yax, 0], [X(v), yax - 0.1, 0], color=GREY_B), T(str(v), 18).move_to([X(v), yax - 0.35, 0]))
                             for v in range(72, 81)])
            axl = T("độ chính xác (%)", 20).move_to([0, yax - 0.8, 0])
            self.play(Create(axis), FadeIn(ticks), FadeIn(axl))

            cap = swap_caption(self, None, "5 seed trên một test set 1000 mẫu: 76.4, 76.6, 77.4, 76.8, 76.0")
            dots = VGroup(*[Dot([X(v), 1.3, 0], color=WHITE, radius=0.07) for v in (76.4, 76.6, 77.4, 76.8, 76.0)])
            mean_d = Dot([X(76.64), 1.3, 0], color=C_ACC, radius=0.13)
            ml = T("trung bình 76.64", 22, C_ACC).next_to(mean_d, UP, buff=0.2)
            self.play(FadeIn(dots), FadeIn(mean_d), FadeIn(ml))
        with noi(self, "Với chỉ năm mẫu phải dùng hệ số t bằng hai phẩy bảy bảy sáu. Dùng một phẩy chín sáu thì khoảng tin cậy quá hẹp, chỉ phủ khoảng tám mươi tám phần trăm."):
            ci_t = Line([X(76.00), 0.6, 0], [X(77.28), 0.6, 0], color=C1, stroke_width=10)
            tl = T("CI t: [76.00, 77.28]", 22, C1).next_to(ci_t, RIGHT, buff=0.3)
            ci_z = Line([X(76.19), 0.2, 0], [X(77.09), 0.2, 0], color=GREY_B, stroke_width=6)
            zl = T("hệ số 1.96: [76.19, 77.09]", 20, GREY_B).next_to(ci_z, RIGHT, buff=0.3)
            self.wait(0.5)
            cap = swap_caption(self, cap, "Với n = 5 phải dùng hệ số t = 2.776; hệ số 1.96 chỉ phủ 87.8%")
            self.play(Create(ci_t), FadeIn(tl), Create(ci_z), FadeIn(zl))
            self.wait(1.5)

        with noi(self, "Nhưng đo trên bốn mươi nghìn mẫu mới, độ chính xác thật chỉ là bảy mươi ba phẩy bảy. Khoảng tin cậy theo seed hoàn toàn không chứa nó."):
            true_line = DashedLine([X(73.70), 1.9, 0], [X(73.70), yax, 0], color=C2, stroke_width=5)
            tr_l = T("giá trị thật 73.70\n(40 000 mẫu mới)", 22, C2).next_to(true_line, LEFT, buff=0.15).shift(UP * 0.9)
            cap = swap_caption(self, cap, "Một lần rút xấu: lệch 2.94 điểm (2.1 SE); điển hình lệch khoảng 1 SE")
            self.play(Create(true_line), FadeIn(tr_l))
            self.wait(1.5)
        with noi(self, "Vì khoảng tin cậy theo seed bỏ qua sai số do chọn mẫu test. Riêng sai số đó đã là cộng trừ hai phẩy bảy điểm với một nghìn mẫu."):
            band = Rectangle(width=X(76.64 + 2.72) - X(76.64 - 2.72), height=0.35, color=C_B, fill_color=C_B, fill_opacity=0.4)
            band.move_to([X(76.64), -0.5, 0])
            bl = T("± 1.96 × 1.39 (sai số lấy mẫu test, nhị thức) = ± 2.72", 20, C_B).next_to(band, DOWN, buff=0.12)
            cap = swap_caption(self, cap, "Một lần rút xấu (2.1 SE): dải ± 1.96·SE quanh nó trượt sát 73.70; thường dải này chứa giá trị thật")
            self.play(FadeIn(band), FadeIn(bl))
            self.wait(3)


# ---------------------------------------------------------------- 11.4
class RoRiChonLocTruocKhiChia(Scene):
    """Ví dụ 11.8: nhãn ngẫu nhiên, chọn 20/5000 đặc trưng trước hoặc trong từng fold."""

    def construct(self):
        title = T("Chọn đặc trưng trước khi chia: 0.867 trên nhãn ngẫu nhiên", 30).to_edge(UP, buff=0.3)
        with noi(self, "Một trăm mẫu, năm nghìn đặc trưng ngẫu nhiên, và nhãn hoàn toàn ngẫu nhiên. So sánh hai quy trình: chọn hai mươi đặc trưng trên toàn bộ dữ liệu rồi mới chia fold, và chọn đặc trưng bên trong từng fold."):
            self.play(Write(title))

            def box(txt, x, y, col):
                t = T(txt, 18, WHITE)
                r = RoundedRectangle(corner_radius=0.1, width=max(t.width + 0.3, 1.9), height=t.height + 0.3, color=col, fill_color=col, fill_opacity=0.25)
                g = VGroup(r, t)
                return g.move_to([x, y, 0])

            cap = swap_caption(self, None, "n = 100 mẫu, 5000 đặc trưng Gauss độc lập, nhãn nhị phân hoàn toàn ngẫu nhiên")
            y1, y2 = 1.7, -0.8
            f1 = VGroup(box("Chọn 20 đặc trưng\ntrên cả 100 mẫu", -5.3, y1, C2), box("Chia 5 fold", -2.7, y1, GREY_B), box("Fit trên 4 fold,\nđo trên 1 fold", -0.2, y1, GREY_B))
            a1 = VGroup(Arrow(f1[0].get_right(), f1[1].get_left(), buff=0.05), Arrow(f1[1].get_right(), f1[2].get_left(), buff=0.05))
            f2 = VGroup(box("Chia 5 fold", -5.3, y2, GREY_B), box("Chọn 20 đặc trưng\nchỉ trên 4 fold train", -2.5, y2, C1), box("Fit trên 4 fold,\nđo trên 1 fold", 0.2, y2, GREY_B))
            a2 = VGroup(Arrow(f2[0].get_right(), f2[1].get_left(), buff=0.05), Arrow(f2[1].get_right(), f2[2].get_left(), buff=0.05))
            n1 = T("CHỌN TRƯỚC (rò rỉ)", 22, C2).next_to(f1, UP, buff=0.15).align_to(f1, LEFT)
            n2 = T("CHỌN TRONG TỪNG FOLD (đúng)", 22, C1).next_to(f2, UP, buff=0.15).align_to(f2, LEFT)
            self.play(FadeIn(n1), FadeIn(f1), Create(a1))
            self.play(FadeIn(n2), FadeIn(f2), Create(a2))
            self.wait(1)

        # biểu đồ cột bên phải
        with noi(self, "Chọn trước khi chia cho độ chính xác cross validation không phẩy tám sáu bảy, trên nhãn ngẫu nhiên. Chọn trong từng fold cho không phẩy bốn chín một, đúng như đoán mò."):
            base_y, sc, bx = -2.6, 4.0, [3.9, 5.7]
            axis = Line([3.0, base_y, 0], [6.6, base_y, 0], color=GREY_B)
            h1, h2 = 0.867 * sc, 0.491 * sc
            b1 = Rectangle(width=1.1, height=h1, color=C2, fill_color=C2, fill_opacity=0.85).move_to([bx[0], base_y + h1 / 2, 0])
            b2 = Rectangle(width=1.1, height=h2, color=C1, fill_color=C1, fill_opacity=0.85).move_to([bx[1], base_y + h2 / 2, 0])
            v1 = T("0.867", 24, C2).next_to(b1, UP, buff=0.1)
            v2 = T("0.491", 24, C1).next_to(b2, UP, buff=0.1)
            half = DashedLine([3.0, base_y + 0.5 * sc, 0], [6.6, base_y + 0.5 * sc, 0], color=C_ACC)
            hl = T("0.5", 20, C_ACC).next_to(half, LEFT, buff=0.08)
            cap = swap_caption(self, cap, "Độ chính xác CV trung bình của 30 lần lặp")
            self.play(Create(axis), GrowFromEdge(b1, DOWN), GrowFromEdge(b2, DOWN), FadeIn(v1), FadeIn(v2), Create(half), FadeIn(hl))
            self.wait(1.5)
        with noi(self, "Thử mô hình rò rỉ đó trên hai nghìn mẫu mới: chỉ được không phẩy bốn chín chín. Mọi bước học từ dữ liệu đều phải nằm trong fold."):
            cap = swap_caption(self, cap, "Nhãn ngẫu nhiên thì độ chính xác thật là 0.5; mô hình rò rỉ trên 2000 mẫu mới: 0.499")
            self.wait(3.5)


class BestOfK(Scene):
    """Ví dụ 11.10: chọn tốt nhất trong K mô hình có chất lượng thật bằng nhau (0.80) trên test 1000 mẫu."""

    def construct(self):
        title = T("Chọn tốt nhất trong K mô hình theo cùng một test set", 32).to_edge(UP, buff=0.3)
        with noi(self, "K mô hình có độ chính xác thật bằng nhau, đều là không phẩy tám. Ta chọn mô hình tốt nhất theo cùng một test set một nghìn mẫu."):
            self.play(Write(title))
            cap = swap_caption(self, None, "K mô hình đều có độ chính xác thật 0.80; test 1000 mẫu, SE nhị thức 0.0126")
            Ks = [1, 5, 20, 100, 1000]
            bias = [0.00, 1.44, 2.33, 3.12, 4.06]
            obs = [0.8000, 0.8144, 0.8233, 0.8311, 0.8400]
            base_y, sc = -1.4, 0.6
            xs = [-4.6, -2.3, 0.0, 2.3, 4.6]
            base = DashedLine([-6.0, base_y, 0], [6.0, base_y, 0], color=C_ACC)
            bl = T("độ chính xác thật (tập mới): 0.8000", 22, C_ACC).move_to([-3.0, base_y - 0.45, 0])
            self.play(Create(base), FadeIn(bl))
        with noi(self, "Càng nhiều mô hình, số đo cao nhất càng cao hơn giá trị thật: hơn một phẩy bốn điểm với năm mô hình, và hơn bốn điểm với một nghìn mô hình."):
            bars = VGroup()
            for K, b, o, x in zip(Ks, bias, obs, xs):
                h = max(b * sc, 0.02)
                r = Rectangle(width=1.2, height=h, color=C2, fill_color=C2, fill_opacity=0.85).move_to([x, base_y + h / 2, 0])
                v = T(f"+{b:.2f}", 26, C2).next_to(r, UP, buff=0.1)
                ob = T(f"{o:.4f}", 20, GREY_A).next_to(v, UP, buff=0.08)
                kl = T(f"K = {K}", 24).move_to([x, base_y - 1.0, 0])
                bars.add(VGroup(r, v, ob, kl))
            for g in bars:
                self.play(FadeIn(g[3]), GrowFromEdge(g[0], DOWN), FadeIn(g[1]), FadeIn(g[2]), run_time=0.9)
            cap = swap_caption(self, cap, "Số đo cao nhất trên test (xám) cao hơn giá trị thật; cột đỏ là thiên lệch, đơn vị điểm phần trăm")
            self.wait(2)
        with noi(self, "Thiên lệch tăng chậm theo K nhưng không bao giờ về không. Chọn theo test rồi báo cáo đúng con số đó là báo cáo một ước lượng lạc quan."):
            note = T("Thiên lệch ≈ SE × E[max của K biến chuẩn]: tăng chậm theo K, nhưng không về 0", 22, C_ACC).move_to([0, 2.55, 0])
            self.play(FadeIn(note))
            cap = swap_caption(self, cap, "Mỗi lần \"chọn theo test\" rồi báo cáo đúng con số test là báo cáo một ước lượng lạc quan")
            self.wait(3)
