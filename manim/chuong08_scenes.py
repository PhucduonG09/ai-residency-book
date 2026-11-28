"""Các cảnh Manim cho Chương 8 — Attention và Transformer.

Render một cảnh (chất lượng 720p):
    manim -qm chuong08_scenes.py AttentionBaTu

Chỉ dùng Text (không cần LaTeX), nên chỉ cần Manim Community + ffmpeg.
Mỗi cảnh dùng đúng số liệu của ví dụ giải mẫu trong sách (và của lần chạy lab).
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
CQ, CK, CV = BLUE, ORANGE, GREEN


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
        scene.play(FadeOut(old), FadeIn(new), run_time=0.5)
    return new


def matrix(rows, size=24, color=WHITE, dx=1.2, dy=0.55):
    """Ma trận dạng Text có ngoặc vuông. Ô (i, j) là m[0][i * ncol + j]."""
    nr, nc = len(rows), len(rows[0])
    cells = VGroup(*[T(str(v), size, color).move_to([(j - (nc - 1) / 2) * dx, -(i - (nr - 1) / 2) * dy, 0])
                     for i, r in enumerate(rows) for j, v in enumerate(r)])
    h, w, t = nr * dy + 0.1, nc * dx + 0.1, 0.12
    lb = VMobject(color=color).set_points_as_corners([[-w / 2 + t, h / 2, 0], [-w / 2, h / 2, 0], [-w / 2, -h / 2, 0], [-w / 2 + t, -h / 2, 0]])
    rb = VMobject(color=color).set_points_as_corners([[w / 2 - t, h / 2, 0], [w / 2, h / 2, 0], [w / 2, -h / 2, 0], [w / 2 - t, -h / 2, 0]])
    return VGroup(cells, lb, rb)


# ---------------------------------------------------------------- 8.1
class NutThatCoChai(Scene):
    """Ví dụ 8.1 và Lab 6: nút thắt c của seq2seq, rồi attention cộng tính với s = (1, 0), h₁ = (1, 1), h₂ = (0, 1)."""

    def construct(self):
        title = T("Nút thắt thông tin và attention", 34).to_edge(UP, buff=0.3)
        with noi(self, "Trong seq2seq cổ điển, encoder đọc cả câu nguồn, rồi nén toàn bộ vào một vector c duy nhất có kích thước cố định."):
            self.play(Write(title))
            # --- phần 1: nút thắt
            hs = VGroup(*[VGroup(Rectangle(width=1.0, height=0.7, color=CK), T(f"h{'₁₂₃₄'[i]}", 24, CK)) for i in range(4)])
            for g in hs:
                g[1].move_to(g[0])
            hs.arrange(RIGHT, buff=0.35).move_to(LEFT * 3.9 + UP * 0.8)
            c_box = VGroup(Rectangle(width=0.9, height=0.7, color=C_ACC, fill_opacity=0.35), T("c", 26, C_ACC))
            c_box[1].move_to(c_box[0])
            c_box.move_to(UP * 0.8 + RIGHT * 0.1)
            ys = VGroup(*[VGroup(Rectangle(width=1.0, height=0.7, color=C1), T(f"y{'₁₂₃'[i]}", 24, C1)) for i in range(3)])
            for g in ys:
                g[1].move_to(g[0])
            ys.arrange(RIGHT, buff=0.35).move_to(RIGHT * 3.9 + UP * 0.8)
            arrows_in = VGroup(*[Arrow(h.get_right() + RIGHT * 0.02, c_box.get_left(), buff=0.05, stroke_width=3,
                                       max_tip_length_to_length_ratio=0.3) for h in hs[-1:]])
            arr_out = Arrow(c_box.get_right(), ys[0].get_left(), buff=0.05, stroke_width=3)
            enc_lab = T("encoder (4 token nguồn)", 22).next_to(hs, UP, buff=0.3)
            dec_lab = T("decoder", 22).next_to(ys, UP, buff=0.3)
            self.play(FadeIn(hs), FadeIn(enc_lab))
            cap = swap_caption(self, None, "Seq2seq cổ điển: encoder đọc cả câu nguồn")
            self.play(FadeIn(c_box), FadeIn(arrows_in), run_time=1)
        with noi(self, "Decoder chỉ được nhìn vector c đó để sinh câu đích."):
            cap = swap_caption(self, cap, "Cả câu bị nén vào MỘT vector c có kích thước cố định")
            self.play(FadeIn(arr_out), FadeIn(ys), FadeIn(dec_lab), run_time=1)
            self.wait(0.5)
        with noi(self, "Câu càng dài, c càng phải chứa nhiều thông tin, và độ chính xác tụt mạnh: từ không phẩy sáu sáu bảy ở độ dài tám xuống không phẩy ba lăm ở độ dài mười sáu. Có attention thì gần như hoàn hảo."):
            stat = T("Bài chép chuỗi (Lab 6), không attention: acc 0.667 ở độ dài 8, 0.350 ở độ dài 16", 24, C2).move_to(DOWN * 1.2)
            stat2 = T("Có attention: 0.994 và 0.986", 24, C1).next_to(stat, DOWN, buff=0.3)
            cap = swap_caption(self, cap, "Câu càng dài, c càng phải chứa nhiều: độ chính xác tụt")
            self.play(FadeIn(stat))
            self.play(FadeIn(stat2))
            self.wait(1.0)
        with noi(self, "Xét một ví dụ nhỏ với hai trạng thái encoder, h một và h hai, và trạng thái decoder s."):
            self.play(FadeOut(VGroup(hs, c_box, ys, arrows_in, arr_out, enc_lab, dec_lab, stat, stat2)))
            cap = swap_caption(self, cap, "Sang ví dụ 8.1: chỉ xét hai trạng thái encoder, h₁ = (1, 1) và h₂ = (0, 1)")

            # --- phần 2: attention cộng tính (ví dụ 8.1)
            h1 = VGroup(Rectangle(width=2.4, height=0.7, color=CK), T("h₁ = (1, 1)", 24, CK))
            h2 = VGroup(Rectangle(width=2.4, height=0.7, color=CK), T("h₂ = (0, 1)", 24, CK))
            s = VGroup(Rectangle(width=2.4, height=0.7, color=C1), T("s = (1, 0)", 24, C1))
            for g in (h1, h2, s):
                g[1].move_to(g[0])
            h1.move_to(LEFT * 4.5 + UP * 1.5)
            h2.move_to(LEFT * 4.5 + DOWN * 0.1)
            s.move_to(RIGHT * 2.0 + UP * 0.7)
            self.play(FadeIn(h1), FadeIn(h2), FadeIn(s))
        with noi(self, "Decoder chấm điểm từng trạng thái encoder bằng attention cộng tính, rồi softmax biến điểm thành trọng số: không phẩy năm năm và không phẩy bốn năm."):
            cap = swap_caption(self, cap, "Decoder chấm điểm từng trạng thái encoder: e = vᵀ tanh(Wsᵢ₋₁ + Uhⱼ)")
            a1 = Arrow(s.get_left() + UP * 0.1, h1.get_right(), buff=0.1, stroke_width=3 + 8 * 0.5504, color=C_ACC)
            a2 = Arrow(s.get_left() + DOWN * 0.1, h2.get_right(), buff=0.1, stroke_width=3 + 8 * 0.4496, color=C_ACC)
            l1 = T("e₁ = 1.7256  →  α₁ = 0.5504", 22, C_ACC).next_to(a1, UP, buff=0.12).shift(RIGHT * 0.3)
            l2 = T("e₂ = 1.5232  →  α₂ = 0.4496", 22, C_ACC).next_to(a2, DOWN, buff=0.12).shift(RIGHT * 0.3)
            self.play(GrowArrow(a1), GrowArrow(a2), FadeIn(l1), FadeIn(l2), run_time=1.5)
        with noi(self, "Vector ngữ cảnh c là trung bình có trọng số của các trạng thái encoder. Mỗi bước giải mã có một c riêng."):
            cap = swap_caption(self, cap, "softmax biến điểm thành trọng số; mũi tên dày = chú ý nhiều")
            ctx = T("c = 0.5504·h₁ + 0.4496·h₂ = (0.5504, 1.0000)", 26, WHITE).move_to(DOWN * 1.5)
            self.play(FadeIn(ctx))
            self.wait(1)
        with noi(self, "Nếu dùng tích vô hướng làm hàm điểm, trọng số thành không phẩy bảy ba và không phẩy hai bảy. Cơ chế vẫn như nhau: chấm điểm, softmax, rồi lấy trung bình có trọng số."):
            dot = T("Nếu dùng dot-product e = h·s: α = (0.7311, 0.2689), c = (0.7311, 1.0000)", 24, CV).next_to(ctx, DOWN, buff=0.35)
            cap = swap_caption(self, cap, "Cùng cơ chế (điểm, softmax, trung bình có trọng số), khác hàm điểm")
            self.play(FadeIn(dot))
            self.wait(2)


# ---------------------------------------------------------------- 8.2
class AttentionBaTu(Scene):
    """Ví dụ 8.3: Q = K = (1,0), (0,1), (1,1); V = (1,2), (3,4), (5,6); dₖ = 2; rồi mặt nạ nhân quả."""

    def construct(self):
        title = T("Attention với 3 token, dₖ = 2", 34).to_edge(UP, buff=0.3)
        with noi(self, "Attention với ba token. Bước một: tính điểm giữa mọi cặp query và key, tức Q nhân K chuyển vị."):
            self.play(Write(title))
            S = [["1", "0", "1"], ["0", "1", "1"], ["1", "1", "2"]]
            Sd = [["0.707", "0", "0.707"], ["0", "0.707", "0.707"], ["0.707", "0.707", "1.414"]]
            A = [["0.4011", "0.1978", "0.4011"], ["0.1978", "0.4011", "0.4011"], ["0.2483", "0.2483", "0.5035"]]
            Ac = [["1", "0", "0"], ["0.3302", "0.6698", "0"], ["0.2483", "0.2483", "0.5035"]]
            Vm = [["1", "2"], ["3", "4"], ["5", "6"]]
            O = [["3.0000", "4.0000"], ["3.4067", "4.4067"], ["3.5105", "4.5105"]]
            Oc = [["1.0000", "2.0000"], ["2.3395", "3.3395"], ["3.5105", "4.5105"]]
            pos_a = LEFT * 4.3 + UP * 0.2
            m = matrix(S, color=CQ, dx=1.45).move_to(pos_a)
            lab = T("QKᵀ", 26, CQ).next_to(m, UP, buff=0.3)
            self.play(FadeIn(m), FadeIn(lab))
            cap = swap_caption(self, None, "Điểm giữa mọi cặp query–key: QKᵀ")
            self.wait(0.6)
        with noi(self, "Chia cho căn d k, ở đây là căn hai, để giữ độ lớn của điểm ổn định."):
            m2 = matrix(Sd, color=CQ, dx=1.45).move_to(pos_a)
            lab2 = T("QKᵀ / √2", 26, CQ).move_to(lab)
            self.play(ReplacementTransform(m, m2), ReplacementTransform(lab, lab2))
            cap = swap_caption(self, cap, "Chia √dₖ = √2 để giữ độ lớn của điểm")
            self.wait(0.6)
        with noi(self, "Softmax theo từng hàng: mỗi hàng thành một phân phối, không âm và tổng bằng một."):
            m3 = matrix(A, color=C_ACC, dx=1.45).move_to(pos_a)
            lab3 = T("A = softmax từng hàng", 26, C_ACC).move_to(lab2)
            sums = VGroup(*[T("tổng 1", 20, C_ACC).move_to(pos_a + RIGHT * 2.95 + UP * (1 - i) * 0.55) for i in range(3)])
            self.play(ReplacementTransform(m2, m3), ReplacementTransform(lab2, lab3))
            self.play(FadeIn(sums))
            cap = swap_caption(self, cap, "Mỗi hàng là một phân phối: không âm, tổng bằng 1")
            self.wait(0.6)
            self.play(FadeOut(sums))
        with noi(self, "Nhân với V: đầu ra mỗi vị trí là trung bình có trọng số các hàng của V. Hàng đầu tiên ra ba, bốn."):
            times = T("×", 30).move_to(LEFT * 1.3 + UP * 0.2)
            mv = matrix(Vm, color=CV, dx=0.9).move_to(LEFT * 0.2 + UP * 0.2)
            labv = T("V", 26, CV).next_to(mv, UP, buff=0.3)
            eq = T("=", 30).move_to(RIGHT * 1.15 + UP * 0.2)
            mo = matrix(O, color=WHITE, dx=1.3).move_to(RIGHT * 3.3 + UP * 0.2)
            labo = T("O = A·V", 26).next_to(mo, UP, buff=0.3)
            self.play(FadeIn(times), FadeIn(mv), FadeIn(labv))
            self.play(FadeIn(eq), FadeIn(mo), FadeIn(labo))
            cap = swap_caption(self, cap, "Đầu ra mỗi vị trí = trung bình có trọng số các hàng của V: O₁ = (3.0, 4.0)")
            self.wait(1.5)
        # mặt nạ nhân quả
        with noi(self, "Với mặt nạ nhân quả, vị trí i không được nhìn các vị trí sau nó. Ta che trước khi lấy softmax."):
            mc = matrix(Ac, color=C_ACC, dx=1.45).move_to(pos_a)
            grey = VGroup(*[m3[0][i * 3 + j] for i in range(3) for j in range(3) if j > i])
            self.play(grey.animate.set_color(GREY_D), run_time=0.8)
            cap = swap_caption(self, cap, "Mặt nạ nhân quả: vị trí i không được nhìn j > i (che TRƯỚC softmax)")
            self.wait(0.6)
        with noi(self, "Hàng đầu chỉ nhìn chính nó. Hàng hai chia lại thành không phẩy ba ba và không phẩy sáu bảy. Hàng ba vốn nhìn được tất cả nên không đổi."):
            moc = matrix(Oc, color=WHITE, dx=1.3).move_to(RIGHT * 3.3 + UP * 0.2)
            self.play(ReplacementTransform(m3, mc), ReplacementTransform(mo, moc))
            cap = swap_caption(self, cap, "Hàng 2 chỉ còn (0.3302, 0.6698): O₂ = (2.3395, 3.3395); hàng 3 không đổi")
            self.wait(2.5)


class BaoHoaSoftmax(Scene):
    """Mục 8.2: logit (8, 0) so với (1, 0) sau khi chia √64 = 8; đạo hàm p(1 − p)."""

    def construct(self):
        title = T("Vì sao chia √dₖ: tránh bão hòa softmax", 34).to_edge(UP, buff=0.3)
        with noi(self, "Vì sao phải chia cho căn d k? Với d k bằng sáu mươi tư, tích q nhân k có độ lệch chuẩn khoảng tám, nên logit cỡ tám là bình thường."):
            self.play(Write(title))
            base_y = -1.2
            scale = 2.4

            def panel(cx, logits, probs, head):
                line = Line([cx - 2.0, base_y, 0], [cx + 2.0, base_y, 0], color=GREY)
                bars = VGroup()
                labs = VGroup()
                for k, (p, col) in enumerate(zip(probs, (CQ, C2))):
                    bar = Rectangle(width=1.1, height=max(p * scale, 0.02), color=col, fill_opacity=0.8, stroke_width=1)
                    bar.move_to([cx - 1.1 + 2.2 * k, base_y + max(p * scale, 0.02) / 2, 0])
                    bars.add(bar)
                    labs.add(T(f"p{'₁₂'[k]} = {p:.4f}", 20, col).next_to(line, DOWN, buff=0.15).set_x(cx - 1.1 + 2.2 * k))
                hd = T(head, 24).move_to([cx, 2.5, 0])
                lg = T(f"logit = {logits}", 24, C_ACC).next_to(hd, DOWN, buff=0.2)
                return VGroup(line, bars, labs, hd, lg), bars

            left, bl = panel(-3.6, "(8, 0)", (0.9997, 0.0003), "Không chia")
            right, br = panel(3.6, "(1, 0)", (0.7311, 0.2689), "Chia √64 = 8")
            self.play(FadeIn(left))
            cap = swap_caption(self, None, "dₖ = 64: tích q·k có độ lệch chuẩn √dₖ = 8, nên logit cỡ 8 là bình thường")
            self.wait(1.5)
        with noi(self, "Với logit tám và không, softmax gần như one-hot. Đạo hàm theo logit chỉ còn khoảng ba phần mười nghìn: gradient gần như biến mất."):
            d1 = T("p₁(1 − p₁) = 3.35e-4", 24, C2).move_to([-3.6, -2.4, 0])
            self.play(FadeIn(d1))
            cap = swap_caption(self, cap, "Softmax gần one-hot: đạo hàm theo logit chỉ còn 3.35e-4")
            self.wait(2)
        with noi(self, "Chia cho tám đưa logit về một và không. Đạo hàm lên không phẩy hai, lớn gấp gần sáu trăm lần."):
            arr = Arrow(LEFT * 0.9 + UP * 0.4, RIGHT * 0.9 + UP * 0.4, buff=0, color=C_ACC)
            arr_l = T("÷ 8", 26, C_ACC).next_to(arr, UP, buff=0.1)
            self.play(GrowArrow(arr), FadeIn(arr_l))
            self.play(FadeIn(right))
            d2 = T("p₁(1 − p₁) = 0.1966", 24, C1).move_to([3.6, -2.4, 0])
            self.play(FadeIn(d2))
        cap = swap_caption(self, cap, "Chia √dₖ đưa logit về cỡ 1: đạo hàm lớn gấp gần 600 lần")
        self.wait(3)


# ---------------------------------------------------------------- 8.3
class SongSinViTri(Scene):
    """Ví dụ 8.5: PE sinusoidal d = 4, vị trí 0 đến 3; cặp đầu quay 1 rad/bước, cặp sau 0.01 rad/bước."""

    def construct(self):
        title = T("Mã hóa vị trí sinusoidal: các kim đồng hồ", 34).to_edge(UP, buff=0.3)
        with noi(self, "Mã hóa vị trí sinusoidal giống như các kim đồng hồ. Mỗi cặp chiều là một kim quay với tốc độ riêng: kim một quay một radian mỗi bước, kim hai chỉ không phẩy không một radian."):
            self.play(Write(title))
            R = 1.15
            centers = [LEFT * 5.2 + UP * 0.6, LEFT * 2.1 + UP * 0.6]
            omegas = [1.0, 0.01]
            names = ["cặp 1\n1 rad mỗi bước", "cặp 2\n0.01 rad mỗi bước"]
            dials, hands = VGroup(), []
            for c, w, nm in zip(centers, omegas, names):
                circ = Circle(radius=R, color=GREY).move_to(c)
                h = Arrow(c, c + UP * R, buff=0, color=C_ACC, stroke_width=5, max_tip_length_to_length_ratio=0.2)
                lb = T(nm, 18).next_to(circ, DOWN, buff=0.2)
                dials.add(circ, lb)
                hands.append(h)
            self.play(FadeIn(dials), *[GrowArrow(h) for h in hands])
            head = VGroup(T("pos", 22).move_to([1.2, 2.2, 0]), T("PE = (sin, cos, sin, cos)", 22).move_to([1.8, 2.2, 0]))
            head[1].align_to([1.8, 0, 0], LEFT)
            self.play(FadeIn(head))
        with noi(self, "Ở vị trí không, mọi kim chỉ lên. Qua từng vị trí, kim một quay nhanh, còn kim hai gần như đứng yên."):
            rows = [("0", "(0, 1, 0, 1)"), ("1", "(0.8415, 0.5403, 0.0100, 1.0000)"),
                    ("2", "(0.9093, −0.4161, 0.0200, 0.9998)"), ("3", "(0.1411, −0.9900, 0.0300, 0.9996)")]
            row_m = VGroup()
            cap = None
            for pos, (a, b) in enumerate(rows):
                r = VGroup(T(a, 20).move_to([1.2, 1.5 - 0.6 * pos, 0]), T(b, 20).move_to([1.8, 1.5 - 0.6 * pos, 0]))
                r[1].align_to([1.8, 0, 0], LEFT)
                row_m.add(r)
                if pos == 0:
                    self.play(FadeIn(r))
                    cap = swap_caption(self, cap, "Vị trí 0: mọi kim chỉ lên, PE = (0, 1, 0, 1)")
                else:
                    anims = [FadeIn(r)]
                    for k, (c, w) in enumerate(zip(centers, omegas)):
                        th = pos * w
                        new = Arrow(c, c + R * np.array([np.sin(th), np.cos(th), 0]), buff=0, color=C_ACC, stroke_width=5,
                                    max_tip_length_to_length_ratio=0.2)
                        anims.append(Transform(hands[k], new))
                    self.play(*anims, run_time=1.4)
                    if pos == 1:
                        cap = swap_caption(self, cap, "Kim 1 quay nhanh (1 rad), kim 2 gần như đứng yên (0.01 rad)")
                self.wait(0.6)
        with noi(self, "Kim nhanh giúp phân biệt các vị trí gần nhau, kim chậm phân biệt các vị trí xa."):
            cap = swap_caption(self, cap, "Kim nhanh phân biệt vị trí gần, kim chậm phân biệt vị trí xa")
            self.wait(1.5)
        with noi(self, "Dịch vị trí đi k bước tương đương quay mỗi cặp một góc cố định, tức một phép biến đổi tuyến tính. Với d bằng năm trăm mười hai, bước sóng trải từ khoảng sáu tới sáu mươi nghìn vị trí."):
            note = T("PE(pos + k) = quay từng cặp một góc ωk: dịch vị trí là phép biến đổi tuyến tính", 22, C_ACC).move_to(DOWN * 2.2)
            note2 = T("d = 512: bước sóng từ 6.28 đến khoảng 60 611 vị trí", 22, WHITE).next_to(note, DOWN, buff=0.25)
            self.play(FadeIn(note))
            self.play(FadeIn(note2))
        self.wait(3)


class RoPEQuay(Scene):
    """Ví dụ 8.6: q = (1, 2), k = (3, 1), θ = π/6; điểm chỉ phụ thuộc n − m."""

    def construct(self):
        title = T("RoPE: xoay q và k, điểm chỉ phụ thuộc n − m", 34).to_edge(UP, buff=0.3)
        with noi(self, "RoPE mã hóa vị trí bằng cách xoay. Không xoay thì q nhân k bằng năm. Token ở vị trí m có q bị xoay một góc m theta, token ở vị trí n có k xoay n theta."):
            self.play(Write(title))
            ax = Axes(x_range=[-4, 4, 1], y_range=[-4, 4, 1], x_length=5.6, y_length=5.6, tips=False,
                      axis_config={"stroke_opacity": 0.5}).shift(LEFT * 4.0 + DOWN * 0.1)
            self.play(Create(ax))
            q, k = np.array([1.0, 2.0]), np.array([3.0, 1.0])
            th = np.pi / 6

            def rot(v, a):
                c, s = np.cos(a), np.sin(a)
                return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

            def vec(v, col):
                return Arrow(ax.c2p(0, 0), ax.c2p(*v), buff=0, color=col, stroke_width=5, max_tip_length_to_length_ratio=0.15)

            qa, ka = vec(q, C1), vec(k, C2)
            def place(lbl, pt):
                quad = (UR if pt[0] >= 0 else UL) if pt[1] >= 0 else (DR if pt[0] >= 0 else DL)
                return lbl.next_to(ax.c2p(*pt), quad, buff=0.08)

            lq = place(T("q = (1, 2)", 22, C1), q)
            lk = place(T("k = (3, 1)", 22, C2), k)
            self.play(GrowArrow(qa), GrowArrow(ka), FadeIn(lq), FadeIn(lk))
            t1 = T("θ = π/6 mỗi vị trí", 24).move_to(RIGHT * 2.0 + UP * 2.4)
            t1.align_to([0.0, 0, 0], LEFT)
            t2 = T("Không quay: q·k = 5", 24, WHITE).next_to(t1, DOWN, buff=0.3).align_to(t1, LEFT)
            self.play(FadeIn(t1), FadeIn(t2))
            cap = swap_caption(self, None, "Vị trí m quay q một góc mθ, vị trí n quay k một góc nθ")
            self.wait(1)
        with noi(self, "Với cặp vị trí hai và không, điểm là trừ một phẩy tám ba. Cặp năm và ba, rồi cặp mười hai và mười, cả hai vector cùng xoay thêm, nhưng góc giữa chúng không đổi, nên điểm vẫn y như vậy."):
            results = VGroup()
            prev = None
            for (m, n) in [(2, 0), (5, 3), (12, 10)]:
                qn = rot(q, m * th)
                kn = rot(k, n * th)
                qa2, ka2 = vec(qn, C1), vec(kn, C2)
                score = float(qn @ kn)
                assert abs(score + 1.8301) < 1e-3
                row = T(f"({m}, {n}): điểm = −1.8301", 24, C_ACC).next_to(t2, DOWN, buff=0.45 + 0.5 * len(results)).align_to(t1, LEFT)
                results.add(row)
                lq2 = place(T(f"R({m}θ)q" if m else "q", 22, C1), qn)
                lk2 = place(T(f"R({n}θ)k" if n else "k = (3, 1)", 22, C2), kn)
                self.play(Transform(qa, qa2), Transform(ka, ka2),
                          ReplacementTransform(lq, lq2), ReplacementTransform(lk, lk2), run_time=1.2)
                lq, lk = lq2, lk2
                self.play(FadeIn(row))
                if prev is None:
                    cap = swap_caption(self, cap, "(2, 0): q quay 60°, k đứng yên; góc giữa hai vector tăng từ 45° lên 105°")
                elif (m, n) == (5, 3):
                    cap = swap_caption(self, cap, "(5, 3): cả hai cùng quay thêm, góc giữa chúng không đổi")
                else:
                    cap = swap_caption(self, cap, "(12, 10): vẫn n − m = −2, vẫn cùng điểm")
                prev = row
                self.wait(0.7)
        with noi(self, "Điểm attention chỉ phụ thuộc khoảng cách tương đối n trừ m, không phụ thuộc vị trí tuyệt đối."):
            fin = T("Chỉ phụ thuộc n − m = −2", 24, WHITE).next_to(results, DOWN, buff=0.4).align_to(t1, LEFT)
            self.play(FadeIn(fin))
        self.wait(2)


# ---------------------------------------------------------------- 8.4
class PreLNvsPostLN(Scene):
    """Mục 8.4: hai cách đặt LayerNorm và độ lớn dòng residual của mô hình pre-LN (số đo ở Lab 2)."""

    def construct(self):
        title = T("Post-LN và pre-LN", 34).to_edge(UP, buff=0.3)
        with noi(self, "Có hai cách đặt LayerNorm trong khối Transformer. Post LN, như paper gốc, chuẩn hóa sau khi cộng residual, nên cả đường tắt cũng đi qua LayerNorm."):
            self.play(Write(title))

            def box(txt, col=WHITE, w=1.5):
                r = Rectangle(width=w, height=0.6, color=col)
                t = T(txt, 20, col).move_to(r)
                return VGroup(r, t)

            def diagram(cx, kind):
                x_in = T("x", 24).move_to([cx, 2.3, 0])
                if kind == "post":
                    sub, add, ln = box("Sublayer", CK), box("+", WHITE, 0.6), box("LayerNorm", C_ACC, 1.9)
                    col_items = [sub, add, ln]
                else:
                    ln, sub, add = box("LayerNorm", C_ACC, 1.9), box("Sublayer", CK), box("+", WHITE, 0.6)
                    col_items = [ln, sub, add]
                for i, it in enumerate(col_items):
                    it.move_to([cx, 1.4 - 0.95 * i, 0])
                out = T("x₂", 24).move_to([cx, 1.4 - 0.95 * 3 + 0.1, 0])
                arrows = VGroup()
                pts = [x_in] + col_items + [out]
                for a, b in zip(pts[:-1], pts[1:]):
                    arrows.add(Arrow(a.get_bottom(), b.get_top(), buff=0.05, stroke_width=3, max_tip_length_to_length_ratio=0.25))
                # đường tắt (residual)
                add_box = add
                skip = VMobject(color=C1, stroke_width=3).set_points_as_corners(
                    [x_in.get_bottom() + DOWN * 0.1 + RIGHT * 0.0, [cx + 1.3, x_in.get_bottom()[1] - 0.1, 0],
                     [cx + 1.3, add_box.get_center()[1], 0], add_box.get_right()])
                skip_arrow = Arrow([cx + 1.3, add_box.get_center()[1], 0], add_box.get_right() + RIGHT * 0.02, buff=0, color=C1, stroke_width=3,
                                   max_tip_length_to_length_ratio=0.4)
                return VGroup(x_in, *col_items, out, arrows, skip, skip_arrow)

            dp = diagram(-4.6, "post")
            dq = diagram(-0.9, "pre")
            hp = T("post-LN (paper gốc)", 22).move_to([-4.6, 2.9, 0])
            hq = T("pre-LN (GPT-2, nanoGPT)", 22).move_to([-0.9, 2.9, 0])
            fp = T("x₂ = LN(x + Sub(x))", 22, C_ACC).move_to([-4.6, -2.1, 0])
            fq = T("x₂ = x + Sub(LN(x))", 22, C_ACC).move_to([-0.9, -2.1, 0])
            self.play(FadeIn(dp), FadeIn(hp))
            self.play(FadeIn(fp))
            cap = swap_caption(self, None, "Post-LN: chuẩn hóa SAU khi cộng; đường tắt cũng đi qua LN")
            self.wait(1.5)
        with noi(self, "Pre LN, dùng trong GPT hai, chỉ chuẩn hóa đầu vào của nhánh con. Đường tắt đi thẳng, gradient chảy về dễ hơn, nên huấn luyện ổn định hơn."):
            self.play(FadeIn(dq), FadeIn(hq))
            self.play(FadeIn(fq))
            cap = swap_caption(self, cap, "Pre-LN: chuẩn hóa nhánh con; đường tắt đi thẳng, không qua LN")
            self.wait(2)

        # đồ thị cột: std dòng residual của pre-LN sau 1, 6, 12, 24 khối (Lab 2)
        with noi(self, "Đổi lại, dòng residual không được chuẩn hóa, nên độ lệch chuẩn tăng dần qua các khối, từ một phẩy không hai lên một phẩy năm năm sau hai mươi tư khối. Vì vậy cần thêm một LayerNorm cuối cùng trước lớp ra."):
            vals = [1.020, 1.128, 1.257, 1.555]
            labels = ["1", "6", "12", "24"]
            base_y = -1.2
            sc = 2.2
            bars = VGroup()
            bl = VGroup()
            for i, (v, lb) in enumerate(zip(vals, labels)):
                h = (v - 0.9) * 4.0
                b = Rectangle(width=0.7, height=h, color=C_ACC, fill_opacity=0.8, stroke_width=1)
                b.move_to([3.2 + 1.0 * i, base_y + h / 2, 0])
                bars.add(b)
                bl.add(T(f"{v:.3f}", 20, WHITE).next_to(b, UP, buff=0.08))
                bl.add(T(f"{lb} khối", 16, GREY_B).move_to([3.2 + 1.0 * i, base_y - 0.3, 0]))
            axis = Line([2.6, base_y, 0], [6.4, base_y, 0], color=GREY)
            hd = T("pre-LN: std của dòng residual", 22).move_to([4.6, 2.9, 0])
            hd2 = T("(d = 256, 24 khối, đầu vào N(0, 1))", 18, GREY_B).next_to(hd, DOWN, buff=0.1)
            self.play(FadeIn(hd), FadeIn(hd2), FadeIn(axis))
            self.play(FadeIn(bars), FadeIn(bl), run_time=1.5)
            cap = swap_caption(self, cap, "Dòng residual không được chuẩn hóa nên std tăng dần: cần LN cuối trước lớp ra")
            self.wait(3)


# ---------------------------------------------------------------- 8.5
class KVCache(Scene):
    """Mục 8.5: giải mã từng token, K và V của token cũ được giữ lại; GPT-2 small 36 864 byte mỗi token."""

    def construct(self):
        title = T("KV cache khi sinh từng token", 34).to_edge(UP, buff=0.3)
        with noi(self, "Khi sinh từng token, mỗi token mới chỉ cần tính q, k, v của riêng nó, rồi nối k và v vào cache. Query của nó chấm với toàn bộ K trong cache. Nhờ mặt nạ nhân quả, K và V của token cũ không đổi, nên không phải tính lại."):
            self.play(Write(title))
            x0 = -4.8
            dxx = 1.35
            tok_y, k_y, v_y = 1.9, 0.5, -0.5
            self.play(FadeIn(T("token", 22).move_to([-6.2, tok_y, 0])),
                      FadeIn(T("K cache", 22, CK).move_to([-6.2, k_y, 0])),
                      FadeIn(T("V cache", 22, CV).move_to([-6.2, v_y, 0])))
            cap = None
            toks = VGroup()
            for i in range(5):
                x = x0 + dxx * i + 0.6
                tk = VGroup(Rectangle(width=1.0, height=0.55, color=WHITE), T(f"t{'₁₂₃₄₅'[i]}", 22)).move_to([x, tok_y, 0])
                tk[1].move_to(tk[0])
                kk = Rectangle(width=1.0, height=0.55, color=CK, fill_opacity=0.5).move_to([x, k_y, 0])
                vv = Rectangle(width=1.0, height=0.55, color=CV, fill_opacity=0.5).move_to([x, v_y, 0])
                self.play(FadeIn(tk), run_time=0.5)
                self.play(FadeIn(kk, shift=DOWN * 0.3), FadeIn(vv, shift=DOWN * 0.3), run_time=0.6)
                msg = {0: "Token mới: tính q, k, v của RIÊNG nó, nối k và v vào cache",
                       1: "Query của token mới chấm với toàn bộ K trong cache, rồi lấy trung bình V",
                       2: "K và V của token cũ không đổi (mặt nạ nhân quả): không tính lại",
                       3: "Mỗi bước chỉ thêm một cột vào cache", 4: "Cache lớn dần theo độ dài chuỗi"}[i]
                cap = swap_caption(self, cap, msg)
                toks.add(tk)
                self.wait(0.2)
            self.wait(0.5)
        with noi(self, "Sinh một nghìn token mà không có cache phải xử lý hơn năm trăm nghìn lượt token. Có cache chỉ cần một nghìn lượt."):
            c1 = T("n = 1000: không cache cần n(n + 1)/2 = 500 500 lượt xử lý token", 22, C2).move_to(DOWN * 1.2)
            c2 = T("có cache: chỉ n = 1000 lượt", 22, C1).next_to(c1, DOWN, buff=0.25)
            self.play(FadeIn(c1))
            self.play(FadeIn(c2))
        with noi(self, "Đổi lại, cache tốn bộ nhớ. Với GPT hai small ở fp mười sáu, mỗi token tốn khoảng ba mươi sáu kilobyte, tức ba mươi sáu megabyte cho chuỗi một nghìn không trăm hai mươi tư token."):
            cap = swap_caption(self, cap, "Đổi lại cache tốn bộ nhớ: 2 · L · (số đầu K/V) · dₕ · byte, mỗi token")
            self.wait(1.0)
            c3 = T("GPT-2 small, fp16: 36 864 byte mỗi token, tức 36 MiB ở T = 1024", 22, C_ACC).next_to(c2, DOWN, buff=0.3)
            self.play(FadeIn(c3))
        self.wait(2.5)
