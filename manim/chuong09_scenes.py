"""Các cảnh Manim cho Chương 9 — Mô hình ngôn ngữ lớn.

Render một cảnh (720p):
    manim -qm chuong09_scenes.py BPEGhepCap

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
    if t.width > 12.6:
        t.scale_to_fit_width(12.6)
    return VGroup(BackgroundRectangle(t, fill_opacity=0.85, buff=0.12), t).to_edge(DOWN, buff=0.25)


def swap_caption(scene, old, text, size=24):
    new = caption(text, size)
    if old is None:
        scene.play(FadeIn(new), run_time=0.5)
    else:
        scene.play(FadeOut(old), FadeIn(new), run_time=0.5)
    return new


def title_of(s):
    return T(s, 34).to_edge(UP, buff=0.3)


# ---------------------------------------------------------------- 9.1
class BPEGhepCap(Scene):
    """Ví dụ 9.1: bốn từ của Sennrich; ba phép gộp đầu hòa nhau ở 9."""

    def construct(self):
        with noi(self, "Byte Pair Encoding xây từ vựng bằng cách gộp dần các cặp ký tự. Bắt đầu với bốn từ và tần số của chúng, mỗi từ tách thành từng ký tự."):
            self.play(Write(title_of("BPE: gộp cặp phổ biến nhất")))
            forms = [
                ["l o w </w>", "l o w e r </w>", "n e w e s t </w>", "w i d e s t </w>"],
                ["l o w </w>", "l o w e r </w>", "n e w est</w>", "w i d est</w>"],
                ["low </w>", "low e r </w>", "n e w est</w>", "w i d est</w>"],
            ]
            freq = ["5", "2", "6", "3"]

            def rows(f):
                g = VGroup(*[T(f"{w}   ×{n}", 26) for w, n in zip(f, freq)]).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
                return g.move_to(LEFT * 3.6 + UP * 0.3)

            cur = rows(forms[0])
            self.play(FadeIn(cur))
        with noi(self, "Đếm mọi cặp liền kề, có nhân với tần số, rồi gộp cặp xuất hiện nhiều nhất. Ba phép gộp đầu tạo ra e s, e s t, rồi e s t kèm dấu kết thúc từ."):
            merges = [("1. (e, s) 9 → es", C_ACC), ("2. (es, t) 9 → est", C_ACC), ("3. (est, </w>) 9 → est</w>", C_ACC),
                      ("4. (l, o) 7 → lo", WHITE), ("5. (lo, w) 7 → low", WHITE)]
            col = VGroup(*[T(m, 24, c) for m, c in merges]).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 3.6 + UP * 0.3)
            cap = swap_caption(self, None, "Đếm mọi cặp liền kề (có trọng số tần số), gộp cặp nhiều nhất")
            for i in range(3):
                self.play(FadeIn(col[i]), run_time=0.7)
            self.play(FadeTransform(cur, rows(forms[1])), run_time=0.8)
            cur = rows(forms[1])
            self.wait(0.5)
        with noi(self, "Để ý ba cặp đầu hòa nhau, cùng chín lần. Chọn cặp nào trước là một lựa chọn thiết kế của thuật toán."):
            cap = swap_caption(self, cap, "Ba cặp đầu hòa nhau ở 9: tie-break là một lựa chọn thiết kế")
            self.wait(1.5)
        with noi(self, "Hai phép gộp tiếp theo tạo ra l o, rồi l o w."):
            for i in range(3, 5):
                self.play(FadeIn(col[i]), run_time=0.7)
            self.remove(*[m for m in self.mobjects if isinstance(m, VGroup) and m is not col and m is not cap and m.height > 2 and m.get_x() < 0 and not nen.la_luoi(m)])
            self.play(FadeIn(rows(forms[2])), run_time=0.8)
        with noi(self, "Gặp từ lạ lowest, ta áp các phép gộp theo đúng thứ tự đã học, được hai token: low và est."):
            cap = swap_caption(self, cap, "Từ lạ lowest: áp phép gộp theo thứ tự đã học → low + est</w> (2 token)")
        self.wait(3)


# ---------------------------------------------------------------- 9.1
class GiaiMaNhietDo(Scene):
    """Ví dụ 9.4: logit (3, 2, 1, 0, −1) ở T = 0.5, 1, 2; top-p = 0.9."""

    def construct(self):
        with noi(self, "Cùng một bộ logit, ba mức nhiệt độ. Nhiệt độ nhỏ làm phân phối nhọn hơn, gần như luôn chọn token đầu. Nhiệt độ lớn làm phân phối phẳng hơn, đa dạng hơn."):
            self.play(Write(title_of("Nhiệt độ và top-p")))
            z = np.array([3, 2, 1, 0, -1.0])
            groups = VGroup()
            for k, Tm in enumerate([0.5, 1.0, 2.0]):
                p = np.exp(z / Tm) / np.exp(z / Tm).sum()
                bars = VGroup()
                for i, v in enumerate(p):
                    b = Rectangle(width=0.5, height=max(3.2 * v, 0.03), fill_color=BLUE, fill_opacity=0.9, stroke_width=1)
                    b.move_to([i * 0.9, 0, 0], aligned_edge=DOWN)
                    lab = T(f"{v:.4f}", 15).next_to(b, UP, buff=0.08)
                    tk = T(f"x{'₁₂₃₄₅'[i]}", 18, GRAY_B).move_to([i * 0.9, -0.25, 0])
                    bars.add(b, lab, tk)
                head = T(f"T = {Tm}", 26, C_ACC).next_to(bars, DOWN, buff=0.25)
                g = VGroup(bars, head)
                g.shift(RIGHT * (k * 4.6 - 6.3) + DOWN * 0.3)
                groups.add(g)
            cap = swap_caption(self, None, "Cùng logit (3, 2, 1, 0, −1) của x₁ đến x₅: T nhỏ làm nhọn, T lớn làm phẳng")
            for g in groups:
                self.play(FadeIn(g), run_time=0.8)
            self.wait(2)
        with noi(self, "Top p giữ tập token nhỏ nhất có tổng xác suất ít nhất bằng p. Với nhiệt độ một và p bằng không phẩy chín, cần giữ ba token, rồi chuẩn hóa lại."):
            self.play(FadeOut(groups))
            lines = VGroup(
                T("T = 1:  p = (0.6364, 0.2341, 0.0861, 0.0317, 0.0117)", 24),
                T("tích lũy: 0.6364, 0.8705, 0.9567", 24, C_ACC),
                T("top-p = 0.9: token thứ ba làm tích lũy vượt 0.9 nên phải giữ", 24),
                T("chuẩn hóa lại: (0.6652, 0.2447, 0.0900, 0, 0)", 24, C1),
                T("cùng p = 0.9:  T = 0.5 giữ 2 token,  T = 2 giữ 4 token", 24, C2),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to(UP * 0.2)
            cap = swap_caption(self, cap, "Top-p giữ tập nhỏ nhất có tổng xác suất ít nhất bằng p")
            for ln in lines:
                self.play(FadeIn(ln), run_time=0.9)
            self.wait(1.5)
        with noi(self, "Top p tự thích nghi theo độ nhọn của phân phối: nhiệt độ không phẩy năm chỉ giữ hai token, nhiệt độ hai giữ bốn token."):
            cap = swap_caption(self, cap, "Top-p thích nghi theo độ nhọn của phân phối: T nhỏ giữ ít token, T lớn giữ nhiều")
        self.wait(2.5)


# ---------------------------------------------------------------- 9.2
class TokenMoiThamSo(Scene):
    """Ví dụ 9.5 và 9.6: D/N của các mô hình, rồi các điểm tối ưu của cách 3 (hai cách ghi số mũ) và của Epoch ở C = 5.88e23."""

    def construct(self):
        with noi(self, "Bao nhiêu token huấn luyện cho mỗi tham số? GPT ba dùng khoảng một phẩy bảy, Chinchilla khoảng hai mươi, còn Llama ba bốn trăm lẻ năm tỉ dùng tới ba mươi tám phẩy năm. Con số hai mươi chỉ là cách tóm tắt, không phải hằng số."):
            self.play(Write(title_of("Bao nhiêu token cho mỗi tham số?")))

            def chart(items, anchor, size=22):
                g = VGroup()
                for name, v, col in items:
                    ln = 1.1 * (np.log10(v) + 0.2)
                    bar = Rectangle(width=ln, height=0.5, fill_color=col, fill_opacity=0.9, stroke_width=0)
                    lab = T(name, size).move_to([0, 0, 0])
                    val = T(f"{v}", 22, col)
                    g.add(VGroup(lab, bar, val))
                rows = []
                for i, r in enumerate(g):
                    lab, bar, val = r
                    y = 1.2 - 0.95 * i
                    lab.move_to([-5.3, y, 0], aligned_edge=LEFT) if False else lab.move_to([-4.2, y, 0])
                    lab.shift(LEFT * (lab.width / 2 - 0.0)).move_to([-6.6 + lab.width / 2, y, 0])
                    bar.move_to([-3.2 + bar.width / 2, y, 0])
                    val.next_to(bar, RIGHT, buff=0.2)
                return g

            def log_axis(y=-2.45):
                xs = lambda v: -3.2 + 1.1 * (np.log10(v) + 0.2)
                ax = Line([xs(1), y, 0], [xs(100), y, 0], color=GRAY_B, stroke_width=2)
                ticks = VGroup(*[VGroup(Line([xs(v), y, 0], [xs(v), y - 0.12, 0], color=GRAY_B, stroke_width=2),
                                        T(str(v), 18, GRAY_B).move_to([xs(v), y - 0.35, 0])) for v in (1, 10, 100)])
                return VGroup(ax, ticks)

            a = chart([("GPT-3", 1.7, BLUE), ("Gopher", 1.1, BLUE), ("Chinchilla", 20.0, C1), ("Llama 3 405B", 38.5, C_ACC)], 0)
            x20 = 1.1 * (np.log10(20) + 0.2) - 3.2
            ref = DashedLine([x20, 2.1, 0], [x20, -2.0, 0], color=WHITE)
            rl = T("20", 20).next_to(ref, UP, buff=0.05)
            cap = swap_caption(self, None, "D/N = token huấn luyện chia số tham số (trục log: 1, 10, 100): quy tắc cỡ 20 là cách tóm tắt, không phải hằng số")
            ax1 = log_axis()
            self.play(FadeIn(a), FadeIn(ax1), Create(ref), FadeIn(rl), run_time=1.5)
            self.wait(3)
        with noi(self, "Ngay cả điểm tối ưu cũng tùy cách fit. Tại cùng ngân sách tính toán, cách ba của paper Chinchilla với số mũ đủ chữ số cho khoảng năm mươi tám token mỗi tham số, còn bản fit lại của Epoch cho khoảng mười tám."):
            self.play(FadeOut(a), FadeOut(ax1), FadeOut(ref), FadeOut(rl))
            b = chart([("Chinchilla thực tế", 20.0, C1), ("fit Epoch", 18.4, C_ACC),
                       ("cách 3: 0.339/0.285", 57.9, C2), ("cách 3: 0.34/0.28", 92.8, C2)], 0, 20)
            cap = swap_caption(self, cap, "Tại C = 5.88·10²³: cách 3 với số mũ đủ chữ số cho 41.1B / 2.38T (57.9); Epoch 73.0B / 1.34T (18.4)", 20)
            self.play(FadeIn(b), FadeIn(log_axis()), run_time=1.5)
            self.wait(3.5)
        with noi(self, "Nếu dùng số mũ đã làm tròn, kết quả nhảy lên chín mươi ba. Con số đó chỉ do làm tròn, nên cần đọc kỹ paper gốc trước khi trích dẫn."):
            cap = swap_caption(self, cap, "Số mũ làm tròn cho 32.5B (92.8): con số 93 chỉ do làm tròn. Chính paper nêu cách 3 ≈ 40B ở ngân sách Gopher (Figure 4)", 20)
            self.wait(4.5)


# ---------------------------------------------------------------- 9.3
class DPOHaiLogTiSo(Scene):
    """Ví dụ 9.7: loss DPO 0.5130 (b) và likelihood displacement: loss 0.4741 so với 0.6931 lúc khởi đầu (c)."""

    def construct(self):
        with noi(self, "Loss DPO chỉ nhìn hiệu của hai log tỉ số. Với mỗi đáp án, lấy log xác suất của mô hình mới trừ log xác suất của mô hình tham chiếu, rồi nhân beta. Đáp án thắng có phần thưởng ngầm không phẩy hai, đáp án thua trừ không phẩy hai, cho loss không phẩy năm một ba."):
            self.play(Write(title_of("Loss DPO chỉ nhìn hiệu của hai log-tỉ số")))
            lines = VGroup(
                T("β = 0.1", 26, C_ACC),
                T("đáp án thắng:  ln πθ = −12,  ln πref = −14  →  hiệu +2", 24, C1),
                T("đáp án thua:   ln πθ = −15,  ln πref = −13  →  hiệu −2", 24, C2),
                T("phần thưởng ngầm:  r̂w = 0.2,  r̂l = −0.2,  z = 0.4", 24),
                T("σ(0.4) = 0.5987     loss = −ln σ(0.4) = 0.5130", 24, C_ACC),
                T("trọng số gradient σ(−0.4) = 0.4013", 24),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(UP * 0.3)
            cap = swap_caption(self, None, "Mỗi đáp án: log xác suất mới trừ log xác suất tham chiếu, nhân β")
            for ln in lines:
                self.play(FadeIn(ln), run_time=0.9)
            self.wait(1.5)
        with noi(self, "Nhưng loss giảm không có nghĩa là xác suất đáp án tốt tăng. Nếu cả hai đáp án đều giảm xác suất, đáp án thua giảm nhiều hơn, loss vẫn giảm. Hiện tượng này gọi là likelihood displacement."):
            self.play(FadeOut(lines))
            cap = swap_caption(self, cap, "Nhưng loss giảm không có nghĩa xác suất đáp án tốt tăng")
            l2 = VGroup(
                T("đáp án thắng:  ln πθ từ −14 xuống −16  (giảm)", 24, C1),
                T("đáp án thua:   ln πθ từ −13 xuống −20  (giảm nhiều hơn)", 24, C2),
                T("z = 0.1 · [(−2) − (−7)] = 0.5", 24),
                T("loss = 0.4741, thấp hơn 0.6931 lúc khởi đầu (πθ = πref)", 26, C_ACC),
                T("likelihood displacement: xác suất đáp án được chọn đã giảm", 24, C2),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to(UP * 0.3)
            for ln in l2:
                self.play(FadeIn(ln), run_time=0.9)
            self.wait(3)


# ---------------------------------------------------------------- 9.3
class GRPONhom(Scene):
    """Ví dụ 9.8: advantage theo nhóm G = 8; nhóm toàn 0 không có gradient."""

    def construct(self):
        with noi(self, "GRPO tính advantage trong từng nhóm. Một nhóm tám đầu ra, mỗi đầu ra nhận phần thưởng không hoặc một. Trung bình là không phẩy ba bảy lăm."):
            self.play(Write(title_of("GRPO: advantage tính trong nhóm")))
            rw = [1, 0, 0, 1, 0, 0, 0, 1]
            adv = [1.2910 if r else -0.7746 for r in rw]
            base = DOWN * 0.2

            def row(vals, texts, y, mean_line):
                g = VGroup()
                for i, (v, t) in enumerate(zip(vals, texts)):
                    b = Rectangle(width=0.7, height=max(1.4 * v, 0.04), fill_color=C1 if v else GREY, fill_opacity=0.9, stroke_width=1)
                    b.move_to([i * 1.2 - 4.2, y, 0], aligned_edge=DOWN)
                    lab = T(t, 20, C1 if t.startswith("+") else (C2 if t.startswith("−") else WHITE)).next_to(b, UP, buff=0.1)
                    g.add(b, lab)
                if mean_line is not None:
                    ln = DashedLine([-5.0, y + 1.4 * mean_line, 0], [4.6, y + 1.4 * mean_line, 0], color=C_ACC)
                    g.add(ln, T("trung bình 0.375", 18, C_ACC).move_to([6.0, y + 1.4 * mean_line, 0]))
                return g

            r1 = row(rw, [""] * 8, 0.6, 0.375)
            cap = swap_caption(self, None, "Một nhóm G = 8 đầu ra với phần thưởng 0 hoặc 1: trung bình 0.375, độ lệch chuẩn 0.4841")
            self.play(FadeIn(r1), run_time=1)
            self.wait(1)
        with noi(self, "Advantage bằng phần thưởng trừ trung bình, chia độ lệch chuẩn. Đầu ra đúng được cộng một phẩy hai chín, đầu ra sai bị trừ không phẩy bảy bảy, tổng bằng không."):
            self.remove(r1)
            r1 = row(rw, [f"{a:+.3f}".replace("-", "−") for a in adv], 0.6, 0.375)
            self.add(r1)
            cap = swap_caption(self, cap, "Advantage = (r − trung bình)/độ lệch chuẩn: đúng +1.291, sai −0.775, tổng bằng 0")
            self.wait(3)
        with noi(self, "Nếu cả nhóm đều sai, độ lệch chuẩn bằng không, mọi advantage bằng không, và không có gradient. Nhóm đồng nhất không dạy được gì, nên DAPO bỏ những nhóm này."):
            r2 = row([0] * 8, ["0"] * 8, -1.7, None)
            t2 = T("nhóm toàn sai: độ lệch chuẩn 0, mọi advantage 0, không có gradient", 22, C2).move_to([0, -2.5, 0])
            self.play(FadeIn(r2), run_time=0.8)
            cap = swap_caption(self, cap, "Nhóm đồng nhất không dạy được gì; DAPO bỏ các nhóm này (Dynamic Sampling)")
            self.play(FadeIn(t2))
        self.wait(3)


# ---------------------------------------------------------------- 9.4
class LoRAHaiMaTran(Scene):
    """Ví dụ 9.9: 4096 × 4096 = 16 777 216; LoRA r = 8: 65 536 (0.39%)."""

    def construct(self):
        with noi(self, "Tinh chỉnh toàn phần phải học cả mười sáu triệu bảy trăm nghìn số của một ma trận bốn nghìn không trăm chín mươi sáu nhân bốn nghìn không trăm chín mươi sáu."):
            self.play(Write(title_of("LoRA: thay ΔW bằng tích BA")))
            W = Square(side_length=2.6, fill_color=GREY, fill_opacity=0.6, stroke_color=WHITE).move_to(LEFT * 4.6 + DOWN * 0.1)
            Wl = T("W₀ (đóng băng)", 22).next_to(W, UP, buff=0.15)
            Wn = T("4096 × 4096 = 16 777 216", 20).next_to(W, DOWN, buff=0.15)
            plus = T("+", 40).move_to(LEFT * 3.0 + DOWN * 0.1)
            B = Rectangle(width=0.4, height=2.6, fill_color=BLUE, fill_opacity=0.9).move_to(LEFT * 2.3 + DOWN * 0.1)
            A = Rectangle(width=2.6, height=0.4, fill_color=C_ACC, fill_opacity=0.9).move_to(RIGHT * 0.1 + DOWN * 0.1)
            dot = T("×", 36).move_to(LEFT * 1.4 + DOWN * 0.1)
            Bl = T("B: 4096 × 8", 20, BLUE).next_to(B, UP, buff=0.15)
            Al = T("A: 8 × 4096", 20, C_ACC).next_to(A, UP, buff=0.15)
            cnt = T("mỗi ma trận 32 768, tổng 65 536", 20).move_to(RIGHT * 0.8 + DOWN * 1.7)
            self.play(FadeIn(W), FadeIn(Wl), FadeIn(Wn))
            cap = swap_caption(self, None, "Tinh chỉnh toàn phần học cả 16 777 216 số của một ma trận 4096 × 4096")
            self.wait(1.5)
        with noi(self, "LoRA giữ W không đóng băng, và chỉ học hai ma trận mỏng B và A với hạng r bằng tám. Tổng cộng sáu mươi lăm nghìn năm trăm ba mươi sáu tham số."):
            self.play(FadeIn(plus), FadeIn(B), FadeIn(A), FadeIn(dot), FadeIn(Bl), FadeIn(Al))
            cap = swap_caption(self, cap, "LoRA chỉ học B và A với hạng r = 8; h = W₀x + (α/r)·BAx")
            self.play(FadeIn(cnt))
            self.wait(2)
        with noi(self, "Chỉ bằng không phẩy ba chín phần trăm. B khởi tạo bằng không nên lúc đầu mô hình không đổi. Sau khi huấn luyện, gộp B nhân A vào W không, nên không thêm độ trễ khi suy luận."):
            r1 = T("65 536 / 16 777 216 = 2r/d = 0.39%", 26, C_ACC).move_to(RIGHT * 4.4 + UP * 1.0)
            r2 = T("B = 0  ⇒  ΔW = 0 lúc đầu", 26, C1).move_to(RIGHT * 4.4 + DOWN * 0.1)
            r1.scale_to_fit_width(min(r1.width, 4.4)); r2.scale_to_fit_width(min(r2.width, 4.4))
            self.play(FadeIn(r1), FadeIn(r2))
            cap = swap_caption(self, cap, "B bằng 0 nên mô hình ban đầu không đổi; sau huấn luyện gộp BA vào W₀, không thêm độ trễ")
            self.wait(3)


# ---------------------------------------------------------------- 9.4
class NgoaiLeLuongTu(Scene):
    """Ví dụ 9.11: absmax int8 với và không có ngoại lệ 12.0; lượng tử theo khối."""

    def construct(self):
        with noi(self, "Lượng tử hóa absmax sang int tám: chia cho giá trị tuyệt đối lớn nhất, ở đây là không phẩy năm, rồi nhân một trăm hai mươi bảy và làm tròn. Sai số lớn nhất chỉ khoảng không phẩy không không hai."):
            self.play(Write(title_of("Một ngoại lệ làm hỏng cả vector")))
            xs = [0.1, -0.5, 0.3, -0.2, 0.05, 0.4, -0.3, 0.2]

            def row(vals, y, color, label):
                g = VGroup(T(label, 20, color).move_to([-5.9, y, 0]))
                g[0].move_to([-6.6 + g[0].width / 2, y, 0])
                for i, v in enumerate(vals):
                    g.add(T(v, 22, color).move_to([-2.6 + i * 1.15, y, 0]))
                return g

            fmt = lambda v: f"{v:g}".replace("-", "−")
            r0 = row([fmt(v) for v in xs], 1.7, WHITE, "gốc")
            r1 = row(["25", "−127", "76", "−51", "13", "102", "−76", "51"], 0.9, C1, "mã int8")
            e1 = T("sai số lớn nhất 0.0016", 22, C1).move_to([3.6, 0.2, 0])
            cap = swap_caption(self, None, "absmax: mã = làm tròn(127·x/0.5), bước lượng tử 0.5/127")
            self.play(FadeIn(r0), FadeIn(r1), FadeIn(e1))
            self.wait(2.5)
        with noi(self, "Giờ thay một giá trị bằng ngoại lệ mười hai. Bước lượng tử to gấp hai mươi tư lần, các giá trị nhỏ bị dồn về vài mã, sai số lên tới không phẩy không bốn."):
            self.play(FadeOut(r0), FadeOut(r1), FadeOut(e1))
            xs2 = list(xs); xs2[3] = 12.0
            r0 = row([fmt(v) for v in xs2], 1.7, WHITE, "gốc")
            r1 = row(["1", "−5", "3", "127", "1", "4", "−3", "2"], 0.9, C2, "mã int8")
            e1 = T("bước 12/127 = 0.0945; sai số tới 0.0445", 22, C2).move_to([2.9, 0.1, 0])
            cap = swap_caption(self, cap, "Thay −0.2 bằng 12.0: bước lượng tử to gấp 24 lần, các giá trị nhỏ dồn về 1, −5, 3")
            self.play(FadeIn(r0), FadeIn(r1), FadeIn(e1))
            self.wait(3)
        with noi(self, "Cách khắc phục: lượng tử theo khối bốn phần tử. Ngoại lệ chỉ ảnh hưởng khối chứa nó. Khối thứ hai lại chính xác như cũ."):
            sep = DashedLine([1.425, 2.2, 0], [1.425, 0.5, 0], color=C_ACC)
            cap = swap_caption(self, cap, "Lượng tử theo khối 4 phần tử: ngoại lệ chỉ ảnh hưởng khối chứa nó", 22)
            self.play(Create(sep))
            olds = [r1[5 + k] for k in range(4)]
            news = [T(c, 22, C_ACC).move_to(olds[k]) for k, c in enumerate(["16", "127", "−95", "64"])]
            self.play(*[Transform(o, n) for o, n in zip(olds, news)])
            e3 = T("khối 1 (có ngoại lệ) vẫn sai tới 0.0276", 22, C2).move_to([0, -0.9, 0])
            e2 = T("khối 2: mã 16, 127, −95, 64; sai số chỉ 0.0016", 22, C_ACC).move_to([0, -1.6, 0])
            self.play(FadeIn(e3), FadeIn(e2))
            self.wait(3.5)
