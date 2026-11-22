"""Các cảnh Manim cho Chương 14 — Technical interview.

Render một cảnh (chất lượng 720p):
    manim -qm chuong14_scenes.py DongHo12Phut

Chỉ dùng Text (không cần LaTeX). Số liệu lấy từ các ví dụ giải mẫu và lần chạy lab của chương.
Khung 60 phút, 12 phút mỗi chuỗi, thang tự chấm là QUY TẮC CỦA SÁCH, không phải định dạng của chương trình tuyển nào.
Mọi dung lượng GPU, băng thông là số giả định của đề.
"""
import sys
from manim import *

FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"
C_OK, C_BAD, C_ACC, C_DIM = GREEN, RED, YELLOW, GREY_B


def T(s, size=28, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


def caption(s):
    t = T(s, 24)
    if t.width > 12.8:
        t.scale_to_fit_width(12.8)
    return VGroup(BackgroundRectangle(t, fill_opacity=0.85, buff=0.12), t).to_edge(DOWN, buff=0.25)


# ---------------------------------------------------------------- 14.1
class DongHo12Phut(Scene):
    """Một chuỗi 4 tầng trong 12 phút: mỗi tầng 3 phút = 2 phút trả lời (khoảng 280 từ ở 140 từ/phút) + 1 phút hỏi lại."""

    def construct(self):
        title = T("Một chuỗi 4 tầng trong 12 phút", 34).to_edge(UP, buff=0.3)
        self.play(Write(title))
        center = np.array([-3.0, -0.2, 0])
        R = 2.3
        colors = [BLUE, TEAL, GREEN, ORANGE]
        names = ["Tầng 1: định nghĩa", "Tầng 2: cơ chế, suy dẫn", "Tầng 3: con số, điều kiện", "Tầng 4: giới hạn, hướng mở"]
        rim = Circle(radius=R, color=WHITE, stroke_width=3).move_to(center)
        self.play(Create(rim))
        sectors = VGroup()
        for i in range(4):
            start = PI / 2 - i * PI / 2                      # chạy theo chiều kim đồng hồ từ 12 giờ
            ans = AnnularSector(inner_radius=0, outer_radius=R, angle=-PI / 3, start_angle=start,
                                color=colors[i], fill_opacity=0.75, stroke_width=0).move_arc_center_to(center)
            ask = AnnularSector(inner_radius=0, outer_radius=R, angle=-PI / 6, start_angle=start - PI / 3,
                                color=colors[i], fill_opacity=0.25, stroke_width=0).move_arc_center_to(center)
            sectors.add(VGroup(ans, ask))
        for i, s in enumerate(sectors):
            self.play(FadeIn(s), run_time=0.6)
        ticks = VGroup()
        for m in range(0, 12, 3):
            ang = PI / 2 - m * PI / 6
            p = center + 1.18 * R * np.array([np.cos(ang), np.sin(ang), 0])
            ticks.add(T(f"{m}'", 22).move_to(p))
        self.play(FadeIn(ticks))
        legend = VGroup()
        for i, n in enumerate(names):
            legend.add(VGroup(Square(side_length=0.28, color=colors[i], fill_opacity=0.8), T(n, 24)).arrange(RIGHT, buff=0.2))
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.4, 1.0, 0])
        self.play(FadeIn(legend, lag_ratio=0.2), run_time=1.2)
        key = VGroup(
            VGroup(Square(side_length=0.28, color=GREY_A, fill_opacity=0.8), T("đậm: 2 phút trả lời", 22)).arrange(RIGHT, buff=0.2),
            VGroup(Square(side_length=0.28, color=GREY_A, fill_opacity=0.25), T("nhạt: 1 phút hỏi lại, chuyển tầng", 22)).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to(legend, DOWN, buff=0.45, aligned_edge=LEFT)
        self.play(FadeIn(key))
        words = T("2 phút ≈ 280 từ ở 140 từ/phút", 26, C_ACC).next_to(key, DOWN, buff=0.45, aligned_edge=LEFT)
        self.play(FadeIn(words))
        cap = caption("Khung 12 phút và 4 tầng là quy tắc luyện của sách; người phỏng vấn thật có thể dừng ở bất kỳ tầng nào")
        self.play(FadeIn(cap))
        self.wait(2.5)


class LichLeitner(Scene):
    """Lab 2: 18 chuỗi, 28 ngày, hộp 1-3-7-14 ngày, quy tắc qua/trượt xác định. Tổng 69 lượt; hai ngày nặng nhất (12 và 17) có 6 chuỗi."""

    def construct(self):
        title = T("Lịch ôn Leitner cho 18 chuỗi trong 28 ngày", 32).to_edge(UP, buff=0.3)
        self.play(Write(title))
        load = [1, 2, 1, 1, 3, 2, 3, 2, 2, 3, 2, 6, 3, 3, 3, 2, 6, 3, 3, 2, 1, 3, 1, 2, 0, 2, 4, 3]
        base_y, k = -2.0, 0.55
        x0, step = -6.2, 0.45
        axis = Line([x0 - 0.3, base_y, 0], [x0 + 27 * step + 0.3, base_y, 0], color=WHITE)
        self.play(Create(axis))
        bars = VGroup()
        for i, v in enumerate(load):
            col = C_BAD if v == 6 else BLUE
            h = max(v * k, 0.02)
            r = Rectangle(width=0.34, height=h, color=col, fill_opacity=0.75, stroke_width=1)
            r.move_to([x0 + i * step, base_y + h / 2, 0])
            bars.add(r)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.06), run_time=2.5)
        days = VGroup(*[T(str(d), 18, C_DIM).move_to([x0 + (d - 1) * step, base_y - 0.3, 0]) for d in (1, 7, 14, 21, 28)])
        self.play(FadeIn(days))
        for d in (12, 17):
            lab = T("6", 24, C_BAD).next_to(bars[d - 1], UP, buff=0.1)
            dl = T(f"ngày {d}", 20, C_BAD).next_to(lab, UP, buff=0.08)
            self.play(FadeIn(lab), FadeIn(dl), run_time=0.5)
        note = VGroup(T("tổng 69 lượt; ngày nặng nhất 6 chuỗi = 72 phút nói", 24, C_ACC),
                      T("hộp 1, 2, 3, 4 ôn lại sau 1, 3, 7, 14 ngày; quên thì về hộp 1", 22)).arrange(DOWN, buff=0.15)
        note.move_to([0, -2.95, 0])
        self.play(FadeIn(note))
        self.wait(2.8)


# ---------------------------------------------------------------- 14.2
class BoDePhuongSai(Scene):
    """VD 14.4: Var(∑wᵢxᵢ) = n·Var(w)·𝔼[x²] dùng cho khởi tạo, √dₖ và dòng residual; số từ mô phỏng một seed (seed 0; 20 seed cho 1.35–3.21 ở lớp 10)."""

    def construct(self):
        title = T("Một bổ đề phương sai, ba chỗ dùng", 34).to_edge(UP, buff=0.3)
        self.play(Write(title))
        lemma = T("Var(Σ wᵢxᵢ) = n · Var(w) · E[x²]", 32, C_ACC).move_to([0, 2.3, 0])
        cond = T("w, x độc lập, w trung bình 0", 22, C_DIM).next_to(lemma, DOWN, buff=0.15)
        self.play(FadeIn(lemma), FadeIn(cond))

        def box(head, lines, x, color):
            r = Rectangle(width=4.1, height=2.9, color=color, fill_opacity=0.15, stroke_width=3).move_to([x, -0.45, 0])
            h = T(head, 26, color)
            body = VGroup(*[T(s, 21) for s in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
            g = VGroup(h, body).arrange(DOWN, buff=0.25)
            if g.width > 3.8:
                g.scale_to_fit_width(3.8)
            g.move_to(r)
            return VGroup(r, g)

        b1 = box("Khởi tạo, 10 lớp ReLU", ["n = 512, đầu vào phương sai 1", "He 2/n: E[z²] 1.995 → 1.808", "1/n: 1.001 → 0.0017",
                                         "(lý thuyết 2⁻⁹ ≈ 0.0020)"], -4.45, BLUE)
        b2 = box("Attention, q·k", ["dₖ = 64: phương sai 63.9", "std 7.99", "chia √dₖ: std 0.999", "softmax không bão hòa"], 0, GREEN)
        b3 = box("Dòng residual, 12 lớp", ["24 nhánh phương sai 1", "không co: 25.13 (lý thuyết 25)", "nhân 1/√(2L): 2.000",
                                        "GPT-2: 0.02/√24 = 0.00408"], 4.45, ORANGE)
        for b in (b1, b2, b3):
            arr = Arrow(cond.get_bottom() + DOWN * 0.12, b[0].get_top(), buff=0.1, color=C_DIM, stroke_width=3)
            self.play(GrowArrow(arr), FadeIn(b), run_time=0.9)
            self.wait(0.6)
        cap = caption("Một seed. Qua 20 seed, lớp 10 với He dao động 1.35–3.21 (cỡ ±25% quanh 2): chỉ kết luận \"giữ cỡ hằng số\"")
        self.play(FadeIn(cap))
        self.wait(2.6)


# ---------------------------------------------------------------- 14.3
class NganSachBoNho(Scene):
    """VD 14.6: GPU giả định 80 GiB (trọng số 12.55 + KV 2 GiB/chuỗi MHA hoặc 0.5 GiB với 8 đầu K/V) và laptop 6 GiB (3.33 + 2.00)."""

    def construct(self):
        title = T("Ngân sách bộ nhớ khi phục vụ mô hình 7B", 32).to_edge(UP, buff=0.3)
        self.play(Write(title))
        base_y, top_y = -2.2, 1.6
        k = (top_y - base_y) / 80                       # đơn vị cao mỗi GiB cho cột 80 GiB

        def column(x, total, parts, label, kscale, w=1.4):
            frame = Rectangle(width=w, height=total * kscale, color=WHITE, stroke_width=2).move_to([x, base_y + total * kscale / 2, 0])
            lab = T(label, 22).next_to(frame, DOWN, buff=0.15)
            blocks, y = VGroup(), base_y
            for gib, col in parts:
                h = gib * kscale
                blocks.add(Rectangle(width=w, height=h, color=col, fill_opacity=0.7, stroke_width=0.5).move_to([x, y + h / 2, 0]))
                y += h
            return frame, lab, blocks

        f1, l1, w1 = column(-5.3, 80, [(12.55, BLUE)] + [(2.0, GREEN)] * 33, "80 GiB, MHA", k)
        f2, l2, w2 = column(-2.1, 80, [(12.55, BLUE)] + [(0.5, TEAL)] * 134, "80 GiB, 8 đầu K/V", k)
        self.play(Create(f1), Create(f2), FadeIn(l1), FadeIn(l2))
        wt1 = T("12.55", 20).move_to(w1[0])
        wt2 = T("12.55", 20).move_to(w2[0])
        self.play(FadeIn(w1[0]), FadeIn(w2[0]), FadeIn(wt1), FadeIn(wt2))
        self.play(LaggedStart(*[FadeIn(b) for b in w1[1:]], lag_ratio=0.03), run_time=1.6)
        self.play(LaggedStart(*[FadeIn(b) for b in w2[1:]], lag_ratio=0.008), run_time=1.6)
        wnote = T("khối xanh dưới đáy: trọng số fp16 12.55 GiB", 18, BLUE).next_to(VGroup(l1, l2), DOWN, buff=0.1)
        self.play(FadeIn(wnote), run_time=0.5)
        n1 = VGroup(T("33 chuỗi", 22, GREEN), T("× 2 GiB", 22, GREEN)).arrange(DOWN, buff=0.08).next_to(f1, UP, buff=0.12)
        n2 = VGroup(T("134 chuỗi", 22, TEAL), T("× 0.5 GiB", 22, TEAL)).arrange(DOWN, buff=0.08).next_to(f2, UP, buff=0.12)
        self.play(FadeIn(n1), FadeIn(n2))
        # laptop 6 GiB, vẽ theo thang riêng (phóng to) cho dễ nhìn
        k2 = (top_y - base_y) / 6
        f3, l3, w3 = column(1.3, 6, [(3.33, BLUE), (2.00, GREEN)], "laptop 6 GiB", k2)
        sc = T("thang riêng", 18, C_DIM).next_to(f3, UP, buff=0.12)
        self.play(Create(f3), FadeIn(l3), FadeIn(w3), FadeIn(sc))
        t3 = VGroup(T("4-bit + scale: 3.33 GiB", 22, BLUE),
                    T("KV 4096 token: 2.00 GiB", 22, GREEN), T("tổng 5.33 GiB", 24, C_ACC),
                    T("chưa tính activation", 20, C_BAD), T("và overhead framework", 20, C_BAD)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        t3.next_to(f3, RIGHT, buff=0.4)
        if t3.get_right()[0] > 6.8:
            t3.shift(LEFT * (t3.get_right()[0] - 6.8))
        self.play(FadeIn(t3, lag_ratio=0.2), run_time=1.2)
        cap = caption("80 GiB và 6 GiB là giả định của đề; con số trên laptop thật phải đo")
        self.play(FadeIn(cap))
        self.wait(2.5)


class CayChuoiNoiChuong(Scene):
    """Capstone P1: từ một logit tới một token được phục vụ. Mỗi nút một chương và một con số đã kiểm."""

    def construct(self):
        title = T("Từ một logit tới một token được phục vụ", 32).to_edge(UP, buff=0.3)
        self.play(Write(title))
        nodes = [
            ("logit → p − y", "ch2", "p − y = (0.24, −0.33, 0.09)", BLUE),
            ("nhiệt độ khi giải mã", "ch9", "T = 0.5 / 1 / 2: H = 0.44 / 0.83 / 1.02 nats", TEAL),
            ("KV cache", "ch8", "0.5 MiB mỗi token; 2 GiB ở 4096", GREEN),
            ("speculative decoding", "ch9 + 14.4", "α = 0.8, k = 4: 3.36 token/lượt", ORANGE),
        ]
        ys = [1.75, 0.45, -0.85, -2.15]
        prev = None
        for (name, ch, num, col), y in zip(nodes, ys):
            r = RoundedRectangle(corner_radius=0.15, width=4.2, height=0.95, color=col, fill_opacity=0.2, stroke_width=3).move_to([-2.6, y, 0])
            nm = T(name, 24).move_to(r)
            chl = T(ch, 20, col).next_to(r, LEFT, buff=0.15)
            nb = T(num, 22, C_ACC).next_to(r, RIGHT, buff=0.35)
            grp = VGroup(r, nm, chl, nb)
            if prev is not None:
                arr = Arrow(prev.get_bottom(), r.get_top(), buff=0.05, color=C_DIM, stroke_width=3, max_tip_length_to_length_ratio=0.35)
                self.play(GrowArrow(arr), run_time=0.4)
            self.play(FadeIn(grp, shift=DOWN * 0.1), run_time=0.8)
            self.wait(0.5)
            prev = r
        cap = caption("Mỗi tầng chuỗi là một chương; mỗi nút phải có một con số đúng đơn vị")
        self.play(FadeIn(cap))
        self.wait(2.5)


# ---------------------------------------------------------------- 14.4
class TokenNhapHinhHoc(Scene):
    """VD 14.10: α = 0.8, k = 4; số token kỳ vọng mỗi lượt = 1 + 0.8 + 0.64 + 0.512 + 0.4096 = 3.3616 (giả định chấp nhận độc lập)."""

    def construct(self):
        title = T("Speculative decoding: token kỳ vọng mỗi lượt", 32).to_edge(UP, buff=0.3)
        self.play(Write(title))
        vals = [1.0, 0.8, 0.64, 0.512, 0.4096]
        labs = ["α⁰ = 1", "α¹ = 0.8", "α² = 0.64", "α³ = 0.512", "α⁴ = 0.4096"]
        base_y, k = -1.7, 2.6
        x0, step = -5.4, 1.55
        axis = Line([x0 - 0.8, base_y, 0], [x0 + 4 * step + 0.8, base_y, 0], color=WHITE)
        self.play(Create(axis))
        bars = VGroup()
        for i, (v, s) in enumerate(zip(vals, labs)):
            col = C_ACC if i == 0 else BLUE
            r = Rectangle(width=1.0, height=v * k, color=col, fill_opacity=0.7, stroke_width=2).move_to([x0 + i * step, base_y + v * k / 2, 0])
            a_, v_ = s.split(" = ")
            t = VGroup(T(a_, 22), T(v_, 20)).arrange(DOWN, buff=0.08).next_to(r, DOWN, buff=0.15)
            bars.add(VGroup(r, t))
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in bars], lag_ratio=0.25), run_time=2.2)
        bonus = T("token của mô hình đích: luôn có", 20, C_ACC).next_to(bars[0][0], UP, buff=0.15).shift(RIGHT * 0.9)
        self.play(FadeIn(bonus))
        side = VGroup(T("cộng dồn:", 24), T("1 + 0.8 + 0.64 + 0.512 + 0.4096", 20),
                      T("= 3.3616 token mỗi lượt", 28, C_ACC), T("mô phỏng: 3.3511", 22, C_DIM),
                      T("công thức (1 − αᵏ⁺¹)/(1 − α)", 22)).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.75, 0.6, 0])
        self.play(FadeIn(side, lag_ratio=0.3), run_time=1.8)
        warn = VGroup(T("sai hay gặp: nói kα = 3.2,", 22, C_BAD), T("quên số hạng α⁰", 22, C_BAD)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        warn.next_to(side, DOWN, buff=0.45, aligned_edge=LEFT)
        self.play(FadeIn(warn))
        cap = caption("Giả định: mỗi token nháp được chấp nhận độc lập với xác suất α; thực tế α đổi theo ngữ cảnh")
        self.play(FadeIn(cap))
        self.wait(2.5)


# ---------------------------------------------------------------- 14.5
class NganSachSTAR(Scene):
    """VD 14.2 và 14.11: câu chuyện 2 phút theo gợi ý MIT CAPD 20/10/60/10 (24, 12, 72, 12 giây) so với câu chuyện mẫu
    206 từ = 88 giây ở 140 từ/phút (S 30, T 23, A 116, R 37 từ). Thanh câu chuyện mẫu dài theo số giây thật (88/120)."""

    def construct(self):
        title = T("Ngân sách giây cho một câu chuyện STAR", 32).to_edge(UP, buff=0.3)
        self.play(Write(title))
        names = ["S", "T", "A", "R"]
        colors = [BLUE, TEAL, GREEN, ORANGE]
        full_w = 11.0                                   # thanh 120 giây
        x_left = -full_w / 2

        def bar(seconds, y, label):
            g, x = VGroup(), x_left
            tot = sum(seconds)
            for n, sec, c in zip(names, seconds, colors):
                w = sec * full_w / 120
                r = Rectangle(width=w, height=0.9, color=c, fill_opacity=0.55, stroke_width=2).move_to([x + w / 2, y, 0])
                txt = T(f"{n} {round(100 * sec / tot)}%", 22).move_to(r)
                if txt.width > w - 0.1:
                    txt = T(n, 22).move_to(r)
                g.add(VGroup(r, txt))
                x += w
            lab = T(label, 22, C_DIM).next_to(g, UP, buff=0.15).align_to(g, LEFT)
            return VGroup(lab, g)

        top = bar([24, 12, 72, 12], 1.2, "gợi ý MIT CAPD, 2 phút: S 24 s, T 12 s, A 72 s, R 12 s")
        self.play(FadeIn(top, lag_ratio=0.2), run_time=1.5)
        self.wait(0.8)
        words = [30, 23, 116, 37]                        # đếm từ, không tính nhãn "For illustration only."
        secs = [w / 140 * 60 for w in words]
        bot = bar(secs, -0.9, "câu chuyện mẫu: 206 từ = 88 s ở 140 từ/phút (nhãn của sách không đếm)")
        self.play(FadeIn(bot, lag_ratio=0.2), run_time=1.5)
        end_x = x_left + sum(secs) * full_w / 120
        slack = DashedLine([end_x, -0.9, 0], [x_left + full_w, -0.9, 0], color=C_DIM)
        slack_t = T("còn 32 s", 20, C_DIM).next_to(slack, DOWN, buff=0.12)
        self.play(Create(slack), FadeIn(slack_t))
        notes = VGroup(T("Action dài nhất: việc của TÔI, có số (200 câu, kappa 0.78, 0.05 so với 0.11)", 22),
                       T("vai trò khớp hồ sơ chương 13: \"my graph step\"; đếm: I 12 lần, we 5 lần", 22)
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        notes.move_to([0, -2.45, 0])
        self.play(FadeIn(notes, lag_ratio=0.3))
        cap = caption("Tỉ lệ là gợi ý của MIT CAPD; 6 câu chuyện và 2 phút mỗi câu là quy tắc của sách")
        self.play(FadeIn(cap))
        self.wait(2.5)
