"""Các cảnh Manim cho Chương 13 — Hồ sơ và video presentation.

Render một cảnh (chất lượng 720p):
    manim -qm chuong13_scenes.py BaPhutPresentation

Chỉ dùng Text (không cần LaTeX). Số liệu lấy từ các ví dụ giải mẫu và từ lần chạy lab của chương;
mọi mẫu CV, statement, kịch bản là MẪU MINH HỌA với số giả.
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
C_OK, C_BAD, C_ACC, C_DIM = GREEN, RED, YELLOW, GREY_B


def T(s, size=28, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


def caption(s):
    t = T(s, 24)
    if t.width > 12.8:
        t.scale_to_fit_width(12.8)
    return VGroup(BackgroundRectangle(t, fill_opacity=0.85, buff=0.12), t).to_edge(DOWN, buff=0.25)


# ---------------------------------------------------------------- 13.1
class PheuTuyenRRP(Scene):
    """Năm vòng công khai của Qualcomm RRP Batch 17 (mức B, kiểm 03/11/2026); vòng video không có định dạng công khai."""

    def construct(self):
        title = T("Qualcomm RRP, Batch 17: năm vòng công khai", 32).to_edge(UP, buff=0.3)
        with noi(self, "Chương trình Qualcomm RRP, batch mười bảy, có năm vòng công khai: hồ sơ, bài kiểm tra đầu vào, phỏng vấn lập trình, video presentation và phỏng vấn kỹ thuật."):
            self.play(Write(title))
            names = ["1. Application", "2. Entrance test", "3. Coding interview", "4. Video presentation", "5. Technical interview"]
            widths = [7.0, 6.4, 5.8, 5.2, 4.6]
            cx = -2.0
            rows = VGroup()
            for i, (n, w) in enumerate(zip(names, widths)):
                r = Rectangle(width=w, height=0.78, color=C_DIM, fill_opacity=0.25, stroke_width=3)
                lab = T(n, 26).move_to(r)
                rows.add(VGroup(r, lab))
            rows.arrange(DOWN, buff=0.18).move_to([cx, 0.1, 0])
            for g in rows:
                self.play(FadeIn(g, shift=DOWN * 0.15), run_time=0.45)
        # nhấn mạnh vòng 4
        with noi(self, "Vòng video là vòng thứ tư. Định dạng, hạn nộp và tiêu chí chấm của nó không được công khai."):
            r4 = rows[3][0]
            self.play(r4.animate.set_color(C_ACC).set_fill(C_ACC, opacity=0.3), run_time=0.8)
            q = VGroup(T("định dạng, hạn nộp, tiêu chí chấm:", 24, C_ACC), T("không công khai (mức D)", 26, C_ACC)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            q.next_to(r4, RIGHT, buff=0.4)
            self.play(FadeIn(q))
            note = T("Video là vòng 4, sau entrance test và coding interview", 22).next_to(rows, DOWN, buff=0.3)
            self.play(FadeIn(note))
        with noi(self, "Thông tin này lấy từ bản sao tin tuyển, chưa đối chiếu với cổng gốc, kiểm ngày ba tháng mười một năm hai nghìn không trăm hai mươi sáu."):
            cap = caption("Mức B: bản sao tin tuyển, chưa đối chiếu cổng gốc. Kiểm ngày 03/11/2026")
            self.play(FadeIn(cap))
            self.wait(2.8)


# ---------------------------------------------------------------- 13.2
class BulletCV(Scene):
    """Một bullet CV đi qua ba bước: động từ hành động, bối cảnh, kết quả có số. Cờ linter 2 -> 1 -> 1 -> 0."""

    def construct(self):
        title = T("Sửa một bullet CV theo công thức", 34).to_edge(UP, buff=0.3)
        with noi(self, "Sửa một bullet trong CV theo công thức. Bản gốc mở bằng responsible for, động từ yếu, và không có con số nào."):
            self.play(Write(title))
            steps = [
                ("Bước 0 · bullet gốc",
                 "Responsible for working on a graph-based question\nanswering project for a course.",
                 "cờ: weak-verb, no-number", 2),
                ("Bước 1 · động từ hành động",
                 "Built a graph-based question answering system\nfor a course project.",
                 "cờ: no-number", 1),
                ("Bước 2 · bối cảnh cụ thể",
                 "Built hybrid dense + citation-graph retrieval\nfor arXiv abstracts.",
                 "cờ: no-number", 1),
                ("Bước 3 · kết quả có số",
                 "Built hybrid dense + citation-graph retrieval over 100 000\narXiv abstracts; raised recall@5 from 0.61 to 0.72 (3 seeds)",
                 "không còn cờ", 0),
            ]
            label = T(steps[0][0], 28, C_ACC).move_to([0, 1.7, 0])
            bullet = Text(steps[0][1], font=FONT, font_size=26, color=WHITE, line_spacing=1.5).move_to([0, 0.35, 0])
            flags = T(steps[0][2], 26, C_BAD).move_to([-2.2, -1.0, 0])
            count = T("2 cờ", 30, C_BAD).move_to([3.6, -1.0, 0])
            self.play(FadeIn(label), FadeIn(bullet), FadeIn(flags), FadeIn(count))
            self.wait(1.2)
        with noi(self, "Bước một, dùng động từ hành động. Bước hai, nêu bối cảnh cụ thể. Bước ba, thêm kết quả có số: recall tại năm tăng từ không phẩy sáu mốt lên không phẩy bảy hai, qua ba seed."):
            for (lab, txt, fl, n) in steps[1:]:
                color = C_OK if n == 0 else C_BAD
                n_lab = T(lab, 28, C_ACC).move_to(label)
                n_bul = Text(txt, font=FONT, font_size=26, color=WHITE, line_spacing=1.5).move_to(bullet)
                n_fl = T(fl, 26, color).move_to([-2.2, -1.0, 0])
                n_cnt = T(f"{n} cờ" if n else "0 cờ", 30, color).move_to(count)
                self.play(Transform(label, n_lab), Transform(bullet, n_bul), Transform(flags, n_fl), Transform(count, n_cnt), run_time=1.1)
                self.wait(1.4)
        with noi(self, "Linter chỉ kiểm có chữ số hay không. Người đọc mới kiểm con số đó có ý nghĩa hay không."):
            tick = T("✔", 44, C_OK).next_to(bullet, RIGHT, buff=0.3).set_x(6.2)
            tick.move_to([6.3, 0.35, 0])
            self.play(FadeIn(tick))
            cap = caption("Linter chỉ kiểm có chữ số hay không; người đọc mới kiểm con số có ý nghĩa không")
            self.play(FadeIn(cap))
            self.wait(2.2)


# ---------------------------------------------------------------- 13.3
class MuiTenStatement(Scene):
    """Statement mẫu 306 từ, sáu đoạn (34, 75, 48, 34, 59, 56). Đoạn 5 lạc đề; bỏ đi còn 247 từ."""

    def construct(self):
        title = T("Statement như một đề xuất nhỏ", 34).to_edge(UP, buff=0.3)
        with noi(self, "Viết statement như một đề xuất nhỏ: mở bằng một vấn đề cụ thể, kể những gì đã làm với vai trò rõ ràng, rồi nêu bước tiếp theo."):
            self.play(Write(title))

            def box(lines, x, y, color, w=3.6, h=1.5):
                r = Rectangle(width=w, height=h, color=color, fill_opacity=0.2, stroke_width=3).move_to([x, y, 0])
                t = VGroup(*[T(s, 22, WHITE if i else color) for i, s in enumerate(lines)]).arrange(DOWN, buff=0.14).move_to(r)
                if t.width > w - 0.3:
                    t.scale_to_fit_width(w - 0.3)
                return VGroup(r, t)

            b1 = box(["Mở: vấn đề cụ thể", "đoạn 1 · 34 từ"], -4.7, 1.0, BLUE)
            b2 = box(["Đã làm + vai trò rõ", "đoạn 2, 3, 4 · 157 từ"], 0, 1.0, GREEN)
            b3 = box(["Bước tiếp theo", "đoạn 6 · 56 từ"], 4.7, 1.0, ORANGE)
            a1 = Arrow([-2.8, 1.0, 0], [-1.95, 1.0, 0], buff=0, color=WHITE)
            a2 = Arrow([1.95, 1.0, 0], [2.8, 1.0, 0], buff=0, color=WHITE)
            self.play(FadeIn(b1))
            self.play(GrowArrow(a1), FadeIn(b2))
            self.play(GrowArrow(a2), FadeIn(b3))
            self.wait(0.8)
        with noi(self, "Đoạn năm kể lại môn NLP nhưng không phục vụ đề xuất, nên bỏ đi. Bài còn hai trăm bốn mươi bảy từ, dưới mức sàn hai trăm năm mươi, nên hãy bù bằng bằng chứng, không bằng giai thoại."):
            off = box(["Đoạn 5 · 59 từ", "kể lại môn NLP, không phục vụ đề xuất"], 0, -1.35, GREY, w=6.6, h=1.2)
            self.play(FadeIn(off))
            tag = T("bỏ đoạn này", 24, C_BAD).next_to(off, RIGHT, buff=0.3)
            self.play(off[0].animate.set_color(C_BAD), FadeIn(tag))
            self.play(off[1].animate.set_opacity(0.55))
            cap = caption("Bỏ đoạn 5: 306 → 247 từ, dưới sàn 250 của sách; bù bằng bằng chứng, không bằng giai thoại")
            self.play(FadeIn(cap))
            self.wait(3.0)


# ---------------------------------------------------------------- 13.4
class LichNguocThu(Scene):
    """Lịch ngược từ hạn GIẢ ĐỊNH 12/11/2027: 15/07 hỏi, 17/09 gửi gói, 22/10 và 05/11 nhắc."""

    def construct(self):
        title = T("Lịch ngược cho thư giới thiệu", 34).to_edge(UP, buff=0.3)
        sub = T("hạn 12/11/2027 là giả định, suy từ Batch 17, chưa xác minh", 24, C_ACC).next_to(title, DOWN, buff=0.2)
        with noi(self, "Lịch ngược cho thư giới thiệu, tính từ hạn nộp giả định mười hai tháng mười một năm hai nghìn không trăm hai mươi bảy."):
            self.play(Write(title), FadeIn(sub))
            x0, x1 = -5.8, 5.8
            axis = Line([x0 - 0.3, -0.1, 0], [x1 + 0.3, -0.1, 0], color=WHITE, stroke_width=4)
            self.play(Create(axis))

        def xpos(days_before):
            return x0 + (120 - days_before) * (x1 - x0) / 120

        pts = [(120, "15/07", "hỏi người viết", +1, 0.95), (56, "17/09", "gửi gói tài liệu", -1, 0.95),
               (21, "22/10", "nhắc lần 1", +1, 0.95), (7, "05/11", "nhắc lần cuối", -1, 0.95), (0, "12/11", "hạn nộp", +1, 1.95)]
        with noi(self, "Hỏi người viết từ mười lăm tháng bảy. Gửi gói tài liệu ngày mười bảy tháng chín. Nhắc lần một ngày hai mươi hai tháng mười, và nhắc lần cuối ngày năm tháng mười một."):
            for d, date_s, what, side, h in pts:
                x = xpos(d)
                col = C_ACC if d == 0 else GREEN
                dot = Dot([x, -0.1, 0], color=col, radius=0.11)
                y = -0.1 + side * h
                conn = Line([x, -0.1, 0], [x, y - side * 0.48, 0], color=C_DIM, stroke_width=2)
                lab = VGroup(T(date_s, 24, col), T(what, 20, WHITE)).arrange(DOWN, buff=0.06).move_to([x, y, 0])
                if d == 120:
                    lab.shift(RIGHT * 0.6)
                if d == 0:
                    lab.shift(LEFT * 0.5)
                self.play(FadeIn(dot), Create(conn), FadeIn(lab), run_time=0.7)
        with noi(self, "Từ lúc hỏi đến hạn là một trăm hai mươi ngày, khoảng mười bảy tuần. Người viết có tám tuần sau khi nhận gói. Các mốc này là quy tắc riêng của sách, không phải của chương trình."):
            brace = T("120 ngày ≈ 17 tuần từ lúc hỏi đến hạn", 24, WHITE).move_to([-1.0, -2.2, 0])
            gap = T("56 ngày trước hạn: người viết có 8 tuần", 22, WHITE).move_to([2.0, -2.8, 0])
            self.play(FadeIn(brace))
            self.play(FadeIn(gap))
            cap = caption("Các mốc 120 và 56 ngày là quy tắc riêng của sách, không phải của chương trình")
            self.play(FadeIn(cap))
            self.wait(2.5)


# ---------------------------------------------------------------- 13.5
class BaPhutPresentation(Scene):
    """Kịch bản 3 phút: sáu đoạn 20, 25, 45, 45, 25, 20 giây; 47, 58, 105, 105, 58, 47 từ ở 140 từ/phút; tổng 420 từ."""

    def construct(self):
        title = T("Ngân sách cho video 3 phút", 34).to_edge(UP, buff=0.3)
        with noi(self, "Chia ngân sách cho video ba phút thành sáu đoạn: hook, vấn đề, cách làm, bằng chứng, vai trò và kết. Hai đoạn giữa dài nhất, mỗi đoạn bốn mươi lăm giây."):
            self.play(Write(title))
            names = ["Hook", "Vấn đề", "Cách làm", "Bằng chứng", "Vai trò", "Kết"]
            secs = [20, 25, 45, 45, 25, 20]
            wds = [47, 58, 105, 105, 58, 47]
            colors = [BLUE, TEAL, GREEN, GREEN, ORANGE, PURPLE]
            total_w = 12.0
            scale = total_w / sum(secs)
            x = -total_w / 2
            blocks = VGroup()
            for n, s, w, c in zip(names, secs, wds, colors):
                width = s * scale
                r = Rectangle(width=width, height=1.5, color=c, fill_opacity=0.35, stroke_width=3)
                r.move_to([x + width / 2, 0.3, 0])
                inner = VGroup(T(f"{s} s", 24, WHITE), T(f"{w} từ", 22, WHITE)).arrange(DOWN, buff=0.1).move_to(r)
                nm = T(n, 22, c).next_to(r, UP, buff=0.18)
                blocks.add(VGroup(r, inner, nm))
                x += width
            for g in blocks:
                self.play(FadeIn(g, shift=UP * 0.15), run_time=0.55)
        # thang thời gian
        with noi(self, "Tổng một trăm tám mươi giây, tương đương bốn trăm hai mươi từ ở tốc độ một trăm bốn mươi từ mỗi phút. Nếu nói một trăm năm mươi từ mỗi phút, bạn sẽ xong sớm mười hai giây."):
            base_y = -0.8
            ticks = VGroup()
            for t in (0, 60, 120, 180):
                xx = -total_w / 2 + t * scale
                ticks.add(Line([xx, base_y, 0], [xx, base_y - 0.18, 0], color=WHITE))
                ticks.add(T(f"{t} s", 20).move_to([xx, base_y - 0.48, 0]))
            axis = Line([-total_w / 2, base_y, 0], [total_w / 2, base_y, 0], color=WHITE, stroke_width=3)
            self.play(Create(axis), FadeIn(ticks))
            tot = T("Tổng: 180 giây = 420 từ ở 140 từ/phút", 30, C_ACC).move_to([0, -2.1, 0])
            self.play(FadeIn(tot))
            alt = T("ở 150 từ/phút, 420 từ chỉ nói hết trong 168 giây", 24, WHITE).move_to([0, -2.7, 0])
            self.play(FadeIn(alt))
            cap = caption("140 từ/phút và sáu đoạn là quy tắc của sách; định dạng thật theo cổng nộp")
            self.play(FadeIn(cap))
            self.wait(2.2)


class NhipNoiWPM(Scene):
    """Bản ghi giả lập (seed 0): 440 từ, 203.0 giây, trung bình 130.0 từ/phút; sáu cửa sổ 30 giây và một cửa sổ cuối 23 giây."""

    def construct(self):
        title = T("Trung bình 130 từ/phút che mất điều gì?", 34).to_edge(UP, buff=0.3)
        with noi(self, "Một bản ghi giả lập: bốn trăm bốn mươi từ trong hai trăm lẻ ba giây, trung bình một trăm ba mươi từ mỗi phút. Hãy xem theo từng cửa sổ ba mươi giây."):
            self.play(Write(title))
            vals = [172, 140, 136, 118, 122, 106, 112]
            wins = ["0-30", "30-60", "60-90", "90-120", "120-150", "150-180", "180-203"]
            base_y = -2.0
            k = 0.021                      # đơn vị mỗi từ/phút
            x_left, step = -5.6, 1.45

            def y_of(v):
                return base_y + v * k

            band = Rectangle(width=10.4, height=(150 - 130) * k, color=GREEN, fill_opacity=0.25, stroke_width=0)
            band.move_to([x_left + 4.35, (y_of(130) + y_of(150)) / 2, 0])
            axis = Line([x_left - 0.7, base_y, 0], [x_left + 9.2, base_y, 0], color=WHITE)
            band_lab = T("vùng 130-150", 20, C_OK).next_to(band, RIGHT, buff=0.15).align_to(band, UP)
            self.play(Create(axis), FadeIn(band), FadeIn(band_lab))
            bars = VGroup()
            for i, v in enumerate(vals):
                c = C_BAD if v > 150 else (C_OK if v >= 130 else ORANGE)
                r = Rectangle(width=0.95, height=v * k, color=c, fill_opacity=0.7, stroke_width=2)
                r.move_to([x_left + i * step, base_y + v * k / 2, 0])
                vt = T(str(v), 22).move_to(r.get_top() + (UP * 0.26 if v >= 130 else DOWN * 0.28))
                wt = T(wins[i], 17, C_DIM).next_to(r, DOWN, buff=0.1)
                bars.add(VGroup(r, vt, wt))
            for g in bars:
                self.play(FadeIn(g, shift=UP * 0.1), run_time=0.45)
        with noi(self, "Trung bình nằm giữa hai lỗi: mở đầu quá nhanh, một trăm bảy mươi hai từ mỗi phút, và đoạn cuối chậm hẳn. Lưu ý cửa sổ cuối chỉ dài hai mươi ba giây."):
            mean = DashedLine([x_left - 0.7, y_of(130), 0], [x_left + 9.2, y_of(130), 0], color=C_ACC, stroke_width=4)
            mean_lab = T("trung bình 130.0", 20, C_ACC).next_to(mean.get_end(), RIGHT, buff=0.1).shift(DOWN * 0.1)
            self.play(Create(mean), FadeIn(mean_lab))
            note = T("cửa sổ cuối chỉ dài 23 s: chia cho 23, không phải 30", 20, C_DIM).move_to([x_left + 6 * step - 0.3, base_y - 0.85, 0])
            self.play(FadeIn(note))
            cap = caption("Trung bình nằm giữa hai lỗi: mở đầu quá nhanh (172) và cuối chậm hẳn (106-112)")
            self.play(FadeIn(cap))
            self.wait(2.6)
