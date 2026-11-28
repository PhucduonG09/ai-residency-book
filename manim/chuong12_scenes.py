"""Các cảnh Manim cho Chương 12 — Chuyên sâu NLP và đạo đức AI.

Render một cảnh (720p):  manim -qm chuong12_scenes.py BM25BaoHoa
Chỉ dùng Text (không cần LaTeX). Mỗi cảnh dùng đúng số liệu của ví dụ giải mẫu trong sách (dữ liệu tự dựng, đồ chơi).
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
C_OK, C_BAD, C_ACC = GREEN, RED, YELLOW
C_A, C_B = BLUE_C, ORANGE


def T(s, size=28, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


def caption(s, size=24):
    t = T(s, size)
    if t.width > 12.6:
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


# ---------------------------------------------------------------- 12.1
class BM25BaoHoa(Scene):
    """Ví dụ 12.1: phần tf bão hòa về k₁ + 1 = 2.2; bốn đoạn, truy vấn 'graph retrieval'."""

    def construct(self):
        title = T("BM25: tf bão hòa, và idf phân biệt từ hiếm", 32).to_edge(UP, buff=0.3)
        with noi(self, "BM25 cho điểm theo số lần một từ xuất hiện, nhưng phần tf bão hòa dần về k một cộng một, tức hai phẩy hai."):
            self.play(Write(title))
            x0, y0, w, h = -6.3, -1.6, 4.6, 3.2           # gốc và kích thước vùng vẽ
            ymax = 2.4
            ax = VGroup(Line([x0, y0, 0], [x0 + w, y0, 0], color=GREY_B), Line([x0, y0, 0], [x0, y0 + h, 0], color=GREY_B))
            fs = [1, 2, 5, 10, 50]
            vals = [1.000, 1.375, 1.774, 1.964, 2.148]
            xs = [x0 + 0.5 + i * 0.95 for i in range(5)]
            ys = [y0 + v / ymax * h for v in vals]
            xl = VGroup(*[T(str(f), 20).move_to([xs[i], y0 - 0.3, 0]) for i, f in enumerate(fs)])
            xt = VGroup(T("f = số lần từ xuất hiện", 20, GREY_B),
                        T("(các mốc cách đều, trục f không theo tỉ lệ)", 17, GREY_B)).arrange(DOWN, buff=0.08).move_to([x0 + w / 2, y0 - 0.85, 0])
            yt = T("phần tf", 20, GREY_B).next_to(ax[1], UP, buff=0.1)
            lim_y = y0 + 2.2 / ymax * h
            lim = DashedLine([x0, lim_y, 0], [x0 + w, lim_y, 0], color=C_BAD)
            lim_t = T("k₁ + 1 = 2.2", 22, C_BAD).move_to([x0 + 2.9, lim_y + 0.3, 0])
            self.play(Create(ax), FadeIn(xl), FadeIn(xt), FadeIn(yt))
            self.play(Create(lim), FadeIn(lim_t))
            dots = VGroup(*[Dot([xs[i], ys[i], 0], color=C_ACC) for i in range(5)])
            lab = VGroup(*[T(f"{vals[i]:.3f}", 18, C_ACC).move_to([xs[i], ys[i] - 0.32, 0]) for i in range(5)])
            path = VMobject(color=C_ACC).set_points_as_corners([d.get_center() for d in dots])
            self.play(Create(path), FadeIn(dots), FadeIn(lab), run_time=2)
            cap = swap_caption(self, None, "k₁ = 1.2: lặp từ 50 lần chỉ được 2.148, không phải 50 lần điểm")
        with noi(self, "Với k một bằng một phẩy hai, lặp một từ năm mươi lần chỉ được hai phẩy một bốn tám, không phải năm mươi lần điểm."):
            self.wait(2)

        with noi(self, "Truy vấn graph retrieval. Từ graph hiếm hơn nên idf cao hơn: không phẩy sáu chín, so với không phẩy ba năm của retrieval."):
            cap = swap_caption(self, cap, "Truy vấn \"graph retrieval\": idf(graph) = 0.6931, idf(retrieval) = 0.3567")
            rows = [("d₄  graph×2, retrieval×1", "1.2683", C_OK), ("d₁  graph×2, không có retrieval", "0.9624", C_OK),
                    ("d₂  retrieval×2", "0.5154", WHITE), ("d₃  retrieval×2", "0.4766", WHITE)]
            tab = VGroup()
            head = T("điểm BM25", 24, GREY_B).move_to([3.2, 2.4, 0])
            for i, (a, b, c) in enumerate(rows):
                tab.add(VGroup(T(a, 22, c), T(b, 26, c)).arrange(RIGHT, buff=0.5))
            tab.arrange(DOWN, buff=0.35, aligned_edge=RIGHT).move_to([2.7, 0.7, 0])
            self.play(FadeIn(head), FadeIn(tab))
            self.wait(1.5)
        with noi(self, "Đoạn d một không chứa retrieval vẫn thắng d hai và d ba, vì idf của graph gấp gần hai lần. Từ hiếm nặng hơn từ phổ biến."):
            note = T("d₁ không có \"retrieval\" vẫn thắng d₂ và d₃", 26, C_ACC).move_to([2.7, -1.6, 0])
            note2 = VGroup(T("vì idf(graph) gấp 1.94 lần idf(retrieval);", 21, GREY_B),
                           T("cả ba đoạn cùng f = 2 nên tf không phân biệt", 21, GREY_B)).arrange(DOWN, buff=0.1).next_to(note, DOWN, buff=0.2)
            cap = swap_caption(self, cap, "Chênh lệch này do idf: từ hiếm nặng hơn từ phổ biến")
            self.play(FadeIn(note), FadeIn(note2))
            self.wait(2.5)


class RRFHopNhat(Scene):
    """Ví dụ 12.3: gộp hai danh sách theo hạng, k = 60."""

    def construct(self):
        title = T("RRF: gộp theo hạng, điểm = ∑ 1 / (k + hạng)", 32).to_edge(UP, buff=0.3)
        with noi(self, "Reciprocal rank fusion gộp hai danh sách theo hạng. Mỗi tài liệu được cộng một chia cho k cộng hạng, ở mỗi danh sách chứa nó."):
            self.play(Write(title))
            bm = ["d3", "d1", "d7", "d2", "d9"]
            de = ["d1", "d5", "d3", "d8", "d2"]
            xb, xd = -4.6, -1.0
            yy = [1.9 - 0.7 * i for i in range(5)]
            hb = T("BM25", 24, C_A).move_to([xb, 2.6, 0]); hd = T("dense", 24, C_B).move_to([xd, 2.6, 0])
            bb, dd = {}, {}
            grp = VGroup(hb, hd)
            for i in range(5):
                bb[bm[i]] = T(f"{i + 1}. {bm[i]}", 26, C_A).move_to([xb, yy[i], 0])
                dd[de[i]] = T(f"{i + 1}. {de[i]}", 26, C_B).move_to([xd, yy[i], 0])
                grp.add(bb[bm[i]], dd[de[i]])
            self.play(FadeIn(grp))
        with noi(self, "Điểm gốc của BM25 và của dense retrieval không cùng thang đo, nên ta chỉ dùng hạng."):
            cap = swap_caption(self, None, "Hai danh sách xếp hạng; điểm gốc của hai bộ không cùng thang")
            links = {}
            for name, col in (("d1", C_OK), ("d3", C_ACC), ("d2", WHITE)):
                links[name] = Line(bb[name].get_right() + RIGHT * 0.15, dd[name].get_left() + LEFT * 0.15, color=col, stroke_width=3)
            self.play(*[Create(l) for l in links.values()])
        with noi(self, "D một đứng hạng hai và hạng một, d ba đứng hạng một và hạng ba. Tài liệu nằm trong cả hai danh sách sẽ lên đầu."):
            res = VGroup(
                T("d1: 1/62 + 1/61 = 0.03252", 26, C_OK),
                T("d3: 1/61 + 1/63 = 0.03227", 26, C_ACC),
                T("d2: 1/64 + 1/65 = 0.03101", 26, WHITE),
                T("d5 0.01613 > d7 0.01587", 20, GREY_B),
                T("d8 0.01562 > d9 0.01538", 20, GREY_B),
            ).arrange(DOWN, buff=0.25, aligned_edge=LEFT).move_to([3.9, 1.0, 0])
            cap = swap_caption(self, cap, "d1 hạng 2 và hạng 1; d3 hạng 1 và hạng 3: cùng nằm trong cả hai danh sách")
            self.play(FadeIn(res))
            self.wait(2)
        with noi(self, "Khi k nhỏ, hạng một áp đảo. Khi k lớn, điểm phẳng hơn. Bài báo gốc dùng k bằng sáu mươi."):
            kk = VGroup(T("hạng đầu áp đảo khi k nhỏ", 22, GREY_B),
                        T("k = 60:  d1 0.03252   d3 0.03227", 24),
                        T("k = 10:  d1 0.17424   d3 0.16783", 24),
                        T("k = 1:   d1 0.8333    d3 0.7500", 24)).arrange(DOWN, buff=0.2, aligned_edge=LEFT).move_to([3.4, -1.75, 0])
            cap = swap_caption(self, cap, "k nhỏ thì hạng 1 áp đảo, k lớn thì phẳng hơn; bài gốc dùng k = 60")
            self.play(FadeIn(kk))
            self.wait(3)


# ---------------------------------------------------------------- 12.2
class ChunkBienGioi(Scene):
    """Ví dụ 12.4: sự thật hai câu (câu 4 và 5) bị cắt ngang bởi biên chunk."""

    def construct(self):
        title = T("Chunking: sự thật hai câu bị cắt ngang biên chunk", 32).to_edge(UP, buff=0.3)
        with noi(self, "Một sự thật cần hai câu: câu bốn nêu thực thể và thuộc tính, câu năm chứa đáp án."):
            self.play(Write(title))
            cells = VGroup()
            for i in range(12):
                col = C_ACC if i in (3, 4) else GREY_D
                sq = Square(0.8, color=GREY_B, fill_color=col, fill_opacity=0.9).move_to([-4.4 + 0.8 * i, 1.6, 0])
                cells.add(VGroup(sq, T(str(i + 1), 22, BLACK if i in (3, 4) else WHITE).move_to(sq)))
            lab = T("câu 4: tên thực thể và thuộc tính;  câu 5: đáp án", 22, C_ACC).move_to([0, 2.5, 0])
            self.play(FadeIn(cells), FadeIn(lab))
        with noi(self, "Chia chunk bốn câu không chồng lấn, biên cắt ngang giữa câu bốn và câu năm. Tên nằm ở chunk một, đáp án ở chunk hai."):
            cap = swap_caption(self, None, "Chunk 4 câu, không chồng lấn: biên nằm giữa câu 4 và câu 5")
            b1 = [Rectangle(width=3.2, height=1.0, color=C_A).move_to([-4.4 + 0.8 * (4 * j + 1.5), 1.6, 0]) for j in range(3)]
            self.play(*[Create(b) for b in b1])
            cut = Line([-4.4 + 0.8 * 3.5, 0.95, 0], [-4.4 + 0.8 * 3.5, 2.2, 0], color=C_BAD, stroke_width=6)
            cutl = T("cắt ngang: tên ở chunk 1, đáp án ở chunk 2", 22, C_BAD).move_to([0, 0.5, 0])
            self.play(Create(cut), FadeIn(cutl))
            self.wait(2)
        with noi(self, "Nếu chồng lấn một câu, với bước ba câu, chunk từ câu bốn đến câu bảy chứa trọn cả hai câu."):
            self.play(*[FadeOut(b) for b in b1], FadeOut(cut), FadeOut(cutl))
            b2 = [Rectangle(width=3.2, height=1.0, color=C_OK).move_to([-4.4 + 0.8 * (3 * j + 1.5), 1.6 + 0.0, 0]) for j in range(3)]
            self.play(Create(b2[1]))
            cap = swap_caption(self, cap, "Chồng lấn 1 câu (bước 3): chunk 4–7 chứa cả hai câu")
            self.wait(1.5)
        with noi(self, "Chunk càng dài, xác suất cắt ngang càng nhỏ. Với chồng lấn, top ba luôn có đáp án. Cái giá là số chunk gấp một phẩy ba mốt lần, và các chunk kề nhau có thể hòa điểm."):
            tab = VGroup(T("c", 22, GREY_B), T("xác suất cắt", 22, GREY_B)).arrange(RIGHT, buff=1.2)
            rows = VGroup(*[VGroup(T(a, 24), T(b, 24)).arrange(RIGHT, buff=1.6) for a, b in
                            (("2", "0.492"), ("4", "0.238"), ("8", "0.111"), ("16", "0.048"))])
            left = VGroup(tab, rows.arrange(DOWN, buff=0.2)).arrange(DOWN, buff=0.3).move_to([-3.2, -1.4, 0])
            right = VGroup(T("chồng lấn 1 câu, c = 4:", 24, C_OK), T("ans@top-3 = 1.000", 26, C_OK),
                           T("ans@top-1 = 0.833 (không chồng lấn 0.743)", 20, C_ACC),
                           T("hai chunk kề nhau hòa điểm: top-1 phụ thuộc", 19, GREY_B),
                           T("quy tắc phá hòa; phá theo chỉ số nhỏ cho 0.677", 19, GREY_B),
                           T("cái giá: số chunk gấp 1.31 lần, token trùng", 19, GREY_B)).arrange(DOWN, buff=0.16, aligned_edge=LEFT).move_to([3.2, -1.3, 0])
            cap = swap_caption(self, cap, "Chồng lấn: top-3 luôn đủ đáp án, không làm top-1 tệ hơn; cái giá là chi phí và hòa điểm")
            self.play(FadeIn(left), FadeIn(right))
            self.wait(3)


class PhanRaLoiRAG(Scene):
    """Ví dụ 12.6: Acc = R·a₁ + (1 − R)·a₀ = 0.655, và hai kịch bản cải thiện."""

    def construct(self):
        title = T("Phân rã lỗi RAG: 200 câu hỏi", 32).to_edge(UP, buff=0.3)
        with noi(self, "Phân rã lỗi của một hệ RAG trên hai trăm câu hỏi."):
            self.play(Write(title))
            scale = 0.036                                    # chiều rộng mỗi câu hỏi
            x_left = -6.0
            w1, w0 = 140 * scale, 60 * scale
            r1 = Rectangle(width=w1, height=0.9, color=C_A, fill_color=C_A, fill_opacity=0.35).move_to([x_left + w1 / 2, 1.6, 0])
            r0 = Rectangle(width=w0, height=0.9, color=GREY_B, fill_color=GREY_D, fill_opacity=0.6).move_to([x_left + w1 + w0 / 2, 1.6, 0])
            t1 = T("có đoạn vàng: 140 (R = 0.70)", 22).move_to(r1)
            t0 = T("không có: 60", 20).move_to(r0)
            self.play(FadeIn(r1), FadeIn(r0), FadeIn(t1), FadeIn(t0))
        with noi(self, "Đoạn vàng nằm trong ngữ cảnh ở một trăm bốn mươi câu. Khi có nó, mô hình trả lời đúng một trăm mười chín câu. Khi không có, vẫn đúng mười hai câu."):
            cap = swap_caption(self, None, "Đoạn vàng nằm trong ngữ cảnh ở 140 trên 200 câu")
            g1 = Rectangle(width=119 * scale, height=0.5, color=C_OK, fill_color=C_OK, fill_opacity=0.8).move_to([x_left + 119 * scale / 2, 0.55, 0])
            g0 = Rectangle(width=12 * scale, height=0.5, color=C_OK, fill_color=C_OK, fill_opacity=0.8).move_to([x_left + w1 + 12 * scale / 2, 0.55, 0])
            a1 = T("đúng 119 (a₁ = 0.85)", 20, C_OK).next_to(g1, DOWN, buff=0.15)
            a0 = T("đúng 12 (a₀ = 0.20)", 20, C_OK).next_to(g0, DOWN, buff=0.15).shift(RIGHT * 0.9)
            cap = swap_caption(self, cap, "Khi có đoạn vàng đúng 119; khi không có vẫn đúng 12 (không có đoạn vàng làm cơ sở)")
            self.play(FadeIn(g1), FadeIn(g0), FadeIn(a1), FadeIn(a0))
            self.wait(1.5)
        with noi(self, "Độ chính xác cuối bằng không phẩy bảy nhân không phẩy tám lăm, cộng không phẩy ba nhân không phẩy hai, bằng không phẩy sáu năm năm."):
            f = T("Acc = 0.70 · 0.85 + 0.30 · 0.20 = 0.655", 32, C_ACC).move_to([0, -1.4, 0])
            cap = swap_caption(self, cap, "Độ chính xác cuối = 131/200 = 0.655")
            self.play(Write(f))
            self.wait(1.5)
        with noi(self, "Tăng recall lên không phẩy tám lăm, hay tăng độ chính xác khi có đoạn vàng lên không phẩy chín hai, đều có giá trị. Chọn theo chi phí và độ khó."):
            s1 = T("R lên 0.85  ⇒  0.7525", 26).move_to([-3.0, -2.4, 0])
            s2 = T("a₁ lên 0.92  ⇒  0.704", 26).move_to([3.0, -2.4, 0])
            cap = swap_caption(self, cap, "Mỗi điểm R và mỗi điểm a₁ có giá gần bằng nhau; chọn theo chi phí và độ khó")
            self.play(FadeIn(s1), FadeIn(s2))
            self.wait(3)


# ---------------------------------------------------------------- 12.3
class PPRLanToa(Scene):
    """Ví dụ 12.7: PPR trên đồ thị 7 nút, hạt giống X, α = 0.5."""

    def construct(self):
        title = T("Personalized PageRank: khối lượng lan theo cạnh", 32).to_edge(UP, buff=0.3)
        with noi(self, "Personalized PageRank trên đồ thị bảy nút: các thực thể và các đoạn văn."):
            self.play(Write(title))
            pos = {"X": [-2.0, 0.6, 0], "Y": [1.0, 0.6, 0], "Z": [-4.0, 0.6, 0], "D": [3.0, 0.6, 0],
                   "P1": [-0.5, 1.9, 0], "P2": [2.0, 1.9, 0], "P3": [-3.0, 1.9, 0]}
            pos = {k: [v[0] + 0.0, v[1] - 0.2, 0] for k, v in pos.items()}
            edges = [("X", "P1"), ("Y", "P1"), ("Y", "P2"), ("D", "P2"), ("X", "P3"), ("Z", "P3")]
            lines = VGroup(*[Line(pos[a], pos[b], color=GREY_B) for a, b in edges])
            nodes = {}
            for k, p in pos.items():
                shape = Circle(0.36, color=C_A if k[0] != "P" else C_B, fill_color=GREY_D, fill_opacity=1)
                nodes[k] = VGroup(shape, T(k, 20).move_to(shape)).move_to(p)
            self.play(Create(lines), *[FadeIn(n) for n in nodes.values()])
        vtxt = {}

        def show(vals, shade=True):
            anims = []
            for k, v in vals.items():
                sh = nodes[k][0]
                anims.append(sh.animate.set_fill(interpolate_color(GREY_D, C_ACC, min(1.0, 0.1 + v * 1.6)), opacity=1))
                new = T(f"{v:.4g}" if v not in (0,) else "0", 18, C_ACC).next_to(nodes[k], UP if k[0] == "P" else DOWN, buff=0.12)
                if k in vtxt:
                    anims.append(Transform(vtxt[k], new))
                else:
                    vtxt[k] = new
                    anims.append(FadeIn(new))
            self.play(*anims, run_time=1.0)

        base = dict.fromkeys(pos, 0.0)
        with noi(self, "Toàn bộ khối lượng bắt đầu ở hạt giống X, với xác suất khởi động lại alpha bằng không phẩy năm."):
            cap = swap_caption(self, None, "Hạt giống X: toàn bộ khối lượng bắt đầu ở X, α = 0.5 (xác suất khởi động lại)")
            show({**base, "X": 1.0})
            self.wait(1)
        with noi(self, "Lặp một: X giữ một nửa, chia đều phần còn lại cho P một và P ba."):
            cap = swap_caption(self, cap, "Lặp 1: X giữ 0.5, chia 0.25 cho P₁ và 0.25 cho P₃")
            show({**base, "X": 0.5, "P1": 0.25, "P3": 0.25})
            self.wait(1)
        with noi(self, "Lặp hai: khối lượng chạm tới Y và Z, nhưng P hai và D vẫn bằng không."):
            cap = swap_caption(self, cap, "Lặp 2: khối lượng chạm Y và Z; P₂ và D vẫn bằng 0")
            show({**base, "X": 0.625, "Y": 0.0625, "Z": 0.0625, "P1": 0.125, "P3": 0.125})
            self.wait(1)
        with noi(self, "Khi hội tụ, P ba lớn hơn P một, lớn hơn P hai. Đoạn đích hai bước xếp cuối, nhưng vẫn lớn hơn không."):
            cap = swap_caption(self, cap, "Hội tụ: trong đồ thị này P₃ > P₁ > P₂ (thường vậy, không phải luật chung)")
            show({"X": 0.5803, "Y": 0.0419, "Z": 0.0415, "D": 0.0030, "P1": 0.1556, "P2": 0.0120, "P3": 0.1658})
            box = T("P₂ (đoạn đích hai bước) xếp cuối nhưng 0.0120 > 0", 26, C_ACC).move_to([0, -2.5, 0])
            self.play(FadeIn(box))
            self.wait(3)


# ---------------------------------------------------------------- 12.5
class TamGiacBatKhaThi(Scene):
    """Ví dụ 12.10: cùng PPV 0.8 và FNR 0.2 nhưng tỉ lệ nền khác nhau thì FPR phải khác."""

    def construct(self):
        title = T("Công bằng: không thể có tất cả khi tỉ lệ nền khác nhau", 32).to_edge(UP, buff=0.3)
        with noi(self, "Hai nhóm, mỗi nhóm một nghìn người, tỉ lệ nền là không phẩy năm và không phẩy hai. Ta ép cả hai có cùng precision và cùng tỉ lệ bỏ sót."):
            self.play(Write(title))

            def table(name, tp, fp, fn, tn, x, col):
                head = T(name, 26, col).move_to([x, 2.5, 0])
                cs = VGroup()
                for i, (lab, v) in enumerate((("TP", tp), ("FP", fp), ("FN", fn), ("TN", tn))):
                    sq = Rectangle(width=2.0, height=0.9, color=GREY_B, fill_color=GREY_D, fill_opacity=0.9)
                    sq.move_to([x - 1.05 + 2.1 * (i % 2), 1.5 - 1.0 * (i // 2), 0])
                    cs.add(VGroup(sq, T(f"{lab} {v}", 24).move_to(sq)))
                return VGroup(head, cs)

            ta = table("Nhóm A: tỉ lệ nền 0.5", 400, 100, 100, 400, -3.4, C_A)
            tb = table("Nhóm B: tỉ lệ nền 0.2", 160, 40, 40, 760, 3.4, C_B)
            self.play(FadeIn(ta), FadeIn(tb))
            cap = swap_caption(self, None, "Mỗi nhóm 1 000 người; ép cả hai có cùng PPV = 0.8 và cùng FNR = 0.2")
            self.wait(2)
        with noi(self, "Precision đều bằng không phẩy tám, tỉ lệ bỏ sót đều bằng không phẩy hai."):
            eq = T("PPV = 400 / 500 = 160 / 200 = 0.8      FNR = 100 / 500 = 40 / 200 = 0.2", 24).move_to([0, -0.7, 0])
            self.play(FadeIn(eq))
            self.wait(1.5)
        with noi(self, "Khi đó tỉ lệ dương tính giả bắt buộc phải khác: không phẩy hai ở nhóm A, và không phẩy không năm ở nhóm B."):
            fa = T("FPR nhóm A = 100 / 500 = 0.200", 28, C_ACC).move_to([-3.4, -1.7, 0])
            fb = T("FPR nhóm B = 40 / 800 = 0.050", 28, C_ACC).move_to([3.4, -1.7, 0])
            cap = swap_caption(self, cap, "Cùng PPV và cùng tỉ lệ bỏ sót thì FPR bắt buộc khác nhau")
            self.play(FadeIn(fa), FadeIn(fb))
            self.wait(1.5)
        with noi(self, "Không thể thỏa mọi tiêu chí công bằng cùng lúc khi tỉ lệ nền khác nhau. Chọn chỉ số nào là một quyết định về giá trị, phải nêu lý do."):
            sel = T("tỉ lệ được chọn: 0.500 so với 0.200 (chênh 0.30)", 24, GREY_B).move_to([0, -2.5, 0])
            cap = swap_caption(self, cap, "Chọn chỉ số công bằng là một quyết định có giá trị, phải nêu lý do")
            self.play(FadeIn(sel))
            self.wait(3)
