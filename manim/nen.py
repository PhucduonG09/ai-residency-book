"""Nền lưới mờ cho mọi cảnh Manim (thay cho nền đen trơn).

Trong file cảnh, sau `from manim import *`:

    import nen
    nen.bat()          # từ đây mọi Scene trong file tự có lưới mờ phía sau

Lưới được thêm trong Scene.setup() nên nằm dưới mọi hình vẽ sau đó.
Muốn bỏ lưới khi xóa cảnh: lọc theo nen.la_luoi(m).
"""
from manim import BLUE_E, GREY_D, NumberPlane, Scene

DO_MO = 0.32      # độ đậm đường kẻ chính
DO_MO_PHU = 0.12  # độ đậm đường kẻ phụ (chia đôi ô)


def luoi() -> NumberPlane:
    g = NumberPlane(
        x_range=[-8, 8, 1], y_range=[-4.5, 4.5, 1],
        background_line_style={"stroke_color": BLUE_E, "stroke_width": 1.2, "stroke_opacity": DO_MO},
        faded_line_style={"stroke_color": GREY_D, "stroke_width": 0.8, "stroke_opacity": DO_MO_PHU},
        faded_line_ratio=2,
        axis_config={"stroke_opacity": 0},   # không vẽ trục giữa: chỉ là nền
    )
    g._la_luoi_nen = True
    return g


def la_luoi(m) -> bool:
    return getattr(m, "_la_luoi_nen", False)


def bat() -> None:
    goc = Scene.setup

    def setup(self):
        goc(self)
        self.add(luoi())

    if not getattr(Scene.setup, "_co_luoi", False):
        setup._co_luoi = True
        Scene.setup = setup
