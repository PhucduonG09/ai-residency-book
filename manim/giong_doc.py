"""Giọng đọc Google (gTTS) cho cảnh Manim — không cần API key.

Cách dùng trong một cảnh:

    from giong_doc import noi

    with noi(self, "Gradient luôn vuông góc với đường đồng mức."):
        self.play(GrowArrow(mui_ten))     # hình chạy trong lúc giọng đọc
    # ra khỏi khối: nếu câu đọc dài hơn hình thì cảnh tự chờ đọc xong

- Lần đầu mỗi câu được gửi lên Google Translate TTS (cần Internet), file mp3
  lưu ở manim/giong_doc_cache/ theo mã băm của câu. Lần sau đọc lại từ cache,
  nên render offline được nếu cache đã có (cache được commit vào repo).
- Sửa câu thoại thì mã băm đổi và câu được tải lại.
- Đặt biến môi trường MANIM_KHONG_GIONG=1 để render câm như cũ (bỏ qua giọng đọc).

Cần: pip install gTTS, và ffmpeg trong PATH để tăng tốc giọng (environment.yml đã có).
(Manim Community ≥ 0.18 đã kèm PyAV để đo độ dài mp3.)
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

CACHE = Path(__file__).resolve().parent / "giong_doc_cache"
LANG = "vi"
NGHI = 0.25  # khoảng lặng (giây) sau mỗi câu trước khi sang câu kế
TOC_DO = 1.25  # gTTS đọc khá chậm: tăng tốc 1.25× (giữ cao độ) bằng bộ lọc atempo của ffmpeg


def file_giong(text: str, lang: str = LANG, toc_do: float = TOC_DO) -> Path:
    """Trả về mp3 của câu (đã chỉnh tốc độ), tải từ gTTS nếu chưa có trong cache."""
    key = hashlib.sha1(f"{lang}|{toc_do}|{text}".encode("utf-8")).hexdigest()[:16]
    f = CACHE / f"{key}.mp3"
    if not f.exists():
        from gtts import gTTS  # chỉ cần khi phải tải câu mới

        CACHE.mkdir(exist_ok=True)
        goc = f.with_suffix(".goc.mp3")
        gTTS(text, lang=lang).save(str(goc))
        if toc_do != 1:
            tmp = f.with_suffix(".tmp.mp3")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(goc), "-filter:a", f"atempo={toc_do}",
                            str(tmp)], check=True)
            goc.unlink()
        else:
            tmp = goc
        tmp.replace(f)  # ghi xong mới đổi tên: lỗi mạng giữa chừng không để lại file hỏng
    return f


def do_dai(f: Path) -> float:
    """Độ dài mp3 (giây), đọc bằng PyAV — thư viện Manim đã dùng."""
    import av

    with av.open(str(f)) as c:
        return c.duration / 1_000_000  # av.time_base = 1e6


class _Giong:
    def __init__(self, scene, t0: float, duration: float):
        self.scene, self.t0, self.duration = scene, t0, duration

    def con_lai(self) -> float:
        """Số giây giọng còn đọc tính từ thời điểm hiện tại của cảnh."""
        return max(0.0, self.t0 + self.duration - self.scene.renderer.time)


@contextmanager
def noi(scene, text: str, nghi: float = NGHI):
    """Phát câu `text` từ thời điểm hiện tại; khi thoát khối thì chờ đọc xong."""
    if os.environ.get("MANIM_KHONG_GIONG"):
        yield _Giong(scene, scene.renderer.time, 0.0)
        return
    f = file_giong(text)
    g = _Giong(scene, scene.renderer.time, do_dai(f))
    scene.add_sound(str(f))
    yield g
    if g.con_lai() > 0:
        scene.wait(g.con_lai() + nghi)
