"""Giọng đọc tiếng Việt cho cảnh Manim — không cần API key.

Mặc định giọng nữ Google (gTTS); MANIM_GIONG=edge để dùng giọng nam NamMinh (Microsoft Edge TTS).

Cách dùng trong một cảnh:

    from giong_doc import noi

    with noi(self, "Gradient luôn vuông góc với đường đồng mức."):
        self.play(GrowArrow(mui_ten))     # hình chạy trong lúc giọng đọc
    # ra khỏi khối: nếu câu đọc dài hơn hình thì cảnh tự chờ đọc xong

- Lần đầu mỗi câu được gửi lên dịch vụ TTS (cần Internet), file mp3
  lưu ở manim/giong_doc_cache/ theo mã băm của câu. Lần sau đọc lại từ cache,
  nên render offline được nếu cache đã có (cache được commit vào repo).
- Sửa câu thoại thì mã băm đổi và câu được tải lại.
- Đặt biến môi trường MANIM_KHONG_GIONG=1 để render câm như cũ (bỏ qua giọng đọc).

Cần: pip install edge-tts (giọng nam) hoặc gTTS + ffmpeg (giọng nữ).
(Manim Community ≥ 0.18 đã kèm PyAV để đo độ dài mp3.)
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

CACHE = Path(__file__).resolve().parent / "giong_doc_cache"
NGHI = 0.25  # khoảng lặng (giây) sau mỗi câu trước khi sang câu kế

# Giọng mặc định: giọng nữ tiếng Việt của Google Translate (gTTS), nhanh 1.25×.
# MANIM_GIONG=edge để dùng giọng nam NamMinh của Microsoft Edge TTS (miễn phí, không cần key).
GIONG = os.environ.get("MANIM_GIONG", "gtts")
EDGE_VOICE = "vi-VN-NamMinhNeural"
EDGE_RATE = "+10%"   # nhanh hơn tốc độ chuẩn một chút
GTTS_TOC_DO = 1.25   # gTTS đọc chậm: tăng tốc 1.25× (giữ cao độ) bằng atempo của ffmpeg


def _edge(text: str, out: Path) -> None:
    import asyncio

    import edge_tts

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")  # chỉ cần trên máy sau proxy
    asyncio.run(edge_tts.Communicate(text, EDGE_VOICE, rate=EDGE_RATE, proxy=proxy).save(str(out)))


def _gtts(text: str, out: Path) -> None:
    from gtts import gTTS

    goc = out.with_suffix(".goc.mp3")
    gTTS(text, lang="vi").save(str(goc))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(goc), "-filter:a", f"atempo={GTTS_TOC_DO}",
                    str(out)], check=True)
    goc.unlink()


def file_giong(text: str) -> Path:
    """Trả về mp3 của câu, tạo mới (cần Internet) nếu chưa có trong cache."""
    cfg = f"edge|{EDGE_VOICE}|{EDGE_RATE}" if GIONG == "edge" else f"gtts|vi|{GTTS_TOC_DO}"
    key = hashlib.sha1(f"{cfg}|{text}".encode("utf-8")).hexdigest()[:16]
    f = CACHE / f"{key}.mp3"
    if not f.exists():
        CACHE.mkdir(exist_ok=True)
        tmp = f.with_suffix(".tmp.mp3")
        for lan in range(4):  # dịch vụ thỉnh thoảng lỗi tạm thời: thử lại vài lần
            try:
                (_edge if GIONG == "edge" else _gtts)(text, tmp)
                break
            except Exception:
                if lan == 3:
                    raise
                time.sleep(2 * (lan + 1))
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
