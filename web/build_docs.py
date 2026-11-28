"""Gom nội dung sách thành thư mục web/_docs/ để MkDocs dựng trang web tĩnh.

Không sửa gì trong book/: script chỉ đọc book/, plan/lo-trinh.png và manim/videos/,
rồi viết bản đã xử lý vào web/_docs/ (thư mục tạm, không đưa lên git).

Chạy:  python web/build_docs.py && mkdocs build   (hoặc mkdocs serve để xem thử)
"""
import re
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "_docs"

LIST_ITEM = re.compile(r"^(\s*)([-*+]|\d+[.)])\s")


def add_blank_before_lists(text: str) -> str:
    """Python-Markdown chỉ nhận danh sách khi có dòng trống phía trước.

    Sách viết kiểu MyST/GitHub (danh sách ngay sau đoạn văn), nên chèn dòng trống
    trước dòng đầu tiên của mỗi danh sách. Bỏ qua code block và khối công thức $$.
    """
    out, fence, math = [], False, False
    prev = ""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("```"):
            fence = not fence
        elif not fence and s == "$$":
            math = not math
        elif (not fence and not math and LIST_ITEM.match(line)
              and prev.strip() and not LIST_ITEM.match(prev)
              and not prev.startswith((" ", "\t", ">"))):
            out.append("")
        out.append(line)
        prev = line
    return "\n".join(out) + "\n"


def video_block(chap_key: str, scene: dict) -> str:
    src = f"videos/{chap_key}/{scene['cls']}.mp4"
    title = scene.get("title", scene["cls"])
    desc = scene.get("desc", "")
    return (
        f'??? video "Video: {title}"\n'
        f"    {desc}\n\n"
        f'    <video controls preload="metadata" width="100%" src="{src}"></video>\n'
    )


def insert_videos(text: str, chap_key: str, catalog: dict) -> str:
    """Chèn video Manim ngay sau tiêu đề mục tương ứng (ví dụ '## 1.3 ...')."""
    scenes = catalog.get(chap_key, {}).get("scenes", [])
    if not scenes:
        return text
    by_section: dict[str, list] = {}
    for sc in scenes:
        by_section.setdefault(str(sc["section"]), []).append(sc)

    out = []
    for line in text.splitlines():
        m = re.match(r"^##\s+(\d+\.\d+)\s", line)
        out.append(line)
        if m and m.group(1) in by_section:
            out.append("")
            for sc in by_section.pop(m.group(1)):
                out.append(video_block(chap_key, sc))
    # Cảnh không khớp tiêu đề nào: gom ở cuối chương
    leftover = [sc for scs in by_section.values() for sc in scs]
    if leftover:
        out += ["", "## Video minh họa khác", ""]
        out += [video_block(chap_key, sc) for sc in leftover]
    return "\n".join(out) + "\n"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    chapters = yaml.safe_load((ROOT / "book" / "chapters.yaml").read_text(encoding="utf-8"))
    catalog = yaml.safe_load((ROOT / "manim" / "catalog.yaml").read_text(encoding="utf-8"))

    for ch in chapters["chapters"]:
        text = (ROOT / "book" / ch["file"]).read_text(encoding="utf-8")
        text = text.replace("](../plan/lo-trinh.png)", "](assets/lo-trinh.png)")
        text = add_blank_before_lists(text)
        if ch.get("manim"):
            text = insert_videos(text, ch["manim"], catalog)
        name = "index.md" if ch["id"] == "huong-dan" else ch["file"]
        (OUT / name).write_text(text, encoding="utf-8")

    (OUT / "assets").mkdir()
    shutil.copy(ROOT / "plan" / "lo-trinh.png", OUT / "assets" / "lo-trinh.png")
    shutil.copytree(ROOT / "manim" / "videos", OUT / "videos")
    shutil.copytree(ROOT / "web" / "extra", OUT / "extra")
    print(f"Đã gom {len(chapters['chapters'])} trang vào {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
