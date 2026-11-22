"""Sổ tay luyện thi AI Residency — app học tập chạy local.

Chạy:  streamlit run app/app.py      (hoặc nhấp đúp run_app.bat trên Windows)
Toàn bộ nội dung là file trong thư mục dự án; tiến độ lưu ở progress/progress.json.
"""
from __future__ import annotations

import ast
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
BOOK, PLAN, LABS, MANIM, CARDS = (ROOT / p for p in ("book", "plan", "labs", "manim", "flashcards"))
PROGRESS_FILE = ROOT / "progress" / "progress.json"

STREAMS = {
    "chuong": "Chương học",
    "A": "A · Dự án nghiên cứu",
    "B": "B · Coding",
    "C": "C · Tiếng Anh và trình bày",
    "chung": "Chung",
}
LEITNER_DAYS = {1: 0, 2: 1, 3: 3, 4: 7, 5: 14}   # hộp → số ngày tới lần ôn sau

st.set_page_config(page_title="Sổ tay Residency", page_icon="📘", layout="wide")


# ----------------------------------------------------------------- dữ liệu
@st.cache_data(show_spinner=False)
def _yaml_cached(path: str, mtime: float):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_yaml(path: Path):
    """Đọc YAML, cache theo thời điểm sửa file (sửa file là tự nạp lại)."""
    return _yaml_cached(str(path), path.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _read_text_cached(path: str, mtime: float) -> str:
    return Path(path).read_text(encoding="utf-8")


def read_text(path: Path) -> str:
    return _read_text_cached(str(path), path.stat().st_mtime)


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    else:
        data = {}
    data.setdefault("done", {})        # task id → ngày hoàn thành
    data.setdefault("cards", {})       # card id → {"box": int, "due": "YYYY-MM-DD"}
    data.setdefault("minitest", {})    # chương → điểm
    return data


def save_progress(data: dict) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(PROGRESS_FILE)


def today() -> dt.date:
    return dt.date.today()


def as_date(x) -> dt.date:
    return x if isinstance(x, dt.date) else dt.date.fromisoformat(str(x))


def weeks() -> list[dict]:
    ws = load_yaml(PLAN / "weeks.yaml")["weeks"]
    for w in ws:
        w["start"] = as_date(w["start"])
        w["end"] = w["start"] + dt.timedelta(days=6)
    return ws


def current_week_index(ws: list[dict]) -> int:
    t = today()
    for i, w in enumerate(ws):
        if w["start"] <= t <= w["end"]:
            return i
    return 0 if t < ws[0]["start"] else len(ws) - 1


def chapters() -> list[dict]:
    return load_yaml(BOOK / "chapters.yaml")["chapters"]


def mock_tests() -> list[dict]:
    return load_yaml(BOOK / "chapters.yaml").get("mock_tests", [])


# ----------------------------------------------------------------- hiển thị markdown
IMG_RE = re.compile(r"^!\[(?P<alt>.*)\]\((?P<src>[^)]+)\)\s*$")


def render_markdown(text: str, chapter_manim: str | None) -> None:
    """Hiện markdown; dòng '% manim: Cảnh' thành video, dòng ảnh ![..](..) thành st.image."""
    buf: list[str] = []
    in_code = False

    def flush():
        if buf:
            st.markdown("\n".join(buf))
            buf.clear()

    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and line.startswith("% manim:"):
            flush()
            scene = line.split(":", 1)[1].strip()
            video = MANIM / "videos" / (chapter_manim or "") / f"{scene}.mp4"
            if video.exists():
                with st.container(border=True):
                    st.caption(f"Minh họa Manim — {scene}")
                    st.video(str(video))
            else:
                st.info(f"Chưa có video cho cảnh {scene}. Render ở trang Manim.")
            continue
        m = IMG_RE.match(line) if not in_code else None
        if m:
            flush()
            path = (BOOK / m["src"]).resolve()
            if path.exists():
                st.image(str(path), caption=m["alt"])
            continue
        buf.append(line)
    flush()


@st.cache_data(show_spinner=False)
def split_sections(md: str) -> tuple[str, list[tuple[str, str]]]:
    """Tách chương thành (tiêu đề, [(tên mục, nội dung)]) theo các dòng '## '."""
    title, sections, cur_name, cur = "", [], None, []
    in_code = False
    for line in md.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and line.startswith("# ") and not title:
            title = line[2:].strip()
            continue
        if not in_code and line.startswith("## "):
            if cur_name is not None:
                sections.append((cur_name, "\n".join(cur)))
            cur_name, cur = line[3:].strip(), []
        else:
            cur.append(line)
    if cur_name is not None:
        sections.append((cur_name, "\n".join(cur)))
    return title, sections


# ----------------------------------------------------------------- trang: Tuần này
@st.fragment
def page_week(progress: dict) -> None:
    ws = weeks()
    idx = current_week_index(ws)
    labels = [f"{w['title']}  ({w['start']:%d/%m} – {w['end']:%d/%m})" for w in ws]
    pick = st.selectbox("Tuần", range(len(ws)), index=idx, format_func=lambda i: labels[i])
    w = ws[pick]
    st.title(w["title"])
    if pick == idx and ws[idx]["start"] <= today() <= ws[idx]["end"]:
        st.caption(f"Tuần hiện tại · hôm nay {today():%d/%m/%Y}")

    tasks = w["tasks"]
    done = [t for t in tasks if t["id"] in progress["done"]]
    h_total = sum(t["hours"] for t in tasks)
    h_done = sum(t["hours"] for t in done)
    nxt = next((m for m in mock_tests() if as_date(m["date"]) >= today()), None)
    c1, c2, c3 = st.columns(3)
    c1.metric("Việc đã xong", f"{len(done)}/{len(tasks)}")
    c2.metric("Giờ đã học", f"{h_done:g}/{h_total:g} giờ")
    if nxt:
        c3.metric(f"Tới {nxt['name']}", f"{(as_date(nxt['date']) - today()).days} ngày")
    st.progress(h_done / h_total if h_total else 0.0)

    def toggle(task_id: str):
        if st.session_state[f"cb_{task_id}"]:
            progress["done"][task_id] = today().isoformat()
        else:
            progress["done"].pop(task_id, None)
        save_progress(progress)

    for key, name in STREAMS.items():
        group = [t for t in tasks if t["stream"] == key]
        if not group:
            continue
        st.subheader(name)
        for t in group:
            st.checkbox(f"{t['text']} · {t['hours']:g} giờ", value=t["id"] in progress["done"],
                        key=f"cb_{t['id']}", on_change=toggle, args=(t["id"],))

    st.divider()
    st.caption("Kế hoạch các tuần nằm ở plan/weeks.yaml; tuần mới được thêm khi chương tiếp theo được viết.")


# ----------------------------------------------------------------- trang: Sách
def page_book() -> None:
    chs = chapters()
    names = [c["title"] + ("" if c["status"] == "ready" else "  (chưa viết)") for c in chs]
    default = next((i for i, c in enumerate(chs) if c["id"] == "chuong-01"), 0)
    i = st.sidebar.selectbox("Chương", range(len(chs)), index=default, format_func=lambda k: names[k])
    ch = chs[i]
    if ch["status"] != "ready":
        st.title(ch["title"])
        st.info("Chương này chưa viết. Sách được viết cuốn chiếu, luôn đi trước tiến độ học 1–2 chương.")
        return
    md = read_text(BOOK / ch["file"])
    title, sections = split_sections(md)
    st.title(title or ch["title"])
    if not sections:
        render_markdown(md, ch.get("manim"))
        return
    # Mặc định chỉ vẽ một mục (nhanh); "Cả chương" để cuối vì chương dài hàng trăm KB làm trình duyệt chậm
    sec_names = [s[0] for s in sections] + ["Cả chương (chậm với chương dài)"]
    s = st.sidebar.radio("Mục", sec_names, key=f"sec_{ch['id']}")
    chosen = sections if s.startswith("Cả chương") else [x for x in sections if x[0] == s]
    if len(chosen) == 1 and len(sections) > 1:
        idx = [x[0] for x in sections].index(chosen[0][0])
        nav_prev, nav_next = st.columns(2)
        if idx > 0:
            nav_prev.caption(f"← Mục trước: {sections[idx - 1][0]}")
        if idx < len(sections) - 1:
            nav_next.caption(f"Mục sau: {sections[idx + 1][0]} →")
    for name, body in chosen:
        st.header(name)
        if name.startswith("Lời giải"):
            st.warning("Chỉ mở sau khi đã tự làm ít nhất 20 phút.")
            with st.expander("Hiện lời giải"):
                render_markdown(body, ch.get("manim"))
        else:
            render_markdown(body, ch.get("manim"))


# ----------------------------------------------------------------- trang: Lab
JUPYTER_PORT = 8866


@st.cache_resource
def jupyter_server() -> dict:
    """Khởi động JupyterLab nền một lần (chỉ nghe 127.0.0.1, có token ngẫu nhiên) để nhúng vào app."""
    import secrets
    import socket
    token = secrets.token_urlsafe(24)
    with socket.socket() as sock:
        sock.settimeout(0.3)
        if sock.connect_ex(("127.0.0.1", JUPYTER_PORT)) == 0:
            return {"ok": False, "error": f"Cổng {JUPYTER_PORT} đang bị chương trình khác dùng."}
    app_port = st.get_option("server.port")
    csp = f"frame-ancestors 'self' http://localhost:{app_port} http://127.0.0.1:{app_port}"
    cmd = [sys.executable, "-m", "jupyterlab", "--no-browser", "--ServerApp.ip=127.0.0.1",
           f"--ServerApp.port={JUPYTER_PORT}", "--ServerApp.port_retries=0",
           f"--ServerApp.token={token}", "--ServerApp.password=",
           f"--ServerApp.root_dir={ROOT}",
           f"--ServerApp.tornado_settings={{'headers': {{'Content-Security-Policy': \"{csp}\"}}}}",
           "--ServerApp.allow_origin=", f"--LabServerApp.app_settings_dir={ROOT / 'app' / 'jupyter_settings'}", "--LabApp.news_url=", "--LabApp.check_for_updates_class=jupyterlab.NeverCheckForUpdate"]
    if hasattr(os, "geteuid") and os.geteuid() == 0:           # chỉ gặp khi chạy thử trong container Linux
        cmd.append("--allow-root")
    flags = 0x08000000 if sys.platform == "win32" else 0      # CREATE_NO_WINDOW
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=flags)
    import atexit
    atexit.register(lambda: proc.poll() is None and proc.terminate())
    return {"ok": True, "token": token, "proc": proc}


def jupyter_ready(timeout: float = 40.0) -> dict:
    import socket
    import time
    info = jupyter_server()
    if not info["ok"]:
        return info
    t0 = time.time()
    while time.time() - t0 < timeout:
        if info["proc"].poll() is not None:
            jupyter_server.clear()
            return {"ok": False, "error": "JupyterLab thoát ngay khi khởi động (thiếu gói jupyterlab trong môi trường? "
                                          "chạy lại setup.bat)."}
        with socket.socket() as sock:
            sock.settimeout(0.3)
            if sock.connect_ex(("127.0.0.1", JUPYTER_PORT)) == 0:
                return info
        time.sleep(0.4)
    return {"ok": False, "error": "JupyterLab khởi động quá lâu."}


def jupyter_url(path: Path, token: str) -> str:
    rel = path.resolve().relative_to(ROOT).as_posix()
    return f"http://localhost:{JUPYTER_PORT}/lab/tree/{rel}?token={token}"


@st.cache_data(show_spinner=False)
def _cells_cached(path: str, mtime: float) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["cells"]


def notebook_cells(nb: Path) -> list[dict]:
    return _cells_cached(str(nb), nb.stat().st_mtime)


def page_lab() -> None:
    import streamlit.components.v1 as components
    st.title("Lab code")
    st.write("Mỗi chương có hai notebook: **lab.ipynb** có chỗ `# TODO` để bạn tự viết, kèm ô kiểm tra tự động; "
             "**loigiai.ipynb** là bản đã chạy kiểm. Nút **Mở và sửa code ngay trong app** nhúng JupyterLab vào trang này: "
             "bạn sửa, chạy ô code và lưu thẳng vào file trong thư mục `labs/`.")
    chs = [c for c in chapters() if c.get("lab")]
    if not chs:
        st.info("Chưa có lab nào.")
        return
    ch = st.selectbox("Chương", chs, format_func=lambda c: c["title"])
    folder = LABS / ch["lab"]
    for nb in sorted(folder.glob("*.ipynb")):
        is_sol = nb.stem.startswith("loigiai")
        key = f"open_{nb}"
        with st.container(border=True):
            st.subheader(("Lời giải — " if is_sol else "Bài lab — ") + nb.name)
            if is_sol:
                st.caption("Chỉ mở sau khi đã tự làm. Bí quá 20 phút mới mở.")
            c1, c2 = st.columns([1, 1])
            if c1.button("Mở và sửa code ngay trong app", key=f"btn_{key}", type="primary"):
                st.session_state["lab_open"] = str(nb)
            if st.session_state.get("lab_open") == str(nb):
                if c2.button("Đóng trình soạn thảo", key=f"close_{key}"):
                    st.session_state["lab_open"] = None
                    st.rerun()
                with st.spinner("Đang khởi động JupyterLab (lần đầu mất vài giây)..."):
                    info = jupyter_ready()
                if not info["ok"]:
                    st.error(info["error"])
                else:
                    url = jupyter_url(nb, info["token"])
                    components.iframe(url, height=820, scrolling=False)
                    st.caption("Nhớ lưu (Ctrl+S) trước khi đóng. Cần màn hình rộng hơn? "
                               f"[Mở trong cửa sổ riêng]({url}).")
            if st.toggle("Xem nhanh nội dung (chỉ đọc)", key=f"prev_{nb}"):
                cells = notebook_cells(nb)
                for c in cells:
                    src = "".join(c["source"])
                    if c["cell_type"] == "markdown":
                        st.markdown(src)
                    else:
                        st.code(src, language="python")


# ----------------------------------------------------------------- trang: Manim
def scene_classes(path: Path) -> dict[str, str]:
    return _scene_classes_cached(str(path), path.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _scene_classes_cached(path_str: str, mtime: float) -> dict[str, str]:
    """Tên lớp Scene → mã nguồn của lớp, đọc bằng ast."""
    path = Path(path_str)
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and any(
                getattr(b, "id", getattr(b, "attr", "")).endswith("Scene") for b in node.bases):
            out[node.name] = ast.get_source_segment(src, node) or ""
    return out


def render_scene(py_file: Path, cls: str, out_mp4: Path, quality: str = "-qm") -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, "-m", "manim", quality, "--disable_caching", "--media_dir", tmp, str(py_file), cls]
        proc = subprocess.run(cmd, cwd=str(py_file.parent), capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        found = [p for p in Path(tmp).rglob(f"{cls}.mp4") if "partial_movie_files" not in p.parts]
        if proc.returncode != 0 or not found:
            return False, (proc.stderr or proc.stdout)[-3000:]
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(found[0], out_mp4)
        return True, ""


TEMPLATE = '''"""Cảnh của tôi. Render ở trang Manim của app."""
import sys
from manim import *

FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"


class PowerIteration(Scene):
    def construct(self):
        title = Text("Power iteration", font=FONT, font_size=34).to_edge(UP)
        self.play(Write(title))
        # TODO: vẽ vector v₀, áp A rồi chuẩn hóa 6 lần, hiện góc tới vector riêng (1, 1)/√2
        self.wait(1)
'''


def page_manim() -> None:
    st.title("Minh họa Manim")
    cat = load_yaml(MANIM / "catalog.yaml") or {}
    tab_view, tab_mine = st.tabs(["Cảnh minh họa của sách", "Cảnh của tôi"])
    with tab_view:
        if not cat:
            st.info("Chưa có cảnh nào.")
        else:
            key = st.selectbox("Chương", list(cat), format_func=lambda k: k.replace("chuong", "Chương "))
            entry = cat[key]
            py = MANIM / entry["file"]
            sources = scene_classes(py)
            scenes = entry["scenes"]
            pick = st.selectbox("Cảnh", range(len(scenes)), key=f"scene_{key}",
                                format_func=lambda i: f"{scenes[i]['section']} · {scenes[i]['title']}")
            for sc in [scenes[pick]]:          # chỉ nạp một video mỗi lần cho nhẹ trình duyệt
                with st.container(border=True):
                    st.subheader(f"{sc['section']} · {sc['title']}")
                    st.caption(sc["desc"])
                    video = MANIM / "videos" / key / f"{sc['cls']}.mp4"
                    if video.exists():
                        st.video(str(video))
                    else:
                        st.info("Chưa có video — bấm Render.")
                    if st.toggle("Xem code", key=f"code_{key}_{sc['cls']}"):
                        st.code(sources.get(sc["cls"], "# không tìm thấy lớp"), language="python")
                    if st.button("Render lại (720p)", key=f"r_{key}_{sc['cls']}"):
                        with st.spinner("Đang render, thường mất 10–60 giây…"):
                            ok, err = render_scene(py, sc["cls"], video)
                        st.success("Xong.") if ok else st.error(err)
    with tab_mine:
        mine = MANIM / "cua-toi"
        mine.mkdir(exist_ok=True)
        files = sorted(mine.glob("*.py"))
        st.write("Đặt file cảnh của bạn trong `manim/cua-toi/`. Chọn file và lớp cảnh rồi bấm Render.")
        if not files:
            if st.button("Tạo file mẫu cho bài dựng cảnh chương 1"):
                (mine / "chuong01_power_iteration.py").write_text(TEMPLATE, encoding="utf-8")
                st.rerun()
            return
        f = st.selectbox("File", files, format_func=lambda p: p.name)
        try:
            classes = scene_classes(f)
        except SyntaxError as e:
            st.error(f"File có lỗi cú pháp: {e}")
            return
        if not classes:
            st.warning("Không tìm thấy lớp nào kế thừa Scene trong file này.")
            return
        cls = st.selectbox("Cảnh", list(classes))
        quality = st.radio("Chất lượng", ["-ql (480p, nhanh)", "-qm (720p)"], horizontal=True)
        out = mine / "videos" / f"{cls}.mp4"
        if st.button("Render cảnh của tôi"):
            with st.spinner("Đang render…"):
                ok, err = render_scene(f, cls, out, quality.split()[0])
            if not ok:
                st.error(err)
        if out.exists():
            st.video(str(out))


# ----------------------------------------------------------------- trang: Ôn tập
def load_cards() -> list[dict]:
    files = sorted(CARDS.glob("*.csv"))
    return _cards_cached(tuple((str(f), f.stat().st_mtime) for f in files))


@st.cache_data(show_spinner=False)
def _cards_cached(files: tuple) -> list[dict]:
    cards = []
    for f in (Path(x[0]) for x in files):
        with f.open(encoding="utf-8", newline="") as fh:
            for i, row in enumerate(csv.reader(fh)):
                if len(row) >= 2:
                    cards.append({"id": f"{f.stem}:{i}", "deck": f.stem, "front": row[0], "back": row[1]})
    return cards


@st.fragment
def page_review(progress: dict) -> None:
    st.title("Ôn tập flashcard")
    cards = load_cards()
    decks = sorted({c["deck"] for c in cards})
    deck = st.selectbox("Bộ thẻ", ["Tất cả"] + decks)
    pool = [c for c in cards if deck == "Tất cả" or c["deck"] == deck]
    state = lambda c: progress["cards"].get(c["id"], {"box": 1, "due": today().isoformat()})
    due = [c for c in pool if as_date(state(c)["due"]) <= today()]
    import pandas as pd
    boxes = pd.Series([state(c)["box"] for c in pool]).value_counts().reindex(range(1, 6), fill_value=0)
    c1, c2 = st.columns([1, 2])
    c1.metric("Thẻ đến hạn hôm nay", len(due))
    c2.bar_chart(pd.DataFrame({"Số thẻ": boxes.values}, index=[f"Hộp {b}" for b in boxes.index]))

    qkey = f"queue_{deck}"
    if qkey not in st.session_state or st.session_state.get(f"{qkey}_day") != today().isoformat():
        st.session_state[qkey] = [c["id"] for c in due]
        st.session_state[f"{qkey}_day"] = today().isoformat()
    queue = st.session_state[qkey]
    if not queue:
        st.success("Hết thẻ cần ôn hôm nay.")
        return
    card = next(c for c in cards if c["id"] == queue[0])
    with st.container(border=True):
        st.caption(f"Còn {len(queue)} thẻ · hộp {state(card)['box']}")
        st.markdown(f"### {card['front']}")
        if st.session_state.get("flipped") == card["id"]:
            st.markdown(f"**{card['back']}**")
            a, b = st.columns(2)

            def answer(ok: bool):
                s = state(card)
                box = min(s["box"] + 1, 5) if ok else 1
                progress["cards"][card["id"]] = {
                    "box": box, "due": (today() + dt.timedelta(days=LEITNER_DAYS[box])).isoformat()}
                save_progress(progress)
                queue.pop(0)
                if not ok:
                    queue.append(card["id"])       # quên: ôn lại cuối lượt hôm nay
                st.session_state["flipped"] = None

            a.button("Nhớ", on_click=answer, args=(True,), use_container_width=True)
            b.button("Quên", on_click=answer, args=(False,), use_container_width=True)
        else:
            st.button("Lật thẻ", on_click=lambda: st.session_state.update(flipped=card["id"]))


# ----------------------------------------------------------------- trang: Tiến độ
@st.fragment
def page_progress(progress: dict) -> None:
    st.title("Tiến độ")
    ws = weeks()
    rows = []
    for w in ws:
        plan_h = sum(t["hours"] for t in w["tasks"])
        done_h = sum(t["hours"] for t in w["tasks"] if t["id"] in progress["done"])
        rows.append({"Tuần": w["start"].strftime("%d/%m"), "Kế hoạch (giờ)": plan_h, "Đã làm (giờ)": done_h})
    import pandas as pd
    df = pd.DataFrame(rows).set_index("Tuần")
    st.subheader("Giờ học theo tuần")
    st.bar_chart(df, stack=False)

    st.subheader("Việc còn tồn của các tuần đã qua")
    late = [(w, t) for w in ws if w["end"] < today() for t in w["tasks"] if t["id"] not in progress["done"]]
    if late:
        for w, t in late:
            st.write(f"- {w['start']:%d/%m} · {STREAMS[t['stream']]} · {t['text']}")
    else:
        st.write("Không có việc nào tồn.")

    st.subheader("Điểm mini-test (thang 30)")
    st.caption("Từ 24: sang chương sau · 18 đến dưới 24: ôn lại mục sai rồi làm lại sau 3 ngày · dưới 18: học lại chương.")
    for ch in [c for c in chapters() if c["status"] == "ready" and c["id"].startswith("chuong")]:
        cur = progress["minitest"].get(ch["id"])
        val = st.number_input(ch["title"], 0, 30, value=int(cur) if cur is not None else 0, key=f"mt_{ch['id']}")
        if st.button("Lưu điểm", key=f"save_{ch['id']}"):
            progress["minitest"][ch["id"]] = int(val)
            save_progress(progress)
            msg = ("Đạt — sang chương sau." if val >= 24 else
                   "Ôn lại các mục ứng với câu sai, làm lại sau 3 ngày." if val >= 18 else "Nên học lại chương.")
            st.success(f"Đã lưu {val}/30. {msg}")


# ----------------------------------------------------------------- khung chính
PAGES = {
    "Tuần này": page_week,
    "Sách": page_book,
    "Lab": page_lab,
    "Manim": page_manim,
    "Ôn tập": page_review,
    "Tiến độ": page_progress,
}


def main() -> None:
    progress = load_progress()
    st.sidebar.title("Sổ tay Residency")
    st.sidebar.caption("Track Research · NLP")
    page = st.sidebar.radio("Trang", list(PAGES), label_visibility="collapsed")
    st.sidebar.divider()
    fn = PAGES[page]
    if page in ("Tuần này", "Ôn tập", "Tiến độ"):
        fn(progress)
    else:
        fn()


main()
