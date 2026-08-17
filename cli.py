import os
import sys
import json
import shutil
import tempfile
import logging
import argparse
from pathlib import Path

# Force UTF-8 on Windows stdout/stderr
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Setup logging to stderr so JSON stdout remains clean for IPC
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)
logger = logging.getLogger("video-desktop.engine")

# --- JSON-RPC 2.0 over stdio (Medal-style: request/response/notification) ---
#
# Main -> Engine (stdin):   {"jsonrpc":"2.0","id":<int>,"method":"<action>","params":{...}}
# Engine -> Main (stdout):  {"jsonrpc":"2.0","id":<int>,"result":{...}}            (response)
#                           {"jsonrpc":"2.0","id":<int>,"error":{"code":..,"message":..}}
#                           {"jsonrpc":"2.0","method":"<event>","params":{...}}     (notification)
#
# Handshake: engine sends "engine_ready" notification ONLY after ffmpeg/disk checks;
# main buffers requests until it arrives (A2/A3). Heartbeat: main pings, engine answers.
# Graceful shutdown: main sends "shutdown" notification, engine replies "engine_stopped".

_JSONRPC = "2.0"


def _write_line(payload: dict):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def notify(method: str, **params):
    """Engine -> Main notification (fire-and-forget event)."""
    _write_line({"jsonrpc": _JSONRPC, "method": method, "params": params})


def respond(request_id, result=None, error=None):
    """Engine -> Main response to a request (matched by id)."""
    msg = {"jsonrpc": _JSONRPC, "id": request_id}
    if error is not None:
        msg["error"] = error if isinstance(error, dict) else {"code": -32000, "message": str(error)}
    else:
        msg["result"] = {} if result is None else result
    _write_line(msg)


# --- Readiness / handshake (A2) ---

def readiness_payload() -> dict:
    """Check ffmpeg/ffprobe + disk space BEFORE declaring engine ready."""
    from engine.media.ffmpeg_utils import get_ffmpeg_executable, get_ffprobe_executable

    ffmpeg = get_ffmpeg_executable()
    ffprobe = get_ffprobe_executable()

    ffmpeg_ok = ffprobe_ok = False
    try:
        import subprocess
        r = subprocess.run([ffmpeg, "-version"], capture_output=True, timeout=15)
        ffmpeg_ok = r.returncode == 0
    except Exception as e:
        logger.warning(f"ffmpeg version check failed: {e}")
    try:
        import subprocess
        r = subprocess.run([ffprobe, "-version"], capture_output=True, timeout=15)
        ffprobe_ok = r.returncode == 0
    except Exception as e:
        logger.warning(f"ffprobe version check failed: {e}")

    disk_free_gb = None
    try:
        usage = shutil.disk_usage(tempfile.gettempdir())
        disk_free_gb = round(usage.free / (1024 ** 3), 1)
    except Exception:
        pass

    return {
        "status": "online",
        "version": "2.0.0",
        "ffmpeg": {"path": ffmpeg, "ok": ffmpeg_ok},
        "ffprobe": {"path": ffprobe, "ok": ffprobe_ok},
        "diskFreeGB": disk_free_gb,
    }


# --- Legacy event emitter kept for the old renderer contract ---

def send_ipc(event_type: str, data: dict):
    """Output a JSON-RPC notification (previously: {type, ...data})."""
    notify(event_type, **data)


# --- Handlers ---

def handle_ping(payload: dict) -> dict:
    return {"pong": True, "ts": None}


def handle_probe(payload: dict) -> dict:
    from engine.media.ffmpeg_utils import video_info_local
    filepath = payload.get("filepath")
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"Source file not found: {filepath}")
    info = video_info_local(filepath)
    return {"success": True, "info": info}


def _consumer_config() -> "DownloaderConfig":
    """Строит конфиг довнлоадера из конфига движка (движок не знает про
    библиотеку — библиотека не знает про движок)."""
    from engine.config import get_config
    from ytdlp import DownloaderConfig

    e = get_config().engine
    return DownloaderConfig(
        temp_dir=e.temp_dir,
        ytdlp_cookies=e.ytdlp_cookies,
        ytdlp_cookies_from_browser=e.ytdlp_cookies_from_browser,
        ytdlp_proxy=e.ytdlp_proxy,
        ytdlp_max_height=e.ytdlp_max_height,
        ytdlp_bgutil_script=e.ytdlp_bgutil_script,
        ytdlp_bgutil_url=e.ytdlp_bgutil_url,
    )


def _warn_running_cookie_browsers(
    consumer_cfg: "DownloaderConfig",
    job_id: str | None = None,
    cookie_source: str | None = None,
    cookie_file: str = "",
    cookie_browser: str = "",
) -> set[str]:
    """Если браузер, из которого планируем брать куки, сейчас запущен — шлём
    событие cookies_warning, чтобы UI показал модалку «закройте браузер».
    Возвращает множество запущенных браузеров, которые стоит исключить из
    вариантов куки (их БД недоступна, пока процесс жив).

    cookie_source/cookie_file/cookie_browser — явный выбор из настроек
    приложения (none/file/login/browser), см. ytdlp.cookies_opts."""
    from ytdlp import running_browsers

    from ytdlp.cookies import cookies_opts
    running = running_browsers()
    if not running:
        return set()
    wanted = set()
    for opt in cookies_opts(
        consumer_cfg,
        source=cookie_source,
        cookie_file=cookie_file,
        cookie_browser=cookie_browser,
    ):
        cb = opt.get("cookiesfrombrowser")
        if cb:
            wanted.add(cb[0])
    busy = wanted & running
    if busy:
        send_ipc("cookies_warning", {"browsers": sorted(busy), "job_id": job_id})
    return busy


def _cookie_params(payload: dict) -> tuple[str | None, str, str]:
    """Достаёт из payload явный выбор куки (none/file/login/browser). Значение
    из настроек приложения передаётся на каждый запрос, потому что конфигурация
    движка (env) живёт от запуска процесса и не обновляется на лету."""
    source = payload.get("cookie_source")
    if source not in ("none", "file", "login", "browser"):
        source = None
    return source, payload.get("cookie_file") or "", payload.get("cookie_browser") or ""


def handle_video_info(payload: dict) -> dict:
    """Получает метаданные видео по URL (yt-dlp, без скачивания):
    title/thumbnail/duration/размеры/экстрактор — для карточки на фронте."""
    from ytdlp import fetch_video_info

    url = payload.get("url")
    if not url:
        raise ValueError("Missing url")
    cookie_source, cookie_file, cookie_browser = _cookie_params(payload)
    consumer_cfg = _consumer_config()
    _warn_running_cookie_browsers(
        consumer_cfg,
        cookie_source=cookie_source,
        cookie_file=cookie_file,
        cookie_browser=cookie_browser,
    )
    return fetch_video_info(
        url,
        cfg=consumer_cfg,
        cookie_source=cookie_source,
        cookie_file=cookie_file,
        cookie_browser=cookie_browser,
    )


def handle_download(payload: dict) -> dict:
    """Скачивает видео по URL (yt-dlp) с выбором наилучшего доступного
    качества и деградацией на худшее, если лучшего нет. Кап по высоте
    YTDLP_MAX_HEIGHT (по умолчанию 1080)."""
    from ytdlp import download_url

    url = payload.get("url")
    output_dir = payload.get("output_dir")
    if not url:
        raise ValueError("Missing url")
    if not output_dir:
        raise ValueError("Missing output_dir")
    cookie_source, cookie_file, cookie_browser = _cookie_params(payload)
    consumer_cfg = _consumer_config()
    # Apply per-request overrides from app settings
    if payload.get("max_height"):
        consumer_cfg.ytdlp_max_height = int(payload["max_height"])
    busy = _warn_running_cookie_browsers(
        consumer_cfg,
        payload.get("job_id"),
        cookie_source=cookie_source,
        cookie_file=cookie_file,
        cookie_browser=cookie_browser,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if busy:
        # Все источники куки заняты запущенными браузерами — качать без куки
        # бесполезно (YouTube сразу отдаст бот-проверку), а перебор профилей
        # с паузами заставляет задачу «висеть» минуты. Быстро падаем, чтобы
        # модалка «Закройте браузер» отработала и перезапустила задачу.
        raise RuntimeError(
            "Browser is running, cannot read its cookies. "
            "Close the browser and restart the job."
        )
    meta = download_url(
        url,
        out / "source",
        on_progress=lambda p: send_ipc("download_progress", {"percent": int(p)}),
        cfg=consumer_cfg,
        cookie_source=cookie_source,
        cookie_file=cookie_file,
        cookie_browser=cookie_browser,
    )
    return {"success": True, **meta}


def handle_transcribe(payload: dict) -> dict:
    """Транскрибирует секцию [start, end] → реплики для редактора."""
    filepath = payload.get("filepath")
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"Source file not found: {filepath}")
    from pathlib import Path
    from engine.media.subtitles import transcribe_section
    start = float(payload.get("start", 0))
    end = float(payload.get("end", start + 30))
    result = transcribe_section(Path(filepath), start, end)
    return {
        "success": True,
        "language": result.get("language", ""),
        "cues": result.get("cues", []),
    }


def handle_scene_detect(payload: dict) -> dict:
    """Детекция сцен для редактора: границы сцен в секундах."""
    filepath = payload.get("filepath")
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"Source file not found: {filepath}")
    from engine.reframe.scene_detection import detect_scenes
    scenes, fps = detect_scenes(filepath)
    return {
        "success": True,
        "fps": float(fps) if fps else None,
        "scenes": [
            {"start": float(s.get_seconds()), "end": float(e.get_seconds())}
            for s, e in scenes
        ],
    }


def handle_face_scan(payload: dict) -> dict:
    """Скан лиц для редактора: временные диапазоны, где на видео есть лица.

    Выборка с шагом stride кадров; диапазоны сливаются, если между
    обнаружениями было не больше merge_gap кадров."""
    filepath = payload.get("filepath")
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"Source file not found: {filepath}")
    stride = max(int(payload.get("stride", 4)), 1)
    merge_gap = max(int(payload.get("merge_gap", 30)), 0)
    max_frames = max(int(payload.get("max_frames", 0)), 0)

    from engine.reframe.analysis import detect_face_candidates
    import cv2

    cap = cv2.VideoCapture(filepath)
    processed = 0
    intervals = []
    last_end = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if max_frames > 0 and total > max_frames:
            stride = max(1, int(round(total / max_frames)))
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                processed += 1
                found = bool(detect_face_candidates(frame))
                t = frame_idx / fps
                if found:
                    if last_end is not None and frame_idx - last_end <= merge_gap:
                        intervals[-1] = (intervals[-1][0], t)
                    else:
                        intervals.append((t, t))
                    last_end = frame_idx
            frame_idx += 1
    finally:
        cap.release()

    return {
        "success": True,
        "fps": float(fps),
        "total_frames": processed,
        "faces": [{"start": float(s), "end": float(e)} for s, e in intervals],
    }


def handle_render_edit(payload: dict) -> dict:
    """Рендер отредактированного клипа: trim + transform + субтитры."""
    from engine.editor import render_edit
    result = render_edit(payload)
    return {"success": True, **result}


def _check_disk_space(output_dir: str | Path, min_free_gb: float = 10.0) -> None:
    """Preflight disk check (B5): fail before starting a long render, not mid-way."""
    try:
        usage = shutil.disk_usage(Path(output_dir))
        free_gb = usage.free / (1024 ** 3)
    except Exception as e:
        raise RuntimeError(f"Failed to check disk space for {output_dir}: {e}")
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Not enough free disk space: {free_gb:.1f} GB available, need >= {min_free_gb:.0f} GB"
        )


def handle_render(payload: dict) -> dict:
    from engine.media.ffmpeg_utils import cut_local_section, video_info_local
    from engine.reframe.engine import render as reframe_render

    source_path = payload.get("source_path")
    output_dir = payload.get("output_dir")
    highlights = payload.get("highlights", [])
    aspect_ratio = payload.get("aspect_ratio", "9:16")
    enable_subtitles = payload.get("subtitles", False)

    if not source_path or not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    _check_disk_space(out_dir_path)  # B5

    total_clips = len(highlights)
    send_ipc("render_started", {"total_clips": total_clips, "source": source_path})

    for idx, hl in enumerate(highlights):
        start_sec = int(hl.get("start", 0))
        end_sec = int(hl.get("end", start_sec + 5))
        clip_name = f"clip_{idx + 1}_{start_sec}s_{end_sec}s.mp4"
        out_clip_path = str(out_dir_path / clip_name)

        send_ipc("clip_progress", {
            "clip_index": idx + 1,
            "total_clips": total_clips,
            "step": "cutting",
            "percent": int((idx / total_clips) * 100)
        })

        temp_cut_dir = out_dir_path / f"temp_{idx}"
        temp_cut_dir.mkdir(parents=True, exist_ok=True)
        try:
            # 1. Cut section (B1 fast-path inside cut_local_section)
            cut_file = cut_local_section(source_path, temp_cut_dir, start_sec, end_sec, idx=idx)

            # 2. Reframe to vertical (9:16)
            send_ipc("clip_progress", {
                "clip_index": idx + 1,
                "total_clips": total_clips,
                "step": "reframing",
                "percent": int(((idx + 0.5) / total_clips) * 100)
            })

            reframe_render(str(cut_file), out_clip_path, aspect_ratio=aspect_ratio)

            # 3. Subtitles if requested
            if enable_subtitles:
                send_ipc("clip_progress", {
                    "clip_index": idx + 1,
                    "total_clips": total_clips,
                    "step": "subtitles",
                    "percent": int(((idx + 0.8) / total_clips) * 100)
                })
                from engine.media.subtitles import generate_and_burn_subtitles
                generate_and_burn_subtitles(Path(out_clip_path), Path(out_clip_path))

            # 4. Verify the delivered clip (B4): never ship a broken file.
            verify_info = video_info_local(out_clip_path)
            if not (verify_info.get("width") and verify_info.get("height")):
                raise RuntimeError(
                    f"Delivered clip failed verification (no video stream): {out_clip_path}"
                )
            if not (verify_info.get("duration") or 0) > 0:
                raise RuntimeError(
                    f"Delivered clip failed verification (zero duration): {out_clip_path}"
                )

            send_ipc("clip_completed", {
                "clip_index": idx + 1,
                "total_clips": total_clips,
                "path": out_clip_path
            })
        except Exception as e:
            logger.exception(f"Failed rendering clip {idx + 1}")
            send_ipc("clip_error", {"clip_index": idx + 1, "error": str(e)})
        finally:
            if temp_cut_dir.exists():
                shutil.rmtree(temp_cut_dir, ignore_errors=True)

    send_ipc("render_finished", {"output_dir": str(output_dir), "total_clips": total_clips})
    return {"output_dir": str(output_dir), "total_clips": total_clips}


def handle_ensure_job_dir(payload: dict) -> dict:
    """Создаёт рабочую папку проекта сразу при создании джоба, чтобы
    «Открыть папку» работала ещё до рендера. Возвращает готовый путь."""
    output_dir = payload.get("output_dir")
    if not output_dir:
        raise ValueError("Missing output_dir")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {"success": True, "dir": str(out)}


HANDLERS = {
    "ping": handle_ping,
    "probe": handle_probe,
    "video_info": handle_video_info,
    "download": handle_download,
    "transcribe": handle_transcribe,
    "scene_detect": handle_scene_detect,
    "face_scan": handle_face_scan,
    "render_edit": handle_render_edit,
    "render": handle_render,
    "ensure_job_dir": handle_ensure_job_dir,
}


def handle_shutdown():
    """Graceful shutdown (A5): stop render work and exit cleanly."""
    send_ipc("engine_stopped", {"reason": "shutdown"})
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="video-desktop engine CLI")
    parser.add_argument("--listen", action="store_true", help="Listen for JSON-RPC commands from stdin")
    args = parser.parse_args()

    if args.listen:
        # Handshake FIRST, after readiness checks — main will not send
        # requests until this notification arrives (A2/A3).
        notify("engine_ready", **readiness_payload())

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
            except Exception as exc:
                logger.exception("Error parsing JSON-RPC command: %r", line)
                send_ipc("error", {"message": str(exc)})
                continue

            # Notification (no id): shutdown etc.
            if "id" not in cmd or cmd.get("id") is None:
                method = cmd.get("method")
                if method == "shutdown":
                    handle_shutdown()
                else:
                    send_ipc("unknown_action", {"action": method})
                continue

            # Request (has id): dispatch to handler, respond with matching id.
            request_id = cmd.get("id")
            method = cmd.get("method")
            params = cmd.get("params") or {}

            if method not in HANDLERS:
                respond(request_id, error={"code": -32601, "message": f"Method not found: {method}"})
                continue

            try:
                result = HANDLERS[method](params)
                respond(request_id, result=result)
            except Exception as exc:
                logger.exception(f"Error handling '{method}'")
                respond(request_id, error={"code": -32000, "message": str(exc)})


if __name__ == "__main__":
    main()
