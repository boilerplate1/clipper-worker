"""Редактор клипов: точный трим + трансформация + субтитры.

Используется IPC-командами ``transcribe`` и ``render_edit`` из ``cli.py``.
В отличие от пайплайна highlights (паддинг 2с, авто-рефрейм), здесь:

* точный [start, end] из исходника — без паддинга;
* пользовательская трансформация: кроп-регион, зум, панорамирование;
* субтитры: реплики с текстом/временем/позицией/размером → .srt (всегда)
  и опционально вжжённые в кадр через ASS;
* на выходе .mp4 + .srt рядом.

Входной payload ``render_edit``::

    {
      "source_path": str,
      "output_dir": str,
      "filename": str,                 // базовое имя без расширения
      "start": float, "end": float,    // сек, в исходнике
      "aspect_ratio": "9:16",          // формат вывода
      "transform": {                   // необязательно
        "mode": "crop",                // "crop" | "none"
        "crop": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},  // регион исходника 0..1
        "zoom": 1.0,                   // >1 = ближе
        "pan_x": 0.0, "pan_y": 0.0     // смещение центра региона, доля региона
      },
      "subtitles": {                   // необязательно
        "cues": [{"start": float, "end": float, "text": str,
                   "x": float, "y": float, "scale": float}],
        "style": {"font_name": str, "font_size": int, "font_color": str,
                   "border_color": str, "border_width": int, "uppercase": bool},
        "burn": true                   // вжигать ASS в кадр
      }
    }

Координаты реплик (x, y) — нормализованные (0..1) относительно ВЫХОДНОГО
кадра, центр блока. start/end реплик — относительно НАЧАЛА КЛИПА.
"""

import os
import tempfile
from pathlib import Path
from typing import Any

from engine.media.ffmpeg_utils import (
    METADATA_SCRUB,
    audio_encode_args,
    get_ffmpeg_executable,
    hwaccel_args,
    run_ffmpeg_progress,
    video_encode_args,
    video_info_local,
)
from engine.media.subtitles import (
    generate_ass_cues,
    generate_srt,
    transcribe_section,
)

# Минимальная ширина доставки (как в reframe/engine.py): короткие платформы
# ожидают вертикаль 1080px; апскейлим до неё, а не отдаём размытое.
DELIVERY_MIN_WIDTH = 1080


def _clamp(value: float, lo: float, hi: float, default: float) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        num = float(default)
    return max(lo, min(hi, num))


def delivery_size(orig_w: float, orig_h: float, aspect_ratio: str) -> tuple[int, int]:
    """Выходной (w, h) под aspect_ratio, чётный, >= DELIVERY_MIN_WIDTH."""
    if ":" in aspect_ratio:
        num, den = aspect_ratio.split(":")
        ar = float(num) / float(den)
    else:
        ar = float(aspect_ratio)
    out_h = int(round(orig_h))
    out_w = int(round(out_h * ar))
    if out_w > orig_w:
        out_w = int(round(orig_w))
        out_h = int(round(out_w / ar))
    if out_w < DELIVERY_MIN_WIDTH:
        out_w = DELIVERY_MIN_WIDTH
        out_h = int(round(out_w / ar))
    out_w += out_w % 2
    out_h += out_h % 2
    return out_w, out_h


def _crop_filter(orig_w: float, orig_h: float, transform: dict[str, Any]) -> str:
    """ffmpeg crop-фильтр для пользовательского региона/зума/пана.

    Регион задаётся в долях исходного кадра (0..1); зум и панорама применяются
    к центру региона. Возвращает строку "crop=w:h:x:y" (чётные значения).
    """
    crop = transform.get("crop") or {}
    cx = _clamp(crop.get("x", 0.0) + crop.get("w", 1.0) / 2, 0.0, 1.0, 0.5)
    cy = _clamp(crop.get("y", 0.0) + crop.get("h", 1.0) / 2, 0.0, 1.0, 0.5)
    reg_w = max(_clamp(crop.get("w", 1.0), 0.01, 1.0, 1.0), 0.01)
    reg_h = max(_clamp(crop.get("h", 1.0), 0.01, 1.0, 1.0), 0.01)
    zoom = max(_clamp(transform.get("zoom", 1.0), 1.0, 8.0, 1.0), 1.0)
    pan_x = _clamp(transform.get("pan_x", 0.0), -1.0, 1.0, 0.0)
    pan_y = _clamp(transform.get("pan_y", 0.0), -1.0, 1.0, 0.0)

    w = int(round(orig_w * reg_w / zoom))
    h = int(round(orig_h * reg_h / zoom))
    w += w % 2
    h += h % 2
    w = max(w, 2)
    h = max(h, 2)
    x = int(round(cx * orig_w - w / 2 + pan_x * w))
    y = int(round(cy * orig_h - h / 2 + pan_y * h))
    x = max(0, min(int(orig_w) - w, x))
    y = max(0, min(int(orig_h) - h, y))
    return f"crop={w}:{h}:{x}:{y}"


def trim_and_transform(
    source_path: Path,
    start: float,
    end: float,
    output_path: Path,
    aspect_ratio: str,
    transform: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Точный трим [start, end] + трансформация, возвращает (out_w, out_h)."""
    transform = transform or {}
    info = video_info_local(source_path)
    orig_w = float(info.get("width") or 0)
    orig_h = float(info.get("height") or 0)
    if not orig_w or not orig_h:
        raise RuntimeError(f"Failed to read resolution of {source_path}")

    out_w, out_h = delivery_size(orig_w, orig_h, aspect_ratio)

    vf_parts: list[str] = []
    if transform.get("mode") == "crop":
        vf_parts.append(_crop_filter(orig_w, orig_h, transform))
    vf_parts.append(f"scale={out_w}:{out_h},setsar=1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dur = max(end - start, 0.1)
    result = run_ffmpeg_progress(
        [get_ffmpeg_executable(), "-y", "-hide_banner", "-loglevel", "error",
         "-progress", "pipe:1", "-nostats",
         "-ss", f"{start:.3f}", *hwaccel_args(), "-i", str(source_path),
         "-t", f"{dur:.3f}", "-vf", ",".join(vf_parts),
         *video_encode_args(), *audio_encode_args(), "-b:a", "192k",
         "-avoid_negative_ts", "make_zero", *METADATA_SCRUB,
         "-use_editlist", "0", "-movflags", "+faststart", str(output_path)]
    )
    if result != 0:
        raise RuntimeError(f"FFmpeg trim/transform failed (rc={result})")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise FileNotFoundError(f"Trim output not produced from {source_path}")
    return out_w, out_h


def render_edit(payload: dict[str, Any]) -> dict[str, Any]:
    """Полный рендер отредактированного клипа. Возвращает {video_path, srt_path}."""
    source_path = Path(payload["source_path"])
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    output_dir = Path(payload.get("output_dir") or source_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = str(payload.get("filename") or "clip").strip() or "clip"
    if not filename.lower().endswith(".mp4"):
        filename += ".mp4"
    video_path = output_dir / filename
    srt_path = video_path.with_suffix(".srt")

    start = float(payload.get("start", 0))
    end = float(payload.get("end", start + 5))
    aspect_ratio = str(payload.get("aspect_ratio") or "9:16")
    transform = payload.get("transform") or {}
    subs = payload.get("subtitles") or {}
    cues = list(subs.get("cues") or [])
    burn = bool(subs.get("burn", False))

    workdir = Path(tempfile.mkdtemp(prefix="edit_render_"))
    try:
        cut_path = workdir / "cut.mp4"
        out_w, out_h = trim_and_transform(
            source_path, start, end, cut_path, aspect_ratio, transform
        )

        # SRT всегда (это формат по умолчанию редактора).
        if cues:
            generate_srt(cues, srt_path)

        if burn and cues:
            ass_path = workdir / "subs.ass"
            if generate_ass_cues(
                cues, ass_path, style=subs.get("style"), play_w=out_w, play_h=out_h
            ):
                from engine.media.subtitles import burn_subtitles

                burn_subtitles(cut_path, ass_path, video_path)
                return {
                    "video_path": str(video_path),
                    "srt_path": str(srt_path),
                    "burned": True,
                }

        # Без вжигания — перемещаем уже готовый cut.
        import shutil

        shutil.move(str(cut_path), str(video_path))
        return {
            "video_path": str(video_path),
            "srt_path": str(srt_path) if cues else None,
            "burned": False,
        }
    finally:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)
