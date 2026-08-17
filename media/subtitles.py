"""Субтитры: Whisper-транскрипция + генерация ASS/SRT + вжигание через ffmpeg.

Поток:
  1. ``transcribe_audio`` — faster-whisper распознаёт речь (таймкод на слово).
  2. ``generate_ass`` — по транскрипту собираем караоке-ASS (активное слово
     подсвечено), разбитое на короткие блоки под вертикальное видео.
  3. ``burn_subtitles`` — вжигаем ASS в клип через ffmpeg/libass.


"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from engine.media.ffmpeg_utils import (
    METADATA_SCRUB,
    get_ffmpeg_executable,
    get_ffprobe_executable,
    hwaccel_args,
    video_encode_args,
)

# Поправка синхронизации субтитров (сек, добавляется к таймкоду каждого слова).
# Whisper-таймкоды иногда запаздывают на доли секунды; отрицательное значение
# сдвигает их раньше.
SUBTITLE_OFFSET = float(os.getenv("CLIPPER_SUBTITLE_OFFSET", "-0.12"))

# Нижний отступ субтитров (в единицах PlayResY=288) — поднимает текст выше
# нижнего UI TikTok/Reels, иначе платформа перекрывает его своими кнопками.
SAFE_MARGIN_V = 43

# Стиль по умолчанию: современный TikTok/CapCut-вид (караоке с подсветкой слова).
# Размер задаётся в единицах PlayResY=288: на 1080p эквивалент ~6.7x, поэтому
# font_size 16 — это ~90px на экране 1080x1920, а не гигантские 120px как при 22.
DEFAULT_STYLE = {
    "font_name": "Anton",
    "font_size": 16,
    "font_color": "#FFFFFF",
    "highlight_color": "#FFE500",
    "border_color": "#000000",
    "border_width": 3,
    "effect": "pop",
    "max_chars": 16,
    "max_duration": 1.4,
    "uppercase": True,
}

_HEX_RE = re.compile(r"^[0-9A-Fa-f]{6}$")
_FONT_RE = re.compile(r"[^A-Za-z0-9 _-]")


def _clamp(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        num = float(default)
    return max(lo, min(hi, num))


def _sanitize_font(name: Any) -> str:
    # Убираем всё, кроме букв/цифр/пробелов — чтобы имя шрифта не влезло в
    # force_style (запятые/скобки сломали бы разбор фильтра).
    cleaned = _FONT_RE.sub("", str(name or "")).strip()
    return cleaned or "Verdana"


def _hex_to_ass(hex_color: Any, opacity: float = 1.0, fallback: str = "FFFFFF") -> str:
    """#RRGGBB -> ASS &HAABBGGRR. Невалидный цвет не падает, а берёт fallback."""
    digits = str(hex_color or "").lstrip("#")
    if not _HEX_RE.match(digits):
        digits = fallback
    opacity = _clamp(opacity, 0.0, 1.0, 1.0)
    r = int(digits[0:2], 16)
    g = int(digits[2:4], 16)
    b = int(digits[4:6], 16)
    alpha = round((1.0 - opacity) * 255)
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def _hex_to_ass_inline(hex_color: Any, fallback: str = "FFFFFF") -> str:
    """#RRGGBB -> &HBBGGRR (инлайн-тег \\c для подсветки активного слова)."""
    digits = str(hex_color or "").lstrip("#")
    if not _HEX_RE.match(digits):
        digits = fallback
    return f"&H{digits[4:6]}{digits[2:4]}{digits[0:2]}&".upper()


def _escape_ass(text: Any) -> str:
    return str(text).replace("\\", "/").replace("{", "(").replace("}", ")")


def _ass_time(seconds: float) -> str:
    """Формат ASS H:MM:SS.cc (сотые доли)."""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        centis = 99
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _escape_filter_path(value: str) -> str:
    # Путь в подстановке ffmpeg-фильтра: слэши вперёд, двоеточие экранируем.
    return value.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def _srt_time(seconds: float) -> str:
    """SRT H:MM:SS,mmm."""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        secs += 1
        millis = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(cues: list[dict[str, Any]], output_path: Path) -> bool:
    """Пишет SRT по репликам [{start, end, text}]. Возвращает False, если пусто."""
    lines: list[str] = []
    for i, cue in enumerate(cues, start=1):
        text = str(cue.get("text") or "").strip()
        if not text:
            continue
        start = float(cue.get("start", 0))
        end = float(cue.get("end", start + 1))
        if end <= start:
            end = start + 1
        lines.append(str(i))
        lines.append(f"{_srt_time(start)} --> {_srt_time(end)}")
        lines.append(text)
        lines.append("")
    if not lines:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return True


def extract_audio_section(video_path: Path, start: float, end: float, output_path: Path) -> Path:
    """Вырезает аудио [start, end] секции в моно-WAV 16 kHz (для Whisper)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dur = max(end - start, 0.1)
    cmd = [
        get_ffmpeg_executable(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        *hwaccel_args(),
        "-i",
        str(video_path),
        "-t",
        f"{dur:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=600)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise FileNotFoundError(f"Audio section not produced from {video_path}")
    return output_path


def transcribe_section(video_path: Path, start: float, end: float) -> dict[str, Any]:
    """Транскрибирует секцию [start, end], таймкоды относительно start (0 = начало клипа)."""
    workdir = Path(tempfile.mkdtemp(prefix="clip_transcribe_"))
    try:
        wav = extract_audio_section(video_path, start, end, workdir / "section.wav")
        transcript = transcribe_audio(wav)
        # Для готовых реплик: group words в короткие блоки (правила из караоке),
        # но без караоке-разбивки по словам — это пользовательские куски.
        cues = cues_from_words(transcript.get("words") or [])
        transcript["cues"] = cues
        return transcript
    finally:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)


def cues_from_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Группирует слова в реплики [{start, end, text}] для редактора.

    Правила те же, что в _collect_blocks (max_chars/max_duration), но текст
    склеивается целиком — каждая реплика редактируется как один блок.
    """
    opts = {**DEFAULT_STYLE}
    blocks = _collect_blocks({"words": words}, int(opts["max_chars"]), float(opts["max_duration"]))
    cues: list[dict[str, Any]] = []
    for block in blocks:
        text = " ".join(w["word"] for w in block).strip()
        if not text:
            continue
        start = block[0]["start"]
        end = block[-1]["end"]
        if end <= start:
            end = start + 1
        cues.append({"start": round(start, 3), "end": round(end, 3), "text": text})
    return cues


def generate_ass(
    transcript: dict[str, Any],
    output_path: Path,
    style: dict[str, Any] | None = None,
) -> bool:
    """Собирает караоке-ASS: активное слово подсвечено, остальные затемнены.

    Одно событие Dialogue на слово (подряд, без мерцания), эффект применяется
    инлайн-тегами к активному слову, затем {\\r} сбрасывает стиль.
    """
    opts = {**DEFAULT_STYLE, **(style or {})}
    # Размер шрифта и обводку можно переопределить через env без правки кода.
    if os.getenv("CLIPPER_SUBTITLE_FONT_SIZE"):
        opts["font_size"] = float(os.environ["CLIPPER_SUBTITLE_FONT_SIZE"])
    if os.getenv("CLIPPER_SUBTITLE_BORDER"):
        opts["border_width"] = float(os.environ["CLIPPER_SUBTITLE_BORDER"])
    blocks = _collect_blocks(transcript, int(opts["max_chars"]), float(opts["max_duration"]))
    if not blocks:
        return False

    font = _sanitize_font(opts["font_name"])
    fontsize = max(10, int(_clamp(opts["font_size"], 10, 200, 44) * 0.85))
    border = max(1, int(_clamp(opts["border_width"], 0, 10, 4)))
    border_colour = _hex_to_ass(opts["border_color"], 1.0)
    primary_colour = _hex_to_ass(opts["font_color"], 1.0)
    back_colour = _hex_to_ass("#000000", 0.0)
    highlight = _hex_to_ass_inline(opts["highlight_color"])

    effect = str(opts.get("effect") or "none").lower()
    if effect == "glow":
        active_prefix = f"{{\\c&HFFFFFF&\\3c{highlight}\\bord{border + 2}\\blur4}}"
    elif effect == "box":
        active_prefix = f"{{\\c&HFFFFFF&\\3c{highlight}\\bord{border + 3}\\blur0}}"
    elif effect == "pop":
        active_prefix = f"{{\\c{highlight}\\fscx90\\fscy90\\t(0,110,\\fscx108\\fscy108)}}"
    else:
        active_prefix = f"{{\\c{highlight}}}"

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResY: 288\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{fontsize},{primary_colour},{primary_colour},"
        f"{border_colour},{back_colour},1,0,0,0,100,100,0,0,1,"
        f"{border},0,2,10,10,{SAFE_MARGIN_V},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events: list[str] = []
    for block in blocks:
        for i, word in enumerate(block):
            ev_start = block[0]["start"] if i == 0 else word["start"]
            ev_end = block[i + 1]["start"] if i < len(block) - 1 else block[-1]["end"]
            if ev_end <= ev_start:
                continue
            parts = []
            for j, other in enumerate(block):
                text = _escape_ass(other["word"])
                if opts.get("uppercase"):
                    text = text.upper()
                parts.append(f"{active_prefix}{text}{{\\r}}" if j == i else text)
            events.append(
                f"Dialogue: 0,{_ass_time(ev_start)},{_ass_time(ev_end)},"
                f"Default,,0,0,0,,{' '.join(parts)}"
            )

    if not events:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 с BOM, чтобы Windows/ffmpeg надёжно определили Unicode.
    output_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8-sig")
    return True


_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


def _load_whisper_model(model_size: str, device: str, compute_type: str) -> Any:
    """Загружает WhisperModel один раз на процесс (idempotent по конфигу).

    Модель тяжело инициализировать (особенно на GPU), а вызываем мы её и для
    generate_highlights (всё видео), и на каждый клип — кэш экономит секунды.
    faster-whisper сам поддерживает последовательные transcribe() из одного
    инстанса, т.к. в воркере транскрипция строго по одному.
    """
    from faster_whisper import WhisperModel

    key = (model_size, device, compute_type)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _MODEL_CACHE[key] = model
    return model


def transcribe_audio(video_path: Path) -> dict[str, Any]:
    """Распознаёт речь faster-whisper'ом, возвращает полный транскрипт.

    Формат на выходе (тот же контракт, что у OpenShorts):
        {"text": str, "language": str,
         "segments": [{"start": float, "end": float, "text": str,
                       "words": [{"word": str, "start": float, "end": float}]}],
         "words": [...]}   — плоский список слов, для караоке/generate_ass.

    Модель/устройство задаются через env (см. pyproject).
    """
    model_size = os.environ.get("WHISPER_MODEL", "small")
    device = os.environ.get("WHISPER_DEVICE", "cpu")
    # int8 существует только для CPU; на CUDA дефолт — float16 (rtx2080ti и
    # выше). Если WHISPER_COMPUTE задан явно — уважаем его.
    default_compute = "float16" if device.startswith("cuda") else "int8"
    compute_type = os.environ.get("WHISPER_COMPUTE", default_compute)

    model = _load_whisper_model(model_size, device, compute_type)
    segments, info = model.transcribe(
        str(video_path),
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
        word_timestamps=True,
    )
    text_parts: list[str] = []
    segments_out: list[dict[str, Any]] = []
    words_flat: list[dict[str, Any]] = []
    # faster-whisper отдаёт сегменты лениво (генератор), собираем в списки.
    for segment in segments:
        seg_words: list[dict[str, Any]] = []
        for w in segment.words or []:
            item = {"word": w.word, "start": float(w.start), "end": float(w.end)}
            seg_words.append(item)
            words_flat.append(item)
        seg_text = str(segment.text or "").strip()
        if not seg_text and not seg_words:
            continue
        text_parts.append(seg_text)
        segments_out.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": seg_text,
                "words": seg_words,
            }
        )
    return {
        "text": " ".join(text_parts).strip(),
        "language": getattr(info, "language", ""),
        "segments": segments_out,
        "words": words_flat,
    }


def _collect_blocks(
    transcript: dict[str, Any],
    max_chars: int,
    max_duration: float,
) -> list[list[dict[str, Any]]]:
    """Группирует слова в короткие блоки под вертикальный экран.

    Правила: не больше max_chars символов и не дольше max_duration секунд.
    Время уже относительно клипа (0 = начало), плюс поправка SUBTITLE_OFFSET.
    """
    flat = transcript.get("words") or []
    blocks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    block_start: float | None = None

    for raw in flat:
        word = str(raw.get("word") or "").strip()
        if not word:
            continue
        start = max(0.0, float(raw.get("start", 0)) + SUBTITLE_OFFSET)
        end = max(start, float(raw.get("end", 0)) + SUBTITLE_OFFSET)
        item = {"word": word, "start": start, "end": end}

        if not current:
            current = [item]
            block_start = start
            continue

        text_len = sum(len(w["word"]) + 1 for w in current)
        duration = end - (block_start or start)
        if text_len + len(word) > max_chars or duration > max_duration:
            blocks.append(current)
            current = [item]
            block_start = start
        else:
            current.append(item)

    if current:
        blocks.append(current)
    return blocks


def burn_subtitles(video_path: Path, ass_path: Path, output_path: Path) -> Path:
    """Вжигает ASS в клип через ffmpeg/libass. Возвращает путь к результату.

    Если output_path совпадает с video_path (in-place, как в cli.py) — сначала
    пишем во временный файл рядом, а потом атомарно заменяем оригинал
    (ffmpeg отказывается писать в тот же файл, что читает: "same as Input")."""
    import tempfile
    import os

    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    in_place = os.path.normcase(str(output_path.resolve())) == os.path.normcase(str(video_path.resolve()))
    if in_place:
        fd, tmp_name = tempfile.mkstemp(prefix=".burn_", suffix=".mp4", dir=str(output_path.parent))
        os.close(fd)
        os.unlink(tmp_name)
        tmp_path = output_path.parent / os.path.basename(tmp_name)
        render_target = tmp_path
    else:
        tmp_path = None
        render_target = output_path

    vf = f"ass=filename='{_escape_filter_path(str(ass_path))}'"
    cmd = [
        get_ffmpeg_executable(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        *hwaccel_args(),
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-c:a",
        "copy",
        *video_encode_args(),
        *METADATA_SCRUB,
        "-use_editlist", "0",
        "-movflags",
        "+faststart",
        str(render_target),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        stderr_text = result.stderr.decode(errors="replace")
        raise RuntimeError(f"FFmpeg subtitle burn failed: {stderr_text[:500]}")
    if in_place:
        os.replace(tmp_path, output_path)
    return output_path


def generate_ass_cues(
    cues: list[dict[str, Any]],
    output_path: Path,
    style: dict[str, Any] | None = None,
    play_w: int = 1080,
    play_h: int = 1920,
) -> bool:
    """ASS по пользовательским репликам [{start, end, text, x?, y?, scale?}].

    В отличие от караоке-generate_ass — это простой статичный стиль (без
    подсветки активного слова), где каждая реплика — отдельный Dialogue с
    инлайн-позицией ``\\pos(x,y)``. ``x``/``y`` — нормализованные (0..1)
    координаты ЦЕНТРА блока, масштабируются в пиксели PlayRes. ``scale`` —
    множитель размера шрифта (1.0 = базовый).
    """
    opts = {**DEFAULT_STYLE, **(style or {})}
    if os.getenv("CLIPPER_SUBTITLE_FONT_SIZE"):
        opts["font_size"] = float(os.environ["CLIPPER_SUBTITLE_FONT_SIZE"])
    if os.getenv("CLIPPER_SUBTITLE_BORDER"):
        opts["border_width"] = float(os.environ["CLIPPER_SUBTITLE_BORDER"])

    font = _sanitize_font(opts["font_name"])
    base_fontsize = int(_clamp(opts["font_size"], 10, 400, 90))
    border = max(0, int(_clamp(opts["border_width"], 0, 20, 4)))
    border_colour = _hex_to_ass(opts["border_color"], 1.0)
    primary_colour = _hex_to_ass(opts["font_color"], 1.0)
    back_colour = _hex_to_ass("#000000", 0.0)

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_w}\n"
        f"PlayResY: {play_h}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{base_fontsize},{primary_colour},{primary_colour},"
        f"{border_colour},{back_colour},1,0,0,0,100,100,0,0,1,"
        f"{border},0,5,10,10,40,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events: list[str] = []
    for cue in cues:
        text = str(cue.get("text") or "").strip()
        if not text:
            continue
        start = float(cue.get("start", 0))
        end = float(cue.get("end", start + 1))
        if end <= start:
            end = start + 1
        x = _clamp(cue.get("x", 0.5), 0.0, 1.0, 0.5)
        y = _clamp(cue.get("y", 0.88), 0.0, 1.0, 0.88)
        scale = _clamp(cue.get("scale", 1.0), 0.3, 3.0, 1.0)
        fontsize = int(round(base_fontsize * scale))
        pos_x = int(round(x * play_w))
        pos_y = int(round(y * play_h))
        escaped = _escape_ass(text)
        if opts.get("uppercase"):
            escaped = escaped.upper()
        events.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},"
            f"Default,,0,0,0,,{{\\an5\\pos({pos_x},{pos_y})"
            f"\\fs{fontsize}\\bord{border}}}{escaped}"
        )

    if not events:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8-sig")
    return True


def generate_and_burn_subtitles(
    video_path: Path,
    output_path: Path,
    cues: list[dict[str, Any]] | None = None,
    style: dict[str, Any] | None = None,
) -> Path:
    """Транскрибирует клип и вжигает караоке-субтитры (или по готовым cues).

    Используется cli.py (режим ``subtitles=True``) и редактором. Возвращает
    путь к видео с вожжёнными субтитрами.
    """
    import tempfile

    video_path = Path(video_path)
    output_path = Path(output_path)

    if cues is None:
        transcript = transcribe_audio(video_path)
        cues = transcript.get("cues") or cues_from_words(transcript.get("words") or [])
    workdir = Path(tempfile.mkdtemp(prefix="subs_burn_"))
    try:
        ass_path = workdir / "subs.ass"
        if not generate_ass_cues(cues, ass_path, style=style):
            return video_path
        return burn_subtitles(video_path, ass_path, output_path)
    finally:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)

