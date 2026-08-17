"""Выбор клипов (highlights) через Gemini — двухпроходный пайплайн.

Если задача пришла без готовой разметки (новое видео по URL/файлу), воркер сам
находит, что нарезать: Whisper-транскрипция -> окна -> скоринг Gemini ->
детализация лучших окон в готовые клипы. Логика портирована из OpenShorts
(main.get_viral_clips + gemini_worker) и использует её же хелперы
(worker.clip_selection).

Формат на выходе совпадает с тем, что ждёт бекенд (start/end — "mm:ss"):
    {"start": str, "end": str, "title": str, "reason": str,
     "hook_sentence": str, "virality_score": int, "virality_prediction": str}
"""

import logging
import os
from pathlib import Path
from typing import Any

from engine.ai.clip_selection import (
    build_transcript_windows,
    clip_count_targets,
    snap_clip_to_words,
)
from engine.media.subtitles import transcribe_audio

logger = logging.getLogger("Clipper.Highlights")

# Группа окон для одного score-запроса: дешевый проход должен держать промпт
# коротким, чтобы дорогой detail-разбор оставался сфокусированным на коротком списке.
SCORE_BATCH = 8


def _mmss(seconds: float) -> str:
    """Секунды -> "mm:ss" (как хранит бекенд в startTs/endTs)."""
    seconds = max(0, int(round(seconds)))
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def _flat_words(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """Слова по всему видео в формате clip_selection: [{'w','s','e'}, ...]."""
    words: list[dict[str, Any]] = []
    for segment in transcript.get("segments") or []:
        for w in segment.get("words") or []:
            words.append(
                {
                    "w": str(w.get("word") or ""),
                    "s": float(w.get("start", 0)),
                    "e": float(w.get("end", 0)),
                }
            )
    return words


def _to_backend_format(short: dict[str, Any]) -> dict[str, Any]:
    """Мапит клип openshorts в формат бекенда (HighlightDto)."""
    return {
        "start": _mmss(float(short.get("start", 0))),
        "end": _mmss(float(short.get("end", 0))),
        "title": str(short.get("video_title_for_youtube_short") or "").strip(),
        "reason": str(short.get("source_window_id") or "gemini").strip(),
        "hook_sentence": str(short.get("viral_hook_text") or "").strip(),
        "virality_score": int(short.get("predicted_score", 0) or 0),
        "virality_prediction": "gemini",
    }


def _score_windows(
    client: Any,
    model_name: str,
    windows: list[dict[str, Any]],
    language: str,
    video_duration: float,
) -> list[dict[str, Any]]:
    """Проход 1: скоринг окон батчами, собирает лучшие кандидаты."""
    import json

    from engine.ai import gemini_worker

    scored: list[dict[str, Any]] = []
    for b in range(0, len(windows), SCORE_BATCH):
        batch = windows[b : b + SCORE_BATCH]
        payload = [
            {"id": w["id"], "start": w["start"], "end": w["end"], "text": w["text"]}
            for w in batch
        ]
        prompt = gemini_worker.SCORE_PROMPT_TEMPLATE.format(
            video_duration=video_duration,
            language=language,
            windows_json=json.dumps(payload, ensure_ascii=False),
        )
        parsed, _ = gemini_worker.run_gemini_stage(
            client, model_name, prompt, gemini_worker.ScoreResponse
        )
        scored.extend(parsed.get("windows") or [])
    return scored


def _detail_shorts(
    client: Any,
    model_name: str,
    shortlist: list[dict[str, Any]],
    language: str,
    video_duration: float,
) -> list[dict[str, Any]]:
    """Проход 2: детализация короткого списка окон в готовые клипы."""
    import json

    from engine.ai import gemini_worker

    payload = [
        {"id": w["id"], "start": w["start"], "end": w["end"], "text": w["text"]}
        for w in shortlist
    ]
    min_clips, max_clips = clip_count_targets(len(shortlist))
    prompt = gemini_worker.DETAIL_PROMPT_TEMPLATE.format(
        video_duration=video_duration,
        language=language,
        min_clips=min_clips,
        max_clips=max_clips,
        windows_json=json.dumps(payload, ensure_ascii=False),
    )
    parsed, _ = gemini_worker.run_gemini_stage(
        client, model_name, prompt, gemini_worker.DetailResponse
    )
    return parsed.get("shorts") or []


def generate_highlights(
    video_path: Path,
    clips_count: int,
    total_duration: float,
) -> list[dict[str, Any]]:
    """Выбирает highlights для задачи через Gemini (2-pass: score -> detail).

    Пустой список — если нет речи, нет GEMINI_API_KEY или Gemini ничего не
    нашёл; рендер такой задачи заканчивается без клипов.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("[HIGHLIGHTS] GEMINI_API_KEY is not set, cannot generate highlights.")
        return []

    transcript = transcribe_audio(video_path)
    words = _flat_words(transcript)
    if not words:
        logger.warning("[HIGHLIGHTS] no speech detected, no highlights generated.")
        return []

    duration = float(total_duration or 0)
    if duration <= 0:
        segments = transcript.get("segments") or []
        duration = float(segments[-1]["end"]) if segments else 60.0

    language = str(transcript.get("language") or "unknown")
    model_name = os.getenv("GEMINI_MODEL") or "gemini-3.1-flash-lite"
    logger.info(f"[HIGHLIGHTS] Gemini model={model_name}, language={language}, "
                f"duration={duration:.0f}s.")


    from engine.ai import gemini_worker
    from engine.ai.gemini_client import make_gemini_client

    try:
        client = make_gemini_client(api_key)
        windows = build_transcript_windows(transcript, duration)
        if not windows:
            logger.warning("[HIGHLIGHTS] no transcript windows, no highlights generated.")
            return []
        logger.info(f"[HIGHLIGHTS] built {len(windows)} scoring window(s).")

        scored = _score_windows(client, model_name, windows, language, duration)

        # Короткий список: топ окон с масштабированием по длительности, чтобы
        # длинные видео давали больше кандидатов без взрыва detail-запроса.
        scored.sort(key=lambda w: w.get("score", 0), reverse=True)
        target = max(3, min(10, int(duration // 90) + 2))
        by_id = {w["id"]: w for w in windows}
        shortlist = [by_id[w["id"]] for w in scored[:target] if w.get("id") in by_id]
        if not shortlist:
            shortlist = windows[:target]
        logger.info(f"[HIGHLIGHTS] shortlisted {len(shortlist)} window(s) for detail.")

        shorts = _detail_shorts(client, model_name, shortlist, language, duration)

        # Привязываем границы к реальным границам слов (+ кусочек тишины).
        for s in shorts:
            ns, ne = snap_clip_to_words(
                float(s.get("start", 0)), float(s.get("end", 0)), words, duration
            )
            s["start"], s["end"] = ns, ne
    except gemini_worker.GeminiBlockedError as e:
        # Policy-отказ: поднимаем как есть — задача должна упасть с настоящей
        # причиной, а не с общим «клипы не найдены».
        logger.error(f"[HIGHLIGHTS] Gemini blocked: {e}")
        raise
    except Exception as e:
        logger.error(f"[HIGHLIGHTS] Gemini analysis failed: {e}")
        return []

    if not shorts:
        logger.warning("[HIGHLIGHTS] Gemini returned no clips.")
        return []

    # Сортируем по предсказанному виральному потенциалу, режем до запрошенного
    # числа клипов.
    shorts.sort(key=lambda s: s.get("predicted_score", 0) or 0, reverse=True)
    highlights = [_to_backend_format(s) for s in shorts[: max(1, int(clips_count))]]
    logger.info(f"[HIGHLIGHTS] generated {len(highlights)} highlight(s) via Gemini.")
    return highlights

