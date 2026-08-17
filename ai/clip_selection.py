"""Чистые хелперы пайплайна выбора клипов через Gemini.

Только стандартная библиотека: логика остаётся юнит-тестируемой без тяжёлых
видео-зависимостей и SDK. Портировано из OpenShorts (clip_selection.py).

Транскрипт-контракт, на котором всё строится:
    {"text": str, "language": str,
     "segments": [{"start": float, "end": float, "text": str,
                   "words": [{"word": str, "start": float, "end": float}]}]}
"""

import os
from typing import Any

# USD за 1M токенов (input, output вместе с thinking), цены ai.google.dev.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),  # deprecated (shut down 2026-06-01)
}


def lookup_model_prices(model_name: str | None) -> tuple[float, float] | None:
    """Longest-prefix поиск цены в MODEL_PRICES; None если модель неизвестна."""
    name = str(model_name or "").lower()
    best_key: str | None = None
    for key in MODEL_PRICES:
        if name.startswith(key) and (best_key is None or len(key) > len(best_key)):
            best_key = key
    return MODEL_PRICES[best_key] if best_key else None


def clip_count_targets(n_windows: int) -> tuple[int, int]:
    """Сколько клипов просить у detail-прохода при данном размере шортлиста.

    Измерено на проде 3-ago-2026: 408 из 429 задач (95%) отдавали 3 клипа и
    меньше, мода — ОДИН, тогда как промпт был волен вернуть по клипу на окно.
    Пользователи с 1-3 клипами возвращались на второй день в 0.4% случаев,
    с 4-9 — в 16.1%, так что именно число клипов, а не их качество, держит
    кривую удержания.

    Старый промпт сильно давил в другую сторону («лучше один отличный клип на
    окно») и давал модели два безграничных разрешения выкидывать клипы
    (2-second rule и STANDS ALONE — оба заканчиваются на «or skip it»), без
    нижней границы — и оно схлопывалось в один клип. Здесь вместо этого есть
    и нижняя граница, и реалистичный потолок.

    ``CLIP_TARGET_MIN`` / ``CLIP_TARGET_MAX`` перекрывают обе границы для
    A/B-прогонов без деплоя.
    """
    n = max(1, int(n_windows or 1))
    # Нижняя граница растёт вместе с материалом: 3 окна -> 3, 5 -> 4, 10+ -> 6.
    low = max(2, min(6, n // 2 + 2))
    # Потолок разрешает богатому окну дать больше одного клипа без «набивки».
    high = min(12, max(4, n * 2))
    low = min(low, high)

    def _override(name: str, current: int) -> int:
        raw = os.environ.get(name)
        if not raw:
            return current
        try:
            return max(1, int(raw))
        except ValueError:
            return current

    low = _override("CLIP_TARGET_MIN", low)
    high = _override("CLIP_TARGET_MAX", high)
    return low, max(low, high)


def compact_words(words: list[dict[str, Any]], precision: int = 2) -> list[dict[str, Any]]:
    """Округляет таймкоды слов для промптов — полная точность float жжёт токены."""
    return [
        {
            "w": w.get("w", ""),
            "s": round(float(w.get("s", 0)), precision),
            "e": round(float(w.get("e", 0)), precision),
        }
        for w in words
    ]


def build_transcript_windows(
    transcript_result: dict[str, Any],
    video_duration: float,
    window_seconds: float = 90,
    overlap_seconds: float = 30,
) -> list[dict[str, Any]]:
    """Строит окна скоринга по границам Whisper-сегментов.

    Предложение (а обычно и виральный момент) никогда не режется пополам
    окном: окно растёт по сегментам до ~window_seconds (до 1.25x для
    завершающего сегмента), а следующее окно начинается на ~overlap_seconds
    раньше конца предыдущего, тоже по началу сегмента.
    """
    segments: list[tuple[float, float, str]] = []
    for segment in transcript_result.get("segments", []):
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        segments.append((float(segment.get("start", 0)), float(segment.get("end", 0)), text))

    windows: list[dict[str, Any]] = []
    window_index = 1
    i = 0
    n = len(segments)
    while i < n:
        w_start = segments[i][0]
        j = i
        # Растём, пока следующий сегмент влезает в толерантный лимит — окно
        # закрывается на границе сегмента вблизи window_seconds.
        while j + 1 < n and segments[j + 1][1] - w_start <= window_seconds * 1.25:
            j += 1
            if segments[j][1] - w_start >= window_seconds:
                break
        w_end = segments[j][1]
        windows.append(
            {
                "id": f"window_{window_index:03d}",
                "start": round(w_start, 3),
                "end": round(w_end, 3),
                "text": " ".join(seg[2] for seg in segments[i : j + 1]),
            }
        )
        window_index += 1

        if j >= n - 1:
            break
        # Следующее окно начинается с первого сегмента после (end - overlap),
        # но всегда с прогрессом.
        target = w_end - overlap_seconds
        k = i + 1
        while k <= j and segments[k][0] < target:
            k += 1
        i = max(k, i + 1)

    if not windows:
        windows.append(
            {
                "id": "window_001",
                "start": 0.0,
                "end": round(float(video_duration), 3),
                "text": str(transcript_result.get("text", "") or ""),
            }
        )
    return windows


def snap_clip_to_words(
    start: float,
    end: float,
    words: list[dict[str, Any]],
    video_duration: float,
    min_duration: float = 15.0,
    max_duration: float = 60.0,
    search_window: float = 1.5,
    max_lead: float = 0.35,
    max_tail: float = 0.45,
) -> tuple[float, float]:
    """Привязывает предложенные Gemini границы к реальным границам слов.

    Плюс кусочек окружающей тишины. LLM плохи в миллисекундной арифметике;
    таймкоды слов — эталонная истина, поэтому резы попадают в паузы, а не
    в середину слова.

    words: [{'w','s','e'}, ...] по всему видео, отсортированы по start.
    Возвращает (start, end); откатывается к исходному, если рядом нет слов
    или привязка не укладывается в границы длительности.
    """
    original = (round(float(start), 3), round(float(end), 3))
    if not words:
        return original

    starts = [float(w.get("s", 0)) for w in words]
    ends = [float(w.get("e", 0)) for w in words]

    # START: привязка к ближайшему началу слова, затем отступ в тишину перед ним.
    new_start = float(start)
    candidates = [s for s in starts if abs(s - new_start) <= search_window]
    if candidates:
        word_start = min(candidates, key=lambda s: abs(s - new_start))
        prev_ends = [e for e in ends if e <= word_start]
        if prev_ends:
            gap = max(0.0, word_start - max(prev_ends))
            lead = min(max_lead, gap / 2)
        else:
            lead = max_lead
        new_start = max(0.0, word_start - lead)

    # END: привязка к ближайшему концу слова, затем хвост в тишину после него.
    new_end = float(end)
    candidates = [e for e in ends if abs(e - new_end) <= search_window]
    if candidates:
        word_end = min(candidates, key=lambda e: abs(e - new_end))
        next_starts = [s for s in starts if s >= word_end]
        if next_starts:
            gap = max(0.0, min(next_starts) - word_end)
            tail = min(max_tail, gap / 2)
        else:
            tail = max_tail
        new_end = min(float(video_duration), word_end + tail)

    # Чиним границы длительности, оставаясь на границах слов.
    if new_end - new_start < min_duration:
        target = new_start + min_duration
        later = sorted(e for e in ends if e >= target)
        if later and later[0] - new_start <= max_duration:
            new_end = min(float(video_duration), later[0] + 0.2)
        else:
            return original
    if new_end - new_start > max_duration:
        target = new_start + max_duration
        earlier = [e for e in ends if new_start < e <= target]
        new_end = (max(earlier) + 0.2) if earlier else target
        new_end = min(new_end, new_start + max_duration, float(video_duration))

    if new_end <= new_start or new_end - new_start < min_duration:
        return original
    return (round(new_start, 3), round(new_end, 3))

