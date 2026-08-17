"""FFmpeg/ffprobe helpers and encoder selection — единый ffmpeg-модуль воркера.

Находит бинарники ffmpeg/ffprobe (bundled ``bin/``, системные, imageio_ffmpeg),
выбирает видеоэнкодер (tiers libx264 / h264_nvenc) и держит общие аргументы
кодирования (METADATA_SCRUB, нормализация громкости, hwaccel), плюс локальные
нарезку и пробинг для пайплайна. Загрузка по URL живёт в ``worker.media.ingest``
(yt-dlp), не здесь.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger("Clipper.FFmpeg")


def _repo_bin_dir() -> str | None:
    bin_dir = Path(__file__).resolve().parent.parent / "bin"
    exe = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    return str(bin_dir) if (bin_dir / exe).exists() else None


def get_ffmpeg_executable() -> str:
    if sys.platform != "win32":
        system = shutil.which("ffmpeg")
        if system:
            return system
    d = _repo_bin_dir()
    if d:
        return os.path.join(d, "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    try:
        import imageio_ffmpeg

        return cast(str, imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    return "ffmpeg"


def _repo_probe_dir() -> str | None:
    bin_dir = Path(__file__).resolve().parent.parent / "bin"
    exe = "ffprobe.exe" if sys.platform == "win32" else "ffprobe"
    return str(bin_dir) if (bin_dir / exe).exists() else None


def get_ffprobe_executable() -> str:
    """Resolve ffprobe next to ffmpeg (system, bundled bin/, or bare name)."""
    if sys.platform != "win32":
        system = shutil.which("ffprobe")
        if system:
            return system
    d = _repo_probe_dir()
    if d:
        return os.path.join(d, "ffprobe.exe" if sys.platform == "win32" else "ffprobe")
    return "ffprobe"


# --- Encoder selection ---

# Quality tiers pinning the historical libx264 settings.
QUALITY = "quality"  # was: -preset medium -crf 18
QUALITY_FAST = "quality_fast"  # was: -preset fast -crf 18
DELIVERY = "delivery"  # was: -preset fast -crf 22

_X264_ARGS = {
    QUALITY: ["-c:v", "libx264", "-preset", "medium", "-crf", "18"],
    QUALITY_FAST: ["-c:v", "libx264", "-preset", "fast", "-crf", "18"],
    DELIVERY: ["-c:v", "libx264", "-preset", "fast", "-crf", "22"],
}

# NVENC -cq is not 1:1 with x264 CRF: benchmarked on the prod GPU (RTX 4000
# Ada), cq ≈ crf + 7 lands in the same file-size ballpark, with vbr + AQ for
# quality. Presets p1-p7: p5 ≈ "medium", p4 ≈ "fast".
# Lowered vs the original 25/25/29 after a complaint about soft/blocky output:
# the 9:16 pipeline upscales the crop (720->1080 wide), so a higher bit budget
# is worth it. cq 19 is visibly cleaner than 25 on that upscale.
# -pix_fmt yuv420p is REQUIRED: with RGB input (the bgr24 rawvideo pipe from
# OpenCV) nvenc otherwise emits H.264 in gbrp/GBR colorspace, which ffmpeg
# reads fine but web players render as a magenta/green mess.
_NVENC_ARGS = {
    QUALITY: [
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p5",
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        "19",
        "-b:v",
        "0",
        "-spatial-aq",
        "1",
        "-temporal-aq",
        "1",
        "-pix_fmt",
        "yuv420p",
    ],
    QUALITY_FAST: [
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        "21",
        "-b:v",
        "0",
        "-spatial-aq",
        "1",
        "-pix_fmt",
        "yuv420p",
    ],
    DELIVERY: [
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-rc",
        "vbr",
        "-cq",
        "24",
        "-b:v",
        "0",
        "-spatial-aq",
        "1",
        "-pix_fmt",
        "yuv420p",
    ],
}

# Encoder fallback cascade (Medal-style: av1_nvenc -> av1_qsv -> av1_amf ->
# h264_nvenc -> h264_qsv -> h264_amf -> libx264). The first encoder that
# survives a real short encode probe wins; libx264 is the guaranteed last resort.
# Kept as name -> per-tier args so quality mapping stays per-vendor.
_AV1_NVENC_ARGS = {
    QUALITY: ["-c:v", "av1_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "26", "-b:v", "0", "-pix_fmt", "yuv420p"],
    QUALITY_FAST: ["-c:v", "av1_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "28", "-b:v", "0", "-pix_fmt", "yuv420p"],
    DELIVERY: ["-c:v", "av1_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "30", "-b:v", "0", "-pix_fmt", "yuv420p"],
}

_QSV_ARGS = {
    QUALITY: ["-c:v", "h264_qsv", "-preset", "medium", "-global_quality", "22", "-pix_fmt", "yuv420p"],
    QUALITY_FAST: ["-c:v", "h264_qsv", "-preset", "fast", "-global_quality", "24", "-pix_fmt", "yuv420p"],
    DELIVERY: ["-c:v", "h264_qsv", "-preset", "fast", "-global_quality", "26", "-pix_fmt", "yuv420p"],
}

_AMF_ARGS = {
    QUALITY: ["-c:v", "h264_amf", "-usage", "transcoding", "-quality", "quality", "-rc", "cqp", "-qp_i", "22", "-qp_p", "22", "-pix_fmt", "yuv420p"],
    QUALITY_FAST: ["-c:v", "h264_amf", "-usage", "transcoding", "-quality", "speed", "-rc", "cqp", "-qp_i", "24", "-qp_p", "24", "-pix_fmt", "yuv420p"],
    DELIVERY: ["-c:v", "h264_amf", "-usage", "transcoding", "-quality", "speed", "-rc", "cqp", "-qp_i", "26", "-qp_p", "26", "-pix_fmt", "yuv420p"],
}

# Encoder cascade in priority order. Each entry: (key, args_map, probe_codec).
# The probe_codec is what a tiny lavfi encode is run with (some encoders need
# full names / HW init). libx264 always works -> guaranteed fallback.
_ENCODER_CASCADE = [
    ("av1_nvenc", _AV1_NVENC_ARGS, "av1_nvenc"),
    ("h264_nvenc", _NVENC_ARGS, "h264_nvenc"),
    ("h264_qsv", _QSV_ARGS, "h264_qsv"),
    ("h264_amf", _AMF_ARGS, "h264_amf"),
    ("libx264", _X264_ARGS, "libx264"),
]
_DEFAULT_CASCADE_KEYS = [k for k, _, _ in _ENCODER_CASCADE]

# Output args that drop container/stream metadata carried over from the source
# — most notably YouTube's "produced by Google Inc." stream handler, which
# otherwise survives every re-encode (ffmpeg copies input metadata by default)
# and rides into the published clip. The per-stream specifiers are required:
# global -map_metadata -1 alone leaves the audio handler_name intact on a
# stream copy. Empty audio/video specifiers are harmless when a clip lacks that
# stream (ffmpeg ignores them, verified). Spliced in before the output filename
# at each final-artifact producer; kept out of video_encode_args() so that
# stays purely codec/quality args.
METADATA_SCRUB = [
    "-map_metadata",
    "-1",
    "-map_chapters",
    "-1",
    "-map_metadata:s:v",
    "-1",
    "-map_metadata:s:a",
    "-1",
]

# Loudness normalisation for the delivered clip.
#
# Without this the clip inherits whatever the source was mastered at, so a
# user's clips land anywhere: measured across real delivered clips on
# 26-jul-2026, from -13.8 LUFS on a loud upload down to -28 LUFS on a quiet
# talk. TikTok, Reels and Shorts all normalise playback to roughly -14 LUFS,
# which means the quiet ones just sound thin next to everything else in the
# feed — the loud ones aren't rewarded, the quiet ones are punished.
#
# I=-14 matches the platforms' target, LRA=11 is the usual allowance for speech.
# Applied at the clip cut, where the audio is being encoded to AAC anyway, so it
# costs nothing extra. AUDIO_NORMALIZE=0 turns it off.
#
# TP=-2.0, not the -1.5 that matches the platforms' own advice, because the
# ceiling is enforced BEFORE the AAC encode and the encoder then adds
# inter-sample peaks on top. Measured over 14 corpus clips (31-jul-2026):
#
#   TP=-1.5   peak reached +0.2 dBTP, 1 clip clipping,  8 above -1.0
#   TP=-2.0   peak reached -0.3 dBTP, 0 clipping,       5 above -1.0
#   TP=-3.0   peak reached -0.9 dBTP, 0 clipping,       1 above -1.0
#
# -3.0 also costs level: only 8 of 14 stayed inside -15..-13 LUFS versus 12 at
# -2.0, and level is what the listener notices. Two other fixes were tried and
# do NOT work, so don't reach for them again: an `alimiter` after loudnorm
# (limits sample peaks, not inter-sample, and measured WORSE at +0.7), and
# two-pass loudnorm with linear=true (+0.4, still clipping). The overshoot is
# the codec's, so the only lever is headroom.
LOUDNORM_FILTER = "loudnorm=I=-14:TP=-2.0:LRA=11"


def audio_encode_args():
    """AAC encode args for a delivered clip, with loudness normalisation."""
    args = ["-c:a", "aac"]
    if os.environ.get("AUDIO_NORMALIZE", "1").strip() != "0":
        args = ["-af", LOUDNORM_FILTER] + args
    return args


_probe_lock = threading.Lock()
_encoder_probe_cache: dict[str, bool] = {}
_known_encoders: set[str] | None = None
_picked_encoder = None  # None = not decided yet
_announced = False


def _probe_encoder(codec: str) -> bool:
    """One tiny lavfi encode to prove an encoder works end-to-end.

    NVENC rejects frames smaller than ~145px, so the probe uses 256x256.
    Any failure (no ffmpeg binary, no GPU, no driver libs) means False.
    """
    cmd = [
        get_ffmpeg_executable(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=black:s=256x256:d=0.1",
        "-c:v",
        codec,
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15
        )
        return result.returncode == 0
    except Exception:
        return False


def _buildin_encoder_list() -> set[str] | None:
    """Fast pre-filter: codec names ffmpeg reports in `-encoders`.

    Returns None when listing failed (then we fall back to real probing).
    """
    global _known_encoders
    if _known_encoders is not None:
        return _known_encoders
    try:
        r = subprocess.run(
            [get_ffmpeg_executable(), "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return None
    if r.returncode != 0:
        return None
    names: set[str] = set()
    for line in (r.stdout or "").splitlines():
        parts = line.split()
        # Lines look like: " V..... h264_nvenc ..." (name is the 3rd token)
        if len(parts) >= 3 and parts[2] and parts[2][0].isalnum():
            names.add(parts[2])
    _known_encoders = names
    return names


def encoder_available(codec: str) -> bool:
    """Probe an encoder once and cache the verdict (thread-safe).

    Cheap gate first: if ffmpeg's own `-encoders` list does not contain the
    codec, skip the real (costly) encode probe entirely.
    """
    global _encoder_probe_cache
    cached = _encoder_probe_cache.get(codec)
    if cached is not None:
        return cached
    known = _buildin_encoder_list()
    if known is not None and codec not in known:
        _encoder_probe_cache[codec] = False
        return False
    with _probe_lock:
        cached = _encoder_probe_cache.get(codec)
        if cached is not None:
            return cached
        ok = _probe_encoder(codec)
        _encoder_probe_cache[codec] = ok
        logger.info("ffmpeg encoder probe: %s -> %s", codec, "ok" if ok else "unavailable")
        return ok


def nvenc_available():
    """Backwards-compat: is h264_nvenc usable?"""
    return encoder_available("h264_nvenc")


def reset_encoder_cache():
    """Test hook: forget the cached probe results."""
    global _encoder_probe_cache, _known_encoders, _picked_encoder, _announced
    with _probe_lock:
        _encoder_probe_cache = {}
        _known_encoders = None
        _picked_encoder = None
        _announced = False


def _resolve_encoder(mode: str) -> str | None:
    """Pick the encoder key for a mode.

    - "x264": force libx264 (historical default)
    - "nvenc": h264_nvenc if usable else libx264
    - "auto" (default): first usable encoder in the cascade
    """
    if mode == "x264":
        return "libx264"
    if mode == "nvenc":
        return "h264_nvenc" if encoder_available("h264_nvenc") else "libx264"
    # auto: walk the priority cascade, libx264 is the guaranteed last resort
    for key, _, probe in _ENCODER_CASCADE:
        if key == "libx264" or encoder_available(probe):
            return key
    return "libx264"


def video_encode_args(tier=QUALITY):
    """Return the codec/quality args for one encode, honoring FFMPEG_ENCODER.

    Mode is x264 | nvenc | auto. `auto` walks the hardware cascade
    (av1_nvenc -> h264_nvenc -> h264_qsv -> h264_amf -> libx264), probing
    each encoder with a real short encode (Medal-style).
    """
    global _picked_encoder, _announced
    if tier not in _X264_ARGS and tier not in _NVENC_ARGS:
        raise ValueError(f"Unknown encode tier: {tier!r}")

    mode = os.environ.get("FFMPEG_ENCODER", "auto").strip().lower()

    if _picked_encoder is None:
        with _probe_lock:
            if _picked_encoder is None:
                picked = _resolve_encoder(mode)
                _picked_encoder = picked
                if mode == "nvenc" and picked == "libx264":
                    print(
                        "⚠️ [Encoder] FFMPEG_ENCODER=nvenc but h264_nvenc is not "
                        "usable here — falling back to libx264"
                    )

    if not _announced:
        _announced = True
        print(
            f"🎞️ [Encoder] video encoder: {_picked_encoder} "
            f"(FFMPEG_ENCODER={mode})"
        )

    args_map = {key: args for key, args, _ in _ENCODER_CASCADE}
    return list(args_map[_picked_encoder][tier])


def hwaccel_args() -> list[str]:
    """Decode acceleration for ffmpeg input, gated by FFMPEG_HWACCEL.

    ffmpeg automatically falls back to software decoding for codecs NVDEC
    cannot handle, so enabling it is safe even on mixed sources.
    """
    mode = os.environ.get("FFMPEG_HWACCEL", "").strip().lower()
    return ["-hwaccel", mode] if mode else []


# Pad (s) around a section so cuts land cleanly; the final trim removes it.
SECTION_PAD_SECONDS = 2

# Watchdog: if no new out_time progress for this many seconds, kill ffmpeg
# and retry once (Medal CaptureStuckChecker equivalent).
WATCHDOG_STALL_SECONDS = float(os.environ.get("FFMPEG_WATCHDOG_STALL", "60"))
WATCHDOG_MAX_RETRIES = 1

# Minimum free disk space (GB) before a render is allowed to start (B5).
MIN_FREE_DISK_GB = float(os.environ.get("FFMPEG_MIN_FREE_GB", "10"))


def run_ffmpeg_progress(
    cmd: list[str],
    *,
    stall_seconds: float = WATCHDOG_STALL_SECONDS,
    on_progress: Any = None,
    timeout: int = 3600,
) -> int:
    """Run ffmpeg while watching -progress output; kill/retry if it stalls.

    Requires ffmpeg args to include ``-progress pipe:1 -nostats`` so progress
    (out_time_us=...) lands on stdout. Returns the final returncode. On a
    watchdog stall it terminates the process and returns -1 so the caller can
    retry; the caller is expected to drop the partial output.
    """
    cmd = list(cmd)
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    last_activity = time.monotonic()
    deadline = time.monotonic() + timeout
    try:
        assert p.stdout is not None
        for raw in p.stdout:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("out_time_us="):
                last_activity = time.monotonic()
                try:
                    out_us = int(line.split("=", 1)[1])
                except ValueError:
                    out_us = 0
                if on_progress is not None:
                    on_progress(out_us / 1_000_000.0)
            if time.monotonic() - last_activity > stall_seconds:
                logger.warning(
                    "ffmpeg stalled (no progress for %.0fs) — killing",
                    stall_seconds,
                )
                p.kill()
                p.wait(timeout=15)
                return -1
            if time.monotonic() > deadline:
                logger.warning("ffmpeg exceeded total timeout (%ss) — killing", timeout)
                p.kill()
                p.wait(timeout=15)
                return -1
    finally:
        try:
            p.stdout.close()
        except Exception:
            pass
    rc = p.wait(timeout=15)
    return rc


def video_stream_codecs(path: str | Path) -> dict[str, str]:
    """Return {video: codec_name, audio: codec_name} for a local file.

    Used to decide whether a stream copy (-c copy) is safe: h264 video + aac
    audio (or no audio) can be cut losslessly instead of re-encoding (B1).
    """
    path = Path(path)
    probe = get_ffprobe_executable()
    r = subprocess.run(
        [
            probe,
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    try:
        data = json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        return {}
    video = audio = None
    for s in data.get("streams") or []:
        if s.get("codec_type") == "video" and not video:
            video = s.get("codec_name")
        elif s.get("codec_type") == "audio" and not audio:
            audio = s.get("codec_name")
    return {"video": video, "audio": audio}


def can_stream_copy(path: str | Path) -> bool:
    """True if the source can be cut with -c copy (h264 + aac/absent audio)."""
    codecs = video_stream_codecs(path)
    if codecs.get("video") not in ("h264",):
        return False
    audio = codecs.get("audio")
    return audio is None or audio == "aac"


def cut_local_section(
    file_path: str | Path,
    out_dir: str | Path,
    start_sec: int,
    end_sec: int,
    idx: int = 0,
    extra_end_sec: float = 0.0,
) -> Path:
    """Cut a [start_sec, end_sec] section from a locally uploaded file.

    Keeps a small padding before the start so cuts land cleanly;
    `extra_end_sec` keeps a trailing buffer the caller trims.

    Fast path (B1): if the source is h264+aac (or silent) it is cut with
    ``-c copy -copyinkf`` — lossless and near-instant instead of a full
    re-encode. Otherwise (and as the watchdog retry) it re-encodes.
    """
    file_path = Path(file_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"orig_{idx}.mp4"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    pad = SECTION_PAD_SECONDS
    in_seek = max(0, start_sec - pad)
    out_seek = float(start_sec - in_seek)
    duration = max(end_sec - start_sec, 1) + extra_end_sec

    copy_ok = can_stream_copy(file_path)
    attempt = 0
    while True:
        attempt += 1
        if copy_ok and attempt <= 1:
            # Lossless cut: no re-encode, keep keyframes at the boundary.
            cmd = [
                get_ffmpeg_executable(),
                "-y",
                "-ss",
                str(in_seek),
                *hwaccel_args(),
                "-i",
                str(file_path),
                "-ss",
                str(out_seek),
                "-t",
                str(duration),
                "-c",
                "copy",
                "-copyinkf",
                "-avoid_negative_ts",
                "make_zero",
                str(out_path),
            ]
        else:
            cmd = [
                get_ffmpeg_executable(),
                "-y",
                "-ss",
                str(in_seek),
                *hwaccel_args(),
                "-i",
                str(file_path),
                "-ss",
                str(out_seek),
                "-t",
                str(duration),
                *video_encode_args(QUALITY),
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-avoid_negative_ts",
                "make_zero",
                str(out_path),
            ]

        rc = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600
        ).returncode
        if rc == 0:
            break
        # A copy-mode failure (e.g. VFR/odd GOP) is NOT fatal: fall back to a
        # re-encode once, then give up.
        if rc != 0 and copy_ok and attempt <= 1:
            logger.warning("stream-copy cut failed (rc=%s), retrying with re-encode", rc)
            copy_ok = False
            continue
        raise subprocess.CalledProcessError(rc, cmd)

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise FileNotFoundError(f"Cut section was not produced from {file_path}")
    return out_path


def video_info_local(path: str | Path) -> dict[str, Any]:
    """Read metadata of a locally uploaded video via ffprobe."""
    path = Path(path)
    probe = get_ffprobe_executable()
    r = subprocess.run(
        [
            probe,
            "-v",
            "error",
        "-show_entries",
        "format=duration:stream=width,height,avg_frame_rate",
        "-of",
        "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    data = json.loads(r.stdout or "{}")
    fmt = data.get("format") or {}
    streams = data.get("streams") or []
    video_stream = next((s for s in streams if s.get("width")), None) or (
        streams[0] if streams else {}
    )

    try:
        duration = float(fmt.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0

    fps = None
    raw_fps = video_stream.get("avg_frame_rate")
    if raw_fps and isinstance(raw_fps, str) and "/" in raw_fps:
        try:
            num, den = raw_fps.split("/")
            if float(den) != 0:
                fps = float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            fps = None

    return {
        "title": path.stem,
        "thumbnail": None,
        "duration": duration,
        "sizeBytes": path.stat().st_size if path.exists() else None,
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "fps": fps,
        "extractor": "local",
        "extractorKey": "local",
        "webpage_url": str(path),
    }


def detect_letterbox(
    path: str | Path, max_samples: int = 5, threshold: float = 12.0
) -> tuple[int, int]:
    """Detect top and bottom black bar padding (letterbox) in pixels.

    Returns (top_pad, bottom_pad). If no letterboxing, returns (0, 0).
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0, 0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not orig_w or not orig_h or orig_h < 100:
        cap.release()
        return 0, 0

    sample_indices = np.linspace(
        0, max(0, total_frames - 1), num=max(1, min(max_samples, total_frames)), dtype=int
    )
    top_cuts: list[int] = []
    bot_cuts: list[int] = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        row_means = frame.mean(axis=(1, 2))
        non_black = np.where(row_means > threshold)[0]
        if len(non_black) > 0:
            top_cuts.append(int(non_black[0]))
            bot_cuts.append(int(orig_h - 1 - non_black[-1]))
    cap.release()

    if not top_cuts or not bot_cuts:
        return 0, 0

    top = int(np.median(top_cuts))
    bot = int(np.median(bot_cuts))

    if top < 12 or top > int(orig_h * 0.4):
        top = 0
    if bot < 12 or bot > int(orig_h * 0.4):
        bot = 0

    top -= top % 2
    bot -= bot % 2
    return top, bot


