"""Создание Gemini-клиента воркера с пулом прокси.

Сервер, где крутится воркер, живёт в регионе, который Google не поддерживает
для Gemini API ("User location is not supported"), поэтому прямые запросы
падают с 400 FAILED_PRECONDITION. Решение — ходить в Gemini через прокси,
у которых egress-IP в поддерживаемом регионе:

   1. Cloudflare Worker (GEMINI_WORKER_URL) — реверс-прокси на
      generativelanguage.googleapis.com: база клиента указывает на воркер,
      прокси не нужны вовсе, не зависит от ПК. Самый надёжный путь;
   2. туннель воркера (ytdlp_proxy, PC в поддерживаемом регионе) — надёжный
      (бесплатные прокси умирают за минуты и иногда отдают мусор, из-за чего
      SDK падал с "Malformed reply");
   3. прокси из переменной окружения GEMINI_PROXIES (через запятую);
   4. рабочие прокси, пойманные в прошлый раз (кэш в /tmp, TTL);
   5. свежий снапшот бесплатных прокси proxyscrape (TTL 15 мин).

Прокси проверяются дешёвым REST-запросом к Gemini (согласие региона == HTTP 200).
google-genai (httpx внутри) читает прокси из окружения в момент конструирования
Client, поэтому HTTPS_PROXY выставляется на время создания и сразу
восстанавливается — на S3/прочие вызовы это не влияет.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from google import genai

from engine.config import get_config

logger = logging.getLogger("Clipper.GeminiProxy")

_SNAPSHOT_URL = (
    "https://api.proxyscrape.com/v4/free-proxy-list/get?"
    "request=display_proxies&proxy_format=protocolipport&format=text"
)
_SNAPSHOT_PATH = Path("/tmp/clipper-gemini-proxies.txt")
_WORKING_PATH = Path("/tmp/clipper-gemini-proxies-working.txt")
_SNAPSHOT_TTL = 900.0
_TEST_LIMIT = 24
_TEST_TIMEOUT = 10.0
_TEST_WORKERS = 10

_PROXY_ENV_KEYS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY")


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return []


def _write_lines(path: Path, lines: list[str]) -> None:
    try:
        path.write_text("\n".join(lines), encoding="utf-8")
    except OSError:
        pass


def _fresh_snapshot() -> list[str]:
    if _SNAPSHOT_PATH.exists() and time.time() - _SNAPSHOT_PATH.stat().st_mtime < _SNAPSHOT_TTL:
        return _read_lines(_SNAPSHOT_PATH)
    try:
        r = httpx.get(_SNAPSHOT_URL, timeout=40)
        if r.status_code == 200 and r.text.strip():
            _write_lines(_SNAPSHOT_PATH, r.text.splitlines())
            return _read_lines(_SNAPSHOT_PATH)
    except Exception as e:
        logger.warning(f"[GEMINI_PROXY] snapshot fetch failed: {e}")
    return _read_lines(_SNAPSHOT_PATH)


def _candidates() -> list[str]:
    pool: list[str] = []
    tunnel = (get_config().engine.ytdlp_proxy or "").strip()
    if tunnel:
        pool.append(tunnel)
    for raw in (os.getenv("GEMINI_PROXIES") or "").split(","):
        p = raw.strip()
        if p and p not in pool:
            pool.append(p)
    for p in _read_lines(_WORKING_PATH):
        if p and p not in pool:
            pool.append(p)
    for p in _fresh_snapshot():
        if p and p not in pool:
            pool.append(p)
    return pool


def _region_ok(proxy: str, api_key: str, model: str) -> bool:
    try:
        with httpx.Client(proxy=proxy, timeout=_TEST_TIMEOUT) as c:
            r = c.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                params={"key": api_key},
                json={"contents": [{"parts": [{"text": "Say OK"}]}]},
            )
        return r.status_code == 200
    except Exception:
        return False


def _pick_proxy(api_key: str, model: str) -> str | None:
    candidates = _candidates()[:_TEST_LIMIT]
    if not candidates:
        return None
    found: list[str] = []
    with ThreadPoolExecutor(max_workers=_TEST_WORKERS) as ex:
        futs = {ex.submit(_region_ok, p, api_key, model): p for p in candidates}
        try:
            for fut in as_completed(futs):
                p = futs[fut]
                if fut.result():
                    found.append(p)
                    for other in futs:
                        other.cancel()
                    break
        except Exception:
            pass
    if found:
        best = found[0]
        _write_lines(_WORKING_PATH, (found + _read_lines(_WORKING_PATH))[:50])
        return best
    return None


def make_gemini_client(api_key: str) -> genai.Client:
    """Возвращает genai.Client, ходящий в Gemini через живой прокси.

    Если ни один прокси не подтвердился — обычный клиент (без прокси, на
    сервере это обычно означает geo-блок, поведение не меняется).
    """
    model = os.getenv("GEMINI_MODEL") or "gemini-3.1-flash-lite"

    worker_url = (os.getenv("GEMINI_WORKER_URL") or "").strip()
    if worker_url:
        logger.info(f"[GEMINI] using Cloudflare worker {worker_url}")
        return genai.Client(
            api_key=api_key,
            http_options=genai.types.HttpOptions(base_url=worker_url),
        )

    proxy = _pick_proxy(api_key, model)
    if not proxy:
        logger.warning("[GEMINI] no working proxy found, using direct client")
        return genai.Client(api_key=api_key)

    logger.info(f"[GEMINI] using proxy {proxy}")
    saved = {k: os.environ.get(k) for k in _PROXY_ENV_KEYS}
    try:
        for k in _PROXY_ENV_KEYS:
            os.environ[k] = proxy
        return genai.Client(api_key=api_key)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

