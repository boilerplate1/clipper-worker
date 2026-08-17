import os
import tempfile
from dataclasses import dataclass, field

@dataclass
class StorageConfig:
    endpoint: str = ""
    bucket: str = ""
    access_key: str = ""
    secret_key: str = ""
    region: str = ""
    public_url: str = ""

@dataclass
class EngineConfig:
    temp_dir: str = tempfile.gettempdir()
    ytdlp_cookies: str = os.getenv("YDLP_COOKIES", "")
    ytdlp_cookies_from_browser: str = os.getenv("YDLP_COOKIES_FROM_BROWSER", "")
    ytdlp_proxy: str = ""
    ytdlp_max_height: int = 1080
    ytdlp_bgutil_script: str = ""
    ytdlp_bgutil_url: str = ""

@dataclass
class WorkerConfig:
    worker_id: str = "desktop-local"
    storage: StorageConfig = field(default_factory=StorageConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute: str = "int8"
    ffmpeg_encoder: str = "auto"
    ffmpeg_stall_seconds: int = 60
    min_free_disk_gb: int = 10
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

_CONFIG = WorkerConfig()

def get_config() -> WorkerConfig:
    return _CONFIG
