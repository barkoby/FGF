from __future__ import annotations

from datetime import datetime


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log_info(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)
