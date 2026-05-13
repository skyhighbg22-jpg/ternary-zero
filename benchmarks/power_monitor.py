import shutil
import subprocess
import threading
import time
from typing import Dict, List, Optional


def parse_power_draw_w(raw: str) -> Optional[float]:
    text = raw.strip()
    if not text or text.upper() == "N/A":
        return None
    if text.lower().endswith("w"):
        text = text[:-1].strip()
    try:
        return float(text)
    except ValueError:
        return None


def energy_per_token_mj(power_w: float, duration_s: float, token_count: int) -> float:
    if token_count <= 0 or duration_s <= 0 or power_w < 0:
        return 0.0
    return power_w * duration_s * 1000.0 / float(token_count)


def query_gpu_power_w(gpu_index: int = 0) -> Optional[float]:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None
    first_line = result.stdout.strip().splitlines()
    if not first_line:
        return None
    return parse_power_draw_w(first_line[0])


class PowerSampler:
    def __init__(
        self,
        enabled: bool = False,
        interval_s: float = 0.1,
        gpu_index: int = 0,
    ) -> None:
        self.enabled = enabled
        self.interval_s = max(0.01, float(interval_s))
        self.gpu_index = gpu_index
        self.available = enabled and shutil.which("nvidia-smi") is not None
        self._samples: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def __enter__(self) -> "PowerSampler":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        self._start_time = time.perf_counter()
        if not self.available:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._end_time = time.perf_counter()
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self.interval_s * 4.0)
        self._thread = None

    def summary(self, duration_s: Optional[float] = None) -> Dict[str, object]:
        elapsed = (
            float(duration_s)
            if duration_s is not None
            else self.elapsed_s
        )
        if not self._samples:
            return {
                "enabled": self.enabled,
                "available": self.available,
                "sample_count": 0,
                "avg_power_w": 0.0,
                "peak_power_w": 0.0,
                "energy_j": 0.0,
            }

        avg_power = sum(self._samples) / len(self._samples)
        peak_power = max(self._samples)
        return {
            "enabled": self.enabled,
            "available": self.available,
            "sample_count": len(self._samples),
            "avg_power_w": avg_power,
            "peak_power_w": peak_power,
            "energy_j": avg_power * max(elapsed, 0.0),
        }

    @property
    def samples(self) -> List[float]:
        return list(self._samples)

    @property
    def elapsed_s(self) -> float:
        if self._start_time is None:
            return 0.0
        end_time = self._end_time if self._end_time is not None else time.perf_counter()
        return max(0.0, end_time - self._start_time)

    def _run(self) -> None:
        while not self._stop.is_set():
            power = query_gpu_power_w(self.gpu_index)
            if power is not None:
                self._samples.append(power)
            self._stop.wait(self.interval_s)
