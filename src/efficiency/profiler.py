"""Efficiency measurement: latency, throughput, peak VRAM, energy (spec §4).

Energy methodology (two independent sources, both reported):
  * nvml_gpu_kwh : integral of GPU board power sampled via NVML every 100 ms
                   (trapezoidal rule). Measures the GPU only; it is a measurement,
                   not a TDP-based estimate.
  * codecarbon_* : CodeCarbon estimate of CPU + GPU + RAM energy and CO2-eq.
                   (CPU/RAM are modelled, not measured, on most cloud VMs.)
If NVML is unavailable (no GPU), energy fields are null rather than guessed.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Callable


class NvmlPowerSampler:
    def __init__(self, interval_s: float = 0.1, device: int = 0):
        self.interval, self.device = interval_s, device
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self.available = False
        try:
            import pynvml

            pynvml.nvmlInit()
            self._h = pynvml.nvmlDeviceGetHandleByIndex(device)
            self._nvml = pynvml
            self.available = True
        except Exception:  # noqa: BLE001
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            mw = self._nvml.nvmlDeviceGetPowerUsage(self._h)
            self.samples.append((time.perf_counter(), mw / 1000.0))
            time.sleep(self.interval)

    def start(self) -> None:
        if self.available:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> dict[str, Any]:
        if not self.available:
            return {"nvml_gpu_kwh": None, "nvml_mean_power_w": None}
        self._stop.set()
        self._thread.join()
        s = self.samples
        if len(s) < 2:
            return {"nvml_gpu_kwh": None, "nvml_mean_power_w": None}
        joules = sum((t2 - t1) * (p1 + p2) / 2 for (t1, p1), (t2, p2) in zip(s, s[1:]))
        return {"nvml_gpu_kwh": joules / 3.6e6, "nvml_mean_power_w": joules / (s[-1][0] - s[0][0]),
                "nvml_samples": len(s)}


@contextmanager
def track(stage: str, out: dict, codecarbon_dir: str | None = None):
    """Measure wall time, peak VRAM, NVML energy and CodeCarbon energy of a block."""
    import torch

    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    sampler = NvmlPowerSampler()
    tracker = None
    try:
        from codecarbon import EmissionsTracker

        tracker = EmissionsTracker(project_name=stage, output_dir=codecarbon_dir or ".",
                                   log_level="error", save_to_file=codecarbon_dir is not None)
        tracker.start()
    except Exception:  # noqa: BLE001
        tracker = None
    sampler.start()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if cuda:
            torch.cuda.synchronize()
        rec: dict[str, Any] = {"wall_s": time.perf_counter() - t0}
        rec.update(sampler.stop())
        rec["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1024**3 if cuda else None
        rec["peak_vram_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024**3 if cuda else None
        if tracker is not None:
            try:
                tracker.stop()
                d = tracker.final_emissions_data
                rec.update({"codecarbon_kwh": d.energy_consumed, "codecarbon_kg_co2eq": d.emissions,
                            "codecarbon_gpu_kwh": d.gpu_energy, "codecarbon_cpu_kwh": d.cpu_energy,
                            "codecarbon_ram_kwh": d.ram_energy, "codecarbon_country": d.country_iso_code})
            except Exception as e:  # noqa: BLE001
                rec["codecarbon_error"] = repr(e)
        out[stage] = rec


def measure_latency(run_one: Callable[[int], None], n: int, warmup: int) -> dict[str, float]:
    """run_one(i) processes sample i at batch size 1. Returns ms/sample statistics."""
    import numpy as np
    import torch

    for i in range(warmup):
        run_one(i)
    times = []
    for i in range(n):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_one(warmup + i)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(times)
    return {"latency_ms_mean": float(a.mean()), "latency_ms_std": float(a.std(ddof=1)),
            "latency_ms_p50": float(np.median(a)), "latency_ms_p95": float(np.quantile(a, 0.95)),
            "latency_n": n, "throughput_bs1_samples_per_s": float(1000.0 / a.mean())}

