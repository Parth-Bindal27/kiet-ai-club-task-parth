"""Compute tracking utilities.

Tracks training time, generation time, memory usage, and other compute metrics.
"""
import time
from contextlib import contextmanager

try:
    import torch
except ImportError:
    torch = None


class ComputeTracker:
    """Tracks execution time and memory usage for compute budgeting."""

    def __init__(self):
        self.stats = {}
        self._start_time = None

    def start(self):
        """Start tracking wall-clock time."""
        self._start_time = time.time()

    def stop(self):
        """Stop tracking and record elapsed time."""
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            self.stats["wall_time_seconds"] = elapsed
            self._start_time = None

    def get_stats(self) -> dict:
        """Return accumulated stats."""
        stats = dict(self.stats)

        # Add GPU memory info if available
        if torch is not None:
            if torch.cuda.is_available():
                stats["gpu_memory_allocated_gb"] = (
                    torch.cuda.max_memory_allocated() / (1024 ** 3)
                )
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                stats["device"] = "mps"
            else:
                stats["device"] = "cpu"

        return stats

    @contextmanager
    def track(self, name: str):
        """Context manager to track execution time for a named operation."""
        start_time = time.time()
        yield
        elapsed = time.time() - start_time

        if name not in self.stats:
            self.stats[name] = {"time": 0.0, "count": 0}

        self.stats[name]["time"] += elapsed
        self.stats[name]["count"] += 1
