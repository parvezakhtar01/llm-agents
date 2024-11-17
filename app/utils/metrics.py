# app/utils/metrics.py

import time
import psutil
import tracemalloc
from typing import Dict, Any, Optional
from datetime import datetime
import logging


class MetricsTracker:
    def __init__(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss
        tracemalloc.start()
        self.metrics = {
            "timing": {},
            "memory": {},
            "tokens": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            },
            "model_usage": {
                "calls": []
            }
        }

    def start_operation(self, operation_name: str):
        """Start timing an operation"""
        self.metrics["timing"][operation_name] = {
            "start": time.time(),
            "end": None,
            "duration": None
        }

    def end_operation(self, operation_name: str):
        """End timing an operation"""
        if operation_name in self.metrics["timing"]:
            self.metrics["timing"][operation_name]["end"] = time.time()
            self.metrics["timing"][operation_name]["duration"] = (
                    self.metrics["timing"][operation_name]["end"] -
                    self.metrics["timing"][operation_name]["start"]
            )

    def track_model_usage(
            self,
            model_name: str,
            operation_type: str,
            prompt_tokens: int,
            completion_tokens: int,
            parameters: Optional[Dict] = None
    ):
        """Track model usage including tokens and parameters"""
        self.metrics["tokens"]["prompt_tokens"] += prompt_tokens
        self.metrics["tokens"]["completion_tokens"] += completion_tokens
        self.metrics["tokens"]["total_tokens"] += prompt_tokens + completion_tokens

        self.metrics["model_usage"]["calls"].append({
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "operation": operation_type,
            "parameters": parameters or {},
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": prompt_tokens + completion_tokens
            }
        })

    def get_memory_usage(self):
        """Get current memory metrics"""
        current, peak = tracemalloc.get_traced_memory()
        self.metrics["memory"] = {
            "current_memory_mb": current / 10 ** 6,
            "peak_memory_mb": peak / 10 ** 6,
            "memory_increase_mb": (psutil.Process().memory_info().rss - self.start_memory) / 10 ** 6
        }

    def get_final_metrics(self) -> Dict[str, Any]:
        """Get final metrics summary"""
        self.get_memory_usage()
        total_duration = time.time() - self.start_time

        return {
            "execution_metrics": {
                "total_duration": total_duration,
                "operation_timings": self.metrics["timing"],
                "memory_usage": self.metrics["memory"],
                "token_usage": self.metrics["tokens"],
                "model_usage": self.metrics["model_usage"],
                "timestamp": datetime.now().isoformat()
            }
        }

    def __del__(self):
        tracemalloc.stop()
