import time
from contextlib import contextmanager
from typing import Dict


class PipelineMonitor:
    """Monitor and track performance of pipeline components"""

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.start_time = time.time()
        self.components = {}
    def register_component(self, name: str):
        """Register a component for monitoring"""
        self.components[name] = {
            "times": [],
            "errors": 0
        }

    @contextmanager
    def track(self, component_name: str):
        """Track execution time of a component"""
        start = time.time()
        try:
            yield
        except Exception as e:
            # Track the error
            self.components[component_name]["errors"] += 1
            # Re-raise the exception
            raise e
        finally:
            duration = time.time() - start
            self.components[component_name]["times"].append(duration)

    def get_stats(self) -> Dict:
        """Get performance statistics"""
        stats = {
            "pipeline_name": self.pipeline_name,
            "uptime": time.time() - self.start_time,
            "components": {}
        }

        for name, data in self.components.items():
            times = data["times"]
            if times:
                stats["components"][name] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "count": len(times),
                    "errors": data["errors"]
                }
            else:
                stats["components"][name] = {
                    "avg_time": 0,
                    "min_time": 0,
                    "max_time": 0,
                    "count": 0,
                    "errors": data["errors"]
                }

        return stats