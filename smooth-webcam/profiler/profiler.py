"""
Profiler
Uses: profiler/tree
Used by: all modules

Hierarchical timing profiler using context managers.
"""

import time
from contextlib import contextmanager

# === PARAMETERS ===
SMOOTHING_ALPHA = 0.1


class Profiler:
    """Hierarchical timing profiler with context manager support."""

    def __init__(self):
        self.stack = []
        self.avg_times = {}
        self.total_ms = 0

    @contextmanager
    def section(self, name):
        """
        Time a section using context manager.

        Usage:
            with profiler.section("warp"):
                with profiler.section("rbf"):
                    ...
        """
        self.stack.append(name)
        path = ".".join(self.stack)
        start = time.perf_counter()

        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000

            if path not in self.avg_times:
                self.avg_times[path] = elapsed_ms
            else:
                self.avg_times[path] = (
                    SMOOTHING_ALPHA * elapsed_ms
                    + (1 - SMOOTHING_ALPHA) * self.avg_times[path]
                )

            self.stack.pop()

    def format_node(
        self,
        name,
        ms,
    ):
        """Format a node with time and percentage."""
        pct = (ms / self.total_ms * 100) if self.total_ms > 0 else 0
        bar = "█" * int(pct / 5) if pct >= 5 else ""
        return f"{name}: {ms:.0f}ms ({pct:.0f}%) {bar}"

    def print_tree(
        self,
        paths,
        prefix="",
        parent_path="",
    ):
        """Recursively print timing tree."""
        # Get direct children of current parent
        if parent_path:
            children = [
                p
                for p in paths
                if p.startswith(parent_path + ".")
                and p.count(".") == parent_path.count(".") + 1
            ]
        else:
            children = [p for p in paths if "." not in p]

        children = sorted(children)

        for i, path in enumerate(children):
            is_last = i == len(children) - 1
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")

            name = path.split(".")[-1]
            ms = self.avg_times[path]

            print(f"{prefix}{connector}{self.format_node(name, ms)}")

            # Recurse for children
            self.print_tree(paths, child_prefix, path)

    def report(self):
        """Print hierarchical timing tree."""
        if not self.avg_times:
            return

        top_level = [k for k in self.avg_times.keys() if "." not in k]
        self.total_ms = sum(self.avg_times[k] for k in top_level)
        fps = 1000 / self.total_ms if self.total_ms > 0 else 0

        print(f"\n[{fps:.1f} fps | {self.total_ms:.0f}ms total]")
        self.print_tree(list(self.avg_times.keys()))


# Global profiler instance
profiler = Profiler()
