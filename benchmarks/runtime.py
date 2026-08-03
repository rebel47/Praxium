"""Minimal reproducible scheduler benchmark without third-party harnesses."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import time

from praxium import GraphBuilder, Runtime, State, __version__


async def increment(state: State, _context: object) -> dict[str, int]:
    return {"count": int(state.data.get("count", 0)) + 1}


async def benchmark(iterations: int) -> dict[str, object]:
    graph = (
        GraphBuilder("benchmark")
        .add_node("increment", increment)
        .set_entrypoint("increment")
        .set_finish_point("increment")
        .build()
    )
    runtime = Runtime()
    timings = []
    for _ in range(iterations):
        started = time.perf_counter()
        result = await runtime.run(graph, {"count": 0})
        if result.state.data["count"] != 1:
            raise RuntimeError("benchmark produced an invalid result")
        timings.append(time.perf_counter() - started)
    ordered = sorted(timings)
    return {
        "framework_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "iterations": iterations,
        "mean_ms": statistics.mean(timings) * 1000,
        "median_ms": statistics.median(timings) * 1000,
        "p95_ms": ordered[max(int(len(ordered) * 0.95) - 1, 0)] * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(benchmark(args.iterations)), indent=2))


if __name__ == "__main__":
    main()
