#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import os
import queue
import re
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_TASKS = [
    "binary-addition",
    "binary-multiplication",
    "bucket-sort",
    "compute-sqrt",
    "cycle-navigation",
    "dyck-2-3",
    "even-pairs",
    "first",
    "majority",
    "marked-copy",
    "marked-reversal",
    "missing-duplicate-string",
    "modular-arithmetic-simple",
    "odds-first",
    "parity",
    "repeat-01",
    "stack-manipulation",
    "unmarked-reversal",
]


@dataclass
class TaskResult:
    task: str
    status: str
    mean: Optional[float]
    std: Optional[float]
    max_acc: Optional[float]
    gpu: int
    duration_sec: float
    log_path: Path
    error: Optional[str] = None


def parse_metrics(output: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    multi = re.search(
        r"Inductive-bias test accuracy over\s+\d+\s+runs.*?mean=([0-9]*\.?[0-9]+)\s+std=([0-9]*\.?[0-9]+)\s+max=([0-9]*\.?[0-9]+)",
        output,
        flags=re.DOTALL,
    )
    if multi:
        return float(multi.group(1)), float(multi.group(2)), float(multi.group(3))

    single = re.search(
        r"Inductive-bias test accuracy \(len 0\.\.500\):\s*([0-9]*\.?[0-9]+)",
        output,
    )
    if single:
        v = float(single.group(1))
        return v, 0.0, v

    return None, None, None


def maybe_download_all(train_script: Path, flare_root: Path, tasks: list[str]) -> None:
    spec = importlib.util.spec_from_file_location("flare_reconstruct_lstm", str(train_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import training script: {train_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "download_official_flare_language"):
        raise RuntimeError(
            "Training script does not expose download_official_flare_language()."
        )

    downloader = module.download_official_flare_language
    flare_root.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        print(f"[download] {task}")
        downloader(flare_root, task)


def run_task(
    task: str,
    gpu_id: int,
    args: argparse.Namespace,
    logs_dir: Path,
) -> TaskResult:
    log_path = logs_dir / f"{task}.log"

    cmd = [
        args.python,
        str(args.train_script),
        "--dataset-source",
        "official",
        "--language",
        task,
        "--flare-root",
        str(args.flare_root),
        "--epochs",
        str(args.epochs),
        "--num-runs",
        str(args.num_runs),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
    ]

    if args.cpu_only:
        cmd.append("--cpu-only")

    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))

    env = os.environ.copy()
    if not args.cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    duration = time.time() - start

    combined_output = proc.stdout + "\n" + proc.stderr
    log_path.write_text(
        f"$ {' '.join(shlex.quote(c) for c in cmd)}\n\n"
        + combined_output,
        encoding="utf-8",
    )

    mean, std, max_acc = parse_metrics(combined_output)

    if proc.returncode != 0:
        return TaskResult(
            task=task,
            status="FAIL",
            mean=mean,
            std=std,
            max_acc=max_acc,
            gpu=gpu_id,
            duration_sec=duration,
            log_path=log_path,
            error=f"exit={proc.returncode}",
        )

    if mean is None or std is None:
        return TaskResult(
            task=task,
            status="FAIL",
            mean=mean,
            std=std,
            max_acc=max_acc,
            gpu=gpu_id,
            duration_sec=duration,
            log_path=log_path,
            error="could not parse mean/std from output",
        )

    return TaskResult(
        task=task,
        status="PASS",
        mean=mean,
        std=std,
        max_acc=max_acc,
        gpu=gpu_id,
        duration_sec=duration,
        log_path=log_path,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run FLARE task suite in parallel across multiple GPUs and aggregate "
            "mean/std accuracy per task."
        )
    )
    p.add_argument(
        "--train-script",
        type=Path,
        default=Path(
            "/Users/aviv/Repositories/Codex/experiments/robot_sequence/flare_reconstruct_lstm.py"
        ),
    )
    p.add_argument(
        "--flare-root",
        type=Path,
        default=Path(
            "/Users/aviv/Repositories/Codex/experiments/robot_sequence/flare_data"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/Users/aviv/Repositories/Codex/experiments/robot_sequence/flare_runs"
        ),
    )
    p.add_argument("--python", type=str, default="python3")
    p.add_argument("--num-gpus", type=int, default=4)
    p.add_argument("--num-runs", type=int, default=10)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated list of FLARE tasks. Default is all tasks.",
    )
    p.add_argument(
        "--download-official",
        action="store_true",
        help="Download missing official FLARE files before starting runs.",
    )
    p.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run on CPU only (ignores --num-gpus).",
    )
    p.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="Extra args appended to each train-script invocation.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        raise ValueError("No tasks specified.")

    args.train_script = args.train_script.expanduser().resolve()
    args.flare_root = args.flare_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"run_{run_stamp}"
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.download_official:
        maybe_download_all(args.train_script, args.flare_root, tasks)

    gpus = [0] if args.cpu_only else list(range(args.num_gpus))
    task_queue: queue.Queue[str] = queue.Queue()
    for task in tasks:
        task_queue.put(task)

    results: dict[str, TaskResult] = {}
    lock = threading.Lock()

    def worker(gpu_id: int) -> None:
        while True:
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                return

            print(f"[gpu {gpu_id}] starting {task}")
            result = run_task(task, gpu_id, args, logs_dir)
            with lock:
                results[task] = result
            status_line = (
                f"[gpu {gpu_id}] {result.status} {task} "
                f"mean={result.mean} std={result.std} "
                f"time={result.duration_sec/60:.1f}m"
            )
            print(status_line)
            task_queue.task_done()

    threads = [threading.Thread(target=worker, args=(gpu_id,), daemon=True) for gpu_id in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ordered = [results[t] for t in tasks if t in results]

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "status", "mean", "std", "max", "gpu", "duration_sec", "log_path", "error"])
        for r in ordered:
            writer.writerow(
                [
                    r.task,
                    r.status,
                    "" if r.mean is None else f"{r.mean:.6f}",
                    "" if r.std is None else f"{r.std:.6f}",
                    "" if r.max_acc is None else f"{r.max_acc:.6f}",
                    r.gpu,
                    f"{r.duration_sec:.2f}",
                    str(r.log_path),
                    "" if r.error is None else r.error,
                ]
            )

    md_path = out_dir / "summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| Task | Mean | Std | Status |\n")
        f.write("|---|---:|---:|---|\n")
        for r in ordered:
            mean_s = "-" if r.mean is None else f"{r.mean:.4f}"
            std_s = "-" if r.std is None else f"{r.std:.4f}"
            f.write(f"| {r.task} | {mean_s} | {std_s} | {r.status} |\n")

    print("\nResults table")
    print("| Task | Mean | Std | Status |")
    print("|---|---:|---:|---|")
    for r in ordered:
        mean_s = "-" if r.mean is None else f"{r.mean:.4f}"
        std_s = "-" if r.std is None else f"{r.std:.4f}"
        print(f"| {r.task} | {mean_s} | {std_s} | {r.status} |")

    n_fail = sum(1 for r in ordered if r.status != "PASS")
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {md_path}")
    if n_fail:
        print(f"Completed with {n_fail} failed task(s).")
        return 1
    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
