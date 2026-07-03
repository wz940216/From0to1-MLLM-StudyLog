#!/usr/bin/env python3
"""Plot train/validation loss curves from a JSONL training log."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt


DEFAULT_LOG = (
    "week16_multitask_tuning_summary/outputs/logs/"
    "multitask_balanced/pretrain/train.jsonl"
)


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def format_duration(start: datetime | None, end: datetime | None) -> str:
    if start is None or end is None or end < start:
        return "unknown"

    total_seconds = int((end - start).total_seconds())
    days, rem = divmod(total_seconds, 24 * 3600)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if minutes or hours or days:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def iter_records(log_path: Path) -> Iterable[dict]:
    with log_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skip invalid JSON at line {line_no}: {exc}")
                continue
            if isinstance(record, dict):
                yield record


def collect_points(log_path: Path) -> tuple[list[int], list[float], list[int], list[float], datetime | None, datetime | None]:
    train_steps: list[int] = []
    train_losses: list[float] = []
    val_steps: list[int] = []
    val_losses: list[float] = []
    first_time: datetime | None = None
    last_time: datetime | None = None

    for record in iter_records(log_path):
        timestamp = parse_time(record.get("time"))
        if timestamp is not None:
            first_time = timestamp if first_time is None else min(first_time, timestamp)
            last_time = timestamp if last_time is None else max(last_time, timestamp)

        step = record.get("step")
        if step is None:
            continue

        event = str(record.get("event", "")).lower()
        loss = record.get("loss")
        val_loss = record.get("val_loss", record.get("validation_loss"))

        try:
            step_int = int(step)
        except (TypeError, ValueError):
            continue

        if event == "train_step" and loss is not None:
            train_steps.append(step_int)
            train_losses.append(float(loss))
        elif val_loss is not None:
            val_steps.append(step_int)
            val_losses.append(float(val_loss))
        elif any(name in event for name in ("val", "valid", "eval")) and loss is not None:
            val_steps.append(step_int)
            val_losses.append(float(loss))

    return train_steps, train_losses, val_steps, val_losses, first_time, last_time


def moving_average(values: Sequence[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return list(values)

    smoothed: list[float] = []
    running_sum = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        running_sum += value
        if len(queue) > window:
            running_sum -= queue.pop(0)
        smoothed.append(running_sum / len(queue))
    return smoothed


def downsample_points(steps: Sequence[int], losses: Sequence[float], max_points: int) -> tuple[list[int], list[float]]:
    if max_points <= 0 or len(steps) <= max_points:
        return list(steps), list(losses)

    stride = max(1, len(steps) // max_points)
    sampled_steps = list(steps[::stride])
    sampled_losses = list(losses[::stride])

    if sampled_steps[-1] != steps[-1]:
        sampled_steps.append(steps[-1])
        sampled_losses.append(losses[-1])

    return sampled_steps, sampled_losses


def plot_loss_curve(log_path: Path, output_path: Path, dpi: int, max_train_points: int, smooth_window: int) -> None:
    train_steps, train_losses, val_steps, val_losses, start_time, end_time = collect_points(log_path)

    if not train_steps and not val_steps:
        raise RuntimeError(f"No train or validation loss points found in {log_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6.2))

    plot_train_losses = moving_average(train_losses, smooth_window)
    plot_train_steps, plot_train_losses = downsample_points(train_steps, plot_train_losses, max_train_points)

    if plot_train_steps:
        ax.plot(
            plot_train_steps,
            plot_train_losses,
            label=f"train loss (smoothed, {len(plot_train_steps)} points)",
            color="#1f77b4",
            linewidth=1.4,
            alpha=0.9,
        )
    if val_steps:
        ax.plot(val_steps, val_losses, label="val loss", color="#d62728", marker="o", markersize=3, linewidth=1.7)

    duration = format_duration(start_time, end_time)
    time_text = f"Duration: {duration}"
    if start_time and end_time:
        time_text += f"\nStart: {start_time:%Y-%m-%d %H:%M:%S}\nEnd: {end_time:%Y-%m-%d %H:%M:%S}"

    ax.set_title("Training Loss Curve")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    ax.text(
        0.99,
        0.98,
        time_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )

    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    print(f"Saved: {output_path}")
    print(
        f"Train points: {len(train_steps)} -> {len(plot_train_steps)}, "
        f"val points: {len(val_steps)}, duration: {duration}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train and validation loss from JSONL logs.")
    parser.add_argument("--log", default=DEFAULT_LOG, help="Path to train.jsonl")
    parser.add_argument("--output", default=None, help="Output image path. Default: <log_dir>/loss_curve.png")
    parser.add_argument("--dpi", type=int, default=160, help="Output image DPI")
    parser.add_argument("--max-train-points", type=int, default=1200, help="Maximum train points to draw")
    parser.add_argument("--smooth-window", type=int, default=80, help="Moving-average window for train loss")
    args = parser.parse_args()

    log_path = Path(args.log)
    output_path = Path(args.output) if args.output else log_path.with_name("loss_curve.png")
    plot_loss_curve(log_path, output_path, args.dpi, args.max_train_points, args.smooth_window)


if __name__ == "__main__":
    main()
