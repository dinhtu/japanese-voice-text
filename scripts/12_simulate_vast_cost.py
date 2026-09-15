"""Estimate end-to-end training time/cost with setup overhead.

Examples:
    uv run python scripts/12_simulate_vast_cost.py --preset h100_sxm --setup copy
    uv run python scripts/12_simulate_vast_cost.py --preset h100_sxm --setup copy --epochs 20
    uv run python scripts/12_simulate_vast_cost.py --preset h200 --setup redownload --epochs-list 1,3,5,10,20
"""

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class GPUPreset:
    rate_per_hour: float
    speedup_vs_a100_40: float


PRESETS = {
    # Snapshot from 2026-03-01 (Vast search, storage=600GiB, median offer).
    "a100_40gb": GPUPreset(rate_per_hour=0.7889, speedup_vs_a100_40=1.0),
    "a100_80gb": GPUPreset(rate_per_hour=1.1833, speedup_vs_a100_40=1.3805),
    "h100_sxm": GPUPreset(rate_per_hour=1.9222, speedup_vs_a100_40=3.6174),
    "h100_nvl": GPUPreset(rate_per_hour=2.3838, speedup_vs_a100_40=3.8260),
    "h200": GPUPreset(rate_per_hour=2.5591, speedup_vs_a100_40=5.1719),
    "h200_nvl": GPUPreset(rate_per_hour=2.6222, speedup_vs_a100_40=4.7327),
    "b200": GPUPreset(rate_per_hour=2.9706, speedup_vs_a100_40=5.8291),
}

SETUP_PRESETS = {
    "copy": (2.0, 4.0),
    "redownload": (7.0, 13.0),
}


def parse_epochs_list(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("--epochs-list is empty")
    if any(v <= 0 for v in values):
        raise ValueError("--epochs-list must contain positive integers")
    return values


def parse_args():
    p = argparse.ArgumentParser(description="Simulate setup+training total time/cost on Vast")
    p.add_argument("--preset", choices=sorted(PRESETS), default="h100_sxm")
    p.add_argument("--setup", choices=sorted(SETUP_PRESETS), default="copy")
    p.add_argument("--epochs", type=int, default=None, help="Single epoch count to print")
    p.add_argument("--epochs-list", default="1,3,5,10,20", help="Comma-separated epoch counts")
    p.add_argument("--a100-epoch-min", type=float, default=1.8)
    p.add_argument("--a100-epoch-max", type=float, default=3.0)
    return p.parse_args()


def simulate(
    rate_per_hour: float,
    speedup: float,
    setup_min_h: float,
    setup_max_h: float,
    a100_epoch_min_h: float,
    a100_epoch_max_h: float,
    epochs: int,
) -> tuple[float, float, float, float]:
    epoch_min_h = a100_epoch_min_h / speedup
    epoch_max_h = a100_epoch_max_h / speedup
    total_min_h = setup_min_h + epochs * epoch_min_h
    total_max_h = setup_max_h + epochs * epoch_max_h
    total_min_cost = total_min_h * rate_per_hour
    total_max_cost = total_max_h * rate_per_hour
    return total_min_h, total_max_h, total_min_cost, total_max_cost


def main():
    args = parse_args()
    preset = PRESETS[args.preset]
    setup_min_h, setup_max_h = SETUP_PRESETS[args.setup]

    if args.a100_epoch_min < 0 or args.a100_epoch_max <= 0:
        raise ValueError("A100 epoch range must be positive")
    if args.a100_epoch_min > args.a100_epoch_max:
        raise ValueError("--a100-epoch-min must be <= --a100-epoch-max")

    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs must be positive")

    if args.epochs is not None:
        epochs_list = [args.epochs]
    else:
        epochs_list = parse_epochs_list(args.epochs_list)

    print(f"preset={args.preset} setup={args.setup}")
    print(
        f"rate=${preset.rate_per_hour:.4f}/h "
        f"speedup={preset.speedup_vs_a100_40:.4f}x "
        f"a100_epoch={args.a100_epoch_min:.2f}-{args.a100_epoch_max:.2f}h"
    )
    print("epochs,total_hours_range,total_cost_range")
    for e in epochs_list:
        tmin, tmax, cmin, cmax = simulate(
            rate_per_hour=preset.rate_per_hour,
            speedup=preset.speedup_vs_a100_40,
            setup_min_h=setup_min_h,
            setup_max_h=setup_max_h,
            a100_epoch_min_h=args.a100_epoch_min,
            a100_epoch_max_h=args.a100_epoch_max,
            epochs=e,
        )
        print(f"{e},{tmin:.2f}-{tmax:.2f}h,${cmin:.2f}-${cmax:.2f}")


if __name__ == "__main__":
    main()
