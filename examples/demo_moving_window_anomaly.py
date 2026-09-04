"""End-to-end demo of moving-window anomaly detection + lag analysis on
synthetic data (no external dataset required).

Simulates two proxy signals ("Q" and "LET", named after the solar-wind
turbulence proxies this package was originally built for, but the method
applies to any pair of related time series): both share the same slow
background wobble, "LET" reacts to the same events as "Q" but ~90 seconds
later, and a handful of independent spikes are injected into each. Run:

    python examples/demo_moving_window_anomaly.py

Figures are written to ``examples/output/``.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless: this script only writes figures to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from moving_window_anomaly import (
    extract_anomaly_events,
    proximity_lag,
    run_and_plot_windowed_anomaly_detection,
    summarize_lag,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def make_synthetic_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 3600  # 1 hour at 1s cadence
    index = pd.date_range("2022-02-24 00:00:00", periods=n, freq="1s")
    t = np.arange(n)

    background = 0.3 * np.sin(2 * np.pi * t / 900.0)  # slow ~15min wobble
    q = background + rng.normal(scale=0.08, size=n)
    let = background + rng.normal(scale=0.08, size=n)

    lag_seconds = 90
    event_positions = [500, 1500, 2600]
    for pos in event_positions:
        q[pos : pos + 5] += 5.0
        if pos + lag_seconds + 5 < n:
            let[pos + lag_seconds : pos + lag_seconds + 5] += 4.5

    # A couple of spikes independent to each series (no matching counterpart).
    q[3300] += 5.0
    let[3400] += 5.0

    return (
        pd.Series(q, index=index, name="Q"),
        pd.Series(let, index=index, name="LET"),
        lag_seconds,
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    q, let, true_lag_seconds = make_synthetic_data()

    window = 120  # samples (~2 minutes at 1s cadence)
    alpha = 5.0

    fig, lag_figs, detections = run_and_plot_windowed_anomaly_detection(
        {"Q": q, "LET": let}, window=window, alpha=alpha, max_lag=180, sample_spacing=1.0,
        merge_gap_points=5,
    )
    fig.savefig(os.path.join(OUTPUT_DIR, "windowed_anomalies.png"), dpi=150, bbox_inches="tight")
    for name, lag_fig in lag_figs.items():
        lag_fig.savefig(os.path.join(OUTPUT_DIR, f"lag_scan_{name}.png"), dpi=150, bbox_inches="tight")
    plt.close("all")

    result_q, events_q = detections["Q"]
    result_let, events_let = detections["LET"]
    print(f"Q: {len(events_q)} anomaly events detected")
    print(f"LET: {len(events_let)} anomaly events detected")

    matches = proximity_lag(events_q, events_let, max_lag="4min")
    summary = summarize_lag(matches)
    print("\nEvent-proximity lag (Q -> LET), seconds:")
    print(matches)
    print(summary)
    print(f"\n(true injected lag was {true_lag_seconds}s; "
          f"cross-correlation best_lag should land close to it too)")

    print(f"\nFigures written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
