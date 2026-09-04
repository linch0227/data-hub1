"""Plotting helpers for moving-window anomaly detection and lag analysis."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .anomaly import AnomalyEvent, WindowedAnomalyResult, find_anomalies_windowed
from .lag import LagScanResult, cross_correlation_lag


def plot_windowed_anomalies(
    result: WindowedAnomalyResult,
    alpha: float,
    ax=None,
    label: Optional[str] = None,
    events: Optional[List[AnomalyEvent]] = None,
):
    """Plot a series with its rolling ±alpha*sigma band and flagged anomalies."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(result.series.index, result.series.values, color="tab:blue", lw=1, label=label or "series")

    band_upper = result.rolling_mean + alpha * result.rolling_std
    band_lower = result.rolling_mean - alpha * result.rolling_std
    ax.plot(result.rolling_mean.index, result.rolling_mean.values, color="tab:gray", lw=1, ls="--", label="rolling mean")
    ax.fill_between(
        result.rolling_mean.index, band_lower.values, band_upper.values,
        color="tab:gray", alpha=0.2, label=f"±{alpha}σ band",
    )

    anomalies = result.anomalies
    if len(anomalies):
        ax.scatter(anomalies.index, anomalies.values, color="tab:red", zorder=5, s=20, label="anomaly")

    if events:
        for ev in events:
            ax.axvspan(ev.start, ev.end, color="tab:red", alpha=0.08)

    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_lag_scan(result: LagScanResult, ax=None, title: Optional[str] = None):
    """Plot correlation vs. lag, marking the lag of maximum |correlation|."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.plot(result.lags, result.correlations, marker="o", ms=3, lw=1)
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    if np.isfinite(result.best_lag):
        ax.axvline(result.best_lag, color="tab:red", lw=1, ls=":", label=f"best lag={result.best_lag:g}")
        ax.legend(loc="best", fontsize=8)

    ax.set_xlabel("lag")
    ax.set_ylabel("correlation")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def run_and_plot_windowed_anomaly_detection(
    series_dict: Mapping[str, pd.Series],
    window,
    alpha: float = 2.0,
    max_lag: int = 60,
    sample_spacing: float = 1.0,
    min_periods: int = 2,
    merge_gap_points: int = 0,
) -> Tuple[plt.Figure, Dict[str, plt.Figure], dict]:
    """End-to-end: detect windowed anomalies for each series, plot them
    stacked in one figure, then cross-correlation-scan every pair of series
    for lag and plot each pair's scan as its own figure.

    :param series_dict: mapping of name -> series, e.g. ``{"Q": q, "LET": let}``.
        All series should share a comparable time grid for the lag scan to
        be meaningful (they are aligned/intersected automatically, but a
        wildly different sampling rate between series will make "lag in
        samples" hard to interpret -- pass ``sample_spacing`` to convert to
        real time units).
    :param window: rolling window for anomaly detection (int samples or
        pandas offset string).
    :param alpha: anomaly threshold in rolling standard deviations.
    :param max_lag: largest lag (in samples) to scan for each pair.
    :param sample_spacing: real-world duration of one sample, used only to
        label the lag axis in real units.
    :return: ``(fig, lag_figs, detections)`` where ``fig`` is the stacked
        anomaly-detection figure, ``lag_figs`` maps ``"A_vs_B"`` -> figure
        for every unordered pair of series, and ``detections`` is the
        ``find_anomalies_windowed`` output (name -> (result, events)).
    """
    detections = find_anomalies_windowed(
        series_dict, window=window, alpha=alpha, min_periods=min_periods,
        merge_gap_points=merge_gap_points,
    )

    names = list(series_dict.keys())
    fig, axes = plt.subplots(len(names), 1, figsize=(11, 3 * len(names)), sharex=True)
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        result, events = detections[name]
        plot_windowed_anomalies(result, alpha=alpha, ax=ax, label=name, events=events)
        ax.set_ylabel(name)

    axes[0].set_title(f"Moving-window anomaly detection (window={window}, alpha={alpha})")
    fig.tight_layout()

    lag_figs: Dict[str, plt.Figure] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            lag_result = cross_correlation_lag(
                series_dict[name_a], series_dict[name_b],
                max_lag=max_lag, sample_spacing=sample_spacing,
            )
            lag_fig, lag_ax = plt.subplots(figsize=(8, 4))
            plot_lag_scan(lag_result, ax=lag_ax, title=f"{name_a} vs {name_b}")
            lag_figs[f"{name_a}_vs_{name_b}"] = lag_fig

    return fig, lag_figs, detections
