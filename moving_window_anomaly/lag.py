"""Lag / lead analysis between two time series or between anomaly events.

Two complementary approaches are provided:

* :func:`cross_correlation_lag` -- shifts one series against the other by a
  range of discrete lags and finds the lag that maximizes their (Pearson or
  Spearman) correlation. Good for "does the whole shape of B follow A by
  roughly N samples?".
* :func:`proximity_lag` -- matches individual anomaly *events* (e.g. from
  :func:`moving_window_anomaly.anomaly.extract_anomaly_events`) between two
  series by nearest-neighbor in time. Good for "when A spikes, how long
  until B spikes too?", including when spikes are irregular/sparse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class LagScanResult:
    """Output of :func:`cross_correlation_lag`."""

    lags: np.ndarray
    correlations: np.ndarray
    best_lag: float
    best_correlation: float


def cross_correlation_lag(
    x: pd.Series,
    y: pd.Series,
    max_lag: int,
    sample_spacing: float = 1.0,
    method: str = "pearson",
) -> LagScanResult:
    """Scan discrete lags and find the one maximizing |correlation(x, shift(y))|.

    ``x`` and ``y`` are first aligned on their shared index (via an inner
    join, dropping any rows where either is missing) so they must already
    be sampled on a common/comparable grid (e.g. both resampled to the same
    frequency) for the lag in samples to be meaningful.

    Sign convention: for ``lag > 0``, ``x`` is compared against ``y`` shifted
    ``lag`` samples earlier (``y[lag:]`` vs ``x[:-lag]``), i.e. a positive
    ``best_lag`` means **x leads y** (y's pattern shows up ``best_lag``
    samples after x's). A negative ``best_lag`` means y leads x.

    :param x: reference series.
    :param y: series to shift against ``x``.
    :param max_lag: largest lag (in samples) to scan in either direction.
    :param sample_spacing: real-world duration of one sample (e.g. seconds
        between rows); only used to rescale the returned ``lags`` array,
        it does not affect ``best_correlation``.
    :param method: correlation method, forwarded to ``pandas.Series.corr``.
    :return: a :class:`LagScanResult`.
    """
    aligned = pd.concat(
        [pd.Series(x).rename("x"), pd.Series(y).rename("y")], axis=1
    ).dropna()
    xs_full = aligned["x"].reset_index(drop=True)
    ys_full = aligned["y"].reset_index(drop=True)
    n = len(xs_full)

    lags = np.arange(-max_lag, max_lag + 1)
    corrs = np.full(lags.shape, np.nan, dtype=float)

    for i, lag in enumerate(lags):
        if lag < 0:
            xs = xs_full.iloc[-lag:]
            ys = ys_full.iloc[: n + lag]
        elif lag > 0:
            xs = xs_full.iloc[: n - lag]
            ys = ys_full.iloc[lag:]
        else:
            xs, ys = xs_full, ys_full

        if len(xs) < 2:
            continue
        corrs[i] = xs.reset_index(drop=True).corr(ys.reset_index(drop=True), method=method)

    scaled_lags = lags * sample_spacing
    if not np.isfinite(corrs).any():
        return LagScanResult(lags=scaled_lags, correlations=corrs, best_lag=np.nan, best_correlation=np.nan)

    best_i = int(np.nanargmax(np.abs(corrs)))
    return LagScanResult(
        lags=scaled_lags,
        correlations=corrs,
        best_lag=float(scaled_lags[best_i]),
        best_correlation=float(corrs[best_i]),
    )


def _to_seconds_or_float(value):
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    if isinstance(value, str):
        return pd.Timedelta(value).total_seconds()
    return float(value)


def proximity_lag(
    events_a: Sequence,
    events_b: Sequence,
    max_lag: Optional[object] = None,
) -> pd.DataFrame:
    """Match anomaly events between two series by nearest neighbor in time.

    For every event in ``events_a`` (objects exposing a ``peak_time``
    attribute, such as :class:`moving_window_anomaly.anomaly.AnomalyEvent`),
    find the closest event in ``events_b`` by ``peak_time`` and record the
    signed time lag between them.

    :param events_a: events for series A.
    :param events_b: events for series B.
    :param max_lag: optional cap (as a ``pandas.Timedelta``, a timedelta
        string like ``"2min"``, or a plain number matching the events'
        index units) -- matches farther apart than this are dropped.
    :return: DataFrame with columns ``a_peak_time``, ``b_peak_time`` and
        ``lag`` (``b_peak_time - a_peak_time``; positive = b follows a).
        ``lag`` is in seconds when times are datetime-like, otherwise in
        the same units as the underlying index.
    """
    columns = ["a_peak_time", "b_peak_time", "lag"]
    if len(events_a) == 0 or len(events_b) == 0:
        return pd.DataFrame(columns=columns)

    b_index = pd.Index([e.peak_time for e in events_b])
    max_lag_value = _to_seconds_or_float(max_lag) if max_lag is not None else None

    rows = []
    for ea in events_a:
        diffs = b_index - ea.peak_time
        if isinstance(diffs, pd.TimedeltaIndex):
            signed = diffs.total_seconds()
        else:
            signed = np.asarray(diffs, dtype=float)
        abs_diffs = np.abs(signed)

        best_j = int(np.argmin(abs_diffs))
        if max_lag_value is not None and abs_diffs[best_j] > max_lag_value:
            continue

        rows.append(
            {
                "a_peak_time": ea.peak_time,
                "b_peak_time": events_b[best_j].peak_time,
                "lag": float(signed[best_j]),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def summarize_lag(matches: pd.DataFrame, lag_col: str = "lag") -> dict:
    """Summary statistics (count/mean/median/std) of a :func:`proximity_lag` result."""
    if matches.empty:
        return {"n_matches": 0, "mean_lag": np.nan, "median_lag": np.nan, "std_lag": np.nan}
    return {
        "n_matches": int(len(matches)),
        "mean_lag": float(matches[lag_col].mean()),
        "median_lag": float(matches[lag_col].median()),
        "std_lag": float(matches[lag_col].std()),
    }
