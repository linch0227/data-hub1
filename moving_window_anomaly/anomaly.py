"""Moving-window (rolling) anomaly detection.

A point is flagged as anomalous when it deviates from the trailing rolling
mean by more than ``alpha`` rolling standard deviations. Both the window
and the threshold are computed *locally*, so the detector adapts to slow
drifts/trends in the signal instead of relying on a single global
mean/std for the whole series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd


@dataclass
class WindowedAnomalyResult:
    """Container for the output of :func:`detect_windowed_anomalies`."""

    series: pd.Series
    rolling_mean: pd.Series
    rolling_std: pd.Series
    zscore: pd.Series
    is_anomaly: pd.Series

    @property
    def anomalies(self) -> pd.Series:
        """The subset of ``series`` flagged as anomalous."""
        return self.series[self.is_anomaly]


def detect_windowed_anomalies(
    series: pd.Series,
    window,
    alpha: float = 2.0,
    min_periods: int = 2,
    center: bool = False,
) -> WindowedAnomalyResult:
    """Flag points that fall outside ``alpha`` rolling standard deviations.

    :param series: input time series (``DatetimeIndex`` or integer index).
    :param window: rolling window, either an integer number of samples or a
        pandas offset string (e.g. ``"5min"``) when ``series`` has a
        ``DatetimeIndex``.
    :param alpha: number of rolling standard deviations a point must exceed
        to be flagged as an anomaly.
    :param min_periods: minimum number of samples required in a window
        before it produces a (non-NaN) mean/std; windows with fewer points
        never flag an anomaly.
    :param center: whether the rolling window is centered on each point
        (``False`` = trailing window, matching a live/causal detector).
    :return: a :class:`WindowedAnomalyResult`.
    """
    s = series.astype(float)
    rolling = s.rolling(window=window, min_periods=min_periods, center=center)
    rolling_mean = rolling.mean()
    rolling_std = rolling.std()

    with np.errstate(invalid="ignore", divide="ignore"):
        zscore = (s - rolling_mean) / rolling_std

    is_anomaly = (zscore.abs() > alpha).fillna(False)

    return WindowedAnomalyResult(
        series=s,
        rolling_mean=rolling_mean,
        rolling_std=rolling_std,
        zscore=zscore,
        is_anomaly=is_anomaly,
    )


@dataclass
class AnomalyEvent:
    """A contiguous run of anomalous points, summarized as one event."""

    start: Any
    end: Any
    peak_time: Any
    peak_value: float
    peak_zscore: float
    n_points: int


def extract_anomaly_events(
    result: WindowedAnomalyResult, merge_gap_points: int = 0
) -> List[AnomalyEvent]:
    """Group anomalous points into discrete events.

    Points are merged into the same event when they are separated by at
    most ``merge_gap_points`` non-anomalous samples (``0`` = only strictly
    consecutive anomalous points are merged). Working in integer sample
    positions (rather than index labels) keeps this agnostic to whether
    the index is a ``DatetimeIndex`` or a plain integer index.

    :param result: output of :func:`detect_windowed_anomalies`.
    :param merge_gap_points: max number of non-anomalous samples allowed
        between two anomalous points for them to be merged into one event.
    :return: list of :class:`AnomalyEvent`, ordered by time.
    """
    mask = result.is_anomaly.to_numpy()
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        return []

    groups: List[List[int]] = []
    current = [int(positions[0])]
    for p in positions[1:]:
        if p - current[-1] <= merge_gap_points + 1:
            current.append(int(p))
        else:
            groups.append(current)
            current = [int(p)]
    groups.append(current)

    index = result.series.index
    events: List[AnomalyEvent] = []
    for group in groups:
        sub_idx = index[group]
        sub_vals = result.series.iloc[group]
        sub_z = result.zscore.iloc[group]
        rel_peak = int(np.argmax(np.abs(sub_z.to_numpy())))
        events.append(
            AnomalyEvent(
                start=sub_idx[0],
                end=sub_idx[-1],
                peak_time=sub_idx[rel_peak],
                peak_value=float(sub_vals.iloc[rel_peak]),
                peak_zscore=float(sub_z.iloc[rel_peak]),
                n_points=len(group),
            )
        )
    return events


def find_anomalies_windowed(
    series_dict: Mapping[str, pd.Series],
    window,
    alpha: float = 2.0,
    min_periods: int = 2,
    merge_gap_points: int = 0,
) -> Dict[str, Tuple[WindowedAnomalyResult, List[AnomalyEvent]]]:
    """Run windowed anomaly detection over several named series at once.

    :param series_dict: mapping of name -> series (e.g. ``{"Q": q, "LET": let}``).
    :return: mapping of name -> ``(WindowedAnomalyResult, list[AnomalyEvent])``.
    """
    out: Dict[str, Tuple[WindowedAnomalyResult, List[AnomalyEvent]]] = {}
    for name, s in series_dict.items():
        result = detect_windowed_anomalies(
            s, window=window, alpha=alpha, min_periods=min_periods
        )
        events = extract_anomaly_events(result, merge_gap_points=merge_gap_points)
        out[name] = (result, events)
    return out
