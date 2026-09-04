import numpy as np
import pandas as pd

from moving_window_anomaly.anomaly import AnomalyEvent
from moving_window_anomaly.lag import cross_correlation_lag, proximity_lag, summarize_lag


def test_cross_correlation_lag_recovers_known_shift():
    n = 500
    index = pd.date_range("2022-01-01", periods=n, freq="1s")
    t = np.arange(n)
    x = pd.Series(np.sin(2 * np.pi * t / 50.0), index=index)

    true_shift = 7  # y is x delayed by 7 samples -> x leads y by 7 samples
    y_values = np.roll(x.values, true_shift)
    y = pd.Series(y_values, index=index).iloc[true_shift:]
    x_trimmed = x.iloc[true_shift:]

    result = cross_correlation_lag(x_trimmed, y, max_lag=15, sample_spacing=1.0)

    assert result.best_lag == true_shift
    assert result.best_correlation > 0.99


def test_cross_correlation_lag_negative_shift():
    # A single non-periodic pulse (rather than a pure sine) avoids the
    # ambiguity of a periodic signal having equally strong correlation at a
    # half-period offset, so there is one unambiguous best lag.
    n = 400
    index = pd.date_range("2022-01-01", periods=n, freq="1s")
    t = np.arange(n)
    x_values = np.exp(-((t - 200) ** 2) / (2 * 8.0 ** 2))
    x = pd.Series(x_values, index=index)

    true_shift = -5  # y[i] = x[i + 5] -> y previews x's future -> y leads x by 5
    y = pd.Series(np.roll(x.values, true_shift), index=index)
    # trim the wraparound edges for a clean comparison window
    x_trimmed = x.iloc[50:-50]
    y_trimmed = y.iloc[50:-50]

    result = cross_correlation_lag(x_trimmed, y_trimmed, max_lag=15, sample_spacing=1.0)

    assert result.best_lag == true_shift
    assert result.best_correlation > 0.99


def test_cross_correlation_lag_handles_insufficient_overlap():
    x = pd.Series([1.0, 2.0], index=pd.date_range("2022-01-01", periods=2, freq="1s"))
    y = pd.Series([1.0, 2.0], index=pd.date_range("2022-01-01", periods=2, freq="1s"))
    result = cross_correlation_lag(x, y, max_lag=5)
    # With only 2 points, most shifted windows have < 2 overlapping samples.
    assert np.isfinite(result.best_correlation)


def _event(peak_time, value=5.0):
    return AnomalyEvent(start=peak_time, end=peak_time, peak_time=peak_time, peak_value=value, peak_zscore=3.0, n_points=1)


def test_proximity_lag_matches_nearest_events_and_sign():
    t0 = pd.Timestamp("2022-01-01 00:00:00")
    events_a = [_event(t0), _event(t0 + pd.Timedelta("10min"))]
    # b follows a by 2 minutes each time
    events_b = [_event(t0 + pd.Timedelta("2min")), _event(t0 + pd.Timedelta("12min"))]

    matches = proximity_lag(events_a, events_b)
    assert len(matches) == 2
    assert (matches["lag"] == 120.0).all()  # seconds, positive = b follows a


def test_proximity_lag_respects_max_lag():
    t0 = pd.Timestamp("2022-01-01 00:00:00")
    events_a = [_event(t0)]
    events_b = [_event(t0 + pd.Timedelta("10min"))]

    matches = proximity_lag(events_a, events_b, max_lag="2min")
    assert matches.empty

    matches_ok = proximity_lag(events_a, events_b, max_lag="20min")
    assert len(matches_ok) == 1


def test_proximity_lag_empty_inputs():
    assert proximity_lag([], []).empty
    assert proximity_lag([_event(pd.Timestamp("2022-01-01"))], []).empty


def test_summarize_lag():
    t0 = pd.Timestamp("2022-01-01")
    events_a = [_event(t0), _event(t0 + pd.Timedelta("1min"))]
    events_b = [_event(t0 + pd.Timedelta("20s")), _event(t0 + pd.Timedelta("80s"))]
    matches = proximity_lag(events_a, events_b)
    summary = summarize_lag(matches)
    assert summary["n_matches"] == 2
    assert summary["mean_lag"] == 20.0

    empty_summary = summarize_lag(proximity_lag([], []))
    assert empty_summary["n_matches"] == 0
