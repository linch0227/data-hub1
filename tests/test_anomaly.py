import numpy as np
import pandas as pd

from moving_window_anomaly import (
    detect_windowed_anomalies,
    extract_anomaly_events,
    find_anomalies_windowed,
)


def _make_series_with_spikes(n=300, freq="1s", spike_positions=(100, 200), spike_magnitude=10.0, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-01", periods=n, freq=freq)
    values = rng.normal(loc=0.0, scale=1.0, size=n)
    for pos in spike_positions:
        values[pos] += spike_magnitude
    return pd.Series(values, index=index)


def test_detect_windowed_anomalies_flags_injected_spikes():
    series = _make_series_with_spikes()
    result = detect_windowed_anomalies(series, window=20, alpha=3.0)

    assert result.is_anomaly.loc[series.index[100]]
    assert result.is_anomaly.loc[series.index[200]]
    # Sanity: normal (non-spiked) points shouldn't dominate the flagged set.
    assert result.is_anomaly.sum() < 20


def test_detect_windowed_anomalies_respects_min_periods():
    series = _make_series_with_spikes(n=50, spike_positions=(0,))
    result = detect_windowed_anomalies(series, window=10, alpha=2.0, min_periods=5)
    # First few points don't have enough samples in the window -> never flagged.
    assert not result.is_anomaly.iloc[:4].any()


def test_extract_anomaly_events_groups_contiguous_points():
    index = pd.date_range("2022-01-01", periods=10, freq="1s")
    series = pd.Series(np.zeros(10), index=index)
    is_anomaly = pd.Series([False, True, True, False, False, True, False, False, False, False], index=index)

    result = detect_windowed_anomalies(series, window=3, alpha=1e9)  # force no auto anomalies
    result.is_anomaly = is_anomaly  # override with a controlled mask

    events = extract_anomaly_events(result, merge_gap_points=0)
    assert len(events) == 2
    assert events[0].start == index[1]
    assert events[0].end == index[2]
    assert events[0].n_points == 2
    assert events[1].start == index[5]
    assert events[1].n_points == 1


def test_extract_anomaly_events_merges_small_gaps():
    index = pd.date_range("2022-01-01", periods=10, freq="1s")
    series = pd.Series(np.zeros(10), index=index)
    is_anomaly = pd.Series([False, True, False, True, False, False, False, False, False, False], index=index)

    result = detect_windowed_anomalies(series, window=3, alpha=1e9)
    result.is_anomaly = is_anomaly

    events_no_merge = extract_anomaly_events(result, merge_gap_points=0)
    assert len(events_no_merge) == 2

    events_merged = extract_anomaly_events(result, merge_gap_points=1)
    assert len(events_merged) == 1
    assert events_merged[0].n_points == 2


def test_find_anomalies_windowed_runs_over_multiple_series():
    series_dict = {
        "A": _make_series_with_spikes(spike_positions=(50,)),
        "B": _make_series_with_spikes(spike_positions=(60,), seed=1),
    }
    out = find_anomalies_windowed(series_dict, window=15, alpha=3.0)
    assert set(out.keys()) == {"A", "B"}
    result_a, events_a = out["A"]
    assert len(events_a) >= 1
    assert result_a.is_anomaly.loc[series_dict["A"].index[50]]
