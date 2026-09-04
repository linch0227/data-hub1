# moving_window_anomaly

A small, dependency-light toolkit (pandas + numpy + matplotlib) for:

1. **Moving-window anomaly detection** -- flag points that deviate from a
   trailing rolling mean by more than `alpha` rolling standard deviations,
   and group flagged points into discrete anomaly *events*.
2. **Lag / lead analysis** -- figure out whether/how much one series lags
   or leads another, either by scanning cross-correlation over a range of
   discrete lags (`cross_correlation_lag`), or by matching individual
   anomaly events between two series by nearest-neighbor in time
   (`proximity_lag`).

It is generic over any `pandas.Series` (datetime-indexed or plain integer
index), so the same code works for solar-wind proxies, sensor telemetry,
or any other time series.

## Quick start

```python
import pandas as pd
from moving_window_anomaly import (
    detect_windowed_anomalies, extract_anomaly_events,
    cross_correlation_lag, proximity_lag, summarize_lag,
    run_and_plot_windowed_anomaly_detection,
)

# --- single series ---
result = detect_windowed_anomalies(series, window="5min", alpha=2.0)
events = extract_anomaly_events(result, merge_gap_points=2)

# --- lag between two series (whole-shape cross-correlation) ---
lag_result = cross_correlation_lag(series_a, series_b, max_lag=60, sample_spacing=1.0)
print(lag_result.best_lag, lag_result.best_correlation)

# --- lag between two series (event-by-event) ---
matches = proximity_lag(events_a, events_b, max_lag="2min")
print(summarize_lag(matches))

# --- everything at once, with plots ---
fig, lag_figs, detections = run_and_plot_windowed_anomaly_detection(
    {"Q": series_a, "LET": series_b}, window=120, alpha=3.0, max_lag=180,
)
```

See `examples/demo_moving_window_anomaly.py` for a runnable, fully
synthetic end-to-end example (no external data needed), and `tests/` for
focused unit tests of each function.

## Notes on parameters

- `window`: number of samples (int) or a pandas offset string (e.g.
  `"5min"`, requires a `DatetimeIndex`).
- `alpha`: anomaly threshold in rolling standard deviations. Larger
  `alpha` -> fewer, higher-confidence anomalies.
- `merge_gap_points`: how many non-anomalous samples can separate two
  anomalous points and still have them merged into one event (`0` = only
  strictly consecutive points are merged).
- `cross_correlation_lag` sign convention: `best_lag > 0` means the first
  series leads the second (the second series repeats the first's pattern
  `best_lag` samples later); `best_lag < 0` means the second series leads.
- `proximity_lag` returns `lag = b_peak_time - a_peak_time`; positive
  means B's event follows A's.
