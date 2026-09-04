"""Moving-window anomaly detection and lag/lead analysis for time series.

This package is deliberately generic: it works on any ``pandas.Series``
(with a ``DatetimeIndex`` or a plain integer index), so it can be reused
across different signals (e.g. solar wind proxies, sensor readings, ...)
without depending on how the data was loaded.

Typical workflow::

    from moving_window_anomaly import (
        detect_windowed_anomalies, extract_anomaly_events,
        cross_correlation_lag, proximity_lag,
    )

    result = detect_windowed_anomalies(series, window="5min", alpha=2.0)
    events = extract_anomaly_events(result)

    lag_result = cross_correlation_lag(series_a, series_b, max_lag=30)
    matches = proximity_lag(events_a, events_b, max_lag="2min")

See ``examples/demo_moving_window_anomaly.py`` for an end-to-end example on
synthetic data, and the ``tests/`` folder for smaller, focused examples of
each function.
"""

from .anomaly import (
    AnomalyEvent,
    WindowedAnomalyResult,
    detect_windowed_anomalies,
    extract_anomaly_events,
    find_anomalies_windowed,
)
from .lag import (
    LagScanResult,
    cross_correlation_lag,
    proximity_lag,
    summarize_lag,
)
from .plotting import (
    plot_lag_scan,
    plot_windowed_anomalies,
    run_and_plot_windowed_anomaly_detection,
)

__all__ = [
    "AnomalyEvent",
    "WindowedAnomalyResult",
    "detect_windowed_anomalies",
    "extract_anomaly_events",
    "find_anomalies_windowed",
    "LagScanResult",
    "cross_correlation_lag",
    "proximity_lag",
    "summarize_lag",
    "plot_lag_scan",
    "plot_windowed_anomalies",
    "run_and_plot_windowed_anomaly_detection",
]
