"""
自動偵測 PSD 冪律擬合區間 (Auto Power-Law Fitting Range Detection)。

設計理念：
    人工判讀 PSD 慣性尺度時，通常是「找一段局部斜率 (local slope) 幾乎不變的
    連續頻段」。這裡用相鄰資料點在 log-log 空間的斜率序列（先做輕度移動平均
    去雜訊），在指定搜尋範圍內找出局部斜率標準差夠小、且點數最多的連續頻段，
    以此取代手動 input() 選取起訖頻率。

    找到的只是候選起訖頻率 (f_start, f_end)；實際擬合仍呼叫既有的
    compute_PSD_fitting()，回傳格式與手動流程完全相同，可直接放進
    self.params['PSD_fit']，跟既有繪圖/匯出/快取邏輯無縫接軌。

    注意：這是幾何上的啟發式方法，不是物理上保證正確的慣性尺度判定。
    建議搭配自動存檔的補償譜驗證圖 (plot_fitting_check) 人工複核，
    尤其是論文要引用的數值。
"""

import numpy as np


def _local_slope_regression(freqs, psd, window_points=9):
    """
    以滑動窗口在 log-log 空間做線性回歸取得局部斜率。

    比逐點差分 (point-to-point diff) 抗雜訊得多：差分法對相鄰點的雜訊會直接
    放大 (尤其頻率取樣很密時，分母 d(log f) 很小)，而窗口回歸是用多個點一起
    估計斜率，雜訊會被平均掉。
    """
    log_f = np.log10(freqs)
    log_p = np.log10(psd)
    n = len(freqs)
    if n < window_points:
        return np.array([]), np.array([])

    n_out = n - window_points + 1
    centers = np.empty(n_out)
    slopes = np.empty(n_out)
    for i in range(n_out):
        x = log_f[i:i + window_points]
        y = log_p[i:i + window_points]
        if not np.all(np.isfinite(y)):
            centers[i] = freqs[i + window_points // 2]
            slopes[i] = np.nan
            continue
        slope, _ = np.polyfit(x, y, 1)
        centers[i] = freqs[i + window_points // 2]
        slopes[i] = slope

    return centers, slopes


def auto_detect_fit_range(freqs, psd, f_min=9e-4, f_max=9e-2,
                           slope_tol=0.15, min_points=8, window_points=9,
                           require_negative_slope=True):
    """
    在 [f_min, f_max] 搜尋範圍內，自動尋找「局部斜率標準差 <= slope_tol」
    且點數最多的連續頻段，做為冪律擬合的候選區間。

    :param freqs, psd: 完整的 PSD 頻率與功率陣列 (未經頻段裁切)。
    :param f_min, f_max: 搜尋的頻率上下界 (Hz)。預設對齊既有程式碼中
        K41 參考線的 0.0009 ~ 0.09 Hz。
    :param slope_tol: 局部斜率標準差門檻，越小代表對「單一冪律」的要求越嚴格。
    :param min_points: 候選區間至少要包含的局部斜率估計點數，避免擬合區間過窄。
    :param window_points: 局部斜率回歸視窗的資料點數；越大越抗雜訊，但解析度
        (能分辨的最短頻段) 會變粗。
    :param require_negative_slope: 是否要求區間內平均斜率為負值 (符合能量
        串級 cascade 的物理預期，排除雜訊平台或上升段被誤選)。
    :return: dict {'start', 'end', 'mean_slope', 'std_slope', 'n_points'}，
        找不到符合條件的區間時回傳 None。
    """
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)

    valid = np.isfinite(freqs) & np.isfinite(psd) & (freqs > 0) & (psd > 0)
    freqs, psd = freqs[valid], psd[valid]

    order = np.argsort(freqs)
    freqs, psd = freqs[order], psd[order]

    mask = (freqs >= f_min) & (freqs <= f_max)
    freqs, psd = freqs[mask], psd[mask]
    if len(freqs) < min_points:
        return None

    mid_freqs, slope = _local_slope_regression(freqs, psd, window_points)
    n = len(slope)
    if n < min_points:
        return None

    best = None  # (span_points, -std, i, j)
    for i in range(n):
        if not np.isfinite(slope[i]):
            continue
        for j in range(i + min_points - 1, n):
            window = slope[i:j + 1]
            window = window[np.isfinite(window)]
            if len(window) < min_points:
                continue

            std = np.std(window)
            if std > slope_tol:
                continue

            mean_slope = np.mean(window)
            if require_negative_slope and mean_slope >= 0:
                continue

            span = j - i + 1
            candidate = (span, -std, i, j)
            if best is None or candidate > best:
                best = candidate

    if best is None:
        return None

    _, _, i, j = best
    window = slope[i:j + 1]
    window = window[np.isfinite(window)]

    return {
        'start': float(mid_freqs[i]),
        'end': float(mid_freqs[j]),
        'mean_slope': float(np.mean(window)),
        'std_slope': float(np.std(window)),
        'n_points': int(j - i + 1),
    }
