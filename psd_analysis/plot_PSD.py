import cdflib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import astropy.units as u
import plasmapy
from scipy import signal


from data_loading_and_converted import load_and_preprocess_mag, load_and_preprocess_v_p_n_t, get_resampled_data, convert_FA
from compute_psd import compute_PSD_params, process_multi_window_psd, compute_PSD_fitting, power_law_func, compute_log_binning, fin_diff_deriv, bkn_pow
from compute_other_function import get_fci, compute_acf_at_lag, compute_pdf_analysis, compute_correlation_time, compute_sf_analysis

# ==========================================
# 模組 C：繪圖函數
# ==========================================

def plot_psd_with_markers(psd_df, tau_c_dict, fci_dict, inertial_time=None, type=None, fitting_params=None):
    plt.figure(figsize=(10, 7))
    
    # 建立固定顏色對應
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    comp_colors = {comp: color_cycle[i % len(color_cycle)] for i, comp in enumerate(psd_df.columns)}

    B_components = ['B_para', 'B_perp1', 'B_perp2', 'Trace']
    V_components = ['V_para', 'V_perp1', 'V_perp2', 'Trace']

    correlation = 0  # Initialize
    
    # --- 1. 確定正確的組成分量名稱 (comp) ---
    if type == 'Trace':
        if len(psd_df.columns) > 1 and psd_df.columns[1] in B_components:
            comp = 'Trace'
        elif len(psd_df.columns) > 1 and psd_df.columns[1] in V_components:
            comp = 'Trace'
        else:
            comp = psd_df.columns[-1] # 降級安全機制
    else:
        comp = type if type is not None else psd_df.columns[0]

    # 確保全域畫筆顏色存在，避免 type == None 時文本著色崩潰
    c = comp_colors.get(comp, 'gray')

    # --- 2. 畫 PSD 曲線 ---
    if type is None:
        for col in psd_df.columns:
            plt.loglog(psd_df.index, psd_df[col], label=col, color=comp_colors[col], alpha=0.8)
        
        # 多分量模式下取關聯時間的最小值
        if len(psd_df.columns) > 1 and psd_df.columns[1] in B_components:
            correlation = min(tau_c_dict[k] for k in B_components if k in tau_c_dict)
        elif len(psd_df.columns) > 1 and psd_df.columns[1] in V_components:
            correlation = min(tau_c_dict[k] for k in V_components if k in tau_c_dict)            
    else:
        # 【修正核心】：將 psd_df[type] 改為對齊轉換後的真實分量 psd_df[comp]
        plt.loglog(psd_df.index, psd_df[comp], label=comp, color=comp_colors[comp], alpha=0.8)

    plt.tight_layout()  # 先整理佈局以獲得正確的軸範圍
    ymin, ymax = plt.ylim()
    freqs = psd_df.index.values

    # --- 3. 畫 Correlation Frequencies (fc) ---
    if correlation == 0:
        correlation = tau_c_dict.get(comp, 0)
        c = comp_colors.get(comp, 'gray')
        
    freq_correlation = 1 / correlation if correlation > 0 else None
    text_y_bottom = ymin * (ymax / ymin)**0.05
    text_y_top = ymin * (ymax / ymin)**0.85
    
    if freq_correlation is not None:
        plt.axvline(x=freq_correlation, color=c, linestyle='--', alpha=0.7)
        plt.text(freq_correlation, text_y_bottom, f'correlation time {comp}', color=c, rotation=90, va='bottom', ha='right', fontsize=9)
    
    # --- 4. 畫 Ion Gyrofrequency (fci) ---
    # fci_dict 應為單一數值 (自 self.params['f_gyro'] 傳入)
    if fci_dict is not None:
        plt.axvline(x=fci_dict, color='gray', linestyle=':', linewidth=2)
        plt.text(fci_dict, text_y_top, f'ion gyrofrequency', color='gray', rotation=90, va='top', ha='left', fontsize=9)

    # --- 5. Ion Inertial Scale：依論文可讀性需求，不再繪製此參考線 ---

    # --- 6. 多段冪律擬合曲線疊加繪製 ---
    # 【核心優化】：無論外層傳遞單一 Dict 還是多段 List，此處皆能完美遍歷渲染
    if fitting_params is not None and comp in fitting_params:
        records = fitting_params[comp]
        if not isinstance(records, list):
            records = [records]
            
        for seg_idx, seg_data in enumerate(records):
            f_start = seg_data['start']
            f_end = seg_data['end']
            f_ref = np.array([f_start, f_end])
            fit_y = power_law_func(f_ref, *seg_data['popt'])

            # 呼叫外部波段函數繪製擬合實線（圖例保持簡短，斜率資訊改用大字體標註於線段旁）
            plt.loglog(f_ref, fit_y, linestyle='--', linewidth=2.5,
                       label=f"Seg {seg_idx+1} Fit")

            # 【放大斜率資訊區塊】在擬合線中點旁標註斜率數值，字體加大並加上底框以利論文閱讀
            f_mid = np.sqrt(f_start * f_end)  # log 空間中點
            y_mid = power_law_func(f_mid, *seg_data['popt'])
            plt.annotate(
                f"slope = {seg_data['slope']:.2f} $\\pm$ {seg_data['slope_error']:.2f}",
                xy=(f_mid, y_mid), xytext=(0, 12), textcoords='offset points',
                fontsize=14, fontweight='bold', ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.85)
            )

    # --- 7. Kolmogorov K41 參考斜率線渲染 ---
    # 【修正核心】：target_comp 綁定為當前作用中的真實分量 comp，不再發生 'Trace' 找不到造成的崩潰
    target_comp = comp
    k_start = 0.0009  # 低頻起點
    k_end = 0.09      # 高頻終點

    if freqs.min() <= k_start <= freqs.max() and target_comp in psd_df.columns:
        idx = (np.abs(freqs - k_start)).argmin()
        p_start = psd_df[target_comp].iloc[idx]

        k_ref = np.array([k_start, k_end])
        offset_factor = 0.5 
        p_ref = (p_start * offset_factor) * (k_ref / k_start)**(-5/3)

        plt.loglog(k_ref, p_ref, 'k--', linewidth=2, label=r'$f^{-5/3}$ (K41)')
        plt.text(k_end * 1.05, p_ref[-1], r'$f^{-5/3}$', fontsize=11, fontweight='bold', va='center', ha='left')

    # --- 8. 視窗坐標軸收尾樣式優化 ---
    plt.xlabel("Frequency [Hz]", fontsize=16)
    plt.ylabel("PSD [nT²/Hz]", fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(True, which='both', alpha=0.3, ls=':')
    plt.xlim(1e-4, 1)  # 限制頻率範圍 0.0001 - 1 Hz，提升論文圖片可讀性
    plt.legend(loc='upper right', frameon=True, fontsize=9)

def plot_fitting_check(psd_df, type, fit_params):
    """
    進階譜補償驗證圖：同時顯示「自訂非線性擬合」與「K41 (-5/3) 理論值」兩種方案的補償譜。
    """
    f_start = fit_params['start']
    f_end = fit_params['end']
    freqs = psd_df.index
    raw_psd = psd_df[type].values

    # ==========================================
    # 1. 方案 A：自訂非線性擬合的補償譜 (Compensated A)
    # ==========================================
    slope_A = fit_params["slope"]
    compensated_A = raw_psd * (freqs ** (-slope_A))

    # ==========================================
    # 2. 方案 B：K41 理論值 (-5/3) 的補償譜 (Compensated B)
    # ==========================================
    slope_B = -5/3
    compensated_B = raw_psd * (freqs ** (-slope_B))

    # ==========================================
    # 3. 動態計算兩組方案在擬合區間內的中位數（用於水平對齊）
    # ==========================================
    mask_inertial = (freqs >= f_start) & (freqs <= f_end)
    
    if np.sum(mask_inertial) > 0:
        y_level_A = np.median(compensated_A[mask_inertial])
        y_level_B = np.median(compensated_B[mask_inertial])
    else:
        y_level_A = np.median(compensated_A)
        y_level_B = np.median(compensated_B)

    # ==========================================
    # 4. 開始繪圖
    # ==========================================
    plt.figure(figsize=(9, 6))
    
    # --- 繪製方案 A (自訂擬合) ---
    plt.loglog(freqs, compensated_A, color='blue', linestyle='-', linewidth=2,
               label=rf'Compensated (Your Fit: $f^{{{-slope_A:.2f}}}$)')
    plt.axhline(y=y_level_A, color='blue', linestyle='--', alpha=0.6, 
                label=f'Ideal Horizon (Your Fit)')

    # --- 繪製方案 B (K41 理論) ---
    plt.loglog(freqs, compensated_B, color='gray', linestyle='-.', linewidth=1.8,alpha=0.4,
               label=r'Compensated (K41: $f^{5/3}$)')
    plt.axhline(y=y_level_B, color='gray', linestyle=':', alpha=0.4, 
                label=f'Ideal Horizon (K41)')

    # --- 繪製邊界垂直虛線 ---
    plt.axvline(x=f_start, color='c', linestyle='--', alpha=0.7, label='Fit Start')
    plt.axvline(x=f_end, color='c', linestyle='--', alpha=0.7, label='Fit End')

    # ==========================================
    # 5. 圖表修飾
    # ==========================================
    plt.xlabel('Frequency (Hz)', fontsize=11)
    plt.ylabel('Compensated Amplitude', fontsize=11)
    plt.title(f'Fig 2: [{type}] Compensated Spectrum Validation', fontsize=12, fontweight='bold')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # 為了防範兩組數據高度差太多，把圖例移到最適位置
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()