import cdflib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import math
import astropy.units as u
import plasmapy
from scipy import signal
import scipy.constants as const
import sys
import pickle

#date and time
SAVE_DATE = "20220224"
START_TIME_ID = "000000"
#START_TIME_ID = "150000"
start_time_str = f"{SAVE_DATE} {START_TIME_ID[:2]}:{START_TIME_ID[2:4]}:{START_TIME_ID[4:]}"
END_TIME_ID = "073100" 
#END_TIME_ID = "213100"
end_time_str = f"{SAVE_DATE} {END_TIME_ID[:2]}:{END_TIME_ID[2:4]}:{END_TIME_ID[4:]}"

#file
input_DIR = r'C:\Users\Linch\OneDrive\文件\assignment\MS\program\input'
output_DIR = r'C:\Users\Linch\OneDrive\文件\assignment\MS\program\output'
# 移除了原本的 sys.path.append(input_DIR) / sys.path.append(output_DIR)：
# 這兩行把「資料夾」加進 import 路徑，但下面 import 的自訂模組
# (data_loading_and_converted, compute_psd, ...) 並不放在 input/output 資料夾裡，
# 而是本來就跟這支腳本放在一起 (Python 預設會把腳本所在目錄加入 sys.path)。
# 這兩行對 import 沒有實際作用，反而有風險：如果 input/output 資料夾裡剛好出現
# 同名的 .py 檔，會意外覆蓋掉真正要用的模組。如果你的專案結構跟這裡假設的不同、
# 拿掉後 import 失敗，代表自訂模組其實放在別的地方，把正確的路徑加回來即可。
folder_name = "SOLO_20220224"
file_mkp = "summary_EPS_MKP_20220224_000000_080000.csv"
file_mkp_total = "Total_EPS_MKP_20220224_000000_080000.csv"
file_megn = "SOLO_L2_mag-rtn-normal_20220224_V02.cdf"
file_swa = "SOLO_L2_swa-pas-grnd-mom_20220224_V02.cdf"
file_vdf = "SOLO_L2_swa-pas-vdf_20220224_V02.cdf"
folder_output = f"Turbulence_{START_TIME_ID}_{END_TIME_ID}"
#folder_input = f"Turbulence_{START_TIME_ID}_{END_TIME_ID}"

mag_path = os.path.join(input_DIR, folder_name, file_megn)
swa_path = os.path.join(input_DIR, folder_name, file_swa)
vdf_path = os.path.join(input_DIR, folder_name, file_vdf)
Mkp_path = os.path.join(input_DIR, folder_name, file_mkp)
#Mkp_path = os.path.join(input_DIR, folder_name, file_mkp_total)################
output_path = os.path.join(output_DIR, folder_name, folder_output)
input_path = os.path.join(input_DIR,folder_name,folder_output)

from data_loading_and_converted import load_and_preprocess_mag, load_and_preprocess_v_p_n_t, get_resampled_data, convert_FA
from compute_psd_v1 import compute_PSD_params, process_multi_window_psd, compute_PSD_fitting, power_law_func, compute_log_binning
from compute_other_function import get_fci, compute_acf_at_lag, compute_pdf_analysis, compute_correlation_time, compute_sf_analysis
from plot_PSD import plot_psd_with_markers, plot_fitting_check
from auto_fit import auto_detect_fit_range
from compute_structure_function import plot_structure_functions, plot_flatness_multi_components, plot_single_flatness_with_slope, compute_yagmolaw, compute_yaglom_law
from compute_yaglom import preprocess_elsasser_variable, analyze_yaglom_law, plot_sigma_c_r
from compute_agyro_pressure import compute_agyrotropicity, plot_Q, plot_P
from compute_anomaly import find_the_anomaly_guieds
from plot_time_series import plot_mag_timeseries_dynamic
from load_vdf import load_vdf
from generate_report import generate_report
from function_structure_function import compute_turbulence_structure_function

class SolarOrbiterAnalyzer:
    def __init__(self):
        self.role = "solar orbiter data analyst assistant"
        self.data = {}
        self.params = {}
        self.results = {}
        print(f"Hi, I am {self.role}。I am ready to assist you in analyzing space physics data.")
        print("Type 'help' to see available commands.")

        # ====== 初始化輸出資料夾結構 ======
        self.output_dir = output_path
        self.fig_dir = os.path.join(self.output_dir, "figures")
        self.input_dir = input_path
        os.makedirs(self.fig_dir, exist_ok=True)
        # 修正：原本只建立了 fig_dir，self.input_dir 從未被建立過。
        # calc_psd_fitting()/calc_flatness_fitting() 會把快取 pickle 寫進 self.input_dir，
        # 如果這個資料夾不存在，pickle.dump 會丟 FileNotFoundError，
        # 而外層又用了寬鬆的 except Exception，導致快取「靜默地」存不進去。
        os.makedirs(self.input_dir, exist_ok=True)
        # ======================================

    def load_Mkp(self, file_path, set_time_index=True):
        """
        載入由數據源生成的 MKP CSV 檔案，並儲存完整的資料與壓力張量。

        :param file_path: CSV 檔案的路徑
        :param set_time_index: 是否將 Time_Index 設為 DataFrame 的 Index
        """
        print("Loading MKP data...")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到檔案：{file_path}")

        # 1. 讀取 CSV 檔案
        if set_time_index:
            # read_csv 會自動將資料轉換為帶有 P_xx, P_yy 等欄位的表格
            df = pd.read_csv(file_path, parse_dates=['Time_Index'])
            df = df.set_index('Time_Index')
        else:
            df = pd.read_csv(file_path)

        # 2. 儲存完整的原始資料 (包含 folder, epsilon, M_kp, Np 等)
        self.results['Mkp'] = df

        name_mapping = {
        "P_xx": "Pr",
        "P_yy": "Pt",
        "P_zz": "Pn",
        "P_xy": "P_rt",
        "P_xz": "P_rn",
        "P_yz": "P_tn"
        }

        self.data['P_tensor'] = df[list(name_mapping.keys())].rename(columns=name_mapping, errors='ignore')
        #######load and handle proton velocity
        # 1. 建立一個全新的子 DataFrame，並完美繼承原本 df 的 Time Index
        proton_v_df = pd.DataFrame(index=df.index)

        # 2. 把計算好的 RTN 速度塞進這個子 DataFrame
        proton_v_df['Vr'] = -df['Vx']
        proton_v_df['Vt'] = -df['Vy']
        proton_v_df['Vn'] = df['Vz']
        proton_v_df['V_magnitude'] = np.sqrt(df['Vx']**2 + df['Vy']**2 + df['Vz']**2)

        # 3. 最後把整個 DataFrame 當成一個物件，塞進 self.data 字典中
        self.data['proton_velocity'] = proton_v_df

        print("MKP data loaded successfully.")
        print(f"Stored full data in self.results['Mkp'] (Shape: {df.shape})")
        print(f"Stored tensor columns in self.data['P_tensor']")

    def load_data(self, start=None, end=None):
        """步驟 1: 載入並預處理資料"""
        print("loading MAG and SWA data...")

        self.data = {} # 清空之前的資料
        self.params = {} # 清空之前的參數

        # 這裡套用你原本的 import 函式
        df_raw = load_and_preprocess_mag(mag_path).set_index('Time_UTC')
        df_raw_v, df_raw_p, df_raw_t = load_and_preprocess_v_p_n_t(swa_path)
        vdf, Energy, Azimuth, Elevation,velocity, timeVdf = load_vdf(vdf_path)

        self.load_Mkp(Mkp_path, set_time_index=True)

        df_raw_v = df_raw_v.set_index('Time_UTC')
        df_raw_p = df_raw_p.set_index('Time_UTC')
        df_raw_t = df_raw_t.set_index('Time_UTC')

        # 設定時間切片 (這裡可改為互動式輸入)
        #start, end = '2022-02-24 00:00:01', '2022-02-24 07:49:30'
        #start, end = '2022-02-24 15:00:01', '2022-02-24 21:30:30'
        df_selected = df_raw.loc[start : end].copy()
        df_selected_v = df_raw_v.loc[start : end].copy()
        df_selected_p = df_raw_p.loc[start : end].copy()
        df_selected_t = df_raw_t.loc[start : end].copy()
        df_selected_n = df_raw_v['N'].loc[start : end].copy()

        # align time
        df_selected = get_resampled_data(df_selected.reset_index())
        df_selected_v = get_resampled_data(df_selected_v.reset_index(),freq="4s")
        df_selected_p = get_resampled_data(df_selected_p.reset_index(),freq="4s")
        df_selected_t = get_resampled_data(df_selected_t.reset_index(),freq="4s")
        df_selected_n = get_resampled_data(df_selected_n.reset_index(),freq="4s")

        # if self.data.get('proton_velocity') is not None:
        #     df_selected_v = self.data['proton_velocity'].copy()
        #     df_selected_v.index = df_selected_v.index.tz_localize(None)
        #     start_time = max(df_selected_v.index.min(),df_selected.index.min())
        #     end_time = min(df_selected_v.index.max(),df_selected.index.max())
        #     df_selected_v = df_selected_v.loc[start_time:end_time]
        #     df_selected = df_selected.loc[start_time:end_time]

        # field alignment
        df_rotated, B_matrix = convert_FA(df_selected)
        #print(B_matrix)
        df_rotated_v, matrix = convert_FA(df_selected_v,columns=['Vr', 'Vt', 'Vn'], output_prefix= 'V',B_matrix=B_matrix)

        # 儲存到物件屬性
        self.data['start'] = start
        self.data['end'] = end
        self.data['raw_mag'] = df_raw
        self.data['raw_v'] = df_raw_v
        self.data['raw_p'] = df_raw_p
        self.data['raw_t'] = df_raw_t
        self.data['mag'] = df_selected
        self.data['v'] = df_selected_v
        self.data['n'] = df_selected_n
        self.data['p'] = df_selected_p
        self.data['t'] = df_selected_t
        self.data['rotated_mag'] = df_rotated
        self.data['rotated_v'] = df_rotated_v
        # 新增：保存場向對齊旋轉矩陣，calc_plasma_params() 用它算 theta_BR/theta_BV
        self.data['B_matrix'] = B_matrix

        self.params['Elevation'] = Elevation
        self.params['Azimuth'] = Azimuth
        self.params['Energy'] = Energy
        self.params['velocity'] = velocity

        print(f"loading and prehandling complete。time series from {start} to {end}")

    def load_pickle_data(self, pkl_path):
        """額外功能：從 Pickle 檔案載入之前計算好的全域小波資料"""
        # 修正：原本這裡會立刻用寫死的路徑覆蓋掉傳入的 pkl_path 參數，
        # 導致呼叫端永遠無法指定別的檔案。改為：優先使用傳入路徑，沒傳才用預設值。
        if not pkl_path:
            pkl_path = r'C:\Users\Linch\OneDrive\文件\assignment\MS\program\wavelet_analysis_results.pkl'
        if not os.path.exists(pkl_path):
            print(f"錯誤：指定的 Pickle 檔案不存在: {pkl_path}")
            return None

        print(f"正在從 Pickle 檔案載入全域小波資料...")
        with open(pkl_path, 'rb') as f:
            loaded_data = pickle.load(f)
            self.data['LIM'] = loaded_data['LIM']
            self.data['time_axis_dt'] = loaded_data['time_axis_dt']
            self.data['freqs'] = loaded_data['freqs']
            self.data['coi'] = loaded_data['coi']
            self.data['signal'] = loaded_data['signal']
        print(f"成功從 Pickle 檔案載入全域小波資料！")
        return loaded_data

    def calc_plasma_params(self):
        """步驟 2: 計算等離子體物理參數"""
        if 'rotated_mag' not in self.data:
            print("錯誤：請先執行 'load' 載入資料。")
            return

        n_data = self.data['n']
        if isinstance(n_data, pd.DataFrame):
            # 如果是 DataFrame，指定拿 'N' 欄位（或第一欄）算平均，並轉成 float
            col_name = 'N' if 'N' in n_data.columns else n_data.columns[0]
            mean_N = float(n_data[col_name].mean())
        elif isinstance(n_data, pd.Series):
            mean_N = float(n_data.mean())
        else:
            mean_N = float(n_data)
        mean_B = self.data['rotated_mag'].mean()
        mean_v = self.data['rotated_v'].mean()
        mean_t = self.data['t'].mean()

        # 使用 PlasmaPy 計算
        self.params['plasma_freq'] = plasmapy.formulary.frequencies.plasma_frequency(mean_N * u.cm**-3, particle="p+", to_hz=True)
        self.params['gyro_freq'] = plasmapy.formulary.frequencies.gyrofrequency(mean_B['B_magnitude'] * u.nT, particle="p+", to_hz=True)
        self.params['gyro_period'] = 1 / self.params['gyro_freq']
        self.params['i_l'] = plasmapy.formulary.lengths.inertial_length(mean_N * u.cm**-3, particle="p+").to(u.km)
        self.params['i_t'] = self.params['i_l'] / (mean_v['V_magnitude'] * u.km/u.s)
        self.params['beta'] = plasmapy.formulary.dimensionless.beta(mean_t['T'] *u.eV, mean_N * u.cm**-3, mean_B['B_magnitude']*u.nT)
        self.params['f_gyro'] = self.params['gyro_freq'].value if hasattr(self.params['gyro_freq'], 'value') else self.params['gyro_freq']
        self.params['t_inertial'] = self.params['i_t'].value if hasattr(self.params['i_t'], 'value') else self.params['i_t']

        # 新增：theta_BR / theta_BV。print_all_params() 一直在讀這兩個值，
        # 但先前整個程式沒有任何地方計算或儲存它們，呼叫 status 一定會 KeyError。
        # rot_matrix 的第一列 (convert_FA 回傳的 B_matrix[0]) 就是平均磁場方向的 RTN 單位向量 e_para，
        # 所以跟 R 軸 ([1,0,0]) 的夾角、跟平均質子速度方向的夾角都能直接算出來。
        if 'B_matrix' in self.data:
            e_para = self.data['B_matrix'][0]
            theta_BR = np.degrees(np.arccos(np.clip(e_para[0], -1.0, 1.0)))

            v_rtn_mean = np.array([
                self.data['v']['Vr'].mean(),
                self.data['v']['Vt'].mean(),
                self.data['v']['Vn'].mean(),
            ])
            v_norm = np.linalg.norm(v_rtn_mean)
            if v_norm > 0:
                theta_BV = np.degrees(np.arccos(np.clip(np.dot(e_para, v_rtn_mean) / v_norm, -1.0, 1.0)))
            else:
                theta_BV = np.nan

            self.data['theta_BR'] = theta_BR
            self.data['theta_BV'] = theta_BV
        else:
            print("⚠ 找不到 B_matrix（尚未執行 load_data），無法計算 theta_BR/theta_BV。")

        print(f"plasma parameters：Plasma Freq = {self.params['plasma_freq']:.2f},proton Gyro Frequency = {self.params['gyro_freq']:.2f},proton Gyro period = {self.params['gyro_period']:.2f}, Inertial Length = {self.params['i_l']:.2f}, Inertial Time = {self.params['i_t']:.2f}, Beta = {self.params['beta']:.2f}")

    def calc_turbulence_params(self):
        """步驟 3: 計算湍流相關參數 (如 FCI, ACF, PDF, Correlation Time, Structure Function)"""
        if 'rotated_mag' not in self.data:
            print("錯誤：請先執行 'load' 載入資料。")
            return
        # 修正：原本這裡檢查的是 self.data 有沒有 'correlation_time'/'acf'，
        # 但實際要初始化、之後也一直在讀寫的是 self.params['correlation_time']/self.params['acf']。
        # 'correlation_time' 這個 key 永遠不會出現在 self.data 裡，
        # 所以判斷式恆為 True，等於每次呼叫這個方法都會把之前累積的結果整個清空重來。
        if 'correlation_time' not in self.params:
            self.params['correlation_time'] = {}
        if 'acf' not in self.params:
            self.params['acf'] = {}
        if 'elsasser' not in self.data:
            self.data['elsasser'] = {}

        # compute correlation time for each component
        df_conv = self.data['rotated_mag']
        components_conv = ['B_magnitude','B_para', 'B_perp1', 'B_perp2']
        # 修正：Trace_Mag 原本是把 B_magnitude 也一起加進平方和開根號，
        # 但 B_magnitude^2 本來就等於 B_para^2+B_perp1^2+B_perp2^2 (場向對齊分解)，
        # 等於把模長重複算了一次 (Trace_Mag = sqrt(2)*B_magnitude，不是有意義的物理量)。
        # Trace 應該只由三個正交分量組成。
        trace_components_B = ['B_para', 'B_perp1', 'B_perp2']
        trace_data = np.sqrt((df_conv[trace_components_B]**2).sum(axis=1)) # 向量模長差值
        df_conv['Trace_Mag'] = trace_data # 存入 DF 方便後續使用
        for comp in components_conv + ['Trace_Mag']:
            acf, lags, tc = compute_correlation_time(df_conv[comp].values, sampling_rate=8)

            self.params['correlation_time'][comp] = tc
            self.params['acf'][comp] = (acf, lags)
            print(f"Component {comp} Correlation Time: {tc:.2f} s")

        df_conv_v = self.data['rotated_v']
        components_conv_V = ['V_magnitude','V_para', 'V_perp1', 'V_perp2']
        trace_components_V = ['V_para', 'V_perp1', 'V_perp2']
        trace_data = np.sqrt((df_conv_v[trace_components_V]**2).sum(axis=1)) # 向量模長差值
        df_conv_v['Trace_V'] = trace_data # 存入 DF 方便後續使用
        for comp in components_conv_V + ['Trace_V']:
            acf, lags, tc = compute_correlation_time(df_conv_v[comp].values, sampling_rate=0.25)

            self.params['correlation_time'][comp] = tc
            self.params['acf'][comp] = (acf, lags)
            print(f"Component {comp} Correlation Time: {tc:.2f} s")

        # compute elasser variables
        dV, dB, z_p, z_n = preprocess_elsasser_variable(self.data['rotated_mag'], self.data['rotated_v'], self.data['n'])
        self.data['elsasser'] = {
            'dV': dV,
            'dB': dB,
            'z_p': z_p,
            'z_n': z_n
        }

    def run_psd(self):
        """步驟 3: 計算並繪製 PSD"""
        print(" PSD computing...")
        components_Z = ['V_para', 'V_perp1', 'V_perp2']
        components_conv = ['B_magnitude','B_para', 'B_perp1', 'B_perp2']
        components_conv_V = ['V_magnitude','V_para', 'V_perp1', 'V_perp2']
        # 修正：components_conv/components_conv_V 裡混了模長 (B_magnitude/V_magnitude)
        # 跟三個正交分量，如果直接把整個 list 拿去加總算 'Trace'，會跟上面
        # Trace_Mag/Trace_V 一樣重複計入模長。這裡额外傳入只含三個正交分量的
        # trace_components，讓 PSD 的 'Trace' 欄位維持正確，同時仍然畫得出
        # B_magnitude/V_magnitude 各自的頻譜（診斷用）。
        self.results['psd_mag'] = process_multi_window_psd(self.data['rotated_mag'], components_conv, trace_components=['B_para', 'B_perp1', 'B_perp2'])

        self.results['psd_v'] = process_multi_window_psd(self.data['rotated_v'], components_conv_V, sampling_rate=0.25, trace_components=['V_para', 'V_perp1', 'V_perp2'])

        self.results['psd_z_p'] = process_multi_window_psd(self.data['elsasser']['z_p'], components_Z, sampling_rate=0.25)
        self.results['psd_z_n'] = process_multi_window_psd(self.data['elsasser']['z_n'], components_Z, sampling_rate=0.25)

        self.results['psd_dB'] = process_multi_window_psd(self.data['elsasser']['dB'], ['B_para', 'B_perp1', 'B_perp2'], sampling_rate=0.25)
        self.results['psd_dV'] = process_multi_window_psd(self.data['elsasser']['dV'], ['V_para', 'V_perp1', 'V_perp2'], sampling_rate=0.25)
        print("PSD analysis complete.")

    def load_or_run_psd_fitting(self):
            """
            自動檢查是否存在歷史擬合紀錄：
            - 若有，直接載入歷史數據，不中斷自動化流程。
            - 若無，則啟動互動式擬合介面供使用者手動擬合。
            """
            cache_file = os.path.join(self.input_dir, f"PSD_Fitting_Cache_{SAVE_DATE}.pkl")

            if os.path.exists(cache_file):
                print(f"\n✨ [快取命中] 發現歷史擬合紀錄檔案：\n📂 {cache_file}")
                try:
                    with open(cache_file, 'rb') as f:
                        self.params['PSD_fit'] = pickle.load(f)
                    print("🎉 成功載入歷史擬合參數！跳過互動式擬合。")

                    # 顯示目前載入了哪些歷史分量
                    for k, v in self.params['PSD_fit'].items():
                        print(f"  - {k}: 已載入 {len(v)} 段擬合區間")
                    return
                except Exception as e:
                    print(f"⚠ 讀取快取檔案失敗 ({e})，將重新啟動互動式擬合...")

            # 如果快取檔案不存在或讀取失敗，執行原本的手動擬合
            print("\n🔍 未找到歷史擬合紀錄，啟動互動式擬合工具...")
            self.calc_psd_fitting()

    def load_or_run_flatness_fitting(self):
        """
        自動檢查是否存在歷史平坦度擬合快取：
        - 若有，直接載入歷史數據，不中斷自動化流程。
        - 若無，則啟動互動式擬合介面供使用者手動擬合。

        修正：原本這個方法只有「快取存在」的分支，如果快取不存在或讀取失敗，
        函式就直接結束，從來不會呼叫 calc_flatness_fitting()。
        跟同樣模式的 load_or_run_psd_fitting() 對照就看得出不對稱——
        PSD 那邊沒快取時會自動啟動互動擬合，flatness 這邊卻悄悄跳過，
        導致全新環境第一次跑 auto_load 時，平坦度擬合永遠不會發生。
        """
        cache_file = os.path.join(self.input_dir, f"Flatness_Fitting_Cache_{SAVE_DATE}.pkl")
        if os.path.exists(cache_file):
            print(f"✨ [快取命中] 發現歷史平坦度擬合紀錄檔案：\n📂 {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    self.params['flatness_fit'] = pickle.load(f)
                print("🎉 成功載入歷史平坦度擬合參數！")
                return
            except Exception as e:
                print(f"⚠ 讀取快取檔案失敗 ({e})，將重新啟動互動式擬合...")

        print("\n🔍 未找到歷史平坦度擬合紀錄，啟動互動式擬合工具...")
        self.calc_flatness_fitting()

    def calc_psd_fitting(self, adjust=True):
        """
        多分量、多區段互動式 PSD 擬合工具：
        支援自選數據源類型 (磁場 或 速度場) 與分量，並支援單一分量紀錄多段 Fitting 結果。
        """
        # ==========================================
        # 1. 選擇要擬合的數據源類型 (Data Source Type)
        # ==========================================
        print("\n==================================================")
        print("⚡ 請選擇你想進行擬合的數據源類型：")
        print("   [1] 磁場數據 (Magnetic Field PSD)")
        print("   [2] 速度數據 (Velocity PSD)")
        print("==================================================")

        while True:
            type_choice = input("請輸入對應的數字編號 (1 或 2) >>> ").strip()
            if type_choice == '1':
                data_key = 'psd_mag'
                source_label = "Magnetic"
                break
            elif type_choice == '2':
                data_key = 'psd_v'
                source_label = "Velocity"
                break
            else:
                print("❌ 輸入錯誤，請輸入 1 或 2！")

        if data_key not in self.results:
            print(f"❌ 錯誤：找不到 {source_label} 的 PSD 數據，請先確定是否已執行相關分析。")
            return

        # 確保 PSD_fit 字典存在
        if 'PSD_fit' not in self.params:
            self.params['PSD_fit'] = {}

        # ==========================================
        # 2. 選擇要擬合的數據分量 (Component)
        # ==========================================
        available_comps = list(self.results[data_key].columns)

        print("\n==================================================")
        print(f"📊 請選擇你想進行擬合的【{source_label}】數據分量 (Component)：")
        for idx, comp_name in enumerate(available_comps):
            # 建立一個唯一的複合 Key (例如: "Magnetic_Trace_Mag" 或 "Velocity_V_para") 避免分量名稱衝突
            unique_key = f"{source_label}_{comp_name}"
            existing_count = len(self.params['PSD_fit'].get(unique_key, []))
            status = f" (📈 已有 {existing_count} 段擬合紀錄)" if existing_count > 0 else ""
            print(f"   [{idx + 1}] {comp_name}{status}")
        print("==================================================")

        while True:
            comp_choice = input("請輸入對應的數字編號 >>> ").strip()
            try:
                comp_idx = int(comp_choice) - 1
                if 0 <= comp_idx < len(available_comps):
                    target_comp = available_comps[comp_idx]
                    unique_key = f"{source_label}_{target_comp}"
                    print(f"🎯 已選擇目標分量：【{source_label} -> {target_comp}】")
                    break
                else:
                    print("❌ 編號超出範圍，請重新輸入！")
            except ValueError:
                print("❌ 請輸入正確的數字編號！")

        # 初始化該唯一分量的紀錄列表 (如果之前沒有的話)
        if unique_key not in self.params['PSD_fit']:
            self.params['PSD_fit'][unique_key] = []

        # ==========================================
        # 3. 互動擬合與驗證循環
        # ==========================================
        while True:
            # 彈出原始 PSD 圖像
            print(f"\n📢 正在開啟【{source_label} - {target_comp}】的原始 PSD 圖表，請觀察曲線並規劃你的 Fitting 範圍...")
            # 這裡沿用你原本傳入的參數，並指定繪製選定的數據源與分量
            plot_psd_with_markers(self.results[data_key], self.params['correlation_time'], self.params['f_gyro'], self.params['t_inertial'], type=target_comp)
            plt.title(f"Observe PSD for {source_label}-{target_comp} (Close window to enter ranges)")
            plt.show()

            # 輸入頻率範圍
            print(f"\n👉 請輸入【{target_comp}】的擬合起始頻率 (例如: 10e-4):")
            try:
                start_input = input("Start Frequency >>> ").strip()
                f_start = float(start_input)

                print(f"👉 請輸入【{target_comp}】的擬合結束頻率 (例如: 1e-1):")
                end_input = input("End Frequency >>> ").strip()
                f_end = float(end_input)
            except ValueError:
                print("❌ 輸入格式錯誤！請重新輸入數字。")
                plt.close('all')
                continue

            # 計算自訂非線性擬合
            print(f"⚙️ 正在針對【{target_comp}】區間 [{f_start} ~ {f_end}] Hz 進行擬合計算...")
            fit_custom = compute_PSD_fitting(self.results[data_key][target_comp], fit_range=(f_start, f_end))

            # 彈出雙方案補償驗證圖
            # 注意：為配合原繪圖函式，這裡臨時用 target_comp 作為 key 傳入
            temp_params = {target_comp: fit_custom}

            plot_psd_with_markers(self.results[data_key], self.params['correlation_time'], self.params['f_gyro'], self.params['t_inertial'], type=target_comp, fitting_params=temp_params)

            plt.title(f"Fig 1: {source_label} {target_comp} Fitting Spectrum")

            plot_fitting_check(self.results[data_key], target_comp, fit_custom)
            plt.title(f"Fig 2: {source_label} {target_comp} Compensated Spectrum (Validation)")

            print("📊 圖像已生成。請檢查 [Fig 2] 的藍色補償譜是否完美呈現水平平台。")
            plt.show()

            # ==========================================
            # 4. 儲存機制
            # ==========================================
            print(f"\n==================================================")
            print(f"🤔 檢查結果（區間: {f_start} ~ {f_end} Hz）：")
            print(f"   當前擬合斜率 (Slope): {fit_custom['slope']:.2f} ± {fit_custom['slope_error']:.2f}")
            print(f"==================================================")
            print("   [1] 滿意結果，存入該分量的多段紀錄中")
            print("   [2] 不滿意，放棄本次計算並重新調整區間")

            choice = input("請選擇 (1/2) >>> ").strip()

            if choice == '1':
                # 將這次的擬合結果追加到唯一的 unique_key 列表中
                self.params['PSD_fit'][unique_key].append(fit_custom)

                current_segments_num = len(self.params['PSD_fit'][unique_key])
                print(f"💾 [成功] 已將此波段存為【{source_label}_{target_comp}】的第 {current_segments_num} 段紀錄！")

                cache_file = os.path.join(self.input_dir, f"PSD_Fitting_Cache_{SAVE_DATE}.pkl")
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(self.params['PSD_fit'], f)
                    print(f"💾 [快取更新] 最新擬合結果已同步備份至快取。")
                except Exception as e:
                    print(f"⚠ 快取備份失敗: {e}")

                more = input(f"\n➕ 是否要繼續對【{source_label}_{target_comp}】進行其他波段的擬合？(y/n) >>> ").strip().lower()
                if more != 'y':
                    break
                else:
                    plt.close('all')
            else:
                print("🔄 [放棄] 已捨棄本次計算，重新啟動區間輸入...")
                plt.close('all')

    def calc_psd_fitting_auto(self, components=None, f_search=(9e-4, 9e-2),
                               slope_tol=0.15, min_points=8, save_check_fig=True):
        """
        非互動式自動擬合：不用 input() 讓人手動選頻段，改用
        auto_fit.auto_detect_fit_range() 在指定搜尋範圍內自動找出「局部斜率
        最穩定」的連續頻段，做為慣性尺度冪律擬合區間。

        適合 agent / 批次腳本一次跑完多個事件；每個分量最多新增一段擬合紀錄，
        沿用既有的快取機制 (self.params['PSD_fit'] + pickle 備份)，跟手動流程
        完全相容 —— 之後照樣可以用 plot_psd(auto_save=True) 出圖、
        export_fitting_data() 匯出。

        :param components: {'Magnetic': [...], 'Velocity': [...]}，指定要
            擬合的分量；預設 None 代表兩個資料源的所有欄位都嘗試自動擬合。
        :param f_search: (f_min, f_max)，自動搜尋的頻率上下界 (Hz)。
            預設對齊既有 K41 參考線範圍 (0.0009 ~ 0.09 Hz)。
        :param slope_tol: 局部斜率標準差門檻，越小要求越嚴格。
        :param min_points: 候選區間最少局部斜率點數。
        :param save_check_fig: 是否把補償譜驗證圖存檔，方便事後人工複核
            自動擬合結果（強烈建議論文用的數值都要複核過）。
        """
        if 'PSD_fit' not in self.params:
            self.params['PSD_fit'] = {}

        sources = {
            'Magnetic': ('psd_mag', self.results.get('psd_mag')),
            'Velocity': ('psd_v', self.results.get('psd_v')),
        }

        cache_file = os.path.join(self.input_dir, f"PSD_Fitting_Cache_{SAVE_DATE}.pkl")
        any_success = False

        for source_label, (data_key, psd_df) in sources.items():
            if psd_df is None:
                print(f"⚠ 跳過 {source_label}：找不到 {data_key}，請先執行 run_psd()。")
                continue

            target_comps = components.get(source_label) if components else list(psd_df.columns)

            for target_comp in target_comps:
                if target_comp not in psd_df.columns:
                    print(f"⚠ 跳過 {source_label}-{target_comp}：欄位不存在。")
                    continue

                unique_key = f"{source_label}_{target_comp}"
                series = psd_df[target_comp].dropna()

                candidate = auto_detect_fit_range(
                    series.index.values, series.values,
                    f_min=f_search[0], f_max=f_search[1],
                    slope_tol=slope_tol, min_points=min_points,
                )

                if candidate is None:
                    print(f"❌ {unique_key}：在 {f_search[0]}~{f_search[1]} Hz 範圍內找不到符合條件 "
                          f"(std<={slope_tol}, n>={min_points}) 的單一冪律區段，略過。")
                    continue

                f_start, f_end = candidate['start'], candidate['end']
                print(f"🎯 {unique_key}：自動偵測擬合區間 [{f_start:.5f}, {f_end:.5f}] Hz "
                      f"(局部斜率 {candidate['mean_slope']:.2f} ± {candidate['std_slope']:.2f}, "
                      f"{candidate['n_points']} 點)")

                fit_result = compute_PSD_fitting(series, fit_range=(f_start, f_end))

                if unique_key not in self.params['PSD_fit']:
                    self.params['PSD_fit'][unique_key] = []
                self.params['PSD_fit'][unique_key].append(fit_result)
                any_success = True

                print(f"   ✅ 擬合斜率 (Slope): {fit_result['slope']:.2f} ± {fit_result['slope_error']:.2f}")

                if save_check_fig:
                    plot_fitting_check(psd_df, target_comp, fit_result)
                    seg_num = len(self.params['PSD_fit'][unique_key])
                    fig_path = os.path.join(
                        self.fig_dir, f"auto_fit_check_{unique_key.lower()}_seg{seg_num}.png"
                    )
                    plt.savefig(fig_path, bbox_inches='tight', dpi=150)
                    plt.close('all')
                    print(f"   💾 驗證圖已存檔：{fig_path}")

        if any_success:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.params['PSD_fit'], f)
                print(f"💾 [快取更新] 自動擬合結果已同步備份至：\n📂 {cache_file}")
            except Exception as e:
                print(f"⚠ 快取備份失敗: {e}")
        else:
            print("⚠ 本次沒有任何分量成功自動擬合。")

    def export_fitting_data(self, format='json'):
        """
        將 self.params['PSD_fit'] 中儲存的多數據源、多分量、多區段擬合數據匯出成檔案。
        """
        if 'PSD_fit' not in self.params or not self.params['PSD_fit']:
            print("⚠ 提示：目前系統中沒有任何已儲存的擬合數據（PSD_fit 爲空）。")
            return

        flattened_records = []

        # 遍歷所有複合 Key (如 Magnetic_Trace_Mag, Velocity_V_para 等)
        for unique_key, segments in self.params['PSD_fit'].items():
            # 自動拆分出 數據源類型 與 實際分量名稱
            if "_" in unique_key:
                data_type, comp_name = unique_key.split("_", 1)
            else:
                data_type = "Unknown"
                comp_name = unique_key

            for seg_idx, seg_data in enumerate(segments):
                record = {
                    'DataType': data_type,          # 新增：區分 Magnetic 或 Velocity
                    'Component': comp_name,
                    'Segment_ID': seg_idx + 1,
                    'Freq_Start_Hz': seg_data['start'],
                    'Freq_End_Hz': seg_data['end'],
                    'Slope_Alpha': seg_data['slope'],
                    'Slope_Error_1Sigma': seg_data['slope_error'],
                    'Intercept_LogA': seg_data['intercept'],
                    'Local_Slope_Mean': np.mean(seg_data['d(log_e)/d(log_f)']),
                    'Local_Slope_SE': seg_data['local_slope_error'],
                    'Total_Data_Points': seg_data['n_points']
                }
                flattened_records.append(record)

        df_export = pd.DataFrame(flattened_records)
        file_base_path = os.path.join(self.input_dir, f"PSD_Fitting_Results_{SAVE_DATE}")

        if format.lower() == 'csv':
            file_path = f"{file_base_path}.csv"
            df_export.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"🎉 [匯出成功] 擬合數據已儲存為 CSV 檔：\n📂 {file_path}")
        elif format.lower() == 'json':
            file_path = f"{file_base_path}.json"
            df_export.to_json(file_path, orient='records', indent=4)
            print(f"🎉 [匯出成功] 擬合數據已儲存為 JSON 檔：\n📂 {file_path}")

        return df_export

    def plot_psd(self, auto_save=False):
        """
        繪製 PSD 圖表，並修正原本只抓 [-1] 的問題：
        改為傳遞完整的多段擬合段落清單 (Full List of Segments) 給繪圖常式。
        """
        print("Plotting (1) Magnetic field PSD (2) Velocity PSD with ALL segments...")

        if auto_save:
            # 建立一個結構來幫磁場與速度場收集它們各自的擬合參數
            # 格式優化為：{'psd_mag': {分量名: [擬合區段1, 擬合區段2, ...]}, 'psd_v': {...}}
            compiled_fits = {'psd_mag': {}, 'psd_v': {}}

            # 1. 巡覽快取中所有的唯一 Key (例如: "Magnetic_B_para", "Velocity_Trace")
            if 'PSD_fit' in self.params and self.params['PSD_fit']:
                for unique_key, segments in self.params['PSD_fit'].items():
                    if not segments or "_" not in unique_key:
                        continue

                    # 拆解複合 Key 取得數據源與實際分量名稱
                    source_label, comp_name = unique_key.split("_", 1)
                    dict_key = 'psd_mag' if source_label == "Magnetic" else 'psd_v'

                    # 【關鍵修正】：不再傳送 segments[-1]，直接把完整的多段清單塞過去！
                    compiled_fits[dict_key][comp_name] = segments

            # 2. 自動繪製並儲存【磁場】的所有已擬合分量
            mag_fit_params = compiled_fits['psd_mag']
            mag_comps_to_plot = list(mag_fit_params.keys()) if mag_fit_params else ['Trace']

            for comp in mag_comps_to_plot:
                print(f"💾 正在自動匯出磁場分量多段擬合圖表: {comp}...")
                plot_psd_with_markers(
                    self.results['psd_mag'],
                    self.params['correlation_time'],
                    self.params['f_gyro'],
                    self.params['t_inertial'],
                    type=comp,
                    fitting_params=mag_fit_params  # 傳遞包含完整多段 List 的字典
                )
                plt.savefig(os.path.join(self.fig_dir, f"psd_magnetic_{comp.lower()}.png"), bbox_inches='tight', dpi=150)
                plt.close('all')  # 釋放記憶體

            # 3. 自動繪製並儲存【速度場】的所有已擬合分量
            vel_fit_params = compiled_fits['psd_v']
            vel_comps_to_plot = list(vel_fit_params.keys()) if vel_fit_params else ['Trace']

            for comp in vel_comps_to_plot:
                print(f"💾 正在自動匯出速度分量多段擬合圖表: {comp}...")
                plot_psd_with_markers(
                    self.results['psd_v'],
                    self.params['correlation_time'],
                    self.params['f_gyro'],
                    self.params['t_inertial'],
                    type=comp,
                    fitting_params=vel_fit_params  # 傳遞包含完整多段 List 的字典
                )
                plt.savefig(os.path.join(self.fig_dir, f"psd_velocity_{comp.lower()}.png"), bbox_inches='tight', dpi=150)
                plt.close('all')

            return
        else:
            # 修正：原本 auto_save=False 時整個方法沒有 else 分支，是完全的 no-op
            # （呼叫了也不會畫任何東西）。這裡補上互動顯示模式，畫出 Trace 分量給使用者看。
            mag_fit_params = {}
            vel_fit_params = {}
            if 'PSD_fit' in self.params and self.params['PSD_fit']:
                for unique_key, segments in self.params['PSD_fit'].items():
                    if not segments or "_" not in unique_key:
                        continue
                    source_label, comp_name = unique_key.split("_", 1)
                    if source_label == "Magnetic":
                        mag_fit_params[comp_name] = segments
                    else:
                        vel_fit_params[comp_name] = segments

            if 'psd_mag' in self.results:
                plot_psd_with_markers(self.results['psd_mag'], self.params['correlation_time'], self.params['f_gyro'], self.params['t_inertial'], type='Trace', fitting_params=mag_fit_params)
                plt.title("Magnetic Trace PSD")
            if 'psd_v' in self.results:
                plot_psd_with_markers(self.results['psd_v'], self.params['correlation_time'], self.params['f_gyro'], self.params['t_inertial'], type='Trace', fitting_params=vel_fit_params)
                plt.title("Velocity Trace PSD")
            plt.show()

    def run_structure_function(self, auto_save=True):
        """步驟 4: 計算、儲存並繪製結構函數與平坦度 (Flatness)"""
        print("Structure Function & Flatness computing...")

        if 'sf' not in self.results:
            self.results['sf'] = {}

        components_to_analyze = {
            'Magnetic': ['B_para', 'B_perp1', 'B_perp2'],
            'Velocity': ['V_para', 'V_perp1', 'V_perp2']
        }

        for source_label, comps in components_to_analyze.items():
            df_key = 'rotated_mag' if source_label == 'Magnetic' else 'rotated_v'

            if df_key not in self.data:
                print(f"⚠ 提示：找不到 {source_label} 的旋轉資料欄位 ({df_key})，跳過此處。")
                continue

            for comp in comps:
                if comp not in self.data[df_key].columns:
                    continue

                unique_key = f"{source_label}_{comp}"
                print(f"  -> Processing component: {unique_key}")

                s_rate = 8.0 if source_label == 'Magnetic' else 0.25

                # Compute structure functions
                scales, sf_data = compute_turbulence_structure_function(
                    self.data[df_key][comp],
                    sampling_rate=s_rate
                )

                # --- NEW: Compute Flatness F(tau) = S4 / (S2)^2 ---
                s2 = np.array(sf_data[2])
                s4 = np.array(sf_data[4])

                # Avoid division by zero warnings
                with np.errstate(divide='ignore', invalid='ignore'):
                    flatness = s4 / (s2 ** 2)

                # Save scales, sf_data, and flatness together
                self.results['sf'][unique_key] = {
                    'scales': scales,
                    'sf_data': sf_data,
                    'flatness': flatness
                }

        print("Structure Function & Flatness analysis complete.")

        if auto_save:
            self.plot_all_structure_functions(save_to_disk=True)
            self.plot_all_flatness(save_to_disk=True)  # New plotting call

    def plot_all_structure_functions(self, save_to_disk=True):
        """
        將儲存於系統內的所有分量結構函數 (1~4階) 繪製於獨立圖表中並完美對齊輸出。
        """
        if 'sf' not in self.results or not self.results['sf']:
            print("❌ 錯誤：查無結構函數/平坦度計算紀錄，請先執行 run_structure_function()。")
            return

        print("Plotting structure functions for all components...")

        styles = {
            1: {'color': '#1f77b4', 'marker': 'o', 'label': '$S_1(\\tau) = \\langle|\\Delta|\\rangle$'},
            2: {'color': '#ff7f0e', 'marker': 's', 'label': '$S_2(\\tau) = \\langle\\Delta^2\\rangle$'},
            3: {'color': '#2ca02c', 'marker': '^', 'label': '$S_3(\\tau) = |\\langle\\Delta^3\\rangle|$ (Abs)'},
            4: {'color': '#d62728', 'marker': 'd', 'label': '$S_4(\\tau) = \\langle\\Delta^4\\rangle$'}
        }

        for unique_key, payload in self.results['sf'].items():
            scales = payload['scales']
            sf_data = payload['sf_data']

            # 從 "Magnetic_B_para" 或 "Velocity_V_para" 中萃取出正確的組成分量字串
            # 以便與 self.params['correlation_time'] 中的 Key 對齊
            comp_name = unique_key.split("_", 1)[1] if "_" in unique_key else unique_key

            fig, ax = plt.subplots(figsize=(8, 6))

            # 依序疊加 1 至 4 階曲線
            for order in [1, 2, 3, 4]:
                y_vals = np.array(sf_data[order])
                if order == 3:
                    y_vals = np.abs(y_vals)

                ax.loglog(scales, y_vals,
                           linestyle='-',
                           linewidth=1.2,
                           color=styles[order]['color'],
                           marker=styles[order]['marker'],
                           markersize=3.5,
                           alpha=0.8,
                           label=styles[order]['label'])

            # --- 理論參考斜率線 ---
            try:
                mid_idx = len(scales) // 2
                ref_x = scales[max(0, mid_idx-4) : min(len(scales), mid_idx+5)]
                ref_y2 = ref_x**(2/3) * (sf_data[2][mid_idx] / (scales[mid_idx]**(2/3)))
                ax.loglog(ref_x, ref_y2, color='black', linestyle='--', alpha=0.6, label='Slope = 2/3 ($K41$)')
            except Exception as e:
                # 修正：原本是完全裸的 except:，會連同拼字錯誤、邏輯錯誤都一併吞掉。
                # 這裡只是理論參考線的裝飾，缺資料點時跳過即可，但至少把原因印出來。
                print(f"⚠ 無法繪製 K41 參考線 ({unique_key}): {e}")

            # --- 新增：動態繪製物理時間尺度垂線與標籤 ---
            # 使用 ax.get_ylim() 動態算出文字合適的 Y 軸高度（放置於圖表底部上方一點點，避免遮擋數據）
            y_min, y_max = ax.get_ylim()
            text_y = y_min * 1.5

            # 1. 關聯時間 (Correlation Time) - 動態從字典配對
            if 'correlation_time' in self.params and comp_name in self.params['correlation_time']:
                tc = self.params['correlation_time'][comp_name]
                ax.axvline(x=tc, color="#414141", linestyle='--', alpha=0.7)
                ax.text(tc, text_y, f'correlation time ({tc:.1f}s)', color="#414141", rotation=90, va='bottom', ha='right', fontsize=8)

            # 2. 迴旋週期 (Gyro-period) - 從 Astropy 物理單位物件安全轉換為純數值
            if 'gyro_period' in self.params:
                gp = self.params['gyro_period']
                gp_val = gp.value if hasattr(gp, 'value') else gp
                ax.axvline(x=gp_val, color="#414141", linestyle='--', alpha=0.7)
                ax.text(gp_val, text_y, f'gyro-period ({gp_val:.4f}s)', color="#414141", rotation=90, va='bottom', ha='right', fontsize=8)

            # 3. 慣性時間 (Inertial Time)
            if 't_inertial' in self.params:
                ti = self.params['t_inertial']
                ti_val = ti.value if hasattr(ti, 'value') else ti
                ax.axvline(x=ti_val, color="#414141", linestyle='--', alpha=0.7)
                ax.text(ti_val, text_y, f'inertial time ({ti_val:.3f}s)', color="#414141", rotation=90, va='bottom', ha='right', fontsize=8)

            # 視窗樣式優化
            ax.set_title(f"Structure Functions - {unique_key.replace('_', ' ')}", fontsize=13, fontweight='bold')
            ax.set_xlabel("Scale $\\tau$ (seconds)", fontsize=11)
            ax.set_ylabel("Structure Function Magnitude", fontsize=11)
            ax.grid(True, which="both", ls="--", alpha=0.3)
            ax.legend(loc="upper left", fontsize=9, frameon=True)
            plt.tight_layout()

            if save_to_disk:
                filename = f"structure_function_{unique_key.lower()}.png"
                save_path = os.path.join(self.fig_dir, filename)
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                print(f"  💾 Saved plot to: {save_path}")
                plt.close()
            else:
                plt.show()

    def plot_all_flatness(self, save_to_disk=True, target_key=None):
        """
        將儲存於系統內的所有（或指定）分量平坦度 (Flatness) 繪製於獨立圖表中。
        """
        if 'sf' not in self.results or not self.results['sf']:
            print("❌ 錯誤：查無結構函數/平坦度計算紀錄，請先執行 run_structure_function()。")
            return

        print("Plotting flatness...")

        for unique_key, payload in self.results['sf'].items():
            # ======= 若指定了目標分量，跳過其他無關分量 =======
            if target_key is not None and unique_key != target_key:
                continue
            # ==================================================

            scales = payload['scales']
            flatness = payload['flatness']

            comp_name = unique_key.split("_", 1)[1] if "_" in unique_key else unique_key

            fig, ax = plt.subplots(figsize=(8, 5))

            # 繪製平坦度主線
            ax.loglog(scales, flatness,
                       linestyle='-',
                       linewidth=1.5,
                       color='#9467bd',
                       marker='d',
                       markersize=4,
                       alpha=0.8,
                       label=f'Flatness $F(\\tau)$')

            # 高斯分佈基準線 (F = 3)
            ax.axhline(y=3.0, color='r', linestyle='--', linewidth=1.2, label='Gaussian Baseline ($F=3$)')

            # 修正：這裡原本完全不知道 self.params['flatness_fit'] 的存在，
            # 所以無論是 run_structure_function(auto_save=True) 存到硬碟的圖，
            # 還是 calc_flatness_fitting() 擬合前彈出的預覽圖，都只畫得出原始
            # Flatness 曲線本身，看不到任何已接受的冪律擬合線 ——
            # 這正是「匯出的圖沒有擬合結果」的原因。這裡把已存入
            # self.params['flatness_fit'][unique_key] 的每一段擬合結果疊加上去。
            fit_segments = self.params.get('flatness_fit', {}).get(unique_key, [])
            fit_colors = ['black', '#2ca02c', '#1f77b4', '#e377c2', '#8c564b']
            for i, seg in enumerate(fit_segments):
                seg_start, seg_end = seg['start'], seg['end']
                slope = -seg['kappa_slope']  # calc_flatness_fitting() 存的是 kappa_slope = -slope
                intercept = seg['intercept']
                seg_mask = (scales >= seg_start) & (scales <= seg_end)
                x_line = scales[seg_mask] if np.any(seg_mask) else np.array([seg_start, seg_end])
                y_line = 10**(slope * np.log10(x_line) + intercept)
                color = fit_colors[i % len(fit_colors)]
                ax.loglog(x_line, y_line, linestyle='-', linewidth=2.2, color=color,
                          label=f"Fit {i+1}: $\\kappa$={seg['kappa_slope']:.2f}$\\pm${seg['slope_error']:.2f}")
                ax.axvspan(seg_start, seg_end, color=color, alpha=0.08)

            # --- 新增：動態繪製物理時間尺度垂線與標籤 ---
            text_y = 1.1

            if 'correlation_time' in self.params and comp_name in self.params['correlation_time']:
                tc = self.params['correlation_time'][comp_name]
                ax.axvline(x=tc, color="#414141", linestyle='--', alpha=0.7)
                ax.text(tc, text_y, f'correlation time ({tc:.1f}s)', color="#414141", rotation=90, va='bottom', ha='right', fontsize=8)

            if 'gyro_period' in self.params:
                gp = self.params['gyro_period']
                gp_val = gp.value if hasattr(gp, 'value') else gp
                ax.axvline(x=gp_val, color="#414141", linestyle='--', alpha=0.7)
                ax.text(gp_val, text_y, f'gyro-period ({gp_val:.4f}s)', color="#414141", rotation=90, va='bottom', ha='right', fontsize=8)

            if 't_inertial' in self.params:
                ti = self.params['t_inertial']
                ti_val = ti.value if hasattr(ti, 'value') else ti
                ax.axvline(x=ti_val, color="#414141", linestyle='--', alpha=0.7)
                ax.text(ti_val, text_y, f'inertial time ({ti_val:.3f}s)', color="#414141", rotation=90, va='bottom', ha='right', fontsize=8)

            ax.set_title(f"Flatness Factor - {unique_key.replace('_', ' ')}", fontsize=13, fontweight='bold')
            ax.set_xlabel("Scale $\\tau$ (seconds)", fontsize=11)
            ax.set_ylabel("Flatness $S_4 / (S_2)^2$", fontsize=11)

            try:
                ax.set_ylim(bottom=1.0, top=max(10.0, np.nanmax(flatness) * 1.5))
            except Exception as e:
                # 修正：原本是裸的 except:，這裡改為只吞掉真的會發生的情況 (flatness 全是 NaN)
                # 並印出原因，避免其他錯誤被靜默忽略。
                print(f"⚠ 無法設定 y 軸範圍 ({unique_key}): {e}")

            ax.grid(True, which="both", ls="--", alpha=0.3)
            ax.legend(loc="upper right", fontsize=10, frameon=True)
            plt.tight_layout()

            if save_to_disk:
                filename = f"flatness_{unique_key.lower()}.png"
                save_path = os.path.join(self.fig_dir, filename)
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                print(f"  💾 Saved flatness plot to: {save_path}")
                plt.close()
            else:
                plt.show()

    def calc_flatness_fitting(self):
        """
        多分量互動式 平坦度 (Flatness) 冪律擬合工具：
        對齊 PSD 擬合架構，計算 F(tau) = A * tau^(-kappa) 的交點與斜率，並備份至快取。
        """
        if 'sf' not in self.results or not self.results['sf']:
            print("❌ 錯誤：查無結構函數/平坦度資料，請先執行 run_structure_function()。")
            return

        if 'flatness_fit' not in self.params:
            self.params['flatness_fit'] = {}

        cache_file = os.path.join(self.input_dir, f"Flatness_Fitting_Cache_{SAVE_DATE}.pkl")

        available_keys = list(self.results['sf'].keys())

        print("\n==================================================")
        print("📊 請選擇你想進行平坦度擬合的數據分量 (Component)：")
        for idx, unique_key in enumerate(available_keys):
            existing_count = len(self.params['flatness_fit'].get(unique_key, []))
            status = f" (📈 已有 {existing_count} 段擬合紀錄)" if existing_count > 0 else ""
            print(f"   [{idx + 1}] {unique_key}{status}")
        print("==================================================")

        while True:
            comp_choice = input("請輸入對應的數字編號 >>> ").strip()
            try:
                comp_idx = int(comp_choice) - 1
                if 0 <= comp_idx < len(available_keys):
                    target_key = available_keys[comp_idx]
                    print(f"🎯 已選擇目標分量：【{target_key}】")
                    break
                else:
                    print("❌ 編號超出範圍，請重新輸入！")
            except ValueError:
                print("❌ 請輸入正確的數字編號！")

        payload = self.results['sf'][target_key]
        scales = np.array(payload['scales'])
        flatness = np.array(payload['flatness'])

        valid_mask = np.isfinite(scales) & np.isfinite(flatness) & (flatness > 0) & (scales > 0)
        scales = scales[valid_mask]
        flatness = flatness[valid_mask]

        while True:
            # ======= 只彈出使用者選定分量的圖表，避免多圖干擾 =======
            print(f"\n📢 正在開啟【{target_key}】的平坦度圖表，請觀察曲線並規劃擬合範圍...")
            self.plot_all_flatness(save_to_disk=False, target_key=target_key)

            print(f"\n👉 請輸入【{target_key}】的擬合起始時間 scale (秒, 例如: 0.5):")
            try:
                t_start = float(input("Start Scale (seconds) >>> ").strip())
                print(f"👉 請輸入【{target_key}】的擬合結束時間 scale (秒, 例如: 20.0):")
                t_end = float(input("End Scale (seconds) >>> ").strip())
            except ValueError:
                print("❌ 輸入格式錯誤！請重新輸入數字。")
                plt.close('all')
                continue

            fit_mask = (scales >= t_start) & (scales <= t_end)
            x_fit = scales[fit_mask]
            y_fit = flatness[fit_mask]

            if len(x_fit) < 4:
                print("❌ 警告：選取區間內的可計算數據點太少（小於 4 點），請放大範圍。")
                plt.close('all')
                continue

            log_x = np.log10(x_fit)
            log_y = np.log10(y_fit)

            slope, intercept = np.polyfit(log_x, log_y, 1)

            y_pred = slope * log_x + intercept
            residuals = log_y - y_pred
            residual_ste = np.sqrt(np.sum(residuals**2) / (len(log_x) - 2))
            x_mean = np.mean(log_x)
            slope_err = residual_ste / np.sqrt(np.sum((log_x - x_mean)**2))

            fit_result = {
                'start': t_start,
                'end': t_end,
                'kappa_slope': -slope,
                'slope_error': slope_err,
                'intercept': intercept,
                'n_points': len(x_fit)
            }

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

            ax1.loglog(scales, flatness, color='#9467bd', marker='d', ms=3, label='Data $F(\\tau)$')
            ax1.axhline(y=3.0, color='r', linestyle='--', alpha=0.5, label='Gaussian (F=3)')

            fit_y_lines = 10**(slope * np.log10(scales) + intercept)
            ax1.loglog(scales, fit_y_lines, color='black', linestyle='-', linewidth=1.5,
                       label=f'Fit Line (Slope: {slope:.3f})')
            ax1.axvspan(t_start, t_end, color='gray', alpha=0.15, label='Fitted Range')
            ax1.set_title(f"{target_key} Flatness Power-Law Fit")
            ax1.set_xlabel("Scale $\\tau$ (seconds)")
            ax1.set_ylabel("Flatness")
            ax1.grid(True, which='both', ls='--', alpha=0.3)
            ax1.legend(loc='lower left', fontsize=9)

            compensated_y = flatness * (scales ** (-slope))
            ax2.loglog(scales, compensated_y, color='#17becf', marker='o', ms=3, label='Compensated $F(\\tau) \\cdot \\tau^{\\kappa}$')
            ax2.axvspan(t_start, t_end, color='gray', alpha=0.15)
            ax2.set_title("Compensated Spectrum (Should be horizontal flat)")
            ax2.set_xlabel("Scale $\\tau$ (seconds)")
            ax2.set_ylabel("Compensated Amplitude")
            ax2.grid(True, which='both', ls='--', alpha=0.3)
            ax2.legend(loc='upper right', fontsize=9)

            plt.tight_layout()
            # 修正：原本這裡完全沒有 plt.show()，這個帶擬合線/補償譜的驗證圖
            # 只是建立在記憶體裡，使用者選完 1/2 之後就直接被 plt.close('all') 關掉，
            # 從來沒有被看到過，也沒有被存檔過 —— 這就是「匯出的圖沒有擬合結果」的來源之一。
            plt.show()
            print(f"\n==================================================")
            print(f"🤔 檢查結果（區間: {t_start} ~ {t_end} 秒）：")
            print(f"   間歇性指數 (Kappa): {(-slope):.3f} ± {slope_err:.3f}")
            print(f"==================================================")
            print("   [1] 滿意結果，存入該分量的多段紀錄中")
            print("   [2] 不滿意，放棄本次計算並重新調整區間")

            choice = input("請選擇 (1/2) >>> ").strip()

            if choice == '1':
                if target_key not in self.params['flatness_fit']:
                    self.params['flatness_fit'][target_key] = []
                self.params['flatness_fit'][target_key].append(fit_result)

                print(f"💾 [成功] 已存入歷史擬合清單。")

                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(self.params['flatness_fit'], f)
                except Exception as e:
                    print(f"⚠ 快取同步失敗: {e}")

                # 新增：把這張帶擬合線的驗證圖存到 fig_dir，跟 PSD 那邊的行為對齊。
                seg_num = len(self.params['flatness_fit'][target_key])
                fit_fig_path = os.path.join(self.fig_dir, f"flatness_fit_{target_key.lower()}_seg{seg_num}.png")
                try:
                    fig.savefig(fit_fig_path, bbox_inches='tight', dpi=150)
                    print(f"💾 擬合驗證圖已存檔：\n📂 {fit_fig_path}")
                except Exception as e:
                    print(f"⚠ 擬合圖存檔失敗: {e}")

                plt.close('all')
                break
            else:
                print("🔄 [放棄] 重新啟動區間輸入...")
                plt.close('all') # 丟棄時清空當前畫布，避免視窗堆疊

    def run_yaglom(self, tau_values=40, ratio_in=0.008, start_idx=40, middle1_idx=1500, middle2_idx=1000, end_idx=3000):
        """
        新增：Yaglom's law 三階能量串級率分析。之前 compute_yaglom.py 裡的
        analyze_yaglom_law() 完全沒有被任何方法呼叫過 —— import 了但沒接上。
        這裡用 self.data['elsasser'] 裡已經算好的 dV/dB/z_p/z_n 以及
        self.params['gyro_period'] 接起來。
        """
        if 'elsasser' not in self.data or not self.data['elsasser']:
            print("錯誤：請先執行 'turb' 計算 Elsasser 變量 (preprocess_elsasser_variable)。")
            return
        if 'gyro_period' not in self.params:
            print("錯誤：請先執行 'plasma' 計算質子迴旋週期。")
            return

        v_bulk = self.data['rotated_v']['V_magnitude'].mean()

        results, eps_ce, results2, eps, ep_p, ep_n, mean_eps = analyze_yaglom_law(
            self.data['elsasser']['dV'], self.data['elsasser']['dB'],
            self.data['elsasser']['z_p'], self.data['elsasser']['z_n'],
            v_bulk=v_bulk, tau_values=tau_values, sampling_rate=0.25,
            ratio_in=ratio_in, start_idx=start_idx, middle1_idx=middle1_idx,
            middle2_idx=middle2_idx, end_idx=end_idx,
            ion_gyro_period=self.params['gyro_period']
        )

        self.results['yaglom'] = {'results': results, 'results2': results2}
        # eps_ce 含 LET_total 欄位，run_agyrotropy() 會需要它來對照磁重聯異常點
        self.data['eps_ce'] = eps_ce
        self.data['eps'] = eps
        self.params['epsilon'] = mean_eps
        print(f"Yaglom law analysis complete. Mean cascade rate epsilon ≈ {mean_eps:.2e} m^2/s^3")
        plt.show()

    def run_agyrotropy(self):
        """
        新增：Swisdak (2016) 非迴轉性指標 Q，之前 compute_agyro_pressure.py 裡的
        compute_agyrotropicity()/plot_Q() 也是 import 了但完全沒被呼叫。
        需要先跑過 'mkp' (拿 P_tensor) 跟 'yaglom' (拿 eps_ce 裡的 LET_total)。
        """
        if 'P_tensor' not in self.data:
            print("錯誤：請先執行 'mkp' 載入壓力張量資料。")
            return
        if 'raw_mag' not in self.data:
            print("錯誤：請先執行 'load' 載入磁場資料。")
            return
        if 'eps_ce' not in self.data:
            print("錯誤：請先執行 'yaglom' 計算能量耗散率 (LET_total)。")
            return

        Q = compute_agyrotropicity(self.data['P_tensor'], self.data['raw_mag'])
        self.results['agyrotropy'] = Q
        plot_Q(Q, self.data['eps_ce'])
        plt.show()
        print("Agyrotropicity (Q) analysis complete.")

    def plot_mkp(self):
        """新增：'pmkp' 指令對應的實作 —— 原本 show_help() 有列這個指令，但完全沒有方法可以呼叫。"""
        if 'Mkp' not in self.results:
            print("錯誤：請先執行 'mkp' 載入 MKP 資料。")
            return
        df = self.results['Mkp']
        candidate_cols = [c for c in ['epsilon', 'M_kp', 'Np'] if c in df.columns]
        if not candidate_cols:
            print(f"MKP 資料中找不到常見欄位 (epsilon/M_kp/Np)，可用欄位: {list(df.columns)}")
            return
        fig, axes = plt.subplots(len(candidate_cols), 1, figsize=(10, 3 * len(candidate_cols)), sharex=True)
        if len(candidate_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, candidate_cols):
            ax.plot(df.index, df[col])
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time")
        plt.suptitle("MKP Summary Data")
        plt.tight_layout()
        plt.show()

    def plot_all_proxies(self):
        """新增：'pp' 指令對應的實作，同樣是 show_help() 列了但沒接上。"""
        self.plot_time_series()
        if 'psd_mag' in self.results or 'psd_v' in self.results:
            self.plot_psd(auto_save=True)
        if 'sf' in self.results:
            self.plot_all_structure_functions(save_to_disk=False)
            self.plot_all_flatness(save_to_disk=False)
        plt.show()

    def plot_time_series(self, choice=None):
        """步驟 6: 繪製時間序列圖"""
        print("Time series plotting...")
        if choice is None:
            plot_mag_timeseries_dynamic(self.data['rotated_mag'], components=['B_para', 'B_perp1', 'B_perp2'], title_suffix="rotated")
            plot_mag_timeseries_dynamic(self.data['rotated_v'], components=['V_para', 'V_perp1', 'V_perp2'], title_suffix="Velocity rotated", data_type="Velocity")
            #plot_mag_timeseries_dynamic(self.data['v'], components=['Vr', 'Vt', 'Vn'], title_suffix="Velocity original", data_type="Velocity")
            #plot_mag_timeseries_dynamic(self.data['proton_velocity'], components=['Vr', 'Vt', 'Vn'], title_suffix="Velocity proton", data_type="Velocity")
        elif choice == 'm':
            plot_mag_timeseries_dynamic(self.data['rotated_mag'], components=['B_para', 'B_perp1', 'B_perp2'], title_suffix="rotated")
        elif choice == 'v':
            plot_mag_timeseries_dynamic(self.data['rotated_v'], components=['V_para', 'V_perp1', 'V_perp2'], title_suffix="Velocity rotated", data_type="Velocity")
        elif choice == 's': # 無法顯示
                print("input the component you want to plot (e.g., B_para, V_perp1, etc.)")
                comp = input(">>> ").strip().lower()
                if comp in self.data['rotated_mag'].columns:
                    plot_mag_timeseries_dynamic(self.data['rotated_mag'], components=[comp], title_suffix="rotated")
                elif comp in self.data['rotated_v'].columns:
                    plot_mag_timeseries_dynamic(self.data['rotated_v'], components=[comp], title_suffix="Velocity rotated", data_type="Velocity")
        else:
            plot_mag_timeseries_dynamic(self.data['raw_mag'], components=['Br', 'Bt', 'Bn'])
            plot_mag_timeseries_dynamic(self.data['raw_v'], components=['Vr', 'Vt', 'Vn'], data_type="Velocity")
        print("Time series 圖表已生成。")

    def print_all_params(self):
        """顯示目前所有計算出的參數"""
        print("目前計算出的參數：")
        print(f"parameters:")
        print(f"interval: {self.data['start']} to {self.data['end']}")
        print(f"interval duration: {(pd.to_datetime(self.data['end']) - pd.to_datetime(self.data['start'])).total_seconds()/3600:.2f} hours")
        print(f"interval length: {len(self.data['mag'])} data points")
        print(f"Plasma Frequency: {self.params['plasma_freq']:.2f}")
        print(f"ion gyrofrequency: {self.params['gyro_freq']:.2f}")
        print(f"Inertial Length: {self.params['i_l']:.2f}")
        print(f"Inertial Time: {self.params['i_t']:.2f}")

        # 修正：theta_BR/theta_BV/beta 之前完全沒被計算/存到正確位置，呼叫這裡必定 KeyError。
        # theta_BR/theta_BV 現在由 calc_plasma_params() 算好存在 self.data；
        # beta 本來就存在 self.params，不是 self.data。這裡都加上 .get 防呆，
        # 避免使用者忘記先跑 'plasma' 就呼叫 'status' 導致整個程式崩潰。
        theta_BR = self.data.get('theta_BR')
        theta_BV = self.data.get('theta_BV')
        beta = self.params.get('beta')
        print(f"angle between B and R  (theta_BR): {theta_BR:.2f}°" if theta_BR is not None else "angle between B and R  (theta_BR): N/A (run 'plasma' first)")
        print(f"angle between B and V: {theta_BV:.2f} °" if theta_BV is not None else "angle between B and V: N/A (run 'plasma' first)")
        print(f"Beta: {beta:.2f}" if beta is not None else "Beta: N/A (run 'plasma' first)")

        print(f"mean B: {self.data['rotated_mag']['B_magnitude'].mean():.2f} nT")
        print(f"mean V: {self.data['rotated_v']['V_magnitude'].mean():.2f} km/s")
        print(f"mean T: {self.data['t'].mean():.2f} eV")
        print(f"mean N: {self.data['rotated_v']['N'].mean():.2f} cm^-3")
        print(f"mean p: {self.data['p']['P_magnitude'].mean()} J*cm^-3")

    def show_help(self):
        print("\nAvailable commands:")
        print("  load        - load CDF data and perform initial slicing and conversion")
        print("  plasma      - calculate basic plasma parameters (frequency, inertial length, etc.)")
        print("  turb        - calculate turbulence parameters (correlation time, ACF, elasser variables, etc.)")
        print("  psd         - perform PSD analysis and plotting")
        print("  fit         - interactively fit a power-law slope to a PSD component")
        print("  export      - export saved PSD fitting results to CSV/JSON")
        print("  structure   - perform structure function & flatness analysis")
        print("  sf          - alias for 'structure'")
        print("  ff          - interactively fit a power-law slope to a flatness curve")
        print("  yaglom      - perform Yaglom's law (3rd order) cascade-rate analysis")
        print("  agyro       - compute Swisdak agyrotropy index Q (needs 'mkp' + 'yaglom' first)")
        print("  mkp         - load Mkp data")
        print("  pmkp        - plot Mkp data")
        print("  pp          - plot all proxies data")
        print("  time_series - plot time series")
        print("  status      - display summary of currently calculated parameters")
        print("  report      - generate the summary report")
        print("  help        - show this message")
        print("  exit        - exit the program")

    def generate_report(self):

        generate_report(self, output_path, SAVE_DATE)

    def interactive_shell(self, auto_load=True):

        if auto_load:
            print("正在執行初始自動化流程...")
            try:
                self.load_data(start = start_time_str, end = end_time_str)
                #self.plot_time_series(choice = 'initial')
                #self.plot_time_series()
                self.calc_plasma_params()
                self.calc_turbulence_params()

                self.run_psd()
                self.run_structure_function()

                self.load_or_run_psd_fitting()
                #self.calc_psd_fitting()
                self.export_fitting_data(format='json')
                # 修正：原本這裡重複呼叫了一次 self.run_structure_function()
                # (跟上面第一次呼叫中間沒有任何東西改變輸入資料)，等於白白算了兩遍結構函數。
                self.load_or_run_flatness_fitting()
                # 修正：run_structure_function() 在上面第 1270 行已經把「還沒擬合過」
                # 的平坦度圖存過一次檔了。擬合完成後這裡要重新存一次，
                # 圖檔裡才會真的畫出剛剛擬合出來的冪律線，否則跟 PSD 不對稱
                # (PSD 是先 load_or_run_psd_fitting() 再 plot_psd(auto_save=True) 存最終圖，
                # 但 flatness 這邊擬合完後從沒有重新存圖)。
                self.plot_all_flatness(save_to_disk=True)
                self.plot_psd(auto_save=True)
                print("初始資料載入完成。")
                plt.show()

            except Exception as e:
                print(f"自動載入失敗: {e}")

        """指令式介面主迴圈"""
        while True:
            cmd = input("\n[指令] >>> ").strip().lower()
            if cmd == 'exit':
                break
            elif cmd == 'help':
                self.show_help()
            elif cmd == 'load':
                s = input(f"Start time (預設 {start_time_str}) >>> ").strip() or start_time_str
                e = input(f"End time (預設 {end_time_str}) >>> ").strip() or end_time_str
                self.load_data(start=s, end=e)
            elif cmd == 'plasma':
                self.calc_plasma_params()
            elif cmd == 'turb':
                self.calc_turbulence_params()
            elif cmd == 'psd':
                self.run_psd()
                self.plot_psd(auto_save=True)
            elif cmd == 'fit':
                self.calc_psd_fitting()
            elif cmd == 'export':
                self.export_fitting_data()
            elif cmd == 'structure' or cmd == 'sf':
                self.run_structure_function()
            elif cmd == 'ff':
                self.calc_flatness_fitting()
                # 手動執行 'ff' 之後也重新存一次總覽圖，讓 fig_dir 裡的
                # flatness_<key>.png 反映最新的擬合結果（跟自動流程一致）。
                self.plot_all_flatness(save_to_disk=True)
            elif cmd == 'yaglom':
                self.run_yaglom()
            elif cmd == 'agyro':
                self.run_agyrotropy()
            elif cmd == 'mkp':
                self.load_Mkp(Mkp_path)
            elif cmd == 'pmkp':
                self.plot_mkp()
            elif cmd == 'pp':
                self.plot_all_proxies()
            elif cmd == 'time_series':
                self.plot_time_series()
            elif cmd == 'status':
                self.print_all_params()
            elif cmd == 'report':
                self.generate_report()
            else:
                print(f"未知指令：'{cmd}'。輸入 'help' 查看可用指令。")

if __name__ == "__main__":
    analyzer = SolarOrbiterAnalyzer()
    analyzer.interactive_shell()
