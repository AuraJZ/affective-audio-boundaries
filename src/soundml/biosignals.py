"""BIRAFFE2 生理信号的逐试次特征。

## 信号

`SUBxxx-BioSigs.csv`：`TIMESTAMP, ECG, EDA`，约 1000 Hz（BITalino）。
每名受试约 30 次设备掉线，必须按时间戳间隔检测并剔除受影响的试次 ——
否则掉线段会被当成「平坦的生理响应」，系统性压低效应量。

## 逐试次窗口

IADS 刺激时长 6 s。取：

    基线    −2 … 0 s
    响应    ECG  0 … 6 s      （心率变化几乎即时）
            EDA  1 … 8 s      （皮电响应有 1–5 s 潜伏期，窗口需后移并加长）

## 特征

| 特征 | 含义 |
|---|---|
| `eda_rise` | 紧张性抬升：响应窗均值 − 基线均值 |
| `eda_scr` | 相位性幅度：去趋势后响应窗的峰值 |
| `hr_change` | 心率变化：响应窗均值 − 基线均值（bpm） |
| `hr_baseline` | 基线心率，用于个体标准化 |

⚠️ 6 s 窗口内只有约 6–8 个心搏，**RMSSD 之类的 HRV 指标在此长度上不稳定**，
故不计算 —— 这是有意省略，不是遗漏。
"""

from __future__ import annotations

import numpy as np
import scipy.signal

FS = 1000.0            # 标称采样率
BASELINE_S = (-2.0, 0.0)
# ECG 基线单独放宽到 5 s：2 s 内只有 2–3 个心搏，R 峰检测（要求 ≥3 峰）
# 在多数试次上失败 —— 试跑时基线心率完整率仅 38%。试次间隔约 18 s，5 s 可行。
ECG_BASELINE_S = (-5.0, 0.0)
ECG_WIN_S = (0.0, 6.0)
EDA_WIN_S = (1.0, 8.0)
MAX_GAP_S = 0.5        # 时间戳间隔超过此值视为掉线


PHASIC_HP_HZ = 0.05    # 相位性/紧张性分界，标准取值


def phasic_component(eda: np.ndarray, fs: float = FS) -> np.ndarray:
    """高通分离相位性成分（SCR）。

    ⚠️ 早期版本**没有做这一步**，只在 6 s 窗内做线性去趋势 ——
    而 EDA 的主要方差是分钟尺度的慢漂移，6 s 窗根本扣不掉。
    结果是试次级响应被漂移淹没，测出来全是零。

    功率谱显示 0.05–1 Hz 占总能量 28.6%，相位性成分**是存在的**，
    问题出在提取而非数据。
    """
    if eda.size < int(10 * fs):
        return eda - np.mean(eda)
    try:
        b, a = scipy.signal.butter(2, PHASIC_HP_HZ / (fs / 2), btype="high")
        return scipy.signal.filtfilt(b, a, eda)
    except Exception:
        return eda - np.mean(eda)


def detect_gaps(ts: np.ndarray, max_gap_s: float = MAX_GAP_S) -> np.ndarray:
    """返回掉线区间 [(start, end), ...]（秒，绝对时间戳）。"""
    d = np.diff(ts)
    idx = np.flatnonzero(d > max_gap_s)
    return np.array([(ts[i], ts[i + 1]) for i in idx]) if idx.size else np.empty((0, 2))


def window_ok(t0: float, t1: float, gaps: np.ndarray) -> bool:
    """窗口是否与任一掉线区间重叠。"""
    if gaps.size == 0:
        return True
    return not np.any((gaps[:, 0] < t1) & (gaps[:, 1] > t0))


def _slice(ts: np.ndarray, x: np.ndarray, t0: float, t1: float) -> np.ndarray:
    m = (ts >= t0) & (ts < t1)
    return x[m]


def heart_rate(ecg: np.ndarray, fs: float = FS) -> float:
    """由 R 峰间期估计平均心率（bpm）。样本过短或峰过少返回 NaN。"""
    if ecg.size < int(2 * fs):
        return np.nan
    x = ecg - np.mean(ecg)
    sd = np.std(x)
    if sd == 0:
        return np.nan
    try:
        b, a = scipy.signal.butter(3, [5 / (fs / 2), 20 / (fs / 2)], btype="band")
        f = scipy.signal.filtfilt(b, a, x)
    except Exception:
        return np.nan
    peaks, _ = scipy.signal.find_peaks(np.abs(f), height=2 * np.std(f),
                                       distance=int(0.35 * fs))
    if peaks.size < 3:
        return np.nan
    ibi = np.diff(peaks) / fs
    ibi = ibi[(ibi > 0.35) & (ibi < 1.7)]          # 生理合理区间
    return float(60.0 / np.mean(ibi)) if ibi.size >= 2 else np.nan


def trial_features(ts: np.ndarray, ecg: np.ndarray, eda: np.ndarray,
                   onset: float, gaps: np.ndarray) -> dict[str, float]:
    """单个试次的生理特征。窗口若与掉线重叠则返回 NaN。"""
    nan = {"eda_rise": np.nan, "eda_scr": np.nan,
           "hr_change": np.nan, "hr_baseline": np.nan}

    b0, b1 = onset + BASELINE_S[0], onset + BASELINE_S[1]
    eb0, eb1 = onset + ECG_BASELINE_S[0], onset + ECG_BASELINE_S[1]
    e0, e1 = onset + ECG_WIN_S[0], onset + ECG_WIN_S[1]
    d0, d1 = onset + EDA_WIN_S[0], onset + EDA_WIN_S[1]
    if not (window_ok(eb0, b1, gaps) and window_ok(e0, e1, gaps)
            and window_ok(d0, d1, gaps)):
        return nan

    eda_b = _slice(ts, eda, b0, b1)
    eda_r = _slice(ts, eda, d0, d1)
    ecg_b = _slice(ts, ecg, eb0, eb1)
    ecg_r = _slice(ts, ecg, e0, e1)
    if min(eda_b.size, eda_r.size, ecg_r.size) < int(0.5 * FS):
        return nan

    out = dict(nan)
    base = float(np.mean(eda_b))
    out["eda_rise"] = float(np.mean(eda_r) - base)
    # 去线性趋势后的峰值 = 相位性成分幅度
    if eda_r.size > 8:
        det = eda_r - np.linspace(eda_r[0], eda_r[-1], eda_r.size)
        out["eda_scr"] = float(np.max(det) - np.min(det))

    hr_b = heart_rate(ecg_b) if ecg_b.size >= int(2 * FS) else np.nan
    hr_r = heart_rate(ecg_r)
    out["hr_baseline"] = hr_b
    if np.isfinite(hr_b) and np.isfinite(hr_r):
        out["hr_change"] = float(hr_r - hr_b)
    return out


FEATURES = ["eda_rise", "eda_scr", "hr_change"]
