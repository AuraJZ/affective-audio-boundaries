"""Layer A —— 低层声学描述符（对应笔记 §2.1，≈ 论文的 RDKit 描述符）。

对笔记的一处补强（回应 §8.3「时间动态」风险）：
帧级特征不只取 mean/std，另加 p10/p50/p90 和一阶差分均值。
成本几乎为零，但保住了 mean 会抹掉的时序信息。

版本号变更规则：只要改动会影响输出数值，就 bump EXTRACTOR_VERSION。
run record 里记的是这个号，不是 git commit——因为特征可能跨多个 commit 复用。
"""

from __future__ import annotations

import numpy as np
import librosa
import pyloudnorm

EXTRACTOR_VERSION = "layer_a/1.2.0"  # 1.1.0: 30s 居中截窗；1.2.0: 可选响度归一化

# 响度归一化的目标（ITU-R BS.1770 积分响度）。
# 用途见 reports/step2_real_labels.md §4：判别"模型学到的是助眠规律还是单纯的响度"。
TARGET_LUFS = -23.0

SR = 22050  # Nyquist 11 kHz，够覆盖 2–5 kHz 刺耳带；Layer B 心理声学再上全带宽
N_FFT = 2048
HOP = 512
# 跨数据集统一时长：DEAM 是 45s 摘录、PMEmo 是变长副歌、ESC-50 是 5s。
# 一律取居中窗口，短于此长度的整段使用（ESC-50 因此不受影响）。
MAX_DURATION_S = 30.0
SLOPE_BAND = (20.0, 8000.0)  # 1/f 斜率拟合区间
LOW_BAND = (0.0, 500.0)  # 低频能量占比
HARSH_BAND = (2000.0, 5000.0)  # 刺耳带能量占比


def _agg(x: np.ndarray, name: str) -> dict[str, float]:
    """把帧级序列压成 6 个统计量。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0 or not np.all(np.isfinite(x)):
        x = x[np.isfinite(x)] if x.size else x
    if x.size == 0:
        return {f"{name}_{s}": np.nan for s in ("mean", "std", "p10", "p50", "p90", "dmean")}
    d = np.abs(np.diff(x)) if x.size > 1 else np.array([0.0])
    return {
        f"{name}_mean": float(np.mean(x)),
        f"{name}_std": float(np.std(x)),
        f"{name}_p10": float(np.percentile(x, 10)),
        f"{name}_p50": float(np.percentile(x, 50)),
        f"{name}_p90": float(np.percentile(x, 90)),
        f"{name}_dmean": float(np.mean(d)),  # 一阶差分 —— 时序变化速率
    }


def _spectral_slope(power: np.ndarray, freqs: np.ndarray) -> float:
    """1/f 谱斜率（粉噪指数）。理想粉噪 ≈ -1，白噪 ≈ 0。

    在 log-log 域对平均功率谱做一次线性拟合。
    """
    band = (freqs >= SLOPE_BAND[0]) & (freqs <= SLOPE_BAND[1]) & (freqs > 0)
    p = power[band].mean(axis=1) if power.ndim == 2 else power[band]
    f = freqs[band]
    ok = p > 0
    if ok.sum() < 8:
        return float("nan")
    slope, _ = np.polyfit(np.log10(f[ok]), np.log10(p[ok]), 1)
    return float(slope)


def _band_ratio(power: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    total = power.sum()
    if total <= 0:
        return float("nan")
    band = (freqs >= lo) & (freqs < hi)
    return float(power[band].sum() / total)


def _center_crop(y: np.ndarray, sr: int) -> np.ndarray:
    """取居中的 MAX_DURATION_S 秒；不足则原样返回。

    掐头去尾还有个附带好处：躲开淡入淡出和片头静音。
    """
    n = int(MAX_DURATION_S * sr)
    if y.size <= n:
        return y
    start = (y.size - n) // 2
    return y[start : start + n]


_METER: pyloudnorm.Meter | None = None


def _normalize_loudness(y: np.ndarray, sr: int) -> np.ndarray:
    """按 ITU-R BS.1770 积分响度归一到 TARGET_LUFS。

    只做增益缩放，不限幅——特征提取不输出音频，削顶反而会污染频谱。
    绝对电平因此被抹平，**相对**动态（rms_std / rms_dmean）仍然保留，
    这正是判别实验想要的：看剩下的相对动态和频谱形状还能不能分类。
    """
    global _METER
    if _METER is None or _METER.rate != sr:
        _METER = pyloudnorm.Meter(sr)
    if y.size < int(0.4 * sr):  # BS.1770 最短分析块
        return y
    loudness = _METER.integrated_loudness(y)
    if not np.isfinite(loudness):  # 全静音
        return y
    return y * (10.0 ** ((TARGET_LUFS - loudness) / 20.0))


def load_prepared(path: str, *, normalize_loudness: bool = False) -> tuple[np.ndarray, int]:
    """载入 → 居中截窗 → （可选）响度归一化。返回可直接送入 `extract_array` 的信号。

    单独暴露出来是为了反事实干预实验：干预要作用在**预处理之后**的信号上，
    且干预后必须重新归一化响度，否则会重新引入响度混淆。
    """
    y, sr = librosa.load(path, sr=SR, mono=True)
    y = _center_crop(y, sr)
    if normalize_loudness:
        y = _normalize_loudness(y, sr)
    return y, sr


def renormalize(y: np.ndarray, sr: int) -> np.ndarray:
    """干预后重新归一化响度 —— 供干预实验调用。"""
    return _normalize_loudness(y, sr)


def extract(path: str, *, normalize_loudness: bool = False) -> dict[str, float]:
    """提取单个音频的 Layer A 特征向量。"""
    y, sr = load_prepared(path, normalize_loudness=normalize_loudness)
    return extract_array(y, sr)


def extract_array(y: np.ndarray, sr: int = SR) -> dict[str, float]:
    """对**已预处理**的信号数组提特征（不再截窗、不再归一化）。"""
    feats: dict[str, float] = {}

    if y.size < N_FFT:
        y = np.pad(y, (0, N_FFT - y.size))

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    power = S**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    # --- 频谱形状 ---
    feats |= _agg(librosa.feature.spectral_centroid(S=S, sr=sr), "spec_centroid")
    feats |= _agg(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85), "spec_rolloff")
    feats |= _agg(librosa.feature.spectral_flatness(S=S), "spec_flatness")
    feats |= _agg(librosa.feature.spectral_bandwidth(S=S, sr=sr), "spec_bandwidth")
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    for i, row in enumerate(contrast):
        feats |= _agg(row, f"spec_contrast{i}")

    # --- 时域 ---
    feats |= _agg(librosa.feature.rms(S=S), "rms")
    feats |= _agg(librosa.feature.zero_crossing_rate(y, hop_length=HOP), "zcr")

    # --- 节奏 / onset ---
    onset_env = librosa.onset.onset_strength(S=librosa.power_to_db(power), sr=sr, hop_length=HOP)
    feats |= _agg(onset_env, "onset_env")
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP)
    dur = len(y) / sr
    feats["onset_rate"] = float(len(onsets) / dur) if dur > 0 else np.nan
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=HOP)
    feats["tempo"] = float(np.atleast_1d(tempo)[0])
    # pulse clarity 代理：节拍图峰值相对能量
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr, hop_length=HOP)
    feats["pulse_clarity"] = float(np.max(pulse) - np.mean(pulse)) if pulse.size else np.nan

    # --- 音高 ---
    f0 = librosa.yin(y, fmin=50, fmax=2000, sr=sr, frame_length=N_FFT, hop_length=HOP)
    voiced = f0[np.isfinite(f0)]
    feats |= _agg(voiced, "f0")

    # --- MFCC 1–13 ---
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(power), sr=sr, n_mfcc=13)
    for i, row in enumerate(mfcc, start=1):
        stats = _agg(row, f"mfcc{i}")
        # MFCC 只留 mean/std，否则维度爆炸（13×6=78 → 13×2=26）
        feats[f"mfcc{i}_mean"] = stats[f"mfcc{i}_mean"]
        feats[f"mfcc{i}_std"] = stats[f"mfcc{i}_std"]

    # --- 助眠相关的关键三项（笔记 §2.1 结尾特别标注）---
    feats["slope_1f"] = _spectral_slope(power, freqs)
    mean_power = power.mean(axis=1)
    feats["low_energy_ratio"] = _band_ratio(mean_power, freqs, *LOW_BAND)
    feats["harsh_energy_ratio"] = _band_ratio(mean_power, freqs, *HARSH_BAND)

    return feats
