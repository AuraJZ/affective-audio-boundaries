"""帧级描述符 —— 与逐时刻标注对齐的时间网格。

## 为什么需要

我们的核心主张是关于**时间变化率**的，却一直用片段级聚合量在检验它。
DEAM / PMEmo 都带逐时刻标注（每 0.5 s），能在**主张所指的时间尺度上**直接检验：
某个描述符的瞬时变化，是否预测唤醒的瞬时变化。

## 为什么组内设计更强

跨曲比较把唤醒与曲风、艺人、制作、录音条件全部混在一起。
组内（同一首歌内部）比较时，**同一段录音自己做自己的对照**，
上述混淆按构造被消掉——相当于固定效应设计。

## 实现

每首歌只做一次 STFT，然后对每个标注时刻取其前后 `HALF_WIN_S` 的帧做聚合。
描述符选取覆盖两类：域无关的变化率族，以及作为对照的频谱水平族。
"""

from __future__ import annotations

import numpy as np
import librosa

from .features_a import HOP, N_FFT, SR, load_prepared

EXTRACTOR_VERSION = "frame/1.0.0"
HALF_WIN_S = 1.0          # 每个标注时刻左右各取 1 s
FRAME_RATE = HOP / SR     # 每帧约 23 ms

# 变化率 / 变异族（域无关候选）与频谱水平族（对照）
VARIABILITY = ["onset_env_mean", "onset_env_dmean", "onset_rate",
               "spec_contrast3_dmean", "spec_flatness_std", "mfcc1_std",
               "zcr_std", "rms_dmean"]
LEVEL = ["spec_rolloff_mean", "spec_centroid_mean", "spec_flatness_p50",
         "zcr_p50", "rms_mean", "mfcc1_mean", "spec_contrast3_mean"]
DESCRIPTORS = VARIABILITY + LEVEL


def _win_stats(x: np.ndarray) -> tuple[float, float, float]:
    """(mean, std, 一阶差分均值) —— 与 features_a._agg 的口径一致。"""
    if x.size == 0:
        return np.nan, np.nan, np.nan
    d = np.abs(np.diff(x)) if x.size > 1 else np.array([0.0])
    return float(np.mean(x)), float(np.std(x)), float(np.mean(d))


def frame_series(path: str, times_s: np.ndarray) -> dict[str, np.ndarray]:
    """返回 {描述符: 长度 = len(times_s) 的序列}。

    `times_s` 为标注时刻（秒）。窗口超出音频范围的时刻返回 NaN。
    """
    y, sr = load_prepared(path, normalize_loudness=True)
    if y.size < N_FFT:
        return {d: np.full(len(times_s), np.nan) for d in DESCRIPTORS}

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    power = S ** 2
    logp = librosa.power_to_db(power)

    # 帧级基础量，全曲算一次
    base = {
        "centroid": librosa.feature.spectral_centroid(S=S, sr=sr)[0],
        "rolloff": librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0],
        "flatness": librosa.feature.spectral_flatness(S=S)[0],
        "rms": librosa.feature.rms(S=S)[0],
        "zcr": librosa.feature.zero_crossing_rate(y, hop_length=HOP)[0],
        "contrast3": librosa.feature.spectral_contrast(S=S, sr=sr)[3],
        "onset_env": librosa.onset.onset_strength(S=logp, sr=sr, hop_length=HOP),
        "mfcc1": librosa.feature.mfcc(S=logp, sr=sr, n_mfcc=13)[0],
    }
    onsets = librosa.onset.onset_detect(onset_envelope=base["onset_env"],
                                        sr=sr, hop_length=HOP)
    onset_times = onsets * FRAME_RATE

    n_frames = len(base["centroid"])
    half = int(HALF_WIN_S / FRAME_RATE)
    out = {d: np.full(len(times_s), np.nan) for d in DESCRIPTORS}

    for i, t in enumerate(times_s):
        c = int(t / FRAME_RATE)
        lo, hi = max(c - half, 0), min(c + half, n_frames)
        if hi - lo < 8:
            continue

        oe_m, _, oe_d = _win_stats(base["onset_env"][lo:hi])
        out["onset_env_mean"][i] = oe_m
        out["onset_env_dmean"][i] = oe_d
        out["onset_rate"][i] = float(
            np.sum((onset_times >= t - HALF_WIN_S) & (onset_times < t + HALF_WIN_S))
            / (2 * HALF_WIN_S))

        c3_m, _, c3_d = _win_stats(base["contrast3"][lo:hi])
        out["spec_contrast3_mean"][i] = c3_m
        out["spec_contrast3_dmean"][i] = c3_d

        fl_m, fl_s, _ = _win_stats(base["flatness"][lo:hi])
        out["spec_flatness_p50"][i] = float(np.median(base["flatness"][lo:hi]))
        out["spec_flatness_std"][i] = fl_s

        m1_m, m1_s, _ = _win_stats(base["mfcc1"][lo:hi])
        out["mfcc1_mean"][i] = m1_m
        out["mfcc1_std"][i] = m1_s

        z_m, z_s, _ = _win_stats(base["zcr"][lo:hi])
        out["zcr_p50"][i] = float(np.median(base["zcr"][lo:hi]))
        out["zcr_std"][i] = z_s

        r_m, _, r_d = _win_stats(base["rms"][lo:hi])
        out["rms_mean"][i] = r_m
        out["rms_dmean"][i] = r_d

        out["spec_rolloff_mean"][i] = float(np.mean(base["rolloff"][lo:hi]))
        out["spec_centroid_mean"][i] = float(np.mean(base["centroid"][lo:hi]))

    return out
