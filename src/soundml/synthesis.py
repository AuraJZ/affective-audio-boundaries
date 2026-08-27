"""参数化声景合成 —— 干净的剂量–反应刺激。

## 为什么合成而不是修改

`interventions.py` 修改真实音频，代价是**特异性不足**：注入一个瞬态必然同时
改变宽带谱对比度动态，躲不开（见 Fig. 4c，目标描述符从未是被推动最多的那个）。

此前把这记为「物理约束，无法消除」。**那句话只对修改成立。**
从参数直接合成时，目标属性是**按构造控制的** —— 要 onset 密度 2/s 就生成 2/s。
连带效应仍会存在（声学量本就不正交），但目标量不再淹没在其中，
且每一格的参数是**已知真值**而非事后测量。

附带好处：合成音频版权完全干净，不受任何语料的再分发条款约束。

## 三个受控轴

对应 Fig. 2 的域无关特征族：

| 轴 | 参数 | 对应描述符族 |
|---|---|---|
| 谱倾斜 | `slope` —— 1/f^(−slope/2) 整形 | 频谱水平（域相关，作对照） |
| 突发密度 | `onset_rate` —— 每秒瞬态个数 | `onset_rate`、`onset_env_*` |
| 频谱变化率 | `flux` —— 滤波器截止的时变速率 | `spec_*_dmean`、`spec_flatness_std` |

底噪用 FFT 法生成有色噪声（与 `02-技术方案/audio/generate_pink.py` 同法），
保证循环无缝；瞬态是带通滤波的指数衰减脉冲。
所有输出统一归一到 −23 LUFS，与全部特征提取口径一致。
"""

from __future__ import annotations

import librosa
import numpy as np
import scipy.signal

from .features_a import SR, renormalize

DURATION_S = 10.0


def colored_noise(n: int, slope: float, rng) -> np.ndarray:
    """1/f^(slope/2) 有色噪声。slope=0 白噪，1 粉噪，2 布朗噪。

    频域整形 + IFFT：输出天然周期，无需交叉淡化即可无缝循环。
    """
    spec = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
    f = np.arange(len(spec))
    f[0] = 1
    spec = spec / f ** (slope / 2.0)
    spec[0] = 0
    y = np.fft.irfft(spec, n=n)
    return y / (np.abs(y).max() + 1e-12)


def add_transients(y: np.ndarray, rate_per_s: float, sr: int, rng,
                   level: float = 0.55) -> np.ndarray:
    """按给定密度注入带通瞬态。rate_per_s 即为该刺激的真值 onset 密度。"""
    if rate_per_s <= 0:
        return y
    n_click = int(rate_per_s * len(y) / sr)
    if n_click == 0:
        return y
    clen = int(0.025 * sr)
    b, a = scipy.signal.butter(4, [400 / (sr / 2), 5000 / (sr / 2)], btype="band")
    decay = np.exp(-np.linspace(0, 7, clen))
    peak = float(np.abs(y).max()) or 1.0

    out = y.copy()
    # 均匀分布 + 抖动，避免规律脉冲产生可听的节拍
    slots = np.linspace(0, len(y) - clen, n_click, dtype=int)
    jitter = rng.integers(-int(0.4 * sr / max(rate_per_s, 1)),
                          int(0.4 * sr / max(rate_per_s, 1)) + 1, size=n_click)
    for s, j in zip(slots, jitter):
        st = int(np.clip(s + j, 0, len(y) - clen))
        click = scipy.signal.lfilter(b, a, rng.normal(size=clen)) * decay
        click = click / (np.abs(click).max() + 1e-9) * peak * level
        out[st:st + clen] += click
    return out


def tilted_noise(n: int, slope: float, flux_hz: float, flux_depth: float,
                 sr: int, rng) -> np.ndarray:
    """谱倾斜随时间正弦摆动的噪声。

    slope_t = slope + flux_depth · sin(2π · flux_hz · t)

    ⚠️ 这个实现取代了早期的「时变低通」版本。低通版有两个致命缺陷，
    在操纵检查中暴露：

    1. 低通的滚降主导了测得的谱斜率，使 `slope` 轴完全失控
       （参数 1.0 → 实测 −3.08；参数 2.0 → 实测 −1.34，非单调）
    2. 滤波器切换本身产生瞬态，被 onset 检测器计入，污染 `onset_rate` 轴
       （参数 0 → 实测 3.30）

    改为在 STFT 幅度上直接施加时变倾斜后：
    **均值斜率按构造等于 `slope`，flux 只控制它变化多快，两轴天然正交。**
    倾斜是平滑连续的，也不再制造瞬态。
    """
    n_fft, hop = 2048, 512
    y0 = rng.normal(size=n)
    D = librosa.stft(y0, n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(D), np.angle(D)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    freqs[0] = freqs[1]                       # 避免 log(0)
    t = np.arange(mag.shape[1]) * hop / sr
    slope_t = slope + flux_depth * np.sin(2 * np.pi * flux_hz * t)

    # 逐帧整形：|X| *= f^(-slope_t/2)
    shaped = mag * (freqs[:, None] ** (-slope_t[None, :] / 2.0))
    out = librosa.istft(shaped * np.exp(1j * phase), hop_length=hop, length=n)
    m = np.abs(out).max()
    return out / m if m > 0 else out


def synthesize(slope: float, onset_rate: float, flux_hz: float,
               seed: int = 0, duration_s: float = DURATION_S,
               sr: int = SR, flux_depth: float = 0.8) -> np.ndarray:
    """按三个受控参数合成一段刺激，归一到 −23 LUFS。

    `flux_hz = 0` 时倾斜恒定；`flux_depth` 固定为 0.8，
    使 flux 轴只改变**变化速率**而不改变变化幅度。
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    y = tilted_noise(n, slope, flux_hz, flux_depth if flux_hz > 0 else 0.0, sr, rng)
    y = add_transients(y, onset_rate, sr, rng)
    return renormalize(y, sr)
