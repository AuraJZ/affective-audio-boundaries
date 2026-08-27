"""反事实音频干预 —— 我们相对分子领域的结构性优势。

分子领域做不到「把这个分子的 C–N 键强度调高 10%、其余不变、再测一次」。
**但音频可以。**

SHAP 说的是「模型在看什么」，干预说的是「改变它会怎样」——
后者才是一条规则该有的形态：一个数值区间只有配上「推动它会发生什么」
才站得住，相关性本身不提供这一点。

## 三种干预，各对应一个域无关特征族

| 干预 | 目标特征 | 手段 |
|---|---|---|
| `envelope_smooth` | `onset_rate`、`onset_env_dmean` | 提取宽带包络 → 平滑 → 反向增益施加 |
| `spectral_smooth` | `spec_contrast3_dmean`、`spec_flatness_std` | STFT 幅度谱沿**时间轴**平滑 |
| `transient_inject` | `onset_rate`（反向） | 注入短促滤波噪声脉冲 |

## 诚实前提：音频干预不可能完美外科手术式

压缩包络必然同时改变响度动态与起音包络；时间平滑必然同时影响多个谱统计量。
因此每次干预都必须报告：

1. **操纵检查**：目标特征是否真的随剂量单调移动？（没动 = 干预无效）
2. **特异性**：目标特征的标准化位移 ÷ 其他特征的最大标准化位移。
   比值低 = 干预不特异，不能把预测变化归因于目标特征。
3. **剂量–反应**：预测是否随剂量单调变化？

只报第 3 项而不报第 1、2 项，是这类实验最常见的过度主张。

## 响度控制

所有干预后一律重新归一化到 −23 LUFS。
否则压缩会顺带抬高响度，而响度已被证明是数据集指纹（`step2b`）。
"""

from __future__ import annotations

import numpy as np
import librosa
import scipy.signal

from .features_a import HOP, N_FFT, SR, renormalize

# 剂量等级：0 = 对照（不干预）
DOSES = [0, 1, 2, 3]


def _envelope(y: np.ndarray, hop: int) -> np.ndarray:
    """帧级 RMS 包络。"""
    return librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]


def envelope_smooth(y: np.ndarray, dose: int, sr: int = SR) -> np.ndarray:
    """压缩振幅包络的动态范围 —— 削平起音与突发。

    在**对数域**把包络向其均值收缩：

        env_db_new = mean_db + (env_db − mean_db) × (1 − strength)

    这在数学上保证包络方差单调下降，因此 onset 强度与突发密度必然降低。

    ⚠️ 早期实现用 `gain = 平滑包络 / 原包络`，在安静段 env→0 导致增益被放大
    （即使截断在 10 倍），**反而制造出新的起音**——试跑时 onset_rate 的
    操纵检查 ρ 是 +0.213（方向相反）。对数域收缩没有这个病态。
    """
    if dose == 0:
        return y
    hop = HOP
    env = _envelope(y, hop)
    eps = 1e-8
    env_db = 20.0 * np.log10(env + eps)
    strength = {1: 0.4, 2: 0.7, 3: 0.9}[dose]
    target_db = env_db.mean() + (env_db - env_db.mean()) * (1.0 - strength)
    gain_frames = 10.0 ** ((target_db - env_db) / 20.0)

    frame_pos = np.arange(len(gain_frames)) * hop
    gain = np.interp(np.arange(len(y)), frame_pos, gain_frames)
    return renormalize(y * gain, sr)


def spectral_smooth(y: np.ndarray, dose: int, sr: int = SR) -> np.ndarray:
    """沿**时间轴**平滑幅度谱 —— 削平音色的变化速率。

    保留相位，只平滑幅度，因此音高与整体音色不变，变的是"变化多快"。
    剂量 = 平滑窗长度（帧）。
    """
    if dose == 0:
        return y
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    mag, phase = np.abs(D), np.angle(D)
    win = int(2 ** (dose + 1) + 1)  # 5 / 9 / 17 帧
    win = min(win, max(3, (mag.shape[1] // 2) * 2 - 1))
    mag_s = scipy.signal.savgol_filter(mag, win, polyorder=1, axis=1, mode="nearest")
    mag_s = np.clip(mag_s, 0, None)
    out = librosa.istft(mag_s * np.exp(1j * phase), hop_length=HOP, length=len(y))
    return renormalize(out, sr)


def transient_inject(y: np.ndarray, dose: int, sr: int = SR, seed: int = 0) -> np.ndarray:
    """注入短促滤波噪声脉冲 —— 提高突发密度（与前两者反向）。

    脉冲经带通滤波（500–4000 Hz）并加指数衰减包络，模拟真实瞬态。
    剂量 = 每秒注入个数。
    """
    if dose == 0:
        return y
    rng = np.random.default_rng(seed)
    dur_s = len(y) / sr
    n_click = int(dose * 4 * dur_s)  # 4 / 8 / 12 个每秒
    click_len = int(0.02 * sr)

    b, a = scipy.signal.butter(4, [500 / (sr / 2), 4000 / (sr / 2)], btype="band")
    decay = np.exp(-np.linspace(0, 6, click_len))

    out = y.copy()
    peak = float(np.abs(y).max()) or 1e-6
    for _ in range(n_click):
        start = int(rng.integers(0, max(1, len(y) - click_len)))
        click = scipy.signal.lfilter(b, a, rng.normal(size=click_len)) * decay
        # 相对**峰值**而非 RMS 定标，确保脉冲能被 onset 检测器识别
        click = click / (np.abs(click).max() + 1e-9) * peak * 0.8
        out[start : start + click_len] += click
    return renormalize(out, sr)


INTERVENTIONS = {
    "envelope_smooth": (envelope_smooth, ["onset_rate", "onset_env_dmean", "onset_env_mean"]),
    "spectral_smooth": (spectral_smooth, ["spec_contrast3_dmean", "spec_flatness_std", "mfcc1_std"]),
    "transient_inject": (transient_inject, ["onset_rate", "onset_env_dmean"]),
}
