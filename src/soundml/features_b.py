"""Layer B —— 心理声学描述符（笔记 §2.2，≈ 论文的 MACCS 关键子结构）。

这一层每个维度都有明确感知意义，是最像 MACCS「功能基团位」的一层，
也是让 stacking 成立的关键（见 reports/step2_real_labels.md §2：
基模型必须吃不同的**表示**，只换算法没用）。

## 电平校准

心理声学量（sone / acum / asper）依赖**绝对声压**，而 librosa 给的是归一化浮点。
处理方式：先按 BS.1770 归一到 −23 LUFS（与 Layer A 一致），
再缩放到 REFERENCE_SPL_DB 的等效声压。

即：所有音频都在**同一回放音量**下比较音色。这是有意的取舍——
把响度固定住，剩下的差异才归因于音色而不是音量。

⚠️ REFERENCE_SPL_DB 取 60 dB 是心理声学文献的常规分析电平，不是产品实际回放电平。
   出具体规则给设备时需要在目标电平（20–45 dB）复核一次——
   sharpness/roughness 的排序对电平较稳健，loudness 的绝对值不稳健。

## 与笔记 §2.2 的差异

- **fluctuation strength**：MoSQITo 1.2.1 **没有实现**。本模块用调制谱代理量
  （见 `_fluctuation_proxy`），已明确标注 `_proxy` 后缀，不是 ISO 标准量。
- **roughness**：`roughness_dw` 在 30s 上要 27.5 s/段（全量 7.4 小时），
  改在**居中 10s** 上计算。
- **双耳 ITD/ILD**：训练语料是音乐，双耳线索意义不大且成本高。
  改用两个廉价的立体声宽度量（L/R 相关、side/mid 能量比）。
  真正的双耳节拍检测留到 Step 6 筛候选音频时再做。
"""

from __future__ import annotations

import numpy as np
import librosa
import pyloudnorm
from mosqito.sq_metrics import (
    loudness_zwst_perseg,
    roughness_dw,
    sharpness_din_perseg,
)

from .features_a import TARGET_LUFS, _agg

EXTRACTOR_VERSION = "layer_b/1.0.0"

FS = 48000  # MoSQITo 的 ISO 532-1 / DIN 45692 实现要求 48 kHz
MAX_DURATION_S = 30.0
ROUGHNESS_DURATION_S = 10.0  # roughness_dw 成本高，只在居中 10s 上算
SEG_S = 2.0  # loudness / sharpness 的分段长度
REFERENCE_SPL_DB = 60.0
P_REF = 20e-6  # 20 µPa

# fluctuation strength 代理量的参数
FS_ENV_RATE = 200.0  # 包络采样率，够覆盖 20 Hz 调制
FS_MOD_BAND = (0.25, 20.0)

_METER: pyloudnorm.Meter | None = None


def _load_calibrated(path: str) -> tuple[np.ndarray, np.ndarray]:
    """返回 (校准后的单声道信号, 原始立体声 [2, n])。"""
    global _METER
    stereo, _ = librosa.load(path, sr=FS, mono=False, duration=None)
    stereo = np.atleast_2d(stereo)

    y = stereo.mean(axis=0)
    n = int(MAX_DURATION_S * FS)
    if y.size > n:
        start = (y.size - n) // 2
        y, stereo = y[start : start + n], stereo[:, start : start + n]

    if _METER is None:
        _METER = pyloudnorm.Meter(FS)
    if y.size >= int(0.4 * FS):
        lufs = _METER.integrated_loudness(y)
        if np.isfinite(lufs):
            y = y * 10.0 ** ((TARGET_LUFS - lufs) / 20.0)

    rms = float(np.sqrt(np.mean(y**2)))
    if rms > 0:  # 缩放到 REFERENCE_SPL_DB 的等效声压（单位: Pa）
        y = y * (P_REF * 10 ** (REFERENCE_SPL_DB / 20.0)) / rms
    return y, stereo


def _fluctuation_proxy(y: np.ndarray) -> dict[str, float]:
    """Fluctuation strength 的调制谱代理量。

    ⚠️ 不是 ISO 标准量。MoSQITo 1.2.1 未实现 fluctuation strength，
       这里用经典的 4 Hz 峰值加权对包络调制谱积分：

           w(f) = 1 / (f/4 + 4/f)      —— 在 4 Hz 处取最大值

    做法：取宽带包络 → 归一化成调制深度 → FFT → 按 w(f) 加权求和。
    """
    hop = max(1, int(FS / FS_ENV_RATE))
    env = np.sqrt(
        np.mean(np.square(librosa.util.frame(y, frame_length=hop * 2, hop_length=hop)), axis=0)
    )
    if env.size < 16 or env.mean() <= 0:
        return {"fluct_proxy": np.nan, "fluct_peak_hz": np.nan}

    depth = env / env.mean() - 1.0  # 调制深度，去均值
    spec = np.abs(np.fft.rfft(depth * np.hanning(depth.size))) / depth.size
    freqs = np.fft.rfftfreq(depth.size, d=hop / FS)

    band = (freqs >= FS_MOD_BAND[0]) & (freqs <= FS_MOD_BAND[1])
    f, s = freqs[band], spec[band]
    if f.size == 0:
        return {"fluct_proxy": np.nan, "fluct_peak_hz": np.nan}

    w = 1.0 / (f / 4.0 + 4.0 / f)
    return {
        "fluct_proxy": float(np.sum(s * w)),
        "fluct_peak_hz": float(f[np.argmax(s)]),  # 主导调制频率
    }


def _stereo_width(stereo: np.ndarray) -> dict[str, float]:
    """立体声宽度的两个廉价指标（§2.2 双耳项的简化替代）。"""
    if stereo.shape[0] < 2:
        return {"lr_corr": 1.0, "side_mid_ratio": 0.0}
    left, right = stereo[0], stereo[1]
    mid, side = (left + right) / 2, (left - right) / 2
    e_mid = float(np.sum(mid**2))
    corr = float(np.corrcoef(left, right)[0, 1]) if left.std() > 0 and right.std() > 0 else 1.0
    return {
        "lr_corr": corr,
        "side_mid_ratio": float(np.sum(side**2) / e_mid) if e_mid > 0 else 0.0,
    }


def extract(path: str) -> dict[str, float]:
    """提取单个音频的 Layer B 特征向量。"""
    y, stereo = _load_calibrated(path)
    feats: dict[str, float] = {}
    nperseg = int(SEG_S * FS)

    # --- Loudness (ISO 532-1 / Zwicker)，单位 sone ---
    n_seg = loudness_zwst_perseg(y, FS, nperseg=nperseg, noverlap=0)[0]
    feats |= _agg(np.asarray(n_seg), "loudness_sone")

    # --- Sharpness (DIN 45692)，单位 acum ---
    s_seg = sharpness_din_perseg(y, FS, nperseg=nperseg, noverlap=0)[0]
    feats |= _agg(np.asarray(s_seg), "sharpness_acum")

    # --- Roughness (Daniel & Weber)，单位 asper ---
    n_r = int(ROUGHNESS_DURATION_S * FS)
    y_r = y[(y.size - n_r) // 2 :][:n_r] if y.size > n_r else y
    feats |= _agg(np.asarray(roughness_dw(y_r, FS, overlap=0)[0]), "roughness_asper")

    # --- Fluctuation strength（代理量）与立体声宽度 ---
    feats |= _fluctuation_proxy(y)
    feats |= _stereo_width(stereo)

    return feats
