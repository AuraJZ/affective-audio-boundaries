"""Layer D：调性与和声描述符 —— 补 Layer A 的已知盲区。

Layer A 是频谱-时序描述子（MFCC、质心、通量、起始密度……），
几乎不含**调性**信息。数据准确地指出了这个缺口：Soundtracks 的七个情绪轴上，
`happy` 是唯一显著弱的一个（ρ=0.474，R²=0.138），
而其余六个（tension 0.804、tender 0.754、valence 0.735…）都在 0.6 以上。

音乐中的「快乐」高度依赖**调式（大调/小调）与协和度** —— 正是本层要补的。

## 特征（39 维）

| 组 | 维度 | 说明 |
|---|---|---|
| 调式 | 4 | Krumhansl-Kessler 调性剖面：大调强度、小调强度、**调式得分**、调性清晰度 |
| **调不变色度** | 13 | 色度向量旋转到估计主音后取均值 —— **去掉了「唱什么调」，只留「调内结构」** |
| 色度变化 | 3 | 色度通量（和声变化率）均值/标准差、色度熵（调性 vs 无调性） |
| Tonnetz | 12 | 调性中心（五度圈 / 大三度 / 小三度环）均值与标准差 |
| 协和度 | 4 | 色度加权音程协和度 + Plomp-Levelt 感觉粗糙度均值/标准差 |
| 谐波性 | 3 | 谐波/打击成分能量比、基频中位数、有调帧占比 |

## 为什么要「调不变」色度

原始色度向量编码的是「这首曲子在哪个调」，那是**任意的**（同一首歌移调后
情绪不变）。直接喂进模型，学到的会是数据集里各调的分布偏差。
因此先估主音、再旋转，得到「相对主音的音级分布」——
大三度是否突出、小三度是否突出、导音有多强，这些才是承载情绪的量。

## 音程协和度权重

取自感觉协和度的常规排序（同度 > 纯五 > 纯四 > 大三/大六 > 小三/小六 >
大二/小七 > 小二/大七 > 三全音）。这是**先验设定的常数，不从数据拟合**，
因此不构成自由度。
"""

from __future__ import annotations

import numpy as np

from . import features_a

EXTRACTOR_VERSION = "layer_d/1.0.0"

# Krumhansl-Kessler 调性剖面（1982），大调与小调
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# 音程协和度（半音距 0–11），感觉协和度的常规排序，先验常数
INTERVAL_CONSONANCE = np.array([1.00, 0.10, 0.30, 0.60, 0.70, 0.80,
                                0.05, 0.90, 0.60, 0.70, 0.30, 0.15])

N_PEAKS = 20            # Plomp-Levelt 粗糙度取的谱峰数
PITCH_FMIN, PITCH_FMAX = 65.0, 2093.0


def _rotate_to_tonic(chroma_mean: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """估计主音与调式，返回调不变色度和四个调式量。

    对 12 个可能的主音，分别与大调、小调剖面求相关，取最大者为估计。
    `mode_score = 大调最优 − 小调最优`：正值偏大调，负值偏小调。
    """
    c = chroma_mean / max(chroma_mean.sum(), 1e-12)
    maj, mnr = np.empty(12), np.empty(12)
    for k in range(12):
        r = np.roll(c, -k)
        maj[k] = np.corrcoef(r, KK_MAJOR)[0, 1]
        mnr[k] = np.corrcoef(r, KK_MINOR)[0, 1]
    maj = np.nan_to_num(maj)
    mnr = np.nan_to_num(mnr)
    best_maj, best_mnr = float(maj.max()), float(mnr.max())
    tonic = int(maj.argmax() if best_maj >= best_mnr else mnr.argmax())
    allk = np.concatenate([maj, mnr])
    return np.roll(c, -tonic), {
        "key_major_strength": best_maj,
        "key_minor_strength": best_mnr,
        "mode_score": best_maj - best_mnr,
        "key_clarity": float(allk.max() - allk.mean()),
    }


def _chroma_consonance(chroma_mean: np.ndarray) -> float:
    """色度加权的音程协和度：同时发声的音级对，按其音程的协和度加权求和。"""
    c = chroma_mean / max(chroma_mean.sum(), 1e-12)
    total = 0.0
    for i in range(12):
        for j in range(12):
            total += c[i] * c[j] * INTERVAL_CONSONANCE[(j - i) % 12]
    return float(total)


def _plomp_levelt(freqs: np.ndarray, amps: np.ndarray) -> float:
    """谱峰间的感觉粗糙度（Plomp & Levelt 1965 的常用参数化）。

    两个正弦成分的粗糙度随其频差先升后降，峰值约在临界带宽的 25%。
    这是**声学层面**的不协和，与上面基于音级的音乐层面协和度互补。
    """
    if len(freqs) < 2:
        return 0.0
    a, b = 3.5, 5.75
    s1, s2 = 0.0207, 18.96
    total = 0.0
    for i in range(len(freqs)):
        for j in range(i + 1, len(freqs)):
            f1, f2 = sorted((freqs[i], freqs[j]))
            if f1 <= 0:
                continue
            s = 0.24 / (s1 * f1 + s2)
            d = f2 - f1
            total += (min(amps[i], amps[j])
                      * (np.exp(-a * s * d) - np.exp(-b * s * d)))
    return float(total / max(len(freqs), 1))


def extract_array(y: np.ndarray, sr: int = features_a.SR) -> dict[str, float]:
    import librosa

    hop = features_a.HOP
    d: dict[str, float] = {}

    # ---- 谐波/打击分离 ----
    y_h, y_p = librosa.effects.hpss(y)
    e_h, e_p = float(np.sum(y_h ** 2)), float(np.sum(y_p ** 2))
    d["harmonic_ratio"] = e_h / max(e_h + e_p, 1e-12)

    # ---- 色度（CQT，对调性比 STFT 色度更可靠）----
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr, hop_length=hop)
    cmean = chroma.mean(axis=1)

    rel, mode = _rotate_to_tonic(cmean)
    d.update(mode)
    for i in range(12):
        d[f"rel_chroma_{i}"] = float(rel[i])
    d["rel_chroma_spread"] = float(rel.std())

    # 和声变化率：相邻帧色度向量的欧氏距离
    flux = np.sqrt(((np.diff(chroma, axis=1)) ** 2).sum(axis=0))
    d["chroma_flux_mean"] = float(flux.mean()) if flux.size else 0.0
    d["chroma_flux_std"] = float(flux.std()) if flux.size else 0.0

    # 色度熵：调性明确 → 低熵；无调性/噪声 → 高熵
    p = chroma / np.maximum(chroma.sum(axis=0, keepdims=True), 1e-12)
    d["chroma_entropy"] = float(np.mean(-(p * np.log(p + 1e-12)).sum(axis=0)))

    # ---- Tonnetz（五度圈 / 大三度环 / 小三度环）----
    ton = librosa.feature.tonnetz(chroma=chroma, sr=sr)
    for i in range(ton.shape[0]):
        d[f"tonnetz{i}_mean"] = float(ton[i].mean())
        d[f"tonnetz{i}_std"] = float(ton[i].std())

    # ---- 协和度 ----
    d["chroma_consonance"] = _chroma_consonance(cmean)

    S = np.abs(librosa.stft(y, n_fft=features_a.N_FFT, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=features_a.N_FFT)
    rough = []
    for t in range(0, S.shape[1], max(S.shape[1] // 40, 1)):
        col = S[:, t]
        idx = np.argsort(col)[-N_PEAKS:]
        idx = idx[col[idx] > col.max() * 0.02]
        if idx.size >= 2:
            rough.append(_plomp_levelt(freqs[idx], col[idx] / max(col.max(), 1e-12)))
    d["roughness_pl_mean"] = float(np.mean(rough)) if rough else 0.0
    d["roughness_pl_std"] = float(np.std(rough)) if rough else 0.0

    # ---- 基频（piptrack：比 pyin 快两个数量级，此处只需分布统计）----
    pitches, mags = librosa.piptrack(y=y_h, sr=sr, hop_length=hop,
                                     fmin=PITCH_FMIN, fmax=PITCH_FMAX)
    sel = mags > np.median(mags[mags > 0]) if np.any(mags > 0) else np.zeros_like(mags, bool)
    f0 = pitches[sel]
    f0 = f0[f0 > 0]
    d["f0_median"] = float(np.median(f0)) if f0.size else 0.0
    d["f0_iqr"] = float(np.subtract(*np.percentile(f0, [75, 25]))) if f0.size else 0.0
    d["voiced_frac"] = float(sel.any(axis=0).mean())
    return d


def extract(path: str, *, normalize_loudness: bool = False) -> dict[str, float]:
    y, sr = features_a.load_prepared(path, normalize_loudness=normalize_loudness)
    return extract_array(y, sr)
