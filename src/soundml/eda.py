"""PMEmo 皮电（EDA）生理唤醒指标。

## 为什么重要

整个项目最大的软肋是**构念鸿沟**：我们的标签是「主观 arousal 打分」，
而产品要的是「助眠」。两者之间隔着一层。

PMEmo 附带 **794 段音频 × 457 名被试的皮电记录** ——
皮电是**生理**指标，不受语言、量表锚点、社会期许影响，比主观打分硬得多。
这批数据一直在硬盘上没用过。

它能回答一个关键问题：
**用主观 arousal 学到的声学规律，能不能预测生理唤醒？**
若能，构念鸿沟收窄；若不能，说明我们一直在建模「人怎么描述声音」
而不是「声音怎么影响人」。

## 指标构造

皮电分两个成分：

- **紧张性（tonic, SCL）**：缓慢基线漂移，反映整体唤醒水平
- **相位性（phasic, SCR）**：快速峰，反映事件性唤醒响应

两个都用：

    tonic_rise = mean(后半段) − mean(前 10% 基线)
    phasic     = 高通后信号的标准差（≈ SCR 活动强度）

## 被试标准化（关键）

被试间皮电绝对值差异极大（同一时刻 3.1 vs 15.1 μS），
且受电极位置、皮肤湿度、季节影响。因此**必须在被试内标准化**：
先收集每名被试在其全部音频上的取值，再做 z 变换，最后跨被试聚合。

不做这一步，聚合结果会被少数高基线被试主导。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal

FS = 50.0          # PMEmo EDA 采样率（时间列步长 0.02 s）
BASELINE_FRAC = 0.10
HIGHPASS_HZ = 0.05  # 分离 phasic 成分


def _clip_features(t: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """单个被试在单段音频上的 (tonic_rise, phasic)。"""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    if ok.sum() < FS * 2:  # 不足 2 秒
        return np.nan, np.nan
    x = x[ok]

    n_base = max(int(len(x) * BASELINE_FRAC), 1)
    baseline = float(np.mean(x[:n_base]))
    tonic_rise = float(np.mean(x[len(x) // 2:]) - baseline)

    # 一阶 Butterworth 高通分离 phasic
    try:
        b, a = scipy.signal.butter(1, HIGHPASS_HZ / (FS / 2), btype="high")
        phasic = float(np.std(scipy.signal.filtfilt(b, a, x)))
    except Exception:
        phasic = float(np.std(np.diff(x)))
    return tonic_rise, phasic


def load_eda_arousal(eda_dir: Path) -> pd.DataFrame:
    """扫描全部 `<musicId>_EDA.csv`，返回逐音频的生理唤醒指标。

    返回列：music_id, eda_tonic, eda_phasic, eda_arousal, n_subjects
    `eda_arousal` = 两个被试内 z 分数的均值。
    """
    records = []  # (music_id, subject, tonic, phasic)
    for f in sorted(eda_dir.glob("*_EDA.csv")):
        mid = f.stem.split("_")[0]
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.shape[1] < 2:
            continue
        t = df.iloc[:, 0].to_numpy(dtype=np.float64)
        for subj in df.columns[1:]:
            tonic, phasic = _clip_features(t, df[subj].to_numpy())
            if np.isfinite(tonic) and np.isfinite(phasic):
                records.append((mid, str(subj), tonic, phasic))

    R = pd.DataFrame(records, columns=["music_id", "subject", "tonic", "phasic"])
    if R.empty:
        return pd.DataFrame(columns=["music_id", "eda_tonic", "eda_phasic",
                                     "eda_arousal", "n_subjects"])

    # --- 被试内 z 标准化 ---
    for col in ("tonic", "phasic"):
        grp = R.groupby("subject")[col]
        R[f"z_{col}"] = (R[col] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)

    agg = R.groupby("music_id").agg(
        eda_tonic=("z_tonic", "mean"),
        eda_phasic=("z_phasic", "mean"),
        n_subjects=("subject", "nunique"),
    ).reset_index()
    agg["eda_arousal"] = agg[["eda_tonic", "eda_phasic"]].mean(axis=1)
    return agg
