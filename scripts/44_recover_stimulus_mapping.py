"""恢复 ds002721 刺激码 → Soundtracks 音频的映射（有界尝试，允许失败）。

    .venv/Scripts/python scripts/44_recover_stimulus_mapping.py

## 问题

`events.json` 说「刺激码 301–360，减 300 得 mp3 编号」，但实测码达 **657**，
且 `code − 300` 指向的音频与受试评分**毫无关系**（三条独立证据）：

| 检验 | 结果 |
|---|---|
| 受试 8 项自评 × 组均值 8 量表 | 8×8 全部相关系数绝对值 ≤ 0.16，无置换对得上 |
| 谱质心 → 受试 energetic | −0.006（同一探针对组均值 energy 是 +0.374） |
| 偏移量 −3…+3 扫描 | 全部 ≈0 |

但这些码**确实是刺激身份** —— 听到同一个码的不同受试评分高度一致
（分半 0.80–0.87，ICC 0.16–0.34，p=0.002）。因此是**编号体系对不上**。

## 思路

受试对每个码给出 8 维情绪剖面；Set1 每段音频有 8 维组均值剖面。
若两套编号只是置换关系，存在一个指派使剖面对齐。
用匈牙利算法（`linear_sum_assignment`）求最大总相关。

## 必须有的零对照 ⚠️

307 × 360 的指派空间极大，**任何数据都能找出一个「最优」指派**。
因此同时在打乱的数据上跑同一套流程。

零分布必须打乱的是「**码 ↔ 剖面**」这一关联本身：在试次层面
**被试内打乱刺激码**，再重新估剖面、重新指派。

⚠️ 第一版的零分布是**打乱剖面矩阵的行**，那是空操作 ——
匈牙利指派的最优总和对行置换不变，打乱行只是给码重新贴标签。
当时零分布均值与观测值完全相等（都是 +0.6716），这个等号本身就是失效的信号。

判据：真实指派的平均剖面相关须超过零分布 **95 分位**，且
`code − 300` 若为真解，应在指派结果中大量复现。

## 失败是可接受的结局

若恢复不成功，则「声学 ↔ 脑电」一线**在本数据集上无法进行**，
须向数据集作者索取刺激清单。脑电信度结论不受影响 ——
ICC 按码分组，码在数据集内部自洽。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

TRIALS = REPO_ROOT / "data" / "features" / "ds002721_trials_raw.parquet"
Q = ["pleasant", "energetic", "tense", "angry", "afraid", "happy", "sad", "tender"]
REF = ["valence", "energy", "tension", "anger", "fear", "happy", "sad", "tender"]
MIN_SUBS = 5            # 剖面估计所需最少受试
N_NULL = 200


def profiles(D: pd.DataFrame, min_subs: int) -> pd.DataFrame:
    """每个刺激码的 8 维评分剖面（先被试内 z，再按码取均值）。"""
    cols = [f"alt_{q}" for q in Q]                 # 901–909 才是真作答码
    S = D[["subject", "stim"] + cols].copy()
    for c in cols:
        g = S.groupby("subject")[c]
        S[c] = (S[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    n = S.groupby("stim")["subject"].nunique()
    S = S[S.stim.isin(n[n >= min_subs].index)]
    P = S.groupby("stim")[cols].mean()
    return P.dropna()


def assign_quality(P: np.ndarray, C: np.ndarray) -> tuple[float, np.ndarray]:
    """匈牙利指派 + 平均剖面相关。P/C 均已按行标准化。"""
    S = P @ C.T / P.shape[1]                       # (n_code, n_clip) 余弦式相似
    r, c = linear_sum_assignment(-S)
    return float(S[r, c].mean()), c


def main() -> None:
    D = pd.read_parquet(TRIALS)
    A = pd.read_csv(REPO_ROOT / "data" / "reg_soundtracks.csv")
    A = A.set_index("stim")[REF]

    with RunRecord(
        "recover_stimulus_mapping", seed=SEED,
        params={"method": "8 维情绪剖面 + 匈牙利指派",
                "min_subjects_per_code": MIN_SUBS, "n_null": N_NULL,
                "null": "打乱受试评分后跑同一流程",
                "acceptable_outcome": "失败 —— 则声学↔脑电线在本数据集不可行"},
    ) as run:
        P = profiles(D, MIN_SUBS)
        run.log_metric("n_codes_with_profile", int(len(P)))
        print(f"可估剖面的刺激码 {len(P)}（≥{MIN_SUBS} 名受试）   "
              f"候选音频 {len(A)}")
        if len(P) < 20:
            print("🔴 可用码太少，无法尝试恢复。")
            run.log_metric("verdict", "insufficient_data")
            return

        def zrow(M):
            M = np.asarray(M, dtype=float)
            return (M - M.mean(1, keepdims=True)) / np.maximum(M.std(1, keepdims=True), 1e-9)

        Pz, Cz = zrow(P.to_numpy()), zrow(A.to_numpy())
        obs, col = assign_quality(Pz, Cz)

        # 零分布：被试内打乱刺激码 → 重估剖面 → 重新指派。
        # 打乱剖面矩阵的行是**空操作**（指派最优值对行置换不变），不可用。
        rng = np.random.default_rng(SEED)
        null = []
        for _ in range(N_NULL):
            Dp = D.copy()
            s = Dp.stim.to_numpy().copy()
            subj = Dp.subject.to_numpy()
            for u in np.unique(subj):
                m = subj == u
                s[m] = rng.permutation(s[m])
            Dp["stim"] = s
            Pp = profiles(Dp, MIN_SUBS)
            if len(Pp) >= 20:
                null.append(assign_quality(zrow(Pp.to_numpy()), Cz)[0])
        null = np.array(null)
        p = float((np.sum(null >= obs) + 1) / (N_NULL + 1))
        thr = float(np.percentile(null, 95))

        # `code − 300` 若为真解，应在指派中大量复现
        codes = P.index.to_numpy()
        clips = A.index.to_numpy()[col]
        hit = float(np.mean(codes - 300 == clips))

        run.log_metric("assignment", {
            "mean_profile_corr": obs, "null_mean": float(null.mean()),
            "null_95th": thr, "p": p,
            "frac_matching_code_minus_300": hit})

        print(f"\n  指派平均剖面相关 {obs:+.4f}")
        print(f"  零分布（被试内打乱刺激码，{len(null)} 次）"
              f"均值 {null.mean():+.4f}   95 分位 {thr:+.4f}   "
              f"SD {null.std():.4f}")
        if null.std() < 1e-6:
            print("  ⚠️ 零分布方差为零 —— 置换未生效，判决无效，不得采用。")
        print(f"  p = {p:.4f}")
        print(f"  指派结果与 `code − 300` 一致的比例 {100*hit:.1f}%")

        valid = null.size >= 50 and null.std() > 1e-6
        ok = valid and obs > thr
        run.log_metric("null_valid", bool(valid))
        run.log_metric("verdict", ("recovered" if ok else
                                   "failed" if valid else "null_invalid"))
        if not valid:
            print("\n🔴 零分布无效（方差为零或样本不足）—— **本次不给判决**。")
            return
        print("\n" + "=" * 74)
        if ok:
            print("✅ 指派质量超过零分布 —— 存在可辨的映射，可进一步核验。")
            out = pd.DataFrame({"code": codes, "recovered_clip": clips,
                                "code_minus_300": codes - 300})
            out.to_csv(REPO_ROOT / "reports" / "source_data" /
                       "ds002721_recovered_mapping.csv", index=False)
            print(f"   → reports/source_data/ds002721_recovered_mapping.csv")
        else:
            print("🔴 指派质量落在零分布之内 —— **恢复失败**。")
            print("   任何指派在 307×360 的空间里都能「看起来不错」，本次没有超过随机。")
            print("\n   结论：**声学 ↔ 脑电一线在本数据集上无法进行**，")
            print("   须向数据集作者（Daly / Nicolaou, Univ. of Reading）索取刺激清单。")
            print("\n   不受影响：脑电与主观评分的刺激层面信度 —— ")
            print("   ICC 按码分组，码在数据集内部自洽，与它指向哪个 mp3 无关。")


if __name__ == "__main__":
    main()
