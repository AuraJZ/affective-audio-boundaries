"""含连接性的完整脑电信度 —— 带功率 26 维 + 相干/不对称 130 维。

    .venv/Scripts/python scripts/48_eeg_reliability_full.py

## 为什么重算

`step17` 的「脑电刺激层面信度 ≈ 0」只用了**区域带功率**。
但 Daly et al. (2014) 在**同一批数据**上报告的效应在**通道间相干**与
**beta/gamma 不对称** —— 我此前的特征集恰好不含这两类。

**用一个不包含目标效应的特征集去否定该效应，是循环论证。**
因此在下任何结论前必须把这两类补进来重算。

## 特征

| 族 | 维度 | 来源 |
|---|---|---|
| 区域带功率变化 | 26 | `39_eeg_reliability.py` |
| 幅度平方相干 | 50 | `46_connectivity_features.py` |
| **虚部相干**（容积传导稳健） | 50 | 同上 |
| 电极对不对称（含 beta/gamma） | 30 | 同上 |
| **合计** | **156** | |

## 统计

与 `step17` 完全一致，便于直接比较：被试内 z → 单因素随机效应 ICC
（全部 1,240 试次）+ 跨受试分半（≥8 人的子集）。

156 次检验，**必须做 BH-FDR**。`step17` 里 beta_central 的 p=0.030 在
26 次检验下已不成立，156 次下更不成立 —— 本脚本直接报 q 值。

置换零分布（被试内打乱刺激码）只对 ICC 最高的 12 个跑，代价所限。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

warnings.filterwarnings("ignore")

FEAT = REPO_ROOT / "data" / "features"
QUESTIONS = ["pleasant", "energetic", "tense", "angry",
             "afraid", "happy", "sad", "tender"]
BANDS = ("delta", "theta", "alpha", "beta", "gamma")
N_PERM, N_SPLIT, MIN_SUBS_SPLIT, N_TOP_PERM = 500, 400, 8, 12


def icc_oneway(v: np.ndarray, g: np.ndarray) -> float:
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    uniq, inv = np.unique(g, return_inverse=True)
    k, N = len(uniq), len(v)
    if k < 3 or N <= k:
        return np.nan
    cnt = np.bincount(inv)
    gm = v.mean()
    gmeans = np.bincount(inv, weights=v) / cnt
    msb = float((cnt * (gmeans - gm) ** 2).sum()) / (k - 1)
    msw = float(((v - gmeans[inv]) ** 2).sum()) / (N - k)
    n0 = (N - (cnt ** 2).sum() / N) / (k - 1)
    if n0 <= 0:
        return np.nan
    va = (msb - msw) / n0
    return float(va / (va + msw)) if (va + msw) > 0 else np.nan


def split_half(D: pd.DataFrame, col: str, rng) -> float:
    per = D.groupby("stim")["subject"].nunique()
    S = D[D.stim.isin(per[per >= MIN_SUBS_SPLIT].index)]
    if S.stim.nunique() < 10:
        return np.nan
    subs = S.subject.unique()
    raw = []
    for _ in range(N_SPLIT):
        perm = rng.permutation(subs)
        a = set(perm[: len(perm) // 2])
        ga = S[S.subject.isin(a)].groupby("stim")[col].agg(["mean", "size"])
        gb = S[~S.subject.isin(a)].groupby("stim")[col].agg(["mean", "size"])
        common = ga.index[ga["size"] >= 2].intersection(gb.index[gb["size"] >= 2])
        if len(common) >= 10:
            r = spearmanr(ga.loc[common, "mean"], gb.loc[common, "mean"],
                          nan_policy="omit").statistic
            if np.isfinite(r):
                raw.append(float(r))
    if not raw:
        return np.nan
    r = np.array(raw)
    return float(np.mean(2 * r / (1 + r)))


def perm_p(D: pd.DataFrame, col: str, obs: float, rng) -> tuple[float, float]:
    v, g, s = D[col].to_numpy(), D.stim.to_numpy(), D.subject.to_numpy()
    null = []
    for _ in range(N_PERM):
        gp = g.copy()
        for u in np.unique(s):
            m = s == u
            gp[m] = rng.permutation(gp[m])
        x = icc_oneway(v, gp)
        if np.isfinite(x):
            null.append(x)
    if not null or not np.isfinite(obs):
        return np.nan, np.nan
    null = np.array(null)
    return float((np.sum(null >= obs) + 1) / (len(null) + 1)), float(null.mean())


def main() -> None:
    A = pd.read_parquet(FEAT / "ds002721_trials_raw.parquet")
    B = pd.read_parquet(FEAT / "ds002721_connectivity.parquet")
    band = [c for c in A.columns if any(c.startswith(b) for b in BANDS) or c == "faa"]
    conn = [c for c in B.columns if c.startswith(("coh_", "icoh_", "asym_"))]
    D = A[["subject", "run", "trial", "stim"] + band].merge(
        B[["subject", "run", "trial"] + conn], on=["subject", "run", "trial"],
        how="inner")
    feats = band + conn

    fam = {c: ("band power" if c in band else
               "coherence" if c.startswith("coh_") else
               "imaginary coherence" if c.startswith("icoh_") else "asymmetry")
           for c in feats}

    with RunRecord(
        "ds002721_reliability_full", seed=SEED,
        params={"n_features": len(feats),
                "families": {k: sum(v == k for v in fam.values())
                             for k in set(fam.values())},
                "n_trials": int(len(D)), "n_perm": N_PERM,
                "why": "step17 只用带功率，不含 Daly et al. 报告效应的相干与"
                       "beta/gamma 不对称；不补齐则结论是循环论证",
                "multiple_comparison": "BH-FDR over all features"},
    ) as run:
        rng = np.random.default_rng(SEED)
        for c in feats:
            g = D.groupby("subject")[c]
            D[f"z_{c}"] = ((D[c] - g.transform("mean"))
                           / g.transform("std").replace(0, np.nan))

        res = []
        for i, c in enumerate(feats, 1):
            s = D.dropna(subset=[f"z_{c}"])
            if len(s) < 200:
                continue
            res.append({"measure": c, "family": fam[c], "n": int(len(s)),
                        "icc": icc_oneway(s[f"z_{c}"].to_numpy(),
                                          s.stim.to_numpy()),
                        "split_half_sb": split_half(s, f"z_{c}", rng)})
            if i % 40 == 0:
                print(f"  {i}/{len(feats)}", flush=True)
        R = pd.DataFrame(res).sort_values("icc", ascending=False)

        top = R.head(N_TOP_PERM).measure.tolist()
        pv = {}
        for c in top:
            s = D.dropna(subset=[f"z_{c}"])
            p, nm = perm_p(s, f"z_{c}", float(R.loc[R.measure == c, "icc"].iloc[0]), rng)
            pv[c] = (p, nm)
        R["perm_p"] = R.measure.map(lambda c: pv.get(c, (np.nan, np.nan))[0])
        R["null_icc"] = R.measure.map(lambda c: pv.get(c, (np.nan, np.nan))[1])
        # BH-FDR 按全部特征数校正（只跑了前 12 个置换，故用保守的 len(R)）
        m = R.perm_p.notna()
        order = R.loc[m, "perm_p"].rank(method="first")
        R.loc[m, "q"] = np.minimum(1.0, R.loc[m, "perm_p"] * len(R) / order)

        run.log_metric("n_features_tested", int(len(R)))
        run.log_metric("best_by_family",
                       R.groupby("family").icc.max().round(4).to_dict())
        run.log_metric("n_fdr_significant", int((R.q < 0.05).sum()))
        R.to_csv(REPO_ROOT / "reports" / "source_data" /
                 "ds002721_reliability_full.csv", index=False)

        print("\n" + "=" * 82)
        print(f"run_id: {run.run_id}   试次 {len(D):,}   特征 {len(R)}")
        print("\n--- 各特征族的最强 ICC ---")
        for f_, g in R.groupby("family"):
            b = g.iloc[0]
            print(f"  {f_:22s} n={len(g):3d}   最强 {b.measure:22s} "
                  f"ICC {b.icc:+.3f}   分半SB {b.split_half_sb:+.3f}")

        print(f"\n--- 全体前 {N_TOP_PERM}（置换 + BH-FDR，校正基数 {len(R)}）---")
        for r_ in R.head(N_TOP_PERM).itertuples():
            q = "" if not np.isfinite(getattr(r_, "q", np.nan)) else f"  q={r_.q:.3f}"
            print(f"  {r_.measure:24s} {r_.family:20s} ICC {r_.icc:+.3f}   "
                  f"分半SB {r_.split_half_sb:+.3f}   p={r_.perm_p:.4f}{q}")

        nsig = int((R.q < 0.05).sum())
        print("\n" + "=" * 82)
        if nsig:
            print(f"→ {nsig} 个特征在 FDR<0.05 下显著。最强 ICC {R.icc.max():+.3f}，")
            print("  仍远低于同批受试主观评分的 0.335。结论方向不变，但须报告这些特征。")
        else:
            print("→ **补上相干与不对称后，仍无特征通过 FDR<0.05。**")
            print(f"  最强 ICC {R.icc.max():+.3f}（{R.iloc[0].measure}，"
                  f"{R.iloc[0].family}），主观评分为 +0.335。")
            print("  step17 的结论在包含文献效应类型的特征集上依然成立。")


if __name__ == "__main__":
    main()
