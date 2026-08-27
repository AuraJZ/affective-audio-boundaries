r"""A2 追加：「经评分中转」的增益是真的，还是任何良态方向都行。

    .venv/Scripts/python scripts/90_mediated_route_control.py

## 要排除什么

`89` 发现一个反直觉的事实：

    声学 → 皮电（直接回归）          ρ = 0.032
    声学 → 预测评分 → 皮电（中转）    ρ = 0.119     ← 4 倍

两者都是同一批 161 维特征的函数。差别在于**方向是怎么定下来的**：
直接回归要用信噪比极低的皮电来定方向，岭回归把系数压向零；
而中转路线用信号极强的评分（ρ=0.82）定方向，方向估得准。

**但这需要一个对照。** 若声学空间里**任何一个良态方向**都能与皮电相关到 0.12，
那这个增益与「唤醒度/效价」无关，只是「稳定的低维投影比过拟合的高维回归好」。

## 三个对照，逐个更严

| 对照 | 问什么 |
|---|---|
| **声学 PC1–PC5** | 无监督的主方向能否达到同样水平 |
| **随机方向** | 在特征协方差下随机取方向的分布，0.119 落在第几分位 |
| **打乱的评分** | 用打乱后的评分训练中转模型 —— 破坏「评分—歌曲」对应但保留一切别的 |

第三个最关键：它保留了「用一个强信号目标定方向」这件事本身，
只破坏该目标与歌曲的真实对应。若它也给 0.12，则增益来自流程而非评分内容。

## 判据先写在这里

- 三个对照都远低于 0.119 → 中转增益**特异于评分内容**，可写进正文并作为产品建议
- PC1 或随机方向也能到 0.12 → 增益只是「低维稳定投影」，须如此表述，不能归功于评分
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")

FEAT = REPO_ROOT / "data" / "features"
PM = REPO_ROOT / "data" / "raw" / "PMEmo" / "PMEmo2019"
OUT = REPO_ROOT / "reports" / "source_data" / "mediated_route_control.csv"
SEED = 20260801
EDA = ["scr_rate", "phasic_mean", "scl_slope"]
RATINGS = ["Arousal(mean)", "Valence(mean)"]
N_PC = 120
N_RAND = 500
N_SHUFFLE = 200
MIN_LISTENERS = 8


def oof_predict(X, y, groups):
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = make_pipeline(StandardScaler(), PCA(N_PC),
                          RidgeCV(alphas=np.logspace(-2, 4, 25)))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("90_mediated_route_control", SEED)

    from soundml.regression import feature_cols
    a_cols = feature_cols(pd.read_parquet(FEAT / "layer_a_reg_deam_ln.parquet"))
    A = pd.read_parquet(FEAT / "layer_a_reg_pmemo_ln.parquet")
    Dl = pd.read_parquet(FEAT / "layer_d_reg_pmemo_ln.parquet")
    d_cols = [c for c in Dl.columns if c != "clip_id"]
    F = A[["clip_id"] + a_cols].merge(Dl, on="clip_id")
    F["musicId"] = F["clip_id"].astype(str).str.removeprefix("pmemo_").astype(int)
    feats = a_cols + d_cols

    Am = pd.read_csv(PM / "annotations" / "static_annotations.csv")
    T = pd.read_parquet(FEAT / "pmemo_eda_song_level.parquet")
    for m in EDA:
        g = T.groupby("listener")[m]
        T[m] = (T[m] - g.transform("mean")) / (g.transform("std") + 1e-9)
    agg = T.groupby("musicId").agg(**{m: (m, "mean") for m in EDA},
                                   n=("listener", "size")).reset_index()
    M = (F.merge(Am, on="musicId").merge(agg, on="musicId")
         .merge(pd.read_csv(PM / "metadata.csv")[["musicId", "artist"]],
                on="musicId", how="left"))
    M["artist"] = M["artist"].fillna("u" + M.musicId.astype(str))
    M = M[M.n >= MIN_LISTENERS].reset_index(drop=True)
    X = np.nan_to_num(M[feats].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    Xs = StandardScaler().fit_transform(X)
    groups = M["artist"].to_numpy()
    print(f"{len(M)} 首 × {X.shape[1]} 维\n")

    rows = []

    # ── 基准：直接回归 vs 经评分中转 ─────────────────────────────────
    print("=" * 84)
    print("[基准] 直接回归 vs 经评分中转")
    direct = {m: float(spearmanr(oof_predict(X, M[m].to_numpy(float), groups),
                                 M[m].to_numpy(float)).statistic)
              for m in EDA}
    med = {}
    for r_ in RATINGS:
        p = oof_predict(X, M[r_].to_numpy(float), groups)
        med[r_] = {m: float(spearmanr(p, M[m].to_numpy(float)).statistic)
                   for m in EDA}
    for m in EDA:
        best_med = max(abs(med[r_][m]) for r_ in RATINGS)
        print(f"  {m:<14} 直接 {direct[m]:+.3f}   "
              f"经评分中转（最好）{best_med:+.3f}   倍数 "
              f"{best_med/max(abs(direct[m]),1e-9):.1f}×")
        rows.append({"route": "direct", "eda": m, "rho": direct[m]})
    for r_ in RATINGS:
        for m in EDA:
            rows.append({"route": f"mediated_{r_}", "eda": m, "rho": med[r_][m]})

    # ── 对照 ① 无监督主方向 ──────────────────────────────────────────
    print("\n" + "=" * 84)
    print("[对照 1] 声学 PC1–PC5 —— 无监督的主方向能否达到同样水平")
    P = PCA(5).fit_transform(Xs)
    for k in range(5):
        rr = {m: float(spearmanr(P[:, k], M[m].to_numpy(float)).statistic)
              for m in EDA}
        print(f"  PC{k+1}   " + "   ".join(f"{m}={rr[m]:+.3f}" for m in EDA))
        for m in EDA:
            rows.append({"route": f"pc{k+1}", "eda": m, "rho": rr[m]})

    # ── 对照 ② 随机方向 ──────────────────────────────────────────────
    print("\n" + "=" * 84)
    print(f"[对照 2] {N_RAND} 个随机方向 —— 中转值落在第几分位")
    for m in EDA:
        y = M[m].to_numpy(float)
        null = []
        for _ in range(N_RAND):
            w = rng.standard_normal(Xs.shape[1])
            v = Xs @ (w / np.linalg.norm(w))
            r = spearmanr(v, y).statistic
            if np.isfinite(r):
                null.append(abs(r))
        null = np.array(null)
        obs = max(abs(med[r_][m]) for r_ in RATINGS)
        pct = float((null < obs).mean())
        print(f"  {m:<14} 随机方向 |ρ| 中位 {np.median(null):.3f}   "
              f"95 分位 {np.quantile(null,0.95):.3f}   "
              f"中转值 {obs:.3f} 落在第 {pct:.1%} 分位"
              f"{'   ✅ 超出' if pct > 0.95 else '   ⚠️ 未超出'}")
        rows.append({"route": "random_dir_p95", "eda": m,
                     "rho": float(np.quantile(null, 0.95))})
        run.log_metric(f"random_dir_pct_{m}", pct)

    # ── 对照 ③ 打乱评分后重训中转模型 ────────────────────────────────
    print("\n" + "=" * 84)
    print(f"[对照 3] 打乱评分后重训中转模型（{N_SHUFFLE} 次）")
    print("         保留「用强信号目标定方向」这件事，只破坏评分与歌曲的对应")
    for r_ in RATINGS:
        y_rate = M[r_].to_numpy(float)
        for m in EDA:
            y = M[m].to_numpy(float)
            null = []
            for _ in range(N_SHUFFLE):
                p = oof_predict(X, rng.permutation(y_rate), groups)
                v = spearmanr(p, y).statistic
                if np.isfinite(v):
                    null.append(abs(v))
            null = np.array(null)
            obs = abs(med[r_][m])
            pv = float((1 + (null >= obs).sum()) / (1 + len(null)))
            print(f"  {r_:<18}{m:<14} 观测 {obs:.3f}   "
                  f"打乱后 中位 {np.median(null):.3f}   p = {pv:.4f}"
                  f"{'   ✅' if pv < 0.05 else '   ⚠️'}", flush=True)
            rows.append({"route": f"shuffled_{r_}", "eda": m,
                         "rho": float(np.median(null)), "p": pv})
            run.log_metric(f"shuffle_p_{r_}_{m}", pv)

    D = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT, index=False, encoding="utf-8")
    run.write()
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
