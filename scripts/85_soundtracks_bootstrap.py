r"""Soundtracks 上「调性补上了 happy 缺口」这一论证的不确定性估计。

    .venv/Scripts/python scripts/85_soundtracks_bootstrap.py

## 为什么必须做

论文第三部分（表征缺口）全部压在一个对比上：**happy 增益 +0.132
对其余七轴均值 +0.045**。但 Soundtracks 只有 n = 360，五折意味着每折 72 段，
而且是八个情绪维度。审稿意见（R1-M3）说得直接：

> 若 +0.132 的 95% 置信区间与 +0.045 重叠，第三部分的核心论证就不成立。

原文只给了点估计，没有任何区间。这是个必须回答的问题。

## 三件事

| | |
|---|---|
| **① 逐轴增益的 bootstrap CI** | 按**片段**重抽，看 happy 的区间与其余七轴的是否分开 |
| **② 六个算法各自的增益** | 原文只报「六者最优」，重复取最优会引入乐观偏差（R3-m4） |
| **③ D 单独 vs A 单独的对比** | 「39 维调性 0.580 > 122 维谱时 0.474」同样需要区间 |

## bootstrap 的边界

按片段重抽 **折外预测**，模型本身不重拟合。这给出的是
「给定已拟合的模型，ρ 的抽样不确定性」—— 正是「+0.132 与 +0.045
是否可区分」所需要的量。

它**不包含**训练集重抽带来的模型不确定性，因此是偏窄的区间。
若在这个偏窄的区间下两者仍然重叠，论证就更站不住；
若不重叠，则只能说「在模型固定的前提下可区分」。这一限制须写进正文。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import (N_SPLITS, SEED, build_regressors, feature_cols,
                                reg_score)

warnings.filterwarnings("ignore")

FEAT = REPO_ROOT / "data" / "features"
OUT = REPO_ROOT / "reports" / "source_data" / "soundtracks_bootstrap.csv"
PRED = FEAT / "soundtracks_oof_predictions.parquet"
DATASET = "reg_soundtracks"
AXES = ["happy", "tension", "tender", "valence", "fear", "anger", "sad", "energy"]
N_BOOT = 5000


def build_predictions() -> pd.DataFrame:
    """跑一次 CV，把**全部六个算法**的折外预测存下来。

    特征列以 DEAM 为准显式取 —— Soundtracks 的表里带着 energy/tension 等
    九个标签列，任何「排除已知元数据」式的写法都会把它们当特征喂进去。
    这是 `40`/`55` 已经定下的口径。
    """
    lab = pd.read_csv(REPO_ROOT / "data" / f"{DATASET}.csv")
    A = pd.read_parquet(FEAT / f"layer_a_{DATASET}_ln.parquet")
    Dl = pd.read_parquet(FEAT / f"layer_d_{DATASET}_ln.parquet")
    a_cols = feature_cols(pd.read_parquet(FEAT / "layer_a_reg_deam_ln.parquet"))
    d_cols = [c for c in Dl.columns if c != "clip_id"]
    leak = [c for c in a_cols + d_cols if c in AXES + ["arousal", "valence"]]
    assert not leak, f"标签泄漏：{leak}"

    M = lab.merge(A[["clip_id"] + a_cols], on="clip_id").merge(Dl, on="clip_id")
    g = M["group"].to_numpy()
    XA, XD = M[a_cols].to_numpy(float), M[d_cols].to_numpy(float)
    sets = {"A": XA, "D": XD, "AD": np.c_[XA, XD]}
    print(f"  片段 {len(M)}   A {XA.shape[1]} 维   D {XD.shape[1]} 维")

    rows = []
    for ax in AXES:
        if ax not in M.columns:
            continue
        y = M[ax].to_numpy(float)
        folds = list(GroupKFold(n_splits=N_SPLITS).split(XA, y, g))
        for sname, X in sets.items():
            for algo in build_regressors():
                p = np.zeros(len(y))
                for tr, te in folds:
                    m = build_regressors()[algo]
                    m.fit(X[tr], y[tr])
                    p[te] = m.predict(X[te])
                rows.append(pd.DataFrame({"axis": ax, "featset": sname,
                                          "algo": algo, "clip": M.clip_id,
                                          "y": y, "pred": p}))
        print(f"    {ax}", flush=True)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("85_soundtracks_bootstrap", SEED)

    print("=" * 80)
    print("[1/3] 折外预测")
    if PRED.exists():
        P = pd.read_parquet(PRED)
        print(f"  复用 {PRED.name}")
    else:
        P = build_predictions()
        P.to_parquet(PRED, index=False)
    print(f"  {len(P)} 行   轴 {P.axis.nunique()}   算法 {P.algo.nunique()}")

    # ── ② 六个算法各自的增益（不择优）────────────────────────────────
    print("\n" + "=" * 80)
    print("[2/3] 六个算法各自的 A → A+D 增益（原文只报了最优）")
    per_algo = []
    for (ax, algo), g in P.groupby(["axis", "algo"]):
        r = {}
        for s in ("A", "D", "AD"):
            x = g[g.featset == s]
            r[s] = reg_score(x.y.to_numpy(), x.pred.to_numpy())["spearman"]
        per_algo.append({"axis": ax, "algo": algo, "rho_A": r["A"],
                         "rho_D": r["D"], "rho_AD": r["AD"],
                         "gain": r["AD"] - r["A"]})
    PA = pd.DataFrame(per_algo)
    piv = PA.pivot(index="axis", columns="algo", values="gain")
    print(piv.round(3).to_string())
    hg = PA[PA.axis == "happy"].gain
    og = PA[PA.axis != "happy"].groupby("algo").gain.mean()
    print(f"\n  happy 增益跨算法：{hg.min():+.3f} ~ {hg.max():+.3f}"
          f"   中位 {hg.median():+.3f}")
    print(f"  其余七轴均值跨算法：{og.min():+.3f} ~ {og.max():+.3f}"
          f"   中位 {og.median():+.3f}")
    print(f"  **六个算法中有 {int((PA[PA.axis=='happy'].set_index('algo').gain > og).sum())}"
          f"/{len(og)} 个满足「happy 增益 > 其余七轴均值」**")
    run.log_metric("happy_gain_min_across_algos", float(hg.min()))
    run.log_metric("happy_gain_max_across_algos", float(hg.max()))

    # ── ① bootstrap CI（用原文口径：每格取六者最优）───────────────────
    print("\n" + "=" * 80)
    print(f"[3/3] 按片段重抽的 bootstrap CI（{N_BOOT} 次）")
    best = {(ax, s): PA.loc[PA[PA.axis == ax][f"rho_{s}"].idxmax(), "algo"]
            for ax in P.axis.unique() for s in ("A", "D", "AD")}
    # P["clip"] 而非 P.clip —— `clip` 撞上 DataFrame.clip() 方法，
    # 属性式访问会静默取到方法对象而不是列。
    clips = P["clip"].unique()
    idx = {(ax, s): P[(P.axis == ax) & (P.featset == s)
                      & (P.algo == best[(ax, s)])].set_index("clip")
           for ax in P.axis.unique() for s in ("A", "D", "AD")}

    from scipy.stats import spearmanr
    boot = {ax: [] for ax in P.axis.unique()}
    boot_d = {ax: [] for ax in P.axis.unique()}
    for _ in range(N_BOOT):
        pick = rng.choice(clips, len(clips), replace=True)
        for ax in boot:
            ra, rad, rd = (spearmanr(idx[(ax, s)].loc[pick, "pred"],
                                     idx[(ax, s)].loc[pick, "y"]).statistic
                           for s in ("A", "AD", "D"))
            if np.isfinite(ra) and np.isfinite(rad):
                boot[ax].append(rad - ra)
            if np.isfinite(rd) and np.isfinite(ra):
                boot_d[ax].append(rd - ra)

    rows = []
    for ax, v in boot.items():
        a = np.array(v)
        rows.append({"axis": ax, "gain": float(a.mean()),
                     "lo": float(np.quantile(a, .025)),
                     "hi": float(np.quantile(a, .975))})
    G = pd.DataFrame(rows).sort_values("gain", ascending=False)
    print(f"  {'轴':<10}{'增益':>8}{'95% CI':>22}")
    for _, x in G.iterrows():
        mark = "  ← happy" if x.axis == "happy" else ""
        print(f"  {x.axis:<10}{x.gain:>+8.3f}   [{x.lo:+.3f}, {x.hi:+.3f}]{mark}")

    hb = np.array(boot["happy"])
    ob = np.mean([np.array(boot[a]) for a in boot if a != "happy"], axis=0)
    diff = hb - ob[: len(hb)]
    lo, hi = np.quantile(diff, [.025, .975])
    print(f"\n  happy 增益 − 其余七轴均值：{diff.mean():+.3f}   "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]")
    sep = lo > 0
    print(f"  → {'✅ 区间不含 0，第三部分的论证成立' if sep else '🔴 区间含 0，第三部分的核心论证不成立'}")
    run.log_metric("happy_minus_others", float(diff.mean()))
    run.log_metric("happy_minus_others_ci_lo", float(lo))
    run.log_metric("happy_minus_others_ci_hi", float(hi))
    run.log_metric("separated", bool(sep))

    hd = np.array(boot_d["happy"])
    dlo, dhi = np.quantile(hd, [.025, .975])
    print(f"\n  happy 上「39 维调性 − 122 维谱时」：{hd.mean():+.3f}   "
          f"95% CI [{dlo:+.3f}, {dhi:+.3f}]"
          f"   {'✅ 不含 0' if dlo > 0 else '🔴 含 0'}")
    run.log_metric("happy_D_minus_A", float(hd.mean()))
    run.log_metric("happy_D_minus_A_ci_lo", float(dlo))

    pd.concat([G.assign(kind="gain_ci"), PA.assign(kind="per_algo")]).to_csv(
        OUT, index=False, encoding="utf-8")
    run.write()
    print(f"\n⚠️ bootstrap 只重抽折外预测，模型不重拟合 —— 区间偏窄，")
    print("   不含训练集重抽带来的模型不确定性。此限制须写进正文。")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
