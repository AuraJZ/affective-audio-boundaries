"""调性是不是域之墙的机制？—— 判据先于数据。

    .venv/Scripts/python scripts/59_tonality_domain_wall.py

## 问题

`step15` 确立了域之墙是**现象**：同域迁移 0.592，跨域 0.054。
但没有解释**为什么**。

一个可检验的机制假设：**调性结构是音乐域独有的**。环境声（雨、交通、
人声环境）几乎没有稳定的调式与和声。若音乐模型所依赖的表征中有相当一部分
是调性，那这部分在跨越边界时**不仅无用，而且有害** —— 它会把噪声当信号。

## 判据（写在跑之前）

| # | 预期 | 若成立 |
|---|---|---|
| 1 | **音乐域内** A+D > A | 调性在域内有信息 |
| 2 | **音乐 → 环境声** A+D **≤** A | 调性是不可共享的那部分 ⭐ |
| 3 | **环境声域内** A+D ≈ A | 环境声本就没有调性结构 |

**第 2 条是关键**：加特征通常不会让迁移变差。若加了调性反而使跨域迁移下降，
那是「该表征为音乐域专有」的直接证据，而不只是相关性描述。

若第 2 条不成立（A+D 在跨域上也不劣于 A），则调性只是**又一个**
不可迁移的特征，不构成对墙的机制解释 —— 结论须相应收窄。
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import (
    N_SPLITS, SEED, build_regressors, feature_cols, reg_score,
)

FEAT = REPO_ROOT / "data" / "features"
SETS = {"deam": "music", "pmemo": "music",
        "soundtracks": "music", "emo_mix": "ambient"}


def main() -> None:
    a_cols = feature_cols(pd.read_parquet(FEAT / "layer_a_reg_deam_ln.parquet"))
    data, d_cols = {}, None
    for k in SETS:
        A = pd.read_parquet(FEAT / f"layer_a_reg_{k}_ln.parquet")
        D = pd.read_parquet(FEAT / f"layer_d_reg_{k}_ln.parquet")
        lab = pd.read_csv(REPO_ROOT / "data" / f"reg_{k}.csv")
        if d_cols is None:
            d_cols = [c for c in D.columns if c != "clip_id"]
        M = lab.merge(A[["clip_id"] + a_cols], on="clip_id").merge(D, on="clip_id")
        data[k] = M

    X = {"A": {k: M[a_cols].to_numpy(dtype=float) for k, M in data.items()},
         "A+D": {k: M[a_cols + d_cols].to_numpy(dtype=float)
                 for k, M in data.items()}}
    y = {k: M["arousal"].to_numpy(dtype=float) for k, M in data.items()}
    g = {k: M["group"].to_numpy() for k, M in data.items()}
    algos = list(build_regressors())

    with RunRecord(
        "tonality_domain_wall", seed=SEED,
        params={"n_a": len(a_cols), "n_d": len(d_cols),
                "corpora": {k: int(len(v)) for k, v in y.items()},
                "hypothesis": "调性结构为音乐域专有；跨域时不仅无用而且有害",
                "predictions": ["音乐域内 A+D > A",
                                "音乐→环境声 A+D ≤ A（关键）",
                                "环境声域内 A+D ≈ A"]},
    ) as run:
        rows = []
        for tag in ("A", "A+D"):
            for ds in SETS:
                folds = list(GroupKFold(n_splits=N_SPLITS)
                             .split(X[tag][ds], y[ds], g[ds]))
                best = -9
                for algo in algos:
                    p = np.zeros(len(y[ds]))
                    for tr, te in folds:
                        m = build_regressors()[algo]
                        m.fit(X[tag][ds][tr], y[ds][tr])
                        p[te] = m.predict(X[tag][ds][te])
                    best = max(best, reg_score(y[ds], p)["spearman"])
                rows.append({"features": tag, "src": ds, "dst": ds,
                             "kind": "within", "spearman": best})
            for src, dst in itertools.permutations(SETS, 2):
                for algo in algos:
                    m = build_regressors()[algo].fit(X[tag][src], y[src])
                    s = reg_score(y[dst], m.predict(X[tag][dst]))["spearman"]
                    rows.append({"features": tag, "src": src, "dst": dst,
                                 "kind": "same-domain" if SETS[src] == SETS[dst]
                                 else "cross-domain", "spearman": s,
                                 "algo": algo})
            print(f"  {tag} 完成", flush=True)

        R = pd.DataFrame(rows)
        summ = (R[R.kind != "within"].groupby(["features", "kind"]).spearman
                .mean().unstack(0))

        # ---- 三条预测的逐条裁定 ----
        mus_in = R[(R.kind == "within") & (R.src != "emo_mix")].groupby("features").spearman.mean()
        m2a = R[(R.kind == "cross-domain") & (R.src != "emo_mix")].groupby("features").spearman.mean()
        a2a = R[(R.kind == "within") & (R.src == "emo_mix")].groupby("features").spearman.mean()

        preds = {
            "music_within_improves": {"A": float(mus_in["A"]), "A+D": float(mus_in["A+D"]),
                                      "pass": bool(mus_in["A+D"] > mus_in["A"])},
            "music_to_ambient_not_better": {"A": float(m2a["A"]), "A+D": float(m2a["A+D"]),
                                            "pass": bool(m2a["A+D"] <= m2a["A"])},
            "ambient_within_unchanged": {"A": float(a2a["A"]), "A+D": float(a2a["A+D"]),
                                         "pass": bool(abs(a2a["A+D"] - a2a["A"]) < 0.05)},
        }
        run.log_metric("summary", summ.round(4).to_dict())
        run.log_metric("predictions", preds)
        R.to_csv(REPO_ROOT / "reports" / "source_data" /
                 "tonality_domain_wall.csv", index=False)

        print("\n" + "=" * 74)
        print(f"run_id: {run.run_id}")
        print("\n--- 迁移均值 ---")
        print(summ.round(3).to_string())

        print("\n--- 三条预测 ---")
        names = {"music_within_improves": "音乐域内 A+D > A",
                 "music_to_ambient_not_better": "音乐 → 环境声 A+D ≤ A ⭐",
                 "ambient_within_unchanged": "环境声域内 A+D ≈ A"}
        for k, v in preds.items():
            print(f"  {'✅' if v['pass'] else '❌'} {names[k]:26s} "
                  f"A {v['A']:+.3f} → A+D {v['A+D']:+.3f}"
                  f"（Δ {v['A+D']-v['A']:+.3f}）")

        print("\n" + "=" * 74)
        if preds["music_to_ambient_not_better"]["pass"] and \
           preds["music_within_improves"]["pass"]:
            d = preds["music_to_ambient_not_better"]
            print("→ **调性在域内有信息，跨域时无用甚至有害** ——")
            print(f"   音乐→环境声 {d['A']:+.3f} → {d['A+D']:+.3f}。")
            print("   这为域之墙提供了机制层面的解释：")
            print("   墙的一部分是「音乐域专有的表征在另一域中不成立」。")
        else:
            print("→ 关键预测未中：调性只是**又一个**不可迁移的特征，")
            print("   不构成对墙的机制解释。A2 的表述维持现状，不作机制主张。")


if __name__ == "__main__":
    main()
