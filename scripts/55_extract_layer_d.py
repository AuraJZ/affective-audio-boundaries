"""提取 Layer D（调性与和声），并检验它是否真的补上了 happy 轴的缺口。

    .venv/Scripts/python scripts/55_extract_layer_d.py --dataset reg_soundtracks
    .venv/Scripts/python scripts/55_extract_layer_d.py --dataset reg_soundtracks --evaluate

## 判据先于数据

Layer D 是为一个**具体的、被数据指出的缺口**而加的：
Soundtracks 七个情绪轴中 `happy` 是唯一显著弱的（R²=0.138，其余 ≥0.35）。

因此判据不是「A+D 是否比 A 好」（加特征几乎总能好一点），而是：

| 预期 | 若成立 | 若不成立 |
|---|---|---|
| **`happy` 的增益应显著大于其他轴** | 缺口确为调性 | Layer D 只是泛泛地加了信息量 |
| `mode_score` 应在 `happy` 的重要性中靠前 | 机制与假设一致 | 增益来源不明，需重新解释 |

**只看总体 R² 上升就宣布成功，是我在 B1 上犯过的那类错误**——
一个数字在没有针对性预期时，无法区分「补上了缺口」与「多给了自由度」。
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from soundml import features_a, features_d
from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import (
    N_SPLITS, SEED, build_regressors, feature_cols, reg_score,
)

FEAT = REPO_ROOT / "data" / "features"
AXES = ["happy", "tension", "tender", "valence", "fear", "anger", "sad", "energy"]


def extract(dataset: str, normalize: bool) -> None:
    labels = pd.read_csv(REPO_ROOT / "data" / f"{dataset}.csv")
    tag = dataset + ("_ln" if normalize else "")
    out = FEAT / f"layer_d_{tag}.parquet"

    with RunRecord(
        f"extract_layer_d_{tag}", seed=0,
        params={"dataset": dataset, "normalize_loudness": normalize,
                "extractor_version": features_d.EXTRACTOR_VERSION,
                "sr": features_a.SR, "hop": features_a.HOP,
                "max_duration_s": features_a.MAX_DURATION_S,
                "gap_addressed": "Layer A 无调性信息；Soundtracks 的 happy 轴 R²=0.138"},
    ) as run:
        rows, failed = [], []
        t0 = time.perf_counter()
        for i, rec in enumerate(labels.itertuples(index=False), 1):
            try:
                rows.append({"clip_id": rec.clip_id,
                             **features_d.extract(rec.path,
                                                  normalize_loudness=normalize)})
            except Exception as e:                           # noqa: BLE001
                failed.append((rec.clip_id, f"{type(e).__name__}: {e}"))
            if i % 60 == 0:
                print(f"  {i}/{len(labels)}  "
                      f"({i/(time.perf_counter()-t0):.1f} clip/s, 失败 {len(failed)})",
                      flush=True)
        D = pd.DataFrame(rows)
        D.to_parquet(out)
        run.log_metric("n_clips", int(len(D)))
        run.log_metric("n_features", int(D.shape[1] - 1))
        run.log_metric("n_failed", len(failed))
        print(f"\n→ {out.name}   {len(D)} 段 × {D.shape[1]-1} 维（失败 {len(failed)}）")
        for cid, why in failed[:5]:
            print(f"   {cid}  {why}")


def evaluate(dataset: str) -> None:
    lab = pd.read_csv(REPO_ROOT / "data" / f"{dataset}.csv")
    A = pd.read_parquet(FEAT / f"layer_a_{dataset}_ln.parquet")
    Dl = pd.read_parquet(FEAT / f"layer_d_{dataset}_ln.parquet")
    a_cols = feature_cols(pd.read_parquet(FEAT / "layer_a_reg_deam_ln.parquet"))
    d_cols = [c for c in Dl.columns if c != "clip_id"]

    M = lab.merge(A[["clip_id"] + a_cols], on="clip_id").merge(Dl, on="clip_id")
    g = M["group"].to_numpy()
    XA = M[a_cols].to_numpy(dtype=float)
    XD = M[d_cols].to_numpy(dtype=float)
    XAD = np.c_[XA, XD]
    algos = list(build_regressors())

    with RunRecord(
        "layer_d_evaluation", seed=SEED,
        params={"dataset": dataset, "n_clips": int(len(M)),
                "n_a": len(a_cols), "n_d": len(d_cols),
                "axes": AXES, "cv": f"GroupKFold({N_SPLITS})",
                "prediction": "happy 的增益应显著大于其他轴；否则 Layer D 只是"
                              "泛泛增加信息量，不构成「补上了调性缺口」"},
    ) as run:
        res = []
        for ax in AXES:
            if ax not in M.columns:
                continue
            y = M[ax].to_numpy(dtype=float)
            folds = list(GroupKFold(n_splits=N_SPLITS).split(XA, y, g))
            row = {"axis": ax}
            for name, X in (("A", XA), ("D", XD), ("AD", XAD)):
                best = None
                for algo in algos:
                    p = np.zeros(len(y))
                    for tr, te in folds:
                        m = build_regressors()[algo]
                        m.fit(X[tr], y[tr])
                        p[te] = m.predict(X[te])
                    s = reg_score(y, p)
                    if best is None or s["spearman"] > best["spearman"]:
                        best = {**s, "algo": algo}
                row[f"rho_{name}"] = best["spearman"]
                row[f"r2_{name}"] = best["r2"]
                row[f"algo_{name}"] = best["algo"]
            row["gain_rho"] = row["rho_AD"] - row["rho_A"]
            row["gain_r2"] = row["r2_AD"] - row["r2_A"]
            res.append(row)
            print(f"  {ax:10s} A {row['rho_A']:+.3f} → A+D {row['rho_AD']:+.3f}"
                  f"   （D 单独 {row['rho_D']:+.3f}）", flush=True)
        R = pd.DataFrame(res)

        happy_gain = float(R.loc[R.axis == "happy", "gain_rho"].iloc[0])
        other_gain = float(R.loc[R.axis != "happy", "gain_rho"].mean())
        targeted = bool(happy_gain > other_gain * 2 and happy_gain > 0.05)
        run.log_metric("results", R.to_dict("records"))
        run.log_metric("happy_gain_rho", happy_gain)
        run.log_metric("other_axes_mean_gain_rho", other_gain)
        run.log_metric("targeted_gain", targeted)

        # ---- 机制核查：mode_score 是否在 happy 上真的重要 ----
        from sklearn.ensemble import RandomForestRegressor
        imp = {}
        for ax in ("happy", "tension"):
            if ax not in M.columns:
                continue
            rf = RandomForestRegressor(n_estimators=400, random_state=SEED,
                                       n_jobs=-1).fit(XAD, M[ax].to_numpy())
            names = a_cols + d_cols
            order = np.argsort(rf.feature_importances_)[::-1]
            imp[ax] = [(names[i], float(rf.feature_importances_[i]))
                       for i in order[:8]]
            d_ranks = {names[i]: int(np.where(order == i)[0][0]) + 1
                       for i in range(len(names)) if names[i] in
                       ("mode_score", "chroma_consonance", "key_major_strength")}
            imp[f"{ax}_key_feature_ranks"] = d_ranks
        run.log_metric("feature_importance", imp)

        print("\n" + "=" * 78)
        print(f"run_id: {run.run_id}   {len(M)} 段   A {len(a_cols)} 维 + D {len(d_cols)} 维")
        print(f"\n{'轴':<10}{'A':>8}{'D 单独':>9}{'A+D':>8}{'ρ 增益':>9}{'R² 增益':>9}")
        for r_ in R.sort_values("gain_rho", ascending=False).itertuples():
            print(f"{r_.axis:<10}{r_.rho_A:>+8.3f}{r_.rho_D:>+9.3f}"
                  f"{r_.rho_AD:>+8.3f}{r_.gain_rho:>+9.3f}"
                  f"{r_.gain_r2:>+9.3f}")

        print(f"\n--- 判据 ---")
        print(f"  happy 的 ρ 增益      {happy_gain:+.3f}")
        print(f"  其余各轴平均增益     {other_gain:+.3f}")
        print(f"  {'✅ 增益具有针对性' if targeted else '❌ 增益不具针对性'}")

        for ax in ("happy", "tension"):
            if ax in imp:
                print(f"\n--- {ax} 的前 8 个重要特征 ---")
                for nm, v in imp[ax]:
                    star = " ⭐" if nm in d_cols else ""
                    print(f"    {nm:24s} {v:.4f}{star}")
                print(f"    调性关键特征排名：{imp[f'{ax}_key_feature_ranks']}")

        R.to_csv(REPO_ROOT / "reports" / "source_data" /
                 "layer_d_evaluation.csv", index=False)

        print("\n" + "=" * 78)
        if targeted:
            print("→ **增益集中在 happy 上，与「缺口是调性」的假设一致。**")
        else:
            print("→ 增益不具针对性 —— Layer D 提供的是泛泛的信息量，")
            print("  不能主张「补上了调性缺口」。须重新解释增益来源。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="reg_soundtracks")
    ap.add_argument("--normalize-loudness", action="store_true", default=True)
    ap.add_argument("--evaluate", action="store_true",
                    help="只做评估（特征需已提取）")
    args = ap.parse_args()

    if not args.evaluate:
        extract(args.dataset, args.normalize_loudness)
    evaluate(args.dataset)


if __name__ == "__main__":
    main()
