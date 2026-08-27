"""Layer A vs Layer C：手工描述符 vs 预训练嵌入。

    uv run python scripts/18_representation_comparison.py --dataset reg_emo_mix

## 两个问题

**1. 精度天花板在哪？**
若 CLAP 这样的大规模预训练嵌入也只能到 ρ≈0.77，
说明限制来自**标签噪声**而非特征表达力 —— 再堆特征无用，
该去改进标签或换测量方式。
这直接决定后续该往哪投入。

**2. 「表示冗余」的解释站得住吗？**
此前 stacking 四次失败，我们的解释是「基模型吃同一份特征、预测高度相关」。
Layer A（手工描述符）与 Layer C（深度嵌入）的差异远大于 A 与 B，
若 A+C 的 stacking 仍不成立，则该解释被彻底证伪，需另找原因。

## 附带：跨域迁移在深度嵌入上是否也崩？

域之墙目前是在手工特征上观察到的。
若 CLAP 嵌入能跨域迁移，说明墙来自特征设计而非任务本身 —— 那是好消息。
若同样崩，说明域分离是任务的固有属性。
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import N_SPLITS, SEED, build_regressors, feature_cols, reg_score

FEAT = REPO_ROOT / "data" / "features"


def load_rep(tag: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    a = pd.read_parquet(FEAT / f"layer_a_{tag}_ln.parquet")
    cols_a = feature_cols(a)
    c_path = FEAT / f"layer_c_{tag}.parquet"
    if not c_path.exists():
        return a, {"A": cols_a}
    c = pd.read_parquet(c_path)
    cols_c = [x for x in c.columns if x.startswith("clap")]
    df = a.merge(c[["clip_id", *cols_c]], on="clip_id", validate="one_to_one")
    return df, {"A": cols_a, "C": cols_c, "A+C": cols_a + cols_c}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="reg_emo_mix")
    ap.add_argument("--cross", default=None, help="另一语料 tag，用于跨域测试")
    args = ap.parse_args()

    df, reps = load_rep(args.dataset)
    if "C" not in reps:
        raise SystemExit(f"未找到 layer_c_{args.dataset}.parquet，请先跑 15_extract_layer_c.py")

    y = df["arousal"].to_numpy()
    groups = df["group"].to_numpy()
    folds = list(GroupKFold(n_splits=N_SPLITS).split(np.zeros(len(y)), y, groups))

    # 嵌入维度高（512），除树模型外加一个岭回归作为线性探针
    ALGOS = ["RF", "XGBoost", "SVR"]

    with RunRecord("representation_comparison", seed=SEED,
                   params={"dataset": args.dataset,
                           "representations": {k: len(v) for k, v in reps.items()},
                           "algorithms": ALGOS + ["RidgeProbe"],
                           "question": "精度天花板 & 表示冗余解释是否成立"}) as run:
        oof, perf = {}, {}
        for rep, cols in reps.items():
            X = SimpleImputer(strategy="median").fit_transform(df[cols].to_numpy(dtype=np.float64))
            for algo in ALGOS:
                preds = np.zeros(len(y))
                for tr, te in folds:
                    m = build_regressors()[algo]
                    m.fit(X[tr], y[tr])
                    preds[te] = m.predict(X[te])
                oof[f"{algo}|{rep}"] = preds
                perf[f"{algo}|{rep}"] = reg_score(y, preds)
            # 线性探针：嵌入质量的标准评估方式
            preds = np.zeros(len(y))
            for tr, te in folds:
                m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X[tr], y[tr])
                preds[te] = m.predict(X[te])
            oof[f"RidgeProbe|{rep}"] = preds
            perf[f"RidgeProbe|{rep}"] = reg_score(y, preds)
            print(f"  表示 {rep} 完成", flush=True)

        run.log_metric("performance", perf)

        # ---------- 表示多样性：A 与 C 的预测相关 ----------
        corr = pd.DataFrame(oof).corr()
        pairs = {"within_A": [], "within_C": [], "across_AC": []}
        for k1, k2 in itertools.combinations(oof, 2):
            r1, r2 = k1.split("|")[1], k2.split("|")[1]
            v = float(corr.loc[k1, k2])
            if r1 == "A" and r2 == "A":
                pairs["within_A"].append(v)
            elif r1 == "C" and r2 == "C":
                pairs["within_C"].append(v)
            elif {r1, r2} == {"A", "C"}:
                pairs["across_AC"].append(v)
        diversity = {k: float(np.mean(v)) for k, v in pairs.items() if v}
        run.log_metric("oof_diversity", diversity)

        # ---------- stacking 检验 ----------
        best_a = max((k for k in perf if k.endswith("|A")), key=lambda k: perf[k]["spearman"])
        best_c = max((k for k in perf if k.endswith("|C")), key=lambda k: perf[k]["spearman"])
        Z = np.column_stack([oof[best_a], oof[best_c]])
        stack = np.zeros(len(y))
        for tr, te in folds:
            m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Z[tr], y[tr])
            stack[te] = m.predict(Z[te])
        perf["Stacking(A+C)"] = reg_score(y, stack)
        run.log_metric("stacking", {
            "base": [best_a, best_c],
            "stacking_spearman": perf["Stacking(A+C)"]["spearman"],
            "best_single": max(perf[best_a]["spearman"], perf[best_c]["spearman"]),
            "gain": perf["Stacking(A+C)"]["spearman"] - max(perf[best_a]["spearman"],
                                                            perf[best_c]["spearman"]),
        })

        # ---------- 跨域：深度嵌入能否翻墙 ----------
        cross = None
        if args.cross:
            df2, reps2 = load_rep(args.cross)
            if "C" in reps2:
                y2 = df2["arousal"].to_numpy()
                cross = {}
                for rep in ("A", "C"):
                    X1 = SimpleImputer(strategy="median").fit_transform(
                        df[reps[rep]].to_numpy(dtype=np.float64))
                    X2 = SimpleImputer(strategy="median").fit_transform(
                        df2[reps2[rep]].to_numpy(dtype=np.float64))
                    m = build_regressors()["XGBoost"].fit(X1, y)
                    cross[rep] = reg_score(y2, m.predict(X2))
                run.log_metric("cross_domain", cross)

        # ---------- 打印 ----------
        P = pd.DataFrame(perf).T[["spearman", "r2", "mae"]].sort_values("spearman", ascending=False)
        print("\n" + "=" * 68)
        print(f"run_id: {run.run_id}   {args.dataset}  n={len(y)}")
        print(f"表示维度: " + "  ".join(f"{k}={len(v)}" for k, v in reps.items()))
        print("\n--- 各表示 × 各算法（Spearman ρ）---")
        print(P.round(4).to_string())

        print("\n--- 表示多样性（OOF 预测相关）---")
        for k, v in diversity.items():
            print(f"  {k:<12} {v:.3f}")

        st = run.meta["metrics"]["stacking"]
        print(f"\n--- Stacking(A+C) ---")
        print(f"  底座: {st['base']}")
        print(f"  最佳单模型 {st['best_single']:.4f} → stacking {st['stacking_spearman']:.4f}"
              f"  (Δ = {st['gain']:+.4f})")
        if st["gain"] > 0.01:
            print("  → 跨表示 stacking 有效：此前的「表示冗余」解释成立。")
        else:
            print("  → 即便手工特征 + 深度嵌入，stacking 仍无增益：")
            print("    「表示冗余」不是全部原因，限制更可能来自标签噪声。")

        if cross:
            print("\n--- 跨域迁移：深度嵌入能否翻墙 ---")
            for rep, s in cross.items():
                print(f"  Layer {rep}: ρ = {s['spearman']:+.3f}")


if __name__ == "__main__":
    main()
