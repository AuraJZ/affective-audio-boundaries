"""PLAN II-2：Rashomon set —— 近乎同样好的模型里，哪些特征始终重要？

    uv run python scripts/16_rashomon.py --dataset reg_emo_mix

## 为什么

「选一个最佳模型再解释它」有个隐患：往往存在**大量精度几乎相同**的模型，
而它们对「哪个特征重要」的看法可以差很多（Rashomon effect）。
只报最佳模型的解释，等于在多个同样合理的说法里任选了一个。

正确做法（Fisher/Rudin 的 model class reliance / variable importance clouds）：
枚举所有落在最优 ε 邻域内的模型，看**哪些特征在全部这些模型里都重要**。
这直接回答「规则是否稳健」，而非「哪个模型最准」。

## 实现

在超参空间随机采样 N 个模型（不同算法、深度、正则、特征子采样、随机种子），
用 GroupKFold 评估。保留 ρ ≥ ρ_max − ε 的那些，
对每个特征统计它在这些模型中进入 top-k 的比例。
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import N_SPLITS, SEED, feature_cols, reg_score

EPSILON = 0.02   # Rashomon 邻域：ρ 落后最优不超过此值
TOP_K = 15


def sample_model(rng, i: int):
    """从超参空间采一个模型。"""
    if i % 2 == 0:
        return XGBRegressor(
            n_estimators=int(rng.choice([200, 300, 400, 600])),
            max_depth=int(rng.choice([3, 4, 5, 6, 8])),
            learning_rate=float(rng.choice([0.03, 0.05, 0.08, 0.12])),
            subsample=float(rng.choice([0.6, 0.8, 1.0])),
            colsample_bytree=float(rng.choice([0.5, 0.7, 0.9])),
            reg_lambda=float(rng.choice([0.5, 1.0, 3.0])),
            random_state=int(rng.integers(0, 10_000)), n_jobs=-1)
    return RandomForestRegressor(
        n_estimators=int(rng.choice([200, 400, 600])),
        max_depth=int(rng.choice([6, 10, 16, 24])),
        max_features=float(rng.choice([0.2, 0.4, 0.6, 0.9])),
        min_samples_leaf=int(rng.choice([1, 2, 4, 8])),
        random_state=int(rng.integers(0, 10_000)), n_jobs=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="reg_emo_mix")
    ap.add_argument("--n-models", type=int, default=60)
    args = ap.parse_args()

    df = pd.read_parquet(REPO_ROOT / "data" / "features" / f"layer_a_{args.dataset}_ln.parquet")
    cols = feature_cols(df)
    X = SimpleImputer(strategy="median").fit_transform(df[cols].to_numpy(dtype=np.float64))
    y = df["arousal"].to_numpy()
    groups = df["group"].to_numpy()
    folds = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups))
    tr, te = folds[0]  # 单折用于置换重要性，节省成本

    with RunRecord("rashomon_set", seed=SEED,
                   params={"dataset": args.dataset, "n_models": args.n_models,
                           "epsilon": EPSILON, "top_k": TOP_K,
                           "importance": "permutation importance (单折)"}) as run:
        rng = np.random.default_rng(SEED)
        perf, imps = [], []
        for i in range(args.n_models):
            m = sample_model(rng, i)
            m.fit(X[tr], y[tr])
            rho = reg_score(y[te], m.predict(X[te]))["spearman"]
            perf.append(rho)
            pi = permutation_importance(m, X[te], y[te], n_repeats=3,
                                        random_state=SEED, n_jobs=-1)
            imps.append(pi.importances_mean)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{args.n_models} 模型  当前最优 ρ={max(perf):.4f}", flush=True)

        perf = np.array(perf)
        I = np.vstack(imps)
        keep = perf >= perf.max() - EPSILON
        run.log_metric("n_in_rashomon_set", int(keep.sum()))
        run.log_metric("rho_max", float(perf.max()))
        run.log_metric("rho_threshold", float(perf.max() - EPSILON))
        run.log_metric("rho_range_in_set", [float(perf[keep].min()), float(perf[keep].max())])

        # 每个特征进入 top-k 的比例（仅在 Rashomon 集合内统计）
        Ik = I[keep]
        in_top = np.zeros(len(cols))
        for row in Ik:
            in_top[np.argsort(row)[::-1][:TOP_K]] += 1
        stability = pd.Series(in_top / keep.sum(), index=cols).sort_values(ascending=False)

        run.log_metric("stability_top20", stability.head(20).round(3).to_dict())
        run.log_metric("n_always_important",
                       int((stability >= 0.99).sum()))
        stability.to_frame("top15_frequency").to_csv(
            run.artifact_path("rashomon_stability.csv"), encoding="utf-8")

        print("\n" + "=" * 66)
        print(f"run_id: {run.run_id}")
        print(f"采样 {args.n_models} 个模型，ρ ∈ [{perf.min():.3f}, {perf.max():.3f}]")
        print(f"Rashomon 集合（ρ ≥ {perf.max() - EPSILON:.3f}）: {keep.sum()} 个模型")
        print(f"\n--- 在全部近优模型中进入 top-{TOP_K} 的频率 ---")
        print(stability.head(18).round(3).to_string())
        print(f"\n频率 = 1.00 的特征数（**所有**近优模型都认为重要）: "
              f"{int((stability >= 0.99).sum())}")


if __name__ == "__main__":
    main()
