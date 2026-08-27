"""逐特征的 SHAP 重要性显著性检验（置换零分布 + BH-FDR）。

    uv run python scripts/07_feature_significance.py --dataset deam --n-perm 200

## 为什么需要这个脚本

`diag_permutation.py` 证明了「SHAP top-20 三模型交集」是废统计量：
标签完全打乱后，三模型仍平均"共识" 3.75 个特征（最多 7 个），
观测值 4 个的 p = 0.381。

原因：模型挑特征受**特征本身的性质**驱动（方差、取值多样性、易过拟合程度），
与有无信号无关。三个模型看同一张特征表，自然被同一批"好用"的特征吸引。
「取交集」只看排名、不看效应量，且没有任何零假设校准。

## 正确做法

把校准下沉到**每个特征**：

    对特征 j，问「它的 mean|SHAP| 是否显著高于它自己在随机标签下的分布？」

得到逐特征经验 p 值，再对 F 个特征做 Benjamini-Hochberg FDR 校正。

这之后再取跨模型交集才有意义 —— 因为每个分量都已经过校准。

## 成本权衡

零分布需要几百次重复，所以只用 TreeExplainer 能覆盖的模型（RF / XGBoost）。
SVM 的 KernelExplainer 太慢（每次约 30 s），200 次重复要 1.7 小时。
这是有意识的取舍，不是遗漏 —— 记录在 run record 里。
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer

from soundml.data import load_joined, source_paths
from soundml.modeling import SEED, build_models
from soundml.provenance import RunRecord

TREE_MODELS = ("RF", "XGBoost")
FDR_Q = 0.10


def shap_importance(algo: str, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """mean|SHAP| per feature。"""
    clf = build_models()[algo].named_steps["clf"]
    clf.fit(X, y)
    sv = np.asarray(shap.TreeExplainer(clf).shap_values(X))
    if sv.ndim == 3:
        sv = sv[..., 1]
    return np.abs(sv).mean(axis=0)


def benjamini_hochberg(p: np.ndarray, q: float) -> np.ndarray:
    """返回布尔掩码：在 FDR = q 下显著的项。"""
    n = len(p)
    order = np.argsort(p)
    thresholds = q * np.arange(1, n + 1) / n
    passed = p[order] <= thresholds
    mask = np.zeros(n, dtype=bool)
    if passed.any():
        cutoff = np.max(np.flatnonzero(passed))  # 最大的通过秩
        mask[order[: cutoff + 1]] = True
    return mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--rep", default="A+B", choices=["A", "B", "A+B"])
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()

    df, reps = load_joined(args.dataset)
    cols = reps[args.rep]
    X = SimpleImputer(strategy="median").fit_transform(df[cols].to_numpy(dtype=np.float64))
    y = df["label"].to_numpy()
    F = len(cols)

    with RunRecord(
        "feature_significance",
        seed=SEED,
        params={
            "dataset": args.dataset,
            "representation": args.rep,
            "n_features": F,
            "n_perm": args.n_perm,
            "models": list(TREE_MODELS),
            "fdr_q": FDR_Q,
            "importance": "mean|SHAP| (TreeExplainer)",
            "note": "仅用树模型；SVM KernelExplainer 在数百次重复下成本不可行，为有意取舍",
        },
    ) as run:
        run.add_input_snapshot("features", source_paths(args.dataset))

        observed = {m: shap_importance(m, X, y) for m in TREE_MODELS}
        null = {m: np.zeros((args.n_perm, F)) for m in TREE_MODELS}

        rng = np.random.default_rng(SEED)
        t0 = time.perf_counter()
        for i in range(args.n_perm):
            y_perm = rng.permutation(y)
            for m in TREE_MODELS:
                null[m][i] = shap_importance(m, X, y_perm)
            if (i + 1) % 20 == 0:
                rate = (i + 1) / (time.perf_counter() - t0)
                eta = (args.n_perm - i - 1) / rate / 60
                print(f"  {i + 1}/{args.n_perm}  ({rate:.2f} perm/s, 剩余 ~{eta:.0f} 分钟)", flush=True)

        results, sig_sets = {}, {}
        for m in TREE_MODELS:
            # 逐特征经验 p 值，+1 平滑 (Phipson & Smyth 2010)
            p = (np.sum(null[m] >= observed[m][None, :], axis=0) + 1) / (args.n_perm + 1)
            mask = benjamini_hochberg(p, FDR_Q)
            results[m] = pd.DataFrame({
                "feature": cols,
                "observed": observed[m],
                "null_mean": null[m].mean(axis=0),
                "null_p95": np.percentile(null[m], 95, axis=0),
                "p_value": p,
                "significant": mask,
            }).sort_values("p_value")
            sig_sets[m] = set(np.array(cols)[mask])
            run.log_metric(f"n_significant_{m}", int(mask.sum()))

        calibrated = sorted(set.intersection(*sig_sets.values()))
        run.log_metric("calibrated_consensus", calibrated)
        run.log_metric("n_calibrated_consensus", len(calibrated))
        run.log_metric("per_model_significant", {m: sorted(s) for m, s in sig_sets.items()})

        merged = results[TREE_MODELS[0]][["feature"]].copy()
        for m in TREE_MODELS:
            r = results[m].set_index("feature")
            merged[f"p_{m}"] = merged["feature"].map(r["p_value"])
            merged[f"sig_{m}"] = merged["feature"].map(r["significant"])
        merged["calibrated"] = merged["feature"].isin(calibrated)
        merged.sort_values([f"p_{m}" for m in TREE_MODELS]).to_csv(
            run.artifact_path("feature_significance.csv"), index=False, encoding="utf-8")

        print("\n" + "=" * 66)
        print(f"run_id: {run.run_id}")
        print(f"特征总数: {F}   置换次数: {args.n_perm}   FDR q = {FDR_Q}")
        for m in TREE_MODELS:
            print(f"  {m:<9} 显著特征 {len(sig_sets[m]):>3} 个")
        print(f"\n经校准的共识特征: {len(calibrated)} 个")
        for f in calibrated:
            ps = " / ".join(f"{results[m].set_index('feature').loc[f, 'p_value']:.4f}" for m in TREE_MODELS)
            print(f"  {f:<26} p = {ps}")
        if not calibrated:
            print("  （无）—— 没有任何特征在两个模型上都通过 FDR 校正。")
            print("  规则库不能出。需要更多数据、更强特征，或重新定义任务。")


if __name__ == "__main__":
    main()
