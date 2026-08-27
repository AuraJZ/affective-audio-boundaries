"""负对照：SHAP 跨模型共识数的置换检验。

    uv run python scripts/diag_permutation.py --features layer_a_deam_ln --n-perm 20

## 要回答的问题

我们定了「SHAP top-20 三模型交集 ≥ 6」作为规则可信度的门槛，
`20260726T203318Z` 那一轮刚好拿到 6 个，"卡线通过"。

但这个门槛是**悬空的**——从没验证过随机标签下会出现几个共识特征。
如果打乱标签后也常常出现 5–6 个，那我们那 6 个就是噪声产物，
基于它出规则库等于在噪声上盖楼。

## 做法

第 0 次重复 = 标签不打乱 → 观测值
第 1..N 次   = 标签随机打乱 → 零假设分布

三个模型、SHAP 参数、分箱方式全部保持一致（含降低成本后的参数），
否则观测值和零分布不可比。这就是为什么观测值也要在这里重算一遍，
而不是直接引用之前那一轮的 6。

同时报告 CV AUC 的零分布——它应当落在 0.5 附近，是"打乱确实生效"的 sanity check。
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

from soundml.data import load_joined, source_paths
from soundml.modeling import N_SPLITS, SEED, build_models
from soundml.provenance import RunRecord

TOP_K = 20
# 为了跑得动 N 次重复而调低的 SHAP 成本参数。
# 观测值也用这套参数重算，保证可比。
SVM_BACKGROUND = 15
SVM_EXPLAIN_N = 60
SVM_NSAMPLES = 100


def consensus_size(X: np.ndarray, y: np.ndarray, cols: list[str], rng) -> tuple[int, list[str]]:
    """给定标签，返回三模型 SHAP top-20 交集的大小与特征名。"""
    ranks: dict[str, pd.Series] = {}

    for algo in ("RF", "XGBoost"):
        clf = build_models()[algo].named_steps["clf"]
        clf.fit(X, y)
        sv = np.asarray(shap.TreeExplainer(clf).shap_values(X))
        if sv.ndim == 3:
            sv = sv[..., 1]
        ranks[algo] = pd.Series(np.abs(sv).mean(axis=0), index=cols).sort_values(ascending=False)

    svm = build_models()["SVM"].fit(X, y)
    sub = rng.choice(len(X), size=min(SVM_EXPLAIN_N, len(X)), replace=False)
    sv_svm = shap.KernelExplainer(
        lambda a: svm.predict_proba(a)[:, 1], shap.kmeans(X, SVM_BACKGROUND)
    ).shap_values(X[sub], nsamples=SVM_NSAMPLES, silent=True)
    ranks["SVM"] = pd.Series(np.abs(np.asarray(sv_svm)).mean(axis=0),
                             index=cols).sort_values(ascending=False)

    tops = [set(r.head(TOP_K).index) for r in ranks.values()]
    inter = sorted(set.intersection(*tops))
    return len(inter), inter


def cv_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    preds = np.zeros(len(y))
    for tr, te in cv.split(X, y, groups):
        m = build_models()["RF"]
        m.fit(X[tr], y[tr])
        preds[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, preds))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="如 deam")
    ap.add_argument("--rep", default="A+B", choices=["A", "B", "A+B"])
    ap.add_argument("--n-perm", type=int, default=20)
    args = ap.parse_args()

    df, reps = load_joined(args.dataset)
    cols = reps[args.rep]
    X = SimpleImputer(strategy="median").fit_transform(df[cols].to_numpy(dtype=np.float64))
    y = df["label"].to_numpy()
    groups = df["group"].to_numpy()

    with RunRecord(
        "diag_permutation",
        seed=SEED,
        params={
            "dataset": args.dataset,
            "representation": args.rep,
            "n_perm": args.n_perm,
            "top_k": TOP_K,
            "n_features": len(cols),
            "svm_shap": {"background": SVM_BACKGROUND, "explain_n": SVM_EXPLAIN_N,
                         "nsamples": SVM_NSAMPLES},
            "note": "观测值用与零分布完全相同的参数重算，保证可比",
        },
    ) as run:
        run.add_input_snapshot("features", source_paths(args.dataset))

        rng = np.random.default_rng(SEED)
        obs_size, obs_feats = consensus_size(X, y, cols, rng)
        obs_auc = cv_auc(X, y, groups)
        print(f"观测值：共识 {obs_size} 个，CV AUC {obs_auc:.4f}", flush=True)
        print(f"        {obs_feats}", flush=True)

        null_sizes, null_aucs = [], []
        for i in range(1, args.n_perm + 1):
            y_perm = rng.permutation(y)
            size, _ = consensus_size(X, y_perm, cols, rng)
            auc = cv_auc(X, y_perm, groups)
            null_sizes.append(size)
            null_aucs.append(auc)
            print(f"  perm {i:>2}/{args.n_perm}: 共识 {size}, AUC {auc:.3f}", flush=True)

        null = np.array(null_sizes)
        # 单尾 p 值，含 +1 平滑（Phipson & Smyth 2010）
        p = float((np.sum(null >= obs_size) + 1) / (len(null) + 1))

        run.log_metric("observed_consensus", obs_size)
        run.log_metric("observed_features", obs_feats)
        run.log_metric("observed_cv_auc", obs_auc)
        run.log_metric("null_consensus", null_sizes)
        run.log_metric("null_cv_auc", [round(a, 4) for a in null_aucs])
        run.log_metric("null_consensus_mean", float(null.mean()))
        run.log_metric("null_consensus_p95", float(np.percentile(null, 95)))
        run.log_metric("null_consensus_max", int(null.max()))
        run.log_metric("null_auc_mean", float(np.mean(null_aucs)))
        run.log_metric("p_value", p)
        run.log_metric("significant_at_0.05", p < 0.05)

        print("\n" + "=" * 60)
        print(f"观测共识数        : {obs_size}")
        print(f"零分布 均值/p95/最大 : {null.mean():.2f} / {np.percentile(null, 95):.1f} / {null.max()}")
        print(f"零分布 AUC 均值   : {np.mean(null_aucs):.4f}  (应接近 0.5)")
        print(f"p 值              : {p:.4f}")
        print("=" * 60)
        if p < 0.05:
            print("→ 共识特征显著多于随机，「≥6」这条门槛站得住。")
        else:
            print("→ 共识特征与随机不可区分。当前规则库建立在噪声上，门槛必须重定。")


if __name__ == "__main__":
    main()
