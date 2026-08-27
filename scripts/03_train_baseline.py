"""Step 1c / 2c: 模型矩阵 + stacking + SHAP + 适用域。

    uv run python scripts/03_train_baseline.py --train deam --test pmemo

对应笔记 §3 / §4。当前只跑 Layer A × 5 个经典算法（矩阵的第一行），
Layer B/C 在 Step 3 补齐后再撑满 §3.2 的表。

四处与笔记不同的地方，都是防止指标虚高：
  1. StratifiedGroupKFold 按来源分组 —— 随机切会把同一段源录音的邻近切片
     分到训练和测试两边。
  2. stacking 的 meta-learner 用外层折的 OOF 预测训练，不用 sklearn 内置 cv，
     避免分组信息丢失。
  3. **跨源 AUC 才是成绩单**（--test 指定另一个数据集）。组内 CV 只是 sanity check：
     同一批标注、同一批录音条件下的高 AUC 说明不了泛化。
  4. 规则可信度看 SHAP top-20 的跨模型重叠数（门槛 ≥6），不是 AUC。
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from soundml.modeling import (
    AD_K, META_COLS, N_SPLITS, SEED, SHAP_OVERLAP_THRESHOLD,
    build_models, fit_ad, mean_std, score,
)
from soundml.provenance import REPO_ROOT, RunRecord

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="训练集名，如 deam")
    ap.add_argument("--test", default=None, help="独立测试集名，如 pmemo")
    args = ap.parse_args()

    feat_path = lambda d: REPO_ROOT / "data" / "features" / f"layer_a_{d}.parquet"
    tr_df = pd.read_parquet(feat_path(args.train))
    te_df = pd.read_parquet(feat_path(args.test)) if args.test else None

    feat_cols = [c for c in tr_df.columns if c not in META_COLS]
    X = tr_df[feat_cols].to_numpy(dtype=np.float64)
    y = tr_df["label"].to_numpy()
    groups = tr_df["group"].to_numpy()

    models = build_models()
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(cv.split(X, y, groups))

    with RunRecord(
        f"train_{args.train}" + (f"_test_{args.test}" if args.test else ""),
        seed=SEED,
        params={
            "train_set": args.train,
            "test_set": args.test,
            "cv": f"StratifiedGroupKFold(n_splits={N_SPLITS})",
            "n_features": len(feat_cols),
            "models": {k: str(v.named_steps["clf"]) for k, v in models.items()},
            "ad": {"Z": AD_Z, "k": AD_K},
        },
    ) as run:
        run.add_input_snapshot("train_features", [feat_path(args.train)])
        if args.test:
            run.add_input_snapshot("test_features", [feat_path(args.test)])
        run.log_metric("n_train", {"total": len(y), "pos": int(y.sum()), "neg": int((1 - y).sum())})

        # ---------- 组内 CV（sanity check）----------
        oof = {name: np.zeros(len(y)) for name in models}
        per_fold: dict[str, list[dict[str, float]]] = {name: [] for name in models}
        for fi, (tr, te) in enumerate(folds, start=1):
            for name in models:
                m = build_models()[name]
                m.fit(X[tr], y[tr])
                p = m.predict_proba(X[te])[:, 1]
                oof[name][te] = p
                per_fold[name].append(score(y[te], p))
            print(f"  fold {fi}/{N_SPLITS} done", flush=True)

        cv_summary = {name: mean_std(per_fold[name]) for name in models}

        # ---------- Stacking（笔记 §3.4）----------
        top4 = sorted(models, key=lambda n: cv_summary[n]["auc_mean"], reverse=True)[:4]
        Z_meta = np.column_stack([oof[n] for n in top4])
        stack_folds = []
        for tr, te in folds:
            meta = LogisticRegression(max_iter=1000, random_state=SEED)
            meta.fit(Z_meta[tr], y[tr])
            stack_folds.append(score(y[te], meta.predict_proba(Z_meta[te])[:, 1]))
        cv_summary["Stacking"] = mean_std(stack_folds)

        run.log_metric("cv_within_source", cv_summary)
        run.log_metric("stacking_base_models", top4)

        # ---------- 全量拟合 + 跨源测试（真正的成绩单）----------
        full = {name: build_models()[name].fit(X, y) for name in models}
        meta_full = LogisticRegression(max_iter=1000, random_state=SEED).fit(Z_meta, y)

        cross = None
        if te_df is not None:
            missing = set(feat_cols) - set(te_df.columns)
            if missing:
                raise ValueError(f"测试集缺少特征列: {sorted(missing)[:5]}")
            Xt = te_df[feat_cols].to_numpy(dtype=np.float64)
            yt = te_df["label"].to_numpy()
            base_t = {n: full[n].predict_proba(Xt)[:, 1] for n in models}
            cross = {n: score(yt, p) for n, p in base_t.items()}
            cross["Stacking"] = score(
                yt, meta_full.predict_proba(np.column_stack([base_t[n] for n in top4]))[:, 1]
            )
            run.log_metric("n_test", {"total": len(yt), "pos": int(yt.sum()), "neg": int((1 - yt).sum())})
            run.log_metric("cross_source", cross)

        # ---------- 适用域 ----------
        scaler = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]).fit(X)
        nn, ad = fit_ad(scaler.transform(X))
        if te_df is not None:
            d_test, _ = nn.kneighbors(scaler.transform(Xt), n_neighbors=AD_K)
            inside = float((d_test.mean(axis=1) <= ad["threshold"]).mean())
            ad["test_fraction_inside"] = inside
        run.log_metric("applicability_domain", ad)

        # ---------- SHAP（笔记 §4）----------
        imputer = SimpleImputer(strategy="median").fit(X)
        Xi = imputer.transform(X)
        shap_rank: dict[str, pd.Series] = {}

        for name in ("RF", "XGBoost"):
            clf = build_models()[name].named_steps["clf"]
            clf.fit(Xi, y)
            sv = np.asarray(shap.TreeExplainer(clf).shap_values(Xi))
            if sv.ndim == 3:
                sv = sv[..., 1]
            shap_rank[name] = pd.Series(np.abs(sv).mean(axis=0), index=feat_cols).sort_values(ascending=False)
            plt.figure()
            shap.summary_plot(sv, pd.DataFrame(Xi, columns=feat_cols), show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(run.artifact_path(f"shap_summary_{name}.png"), dpi=150)
            plt.close()

        rng = np.random.default_rng(SEED)
        bg = shap.kmeans(Xi, 25)
        sub = rng.choice(len(Xi), size=min(100, len(Xi)), replace=False)
        sv_svm = shap.KernelExplainer(lambda a: full["SVM"].predict_proba(a)[:, 1], bg).shap_values(
            Xi[sub], nsamples=200, silent=True
        )
        shap_rank["SVM"] = pd.Series(np.abs(np.asarray(sv_svm)).mean(axis=0),
                                     index=feat_cols).sort_values(ascending=False)

        # ---------- 跨模型稳定性 ----------
        tops = {k: set(v.head(20).index) for k, v in shap_rank.items()}
        consensus = sorted(set.intersection(*tops.values()))
        run.log_metric("shap_top20_overlap_3models", len(consensus))
        run.log_metric("shap_overlap_threshold", SHAP_OVERLAP_THRESHOLD)
        run.log_metric("shap_overlap_passed", len(consensus) >= SHAP_OVERLAP_THRESHOLD)
        run.log_metric("shap_consensus_features", consensus)
        run.log_metric("shap_top10_per_model", {k: list(v.head(10).index) for k, v in shap_rank.items()})

        pd.DataFrame({k: v.rank(ascending=False) for k, v in shap_rank.items()}).assign(
            mean_rank=lambda d: d.mean(axis=1)
        ).sort_values("mean_rank").to_csv(run.artifact_path("shap_ranks.csv"), encoding="utf-8")

        # ---------- 打印 ----------
        cols = ["auc_mean", "auc_std", "accuracy_mean", "recall_mean"]
        print("\n" + "=" * 66)
        print(f"run_id: {run.run_id}\n")
        print(f"--- 组内 CV（{args.train}，sanity check）---")
        print(pd.DataFrame(cv_summary).T[cols].round(4))
        if cross:
            print(f"\n--- 跨源测试（{args.train} → {args.test}，成绩单）---")
            print(pd.DataFrame(cross).T[["auc", "accuracy", "precision", "recall"]].round(4))
            print(f"\n落在适用域内的测试样本: {ad['test_fraction_inside']:.1%}")
        print(f"\nstacking 底座: {top4}")
        mark = "✓" if len(consensus) >= SHAP_OVERLAP_THRESHOLD else "✗"
        print(f"SHAP top-20 三模型交集: {len(consensus)} 个 (门槛 {SHAP_OVERLAP_THRESHOLD}) {mark}")
        print(json.dumps(consensus, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
