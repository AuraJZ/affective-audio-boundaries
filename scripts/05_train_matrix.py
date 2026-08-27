"""Step 3: 多表示模型矩阵 —— 检验 stacking 假设。

    uv run python scripts/05_train_matrix.py --train deam --test pmemo

## 这个脚本要回答的问题

`reports/step2_real_labels.md` §2 的结论是：stacking 三次都没赢，
因为 5 个模型全吃**同一份 Layer A 特征**，OOF 预测相关系数均值 0.912，
没有互补信息可融合。论文的基模型吃的是不同的**分子表示**。

所以：补上 Layer B（心理声学）之后，把矩阵扩成

    5 算法 × 3 表示（A / B / A+B）= 15 个模型

若假设成立，应观察到：
  1. **跨表示**的 OOF 相关显著低于**同表示内**的相关
  2. 由跨表示模型组成的 stacking 首次超过最佳单模型

两条都验证不了，就说明论文的集成优势在声音模态上确实不成立——
那是个值得写进报告的阴性结论，不是失败。
"""

from __future__ import annotations

import argparse
import itertools
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

from soundml.data import load_joined, source_paths
from soundml.modeling import (
    AD_K, N_SPLITS, SEED, SHAP_OVERLAP_THRESHOLD,
    build_models, fit_ad, mean_std, score,
)
from soundml.provenance import RunRecord


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", default=None)
    args = ap.parse_args()

    tr, reps = load_joined(args.train)
    te, _ = load_joined(args.test) if args.test else (None, None)

    y = tr["label"].to_numpy()
    groups = tr["group"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(cv.split(np.zeros(len(y)), y, groups))

    algos = list(build_models())
    cells = [(rep, algo) for rep in reps for algo in algos]  # 15 个模型

    with RunRecord(
        f"matrix_{args.train}" + (f"_test_{args.test}" if args.test else ""),
        seed=SEED,
        params={
            "train_set": args.train,
            "test_set": args.test,
            "representations": {k: len(v) for k, v in reps.items()},
            "algorithms": algos,
            "n_models": len(cells),
            "cv": f"StratifiedGroupKFold(n_splits={N_SPLITS})",
        },
    ) as run:
        run.add_input_snapshot("train_features", source_paths(args.train))
        if args.test:
            run.add_input_snapshot("test_features", source_paths(args.test))
        run.log_metric("n_train", {"total": len(y), "pos": int(y.sum()), "neg": int((1 - y).sum())})

        Xs = {rep: tr[cols].to_numpy(dtype=np.float64) for rep, cols in reps.items()}

        # ---------- 15 模型的组内 CV + OOF ----------
        oof: dict[str, np.ndarray] = {}
        cv_summary: dict[str, dict[str, float]] = {}
        for rep, algo in cells:
            key = f"{algo}|{rep}"
            preds, rows = np.zeros(len(y)), []
            for trn, tst in folds:
                m = build_models()[algo]
                m.fit(Xs[rep][trn], y[trn])
                p = m.predict_proba(Xs[rep][tst])[:, 1]
                preds[tst] = p
                rows.append(score(y[tst], p))
            oof[key] = preds
            cv_summary[key] = mean_std(rows)
            print(f"  {key:<16} AUC {cv_summary[key]['auc_mean']:.4f}", flush=True)

        run.log_metric("cv_matrix", cv_summary)

        # ---------- 关键诊断：跨表示 vs 同表示的 OOF 相关 ----------
        corr = pd.DataFrame(oof).corr()
        within, across = [], []
        for k1, k2 in itertools.combinations(oof, 2):
            r = float(corr.loc[k1, k2])
            (within if k1.split("|")[1] == k2.split("|")[1] else across).append(r)
        diversity = {
            "within_representation_mean": float(np.mean(within)),
            "across_representation_mean": float(np.mean(across)),
            "gap": float(np.mean(within) - np.mean(across)),
        }
        run.log_metric("oof_diversity", diversity)

        # ---------- Stacking 两种底座 ----------
        ranked = sorted(oof, key=lambda k: cv_summary[k]["auc_mean"], reverse=True)
        bases = {"top4": ranked[:4]}
        # 每个表示各取其最佳 —— 最大化表示多样性
        bases["best_per_rep"] = [
            max((k for k in oof if k.endswith(f"|{rep}")), key=lambda k: cv_summary[k]["auc_mean"])
            for rep in reps
        ]

        stack_meta: dict[str, LogisticRegression] = {}
        for name, base in bases.items():
            Z = np.column_stack([oof[k] for k in base])
            rows = []
            for trn, tst in folds:
                meta = LogisticRegression(max_iter=1000, random_state=SEED).fit(Z[trn], y[trn])
                rows.append(score(y[tst], meta.predict_proba(Z[tst])[:, 1]))
            cv_summary[f"Stacking({name})"] = mean_std(rows)
            stack_meta[name] = LogisticRegression(max_iter=1000, random_state=SEED).fit(Z, y)
        run.log_metric("stacking_bases", bases)
        run.log_metric("cv_matrix_with_stacking", cv_summary)

        # ---------- 全量拟合 + 跨源测试 ----------
        cross, ad = None, None
        if te is not None:
            Xt = {rep: te[cols].to_numpy(dtype=np.float64) for rep, cols in reps.items()}
            yt = te["label"].to_numpy()
            full = {f"{a}|{r}": build_models()[a].fit(Xs[r], y) for r, a in cells}
            base_t = {k: m.predict_proba(Xt[k.split("|")[1]])[:, 1] for k, m in full.items()}
            cross = {k: score(yt, p) for k, p in base_t.items()}
            for name, base in bases.items():
                Zt = np.column_stack([base_t[k] for k in base])
                cross[f"Stacking({name})"] = score(yt, stack_meta[name].predict_proba(Zt)[:, 1])
            run.log_metric("n_test", {"total": len(yt), "pos": int(yt.sum()), "neg": int((1 - yt).sum())})
            run.log_metric("cross_source", cross)

            sc = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]).fit(Xs["A+B"])
            nn, ad = fit_ad(sc.transform(Xs["A+B"]))
            d, _ = nn.kneighbors(sc.transform(Xt["A+B"]), n_neighbors=AD_K)
            ad["test_fraction_inside"] = float((d.mean(axis=1) <= ad["threshold"]).mean())
            run.log_metric("applicability_domain", ad)

        # ---------- SHAP（在 A+B 上）----------
        cols = reps["A+B"]
        Xi = SimpleImputer(strategy="median").fit_transform(Xs["A+B"])
        shap_rank: dict[str, pd.Series] = {}
        for algo in ("RF", "XGBoost"):
            clf = build_models()[algo].named_steps["clf"]
            clf.fit(Xi, y)
            sv = np.asarray(shap.TreeExplainer(clf).shap_values(Xi))
            if sv.ndim == 3:
                sv = sv[..., 1]
            shap_rank[algo] = pd.Series(np.abs(sv).mean(axis=0), index=cols).sort_values(ascending=False)
            plt.figure()
            shap.summary_plot(sv, pd.DataFrame(Xi, columns=cols), show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(run.artifact_path(f"shap_summary_{algo}.png"), dpi=150)
            plt.close()

        svm_full = build_models()["SVM"].fit(Xs["A+B"], y)
        rng = np.random.default_rng(SEED)
        sub = rng.choice(len(Xi), size=min(100, len(Xi)), replace=False)
        sv_svm = shap.KernelExplainer(
            lambda a: svm_full.predict_proba(a)[:, 1], shap.kmeans(Xi, 25)
        ).shap_values(Xi[sub], nsamples=200, silent=True)
        shap_rank["SVM"] = pd.Series(np.abs(np.asarray(sv_svm)).mean(axis=0),
                                     index=cols).sort_values(ascending=False)

        tops = {k: set(v.head(20).index) for k, v in shap_rank.items()}
        consensus = sorted(set.intersection(*tops.values()))
        layer_b_cols = set(reps["B"])
        run.log_metric("shap_top20_overlap_3models", len(consensus))
        run.log_metric("shap_overlap_passed", len(consensus) >= SHAP_OVERLAP_THRESHOLD)
        run.log_metric("shap_consensus_features", consensus)
        run.log_metric("shap_consensus_from_layer_b", sorted(set(consensus) & layer_b_cols))
        run.log_metric("shap_top10_per_model", {k: list(v.head(10).index) for k, v in shap_rank.items()})

        ranks_df = pd.DataFrame({k: v.rank(ascending=False) for k, v in shap_rank.items()})
        ranks_df["mean_rank"] = ranks_df.mean(axis=1)
        ranks_df["layer"] = ["B" if c in layer_b_cols else "A" for c in ranks_df.index]
        ranks_df.sort_values("mean_rank").to_csv(run.artifact_path("shap_ranks.csv"), encoding="utf-8")

        # ---------- 打印 ----------
        print("\n" + "=" * 72)
        print(f"run_id: {run.run_id}\n")
        cv_tbl = pd.DataFrame(cv_summary).T[["auc_mean", "auc_std"]].sort_values("auc_mean", ascending=False)
        print("--- 组内 CV（前 10）---")
        print(cv_tbl.head(10).round(4))
        if cross:
            print(f"\n--- 跨源测试 {args.train} → {args.test}（成绩单，前 10）---")
            print(pd.DataFrame(cross).T.sort_values("auc", ascending=False).head(10)[
                ["auc", "accuracy", "recall"]].round(4))
            print(f"\n适用域内测试样本: {ad['test_fraction_inside']:.1%}")

        print("\n--- OOF 相关：表示多样性诊断 ---")
        print(f"同表示内平均相关 : {diversity['within_representation_mean']:.3f}")
        print(f"跨表示平均相关   : {diversity['across_representation_mean']:.3f}")
        print(f"差值             : {diversity['gap']:+.3f}")

        print(f"\n⚠️ 以下为**已作废**的统计量，仅作参考，不得用于可信度判断：")
        print(f"   （置换检验 p = 0.381，随机标签下平均也有 3.75 个共识）")
        print(f"   正确的显著性判断见 scripts/07_feature_significance.py")
        print(f"SHAP top-20 三模型交集: {len(consensus)} 个")
        print(json.dumps(consensus, ensure_ascii=False, indent=2))
        print(f"其中来自 Layer B: {sorted(set(consensus) & layer_b_cols)}")


if __name__ == "__main__":
    main()
