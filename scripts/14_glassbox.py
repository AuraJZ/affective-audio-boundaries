"""PLAN II-1：原生可解释模型（EBM + 单调约束 GBDT）替代「黑盒 + 事后 SHAP」。

    uv run python scripts/14_glassbox.py --dataset reg_deam

## 为什么换

SHAP 是**事后归因** —— 先训黑盒，再猜它在看什么。
而规则库需要的是能**直接读出数值区间**的形状函数。

- **EBM**（Explainable Boosting Machine）：广义可加模型，
  直接输出每个特征的 `特征值 → 贡献` 曲线。规则可从曲线直接读阈值，
  不需要我们此前那套分箱近似。

- **单调约束 XGBoost**：强制「该特征越高（低）→ 预测单调变化」。
  **若加约束后精度不掉，说明单调假设被数据支持** ——
  这本身就是一条证据，比 SHAP 的相关性论证硬。

两者都与 RF/XGBoost 基线对比：可解释性的代价是多少精度？
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import N_SPLITS, SEED, build_regressors, feature_cols, reg_score

# 域无关核心特征及其**期望方向**（负 = 值越低越助眠 = 值越高 arousal 越高）
# 来源 reports/step6_domain_invariant_rules.md §1
EXPECTED_SIGN = {
    "onset_env_dmean": +1, "spec_contrast3_dmean": +1, "onset_rate": +1,
    "spec_flatness_std": +1, "onset_env_mean": +1, "spec_flatness_p50": +1,
    "mfcc1_std": +1,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="reg_deam")
    ap.add_argument("--top-k", type=int, default=25,
                    help="EBM 只在最重要的 K 个特征上跑，避免 122 维过拟合与耗时")
    args = ap.parse_args()

    path = REPO_ROOT / "data" / "features" / f"layer_a_{args.dataset}_ln.parquet"
    df = pd.read_parquet(path)
    cols = feature_cols(df)
    Xall = SimpleImputer(strategy="median").fit_transform(df[cols].to_numpy(dtype=np.float64))
    y = df["arousal"].to_numpy()
    groups = df["group"].to_numpy()
    folds = list(GroupKFold(n_splits=N_SPLITS).split(Xall, y, groups))

    with RunRecord(
        "glassbox_models", seed=SEED,
        params={"dataset": args.dataset, "n_features": len(cols), "top_k": args.top_k,
                "models": ["RF(baseline)", "XGBoost(baseline)", "XGBoost+monotone", "EBM"],
                "expected_signs": EXPECTED_SIGN},
    ) as run:
        results = {}

        # ---------- 基线 ----------
        for algo in ("RF", "XGBoost"):
            preds = np.zeros(len(y))
            for tr, te in folds:
                m = build_regressors()[algo]
                m.fit(Xall[tr], y[tr])
                preds[te] = m.predict(Xall[te])
            results[algo] = reg_score(y, preds)
            print(f"  {algo} baseline 完成", flush=True)

        # ---------- 特征重要性排序（用于选 top-k） ----------
        base = build_regressors()["XGBoost"].fit(Xall, y)
        imp = pd.Series(base.named_steps["m"].feature_importances_, index=cols)
        topk = list(imp.nlargest(args.top_k).index)
        # 确保域无关特征在内
        for f in EXPECTED_SIGN:
            if f in cols and f not in topk:
                topk.append(f)
        idx = [cols.index(c) for c in topk]
        Xk = Xall[:, idx]
        run.log_metric("selected_features", topk)

        # ---------- 单调约束 XGBoost ----------
        constraints = tuple(EXPECTED_SIGN.get(c, 0) for c in topk)
        preds = np.zeros(len(y))
        for tr, te in folds:
            m = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                             n_jobs=-1, monotone_constraints=constraints)
            m.fit(Xk[tr], y[tr])
            preds[te] = m.predict(Xk[te])
        results["XGBoost+monotone"] = reg_score(y, preds)
        run.log_metric("n_constrained", int(sum(1 for c in constraints if c != 0)))
        print("  单调约束 XGBoost 完成", flush=True)

        # 同样特征子集但**无**约束，作为公平对照
        preds = np.zeros(len(y))
        for tr, te in folds:
            m = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1)
            m.fit(Xk[tr], y[tr])
            preds[te] = m.predict(Xk[te])
        results["XGBoost(topk,no constraint)"] = reg_score(y, preds)

        # ---------- EBM ----------
        from interpret.glassbox import ExplainableBoostingRegressor

        preds = np.zeros(len(y))
        for tr, te in folds:
            m = ExplainableBoostingRegressor(feature_names=topk, random_state=SEED,
                                             interactions=0, max_bins=64)
            m.fit(Xk[tr], y[tr])
            preds[te] = m.predict(Xk[te])
        results["EBM"] = reg_score(y, preds)
        print("  EBM 完成", flush=True)

        # 全量 EBM 取形状函数
        ebm = ExplainableBoostingRegressor(feature_names=topk, random_state=SEED,
                                           interactions=0, max_bins=64).fit(Xk, y)
        expl = ebm.explain_global()
        shapes = {}
        for i, name in enumerate(topk):
            d = expl.data(i)
            if d and d.get("type") == "univariate":
                shapes[name] = {"bins": [float(v) for v in d["names"]],
                                "scores": [float(v) for v in d["scores"]]}
        run.log_metric("ebm_shape_functions", shapes)
        (run.dir / "ebm_shapes.json").write_text(
            json.dumps(shapes, ensure_ascii=False), encoding="utf-8")

        ebm_imp = pd.Series(ebm.term_importances(), index=ebm.term_names_).sort_values(ascending=False)
        run.log_metric("ebm_importance_top15", ebm_imp.head(15).round(5).to_dict())
        run.log_metric("performance", results)

        # ---------- 打印 ----------
        R = pd.DataFrame(results).T[["spearman", "pearson", "r2", "mae"]]
        print("\n" + "=" * 68)
        print(f"run_id: {run.run_id}   数据集 {args.dataset}  n={len(y)}")
        print(f"EBM/约束模型使用 {len(topk)} 个特征，其中 "
              f"{sum(1 for c in constraints if c)} 个加了单调约束\n")
        print(R.round(4).to_string())

        base_rho = results["XGBoost(topk,no constraint)"]["spearman"]
        mono_rho = results["XGBoost+monotone"]["spearman"]
        print(f"\n单调约束的代价: Δρ = {mono_rho - base_rho:+.4f}")
        if mono_rho >= base_rho - 0.02:
            print("→ 加约束后精度基本不掉：单调假设被数据支持。")
        else:
            print("→ 加约束后精度明显下降：单调假设不成立，规则不应写成简单阈值。")

        print("\n--- EBM 特征重要性 top 10 ---")
        print(ebm_imp.head(10).round(4).to_string())


if __name__ == "__main__":
    main()
