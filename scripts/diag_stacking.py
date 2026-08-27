"""诊断：stacking 为什么没有超过最佳单模型？

假设：论文的基模型用的是**不同的分子表示**（MACCS / RDKit / ECFP4 / 图 / 预训练），
彼此互补性强，所以 stacking 能把 AUC 从 0.957 拉到 0.994。
我们目前 5 个模型全部吃**同一份 Layer A 特征**，只是算法不同，
预测可能高度相关 —— 那 stacking 就没有互补信息可融合。

检验方式：算 OOF 预测之间的相关矩阵。若普遍 > 0.9，假设成立，
说明必须等 Layer B / Layer C 补进来（表示层多样化）stacking 才可能有意义。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from soundml.modeling import META_COLS, SEED, build_models
from soundml.provenance import REPO_ROOT, RunRecord


def main() -> None:
    df = pd.read_parquet(REPO_ROOT / "data" / "features" / "layer_a_deam.parquet")
    feat_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feat_cols].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy()
    groups = df["group"].to_numpy()

    models = build_models()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = {n: np.zeros(len(y)) for n in models}

    with RunRecord("diag_stacking_correlation", seed=SEED,
                   params={"dataset": "deam", "question": "base model prediction correlation"}) as run:
        for tr, te in cv.split(X, y, groups):
            for n in models:
                m = build_models()[n]
                m.fit(X[tr], y[tr])
                oof[n][te] = m.predict_proba(X[te])[:, 1]

        corr = pd.DataFrame(oof).corr(method="pearson")
        off = corr.to_numpy()[np.triu_indices(len(models), k=1)]

        run.log_metric("oof_correlation", corr.round(4).to_dict())
        run.log_metric("mean_offdiag_corr", float(off.mean()))
        run.log_metric("min_offdiag_corr", float(off.min()))
        run.log_metric("max_offdiag_corr", float(off.max()))
        run.log_metric("hypothesis_supported", bool(off.mean() > 0.9))

        print("--- OOF 预测相关矩阵 (Pearson) ---")
        print(corr.round(3))
        print(f"\n非对角均值: {off.mean():.3f}   最小: {off.min():.3f}   最大: {off.max():.3f}")
        print("→ 假设成立（基模型高度冗余）" if off.mean() > 0.9 else "→ 假设不成立，需另找原因")


if __name__ == "__main__":
    main()
