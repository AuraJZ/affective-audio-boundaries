"""回归版模型矩阵 + 跨源 + 跨域。

    uv run python scripts/12_regression_matrix.py

对三个语料分别做组内 CV，并做全部有意义的跨语料迁移：

    音乐 ↔ 音乐  (DEAM ↔ PMEmo)      —— 同域跨源
    音乐 → 环境声 (DEAM → Emo)        —— 跨域
    环境声 → 音乐 (Emo → DEAM)        —— 跨域（反向）

跨域反向此前从未做过：之前只测了「音乐规则用在环境声上」，
没测「环境声规则用在音乐上」。若两个方向都崩，说明是真正的域分离；
若只有一个方向崩，说明其中一域的规则更普适。
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import (
    N_SPLITS, SEED, build_regressors, feature_cols, reg_score,
)

FEAT = REPO_ROOT / "data" / "features"
SETS = {"deam": "音乐/DEAM", "pmemo": "音乐/PMEmo", "emo_mix": "环境声/Emo-Soundscapes"}
DOMAIN = {"deam": "music", "pmemo": "music", "emo_mix": "ambient"}


def load(tag: str) -> pd.DataFrame:
    return pd.read_parquet(FEAT / f"layer_a_reg_{tag}_ln.parquet")


def main() -> None:
    data = {k: load(k) for k in SETS}
    cols = feature_cols(data["deam"])
    for k, d in data.items():
        assert feature_cols(d) == cols, f"{k} 特征列不一致"

    X = {k: d[cols].to_numpy(dtype=np.float64) for k, d in data.items()}
    y = {k: d["arousal"].to_numpy() for k, d in data.items()}
    g = {k: d["group"].to_numpy() for k, d in data.items()}

    algos = list(build_regressors())

    with RunRecord(
        "regression_matrix", seed=SEED,
        params={"task": "regression", "target": "arousal",
                "datasets": {k: len(v) for k, v in y.items()},
                "algorithms": algos, "n_features": len(cols),
                "cv": f"GroupKFold({N_SPLITS})"},
    ) as run:
        run.log_metric("n_samples", {k: int(len(v)) for k, v in y.items()})

        # ---------- 组内 CV ----------
        within, oof = {}, {}
        for ds in SETS:
            cv = GroupKFold(n_splits=N_SPLITS)
            folds = list(cv.split(X[ds], y[ds], g[ds]))
            for algo in algos:
                preds = np.zeros(len(y[ds]))
                for tr, te in folds:
                    m = build_regressors()[algo]
                    m.fit(X[ds][tr], y[ds][tr])
                    preds[te] = m.predict(X[ds][te])
                within[f"{algo}|{ds}"] = reg_score(y[ds], preds)
                oof[f"{algo}|{ds}"] = preds
            print(f"  组内 CV 完成: {ds}", flush=True)
        run.log_metric("within_source", within)

        # ---------- 跨语料迁移（全部有序对）----------
        cross = {}
        for src, dst in itertools.permutations(SETS, 2):
            for algo in algos:
                m = build_regressors()[algo].fit(X[src], y[src])
                s = reg_score(y[dst], m.predict(X[dst]))
                s["same_domain"] = DOMAIN[src] == DOMAIN[dst]
                cross[f"{algo}|{src}->{dst}"] = s
            print(f"  跨语料完成: {src} -> {dst}", flush=True)
        run.log_metric("cross_corpus", cross)

        # ---------- 汇总：同域 vs 跨域 ----------
        rows = []
        for key, s in cross.items():
            algo, pair = key.split("|")
            src, dst = pair.split("->")
            rows.append({"algo": algo, "src": src, "dst": dst,
                         "same_domain": s["same_domain"], "spearman": s["spearman"]})
        C = pd.DataFrame(rows)
        summary = C.groupby("same_domain")["spearman"].agg(["mean", "std", "min", "max", "count"])
        run.log_metric("transfer_by_domain_match", summary.round(4).to_dict("index"))
        C.to_csv(run.artifact_path("cross_corpus.csv"), index=False, encoding="utf-8")

        # ---------- 打印 ----------
        W = pd.DataFrame(within).T
        W["dataset"] = [k.split("|")[1] for k in W.index]
        print("\n" + "=" * 74)
        print(f"run_id: {run.run_id}")
        print(f"样本量: " + "  ".join(f"{SETS[k]}={len(y[k])}" for k in SETS))

        print("\n--- 组内 CV（Spearman ρ，GroupKFold-5）---")
        piv = W.reset_index().assign(algo=lambda d: d["index"].str.split("|").str[0])
        print(piv.pivot(index="algo", columns="dataset", values="spearman").round(3).to_string())

        print("\n--- 跨语料迁移（Spearman ρ）---")
        print(C.pivot_table(index="algo", columns=["src", "dst"], values="spearman")
              .round(3).to_string())

        print("\n--- 迁移能力：同域 vs 跨域 ---")
        print(summary.round(3).to_string())

        best_within = W["spearman"].max()
        same = C[C.same_domain]["spearman"].mean()
        diff = C[~C.same_domain]["spearman"].mean()
        print(f"\n最佳组内 ρ = {best_within:.3f}")
        print(f"同域跨源 ρ 均值 = {same:.3f}   跨域 ρ 均值 = {diff:.3f}   落差 = {same - diff:.3f}")


if __name__ == "__main__":
    main()
