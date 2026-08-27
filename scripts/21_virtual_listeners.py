"""虚拟听者群 —— 195 名 DEAM 评分者，每人一个模型。

    uv run python scripts/21_virtual_listeners.py

## 这在回答什么

「用数据库里的标注造一个虚拟的人，生成音频让他听，看他什么反应。」

正确形态不是**一个**虚拟人，而是**一群**：生成一段音频 → 问 195 个虚拟听者
→ 得到**分布**（多少人觉得放松、分歧多大），而不是一个点估计。
点估计会掩盖个体差异，而个体差异恰恰是产品「用户分型 → 声音子库」那一层的前提。

## 关键分解：个体差异是「声学响应不同」还是「量表用法不同」？

两者的产品含义完全不同：

- **量表用法不同**（有人整体打分偏高、有人量程窄）→ 校准问题，一个群体模型加个体偏移即可
- **声学响应不同**（有人对高频敏感、有人对突发敏感）→ 必须按人换**规则**，
  笔记 §9.2 的 Layer 3 才有存在意义

做法：把每个评分者的打分拆成
    rater_ij = 群体均值_j + 个体偏差_ij
然后检验**个体偏差能否由声学特征预测**。

- 能预测 → 该评分者对特定声学属性有系统性的不同响应 → 真·声学个体差异
- 不能预测 → 只是偏移/量程差异 → 校准问题

## 诚实前提

单个评分者只有约 91 条打分、122 维特征 —— 直接建模必然过拟合。
因此个体模型一律用**强正则岭回归 + 逐评分者交叉验证**，
并与「群体模型直接套用到该评分者」做配对比较。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED, feature_cols

RATER_DIR = (REPO_ROOT / "data" / "raw" / "DEAM" / "annotations"
             / "annotations per each rater" / "song_level")
FEAT = REPO_ROOT / "data" / "features" / "layer_a_reg_deam_ln.parquet"

MIN_RATINGS = 40          # 每名评分者的最少打分数
ALPHAS = np.logspace(-1, 4, 20)   # 强正则：样本少、维度高
N_FOLDS = 5


def load_raters() -> pd.DataFrame:
    rename = {"workerid": "workerID", "songid": "SongId",
              "arousal": "Arousal", "valence": "Valence",
              "arousal_average": "Arousal", "valence_average": "Valence"}
    frames = []
    for f in sorted(RATER_DIR.glob("*.csv")):
        d = pd.read_csv(f, skipinitialspace=True)
        d.columns = [rename.get(c.strip().lower(), c.strip()) for c in d.columns]
        frames.append(d[["workerID", "SongId", "Arousal"]])
    return pd.concat(frames, ignore_index=True)


def cv_pred(X: np.ndarray, y: np.ndarray, rng_seed: int) -> np.ndarray:
    """岭回归的交叉验证外样本预测。"""
    out = np.zeros(len(y))
    kf = KFold(n_splits=min(N_FOLDS, len(y) // 6), shuffle=True, random_state=rng_seed)
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), y[tr])
        out[te] = m.predict(sc.transform(X[te]))
    return out


def main() -> None:
    raters = load_raters()
    feats = pd.read_parquet(FEAT)
    feats["SongId"] = feats["clip_id"].str.removeprefix("deam_").astype(int)
    cols = [c for c in feature_cols(feats) if c != "SongId"]

    X_all = SimpleImputer(strategy="median").fit_transform(
        feats[cols].to_numpy(dtype=np.float64))
    feat_idx = {s: i for i, s in enumerate(feats["SongId"])}

    raters = raters[raters["SongId"].isin(feat_idx)]
    # 群体均值（每首歌），个体偏差 = 个人打分 − 群体均值
    grp = raters.groupby("SongId")["Arousal"].mean()
    raters = raters.assign(group_mean=raters["SongId"].map(grp))
    raters["deviation"] = raters["Arousal"] - raters["group_mean"]

    counts = raters.groupby("workerID").size()
    keep = counts[counts >= MIN_RATINGS].index
    R = raters[raters["workerID"].isin(keep)]

    with RunRecord(
        "virtual_listeners", seed=SEED,
        params={"corpus": "DEAM", "min_ratings_per_rater": MIN_RATINGS,
                "n_raters_kept": int(len(keep)),
                "model": "RidgeCV (strong regularisation) + per-rater KFold",
                "question": "个体差异是声学响应差异还是量表用法差异"},
    ) as run:
        run.log_metric("n_raters_total", int(raters["workerID"].nunique()))
        run.log_metric("n_raters_modelled", int(len(keep)))
        run.log_metric("ratings_per_rater",
                       {"median": float(counts[keep].median()),
                        "min": int(counts[keep].min()),
                        "max": int(counts[keep].max())})

        rows = []
        for w, g in R.groupby("workerID"):
            idx = [feat_idx[s] for s in g["SongId"]]
            X = X_all[idx]
            y_abs = g["Arousal"].to_numpy(dtype=float)
            y_dev = g["deviation"].to_numpy(dtype=float)
            gm = g["group_mean"].to_numpy(dtype=float)
            if len(y_abs) < 30 or np.std(y_abs) == 0:
                continue

            # 1) 该评分者的打分，群体均值能解释多少（= 该评分者有多「典型」）
            rho_group = spearmanr(gm, y_abs).statistic

            # 2) 个体模型：直接预测该评分者的绝对打分
            rho_pers = spearmanr(cv_pred(X, y_abs, SEED), y_abs).statistic

            # 3) 关键：个体**偏差**能否由声学预测
            rho_dev = (spearmanr(cv_pred(X, y_dev, SEED), y_dev).statistic
                       if np.std(y_dev) > 1e-9 else np.nan)

            rows.append({"workerID": w, "n": int(len(y_abs)),
                         "rho_group_model": float(rho_group),
                         "rho_personal_model": float(rho_pers),
                         "rho_deviation_model": float(rho_dev),
                         "rating_mean": float(y_abs.mean()),
                         "rating_sd": float(y_abs.std())})

        D = pd.DataFrame(rows).dropna(subset=["rho_group_model"])
        D.to_csv(run.artifact_path("virtual_listeners.csv"), index=False, encoding="utf-8")

        dev = D["rho_deviation_model"].dropna()
        # 偏差可预测性是否显著高于 0（配对符号秩，对照 0）
        stat = wilcoxon(dev, alternative="greater") if len(dev) > 10 else None

        summary = {
            "rho_group_model": {"mean": float(D.rho_group_model.mean()),
                                "sd": float(D.rho_group_model.std()),
                                "p10": float(D.rho_group_model.quantile(.1)),
                                "p90": float(D.rho_group_model.quantile(.9))},
            "rho_personal_model": {"mean": float(D.rho_personal_model.mean()),
                                   "sd": float(D.rho_personal_model.std())},
            "rho_deviation_model": {"mean": float(dev.mean()),
                                    "sd": float(dev.std()),
                                    "frac_positive": float((dev > 0).mean()),
                                    "p_value_vs_zero": float(stat.pvalue) if stat else None},
            "rating_offset_spread_sd": float(D.rating_mean.std()),
            "rating_scale_spread_sd": float(D.rating_sd.std()),
        }
        run.log_metric("summary", summary)
        run.log_metric("acoustic_individual_difference",
                       bool(dev.mean() > 0.1 and stat and stat.pvalue < 0.05))

        # ---------- 打印 ----------
        print("=" * 74)
        print(f"run_id: {run.run_id}")
        print(f"虚拟听者 {len(D)} 名（每人 ≥{MIN_RATINGS} 条打分，"
              f"中位 {counts[keep].median():.0f} 条）\n")

        print("--- 每名评分者的可预测性 ---")
        print(f"  群体均值直接套用   ρ = {D.rho_group_model.mean():.3f} "
              f"± {D.rho_group_model.std():.3f}   "
              f"[p10 {D.rho_group_model.quantile(.1):.3f}, "
              f"p90 {D.rho_group_model.quantile(.9):.3f}]")
        print(f"  个体声学模型       ρ = {D.rho_personal_model.mean():.3f} "
              f"± {D.rho_personal_model.std():.3f}")

        print("\n--- 关键检验：个体偏差能否由声学预测 ---")
        print(f"  偏差模型 ρ = {dev.mean():+.3f} ± {dev.std():.3f}   "
              f"为正的比例 {100 * (dev > 0).mean():.0f}%")
        if stat:
            print(f"  Wilcoxon vs 0: p = {stat.pvalue:.2e}")

        print("\n--- 量表用法的个体差异 ---")
        print(f"  个人均分的跨人标准差 = {D.rating_mean.std():.3f}")
        print(f"  个人量程的跨人标准差 = {D.rating_sd.std():.3f}")

        print("\n" + "=" * 74)
        if dev.mean() > 0.1 and stat and stat.pvalue < 0.05:
            print("→ 个体偏差可由声学预测：存在**真·声学个体差异**，")
            print("  「用户分型 → 声音子库」那一层有依据。")
        else:
            print("→ 个体偏差**不能**由声学预测：个体差异主要是量表用法/偏移，")
            print("  而非对声学属性的响应不同。")
            print("  含义：一个群体模型 + 个体校准即可，不需要按人换规则。")


if __name__ == "__main__":
    main()
