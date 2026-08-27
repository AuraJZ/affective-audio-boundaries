"""ARAUS 外部复现 —— 「变异 > 水平」能否在另一实验室的真人数据上重现？

    uv run python scripts/26_araus_external.py

## 为什么这是最强的外部检验

| 维度 | 我们的语料 | ARAUS |
|---|---|---|
| 标注 | 众包 | **实验室，605 名受试** |
| 量表 | 自定 arousal | **ISO/TS 12913-2 Method A** |
| 构念 | arousal | **eventfulness**（声景环形模型的唤醒轴）|
| 设计 | 观察性 | **受控增强**（掩蔽声 × 信噪比 −6…+6 dB）|
| 特征 | 我们自己提的 | **他们自己算的心理声学量** |

⚠️ **eventfulness ≠ arousal**，是相关但不同的构念。
本检验回答「规则能否预测一个**相关但不同**的构念」——更严格，
但结论必须如此表述，不可写成「外部验证了 arousal 预测」。

## 关键：用 ARAUS 自己的特征体系构造两族

`responses.csv` 每段刺激都带心理声学量的**分位数统计**：

    S / N / F / R / T / LA / LC  ×  avg, max, 05, 10, …, 95

于是两族有了自然的操作定义：

- **水平族**：`Xavg`（中心趋势）
- **变异族**：`X95 − X05`（分位间距 = 该量在时间上的变异幅度）

这是**用他们的量、他们的人、他们的量表**来检验我们的主张，
不借用我们自己的任何特征提取代码 —— 最干净的外部检验形式。

## 附带：两个此前无法回答的问题

1. **Fluctuation strength**：MoSQITo 未实现，我们只能自造代理量（p=1.000 无用）。
   ARAUS 有真实 ISO 实现。
2. **Roughness**：我们测得 6 项全部 p≈1.0。ARAUS 能验证这是真阴性还是实现问题。
"""

from __future__ import annotations

import argparse
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED, build_regressors, reg_score

DATA_ZIP = REPO_ROOT / "data" / "raw" / "ARAUS" / "f_114619.zip"

# 心理声学量前缀 → 含义
PSYCHO = {"S": "Sharpness", "N": "Loudness", "F": "FluctuationStrength",
          "R": "Roughness", "T": "Tonality", "LA": "SPL_A", "LC": "SPL_C"}
RATINGS = ["pleasant", "eventful", "chaotic", "vibrant",
           "uneventful", "calm", "annoying", "monotonous"]


def build_families(d: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """从分位数列构造「水平族」与「变异族」。"""
    level, variability = [], []
    out = d.copy()
    for p, label in PSYCHO.items():
        avg, p05, p95 = f"{p}avg_r", f"{p}05_r", f"{p}95_r"
        if avg in d.columns:
            level.append(avg)
        if p05 in d.columns and p95 in d.columns:
            col = f"{p}_range_r"
            out[col] = d[p95] - d[p05]
            variability.append(col)
    return out, level, variability


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="eventful", choices=RATINGS)
    args = ap.parse_args()

    z = zipfile.ZipFile(DATA_ZIP)
    with z.open("datav2/responses.csv") as fh:
        R = pd.read_csv(fh)

    R = R[R.get("is_attention", 0) == 0] if "is_attention" in R.columns else R
    R, level, variability = build_families(R)

    # 刺激 = (底噪, 掩蔽声, 信噪比) 的唯一组合
    key = ["soundscape", "masker", "smr"]
    agg = {args.target: "mean", **{c: "first" for c in level + variability}}
    S = R.groupby(key, as_index=False).agg(agg)
    S["n_raters"] = R.groupby(key).size().to_numpy()

    y = S[args.target].to_numpy(dtype=float)
    groups = S["soundscape"].to_numpy()      # 按底噪分组，防止同源泄漏

    with RunRecord(
        "araus_external_validation", seed=SEED,
        params={"target": args.target, "n_responses": int(len(R)),
                "n_stimuli": int(len(S)),
                "n_participants": int(R["participant"].nunique()),
                "level_features": level, "variability_features": variability,
                "construct_note": "eventfulness ≠ arousal；跨构念检验",
                "cv": "GroupKFold(5) grouped by base soundscape"},
    ) as run:
        run.log_metric("n_stimuli", int(len(S)))
        run.log_metric("n_participants", int(R["participant"].nunique()))
        run.log_metric("raters_per_stimulus",
                       {"median": float(S.n_raters.median()),
                        "min": int(S.n_raters.min()), "max": int(S.n_raters.max())})

        # ---------- 1. 逐特征相关 ----------
        rows = []
        for c in level + variability:
            v = pd.to_numeric(S[c], errors="coerce")
            if v.notna().sum() < 100 or v.std() == 0:
                continue
            r = spearmanr(v, y, nan_policy="omit").statistic
            fam = "Variability" if c in variability else "Level"
            rows.append({"feature": c, "family": fam, "rho": float(r),
                         "abs_rho": abs(float(r))})
        F = pd.DataFrame(rows).sort_values("abs_rho", ascending=False)
        F.to_csv(run.artifact_path("feature_vs_target.csv"), index=False,
                 encoding="utf-8")
        run.log_metric("per_feature", F.round(4).to_dict("records"))

        fam_cmp = F.groupby("family")["abs_rho"].agg(["mean", "max", "count"])
        run.log_metric("family_comparison", fam_cmp.round(4).to_dict("index"))

        # ---------- 2. 两族各自的预测力（模型层面）----------
        cv = GroupKFold(n_splits=5)
        perf = {}
        for fam, cols in (("Level", level), ("Variability", variability),
                          ("Both", level + variability)):
            X = SimpleImputer(strategy="median").fit_transform(
                S[cols].to_numpy(dtype=float))
            pred = np.zeros(len(y))
            for tr, te in cv.split(X, y, groups):
                m = build_regressors()["XGBoost"].fit(X[tr], y[tr])
                pred[te] = m.predict(X[te])
            perf[fam] = reg_score(y, pred)
        run.log_metric("family_model_performance", perf)

        # ---------- 3. 剂量–反应：信噪比（真人）----------
        dose = S.groupby("smr")[args.target].agg(["mean", "std", "count"])
        rho_smr = spearmanr(S["smr"], y).statistic
        run.log_metric("dose_response_smr",
                       {"by_smr": dose.round(4).to_dict("index"),
                        "spearman": float(rho_smr)})

        # ---------- 4. 此前无法回答的两项 ----------
        special = {}
        for pref, label in (("F", "FluctuationStrength"), ("R", "Roughness")):
            for suf in ("avg_r", "_range_r"):
                c = f"{pref}{suf}" if suf.startswith("avg") else f"{pref}{suf}"
                if c in S.columns:
                    v = pd.to_numeric(S[c], errors="coerce")
                    if v.notna().sum() > 100 and v.std() > 0:
                        special[c] = float(spearmanr(v, y, nan_policy="omit").statistic)
        run.log_metric("previously_unanswerable", special)

        # ---------- 打印 ----------
        print("=" * 76)
        print(f"run_id: {run.run_id}")
        print(f"{len(R):,} 条评分 → {len(S):,} 段刺激   "
              f"{R['participant'].nunique()} 名受试   目标: {args.target}")
        print(f"每段刺激评分人数 中位 {S.n_raters.median():.0f}")

        print(f"\n--- 1. 逐特征 |ρ| 前 12 ---")
        print(f"{'特征':<20}{'族':<14}{'ρ':>9}")
        for r in F.head(12).itertuples():
            print(f"{r.feature:<20}{r.family:<14}{r.rho:>+9.3f}")

        print(f"\n--- 2. 族间对比 ---")
        print(fam_cmp.round(4).to_string())
        print()
        for fam, s in perf.items():
            print(f"  {fam:<14} 模型 ρ = {s['spearman']:+.4f}   R² = {s['r2']:+.4f}")

        print(f"\n--- 3. 剂量–反应：信噪比 → {args.target}（真人）---")
        print(f"  Spearman ρ = {rho_smr:+.3f}")
        for k, v in dose.round(3).to_dict("index").items():
            print(f"    SMR {k:>4} dB : {v['mean']:+.3f}  (n={int(v['count'])})")

        print(f"\n--- 4. 此前无法回答的两项 ---")
        for c, r in special.items():
            print(f"    {c:<16} ρ = {r:+.3f}")

        print("\n" + "=" * 76)
        v_mean = fam_cmp.loc["Variability", "mean"] if "Variability" in fam_cmp.index else np.nan
        l_mean = fam_cmp.loc["Level", "mean"] if "Level" in fam_cmp.index else np.nan
        if np.isfinite(v_mean) and np.isfinite(l_mean):
            if v_mean > l_mean:
                print(f"→ 变异族 ({v_mean:.3f}) > 水平族 ({l_mean:.3f})：主张在外部数据上重现。")
            else:
                print(f"→ 变异族 ({v_mean:.3f}) ≤ 水平族 ({l_mean:.3f})：主张**未**重现。")
        print("⚠️ eventfulness 与 arousal 是相关但不同的构念，结论须如此表述。")


if __name__ == "__main__":
    main()
