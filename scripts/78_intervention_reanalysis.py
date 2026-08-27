r"""反事实干预的剂量—反应，在**有功效的层面**重算。

    .venv/Scripts/python scripts/78_intervention_reanalysis.py

## 原来的统计量不可能显著 🔴

`10_counterfactual.py` 报告的 `dose_response_rho` 是把 80 段音频先按剂量
求平均、再在**4 个剂量点**上算秩相关。n = 4 时 Spearman 的最小可能
双侧 p 是 **0.083** —— 也就是说，无论数据多好，那个统计量在 α = 0.05 下
**永远不可能显著**。报告 ρ = +0.632 而不给 p，读者会误以为那是证据。

这不是数据的问题，是把 80 段的信息聚合掉了。

## 重算：在音频层面

底层是 80 段源音频 × 4 剂量 × 2 个域模型。正确的做法是

1. 对**每一段音频**求其自身的剂量—反应斜率（4 个点，只作为个体估计）；
2. 在 **80 段**上检验这些斜率的中位数是否偏离 0 —— Wilcoxon 符号秩，
   外加符号翻转置换零分布。

n = 80 功效充足，且这才是「这个干预是否系统性地推动预测」的正确问法。

## 同时补上特异性的准确表述

`10` 已经算了 `shift_rank_among_122`，但正文把它写成「各自推动了一族相关特征」，
掩盖了更硬的事实：transient_inject 的目标特征只排第 12，
而某个非目标特征的位移是它的 4.5 倍。本脚本把这组数原样列出，
供正文照实改写。
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")
RUNS = REPO_ROOT / "runs"
OUT = REPO_ROOT / "reports" / "source_data"
SEED = 20260731
N_PERM = 10000


def latest(suffix: str) -> Path:
    ds = sorted(d for d in RUNS.glob(f"*{suffix}") if (d / "meta.json").exists())
    return ds[-1]


def signflip_p(v: np.ndarray, rng: np.random.Generator) -> float:
    obs = abs(float(np.mean(v)))
    s = rng.integers(0, 2, (N_PERM, len(v))) * 2 - 1
    null = np.abs(s @ v / len(v))
    return float((1 + (null >= obs).sum()) / (1 + N_PERM))


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("78_intervention_reanalysis", SEED)
    d = latest("_counterfactual_intervention")
    R = pd.read_parquet(d / "counterfactual_raw.parquet")
    print(f"来源 {d.name}")
    print(f"原始表 {R.shape}   列 {list(R.columns)}\n")

    # 两个域模型各有一列预测。第一版只取了最后一列，漏掉音乐域 ——
    # 「取 columns[-1]」这种写法正是本项目屡次出错的来源，此处显式列出。
    PRED = {"p_deam": "Music model", "p_emo": "Ambient model"}
    assert set(PRED) <= set(R.columns), f"缺预测列：{set(PRED) - set(R.columns)}"
    doses = sorted(R.dose.unique())
    print(f"剂量水平 {doses}   预测列 {list(PRED)}")
    print(f"源音频 {R.source.nunique()} 段\n")

    print("=" * 84)
    print("旧统计量 vs 新统计量")
    print("  旧：先按剂量求均值 → 4 个点上算秩相关（n=4，最小可能 p = 0.083）")
    print("  新：逐段音频求斜率 → 80 段上做 Wilcoxon + 符号翻转置换\n")

    rows = []
    for iv in sorted(R.intervention.unique()):
        for key, dom in PRED.items():
            S = R[R.intervention == iv]
            # 旧：4 点聚合
            agg = S.groupby("dose")[key].mean().reindex(doses)
            old_rho = spearmanr(doses, agg.to_numpy()).statistic

            # 新：逐段斜率
            slopes = []
            for src, g in S.groupby("source"):
                g = g.sort_values("dose")
                if len(g) < 3 or g[key].std() < 1e-12:
                    continue
                slopes.append(float(np.polyfit(g.dose, g[key], 1)[0]))
            sv = np.array(slopes)
            if len(sv) < 20:
                continue
            w = wilcoxon(sv).pvalue
            pp = signflip_p(sv, rng)
            frac = float(np.mean(np.sign(sv) == np.sign(np.median(sv))))
            rows.append({"intervention": iv, "domain": dom, "n_clips": len(sv),
                         "old_rho_n4": old_rho, "median_slope": float(np.median(sv)),
                         "mean_slope": float(sv.mean()), "wilcoxon_p": float(w),
                         "signflip_p": pp, "sign_consistency": frac})
            print(f"  {iv:<18} {str(dom):<15} 旧 ρ = {old_rho:+.3f} (n=4, p≥.083)"
                  f"   →  新 n={len(sv)}  中位斜率 {np.median(sv):+.5f}  "
                  f"Wilcoxon p = {w:.4f}  置换 p = {pp:.4f}  "
                  f"同号率 {frac:.0%}")

    D = pd.DataFrame(rows)
    D.to_csv(OUT / "intervention_dose_response_clip_level.csv",
             index=False, encoding="utf-8")

    sig = D[D.signflip_p < 0.05]
    print(f"\n  置换 p < .05 的格子：{len(sig)}/{len(D)}")
    run.log_metric("n_significant_cells", int(len(sig)))
    run.log_metric("n_cells", int(len(D)))

    # ── 特异性：把话说准 ──────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("特异性：目标特征在 122 个描述符里排第几，位移是最大者的几分之一")
    M = pd.read_csv(d / "intervention_summary.csv")
    prim = M.groupby("intervention").first().reset_index()
    for _, x in prim.iterrows():
        print(f"  {x.intervention:<18} 目标 {x.target:<22} "
              f"位移 {x.target_shift_sd:.2f} SD   "
              f"最大非目标 {x.max_other_shift_sd:.2f} SD   "
              f"（{x.max_other_shift_sd/x.target_shift_sd:.1f} 倍）   "
              f"目标排名 {int(x.shift_rank_among_122)}/122")
    run.write()
    print("\n" + "=" * 84)
    print("→ 正文须据此改写：剂量—反应用音频层面的检验，特异性照实说排名。")


if __name__ == "__main__":
    main()
