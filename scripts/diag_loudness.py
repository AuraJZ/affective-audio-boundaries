"""响度混淆判别实验的对比分析。

    uv run python scripts/diag_loudness.py <原始run_id> <归一化run_id>

## 判别标准（先定死，不看结果再改）

问题：模型学到的是助眠规律，还是单纯的"这段音频多响"？
（arousal 标注时响度是最强的直觉线索，见 reports/step2_real_labels.md §4）

响度归一化后：

  解释 A —— 真信号：
    跨源 AUC 下降 < 0.05，且响度类特征仍有一半以上留在 top-20

  解释 B —— 构念混淆：
    跨源 AUC 下降 > 0.10，或响度类特征整体跌出 top-20 且无同等强度的特征顶上

  中间地带：响度贡献了一部分但不是全部 —— 需要在规则库里显式区分
    "绝对电平"与"相对动态"两类规则
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from soundml.provenance import RUNS_DIR

# 直接依赖绝对电平的特征。归一化后它们理应失效——这是实验的对照锚点。
LEVEL_FEATURES = {"mfcc1_mean", "rms_mean", "rms_p10", "rms_p50", "rms_p90"}

AUC_DROP_SIGNAL = 0.05
AUC_DROP_CONFOUND = 0.10


def load(run_id: str) -> tuple[dict, pd.DataFrame]:
    d = RUNS_DIR / run_id
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    ranks = pd.read_csv(d / "shap_ranks.csv", index_col=0)
    return meta, ranks


def best_cross_auc(meta: dict) -> tuple[str, float]:
    cross = meta["metrics"]["cross_source"]
    name = max(cross, key=lambda k: cross[k]["auc"])
    return name, cross[name]["auc"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("normalized")
    args = ap.parse_args()

    m0, r0 = load(args.baseline)
    m1, r1 = load(args.normalized)

    n0, auc0 = best_cross_auc(m0)
    n1, auc1 = best_cross_auc(m1)
    drop = auc0 - auc1

    print("=" * 66)
    print("响度混淆判别实验")
    print("=" * 66)
    print(f"\n基线      : {args.baseline}\n归一化    : {args.normalized}")
    print(f"\n最佳跨源 AUC:  {auc0:.4f} ({n0})  →  {auc1:.4f} ({n1})   Δ = {-drop:+.4f}")

    print("\n--- 绝对电平类特征的 SHAP 平均排名 ---")
    print(f"{'特征':<22} {'基线':>8} {'归一化':>8}  {'变化':>8}")
    for f in sorted(LEVEL_FEATURES):
        if f in r0.index and f in r1.index:
            a, b = r0.loc[f, "mean_rank"], r1.loc[f, "mean_rank"]
            print(f"{f:<22} {a:>8.1f} {b:>8.1f}  {b - a:>+8.1f}")

    in0 = {f for f in LEVEL_FEATURES if f in r0.index and r0.loc[f, "mean_rank"] <= 20}
    in1 = {f for f in LEVEL_FEATURES if f in r1.index and r1.loc[f, "mean_rank"] <= 20}
    print(f"\n留在 top-20 的电平类特征:  基线 {len(in0)} → 归一化 {len(in1)}")

    print("\n--- 归一化后的 top-10 ---")
    print(r1.head(10)[["mean_rank"]].round(1))

    print("\n--- SHAP 跨模型交集 ---")
    print(f"基线   : {m0['metrics']['shap_top20_overlap_3models']} 个")
    print(f"归一化 : {m1['metrics']['shap_top20_overlap_3models']} 个")

    print("\n" + "=" * 66)
    survived = len(in1) >= max(1, len(in0) // 2)
    if drop < AUC_DROP_SIGNAL and survived:
        print("判定：解释 A —— 真信号。响度类特征经得起归一化，可进入规则库。")
    elif drop > AUC_DROP_CONFOUND or (in0 and not in1):
        print("判定：解释 B —— 构念混淆。模型主要在学响度，规则库不能基于当前标签出。")
    else:
        print("判定：中间地带。响度贡献了一部分但非全部——")
        print("      规则库需显式区分「绝对电平」与「相对动态」两类规则。")
    print("=" * 66)


if __name__ == "__main__":
    main()
