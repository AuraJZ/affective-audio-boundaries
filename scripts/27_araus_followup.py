"""ARAUS 追加检验 —— 把我们自己的两条教训用在这份外部数据上。

    uv run python scripts/27_araus_followup.py

## 为什么必须先做这两项，才能解读 26 号脚本的阴性结果

**教训 B2**：说「X 预测不了 Y」之前必须先测 Y 的信度。
`26_araus_external.py` 显示每段刺激**中位仅 1 名评分者**。
单人评分的信度天花板可能很低 —— 若天花板约 0.4，则模型 ρ=0.36 已近顶，
族间比较有意义；若天花板很高，则所有数字都远未到顶，比较可能只是噪声。

**教训（step11 的操纵检查）**：声称控制了某变量，就要验证它真的起作用。
26 号脚本得到信噪比 → eventfulness 的 ρ = −0.002（完全平坦），
这与 ARAUS 的整个实验设计矛盾。最可能的原因是**漏了交互**：
信噪比调的是掩蔽声相对电平，而掩蔽声分平静类（鸟、水）与喧闹类（交通、施工），
两类效应方向相反，池化后互相抵消。

## 两项检验

1. **信度天花板**：用有 ≥2 名评分者的刺激做分半 + Spearman-Brown
2. **信噪比 × 掩蔽声类型交互**：分类型看剂量–反应
"""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

DATA_ZIP = REPO_ROOT / "data" / "raw" / "ARAUS" / "f_114619.zip"
TARGET = "eventful"
N_SPLITS = 200


def main() -> None:
    z = zipfile.ZipFile(DATA_ZIP)
    with z.open("datav2/responses.csv") as fh:
        R = pd.read_csv(fh, usecols=["participant", "soundscape", "masker", "smr",
                                     "is_attention", TARGET, "pleasant", "calm"])
    with z.open("datav2/maskers.csv") as fh:
        M = pd.read_csv(fh, usecols=["masker", "class"])

    R = R[R["is_attention"] == 0]
    R = R.merge(M, on="masker", how="left")
    R["stim"] = (R.soundscape.astype(str) + "|" + R.masker.astype(str)
                 + "|" + R.smr.astype(str))

    with RunRecord(
        "araus_followup", seed=SEED,
        params={"target": TARGET, "n_responses": int(len(R)),
                "checks": ["reliability ceiling", "SMR x masker-class interaction"]},
    ) as run:
        # ---------- 1. 信度天花板 ----------
        counts = R.groupby("stim").size()
        multi = counts[counts >= 2].index
        run.log_metric("raters_per_stimulus",
                       {"n_stimuli": int(counts.size),
                        "median": float(counts.median()),
                        "n_with_2plus": int(len(multi)),
                        "frac_with_2plus": float(len(multi) / counts.size)})

        rng = np.random.default_rng(SEED)
        Rm = R[R.stim.isin(multi)]
        halves = []
        for _ in range(N_SPLITS):
            # 每段刺激内随机把评分者分两半
            sh = Rm.sample(frac=1.0, random_state=int(rng.integers(0, 1 << 31)))
            first = sh.groupby("stim").head(1).set_index("stim")[TARGET]
            rest = sh.groupby("stim").tail(-1)
            if rest.empty:
                continue
            second = rest.groupby("stim")[TARGET].mean()
            common = first.index.intersection(second.index)
            if len(common) > 50:
                r = spearmanr(first.loc[common], second.loc[common]).statistic
                if np.isfinite(r):
                    halves.append(float(r))

        ceiling = None
        if halves:
            h = np.array(halves)
            sb = 2 * h / (1 + h)
            ceiling = {"raw_mean": float(h.mean()),
                       "spearman_brown": float(sb.mean()),
                       "p10": float(np.percentile(sb, 10)),
                       "p90": float(np.percentile(sb, 90)),
                       "n_splits": int(h.size)}
            run.log_metric("reliability_ceiling", ceiling)

        # ---------- 2. 信噪比 × 掩蔽声类型 ----------
        pooled_rho = float(spearmanr(R.smr, R[TARGET]).statistic)
        by_class = {}
        for cls, g in R.groupby("class"):
            if len(g) < 200:
                continue
            by_class[str(cls)] = {
                "n": int(len(g)),
                "rho_smr": float(spearmanr(g.smr, g[TARGET]).statistic),
                "means": g.groupby("smr")[TARGET].mean().round(3).to_dict(),
            }
        run.log_metric("pooled_smr_rho", pooled_rho)
        run.log_metric("smr_by_masker_class", by_class)

        rhos = [v["rho_smr"] for v in by_class.values()]
        # ⚠️ 判据修正：初版只查「符号是否有正有负」，那在 +0.044 vs −0.025 上就会
        # 触发，等于把噪声判成交互。真正的交互要求**至少有一类的效应本身可观**。
        MIN_EFFECT = 0.10
        real_interaction = bool(rhos and max(abs(r) for r in rhos) >= MIN_EFFECT
                                and min(rhos) < 0 < max(rhos))
        run.log_metric("interaction_evidence",
                       {"pooled": pooled_rho,
                        "per_class_min": float(min(rhos)) if rhos else None,
                        "per_class_max": float(max(rhos)) if rhos else None,
                        "max_abs_effect": float(max(abs(r) for r in rhos)) if rhos else None,
                        "min_effect_threshold": MIN_EFFECT,
                        "signs_differ": bool(rhos and min(rhos) < 0 < max(rhos)),
                        "real_interaction": real_interaction})

        # ---------- 打印 ----------
        print("=" * 74)
        print(f"run_id: {run.run_id}")
        rp = run.meta["metrics"]["raters_per_stimulus"]
        print(f"刺激 {rp['n_stimuli']:,} 段，中位评分人数 {rp['median']:.0f}，"
              f"≥2 人的占 {rp['frac_with_2plus']:.1%}")

        print("\n--- 1. eventfulness 的信度天花板 ---")
        if ceiling:
            print(f"  分半 ρ = {ceiling['raw_mean']:+.3f}   "
                  f"Spearman-Brown = {ceiling['spearman_brown']:+.3f}   "
                  f"[{ceiling['p10']:+.3f}, {ceiling['p90']:+.3f}]")
            for name, v in (("水平族模型", 0.3571), ("变异族模型", 0.2903)):
                pct = v / ceiling["spearman_brown"] if ceiling["spearman_brown"] else np.nan
                print(f"    {name} ρ={v:.3f}  → 占天花板 {pct:.0%}")
        else:
            print("  可用的多评分者刺激不足，无法估计。")

        print("\n--- 2. 信噪比 → eventfulness，按掩蔽声类型 ---")
        print(f"  池化（26 号脚本的做法）: ρ = {pooled_rho:+.3f}")
        print(f"\n{'掩蔽声类型':<16}{'n':>8}{'ρ(SMR)':>10}   各 SMR 均值")
        for cls, v in sorted(by_class.items(), key=lambda kv: kv[1]["rho_smr"]):
            means = "  ".join(f"{k:+.0f}:{x:.2f}" for k, x in sorted(v["means"].items()))
            print(f"{cls:<16}{v['n']:>8}{v['rho_smr']:>+10.3f}   {means}")

        ie = run.meta["metrics"]["interaction_evidence"]
        print(f"\n  符号有正有负: {ie['signs_differ']}   "
              f"最大 |ρ| = {ie['max_abs_effect']:.3f}（门槛 {ie['min_effect_threshold']}）")
        if ie["real_interaction"]:
            print("→ **存在实质交互**：池化的平坦是抵消产物。")
        elif ie["signs_differ"]:
            print("→ 符号虽有正负，但**幅度全在噪声水平**，不构成实质交互。")
            print("  剂量–反应是**真的平坦**，不是池化产物 —— 原假设不成立。")
        else:
            print("→ 各类型方向一致，池化平坦不是交互造成的。")


if __name__ == "__main__":
    main()
