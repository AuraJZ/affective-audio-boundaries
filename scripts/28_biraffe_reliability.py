"""BIRAFFE2 信度检验 —— 建模之前先问「这批数据能支撑多强的结论」。

    uv run python scripts/28_biraffe_reliability.py

## 为什么这一步必须在建模之前

教训 B2：说「X 预测不了 Y」之前必须先测 Y 的信度。
PMEmo 的 EDA 我们是**先建模、后测信度**，结果发现信度只有 0.155，
前面的建模全部作废。这次反过来。

三个天花板要分别测：

1. **主观 arousal 的跨受试信度** —— 与 DEAM(0.868)、ARAUS(0.566) 可比
2. **生理指标的跨受试信度** —— 与 PMEmo EDA(0.155) 可比
3. **主观 ↔ 生理的相关**，并对照上面两个天花板

若第 2 项同样很低，则结论不是「BIRAFFE2 不行」，而是
**「用外周生理测单次刺激响应，信噪比本身就不够」** ——
这对依赖同类信号的智能枕产品线是关键情报。

## 数据结构

每名受试 121 个试次：`TIMESTAMP; ID; COND; IADS-ID; IAPS-ID; ANS-VALENCE; ANS-AROUSAL; ANS-TIME`

- `COND` 中 `S±/S0` 为**纯声音**试次，`PS±/PS0` 为图+声
- 本分析**只用纯声音试次**（IAPS-ID 为 None），排除视觉混淆
- 每人约 30 次 BITalino 掉线，需按段剔除
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

ROOT = REPO_ROOT / "data" / "raw" / "BIRAFFE2"
PROC = ROOT / "procedure"
N_SPLITS = 200
MIN_RATERS = 4          # 每个刺激至少这么多受试才纳入分半


def load_procedures() -> pd.DataFrame:
    rows = []
    for f in sorted(PROC.rglob("SUB*-Procedure.csv")):
        try:
            d = pd.read_csv(f, sep=";")
        except Exception:
            continue
        d["subject"] = f.stem.split("-")[0]
        rows.append(d)
    if not rows:
        raise SystemExit(f"未在 {PROC} 找到 Procedure 文件")
    P = pd.concat(rows, ignore_index=True)
    P.columns = [c.strip() for c in P.columns]
    return P


def split_half(df: pd.DataFrame, value: str, item: str, rater: str,
               rng, n_splits: int, min_items: int = 20) -> dict | None:
    """受试随机分半 → 各自的刺激均值 → 相关 → Spearman-Brown。"""
    raters = df[rater].unique()
    if len(raters) < 4:
        return None
    raw = []
    for _ in range(n_splits):
        perm = rng.permutation(raters)
        a = set(perm[: len(perm) // 2])
        ga = df[df[rater].isin(a)].groupby(item)[value].mean()
        gb = df[~df[rater].isin(a)].groupby(item)[value].mean()
        common = ga.index.intersection(gb.index)
        if len(common) >= min_items:
            r = spearmanr(ga.loc[common], gb.loc[common]).statistic
            if np.isfinite(r):
                raw.append(float(r))
    if not raw:
        return None
    raw = np.array(raw)
    sb = 2 * raw / (1 + raw)
    return {"raw_mean": float(raw.mean()),
            "spearman_brown": float(sb.mean()),
            "p10": float(np.percentile(sb, 10)),
            "p90": float(np.percentile(sb, 90)),
            "n_splits": int(raw.size)}


def main() -> None:
    P = load_procedures()

    # 试次行 = 有 IADS-ID 或 IAPS-ID 的行
    trials = P[P["IADS-ID"].notna() & (P["IADS-ID"].astype(str) != "None")].copy()
    trials["IADS-ID"] = pd.to_numeric(trials["IADS-ID"], errors="coerce")
    trials = trials.dropna(subset=["IADS-ID"])
    trials["IADS-ID"] = trials["IADS-ID"].astype(int)

    for c in ("ANS-VALENCE", "ANS-AROUSAL", "ANS-TIME"):
        trials[c] = pd.to_numeric(trials[c], errors="coerce")

    # 纯声音试次：IAPS-ID 缺失或为 'None'
    iaps = trials["IAPS-ID"].astype(str)
    sound_only = trials[iaps.isin(["None", "nan", ""]) | trials["IAPS-ID"].isna()].copy()
    # 排除练习试次
    sound_only = sound_only[sound_only["COND"].astype(str).str.lower() != "train"]

    rng = np.random.default_rng(SEED)

    with RunRecord(
        "biraffe_reliability", seed=SEED,
        params={"dataset": "BIRAFFE2", "licence": "CC BY 4.0",
                "n_splits": N_SPLITS, "min_raters_per_item": MIN_RATERS,
                "trial_filter": "sound-only, non-training",
                "note": "先测信度再建模 —— PMEmo 的教训"},
    ) as run:
        run.log_metric("n_subjects", int(P["subject"].nunique()))
        run.log_metric("n_trials_all", int(len(trials)))
        run.log_metric("n_trials_sound_only", int(len(sound_only)))
        run.log_metric("n_unique_iads", int(sound_only["IADS-ID"].nunique()))

        per_item = sound_only.groupby("IADS-ID")["subject"].nunique()
        run.log_metric("raters_per_stimulus",
                       {"median": float(per_item.median()),
                        "p10": float(per_item.quantile(0.1)),
                        "max": int(per_item.max()),
                        "n_with_min": int((per_item >= MIN_RATERS).sum())})

        keep = per_item[per_item >= MIN_RATERS].index
        S = sound_only[sound_only["IADS-ID"].isin(keep)].dropna(subset=["ANS-AROUSAL"])

        # ---------- 1. 主观 arousal 的跨受试信度 ----------
        ceil_ar = split_half(S, "ANS-AROUSAL", "IADS-ID", "subject", rng, N_SPLITS)
        ceil_va = split_half(S.dropna(subset=["ANS-VALENCE"]),
                             "ANS-VALENCE", "IADS-ID", "subject", rng, N_SPLITS)
        run.log_metric("ceiling_subjective_arousal", ceil_ar)
        run.log_metric("ceiling_subjective_valence", ceil_va)

        # ---------- 2. 生理（需 biosigs，未到位则跳过并说明）----------
        bio_zip = ROOT / "BIRAFFE2-biosigs.zip"
        bio_ready = bio_zip.exists() and bio_zip.stat().st_size > 5e9
        run.log_metric("biosigs_available", bool(bio_ready))

        # ---------- 打印 ----------
        print("=" * 74)
        print(f"run_id: {run.run_id}")
        print(f"受试 {P['subject'].nunique()} 名   "
              f"试次 {len(trials):,}（纯声音 {len(sound_only):,}）   "
              f"IADS 刺激 {sound_only['IADS-ID'].nunique()} 个")
        print(f"每个刺激的受试数：中位 {per_item.median():.0f}  "
              f"p10 {per_item.quantile(0.1):.0f}  最大 {per_item.max()}")
        print(f"≥{MIN_RATERS} 名受试的刺激：{(per_item >= MIN_RATERS).sum()} 个")

        print("\n--- 1. 主观评分的跨受试信度（分半 + Spearman-Brown）---")
        for nm, c in (("arousal", ceil_ar), ("valence", ceil_va)):
            if c:
                print(f"  {nm:<9} 分半 {c['raw_mean']:+.3f}   "
                      f"校正后 **{c['spearman_brown']:+.3f}**   "
                      f"[{c['p10']:+.3f}, {c['p90']:+.3f}]")
            else:
                print(f"  {nm:<9} 可用刺激不足，无法估计")

        print("\n--- 横向对照（同一方法算出的天花板）---")
        rows = [("DEAM 主观 arousal", 0.868), ("ARAUS eventfulness", 0.566),
                ("PMEmo EDA（生理）", 0.155)]
        if ceil_ar:
            rows.append(("**BIRAFFE2 主观 arousal**", ceil_ar["spearman_brown"]))
        for nm, v in rows:
            print(f"  {nm:<26} {v:+.3f}")

        print("\n--- 2. 生理信度 ---")
        if bio_ready:
            print("  biosigs 已就位，可运行下一步脚本提取 ECG/GSR 特征。")
        else:
            sz = bio_zip.stat().st_size / 1e9 if bio_zip.exists() else 0
            print(f"  biosigs 尚未下载完成（{sz:.1f}/5.2 GB），本轮跳过。")
            print("  ⚠️ 生理信度是本次的核心问题 —— 主观信度高不代表生理可用，")
            print("     PMEmo 就是主观可用而生理不可用。")


if __name__ == "__main__":
    main()
