"""BIRAFFE2 生理信度 —— 本项目最关键的一次检验。

    uv run python scripts/29_biraffe_physio.py

## 为什么这一次比之前都重要

到目前为止**全部结论都建立在主观打分上**。唯一一次碰生理数据（PMEmo EDA）
发现信度仅 0.155，无法支撑任何结论。

BIRAFFE2 是我们能拿到的最好条件：102 名受试、实验室 ECG+EDA、
标准情绪诱发范式、每个刺激中位 36 人、CC BY 4.0 可商用。

**如果这个条件下生理信度仍然很低**，结论不是「这批数据不行」，而是：

> **用外周生理测单次声音刺激的响应，信噪比本身就不够。**

这对依赖同类信号（HR/HRV/体动）的智能枕产品线是关键情报 ——
它意味着设备端「检测到唤醒升高 → 切换音频」的实时闭环需要重新论证。

**如果信度够**，我们第一次有了能支撑「声学 → 生理」的数据。

两种结果都必须如实报告。

## 分析

1. 逐试次提取 EDA/ECG 特征（仅纯声音试次，排除掉线段）
2. 被试内 z 标准化（外周生理个体差异极大，不标准化会被少数人主导）
3. **跨受试分半信度**，与主观 arousal 的 0.813 同法可比
4. 主观 ↔ 生理相关，并对照两个天花板
"""

from __future__ import annotations

import argparse
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from soundml.biosignals import FEATURES, detect_gaps, trial_features
from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

ROOT = REPO_ROOT / "data" / "raw" / "BIRAFFE2"
BIO_ZIP = ROOT / "BIRAFFE2-biosigs.zip"
PROC = ROOT / "procedure"
N_SPLITS = 200
MIN_RATERS = 4


def load_trials() -> pd.DataFrame:
    rows = []
    for f in sorted(PROC.rglob("SUB*-Procedure.csv")):
        try:
            d = pd.read_csv(f, sep=";", low_memory=False)
        except Exception:
            continue
        d.columns = [c.strip() for c in d.columns]
        d["subject"] = f.stem.split("-")[0]
        rows.append(d)
    P = pd.concat(rows, ignore_index=True)

    P = P[P["IADS-ID"].notna() & (P["IADS-ID"].astype(str) != "None")].copy()
    P["IADS-ID"] = pd.to_numeric(P["IADS-ID"], errors="coerce")
    P = P.dropna(subset=["IADS-ID"])
    P["IADS-ID"] = P["IADS-ID"].astype(int)
    P["TIMESTAMP"] = pd.to_numeric(P["TIMESTAMP"], errors="coerce")
    for c in ("ANS-VALENCE", "ANS-AROUSAL"):
        P[c] = pd.to_numeric(P[c], errors="coerce")

    iaps = P["IAPS-ID"].astype(str)
    P = P[iaps.isin(["None", "nan", ""]) | P["IAPS-ID"].isna()]
    P = P[P["COND"].astype(str).str.lower() != "train"]
    return P.dropna(subset=["TIMESTAMP"])


def split_half(df, value, item, rater, rng, n_splits=N_SPLITS, min_items=20):
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
    return {"raw_mean": float(raw.mean()), "spearman_brown": float(sb.mean()),
            "p10": float(np.percentile(sb, 10)), "p90": float(np.percentile(sb, 90))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-subjects", type=int, default=None)
    args = ap.parse_args()

    T = load_trials()
    z = zipfile.ZipFile(BIO_ZIP)
    bio_files = {n.split("/")[-1].split("-")[0]: n
                 for n in z.namelist() if n.endswith("-BioSigs.csv")}

    subs = sorted(set(T["subject"]) & set(bio_files))
    if args.limit_subjects:
        subs = subs[: args.limit_subjects]

    with RunRecord(
        "biraffe_physiological_reliability", seed=SEED,
        params={"dataset": "BIRAFFE2", "licence": "CC BY 4.0",
                "n_subjects_available": len(subs),
                "features": FEATURES, "n_splits": N_SPLITS,
                "windows": {"baseline": "-2..0 s", "ecg": "0..6 s", "eda": "1..8 s"},
                "note": "先测信度再建模；PMEmo EDA 天花板为 0.155，本次为对照"},
    ) as run:
        rows, n_gap_skip = [], 0
        for k, s in enumerate(subs, 1):
            with z.open(bio_files[s]) as fh:
                B = pd.read_csv(fh)
            ts = B["TIMESTAMP"].to_numpy(dtype=float)
            ecg = B["ECG"].to_numpy(dtype=float)
            eda = B["EDA"].to_numpy(dtype=float)
            gaps = detect_gaps(ts)

            # 列名含连字符，itertuples 会重命名为位置属性 —— 用显式取值避免错位
            sub_t = T[T.subject == s][["TIMESTAMP", "IADS-ID",
                                       "ANS-AROUSAL", "ANS-VALENCE"]]
            for onset, iads, aro, val in sub_t.to_numpy():
                f = trial_features(ts, ecg, eda, float(onset), gaps)
                if not np.isfinite(list(f.values())).any():
                    n_gap_skip += 1
                rows.append({"subject": s, "iads": int(iads),
                             "arousal": aro, "valence": val, **f})
            if k % 10 == 0:
                print(f"  {k}/{len(subs)} 名受试", flush=True)

        D = pd.DataFrame(rows)
        # 列名回退：itertuples 的位置索引可能因列序变化而错位，做一次校验
        if D["arousal"].notna().mean() < 0.2:
            raise SystemExit("arousal 列映射异常，请检查 Procedure 列序")

        # 被试内 z 标准化
        for c in FEATURES:
            g = D.groupby("subject")[c]
            D[f"z_{c}"] = (D[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)

        D.to_parquet(run.artifact_path("trial_features.parquet"), index=False)
        run.log_metric("n_trials", int(len(D)))
        run.log_metric("n_trials_dropped_by_gaps", int(n_gap_skip))
        run.log_metric("feature_completeness",
                       {c: float(D[c].notna().mean()) for c in FEATURES})

        rng = np.random.default_rng(SEED)
        per_item = D.groupby("iads")["subject"].nunique()
        keep = per_item[per_item >= MIN_RATERS].index
        S = D[D.iads.isin(keep)]

        # ---------- 信度 ----------
        ceilings = {}
        sub_ceiling = split_half(S.dropna(subset=["arousal"]),
                                 "arousal", "iads", "subject", rng)
        ceilings["subjective_arousal"] = sub_ceiling
        for c in FEATURES:
            zc = f"z_{c}"
            ceilings[c] = split_half(S.dropna(subset=[zc]), zc, "iads", "subject", rng)
        run.log_metric("reliability", ceilings)

        # ---------- 主观 ↔ 生理 ----------
        agg = S.groupby("iads").agg(
            arousal=("arousal", "mean"),
            **{c: (f"z_{c}", "mean") for c in FEATURES}).dropna(subset=["arousal"])
        link = {}
        for c in FEATURES:
            v = agg[c]
            ok = v.notna()
            if ok.sum() > 20:
                r = float(spearmanr(agg.loc[ok, "arousal"], v[ok]).statistic)
                cs = ceilings.get(c) or {}
                cb = cs.get("spearman_brown")
                sb = (sub_ceiling or {}).get("spearman_brown")
                # 双向衰减校正上限 = sqrt(r_xx * r_yy)
                cap = float(np.sqrt(abs(cb * sb))) if cb and sb and cb * sb > 0 else None
                link[c] = {"rho": r, "max_possible": cap,
                           "pct_of_max": (r / cap) if cap else None}
        run.log_metric("subjective_vs_physiological", link)

        # ---------- 打印 ----------
        print("\n" + "=" * 76)
        print(f"run_id: {run.run_id}")
        print(f"受试 {len(subs)} 名   试次 {len(D):,}   "
              f"因掉线剔除 {n_gap_skip:,} ({n_gap_skip / max(len(D),1):.1%})")
        print("特征完整率: " + "  ".join(
            f"{c}={D[c].notna().mean():.0%}" for c in FEATURES))

        print("\n--- 信度（跨受试分半 + Spearman-Brown）---")
        print(f"{'量':<22}{'分半':>9}{'校正后':>10}{'[p10, p90]':>22}")
        for nm, c in ceilings.items():
            if c:
                print(f"{nm:<22}{c['raw_mean']:>+9.3f}{c['spearman_brown']:>+10.3f}"
                      f"   [{c['p10']:+.3f}, {c['p90']:+.3f}]")
            else:
                print(f"{nm:<22}{'不足':>9}")

        print("\n--- 横向对照 ---")
        ref = [("DEAM 主观 arousal", 0.868), ("BIRAFFE2 主观 arousal",
               (sub_ceiling or {}).get("spearman_brown", float("nan"))),
               ("ARAUS eventfulness", 0.566), ("PMEmo EDA（生理）", 0.155)]
        for nm, v in ref:
            print(f"  {nm:<26} {v:+.3f}")
        for c in FEATURES:
            cc = ceilings.get(c)
            if cc:
                print(f"  {'BIRAFFE2 ' + c:<26} {cc['spearman_brown']:+.3f}")

        print("\n--- 主观 arousal ↔ 生理（刺激层面）---")
        for c, v in link.items():
            cap = f"{v['max_possible']:.3f}" if v["max_possible"] else "—"
            pct = f"{v['pct_of_max']:.0%}" if v["pct_of_max"] else "—"
            print(f"  {c:<12} ρ = {v['rho']:+.3f}   理论上限 {cap}   占上限 {pct}")

        print("\n" + "=" * 76)
        best = max((ceilings[c]["spearman_brown"] for c in FEATURES
                    if ceilings.get(c)), default=float("nan"))
        if not np.isfinite(best):
            # 算不出信度 ≠ 信度低。试跑（6 名受试）时每个刺激只有约 2 名评分者，
            # 分半无法进行 —— 此时不得输出「不可用」的判定。
            print("→ 受试数不足，无法估计信度。本轮**不构成任何结论**，")
            print("  需在全部受试上重跑。")
        elif best >= 0.4:
            print(f"→ 生理信度最高 {best:.3f}：**可用**。")
            print("  首次具备支撑「声学 → 生理」的数据。")
        elif np.isfinite(best) and best >= 0.2:
            print(f"→ 生理信度最高 {best:.3f}：**勉强**，仅够做粗粒度对比。")
        else:
            print(f"→ 生理信度最高 {best:.3f}：**不可用**。")
            print("  与 PMEmo(0.155) 一致 —— 说明问题不在某一批数据，")
            print("  而在「用外周生理测单次声音刺激响应」这件事本身的信噪比。")
            print("  ⚠️ 这对依赖 HR/HRV 的设备端实时闭环是关键约束。")


if __name__ == "__main__":
    main()
