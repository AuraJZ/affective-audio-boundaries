"""类别层面 vs 逐刺激层面 —— 我的结论与文献是否真的矛盾？

    uv run python scripts/31_biraffe_category.py

## 起因

`step13` 报告「单次声音刺激的外周生理响应无可复现的刺激层面成分」。
用户指出这与主流声学/心理生理学不一致。**这个质疑是对的，必须查。**

## 查完之后我要修正自己的两处表述

**修正 1（对文献）**：我说过「IADS 本身就是用皮电验证过的」。
**不准确。** IADS 的常模是**主观 SAM 评分**的常模。
生理研究通常报告**类别层面**效应（不悦 vs 中性 vs 愉悦的 SCR 差异），
而非**逐刺激**的跨人信度。

**修正 2（对我的结论）**：这两个层面的要求相差很大：

| 层面 | 主张 | 文献支持 |
|---|---|---|
| **类别** | 「不悦的声音比中性的引起更强 SCR」 | 强 |
| **逐刺激** | 「第 290 号声音可靠地比第 291 号引起更强 SCR」 | 罕有报告 |

**两者可以同时成立**：类别有效应，而逐刺激无信度。
若如此，我的结论与文献不矛盾，但**必须重新表述**。

## 检验

BIRAFFE2 的 `COND` 列即为类别标签（`S-` / `S0` / `S+`）。
直接检验：三类之间的 EDA/HR 响应是否有差异。

同时用**更简单的基线**（刺激前 2 s 原始均值），
避免 `30_biraffe_sanity.py` 里「窗口起始再基线一次」带来的额外噪声 ——
试次间隔仅约 18 s，26 s 的窗口会与相邻试次重叠。
"""

from __future__ import annotations

import argparse
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from soundml.biosignals import detect_gaps, heart_rate
from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

ROOT = REPO_ROOT / "data" / "raw" / "BIRAFFE2"
BIO_ZIP = ROOT / "BIRAFFE2-biosigs.zip"
PROC = ROOT / "procedure"

BASE = (-2.0, 0.0)
RESP_EDA = (1.0, 7.0)
RESP_ECG = (0.0, 6.0)


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
    for c in ("TIMESTAMP", "ANS-AROUSAL", "ANS-VALENCE"):
        P[c] = pd.to_numeric(P[c], errors="coerce")
    iaps = P["IAPS-ID"].astype(str)
    P = P[iaps.isin(["None", "nan", ""]) | P["IAPS-ID"].isna()]
    P = P[P["COND"].astype(str).str.lower() != "train"]
    return P.dropna(subset=["TIMESTAMP"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=60)
    args = ap.parse_args()

    T = load_trials()
    print("COND 取值分布:")
    print(T["COND"].value_counts().to_string())

    z = zipfile.ZipFile(BIO_ZIP)
    bio = {n.split("/")[-1].split("-")[0]: n
           for n in z.namelist() if n.endswith("-BioSigs.csv")}
    subs = sorted(set(T.subject) & set(bio))[: args.n_subjects]

    with RunRecord(
        "biraffe_category_level", seed=SEED,
        params={"n_subjects": len(subs),
                "baseline_s": BASE, "eda_window_s": RESP_EDA, "ecg_window_s": RESP_ECG,
                "question": "类别层面是否有效应（与文献可比），逐刺激层面是否无信度"},
    ) as run:
        rows = []
        for k, s in enumerate(subs, 1):
            with z.open(bio[s]) as fh:
                B = pd.read_csv(fh)
            ts = B["TIMESTAMP"].to_numpy(dtype=float)
            ecg = B["ECG"].to_numpy(dtype=float)
            eda = B["EDA"].to_numpy(dtype=float)
            gaps = detect_gaps(ts)

            cols = ["TIMESTAMP", "COND", "IADS-ID", "ANS-AROUSAL"]
            for onset, cond, iads, aro in T[T.subject == s][cols].to_numpy():
                onset = float(onset)
                if gaps.size and np.any((gaps[:, 0] < onset + 8)
                                        & (gaps[:, 1] > onset - 5)):
                    continue
                mb = (ts >= onset + BASE[0]) & (ts < onset + BASE[1])
                me = (ts >= onset + RESP_EDA[0]) & (ts < onset + RESP_EDA[1])
                mc = (ts >= onset + RESP_ECG[0]) & (ts < onset + RESP_ECG[1])
                mcb = (ts >= onset - 5) & (ts < onset)
                if mb.sum() < 500 or me.sum() < 500:
                    continue
                eda_delta = float(np.mean(eda[me]) - np.mean(eda[mb]))
                hr_b = heart_rate(ecg[mcb]) if mcb.sum() > 2000 else np.nan
                hr_r = heart_rate(ecg[mc]) if mc.sum() > 2000 else np.nan
                rows.append({"subject": s, "cond": str(cond), "iads": int(iads),
                             "arousal": aro, "eda_delta": eda_delta,
                             "hr_delta": (hr_r - hr_b) if np.isfinite(hr_b)
                                         and np.isfinite(hr_r) else np.nan})
            if k % 20 == 0:
                print(f"  {k}/{len(subs)}", flush=True)

        D = pd.DataFrame(rows)
        # 被试内标准化：外周生理个体差异极大
        for c in ("eda_delta", "hr_delta"):
            g = D.groupby("subject")[c]
            D[f"z_{c}"] = (D[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        D.to_parquet(run.artifact_path("category_trials.parquet"), index=False)

        run.log_metric("n_trials", int(len(D)))
        run.log_metric("cond_counts", D["cond"].value_counts().to_dict())

        # ---------- 类别层面：被试内配对比较 ----------
        cat = {}
        for c in ("z_eda_delta", "z_hr_delta"):
            by = D.groupby(["subject", "cond"])[c].mean().unstack()
            keep = [x for x in by.columns if x in ("S-", "S0", "S+")]
            if len(keep) < 2:
                continue
            stats = {}
            for a in keep:
                for b in keep:
                    if a >= b:
                        continue
                    pair = by[[a, b]].dropna()
                    if len(pair) < 10:
                        continue
                    diff = pair[a] - pair[b]
                    w = wilcoxon(diff)
                    stats[f"{a} vs {b}"] = {
                        "n_subjects": int(len(pair)),
                        "mean_diff": float(diff.mean()),
                        "p": float(w.pvalue),
                        # 配对 Cohen's d
                        "d": float(diff.mean() / diff.std()) if diff.std() else np.nan}
            cat[c] = {"means": by[keep].mean().round(4).to_dict(), "contrasts": stats}
        run.log_metric("category_level", cat)

        # ---------- 逐刺激层面：跨受试信度（对照）----------
        rng = np.random.default_rng(SEED)
        per = D.groupby("iads")["subject"].nunique()
        S = D[D.iads.isin(per[per >= 4].index)]
        rel = {}
        for c in ("z_eda_delta", "z_hr_delta", "arousal"):
            raters = S["subject"].unique()
            raw = []
            for _ in range(200):
                perm = rng.permutation(raters)
                a = set(perm[: len(perm) // 2])
                ga = S[S.subject.isin(a)].groupby("iads")[c].mean()
                gb = S[~S.subject.isin(a)].groupby("iads")[c].mean()
                common = ga.index.intersection(gb.index)
                if len(common) >= 20:
                    r = spearmanr(ga.loc[common], gb.loc[common],
                                  nan_policy="omit").statistic
                    if np.isfinite(r):
                        raw.append(float(r))
            if raw:
                raw = np.array(raw)
                rel[c] = float(np.mean(2 * raw / (1 + raw)))
        run.log_metric("stimulus_level_reliability", rel)

        # ---------- 打印 ----------
        print("\n" + "=" * 76)
        print(f"run_id: {run.run_id}   试次 {len(D):,}")

        print("\n--- 1. 类别层面（被试内配对，z 标准化后）---")
        for c, v in cat.items():
            print(f"\n  {c}")
            print(f"    各类均值: {v['means']}")
            for name, st in v["contrasts"].items():
                sig = "***" if st["p"] < .001 else "**" if st["p"] < .01 \
                    else "*" if st["p"] < .05 else "n.s."
                print(f"    {name:<12} Δ={st['mean_diff']:+.3f}  d={st['d']:+.3f}  "
                      f"p={st['p']:.2e} {sig}  (n={st['n_subjects']})")

        print("\n--- 2. 逐刺激层面信度（对照）---")
        for c, v in rel.items():
            print(f"    {c:<16} {v:+.3f}")

        print("\n" + "=" * 76)
        any_sig = any(st["p"] < 0.05 for v in cat.values()
                      for st in v["contrasts"].values())
        if any_sig:
            print("→ **类别层面存在效应，逐刺激层面无信度。**")
            print("  两者可同时成立，与文献不矛盾 —— 但 step13 的表述必须收窄：")
            print("  不是「生理无响应」，而是「逐刺激的跨人信度为零」。")
        else:
            print("→ 类别层面也无效应。这与文献确有冲突，需进一步排查")
            print("  （EDA 单位/预处理、刺激时长、时间戳语义）。")


if __name__ == "__main__":
    main()
