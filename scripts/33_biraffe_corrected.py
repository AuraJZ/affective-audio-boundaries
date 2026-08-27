"""BIRAFFE2 修正版 —— 加入标准相位性分解后重跑。

    uv run python scripts/33_biraffe_corrected.py --n-subjects 60

## 修正了什么

`29_biraffe_physio.py` / `step13` 的提取**缺少相位性分解**：
只在 6 s 窗内做线性去趋势，而 EDA 的主要方差是分钟尺度慢漂移，
6 s 窗扣不掉 —— 试次级响应被漂移淹没。

信号质量诊断（`32_biraffe_signal_quality.py`）证明问题在提取端：

| 检查 | 结果 |
|---|---|
| 相位性能量占比（0.05–1 Hz） | **0.286** —— 充足 |
| 任务段 − 基线段 EDA | **+4.3e-5，75% 受试为正** —— 分钟尺度效应存在，时钟对齐 |
| 试次级 6 s 响应 | 零 |

本脚本对**整段**信号先做 0.05 Hz 高通分离相位性成分，再测试次级 SCR。

## 三个层面一起报，避免再次以偏概全

1. 类别层面（S+ / S0 / S−），与文献可比
2. 个体内（自评 arousal ↔ 自身 SCR）
3. 逐刺激跨受试信度
"""

from __future__ import annotations

import argparse
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from soundml.biosignals import detect_gaps, phasic_component
from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

ROOT = REPO_ROOT / "data" / "raw" / "BIRAFFE2"
BIO_ZIP = ROOT / "BIRAFFE2-biosigs.zip"
PROC = ROOT / "procedure"
BASE = (-2.0, 0.0)
RESP = (1.0, 7.0)


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
    for c in ("TIMESTAMP", "ANS-AROUSAL"):
        P[c] = pd.to_numeric(P[c], errors="coerce")
    iaps = P["IAPS-ID"].astype(str)
    P = P[iaps.isin(["None", "nan", ""]) | P["IAPS-ID"].isna()]
    P = P[P["COND"].astype(str).str.lower() != "train"]
    return P.dropna(subset=["TIMESTAMP"])


def split_half(df, val, item, rater, rng, n=200):
    raters = df[rater].unique()
    if len(raters) < 4:
        return np.nan
    raw = []
    for _ in range(n):
        perm = rng.permutation(raters)
        a = set(perm[: len(perm) // 2])
        ga = df[df[rater].isin(a)].groupby(item)[val].mean()
        gb = df[~df[rater].isin(a)].groupby(item)[val].mean()
        common = ga.index.intersection(gb.index)
        if len(common) >= 20:
            r = spearmanr(ga.loc[common], gb.loc[common], nan_policy="omit").statistic
            if np.isfinite(r):
                raw.append(float(r))
    if not raw:
        return np.nan
    raw = np.array(raw)
    return float(np.mean(2 * raw / (1 + raw)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=60)
    args = ap.parse_args()

    T = load_trials()
    z = zipfile.ZipFile(BIO_ZIP)
    bio = {n.split("/")[-1].split("-")[0]: n
           for n in z.namelist() if n.endswith("-BioSigs.csv")}
    subs = sorted(set(T.subject) & set(bio))[: args.n_subjects]

    with RunRecord(
        "biraffe_corrected_phasic", seed=SEED,
        params={"n_subjects": len(subs), "highpass_hz": 0.05,
                "baseline_s": BASE, "response_s": RESP,
                "fix": "加入整段相位性分解；step13 缺此步"},
    ) as run:
        rows = []
        for k, s in enumerate(subs, 1):
            with z.open(bio[s]) as fh:
                B = pd.read_csv(fh)
            ts = B["TIMESTAMP"].to_numpy(dtype=float)
            eda = B["EDA"].to_numpy(dtype=float)
            fs = 1.0 / np.median(np.diff(ts))
            ph = phasic_component(eda, fs)          # ← 关键修正
            gaps = detect_gaps(ts)

            cols = ["TIMESTAMP", "COND", "IADS-ID", "ANS-AROUSAL"]
            for onset, cond, iads, aro in T[T.subject == s][cols].to_numpy():
                onset = float(onset)
                if gaps.size and np.any((gaps[:, 0] < onset + 8) & (gaps[:, 1] > onset - 3)):
                    continue
                mb = (ts >= onset + BASE[0]) & (ts < onset + BASE[1])
                mr = (ts >= onset + RESP[0]) & (ts < onset + RESP[1])
                if mb.sum() < 500 or mr.sum() < 500:
                    continue
                base = float(np.mean(ph[mb]))
                rows.append({
                    "subject": s, "cond": str(cond), "iads": int(iads), "arousal": aro,
                    # 标准 SCR 幅度：相位性成分在响应窗内的峰值减基线
                    "scr_amp": float(np.max(ph[mr]) - base),
                    "scr_mean": float(np.mean(ph[mr]) - base),
                })
            if k % 20 == 0:
                print(f"  {k}/{len(subs)}", flush=True)

        D = pd.DataFrame(rows)
        for c in ("scr_amp", "scr_mean"):
            g = D.groupby("subject")[c]
            D[f"z_{c}"] = (D[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        D.to_parquet(run.artifact_path("corrected_trials.parquet"), index=False)
        run.log_metric("n_trials", int(len(D)))

        # ---- 1. 类别层面 ----
        cat = {}
        by = D.groupby(["subject", "cond"])["z_scr_amp"].mean().unstack()
        keep = [c for c in ("S-", "S0", "S+") if c in by.columns]
        for i, a in enumerate(keep):
            for b in keep[i + 1:]:
                pair = by[[a, b]].dropna()
                if len(pair) >= 10:
                    d = pair[a] - pair[b]
                    w = wilcoxon(d)
                    cat[f"{a} vs {b}"] = {"n": int(len(pair)),
                                          "mean_diff": float(d.mean()),
                                          "d": float(d.mean() / d.std()) if d.std() else np.nan,
                                          "p": float(w.pvalue)}
        run.log_metric("category_level", cat)
        run.log_metric("category_means", by[keep].mean().round(4).to_dict())

        # ---- 2. 个体内 ----
        within = []
        for s, g in D.dropna(subset=["arousal", "scr_amp"]).groupby("subject"):
            if len(g) >= 15:
                r = spearmanr(g["arousal"], g["scr_amp"]).statistic
                if np.isfinite(r):
                    within.append(float(r))
        W = np.array(within)
        wstat = wilcoxon(W) if W.size > 10 else None
        run.log_metric("within_subject",
                       {"n": int(W.size), "mean": float(W.mean()) if W.size else None,
                        "frac_positive": float((W > 0).mean()) if W.size else None,
                        "p": float(wstat.pvalue) if wstat else None})

        # ---- 3. 逐刺激信度 ----
        rng = np.random.default_rng(SEED)
        per = D.groupby("iads")["subject"].nunique()
        S = D[D.iads.isin(per[per >= 4].index)]
        rel = {c: split_half(S.dropna(subset=[c]), c, "iads", "subject", rng)
               for c in ("z_scr_amp", "arousal")}
        run.log_metric("stimulus_level_reliability", rel)

        # ---- 打印 ----
        print("\n" + "=" * 76)
        print(f"run_id: {run.run_id}   试次 {len(D):,}   受试 {len(subs)}")

        print("\n--- 1. 类别层面（被试内配对，SCR 幅度）---")
        print(f"  各类均值: {by[keep].mean().round(4).to_dict()}")
        for name, st in cat.items():
            sig = "***" if st["p"] < .001 else "**" if st["p"] < .01 \
                else "*" if st["p"] < .05 else "n.s."
            print(f"  {name:<12} Δ={st['mean_diff']:+.3f}  d={st['d']:+.3f}  "
                  f"p={st['p']:.2e} {sig}  (n={st['n']})")

        print("\n--- 2. 个体内：自评 arousal ↔ 自身 SCR ---")
        if W.size:
            print(f"  {W.size} 名受试   ρ 均值 {W.mean():+.3f}   "
                  f"为正 {100*(W>0).mean():.0f}%   "
                  f"p={wstat.pvalue:.2e}" if wstat else "")

        print("\n--- 3. 逐刺激跨受试信度 ---")
        for c, v in rel.items():
            print(f"  {c:<14} {v:+.3f}" if np.isfinite(v) else f"  {c:<14} 不足")

        print("\n" + "=" * 76)
        print("三个层面必须一起看，任一层面单独下结论都会以偏概全。")


if __name__ == "__main__":
    main()
