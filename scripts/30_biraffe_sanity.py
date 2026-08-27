"""BIRAFFE2 对齐与效应的 sanity check —— 我的零信度结论可能是错的。

    uv run python scripts/30_biraffe_sanity.py --n-subjects 30

## 为什么要做这个

`step13` 报告「单次声音刺激的外周生理响应无可复现的刺激层面成分」。

**这与主流心理生理学不一致。** IADS 刺激库本身就是用皮电验证过的：
其常模 arousal 与 SCR 的对应是数十年的基础结果。
若我的分析说 IADS 声音的皮电跨人完全不一致，**更可能是我错了**。

## 最可能的错处

`29_biraffe_physio.py` **假设 `TIMESTAMP` 是刺激起始时刻，且从未验证**。

Procedure 表里另有 `ANS-TIME` 列（3.66、5.66、4.20…），形似反应时。
若 `TIMESTAMP` 记的是**受试作答时刻**，则：

    我用的基线窗 −2…0 s   实际落在刺激中间
    我用的响应窗  1…8 s   实际落在试次结束之后

对齐一错，测出来必然是噪声 —— 而且看起来和「真的没有响应」一模一样。

## 三个诊断

1. **事件相关总平均**：把所有试次按 TIMESTAMP 对齐后平均 EDA。
   对齐正确 → 皮电在 0 之后 2–5 s 上升；
   TIMESTAMP 是作答时刻 → 上升出现在 **0 之前**。
2. **个体内效应**：同一人打分高的声音，其皮电是否也更高。
   个体内有效而跨人无效 → 结论完全不同（是个体差异，不是无信号）。
3. **信号本身有无动态**：EDA 的变异幅度是否远大于量化噪声。
"""

from __future__ import annotations

import argparse
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from soundml.biosignals import detect_gaps
from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

ROOT = REPO_ROOT / "data" / "raw" / "BIRAFFE2"
BIO_ZIP = ROOT / "BIRAFFE2-biosigs.zip"
PROC = ROOT / "procedure"

PRE_S, POST_S = 10.0, 16.0     # 总平均窗口：TIMESTAMP 前后
GRID = np.arange(-PRE_S, POST_S, 0.1)


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
    for c in ("TIMESTAMP", "ANS-VALENCE", "ANS-AROUSAL", "ANS-TIME"):
        P[c] = pd.to_numeric(P[c], errors="coerce")
    iaps = P["IAPS-ID"].astype(str)
    P = P[iaps.isin(["None", "nan", ""]) | P["IAPS-ID"].isna()]
    P = P[P["COND"].astype(str).str.lower() != "train"]
    return P.dropna(subset=["TIMESTAMP"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=30)
    args = ap.parse_args()

    T = load_trials()
    z = zipfile.ZipFile(BIO_ZIP)
    bio = {n.split("/")[-1].split("-")[0]: n
           for n in z.namelist() if n.endswith("-BioSigs.csv")}
    subs = sorted(set(T.subject) & set(bio))[: args.n_subjects]

    with RunRecord(
        "biraffe_sanity_check", seed=SEED,
        params={"n_subjects": len(subs), "window": [-PRE_S, POST_S],
                "question": "TIMESTAMP 是刺激起始还是作答时刻？个体内是否有效应？"},
    ) as run:
        traces, within, dyn = [], [], []
        for k, s in enumerate(subs, 1):
            with z.open(bio[s]) as fh:
                B = pd.read_csv(fh)
            ts = B["TIMESTAMP"].to_numpy(dtype=float)
            eda = B["EDA"].to_numpy(dtype=float)
            gaps = detect_gaps(ts)

            # 信号动态范围 vs 量化步长
            d = np.diff(np.unique(eda))
            dyn.append({"subject": s, "eda_sd": float(np.std(eda)),
                        "eda_range": float(np.ptp(eda)),
                        "min_step": float(np.min(d)) if d.size else np.nan})

            sub_t = T[T.subject == s]
            vals, aros = [], []
            for onset, aro in sub_t[["TIMESTAMP", "ANS-AROUSAL"]].to_numpy():
                onset = float(onset)
                if gaps.size and np.any((gaps[:, 0] < onset + POST_S)
                                        & (gaps[:, 1] > onset - PRE_S)):
                    continue
                m = (ts >= onset - PRE_S) & (ts < onset + POST_S)
                if m.sum() < 100:
                    continue
                tt, xx = ts[m] - onset, eda[m]
                # 以窗口前 2 s 为基线做归零，便于跨试次平均
                base = xx[(tt >= -PRE_S) & (tt < -PRE_S + 2)]
                if base.size < 50:
                    continue
                curve = np.interp(GRID, tt, xx - float(np.mean(base)),
                                  left=np.nan, right=np.nan)
                traces.append(curve)
                # 个体内：以 2–7 s 相对 −2–0 s 的抬升作为响应
                pre = np.nanmean(curve[(GRID >= -2) & (GRID < 0)])
                post = np.nanmean(curve[(GRID >= 2) & (GRID < 7)])
                if np.isfinite(pre) and np.isfinite(post) and np.isfinite(aro):
                    vals.append(post - pre)
                    aros.append(float(aro))
            if len(vals) >= 15:
                r = spearmanr(vals, aros).statistic
                if np.isfinite(r):
                    within.append({"subject": s, "n": len(vals), "rho": float(r)})
            if k % 10 == 0:
                print(f"  {k}/{len(subs)}", flush=True)

        A = np.vstack(traces)
        grand = np.nanmean(A, axis=0)
        sem = np.nanstd(A, axis=0) / np.sqrt(np.sum(~np.isnan(A), axis=0))
        peak_i = int(np.nanargmax(grand))
        trough_i = int(np.nanargmin(grand))

        W = pd.DataFrame(within)
        D = pd.DataFrame(dyn)

        run.log_metric("n_trials_averaged", int(A.shape[0]))
        run.log_metric("grand_average_peak_time_s", float(GRID[peak_i]))
        run.log_metric("grand_average_peak_value", float(grand[peak_i]))
        run.log_metric("grand_average_trough_time_s", float(GRID[trough_i]))
        run.log_metric("signal_dynamics",
                       {"eda_sd_median": float(D.eda_sd.median()),
                        "eda_range_median": float(D.eda_range.median()),
                        "min_step_median": float(D.min_step.median())})
        if len(W):
            run.log_metric("within_subject_rho",
                           {"mean": float(W.rho.mean()), "median": float(W.rho.median()),
                            "frac_positive": float((W.rho > 0).mean()),
                            "n_subjects": int(len(W))})

        pd.DataFrame({"t": GRID, "grand": grand, "sem": sem}).to_csv(
            run.artifact_path("grand_average.csv"), index=False, encoding="utf-8")

        # ---------- 打印 ----------
        print("\n" + "=" * 74)
        print(f"run_id: {run.run_id}")
        print(f"受试 {len(subs)}   平均了 {A.shape[0]:,} 个试次\n")

        print("--- 1. 事件相关总平均（EDA，相对窗口起始基线）---")
        print(f"{'时间(s)':>9}{'均值':>12}{'±SEM':>10}")
        for t in (-8, -6, -4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8, 10, 12):
            i = int(np.argmin(np.abs(GRID - t)))
            mark = ""
            if i == peak_i:
                mark = "  ← 峰"
            elif i == trough_i:
                mark = "  ← 谷"
            print(f"{t:>9}{grand[i]:>12.3e}{sem[i]:>10.1e}{mark}")
        print(f"\n  峰出现在 t = {GRID[peak_i]:+.1f} s")

        print("\n--- 2. 个体内：自评 arousal ↔ 自身皮电抬升 ---")
        if len(W):
            print(f"  {len(W)} 名受试   ρ 均值 {W.rho.mean():+.3f}   "
                  f"中位 {W.rho.median():+.3f}   为正 {100*(W.rho>0).mean():.0f}%")
        else:
            print("  可用受试不足")

        print("\n--- 3. 信号动态 ---")
        print(f"  EDA 标准差中位 {D.eda_sd.median():.3e}   "
              f"极差中位 {D.eda_range.median():.3e}   "
              f"最小量化步长中位 {D.min_step.median():.3e}")

        print("\n" + "=" * 74)
        pk = GRID[peak_i]
        if pk < -1:
            print(f"🔴 总平均的峰在 t = {pk:+.1f} s，**出现在 TIMESTAMP 之前**。")
            print("   → TIMESTAMP 很可能是**作答时刻**而非刺激起始。")
            print("   → step13 的窗口对齐是错的，其零信度结论必须作废重算。")
        elif 1 <= pk <= 8:
            print(f"✅ 总平均的峰在 t = {pk:+.1f} s，符合 SCR 潜伏期。")
            print("   → 对齐正确，step13 的结论不是对齐造成的。")
        else:
            print(f"⚠️ 总平均的峰在 t = {pk:+.1f} s，不在典型 SCR 窗口内。")
            print("   → 对齐存疑，需进一步核对 BIRAFFE2 的时间戳定义。")


if __name__ == "__main__":
    main()
