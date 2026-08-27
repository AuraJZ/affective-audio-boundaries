"""BIRAFFE2 信号质量诊断 —— 问题在数据还是在我。

    uv run python scripts/32_biraffe_signal_quality.py

## 状态

`step13` 报告生理无响应。用户指出与文献冲突。追查后发现：
**类别层面也无效应**（S+/S0/S− 两两 p > 0.24），
而同一批试次的主观评分逐刺激信度为 0.741。

IADS 的类别层面 SCR 效应是稳固的基础结果。因此**问题极可能在提取端**。

## 两个决定性检查

**1. EDA 是否含相位性成分。**
观察到相邻样本仅差约 1.4e-10，而整段 SD 为 2.6e-5 ——
8 s 窗口内的变化比整段变异小两个数量级。
若该通道被重度低通、或本身是慢变的紧张性量，**SCR 根本不在数据里**，
任何试次锁时分析都必然为零。

判据：计算 EDA 的功率谱。SCR 的能量主要在 0.05–1 Hz。
若该频段能量占比极低，则相位性成分缺失。

**2. 两个文件的时钟是否对齐。**
Procedure 与 BioSigs 各有绝对时间戳。若来自不同时钟（不同机器/时区），
试次会落在随机位置 —— 这会产生与「真的无响应」完全一样的结果。

判据：用 Procedure 里的 `EVENT` 标记（BASELINE START/END、
STIMULI PART 2 START/END）对齐 —— 基线段与任务段的 EDA 水平应有系统差异。
若无差异，时钟对齐存疑。
"""

from __future__ import annotations

import argparse
import zipfile

import numpy as np
import pandas as pd
import scipy.signal

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

ROOT = REPO_ROOT / "data" / "raw" / "BIRAFFE2"
BIO_ZIP = ROOT / "BIRAFFE2-biosigs.zip"
PROC = ROOT / "procedure"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=12)
    args = ap.parse_args()

    z = zipfile.ZipFile(BIO_ZIP)
    bio = {n.split("/")[-1].split("-")[0]: n
           for n in z.namelist() if n.endswith("-BioSigs.csv")}
    procs = {f.stem.split("-")[0]: f for f in PROC.rglob("SUB*-Procedure.csv")}
    subs = sorted(set(bio) & set(procs))[: args.n_subjects]

    with RunRecord(
        "biraffe_signal_quality", seed=SEED,
        params={"n_subjects": len(subs),
                "checks": ["EDA phasic content (PSD)", "clock alignment via EVENT"]},
    ) as run:
        psd_rows, clock_rows = [], []
        for s in subs:
            with z.open(bio[s]) as fh:
                B = pd.read_csv(fh)
            ts = B["TIMESTAMP"].to_numpy(dtype=float)
            eda = B["EDA"].to_numpy(dtype=float)
            ecg = B["ECG"].to_numpy(dtype=float)

            fs = 1.0 / np.median(np.diff(ts))

            # ---- 1. EDA 功率谱：相位性能量占比 ----
            x = eda - np.mean(eda)
            f, P = scipy.signal.welch(x, fs=fs, nperseg=int(min(60 * fs, len(x) // 4)))
            tot = np.trapezoid(P, f)
            def band(lo, hi):
                m = (f >= lo) & (f < hi)
                return float(np.trapezoid(P[m], f[m]) / tot) if tot > 0 and m.any() else np.nan
            psd_rows.append({
                "subject": s, "fs": float(fs),
                "tonic_0_005": band(0.0, 0.05),      # 紧张性
                "phasic_005_1": band(0.05, 1.0),     # 相位性 SCR
                "hf_1_10": band(1.0, 10.0),
                "eda_sd": float(np.std(eda)),
                # 8 s 窗内的典型变化幅度
                "eda_8s_range": float(np.median([
                    np.ptp(eda[i:i + int(8 * fs)])
                    for i in range(0, max(len(eda) - int(8 * fs), 1), int(60 * fs))][:50] or [np.nan])),
            })

            # ---- 2. 时钟对齐：EVENT 标记 ----
            P2 = pd.read_csv(procs[s], sep=";", low_memory=False)
            P2.columns = [c.strip() for c in P2.columns]
            P2["TIMESTAMP"] = pd.to_numeric(P2["TIMESTAMP"], errors="coerce")
            ev = P2.dropna(subset=["TIMESTAMP"])
            def ev_time(key):
                m = ev["EVENT"].astype(str).str.contains(key, case=False, na=False)
                return float(ev.loc[m, "TIMESTAMP"].iloc[0]) if m.any() else np.nan

            b0, b1 = ev_time("BASELINE START"), ev_time("BASELINE END")
            row = {"subject": s,
                   "proc_span": [float(ev.TIMESTAMP.min()), float(ev.TIMESTAMP.max())],
                   "bio_span": [float(ts.min()), float(ts.max())],
                   "proc_inside_bio": bool(ev.TIMESTAMP.min() >= ts.min() - 60
                                           and ev.TIMESTAMP.max() <= ts.max() + 60)}
            if np.isfinite(b0) and np.isfinite(b1) and b1 > b0:
                mb = (ts >= b0) & (ts < b1)
                mt = (ts >= b1 + 60) & (ts < b1 + 600)     # 任务段
                if mb.sum() > 1000 and mt.sum() > 1000:
                    row["eda_baseline"] = float(np.mean(eda[mb]))
                    row["eda_task"] = float(np.mean(eda[mt]))
                    row["eda_task_minus_baseline"] = row["eda_task"] - row["eda_baseline"]
                    row["ecg_sd_baseline"] = float(np.std(ecg[mb]))
                    row["ecg_sd_task"] = float(np.std(ecg[mt]))
            clock_rows.append(row)

        Q = pd.DataFrame(psd_rows)
        C = pd.DataFrame(clock_rows)
        run.log_metric("eda_psd", Q.round(6).to_dict("records"))
        run.log_metric("clock", C.to_dict("records"))

        print("=" * 76)
        print(f"run_id: {run.run_id}   受试 {len(subs)}")

        print("\n--- 1. EDA 功率谱：各频段能量占比 ---")
        print(f"  采样率中位          {Q.fs.median():.1f} Hz")
        print(f"  紧张性 0–0.05 Hz    {Q.tonic_0_005.median():.4f}")
        print(f"  **相位性 0.05–1 Hz  {Q.phasic_005_1.median():.4f}**")
        print(f"  高频 1–10 Hz        {Q.hf_1_10.median():.4f}")
        print(f"  整段 SD             {Q.eda_sd.median():.3e}")
        print(f"  8 s 窗内典型极差     {Q.eda_8s_range.median():.3e}  "
              f"（占整段 SD 的 {Q.eda_8s_range.median()/Q.eda_sd.median():.1%}）")

        print("\n--- 2. 时钟对齐 ---")
        print(f"  Procedure 时间落在 BioSigs 范围内: "
              f"{C.proc_inside_bio.sum()}/{len(C)}")
        if "eda_task_minus_baseline" in C:
            v = C["eda_task_minus_baseline"].dropna()
            print(f"  任务段 − 基线段 EDA 差（n={len(v)}）: "
                  f"均值 {v.mean():+.3e}   为正 {100*(v>0).mean():.0f}%")

        print("\n" + "=" * 76)
        ph = Q.phasic_005_1.median()
        ratio = Q.eda_8s_range.median() / Q.eda_sd.median()
        if ph < 0.02 or ratio < 0.05:
            print("🔴 **EDA 缺乏相位性成分。**")
            print(f"   0.05–1 Hz 能量占比仅 {ph:.4f}，8 s 窗内变化仅为整段 SD 的 {ratio:.1%}。")
            print("   → SCR 不在数据里，任何试次锁时分析必然为零。")
            print("   → **step13 的结论必须作废**：那不是生理无响应，")
            print("     而是这一路 EDA 数据不含所需成分。")
        else:
            print(f"✅ EDA 含相位性成分（{ph:.4f}），信号本身可用。")
            print("   零响应的原因需继续排查（时钟、单位、刺激时长）。")


if __name__ == "__main__":
    main()
