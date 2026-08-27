"""阳性对照 + 滤波器修正 —— 先证明流水线能测出已知效应，再解释阴性结果。

    uv run python scripts/34_biraffe_positive_control.py --n-subjects 60

## 两个问题

**问题 1：高通滤波器可能在数值上是坏的。**

EDA 采样 1000 Hz，要求 0.05 Hz 高通 → 归一化截止 `0.05/500 = 1e-4`。
二阶 Butterworth 在如此极端的归一化频率下系数病态 —— 这是 DSP 的已知陷阱。
标准做法是**先降采样到 10–50 Hz 再滤波**（EDA 无 5 Hz 以上内容）。
`33_biraffe_corrected.py` 直接在 1000 Hz 上滤，输出可能是数值垃圾。

**问题 2：整个分析缺少阳性对照。**

到目前为止我只报告阴性结果，却从未验证流水线**能否测出任何已知效应**。
这是顺序错误：**先证明能测出已知的，再解释测不出的。**

## 阳性对照：习惯化

SCR 幅度随试次推进系统性下降，是心理生理学最稳固的现象之一，效应大且普遍。

| 若习惯化 | 则 |
|---|---|
| **能测出** | 流水线有效，低刺激信度是实质结果 |
| **测不出** | 流水线坏了，此前所有阴性结论一律无效 |

## 第二个阳性对照：响应窗 vs 随机窗

同一段信号里，刺激后 1–7 s 的 SCR 幅度，应显著大于随机时刻的同长窗口。
这是最低限度的「信号锁时于刺激」检验。
"""

from __future__ import annotations

import argparse
import zipfile

import numpy as np
import pandas as pd
import scipy.signal
from scipy.stats import spearmanr, wilcoxon

from soundml.biosignals import detect_gaps
from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

ROOT = REPO_ROOT / "data" / "raw" / "BIRAFFE2"
BIO_ZIP = ROOT / "BIRAFFE2-biosigs.zip"
PROC = ROOT / "procedure"

TARGET_FS = 20.0      # 降采样目标：EDA 无 5 Hz 以上内容
HP_HZ = 0.05
BASE = (-2.0, 0.0)
RESP = (1.0, 7.0)


def phasic_downsampled(ts: np.ndarray, eda: np.ndarray, fs: float):
    """先降采样再高通 —— 避免 1000 Hz 下归一化截止 1e-4 的病态滤波。"""
    step = max(int(round(fs / TARGET_FS)), 1)
    t2, x2 = ts[::step], eda[::step]
    fs2 = fs / step
    try:
        b, a = scipy.signal.butter(2, HP_HZ / (fs2 / 2), btype="high")
        ph = scipy.signal.filtfilt(b, a, x2)
    except Exception:
        ph = x2 - np.mean(x2)
    return t2, ph, fs2


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
    return P.dropna(subset=["TIMESTAMP"]).sort_values(["subject", "TIMESTAMP"])


def scr(t: np.ndarray, ph: np.ndarray, onset: float) -> float:
    mb = (t >= onset + BASE[0]) & (t < onset + BASE[1])
    mr = (t >= onset + RESP[0]) & (t < onset + RESP[1])
    if mb.sum() < 5 or mr.sum() < 15:
        return np.nan
    return float(np.max(ph[mr]) - np.mean(ph[mb]))


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
        "biraffe_positive_control", seed=SEED,
        params={"n_subjects": len(subs), "target_fs": TARGET_FS, "hp_hz": HP_HZ,
                "fix": "先降采样再高通，避免 1e-4 归一化截止的病态滤波",
                "positive_controls": ["habituation across trials",
                                      "stimulus-locked vs random windows"]},
    ) as run:
        rng = np.random.default_rng(SEED)
        rows, filt_check = [], []

        for k, s in enumerate(subs, 1):
            with z.open(bio[s]) as fh:
                B = pd.read_csv(fh)
            ts = B["TIMESTAMP"].to_numpy(dtype=float)
            eda = B["EDA"].to_numpy(dtype=float)
            fs = 1.0 / np.median(np.diff(ts))

            # 两种滤波并列，检查旧法是否数值失效
            t2, ph2, fs2 = phasic_downsampled(ts, eda, fs)
            try:
                b, a = scipy.signal.butter(2, HP_HZ / (fs / 2), btype="high")
                ph_old = scipy.signal.filtfilt(b, a, eda)
                old_sd = float(np.std(ph_old))
            except Exception:
                old_sd = np.nan
            filt_check.append({"subject": s, "sd_downsampled": float(np.std(ph2)),
                               "sd_direct_1000hz": old_sd,
                               "raw_sd": float(np.std(eda))})

            gaps = detect_gaps(ts)
            sub_t = T[T.subject == s]
            valid_t = t2[(t2 > t2[0] + 30) & (t2 < t2[-1] - 30)]

            for i, (onset, iads, aro) in enumerate(
                    sub_t[["TIMESTAMP", "IADS-ID", "ANS-AROUSAL"]].to_numpy()):
                onset = float(onset)
                if gaps.size and np.any((gaps[:, 0] < onset + 8) & (gaps[:, 1] > onset - 3)):
                    continue
                a_scr = scr(t2, ph2, onset)
                # 随机窗对照：同一段信号里随机时刻
                r_scr = scr(t2, ph2, float(rng.choice(valid_t))) if valid_t.size else np.nan
                rows.append({"subject": s, "trial_index": i, "iads": int(iads),
                             "arousal": aro, "scr": a_scr, "scr_random": r_scr})
            if k % 20 == 0:
                print(f"  {k}/{len(subs)}", flush=True)

        D = pd.DataFrame(rows).dropna(subset=["scr"])
        F = pd.DataFrame(filt_check)
        run.log_metric("n_trials", int(len(D)))
        run.log_metric("filter_comparison",
                       {"sd_downsampled_median": float(F.sd_downsampled.median()),
                        "sd_direct_1000hz_median": float(F.sd_direct_1000hz.median()),
                        "raw_sd_median": float(F.raw_sd.median())})

        # ---------- 阳性对照 1：习惯化 ----------
        hab = []
        for s, g in D.groupby("subject"):
            if len(g) >= 20:
                r = spearmanr(g.trial_index, g.scr).statistic
                if np.isfinite(r):
                    hab.append(float(r))
        H = np.array(hab)
        h_stat = wilcoxon(H) if H.size > 10 else None
        run.log_metric("habituation",
                       {"n_subjects": int(H.size), "mean_rho": float(H.mean()),
                        "frac_negative": float((H < 0).mean()),
                        "p": float(h_stat.pvalue) if h_stat else None})

        # ---------- 阳性对照 2：刺激锁时 vs 随机窗 ----------
        paired = D.dropna(subset=["scr_random"]).groupby("subject")[["scr", "scr_random"]].mean()
        lock_stat = wilcoxon(paired.scr, paired.scr_random) if len(paired) > 10 else None
        run.log_metric("stimulus_locked_vs_random",
                       {"n_subjects": int(len(paired)),
                        "mean_stim": float(paired.scr.mean()),
                        "mean_random": float(paired.scr_random.mean()),
                        "p": float(lock_stat.pvalue) if lock_stat else None})

        # ---------- 逐刺激信度（修正滤波后）----------
        for c in ("scr",):
            g = D.groupby("subject")[c]
            D[f"z_{c}"] = (D[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        per = D.groupby("iads")["subject"].nunique()
        S = D[D.iads.isin(per[per >= 4].index)]
        raters = S.subject.unique()
        raw = []
        for _ in range(200):
            perm = rng.permutation(raters)
            a = set(perm[: len(perm) // 2])
            ga = S[S.subject.isin(a)].groupby("iads")["z_scr"].mean()
            gb = S[~S.subject.isin(a)].groupby("iads")["z_scr"].mean()
            common = ga.index.intersection(gb.index)
            if len(common) >= 20:
                r = spearmanr(ga.loc[common], gb.loc[common], nan_policy="omit").statistic
                if np.isfinite(r):
                    raw.append(float(r))
        rel = float(np.mean(2 * np.array(raw) / (1 + np.array(raw)))) if raw else np.nan
        run.log_metric("stimulus_reliability_corrected", rel)

        # ---------- 打印 ----------
        print("\n" + "=" * 76)
        print(f"run_id: {run.run_id}   试次 {len(D):,}   受试 {len(subs)}")

        print("\n--- 0. 滤波器对比（相位性成分的 SD）---")
        print(f"  原始 EDA              {F.raw_sd.median():.3e}")
        print(f"  降采样后高通(20 Hz)   {F.sd_downsampled.median():.3e}")
        print(f"  直接在 1000 Hz 高通   {F.sd_direct_1000hz.median():.3e}")

        print("\n--- 1. 阳性对照：习惯化（SCR 随试次下降）---")
        if h_stat:
            print(f"  {H.size} 名受试   ρ 均值 {H.mean():+.3f}   "
                  f"为负 {100*(H<0).mean():.0f}%   p={h_stat.pvalue:.2e}")

        print("\n--- 2. 阳性对照：刺激锁时 vs 随机窗 ---")
        if lock_stat:
            print(f"  刺激后 {paired.scr.mean():.3e}   "
                  f"随机窗 {paired.scr_random.mean():.3e}   "
                  f"p={lock_stat.pvalue:.2e}")

        print(f"\n--- 3. 逐刺激跨受试信度（修正滤波后）---")
        print(f"  {rel:+.3f}")

        print("\n" + "=" * 76)
        hab_ok = h_stat is not None and h_stat.pvalue < 0.05 and H.mean() < 0
        lock_ok = lock_stat is not None and lock_stat.pvalue < 0.05 \
            and paired.scr.mean() > paired.scr_random.mean()
        if hab_ok and lock_ok:
            print("✅ 两个阳性对照均通过 —— 流水线能测出已知效应。")
            print(f"   因此逐刺激信度 {rel:+.3f} 是**实质结果**，不是提取失败。")
        elif not hab_ok and not lock_ok:
            print("🔴 两个阳性对照**均未通过** —— 流水线测不出任何已知效应。")
            print("   此前所有生理阴性结论一律无效，需换用成熟工具"
                  "（cvxEDA / Ledalab / neurokit2）重做。")
        else:
            print("⚠️ 阳性对照部分通过，流水线可靠性存疑，暂不下结论。")
            print(f"   习惯化 {'通过' if hab_ok else '未通过'}；"
                  f"刺激锁时 {'通过' if lock_ok else '未通过'}")


if __name__ == "__main__":
    main()
