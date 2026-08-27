"""BIRAFFE2 重查：逐个体阳性对照 —— 之前的判决也是组水平的。

    .venv/Scripts/python scripts/54_biraffe_per_subject.py --n-subjects 102

## 为什么重查

`step14` 撤回全部 BIRAFFE2 结论，依据是两项阳性对照失败：

| 对照 | 结果 |
|---|---|
| 习惯化（SCR 应随试次下降） | ρ 均值 **+0.113**，p=0.002 —— 方向相反 |
| 刺激锁时（刺激后 > 随机窗） | p=0.267 —— 无差别 |

**但这两个判决都是组水平的。** 「ρ 均值为正」与「多数个体为正」不是一回事：
若一部分个体的数据是好的（习惯化为负）、另一部分是坏的（设备问题、电极脱落），
组平均可以被坏的那部分拖成正值，而好的那部分仍然可用。

BIRAFFE2 的数据集论文本身记录了大量设备错误，这个情形是有先验的。

## 本脚本做什么

对每一名受试**单独**跑两项阳性对照，并给出逐个体判据：

1. **习惯化**：该个体的 SCR 与试次序号的 Spearman ρ < 0，且置换 p < 0.05
2. **刺激锁时**：该个体刺激后窗的 SCR > 其随机窗（配对检验）

**只有两项都过的个体，其数据才进入后续分析。**

若存在这样一批个体，则 BIRAFFE2 部分可用，`step14` 的撤回范围需要收窄
（从「全部撤回」改为「组水平撤回，个体子集可用」）。
若一个都没有，则撤回维持原样，且这次是逐个体验证过的。

## 提取

沿用 `35_biraffe_neurokit.py` 的流水线（neurokit2，经文献验证），
参数完全不变 —— 唯一的差别是判决的层次。
"""

from __future__ import annotations

import argparse
import warnings
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from soundml.biosignals import detect_gaps
from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

warnings.filterwarnings("ignore")

ROOT = REPO_ROOT / "data" / "raw" / "BIRAFFE2"
BIO_ZIP = ROOT / "BIRAFFE2-biosigs.zip"
PROC = ROOT / "procedure"
TARGET_FS = 20.0
BASE, RESP = (-2.0, 0.0), (1.0, 7.0)
MIN_TRIALS = 20
N_PERM = 500


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=102)
    args = ap.parse_args()

    import importlib
    import neurokit2 as nk
    m35 = importlib.import_module("35_biraffe_neurokit")

    T = m35.load_trials()
    z = zipfile.ZipFile(BIO_ZIP)
    bio = {n.split("/")[-1].split("-")[0]: n
           for n in z.namelist() if n.endswith("-BioSigs.csv")}
    subs = sorted(set(T.subject) & set(bio))[: args.n_subjects]

    with RunRecord(
        "biraffe_per_subject_controls", seed=SEED,
        params={"n_subjects": len(subs), "tool": "neurokit2",
                "target_fs": TARGET_FS, "n_perm": N_PERM,
                "change_vs_step14": "判决从组水平下沉到逐个体",
                "hypothesis": "设备问题可能只影响一部分个体，"
                              "组平均被坏数据拖偏，好数据仍可用"},
    ) as run:
        rng = np.random.default_rng(SEED)
        rows = []

        for k, s in enumerate(subs, 1):
            with z.open(bio[s]) as fh:
                B = pd.read_csv(fh)
            ts = B["TIMESTAMP"].to_numpy(dtype=float)
            eda = B["EDA"].to_numpy(dtype=float)
            fs = 1.0 / np.median(np.diff(ts))
            step = max(int(round(fs / TARGET_FS)), 1)
            t2, x2, fs2 = ts[::step], eda[::step], fs / step
            try:
                sig, _ = nk.eda_process(x2, sampling_rate=fs2)
                phasic = sig["EDA_Phasic"].to_numpy()
            except Exception:                                # noqa: BLE001
                rows.append({"subject": s, "status": "eda_process_failed"})
                continue

            gaps = detect_gaps(ts)
            valid = t2[(t2 > t2[0] + 30) & (t2 < t2[-1] - 30)]

            def amp(onset: float) -> float:
                mb = (t2 >= onset + BASE[0]) & (t2 < onset + BASE[1])
                mr = (t2 >= onset + RESP[0]) & (t2 < onset + RESP[1])
                if mb.sum() < 5 or mr.sum() < 15:
                    return np.nan
                return float(np.max(phasic[mr]) - np.mean(phasic[mb]))

            sub_t = T[T.subject == s]
            scr, rnd, idx = [], [], []
            for i, onset in enumerate(sub_t["TIMESTAMP"].to_numpy(dtype=float)):
                if gaps.size and np.any((gaps[:, 0] < onset + 8)
                                        & (gaps[:, 1] > onset - 3)):
                    continue
                a = amp(onset)
                if np.isfinite(a):
                    scr.append(a); idx.append(i)
                    rnd.append(amp(float(rng.choice(valid))) if valid.size else np.nan)
            scr, idx = np.asarray(scr), np.asarray(idx)
            rnd = np.asarray(rnd, dtype=float)

            if len(scr) < MIN_TRIALS:
                rows.append({"subject": s, "status": "too_few_trials",
                             "n_trials": len(scr)})
                continue

            # --- 对照 1：习惯化（该个体自己的置换零分布）---
            rho = spearmanr(idx, scr).statistic
            null = np.array([spearmanr(rng.permutation(idx), scr).statistic
                             for _ in range(N_PERM)])
            p_hab = float((np.sum(null <= rho) + 1) / (N_PERM + 1))   # 单尾：应为负

            # --- 对照 2：刺激锁时 ---
            ok = np.isfinite(rnd)
            p_lock = (float(wilcoxon(scr[ok], rnd[ok], alternative="greater").pvalue)
                      if ok.sum() >= MIN_TRIALS else np.nan)

            rows.append({"subject": s, "status": "ok", "n_trials": len(scr),
                         "hab_rho": float(rho), "hab_p": p_hab,
                         "lock_p": p_lock,
                         "mean_scr": float(np.mean(scr)),
                         "mean_random": float(np.nanmean(rnd)),
                         "hab_pass": bool(rho < 0 and p_hab < 0.05),
                         "lock_pass": bool(np.isfinite(p_lock) and p_lock < 0.05)})
            if k % 20 == 0:
                print(f"  {k}/{len(subs)}", flush=True)

        R = pd.DataFrame(rows)
        ok = R[R.status == "ok"].copy()
        ok["both_pass"] = ok.hab_pass & ok.lock_pass

        n = len(ok)
        n_hab = int(ok.hab_pass.sum())
        n_lock = int(ok.lock_pass.sum())
        n_both = int(ok.both_pass.sum())
        exp_both = n * 0.05 * 0.05

        run.log_metric("n_subjects_processed", int(len(R)))
        run.log_metric("n_usable_signal", n)
        run.log_metric("controls", {
            "habituation_pass": n_hab, "expected_by_chance": n * 0.05,
            "stimulus_locking_pass": n_lock,
            "both_pass": n_both, "both_expected_by_chance": exp_both,
            "frac_hab_rho_negative": float((ok.hab_rho < 0).mean())})

        print("\n" + "=" * 84)
        print(f"run_id: {run.run_id}   {len(R)} 名受试，可分析 {n}")
        print("\n--- 逐个体阳性对照 ---")
        print(f"  习惯化通过（ρ<0 且 p<0.05）   **{n_hab}**/{n}"
              f"   随机期望 {n*0.05:.1f}")
        print(f"    （其中 ρ 为负的个体占比 {100*(ok.hab_rho<0).mean():.0f}%，"
              f"应远超 50%）")
        print(f"  刺激锁时通过（p<0.05）        **{n_lock}**/{n}"
              f"   随机期望 {n*0.05:.1f}")
        print(f"  **两项都过**                  **{n_both}**/{n}"
              f"   随机期望 {exp_both:.2f}")

        if n_both:
            print("\n--- 两项都过的个体 ---")
            for r_ in ok[ok.both_pass].itertuples():
                print(f"  {r_.subject}  试次 {r_.n_trials:3d}  "
                      f"习惯化 ρ={r_.hab_rho:+.3f} p={r_.hab_p:.4f}   "
                      f"锁时 p={r_.lock_p:.4f}")

        R.to_csv(REPO_ROOT / "reports" / "source_data" /
                 "biraffe_per_subject_controls.csv", index=False)

        print("\n" + "=" * 84)
        if n_both > max(exp_both * 3, 3):
            print(f"→ **{n_both} 名个体两项对照都通过**（随机期望 {exp_both:.2f}）。")
            print("  BIRAFFE2 并非整体不可用，而是**存在一个可用的个体子集**。")
            print("  step14 的撤回范围应收窄为「组水平撤回，该子集可用」。")
        else:
            print(f"→ 两项都过的个体 {n_both} 名（随机期望 {exp_both:.2f}）—— "
                  "不构成可用子集。")
            print("  step14 的全撤维持原样，且这次是**逐个体**验证过的，")
            print("  不再存在「组平均被坏数据拖偏」这一未排除的可能。")


if __name__ == "__main__":
    main()
