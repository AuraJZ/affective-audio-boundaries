"""用经过验证的 neurokit2 重做 —— 手写流水线已两次失败。

    uv run python scripts/35_biraffe_neurokit.py --n-subjects 60

## 为什么换工具

我手写的 EDA 处理**两次都失败**，且第二次的阳性对照结果是决定性的：

| 阳性对照 | 结果 |
|---|---|
| 习惯化（SCR 应随试次下降） | ρ = **+0.113**，p=0.002 —— **方向反了** |
| 刺激锁时 vs 随机窗 | p = 0.267 —— **无差别** |

「测到一个不该存在的效应」+「测不到该存在的效应」= 提取本身是坏的。
此前所有生理阴性结论（含刚修正的 0.110）**一律无效**。

滤波器数值问题也被排除：降采样后 8.3e-7 vs 直接滤 7.8e-7，几乎一致。

**因此不再手写第三版**，改用 `neurokit2` ——
其 EDA 分解与 SCR 检测经过文献验证。

## 判据不变

**先过阳性对照，再看任何阴性结果。**

1. 习惯化：SCR 幅度随试次下降
2. 刺激锁时：刺激后窗口的 SCR 大于随机窗口

两个都过，才允许解读逐刺激信度。
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
    return P.dropna(subset=["TIMESTAMP"]).sort_values(["subject", "TIMESTAMP"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=60)
    args = ap.parse_args()

    import neurokit2 as nk

    T = load_trials()
    z = zipfile.ZipFile(BIO_ZIP)
    bio = {n.split("/")[-1].split("-")[0]: n
           for n in z.namelist() if n.endswith("-BioSigs.csv")}
    subs = sorted(set(T.subject) & set(bio))[: args.n_subjects]

    with RunRecord(
        "biraffe_neurokit", seed=SEED,
        params={"n_subjects": len(subs), "tool": "neurokit2 0.2.12",
                "target_fs": TARGET_FS, "baseline_s": BASE, "response_s": RESP,
                "reason": "手写流水线两次未通过阳性对照"},
    ) as run:
        rng = np.random.default_rng(SEED)
        rows, failed = [], 0

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
            except Exception:
                failed += 1
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
            for i, (onset, cond, iads, aro) in enumerate(
                    sub_t[["TIMESTAMP", "COND", "IADS-ID", "ANS-AROUSAL"]].to_numpy()):
                onset = float(onset)
                if gaps.size and np.any((gaps[:, 0] < onset + 8) & (gaps[:, 1] > onset - 3)):
                    continue
                rows.append({"subject": s, "trial_index": i, "cond": str(cond),
                             "iads": int(iads), "arousal": aro,
                             "scr": amp(onset),
                             "scr_random": amp(float(rng.choice(valid))) if valid.size else np.nan})
            if k % 20 == 0:
                print(f"  {k}/{len(subs)}", flush=True)

        D = pd.DataFrame(rows).dropna(subset=["scr"])
        run.log_metric("n_trials", int(len(D)))
        run.log_metric("n_subjects_failed", int(failed))

        # ---------- 阳性对照 1：习惯化 ----------
        hab = [spearmanr(g.trial_index, g.scr).statistic
               for _, g in D.groupby("subject") if len(g) >= 20]
        H = np.array([h for h in hab if np.isfinite(h)])
        h_stat = wilcoxon(H) if H.size > 10 else None
        run.log_metric("habituation",
                       {"n": int(H.size), "mean_rho": float(H.mean()),
                        "frac_negative": float((H < 0).mean()),
                        "p": float(h_stat.pvalue) if h_stat else None})

        # ---------- 阳性对照 2：刺激锁时 ----------
        pr = D.dropna(subset=["scr_random"]).groupby("subject")[["scr", "scr_random"]].mean()
        l_stat = wilcoxon(pr.scr, pr.scr_random) if len(pr) > 10 else None
        run.log_metric("stimulus_locked",
                       {"n": int(len(pr)), "mean_stim": float(pr.scr.mean()),
                        "mean_random": float(pr.scr_random.mean()),
                        "p": float(l_stat.pvalue) if l_stat else None})

        # ---------- 逐刺激信度 ----------
        g = D.groupby("subject")["scr"]
        D["z_scr"] = (D.scr - g.transform("mean")) / g.transform("std").replace(0, np.nan)
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
        run.log_metric("stimulus_reliability", rel)

        # ---------- 打印 ----------
        print("\n" + "=" * 76)
        print(f"run_id: {run.run_id}   试次 {len(D):,}   "
              f"受试 {len(subs)}（处理失败 {failed}）")

        print("\n--- 阳性对照 1：习惯化（应为负）---")
        if h_stat:
            print(f"  ρ 均值 {H.mean():+.3f}   为负 {100*(H<0).mean():.0f}%   "
                  f"p={h_stat.pvalue:.2e}")

        print("\n--- 阳性对照 2：刺激锁时 vs 随机窗（刺激应更大）---")
        if l_stat:
            print(f"  刺激后 {pr.scr.mean():.4f}   随机 {pr.scr_random.mean():.4f}   "
                  f"p={l_stat.pvalue:.2e}")

        print(f"\n--- 逐刺激跨受试信度 ---")
        print(f"  {rel:+.3f}")

        print("\n" + "=" * 76)
        hab_ok = h_stat and h_stat.pvalue < .05 and H.mean() < 0
        lock_ok = l_stat and l_stat.pvalue < .05 and pr.scr.mean() > pr.scr_random.mean()
        if hab_ok and lock_ok:
            print(f"✅ 两个阳性对照通过 —— 可以解读信度 {rel:+.3f}。")
        else:
            print("🔴 阳性对照仍未全部通过：")
            print(f"   习惯化 {'✅' if hab_ok else '❌'}   刺激锁时 {'✅' if lock_ok else '❌'}")
            print("   **不得解读任何生理结果。** 需核对 EDA 单位/极性、")
            print("   时间戳语义，或直接联系数据集作者。")


if __name__ == "__main__":
    main()
