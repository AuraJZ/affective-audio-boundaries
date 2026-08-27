"""补上 ICA 去眼电，重抽全部脑电特征 —— 补一个属于我的方法学缺口。

    .venv/Scripts/python scripts/52_ica_cleaned_features.py --n-subjects 31

## 为什么必须做

此前的试次级带功率（`39`）与相干（`46`）**都没有做伪迹去除**。
眨眼与肌电混在 13 秒响应窗里，会抬高噪声、**压低所有效应量** ——
方向上不利于「测不到刺激特异成分」这个结论。

不补上，「我们测不到」就可以被一句「你没清伪迹」驳回。

## 做法

- ICA（FastICA，19 通道取 15 个成分），拟合在 1–45 Hz 带通后的连续数据上
- 用 **FP1/FP2 作为 EOG 代理**（该数据集无专用眼电导）自动识别眼动成分
- 剔除相关成分后重建，再按与 `39`/`46` **完全相同**的窗口与参数抽特征

参数不变是关键：唯一的差别必须是 ICA，否则无法归因。

## 预期

我预计 ICA 会**小幅抬高**效应量而不会翻转结论 ——
眨眼不锁时于刺激，也不与评分相关，主要作用是加噪。
**但预期不能替代验证**，所以要跑。
"""

from __future__ import annotations

import argparse
import warnings

import mne
import numpy as np
import pandas as pd

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

ROOT = REPO_ROOT / "data" / "raw" / "ds002721"
OUT_BAND = REPO_ROOT / "data" / "features" / "ds002721_trials_ica.parquet"
OUT_CONN = REPO_ROOT / "data" / "features" / "ds002721_connectivity_ica.parquet"
MUSIC_RUNS = (2, 3, 4, 5)
DS_HZ = 250.0
N_ICA = 15
EOG_PROXY = ("FP1", "FP2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=31)
    args = ap.parse_args()

    import importlib
    m39 = importlib.import_module("39_eeg_reliability")
    m46 = importlib.import_module("46_connectivity_features")

    subs = sorted(d.name for d in ROOT.glob("sub-*") if d.is_dir())[: args.n_subjects]

    with RunRecord(
        "ds002721_ica_features", seed=SEED,
        params={"n_subjects": len(subs), "n_ica_components": N_ICA,
                "eog_proxy": list(EOG_PROXY),
                "identical_to": "39_eeg_reliability + 46_connectivity_features",
                "only_difference": "ICA 去眼电",
                "why": "未做伪迹去除会压低效应量，方向上不利于本项目的阴性结论"},
    ) as run:
        band_rows, conn_rows, n_excl = [], [], []

        for k, sub in enumerate(subs, 1):
            for r in MUSIC_RUNS:
                ep = ROOT / sub / "eeg" / f"{sub}_task-run{r}_events.tsv"
                pp = ROOT / sub / "eeg" / f"{sub}_task-run{r}_eeg.edf"
                if not ep.exists() or not pp.exists():
                    continue
                e = pd.read_csv(ep, sep="\t")
                e["trial_type"] = pd.to_numeric(e["trial_type"], errors="coerce")
                e = e.dropna(subset=["trial_type", "onset"])
                tr = m46.trials_of(e)
                if not tr:
                    continue
                try:
                    raw = mne.io.read_raw_edf(pp, preload=True, verbose=False)
                except Exception:                            # noqa: BLE001
                    continue
                raw.rename_channels({c: c.strip().upper() for c in raw.ch_names})
                raw.notch_filter(50.0, verbose=False)
                raw.filter(1.0, 45.0, verbose=False)
                if raw.info["sfreq"] > DS_HZ:
                    raw.resample(DS_HZ, verbose=False)

                # ---- ICA：用 FP1/FP2 作 EOG 代理 ----
                try:
                    ica = mne.preprocessing.ICA(
                        n_components=N_ICA, method="fastica",
                        random_state=SEED, max_iter=800)
                    ica.fit(raw, verbose=False)
                    bad = []
                    for ch in EOG_PROXY:
                        if ch in raw.ch_names:
                            idx, _ = ica.find_bads_eog(raw, ch_name=ch, verbose=False)
                            bad.extend(idx)
                    ica.exclude = sorted(set(bad))
                    n_excl.append(len(ica.exclude))
                    ica.apply(raw, verbose=False)
                except Exception:                            # noqa: BLE001
                    n_excl.append(-1)                        # ICA 失败，记下不静默

                sf = raw.info["sfreq"]
                x = raw.get_data()
                ch = {c: i for i, c in enumerate(raw.ch_names)}

                # ---- 带功率（与 39 同参数，平均参考）----
                raw_avg = raw.copy().set_eeg_reference("average", verbose=False)
                for i, (onset, stim) in enumerate(tr):
                    f = m39.eeg_features(raw_avg, onset)
                    if f:
                        band_rows.append({"subject": sub, "run": r, "trial": i,
                                          "stim": stim, **f})
                del raw_avg

                # ---- 相干（与 46 同参数，保留原参考）----
                for i, (onset, stim) in enumerate(tr):
                    f = m46.trial_features(x, sf, ch, onset)
                    if f:
                        conn_rows.append({"subject": sub, "run": r, "trial": i,
                                          "stim": stim, **f})
                del raw, x
            print(f"  {k}/{len(subs)}  {sub}", flush=True)

        Bd = pd.DataFrame(band_rows)
        Cn = pd.DataFrame(conn_rows)
        Bd.to_parquet(OUT_BAND)
        Cn.to_parquet(OUT_CONN)

        ex = np.array([v for v in n_excl if v >= 0])
        run.log_metric("n_runs_processed", len(n_excl))
        run.log_metric("n_ica_failed", int(sum(v < 0 for v in n_excl)))
        run.log_metric("components_excluded",
                       {"mean": float(ex.mean()) if ex.size else None,
                        "median": float(np.median(ex)) if ex.size else None,
                        "max": int(ex.max()) if ex.size else None,
                        "runs_with_none_excluded": int((ex == 0).sum())})
        run.log_metric("n_trials", {"band": int(len(Bd)), "conn": int(len(Cn))})

        print(f"\n→ {OUT_BAND.name}   {len(Bd)} 试次")
        print(f"→ {OUT_CONN.name}   {len(Cn)} 试次")
        if ex.size:
            print(f"  剔除的 ICA 成分数：中位 {np.median(ex):.0f}   "
                  f"均值 {ex.mean():.1f}   最多 {ex.max()}   "
                  f"一个都没剔的 run {int((ex==0).sum())}/{len(ex)}")
        if any(v < 0 for v in n_excl):
            print(f"  ⚠️ ICA 失败 {sum(v < 0 for v in n_excl)} 个 run（已按原样保留）")


if __name__ == "__main__":
    main()
