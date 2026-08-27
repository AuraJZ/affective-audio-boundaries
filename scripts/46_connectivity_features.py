"""补上连接性与不对称特征 —— 文献发现效应的正是这两类，而我此前只有带功率。

    .venv/Scripts/python scripts/46_connectivity_features.py

## 为什么必须补

Daly et al. (2014) 用**同一批数据**报告的效应在两处：
**通道间相干（inter-channel coherence）** 与 **beta/gamma 频段不对称**。
我此前的 26 维只有区域带功率 + 一个额区 alpha 不对称。

因此「脑电测不到刺激特异成分」这一结论**在补上这两类特征之前不成立** ——
否则就是拿一个不包含文献效应的特征集，去否定文献的效应。

## 特征

| 类 | 维度 | 说明 |
|---|---|---|
| 幅度平方相干 | 10 区域对 × 5 频带 = 50 | 经典做法，但受容积传导污染 |
| **虚部相干** | 10 × 5 = 50 | 对容积传导稳健 —— 零时延的伪相关虚部为 0 |
| 电极对不对称 | 6 对 × 5 频带 = 30 | log(右) − log(左)，含 beta/gamma |

合计 **130 维**，与既有 26 维带功率并列。

## 参考电极

相干**不做平均参考** —— 平均参考会在所有通道间引入共同成分，
系统性抬高相干。保留记录时的 FCz 参考，与文献做法一致。

## 相干的基线扣除

基线窗仅 2 s，在 250 Hz 下不足以稳定估计相干（分段太少）。
因此相干**只取响应窗的值**，不做基线扣除；被试内 z 标准化在后续分析中完成。
"""

from __future__ import annotations

import argparse
import warnings

import mne
import numpy as np
import pandas as pd
from scipy.signal import csd, welch

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

ROOT = REPO_ROOT / "data" / "raw" / "ds002721"
OUT = REPO_ROOT / "data" / "features" / "ds002721_connectivity.parquet"
MUSIC_RUNS = (2, 3, 4, 5)

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
REGIONS = {"F": ("FP1", "FP2", "F7", "F3", "FZ", "F4", "F8"),
           "C": ("C3", "CZ", "C4"),
           "T": ("T3", "T4", "T5", "T6"),
           "P": ("P3", "PZ", "P4"),
           "O": ("O1", "O2")}
ASYM_PAIRS = [("F4", "F3"), ("F8", "F7"), ("T4", "T3"),
              ("T6", "T5"), ("P4", "P3"), ("O2", "O1")]
RESP_WIN = (1.0, 14.0)
DS_HZ = 250.0
NPERSEG = 256          # ~1 s @250 Hz；13 s 响应窗可分 ~12 段


def trials_of(e: pd.DataFrame) -> list[tuple[float, int]]:
    on = np.sort(e.loc[e.trial_type == 788, "onset"].to_numpy())
    st = e[e.trial_type.between(301, 660)].sort_values("onset")
    sv, sc = st.onset.to_numpy(), st.trial_type.to_numpy().astype(int)
    out = []
    for t in on:
        j = np.searchsorted(sv, t, side="right") - 1
        if j >= 0:
            out.append((float(t), int(sc[j]) - 300))
    return out


def band_mask(f: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (f >= lo) & (f <= hi)


def trial_features(x: np.ndarray, sf: float, ch: dict[str, int],
                   onset: float) -> dict[str, float] | None:
    a = int(round((onset + RESP_WIN[0]) * sf))
    b = int(round((onset + RESP_WIN[1]) * sf))
    if a < 0 or b > x.shape[1] or b - a < NPERSEG * 3:
        return None
    seg = x[:, a:b]

    # 区域平均信号
    R = {}
    for name, chans in REGIONS.items():
        idx = [ch[c] for c in chans if c in ch]
        if idx:
            R[name] = seg[idx].mean(axis=0)
    names = sorted(R)
    d: dict[str, float] = {}

    # ---- 相干（幅度平方 + 虚部）----
    pxx = {n: welch(R[n], fs=sf, nperseg=NPERSEG)[1] for n in names}
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            f, p12 = csd(R[n1], R[n2], fs=sf, nperseg=NPERSEG)
            denom = np.sqrt(pxx[n1] * pxx[n2])
            denom = np.where(denom > 0, denom, np.nan)
            coh = np.abs(p12) ** 2 / (pxx[n1] * np.where(pxx[n2] > 0, pxx[n2], np.nan))
            icoh = np.abs(np.imag(p12 / denom))
            for band, (lo, hi) in BANDS.items():
                m = band_mask(f, lo, hi)
                d[f"coh_{n1}{n2}_{band}"] = float(np.nanmean(coh[m]))
                d[f"icoh_{n1}{n2}_{band}"] = float(np.nanmean(icoh[m]))

    # ---- 电极对不对称：log(右) − log(左) ----
    for right, left in ASYM_PAIRS:
        if right not in ch or left not in ch:
            continue
        f, pr = welch(seg[ch[right]], fs=sf, nperseg=NPERSEG)
        _, pl = welch(seg[ch[left]], fs=sf, nperseg=NPERSEG)
        for band, (lo, hi) in BANDS.items():
            m = band_mask(f, lo, hi)
            vr, vl = float(np.trapezoid(pr[m], f[m])), float(np.trapezoid(pl[m], f[m]))
            d[f"asym_{right}{left}_{band}"] = float(
                np.log(max(vr, 1e-20)) - np.log(max(vl, 1e-20)))
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=31)
    args = ap.parse_args()
    subs = sorted(d.name for d in ROOT.glob("sub-*") if d.is_dir())[: args.n_subjects]

    with RunRecord(
        "ds002721_connectivity", seed=SEED,
        params={"dataset": "ds002721", "n_subjects": len(subs),
                "features": "10 区域对 × 5 频带 ×（幅度平方相干 + 虚部相干）"
                            " + 6 电极对 × 5 频带不对称 = 130 维",
                "reference": "保留记录时的 FCz —— 平均参考会抬高相干",
                "resp_win_s": RESP_WIN, "nperseg": NPERSEG,
                "why": "Daly et al. (2014) 在同一数据上的效应正在相干与"
                       "beta/gamma 不对称，此前的特征集不含这两类"},
    ) as run:
        rows = []
        for k, sub in enumerate(subs, 1):
            for r in MUSIC_RUNS:
                ep = ROOT / sub / "eeg" / f"{sub}_task-run{r}_events.tsv"
                pp = ROOT / sub / "eeg" / f"{sub}_task-run{r}_eeg.edf"
                if not ep.exists() or not pp.exists():
                    continue
                e = pd.read_csv(ep, sep="\t")
                e["trial_type"] = pd.to_numeric(e["trial_type"], errors="coerce")
                e = e.dropna(subset=["trial_type", "onset"])
                tr = trials_of(e)
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
                sf, x = raw.info["sfreq"], raw.get_data()
                ch = {c: i for i, c in enumerate(raw.ch_names)}
                del raw
                for i, (onset, stim) in enumerate(tr):
                    f = trial_features(x, sf, ch, onset)
                    if f:
                        rows.append({"subject": sub, "run": r, "trial": i,
                                     "stim": stim, **f})
                del x
            print(f"  {k}/{len(subs)}  {sub}", flush=True)

        D = pd.DataFrame(rows)
        D.to_parquet(OUT)
        feat = [c for c in D.columns
                if c.startswith(("coh_", "icoh_", "asym_"))]
        run.log_metric("n_trials", int(len(D)))
        run.log_metric("n_features", len(feat))
        run.log_metric("feature_families",
                       {"coh": sum(c.startswith("coh_") for c in feat),
                        "icoh": sum(c.startswith("icoh_") for c in feat),
                        "asym": sum(c.startswith("asym_") for c in feat)})
        print(f"\n→ {OUT.name}   {len(D)} 试次 × {len(feat)} 维")
        print(f"  相干 {sum(c.startswith('coh_') for c in feat)}   "
              f"虚部相干 {sum(c.startswith('icoh_') for c in feat)}   "
              f"不对称 {sum(c.startswith('asym_') for c in feat)}")


if __name__ == "__main__":
    main()
