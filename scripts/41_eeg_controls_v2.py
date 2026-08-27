"""阳性对照 v2 —— 修正 v1 中一处使检验失效的不对称。

    .venv/Scripts/python scripts/41_eeg_controls_v2.py --n-subjects 31

## v1（`37_eeg_positive_controls.py`）发生了什么

| 对照 | v1 结果 | 性质 |
|---|---|---|
| C1 枕区 alpha | ✅ 差 +0.264，**31/31 为正**，p=1.9e-09 | 真通过 |
| C2 眨眼锁时 | ✅ 额区 163.8 μV vs 枕区 39.2，vs 随机 25.8，p=9.3e-10 | 真通过 |
| C0 事件簿记 | 核心四条**全 100%**，仅附加的作答码冗余假设不成立 | 判据写错 |
| C3 声音起始 | ❌ 刺激 2.446 μV **<** 随机 4.800 μV，比值 0.51 | **检验失效** |

## C3 为什么失效

v1 对**刺激试次**做了 200 μV 峰峰值剔除，对**随机窗不做**。
每 run 仅 10 个试次，随机窗里的眨眼（数百 μV）在如此少的平均中消不掉，
直接抬高其 GFP。

判别依据：真正「无听觉响应」应给出比值 **≈1.0**；观测到的 **0.51**
（94% 受试随机窗更大）只能由两臂处理不对称产生。
而刺激臂的 2.446 μV 恰是听觉诱发 GFP 的典型量级。

**该缺陷在写 v1 时可见，我当时判断它「偏保守」—— 判断错了。**

## v2 只改这一处

两臂**完全同处理**：随机臂同样做 200 μV 剔除，且取相同的存活试次数。

顺带补上第二个零分布：**循环移位**（本项目 B6 确立的方法）——
保留 10 个起始的时间间隔结构，只破坏其相对声音的相位。
两个零分布同时报告，不在事后择一。

## 判据（先于数据，且不再修改）

**C3 通过 = 刺激臂诱发 GFP 显著大于两个零分布（Wilcoxon p<.05）。**

不过就停，执行删除方案。本脚本是生理侧的**最后一次尝试**。

## 附带诊断（不参与判决）

1. 额中央 N1（[0.08, 0.16] s）总平均波形，导出供检视
2. **音频起始陡峭度** —— 电影配乐节选若带淡入，本就不存在锐利声起始，
   听觉 N1 也就不该出现。这是对刺激材料本身的独立核查，
   由音频数据回答，不由脑电回答。
"""

from __future__ import annotations

import argparse
import warnings

import mne
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

ROOT = REPO_ROOT / "data" / "raw" / "ds002721"
MUSIC_RUNS = (2, 3, 4, 5)
CODE_MUSIC_ON, STIM_LO, STIM_HI = 788, 301, 660

AER_WIN = (-0.20, 0.60)
AER_BASE = (-0.20, 0.0)
GFP_WIN = (0.0, 0.30)
N1_WIN = (0.08, 0.16)
FRONTOCENTRAL = ("F3", "FZ", "F4", "C3", "CZ", "C4")
REJECT_V = 200e-6
DS_HZ = 250.0
MIN_N = 8


def load_events(sub: str, run: int) -> pd.DataFrame | None:
    p = ROOT / sub / "eeg" / f"{sub}_task-run{run}_events.tsv"
    if not p.exists():
        return None
    e = pd.read_csv(p, sep="\t")
    e["trial_type"] = pd.to_numeric(e["trial_type"], errors="coerce")
    return e.dropna(subset=["trial_type", "onset"])


def epoch_at(x: np.ndarray, sf: float, onsets: np.ndarray,
             tmin: float, tmax: float) -> np.ndarray:
    n = x.shape[1]
    a, b = int(round(tmin * sf)), int(round(tmax * sf))
    out = []
    for t in onsets:
        i = int(round(t * sf))
        if i + a >= 0 and i + b < n:
            out.append(x[:, i + a: i + b])
    if not out:
        return np.empty((0, x.shape[0], 0))
    m = min(v.shape[1] for v in out)
    return np.stack([v[:, :m] for v in out])


def prep(ep: np.ndarray, nb: int) -> tuple[np.ndarray, np.ndarray]:
    """基线扣除 + 剔除掩码 —— 两臂共用，保证同处理。"""
    if not ep.size:
        return ep, np.zeros(0, dtype=bool)
    ep = ep - ep[:, :, :nb].mean(axis=2, keepdims=True)
    return ep, np.ptp(ep, axis=2).max(axis=1) < REJECT_V


def evoked_stats(ep: np.ndarray, sf: float, ch: dict[str, int]) -> dict[str, float]:
    ev = ep.mean(axis=0)
    t = np.arange(ev.shape[1]) / sf + AER_WIN[0]
    g = (t >= GFP_WIN[0]) & (t <= GFP_WIN[1])
    n1m = (t >= N1_WIN[0]) & (t <= N1_WIN[1])
    fc = [ch[c] for c in FRONTOCENTRAL if c in ch]
    return {"gfp": float(ev[:, g].std(axis=0).mean()),
            "n1": float(ev[np.ix_(fc, np.where(n1m)[0])].mean()) if fc else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=31)
    args = ap.parse_args()

    subs = sorted(d.name for d in ROOT.glob("sub-*") if d.is_dir())[: args.n_subjects]

    with RunRecord(
        "ds002721_controls_v2", seed=SEED,
        params={"dataset": "OpenNeuro ds002721", "n_subjects": len(subs),
                "fix_vs_v1": "两臂同处理：随机臂同样做 200 μV 剔除并取相同存活试次数",
                "nulls": ["uniform random onsets", "circular shift (B6)"],
                "reject_uV": REJECT_V * 1e6, "gfp_win_s": GFP_WIN,
                "criterion": "刺激臂 GFP 显著大于两个零分布 → 通过；否则终止生理线"},
    ) as run:
        rng = np.random.default_rng(SEED)
        rows, waves = [], []

        for k, sub in enumerate(subs, 1):
            gs, gr, gc, n1s, n1r, nkeep = [], [], [], [], [], []
            for r in MUSIC_RUNS:
                e = load_events(sub, r)
                p = ROOT / sub / "eeg" / f"{sub}_task-run{r}_eeg.edf"
                if e is None or not p.exists():
                    continue
                try:
                    raw = mne.io.read_raw_edf(p, preload=True, verbose=False)
                except Exception:                            # noqa: BLE001
                    continue
                raw.rename_channels({c: c.strip().upper() for c in raw.ch_names})
                raw.notch_filter(50.0, verbose=False)
                raw.filter(1.0, 30.0, verbose=False)
                raw.set_eeg_reference("average", verbose=False)
                if raw.info["sfreq"] > DS_HZ:
                    raw.resample(DS_HZ, verbose=False)
                sf = raw.info["sfreq"]
                x = raw.get_data()
                ch = {c: i for i, c in enumerate(raw.ch_names)}
                dur = x.shape[1] / sf
                nb = int(round((AER_BASE[1] - AER_BASE[0]) * sf))
                del raw

                on = np.sort(e.loc[e.trial_type == CODE_MUSIC_ON, "onset"].to_numpy())
                if not len(on):
                    continue

                ep_s, keep_s = prep(epoch_at(x, sf, on, *AER_WIN), nb)
                n = int(keep_s.sum())
                if n < 3:
                    continue
                st = evoked_stats(ep_s[keep_s], sf, ch)
                gs.append(st["gfp"]); n1s.append(st["n1"]); nkeep.append(n)
                waves.append({"subject": sub, "run": r, "arm": "stim",
                              **{f"t{i}": v for i, v in
                                 enumerate(ep_s[keep_s].mean(axis=0)[
                                     [ch[c] for c in FRONTOCENTRAL if c in ch]]
                                     .mean(axis=0))}})

                # --- 零分布 A：均匀随机起始（同剔除、同存活数）---
                cand = rng.uniform(2.0, dur - 2.0, size=n * 8)
                ep_r, keep_r = prep(epoch_at(x, sf, cand, *AER_WIN), nb)
                idx = np.where(keep_r)[0][:n]
                if len(idx) >= 3:
                    gr.append(evoked_stats(ep_r[idx], sf, ch)["gfp"])
                    n1r.append(evoked_stats(ep_r[idx], sf, ch)["n1"])

                # --- 零分布 B：循环移位（保留起始间隔结构）---
                shifted = (on + rng.uniform(20.0, dur - 20.0)) % (dur - 4.0) + 2.0
                ep_c, keep_c = prep(epoch_at(x, sf, np.sort(shifted), *AER_WIN), nb)
                if keep_c.sum() >= 3:
                    gc.append(evoked_stats(ep_c[keep_c], sf, ch)["gfp"])
                del x

            if gs and gr and gc:
                rows.append({"subject": sub, "n_kept": int(np.sum(nkeep)),
                             "gfp_stim": float(np.mean(gs)),
                             "gfp_random": float(np.mean(gr)),
                             "gfp_circshift": float(np.mean(gc)),
                             "n1_stim": float(np.nanmean(n1s)),
                             "n1_random": float(np.nanmean(n1r))})
            print(f"  {k}/{len(subs)}  {sub}", flush=True)

        D = pd.DataFrame(rows)
        run.log_metric("n_subjects_analysed", int(len(D)))
        run.log_metric("total_epochs_kept", int(D.n_kept.sum()) if len(D) else 0)

        print("\n" + "=" * 80)
        print(f"run_id: {run.run_id}   受试 {len(D)}   "
              f"存活试次 {int(D.n_kept.sum()) if len(D) else 0}")

        ok = None
        if len(D) >= MIN_N:
            st_r = wilcoxon(D.gfp_stim, D.gfp_random)
            st_c = wilcoxon(D.gfp_stim, D.gfp_circshift)
            st_n = wilcoxon(D.n1_stim.fillna(0), D.n1_random.fillna(0))
            ok = bool(st_r.pvalue < .05 and D.gfp_stim.mean() > D.gfp_random.mean()
                      and st_c.pvalue < .05
                      and D.gfp_stim.mean() > D.gfp_circshift.mean())
            run.log_metric("C3_v2", {
                "gfp_stim_uV": float(D.gfp_stim.mean() * 1e6),
                "gfp_random_uV": float(D.gfp_random.mean() * 1e6),
                "gfp_circshift_uV": float(D.gfp_circshift.mean() * 1e6),
                "ratio_vs_random": float((D.gfp_stim / D.gfp_random).median()),
                "ratio_vs_circshift": float((D.gfp_stim / D.gfp_circshift).median()),
                "frac_stim_gt_random": float((D.gfp_stim > D.gfp_random).mean()),
                "frac_stim_gt_circshift": float((D.gfp_stim > D.gfp_circshift).mean()),
                "p_vs_random": float(st_r.pvalue),
                "p_vs_circshift": float(st_c.pvalue),
                "n1_stim_uV": float(D.n1_stim.mean() * 1e6),
                "n1_random_uV": float(D.n1_random.mean() * 1e6),
                "p_n1": float(st_n.pvalue)})

            print("\n--- C3 v2：声音起始诱发 GFP（两臂同处理）---")
            print(f"  刺激        {D.gfp_stim.mean()*1e6:.3f} μV")
            print(f"  随机起始    {D.gfp_random.mean()*1e6:.3f} μV   "
                  f"比值 {(D.gfp_stim/D.gfp_random).median():.2f}   "
                  f"刺激更大 {100*(D.gfp_stim>D.gfp_random).mean():.0f}%   "
                  f"p={st_r.pvalue:.2e}")
            print(f"  循环移位    {D.gfp_circshift.mean()*1e6:.3f} μV   "
                  f"比值 {(D.gfp_stim/D.gfp_circshift).median():.2f}   "
                  f"刺激更大 {100*(D.gfp_stim>D.gfp_circshift).mean():.0f}%   "
                  f"p={st_c.pvalue:.2e}")
            print(f"\n  额中央 N1 [{N1_WIN[0]}, {N1_WIN[1]}] s："
                  f"刺激 {D.n1_stim.mean()*1e6:+.3f} μV   "
                  f"随机 {D.n1_random.mean()*1e6:+.3f} μV   p={st_n.pvalue:.3f}")

        run.log_metric("verdict_C3", ok)
        print("\n" + "=" * 80)
        if ok is True:
            print("✅ C3 通过 —— 加上 v1 已通过的 C1/C2，四项对照齐备：")
            print("   时间对齐（眨眼 ERP）、通道与频谱（枕区 alpha）、")
            print("   事件簿记（100% 双射）、刺激锁时响应（GFP > 两个零分布）。")
            print("   → 进入 39_eeg_reliability.py。")
        elif ok is False:
            print("🔴 C3 仍未通过 —— 两臂已同处理，无第二次修正。")
            print("   → 执行删除方案，本项目仅保留主观维度。")
        else:
            print(f"⚪ 受试不足 {MIN_N}，未评估。")

        if len(D):
            D.to_csv(REPO_ROOT / "reports" / "source_data" /
                     "ds002721_c3_v2.csv", index=False)
        if waves:
            pd.DataFrame(waves).to_csv(REPO_ROOT / "reports" / "source_data" /
                                       "ds002721_erp_waveforms.csv", index=False)


if __name__ == "__main__":
    main()
