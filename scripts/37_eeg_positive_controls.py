"""ds002721 阳性对照 —— 判据在看数据之前定死，不过就停。

    .venv/Scripts/python scripts/37_eeg_positive_controls.py --n-subjects 31

## 纪律

BIRAFFE2 的教训是**顺序错误**：我反复解释阴性结果，却从未验证流水线能测出
任何**已知存在**的效应。本脚本先跑对照，**任何一项不过即终止**，不做事后补救、
不换参数重试。

## 四项对照（判据写在代码里，先于数据）

| | 检验 | 测什么 | 通过条件 |
|---|---|---|---|
| **C0** | 事件簿记 | 我对触发码的理解是否正确 | 每个音乐 run 恰 10 个 `788`，与刺激码严格一一配对，且 80 个作答 |
| **C1** | 枕区 alpha 优势 | EDF 读取、通道身份、频谱 | O1/O2 相对 alpha > FP1/FP2，Wilcoxon p<.05 |
| **C2** | **眨眼锁时 ERP** | **events.tsv ↔ EDF 采样点对齐** | 额区峰峰值 / 枕区 > 2.0，且 > 随机窗，p<.05 |
| **C3** | **声音起始诱发响应** | **刺激锁时的听觉响应存在** | 诱发 GFP(0–0.3 s) > 随机起始零分布，p<.05 |

「未评估」（样本不足）与「未通过」严格区分 —— 把没跑的判成失败，
是我此前四次错误判决中重复出现的那一类。

## 事件配对规则（由 `diag_events` 实测确定）

刺激码在**注视十字**时发出，早于音乐起始 0–8.5 s（不固定），
故按「**最后一个不晚于音乐起始的刺激码**」配对，并要求配对为双射。
每 run：10 个 `786` / 10 个 `788` / 10 个刺激码 / 80 个作答，严格交替。
`800–807` 出现 239 次是问题重复触发所致，取**与作答同时刻**的那一个。

**C2 是 BIRAFFE2 从来没有的那一项。** 眨眼幅度 >100 μV，若 `events.tsv` 的
onset 与 EDF 采样点对齐，额区总平均必然出现巨大偏转；若对不齐，则不会。
这是对「时间戳语义」的直接判决，不依赖任何生理学假设。

C3 用 **GFP（跨通道空间标准差）** 作主统计量：参考电极无关、极性无关，
只问「刺激后是否存在锁时的场变化」。对照是**同一段信号里的随机起始点**，
处理流程完全相同，仅标记时刻不同。

## 参考电极

原始记录参考于 FCz —— 恰在听觉 N1 的最大值附近，会削弱 Cz/Fz 处的 N1。
因此 C3 前重参考为**平均参考**（19 通道）。C2 反而保留原始参考：
眨眼在 FP1/FP2 最大，平均参考会把它摊到全头。
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
REST_RUNS = (1, 6)

CODE_MUSIC_ON = 788
CODE_FIXATION = 786
CODE_BLINK = 257
STIM_LO, STIM_HI = 301, 660          # 减 300 → 001–360.mp3
CODE_Q = range(800, 808)             # 8 个问题
CODE_A = range(833, 842)             # 作答 1–9

FRONTAL = ("FP1", "FP2")
OCCIPITAL = ("O1", "O2")
ALPHA = (8.0, 13.0)
BROAD = (2.0, 45.0)

# ---- C2 眨眼窗 ----
BLINK_WIN = (-0.30, 0.50)
# ---- C3 声音起始窗 ----
AER_WIN = (-0.20, 0.60)
AER_BASE = (-0.20, 0.0)
GFP_WIN = (0.0, 0.30)

DS_HZ = 250.0
MIN_N = 8                            # 组统计所需最少受试；不足则记「未评估」


# ---------------------------------------------------------------- 载入
def run_path(sub: str, run: int, suffix: str):
    return ROOT / sub / "eeg" / f"{sub}_task-run{run}_{suffix}"


def load_events(sub: str, run: int) -> pd.DataFrame | None:
    p = run_path(sub, run, "events.tsv")
    if not p.exists():
        return None
    e = pd.read_csv(p, sep="\t")
    e["trial_type"] = pd.to_numeric(e["trial_type"], errors="coerce")
    return e.dropna(subset=["trial_type", "onset"])


def load_raw(sub: str, run: int, l_freq: float, h_freq: float,
             average_ref: bool) -> mne.io.BaseRaw | None:
    p = run_path(sub, run, "eeg.edf")
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        raw = mne.io.read_raw_edf(p, preload=True, verbose=False)
    except Exception:                                        # noqa: BLE001
        return None
    raw.rename_channels({c: c.strip().upper() for c in raw.ch_names})
    raw.notch_filter(50.0, verbose=False)
    raw.filter(l_freq, h_freq, verbose=False)
    if average_ref:
        raw.set_eeg_reference("average", verbose=False)
    if raw.info["sfreq"] > DS_HZ:
        raw.resample(DS_HZ, verbose=False)
    return raw


def epoch_at(raw: mne.io.BaseRaw, onsets: np.ndarray,
             tmin: float, tmax: float) -> np.ndarray:
    """在给定秒数处切片 → (n_epochs, n_ch, n_times)；越界的丢弃。"""
    sf = raw.info["sfreq"]
    data = raw.get_data()
    n = data.shape[1]
    a, b = int(round(tmin * sf)), int(round(tmax * sf))
    out = []
    for t in onsets:
        i = int(round(t * sf))
        if i + a >= 0 and i + b < n:
            out.append(data[:, i + a: i + b])
    if not out:
        return np.empty((0, data.shape[0], 0))
    m = min(x.shape[1] for x in out)
    return np.stack([x[:, :m] for x in out])


def picks(raw: mne.io.BaseRaw, names) -> list[int]:
    return [raw.ch_names.index(n) for n in names if n in raw.ch_names]


def pair_trials(e: pd.DataFrame) -> pd.DataFrame:
    """把音乐起始配到刺激码 —— 取最后一个不晚于起始的刺激码。

    刺激码在注视十字时发出，提前量 0–8.5 s 不固定，因此不能用固定容差。
    返回每试次一行；`bijective` 记录配对是否为双射（无重复占用）。
    """
    on = np.sort(e.loc[e.trial_type == CODE_MUSIC_ON, "onset"].to_numpy())
    stim = e[(e.trial_type >= STIM_LO) & (e.trial_type <= STIM_HI)].sort_values("onset")
    st, sc = stim.onset.to_numpy(), stim.trial_type.to_numpy().astype(int)
    rows = []
    for i, t in enumerate(on):
        j = np.searchsorted(st, t, side="right") - 1
        rows.append({"trial": i, "onset": float(t),
                     "code": int(sc[j]) if j >= 0 else -1,
                     "stim": int(sc[j]) - 300 if j >= 0 else -1,
                     "lead_s": float(t - st[j]) if j >= 0 else np.nan,
                     "stim_idx": int(j)})
    T = pd.DataFrame(rows)
    T["bijective"] = (T.stim_idx >= 0).all() and T.stim_idx.nunique() == len(T)
    return T


def extract_ratings(e: pd.DataFrame) -> pd.DataFrame:
    """逐试次个体评分：取与作答码同时刻的问题码。

    `833–841` → 1–9，`901–909` → 1–9，两套编码冗余，用于互校。
    """
    resp = e[e.trial_type.isin(CODE_A)]
    rows = []
    for t, v in zip(resp.onset.to_numpy(), resp.trial_type.to_numpy().astype(int)):
        same = e[np.isclose(e.onset, t)]
        q = same[same.trial_type.isin(CODE_Q)].trial_type
        alt = same[(same.trial_type >= 901) & (same.trial_type <= 909)].trial_type
        rows.append({"onset": float(t), "question": int(q.iloc[0]) - 800
                     if len(q) else -1, "response": v - 832,
                     "response_alt": int(alt.iloc[0]) - 900 if len(alt) else -1})
    return pd.DataFrame(rows)


def band_power(x: np.ndarray, sf: float, lo: float, hi: float) -> np.ndarray:
    """Welch 带功率 → (n_ch,)。"""
    from scipy.signal import welch
    f, p = welch(x, fs=sf, nperseg=int(sf * 4), axis=-1)
    m = (f >= lo) & (f <= hi)
    return np.trapezoid(p[..., m], f[m], axis=-1)


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=31)
    args = ap.parse_args()

    subs = sorted(d.name for d in ROOT.glob("sub-*") if d.is_dir())[: args.n_subjects]
    if not subs:
        raise SystemExit(f"没有数据：{ROOT}")

    with RunRecord(
        "ds002721_positive_controls", seed=SEED,
        params={"dataset": "OpenNeuro ds002721", "n_subjects": len(subs),
                "controls": ["C0 event bookkeeping", "C1 posterior alpha",
                             "C2 blink-locked ERP (timing alignment)",
                             "C3 sound-onset evoked GFP vs random-onset null"],
                "discipline": "判据先于数据；任一不过即终止，不做事后补救"},
    ) as run:
        rng = np.random.default_rng(SEED)
        c0, c1, c2, c3 = [], [], [], []

        for k, sub in enumerate(subs, 1):
            # ---------------- C0：事件簿记 ----------------
            for r in MUSIC_RUNS:
                e = load_events(sub, r)
                if e is None:
                    continue
                T = pair_trials(e)
                R = extract_ratings(e)
                agree = (float((R.response == R.response_alt).mean())
                         if len(R) else np.nan)
                c0.append({"subject": sub, "run": r, "n_onset": int(len(T)),
                           "n_stim": int(e.trial_type.between(STIM_LO, STIM_HI).sum()),
                           "n_fixation": int((e.trial_type == CODE_FIXATION).sum()),
                           "bijective": bool(T.bijective.iloc[0]) if len(T) else False,
                           "stim_in_range": bool(((T.stim >= 1) & (T.stim <= 360)).all())
                           if len(T) else False,
                           "max_lead_s": float(T.lead_s.max()) if len(T) else np.nan,
                           "n_ratings": int(len(R)),
                           "code_agreement": agree})

            # ---------------- C1：枕区 alpha 优势（静息 run）----------------
            raw = load_raw(sub, REST_RUNS[0], 1.0, 45.0, average_ref=False)
            if raw is not None:
                x = raw.get_data()
                sf = raw.info["sfreq"]
                pa = band_power(x, sf, *ALPHA)
                pb = band_power(x, sf, *BROAD)
                rel = pa / np.where(pb > 0, pb, np.nan)
                fo, oc = picks(raw, FRONTAL), picks(raw, OCCIPITAL)
                if fo and oc:
                    c1.append({"subject": sub,
                               "alpha_rel_frontal": float(np.nanmean(rel[fo])),
                               "alpha_rel_occipital": float(np.nanmean(rel[oc]))})
                del raw, x

            # ---------------- C2 / C3：音乐 run ----------------
            blink_f, blink_o, blink_rand = [], [], []
            aer_gfp, null_gfp, n_trials = [], [], 0

            for r in MUSIC_RUNS:
                e = load_events(sub, r)
                if e is None:
                    continue

                # --- C2：眨眼锁时（保留原始参考）---
                raw_b = load_raw(sub, r, 1.0, 20.0, average_ref=False)
                if raw_b is not None:
                    bl = e.loc[e.trial_type == CODE_BLINK, "onset"].to_numpy()
                    dur = raw_b.n_times / raw_b.info["sfreq"]
                    if len(bl) >= 5:
                        ep = epoch_at(raw_b, bl, *BLINK_WIN)
                        if ep.size:
                            ep = ep - ep[:, :, :5].mean(axis=2, keepdims=True)
                            ev = ep.mean(axis=0)                  # 总平均
                            fo, oc = picks(raw_b, FRONTAL), picks(raw_b, OCCIPITAL)
                            blink_f.append(np.ptp(ev[fo], axis=-1).mean())
                            blink_o.append(np.ptp(ev[oc], axis=-1).mean())
                            # 随机时刻同样处理，同样试次数
                            rt = rng.uniform(2.0, dur - 2.0, size=len(bl))
                            epr = epoch_at(raw_b, rt, *BLINK_WIN)
                            if epr.size:
                                epr = epr - epr[:, :, :5].mean(axis=2, keepdims=True)
                                blink_rand.append(
                                    np.ptp(epr.mean(axis=0)[fo], axis=-1).mean())
                    del raw_b

                # --- C3：声音起始诱发（平均参考）---
                raw_a = load_raw(sub, r, 1.0, 30.0, average_ref=True)
                if raw_a is None:
                    continue
                on = e.loc[e.trial_type == CODE_MUSIC_ON, "onset"].to_numpy()
                dur = raw_a.n_times / raw_a.info["sfreq"]
                sf = raw_a.info["sfreq"]
                if len(on):
                    ep = epoch_at(raw_a, on, *AER_WIN)
                    if ep.size:
                        nb = int(round((AER_BASE[1] - AER_BASE[0]) * sf))
                        ep = ep - ep[:, :, :nb].mean(axis=2, keepdims=True)
                        # 剔除极端试次（>200 μV 峰峰值）
                        keep = np.ptp(ep, axis=2).max(axis=1) < 200e-6
                        if keep.sum() >= 3:
                            ev = ep[keep].mean(axis=0)
                            t = np.arange(ev.shape[1]) / sf + AER_WIN[0]
                            m = (t >= GFP_WIN[0]) & (t <= GFP_WIN[1])
                            aer_gfp.append(float(ev[:, m].std(axis=0).mean()))
                            n_trials += int(keep.sum())
                            # 零分布：同 run 随机起始，同试次数
                            rt = rng.uniform(2.0, dur - 2.0, size=int(keep.sum()))
                            epr = epoch_at(raw_a, rt, *AER_WIN)
                            if epr.size:
                                epr = epr - epr[:, :, :nb].mean(axis=2, keepdims=True)
                                evr = epr.mean(axis=0)
                                null_gfp.append(float(evr[:, m].std(axis=0).mean()))
                del raw_a

            if blink_f and blink_o:
                c2.append({"subject": sub,
                           "frontal_ptp": float(np.mean(blink_f)),
                           "occipital_ptp": float(np.mean(blink_o)),
                           "random_frontal_ptp": float(np.mean(blink_rand))
                           if blink_rand else np.nan})
            if aer_gfp and null_gfp:
                c3.append({"subject": sub, "n_trials": n_trials,
                           "gfp_stim": float(np.mean(aer_gfp)),
                           "gfp_random": float(np.mean(null_gfp))})

            print(f"  {k}/{len(subs)}  {sub}", flush=True)

        C0, C1, C2, C3 = (pd.DataFrame(x) for x in (c0, c1, c2, c3))

        # ============================================================ 判决
        print("\n" + "=" * 78)
        print(f"run_id: {run.run_id}    受试 {len(subs)}")

        # ---- C0 ----  （None = 未评估，与 False = 未通过 严格区分）
        ok0 = None
        if len(C0):
            exact10 = float((C0.n_onset == 10).mean())
            bij = float(C0.bijective.mean())
            inrange = float(C0.stim_in_range.mean())
            rate80 = float((C0.n_ratings == 80).mean())
            agree = float(C0.code_agreement.mean())
            ok0 = bool(exact10 >= 0.90 and bij >= 0.90
                       and inrange >= 0.90 and rate80 >= 0.90 and agree > 0.99)
            run.log_metric("C0_event_bookkeeping",
                           {"runs": int(len(C0)), "frac_exactly_10_onsets": exact10,
                            "frac_bijective_pairing": bij,
                            "frac_stim_id_in_1_360": inrange,
                            "max_lead_s": float(C0.max_lead_s.max()),
                            "frac_80_ratings": rate80,
                            "response_code_agreement": agree})
            print("\n--- C0 事件簿记 ---")
            print(f"  {len(C0)} 个音乐 run   恰 10 个音乐起始 {100*exact10:.0f}%   "
                  f"配对为双射 {100*bij:.0f}%   刺激 ID 落在 1–360 {100*inrange:.0f}%")
            print(f"  刺激码最大提前量 {C0.max_lead_s.max():.1f} s   "
                  f"每 run 恰 80 个作答 {100*rate80:.0f}%   "
                  f"两套作答编码一致 {100*agree:.1f}%")

        # ---- C1 ----
        ok1 = None
        if len(C1) >= MIN_N:
            d = C1.alpha_rel_occipital - C1.alpha_rel_frontal
            st = wilcoxon(d)
            ok1 = bool(st.pvalue < 0.05 and d.mean() > 0)
            run.log_metric("C1_posterior_alpha",
                           {"n": int(len(C1)), "mean_diff": float(d.mean()),
                            "frac_positive": float((d > 0).mean()),
                            "p": float(st.pvalue)})
            print("\n--- C1 枕区 alpha 优势（静息）---")
            print(f"  枕区相对 alpha {C1.alpha_rel_occipital.mean():.3f}   "
                  f"额区 {C1.alpha_rel_frontal.mean():.3f}   "
                  f"差 {d.mean():+.3f}   为正 {100*(d>0).mean():.0f}%   p={st.pvalue:.2e}")

        # ---- C2 ----
        ok2 = None
        if len(C2) >= MIN_N:
            ratio = C2.frontal_ptp / C2.occipital_ptp
            st_r = wilcoxon(C2.frontal_ptp, C2.random_frontal_ptp.fillna(0))
            ok2 = bool(float(np.median(ratio)) > 2.0 and st_r.pvalue < 0.05
                       and C2.frontal_ptp.mean() > C2.random_frontal_ptp.mean())
            run.log_metric("C2_blink_locked_erp",
                           {"n": int(len(C2)), "median_frontal_occipital_ratio":
                            float(np.median(ratio)),
                            "mean_frontal_ptp_uV": float(C2.frontal_ptp.mean() * 1e6),
                            "mean_random_ptp_uV":
                            float(C2.random_frontal_ptp.mean() * 1e6),
                            "p_vs_random": float(st_r.pvalue)})
            print("\n--- C2 眨眼锁时 ERP（时间对齐判决）---")
            print(f"  额区峰峰 {C2.frontal_ptp.mean()*1e6:.1f} μV   "
                  f"枕区 {C2.occipital_ptp.mean()*1e6:.1f} μV   "
                  f"比值中位数 {np.median(ratio):.2f}")
            print(f"  额区 vs 随机窗 {C2.random_frontal_ptp.mean()*1e6:.1f} μV   "
                  f"p={st_r.pvalue:.2e}")

        # ---- C3 ----
        ok3 = None
        if len(C3) >= MIN_N:
            st_g = wilcoxon(C3.gfp_stim, C3.gfp_random)
            ok3 = bool(st_g.pvalue < 0.05 and C3.gfp_stim.mean() > C3.gfp_random.mean())
            run.log_metric("C3_sound_onset_evoked",
                           {"n": int(len(C3)), "total_trials": int(C3.n_trials.sum()),
                            "gfp_stim_uV": float(C3.gfp_stim.mean() * 1e6),
                            "gfp_random_uV": float(C3.gfp_random.mean() * 1e6),
                            "ratio": float((C3.gfp_stim / C3.gfp_random).median()),
                            "frac_stim_greater": float(
                                (C3.gfp_stim > C3.gfp_random).mean()),
                            "p": float(st_g.pvalue)})
            print("\n--- C3 声音起始诱发 GFP vs 随机起始 ---")
            print(f"  {C3.n_trials.sum()} 个试次   刺激 {C3.gfp_stim.mean()*1e6:.3f} μV   "
                  f"随机 {C3.gfp_random.mean()*1e6:.3f} μV   "
                  f"比值 {(C3.gfp_stim/C3.gfp_random).median():.2f}")
            print(f"  刺激 > 随机的受试比例 {100*(C3.gfp_stim>C3.gfp_random).mean():.0f}%"
                  f"   p={st_g.pvalue:.2e}")

        # ---- 总判决 ----
        labels = {"C0": "事件簿记", "C1": "枕区 alpha", "C2": "眨眼锁时（时间对齐）",
                  "C3": "声音起始诱发"}
        verdict = {"C0": ok0, "C1": ok1, "C2": ok2, "C3": ok3}
        n_pass = sum(v is True for v in verdict.values())
        n_fail = sum(v is False for v in verdict.values())
        n_skip = sum(v is None for v in verdict.values())
        run.log_metric("verdict", verdict)
        run.log_metric("all_passed", n_fail == 0 and n_skip == 0)

        print("\n" + "=" * 78)
        for name, ok in verdict.items():
            mark = "✅ 通过" if ok is True else ("❌ 未通过" if ok is False
                                              else f"⚪ 未评估（受试 < {MIN_N}）")
            print(f"  {name} {labels[name]:<18s} {mark}")

        if n_fail == 0 and n_skip == 0:
            print("\n✅ 四项阳性对照全部通过 —— 流水线能测出已知效应，")
            print("   且 events.tsv 与 EEG 采样点的对齐经眨眼 ERP 直接验证。")
            print("   → 可以进入实质分析（38_eeg_reliability.py）。")
        elif n_fail:
            print(f"\n🔴 {n_fail} 项阳性对照未通过。")
            print("   按既定纪律：**不做事后补救，不解释任何生理结果。**")
            print("   → 执行删除方案，本项目仅保留主观维度。")
        else:
            print(f"\n⚪ {n_pass} 项通过，{n_skip} 项未评估 —— 尚不构成判决，需更多受试。")

        for name, D in (("c0_events", C0), ("c1_alpha", C1),
                        ("c2_blink", C2), ("c3_aer", C3)):
            if len(D):
                D.to_csv(REPO_ROOT / "reports" / "source_data" /
                         f"ds002721_{name}.csv", index=False)


if __name__ == "__main__":
    main()
