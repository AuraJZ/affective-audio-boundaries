"""ds002721 脑电响应的刺激层面信度 —— B2 原则用在生理上。

    .venv/Scripts/python scripts/39_eeg_reliability.py --n-subjects 31

## 前置条件（已满足）

| 对照 | 脚本 | 结果 |
|---|---|---|
| 事件簿记 | `37` C0 | 恰 10 起始 / 双射配对 / 刺激 ID 全在 1–360，**均 100%** |
| 通道与频谱 | `37` C1 | 枕区相对 alpha 0.316 vs 额区 0.052，**31/31 为正**，p=1.9e-09 |
| **时间对齐** | `37` C2 | 眨眼锁时额区 163.8 μV vs 随机 25.8 μV，**p=9.3e-10** |
| **刺激锁时响应** | `43` | 452 ms 处 t=8.78（绝对值），符号翻转 **p=0.0001**；基线段 p=0.54 |

C2 是 BIRAFFE2 始终无法确立的那一项 —— 眨眼幅度百微伏级，
只有 `events.tsv` 的 onset 真正索引到 EDF 采样点时，额区总平均才会出现该偏转。

⚠️ **须写进任何报告**：诱发响应是 ~450 ms 的晚成分，**没有经典听觉 N1**。
刺激材料本身有锐利起始（`42`：中位 25 ms 达半幅，60% 在 50 ms 内），
故最可能的解释是播放延迟抖动抹掉了快成分。这限制了可做的时间分辨分析，
但不影响本脚本关心的**试次级带功率**。

## 问题

「声学能不能预测生理」之前，必须先问：**生理响应本身有没有可复现的刺激层面成分？**
若目标变量是噪声，任何预测器都到不了，而 ρ≈0 会被误读成「特征不够好」。

这是本项目反复兑现的 B2 原则，也是 BIRAFFE2 那次唯一做对的事
（做对了信度检验，却漏了阳性对照）。

## 设计约束（由 `diag_design` 实测）

刺激分配严重不均衡：307 段音频、1240 试次，**每段中位仅 2 名受试**，
仅 44 段 ≥8 人。两两受试的刺激集重叠中位 6 段。

因此用两条互补的路径，而不是只用一条：

| 方法 | 用到的数据 | 与既有结果可比性 |
|---|---|---|
| **分半信度** | 仅核心子集（≥8 人的 44 段） | ✅ 与评分 0.868 / EDA 0.155 同一算法 |
| **单因素随机效应 ICC** | **全部 1240 试次** | 统计上适配不均衡设计，效率高 |

不均衡设计下 ICC 是正确工具；分半只是为了与既有数字对齐。**两者须一致**，
不一致则说明估计不稳，不下结论。

## 脑电响应量

逐试次，音乐播放段 [1, 14] s 相对基线 [-2, 0] s 的**相对带功率变化**：
5 个频带（δ/θ/α/β/γ）× 5 个区域（额/中央/颞/顶/枕）= 25 维，
外加**额区 alpha 不对称**（F4−F3，经典效价相关量）。

## 零分布

ICC 的显著性由**被试内打乱刺激标签**的置换零分布给出 ——
保留每名受试的响应分布与试次数，只破坏「哪段音频」这一信息。
"""

from __future__ import annotations

import argparse
import warnings

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

ROOT = REPO_ROOT / "data" / "raw" / "ds002721"
MUSIC_RUNS = (2, 3, 4, 5)
CODE_MUSIC_ON, STIM_LO, STIM_HI = 788, 301, 660
CODE_Q, CODE_A = range(800, 808), range(833, 842)

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
REGIONS = {"frontal": ("FP1", "FP2", "F7", "F3", "FZ", "F4", "F8"),
           "central": ("C3", "CZ", "C4"),
           "temporal": ("T3", "T4", "T5", "T6"),
           "parietal": ("P3", "PZ", "P4"),
           "occipital": ("O1", "O2")}
QUESTIONS = ["pleasant", "energetic", "tense", "angry",
             "afraid", "happy", "sad", "tender"]

BASE_WIN = (-2.0, 0.0)
RESP_WIN = (1.0, 14.0)
DS_HZ = 250.0
N_PERM = 500
N_SPLIT = 400
MIN_SUBS_SPLIT = 8            # 分半子集：每段音频至少这么多受试
CACHE = REPO_ROOT / "data" / "features" / "ds002721_trials_raw.parquet"


# ------------------------------------------------------------------ 载入
def load_events(sub: str, run: int) -> pd.DataFrame | None:
    p = ROOT / sub / "eeg" / f"{sub}_task-run{run}_events.tsv"
    if not p.exists():
        return None
    e = pd.read_csv(p, sep="\t")
    e["trial_type"] = pd.to_numeric(e["trial_type"], errors="coerce")
    return e.dropna(subset=["trial_type", "onset"])


def trials_of(e: pd.DataFrame) -> list[tuple[float, int]]:
    """(音乐起始秒, 刺激 1–360)；刺激码取最后一个不晚于起始者。"""
    on = np.sort(e.loc[e.trial_type == CODE_MUSIC_ON, "onset"].to_numpy())
    st = e[e.trial_type.between(STIM_LO, STIM_HI)].sort_values("onset")
    sv, sc = st.onset.to_numpy(), st.trial_type.to_numpy().astype(int)
    out = []
    for t in on:
        j = np.searchsorted(sv, t, side="right") - 1
        if j >= 0:
            out.append((float(t), int(sc[j]) - 300))
    return out


def ratings_of(e: pd.DataFrame, onsets: list[float],
               family: str) -> dict[int, dict[str, int]]:
    """逐试次 8 项评分，按作答时刻归属到最近的先前音乐起始。

    `events.json` 给了两套作答编码：`833–841`「Response 01–09」与
    `901–909`「Answer 01–09」，描述上都是 1–9 的作答值。但实测两者仅
    **7.3%** 一致（`41` 的 C0），至多一套是对的。

    `family` 选用哪一套。由 `validate_rating_family()` 用**独立评分研究的
    组均值**判定 —— 正确的那套应与组均值显著正相关，错的那套不应。
    这是让数据回答编码歧义，而不是靠读文档猜。
    """
    lo = 832 if family == "response" else 900
    codes = CODE_A if family == "response" else range(901, 910)
    resp = e[e.trial_type.isin(codes)]
    out: dict[int, dict[str, int]] = {}
    on = np.asarray(onsets)
    for t, v in zip(resp.onset.to_numpy(), resp.trial_type.to_numpy().astype(int)):
        same = e[np.isclose(e.onset, t)]
        q = same[same.trial_type.isin(CODE_Q)].trial_type
        if not len(q):
            continue
        prev = np.where(on <= t)[0]
        if not len(prev):
            continue
        out.setdefault(int(prev[-1]), {})[QUESTIONS[int(q.iloc[0]) - 800]] = v - lo
    return out


# 受试自评的 8 个问题 ↔ 独立评分研究的 8 个量表
RATING_ANCHOR = {"pleasant": "valence", "energetic": "energy", "tense": "tension",
                 "angry": "anger", "afraid": "fear", "happy": "happy",
                 "sad": "sad", "tender": "tender"}


def validate_rating_family(D: pd.DataFrame, anchors: pd.DataFrame,
                           prefix: str, family: str) -> dict[str, float]:
    """用独立组均值检验某套编码是否为真作答。

    Eerola & Vuoskoski (2011) 的 360 段均值评分与本数据集的受试完全无关。
    若编码解读正确，受试自评应与之显著正相关；解读错误则应接近零。

    `prefix` 为受试自评列的前缀（`""` 或 `"alt_"`）。锚点列一律改名为
    `ref_*` 再合并 —— 两侧有 `happy`/`sad`/`tender` 三个同名列，
    若靠 merge 的 suffix 处理，`RATING_ANCHOR` 会取到受试自己的那一列，
    把自相关（ρ=1）当成验证通过。
    """
    A = anchors.rename(columns={c: f"ref_{c}" for c in anchors.columns
                                if c != "stim"})
    M = D.merge(A, on="stim", how="left")
    out: dict[str, float] = {}
    for q, ref in RATING_ANCHOR.items():
        col, rcol = f"{prefix}{q}", f"ref_{ref}"
        if col not in M.columns or rcol not in M.columns:
            continue
        s = M[[col, rcol]].dropna()
        if len(s) >= 100 and s[col].nunique() > 1:
            out[q] = float(spearmanr(s[col].to_numpy(),
                                     s[rcol].to_numpy()).statistic)
    vals = [v for k, v in out.items() if not k.startswith("_")]
    out["_mean"] = float(np.mean(vals)) if vals else np.nan
    out["_n_axes"] = float(len(vals))
    return out


def eeg_features(raw: mne.io.BaseRaw, onset: float) -> dict[str, float] | None:
    """相对带功率的响应−基线变化 + 额区 alpha 不对称。"""
    sf = raw.info["sfreq"]
    x = raw.get_data()
    n = x.shape[1]

    def seg(w):
        a, b = int(round((onset + w[0]) * sf)), int(round((onset + w[1]) * sf))
        return x[:, a:b] if 0 <= a < b <= n else None

    r, b = seg(RESP_WIN), seg(BASE_WIN)
    if r is None or b is None or b.shape[1] < sf:
        return None

    def relpow(y):
        f, p = welch(y, fs=sf, nperseg=int(sf * 2), axis=-1)
        tot = np.trapezoid(p[:, (f >= 1) & (f <= 45)], f[(f >= 1) & (f <= 45)], axis=-1)
        out = {}
        for name, (lo, hi) in BANDS.items():
            m = (f >= lo) & (f <= hi)
            out[name] = np.trapezoid(p[:, m], f[m], axis=-1) / np.where(tot > 0, tot, np.nan)
        return out

    pr, pb = relpow(r), relpow(b)
    ch = {c: i for i, c in enumerate(raw.ch_names)}
    d: dict[str, float] = {}
    for band in BANDS:
        for reg, names in REGIONS.items():
            idx = [ch[c] for c in names if c in ch]
            if idx:
                d[f"{band}_{reg}"] = float(np.nanmean(pr[band][idx])
                                           - np.nanmean(pb[band][idx]))
    if "F3" in ch and "F4" in ch:
        def faa(p):
            return float(np.log(max(p["alpha"][ch["F4"]], 1e-12))
                         - np.log(max(p["alpha"][ch["F3"]], 1e-12)))
        d["faa"] = faa(pr) - faa(pb)
    return d


# ------------------------------------------------- 信度：ICC 与分半
def icc_oneway(values: np.ndarray, groups: np.ndarray) -> float:
    """单因素随机效应 ICC（不均衡设计的标准矩估计）。"""
    ok = np.isfinite(values)
    values, groups = values[ok], groups[ok]
    uniq, inv = np.unique(groups, return_inverse=True)
    k, N = len(uniq), len(values)
    if k < 3 or N <= k:
        return np.nan
    counts = np.bincount(inv)
    gm = values.mean()
    gmeans = np.bincount(inv, weights=values) / counts
    ssb = float((counts * (gmeans - gm) ** 2).sum())
    ssw = float(((values - gmeans[inv]) ** 2).sum())
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (counts ** 2).sum() / N) / (k - 1)
    if n0 <= 0:
        return np.nan
    va = (msb - msw) / n0
    return float(va / (va + msw)) if (va + msw) > 0 else np.nan


def split_half(D: pd.DataFrame, col: str, rng: np.random.Generator,
               min_subs: int) -> float:
    """跨受试分半 + Spearman-Brown —— 与评分/EDA 用的是同一套算法。"""
    per = D.groupby("stim")["subject"].nunique()
    S = D[D.stim.isin(per[per >= min_subs].index)]
    if S.stim.nunique() < 10:
        return np.nan
    subs = S.subject.unique()
    raw = []
    for _ in range(N_SPLIT):
        perm = rng.permutation(subs)
        a = set(perm[: len(perm) // 2])
        ga = S[S.subject.isin(a)].groupby("stim")[col].agg(["mean", "size"])
        gb = S[~S.subject.isin(a)].groupby("stim")[col].agg(["mean", "size"])
        common = ga.index[(ga["size"] >= 2)].intersection(gb.index[(gb["size"] >= 2)])
        if len(common) >= 10:
            r = spearmanr(ga.loc[common, "mean"], gb.loc[common, "mean"],
                          nan_policy="omit").statistic
            if np.isfinite(r):
                raw.append(float(r))
    if not raw:
        return np.nan
    r = np.array(raw)
    return float(np.mean(2 * r / (1 + r)))


def perm_p(D: pd.DataFrame, col: str, obs: float,
           rng: np.random.Generator) -> tuple[float, float]:
    """被试内打乱刺激标签的 ICC 零分布。"""
    null = []
    v, g, s = D[col].to_numpy(), D.stim.to_numpy(), D.subject.to_numpy()
    for _ in range(N_PERM):
        gp = g.copy()
        for u in np.unique(s):
            m = s == u
            gp[m] = rng.permutation(gp[m])
        x = icc_oneway(v, gp)
        if np.isfinite(x):
            null.append(x)
    if not null or not np.isfinite(obs):
        return np.nan, np.nan
    null = np.array(null)
    return float((np.sum(null >= obs) + 1) / (len(null) + 1)), float(null.mean())


# ------------------------------------------------------------------ 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=31)
    ap.add_argument("--reuse", action="store_true",
                    help="用已缓存的试次特征，跳过 EDF 重抽（约 20 分钟）")
    args = ap.parse_args()

    subs = sorted(d.name for d in ROOT.glob("sub-*") if d.is_dir())[: args.n_subjects]

    with RunRecord(
        "ds002721_eeg_reliability", seed=SEED,
        params={"dataset": "OpenNeuro ds002721", "n_subjects": len(subs),
                "response_win_s": RESP_WIN, "baseline_win_s": BASE_WIN,
                "bands": {k: list(v) for k, v in BANDS.items()},
                "n_perm": N_PERM, "n_split": N_SPLIT,
                "min_subjects_for_split_half": MIN_SUBS_SPLIT,
                "prerequisite": "37_eeg_positive_controls 四项全过"},
    ) as run:
        rng = np.random.default_rng(SEED)

        if args.reuse and CACHE.exists():
            D = pd.read_parquet(CACHE)
            print(f"  复用缓存 {CACHE.name}（{len(D)} 试次，{D.subject.nunique()} 受试）")
            analyse(D, run, rng)
            return

        rows = []
        for k, sub in enumerate(subs, 1):
            for r in MUSIC_RUNS:
                e = load_events(sub, r)
                if e is None:
                    continue
                tr = trials_of(e)
                if not tr:
                    continue
                p = ROOT / sub / "eeg" / f"{sub}_task-run{r}_eeg.edf"
                if not p.exists():
                    continue
                try:
                    raw = mne.io.read_raw_edf(p, preload=True, verbose=False)
                except Exception:                            # noqa: BLE001
                    continue
                raw.rename_channels({c: c.strip().upper() for c in raw.ch_names})
                raw.notch_filter(50.0, verbose=False)
                raw.filter(1.0, 45.0, verbose=False)
                raw.set_eeg_reference("average", verbose=False)
                if raw.info["sfreq"] > DS_HZ:
                    raw.resample(DS_HZ, verbose=False)

                onsets = [t for t, _ in tr]
                r_resp = ratings_of(e, onsets, "response")
                r_ans = ratings_of(e, onsets, "answer")
                for i, (onset, stim) in enumerate(tr):
                    f = eeg_features(raw, onset)
                    if f is None:
                        continue
                    rows.append({
                        "subject": sub, "run": r, "trial": i, "stim": stim, **f,
                        **r_resp.get(i, {}),
                        **{f"alt_{k}": v for k, v in r_ans.get(i, {}).items()}})
                del raw
            print(f"  {k}/{len(subs)}  {sub}", flush=True)

        D = pd.DataFrame(rows)

        # 脑电抽取是本脚本最贵的一步（31 人 × 4 run × 1 kHz EDF），
        # 先落盘再做后续统计 —— 后面任何一步崩掉都不必重抽。
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        D.to_parquet(CACHE)
        print(f"  → 缓存 {CACHE.name}（{len(D)} 试次）")

        analyse(D, run, rng)


def analyse(D: pd.DataFrame, run: RunRecord,
            rng: np.random.Generator) -> None:
    """判定作答编码 → 计算信度。与抽取分离，便于用缓存重跑。"""
    # ---------- 先判定作答编码，再谈信度 ----------
    A = pd.read_csv(REPO_ROOT / "data" / "reg_soundtracks.csv")[
        ["stim", "valence", "energy", "tension",
         "anger", "fear", "happy", "sad", "tender"]]
    v_resp = validate_rating_family(D, A, "", "response")
    v_ans = validate_rating_family(D, A, "alt_", "answer")
    run.log_metric("rating_family_validation",
                   {"response_833_841": v_resp, "answer_901_909": v_ans})
    chosen = ("response" if (v_resp["_mean"] >= v_ans["_mean"]
                             or not np.isfinite(v_ans["_mean"]))
              else "answer")
    run.log_metric("rating_family_chosen", chosen)
    print("\n--- 作答编码判定（对照独立评分研究的组均值）---")
    print(f"  833–841 「Response」  与组均值平均 ρ = {v_resp['_mean']:+.3f}")
    print(f"  901–909 「Answer」    与组均值平均 ρ = {v_ans['_mean']:+.3f}")
    print(f"  → 采用 {chosen}")
    for q in QUESTIONS:
        if q in v_resp or q in v_ans:
            print(f"     {q:10s} response {v_resp.get(q, float('nan')):+.3f}"
                  f"   answer {v_ans.get(q, float('nan')):+.3f}")
    if chosen == "answer":
        D = D.drop(columns=[q for q in QUESTIONS if q in D.columns]).rename(
            columns={f"alt_{q}": q for q in QUESTIONS
                     if f"alt_{q}" in D.columns})
    eeg_cols = [c for c in D.columns
                if any(c.startswith(b) for b in BANDS) or c == "faa"]
    rat_cols = [c for c in QUESTIONS if c in D.columns]
    D = D.drop(columns=[c for c in D.columns if c.startswith("alt_")])

    # 被试内 z 标准化 —— 去掉个体基线差异，只留「哪段音频」的成分
    for c in eeg_cols + rat_cols:
        g = D.groupby("subject")[c]
        D[f"z_{c}"] = (D[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)

    run.log_metric("n_trials", int(len(D)))
    run.log_metric("n_stimuli", int(D.stim.nunique()))
    run.log_metric("n_eeg_features", len(eeg_cols))
    run.log_metric("rating_completeness",
                   {c: float(D[c].notna().mean()) for c in rat_cols})

    # ---------------- 主观评分（同一批受试、同一批试次）= 对照上限 ----------------
    print("\n" + "=" * 84)
    print(f"run_id: {run.run_id}   试次 {len(D):,}   刺激 {D.stim.nunique()}   "
          f"受试 {D.subject.nunique()}")

    res = []
    for c in rat_cols + eeg_cols:
        sub_d = D.dropna(subset=[f"z_{c}"])
        if len(sub_d) < 100:
            continue
        icc = icc_oneway(sub_d[f"z_{c}"].to_numpy(), sub_d.stim.to_numpy())
        sh = split_half(sub_d, f"z_{c}", rng, MIN_SUBS_SPLIT)
        res.append({"measure": c,
                    "kind": "rating" if c in rat_cols else "eeg",
                    "n": int(len(sub_d)), "icc": icc, "split_half_sb": sh})
    R = pd.DataFrame(res)

    # 置换检验只对最强的几个跑（每个 500 次置换，代价高）
    top = pd.concat([R[R.kind == "rating"],
                     R[R.kind == "eeg"].nlargest(5, "icc")])
    pvals = {}
    for c in top.measure:
        sub_d = D.dropna(subset=[f"z_{c}"])
        obs = float(R.loc[R.measure == c, "icc"].iloc[0])
        p, nm = perm_p(sub_d, f"z_{c}", obs, rng)
        pvals[c] = {"p": p, "null_mean": nm}
    R["perm_p"] = R.measure.map(lambda c: pvals.get(c, {}).get("p"))
    R["null_icc"] = R.measure.map(lambda c: pvals.get(c, {}).get("null_mean"))

    R = R.sort_values(["kind", "icc"], ascending=[True, False])
    run.log_metric("reliability_table", R.to_dict("records"))
    R.to_csv(REPO_ROOT / "reports" / "source_data" /
             "ds002721_eeg_reliability.csv", index=False)
    D.to_parquet(REPO_ROOT / "data" / "features" / "ds002721_trials.parquet")

    print("\n--- 主观评分（同批受试同批试次，作为可比上限）---")
    for r_ in R[R.kind == "rating"].itertuples():
        print(f"  {r_.measure:12s} n={r_.n:5d}  ICC {r_.icc:+.3f}  "
              f"分半SB {r_.split_half_sb:+.3f}  "
              f"p={r_.perm_p if r_.perm_p is None else f'{r_.perm_p:.4f}'}")

    print("\n--- 脑电响应（按 ICC 排序，前 10）---")
    for r_ in R[R.kind == "eeg"].head(10).itertuples():
        p_ = "" if r_.perm_p is None or not np.isfinite(r_.perm_p) else \
            f"  p={r_.perm_p:.4f}  零分布均值 {r_.null_icc:+.3f}"
        print(f"  {r_.measure:18s} n={r_.n:5d}  ICC {r_.icc:+.3f}  "
              f"分半SB {r_.split_half_sb:+.3f}{p_}")

    best = R[R.kind == "eeg"].iloc[0] if (R.kind == "eeg").any() else None
    best_rat = R[R.kind == "rating"].iloc[0] if (R.kind == "rating").any() else None
    if best is not None and best_rat is not None:
        print("\n" + "=" * 84)
        print(f"最强脑电量 {best.measure}：ICC {best.icc:+.3f}")
        print(f"最强主观量 {best_rat.measure}：ICC {best_rat.icc:+.3f}")
        run.log_metric("best_eeg", {"measure": best.measure, "icc": float(best.icc)})
        run.log_metric("best_rating",
                       {"measure": best_rat.measure, "icc": float(best_rat.icc)})


if __name__ == "__main__":
    main()
