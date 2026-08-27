r"""`93` 的双通道并排，改为**配平 n + 逐格零分布**。

    .venv/Scripts/python scripts/99f_channel_matched.py

## `93` 问的问题是对的，尺子是错的

`93` 要回答：「个体差异测不出」是**规模结论**还是**通道结论**。
做法是把生理侧（PMEmo 皮电）与主观侧（DEAM 评分）并排报 τ̂。

但两侧的每人观测数不同 —— PMEmo 18，DEAM 44。
而 τ̂ 在小 n 下系统性偏高（`99d`：真值为零时 n=18 报 0.105，n=44 报 0.039）。
**并排的两个数根本不在同一把尺子上**，且偏差恰好偏向生理侧看起来更好。

## 本脚本的三处改动

**① 配平 n。** 全部下采样到 18 —— 生理侧的天花板。
同一把尺子，才谈得上「哪个通道更强」。

**② 零分布逐格算，不用全局常数。** `99d` 的模拟用 x~N(0,1)、残差 SD 0.95，
是个代表性设定，不是每一格的真实设定。这里改为**参数化自举**：
取该格**真实的** x 向量与残差 SD，令 τ=0（所有人共用一个 β）生成 y，
重估 τ̂，重复若干次 —— 得到这一格自己的 λ=0 行。

**③ 加第三行：ARAUS，同样下采样到 18。** 它是产品域的主观数据。
若在同一个 n 下，ARAUS 的 τ 高出零分布而音乐语料不高，
那就是**域**的差别，与规模干净地分开了 —— 这是 `93` 当时做不到的。

## 判读

    三行都不超零分布   → 规模结论。18 次/人对任何通道任何域都不够
    只有主观超         → 通道结论。声学读得到共识，读不到身体
    只有产品域超       → 域结论。门槛随域变化，音乐语料低估了产品域
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")

FEAT = REPO_ROOT / "data" / "features"
AR = REPO_ROOT / "data" / "raw" / "ARAUS" / "datav2"
DEAM_CSV = REPO_ROOT / "data" / "raw" / "DEAM" / "annotations" / \
    "annotations per each rater" / "song_level" / \
    "static_annotations_songs_1_2000.csv"
OUT = REPO_ROOT / "reports" / "source_data" / "channel_matched.csv"
SEED = 20260803
CHI2_95 = chi2.ppf(0.95, 1)
N_MATCH = 18                  # 生理侧的天花板，三行一律配平到它
N_NULL = 40                   # 逐格零分布的自举次数
N_SUB = 12                    # 下采样重复次数
DESCRIPTORS = ["rms_mean", "onset_env_mean", "chroma_flux_mean",
               "roughness_pl_mean", "mode_score"]
EDA = ["scr_rate", "phasic_mean", "scl_slope"]


def slope_se(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    n = len(x)
    if n < 4 or np.std(x) < 1e-9:
        return np.nan, np.nan
    xc = x - x.mean()
    sxx = float((xc ** 2).sum())
    b = float((xc * (y - y.mean())).sum() / sxx)
    resid = y - y.mean() - b * xc
    return b, float(np.sqrt(float((resid ** 2).sum() / (n - 2)) / sxx))


def _ll(t2: float, b: np.ndarray, v: np.ndarray) -> float:
    w = 1.0 / (v + t2)
    mu = float((w * b).sum() / w.sum())
    ll = -0.5 * float((np.log(2 * np.pi / w) + w * (b - mu) ** 2).sum())
    return ll - 0.5 * np.log(w.sum())


def tau_ci(b: np.ndarray, se: np.ndarray, grid_n: int = 1500):
    v = se ** 2
    hi_b = max(float(np.var(b, ddof=1)) * 20, 1e-6)
    grid = np.concatenate([[0.0], np.geomspace(1e-10, hi_b, grid_n)])
    lls = np.array([_ll(t, b, v) for t in grid])
    k = int(np.argmax(lls))
    t2, lm = float(grid[k]), float(lls[k])

    def g(x: float) -> float:
        return 2.0 * (lm - _ll(x, b, v)) - CHI2_95

    lo = brentq(g, 0.0, t2) if (k > 0 and g(0.0) > 0) else 0.0
    hi = brentq(g, t2, hi_b) if g(hi_b) > 0 else hi_b
    w = 1.0 / (v + t2)
    return (float(np.sqrt(t2)), float(np.sqrt(lo)), float(np.sqrt(hi)),
            float((w * b).sum() / w.sum()))


def fit(u: list[tuple[np.ndarray, np.ndarray]], grid_n: int = 1500):
    bs, ses, rsd = [], [], []
    for x, y in u:
        b, se = slope_se(x, y)
        if np.isfinite(b) and np.isfinite(se) and se > 0:
            bs.append(b)
            ses.append(se)
            rsd.append(float(np.std(y - b * (x - x.mean()))))
    if len(bs) < 30:
        return None
    tau, lo, hi, beta = tau_ci(np.array(bs), np.array(ses), grid_n)
    return tau, lo, hi, beta, float(np.median(rsd)), len(bs)


def subsample(u, n, rng):
    out = []
    for x, y in u:
        if len(x) < n:
            continue
        i = rng.choice(len(x), n, replace=False)
        out.append((x[i], y[i]))
    return out


def cell_null(u, beta: float, resid_sd: float, rng) -> tuple[float, float]:
    r"""这一格自己的 λ=0 行：用**真实的 x** 与残差 SD，令 τ=0 生成 y。

    比全局常数可信 —— x 的分布、每人的观测数、残差量级都照搬本格。

    同时返回中位数的**蒙特卡洛标准误**。首版把 `τ̂ > null` 当成一个点比较，
    于是 `phasic_mean ~ mode_score` 以 0.0009 的margin被判为「超零分布」——
    那远小于 N_NULL 次抽样的自身误差。超没超要看 margin 是否大于这个误差。
    """
    # 🔴 模拟出来的 y **必须走一遍真实管线里的被试内 z 分**（`zscore()`）。
    # 少了「除以一个估计出来的 SD」这一步，零分布偏低约 14%。
    # 反直觉之处：理想模拟（x~N(0,1)、各人同分布）上 z 分是**降低**零分布的
    # （方差稳定，比 0.567）；但真实数据里每人 var(x) 差别很大，
    # z 分等于把 b_i 除以一个与 var(x) 相关的个人量，反而**升高**（比 1.06–1.33）。
    # 同质假设会把这条机制抹掉 —— 检验估计器的模拟必须保留真实的异质性。
    taus = []
    for _ in range(N_NULL):
        sim = []
        for x, _ in u:
            y = beta * (x - x.mean()) + rng.normal(0, resid_sd, len(x))
            y = (y - y.mean()) / (y.std() + 1e-9)
            sim.append((x, y))
        r = fit(sim, grid_n=600)
        if r:
            taus.append(r[0])
    if not taus:
        return np.nan, np.nan
    a = np.array(taus)
    # 中位数的 MC 标准误 ≈ 1.253 · sd / √m
    return float(np.median(a)), float(1.253 * a.std(ddof=1) / np.sqrt(len(a)))


def zscore(D, unit, ys, xs):
    D = D.copy()
    for y in ys:
        g = D.groupby(unit)[y]
        D[y] = (D[y] - g.transform("mean")) / (g.transform("std") + 1e-9)
    for x in xs:
        D[x] = (D[x] - D[x].mean()) / (D[x].std() + 1e-9)
    return D


def cells_pmemo():
    T = pd.read_parquet(FEAT / "pmemo_eda_song_level.parquet")
    fa = pd.read_parquet(FEAT / "layer_a_reg_pmemo_ln.parquet")
    fd = pd.read_parquet(FEAT / "layer_d_reg_pmemo_ln.parquet")
    for f in (fa, fd):
        f["musicId"] = (f.clip_id.astype(str)
                        .str.removeprefix("pmemo_").astype(int))
    F = fa.merge(fd.drop(columns=["clip_id"]), on="musicId", suffixes=("", "_d"))
    D = zscore(T.merge(F, on="musicId"), "listener", EDA, DESCRIPTORS)
    for m in EDA:
        for d in DESCRIPTORS:
            u = [(h[d].to_numpy(float), h[m].to_numpy(float))
                 for _, h in D.dropna(subset=[d, m]).groupby("listener")
                 if len(h) >= N_MATCH]
            if len(u) >= 30:
                yield "生理 · 音乐（PMEmo 皮电）", f"{m} ~ {d}", u


def cells_deam():
    R = pd.read_csv(DEAM_CSV, skipinitialspace=True)
    R.columns = [c.strip() for c in R.columns]
    A = pd.read_parquet(FEAT / "layer_a_reg_deam_ln.parquet")
    Dl = pd.read_parquet(FEAT / "layer_d_reg_deam_ln.parquet")
    F = A.merge(Dl, on="clip_id", suffixes=("", "_d"))
    F["SongId"] = F.clip_id.astype(str).str.removeprefix("deam_").astype(int)
    M = zscore(R.merge(F[["SongId"] + DESCRIPTORS], on="SongId"),
               "workerID", ["Arousal", "Valence"], DESCRIPTORS)
    for t in ("Arousal", "Valence"):
        for d in DESCRIPTORS:
            u = [(h[d].to_numpy(float), h[t].to_numpy(float))
                 for _, h in M.dropna(subset=[d, t]).groupby("workerID")
                 if len(h) >= N_MATCH]
            if len(u) >= 30:
                yield "主观 · 音乐（DEAM 评分）", f"{t} ~ {d}", u


def cells_araus():
    P = ["Navg_r", "Savg_r", "Ravg_r", "Favg_r"]
    A = pd.read_csv(AR / "responses.csv")
    # 🔴 剔除 fold_r=−1 的锚点：每人 2 行、同一声景+静默+smr 0、位 1 与 45，
    # 在 Navg_r 第 93 百分位且 749 人完全相同 —— 一个共享的重复高杠杆设计点，
    # 会把所有人的斜率往同一处钉。留着它 τ 变动最大 21.5%（`99` 步骤 4）。
    A = A[(A.is_attention == 0) & (A.fold_r != -1)]
    A = zscore(A, "participant", ["pleasant", "eventful"], P)
    for t in ("pleasant", "eventful"):
        for d in P:
            u = [(h[d].to_numpy(float), h[t].to_numpy(float))
                 for _, h in A.dropna(subset=[d, t]).groupby("participant")
                 if len(h) >= N_MATCH]
            if len(u) >= 30:
                yield "主观 · 声景（ARAUS，产品域）", f"{t} ~ {d}", u


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("99f_channel_matched", SEED)
    rows = []
    print("=" * 100)
    print(f"三行一律配平到每人 n = {N_MATCH}（生理侧的天花板），"
          f"零分布逐格用参数化自举算")
    print(f"  {'通道 · 域':<26}{'格':<26}{'人':>5}{'τ̂':>8}"
          f"{'本格零分布':>11}{'margin':>9}{'MC误差':>8}{'τ 改正':>8}{'超零?':>7}")
    for gen in (cells_pmemo, cells_deam, cells_araus):
        for chan, cell, u in gen():
            taus, nulls, ses, betas = [], [], [], []
            for _ in range(N_SUB):
                s = subsample(u, N_MATCH, rng)
                r = fit(s)
                if not r:
                    continue
                nl, mc = cell_null(s, r[3], r[4], rng)
                taus.append(r[0])
                betas.append(r[3])
                nulls.append(nl)
                ses.append(mc)
            if not taus:
                continue
            tau = float(np.median(taus))
            null = float(np.nanmedian(nulls))
            mcse = float(np.nanmedian(ses))
            corr = float(np.sqrt(max(tau ** 2 - null ** 2, 0.0)))
            margin = tau - null
            # 「超零分布」须 margin 大于 2 倍 MC 误差 —— 点比较会把
            # 0.0009 的差判成阳性，而那小于抽样自身的噪声。
            over = bool(margin > 2 * mcse)
            print(f"  {chan:<26}{cell:<26}{len(u):>5}{tau:>8.4f}"
                  f"{null:>11.4f}{margin:>+9.4f}{mcse:>8.4f}{corr:>8.4f}"
                  f"{'  ✅' if over else '  —':>7}", flush=True)
            rows.append({"channel": chan, "cell": cell, "n_units": len(u),
                         "n_per_unit": N_MATCH, "tau": tau, "null": null,
                         "margin": margin, "mc_se": mcse,
                         "tau_corrected": corr, "above_null": over,
                         "beta": float(np.median(betas))})

    R = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("按通道 · 域汇总（全部 n=18，同一把尺子）")
    print(f"  {'通道 · 域':<28}{'格数':>5}{'超零分布':>10}"
          f"{'τ 改正 中位':>13}{'τ 改正 最大':>13}")
    for ch, g in R.groupby("channel"):
        print(f"  {ch:<28}{len(g):>5}{f'{int(g.above_null.sum())}/{len(g)}':>10}"
              f"{g.tau_corrected.median():>13.4f}{g.tau_corrected.max():>13.4f}")
        run.log_metric(f"above_null_{ch}", int(g.above_null.sum()))
        run.log_metric(f"tau_corr_median_{ch}", float(g.tau_corrected.median()))

    print("\n" + "=" * 100)
    print("判读 —— `93` 问的是「规模结论还是通道结论」，现在能答了")
    frac = R.groupby("channel").above_null.mean()
    # 🔴 `prod` 必须从 `subj` 里排掉。ARAUS 那一行叫「主观 · 声景（ARAUS，产品域）」，
    # 同时满足两个条件 ⇒ prod ⊆ subj ⇒ d_hi ≤ s_hi 恒成立 ⇒
    # 「域结论」那个分支**永远不可能触发**。docstring 承诺了三种判读，
    # 首版只有两种能出现。
    prod = [k for k in frac.index if "产品域" in k]
    subj = [k for k in frac.index if k.startswith("主观") and k not in prod]
    phys = [k for k in frac.index if k.startswith("生理")]
    s_hi = max((frac[k] for k in subj), default=0.0)
    p_hi = max((frac[k] for k in phys), default=0.0)
    d_hi = max((frac[k] for k in prod), default=0.0)
    if s_hi < 0.3 and p_hi < 0.3 and d_hi < 0.3:
        print("  → **规模结论**：n=18 对任何通道、任何域都不够。")
        print("    `93` 的原判成立，且现在是在同一把尺子上成立的。")
    elif s_hi >= 0.3 and p_hi < 0.3:
        print("  → **通道结论**：同规模下主观侧超得出零分布，生理侧超不出。")
        print("    「声学读得到共识，读不到身体」得到配平后的支持。")
    elif d_hi >= 0.3 and s_hi < 0.3:
        print("  → **域结论**：只有产品域超得出。音乐语料低估了产品域的个体差异，")
        print("    门槛必须按域分行报，不能给一个常数。")
    else:
        print("  → 混合：见上表逐行判读，正文不得简化成一句话。")
    print(f"\n  各通道超零分布的比例：",
          "   ".join(f"{k} {v:.0%}" for k, v in frac.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT, index=False, encoding="utf-8")
    run.write()
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
