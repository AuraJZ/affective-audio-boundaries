r"""PMEmo 个体差异检验的**灵敏度**：多大的个体差异才捞得出来。

    .venv/Scripts/python scripts/77_pmemo_individual_sensitivity.py

## 为什么必须做这个

`76` 的分半检验给出接近零的结果（最好 r = +0.077, p = 0.070）。
按本项目一贯的纪律，**报告零结果之前先问它有没有功效** ——
否则「没找到个体差异」与「个体差异不存在」会被混为一谈，
而这正是 `step16`/`step18` 反复栽过的地方。

## 做法：往真实结构里注入已知大小的个体差异

保留真实的一切 —— 谁听了哪些歌、这些歌的真实声学值、皮电的真实噪声量级 ——
只把结果变量换成合成的：

    y_ij = (β + u_i) · x_ij + ε_ij ,   u_i ~ N(0, σ_u²)

σ_u 即**个体间斜率标准差**，是要标定的量。ε 的量级取自真实残差。
对每个 σ_u 跑与 `76` 完全相同的分半检验，看它给出什么 r、以多大概率显著。

## 输出的是一个上界，不是一个阴性

结果形如「若个体间斜率标准差达到 X，本检验有 80% 把握测出；
实测未达显著，故个体差异**若存在也不超过 X**」。
这是可证伪的定量陈述，比「未发现个体差异」有用得多。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")

FEAT = REPO_ROOT / "data" / "features"
OUT = REPO_ROOT / "reports" / "source_data" / "pmemo_individual_sensitivity.csv"
SEED = 20260731

MEASURES = ["scr_rate", "phasic_mean", "scl_slope"]
DESCRIPTORS = ["rms_mean", "onset_env_mean", "chroma_flux_mean",
               "roughness_pl_mean", "mode_score"]
MIN_SONGS = 14
N_SPLIT = 20
N_SIM = 60                 # 每个 σ_u 的重复次数
SIGMAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
ALPHA = 0.05


def load() -> pd.DataFrame:
    T = pd.read_parquet(FEAT / "pmemo_eda_song_level.parquet")
    fa = pd.read_parquet(FEAT / "layer_a_reg_pmemo_ln.parquet")
    fd = pd.read_parquet(FEAT / "layer_d_reg_pmemo_ln.parquet")
    for f in (fa, fd):
        f["musicId"] = (f["clip_id"].astype(str)
                        .str.removeprefix("pmemo_").astype(int))
    F = fa.merge(fd.drop(columns=["clip_id"]), on="musicId", suffixes=("", "_d"))
    D = T.merge(F, on="musicId", how="inner")
    for m in MEASURES:
        g = D.groupby("listener")[m]
        D[m] = (D[m] - g.transform("mean")) / (g.transform("std") + 1e-9)
    for d in DESCRIPTORS:
        D[d] = (D[d] - D[d].mean()) / (D[d].std() + 1e-9)
    return D


def slope(x, y):
    if len(x) < 4 or np.std(x) < 1e-9:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def split_half_r(groups: list[np.ndarray], ys: list[np.ndarray],
                 rng: np.random.Generator, n_split: int = N_SPLIT) -> float:
    """给定每人的 (x, y)，返回逐人斜率的分半相关。"""
    rs = []
    for _ in range(n_split):
        a, b = [], []
        for x, y in zip(groups, ys):
            idx = rng.permutation(len(x))
            h = len(idx) // 2
            sa, sb = slope(x[idx[:h]], y[idx[:h]]), slope(x[idx[h:]], y[idx[h:]])
            if np.isfinite(sa) and np.isfinite(sb):
                a.append(sa)
                b.append(sb)
        if len(a) >= 50:
            r = spearmanr(a, b).statistic
            if np.isfinite(r):
                rs.append(r)
    return float(np.mean(rs)) if rs else np.nan


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("77_pmemo_individual_sensitivity", SEED)
    D = load()

    # 以「声压级 × SCR 速率」为标定对象 —— `76` 里最接近显著的一格。
    # 在最有利的一格上标定，得到的上界对其余各格同样成立（更保守）。
    desc, meas = "rms_mean", "scr_rate"
    xs, ys_real = [], []
    for _, g in D.groupby("listener"):
        g = g.dropna(subset=[desc, meas])
        if len(g) < MIN_SONGS:
            continue
        xs.append(g[desc].to_numpy(float))
        ys_real.append(g[meas].to_numpy(float))
    print(f"标定格：{meas} ~ {desc}   听者 {len(xs)}   "
          f"每人歌数中位 {np.median([len(x) for x in xs]):.0f}\n")

    # 真实数据的残差量级与总体斜率
    beta = np.mean([s for s in (slope(x, y) for x, y in zip(xs, ys_real))
                    if np.isfinite(s)])
    resid = np.mean([np.std(y - beta * x) for x, y in zip(xs, ys_real)])
    r_obs = split_half_r(xs, ys_real, rng, n_split=40)
    print(f"真实数据：总体斜率 β = {beta:+.4f}   残差 SD = {resid:.3f}   "
          f"分半 r = {r_obs:+.4f}")

    # 零分布：σ_u = 0 时分半 r 的分布 → 定出显著阈值
    print("\n注入已知大小的个体差异，看本检验捞不捞得出来")
    print(f"  （每格重复 {N_SIM} 次；显著阈值由 σ_u = 0 的分布定出）\n")
    rows = []
    null_rs: list[float] = []
    for sig in SIGMAS:
        rs = []
        for _ in range(N_SIM):
            u = rng.normal(0, sig, len(xs))
            ys = [(beta + u[i]) * xs[i] + rng.normal(0, resid, len(xs[i]))
                  for i in range(len(xs))]
            rs.append(split_half_r(xs, ys, rng, n_split=6))
        rs = np.array([r for r in rs if np.isfinite(r)])
        if sig == 0.0:
            null_rs = list(rs)
            thr = float(np.quantile(rs, 1 - ALPHA))
            print(f"  σ_u = 0（零分布）  r 均值 {rs.mean():+.4f}   "
                  f"95 分位 = {thr:+.4f}  ← 显著阈值")
        power = float(np.mean(rs > thr))
        rows.append({"sigma_u": sig, "mean_split_half_r": float(rs.mean()),
                     "power": power, "threshold": thr})
        if sig > 0:
            print(f"  σ_u = {sig:.2f}      r 均值 {rs.mean():+.4f}   "
                  f"把握 {power:.0%}")

    R = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT, index=False, encoding="utf-8")

    # 80% 把握对应的 σ_u
    ok = R[(R.sigma_u > 0) & (R.power >= 0.80)]
    lim = float(ok.sigma_u.min()) if len(ok) else np.nan
    print("\n" + "=" * 78)
    print(f"实测分半 r = {r_obs:+.4f}，未超显著阈值 {thr:+.4f}")
    if np.isfinite(lim):
        print(f"本检验对 σ_u ≥ {lim:.2f} 有 80% 把握。")
        print(f"→ 个体间斜率标准差**若存在也不超过约 {lim:.2f}**"
              f"（单位：z 分皮电 / 每 SD 声压级）")
        print(f"   作为参照，总体斜率本身 β = {beta:+.4f} ——"
              f" 上界是它的 {lim/abs(beta):.0f} 倍，")
        print("   也就是说：这批数据排除不掉「个体差异远大于群体平均效应」这种情形。")
    else:
        print("在所测范围内均未达 80% 把握 —— 该检验在此设计下不可用。")
    # ── 个性化的盈亏平衡点 ────────────────────────────────────────────
    #
    # `76` 的增益一律为负：用这个人自己的斜率反而更差。原因是斜率估计
    # 本身有噪声，方差约 σ_resid² / (n · var(x))。个性化要划算，
    # 真实的个体间斜率方差必须**超过**这个估计噪声：
    #
    #     σ_u² > σ_resid² / (n · var(x))
    #
    # 这把「要多少校准数据」翻译成工程上真正要做的决定：
    # **收集多少首之后，按人建模才开始优于用群体模型。**
    print("\n" + "=" * 78)
    print("个性化的盈亏平衡：每人多少首之后，按人建模才开始划算")
    print("  （判据：个体间斜率方差 > 斜率估计方差 σ_resid²/(n·var x)）\n")
    var_x = float(np.mean([np.var(x) for x in xs]))
    print(f"  σ_resid = {resid:.3f}   var(x) = {var_x:.3f}\n")
    print(f"  {'σ_u':>6}   {'盈亏平衡所需每人歌数':>22}")
    be_rows = []
    for sig in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
        n_be = resid ** 2 / (sig ** 2 * var_x)
        be_rows.append({"sigma_u": sig, "breakeven_n": n_be})
        print(f"  {sig:>6.2f}   {n_be:>22.0f}")
    pd.DataFrame(be_rows).to_csv(
        OUT.with_name("pmemo_personalisation_breakeven.csv"),
        index=False, encoding="utf-8")
    n_have = float(np.median([len(x) for x in xs]))
    sig_be = float(np.sqrt(resid ** 2 / (n_have * var_x)))
    print(f"\n  本数据集每人 {n_have:.0f} 首 → 只有当 σ_u > {sig_be:.2f} 时"
          f"个性化才划算。")
    print(f"  而灵敏度分析给出的上界是 σ_u ≤ {lim if np.isfinite(lim) else float('nan'):.2f}。")
    if np.isfinite(lim) and lim < sig_be:
        print("  两者一致：在本数据集的观测量下，个性化本就不该有增益 ——")
        print("  实测的负增益是**设计的必然结果**，不是「个体差异不存在」的证据。")
    run.log_metric("breakeven_sigma_u_at_current_n", sig_be)
    run.log_metric("observed_split_half_r", float(r_obs))
    run.log_metric("significance_threshold", thr)
    run.log_metric("sigma_u_80pct_power", lim)
    run.log_metric("population_slope", float(beta))
    run.write()
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
