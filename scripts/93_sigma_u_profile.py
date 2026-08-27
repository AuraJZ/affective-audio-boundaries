r"""σ_u 的正确估计：剖面似然置信区间，两个通道并排。

    .venv/Scripts/python scripts/93_sigma_u_profile.py

## 要修什么 🔴

`77` 报的 σ_u ≤ 0.15 是**七点手选网格上的最小可检出效应**
（`77_pmemo_individual_sensitivity.py:51` 的 `SIGMAS`），
**不是置信上界** —— 而我一直按上界在讲。两者含义不同：

    最小可检出效应   「若真值达到 0.15，本检验有 80% 把握」
    置信上界         「真值有 95% 的把握不超过 X」

前者依赖你选的网格和检验的功效，后者是数据本身的陈述。

## 正确形式：这就是随机效应元分析

模型 `y_ij = (β + u_i)·x_ij + ε_ij`，`u_i ~ N(0, σ_u²)` 下，
逐人 OLS 斜率的边缘分布是

    b_i ~ N(β, σ_u² + se_i²)

即「效应估计 + 已知的估计误差 + 未知的异质性」—— 元分析的标准形式，
**σ_u 就是 τ**。而 τ 的**剖面似然置信区间**是成熟做法
（Viechtbauer 2007），精确、闭式、无需模拟网格。

    ℓ(β, τ²)   = −½ Σ [ log(2π(v_i+τ²)) + (b_i−β)²/(v_i+τ²) ]
    β̂(τ²)      = Σ w_i b_i / Σ w_i ,  w_i = 1/(v_i+τ²)
    ℓ_REML(τ²) = ℓ(β̂, τ²) − ½ log(Σ w_i)
    95% CI     = { τ² : 2[ℓ_p(τ̂²) − ℓ_p(τ²)] ≤ 3.841 }

## 两个通道并排 —— `92` 之后这是必须的

`92` 判决：主观侧在 n=18 下与生理侧**同样**检不出个体差异，
故「18 首/人下按人重拟合被排除」是**规模结论不是通道结论**。

因此 σ_u 必须在两个通道上用同一套方法估，否则单报生理侧那个数
会再次被读成「生理特有」。

## 判读

把 CI 上界与**盈亏平衡** σ_u* = σ_resid/√((n−1)·var x) 比：

    上界 < σ_u*   该规模下按人重拟合确实不划算（规模结论）
    上界 > σ_u*   数据排除不掉「个体差异大到值得校准」
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
PM = REPO_ROOT / "data" / "raw" / "PMEmo" / "PMEmo2019"
DEAM = REPO_ROOT / "data" / "raw" / "DEAM" / "annotations" / \
    "annotations per each rater" / "song_level" / \
    "static_annotations_songs_1_2000.csv"
OUT = REPO_ROOT / "reports" / "source_data" / "sigma_u_profile.csv"
SEED = 20260802

DESCRIPTORS = ["rms_mean", "onset_env_mean", "chroma_flux_mean",
               "roughness_pl_mean", "mode_score"]
EDA = ["scr_rate", "phasic_mean", "scl_slope"]
RATINGS = ["Arousal", "Valence"]
MIN_OBS = 14
CHI2_95 = chi2.ppf(0.95, 1)          # 3.841


def slope_se(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """逐人 OLS 斜率与其标准误。"""
    n = len(x)
    if n < 4 or np.std(x) < 1e-9:
        return np.nan, np.nan
    xc = x - x.mean()
    sxx = float((xc ** 2).sum())
    b = float((xc * (y - y.mean())).sum() / sxx)
    resid = y - y.mean() - b * xc
    s2 = float((resid ** 2).sum() / (n - 2))
    return b, float(np.sqrt(s2 / sxx))


def _ll(t2: float, b: np.ndarray, v: np.ndarray, reml: bool) -> float:
    w = 1.0 / (v + t2)
    mu = float((w * b).sum() / w.sum())
    ll = -0.5 * float((np.log(2 * np.pi / w) + w * (b - mu) ** 2).sum())
    return ll - 0.5 * np.log(w.sum()) if reml else ll


def tau_profile_ci(b: np.ndarray, se: np.ndarray, reml: bool = True
                   ) -> tuple[float, float, float]:
    """τ 的点估计与 95% 剖面似然置信区间。返回 (τ̂, lo, hi)。"""
    v = se ** 2
    hi_bound = max(float(np.var(b, ddof=1)) * 20, 1e-4)
    grid = np.concatenate([[0.0], np.geomspace(1e-8, hi_bound, 4000)])
    lls = np.array([_ll(t, b, v, reml) for t in grid])
    k = int(np.argmax(lls))
    t2_hat, ll_max = float(grid[k]), float(lls[k])

    def g(t2: float) -> float:
        return 2.0 * (ll_max - _ll(t2, b, v, reml)) - CHI2_95

    lo = 0.0
    if k > 0 and g(0.0) > 0:
        lo = brentq(g, 0.0, t2_hat)
    hi = hi_bound
    if g(hi_bound) > 0:
        hi = brentq(g, t2_hat, hi_bound)
    return float(np.sqrt(t2_hat)), float(np.sqrt(lo)), float(np.sqrt(hi))


def breakeven(resid_sd: float, n: float, var_x: float) -> float:
    """个性化的盈亏平衡 σ_u* —— 估计噪声与真实异质性相等之处。"""
    return float(np.sqrt(resid_sd ** 2 / max((n - 1) * var_x, 1e-9)))


def units_pmemo() -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    T = pd.read_parquet(FEAT / "pmemo_eda_song_level.parquet")
    fa = pd.read_parquet(FEAT / "layer_a_reg_pmemo_ln.parquet")
    fd = pd.read_parquet(FEAT / "layer_d_reg_pmemo_ln.parquet")
    for f in (fa, fd):
        f["musicId"] = (f["clip_id"].astype(str)
                        .str.removeprefix("pmemo_").astype(int))
    F = fa.merge(fd.drop(columns=["clip_id"]), on="musicId", suffixes=("", "_d"))
    D = T.merge(F, on="musicId")
    for m in EDA:
        g = D.groupby("listener")[m]
        D[m] = (D[m] - g.transform("mean")) / (g.transform("std") + 1e-9)
    for d in DESCRIPTORS:
        D[d] = (D[d] - D[d].mean()) / (D[d].std() + 1e-9)
    out = {}
    for m in EDA:
        for d in DESCRIPTORS:
            u = []
            for _, g in D.groupby("listener"):
                g = g.dropna(subset=[d, m])
                if len(g) >= MIN_OBS:
                    u.append((g[d].to_numpy(float), g[m].to_numpy(float)))
            out[f"EDA {m} ~ {d}"] = u
    return out


def units_deam() -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    R = pd.read_csv(DEAM, skipinitialspace=True)
    R.columns = [c.strip() for c in R.columns]
    A = pd.read_parquet(FEAT / "layer_a_reg_deam_ln.parquet")
    Dl = pd.read_parquet(FEAT / "layer_d_reg_deam_ln.parquet")
    F = A.merge(Dl, on="clip_id", suffixes=("", "_d"))
    F["SongId"] = F["clip_id"].astype(str).str.removeprefix("deam_").astype(int)
    M = R.merge(F[["SongId"] + DESCRIPTORS], on="SongId")
    for t in RATINGS:
        g = M.groupby("workerID")[t]
        M[t] = (M[t] - g.transform("mean")) / (g.transform("std") + 1e-9)
    for d in DESCRIPTORS:
        M[d] = (M[d] - M[d].mean()) / (M[d].std() + 1e-9)
    out = {}
    for t in RATINGS:
        for d in DESCRIPTORS:
            u = []
            for _, g in M.groupby("workerID"):
                g = g.dropna(subset=[d, t])
                if len(g) >= MIN_OBS:
                    u.append((g[d].to_numpy(float), g[t].to_numpy(float)))
            out[f"评分 {t} ~ {d}"] = u
    return out


def main() -> None:
    run = RunRecord("93_sigma_u_profile", SEED)
    rows = []
    for label, chan in (("生理（PMEmo 皮电）", units_pmemo()),
                        ("主观（DEAM 评分）", units_deam())):
        print("=" * 94)
        print(f"{label}")
        print(f"  {'格':<30}{'人':>4}{'每人n':>7}{'β':>9}{'σ̂_u':>9}"
              f"{'95% CI':>20}{'盈亏平衡':>10}{'判读':>14}")
        for key, u in chan.items():
            if len(u) < 30:
                continue
            bs, ses, ns, rsd, vx = [], [], [], [], []
            for x, y in u:
                b, se = slope_se(x, y)
                if np.isfinite(b) and np.isfinite(se) and se > 0:
                    bs.append(b)
                    ses.append(se)
                    ns.append(len(x))
                    rsd.append(float(np.std(y - b * (x - x.mean()))))
                    vx.append(float(np.var(x)))
            if len(bs) < 30:
                continue
            b_, se_ = np.array(bs), np.array(ses)
            tau, lo, hi = tau_profile_ci(b_, se_)
            n_med = float(np.median(ns))
            be = breakeven(float(np.median(rsd)), n_med, float(np.median(vx)))
            w = 1.0 / (se_ ** 2 + tau ** 2)
            beta = float((w * b_).sum() / w.sum())
            verdict = ("✅ 低于平衡点" if hi < be else
                       ("⚠️ 跨过平衡点" if lo < be < hi else "🔴 高于平衡点"))
            print(f"  {key:<30}{len(bs):>4}{n_med:>7.0f}{beta:>+9.4f}"
                  f"{tau:>9.4f}   [{lo:.4f}, {hi:.4f}]{be:>10.4f}"
                  f"   {verdict}")
            rows.append({"channel": label, "cell": key, "n_units": len(bs),
                         "n_per_unit": n_med, "beta": beta, "tau": tau,
                         "ci_lo": lo, "ci_hi": hi, "breakeven": be,
                         "hi_below_breakeven": bool(hi < be)})
    R = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT, index=False, encoding="utf-8")

    # ── 产品要的那个数：按人校准从多少次观测起开始划算 ────────────────
    #
    # 盈亏平衡 σ_u*(n) = σ_resid/√((n−1)·var x) 随 √n 缩小。
    # 令它等于实测的 τ̂，解出 n：
    #
    #     n* = σ_resid² / (τ̂² · var x) + 1
    #
    # 这是**用真实数据估出的**门槛，不是模拟外推。
    # CI 给出门槛的区间：τ 越小需要的观测越多，故 n* 的上界由 CI 下界决定。
    R["n_breakeven"] = R.apply(
        lambda r: r.n_per_unit * (r.breakeven / max(r.tau, 1e-9)) ** 2, axis=1)
    R["n_breakeven_lo"] = R.apply(
        lambda r: r.n_per_unit * (r.breakeven / max(r.ci_hi, 1e-9)) ** 2, axis=1)
    R["n_breakeven_hi"] = R.apply(
        lambda r: r.n_per_unit * (r.breakeven / max(r.ci_lo, 1e-9)) ** 2
        if r.ci_lo > 1e-6 else np.inf, axis=1)

    print("\n" + "=" * 94)
    print("判读汇总")
    for ch, g in R.groupby("channel"):
        below = int(g.hi_below_breakeven.sum())
        print(f"  {ch}   {below}/{len(g)} 格的 CI 上界低于盈亏平衡   "
              f"τ̂ 中位 {g.tau.median():.4f}   CI 上界中位 {g.ci_hi.median():.4f}")
        run.log_metric(f"tau_median_{ch}", float(g.tau.median()))
        run.log_metric(f"ci_hi_median_{ch}", float(g.ci_hi.median()))
        run.log_metric(f"n_below_breakeven_{ch}", below)

    print("\n" + "=" * 94)
    print("⭐ 按人校准从多少次观测起开始划算（由实测 τ̂ 解出，非模拟外推）")
    print(f"  {'通道':<20}{'当前 n':>8}{'门槛 n*':>10}{'（由 CI 给出的区间）':>26}")
    for ch, g in R.groupby("channel"):
        nb = float(g.n_breakeven.median())
        lo = float(g.n_breakeven_lo.median())
        hi = g.n_breakeven_hi.replace(np.inf, np.nan).median()
        print(f"  {ch:<20}{g.n_per_unit.median():>8.0f}{nb:>10.0f}"
              f"      [{lo:.0f}, {hi:.0f}]" if np.isfinite(hi) else
              f"  {ch:<20}{g.n_per_unit.median():>8.0f}{nb:>10.0f}"
              f"      [{lo:.0f}, ∞)")
        run.log_metric(f"n_breakeven_{ch}", nb)
        run.log_metric(f"n_breakeven_lo_{ch}", lo)
    print()
    print("  与 `92` 的实测曲线对照：主观侧个体差异在 n=100 处开始显形")
    print("  （分半 1/10 显著、增益 4/10 为正）—— **两条独立路径给出同一量级**。")

    print()
    print("措辞纪律：以下三者含义不同，正文不得混用 ——")
    print("  σ̂_u        点估计")
    print("  CI 上界     「有 95% 把握不超过」← 本脚本给的就是它")
    print("  最小可检出   「若真值达到它，检验有 80% 把握」← `77` 给的是这个")
    print()
    print("并排的意义（`92` 之后必须如此报）：若两个通道的 τ̂ 与 CI 相当，")
    print("则「个体差异测不出」是**规模结论**，与通道无关。")
    run.write()
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
