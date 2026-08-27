r"""全项目 τ 的权威表：各格在**自己的 n** 上，配**自己的零分布**。

    .venv/Scripts/python scripts/99g_canonical_tau_table.py

## 它取代什么

- `99d` 步骤 [4] 的 34 格表 —— 那张用的是**全局**零分布常数（n=18 → 0.1047，
  n=44 → 0.0391）。`99f` 实测各格零分布在 0.064–0.102 之间，一个常数盖不住，
  用它会把 τ 削过头（`98` 因此虚高到 329，本格值下是 183）。
- `99f` 的三行汇总 —— 那张把所有格**配平到 n=18** 以便比较通道与域。
  比较要配平，但**报数**要用各格自己的 n，否则丢掉一半观测。

两张表各有各的用处，本表是「报数」那一张，也是补充材料要的那一张。

## 逐格零分布怎么算

参数化自举：取该格**真实的** x 向量（每人各自的）与残差 SD，
令 τ=0（所有人共用一个 β）生成 y，重估 τ̂，重复 N 次取中位。

比模拟一个代表性设定可信 —— x 的分布、每人的观测数、残差量级都照搬本格。

## 改正

方差可加（`99d` 步骤 3 在真值已知的模拟上验过，平均绝对误差 3.3%）：

    τ = √(max(τ̂² − τ̂²_null, 0))

CI 两端同样处理。**下界改正后为 0 的格，其 n\* 无上界** —— 照实报 ∞。

## 输出

    reports/source_data/canonical_tau_table.csv    数据
    reports/tables/canonical_tau_table.md          补充材料用的 Markdown 表
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
OUT_CSV = REPO_ROOT / "reports" / "source_data" / "canonical_tau_table.csv"
OUT_MD = REPO_ROOT / "reports" / "tables" / "canonical_tau_table.md"
SEED = 20260803
CHI2_95 = chi2.ppf(0.95, 1)
N_NULL = 60
DESCRIPTORS = ["rms_mean", "onset_env_mean", "chroma_flux_mean",
               "roughness_pl_mean", "mode_score"]
EDA = ["scr_rate", "phasic_mean", "scl_slope"]
ARAUS_P = ["Navg_r", "Savg_r", "Ravg_r", "Favg_r"]
MIN_OBS = 14


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


def fit(u, grid_n: int = 1500):
    bs, ses, rsd, vx, ns = [], [], [], [], []
    for x, y in u:
        b, se = slope_se(x, y)
        if np.isfinite(b) and np.isfinite(se) and se > 0:
            bs.append(b)
            ses.append(se)
            rsd.append(float(np.std(y - b * (x - x.mean()))))
            vx.append(float(np.var(x)))
            ns.append(len(x))
    if len(bs) < 30:
        return None
    tau, lo, hi, beta = tau_ci(np.array(bs), np.array(ses), grid_n)
    n_med, r_med, v_med = (float(np.median(ns)), float(np.median(rsd)),
                           float(np.median(vx)))
    # 🔴 盈亏平衡 = 逐人斜率标准误的中位，**直接取**，不再用
    # σ_resid/√((n−1)·var x) 重算一遍。两者本该相等，但代码里
    # `np.var` 是 ddof=0（Σ(x−x̄)² = n·var_x 而非 (n−1)·var_x），
    # `np.std` 的残差也没扣 n−2，净差 √((n−2)/(n−1))：n=18 时 0.9701。
    # 更要紧的是各人 n 差别大时（DEAM 每评分者 n 从 14 到 732），
    # 「中位数的组合」≠「组合的中位数」，那里偏 6.8%，连带 n* 偏高 14%。
    return {"tau": tau, "lo": lo, "hi": hi, "beta": beta, "k": len(bs),
            "n": n_med, "resid_sd": r_med, "var_x": v_med,
            "be": float(np.median(ses))}


def cell_null(u, beta: float, resid_sd: float, rng) -> tuple[float, float]:
    r"""本格的 λ=0 行。返回 (零分布中位, 该中位的蒙特卡洛标准误)。

    🔴 模拟出来的 y **必须走一遍真实管线里的被试内 z 分**。
    首版直接拿生成的 y 去拟合，漏了「除以一个估计出来的 SD」这一步。

    我起初以为漏掉它会让零分布偏**高**（z 分有方差稳定作用），
    在 x~N(0,1)、各人同分布的理想模拟上确实如此（比 0.567）。
    但**真实数据上方向相反**（比 1.06–1.33，中位 1.14）——
    每人看到的刺激子集不同、var(x) 差别很大，z 分等于把 b_i 除以一个
    与 var(x) 相关的个人量，反而引入额外的个体间变异。
    理想化模拟把这条机制抹掉了。**同质假设是这里的陷阱。**

    另返回 MC 标准误：60 次抽样的中位数自身带 ±0.002–0.005 的误差，
    而有些格是以 0.0009 的 margin 被判「超零分布」的。
    """
    taus = []
    for _ in range(N_NULL):
        sim = []
        for x, _ in u:
            y = beta * (x - x.mean()) + rng.normal(0, resid_sd, len(x))
            y = (y - y.mean()) / (y.std() + 1e-9)      # 与真实管线同步
            sim.append((x, y))
        r = fit(sim, grid_n=600)
        if r:
            taus.append(r["tau"])
    if not taus:
        return np.nan, np.nan
    a = np.array(taus)
    return float(np.median(a)), float(1.253 * a.std(ddof=1) / np.sqrt(len(a)))


def zscore(D, unit, ys, xs):
    D = D.copy()
    for y in ys:
        g = D.groupby(unit)[y]
        D[y] = (D[y] - g.transform("mean")) / (g.transform("std") + 1e-9)
    for x in xs:
        D[x] = (D[x] - D[x].mean()) / (D[x].std() + 1e-9)
    return D


def units(D, unit, x, y):
    return [(h[x].to_numpy(float), h[y].to_numpy(float))
            for _, h in D.dropna(subset=[x, y]).groupby(unit)
            if len(h) >= MIN_OBS]


def all_cells():
    T = pd.read_parquet(FEAT / "pmemo_eda_song_level.parquet")
    fa = pd.read_parquet(FEAT / "layer_a_reg_pmemo_ln.parquet")
    fd = pd.read_parquet(FEAT / "layer_d_reg_pmemo_ln.parquet")
    for f in (fa, fd):
        f["musicId"] = (f.clip_id.astype(str)
                        .str.removeprefix("pmemo_").astype(int))
    F = fa.merge(fd.drop(columns=["clip_id"]), on="musicId", suffixes=("", "_d"))
    P = zscore(T.merge(F, on="musicId"), "listener", EDA, DESCRIPTORS)
    for m in EDA:
        for d in DESCRIPTORS:
            u = units(P, "listener", d, m)
            if len(u) >= 30:
                yield "生理 · 音乐", "PMEmo 皮电", f"{m} ~ {d}", u

    R = pd.read_csv(DEAM_CSV, skipinitialspace=True)
    R.columns = [c.strip() for c in R.columns]
    A = pd.read_parquet(FEAT / "layer_a_reg_deam_ln.parquet")
    Dl = pd.read_parquet(FEAT / "layer_d_reg_deam_ln.parquet")
    G = A.merge(Dl, on="clip_id", suffixes=("", "_d"))
    G["SongId"] = G.clip_id.astype(str).str.removeprefix("deam_").astype(int)
    M = zscore(R.merge(G[["SongId"] + DESCRIPTORS], on="SongId"),
               "workerID", ["Arousal", "Valence"], DESCRIPTORS)
    for t in ("Arousal", "Valence"):
        for d in DESCRIPTORS:
            u = units(M, "workerID", d, t)
            if len(u) >= 30:
                yield "主观 · 音乐", "DEAM 评分", f"{t} ~ {d}", u

    S = pd.read_csv(AR / "responses.csv")
    # 🔴 剔除 fold_r=−1 的锚点 —— 每人 2 行、同一声景+静默+smr 0、位 1 与 45，
    # 749 人完全相同的重复高杠杆设计点。留着它 τ 变动最大 21.5%（`99` 步骤 4）。
    S = S[(S.is_attention == 0) & (S.fold_r != -1)]
    S = zscore(S, "participant", ["pleasant", "eventful"], ARAUS_P)
    for t in ("pleasant", "eventful"):
        for d in ARAUS_P:
            u = units(S, "participant", d, t)
            if len(u) >= 30:
                yield "主观 · 声景（产品域）", "ARAUS", f"{t} ~ {d}", u


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("99g_canonical_tau_table", SEED)
    rows = []
    print("=" * 108)
    print("各格在**自己的 n** 上，配**自己的**零分布（参数化自举）")
    print(f"  {'域 · 通道':<20}{'格':<26}{'人':>5}{'n':>4}{'τ̂':>8}"
          f"{'零分布':>8}{'τ 改正':>8}{'CI 改正':>18}{'平衡点':>8}{'n*':>8}")
    for dom, src, cell, u in all_cells():
        r = fit(u)
        if not r:
            continue
        null, mcse = cell_null(u, r["beta"], r["resid_sd"], rng)

        def fix(v: float) -> float:
            return float(np.sqrt(max(v ** 2 - null ** 2, 0.0)))

        tc, lc, hc = fix(r["tau"]), fix(r["lo"]), fix(r["hi"])
        nstar = r["n"] * (r["be"] / tc) ** 2 if tc > 1e-9 else np.inf
        nlo = r["n"] * (r["be"] / hc) ** 2 if hc > 1e-9 else np.inf
        # 「超零分布」须 margin > 2×MC 误差 —— 裸点比较会把 0.0009 的差
        # 判成阳性，而那小于 60 次抽样自身的噪声。`99f` 本来有这条，
        # `99g` 首版丢了，于是同一格在两份表里判读相反。
        above = bool(r["tau"] - null > 2 * mcse)
        print(f"  {dom:<20}{cell:<26}{r['k']:>5}{r['n']:>4.0f}{r['tau']:>8.4f}"
              f"{null:>8.4f}{tc:>8.4f}   [{lc:.4f}, {hc:.4f}]"
              f"{r['be']:>8.4f}"
              + (f"{nstar:>8.0f}" if np.isfinite(nstar) else f"{'∞':>8}")
              + ("" if above else "   —"),
              flush=True)
        rows.append({"domain": dom, "source": src, "cell": cell,
                     "n_units": r["k"], "n_per_unit": r["n"], "beta": r["beta"],
                     "tau_raw": r["tau"], "ci_lo_raw": r["lo"],
                     "ci_hi_raw": r["hi"], "null": null, "mc_se": mcse,
                     "tau": tc, "ci_lo": lc, "ci_hi": hc,
                     "breakeven": r["be"], "n_star": nstar, "n_star_lo": nlo,
                     "above_null": above,
                     "usable_now": bool(above and lc > r["be"])})

    R = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print("\n" + "=" * 108)
    print("按域 · 通道汇总")
    print(f"  {'域 · 通道':<22}{'格':>4}{'超零分布':>10}{'现在就够用':>12}"
          f"{'τ 中位':>9}{'n* 中位':>10}{'零分布 范围':>18}")
    for dom, g in R.groupby("domain"):
        fin = g.n_star.replace(np.inf, np.nan).dropna()
        print(f"  {dom:<22}{len(g):>4}"
              f"{f'{int(g.above_null.sum())}/{len(g)}':>10}"
              f"{f'{int(g.usable_now.sum())}/{len(g)}':>12}"
              f"{g.tau.median():>9.4f}"
              + (f"{fin.median():>10.0f}" if len(fin) else f"{'∞':>10}")
              + f"   [{g.null.min():.3f}, {g.null.max():.3f}]")
        run.log_metric(f"tau_median_{dom}", float(g.tau.median()))
        run.log_metric(f"above_null_{dom}", int(g.above_null.sum()))

    print("\n" + "=" * 108)
    print("零分布本身的分布 —— 这是「不能用一个常数」的证据")
    for n, g in R.groupby(R.n_per_unit.round()):
        print(f"  每人 n={n:.0f}   {len(g)} 格   零分布 "
              f"中位 {g.null.median():.4f}   范围 "
              f"[{g.null.min():.4f}, {g.null.max():.4f}]   "
              f"极差是中位的 {(g.null.max()-g.null.min())/g.null.median():.0%}")
        run.log_metric(f"null_median_n{n:.0f}", float(g.null.median()))

    # ── Markdown 表，供补充材料 ──────────────────────────────────────
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    L = ["# 补充表：个体间斜率 SD（τ）的逐格估计与零分布改正", "",
         "每格在自己的每人观测数 n 上估计。零分布由**参数化自举**逐格算出：",
         "取该格真实的 x 向量与残差 SD，令 τ=0 生成 y，重估 τ̂，"
         f"重复 {N_NULL} 次取中位。", "",
         "改正为方差可加 τ = √(max(τ̂² − τ̂²_null, 0))。", "",
         "⚠️ 该模型在真值已知的模拟上对 **τ̂** 的平均绝对误差是 3.3%，"
         "但那不是本表报的量。", "",
         "残差**全为正**（+1.3% ~ +4.2%），即模型系统性高估 τ̂、"
         "因而系统性**欠改正**。传到本表报的量上：τ 改正值的误差 1.4–14.6%、"
         "n\\* 的误差最大 +37%，且**在 τ 小处最大** —— 正是生理侧那些格。", "",
         "`n*` 为按人校准的盈亏平衡观测数；CI 下界改正后为 0 的格，n\\* 无上界。",
         "",
         "| 域 · 通道 | 格 | 人 | n | β | τ̂ | 零分布 | τ 改正 | 95% CI | 平衡点 | n\\* |",
         "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    for _, r in R.iterrows():
        ns = "∞" if not np.isfinite(r.n_star) else f"{r.n_star:.0f}"
        L.append(f"| {r.domain} | `{r.cell}` | {r.n_units:.0f} | "
                 f"{r.n_per_unit:.0f} | {r.beta:+.3f} | {r.tau_raw:.4f} | "
                 f"{r['null']:.4f} | **{r.tau:.4f}** | "
                 f"[{r.ci_lo:.4f}, {r.ci_hi:.4f}] | {r.breakeven:.4f} | {ns} |")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")

    run.write()
    print(f"\n→ {OUT_CSV}")
    print(f"→ {OUT_MD}")


if __name__ == "__main__":
    main()
