r"""C1（重做）：等预算等高面 —— 多听几首 vs 每首多测几秒。

    .venv/Scripts/python scripts/98_budget_surface_v2.py

## 为什么重做

首版（`96`）用 `pmemo_frames.parquet`，但那张表是 `82` 为**段内**分析建的，
`build_cache` 对每首歌做了 `detrend()` —— 去线性趋势包含去均值，
于是每个 (听众, 歌) 的均值**恒为零**（实测歌曲级 SD = 0.000000）。
拿它做**段间**分析等于把一堆零对声学回归，得出「4 首 × 5 段就够」，
与 `92`、`93` 的门槛结论直接矛盾。

数据源已由 `97` 重建（保留歌曲间差异，被试内 z 分后 SD = 0.974，与 `93` 一致）。

## 框架修正：声学侧不加窗

真实系统**总是知道自己在放什么音频** —— 声学特征可以离线全曲算，不花钱。
花钱的只有「测这个人」。因此：

    声学   全曲值，全局标准化（与 `93` 同口径）
    皮电   **仅测 w 秒**的窗口值，被试内 z 分

预算 = k 首 × w 秒 = 每人的总测量时长。

## 两个轴的性质不同

    k 增大   每多一首是一个**独立**的新观测 → SE 应按 1/√k 降
    w 增大   只是把同一首测得更准 → 受自相关限制，会饱和

`97` 的校验行已经透露了 w 的饱和点：歌曲级 SD 在 2.5→10 秒间从 0.104 升到 0.154，
之后（20、37 秒）基本不动。**测超过约 10 秒，歌曲间的信号不再增加。**

## 判据

SE(k, w) 与实测的个体间 SD τ 比 —— 阈值由 `99g` 权威表算出，
不再是从 `93` 抄来的常数 0.106（那个数埋在自己的零分布里）。
SE < τ 处，个体差异开始压过估计噪声，按人校准才有意义。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")

FEAT = REPO_ROOT / "data" / "features"
SD_DIR = REPO_ROOT / "reports" / "source_data"
OUT = SD_DIR / "budget_surface.csv"
SEED = 20260802
DESC = "rms_mean"               # `76` 起的先验主描述符（全曲值）
MEASURE = "phasic_mean"         # 三个指标中信度最高的
K_GRID = [4, 6, 9, 12, 18]
W_GRID = [2.5, 5.0, 10.0, 20.0, 37.0]
N_REP = 60
MIN_SONGS = 14


def tau_target() -> tuple[float, float, float]:
    r"""阈值 τ 及其 CI —— **算出来，不写死**。🔴

    首版这里是 `TAU_TARGET = 0.106`，一个从 `93` 抄来的常数。
    两个后果：

    1. `98` 与 `93` **不独立**（纪律 #12）。PLAN 里「三条独立路径收敛于
       同一门槛」因此不成立 —— 同一个输入喂两遍，得同一个数是恒等式。
    2. 那个 0.106 本身**埋在自己的零分布里**（纪律 #11、脚本 `99d`）：
       每人 n=18 时，即使真值为零，剖面似然也报 τ̂ ≈ 0.1047。

    改正是方差可加：τ = √(max(τ̂² − τ̂²_null(n), 0))，
    在真值已知的模拟上验过，平均绝对误差 3.3%（`99d` 步骤 3）。

    零分布取**本格**的值，来源是 `99g` 的权威表（各格在自己的 n 上、
    配自己的参数化自举零分布）。不取 `99d` 的全局常数 ——
    实测各格零分布在 n=18 处 0.089–0.105、n=44 处 0.000–0.043，
    一个常数盖不住；全局值偏高，用它会把 τ 削过头。
    """
    C = pd.read_csv(SD_DIR / "canonical_tau_table.csv")
    C = C[C.cell == f"{MEASURE} ~ {DESC}"]
    if len(C) == 0:
        raise SystemExit(f"canonical_tau_table.csv 里没有 {MEASURE} ~ {DESC} 一格")
    # 不要 rename —— 权威表里 `tau` 与 `tau_raw` 同时存在，
    # 把后者改名成前者会造出重复索引，`row.tau` 于是返回一个两元素 Series。
    row = C.iloc[0]
    raw_tau = float(row.tau_raw)
    raw_lo, raw_hi = float(row.ci_lo_raw), float(row.ci_hi_raw)
    null = float(row["null"])

    def fix(v: float) -> float:
        return float(np.sqrt(max(float(v) ** 2 - null ** 2, 0.0)))

    print(f"阈值 τ 取自 `99g` 权威表的 {row.cell} 一格"
          f"（每人 n={row.n_per_unit:.0f}）")
    print(f"  原始 τ̂ = {raw_tau:.4f}   **本格**零分布 = {null:.4f}"
          f"（参数化自举，非全局常数）")
    print(f"  → 改正后 τ = {fix(raw_tau):.4f}")
    print(f"  CI [{raw_lo:.4f}, {raw_hi:.4f}] "
          f"→ 改正后 [{fix(raw_lo):.4f}, {fix(raw_hi):.4f}]")
    if fix(raw_lo) <= 0:
        print("  ⚠️ CI 下界改正后为 **0** —— 真值可能就是零，"
              "则所需预算无上界。下面的外推是**乐观端**。")
    return fix(raw_tau), fix(raw_lo), fix(raw_hi)


def load():
    """(听众 → [(声学全曲值, 皮电窗口值)]) ，按窗长分组。"""
    W = pd.read_parquet(FEAT / "pmemo_window_level.parquet")
    fa = pd.read_parquet(FEAT / "layer_a_reg_pmemo_ln.parquet")
    fa["musicId"] = (fa["clip_id"].astype(str)
                     .str.removeprefix("pmemo_").astype(int))
    fa[DESC] = (fa[DESC] - fa[DESC].mean()) / (fa[DESC].std() + 1e-9)
    A = fa[["musicId", DESC]]

    out = {}
    for w, g in W.groupby("window_s"):
        # 同一 (歌,人,窗长) 有 N_DRAW 个随机位置，取均值代表「测 w 秒」
        g = g.groupby(["musicId", "listener"])[MEASURE].mean().reset_index()
        g = g.merge(A, on="musicId", how="inner")
        # 被试内 z 分 —— 与 `93` 同口径
        gg = g.groupby("listener")[MEASURE]
        g[MEASURE] = (g[MEASURE] - gg.transform("mean")) / \
                     (gg.transform("std") + 1e-9)
        per = {}
        for lid, h in g.groupby("listener"):
            h = h.dropna(subset=[DESC, MEASURE])
            if len(h) >= MIN_SONGS:
                per[lid] = (h[DESC].to_numpy(float), h[MEASURE].to_numpy(float))
        out[float(w)] = per
    return out


def slope_se(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 4:
        return np.nan
    xc = x - x.mean()
    sxx = float((xc ** 2).sum())
    if sxx < 1e-12:
        return np.nan
    b = float((xc * (y - y.mean())).sum() / sxx)
    resid = y - y.mean() - b * xc
    return float(np.sqrt(float((resid ** 2).sum() / (n - 2)) / sxx))


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("98_budget_surface_v2", SEED)
    global TAU_TARGET
    TAU_TARGET, TAU_LO, TAU_HI = tau_target()
    run.log_metric("tau_target_corrected", TAU_TARGET)
    print()
    data = load()
    print("各窗长可用听众数：",
          "  ".join(f"{w:.1f}s={len(p)}" for w, p in sorted(data.items())), "\n")

    rows = []
    print("=" * 90)
    print(f"斜率估计误差 SE(k, w)   阈值 τ = {TAU_TARGET:.4f}"
          f"（`99g` 权威表，已扣本格零分布）")
    print(f"  {'k(首)':>6}{'w(秒)':>7}{'预算(秒)':>10}{'人':>5}"
          f"{'SE':>9}{'SE/τ':>8}{'可用':>6}")
    for w in W_GRID:
        per = data.get(w, {})
        if len(per) < 50:
            print(f"  （窗 {w}s 仅 {len(per)} 人达标，跳过）")
            continue
        for k in K_GRID:
            ses = []
            for lid, (x, y) in per.items():
                if len(x) < k:
                    continue
                v = []
                for _ in range(N_REP):
                    idx = rng.choice(len(x), k, replace=False)
                    s = slope_se(x[idx], y[idx])
                    if np.isfinite(s):
                        v.append(s)
                if len(v) >= N_REP // 2:
                    ses.append(float(np.median(v)))
            if len(ses) < 50:
                continue
            se = float(np.median(ses))
            budget = k * w
            print(f"  {k:>6}{w:>7.1f}{budget:>10.0f}{len(ses):>5}{se:>9.4f}"
                  f"{se/TAU_TARGET:>8.2f}{'  ✅' if se < TAU_TARGET else '':>6}",
                  flush=True)
            rows.append({"k_songs": k, "window_s": w, "budget_s": budget,
                         "se": se, "se_over_tau": se / TAU_TARGET,
                         "usable": bool(se < TAU_TARGET), "n_listeners": len(ses)})

    R = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT, index=False, encoding="utf-8")

    # ── 两个轴的边际收益 ─────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("两个轴的边际收益：加倍之后 SE 变为多少（独立观测的理想值 0.71×）")
    for name, col, other in (("k 首数", "k_songs", "window_s"),
                             ("w 秒数", "window_s", "k_songs")):
        gains = []
        for _, g in R.groupby(other):
            g = g.sort_values(col)
            vals = g[col].tolist()
            for a, b in zip(vals[:-1], vals[1:]):
                if not (1.5 <= b / a <= 2.5):
                    continue
                sa = g[g[col] == a].se.iloc[0]
                sb = g[g[col] == b].se.iloc[0]
                gains.append(float(sb / sa))
        if gains:
            med = float(np.median(gains))
            tag = "← 接近理想，加它划算" if med < 0.80 else "← 已饱和，加它不划算"
            print(f"  {name:<8} {med:.3f}×  (n={len(gains)} 对)   {tag}")
            run.log_metric(f"marginal_{col}", med)

    # ── 固定预算下怎么拆 ─────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("固定总测量时长，怎么拆最好")
    R["bucket"] = pd.cut(R.budget_s, [0, 60, 120, 250, 500, 1e9],
                         labels=["≤60s", "60–120s", "120–250s", "250–500s", ">500s"])
    print(f"  {'预算':>10}{'最优 (k, w)':>16}{'SE':>9}"
          f"{'最差 (k, w)':>16}{'SE':>9}{'差距':>8}")
    for b, g in R.groupby("bucket", observed=True):
        if len(g) < 2:
            continue
        best, worst = g.loc[g.se.idxmin()], g.loc[g.se.idxmax()]
        print(f"  {str(b):>10}"
              # :.1f 而非 :.0f —— 首版把 2.5 秒印成「2s」，表可能被引进正文
              f"{f'({int(best.k_songs)}, {best.window_s:.1f}s)':>16}{best.se:>9.4f}"
              f"{f'({int(worst.k_songs)}, {worst.window_s:.1f}s)':>16}"
              f"{worst.se:>9.4f}{worst.se/best.se:>8.2f}×")

    # ── 判决 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    u = R[R.usable]
    if len(u):
        c = u.loc[u.budget_s.idxmin()]
        print(f"最小可行预算：{int(c.k_songs)} 首 × {c.window_s:.0f} 秒 "
              f"= {c.budget_s:.0f} 秒/人   SE={c.se:.4f} < τ={TAU_TARGET}")
        run.log_metric("min_feasible_budget_s", float(c.budget_s))
    else:
        best = R.loc[R.se.idxmin()]
        print(f"⚠️ 所测范围内**无**任何 (k, w) 使 SE < τ = {TAU_TARGET}。")
        print(f"   最好一格：{int(best.k_songs)} 首 × {best.window_s:.0f} 秒，"
              f"SE = {best.se:.4f}，是 τ 的 {best.se/TAU_TARGET:.1f} 倍。")
        # 外推：SE ∝ 1/√k ⇒ 需要多少首
        need_k = float(best.k_songs) * (best.se / TAU_TARGET) ** 2
        need_lo = float(best.k_songs) * (best.se / TAU_HI) ** 2
        print(f"   按 SE ∝ 1/√k 外推，需 **{need_k:.0f} 首/人**才够。")
        print(f"   由 τ 的 CI 给出区间：[{need_lo:.0f}, "
              + ("∞)" if TAU_LO <= 0 else f"{float(best.k_songs)*(best.se/TAU_LO)**2:.0f}]"))
        print("\n   🔴 与首版的差别 —— 首版写「需 98 首/人，与 `93` 的 97–184 同量级，")
        print("   两条独立路径再次吻合」。那句话两处都错：")
        print("   · 阈值 0.106 是从 `93` 抄来的常数 ⇒ 两条路径**不独立**（纪律 #12）")
        print("   · 那个 0.106 埋在自己的零分布里 ⇒ 阈值本身失效（纪律 #11）")
        print("   · 改成从权威表读之后**仍然不独立** —— 现在共享的是同一格的")
        print("     τ 估计值，堵了「抄常数」掉进「共享估计值」（纪律 14bis）。")
        print("     本脚本的 174 首/人**不是**独立于 `99g` 的第二条证据。")
        print(f"   改正后阈值降到 {TAU_TARGET:.4f}，所需首数按 1/τ² 放大 "
              f"{(0.106/TAU_TARGET)**2:.1f} 倍。")
        run.log_metric("min_se", float(best.se))
        run.log_metric("extrapolated_k_needed", need_k)
        run.log_metric("extrapolated_k_lower", need_lo)
    run.write()
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
