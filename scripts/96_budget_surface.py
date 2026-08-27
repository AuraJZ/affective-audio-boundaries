r"""⚠️ 本脚本已被 98_budget_surface_v2.py 取代，**不要引用其数字**。

数据源 pmemo_frames.parquet 的歌曲级均值恒为零（82 的 detrend 所致），
且阈值 TAU_TARGET 抄自 93 而那个数埋在自己的零分布里。
保留仅为记录首版错在哪。定稿见 reports/tables/canonical_tau_table.md。

C1：等预算等高面 —— 固定总测量时长，多听几首 vs 多测几段。

    .venv/Scripts/python scripts/96_budget_surface.py

## 这张图要回答的工程问题

`93` 给出：按人校准的门槛在每人约 100–180 次观测。
但「一次观测」不是唯一的花钱方式 —— 总预算可以拆成两个轴：

    k = 听几首歌      每多一首是一个**独立**的新观测
    m = 每首测几段    每多测一段只是把**同一首**测得更准

预算 ≈ k × m（总测量时长）。问题：固定预算，怎么分？

## 为什么答案不是显然的

若段与段之间独立，则

    var(斜率) ∝ (σ_真实² + σ_测量²/m) / k

固定 k·m = B 时，这个式子对 m 单调递增 → 永远该多听歌、少测段。

**但帧之间高度自相关。** 多测一倍时长并不能把该歌的测量噪声减半 ——
有效段数 m_有效 远小于 m，且会饱和。饱和点在哪，只能实测。

## 做法

`pmemo_frames.parquet`（585,098 行，735 首 × 401 人 × 中位 74 帧）
同时提供两个轴。对每个 (k, m)：

    1. 该听众随机抽 k 首歌
    2. 每首随机取一段**连续** m 帧（连续而非随机抽帧 —— 真实系统是连续测的，
       且这样才保留自相关，随机抽帧会人为地把噪声降下去）
    3. 段内均值作为该歌的观测值，回归得该人的斜率
    4. 取该次回归的**解析标准误**，重抽 R 次后取中位

🔴 **不能用「重抽之间斜率的标准差」当 SE。** 每人只有 18 首，
反复从中抽 k 首用的是**同一批数据** —— k → 18 时重抽方差趋近 0，
而真实的估计误差并不会。首版这么写，得出「4 首 × 5 段就够」，
与 `92`（n=100 才显形）和 `93`（门槛 97–184）直接矛盾。三者不可能都对。

同样地，描述符必须**全局标准化**（与 `93` 一致）——
否则斜率的单位是「每 dB」，与 τ = 0.106 的「每 SD」不可比。

**判据**：把 SE(k,m) 与 `93` 实测的 τ 比。SE < τ 处，个体差异开始可用。
等预算曲线 k·m = B 与该阈值的交点，即最小可行预算及其最优拆分。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")

FEAT = REPO_ROOT / "data" / "features"
OUT = REPO_ROOT / "reports" / "source_data" / "budget_surface.csv"
SEED = 20260802
DESC = "rms_db"                # `76` 起的先验主描述符
K_GRID = [4, 6, 9, 12, 18]     # 听几首
M_GRID = [5, 10, 20, 40, 74]   # 每首测几段（0.5 s/段 ⇒ 2.5–37 s）
N_REP = 40
MIN_FRAMES = 40
MIN_SONGS = 14
# `93` 在 PMEmo 皮电上实测的个体间斜率 SD（中位），作为可用性阈值
TAU_TARGET = 0.106


def prepare() -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """逐听众 → 其各首歌的 (声学帧序列, 皮电帧序列)。

    描述符全局标准化 —— 与 `93` 同口径，否则斜率单位不可比。
    """
    D = pd.read_parquet(FEAT / "pmemo_frames.parquet")
    D[DESC] = (D[DESC] - D[DESC].mean()) / (D[DESC].std() + 1e-9)
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for lid, g in D.groupby("listener"):
        songs = []
        for _, s in g.groupby("musicId"):
            s = s.sort_values("frame")
            if len(s) >= MIN_FRAMES:
                songs.append((s[DESC].to_numpy(float),
                              s["eda"].to_numpy(float)))
        if len(songs) >= MIN_SONGS:
            out[lid] = songs
    return out


def slope_at(songs, k: int, m: int, rng) -> float:
    """抽 k 首、每首连续 m 帧 → 该人斜率的**解析标准误**。

    返回 SE 而非斜率本身：本分析要的是「这个人的斜率能被估到多准」，
    而不是「斜率是多少」。
    """
    if len(songs) < k:
        return np.nan
    pick = rng.choice(len(songs), k, replace=False)
    xs, ys = [], []
    for i in pick:
        a, e = songs[i]
        if len(a) < m:
            continue
        s0 = rng.integers(0, len(a) - m + 1)      # 连续窗，保留自相关
        xs.append(float(a[s0:s0 + m].mean()))
        ys.append(float(e[s0:s0 + m].mean()))
    if len(xs) < 4:
        return np.nan
    x, y = np.array(xs), np.array(ys)
    n = len(x)
    xc = x - x.mean()
    sxx = float((xc ** 2).sum())
    if sxx < 1e-12:
        return np.nan
    b = float((xc * (y - y.mean())).sum() / sxx)
    resid = y - y.mean() - b * xc
    s2 = float((resid ** 2).sum() / (n - 2))
    return float(np.sqrt(s2 / sxx))


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("96_budget_surface", SEED)
    per = prepare()
    print(f"听众 {len(per)}   每人歌数中位 "
          f"{np.median([len(v) for v in per.values()]):.0f}\n")

    rows = []
    print("=" * 88)
    print("斜率估计误差 SE(k, m) —— 越小越好；阈值 τ = "
          f"{TAU_TARGET:.3f}（`93` 实测个体间 SD）")
    print(f"  {'k(首)':>6}{'m(段)':>7}{'预算 k×m':>10}{'时长(s)':>9}"
          f"{'SE':>9}{'SE/τ':>8}{'可用':>6}")
    for k in K_GRID:
        for m in M_GRID:
            ses = []
            for lid, songs in per.items():
                v = [slope_at(songs, k, m, rng) for _ in range(N_REP)]
                v = np.array([x for x in v if np.isfinite(x)])
                if len(v) >= N_REP // 2:
                    ses.append(float(np.median(v)))   # 解析 SE 的中位
            if len(ses) < 50:
                continue
            se = float(np.median(ses))
            ratio = se / TAU_TARGET
            usable = se < TAU_TARGET
            print(f"  {k:>6}{m:>7}{k*m:>10}{k*m*0.5:>9.0f}{se:>9.4f}"
                  f"{ratio:>8.2f}{'  ✅' if usable else '':>6}", flush=True)
            rows.append({"k_songs": k, "m_frames": m, "budget": k * m,
                         "seconds": k * m * 0.5, "se": se, "se_over_tau": ratio,
                         "usable": usable, "n_listeners": len(ses)})

    R = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT, index=False, encoding="utf-8")

    # ── 固定预算下的最优拆分 ─────────────────────────────────────────
    print("\n" + "=" * 88)
    print("固定预算下，怎么拆最好")
    print(f"  {'预算 k×m':>10}{'最优 (k, m)':>14}{'最优 SE':>10}"
          f"{'最差 (k, m)':>14}{'最差 SE':>10}{'差距':>8}")
    R["bucket"] = pd.cut(R.budget, [0, 60, 120, 250, 500, 10000],
                         labels=["≤60", "60–120", "120–250", "250–500", ">500"])
    for b, g in R.groupby("bucket", observed=True):
        if len(g) < 2:
            continue
        best, worst = g.loc[g.se.idxmin()], g.loc[g.se.idxmax()]
        print(f"  {str(b):>10}{f'({best.k_songs}, {best.m_frames})':>14}"
              f"{best.se:>10.4f}{f'({worst.k_songs}, {worst.m_frames})':>14}"
              f"{worst.se:>10.4f}{worst.se/best.se:>8.2f}×")
        rows.append({"kind": "optimum", "bucket": str(b),
                     "best_k": int(best.k_songs), "best_m": int(best.m_frames),
                     "best_se": float(best.se)})

    # ── 两个轴各自的边际收益 ─────────────────────────────────────────
    print("\n" + "=" * 88)
    print("两个轴的边际收益：加倍之后 SE 降多少（理想独立时应为 0.71×）")
    for axis, grid, other in (("k 首数", K_GRID, "m_frames"),
                              ("m 段数", M_GRID, "k_songs")):
        col = "k_songs" if axis.startswith("k") else "m_frames"
        gains = []
        for fixed, g in R.groupby(other):
            g = g.sort_values(col)
            for a, b in zip(grid[:-1], grid[1:]):
                if b / a < 1.5 or b / a > 2.5:
                    continue
                va = g[g[col] == a].se
                vb = g[g[col] == b].se
                if len(va) and len(vb):
                    gains.append(float(vb.iloc[0] / va.iloc[0]))
        if gains:
            print(f"  {axis:<8} 加倍后 SE 变为 {np.median(gains):.3f}× "
                  f"（n={len(gains)} 对）"
                  f"{'   ← 接近理想' if np.median(gains) < 0.80 else '   ← 远未达理想，已饱和'}")
            run.log_metric(f"marginal_{col}", float(np.median(gains)))

    usable = R[R.usable]
    print("\n" + "=" * 88)
    if len(usable):
        cheapest = usable.loc[usable.budget.idxmin()]
        print(f"最小可行预算：k={int(cheapest.k_songs)} 首 × "
              f"m={int(cheapest.m_frames)} 段 = {int(cheapest.budget)} 帧 "
              f"≈ {cheapest.seconds:.0f} 秒/人   SE={cheapest.se:.4f} < τ={TAU_TARGET}")
        run.log_metric("min_feasible_budget", int(cheapest.budget))
        run.log_metric("min_feasible_k", int(cheapest.k_songs))
        run.log_metric("min_feasible_m", int(cheapest.m_frames))
    else:
        print(f"⚠️ 所测范围内**没有**任何 (k, m) 组合使 SE < τ = {TAU_TARGET}。")
        print(f"   最好的一格 SE = {R.se.min():.4f}，是 τ 的 {R.se.min()/TAU_TARGET:.1f} 倍。")
        print("   → 本数据集的规模（每人 ≤18 首）**够不到**可用区，")
        print("     这正是 §10 要画的那件事：现存语料全部落在阈值之外。")
        run.log_metric("min_se_achieved", float(R.se.min()))
        run.log_metric("min_se_over_tau", float(R.se.min() / TAU_TARGET))
    run.write()
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
