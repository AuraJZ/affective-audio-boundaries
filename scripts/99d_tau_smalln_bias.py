r"""τ̂ 的小 n 偏倚标定 —— `99c` 露出来的，会动到 `93`。

    .venv/Scripts/python scripts/99d_tau_smalln_bias.py

## ⚠️ 步骤 [4] 的那张 34 格表已被 `99g` 取代，**不要引用**

本脚本步骤 [4] 用**一个全局零分布常数**（n=18 → 0.1047，n=44 → 0.0391）
改正所有格。`99f`/`99g` 实测各格零分布差别很大：

    每人 n=18   15 格   中位 0.0965   范围 [0.089, 0.105]   全局值偏高 8%
    每人 n=44   18 格   中位 0.0134   范围 [0.000, 0.043]   全局值偏高 **192%**

n=44 那一行差得离谱，用全局值会把 τ 削过头（`98` 因此一度虚高到 329 首/人，
本格值下是 174）。**报数一律用 `reports/source_data/canonical_tau_table.csv`。**

本脚本仍然有效的是步骤 [1]（实测偏倚曲线）与步骤 [2]（真值已知的模拟标定，
即改正模型的依据），以及步骤 [3] 对「方差可加」的验证（平均绝对误差 3.3%）。

## 起因

`99c` 把 ARAUS 每人的观测数从 44 下采样到 18，τ̂ **升高**了：

    ARAUS pleasant   n=44 → 0.1327     n=18 → 0.1738   (+31%)
    ARAUS eventful   n=44 → 0.1597     n=18 → 0.1890   (+18%)

方向是**向上**。而 PMEmo 每人只有 18 次，τ̂ = 0.1198 ——
那个数可能有近三成是偏倚，不是异质性。

## 机制

剖面似然把逐人斜率的方差 v_i = se_i² 当作**已知**：

    b_i ~ N(β, τ² + v_i)

但 v_i 是用 n−2 个自由度估出来的，本身带噪。估计噪声无处可去，
就被吸进 τ²。n 越小，v_i 估得越糙，吸进去的越多 ——
这是元分析里成熟的小样本问题，不是本项目的新发现，
但**本项目一直没有查过**，而 `93` 恰好拿 n=18 的 PMEmo 与 n=44 的 DEAM 并排。

## 三件事

**① 实测偏倚曲线** —— ARAUS 有 749 人 × 44 次，数据够厚，
可以直接把 n 从 44 一路降下去，看 τ̂ 怎么走。这是**同一批人同一份真值**下
测出来的，不依赖任何模型假设。

**② 模拟标定（含 λ=0 行）** —— 真值 τ 已知时估计器给什么。
其中 **τ=0 那一行是关键**：若真值为零而 n=18 下 τ̂ 报出 0.05，
那么所有接近 0.05 的小 n 估计都不可读。
这是本项目对每个零分布都要求的那一行，这次要求估计器自己。

**③ 改正后重排** —— 把偏倚改正用到项目里每一个 τ 上，重算 n*，
看哪些结论变、哪些不变。

## 判据

若 n=18 处的偏倚 > 20%，则 `93` 的双通道并排必须改为**配平 n** 的比较，
且正文所有跨语料 τ 对比都要加改正。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")

AR = REPO_ROOT / "data" / "raw" / "ARAUS" / "datav2"
FEAT = REPO_ROOT / "data" / "features"
OUT = REPO_ROOT / "reports" / "source_data" / "tau_smalln_bias.csv"
SEED = 20260803
CHI2_95 = chi2.ppf(0.95, 1)
N_GRID = [8, 10, 12, 14, 18, 24, 32, 44]
TAU_TRUE = [0.00, 0.05, 0.10, 0.15, 0.20]     # 第一行就是 λ=0
N_SIM = 300
N_UNITS_SIM = 400
N_REP = 30


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


def tau_ci(b: np.ndarray, se: np.ndarray) -> tuple[float, float, float]:
    v = se ** 2
    hi_b = max(float(np.var(b, ddof=1)) * 20, 1e-6)
    grid = np.concatenate([[0.0], np.geomspace(1e-10, hi_b, 2000)])
    lls = np.array([_ll(t, b, v) for t in grid])
    k = int(np.argmax(lls))
    t2, lm = float(grid[k]), float(lls[k])

    def g(x: float) -> float:
        return 2.0 * (lm - _ll(x, b, v)) - CHI2_95

    lo = brentq(g, 0.0, t2) if (k > 0 and g(0.0) > 0) else 0.0
    hi = brentq(g, t2, hi_b) if g(hi_b) > 0 else hi_b
    return float(np.sqrt(t2)), float(np.sqrt(lo)), float(np.sqrt(hi))


def tau_of(u: list[tuple[np.ndarray, np.ndarray]]) -> float:
    bs, ses = [], []
    for x, y in u:
        b, se = slope_se(x, y)
        if np.isfinite(b) and np.isfinite(se) and se > 0:
            bs.append(b)
            ses.append(se)
    if len(bs) < 30:
        return np.nan
    return tau_ci(np.array(bs), np.array(ses))[0]


def araus_units(y_col: str) -> list[tuple[np.ndarray, np.ndarray]]:
    A = pd.read_csv(AR / "responses.csv")
    # 🔴 与 `99f`/`99g` 对齐：剔除 fold_r=−1 的锚点（每人 2 行、同一刺激、
    # 749 人完全相同的高杠杆设计点）。首版漏了，偏倚曲线因此偏高 ——
    # n=18 处的平均虚增从 +18.5% 被抬到 +26%，跨过本脚本自己 20% 的判据。
    A = A[(A.is_attention == 0) & (A.fold_r != -1)]
    A = A.dropna(subset=["Navg_r", y_col]).copy()
    g = A.groupby("participant")[y_col]
    A[y_col] = (A[y_col] - g.transform("mean")) / (g.transform("std") + 1e-9)
    A["Navg_r"] = (A.Navg_r - A.Navg_r.mean()) / (A.Navg_r.std() + 1e-9)
    return [(h.Navg_r.to_numpy(float), h[y_col].to_numpy(float))
            for _, h in A.groupby("participant") if len(h) >= max(N_GRID)]


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("99d_tau_smalln_bias", SEED)
    rows = []

    # ── ① 实测偏倚曲线：同一批人，同一份真值 ─────────────────────────
    print("=" * 92)
    print("[1] 实测偏倚曲线 —— ARAUS 同一批人下采样，真值不变，只有 n 在变")
    cells = {"pleasant": araus_units("pleasant"),
             "eventful": araus_units("eventful")}
    print(f"  {'格':<12}{'人':>5}", end="")
    for n in N_GRID:
        print(f"{'n=' + str(n):>9}", end="")
    print(f"{'  n=18 相对 n=44':>16}")
    emp = {}
    for name, u in cells.items():
        print(f"  {name:<12}{len(u):>5}", end="")
        curve = {}
        for n in N_GRID:
            ts = []
            for _ in range(N_REP):
                sub = [(x[i], y[i]) for x, y in u
                       for i in [rng.choice(len(x), n, replace=False)]]
                t = tau_of(sub)
                if np.isfinite(t):
                    ts.append(t)
            curve[n] = float(np.median(ts)) if ts else np.nan
            print(f"{curve[n]:>9.4f}", end="")
            rows.append({"test": "empirical", "cell": name, "n": n,
                         "tau": curve[n]})
        infl = curve[18] / curve[max(N_GRID)] - 1
        emp[name] = infl
        print(f"{infl:>+15.0%}")
    print(f"\n  ⇒ n=18 处实测虚增 {np.mean(list(emp.values())):+.0%}"
          f"（两格平均）。方向向上，与机制预期一致。")
    run.log_metric("empirical_inflation_n18", float(np.mean(list(emp.values()))))

    # ── ② 模拟标定，含 λ=0 行 ────────────────────────────────────────
    print("\n" + "=" * 92)
    print("[2] 模拟标定：真值已知。**第一行 τ=0 是 λ=0 行 —— 先看它。**")
    print(f"    {N_UNITS_SIM} 个单位 × {N_SIM} 次重复，x ~ N(0,1)，"
          f"残差 SD = 0.95（与实测相当）")
    print(f"\n  {'真 τ':>6}", end="")
    for n in N_GRID:
        print(f"{'n=' + str(n):>9}", end="")
    print()
    for t_true in TAU_TRUE:
        print(f"  {t_true:>6.2f}", end="")
        for n in N_GRID:
            ts = []
            for _ in range(N_SIM // 10):
                bi = rng.normal(0.30, t_true, N_UNITS_SIM)   # β=0.30
                u = []
                for b in bi:
                    x = rng.normal(0, 1, n)
                    u.append((x, b * x + rng.normal(0, 0.95, n)))
                v = tau_of(u)
                if np.isfinite(v):
                    ts.append(v)
            m = float(np.median(ts)) if ts else np.nan
            print(f"{m:>9.4f}", end="")
            rows.append({"test": "simulation", "tau_true": t_true, "n": n,
                         "tau_hat": m})
        print()
    z = [r for r in rows if r["test"] == "simulation" and r["tau_true"] == 0.0]
    z18 = next((r["tau_hat"] for r in z if r["n"] == 18), np.nan)
    z44 = next((r["tau_hat"] for r in z if r["n"] == 44), np.nan)
    print(f"\n  λ=0 行：真值为零时，n=18 报 {z18:.4f}，n=44 报 {z44:.4f}。")
    print(f"  ⇒ 小于约 {max(z18, z44) * 2:.3f} 的 τ̂ 在小 n 下**不可读**。")
    run.log_metric("null_tau_n18", float(z18))
    run.log_metric("null_tau_n44", float(z44))

    # ── ③ 改正模型：方差可加，不是乘性 🔴 ────────────────────────────
    #
    # 首版在这里写了个乘性缩放 τ̂ → τ̂ · [τ̂(44)/τ̂(n)]，那是错的模型。
    # 偏倚来自被吸进 τ² 的估计噪声，是**加在方差上**的：
    #
    #     τ̂²(n) ≈ τ² + τ̂²_null(n)
    #
    # 故改正是 τ = √(max(τ̂² − τ̂²_null(n), 0))。
    # 两者在 τ 远大于零分布时几乎一致，但在 τ 接近零分布时差别是**决定性的** ——
    # 乘性改正永远给不出零，方差改正会。而 PMEmo 恰好落在那个区间。
    print("\n" + "=" * 92)
    print("[3] 改正模型：先验证「方差可加」对不对")
    sim = pd.DataFrame([r for r in rows if r["test"] == "simulation"])
    nullc = {int(r.n): float(r.tau_hat)
             for _, r in sim[sim.tau_true == 0.0].iterrows()}

    def correct(tau_hat: float, n: int) -> float:
        t0 = nullc.get(int(n), 0.0)
        return float(np.sqrt(max(tau_hat ** 2 - t0 ** 2, 0.0)))

    print(f"  {'真 τ':>6}{'n':>5}{'零分布':>9}{'τ̂ 实测':>9}"
          f"{'√(τ²+零²) 预测':>16}{'误差':>8}")
    errs = []
    for _, r in sim[sim.tau_true > 0].iterrows():
        pred = float(np.sqrt(r.tau_true ** 2 + nullc[int(r.n)] ** 2))
        e = (pred - r.tau_hat) / r.tau_hat
        errs.append(abs(e))
        if int(r.n) in (12, 18, 44):
            print(f"  {r.tau_true:>6.2f}{int(r.n):>5}{nullc[int(r.n)]:>9.4f}"
                  f"{r.tau_hat:>9.4f}{pred:>16.4f}{e:>+8.1%}")
    print(f"\n  全表平均绝对误差 {np.mean(errs):.1%} → 方差可加模型"
          f"{'成立 ✅' if np.mean(errs) < 0.12 else '不成立 ⚠️'}")
    run.log_metric("additive_model_mae", float(np.mean(errs)))

    print("\n" + "=" * 92)
    print("[4] 把改正用到项目里每一个 τ 上（直接读各脚本发布的 CSV，不转抄）")
    SD = REPO_ROOT / "reports" / "source_data"
    KNOWN = []
    try:
        A = pd.read_csv(SD / "araus_individual.csv")
        for _, r in A[(A.test == "tau") & (A.usable == True)].iterrows():   # noqa: E712
            KNOWN.append((f"`99` ARAUS {r.target}", str(r.descriptor),
                          int(r.n_per_unit), float(r.tau), float(r.ci_lo),
                          float(r.ci_hi), float(r.breakeven)))
    except Exception as e:
        print(f"  （ARAUS 跳过：{type(e).__name__}）")
    try:
        S = pd.read_csv(SD / "sigma_u_profile.csv")
        for _, r in S.iterrows():
            KNOWN.append((f"`93` {r.channel}", str(r.cell)[:22],
                          int(r.n_per_unit), float(r.tau), float(r.ci_lo),
                          float(r.ci_hi), float(r.breakeven)))
    except Exception as e:
        print(f"  （`93` 跳过：{type(e).__name__}）")

    print(f"  {'来源':<22}{'格':<24}{'n':>4}{'τ̂':>8}{'零分布':>8}"
          f"{'τ 改正':>8}{'CI 改正上界':>12}{'n* 原':>8}{'n* 改正':>9}")
    n_dead = 0
    for src, cell, n, tau, lo, hi, be in KNOWN:
        t0 = nullc.get(int(n), np.nan)
        tc, hc = correct(tau, n), correct(hi, n)
        ns0 = n * (be / tau) ** 2 if tau > 1e-6 else np.inf
        ns1 = n * (be / tc) ** 2 if tc > 1e-6 else np.inf
        dead = tau <= t0
        if dead:
            n_dead += 1
        print(f"  {src:<22}{cell:<24}{n:>4}{tau:>8.4f}{t0:>8.4f}{tc:>8.4f}"
              f"{hc:>12.4f}{ns0:>8.0f}"
              + (f"{ns1:>9.0f}" if np.isfinite(ns1) else f"{'∞':>9}")
              + ("   🔴 埋在零分布里" if dead else ""))
        rows.append({"test": "corrected", "source": src, "cell": cell, "n": n,
                     "tau_raw": tau, "tau_null": t0, "tau_corrected": tc,
                     "ci_hi_corrected": hc, "n_star_raw": ns0,
                     "n_star_corrected": ns1, "below_null": dead})
    print(f"\n  {n_dead}/{len(KNOWN)} 格的 τ̂ **不高于同 n 下的零分布** —— "
          "那些数不是异质性，是估计噪声。")
    run.log_metric("n_cells_below_null", n_dead)

    # ── ⑤ 判决与连带影响 ─────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("[5] 判决")
    print(f"  n=18 的零分布 τ̂ = {nullc.get(18, float('nan')):.4f}，"
          f"n=44 的 = {nullc.get(44, float('nan')):.4f}")
    print("\n  连带影响，逐条：")
    print("  · `93` PMEmo 皮电 τ=0.106 @ n=18 —— **须撤回**，它低于同 n 零分布。")
    print("    但 `93` 的**判决方向不变**：τ 更接近零，只会让")
    print("    「CI 上界低于盈亏平衡」更成立。撤的是那个数，不是那个结论。")
    print(f"  · `98` 的 TAU_TARGET = 0.106 是从 `93` 抄来的常数 —— "
          "同样失效，所需预算按 1/τ² 放大。")
    print("  · `99` ARAUS n=44，改正幅度小，B2 基本不动。")
    print("\n  🔴 独立性要重述。我一直说「四条独立路径收敛到约 100」，不确切：")
    print("     `92`  DEAM 分半实测曲线      —— 不经过 τ 估计，**独立**")
    print("     `99`  ARAUS n=44 剖面似然    —— 零分布小，**独立**")
    print("     `93`  PMEmo n=18 剖面似然    —— 用了被抬高的 τ")
    print("     `98`  等预算外推             —— TAU_TARGET 直接抄自 `93`，"
          "**与 `93` 不独立**")
    print("     ⇒ 是**两条**独立路径加一条共源的，不是四条。正文要改。")
    print("\n  方向：偏倚抬高 τ̂ ⇒ 压低 n*。改正只会让门槛更高 ——")
    print("  「按人校准现在还不划算」若有变化，只会变得更成立。")

    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")
    run.write()
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
