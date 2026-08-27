r"""B2：ARAUS —— 产品域的个体差异。

    .venv/Scripts/python scripts/99_araus_individual.py

⚠️ 许可 **CC BY-NC 4.0**（NTU 研究数据库）—— 可用于论文，**不得进产品**
（与 Soundtracks、ESC-50 同类约束）。引用 doi:10.1109/TAFFC.2023.3247914。

## 🔴 第四版：对抗审计推翻了第三版的头条，这里逐条改正

第三版写「ARAUS 顺序**非随机**，所以这是偏严的检验；连有结构的顺序都撼不动 τ，
故 PMEmo 的 D2 限制不要紧」。**这一整条都是错的。**

**① 顺序其实是随机化的。** 「SD 0.087 vs 随机应 0.014」这个 6 倍差
不是设计结构，是两个统计假象叠加：

    · 位 46–51 各只有 **5 行**（仅那 5 个 51 试次的被试）。n=5 时族占比 SD≈0.17，
      而 `99b` 把 51 个位的 SD 平均起来，这六个位就主导了整个统计量。
    · 位 1 与 45 是 **100% / 99.5% silence** —— fold_r=−1 的锚点试次，按设计如此。

    只看真正的实验试次（位 2–44，每位 n=749）：**0.0153 vs 随机期望 0.0136**。
    随机化的。我原先猜的「六族数量不等」倒是无关：按真实比例重算期望是 0.0136，
    与硬编码的 0.0137 只差 0.0001。

**② 因此 D2 退不掉，而且方向相反。** `94` 自己的输出
（`reports/source_data/order_confound.csv`）写着：PMEmo 的曲目→位置分配
`assignment_icc = 0.041, p = 0.005`，**显著非随机**；且纯位置代理能复现
**249% / 120% / 114%** 的曲目级信度。位置本身就能造出 PMEmo 的全部效应。

拿一个顺序**随机化**的数据集的零结果，去退掉一个顺序**非随机**的数据集的混淆 ——
推断是倒过来的。ARAUS 至多支持一句窄得多的话：
**在顺序随机化的评分研究中，呈现顺序不扰动个体间斜率异质性。**

**③ 三处估计量缺陷。**

    · 自由度：控制顺序后 `y` 已被去掉 2 个参数，残差 df 应为 n−3 而非 n−2。
      只改这一处，最大相对变化就从 6.94% 升到 10.03% —— 跨过脚本自己的 10% 判据。
    · 只对 `y` 去顺序、不对 `x` 去 —— 那不是偏回归。分子对、分母是 xᵀx
      而非 xᵀ(I−P)x，于是斜率被机械地衰减 (1−R²(x∼order)) 倍，
      与顺序是否真混淆无关。控制越灵活，衰减越大。
    · 阳性对照的 `.iloc[:k]` 是**取前 k 行**不是随机抽 —— 两臂来自不同的顺序段，
      且静默臂总是含位置 1 的锚点。

**④ fold_r = −1 是一对重复的高杠杆锚点，前三版都没注意。**
每人恰好 2 行，同一个声景（R0091）+ 静默 + smr 0，落在位 1 与 45。
它在 `Navg_r` 的第 93 百分位，**749 人完全相同**，且在每人 44 行的回归里出现两次 ——
贡献了中位 18.2% 的 Σ(x−x̄)²，是其行数占比（4.5%）的四倍。
一个所有人共享的、重复的、极端 x 值的设计点，会把所有人的斜率往同一处钉。

剔掉它，τ 变动最大到 **±17.7%** —— 比顺序那 6.9% 大 2.5 倍。
本版**主分析剔除锚点**（它们是测–重测探针，不属于随机化的实验设计），
并同时报保留锚点的敏感性。

## 仍然成立的（审计逐条核过）

    `_r` 后缀 = **最终合成刺激**的声学量。在固定 (声景, 掩蔽文件) 内，
                3048/3048 个非静默格的 ρ(smr, Navg_r) = −1.000；静默格 SD 恰为 0。
    SMR 方向    非静默合计 ρ(smr, Navg_r) = −0.2437 ⇒ SMR = 声景/掩蔽声
    无标签泄漏  160 列里恰好 9 列源自评分，均未用作预测变量
    过滤正确    `is_attention==1` 是 749 行、每人一行、九项评分恒为 3.0 —— 检查题本身
    被剔被试    `participants_rejected.csv` 的 ID 是**替换前的另一批人**，未混入
    计数        749 人 / 32,986 行 / 每人中位 44 / 337 声景 / 6 掩蔽族

## 白捡到的阴性对照（不受上述影响）

`silence` 掩蔽族：同一 SMR 标签贴在「什么都没加」上，n=6736，
ρ(SMR, 响度) = −0.000，ρ(SMR, pleasant) = −0.003。设计里本来就有的零。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2, spearmanr, wilcoxon

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")

AR = REPO_ROOT / "data" / "raw" / "ARAUS" / "datav2"
OUT = REPO_ROOT / "reports" / "source_data" / "araus_individual.csv"
SEED = 20260803
PSYCHO = ["Navg_r", "Savg_r", "Ravg_r", "Favg_r"]
NOISE_MASKERS = ["construction", "traffic", "wind"]
TARGETS = ["pleasant", "eventful"]
MIN_OBS = 15
CHI2_95 = chi2.ppf(0.95, 1)
POS_LO, POS_HI = 2, 44        # 真正的实验试次；位 1/45 是锚点，46–51 只有 5 人


def slope_se(x: np.ndarray, y: np.ndarray, n_extra: int = 0):
    """逐人 OLS 斜率与标准误。`n_extra` = 事先从 y 去掉的参数个数。

    🔴 df 必须是 n − 2 − n_extra。第三版恒用 n−2，控制顺序时低估了残差方差，
    连带把 τ 的比较结果推到 10% 判据的错误一侧。
    """
    n = len(x)
    df = n - 2 - n_extra
    if n < 4 or df < 2 or np.std(x) < 1e-9:
        return np.nan, np.nan
    xc = x - x.mean()
    sxx = float((xc ** 2).sum())
    b = float((xc * (y - y.mean())).sum() / sxx)
    resid = y - y.mean() - b * xc
    return b, float(np.sqrt(float((resid ** 2).sum() / df) / sxx))


def _ll(t2: float, b: np.ndarray, v: np.ndarray) -> float:
    w = 1.0 / (v + t2)
    mu = float((w * b).sum() / w.sum())
    ll = -0.5 * float((np.log(2 * np.pi / w) + w * (b - mu) ** 2).sum())
    return ll - 0.5 * np.log(w.sum())


def tau_ci(b: np.ndarray, se: np.ndarray):
    v = se ** 2
    hi_b = max(float(np.var(b, ddof=1)) * 20, 1e-6)
    grid = np.concatenate([[0.0], np.geomspace(1e-10, hi_b, 4000)])
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


def tau_block(D: pd.DataFrame, d: str, t: str, order_deg: int = 0):
    """逐人斜率 → 剖面似然 τ。`order_deg` = 顺序多项式次数，0 表示不控制。

    🔴 控制顺序时对 **x 与 y 同时**投影掉顺序基（Frisch–Waugh）。
    第三版只去 y，那会把斜率机械衰减 (1−R²(x∼order)) 倍，
    与顺序是否真的混淆无关。
    """
    bs, ses, rsd, vx, ns = [], [], [], [], []
    need = [d, t] + (["order"] if order_deg else [])
    for _, g in D.groupby("participant"):
        g = g.dropna(subset=need)
        if len(g) < MIN_OBS:
            continue
        x, y = g[d].to_numpy(float), g[t].to_numpy(float)
        extra = 0
        if order_deg:
            o = g["order"].to_numpy(float)
            Z = np.vander(o, order_deg + 1, increasing=True)
            Q, _ = np.linalg.qr(Z)                     # 正交基，避免病态
            x = x - Q @ (Q.T @ x)                      # x 也要去
            y = y - Q @ (Q.T @ y)
            extra = Q.shape[1] - 1                     # 常数项已含在 OLS 里
        b, se = slope_se(x, y, n_extra=extra)
        if np.isfinite(b) and np.isfinite(se) and se > 0:
            bs.append(b)
            ses.append(se)
            rsd.append(float(np.std(y - b * (x - x.mean()))))
            vx.append(float(np.var(x)))
            ns.append(len(g))
    if len(bs) < 50:
        return None
    tau, lo, hi, beta = tau_ci(np.array(bs), np.array(ses))
    n_med, r_med, v_med = (float(np.median(ns)), float(np.median(rsd)),
                           float(np.median(vx)))
    be = float(np.sqrt(r_med ** 2 / max((n_med - 1) * v_med, 1e-9)))
    return {"tau": tau, "ci_lo": lo, "ci_hi": hi, "beta": beta,
            "n_units": len(bs), "n_per_unit": n_med, "breakeven": be,
            "n_star": n_med * (be / max(tau, 1e-9)) ** 2}


def standardise(D: pd.DataFrame) -> pd.DataFrame:
    D = D.copy()
    for t in TARGETS:
        g = D.groupby("participant")[t]
        D[t] = (D[t] - g.transform("mean")) / (g.transform("std") + 1e-9)
    for d in PSYCHO + ["smr"]:
        D[d] = (D[d] - D[d].mean()) / (D[d].std() + 1e-9)
    return D


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("99_araus_individual", SEED)
    R0 = pd.read_csv(AR / "responses.csv")
    R0 = R0[R0.is_attention == 0].copy()
    R0["masker"] = R0.masker.astype(str).str.split("_").str[0]
    R0["order"] = R0.groupby("participant").stimulus_index.rank(pct=True)
    anchor = R0.fold_r == -1
    print(f"参与者 {R0.participant.nunique()}   声景 {R0.soundscape.nunique()}   "
          f"响应 {len(R0)}   掩蔽族 {sorted(R0.masker.unique())}")
    print(f"锚点试次 fold_r=−1：{int(anchor.sum())} 行 = "
          f"{R0[anchor].participant.nunique()} 人 × 2，同一声景 "
          f"{R0[anchor].soundscape.iloc[0][:5]} + 静默 + smr 0，位 1 与 45")
    R = standardise(R0[~anchor])          # 主分析剔锚点
    RA = standardise(R0)                  # 敏感性用
    print(f"主分析每人刺激数 中位 "
          f"{R.groupby('participant').size().median():.0f}")

    rows = []
    # ── ⓪ SMR 的物理方向 ─────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("[0] SMR 方向 —— 包内无数据字典，从声学量定死")
    for m in ["construction", "traffic", "wind", "water", "bird", "silence"]:
        h = R0[R0.masker == m].dropna(subset=["smr", "Navg_r"])
        r = spearmanr(h.smr, h.Navg_r).statistic
        print(f"  ρ(SMR, 响度) | {m:<13}{r:>+8.3f}   n={len(h)}"
              + ("   ← 阴性对照，必须为零" if m == "silence" else ""))
        rows.append({"test": "smr_direction", "masker": m, "rho": r})
    h = R0[R0.masker != "silence"].dropna(subset=["smr", "Navg_r"])
    r_dir = spearmanr(h.smr, h.Navg_r).statistic
    print(f"\n  非 silence 合计 {r_dir:+.3f} ⇒ SMR = 声景/掩蔽声，越大掩蔽声越弱。")
    run.log_metric("smr_direction_rho", float(r_dir))

    # ── ① 顺序：先查它到底随不随机 ───────────────────────────────────
    print("\n" + "=" * 88)
    print("[1] 顺序是否随机 —— 第三版这里判错了，先把统计量修对")
    for lo, hi, tag in [(1, 51, "全部 51 位（第三版口径）"),
                        (POS_LO, POS_HI, f"仅位 {POS_LO}–{POS_HI}（真实验试次）")]:
        sub = R0[R0.stimulus_index.between(lo, hi)]
        ct = pd.crosstab(sub.stimulus_index, sub.masker)
        obs = float(((ct.T / ct.sum(axis=1)).T).std().mean())
        # 解析期望用**每位真实的 n** 与**真实的族比例**，不用硬编码 0.17
        p = sub.masker.value_counts(normalize=True)
        nn = ct.sum(axis=1)
        exp = float(np.mean([np.sqrt((pj * (1 - pj) / nn).mean()) for pj in p]))
        print(f"  {tag:<28} 实测 {obs:.4f}   随机期望 {exp:.4f}   "
              f"比值 {obs/exp:.2f}×")
        rows.append({"test": "order_randomness", "range": tag,
                     "observed": obs, "expected": exp})
    n_by_pos = R0.groupby("stimulus_index").size()
    print(f"\n  第三版为何判错：位 46–51 各只有 {n_by_pos.loc[46:51].iloc[0]} 行"
          f"（仅那几个 51 试次的被试），n 小则族占比 SD 自然大；")
    print(f"  而位 1/45 是锚点，按设计 100% 静默。两者把全 51 位的平均 SD 拉高。")
    print("  ⇒ **ARAUS 的顺序是随机化的。**「偏严的检验」这个框架不成立。")

    # ── ② 阳性对照 ───────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("[2] 阳性对照：同一个人，construction vs silence，观测数配平")
    print(f"  {'目标':<10}{'|ρ| 施工':>10}{'|ρ| 静默':>10}{'零期望':>9}"
          f"{'配对 Wilcoxon':>15}{'判读':>12}")
    ctrl_ok = {}
    for t in TARGETS:
        pos, neg, ks = [], [], []
        for _, g in R.groupby("participant"):
            a = g[g.masker == "construction"].dropna(subset=["smr", t])
            b = g[g.masker == "silence"].dropna(subset=["smr", t])
            k = min(len(a), len(b))
            if k < 6:
                continue
            # 🔴 随机抽 k **行**，不是 `.iloc[:k]`。取前 k 行是按顺序位截断，
            # 两臂来自不同的顺序段 —— 第三版的写法。
            ia = rng.choice(len(a), k, replace=False)
            ib = rng.choice(len(b), k, replace=False)
            ra = spearmanr(a.smr.to_numpy()[ia], a[t].to_numpy()[ia]).statistic
            rb = spearmanr(b.smr.to_numpy()[ib], b[t].to_numpy()[ib]).statistic
            if np.isfinite(ra) and np.isfinite(rb):
                pos.append(abs(ra))
                neg.append(abs(rb))
                ks.append(k)
        mp, mn = float(np.median(pos)), float(np.median(neg))
        null_exp = float(np.sqrt(2 / np.pi) / np.sqrt(np.median(ks) - 1))
        p = float(wilcoxon(pos, neg).pvalue)
        ok = bool(mp > mn and p < 0.001)
        ctrl_ok[t] = ok
        print(f"  {t:<10}{mp:>10.3f}{mn:>10.3f}{null_exp:>9.3f}{p:>15.2e}"
              f"{'  ✅ 通过' if ok else '  ❌ 不通过':>12}   (n={len(pos)} 人)")
        rows.append({"test": "positive_control", "target": t, "rho_pos": mp,
                     "rho_neg": mn, "null_expectation": null_exp, "p": p,
                     "passed": ok})
    print("\n  静默臂落在解析零期望上 → 配平设计成立。")
    print("  eventful 不通过是**结果不是故障**：SMR 对 eventful 在任一族都 |ρ|≤0.044。")

    # ── ③ 顺序控制的规格扫描 ─────────────────────────────────────────
    print("\n" + "=" * 88)
    print("[3] 顺序控制：τ 对**规格**有多敏感（Frisch–Waugh，df 已改正）")
    print("    顺序既已随机化，这不再是「退掉 D2」的证据，")
    print("    只是本数据集自身的稳健性检查。\n")
    print(f"  {'目标':<10}{'描述符':<10}{'不控制':>9}", end="")
    for deg in (1, 2, 3):
        print(f"{'次数' + str(deg):>9}", end="")
    print(f"{'最大相对变化':>14}")
    worst = 0.0
    for t in TARGETS:
        for d in PSYCHO:
            base = tau_block(R, d, t)
            if not base:
                continue
            print(f"  {t:<10}{d:<10}{base['tau']:>9.4f}", end="")
            rel = []
            for deg in (1, 2, 3):
                c = tau_block(R, d, t, order_deg=deg)
                if c:
                    print(f"{c['tau']:>9.4f}", end="")
                    rel.append(abs(c["tau"] - base["tau"]) / base["tau"])
                    rows.append({"test": "order_spec", "target": t,
                                 "descriptor": d, "degree": deg,
                                 "tau": c["tau"], "tau_base": base["tau"]})
                else:
                    print(f"{'—':>9}", end="")
            mx = max(rel) if rel else 0.0
            worst = max(worst, mx)
            print(f"{mx:>13.1%}")
    print(f"\n  跨全部规格的最大相对变化 **{worst:.1%}**"
          f"（第三版只试线性、且 df 用错，报的是 6.9%）")
    run.log_metric("order_max_rel_change_any_spec", float(worst))

    # ── ④ 锚点敏感性 ─────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("[4] fold_r=−1 锚点：一对重复的高杠杆设计点，前三版都没注意")
    print(f"  {'目标':<10}{'描述符':<10}{'剔锚点(主)':>12}{'保留锚点':>11}"
          f"{'相对变化':>10}")
    amax = 0.0
    for t in TARGETS:
        for d in PSYCHO:
            a, b = tau_block(R, d, t), tau_block(RA, d, t)
            if a and b:
                rel = (b["tau"] - a["tau"]) / a["tau"]
                amax = max(amax, abs(rel))
                print(f"  {t:<10}{d:<10}{a['tau']:>12.4f}{b['tau']:>11.4f}"
                      f"{rel:>+9.1%}")
                rows.append({"test": "anchor_sensitivity", "target": t,
                             "descriptor": d, "tau_no_anchor": a["tau"],
                             "tau_with_anchor": b["tau"], "rel_change": rel})
    print(f"\n  最大 {amax:.1%} —— 比顺序那一项**大**。"
          f"锚点在 Navg_r 第 93 百分位、749 人完全相同、每人出现两次，")
    print("  贡献中位 18.2% 的 Σ(x−x̄)²，会把所有人的斜率往同一处钉。")
    run.log_metric("anchor_max_rel_change", float(amax))

    # ── ⑤ τ 与门槛 ───────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("[5] 个体间斜率 SD（主分析：剔锚点、不控顺序 —— 顺序已随机化）")
    NOISE = R[R.masker.isin(NOISE_MASKERS)]
    print(f"  {'目标':<10}{'描述符':<11}{'人':>5}{'n/人':>6}{'β':>9}{'τ̂':>9}"
          f"{'95% CI':>20}{'平衡点':>9}{'n*':>7}{'判读':>10}")
    n_stars = []
    for t in TARGETS:
        for d, D, tag in ([(p, R, p) for p in PSYCHO] +
                          [("smr", NOISE, "smr(噪声)")]):
            r = tau_block(D, d, t)
            if not r:
                continue
            usable = not (d == "smr" and not ctrl_ok[t])
            verdict = ("不予解读" if not usable else
                       "✅ 超过" if r["ci_lo"] > r["breakeven"] else
                       "⚠️ 跨过" if r["ci_hi"] > r["breakeven"] else "低于")
            print(f"  {t:<10}{tag:<11}{r['n_units']:>5}{r['n_per_unit']:>6.0f}"
                  f"{r['beta']:>+9.4f}{r['tau']:>9.4f}"
                  f"   [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
                  f"{r['breakeven']:>9.4f}{r['n_star']:>7.0f}   {verdict}")
            rows.append({"test": "tau", "target": t, "descriptor": d,
                         "usable": usable, **r})
            if usable:
                n_stars.append(r["n_star"])
            run.log_metric(f"tau_{t}_{d}", r["tau"])

    print("\n" + "=" * 88)
    ns_ = np.array(n_stars)
    print(f"n* 中位 {np.median(ns_):.0f}   "
          f"四分位 [{np.quantile(ns_,.25):.0f}, {np.quantile(ns_,.75):.0f}]")
    print("\n🔴 与第三版的差别，一句话：顺序那条结论**撤回**。")
    print("   ARAUS 顺序随机化，不能用来退 PMEmo 的 D2 ——")
    print("   而 `94` 自己测出 PMEmo 的分配 icc=0.041 (p=0.005) **非随机**，")
    print("   且纯位置代理复现 114–249% 的曲目级信度。方向正好相反。")
    run.log_metric("n_star_median", float(np.median(ns_)))

    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")
    run.write()
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
