r"""B3：个体差异究竟长在哪里 —— 刺激斜率上，还是状态调节上。

    .venv/Scripts/python scripts/95_task_channel_interaction.py

## 全文唯一一次正面估计个体差异的机会

`93` 用剖面似然在两个通道上估了「声学→反应」的**刺激斜率**的个体间 SD。

🔴 **2026-08-03 更正**：原文写「生理 τ̂ ≈ 0.106、主观 ≈ 0.074，CI 多数排除零」。
两处都要改 ——

    生理那个 0.106 **低于自己的零分布**（n=18 时零分布 0.106–0.123）。
                   逐格改正后 15 格里 11 格为零，τ 中位 = **0.000**。
    「CI 排除零」  那是 τ_eff = √(τ²+零²) 的区间，不是 τ 的。
                   真值为零时 n=18 的原始 CI 有 84% 的重复会「排除零」。

**但本脚本的结论方向不变，反而更强**：斜率侧更接近零，
而下面要说的截距侧 τ/σ* = 1.7–2.7 明确可用 —— 两者的对比因此更鲜明。
定稿数字见 `reports/tables/canonical_tau_table.md`。

但那只是一种个体差异。ds003690 提供了另一种，而且量级可能完全不同：

    主动任务   心率 −0.83 bpm（p=0.0002）   瞳孔 +0.113（p=0.0002）
    被动聆听   心率 +0.079（p=0.30）        瞳孔 −0.0000（p=0.9986）
    脑电       两种条件都显著

**同一个人、同一批纯音，只因「有没有任务」，自主神经通道就从有反应变成无反应。**
这是一种**状态调节**效应。若它的个体间方差远大于刺激斜率的，
则「按人校准」该校的不是「这个人对什么声音反应」，
而是「这个人在什么状态下才会反应」—— 那是完全不同的产品设计。

## 三个量，同一把尺子

对每名受试、每条通道：

    主动效应 = 主动任务里 (真实锚点 − 随机锚点) 的均值
    被动效应 = 被动段里同样的量
    交互     = 主动效应 − 被动效应        ← 状态调节的个体值

然后用与 `93` **逐字相同**的剖面似然（随机效应元分析形式）估交互的 τ 与 CI。

## 可比性：都换算成「相对盈亏平衡」

τ 的绝对值不能跨量纲比。统一除以各自的盈亏平衡
`σ* = σ_within/√(n_eff)`，得到「个体差异是估计噪声的几倍」——
这个比值无量纲，可直接横比。

**判据**：若交互的 τ/σ* 明显大于刺激斜率的，则个体差异主要在状态调节上。

## 一处必须先写明的不对称

被动段每人仅 30 次、主动段 240 次，故被动效应的估计噪声约为主动的 √8 ≈ 2.8 倍，
交互继承之。剖面似然用每人自己的 SE，这一点被正确计入 ——
但它意味着**交互的功效天然低于主动效应**，零结果要按此解读。
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
RAW = REPO_ROOT / "data" / "raw" / "ds003690"
OUT = REPO_ROOT / "reports" / "source_data" / "task_channel_interaction.csv"
SEED = 20260802
MODS = ["eeg", "hr", "pupil"]
ACTIVE = ("simpleRT", "gonogo")
PASSIVE = ("passive",)
MIN_PAIRS = 10
N_PERM = 5000
CHI2_95 = chi2.ppf(0.95, 1)


def pairs(g: pd.DataFrame, mod: str, tasks: tuple[str, ...]) -> np.ndarray:
    """真实锚点 − 随机锚点，按 (task, run, trial) 配对。与 `70` 同口径。"""
    g = g[g.task.isin(tasks)]
    k = ["task", "run", "trial"]
    a = g[g.arm == "real"].set_index(k)[mod]
    b = g[g.arm == "neg"].set_index(k)[mod]
    return (a - b).dropna().to_numpy(float)


def _ll(t2: float, b: np.ndarray, v: np.ndarray) -> float:
    w = 1.0 / (v + t2)
    mu = float((w * b).sum() / w.sum())
    ll = -0.5 * float((np.log(2 * np.pi / w) + w * (b - mu) ** 2).sum())
    return ll - 0.5 * np.log(w.sum())          # REML


def tau_ci(b: np.ndarray, se: np.ndarray) -> tuple[float, float, float, float]:
    """τ 的点估计与 95% 剖面似然 CI，外加加权均值。与 `93` 逐字相同。"""
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


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("95_task_channel_interaction", SEED)
    D = pd.read_parquet(FEAT / "ds003690_trials.parquet")
    P = pd.read_csv(RAW / "participants.tsv", sep="\t").set_index("participant_id")
    subs = sorted(D.subject.unique())
    print(f"受试 {len(subs)}   试次行 {len(D)}\n")

    rows, per_sub = [], {}
    for mod in MODS:
        rec = []
        for s in subs:
            g = D[D.subject == s]
            a, p_ = pairs(g, mod, ACTIVE), pairs(g, mod, PASSIVE)
            if len(a) < MIN_PAIRS or len(p_) < MIN_PAIRS:
                continue
            ea, ep = float(a.mean()), float(p_.mean())
            sa = float(a.std(ddof=1) / np.sqrt(len(a)))
            sp = float(p_.std(ddof=1) / np.sqrt(len(p_)))
            rec.append({"subject": s, "active": ea, "passive": ep,
                        "inter": ea - ep, "se_active": sa, "se_passive": sp,
                        "se_inter": float(np.hypot(sa, sp)),
                        "n_active": len(a), "n_passive": len(p_)})
        per_sub[mod] = pd.DataFrame(rec)

    # ── ① 组水平：三个效应各自是否非零 ───────────────────────────────
    print("=" * 92)
    print("[1/3] 组水平：主动效应 / 被动效应 / 交互")
    print(f"  {'通道':<8}{'人':>4}{'主动':>10}{'p':>8}{'被动':>10}{'p':>8}"
          f"{'交互':>10}{'p':>8}")
    for mod in MODS:
        R = per_sub[mod]
        line = [f"  {mod:<8}{len(R):>4}"]
        for col in ("active", "passive", "inter"):
            v = R[col].to_numpy(float)
            signs = rng.integers(0, 2, (N_PERM, len(v))) * 2 - 1
            null = np.abs(signs @ v / len(v))
            p = float((1 + (null >= abs(v.mean())).sum()) / (1 + N_PERM))
            line.append(f"{v.mean():>+10.4f}{p:>8.4f}")
            rows.append({"kind": "group", "mod": mod, "effect": col,
                         "mean": float(v.mean()), "p": p, "n": len(R)})
        print("".join(line), flush=True)

    # ── ② 个体间 SD：刺激效应 vs 状态调节，同一把尺子 ────────────────
    print("\n" + "=" * 92)
    print("[2/3] 个体间 SD（剖面似然，与 `93` 同一估计量）")
    print("      σ* = 估计噪声中位数；τ/σ* 无量纲，可跨通道横比")
    print(f"  {'通道':<8}{'效应':<10}{'τ̂':>9}{'95% CI':>20}{'σ*':>9}"
          f"{'τ/σ*':>8}{'判读':>16}")
    for mod in MODS:
        R = per_sub[mod]
        for col, se_col in (("active", "se_active"), ("inter", "se_inter")):
            b, se = R[col].to_numpy(float), R[se_col].to_numpy(float)
            ok = np.isfinite(b) & np.isfinite(se) & (se > 0)
            if ok.sum() < 20:
                continue
            tau, lo, hi, mu = tau_ci(b[ok], se[ok])
            sig = float(np.median(se[ok]))
            ratio = tau / sig if sig > 0 else np.nan
            verdict = ("✅ 超过估计噪声" if lo > sig else
                       ("⚠️ 与噪声相当" if hi > sig else "低于噪声"))
            # 🔴 术语：ds003690 只有四个纯音，**没有声学变异** ——
            # 这里的 "active" 测的是「这个人整体反应有多强」（截距 b_i），
            # **不是**「反应如何随声音属性变化」（斜率 c_ij，那是 PMEmo 测的）。
            # 两者是不同的量，混用会得出相反的产品结论。
            label = "反应强度(截距)" if col == "active" else "**状态调节**"
            print(f"  {mod:<8}{label:<10}{tau:>9.4f}   [{lo:.4f}, {hi:.4f}]"
                  f"{sig:>9.4f}{ratio:>8.2f}   {verdict}")
            rows.append({"kind": "tau", "mod": mod, "effect": col,
                         "tau": tau, "ci_lo": lo, "ci_hi": hi,
                         "sigma_star": sig, "ratio": ratio, "beta": mu})
            run.log_metric(f"tau_{mod}_{col}", tau)
            run.log_metric(f"ratio_{mod}_{col}", float(ratio))

    # ── ③ 年龄能解释多少状态调节的个体差异 ───────────────────────────
    print("\n" + "=" * 92)
    print("[3/3] 状态调节的个体差异，年龄能解释多少")
    from scipy.stats import spearmanr
    for mod in MODS:
        R = per_sub[mod].copy()
        R["age"] = R.subject.map(P["age"])
        R["grp"] = R.subject.map(P["group"])
        ok = R.dropna(subset=["age", "inter"])
        if len(ok) < 20:
            continue
        r = spearmanr(ok.age, ok.inter).statistic
        y = ok[ok.grp == "Young"].inter
        o = ok[ok.grp == "Older"].inter
        print(f"  {mod:<8} ρ(年龄, 交互) = {r:+.3f}   "
              f"Young {y.mean():+.4f} (n={len(y)})   "
              f"Older {o.mean():+.4f} (n={len(o)})")
        rows.append({"kind": "age", "mod": mod, "rho_age": float(r),
                     "young": float(y.mean()), "older": float(o.mean())})
        run.log_metric(f"rho_age_inter_{mod}", float(r))

    T = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT, index=False, encoding="utf-8")
    for mod, R in per_sub.items():
        R.assign(mod=mod).to_csv(
            OUT.with_name(f"task_channel_persub_{mod}.csv"),
            index=False, encoding="utf-8")

    # ── 判决 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    tt = T[T.kind == "tau"]
    if len(tt):
        st = tt[tt.effect == "active"].ratio.median()
        it = tt[tt.effect == "inter"].ratio.median()
        print(f"τ/σ* 中位：反应强度(截距) {st:.2f}   状态调节 {it:.2f}")
        if it > st * 1.3:
            print("→ 个体差异主要长在状态调节上。")
        elif st > it * 1.3:
            print("→ **个体差异主要长在「反应强度」上，状态调节相对一致。**")
            print()
            print("  🔴 不要读成「个体差异在刺激斜率上」—— ds003690 只有四个纯音，")
            print("     没有声学变异，测不了斜率。它测的是**截距**。")
            print("  与 `93`（PMEmo/DEAM 测斜率）合起来才是完整图景：")
            print("     截距 b_i    大、易测，τ/σ* = 1.7–2.7（每人 240 次）")
            print("     斜率 c_ij   小，τ ≈ 0.07–0.11，低于盈亏平衡（每人 18 次）")
            print("  产品含义：**「这个人有多敏感」与「哪种声音对他有用」是两件事，")
            print("     前者两三次观测就能校准，后者要 100+ 次。**")
        else:
            print("→ 两者相当，数据不支持把个体差异归到其中任一侧。")
        run.log_metric("ratio_median_stimulus", float(st))
        run.log_metric("ratio_median_state", float(it))
    print("\n⚠️ 被动段每人仅 30 次、主动段 240 次 → 被动效应的噪声约为主动的 2.8 倍，")
    print("   交互继承之。剖面似然已用各人自己的 SE 计入，但交互的功效天然更低。")
    run.write()
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
