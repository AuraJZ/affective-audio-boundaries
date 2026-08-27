r"""⚠️ 已被 99g（权威表）取代，**不要引用其数字**。

本脚本用 93 的原始 τ（未扣零分布）比较三个语料，且 ARAUS 侧未剔 fold_r=−1 锚点。
它的历史价值是发现了 τ̂ 的小 n 偏倚（那条线索引出 99d），结论本身已过时。
定稿见 reports/tables/canonical_tau_table.md。

B2 续：τ 跨语料为何不同 —— 一个我先写死了结论、又被数据打脸的检验。

    .venv/Scripts/python scripts/99c_tau_across_corpora.py

## 起因

`99` 给出 ARAUS 的门槛 n* = 54，低于 `92`(~100) / `93`(97–184) / `98`(98)。
我在 `99` 里预先写了一个解释：ARAUS 是**设计出来的**刺激集（SMR 五档摊开响度），
自然音乐的 var x 是碰巧给多少算多少，所以 ARAUS 的 var x 更大、门槛更低。

`99` 自己算了这个比值：**1.06× 和 1.01×**。假说被它自己的检验否掉。
（而脚本照样把那段解释打印了出来 —— 结论没有随检验结果走。这是规矩 #7
本来要防的错，我自己犯了一次。）

拆开 n* = σ_resid²/(τ²·var x) + 1，三项里：

    σ_resid   0.957 / 0.958 / 0.981   —— 几乎相等（x、y 都标准化了，本该如此）
    var x     0.928 / 0.877 / 0.921   —— 几乎相等
    τ̂         0.1328 / 0.1133 / 0.0688 —— **差异全在这里**

## 所以真问题是：τ 为什么不同

两类可能，答案完全不同：

**甲 · 真实的** —— 声景宜人度的个体差异本来就比音乐唤醒度大。
这在文献里说得通：噪声敏感性是有名的稳定人格特质，
而音乐唤醒度判断高度一致（MIR 能work正是因为这个）。

**乙 · 估计伪影** —— τ̂ 随每人观测数 n 变。ARAUS 每人 44 次，DEAM 每人少得多。
若如此，则**本文所有跨语料的 τ 比较都不可靠**，包括 `93` 的双通道并排 ——
那是主线上的一块砖。

## 判据：配平 n 重估

把 ARAUS 每人的观测数**下采样**到 DEAM 的水平，重估 τ。

    τ̂ 不变        → 甲。跨语料 τ 比较成立，ARAUS 的人确实更不一样
    τ̂ 掉到 DEAM 值 → 乙。τ̂ 是 n 的函数，`93` 的并排要重做

另加两条：

    单位数配平    抽 749→DEAM 的人数。REML 的 τ̂ 点估计不该随之变，
                  只该 CI 变宽。若点估计也变，说明估计器本身有问题。
    结果变量配平  ARAUS `pleasant` 对的是 DEAM `Valence`（同为效价），
                  不是 `Arousal`。`99` 那张表拿错了对手。
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
DEAM_CSV = REPO_ROOT / "data" / "raw" / "DEAM" / "annotations" / \
    "annotations per each rater" / "song_level" / \
    "static_annotations_songs_1_2000.csv"
OUT = REPO_ROOT / "reports" / "source_data" / "tau_across_corpora.csv"
SEED = 20260803
CHI2_95 = chi2.ppf(0.95, 1)
MIN_OBS = 14
N_REP = 40


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
    grid = np.concatenate([[0.0], np.geomspace(1e-10, hi_b, 3000)])
    lls = np.array([_ll(t, b, v) for t in grid])
    k = int(np.argmax(lls))
    t2, lm = float(grid[k]), float(lls[k])

    def g(x: float) -> float:
        return 2.0 * (lm - _ll(x, b, v)) - CHI2_95

    lo = brentq(g, 0.0, t2) if (k > 0 and g(0.0) > 0) else 0.0
    hi = brentq(g, t2, hi_b) if g(hi_b) > 0 else hi_b
    return float(np.sqrt(t2)), float(np.sqrt(lo)), float(np.sqrt(hi))


def units(D: pd.DataFrame, unit: str, x: str, y: str, min_obs: int = MIN_OBS
          ) -> list[tuple[np.ndarray, np.ndarray]]:
    """被试内 z 分（y）+ 全局 z 分（x），逐单位切出 (x, y)。与 `93` 同口径。"""
    D = D.dropna(subset=[x, y]).copy()
    g = D.groupby(unit)[y]
    D[y] = (D[y] - g.transform("mean")) / (g.transform("std") + 1e-9)
    D[x] = (D[x] - D[x].mean()) / (D[x].std() + 1e-9)
    return [(h[x].to_numpy(float), h[y].to_numpy(float))
            for _, h in D.groupby(unit) if len(h) >= min_obs]


def estimate(u: list[tuple[np.ndarray, np.ndarray]]) -> dict | None:
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
    tau, lo, hi = tau_ci(np.array(bs), np.array(ses))
    n_med = float(np.median(ns))
    be = float(np.sqrt(float(np.median(rsd)) ** 2 /
                       max((n_med - 1) * float(np.median(vx)), 1e-9)))
    return {"n_units": len(bs), "n_per_unit": n_med, "tau": tau,
            "ci_lo": lo, "ci_hi": hi, "resid_sd": float(np.median(rsd)),
            "var_x": float(np.median(vx)), "breakeven": be,
            "n_star": n_med * (be / max(tau, 1e-9)) ** 2}


def load_all() -> dict[str, list]:
    out: dict[str, list] = {}
    A = pd.read_csv(AR / "responses.csv")
    A = A[A.is_attention == 0]
    out["ARAUS pleasant ~ 响度"] = units(A, "participant", "Navg_r", "pleasant")
    out["ARAUS eventful ~ 响度"] = units(A, "participant", "Navg_r", "eventful")

    D = pd.read_csv(DEAM_CSV, skipinitialspace=True)
    D.columns = [c.strip() for c in D.columns]
    F = pd.read_parquet(FEAT / "layer_a_reg_deam_ln.parquet")
    F["SongId"] = F.clip_id.astype(str).str.removeprefix("deam_").astype(int)
    M = D.merge(F[["SongId", "rms_mean"]], on="SongId")
    out["DEAM Valence ~ 响度"] = units(M, "workerID", "rms_mean", "Valence")
    out["DEAM Arousal ~ 响度"] = units(M, "workerID", "rms_mean", "Arousal")

    T = pd.read_parquet(FEAT / "pmemo_eda_song_level.parquet")
    P = pd.read_parquet(FEAT / "layer_a_reg_pmemo_ln.parquet")
    P["musicId"] = P.clip_id.astype(str).str.removeprefix("pmemo_").astype(int)
    PM = T.merge(P[["musicId", "rms_mean"]], on="musicId")
    out["PMEmo 皮电相位 ~ 响度"] = units(PM, "listener", "rms_mean", "phasic_mean")
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("99c_tau_across_corpora", SEED)
    data = load_all()
    rows = []

    print("=" * 96)
    print("[1] 原样对表 —— 三项里只有 τ 在动")
    print(f"  {'格':<24}{'单位':>6}{'n/单位':>8}{'σ_resid':>9}{'var x':>8}"
          f"{'τ̂':>9}{'95% CI':>20}{'n*':>7}")
    base = {}
    for k, u in data.items():
        r = estimate(u)
        if not r:
            continue
        base[k] = r
        print(f"  {k:<24}{r['n_units']:>6}{r['n_per_unit']:>8.0f}"
              f"{r['resid_sd']:>9.3f}{r['var_x']:>8.3f}{r['tau']:>9.4f}"
              f"   [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]{r['n_star']:>7.0f}")
        rows.append({"test": "baseline", "cell": k, **r})

    print("\n  注：`99` 拿 ARAUS pleasant 对 DEAM **Arousal** 比，对手拿错了。")
    print("      pleasant 是效价轴，该对的是 DEAM **Valence**。")

    # ── ② 配平每单位观测数 ───────────────────────────────────────────
    print("\n" + "=" * 96)
    print("[2] 配平 n/单位 —— 把 ARAUS 下采样到 DEAM 的水平，τ̂ 变不变")
    print("    τ̂ 不变 → 差异是真的；τ̂ 掉下去 → τ̂ 是 n 的函数，"
          "`93` 的跨通道并排要重做\n")
    targets = sorted({int(base[k]["n_per_unit"]) for k in base
                      if k.startswith("DEAM") or k.startswith("PMEmo")})
    print(f"  {'格':<24}{'原 n':>6}{'原 τ̂':>9}", end="")
    for t in targets:
        print(f"{'n=' + str(t):>12}", end="")
    print()
    for k in [x for x in base if x.startswith("ARAUS")]:
        u = data[k]
        print(f"  {k:<24}{base[k]['n_per_unit']:>6.0f}{base[k]['tau']:>9.4f}",
              end="")
        for t in targets:
            taus = []
            for _ in range(N_REP):
                sub = []
                for x, y in u:
                    if len(x) < t:
                        continue
                    idx = rng.choice(len(x), t, replace=False)
                    sub.append((x[idx], y[idx]))
                r = estimate(sub) if len(sub) >= 30 else None
                if r:
                    taus.append(r["tau"])
            if taus:
                m = float(np.median(taus))
                print(f"{m:>12.4f}", end="")
                rows.append({"test": "match_n", "cell": k, "n_target": t,
                             "tau": m, "tau_full": base[k]["tau"]})
            else:
                print(f"{'—':>12}", end="")
        print()

    # ── ③ 配平单位数（阴性对照：点估计不该动） ───────────────────────
    print("\n" + "=" * 96)
    print("[3] 阴性对照：配平**单位数**。REML 的 τ̂ 点估计不该随单位数变，")
    print("    只该 CI 变宽。若点估计也动，是估计器本身的问题。\n")
    n_small = min(base[k]["n_units"] for k in base)
    print(f"  {'格':<24}{'全量 τ̂':>10}{'抽 ' + str(n_small) + ' 单位后 τ̂':>18}"
          f"{'相对变化':>10}")
    for k, u in data.items():
        if k not in base or len(u) <= n_small:
            continue
        taus = []
        for _ in range(N_REP):
            idx = rng.choice(len(u), n_small, replace=False)
            r = estimate([u[i] for i in idx])
            if r:
                taus.append(r["tau"])
        if taus:
            m = float(np.median(taus))
            rel = (m - base[k]["tau"]) / base[k]["tau"]
            print(f"  {k:<24}{base[k]['tau']:>10.4f}{m:>18.4f}{rel:>+9.1%}")
            rows.append({"test": "match_units", "cell": k, "tau": m,
                         "tau_full": base[k]["tau"], "rel_change": rel})

    # ── ④ 判决 —— 由数字决定，不预写 ─────────────────────────────────
    print("\n" + "=" * 96)
    print("[4] 判决")
    ar_k = "ARAUS pleasant ~ 响度"
    de_k = "DEAM Valence ~ 响度"
    if ar_k in base and de_k in base:
        n_t = int(base[de_k]["n_per_unit"])
        m = [r for r in rows if r["test"] == "match_n"
             and r["cell"] == ar_k and r["n_target"] == n_t]
        if m:
            tau_matched = m[0]["tau"]
            gap_full = base[ar_k]["tau"] / base[de_k]["tau"]
            gap_matched = tau_matched / base[de_k]["tau"]
            closed = (gap_full - gap_matched) / max(gap_full - 1, 1e-9)
            print(f"  原始 τ 比（ARAUS/DEAM 效价）      {gap_full:.2f}×")
            print(f"  配平 n={n_t} 后                    {gap_matched:.2f}×")
            print(f"  → 配平消掉了 {closed:.0%} 的差距")
            if closed > 0.7:
                print("\n  **乙**：τ̂ 主要是 n 的函数。跨语料 τ 比较不成立，")
                print("  `93` 的双通道并排（PMEmo n=18 vs DEAM n=?）须重做为配平比较。")
                print("  这是主线上的一块砖，必须先修再往下走。")
            elif closed < 0.3:
                print("\n  **甲**：配平后差距仍在。ARAUS 的个体差异是**真的**更大。")
                print("  与文献相符：噪声敏感性是稳定的人格特质，")
                print("  而音乐效价判断高度一致（MIR 可做正因于此）。")
                print("  ⇒ 门槛不是一个常数，而是**随域变化**的 —— 价目表要按域分行。")
            else:
                print("\n  **两者兼有**：n 解释一部分，域差异解释一部分。"
                      "两条都要在正文里说。")
            run.log_metric("gap_full", gap_full)
            run.log_metric("gap_matched", gap_matched)
            run.log_metric("frac_closed_by_n", closed)

    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")
    run.write()
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
