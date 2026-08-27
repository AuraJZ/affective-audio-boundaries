r"""组间结果的功效审查 —— 补上本文最明显的一处双重标准。

    .venv/Scripts/python scripts/83_group_level_power_audit.py

## 为什么必须做

本文对**个体层面**的阴性做了严格的功效审查（`56` 的设计敏感度模拟，
Table 1，Fig. 6），却对**组间**的阴性只写了一句「它不受此限制，
因为在刺激层面汇总了 31 位听众」，没有任何定量论证。

这是双重标准，而且组间设计本身就很薄：**307 个刺激，每个中位数只有
2 位听众听过**。在这种设计下 ICC 的抽样分布极宽 ——
「主观 0.335 vs 脑电最好 0.092，差 3.6 倍」这个对比究竟能不能读，
在给出置信区间之前是没有答案的。

## 两件事

| | |
|---|---|
| **① 置信区间** | 对两个 ICC 都做 bootstrap（按刺激重抽），看区间是否重叠 |
| **② 设计敏感度** | 往真实的设计结构里注入**已知**的刺激级信度，看本设计能测出多大的 |

② 与 `56` 对个体层面做的是同一件事，只是换到刺激层面。
输出形如「若脑电的真实刺激级 ICC 达到 X，本设计有 80% 把握测出；
实测 0.092 未达显著，故真实值**若存在也不超过 X**」。

## 一个本脚本回答不了的替代解释

审稿意见还指出：主观评分的 ICC 高，可能部分反映**评分在「人—刺激」层面
本就有更强的共同结构**（所有人都同意某段音乐更响、更快），
而脑电不共享这种低层一致性。

**这个解释本脚本证伪不了**，因为它不是功效问题而是构念问题。
只能在正文中写明，不能靠统计消解。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")

FEAT = REPO_ROOT / "data" / "features"
OUT = REPO_ROOT / "reports" / "source_data" / "group_level_power.csv"
SEED = 20260731
N_BOOT = 2000
N_SIM = 300
TRUE_ICC = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
ALPHA = 0.05


def icc_oneway(v: np.ndarray, g: np.ndarray) -> float:
    """单向随机效应 ICC —— 与 `48` 同一实现，不平衡设计的正确估计量。"""
    df = pd.DataFrame({"v": v, "g": g}).dropna()
    if df.g.nunique() < 3 or len(df) < 6:
        return np.nan
    grand = df.v.mean()
    grp = df.groupby("g").v
    n_i = grp.size().to_numpy()
    m_i = grp.mean().to_numpy()
    k = len(n_i)
    msb = float((n_i * (m_i - grand) ** 2).sum() / (k - 1))
    ssw = float(((df.v - df.g.map(grp.mean())) ** 2).sum())
    dfw = len(df) - k
    if dfw <= 0:
        return np.nan
    msw = ssw / dfw
    n0 = (n_i.sum() - (n_i ** 2).sum() / n_i.sum()) / (k - 1)
    denom = msb + (n0 - 1) * msw
    return float((msb - msw) / denom) if denom > 0 else np.nan


def boot_icc(v: np.ndarray, g: np.ndarray, rng, n=N_BOOT) -> tuple:
    """按**刺激**重抽的 bootstrap —— 刺激才是被推广的单位，不是试次。"""
    uniq = np.unique(g)
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        vv, gg = [], []
        for j, s in enumerate(pick):
            m = g == s
            vv.append(v[m])
            gg.append(np.full(m.sum(), j))      # 重编号，避免重复刺激被合并
        x = icc_oneway(np.concatenate(vv), np.concatenate(gg))
        if np.isfinite(x):
            vals.append(x)
    if len(vals) < 50:
        return np.nan, np.nan, np.nan
    a = np.array(vals)
    return float(a.mean()), float(np.quantile(a, .025)), float(np.quantile(a, .975))


def simulate(sizes: np.ndarray, true_icc: float, rng) -> float:
    """按真实的「每刺激听众数」结构生成数据，注入已知 ICC。"""
    k = len(sizes)
    s_between = np.sqrt(max(true_icc, 0.0))
    s_within = np.sqrt(max(1.0 - true_icc, 1e-9))
    mu = rng.normal(0, s_between, k)
    v = np.concatenate([mu[i] + rng.normal(0, s_within, n)
                        for i, n in enumerate(sizes)])
    g = np.concatenate([np.full(n, i) for i, n in enumerate(sizes)])
    return icc_oneway(v, g)


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("83_group_level_power_audit", SEED)

    D = pd.read_parquet(FEAT / "ds002721_trials_raw.parquet")
    conn = FEAT / "ds002721_connectivity.parquet"
    if conn.exists():
        C = pd.read_parquet(conn)
        on = [c for c in ("subject", "run", "trial") if c in C.columns]
        D = D.merge(C, on=on, how="left", suffixes=("", "_c"))
    stim = "stim"
    # 主观 8 问；`alt_*` 是同一批评分的另一版本，不重复计入
    SUBJ = ["pleasant", "energetic", "tender", "afraid", "angry", "tense",
            "sad", "happy"]
    META = {"subject", "run", "trial", "stim"} | set(SUBJ) | \
           {f"alt_{q}" for q in SUBJ}
    # 合并加了 suffixes=("", "_c")，故 META 里每个名字都可能多出一个 `_c` 版本。
    # 🔴 首版没算上这一条，于是 `stim_c`（**刺激编号本身**）被当成一个脑电测量，
    # 按刺激算 ICC 得 0.892 —— 那是构造出来的，不是信号。
    # 这是本项目第五次栽在「硬编码一张排除表」这个写法上。
    META |= {f"{c}_c" for c in META}
    EEG = [c for c in D.columns
           if c not in META and pd.api.types.is_numeric_dtype(D[c])]

    # 排除表挡不住没预料到的列名，所以再加一道**结构性**闸门：
    # 任何在刺激内方差为零的列，都是刺激的确定性函数（编号、时长、声学量……），
    # 拿它按刺激算 ICC 必然接近 1。这道闸不依赖列名。
    leak = [c for c in EEG
            if float(D.groupby(stim)[c].std().fillna(0).max()) < 1e-9]
    if leak:
        print(f"🔴 刺激内方差为零，判为刺激的确定性函数，剔除 {len(leak)} 列："
              f"{leak[:8]}{' …' if len(leak) > 8 else ''}")
        EEG = [c for c in EEG if c not in leak]
    assert len(EEG) >= 50, f"脑电列数异常 {len(EEG)}"
    print(f"试次表 {D.shape}   主观 {len(SUBJ)} 项   脑电 {len(EEG)} 项")

    # ICC 前先做**被试内 z 分** —— 与 `48` 同口径，去掉个体基线与量程
    for c in SUBJ + EEG:
        g = D.groupby("subject")[c]
        D[f"z_{c}"] = (D[c] - g.transform("mean")) / (g.transform("std") + 1e-9)

    sizes = D.groupby(stim).size().to_numpy()
    print(f"刺激 {len(sizes)} 个   每刺激听众数 中位 {np.median(sizes):.0f}   "
          f"范围 {sizes.min()}–{sizes.max()}   ≥8 的有 {(sizes>=8).sum()} 个\n")

    # ── ① 置信区间 ────────────────────────────────────────────────────
    print("=" * 80)
    print("[1/2] 两个 ICC 的 bootstrap 置信区间（按刺激重抽）")
    rows = []
    for c in SUBJ + EEG:
        zc = f"z_{c}"
        if D[zc].notna().sum() < 100:
            continue
        obs = icc_oneway(D[zc].to_numpy(float), D[stim].to_numpy())
        if np.isfinite(obs):
            rows.append({"measure": c, "icc": obs,
                         "kind": "subjective" if c in SUBJ else "eeg"})
    R = pd.DataFrame(rows).sort_values("icc", ascending=False)
    print(f"  {len(R)} 个测量  ——  主观最好 "
          f"{R[R.kind=='subjective'].iloc[0].measure} {R[R.kind=='subjective'].iloc[0].icc:+.3f}"
          f"   脑电最好 {R[R.kind=='eeg'].iloc[0].measure} "
          f"{R[R.kind=='eeg'].iloc[0].icc:+.3f}")
    # 只对「最好的主观量」与「最好的脑电量」做 bootstrap —— 那正是论文里的对比。
    # ⚠️ 「最好的 156 个之一」本身带择优偏差，脑电的 CI 因此偏乐观；
    #    这对本审查是保守方向（更难说它显著低于主观），故可接受，但须写明。
    subj = list(R[R.kind == "subjective"].itertuples())
    eeg = list(R[R.kind == "eeg"].itertuples())
    pick = ([subj[0]] if subj else []) + ([eeg[0]] if eeg else [])
    ci = {}
    for r in pick:
        v = D[f"z_{r.measure}"].to_numpy(float)
        g = D[stim].to_numpy()
        m, lo, hi = boot_icc(v, g, rng)
        ci[r.measure] = (r.icc, lo, hi)
        print(f"  {r.measure:<22} ICC = {r.icc:+.3f}   "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]")
        run.log_metric(f"icc_{r.measure}", float(r.icc))
        run.log_metric(f"icc_ci_lo_{r.measure}", lo)
        run.log_metric(f"icc_ci_hi_{r.measure}", hi)
    if len(ci) == 2:
        (a, alo, ahi), (b, blo, bhi) = ci.values()
        overlap = not (alo > bhi or blo > ahi)
        print(f"\n  两个区间{'**重叠** —— 3.6 倍之说不成立' if overlap else '不重叠 —— 差异可读'}")
        run.log_metric("ci_overlap", bool(overlap))

    # ── ② 设计敏感度 ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("[2/2] 设计敏感度：本设计能测出多大的刺激级信度")
    print("      （按真实的「每刺激听众数」结构生成，注入已知 ICC）\n")
    null = np.array([simulate(sizes, 0.0, rng) for _ in range(N_SIM * 3)])
    null = null[np.isfinite(null)]
    thr = float(np.quantile(null, 1 - ALPHA))
    print(f"  真实 ICC = 0 时的分布：均值 {null.mean():+.4f}   "
          f"95 分位 = {thr:+.4f}  ← 显著阈值\n")
    srows = []
    for t in TRUE_ICC:
        if t == 0:
            continue
        vals = np.array([simulate(sizes, t, rng) for _ in range(N_SIM)])
        vals = vals[np.isfinite(vals)]
        pw = float(np.mean(vals > thr))
        srows.append({"true_icc": t, "recovered": float(vals.mean()),
                      "power": pw, "threshold": thr})
        print(f"  真实 ICC = {t:.2f}   估计 {vals.mean():+.3f}   把握 {pw:.0%}")
    S = pd.DataFrame(srows)
    ok = S[S.power >= 0.80]
    lim = float(ok.true_icc.min()) if len(ok) else np.nan

    pd.concat([R.assign(kind="observed"),
               S.assign(kind="sensitivity")]).to_csv(
        OUT, index=False, encoding="utf-8")

    # ── ③ 择优的零分布：156 个里的**最大值**，不是单个 ────────────────
    #
    # 🔴 阈值 thr 是**单个**测量的零分布 95 分位。拿 156 个里最好的那个去比它，
    # 正是多重比较错误。第二版我改判决时犯了这个错，且与 `RESULTS_SUMMARY`
    # 里早就有的「FDR 校正后无一通过」直接冲突。
    #
    # 正确的零分布：打乱刺激标签（拆掉刺激级结构，但**保留 156 个测量之间的
    # 相关结构** —— 整行一起打乱，不逐列打乱），取该次的最大 ICC，重复。
    print("\n" + "=" * 80)
    print("[3/3] 择优零分布：156 个测量里**最大**ICC 的零分布")
    print("      （整行打乱刺激标签 —— 拆掉刺激级结构，保留测量间相关）")
    Z = D[[f"z_{c}" for c in EEG]].to_numpy(float)
    g_real = D[stim].to_numpy()
    maxnull = []
    for _ in range(400):
        gp = rng.permutation(g_real)
        vals = [icc_oneway(Z[:, j], gp) for j in range(Z.shape[1])]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            maxnull.append(max(vals))
    maxnull = np.array(maxnull)
    thr_max = float(np.quantile(maxnull, 0.95))
    best_eeg = float(eeg[0].icc) if eeg else np.nan
    # 报 p 值而非「过没过」——「勉强过 FWER、勉强不过 FDR」不是矛盾，
    # 是同一个边界的两面：BH 在秩 1 处的阈值就是 Bonferroni，
    # 而最大值置换就是 FWER，两者在秩 1 上几乎等价。
    p_fwer = float((maxnull >= best_eeg).mean())
    print(f"  单测量阈值 {thr:+.3f}   **择优后**阈值 {thr_max:+.3f}"
          f"   （{Z.shape[1]} 个测量，{len(maxnull)} 次置换）")
    print(f"  脑电最好 {best_eeg:+.3f}   **择优校正后 p = {p_fwer:.3f}**")
    run.log_metric("thr_single", float(thr))
    run.log_metric("thr_max_selected", thr_max)
    run.log_metric("best_eeg_icc", best_eeg)
    run.log_metric("p_fwer_best_eeg", p_fwer)

    print("\n" + "=" * 80)
    if np.isfinite(lim):
        print(f"本设计对刺激级 ICC ≥ {lim:.2f} 有 80% 把握（单测量口径）。")
        if p_fwer > 0.05:
            print(f"→ 择优校正后 p = {p_fwer:.3f} 不显著。"
                  f"脑电刺激级信度**若存在也不超过约 {lim:.2f}**。")
        elif p_fwer > 0.01:
            print(f"→ 择优校正后 p = {p_fwer:.3f} —— **就在边界上**。")
            print("   这解释了与 `RESULTS_SUMMARY`「FDR 无一通过」的表面冲突：")
            print("   BH 在秩 1 处的阈值即 Bonferroni，与最大值置换（FWER）几乎等价，")
            print("   两者一个勉强过、一个勉强不过，说的是同一件事。")
            print("   ⚠️ 且这是 156 选 1 的最大值，点估计带**胜者诅咒**（系统性偏高），")
            print(f"   同时 {best_eeg:+.3f} 恰在本设计的检出下限 {lim:.2f} 附近。")
            print("   **不要写成阳性，也不要写成零。** 正文该写的是：")
            print(f"   脑电刺激级信度若存在也不超过约 {lim:.2f}；"
                  f"主观 {subj[0].icc:+.3f}（CI 不含它）。")
            print("   差的是天花板，而「生理是否恰为零」本文回答不了、也不必回答。")
        else:
            print(f"→ 择优校正后 p = {p_fwer:.3f}，稳健显著。")
            print("   ⚠️ 与 `RESULTS_SUMMARY` 的 FDR 结论冲突，须查清哪个对。")
        run.log_metric("icc_detectable_at_80pct", lim)
        run.log_metric("eeg_above_selected_threshold", bool(best_eeg > thr_max))
    else:
        print("在所测范围内均未达 80% 把握 —— 该设计对刺激级信度不可用。")
    print("\n⚠️ 本脚本只处理功效。审稿意见指出的另一条 ——")
    print("   「主观评分的 ICC 高可能是因为它在人—刺激层面本就有更强的共同结构」")
    print("   —— 是构念问题不是功效问题，统计消解不了，须在正文写明。")
    run.write()
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
