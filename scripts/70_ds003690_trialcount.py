r"""ds003690：逐个体阳性对照，以及「每人多少次才够」的**实测**曲线。

    .venv/Scripts/python scripts/70_ds003690_trialcount.py

## 这个脚本回答什么

`step18` 把 ds002721 的个体层面结论定为「在每人约 30 试次下**不可检验**」，
依据是 `56` 的功效模拟：r=0.40 需 271 次/人。那是**模拟**数 ——
用 Ledoit-Wolf 收缩协方差生成的合成数据。

本脚本用真实生理数据把同一条曲线**测出来**：取一个确定存在的效应
（听觉诱发反应），逐个体地把试次数从 10 抽到 270，看检出率如何随之上升。

这把「ds002721 阴性」从「我们没做出来」变成「**这种设计做不出来**」，
而后者是可以证伪的陈述。

## 统计：配对 + 符号翻转

`69` 每个 run 生成等量的真实锚点与随机锚点。按序号配对求差 `d_i`：

    d_i = 真实试次_i 的响应窗测量 − 随机锚点_i 的响应窗测量

零假设「无事件锁时响应」下，两者可交换，故 `d_i` 关于 0 对称，
**符号翻转是精确检验**（非渐近、无分布假设）。

配对还顺带解决了 `69` 发现的心率漂移偏倚：基线窗与响应窗的时间错位
对两臂完全相同，作差即抵消。

## 论证结构：为什么个体失败只能归因于功效

本项目一贯的问题是「阴性到底意味着什么」。这里的设计让它没有歧义：

1. **组水平**先证明效应确实存在（听觉 N1 是心理生理学里最稳的效应之一）；
2. 效应既已确立，**个体层面的检不出就只能是功效不足**，不可能是「没有效应」。

这正是 ds002721 拿不到的论证 —— 那里效应本身就未确立，
所以阴性无法在「不存在」与「测不到」之间区分。

## 三条曲线，缺一不可

| 曲线 | 作用 |
|---|---|
| **检出率 vs 试次数** | 主结果 |
| **假阳性率 vs 试次数**（配对差随机取符号） | 若不测，上升的检出率可能只是上升的假阳性率 |
| **解析功效曲线** | 实测若与解析吻合，说明 `56` 那套模拟外推是可信的 |

## 按任务分层，以及我在单受试上读错的一次 🔴

分层依据是**设计事实**：主动任务的心率响应窗（1–5 s）内含 go 音（约 +1.8 s）
与按键（约 +2.4 s），被动段两者都没有；且两段每人试次数相差 8 倍
（240 vs 30）。这两条都与结果无关，是先验的。

    试次数曲线   用主动任务（每人 240 次，量级足够）
    纯音条件     用被动段（每人仅 30 次），只在组水平报告

**我据单受试波形写过一个错误断言，此处更正。** 当时看到

    passive −7.5 bpm @ +3.75 s ／ gonogo +3.3 @ +5.25 s ／ simpleRT +2.8 @ +2.00 s

便推断「按键的加速把定向减速反了号」。43 人的组水平否掉了它：
主动段是 **−0.74 bpm 的减速**（p = 0.0002），方向与被动段一致。

错因是取值方式：我取的是整段 |差值| 的最大点，在单受试的噪声里
那就是个噪声尖峰，不是响应。**峰值定位在单受试上不可读** ——
诊断波形只能用来发现「窗口是不是设在了响应之外」（`69` 的 P2/CNV 那次
就是这个用法），不能用来推断响应的方向或幅度。

## 冒烟测试给出的量级（sub-AB10，270 试次）

    平均波形 N1 = −3.28 µV，单试次配对差 SD ≈ 28 µV → d = 0.117
    实测 d = −0.119 ✓ 内部一致

即：**一个在平均波形上一眼可见的教科书效应，在单个人身上要约 270 次
才勉强够到 p = 0.05。**

## 局限，先写明

- 纯音、清醒、任务情境。回答的是**方法学**问题「个体内事件锁时生理推断
  需要多少次观测」，**回答不了**「哪种音乐更助眠」。
- 测量刻意保持简单（单 ROI 固定窗均值、仅 ±150 µV 剔除，无 ICA、无空间滤波）。
  专门优化的 ERP 流程会更灵敏，因此本曲线是**下界**。
  但它也代表非专业流程的实际水平 —— 而那才是消费级产品的现实处境。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, nct
from scipy.stats import t as t_dist

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")

FEAT = REPO_ROOT / "data" / "features"
TRIALS = FEAT / "ds003690_trials.parquet"
WAVES = FEAT / "ds003690_waveforms.parquet"
# 与 `69` 的响应窗一致；此处只用于诊断打印，不参与判定
RESP_WIN = {"eeg": (0.080, 0.130), "hr": (1.0, 5.0), "pupil": (0.5, 2.5)}
OUT_SUBJ = FEAT / "ds003690_per_subject.parquet"
OUT_CURVE = FEAT / "ds003690_trialcount_curve.parquet"

SEED = 20260729
MODS = ["eeg", "hr", "pupil", "eeg_p2n1"]
# 主动任务（含按键）vs 被动聆听（纯音，无运动反应）。
# 主动段每人 240 次，支撑试次数曲线；被动段每人仅 30 次，只做组水平。
ACTIVE = ("simpleRT", "gonogo")
PASSIVE = ("passive",)
N_PERM = 5000                      # 逐个体检验
N_PERM_CURVE = 2000                # 曲线上每个格子
N_REPEAT = 40                      # 每个 (受试, 通路, n) 重抽次数
N_GRID = [10, 15, 20, 30, 50, 75, 100, 150, 200, 250]
ALPHA = 0.05
MIN_PAIRS = 20


def t_power(d: float, n: int, alpha: float = ALPHA) -> float:
    """双侧单样本 t 检验的解析功效。

    两条尾都要算 —— 只取上尾会在 d 接近 0 时少掉 α/2，
    而曲线左端恰恰就在那个区间，那里正是「30 次够不够」的判断处。
    """
    crit, df, nc = t_dist.ppf(1 - alpha / 2, n - 1), n - 1, abs(d) * np.sqrt(n)
    # scipy 的 nct 在非中心参数很大时返回 nan。那一段功效已经饱和，
    # 直接给 1.0 —— 但要设阈值明写，不能让 nan 悄悄传下去变成缺失格。
    if not np.isfinite(nc) or nc > 30:
        return 1.0
    v = float(nct.sf(crit, df, nc) + nct.cdf(-crit, df, nc))
    return v if np.isfinite(v) else 1.0


def signflip_p(d: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """符号翻转 p 值。d 为 (reps, n)，signs 为 (P, n)。返回 (reps,) 双侧 p。"""
    obs = np.abs(d.mean(axis=1))                       # (reps,)
    null = np.abs(signs @ d.T) / d.shape[1]            # (P, reps)
    return (1 + (null >= obs).sum(axis=0)) / (1 + signs.shape[0])


def make_pairs(g: pd.DataFrame, mod: str,
               tasks: tuple[str, ...] | None = None) -> np.ndarray:
    """把一名受试的真实臂与随机锚点臂按 (task, run, trial) 配对求差。

    `tasks` 限定任务。**心率必须分层**：主动任务的响应窗（1–5 s）内含按键，
    运动引起的加速与定向反应的减速方向相反，混在一起会互相抵消。
    分层依据是「响应窗内有无运动反应」这个设计事实，不是看结果挑的。
    """
    if tasks is not None:
        g = g[g.task.isin(tasks)]
    key = ["task", "run", "trial"]
    a = g[g.arm == "real"].set_index(key)[mod]
    b = g[g.arm == "neg"].set_index(key)[mod]
    d = (a - b).dropna()
    return d.to_numpy(float)


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("70_ds003690_trialcount", SEED)
    D = pd.read_parquet(TRIALS)
    subs = sorted(D.subject.unique())
    print(f"受试 {len(subs)}   试次行 {len(D)}\n")

    # ══ 0. 组水平：先确立效应存在 ═════════════════════════════════════
    print("=" * 78)
    print("[0/4] 组水平阳性对照：效应是否存在（存在，则个体失败只能归因于功效）")
    grp = {}
    for mod in MODS:
        per = np.array([np.nanmean(make_pairs(D[D.subject == s], mod, ACTIVE))
                        for s in subs])
        per = per[np.isfinite(per)]
        if len(per) < 5:
            continue
        signs = rng.integers(0, 2, (N_PERM, len(per))) * 2 - 1
        p = signflip_p(per[None, :], signs)[0]
        grp[mod] = (per.mean(), p, len(per))
        print(f"  {mod:<10} 跨 {len(per)} 人均值 {per.mean():+8.4f}   "
              f"p = {p:.4f}   {'✅ 效应确立' if p < ALPHA else '❌ 未确立'}")
        run.log_metric(f"group_p_{mod}", float(p))
        run.log_metric(f"group_mean_{mod}", float(per.mean()))

    # ── 0b. 按任务分层：被动聆听段无按键，是心率的干净条件 ──────────────
    #
    # 主动任务里 go 音（约 +1.8 s）与按键（约 +2.4 s）都落在心率响应窗
    # （1–5 s）内，而按键引起的加速与定向反应的减速方向相反，可能相互抵消。
    # 被动段无运动反应、ISI 约 7.9 s，因此是判断「纯音是否引起心脏反应」
    # 的干净条件。分层是先验的（按任务，不按结果），不是事后挑窗。
    print("\n  按任务分层（被动段 = 无按键的干净条件）：")
    for task in sorted(D.task.unique()):
        for mod in MODS:
            per = np.array([np.nanmean(make_pairs(
                D[(D.subject == s) & (D.task == task)], mod)) for s in subs])
            per = per[np.isfinite(per)]
            if len(per) < 5:
                continue
            signs = rng.integers(0, 2, (N_PERM, len(per))) * 2 - 1
            p = signflip_p(per[None, :], signs)[0]
            print(f"    {task:<10} {mod:<10} {len(per):>2} 人  "
                  f"均值 {per.mean():+8.4f}  p = {p:.4f}"
                  f"{'  ✅' if p < ALPHA else ''}")
            run.log_metric(f"task_{task}_{mod}_p", float(p))

    # ── 0c. 诊断：真实臂 − 随机锚点臂的差值波形 ────────────────────────
    #
    # 判据只看一个窗；波形看整段。ds002721 的 C3 判据窗设错、
    # `69` 的 P2 被 CNV 吃掉，两次都是波形先说出来的。
    # **这里只看，不据此改判据** —— 若波形显示反应在别处，
    # 那是需要独立确认的观察，不是本文的结果。
    if WAVES.exists():
        W = pd.read_parquet(WAVES)
        print("\n  差值波形（真实 − 随机锚点），峰值与其时刻：")
        for task in sorted(W.task.unique()):
            for mod in sorted(W["mod"].unique()):
                w = W[(W.task == task) & (W["mod"] == mod)]
                if not len(w):
                    continue
                g = w.pivot_table(index="t", columns="arm", values="v",
                                  aggfunc="mean")
                if not {"real", "neg"} <= set(g.columns):
                    continue
                d = (g["real"] - g["neg"]).dropna()
                k = d.abs().idxmax()
                print(f"    {task:<10} {mod:<6} 峰 {d[k]:+8.3f} @ {k:+.3f}s"
                      f"   （预设窗 {RESP_WIN[mod][0]:+.3f}–{RESP_WIN[mod][1]:+.3f}s "
                      f"内均值 {d[(d.index >= RESP_WIN[mod][0]) & (d.index < RESP_WIN[mod][1])].mean():+.3f}）")

    # ── 0d. 被动段的零结果：是「没反应」还是「30 次太少」🔴 ─────────────
    #
    # 被动段每人仅 30 次，心率与瞳孔都不显著。这两种解释后果完全不同：
    #   「没反应」→ 纯音在无任务相关性时不驱动自主神经，产品必须知道
    #   「太少」  → 只是功效问题，说明不了任何事
    # 本项目反复强调的正是这个区分（`step16`/`step18` 都栽在这里）。
    #
    # 判据：把主动段测到的效应量代入被动段的实际结构
    # （每人 30 次 × 75 人的组水平检验），算若效应等大时本该有多大把握。
    # 把握很高而实测不显著 → 效应确实更小，不是功效不足。
    print("\n  被动段零结果的功效核验：")
    for mod in ("hr", "pupil"):
        act = np.array([np.nanmean(make_pairs(D[D.subject == s], mod, ACTIVE))
                        for s in subs])
        # 组水平检验的效应量：跨受试均值 / 跨受试标准差
        act = act[np.isfinite(act)]
        pas_n = int(np.median([len(make_pairs(D[D.subject == s], mod, PASSIVE))
                               for s in subs]))
        act_n = int(np.median([len(make_pairs(D[D.subject == s], mod, ACTIVE))
                               for s in subs]))
        # 每人均值的噪声随 1/√n 变化；把主动段的组效应量折算到被动段的 n
        within = np.median([np.nanstd(make_pairs(D[D.subject == s], mod, ACTIVE))
                            for s in subs])
        between = float(np.std(act, ddof=1))
        # 被动段每人均值的标准差 = √(个体间真实差异² + 个体内噪声²/n)
        true_var = max(between**2 - within**2 / act_n, 0.0)
        sd_pas = np.sqrt(true_var + within**2 / max(pas_n, 1))
        d_group = abs(act.mean()) / sd_pas
        pw = t_power(d_group, len(act))
        print(f"    {mod:<6} 主动段效应 {act.mean():+.4f}   被动段每人 {pas_n} 次   "
              f"若等大则组水平把握 {pw:.3f}")
        run.log_metric(f"passive_power_if_equal_{mod}", float(pw))
    print("    → 把握高而实测不显著，则被动段效应确实更小，不是功效不足")

    # ══ 1. 逐个体阳性对照 ═════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("[1/4] 逐个体检验：这个人自己的数据，能否测出事件锁时响应")
    rows = []
    for sid in subs:
        g = D[D.subject == sid]
        for mod in MODS:
            d = make_pairs(g, mod, ACTIVE)
            if len(d) < MIN_PAIRS:
                rows.append({"subject": sid, "mod": mod, "n": len(d),
                             "mean": np.nan, "p": np.nan})
                continue
            signs = rng.integers(0, 2, (N_PERM, len(d))) * 2 - 1
            p = signflip_p(d[None, :], signs)[0]
            rows.append({"subject": sid, "mod": mod, "n": len(d),
                         "mean": float(d.mean()),
                         "d": float(d.mean() / (d.std(ddof=1) + 1e-12)),
                         "p": float(p)})
    S = pd.DataFrame(rows)
    for mod in MODS:
        m = S[(S["mod"] == mod) & S.p.notna()]
        if not len(m):
            continue
        q = false_discovery_control(m.p.to_numpy(), method="bh")
        S.loc[m.index, "q"] = q
        print(f"  {mod:<10} n={len(m):>3} 人   "
              f"α=.05 通过 {int((m.p < ALPHA).sum()):>3}   "
              f"BH-FDR 通过 {int((q < ALPHA).sum()):>3}   "
              f"效应量 d 中位 {m['d'].median():+.3f}   "
              f"每人配对中位 {m.n.median():.0f}")
    S.to_parquet(OUT_SUBJ, index=False)

    # ══ 2. 假阳性对照：随机锚点臂自身对半劈 ═══════════════════════════
    print("\n" + "=" * 78)
    print("[2/4] 假阳性对照：随机锚点臂对半劈，此处不该有效应")
    fp_rows = []
    for sid in subs:
        g = D[(D.subject == sid) & (D.arm == "neg")]
        for mod in MODS:
            v = g[mod].dropna().to_numpy(float).copy()
            rng.shuffle(v)
            h = len(v) // 2
            if h < MIN_PAIRS:
                continue
            d = v[:h] - v[h:2 * h]
            signs = rng.integers(0, 2, (N_PERM, len(d))) * 2 - 1
            fp_rows.append({"subject": sid, "mod": mod,
                            "p": float(signflip_p(d[None, :], signs)[0])})
    F = pd.DataFrame(fp_rows)
    for mod in MODS:
        m = F[F["mod"] == mod]
        if len(m):
            print(f"  {mod:<10} 假阳性率 {(m.p < ALPHA).mean():.3f}   "
                  f"（名义 {ALPHA}，n={len(m)}）")
    print("  注：随机锚点可在时间上互相重叠，脑电又有自相关，"
          "因此对半劈的两半并非独立，")
    print("      该对照略偏宽松。主检验用的符号翻转零分布不受此影响 ——"
          "见 [3] 的假阳列。")

    # ══ 3. 实测曲线：检出率 vs 每人试次数 ═════════════════════════════
    print("\n" + "=" * 78)
    print("[3/4] 实测曲线：检出率如何随每人试次数上升")
    curve = []
    for mod in MODS:
        pairs = {sid: make_pairs(D[D.subject == sid], mod, ACTIVE)
                 for sid in subs}
        pairs = {k: v for k, v in pairs.items() if len(v) >= max(N_GRID[0], MIN_PAIRS)}
        # 解析预期：以各人全量数据估出的 d，代入单样本 t 检验功效
        dhat = {k: v.mean() / (v.std(ddof=1) + 1e-12) for k, v in pairs.items()}
        for n in N_GRID:
            elig = {k: v for k, v in pairs.items() if len(v) >= n}
            if not elig:
                continue
            signs = rng.integers(0, 2, (N_PERM_CURVE, n)) * 2 - 1
            det, fp = [], []
            for sid, v in elig.items():
                idx = np.array([rng.choice(len(v), n, replace=False)
                                for _ in range(N_REPEAT)])
                p = signflip_p(v[idx], signs)
                det.append((p < ALPHA).mean())
                # 同一受试的零对照：把配对差随机取符号（保留噪声量级）后再抽
                vn = v * (rng.integers(0, 2, len(v)) * 2 - 1)
                fp.append((signflip_p(vn[idx], signs) < ALPHA).mean())
            ana = float(np.mean([t_power(dhat[s], n) for s in elig]))
            curve.append({"mod": mod, "n": n, "n_subjects": len(elig),
                          "detect": float(np.mean(det)),
                          "detect_sd": float(np.std(det)),
                          "false_pos": float(np.mean(fp)),
                          "analytic": ana})
            print(f"  {mod:<10} n={n:>4}   {len(elig):>2} 人   "
                  f"检出 {np.mean(det):.3f}   假阳 {np.mean(fp):.3f}   "
                  f"解析 {ana:.3f}", flush=True)
    C = pd.DataFrame(curve)
    C.to_parquet(OUT_CURVE, index=False)

    # 实测 vs 解析（同批数据）：只验证函数形式，**不验证外推** ——
    # dhat 是用全量数据估的，与被预测的子样本同源，存在循环性。
    print("\n  实测与解析功效的偏离（同批数据，|实测 − 解析| 均值）：")
    for mod in MODS:
        c = C[C["mod"] == mod]
        if len(c):
            gap = float((c.detect - c.analytic).abs().mean())
            print(f"    {mod:<10} {gap:.3f}")
            run.log_metric(f"empirical_vs_analytic_gap_{mod}", gap)

    # ══ 3b. 半分交叉：这才是对「外推」本身的检验 ═══════════════════════
    #
    # 论文（`56` / Table 1）的做法是「估出效应量 → 算所需次数」。
    # 上面那个比较用同一批数据估 d 又用它预测，验证不了这一步。
    # 这里用**一半试次估 d，预测另一半的检出率** —— 与论文的外推同构。
    print("\n" + "=" * 78)
    print("[3b] 半分交叉：用一半试次估效应量，预测另一半的检出率")
    for mod in MODS:
        pairs = {s: make_pairs(D[D.subject == s], mod, ACTIVE) for s in subs}
        pairs = {k: v for k, v in pairs.items() if len(v) >= 80}
        if not pairs:
            continue
        rows_sh = []
        for n in (20, 30, 50, 75):
            signs = rng.integers(0, 2, (N_PERM_CURVE, n)) * 2 - 1
            obs, pred = [], []
            for v in pairs.values():
                if len(v) < 2 * n:
                    continue
                perm = rng.permutation(len(v))
                a, b = v[perm[: len(v) // 2]], v[perm[len(v) // 2:]]
                d_est = a.mean() / (a.std(ddof=1) + 1e-12)      # 估自 A 半
                idx = np.array([rng.choice(len(b), n, replace=False)
                                for _ in range(N_REPEAT)])
                obs.append((signflip_p(b[idx], signs) < ALPHA).mean())  # 测自 B 半
                pred.append(t_power(d_est, n))
            if obs:
                rows_sh.append((n, len(obs), np.mean(obs), np.mean(pred)))
        for n, k, o, p in rows_sh:
            print(f"  {mod:<10} n={n:>3}  {k:>2} 人   "
                  f"实测 {o:.3f}   外推预测 {p:.3f}   偏离 {abs(o-p):+.3f}")
        if rows_sh:
            g = float(np.mean([abs(o - p) for _, _, o, p in rows_sh]))
            run.log_metric(f"splithalf_extrapolation_gap_{mod}", g)

    # ══ 4. 按年龄分组：年长用户是否需要更多校准数据 ═══════════════════
    #
    # 直接关系产品：若年长者的效应量系统性更小，则同一套校准时长
    # 在他们身上给出的把握度更低 —— 而助眠产品的主要用户恰恰偏年长。
    print("\n" + "=" * 78)
    print("[4/4] 按年龄分组的所需试次数")
    PT = REPO_ROOT / "data" / "raw" / "ds003690" / "participants.tsv"
    if PT.exists():
        P = pd.read_csv(PT, sep="\t").set_index("participant_id")
        for mod in MODS:
            line = []
            for grp_name in ("Young", "Older"):
                ids = [s for s in subs
                       if s in P.index and P.loc[s, "group"] == grp_name]
                ds = [np.abs(v.mean() / (v.std(ddof=1) + 1e-12))
                      for v in (make_pairs(D[D.subject == s], mod, ACTIVE)
                                for s in ids) if len(v) >= MIN_PAIRS]
                if not ds:
                    continue
                med = float(np.median(ds))
                # 该中位效应量下达到 80% 功效所需的每人试次数
                need = next((n for n in range(10, 5001)
                             if t_power(med, n) >= 0.80), None)
                line.append(f"{grp_name} n={len(ds):>2} d̃={med:.3f} "
                            f"需 {need if need else '>5000':>5}")
                run.log_metric(f"d_median_{mod}_{grp_name}", med)
                if need:
                    run.log_metric(f"n_required_{mod}_{grp_name}", int(need))
            if line:
                print(f"  {mod:<10} " + "   |   ".join(line))

    # ══ 结论行 ════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("关键对照：ds002721 的每人试次数 ≈ 30")
    for mod in MODS:
        c = C[C["mod"] == mod]
        if not len(c):
            continue
        at30 = c.loc[(c.n - 30).abs().idxmin()]
        top = c.iloc[-1]
        print(f"  {mod:<10} n=30 检出 {at30.detect:.1%}   "
              f"→ n={int(top.n)} 检出 {top.detect:.1%}")
        run.log_metric(f"detect_n30_{mod}", float(at30.detect))
        run.log_metric(f"detect_max_{mod}", float(top.detect))

    for mod in MODS:
        m = S[(S["mod"] == mod) & S.p.notna()]
        if len(m):
            run.log_metric(f"n_pass_fdr_{mod}",
                           int((m.get("q", pd.Series(1.0, m.index)) < ALPHA).sum()))
            run.log_metric(f"n_subjects_{mod}", int(len(m)))
    for mod in MODS:
        m = F[F["mod"] == mod]
        if len(m):
            run.log_metric(f"false_pos_{mod}", float((m.p < ALPHA).mean()))
    run.write()
    print(f"\n写出 {OUT_SUBJ.name} / {OUT_CURVE.name}")


if __name__ == "__main__":
    main()
