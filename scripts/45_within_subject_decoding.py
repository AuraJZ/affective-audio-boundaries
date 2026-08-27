"""被试内解码 —— 文献做的是这个问题，我此前做的是另一个。

    .venv/Scripts/python scripts/45_within_subject_decoding.py

## 两个问题不是一个问题

| | 问的是 | 本项目 | 主流文献 |
|---|---|---|---|
| **跨受试刺激层面信度** | 不同的人对同一段音乐的脑电反应是否一致 | `step17`，ICC ≤ 0.040 | 几乎不做 |
| **被试内解码** | 在一个人身上，脑电能否追踪他自己报告的状态 | **本脚本** | 标准做法 |

两者的答案**可以同时是**「跨受试≈0」与「被试内良好」——
只要个体差异足够大。若如此，`step17` 的阴性不与文献冲突，
而是精确地否定了「**通用**声学→生理规则库」这一设想，同时支持「按人校准」。

## 三个层次，逐层加严

1. **被试内相关**：每人 ~40 试次上，26 个脑电量 × 8 个评分维度的 Spearman
2. **被试内交叉验证解码**：留一试次 Ridge，报告 ρ(预测, 实际)
3. **跨受试符号一致性**：不同人学到的关系方向是否相同 ——
   这正是「被试内成功」与「跨受试失败」之间的桥

## 必须有的对照 ⚠️

- **零分布**：被试内打乱试次标签，跑同一流程。40 试次 × 26 特征，
  过拟合风险高，没有零分布的解码准确率无意义。
- **试次序号偏相关**：脑电与评分都可能随疲劳漂移，
  同向漂移会伪造相关。故同时报告偏掉 `trial_index` 后的结果。

## v1 的两处错误（已修）🔴

**1. 留一交叉验证 + 相关系数是坏的组合。** v1 的实测与零分布**都是 −0.5 左右**
（pleasant −0.502 vs −0.549），这是 LOOCV 的已知伪迹：留出一点时，
训练集均值系统性偏离该点，使预测与实际反相关。改用 **5 折**。

**2. `MIN_TRIALS=25` 把大半数据滤掉了。** 每人每维度本应有 40 个评分，
实际只提出 ~22 个 —— 因为 v1 要求作答码与问题码**时刻完全相等**，
而两者常差几十毫秒。改为**窗口匹配**（±0.4 s），并把门槛降到 15。

v1 因此只有 2 个维度、11–16 名受试进入分析。**「被试内不可解码」的结论无效。**
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

warnings.filterwarnings("ignore")

ROOT = REPO_ROOT / "data" / "raw" / "ds002721"
TRIALS = REPO_ROOT / "data" / "features" / "ds002721_trials_raw.parquet"
QUESTIONS = ["pleasant", "energetic", "tense", "angry",
             "afraid", "happy", "sad", "tender"]
BANDS = ("delta", "theta", "alpha", "beta", "gamma")
MIN_TRIALS = 15
N_FOLDS = 5
N_PERM = 200
MATCH_TOL = 0.40          # 作答码与问题码的时刻容差（秒）


def reextract_ratings() -> pd.DataFrame:
    """从 events.tsv 重提逐试次评分，用窗口匹配而非精确相等。

    真作答码是 `901–909`（`step17` §3.1 已判定）。问题码 `800–807` 常与之
    相差几十毫秒，v1 用 `np.isclose` 精确匹配，丢掉了约 45% 的评分。
    """
    rows = []
    for sd in sorted(ROOT.glob("sub-*")):
        for r in (2, 3, 4, 5):
            p = sd / "eeg" / f"{sd.name}_task-run{r}_events.tsv"
            if not p.exists():
                continue
            e = pd.read_csv(p, sep="\t")
            e["trial_type"] = pd.to_numeric(e["trial_type"], errors="coerce")
            e = e.dropna(subset=["trial_type", "onset"]).sort_values("onset")
            on = np.sort(e.loc[e.trial_type == 788, "onset"].to_numpy())
            q = e[e.trial_type.between(800, 807)]
            qt, qv = q.onset.to_numpy(), q.trial_type.to_numpy().astype(int)
            ans = e[e.trial_type.between(901, 909)]
            for t, v in zip(ans.onset.to_numpy(), ans.trial_type.to_numpy().astype(int)):
                if not qt.size:
                    continue
                j = int(np.argmin(np.abs(qt - t)))
                if abs(qt[j] - t) > MATCH_TOL:
                    continue
                prev = np.where(on <= t)[0]
                if not prev.size:
                    continue
                rows.append({"subject": sd.name, "run": r, "trial": int(prev[-1]),
                             "question": QUESTIONS[qv[j] - 800], "rating": v - 900})
    R = pd.DataFrame(rows)
    # 同一试次同一问题可能被多次触发，取众数
    R = (R.groupby(["subject", "run", "trial", "question"])["rating"]
           .agg(lambda s: s.mode().iloc[0]).reset_index())
    return R.pivot_table(index=["subject", "run", "trial"], columns="question",
                         values="rating").reset_index()


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """偏 Spearman：控制 z（此处为试次序号）后的 x–y 秩相关。"""
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    def resid(a):
        A = np.c_[np.ones_like(rz), rz]
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    ex, ey = resid(rx), resid(ry)
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(ex, ey)[0, 1])


def decode(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    """K 折 Ridge 解码，返回 ρ(预测, 实际)。

    **不用留一** —— LOOCV 下预测与实际系统性反相关（训练集均值偏离被留出点），
    v1 的实测与零分布因此都落在 −0.5 附近，检验完全失效。
    """
    if len(y) < MIN_TRIALS or np.std(y) < 1e-9:
        return np.nan
    pred = np.empty(len(y))
    for tr, te in KFold(N_FOLDS, shuffle=True, random_state=seed).split(X):
        if np.std(y[tr]) < 1e-9:
            return np.nan
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(sc.transform(X[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    r = spearmanr(pred, y).statistic
    return float(r) if np.isfinite(r) else np.nan


def main() -> None:
    D = pd.read_parquet(TRIALS)
    eeg = [c for c in D.columns if any(c.startswith(b) for b in BANDS) or c == "faa"]
    D = D[["subject", "run", "trial", "stim"] + eeg]
    R = reextract_ratings()
    D = D.merge(R, on=["subject", "run", "trial"], how="left")
    D["trial_index"] = D.groupby("subject").cumcount()

    comp = {q: float(D[q].notna().mean()) for q in QUESTIONS if q in D.columns}
    print("评分完整率（窗口匹配后）：" +
          "  ".join(f"{q}={100*v:.0f}%" for q, v in comp.items()))
    per = D.groupby("subject")[[q for q in QUESTIONS if q in D.columns]].count()
    print(f"每人每维度评分数：中位 {per.median().median():.0f}  "
          f"最少 {per.min().min()}  最多 {per.max().max()}")

    with RunRecord(
        "within_subject_decoding", seed=SEED,
        params={"dataset": "ds002721", "n_eeg_features": len(eeg),
                "min_trials": MIN_TRIALS, "n_perm": N_PERM,
                "rating_code": "901-909（step17 已判定）",
                "question": "文献做的被试内解码，与 step17 的跨受试信度是两个问题",
                "controls": ["被试内打乱试次的零分布", "偏掉试次序号"]},
    ) as run:
        rng = np.random.default_rng(SEED)

        # ---------- 1. 被试内相关 ----------
        rows = []
        for (sub, ), g in D.groupby(["subject"]):
            for q in QUESTIONS:
                s = g[[q, "trial_index"] + eeg].dropna(subset=[q])
                if len(s) < MIN_TRIALS or s[q].nunique() < 3:
                    continue
                for c in eeg:
                    t = s[[q, c, "trial_index"]].dropna()
                    if len(t) < MIN_TRIALS:
                        continue
                    r = spearmanr(t[c], t[q]).statistic
                    rp = partial_spearman(t[c].to_numpy(), t[q].to_numpy(),
                                          t.trial_index.to_numpy())
                    rows.append({"subject": sub, "question": q, "feature": c,
                                 "rho": r, "rho_partial": rp, "n": len(t)})
        C = pd.DataFrame(rows)
        run.log_metric("n_within_subject_correlations", int(len(C)))

        # ---------- 2. 被试内解码 ----------
        dec, dec_null = [], []
        for (sub, ), g in D.groupby(["subject"]):
            Xall = g[eeg].to_numpy(dtype=float)
            if not np.isfinite(Xall).all():
                Xall = np.nan_to_num(Xall, nan=0.0)
            for q in QUESTIONS:
                m = g[q].notna().to_numpy()
                X, y = Xall[m], g[q].to_numpy(dtype=float)[m]
                r = decode(X, y, SEED)
                if not np.isfinite(r):
                    continue
                dec.append({"subject": sub, "question": q, "rho": r, "n": int(m.sum())})
                # 零分布取 10 次打乱的均值，降低单次抽样噪声
                nv = [decode(X, rng.permutation(y), SEED) for _ in range(10)]
                dec_null.append({"subject": sub, "question": q,
                                 "rho": float(np.nanmean(nv))})
            print(f"  解码完成 {sub}", flush=True)
        DEC = pd.DataFrame(dec)
        NUL = pd.DataFrame(dec_null).dropna()
        run.log_metric("n_decoding_cells", int(len(DEC)))

        # ---------- 3. 跨受试符号一致性 ----------
        cons = []
        for (q, c), g in C.groupby(["question", "feature"]):
            if len(g) >= 10:
                v = g.rho_partial.dropna()
                if len(v) >= 10:
                    st = wilcoxon(v)
                    cons.append({"question": q, "feature": c, "n_subjects": len(v),
                                 "mean_rho": float(v.mean()),
                                 "frac_same_sign": float(max((v > 0).mean(),
                                                             (v < 0).mean())),
                                 "p": float(st.pvalue)})
        CONS = pd.DataFrame(cons)
        # BH-FDR
        if len(CONS):
            o = CONS.p.rank(method="first").to_numpy()
            CONS["q"] = np.minimum(1.0, CONS.p * len(CONS) / o)

        # ================= 打印 =================
        print("\n" + "=" * 84)
        print(f"run_id: {run.run_id}   受试 {D.subject.nunique()}   "
              f"试次 {len(D):,}   脑电特征 {len(eeg)}")

        print("\n--- 1. 被试内解码：ρ(留一预测, 实际) ---")
        print(f"{'评分维度':<12}{'受试':>5}{'实测均值':>10}{'零分布均值':>11}"
              f"{'差':>8}{'p':>10}")
        dec_sum = []
        for q in QUESTIONS:
            a = DEC[DEC.question == q].set_index("subject").rho
            b = NUL[NUL.question == q].set_index("subject").rho
            common = a.index.intersection(b.index)
            if len(common) < 8:
                continue
            st = wilcoxon(a.loc[common], b.loc[common])
            dec_sum.append({"question": q, "n": len(common),
                            "obs": float(a.loc[common].mean()),
                            "null": float(b.loc[common].mean()),
                            "p": float(st.pvalue)})
            print(f"{q:<12}{len(common):>5}{a.loc[common].mean():>+10.3f}"
                  f"{b.loc[common].mean():>+11.3f}"
                  f"{a.loc[common].mean()-b.loc[common].mean():>+8.3f}"
                  f"{st.pvalue:>10.4f}")
        run.log_metric("within_subject_decoding", dec_sum)

        print("\n--- 2. 跨受试符号一致性（偏掉试次序号，BH-FDR）---")
        if len(CONS):
            sig = CONS[CONS.q < 0.05].sort_values("p")
            print(f"  {len(CONS)} 个「维度 × 特征」组合，"
                  f"FDR<0.05 的有 **{len(sig)}** 个")
            for r_ in sig.head(10).itertuples():
                print(f"    {r_.question:10s} {r_.feature:18s} "
                      f"ρ均值 {r_.mean_rho:+.3f}   同号 {100*r_.frac_same_sign:.0f}%"
                      f"   q={r_.q:.4f}")
            run.log_metric("consistent_relations",
                           {"n_tested": int(len(CONS)), "n_fdr_sig": int(len(sig)),
                            "top": sig.head(10).to_dict("records")})
        print(f"\n  全部组合的 |ρ均值| 中位数 "
              f"{CONS.mean_rho.abs().median():.3f}" if len(CONS) else "")

        C.to_csv(REPO_ROOT / "reports" / "source_data" /
                 "ds002721_within_subject_corr.csv", index=False)
        DEC.to_csv(REPO_ROOT / "reports" / "source_data" /
                   "ds002721_within_subject_decoding.csv", index=False)
        if len(CONS):
            CONS.to_csv(REPO_ROOT / "reports" / "source_data" /
                        "ds002721_cross_subject_consistency.csv", index=False)

        print("\n" + "=" * 84)
        any_dec = any(d["p"] < 0.05 and d["obs"] > d["null"] for d in dec_sum)
        n_sig = int((CONS.q < 0.05).sum()) if len(CONS) else 0
        if any_dec and n_sig == 0:
            print("→ **被试内可解码，但跨受试无一致关系** ——")
            print("  文献的「被试内」问题与本项目的「跨受试」问题答案不同，两者不冲突。")
            print("  含义：生理必须**按人校准**，通用规则库的前提不成立。")
        elif any_dec and n_sig:
            print("→ 被试内可解码，且有跨受试一致的关系 —— 两条路都开着。")
        elif not any_dec and n_sig == 0:
            print("→ 被试内也不可解码 —— 与文献的常规结果不一致，需进一步核查。")
        else:
            print("→ 被试内解码不显著，但存在跨受试一致关系 —— 结果矛盾，慎解读。")


if __name__ == "__main__":
    main()
