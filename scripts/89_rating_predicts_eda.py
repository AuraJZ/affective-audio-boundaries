r"""A2：主观评分能不能预测歌曲级皮电 —— 「信息型失败」的直接证据。

    .venv/Scripts/python scripts/89_rating_predicts_eda.py

## 这一步要回答什么

`88` 在同一批 717 首歌上给出：声学达到主观评分可达上限的 **84%**，
达到皮电的 **4–7%**。差 15 倍。

但那还不足以说「瓶颈在声学表征」。有两种解释：

| | 含义 | 如何区分 |
|---|---|---|
| **信息型** | 皮电里那部分可靠方差，**信息不在波形里** | 若**评分**能预测皮电，则信息存在、只是声学读不到 |
| **尺度/构念型** | 皮电的刺激层方差对**任何**刺激层预测器都不可读 | 评分也预测不了 |

两种结局对论文的含义完全不同，**故先写死再跑**：

- **评分能预测皮电** → §8 是「信息型失败」，主线的二分成立，
  且指向「转向语义与情境信息」这个可执行结论。
- **评分也不能** → §8 改为「歌曲级皮电在刺激层对任何刺激层预测器都不可读」，
  主线的二分改为「预算型 vs 尺度/构念型」，**标题不变**
  （「读不到身体」在两种情况下都成立，后者只会更强）。

## 关键设计：把评分拆成两部分 ⭐

不能只报「评分 → 皮电」的相关，那说明不了瓶颈在哪。要拆：

    评分 = 声学可预测部分（折外预测值） + 残差

- 若预测皮电的主要是**残差** → 皮电所依赖的信息**恰好是声学读不到的那部分**，
  这是信息型失败最直接的证据。
- 若主要是**声学可预测部分** → 那么声学本应也能预测皮电，
  与 `74`/`87` 的阴性矛盾，须重新检查。

**不要用 0.82 × 0.13 这类链式乘法论证** —— 相关系数不满足传递性，
Cauchy–Schwarz 一问就破。必须直接回归。

## 混杂控制

歌曲级皮电与评分都可能被**时长**、**艺人**带动。故报三个量：
原始相关、控制时长的偏相关、按艺人分组的交叉验证。

## 一条无法排除的替代解释

PMEmo 的评分与皮电**来自同一批听众的同一次会话**。若两者的共变来自
「这个人这一刻的状态」而非「这首歌」，则相关会被抬高。
数据集只发布了逐首均值，**没有逐标注者原始评分**，
因此无法做「一半人的评分 vs 另一半人的皮电」的不交叉对照。
本脚本会检查该文件是否存在；不存在则把此限制写死在正文。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")

FEAT = REPO_ROOT / "data" / "features"
PM = REPO_ROOT / "data" / "raw" / "PMEmo" / "PMEmo2019"
OUT = REPO_ROOT / "reports" / "source_data" / "rating_predicts_eda.csv"
SEED = 20260801
EDA = ["scr_rate", "phasic_mean", "scl_slope"]
RATINGS = ["Arousal(mean)", "Valence(mean)"]
N_PC = 120
N_PERM = 2000
N_BOOT = 2000
MIN_LISTENERS = 8


def partial_spearman(x, y, z) -> float:
    """控制 z 后 x 与 y 的偏相关（秩残差法）。"""
    from scipy.stats import rankdata
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    Z = np.c_[np.ones(len(rz)), rz]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r = spearmanr(ex, ey).statistic
    return float(r) if np.isfinite(r) else np.nan


def perm_p(x, y, rng, n=N_PERM) -> float:
    obs = abs(spearmanr(x, y).statistic)
    null = np.array([abs(spearmanr(rng.permutation(x), y).statistic)
                     for _ in range(n)])
    return float((1 + (null >= obs).sum()) / (1 + n))


def boot_ci(x, y, rng, n=N_BOOT):
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        r = spearmanr(x[i], y[i]).statistic
        if np.isfinite(r):
            v.append(r)
    return (float(np.quantile(v, .025)), float(np.quantile(v, .975))) if v \
        else (np.nan, np.nan)


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("89_rating_predicts_eda", SEED)

    # ── 同源集合，与 `88` 逐字一致 ────────────────────────────────────
    from soundml.regression import feature_cols
    a_cols = feature_cols(pd.read_parquet(FEAT / "layer_a_reg_deam_ln.parquet"))
    A = pd.read_parquet(FEAT / "layer_a_reg_pmemo_ln.parquet")
    Dl = pd.read_parquet(FEAT / "layer_d_reg_pmemo_ln.parquet")
    d_cols = [c for c in Dl.columns if c != "clip_id"]
    F = A[["clip_id"] + a_cols].merge(Dl, on="clip_id")
    F["musicId"] = F["clip_id"].astype(str).str.removeprefix("pmemo_").astype(int)
    feats = a_cols + d_cols

    Am = pd.read_csv(PM / "annotations" / "static_annotations.csv")
    T = pd.read_parquet(FEAT / "pmemo_eda_song_level.parquet")
    for m in EDA:
        g = T.groupby("listener")[m]
        T[m] = (T[m] - g.transform("mean")) / (g.transform("std") + 1e-9)
    agg = T.groupby("musicId").agg(
        **{m: (m, "mean") for m in EDA},
        n_listeners=("listener", "size"),
        duration=("duration", "first")).reset_index()

    M = (F.merge(Am, on="musicId").merge(agg, on="musicId")
         .merge(pd.read_csv(PM / "metadata.csv")[["musicId", "artist"]],
                on="musicId", how="left"))
    M["artist"] = M["artist"].fillna("u" + M.musicId.astype(str))
    M = M[M.n_listeners >= MIN_LISTENERS].reset_index(drop=True)
    X = np.nan_to_num(M[feats].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    groups = M["artist"].to_numpy()
    print(f"同源集合 {len(M)} 首 × {X.shape[1]} 维   艺人组 {len(set(groups))}\n")

    # ── 是否存在逐标注者原始评分 ─────────────────────────────────────
    raw = list((PM / "annotations").glob("*rater*")) + \
          list((PM / "annotations").glob("*individual*"))
    print("=" * 88)
    print(f"[0] 逐标注者原始评分：{'找到 ' + str(raw) if raw else '**不存在**'}")
    if not raw:
        print("    → 无法做「一半人评分 vs 另一半人皮电」的不交叉对照。")
        print("    → 「评分与皮电来自同一批听众同一次会话」这一混杂**无法排除**，")
        print("      必须写进正文限制。")
    run.log_metric("has_per_rater_ratings", bool(raw))

    # ── ① 评分 → 皮电 ────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("[1/3] 主观评分能否预测歌曲级皮电")
    print(f"  {'评分':<18}{'皮电':<14}{'ρ':>8}{'p':>9}   95% CI"
          f"{'  偏相关(控时长)':>18}")
    rows = []
    for r_ in RATINGS:
        for m in EDA:
            x, y = M[r_].to_numpy(float), M[m].to_numpy(float)
            rho = float(spearmanr(x, y).statistic)
            p = perm_p(x, y, rng)
            lo, hi = boot_ci(x, y, rng)
            pr = partial_spearman(x, y, M["duration"].to_numpy(float))
            print(f"  {r_:<18}{m:<14}{rho:>+8.3f}{p:>9.4f}   "
                  f"[{lo:+.3f}, {hi:+.3f}]{pr:>+18.3f}")
            rows.append({"kind": "rating_to_eda", "rating": r_, "eda": m,
                         "rho": rho, "p": p, "ci_lo": lo, "ci_hi": hi,
                         "partial_duration": pr})

    # ── ② 把评分拆成「声学可预测部分」与「残差」 ─────────────────────
    print("\n" + "=" * 88)
    print("[2/3] 拆分：预测皮电的是评分的哪一部分")
    print("      评分 = 声学折外预测值 + 残差")
    print(f"  {'评分':<18}{'皮电':<14}{'声学部分 ρ':>12}{'残差 ρ':>10}"
          f"{'  判读':>12}")
    for r_ in RATINGS:
        y_rate = M[r_].to_numpy(float)
        pred = np.full(len(y_rate), np.nan)
        for tr, te in GroupKFold(5).split(X, y_rate, groups):
            mdl = make_pipeline(StandardScaler(), PCA(N_PC),
                                RidgeCV(alphas=np.logspace(-2, 4, 25)))
            mdl.fit(X[tr], y_rate[tr])
            pred[te] = mdl.predict(X[te])
        resid = y_rate - pred
        run.log_metric(f"acoustic_rho_{r_}", float(spearmanr(pred, y_rate).statistic))
        for m in EDA:
            y = M[m].to_numpy(float)
            r_ac = float(spearmanr(pred, y).statistic)
            r_re = float(spearmanr(resid, y).statistic)
            p_ac, p_re = perm_p(pred, y, rng, 1000), perm_p(resid, y, rng, 1000)
            verdict = ("残差主导 → 信息型" if abs(r_re) > abs(r_ac) + 0.02
                       else ("声学部分主导 → 与 74/87 矛盾"
                             if abs(r_ac) > abs(r_re) + 0.02 else "相当"))
            print(f"  {r_:<18}{m:<14}{r_ac:>+12.3f}{r_re:>+10.3f}   {verdict}")
            rows.append({"kind": "decomposed", "rating": r_, "eda": m,
                         "rho_acoustic_part": r_ac, "p_acoustic": p_ac,
                         "rho_residual": r_re, "p_residual": p_re})

    # ── ③ 判定 ───────────────────────────────────────────────────────
    D = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT, index=False, encoding="utf-8")

    print("\n" + "=" * 88)
    R2E = D[D.kind == "rating_to_eda"]
    best = R2E.loc[R2E.rho.abs().idxmax()]
    n_sig = int((R2E.p < 0.05).sum())
    # 声学对同一批皮电的最好成绩（来自 88/87），作为参照
    ACOUSTIC_BEST = 0.032
    print(f"评分 → 皮电：{n_sig}/{len(R2E)} 格 p<.05；"
          f"最强 {best.rating} → {best.eda}  ρ = {best.rho:+.3f} "
          f"(p = {best.p:.4f})")
    print(f"声学 → 同一批皮电（`87` 最好）：ρ = {ACOUSTIC_BEST:+.3f}")
    ratio = abs(best.rho) / ACOUSTIC_BEST
    print(f"倍数 {ratio:.1f}×")
    print()
    if n_sig >= 2 and abs(best.rho) > 2 * ACOUSTIC_BEST:
        print("→ **信息型失败成立**：皮电里的可靠方差是可读的，")
        print("  只是声学读不到。§8 按预案 A 写，主线的二分成立。")
    elif n_sig == 0:
        print("→ **评分也读不到**：歌曲级皮电在刺激层对任何刺激层预测器都不可读。")
        print("  §8 按预案 B 改写，二分改为「预算型 vs 尺度/构念型」，标题不变。")
    else:
        print("→ 介于两者之间：评分略优于声学但都很弱。")
        print("  按最保守的预案 B 写，并把评分的优势作为方向性证据报告。")
    run.log_metric("n_sig_rating_to_eda", n_sig)
    run.log_metric("best_rating_to_eda_rho", float(best.rho))
    run.log_metric("ratio_vs_acoustic", float(ratio))
    run.write()
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
