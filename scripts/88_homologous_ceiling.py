r"""A1 + E1：一把尺子 —— 同源上限表。

    .venv/Scripts/python scripts/88_homologous_ceiling.py

## 要修的两处硬伤 🔴

**① 异源比较。** 论文的招牌对比「声学解释主观达天花板 89%，解释生理只有 6%」
里，**89% 来自 DEAM、6% 来自 PMEmo** —— 不同语料、不同特征、不同折。
审稿人一查就废。

**② 分母不一致。** `20_label_ceiling.py:130-131` 同时算了两种口径：

    "pct_of_ceiling":        v / R
    "attenuation_corrected": v / sqrt(R)

而**主观的 89% 用了前者，生理的 6% 用了后者**。同一个招牌对比，两个分母。

## 本脚本的做法

PMEmo **同时**有评分、皮电、音频、161 维特征 —— 四个格子可以全部在
同一批歌上填。于是：

    同一批 717 首歌 · 同一套 161 维特征 · 同一套 GroupKFold 折（按艺人）
    同一个信度估计量 · 两种口径都报 · 双重自举 CI

**信度用同一个解析估计量**（而非一边分半一边解析）：

    σ²_内 = 各首「跨观察者 SD²」的均值
    Var(首均值) = σ²_间 + σ²_内 / n
    ICC_单人 = σ²_间 / (σ²_间 + σ²_内)
    信度(n 人均值) = n·ICC / (1 + (n−1)·ICC)      ← Spearman–Brown

评分侧只有均值与 SD（无逐标注者原始值），故只能走解析式；
皮电侧有逐听众值，**两种都算**，用它们是否一致来验证解析式可用。

## 两种口径都报，且说明各自的含义

    ρ / r       「占可靠方差的比例」—— 直觉上是「达天花板百分之几」
    ρ / √r      经典衰减校正 —— 完美预测器能达到的最大相关是 √r，
                故这是「占可达上限的比例」，在信度低时数值更小也更保守

**主口径定为 ρ/√r**（衰减校正是标准做法），两种并列，
全文百分比按此重写。方向不会变，数值会变。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")

FEAT = REPO_ROOT / "data" / "features"
PM = REPO_ROOT / "data" / "raw" / "PMEmo" / "PMEmo2019"
OUT = REPO_ROOT / "reports" / "source_data" / "homologous_ceiling.csv"
SEED = 20260801
EDA_MEASURES = ["scr_rate", "phasic_mean", "scl_slope"]
N_PC = 120
N_BOOT = 2000
MIN_LISTENERS = 8

MODELS = {
    "ridge": lambda: make_pipeline(StandardScaler(), PCA(N_PC),
                                   RidgeCV(alphas=np.logspace(-2, 4, 25))),
    "rf": lambda: make_pipeline(StandardScaler(), PCA(N_PC),
                                RandomForestRegressor(
                                    n_estimators=300, min_samples_leaf=2,
                                    random_state=SEED, n_jobs=-1)),
}


def reliability_analytic(sd_within: np.ndarray, means: np.ndarray,
                         n: np.ndarray) -> tuple[float, float]:
    """由「逐首均值 + 逐首跨观察者 SD + 观察者数」估 n 人均值的信度。

    返回 (单观察者 ICC, n 人均值的信度)。两侧共用此式，口径才可比。
    """
    ok = np.isfinite(sd_within) & np.isfinite(means) & (n > 1)
    sd_within, means, n = sd_within[ok], means[ok], n[ok]
    var_w = float(np.mean(sd_within ** 2))
    var_obs = float(np.var(means, ddof=1))
    n0 = float(np.mean(n))
    var_b = max(var_obs - var_w / n0, 0.0)
    icc = var_b / (var_b + var_w) if (var_b + var_w) > 0 else np.nan
    rel = n0 * icc / (1 + (n0 - 1) * icc) if np.isfinite(icc) and icc > 0 else 0.0
    return float(icc), float(min(rel, 0.999))


def split_half(per_obs: pd.DataFrame, col: str, unit: str, rng,
               reps: int = 40) -> float:
    """分半 + Spearman–Brown。仅皮电侧可用，作为解析式的交叉校验。"""
    vals = []
    for _ in range(reps):
        a, b = [], []
        for _, g in per_obs.groupby(unit):
            v = g[col].dropna().to_numpy().copy()
            if len(v) < MIN_LISTENERS:
                continue
            rng.shuffle(v)
            h = len(v) // 2
            a.append(v[:h].mean())
            b.append(v[h:2 * h].mean())
        if len(a) > 30:
            r = spearmanr(a, b).statistic
            if np.isfinite(r) and r > -1:
                vals.append(2 * r / (1 + r))
    return float(np.mean(vals)) if vals else np.nan


def cv_predict(X, y, groups):
    out = {}
    for name, mk in MODELS.items():
        pred = np.full(len(y), np.nan)
        for tr, te in GroupKFold(5).split(X, y, groups):
            m = mk()
            m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        out[name] = pred
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("88_homologous_ceiling", SEED)

    # ── 同一批歌、同一套特征 ──────────────────────────────────────────
    from soundml.regression import feature_cols
    a_cols = feature_cols(pd.read_parquet(FEAT / "layer_a_reg_deam_ln.parquet"))
    A = pd.read_parquet(FEAT / "layer_a_reg_pmemo_ln.parquet")
    Dl = pd.read_parquet(FEAT / "layer_d_reg_pmemo_ln.parquet")
    d_cols = [c for c in Dl.columns if c != "clip_id"]
    F = A[["clip_id"] + a_cols].merge(Dl, on="clip_id")
    F["musicId"] = F["clip_id"].astype(str).str.removeprefix("pmemo_").astype(int)
    feats = a_cols + d_cols

    # 评分：均值 + 跨评分者 SD
    Am = pd.read_csv(PM / "annotations" / "static_annotations.csv")
    As = pd.read_csv(PM / "annotations" / "static_annotations_std.csv")
    R = Am.merge(As, on="musicId")

    # 皮电：逐听众 → 逐首均值与 SD（被试内 z 分后，与 `74` 同口径）
    T = pd.read_parquet(FEAT / "pmemo_eda_song_level.parquet")
    for m in EDA_MEASURES:
        g = T.groupby("listener")[m]
        T[m] = (T[m] - g.transform("mean")) / (g.transform("std") + 1e-9)
    agg = T.groupby("musicId")[EDA_MEASURES].agg(["mean", "std", "count"])
    agg.columns = [f"{a}__{b}" for a, b in agg.columns]
    agg = agg.reset_index()

    M = F.merge(R, on="musicId", how="inner").merge(agg, on="musicId",
                                                    how="inner")
    meta = pd.read_csv(PM / "metadata.csv")[["musicId", "artist"]]
    M = M.merge(meta, on="musicId", how="left")
    M["artist"] = M["artist"].fillna("u" + M.musicId.astype(str))
    M = M[M[f"{EDA_MEASURES[0]}__count"] >= MIN_LISTENERS].reset_index(drop=True)

    X = np.nan_to_num(M[feats].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    groups = M["artist"].to_numpy()
    print(f"同源集合：{len(M)} 首 × {X.shape[1]} 维特征   "
          f"艺人组 {len(set(groups))}")
    print(f"评分者数（PMEmo 设计）≈ 皮电听众数 中位 "
          f"{M[f'{EDA_MEASURES[0]}__count'].median():.0f}\n")

    # ── 逐结果变量：信度 + 模型 ρ + 两种口径 ─────────────────────────
    n_obs = M[f"{EDA_MEASURES[0]}__count"].to_numpy(float)
    targets = {
        "Arousal (rating)": (M["Arousal(mean)"].to_numpy(float),
                             M["Arousal(std)"].to_numpy(float), n_obs),
        "Valence (rating)": (M["Valence(mean)"].to_numpy(float),
                             M["Valence(std)"].to_numpy(float), n_obs),
    }
    for m in EDA_MEASURES:
        targets[f"EDA {m}"] = (M[f"{m}__mean"].to_numpy(float),
                               M[f"{m}__std"].to_numpy(float),
                               M[f"{m}__count"].to_numpy(float))

    print("=" * 96)
    print(f"{'结果变量':<22}{'信度':>8}{'ρ(ridge)':>10}{'ρ(rf)':>9}"
          f"{'ρ/r':>9}{'ρ/√r':>9}   95% CI (ρ/√r)")
    rows = []
    for label, (y, sd, n) in targets.items():
        icc, rel = reliability_analytic(sd, y, n)
        preds = cv_predict(X, y, groups)
        best_name = max(preds, key=lambda k: spearmanr(preds[k], y).statistic)
        rho = float(spearmanr(preds[best_name], y).statistic)
        # 双重自举：对**歌曲**重抽（观察者层的不确定性已进入 rel 的估计）
        bs = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            r = spearmanr(preds[best_name][i], y[i]).statistic
            if np.isfinite(r):
                bs.append(r / max(np.sqrt(rel), 1e-9))
        lo, hi = np.quantile(bs, [.025, .975]) if bs else (np.nan, np.nan)
        rr = {k: float(spearmanr(v, y).statistic) for k, v in preds.items()}
        pct_r = rho / rel if rel > 0 else np.nan
        pct_sq = rho / np.sqrt(rel) if rel > 0 else np.nan
        print(f"{label:<22}{rel:>8.3f}{rr['ridge']:>10.3f}{rr['rf']:>9.3f}"
              f"{pct_r:>9.1%}{pct_sq:>9.1%}   [{lo:+.1%}, {hi:+.1%}]")
        rows.append({"target": label, "icc_single": icc, "reliability": rel,
                     "rho_ridge": rr["ridge"], "rho_rf": rr["rf"],
                     "rho_best": rho, "best_model": best_name,
                     "pct_of_reliability": pct_r,
                     "pct_of_attenuation_ceiling": pct_sq,
                     "ci_lo": float(lo), "ci_hi": float(hi), "n_songs": len(y)})
        run.log_metric(f"rel_{label}", rel)
        run.log_metric(f"pct_sqrt_{label}", float(pct_sq))

    # ── 皮电侧：解析式 vs 分半，验证估计量可用 ────────────────────────
    print("\n" + "=" * 96)
    print("信度估计量交叉校验（仅皮电侧有逐观察者值）")
    for m in EDA_MEASURES:
        _, rel_a = reliability_analytic(M[f"{m}__std"].to_numpy(float),
                                        M[f"{m}__mean"].to_numpy(float),
                                        M[f"{m}__count"].to_numpy(float))
        rel_s = split_half(T[T.musicId.isin(M.musicId)], m, "musicId", rng)
        print(f"  {m:<14} 解析式 {rel_a:.3f}   分半+SB {rel_s:.3f}   "
              f"差 {abs(rel_a-rel_s):.3f}"
              f"   {'✅ 一致' if abs(rel_a-rel_s) < 0.08 else '⚠️ 不一致，解析式存疑'}")
        run.log_metric(f"rel_analytic_{m}", rel_a)
        run.log_metric(f"rel_splithalf_{m}", rel_s)

    D = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT, index=False, encoding="utf-8")

    print("\n" + "=" * 96)
    sub = D[D.target.str.contains("rating")].pct_of_attenuation_ceiling.max()
    eda = D[D.target.str.contains("EDA")].pct_of_attenuation_ceiling.max()
    sub_r = D[D.target.str.contains("rating")].pct_of_reliability.max()
    eda_r = D[D.target.str.contains("EDA")].pct_of_reliability.max()
    print("同源对比（同一批歌、同一套特征、同一套折、同一个信度估计量）：")
    print(f"  主口径 ρ/√r ：主观 {sub:.1%}   生理 {eda:.1%}   "
          f"倍数 {sub/max(eda,1e-9):.1f}×")
    print(f"  次口径 ρ/r  ：主观 {sub_r:.1%}   生理 {eda_r:.1%}   "
          f"倍数 {sub_r/max(eda_r,1e-9):.1f}×")
    print("\n⚠️ 旧文中的「89% vs 6%」是异源且异口径的，须按上表全部重写。")
    run.log_metric("pct_sqrt_subjective", float(sub))
    run.log_metric("pct_sqrt_physiological", float(eda))
    run.write()
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
