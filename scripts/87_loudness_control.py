r"""D1：「声学读不到身体」会不会是我们自己把声学抹掉了。

    .venv/Scripts/python scripts/87_loudness_control.py

## 这条不做不能投 🔴

`74` 的结论是：歌曲级皮电可靠（信度 0.246），但 161 维声学特征预测不了它
（CV ρ = 0.016–0.032，置换 p = 0.26–0.52）。

**但本项目的全部音频都被归一到 −23 LUFS。** 而**声级是皮肤电最确凿的
物理驱动** —— 这是心理生理学里最少争议的一条。归一化之后
`rms_mean` 的跨曲标准差掉了约 6 倍。

也就是说，「声学读不到身体」这句话，完全可能是**我们自己把声学里
最相关的那一维抹掉了**。不排除这条，`74` 的阴性有一个平庸解释。

响度归一化本身是对的（`step2b` 证明了不归一化时母带电平会成为语料指纹，
虚高跨语料转移）—— 但那是**跨语料迁移**的需要。
**在单一语料内预测生理反应时，它是有害的**：两个目的相反。

## 三条独立路径

| 路径 | 特征 | 音频 |
|---|---|---|
| **A_ln** | 122 维 Layer A（现状） | −23 LUFS 归一化 |
| **A_raw** | 122 维 Layer A（重提） | **原始响度** |
| **ComParE** | PMEmo 自带 6374 维 | **原始响度**（数据集作者提取） |

第三条尤其重要：它由**数据集作者**在原始音频上提取，与我们的流水线完全独立。
若三条都阴性，「声学读不到身体」才立得住；
若后两条阳性而第一条阴性，则 `74` 的结论作废，须改写为
「响度归一化抹掉了声学—皮电关联」。

## 同时量化归一化到底拿走了多少

报告归一化前后**跨曲响度方差**的比值，以及
「仅用响度一维」在两种条件下对皮电的预测力。
这把「可能有影响」变成一个数。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")

FEAT = REPO_ROOT / "data" / "features"
PM = REPO_ROOT / "data" / "raw" / "PMEmo" / "PMEmo2019"
OUT = REPO_ROOT / "reports" / "source_data" / "loudness_control.csv"
SEED = 20260801
MEASURES = ["scr_rate", "phasic_mean", "scl_slope"]
N_PERM = 100
MIN_LISTENERS = 8

# 三条路径的维度差 50 倍（122 vs 6374）。若不统一降维：
#   ① 随机森林在 6374 维上跑 200 次置换 × 5 折算不完；
#   ② 更要紧的是**比较不公平** —— 高维路径会因维度诅咒被系统性压低，
#      那样「ComParE 也阴性」就成了维度的结论而不是响度的结论。
# 因此所有路径统一走 标准化 → PCA(≤N_PC) → 模型。
# 对 122 维的 Layer A 这几乎是恒等变换，对 ComParE 是标准做法。
N_PC = 120


def _pipe(model):
    from sklearn.decomposition import PCA
    return make_pipeline(StandardScaler(), PCA(n_components=N_PC), model)


MODELS = {
    "ridge": lambda: _pipe(RidgeCV(alphas=np.logspace(-2, 4, 25))),
    "rf": lambda: _pipe(RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1)),
}


def song_level_eda() -> pd.DataFrame:
    """歌曲级皮电（被试内 z 分后取听众均值）—— 与 `74` 同口径。"""
    T = pd.read_parquet(FEAT / "pmemo_eda_song_level.parquet")
    for m in MEASURES:
        g = T.groupby("listener")[m]
        T[m] = (T[m] - g.transform("mean")) / (g.transform("std") + 1e-9)
    S = T.groupby("musicId")[MEASURES].mean()
    S["n_listeners"] = T.groupby("musicId").size()
    return S[S.n_listeners >= MIN_LISTENERS].reset_index()


def load_layer_a(tag: str) -> pd.DataFrame | None:
    p = FEAT / f"layer_a_reg_pmemo{tag}.parquet"
    if not p.exists():
        return None
    D = pd.read_parquet(p)
    D["musicId"] = D["clip_id"].astype(str).str.removeprefix("pmemo_").astype(int)
    return D


def load_compare() -> pd.DataFrame:
    """PMEmo 自带 ComParE 6374 维 —— 作者在**原始音频**上提取，与我们流水线独立。"""
    C = pd.read_csv(PM / "features" / "static_features.csv")
    C = C.rename(columns={C.columns[0]: "musicId"})
    C["musicId"] = pd.to_numeric(C["musicId"], errors="coerce")
    return C.dropna(subset=["musicId"]).astype({"musicId": int})


def evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
             rng) -> dict[str, tuple[float, float]]:
    """GroupKFold-5（按艺人），两个模型都报，各配置换 p。"""
    out = {}
    for name, mk in MODELS.items():
        def cv(yy):
            pred = np.full(len(yy), np.nan)
            for tr, te in GroupKFold(5).split(X, yy, groups):
                m = mk()
                m.fit(X[tr], yy[tr])
                pred[te] = m.predict(X[te])
            r = spearmanr(pred, yy).statistic
            return float(r) if np.isfinite(r) else 0.0
        obs = cv(y)
        null = np.array([cv(rng.permutation(y)) for _ in range(N_PERM)])
        p = float((1 + (null >= obs).sum()) / (1 + len(null)))
        out[name] = (obs, p)
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("87_loudness_control", SEED)
    S = song_level_eda()
    meta = pd.read_csv(PM / "metadata.csv")[["musicId", "artist"]]
    print(f"歌曲级皮电 {len(S)} 首\n")

    # ── 归一化到底拿走了多少 ─────────────────────────────────────────
    A_ln, A_raw = load_layer_a("_ln"), load_layer_a("")
    print("=" * 84)
    print("[1/3] 归一化拿走了多少跨曲响度方差")
    if A_raw is None:
        print("  ⚠️ 未归一化版尚未提取完成 —— 先跑")
        print("     .venv/Scripts/python scripts/02_extract_layer_a.py --dataset reg_pmemo")
        run.write()
        return
    for col in ("rms_mean", "rms_p90", "rms_std"):
        if col in A_ln.columns and col in A_raw.columns:
            a, b = A_ln[col].std(), A_raw[col].std()
            print(f"  {col:<12} 归一化后 SD {a:.4f}   原始 SD {b:.4f}   "
                  f"**丢失 {b/max(a,1e-12):.1f} 倍**")
            run.log_metric(f"sd_ratio_{col}", float(b / max(a, 1e-12)))

    # ── 三条路径 ─────────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("[2/3] 三条独立路径预测歌曲级皮电")
    print("      判据：若后两条阳性而第一条阴性，`74` 的结论作废\n")
    from soundml.regression import feature_cols
    a_cols = feature_cols(pd.read_parquet(FEAT / "layer_a_reg_deam_ln.parquet"))
    C = load_compare()
    c_cols = [c for c in C.columns if c != "musicId"
              and pd.api.types.is_numeric_dtype(C[c])]
    print(f"  Layer A {len(a_cols)} 维   ComParE {len(c_cols)} 维"
          f"   （统一 PCA→{N_PC} 维后比较，避免维度诅咒混入结论）")

    paths = {"A_ln (归一化, 现状)": (A_ln, a_cols),
             "A_raw (原始响度)": (A_raw, a_cols),
             "ComParE (原始, 作者提取)": (C, c_cols)}
    rows = []
    for label, (F, cols) in paths.items():
        M = S.merge(F, on="musicId", how="inner").merge(meta, on="musicId",
                                                        how="left")
        M["artist"] = M["artist"].fillna("u" + M.musicId.astype(str))
        leak = [c for c in cols if c in MEASURES + ["arousal", "valence"]]
        assert not leak, f"标签泄漏 {leak}"
        X = np.nan_to_num(M[cols].to_numpy(float), nan=0.0, posinf=0.0,
                          neginf=0.0)
        g = M["artist"].to_numpy()
        print(f"\n  ── {label}   {len(M)} 首 × {X.shape[1]} 维")
        for m in MEASURES:
            res = evaluate(X, M[m].to_numpy(float), g, rng)
            txt = "   ".join(f"{k} ρ={v[0]:+.3f} (p={v[1]:.3f})"
                             for k, v in res.items())
            print(f"     {m:<13} {txt}", flush=True)
            for k, (r, p) in res.items():
                rows.append({"path": label, "measure": m, "model": k,
                             "rho": r, "p": p, "n_feat": X.shape[1]})

    # ── 仅用响度一维 ─────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("[3/3] 仅用响度一维 —— 归一化前后的直接对照")
    for label, F, col in (("归一化", A_ln, "rms_mean"),
                          ("原始", A_raw, "rms_mean"),
                          ("ComParE 原始", C, "pcm_RMSenergy_sma_amean")):
        if col not in F.columns:
            alt = [c for c in F.columns if "RMSenergy" in c and "amean" in c]
            if not alt:
                print(f"  {label:<14} 找不到响度列，跳过")
                continue
            col = alt[0]
        M = S.merge(F[["musicId", col]], on="musicId", how="inner")
        for m in MEASURES:
            r = spearmanr(M[col], M[m]).statistic
            rows.append({"path": f"loudness_only_{label}", "measure": m,
                         "model": "univariate", "rho": float(r), "p": np.nan,
                         "n_feat": 1})
        rr = {m: spearmanr(M[col], M[m]).statistic for m in MEASURES}
        print(f"  {label:<14} {col:<28} " +
              "   ".join(f"{m}={rr[m]:+.3f}" for m in MEASURES))

    R = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT, index=False, encoding="utf-8")

    print("\n" + "=" * 84)
    best = R[R.model != "univariate"].groupby("path").rho.max()
    print("各路径最好的 ρ：")
    for k, v in best.items():
        print(f"  {k:<26} {v:+.3f}")
    ln = best.get("A_ln (归一化, 现状)", np.nan)
    raw = max(best.get("A_raw (原始响度)", -9),
              best.get("ComParE (原始, 作者提取)", -9))
    print(f"\n判定：原始响度路径最好 {raw:+.3f}   归一化路径 {ln:+.3f}")
    if raw - ln > 0.10:
        print("🔴 **归一化抹掉了关联** —— `74` 的结论必须改写。")
    elif raw - ln > 0.05:
        print("⚠️ 原始响度略优，须在正文报告并以原始路径为准。")
    else:
        print("✅ 两者相当 —— 归一化不是阴性的解释，`74` 的结论成立且加强。")
    run.log_metric("best_rho_normalised", float(ln))
    run.log_metric("best_rho_raw", float(raw))
    run.write()
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
