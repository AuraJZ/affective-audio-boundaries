"""组内动态检验 —— 在主张所指的时间尺度上检验主张。

    uv run python scripts/22_within_song_dynamics.py --n-songs 400

## 检验什么

我们的核心主张是「时间变化率决定唤醒」。此前只用片段级聚合量验证过。
DEAM 的逐 0.5 s 标注允许在**同一首歌内部**直接问：

    描述符的瞬时变化 → 唤醒的瞬时变化？

## 三个设计要点

**1. 一阶差分而非电平相关。**
两条序列都强自相关，直接相关会给出严重偏乐观的 p 值。
差分同时解决两件事：消掉大部分自相关，且 Δ–Δ 关系**正是主张本身**。

**2. 组内设计。**
跨曲比较把唤醒与曲风、艺人、制作混在一起；组内比较时同一段录音自己做对照。

**3. 标注延迟扫描。**
连续标注有反应时延迟（文献通常 1–3 s）。不扫描延迟就等于假设延迟为零，
会系统性低估耦合强度。

## 可证伪的预测

若主张成立：**变化率族的 Δ–Δ 耦合应强于频谱水平族**。
若两族无差别，则「变化率特殊」在此时间尺度上不成立 —— 这会削弱主结论。
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from soundml.features_frame import DESCRIPTORS, LEVEL, VARIABILITY, frame_series
from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

DYN = (REPO_ROOT / "data" / "raw" / "DEAM" / "annotations"
       / "annotations averaged per song" / "dynamic (per second annotations)"
       / "arousal.csv")
LABELS = REPO_ROOT / "data" / "reg_deam.csv"
# 扫到 6 s：试跑时最强耦合落在 3.0 s，即当时扫描范围的边界，
# 说明真实反应延迟可能更长。边界值不能当作估计。
LAGS_S = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]


def load_dynamic() -> pd.DataFrame:
    d = pd.read_csv(DYN, skipinitialspace=True)
    d.columns = [c.strip() for c in d.columns]
    return d.set_index("song_id")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=400)
    args = ap.parse_args()

    dyn = load_dynamic()
    paths = (pd.read_csv(LABELS)
             .assign(song_id=lambda d: d.clip_id.str.removeprefix("deam_").astype(int))
             .set_index("song_id")["path"])
    ids = [i for i in dyn.index if i in paths.index]
    rng = np.random.default_rng(SEED)
    ids = list(rng.choice(ids, size=min(args.n_songs, len(ids)), replace=False))

    # 标注时刻（秒），列名形如 sample_15000ms
    times_ms = np.array([int(c.split("_")[1].removesuffix("ms")) for c in dyn.columns])
    times_s = times_ms / 1000.0

    with RunRecord(
        "within_song_dynamics", seed=SEED,
        params={"corpus": "DEAM", "n_songs": len(ids),
                "annotation_rate_hz": 2.0, "lags_s": LAGS_S,
                "method": "within-song first-difference Spearman, lag-scanned",
                "descriptors": {"variability": VARIABILITY, "level": LEVEL}},
    ) as run:
        # rows: 每首歌 × 每个描述符 × 每个延迟
        rows = []
        for k, sid in enumerate(ids, 1):
            a_full = dyn.loc[sid].to_numpy(dtype=float)
            try:
                feats = frame_series(paths.loc[sid], times_s)
            except Exception:
                continue

            for lag in LAGS_S:
                shift = int(round(lag * 2))          # 标注为 2 Hz
                if shift >= len(a_full) - 10:
                    continue
                # 唤醒滞后于声学：把唤醒序列前移 shift 个样本来对齐
                a = a_full[shift:]
                for d in DESCRIPTORS:
                    x = feats[d][: len(a_full) - shift]
                    ok = np.isfinite(x) & np.isfinite(a)
                    if ok.sum() < 20:
                        continue
                    dx, da = np.diff(x[ok]), np.diff(a[ok])
                    if np.std(dx) < 1e-12 or np.std(da) < 1e-12:
                        continue
                    r = spearmanr(dx, da).statistic
                    if np.isfinite(r):
                        rows.append({"song_id": sid, "descriptor": d, "lag_s": lag,
                                     "rho_delta": float(r), "n": int(ok.sum())})
            if k % 50 == 0:
                print(f"  {k}/{len(ids)} 首", flush=True)

        R = pd.DataFrame(rows)
        R.to_parquet(run.artifact_path("within_song_delta.parquet"), index=False)
        run.log_metric("n_songs_used", int(R.song_id.nunique()))

        # ---------- 延迟扫描：哪个延迟耦合最强 ----------
        by_lag = (R.groupby("lag_s")["rho_delta"]
                  .apply(lambda s: float(np.mean(np.abs(s)))).reset_index(name="mean_abs_rho"))
        best_lag = float(by_lag.loc[by_lag.mean_abs_rho.idxmax(), "lag_s"])
        run.log_metric("lag_scan", by_lag.round(4).to_dict("records"))
        run.log_metric("best_lag_s", best_lag)

        B = R[R.lag_s == best_lag]

        # ---------- 逐描述符：组内 Δ–Δ 耦合 ----------
        out = []
        for d, g in B.groupby("descriptor"):
            v = g["rho_delta"].to_numpy()
            st = wilcoxon(v) if len(v) > 10 else None
            out.append({"descriptor": d,
                        "feature_class": "Temporal variability" if d in VARIABILITY
                                         else "Spectral level",
                        "n_songs": int(len(v)),
                        "mean_rho": float(v.mean()),
                        "median_rho": float(np.median(v)),
                        "frac_same_sign": float((np.sign(v) == np.sign(v.mean())).mean()),
                        "p_value": float(st.pvalue) if st else None})
        D = pd.DataFrame(out).sort_values("mean_rho", key=abs, ascending=False)
        D.to_csv(run.artifact_path("descriptor_delta_coupling.csv"),
                 index=False, encoding="utf-8")
        run.log_metric("descriptor_coupling", D.round(4).to_dict("records"))

        # ---------- 关键对比：两族的耦合强度 ----------
        var_s = D[D.feature_class == "Temporal variability"]["mean_rho"].abs()
        lev_s = D[D.feature_class == "Spectral level"]["mean_rho"].abs()
        run.log_metric("class_comparison",
                       {"variability_mean_abs_rho": float(var_s.mean()),
                        "level_mean_abs_rho": float(lev_s.mean()),
                        "ratio": float(var_s.mean() / lev_s.mean()) if lev_s.mean() else None,
                        "prediction_supported": bool(var_s.mean() > lev_s.mean())})

        # ---------- 打印 ----------
        print("\n" + "=" * 76)
        print(f"run_id: {run.run_id}")
        print(f"{R.song_id.nunique()} 首歌 × {len(DESCRIPTORS)} 描述符 × "
              f"{len(LAGS_S)} 个延迟\n")

        print("--- 标注延迟扫描（组内 |Δρ| 均值）---")
        for r in by_lag.itertuples():
            mark = "  ←最强" if r.lag_s == best_lag else ""
            print(f"  lag {r.lag_s:.1f} s : {r.mean_abs_rho:.4f}{mark}")

        print(f"\n--- 逐描述符 Δ–Δ 耦合（lag = {best_lag:.1f} s）---")
        print(f"{'描述符':<24}{'类':<14}{'均值ρ':>9}{'同号率':>9}{'p':>11}")
        for r in D.itertuples():
            cls = "变化率" if r.feature_class.startswith("Temporal") else "频谱水平"
            p = f"{r.p_value:.2e}" if r.p_value is not None else "—"
            print(f"{r.descriptor:<24}{cls:<14}{r.mean_rho:>+9.4f}"
                  f"{r.frac_same_sign:>8.0%}{p:>11}")

        cc = run.meta["metrics"]["class_comparison"]
        print(f"\n--- 关键对比 ---")
        print(f"  变化率族 |ρ| 均值   = {cc['variability_mean_abs_rho']:.4f}")
        print(f"  频谱水平族 |ρ| 均值 = {cc['level_mean_abs_rho']:.4f}")
        if cc["ratio"]:
            print(f"  比值 = {cc['ratio']:.2f}×")
        print("\n" + "=" * 76)
        if cc["prediction_supported"]:
            print("→ 预测成立：变化率族在逐时刻尺度上的耦合强于频谱水平族。")
            print("  主结论在其所指的时间尺度上得到独立支持。")
        else:
            print("→ 预测**不成立**：两族在逐时刻尺度上无差别，甚至相反。")
            print("  片段级发现可能是聚合产物，主结论需要重新表述。")


if __name__ == "__main__":
    main()
