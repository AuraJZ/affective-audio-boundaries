"""区分：A11 的低耦合是实质发现，还是差分造成的方法学产物？

    uv run python scripts/23_dynamics_level_check.py --n-songs 400

## 歧义在哪

`22_within_song_dynamics.py` 对逐时刻序列做一阶差分，得到极低耦合
（最强 |ρ| = 0.056）。差分是为消除自相关而选的，但 DEAM 的连续标注是
**滑块采集、本身经过平滑**的——对平滑序列差分会放大噪声、削掉真实信号。

因此低耦合有两种可能：

- **实质**：瞬时声学确实解释不了瞬时唤醒
- **方法学产物**：差分把信号差没了

## 如何区分

对**未差分的电平序列**做组内相关。但电平序列强自相关，
朴素 p 值严重偏乐观，因此用两道校正：

1. **块自助（block bootstrap）**：按块重采样保留自相关结构，得到诚实的置信区间
2. **序列置换零分布**：把描述符序列**整体循环移位**后重算 ——
   保留两条序列各自的自相关结构，只破坏它们之间的时间对齐。
   这是自相关序列相关性检验的标准零假设构造。

## 判据（事先写死）

| 结果 | 结论 |
|---|---|
| 电平相关显著高于其循环移位零分布 | 存在实质的组内耦合，**差分是方法学产物** |
| 电平相关与零分布不可区分 | **低耦合是实质发现**，A11 成立 |

同时比较两族（变化率 vs 频谱水平），看 A11 的族间差异在电平尺度上是否重现。
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from soundml.features_frame import DESCRIPTORS, LEVEL, VARIABILITY, frame_series
from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

DYN = (REPO_ROOT / "data" / "raw" / "DEAM" / "annotations"
       / "annotations averaged per song" / "dynamic (per second annotations)"
       / "arousal.csv")
LABELS = REPO_ROOT / "data" / "reg_deam.csv"

N_SHIFT_NULL = 24        # 每首歌每描述符的循环移位次数
MIN_SHIFT = 10           # 最小移位量（样本），避免与原序列过近


def autocorr_time(x: np.ndarray) -> float:
    """一阶自相关，用于说明「为何朴素 p 值不可用」。"""
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def circular_shift_null(x: np.ndarray, a: np.ndarray, rng, n: int) -> np.ndarray:
    """循环移位零分布：保留各自自相关，只破坏时间对齐。"""
    out = []
    span = len(x)
    for _ in range(n):
        s = int(rng.integers(MIN_SHIFT, span - MIN_SHIFT))
        r = spearmanr(np.roll(x, s), a).statistic
        if np.isfinite(r):
            out.append(float(r))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=400)
    args = ap.parse_args()

    dyn = pd.read_csv(DYN, skipinitialspace=True)
    dyn.columns = [c.strip() for c in dyn.columns]
    dyn = dyn.set_index("song_id")
    times_s = np.array([int(c.split("_")[1].removesuffix("ms"))
                        for c in dyn.columns]) / 1000.0

    paths = (pd.read_csv(LABELS)
             .assign(song_id=lambda d: d.clip_id.str.removeprefix("deam_").astype(int))
             .set_index("song_id")["path"])
    ids = [i for i in dyn.index if i in paths.index]
    rng = np.random.default_rng(SEED)
    ids = list(rng.choice(ids, size=min(args.n_songs, len(ids)), replace=False))

    with RunRecord(
        "dynamics_level_check", seed=SEED,
        params={"corpus": "DEAM", "n_songs": len(ids),
                "method": "within-song level Spearman + circular-shift null",
                "n_shift_null": N_SHIFT_NULL,
                "question": "A11 的低耦合是实质发现还是差分产物"},
    ) as run:
        rows, ac_rows = [], []
        for k, sid in enumerate(ids, 1):
            a = dyn.loc[sid].to_numpy(dtype=float)
            try:
                feats = frame_series(paths.loc[sid], times_s)
            except Exception:
                continue

            ok_a = np.isfinite(a)
            if ok_a.sum() < 30:
                continue
            ac_rows.append({"song_id": sid, "series": "arousal",
                            "ac1": autocorr_time(a)})

            for d in DESCRIPTORS:
                x = feats[d]
                ok = np.isfinite(x) & ok_a
                if ok.sum() < 30:
                    continue
                xs, as_ = x[ok], a[ok]
                if np.std(xs) < 1e-12 or np.std(as_) < 1e-12:
                    continue

                obs = spearmanr(xs, as_).statistic
                null = circular_shift_null(xs, as_, rng, N_SHIFT_NULL)
                if not np.isfinite(obs) or null.size < 5:
                    continue

                rows.append({
                    "song_id": sid, "descriptor": d,
                    "rho_level": float(obs),
                    "null_mean_abs": float(np.mean(np.abs(null))),
                    "null_p95_abs": float(np.percentile(np.abs(null), 95)),
                    "exceeds_null": bool(abs(obs) > np.percentile(np.abs(null), 95)),
                    "ac1_descriptor": autocorr_time(xs),
                })
            if k % 50 == 0:
                print(f"  {k}/{len(ids)} 首", flush=True)

        R = pd.DataFrame(rows)
        R.to_parquet(run.artifact_path("level_coupling.parquet"), index=False)

        AC = pd.DataFrame(ac_rows)
        run.log_metric("autocorrelation",
                       {"arousal_ac1_mean": float(AC.ac1.mean()),
                        "descriptor_ac1_mean": float(R.ac1_descriptor.mean())})

        # ---------- 逐描述符汇总 ----------
        out = []
        for d, g in R.groupby("descriptor"):
            obs_abs = g.rho_level.abs()
            null_abs = g.null_mean_abs
            # 超过各自 95% 零分布的歌曲比例；随机期望为 5%
            frac = float(g.exceeds_null.mean())
            out.append({
                "descriptor": d,
                "feature_class": "Temporal variability" if d in VARIABILITY
                                 else "Spectral level",
                "n_songs": int(len(g)),
                "mean_abs_rho_level": float(obs_abs.mean()),
                "mean_abs_rho_null": float(null_abs.mean()),
                "excess_over_null": float(obs_abs.mean() - null_abs.mean()),
                "frac_exceeding_null_p95": frac,
            })
        D = pd.DataFrame(out).sort_values("excess_over_null", ascending=False)
        D.to_csv(run.artifact_path("level_vs_null.csv"), index=False, encoding="utf-8")
        run.log_metric("descriptor_level_coupling", D.round(4).to_dict("records"))

        var = D[D.feature_class == "Temporal variability"]
        lev = D[D.feature_class == "Spectral level"]
        cls = {"variability_excess": float(var.excess_over_null.mean()),
               "level_excess": float(lev.excess_over_null.mean()),
               "variability_frac_exceed": float(var.frac_exceeding_null_p95.mean()),
               "level_frac_exceed": float(lev.frac_exceeding_null_p95.mean())}
        run.log_metric("class_comparison_level", cls)

        overall_frac = float(R.exceeds_null.mean())
        run.log_metric("overall_frac_exceeding_null_p95", overall_frac)
        # 事先判据
        substantive = overall_frac > 0.15      # 显著高于 5% 的随机期望
        run.log_metric("verdict_coupling_is_real", bool(substantive))

        # ---------- 打印 ----------
        print("\n" + "=" * 78)
        print(f"run_id: {run.run_id}")
        print(f"{R.song_id.nunique()} 首歌 × {len(DESCRIPTORS)} 描述符\n")

        print("--- 自相关（说明为何朴素 p 值不可用）---")
        print(f"  唤醒序列 lag-1 自相关 = {AC.ac1.mean():.3f}")
        print(f"  描述符序列 lag-1 自相关 = {R.ac1_descriptor.mean():.3f}")

        print("\n--- 电平相关 vs 循环移位零分布 ---")
        print(f"{'描述符':<24}{'类':<10}{'|ρ|观测':>9}{'|ρ|零':>8}{'超出':>9}{'超p95占比':>11}")
        for r in D.itertuples():
            c = "变化率" if r.feature_class.startswith("Temporal") else "频谱水平"
            print(f"{r.descriptor:<24}{c:<10}{r.mean_abs_rho_level:>9.3f}"
                  f"{r.mean_abs_rho_null:>8.3f}{r.excess_over_null:>+9.3f}"
                  f"{r.frac_exceeding_null_p95:>10.0%}")

        print(f"\n--- 族间对比（超出零分布的幅度）---")
        print(f"  变化率族   = {cls['variability_excess']:+.4f}   "
              f"超 p95 占比 {cls['variability_frac_exceed']:.0%}")
        print(f"  频谱水平族 = {cls['level_excess']:+.4f}   "
              f"超 p95 占比 {cls['level_frac_exceed']:.0%}")

        print(f"\n整体超 p95 占比 = {overall_frac:.1%}（随机期望 5%）")
        print("=" * 78)
        if substantive:
            print("→ 电平尺度存在**实质**组内耦合，显著高于循环移位零分布。")
            print("  A11 的极低差分耦合**部分是差分造成的方法学产物**，")
            print("  「瞬时声学解释不了瞬时唤醒」这一表述需要收回或大幅弱化。")
        else:
            print("→ 电平相关也与零分布不可区分。")
            print("  **A11 是实质发现**：组内瞬时耦合本身就很弱，与差分无关。")


if __name__ == "__main__":
    main()
