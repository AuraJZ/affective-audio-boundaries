"""合成网格 + 四道闸 —— 受控刺激的剂量–反应，以及对抗样本筛查。

    uv run python scripts/24_synthesis_grid.py

## 这一步同时做两件事

**1. 修掉 Fig. 4 的特异性局限。**
修改真实音频时目标描述符从未是被推动最多的那个（物理上躲不开）。
**从参数合成时，目标量是按构造控制的** —— 每一格的参数是已知真值，
而非事后测量。这给出干净的剂量–反应曲面。

**2. 闭环生成的可行性检验。**
把网格当作「生成器」的最简形式，跑一遍四道闸，看：

- 有多少候选能全过 → 闸是否过严
- 各闸分别拒掉多少 → 哪一道最吃紧
- **oracle 给高分但被闸拒掉的候选** → 这就是对抗样本，
  也是「不设闸会发生什么」的直接证据

## 网格

三个受控轴（见 `synthesis.py`）：

    slope       0.0 … 2.0   谱倾斜（白噪→粉噪→布朗噪）
    onset_rate  0 … 6 /s    突发密度
    flux        0 … 1.5 Hz  音色变化速率

每格多个随机种子，避免单次实现的偶然性。
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from soundml import features_a, synthesis
from soundml.closed_loop import LoopGuards, summarise
from soundml.data import load_joined
from soundml.modeling import fit_ad
from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED, build_regressors, feature_cols

SLOPES = [0.0, 0.5, 1.0, 1.5, 2.0]
ONSETS = [0.0, 1.0, 2.0, 4.0, 6.0]
FLUXES = [0.0, 0.25, 0.75, 1.5]
SEEDS = [0, 1, 2]

DOMAINS = {"music": "reg_deam", "ambient": "reg_emo_mix"}
N_RASHOMON = 8


def fit_domain_side(tag: str, cols: list[str]):
    """返回该域的 (主模型, 近优模型集, 训练特征矩阵, 训练标签)。"""
    df, _ = load_joined(tag)
    X = SimpleImputer(strategy="median").fit_transform(
        df[cols].to_numpy(dtype=np.float64))
    y = df["arousal"].to_numpy()
    main = build_regressors()["XGBoost"].fit(X, y)

    rng = np.random.default_rng(SEED)
    rash = []
    for i in range(N_RASHOMON):
        m = build_regressors()["RF" if i % 2 else "XGBoost"]
        est = m.named_steps["m"]
        if hasattr(est, "max_depth"):
            est.set_params(max_depth=int(rng.choice([4, 6, 8, 12])))
        est.set_params(random_state=int(rng.integers(0, 9999)))
        rash.append(m.fit(X, y))
    return main, rash, X, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=10.0)
    args = ap.parse_args()

    # 特征列以音乐域为准（两域同为 Layer A 122 维）
    df_ref, reps = load_joined(DOMAINS["music"])
    cols = [c for c in feature_cols(df_ref) if c in reps["A"]]

    domain_main, domain_rash, Xtr, ytr = {}, [], None, None
    for dom, tag in DOMAINS.items():
        m, rash, X, y = fit_domain_side(tag, cols)
        domain_main[dom] = m
        domain_rash.extend(rash)
        if dom == "ambient":          # 适用域以环境声域为基准：产品实际内容
            Xtr, ytr = X, y
        print(f"  {dom} 域模型完成", flush=True)

    scaler = Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("sc", StandardScaler())]).fit(Xtr)
    ad = fit_ad(scaler.transform(Xtr))

    # 参考分布：真实环境声语料在 oracle 下的得分（保守取两域较高者）
    null_scores = np.maximum(domain_main["music"].predict(Xtr),
                             domain_main["ambient"].predict(Xtr))

    guards = LoopGuards(domain_main, domain_rash, ad, scaler, null_scores)

    grid = list(itertools.product(SLOPES, ONSETS, FLUXES, SEEDS))

    with RunRecord(
        "synthesis_grid_closed_loop", seed=SEED,
        params={"grid": {"slope": SLOPES, "onset_rate": ONSETS,
                         "flux_hz": FLUXES, "seeds": SEEDS},
                "n_candidates": len(grid), "duration_s": args.duration,
                "domains": DOMAINS, "n_rashomon": len(domain_rash),
                "low_arousal_cut": guards.low_arousal_cut},
    ) as run:
        rows, verdicts = [], []
        for k, (sl, on, fl, sd) in enumerate(grid, 1):
            y = synthesis.synthesize(sl, on, fl, seed=sd,
                                     duration_s=args.duration)
            feats = features_a.extract_array(y, features_a.SR)
            x = np.array([[feats.get(c, np.nan) for c in cols]], dtype=np.float64)

            p = {"slope": sl, "onset_rate": on, "flux_hz": fl, "seed": sd}
            v = guards.evaluate(x, p)
            verdicts.append(v)

            row = dict(p)
            row["oracle_score"] = v.oracle_score
            row["accepted"] = v.accepted
            for g in v.gates:
                row[f"gate_{g.name}"] = g.passed
                row[f"val_{g.name}"] = g.value
            # 操纵检查：合成参数是否真的落在测得的描述符上。
            # flux 用 spec_centroid_dmean 而非 spec_contrast3_dmean —— 时变谱倾斜
            # 直接移动的是频谱重心；早期用谱对比度衡量，ρ 仅 0.23，是测量对象选错。
            row["measured_onset_rate"] = feats.get("onset_rate", np.nan)
            row["measured_slope_1f"] = feats.get("slope_1f", np.nan)
            row["measured_centroid_dmean"] = feats.get("spec_centroid_dmean", np.nan)
            row["measured_contrast3_dmean"] = feats.get("spec_contrast3_dmean", np.nan)
            row["measured_flatness_std"] = feats.get("spec_flatness_std", np.nan)
            rows.append(row)

            if k % 50 == 0:
                print(f"  {k}/{len(grid)}", flush=True)

        R = pd.DataFrame(rows)
        R.to_csv(run.artifact_path("synthesis_grid.csv"), index=False, encoding="utf-8")

        summ = summarise(verdicts)
        run.log_metric("gate_summary", summ)

        # ---------- 操纵检查：参数 → 实测描述符 ----------
        from scipy.stats import spearmanr
        manip = {
            "slope->slope_1f": float(spearmanr(R.slope, R.measured_slope_1f).statistic),
            "onset->onset_rate": float(spearmanr(R.onset_rate, R.measured_onset_rate).statistic),
            "flux->centroid_dmean": float(spearmanr(R.flux_hz, R.measured_centroid_dmean).statistic),
            "flux->flatness_std": float(spearmanr(R.flux_hz, R.measured_flatness_std).statistic),
        }
        # 串扰：非目标轴对该描述符的影响，应远小于目标轴
        crosstalk = {
            "slope->onset_rate": float(spearmanr(R.slope, R.measured_onset_rate).statistic),
            "onset->slope_1f": float(spearmanr(R.onset_rate, R.measured_slope_1f).statistic),
            "onset->centroid_dmean": float(spearmanr(R.onset_rate, R.measured_centroid_dmean).statistic),
            "flux->slope_1f": float(spearmanr(R.flux_hz, R.measured_slope_1f).statistic),
        }
        run.log_metric("manipulation_check", manip)
        run.log_metric("crosstalk", crosstalk)

        # ---------- 剂量–反应：每个轴的边际效应 ----------
        dose = {}
        for axis in ("slope", "onset_rate", "flux_hz"):
            g = R.groupby(axis)["oracle_score"].mean()
            dose[axis] = {"means": g.round(4).to_dict(),
                          "spearman": float(spearmanr(R[axis], R.oracle_score).statistic)}
        run.log_metric("dose_response", dose)

        # ---------- 对抗样本：oracle 高分但被闸拒 ----------
        adv = R[(R.oracle_score <= R.oracle_score.quantile(0.25)) & (~R.accepted)]
        run.log_metric("adversarial_candidates",
                       {"n": int(len(adv)),
                        "frac_of_top_quartile": float(len(adv) / max(len(R) // 4, 1)),
                        "examples": adv.head(5)[
                            ["slope", "onset_rate", "flux_hz", "oracle_score"]
                        ].round(3).to_dict("records")})

        # ---------- 打印 ----------
        print("\n" + "=" * 72)
        print(f"run_id: {run.run_id}   候选 {len(R)} 个")

        print("\n--- 操纵检查（合成参数 → 实测描述符）---")
        for k, v in manip.items():
            print(f"  {k:<26} ρ = {v:+.3f}")
        print("--- 串扰（应远小于上表）---")
        for k, v in crosstalk.items():
            print(f"  {k:<26} ρ = {v:+.3f}")

        print("\n--- 剂量–反应（oracle 预测唤醒，越低越助眠）---")
        for axis, d in dose.items():
            vals = "  ".join(f"{k}:{v:+.3f}" for k, v in d["means"].items())
            print(f"  {axis:<12} ρ={d['spearman']:+.3f}   {vals}")

        print("\n--- 四道闸 ---")
        print(f"  候选 {summ['n_candidates']}   通过 {summ['n_accepted']}   "
              f"通过率 {summ['accept_rate']:.1%}")
        for g, r in summ["pass_rate_per_gate"].items():
            print(f"    {g:<24} 通过率 {r:.1%}")

        a = run.meta["metrics"]["adversarial_candidates"]
        print(f"\n--- 对抗样本（oracle 判为最助眠的四分位中被闸拒的）---")
        print(f"  {a['n']} 个，占该四分位的 {a['frac_of_top_quartile']:.1%}")
        if a["n"]:
            print("  若不设闸，这些会被当作「发现」直接进候选库：")
            for e in a["examples"]:
                print(f"    slope={e['slope']} onset={e['onset_rate']} "
                      f"flux={e['flux_hz']} → score={e['oracle_score']:+.3f}")


if __name__ == "__main__":
    main()
