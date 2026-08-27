"""PLAN III-2：反事实音频干预 —— 把规则从「关联」升级到「因果」。

    uv run python scripts/10_counterfactual.py --n-clips 40

## 设计

对每段源音频，在三种干预 × 四个剂量（0 = 对照）下重新提特征、重新预测。
在**两个域的模型上分别预测**：若同一干预在音乐模型与环境声模型上
推动预测的方向一致，则该规则的域无关性获得因果层面的支持。

## 三项必报指标（缺一即为过度主张）

1. **操纵检查** —— 目标特征是否随剂量单调移动（Spearman ρ）。
   没动 = 干预无效，后续一切无意义。
2. **特异性** —— 目标特征标准化位移 ÷ 其他特征最大标准化位移。
   比值低 = 干预不特异，预测变化不能归因于目标特征。
   标准化用训练集各特征的标准差。
3. **剂量–反应** —— 预测概率是否随剂量单调变化，方向是否与规则一致。

只报第 3 项而不报 1、2 项，是这类实验最常见的过度主张。
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer

from soundml import features_a
from soundml.data import load_joined
from soundml.interventions import DOSES, INTERVENTIONS
from soundml.modeling import SEED, build_models
from soundml.provenance import RunRecord

# 域无关核心特征（见 reports/step6_domain_invariant_rules.md §1）
INVARIANT = ["onset_env_dmean", "spec_contrast3_dmean", "onset_rate", "spec_flatness_std"]


def fit_domain_model(dataset: str):
    """返回 (模型, 特征列, 各特征标准差, 中位数填充器)。"""
    df, reps = load_joined(dataset)
    cols = reps["A"]  # 干预只改 Layer A 可测的量
    imp = SimpleImputer(strategy="median").fit(df[cols].to_numpy(dtype=np.float64))
    X = imp.transform(df[cols].to_numpy(dtype=np.float64))
    y = df["label"].to_numpy()
    model = build_models()["RF"].fit(X, y)
    return model, cols, X.std(axis=0), imp, df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=40, help="每个域抽多少段源音频")
    args = ap.parse_args()

    domains = {}
    for ds in ("deam", "emo"):
        model, cols, sd, imp, df = fit_domain_model(ds)
        domains[ds] = {"model": model, "cols": cols, "sd": sd, "imp": imp, "df": df}

    # 两域特征列一致（都是 Layer A 122 维）
    cols = domains["deam"]["cols"]
    assert cols == domains["emo"]["cols"], "两域特征列不一致，无法交叉预测"

    rng = np.random.default_rng(SEED)
    sources = []
    for ds in ("deam", "emo"):
        d = domains[ds]["df"]
        pick = rng.choice(len(d), size=min(args.n_clips, len(d)), replace=False)
        for i in pick:
            sources.append({"domain": ds, "clip_id": d.iloc[i]["clip_id"],
                            "path": d.iloc[i]["path"], "label": int(d.iloc[i]["label"])})

    with RunRecord(
        "counterfactual_intervention", seed=SEED,
        params={"n_sources": len(sources), "doses": DOSES,
                "interventions": {k: v[1] for k, v in INTERVENTIONS.items()},
                "invariant_features": INVARIANT,
                "models": "RF fit on each domain's full data, Layer A only"},
    ) as run:
        records = []
        for si, src in enumerate(sources, start=1):
            y0, sr = features_a.load_prepared(src["path"], normalize_loudness=True)
            for iname, (fn, targets) in INTERVENTIONS.items():
                for dose in DOSES:
                    y = fn(y0, dose, sr) if dose else y0
                    feats = features_a.extract_array(y, sr)
                    row = {"source": src["clip_id"], "src_domain": src["domain"],
                           "label": src["label"], "intervention": iname, "dose": dose}
                    row |= {f"f_{k}": v for k, v in feats.items()}
                    vec = np.array([[feats.get(c, np.nan) for c in cols]], dtype=np.float64)
                    for ds in ("deam", "emo"):
                        d = domains[ds]
                        row[f"p_{ds}"] = float(
                            d["model"].predict_proba(d["imp"].transform(vec))[0, 1])
                    records.append(row)
            if si % 10 == 0:
                print(f"  {si}/{len(sources)} 段", flush=True)

        R = pd.DataFrame(records)
        R.to_parquet(run.artifact_path("counterfactual_raw.parquet"), index=False)

        # ---------- 三项指标 ----------
        # ⚠️ 全部**在每段源音频内部**计算再跨源汇总。
        # 池化计算会被组间方差淹没：不同源音频的基线特征值差异远大于剂量效应。
        def within_source_rho(sub: pd.DataFrame, col: str) -> tuple[float, float]:
            """返回 (组内 ρ 的中位数, 符号与中位数一致的源占比)。"""
            rhos = []
            for _, g in sub.groupby("source"):
                if g[col].nunique() > 1:
                    r = spearmanr(g["dose"], g[col]).statistic
                    if np.isfinite(r):
                        rhos.append(float(r))
            if not rhos:
                return np.nan, np.nan
            med = float(np.median(rhos))
            consist = float(np.mean(np.sign(rhos) == np.sign(med))) if med else np.nan
            return med, consist

        sd = pd.Series(domains["deam"]["sd"], index=cols)
        summary, diagnostics = [], []
        for iname, (_, targets) in INTERVENTIONS.items():
            sub = R[R["intervention"] == iname]

            # 特异性用的位移：每段源音频 0→最大剂量的标准化位移，跨源取均值
            base = sub[sub.dose == 0].set_index("source")
            top = sub[sub.dose == max(DOSES)].set_index("source")
            common = base.index.intersection(top.index)
            shifts = {}
            for c in cols:
                fc = f"f_{c}"
                if fc in sub and sd.get(c, 0) > 0:
                    per_src = (top.loc[common, fc] - base.loc[common, fc]).abs() / sd[c]
                    shifts[c] = float(per_src.mean())

            top5 = sorted(shifts.items(), key=lambda kv: kv[1], reverse=True)[:5]
            diagnostics.append({"intervention": iname,
                                "top5_moved": [{"feature": k, "shift_sd": round(v, 3)}
                                               for k, v in top5]})

            dr = {ds: within_source_rho(sub, f"p_{ds}") for ds in ("deam", "emo")}

            for tgt in targets:
                col = f"f_{tgt}"
                if col not in sub:
                    continue
                rho_t, consist_t = within_source_rho(sub, col)
                tgt_shift = shifts.get(tgt, np.nan)
                others = {k: v for k, v in shifts.items() if k != tgt}
                max_other = max(others.values()) if others else np.nan
                rank = int(sum(v > tgt_shift for v in others.values()) + 1)

                row = {"intervention": iname, "target": tgt,
                       "manip_rho": rho_t, "manip_consistency": consist_t,
                       "target_shift_sd": tgt_shift,
                       "max_other_shift_sd": max_other,
                       "specificity": float(tgt_shift / max_other) if max_other else np.nan,
                       "shift_rank_among_122": rank}
                for ds in ("deam", "emo"):
                    row[f"dose_response_rho_{ds}"], row[f"dr_consistency_{ds}"] = dr[ds]
                summary.append(row)

        run.log_metric("top_moved_features", diagnostics)

        S = pd.DataFrame(summary).drop_duplicates(subset=["intervention", "target"])
        S.to_csv(run.artifact_path("intervention_summary.csv"), index=False, encoding="utf-8")
        run.log_metric("summary", S.round(4).to_dict("records"))

        # 各干预的剂量-预测均值曲线
        curves = (R.groupby(["intervention", "dose"])[["p_deam", "p_emo"]]
                  .mean().round(4).reset_index())
        run.log_metric("dose_response_curves", curves.to_dict("records"))

        print("\n" + "=" * 78)
        print(f"run_id: {run.run_id}   源音频 {len(sources)} 段 × "
              f"{len(INTERVENTIONS)} 干预 × {len(DOSES)} 剂量")
        print("\n--- 剂量 → 平均预测概率（越高 = 越判为助眠）---")
        print(curves.to_string(index=False))
        print("\n--- 各干预实际推动最多的特征（诊断）---")
        for d in diagnostics:
            items = "  ".join(f"{x['feature']}={x['shift_sd']}" for x in d["top5_moved"])
            print(f"  {d['intervention']:<18} {items}")

        print("\n--- 三项指标（均为组内计算后跨源汇总）---")
        print(S[["intervention", "target", "manip_rho", "manip_consistency",
                 "target_shift_sd", "max_other_shift_sd", "specificity",
                 "shift_rank_among_122", "dose_response_rho_deam",
                 "dose_response_rho_emo"]].round(3).to_string(index=False))
        print("\n说明：manip_rho 接近 ±1 且 consistency 高 = 干预确实稳定推动了目标特征；")
        print("      specificity > 1 且 shift_rank = 1 = 目标特征是被推动最多的；")
        print("      两个 dose_response_rho 同号 = 因果效应跨域一致。")


if __name__ == "__main__":
    main()
