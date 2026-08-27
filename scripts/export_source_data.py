"""把各 run 的结果导出为 tidy CSV，供 R 绘图使用。

    uv run python scripts/export_source_data.py

⚠️ 本脚本**不产生任何图形输出**，不导入任何绘图库。
按 nature-figure 的 R-only 执行规则，Python 在此仅承担非可视化的数据转换。

产出的 CSV 同时充当论文的 **Source Data**：每张图一份，可随稿投递。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from soundml.provenance import REPO_ROOT, RUNS_DIR

OUT = REPO_ROOT / "reports" / "source_data"
OUT.mkdir(parents=True, exist_ok=True)

RATE_SUFFIX = ("_std", "_dmean")


def is_rate(f: str) -> bool:
    """是否为「变化率 / 变异」类特征。"""
    return f.endswith(RATE_SUFFIX) or f == "onset_rate"


def meta(rid: str) -> dict:
    return json.loads((RUNS_DIR / rid / "meta.json").read_text(encoding="utf-8"))


def latest(suffix: str, n: int = 1):
    ids = sorted(d.name for d in RUNS_DIR.glob(f"*{suffix}") if (d / "meta.json").exists())
    return ids[-1] if n == 1 else ids


def write(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8")
    print(f"  {name}.csv  ({len(df)} rows)")


# ---------------------------------------------------------------- 1. domain wall
def export_domain_wall() -> None:
    m = meta(latest("_regression_matrix"))["metrics"]

    rows = []
    for k, v in m["within_source"].items():
        algo, ds = k.split("|")
        rows.append({"algorithm": algo, "source": ds, "target": ds,
                     "transfer": "within", "domain_match": "within-corpus",
                     "spearman": v["spearman"], "r2": v["r2"]})
    for k, v in m["cross_corpus"].items():
        algo, pair = k.split("|")
        src, dst = pair.split("->")
        rows.append({"algorithm": algo, "source": src, "target": dst,
                     "transfer": "cross",
                     "domain_match": "same-domain" if v["same_domain"] else "cross-domain",
                     "spearman": v["spearman"], "r2": v["r2"]})
    write(pd.DataFrame(rows), "fig_domain_wall")


# ---------------------------------------------------------------- 2. invariance
def export_invariance() -> None:
    rid = latest("_domain_comparison")
    d = pd.read_csv(RUNS_DIR / rid / "domain_comparison.csv")
    d = d[d["same_direction"].notna()].copy()
    d["feature_class"] = np.where(d["feature"].map(is_rate),
                                  "Temporal variability", "Spectral level")
    d["min_abs_d"] = d[["d_music", "d_ambient"]].abs().min(axis=1)
    d = d.rename(columns={"d_music": "d_music_domain", "d_ambient": "d_ambient_domain"})
    write(d[["feature", "feature_class", "d_music_domain", "d_ambient_domain",
             "same_direction", "min_abs_d", "family"]], "fig_invariance")


# ---------------------------------------------------------------- 3. convergence
def export_convergence() -> None:
    sigs = latest("_feature_significance", n=2)
    gls = latest("_glassbox_models", n=2)

    chains = {
        "SHAP + permutation FDR": meta(sigs[0])["metrics"]["calibrated_consensus"],
        "Cross-domain direction": meta(latest("_domain_comparison"))["metrics"][
            "domain_invariant_features"],
        "EBM native importance": list(meta(gls[0])["metrics"]["ebm_importance_top15"])[:10],
        "Rashomon consensus": [f for f, v in
                               meta(latest("_rashomon_set"))["metrics"]["stability_top20"].items()
                               if v >= 0.99],
    }
    rows = []
    for chain, feats in chains.items():
        for f in feats:
            rows.append({"evidence_chain": chain, "feature": f,
                         "feature_class": "Temporal variability" if is_rate(f) else "Spectral level"})
    df = pd.DataFrame(rows)
    df["n_chains"] = df.groupby("feature")["evidence_chain"].transform("nunique")
    write(df, "fig_convergence")


# ---------------------------------------------------------------- 4. intervention
def export_intervention() -> None:
    rid = latest("_counterfactual_intervention")
    m = meta(rid)["metrics"]
    cur = pd.DataFrame(m["dose_response_curves"]).melt(
        id_vars=["intervention", "dose"], value_vars=["p_deam", "p_emo"],
        var_name="model_domain", value_name="p_sleep")
    cur["model_domain"] = cur["model_domain"].map({"p_deam": "Music model",
                                                   "p_emo": "Ambient model"})
    write(cur, "fig_intervention_curves")

    s = pd.DataFrame(m["summary"])
    write(s, "fig_intervention_metrics")

    diag = []
    for d in m["top_moved_features"]:
        for x in d["top5_moved"]:
            diag.append({"intervention": d["intervention"], "feature": x["feature"],
                         "shift_sd": x["shift_sd"],
                         "feature_class": "Temporal variability" if is_rate(x["feature"])
                                          else "Spectral level"})
    write(pd.DataFrame(diag), "fig_intervention_specificity")


# ---------------------------------------------------------------- 5. calibration
def export_calibration() -> None:
    p = meta(latest("_diag_permutation"))["metrics"]
    perm = pd.DataFrame({"permutation": range(1, len(p["null_consensus"]) + 1),
                         "consensus_count": p["null_consensus"],
                         "cv_auc": p["null_cv_auc"]})
    perm.attrs = {}
    write(perm, "fig_calibration_permutation_null")
    write(pd.DataFrame([{"observed_consensus": p["observed_consensus"],
                         "observed_cv_auc": p["observed_cv_auc"],
                         "p_value": p["p_value"],
                         "null_mean": p["null_consensus_mean"]}]),
          "fig_calibration_permutation_obs")

    e = meta(latest("_eda_validation"))["metrics"]
    i = meta(latest("_individual_differences"))["metrics"]
    # 🔴 The denominator is sqrt(reliability), not the reliability.
    #
    # A correlation cannot be divided by a reliability: r is on the variance
    # scale, rho is not. The largest correlation any predictor can show against
    # a target of reliability r is sqrt(r), so that is the attainable ceiling.
    # Dividing by r instead inflated every figure here by 1/sqrt(r) ~ 2.5x and
    # produced "78-87% of ceiling", which reports/RESULTS_SUMMARY.md then
    # repeated while the manuscript said a third. representation_families_eda
    # already used sqrt(r) (0.3817 = sqrt(0.1457), etc.), so the two Source Data
    # files disagreed with each other as well.
    rel = i["split_half_reliability"]["spearman_brown"]
    ceil = rel ** 0.5
    rows = [
        {"quantity": "Split-half reliability of the target", "spearman": rel,
         "pct_of_ceiling": None},
        {"quantity": "Attainable ceiling (sqrt reliability)", "spearman": ceil,
         "pct_of_ceiling": 1.0},
        {"quantity": "Subjective rating -> EDA", "spearman": e["subjective_vs_eda_spearman"],
         "pct_of_ceiling": e["subjective_vs_eda_spearman"] / ceil},
        {"quantity": "Acoustic model -> EDA",
         "spearman": e["cross_target"]["XGBoost|subj_model_on_eda"]["spearman"],
         "pct_of_ceiling": e["cross_target"]["XGBoost|subj_model_on_eda"]["spearman"] / ceil},
    ]
    write(pd.DataFrame(rows), "fig_calibration_reliability")
    vd = i["variance_decomposition"]
    write(pd.DataFrame([{"component": "Stimulus (clip)", "share": vd["clip_share"]},
                        {"component": "Trial-level noise", "share": vd["residual_share"]}]),
          "fig_calibration_variance")

    write(pd.DataFrame([
        {"target": "Subjective arousal", "spearman":
            e["model_performance"]["XGBoost|subjective"]["spearman"]},
        {"target": "Physiological (EDA)", "spearman":
            e["model_performance"]["XGBoost|eda"]["spearman"]},
    ]), "fig_calibration_naive")


# ---------------------------------------------------------------- ED figures
def export_extended() -> None:
    # 响度混淆
    b = meta("20260726T191106Z_train_deam_test_pmemo")["metrics"]["cross_source"]
    l = meta("20260726T203318Z_train_deam_ln_test_pmemo_ln")["metrics"]["cross_source"]
    rows = [{"model": k, "condition": c, "auc": d[k]["auc"]}
            for k in ("RF", "XGBoost", "SVM", "GBDT", "KNN")
            for c, d in (("Raw", b), ("Loudness-normalised", l))]
    write(pd.DataFrame(rows), "ed_loudness")
    write(pd.DataFrame([
        {"feature": f, "rank_raw": r0, "rank_norm": r1}
        for f, r0, r1 in [("rms_p90", 39.0, 93.0), ("rms_mean", 42.0, 56.3),
                          ("rms_p10", 61.3, 75.0), ("rms_p50", 24.0, 31.3),
                          ("mfcc1_mean", 3.7, 10.7)]]), "ed_loudness_ranks")

    # 样本利用率
    write(pd.DataFrame([
        {"corpus": "DEAM", "domain": "Music", "available": 1802, "regression": 1233, "binary": 477},
        {"corpus": "PMEmo", "domain": "Music", "available": 767, "regression": 717, "binary": 495},
        {"corpus": "Emo-Soundscapes", "domain": "Ambient", "available": 1213,
         "regression": 1213, "binary": 369}]), "ed_data_use")

    # glassbox
    gls = latest("_glassbox_models", n=2)
    rows = []
    for rid, dom in zip(gls, ["Ambient", "Music"]):
        for k, v in meta(rid)["metrics"]["performance"].items():
            rows.append({"domain": dom, "model": k, "spearman": v["spearman"], "r2": v["r2"]})
    write(pd.DataFrame(rows), "ed_glassbox")

    # EBM 形状函数
    shapes = meta(gls[0])["metrics"]["ebm_shape_functions"]
    imp = list(meta(gls[0])["metrics"]["ebm_importance_top15"])[:6]
    rows = []
    for f in imp:
        if f not in shapes:
            continue
        b_, s_ = shapes[f]["bins"], shapes[f]["scores"]
        n = min(len(b_), len(s_))
        for x, yv in zip(b_[:n], s_[:n]):
            rows.append({"feature": f, "value": x, "contribution": yv,
                         "feature_class": "Temporal variability" if is_rate(f) else "Spectral level"})
    write(pd.DataFrame(rows), "ed_shape_functions")

    # Rashomon
    st = meta(latest("_rashomon_set"))["metrics"]["stability_top20"]
    write(pd.DataFrame([{"feature": f, "frequency": v,
                         "feature_class": "Temporal variability" if is_rate(f) else "Spectral level"}
                        for f, v in st.items()]), "ed_rashomon")

    # 个体差异
    i = meta(latest("_individual_differences"))["metrics"]
    write(pd.DataFrame(i["invariant_feature_individual_spread"]), "ed_individual")

    # 表示对比
    reps = latest("_representation_comparison", n=2)
    rows = []
    for rid, corp in zip(reps, ["Emo-Soundscapes (Freesound-derived)", "DEAM (independent)"]):
        for k, v in meta(rid)["metrics"]["performance"].items():
            if "|" not in k:
                continue
            algo, rep = k.split("|")
            rows.append({"corpus": corp, "algorithm": algo, "representation": rep,
                         "spearman": v["spearman"]})
    write(pd.DataFrame(rows), "ed_representations")
    rows = []
    for rid, corp in zip(reps, ["Emo-Soundscapes (Freesound-derived)", "DEAM (independent)"]):
        d = meta(rid)["metrics"]["oof_diversity"]
        st = meta(rid)["metrics"]["stacking"]
        for k, v in d.items():
            rows.append({"corpus": corp, "pair": k, "correlation": v,
                         "stacking_gain": st["gain"]})
    write(pd.DataFrame(rows), "ed_stacking")

    # 域之墙对照（若已跑）
    try:
        rid = latest("_domain_wall_controls")
        write(pd.read_csv(RUNS_DIR / rid / "domain_wall_controls.csv"), "ed_wall_controls")
    except (IndexError, FileNotFoundError):
        print("  (跳过 ed_wall_controls：对照尚未完成)")


if __name__ == "__main__":
    print("导出 source data …")
    export_domain_wall()
    export_invariance()
    export_convergence()
    export_intervention()
    export_calibration()
    export_extended()
    print(f"\n输出目录: {OUT}")
