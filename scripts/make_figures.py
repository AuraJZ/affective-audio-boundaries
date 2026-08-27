"""生成论文用图。

    uv run python scripts/make_figures.py

Fig.1  流程总览与两域对比（数据 → 特征 → 校准 → 规则）
Fig.2  置换检验：为什么「SHAP 跨模型共识」是废统计量
Fig.3  响度是数据集指纹，不是信号
Fig.4  跨域反转：音乐规则在环境声上系统性翻转
Fig.5  域无关性：变化率类跨域，音色类不跨域   ← 主图
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from soundml.provenance import REPO_ROOT, RUNS_DIR

FIG = REPO_ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

C_MUSIC, C_AMB, C_NULL, C_OBS = "#4C72B0", "#DD8452", "#BBBBBB", "#C44E52"


def meta(run_id: str) -> dict:
    return json.loads((RUNS_DIR / run_id / "meta.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- Fig 2
def fig_permutation(run_id: str):
    m = meta(run_id)["metrics"]
    null = np.array(m["null_consensus"])
    obs = m["observed_consensus"]

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))

    bins = np.arange(null.min() - 0.5, null.max() + 1.5)
    ax[0].hist(null, bins=bins, color=C_NULL, edgecolor="white", label="随机标签（零分布）")
    ax[0].axvline(obs, color=C_OBS, lw=2, label=f"真实标签 = {obs}")
    ax[0].axvline(null.mean(), color="k", ls="--", lw=1, label=f"零分布均值 = {null.mean():.2f}")
    ax[0].set_xlabel("SHAP top-20 三模型交集特征数")
    ax[0].set_ylabel("置换次数")
    ax[0].set_title(f"(a) 共识数无法区分信号与噪声  $p$ = {m['p_value']:.3f}")
    ax[0].legend(fontsize=7, frameon=False)

    aucs = np.array(m["null_cv_auc"])
    ax[1].hist(aucs, bins=12, color=C_NULL, edgecolor="white")
    ax[1].axvline(m["observed_cv_auc"], color=C_OBS, lw=2,
                  label=f"真实标签 AUC = {m['observed_cv_auc']:.3f}")
    ax[1].axvline(0.5, color="k", ls="--", lw=1, label="0.5")
    ax[1].set_xlabel("交叉验证 AUC")
    ax[1].set_title(f"(b) 打乱确实生效（零分布均值 {aucs.mean():.3f}）")
    ax[1].legend(fontsize=7, frameon=False)

    fig.suptitle("Fig. 2  「多模型 SHAP 共识」是未经校准的统计量", y=1.04, fontsize=11)
    fig.savefig(FIG / "fig2_permutation.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3
def fig_loudness(base_run: str, ln_run: str):
    b, l = meta(base_run)["metrics"], meta(ln_run)["metrics"]
    models = ["RF", "XGBoost", "SVM", "GBDT", "KNN"]
    bc = [b["cross_source"][m]["auc"] for m in models]
    lc = [l["cross_source"][m]["auc"] for m in models]

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    x = np.arange(len(models))
    ax[0].bar(x - 0.2, bc, 0.4, label="原始", color=C_NULL)
    ax[0].bar(x + 0.2, lc, 0.4, label="响度归一化", color=C_MUSIC)
    ax[0].set_xticks(x); ax[0].set_xticklabels(models, fontsize=8)
    ax[0].set_ylim(0.75, 0.96); ax[0].set_ylabel("跨源 AUC (DEAM→PMEmo)")
    ax[0].set_title("(a) 去掉绝对响度后跨源泛化反而变好")
    ax[0].legend(fontsize=7, frameon=False)

    lvl = ["rms_p90", "rms_mean", "rms_p10", "rms_p50", "mfcc1_mean"]
    before = [39.0, 42.0, 61.3, 24.0, 3.7]
    after = [93.0, 56.3, 75.0, 31.3, 10.7]
    yy = np.arange(len(lvl))
    ax[1].barh(yy - 0.2, before, 0.4, label="原始", color=C_NULL)
    ax[1].barh(yy + 0.2, after, 0.4, label="响度归一化", color=C_MUSIC)
    ax[1].set_yticks(yy); ax[1].set_yticklabels(lvl, fontsize=7.5)
    ax[1].invert_yaxis(); ax[1].set_xlabel("SHAP 平均排名（越大越不重要）")
    ax[1].set_title("(b) 绝对电平类特征随之失效")
    ax[1].legend(fontsize=7, frameon=False)

    fig.suptitle("Fig. 3  响度是数据集指纹，不是助眠信号", y=1.04, fontsize=11)
    fig.savefig(FIG / "fig3_loudness.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 4
def fig_cross_domain(run_id: str):
    m = meta(run_id)["metrics"]
    d = pd.DataFrame(m["rule_feature_directions"]).T.reset_index(names="feature")
    d = d.sort_values("d_music")

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0),
                           gridspec_kw={"width_ratios": [1.55, 1]})

    yy = np.arange(len(d))
    ax[0].barh(yy - 0.2, d["d_music"], 0.4, color=C_MUSIC, label="音乐域 (DEAM)")
    ax[0].barh(yy + 0.2, d["d_ambient"], 0.4, color=C_AMB, label="环境声域 (ESC-50)")
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].set_yticks(yy); ax[0].set_yticklabels(d["feature"], fontsize=7.5)
    ax[0].set_xlabel("Cohen's $d$（正 = 助眠类该特征更高）")
    ax[0].set_title("(a) 11 条音乐规则中 9 条方向翻转")
    ax[0].legend(fontsize=7, frameon=False, loc="lower right")

    agree = [m["direction_agreement_rule_features"] * 100,
             m["direction_agreement_all"] * 100]
    ax[1].bar(["规则特征\n(11)", "全部特征\n(122)"], agree,
              color=[C_OBS, C_NULL], width=0.55)
    ax[1].axhline(50, color="k", ls="--", lw=1)
    ax[1].text(1.45, 51.5, "随机水平 50%", fontsize=7, ha="right")
    ax[1].set_ylim(0, 100); ax[1].set_ylabel("方向一致率 (%)")
    ax[1].set_title("(b) 一致率显著低于随机")
    for i, v in enumerate(agree):
        ax[1].text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=8)

    fig.suptitle("Fig. 4  音乐域规则在环境声上系统性反转", y=1.02, fontsize=11)
    fig.savefig(FIG / "fig4_cross_domain.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 5（主图）
def fig_domain_invariance(run_id: str):
    m = meta(run_id)["metrics"]
    df = pd.read_csv(RUNS_DIR / run_id / "domain_comparison.csv")
    df = df[df["same_direction"].notna()].copy()
    df["min_abs_d"] = df[["d_music", "d_ambient"]].abs().min(axis=1)

    # 变化率/时序类 vs 绝对水平类
    RATE = ("_dmean", "_std", "onset_rate")
    df["kind"] = ["变化率 / 波动性" if (f.endswith(RATE[:2]) or f.endswith("_dmean")
                                  or f.endswith("_std") or f == "onset_rate")
                  else "绝对频谱水平" for f in df["feature"]]

    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.2),
                           gridspec_kw={"width_ratios": [1.15, 1]})

    for kind, mk in [("变化率 / 波动性", "o"), ("绝对频谱水平", "s")]:
        s = df[df["kind"] == kind]
        ok = s[s["same_direction"]]
        no = s[~s["same_direction"]]
        ax[0].scatter(ok["d_music"], ok["d_ambient"], marker=mk, s=52,
                      color=C_MUSIC if kind.startswith("变化") else C_AMB,
                      edgecolor="k", lw=0.6, label=f"{kind}（一致）")
        ax[0].scatter(no["d_music"], no["d_ambient"], marker=mk, s=52,
                      facecolor="white",
                      edgecolor=C_MUSIC if kind.startswith("变化") else C_AMB,
                      lw=1.2, label=f"{kind}（翻转）")

    lim = 1.8
    ax[0].fill_between([-lim, 0], [-lim, -lim], [0, 0], color="green", alpha=0.05)
    ax[0].fill_between([0, lim], [0, 0], [lim, lim], color="green", alpha=0.05)
    ax[0].axhline(0, color="k", lw=0.8); ax[0].axvline(0, color="k", lw=0.8)
    ax[0].plot([-lim, lim], [-lim, lim], ls=":", color="gray", lw=0.8)
    ax[0].set_xlim(-lim, lim); ax[0].set_ylim(-lim, lim)
    ax[0].set_xlabel("音乐域 Cohen's $d$"); ax[0].set_ylabel("环境声域 Cohen's $d$")
    ax[0].set_title("(a) 绿区 = 两域方向一致")
    ax[0].legend(fontsize=6.5, frameon=False, loc="upper left")

    top = df[df["same_direction"]].nlargest(6, "min_abs_d").sort_values("min_abs_d")
    yy = np.arange(len(top))
    ax[1].barh(yy - 0.2, top["d_music"].abs(), 0.4, color=C_MUSIC, label="音乐域")
    ax[1].barh(yy + 0.2, top["d_ambient"].abs(), 0.4, color=C_AMB, label="环境声域")
    ax[1].set_yticks(yy); ax[1].set_yticklabels(top["feature"], fontsize=7.5)
    ax[1].set_xlabel("|Cohen's $d$|")
    ax[1].set_title("(b) 域无关特征（全部为负方向）")
    ax[1].legend(fontsize=7, frameon=False, loc="lower right")

    fig.suptitle("Fig. 5  时间稳定性跨域普适，频谱音色不跨域", y=1.02, fontsize=11)
    fig.savefig(FIG / "fig5_domain_invariance.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 1
def fig_overview(sig_music: str, sig_amb: str):
    mm, ma = meta(sig_music)["metrics"], meta(sig_amb)["metrics"]

    fig, ax = plt.subplots(1, 3, figsize=(8.0, 2.5))

    ax[0].bar(["音乐\nDEAM", "音乐\nPMEmo", "环境声\nEmo-Sound"], [477, 495, 369],
              color=[C_MUSIC, C_MUSIC, C_AMB], width=0.55)
    ax[0].set_ylabel("样本数（正+负）")
    ax[0].set_title("(a) 语料")
    for i, v in enumerate([477, 495, 369]):
        ax[0].text(i, v + 12, str(v), ha="center", fontsize=8)

    stages = ["原始特征", "RF 显著", "XGB 显著", "双模型\n校准", "成规则"]
    music = [144, mm["n_significant_RF"], mm["n_significant_XGBoost"],
             mm["n_calibrated_consensus"], 11]
    amb = [122, ma["n_significant_RF"], ma["n_significant_XGBoost"],
           ma["n_calibrated_consensus"], 6]
    x = np.arange(len(stages))
    ax[1].plot(x, music, "o-", color=C_MUSIC, label="音乐域")
    ax[1].plot(x, amb, "s-", color=C_AMB, label="环境声域")
    ax[1].set_yscale("log"); ax[1].set_xticks(x)
    ax[1].set_xticklabels(stages, fontsize=6.5)
    ax[1].set_ylabel("特征数（对数轴）")
    ax[1].set_title("(b) 逐级筛选")
    ax[1].legend(fontsize=7, frameon=False)

    ax[2].bar(["组内 CV", "跨源\n(DEAM→PMEmo)", "跨域\n(→环境声)"],
              [0.9378, 0.9304, 0.5245],
              color=[C_MUSIC, C_MUSIC, C_OBS], width=0.55)
    ax[2].axhline(0.5, color="k", ls="--", lw=1)
    ax[2].set_ylim(0.4, 1.0); ax[2].set_ylabel("AUC (RF)")
    ax[2].set_title("(c) 泛化在跨域处崩塌")
    for i, v in enumerate([0.9378, 0.9304, 0.5245]):
        ax[2].text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)

    fig.suptitle("Fig. 1  流程总览", y=1.04, fontsize=11)
    fig.savefig(FIG / "fig1_overview.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 6
def fig_intervention(run_id: str):
    m = meta(run_id)["metrics"]
    curves = pd.DataFrame(m["dose_response_curves"])
    S = pd.DataFrame(m["summary"])
    diag = {d["intervention"]: d["top5_moved"] for d in m["top_moved_features"]}

    names = {"spectral_smooth": "谱变化平滑\n(降低音色变化速率)",
             "transient_inject": "注入瞬态\n(提高突发密度)",
             "envelope_smooth": "包络压缩\n(削平动态)"}
    order = ["spectral_smooth", "transient_inject", "envelope_smooth"]

    fig, ax = plt.subplots(1, 3, figsize=(8.6, 3.0),
                           gridspec_kw={"width_ratios": [1.25, 1, 1.15]})

    # (a) 剂量-反应曲线
    for i, iv in enumerate(order):
        c = curves[curves.intervention == iv]
        ls = "-" if iv != "envelope_smooth" else ":"
        ax[0].plot(c["dose"], c["p_deam"], ls, marker="o", ms=4,
                   color=C_MUSIC, alpha=1 - 0.25 * i)
        ax[0].plot(c["dose"], c["p_emo"], ls, marker="s", ms=4,
                   color=C_AMB, alpha=1 - 0.25 * i)
    ax[0].set_xticks([0, 1, 2, 3]); ax[0].set_xlabel("干预剂量（0 = 对照）")
    ax[0].set_ylabel("模型判为「助眠」的概率")
    ax[0].set_title("(a) 剂量–反应")
    ax[0].legend(handles=[
        plt.Line2D([], [], color=C_MUSIC, marker="o", label="音乐域模型"),
        plt.Line2D([], [], color=C_AMB, marker="s", label="环境声域模型"),
        plt.Line2D([], [], color="k", ls="-", label="谱平滑 / 注入瞬态"),
        plt.Line2D([], [], color="k", ls=":", label="包络压缩（失败）"),
    ], fontsize=6.5, frameon=False, loc="center left")

    # (b) 跨域一致性
    dr = S.drop_duplicates("intervention").set_index("intervention")
    yy = np.arange(len(order))
    ax[1].barh(yy - 0.2, [dr.loc[i, "dose_response_rho_deam"] for i in order],
               0.4, color=C_MUSIC, label="音乐域")
    ax[1].barh(yy + 0.2, [dr.loc[i, "dose_response_rho_emo"] for i in order],
               0.4, color=C_AMB, label="环境声域")
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_yticks(yy); ax[1].set_yticklabels([names[i] for i in order], fontsize=6.5)
    ax[1].set_xlabel("剂量–反应 Spearman $\\rho$")
    ax[1].set_title("(b) 同号 = 因果效应跨域一致")
    ax[1].legend(fontsize=7, frameon=False, loc="lower right")

    # (c) 特异性诊断：被推动最多的 5 个特征
    iv = "spectral_smooth"
    top = diag[iv][::-1]
    yy = np.arange(len(top))
    cols = [C_MUSIC if f["feature"].endswith(("_dmean", "_std")) else C_NULL for f in top]
    ax[2].barh(yy, [f["shift_sd"] for f in top], color=cols, height=0.6)
    ax[2].set_yticks(yy); ax[2].set_yticklabels([f["feature"] for f in top], fontsize=6.5)
    ax[2].set_xlabel("标准化位移 (SD)")
    ax[2].set_title("(c) 谱平滑推动的全是变化率量")
    ax[2].legend(handles=[Patch(color=C_MUSIC, label="变化率 / 波动性度量")],
                 fontsize=6.5, frameon=False, loc="lower right")

    fig.suptitle("Fig. 6  反事实干预：操纵「变化率」维度，两域预测同向移动", y=1.03, fontsize=11)
    fig.savefig(FIG / "fig6_intervention.png")
    plt.close(fig)


if __name__ == "__main__":
    import matplotlib.font_manager as fm
    for cand in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if any(cand in f.name for f in fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [cand]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"中文字体: {cand}")
            break

    SIG_MUSIC = "20260726T230352Z_feature_significance"
    SIG_AMB = "20260727T091605Z_feature_significance"

    fig_overview(SIG_MUSIC, SIG_AMB); print("✓ fig1_overview")
    fig_permutation("20260726T225328Z_diag_permutation"); print("✓ fig2_permutation")
    fig_loudness("20260726T191106Z_train_deam_test_pmemo",
                 "20260726T203318Z_train_deam_ln_test_pmemo_ln"); print("✓ fig3_loudness")
    fig_cross_domain("20260727T090607Z_cross_domain_music_to_ambient"); print("✓ fig4_cross_domain")
    fig_domain_invariance("20260727T094544Z_domain_comparison"); print("✓ fig5_domain_invariance")
    fig_intervention("20260727T100401Z_counterfactual_intervention"); print("✓ fig6_intervention")
    print(f"\n输出目录: {FIG}")
