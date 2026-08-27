"""论文用图（第二批）—— 回归、域之墙、信度、可解释模型、Rashomon、个体差异。

    uv run python scripts/make_figures2.py

Fig.7   回归 vs 二分类：样本利用率
Fig.8   域之墙：跨语料迁移矩阵     ← 主图
Fig.9   信度天花板：为什么不能说「声学预测不了生理」  ← 方法学主图
Fig.10  可解释性的代价：EBM / 单调约束 vs 黑盒
Fig.11  EBM 形状函数：直接可读的阈值
Fig.12  Rashomon set：近优模型中的特征稳定性
Fig.13  个体差异与方差来源
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from soundml.provenance import REPO_ROOT, RUNS_DIR

FIG = REPO_ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})
C_MUSIC, C_AMB, C_NULL, C_OBS, C_OK = "#4C72B0", "#DD8452", "#BBBBBB", "#C44E52", "#55A868"


def latest(suffix: str) -> str:
    return sorted(d.name for d in RUNS_DIR.glob(f"*{suffix}") if (d / "meta.json").exists())[-1]


def meta(run_id: str) -> dict:
    return json.loads((RUNS_DIR / run_id / "meta.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- Fig 7
def fig_data_use():
    ds = ["DEAM\n(音乐)", "PMEmo\n(音乐)", "Emo-Sound\n(环境声)"]
    binary = [477, 495, 369]
    reg = [1233, 717, 1213]
    total = [1802, 767, 1213]

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    x = np.arange(len(ds))
    ax[0].bar(x - 0.26, total, 0.25, label="可用标注", color=C_NULL)
    ax[0].bar(x, reg, 0.25, label="回归用满", color=C_OK)
    ax[0].bar(x + 0.26, binary, 0.25, label="二分类（旧）", color=C_OBS)
    ax[0].set_xticks(x); ax[0].set_xticklabels(ds, fontsize=8)
    ax[0].set_ylabel("样本数"); ax[0].set_title("(a) 二分类丢弃了中间地带")
    ax[0].legend(fontsize=7, frameon=False)

    ax[1].bar(["二分类", "回归"], [sum(binary), sum(reg)], color=[C_OBS, C_OK], width=0.5)
    ax[1].set_ylabel("合计样本数"); ax[1].set_title("(b) 合计提升 2.4×")
    for i, v in enumerate([sum(binary), sum(reg)]):
        ax[1].text(i, v + 40, str(v), ha="center", fontsize=9)

    fig.suptitle("Fig. 7  从二分类改为回归，样本利用率从 36% 提升", y=1.03, fontsize=11)
    fig.savefig(FIG / "fig7_data_use.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig 8（主图）
def fig_domain_wall(run_id: str):
    m = meta(run_id)["metrics"]
    W = pd.DataFrame(m["within_source"]).T
    W["algo"] = [k.split("|")[0] for k in W.index]
    W["ds"] = [k.split("|")[1] for k in W.index]

    rows = []
    for k, v in m["cross_corpus"].items():
        algo, pair = k.split("|")
        src, dst = pair.split("->")
        rows.append({"algo": algo, "src": src, "dst": dst,
                     "same": v["same_domain"], "rho": v["spearman"]})
    C = pd.DataFrame(rows)

    label = {"deam": "DEAM\n音乐", "pmemo": "PMEmo\n音乐", "emo_mix": "Emo-Sound\n环境声"}
    order = ["deam", "pmemo", "emo_mix"]

    fig, ax = plt.subplots(1, 3, figsize=(9.2, 3.0),
                           gridspec_kw={"width_ratios": [1.05, 1.15, 0.95]})

    # (a) 组内 CV
    piv = W.pivot(index="algo", columns="ds", values="spearman")[order]
    piv.plot.bar(ax=ax[0], color=[C_MUSIC, "#7FA1CC", C_AMB], width=0.78, legend=False)
    ax[0].set_ylim(0.6, 0.87); ax[0].set_ylabel("Spearman $\\rho$")
    ax[0].set_xlabel(""); ax[0].set_title("(a) 组内 CV：三个语料都能学")
    ax[0].tick_params(axis="x", rotation=45, labelsize=7)
    ax[0].legend([label[o].replace("\n", " ") for o in order], fontsize=6.5, frameon=False)

    # (b) 迁移矩阵热图
    M = np.full((3, 3), np.nan)
    for i, s in enumerate(order):
        for j, d in enumerate(order):
            if s == d:
                M[i, j] = W[(W.ds == s) & (W.algo == "XGBoost")]["spearman"].iloc[0]
            else:
                sel = C[(C.src == s) & (C.dst == d) & (C.algo == "XGBoost")]
                if len(sel):
                    M[i, j] = sel["rho"].iloc[0]
    im = ax[1].imshow(M, cmap="RdYlGn", vmin=-0.3, vmax=0.85)
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels([label[o] for o in order], fontsize=6.5)
    ax[1].set_yticks(range(3)); ax[1].set_yticklabels([label[o] for o in order], fontsize=6.5)
    ax[1].set_xlabel("测试语料"); ax[1].set_ylabel("训练语料")
    for i in range(3):
        for j in range(3):
            ax[1].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                       fontsize=8, color="k")
    ax[1].set_title("(b) XGBoost 迁移矩阵（对角为组内）")
    ax[1].grid(False)
    fig.colorbar(im, ax=ax[1], fraction=0.046, label="Spearman $\\rho$")

    # (c) 同域 vs 跨域
    same = C[C.same]["rho"]; diff = C[~C.same]["rho"]
    parts = ax[2].violinplot([same.values, diff.values], showmeans=True, widths=0.7)
    for pc, col in zip(parts["bodies"], [C_OK, C_OBS]):
        pc.set_facecolor(col); pc.set_alpha(0.65)
    ax[2].axhline(0, color="k", lw=0.8, ls="--")
    ax[2].set_xticks([1, 2]); ax[2].set_xticklabels(
        [f"同域\n(n={len(same)})", f"跨域\n(n={len(diff)})"], fontsize=8)
    ax[2].set_ylabel("跨语料 Spearman $\\rho$")
    ax[2].set_title(f"(c) {same.mean():.3f}  vs  {diff.mean():.3f}")

    fig.suptitle("Fig. 8  域之墙：同域跨源几乎无损，跨域精确为零（双向）", y=1.03, fontsize=11)
    fig.savefig(FIG / "fig8_domain_wall.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig 9（方法学主图）
def fig_reliability(eda_run: str, ind_run: str):
    e, i = meta(eda_run)["metrics"], meta(ind_run)["metrics"]
    ceil = i["split_half_reliability"]["spearman_brown"]

    fig, ax = plt.subplots(1, 3, figsize=(8.8, 2.9),
                           gridspec_kw={"width_ratios": [1, 1.1, 0.9]})

    # (a) 天真读法
    names = ["主观 arousal", "生理 EDA"]
    vals = [e["model_performance"]["XGBoost|subjective"]["spearman"],
            e["model_performance"]["XGBoost|eda"]["spearman"]]
    ax[0].bar(names, vals, color=[C_MUSIC, C_OBS], width=0.5)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_ylabel("Spearman $\\rho$"); ax[0].set_ylim(-0.2, 0.95)
    ax[0].set_title("(a) 天真读法：\n「声学预测不了生理」")
    for k, v in enumerate(vals):
        ax[0].text(k, v + 0.04 * np.sign(v or 1), f"{v:+.3f}", ha="center", fontsize=8)

    # (b) 加上天花板
    labels = ["分半信度\n(天花板)", "主观打分\n→ EDA", "声学模型\n→ EDA"]
    vv = [ceil, e["subjective_vs_eda_spearman"], e["cross_target"]["XGBoost|subj_model_on_eda"]["spearman"]]
    bars = ax[1].bar(labels, vv, color=["k", C_MUSIC, C_AMB], width=0.55)
    bars[0].set_facecolor("none"); bars[0].set_edgecolor("k"); bars[0].set_linewidth(1.8)
    ax[1].axhline(ceil, color="k", ls="--", lw=1)
    ax[1].set_ylabel("Spearman $\\rho$"); ax[1].set_ylim(0, 0.22)
    ax[1].set_title("(b) 对着天花板读：\n主观 87%、声学 78%")
    for k, v in enumerate(vv):
        pct = "" if k == 0 else f"\n({v / ceil:.0%})"
        ax[1].text(k, v + 0.008, f"{v:.3f}{pct}", ha="center", fontsize=7.5)

    # (c) 方差来源
    vd = i["variance_decomposition"]
    ax[2].pie([vd["clip_share"], vd["residual_share"]],
              labels=[f"音频\n{vd['clip_share']:.1%}", f"试次噪声\n{vd['residual_share']:.1%}"],
              colors=[C_OK, C_NULL], startangle=90,
              textprops={"fontsize": 7.5}, wedgeprops={"edgecolor": "white"})
    ax[2].set_title("(c) EDA 方差 86.6% 是噪声")

    fig.suptitle("Fig. 9  在说「X 预测不了 Y」之前，必须先测 Y 的信度", y=1.04, fontsize=11)
    fig.savefig(FIG / "fig9_reliability.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig 10
def fig_glassbox(amb_run: str, mus_run: str):
    A, M = meta(amb_run)["metrics"]["performance"], meta(mus_run)["metrics"]["performance"]
    order = ["XGBoost", "XGBoost(topk,no constraint)", "XGBoost+monotone", "EBM", "RF"]
    short = ["XGB\n(全特征)", "XGB\n(top-k)", "XGB\n+单调约束", "EBM", "RF"]

    fig, ax = plt.subplots(1, 2, figsize=(7.6, 2.8))
    x = np.arange(len(order))
    ax[0].bar(x - 0.2, [A[k]["spearman"] for k in order], 0.4, color=C_AMB, label="环境声域")
    ax[0].bar(x + 0.2, [M[k]["spearman"] for k in order], 0.4, color=C_MUSIC, label="音乐域")
    ax[0].set_xticks(x); ax[0].set_xticklabels(short, fontsize=6.5)
    ax[0].set_ylim(0.70, 0.80); ax[0].set_ylabel("Spearman $\\rho$")
    ax[0].set_title("(a) 可解释性的代价很小")
    ax[0].legend(fontsize=7, frameon=False)

    cost = [A["XGBoost+monotone"]["spearman"] - A["XGBoost(topk,no constraint)"]["spearman"],
            M["XGBoost+monotone"]["spearman"] - M["XGBoost(topk,no constraint)"]["spearman"]]
    ax[1].bar(["环境声域", "音乐域"], cost, color=[C_AMB, C_MUSIC], width=0.45)
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_ylim(-0.03, 0.03); ax[1].set_ylabel("$\\Delta\\rho$（加单调约束后）")
    ax[1].set_title("(b) 单调约束零代价\n→ 单调假设被数据支持")
    for k, v in enumerate(cost):
        ax[1].text(k, v + 0.002 * np.sign(v or 1), f"{v:+.4f}", ha="center", fontsize=8)

    fig.suptitle("Fig. 10  原生可解释模型：EBM 与单调约束", y=1.03, fontsize=11)
    fig.savefig(FIG / "fig10_glassbox.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig 11
def fig_shape_functions(run_id: str):
    shapes = meta(run_id)["metrics"]["ebm_shape_functions"]
    imp = meta(run_id)["metrics"]["ebm_importance_top15"]
    picks = [f for f in list(imp)[:6] if f in shapes][:6]

    fig, axes = plt.subplots(2, 3, figsize=(8.4, 4.2))
    for ax, name in zip(axes.ravel(), picks):
        d = shapes[name]
        b, s = np.array(d["bins"]), np.array(d["scores"])
        n = min(len(b), len(s))
        ax.plot(b[:n], s[:n], color=C_AMB, lw=1.6)
        ax.axhline(0, color="k", lw=0.7, ls="--")
        cross = np.where(np.diff(np.sign(s[:n])))[0]
        if cross.size:
            ax.axvline(b[cross[0]], color=C_OBS, lw=1.2, ls=":")
            ax.text(0.97, 0.06, f"阈值≈{b[cross[0]]:.3g}", transform=ax.transAxes,
                    ha="right", fontsize=6.5, color=C_OBS)
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=6.5)
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    fig.supxlabel("特征值", fontsize=9); fig.supylabel("对 arousal 的贡献", fontsize=9)
    fig.suptitle("Fig. 11  EBM 形状函数：阈值可直接读出，无需事后归因", y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig11_shape_functions.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig 12
def fig_rashomon(run_id: str):
    m = meta(run_id)["metrics"]
    st = pd.Series(m["stability_top20"]).sort_values()
    RATE = lambda f: f.endswith(("_std", "_dmean")) or f == "onset_rate"

    fig, ax = plt.subplots(1, 2, figsize=(7.8, 3.2),
                           gridspec_kw={"width_ratios": [1.25, 0.75]})
    colors = [C_OK if RATE(f) else C_NULL for f in st.index]
    ax[0].barh(range(len(st)), st.values, color=colors, height=0.72)
    ax[0].set_yticks(range(len(st))); ax[0].set_yticklabels(st.index, fontsize=6.5)
    ax[0].set_xlabel(f"进入 top-{m.get('top_k', 15)} 的模型比例")
    ax[0].set_xlim(0, 1.05)
    ax[0].set_title("(a) 近优模型中的特征稳定性")
    ax[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_OK, label="变化率 / 变异量")],
                 fontsize=7, frameon=False, loc="lower right")

    n_set = m["n_in_rashomon_set"]
    rng = m["rho_range_in_set"]
    ax[1].bar(["Rashomon\n集合内", "集合外"], [n_set, m.get("n_models", 60) - n_set],
              color=[C_OK, C_NULL], width=0.5)
    ax[1].set_ylabel("模型数")
    ax[1].set_title(f"(b) ρ ∈ [{rng[0]:.3f}, {rng[1]:.3f}]\n的模型共 {n_set} 个")

    fig.suptitle("Fig. 12  Rashomon set：不是「最佳模型说什么」，"
                 "而是「所有近优模型都同意什么」", y=1.02, fontsize=11)
    fig.savefig(FIG / "fig12_rashomon.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig 13
def fig_individual(run_id: str):
    m = meta(run_id)["metrics"]
    D = pd.DataFrame(m["invariant_feature_individual_spread"])
    sv = m["subject_vs_population_rho"]

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    yy = np.arange(len(D))
    ax[0].errorbar(D["mean_rho"], yy,
                   xerr=[D["mean_rho"] - D["p10"], D["p90"] - D["mean_rho"]],
                   fmt="o", color=C_MUSIC, ecolor=C_NULL, capsize=3, ms=5)
    ax[0].axvline(0, color="k", lw=0.8, ls="--")
    ax[0].set_yticks(yy); ax[0].set_yticklabels(D["feature"], fontsize=6.5)
    ax[0].set_xlabel("单个被试的 Spearman $\\rho$（点=均值，须=p10–p90）")
    ax[0].set_title("(a) 域无关特征的个体间散布")

    ax[1].bar(["被试 vs\n群体均值"], [sv["mean"]], color=C_MUSIC, width=0.35,
              yerr=[[sv["mean"] - sv["p10"]], [sv["p90"] - sv["mean"]]],
              capsize=5, ecolor=C_NULL)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("Spearman $\\rho$")
    ax[1].set_title(f"(b) {sv['frac_negative']:.1%} 的被试与群体反向")

    fig.suptitle("Fig. 13  个体差异：群体平均掩盖了大量个体间异质性", y=1.03, fontsize=11)
    fig.savefig(FIG / "fig13_individual.png"); plt.close(fig)


if __name__ == "__main__":
    import matplotlib.font_manager as fm
    for cand in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if any(cand in f.name for f in fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [cand]
            plt.rcParams["axes.unicode_minus"] = False
            break

    reg = latest("_regression_matrix")
    eda = latest("_eda_validation")
    ind = latest("_individual_differences")
    gls = sorted(d.name for d in RUNS_DIR.glob("*_glassbox_models") if (d / "meta.json").exists())

    fig_data_use(); print("✓ fig7_data_use")
    fig_domain_wall(reg); print("✓ fig8_domain_wall")
    fig_reliability(eda, ind); print("✓ fig9_reliability")
    if len(gls) >= 2:
        fig_glassbox(gls[-2], gls[-1]); print("✓ fig10_glassbox")
        fig_shape_functions(gls[-2]); print("✓ fig11_shape_functions")
    try:
        fig_rashomon(latest("_rashomon_set")); print("✓ fig12_rashomon")
    except (IndexError, KeyError) as e:
        print(f"– fig12 跳过（Rashomon 未完成: {e}）")
    fig_individual(ind); print("✓ fig13_individual")
    print(f"\n输出目录: {FIG}")
