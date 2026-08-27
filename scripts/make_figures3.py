"""论文用图（第三批）—— 表示对比与证据汇聚。

Fig.14  表示对比：CLAP vs 手工描述符，含污染对照
Fig.15  四条独立证据链汇聚到同一结论   ← 主图
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
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})
C_MUSIC, C_AMB, C_NULL, C_OBS, C_OK = "#4C72B0", "#DD8452", "#BBBBBB", "#C44E52", "#55A868"


def meta(rid: str) -> dict:
    return json.loads((RUNS_DIR / rid / "meta.json").read_text(encoding="utf-8"))


def runs(suffix: str) -> list[str]:
    return sorted(d.name for d in RUNS_DIR.glob(f"*{suffix}") if (d / "meta.json").exists())


# ---------------------------------------------------------------- Fig 14
def fig_representations(amb_run: str, mus_run: str | None):
    A = meta(amb_run)["metrics"]
    pa = A["performance"]

    fig, ax = plt.subplots(1, 3, figsize=(9.0, 2.9),
                           gridspec_kw={"width_ratios": [1.1, 0.95, 0.95]})

    # (a) 环境声域：各表示 × 算法
    algos = ["RidgeProbe", "SVR", "XGBoost", "RF"]
    xa = [pa.get(f"{al}|A", {}).get("spearman", np.nan) for al in algos]
    xc = [pa.get(f"{al}|C", {}).get("spearman", np.nan) for al in algos]
    x = np.arange(len(algos))
    ax[0].bar(x - 0.2, xa, 0.4, color=C_NULL, label="Layer A 手工 (122维)")
    ax[0].bar(x + 0.2, xc, 0.4, color=C_MUSIC, label="Layer C CLAP (512维)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(algos, fontsize=7.5)
    ax[0].set_ylim(0.65, 0.90); ax[0].set_ylabel("Spearman $\\rho$")
    ax[0].set_title("(a) 环境声域：表示比模型更重要")
    ax[0].legend(fontsize=6.5, frameon=False)

    # (b) 污染对照：两个语料上 CLAP 相对手工的增益
    labels, gains, cols = ["环境声\n(Freesound来源\n可能污染)"], [], [C_OBS]
    ga = max(v["spearman"] for k, v in pa.items() if k.endswith("|C")) - \
         max(v["spearman"] for k, v in pa.items() if k.endswith("|A"))
    gains.append(ga)
    if mus_run:
        pm = meta(mus_run)["metrics"]["performance"]
        gm = max(v["spearman"] for k, v in pm.items() if k.endswith("|C")) - \
             max(v["spearman"] for k, v in pm.items() if k.endswith("|A"))
        gains.append(gm); labels.append("音乐\n(非Freesound\n干净对照)"); cols.append(C_OK)
    ax[1].bar(labels, gains, color=cols, width=0.5)
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_ylabel("CLAP 相对手工的 $\\Delta\\rho$")
    ax[1].set_title("(b) 预训练污染对照")
    for i, v in enumerate(gains):
        ax[1].text(i, v + 0.004, f"{v:+.3f}", ha="center", fontsize=8)
    ax[1].tick_params(axis="x", labelsize=6.5)

    # (c) 表示多样性与 stacking 增益
    d = A["oof_diversity"]; st = A["stacking"]
    keys = ["within_A", "within_C", "across_AC"]
    ax[2].bar(["A 内部", "C 内部", "跨 A/C"], [d[k] for k in keys],
              color=[C_NULL, C_MUSIC, C_OK], width=0.55)
    ax[2].set_ylim(0.7, 1.0); ax[2].set_ylabel("OOF 预测相关")
    ax[2].set_title(f"(c) 即便跨表示仍相关 {d['across_AC']:.2f}\n"
                    f"→ stacking 增益仅 {st['gain']:+.4f}")
    for i, k in enumerate(keys):
        ax[2].text(i, d[k] + 0.008, f"{d[k]:.3f}", ha="center", fontsize=7.5)

    fig.suptitle("Fig. 14  手工描述符 vs 预训练嵌入", y=1.04, fontsize=11)
    fig.savefig(FIG / "fig14_representations.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig 15（主图）
def fig_convergence(sig_run: str, dom_run: str, gls_run: str, rash_run: str):
    """四条方法学上互不依赖的证据链，是否指向同一批特征。"""
    RATE = lambda f: f.endswith(("_std", "_dmean")) or f == "onset_rate"

    # 1. SHAP + 置换 FDR
    e1 = meta(sig_run)["metrics"]["calibrated_consensus"]
    # 2. 跨域方向一致
    e2 = meta(dom_run)["metrics"]["domain_invariant_features"]
    # 3. EBM 原生重要性 top10
    e3 = list(meta(gls_run)["metrics"]["ebm_importance_top15"])[:10]
    # 4. Rashomon 全模型共识
    st = meta(rash_run)["metrics"]["stability_top20"]
    e4 = [f for f, v in st.items() if v >= 0.99]

    chains = [("SHAP + 置换 FDR", e1), ("跨域方向一致", e2),
              ("EBM 原生重要性", e3), ("Rashomon 全模型共识", e4)]

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.4),
                           gridspec_kw={"width_ratios": [1, 1.05]})

    # (a) 每条链中「变化率/变异」类特征的占比
    fracs, tots = [], []
    for name, feats in chains:
        n = sum(1 for f in feats if RATE(f))
        fracs.append(n / max(len(feats), 1)); tots.append((n, len(feats)))
    yy = np.arange(len(chains))
    ax[0].barh(yy, fracs, color=C_OK, height=0.6)
    ax[0].axvline(0.5, color="k", ls="--", lw=1)
    ax[0].set_yticks(yy); ax[0].set_yticklabels([c[0] for c in chains], fontsize=7.5)
    ax[0].set_xlim(0, 1.05); ax[0].set_xlabel("「变化率 / 变异」类特征占比")
    ax[0].set_title("(a) 四条独立证据链")
    for i, (n, t) in enumerate(tots):
        ax[0].text(fracs[i] + 0.02, i, f"{n}/{t}", va="center", fontsize=7.5)

    # (b) 特征 × 证据链 命中矩阵
    allf = []
    for _, feats in chains:
        for f in feats:
            if f not in allf:
                allf.append(f)
    hit = np.zeros((len(allf), len(chains)))
    for j, (_, feats) in enumerate(chains):
        for f in feats:
            hit[allf.index(f), j] = 1
    order = np.argsort(-hit.sum(axis=1))
    allf = [allf[i] for i in order]; hit = hit[order]
    keep = hit.sum(axis=1) >= 2          # 至少被两条链命中
    allf = [f for f, k in zip(allf, keep) if k]; hit = hit[keep]

    ax[1].imshow(hit, cmap="Greens", vmin=0, vmax=1.4, aspect="auto")
    ax[1].set_xticks(range(len(chains)))
    ax[1].set_xticklabels(["SHAP\nFDR", "跨域\n方向", "EBM", "Rashomon"], fontsize=7)
    ax[1].set_yticks(range(len(allf)))
    ax[1].set_yticklabels(
        [f"{f} ★" if RATE(f) else f for f in allf], fontsize=7)
    ax[1].set_title("(b) 被 ≥2 条链命中的特征（★=变化率类）")
    ax[1].grid(False)
    for i in range(len(allf)):
        for j in range(len(chains)):
            if hit[i, j]:
                ax[1].text(j, i, "✓", ha="center", va="center", fontsize=8)

    fig.suptitle("Fig. 15  四条方法学互不依赖的证据链，汇聚到同一批「变化率」特征",
                 y=1.02, fontsize=11)
    fig.savefig(FIG / "fig15_convergence.png"); plt.close(fig)


if __name__ == "__main__":
    import matplotlib.font_manager as fm
    for cand in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if any(cand in f.name for f in fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [cand]
            plt.rcParams["axes.unicode_minus"] = False
            break

    rep = runs("_representation_comparison")
    gls = runs("_glassbox_models")
    fig_representations(rep[0], rep[1] if len(rep) > 1 else None)
    print("✓ fig14_representations")

    # 环境声域的显著性 run（第二个 feature_significance）
    sig = runs("_feature_significance")
    fig_convergence(sig[0], runs("_domain_comparison")[-1], gls[0], runs("_rashomon_set")[-1])
    print("✓ fig15_convergence")
    print(f"\n输出目录: {FIG}")
