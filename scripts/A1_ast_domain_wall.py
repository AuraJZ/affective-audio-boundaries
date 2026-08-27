r"""域之墙在预训练表征上还立不立着 —— 有几族跑几族。

    .venv/Scripts/python scripts/A1_ast_domain_wall.py

## 这是唯一可能推翻本文核心发现的检验

论文主张「音乐上训练的模型搬到环境声上就废了，反之亦然」，并声称
这不是特征设计不足。但支撑「不是特征不足」的证据此前只有**一个** CLAP 模型，
而它偏偏把 Emo-Soundscapes 的来源（Freesound）吃进了预训练数据。

一个模型 + 一个被污染的语料，撑不住「这堵墙不可逾越」这种强声称。
**若换个表征就能翻过去，那「墙」这个框架本身就错了。**

## 四族，四种预训练范式

| 族 | 预训练 | 为什么选它 |
|---|---|---|
| **AST** | AudioSet **监督**分类，527 类 | 明确覆盖环境声 —— **最该翻过去**的那个 |
| **MERT** | **音乐**自监督 | 音乐专用；若它也翻不过去，「领域知识不足」被排除 |
| **W2V2** | **语音**自监督 | 与音频情感无关的预训练 → 下界对照 |
| **CLAP** | 音频–文本对比，63 万段 | 已有，但与环境声语料共线 |

`79` 逐族提取，本脚本**有几族跑几族**，缺的自动跳过并在结尾点名。

## 口径：与手工特征那张表逐字相同

    算法        `soundml.regression` 的六个，一字未改
    评分        Spearman ρ（各语料标度不同，秩相关才可比）
    训练/测试   源语料全量拟合 → 目标语料全量预测（与 `fig_domain_wall.csv` 同）
    域内参照    同语料 GroupKFold-5（按 group 分组，防泄漏）

**换脚本不换尺子。** 任何一处不同，与手工基线的比较就失去意义。

## 判据（写在跑之前）

墙高 = 1 − 跨域 ρ / 域内 ρ。

    某族的墙高**明显低于**手工特征（>20 个百分点） → 表示不足是主因，§8 要改写
    各族墙高与手工特征**相当**                      → 表示不足被排除
    某族跨域 ρ ≤ 0                                  → 墙在更强的表征上反而更高

**但更要紧的不是墙有多高，而是墙长在哪。** 同一个表征、同样是换语料：
域内换语料掉多少 vs 跨域换语料掉多少。若两者相差悬殊，
就排除了「这个嵌入本来就不迁移」，墙是**特定长在域边界上**的。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import N_SPLITS, SEED, build_regressors

warnings.filterwarnings("ignore")

FEAT = REPO_ROOT / "data" / "features"
OUT = REPO_ROOT / "reports" / "source_data" / "ast_domain_wall.csv"
CORPORA = ["deam", "pmemo", "soundtracks", "emo_mix"]
AMBIENT = "emo_mix"                       # 环境声域；其余三个是音乐域
FAMILIES = {"ast": "AST（AudioSet 监督）", "mert": "MERT（音乐自监督）",
            "w2v2": "W2V2（语音自监督）", "clap": "CLAP（音频–文本对比）"}
# 🔴 预先声明的污染：CLAP 的预训练集（LAION-Audio-630K）含 Freesound，
# 而 Emo-Soundscapes 正是从 Freesound 构建的。这不是事后找补 ——
# `79` 的 docstring 在任何结果出来之前就写着「CLAP 与环境声语料共线」，
# 跑另外三族的**全部理由**就是不信这一族。
# 判决必须把污染族与干净族分开算，否则「反例恰好是我早就不信的那个」
# 会变成一个方便的故事。
CONTAMINATED = {"clap": "预训练集含 Freesound，而 Emo-Soundscapes 由 Freesound 构建"}
MIN_DIM = 300                             # 各族维度 768/768/768/512
BASELINE = REPO_ROOT / "reports" / "source_data" / "fig_domain_wall.csv"
# 基线 CSV 用 `same-domain`，本脚本用 `within-domain` —— 同一概念两个词，都收。
SAME_DOMAIN = {"within-domain", "same-domain"}


def load(fam: str, corpus: str):
    """嵌入 + 标签。非特征列**从标签表推**，不硬编码排除表。

    🔴 本项目已五次栽在「硬编码一张排除表」上（最近一次是 `83` 的 `stim_c`）。
    """
    F = pd.read_parquet(FEAT / f"layer_f_{fam}_reg_{corpus}.parquet")
    L = pd.read_csv(REPO_ROOT / "data" / f"reg_{corpus}.csv")
    D = F.merge(L, on="clip_id", how="inner", suffixes=("", "_lab"))
    nonfeat = set(L.columns) | {"clip_id"} | {f"{c}_lab" for c in L.columns}
    cols = [c for c in D.columns
            if c not in nonfeat and pd.api.types.is_numeric_dtype(D[c])]
    assert len(cols) >= MIN_DIM, f"{fam}/{corpus} 维度异常 {len(cols)}"
    grp = D["group"].to_numpy() if "group" in D.columns else np.arange(len(D))
    return D.dropna(subset=["arousal"]), cols, grp


def within(D, cols, grp, model) -> float:
    X, y = D[cols].to_numpy(float), D["arousal"].to_numpy(float)
    g = grp[: len(D)]
    if len(np.unique(g)) < N_SPLITS:
        g = np.arange(len(D))
    pred = np.empty(len(y))
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(X, y, g):
        pred[te] = model.fit(X[tr], y[tr]).predict(X[te])
    return float(spearmanr(pred, y).statistic)


def across(Dtr, Dte, cols, model) -> float:
    m = model.fit(Dtr[cols].to_numpy(float), Dtr["arousal"].to_numpy(float))
    return float(spearmanr(m.predict(Dte[cols].to_numpy(float)),
                           Dte["arousal"].to_numpy(float)).statistic)


def wall(df: pd.DataFrame, col: str) -> dict:
    w = df[df.domain_match == "within-corpus"][col].median()
    x = df[df.domain_match == "cross-domain"][col].median()
    d = df[df.domain_match.isin(SAME_DOMAIN)][col].median()
    assert np.isfinite(d), (f"「域内跨语料」为空 —— domain_match 实际取值 "
                            f"{sorted(df.domain_match.unique())}")
    return {"within": w, "same_domain": d, "cross_domain": x,
            "wall": 1 - x / w if w > 0 else np.nan,
            "wall_same": 1 - d / w if w > 0 else np.nan}


def run_family(fam: str, rng_label: str) -> pd.DataFrame | None:
    data = {}
    for c in CORPORA:
        try:
            data[c] = load(fam, c)
        except FileNotFoundError:
            pass
    if len(data) < len(CORPORA):
        missing = [c for c in CORPORA if c not in data]
        print(f"  {rng_label:<22}尚缺 {missing} —— 跳过")
        return None
    print(f"  {rng_label:<22}"
          + "  ".join(f"{c}={len(data[c][0])}" for c in CORPORA))

    rows = []
    for algo, model in build_regressors().items():
        for src in CORPORA:
            Ds, cols_s, gs = data[src]
            rows.append({"family": fam, "algorithm": algo, "source": src,
                         "target": src, "rho": within(Ds, cols_s, gs, model),
                         "domain_match": "within-corpus"})
            for tgt in CORPORA:
                if tgt == src:
                    continue
                cols = [c for c in cols_s if c in data[tgt][1]]
                cross = (src == AMBIENT) != (tgt == AMBIENT)
                rows.append({"family": fam, "algorithm": algo, "source": src,
                             "target": tgt,
                             "rho": across(Ds, data[tgt][0], cols, model),
                             "domain_match": ("cross-domain" if cross
                                              else "within-domain")})
    return pd.DataFrame(rows)


def main() -> None:
    run = RunRecord("A1_ast_domain_wall", SEED)
    print("各族语料就绪情况")
    parts = []
    for fam, label in FAMILIES.items():
        R = run_family(fam, label)
        if R is not None:
            parts.append(R)
    if not parts:
        raise SystemExit("没有任何一族四个语料齐全")
    R = pd.concat(parts, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT, index=False, encoding="utf-8")

    print("\n" + "=" * 92)
    print("墙高 = 1 − 跨域 ρ / 域内 ρ   （越高＝墙越难翻）")
    print(f"  {'表征':<24}{'域内':>8}{'域内跨语料':>12}{'跨域':>8}"
          f"{'跨语料墙高':>12}{'跨域墙高':>10}")
    summary = {}
    for fam, label in FAMILIES.items():
        g = R[R.family == fam]
        if not len(g):
            continue
        summary[label] = wall(g, "rho")
        s = summary[label]
        print(f"  {label:<24}{s['within']:>8.3f}{s['same_domain']:>12.3f}"
              f"{s['cross_domain']:>8.3f}{s['wall_same']:>12.1%}"
              f"{s['wall']:>10.1%}")
        for k, v in s.items():
            run.log_metric(f"{fam}_{k}", float(v))
    base = None
    if BASELINE.exists():
        base = wall(pd.read_csv(BASELINE), "spearman")
        print(f"  {'手工 161 维（基线）':<24}{base['within']:>8.3f}"
              f"{base['same_domain']:>12.3f}{base['cross_domain']:>8.3f}"
              f"{base['wall_same']:>12.1%}{base['wall']:>10.1%}")

    # ── 判决：由数字定（纪律 #10） ───────────────────────────────────
    print("\n" + "=" * 92)
    print("判决")
    if base is None:
        print("  🔴 找不到手工基线 CSV —— 不下结论。")
        run.write()
        return

    print("  ⭐ 最要紧的不是墙有多高，而是**墙长在哪**。同一表征、同样换语料：")
    for label, s in list(summary.items()) + [("手工 161 维（基线）", base)]:
        # 🔴 gap 是**分数**，要 ×100 才是百分点。首版写 `{gap:>5.1f}`，
        # 把 0.622 印成「0.6 个百分点」—— 而 62 个百分点的落差
        # 正是「墙只长在域边界上」的全部证据，印成 0.6 反而像没差别。
        gap = (s["wall"] - s["wall_same"]) * 100
        print(f"     {label:<24}域内换 {s['wall_same']:>6.1%}   "
              f"跨域换 {s['wall']:>6.1%}   相差 {gap:>5.1f} 个百分点")
    print("     ⇒ 若各族都在同一处塌陷，就排除了「该嵌入本来就不迁移」——")
    print("       墙是特定长在域边界上的，不是一般意义的迁移损失。")

    print()
    # 判决分**干净族**与**污染族**两栏 —— 见 CONTAMINATED 的说明
    fam_of = {l: f for f, l in FAMILIES.items()}
    clean = {l: s for l, s in summary.items() if fam_of[l] not in CONTAMINATED}
    dirty = {l: s for l, s in summary.items() if fam_of[l] in CONTAMINATED}
    low_clean = [(l, base["wall"] - s["wall"]) for l, s in clean.items()
                 if base["wall"] - s["wall"] > 0.20]
    if low_clean:
        print("  🔴 **干净族**里有把墙明显压低的（>20 个百分点）——")
        print("     表示不足是主因，「墙」的框架必须改写：")
        for l, d in low_clean:
            print(f"       {l}  低 {d:.1%}")
    else:
        print("  → **干净族**的墙与手工特征高度相当（差 ≤20 个百分点）：")
        for l, s in clean.items():
            print(f"       {l:<22}墙高 {s['wall']:>6.1%}"
                  f"（手工 {base['wall']:.1%}）")
        print("    「161 维手工特征不够，换个大模型就行」这条反驳被逐族堵死。")
        if "ast" in R.family.values:
            print("    尤其 AST 是**最该翻过去**的那个（AudioSet 监督预训练，")
            print("    明确覆盖环境声类别），它也没翻过去。")

    for l, s in dirty.items():
        d = base["wall"] - s["wall"]
        print(f"\n  ⚠️ **污染族** {l}：墙高 {s['wall']:.1%}，比手工低 {d:.1%}。")
        print(f"     污染来源：{CONTAMINATED[fam_of[l]]}")
        gen = s["same_domain"] / max(base["same_domain"], 1e-9)
        print(f"     它是不是「总体更好的表征」？在**不含环境声语料**的")
        print(f"     音乐↔音乐迁移上，它是 {s['same_domain']:.3f}，"
              f"手工是 {base['same_domain']:.3f}（{gen:.2f}×）——")
        print(f"     {'没有优势' if abs(gen-1) < 0.10 else '有优势'}。"
              f"优势只出现在必然含该语料的那一列。")
        print("     🔴 **但这条判别有极限，必须照实写**：环境声语料只有一个，")
        print("     「跨域」按定义必然含它，所以「真能跨域」与「记住了目标语料」")
        print("     在本数据上**无法分开**。故：墙的结论建立在干净族上，")
        print("     污染族照报、标注、但既不用作支持也不用作反驳。")
    # 别把「没改善」说过头
    for label, s in summary.items():
        if s["cross_domain"] > base["cross_domain"] * 1.5:
            print(f"\n  ⚠️ 但 {label} 的跨域 ρ 是 {s['cross_domain']:.3f}，"
                  f"手工是 {base['cross_domain']:.3f} —— "
                  f"{s['cross_domain']/base['cross_domain']:.1f} 倍。")
            print(f"     正文不得写「完全没有改善」，该写「抬到 "
                  f"{s['cross_domain']:.2f}，仍不足域内的 "
                  f"{s['cross_domain']/s['within']:.0%}」。")

    miss = [l for f, l in FAMILIES.items() if f not in R.family.values]
    if miss:
        print(f"\n  ⚠️ 尚缺 {len(miss)} 族：{miss} —— 提取完须补跑本脚本。")
    run.write()
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
