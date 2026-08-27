r"""域之墙能不能被领域自适应翻过去 —— 本文核心发现的存亡检验。

    .venv/Scripts/python scripts/80_domain_adaptation.py
    .venv/Scripts/python scripts/80_domain_adaptation.py --reps a,c,ast,mert,w2v2,clap10

## 这是唯一可能推翻核心发现的检验

论文说「音乐上训练的模型搬到环境声上就废了，且这不是特征设计不足」。
但**从未尝试过任何领域自适应**。审稿人的问法一针见血：

> 这堵墙是真的不可逾越，还是只是没做适配？

若 CORAL 这种最简单的协方差对齐就能翻过去，那么「墙」这个框架本身就错了 ——
真正的现象只是协变量偏移，那是个已被解决的问题，不构成发现。

## 三类适配，按「需要多少目标域信息」排序

| 方法 | 需要目标域的什么 | 问的是什么 |
|---|---|---|
| **none** | 无 | 原始跨域转移（论文现状） |
| **z-score** | 目标域的**特征**（无标签） | 是否只是尺度/中心偏移 |
| **CORAL** | 目标域的**特征协方差**（无标签） | 是否只是二阶协变量偏移 |
| **few-shot k** | 目标域的 k 个**标签** | 墙的代价折合多少标注 |

前三种是无监督的，是「墙是否为真」的公平检验。
第四种把结论从「翻不过去」变成**「翻过去要花 k 个标签」**——
后者可证伪、可比较，也直接对应产品要不要为新声音类型标注数据。

## CORAL

Sun & Saenko (2016)：把源域特征白化后，再用目标域的协方差重新着色。

    C_s = cov(X_s) + λI ,  C_t = cov(X_t) + λI
    X_s' = X_s · C_s^(-1/2) · C_t^(1/2)

闭式解，不需要目标标签，不需要训练。它是「协变量偏移」这一解释的
最直接检验：若墙只是协变量偏移，CORAL 应当大幅缩小它。

## 判据先写在这里

- 若任一无监督适配把跨域 ρ 推到**域内水平的一半以上**，
  则「墙」的表述必须改为「协变量偏移」，论文核心声称须重写。
- 若全部无监督适配都无实质改善（Δρ < 0.05），
  则墙的声称**得到加强**，且加强的程度远超现有版本。
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")

FEAT = REPO_ROOT / "data" / "features"
OUT = REPO_ROOT / "reports" / "source_data" / "domain_adaptation.csv"
SEED = 20260731
CORPORA = {"reg_deam": "music", "reg_pmemo": "music",
           "reg_soundtracks": "music", "reg_emo_mix": "ambient"}
FEWSHOT_K = [0, 10, 25, 50, 100, 200]
CORAL_LAMBDA = 1.0
N_SA_COMP = 40       # 子空间对齐保留的主成分数
N_FEWSHOT_REP = 20   # few-shot 每格重抽次数：单次抽样方差过大
N_BOOT = 200

# 🔴 特征列不能靠「排除已知元数据」来定
#
# 本项目已经三次因为这个写法把标签放进特征矩阵：Layer D、Layer C，
# 以及本脚本 —— Soundtracks 的特征表里带着 `energy`（就是它的唤醒度量表）
# 与另外七个情绪维度，全是数值型，全会被「未知列即特征」的规则收进去。
#
# 改为两道防线：
#   1. 减去**任何标签 CSV 里出现过的列名**（不只本语料的）
#   2. 取各语料特征列的**交集** —— 跨域转移本就要求列对齐，
#      交集之外的列一律是某语料独有的元数据或标签
LABEL_COLS = {"clip_id", "path", "arousal", "valence", "group", "source",
              "category", "corpus", "split", "label", "musicId", "stim",
              "target_emotion"}


def _label_cols() -> set[str]:
    cols = set(LABEL_COLS)
    for p in (REPO_ROOT / "data").glob("reg_*.csv"):
        cols |= set(pd.read_csv(p, nrows=1).columns)
    return cols


NONFEAT = _label_cols()


def load_rep(rep: str, corpus: str) -> pd.DataFrame | None:
    """取某表征在某语料上的特征表，附 arousal 标签。"""
    paths = {
        "a": FEAT / f"layer_a_{corpus}_ln.parquet",
        "c": FEAT / f"layer_c_{corpus}.parquet",
        "d": FEAT / f"layer_d_{corpus}_ln.parquet",
    }
    p = paths.get(rep) or FEAT / f"layer_f_{rep}_{corpus}.parquet"
    if not p.exists():
        return None
    D = pd.read_parquet(p)
    # Layer A/D 的特征表自带 arousal / source（这也正是标签泄漏的来源，
    # 由 NONFEAT 排除）；layer_f 的嵌入表只有 clip_id，需要外接标签。
    # 无条件 merge 会产生 arousal_x / arousal_y，之后 `D["arousal"]` 直接 KeyError。
    if "arousal" not in D.columns:
        lab = pd.read_csv(REPO_ROOT / "data" / f"{corpus}.csv")[
            ["clip_id", "arousal", "source", "group"]]
        D = D.merge(lab, on="clip_id", how="inner")
    assert {"arousal", "source"} <= set(D.columns), \
        f"{rep}/{corpus} 缺标签列：{set(D.columns) & {'arousal', 'source'}}"
    return D


def feature_cols(tables: dict[str, pd.DataFrame]) -> list[str]:
    """各语料数值列的交集，减去所有标签列。跨域转移要求列完全对齐。"""
    sets = [{c for c in D.columns
             if c not in NONFEAT and pd.api.types.is_numeric_dtype(D[c])}
            for D in tables.values()]
    common = sorted(set.intersection(*sets))
    dropped = sorted(set.union(*sets) - set(common))
    if dropped:
        print(f"  非共有列已剔除 {len(dropped)} 个: {dropped[:8]}"
              f"{' …' if len(dropped) > 8 else ''}")
    assert common, "特征列交集为空"
    return common


def xy(D: pd.DataFrame, fc: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.nan_to_num(D[fc].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    # `group` 在本项目里是逐段的源录音标识；`source` 是语料名（每语料只有一个值），
    # 拿后者做 GroupKFold 会直接报「只有 1 个组」。
    return X, D["arousal"].to_numpy(float), D["group"].to_numpy()


def coral(Xs: np.ndarray, Xt: np.ndarray, lam: float = CORAL_LAMBDA,
          diag_only: bool = False) -> np.ndarray:
    """把源域特征的二阶统计对齐到目标域（Sun & Saenko 2016）。

    `diag_only=True` 只匹配每维**方差**，不匹配相关结构。两者之差即
    「相关结构对齐」的独立贡献。

    ## 为什么要这个分解 🔴

    原本用「对齐到第三方语料」做 sham 来问「增益是不是只来自白化」。
    但本项目只有**一个**环境声语料：当源是环境声、目标是音乐时，
    第三方也只能是音乐语料 —— 那个 sham 其实对齐到了正确的域，
    根本不构成对照。6 个跨域对里有 3 个如此，会把 sham 抬高。

    对角版本没有这个问题，且总是可用：

        none      → coral_diag   匹配各维方差（尺度对齐）
        coral_diag → coral       额外匹配相关结构

    两段都是目标特异的，都算领域自适应；但若增益几乎全在第一段，
    说明所谓「适配」只是尺度对齐这种浅层操作，结论的分量完全不同。
    """
    d = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + lam * np.eye(d)
    Ct = np.cov(Xt, rowvar=False) + lam * np.eye(d)
    if diag_only:
        Cs, Ct = np.diag(np.diag(Cs)), np.diag(np.diag(Ct))
    W = np.real(fractional_matrix_power(Cs, -0.5) @
                fractional_matrix_power(Ct, 0.5))
    return (Xs - Xs.mean(0)) @ W + Xt.mean(0)


def subspace_align(Xs: np.ndarray, Xt: np.ndarray, k: int = N_SA_COMP
                   ) -> tuple[np.ndarray, np.ndarray]:
    """子空间对齐（Fernando et al. 2013）：把源域主子空间旋转到目标域主子空间。

    与 CORAL 并列的另一条标准无监督路线，同样闭式、同样不需要目标标签。
    多报一条是为了让「我们试过标准做法」这句话站得住 ——
    只试一种方法而下「不可逾越」的结论，是审稿人第一个会攻的点。
    """
    from sklearn.decomposition import PCA
    k = min(k, Xs.shape[1], Xs.shape[0] - 1, Xt.shape[0] - 1)
    mu = Xs.mean(0)
    Ps = PCA(k).fit((Xs - mu)).components_.T
    Pt = PCA(k).fit((Xt - Xt.mean(0))).components_.T
    M = Ps.T @ Pt                       # 源子空间 → 目标子空间的旋转
    return (Xs - mu) @ Ps @ M, (Xt - Xt.mean(0)) @ Pt


# 🔴 不做「择优报告」
#
# 第一版用**训练集拟合度**在 Ridge 与随机森林之间选。随机森林的训练拟合
# 永远接近 1，等于恒选 RF —— 既是错的选择规则，也正是审稿意见 R3-m4
# 批评的「多处重复取最优会引入乐观偏差」。
#
# 改为**两个都算、两个都报**。跨域比较尤其不能择优：线性模型常比树模型
# 更能迁移，用一个被污染的选择规则会把这个差别抹掉。
MODELS = {
    "ridge": lambda: make_pipeline(StandardScaler(),
                                   RidgeCV(alphas=np.logspace(-2, 4, 25))),
    "rf": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                        random_state=SEED, n_jobs=-1),
}


def fit_predict_all(Xtr, ytr, Xte) -> dict[str, np.ndarray]:
    out = {}
    for name, mk in MODELS.items():
        m = mk()
        m.fit(Xtr, ytr)
        out[name] = m.predict(Xte)
    return out


def fit_predict(Xtr, ytr, Xte) -> np.ndarray:
    """向后兼容：主模型取 ridge（跨域比较的标准选择）。"""
    return fit_predict_all(Xtr, ytr, Xte)["ridge"]


def boot_ci(pred: np.ndarray, y: np.ndarray, rng, n=N_BOOT) -> tuple[float, float]:
    rs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        r = spearmanr(pred[i], y[i]).statistic
        if np.isfinite(r):
            rs.append(r)
    return (float(np.quantile(rs, 0.025)), float(np.quantile(rs, 0.975))) \
        if rs else (np.nan, np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="a,c")
    args = ap.parse_args()
    reps = args.reps.split(",")
    rng = np.random.default_rng(SEED)
    run = RunRecord("80_domain_adaptation", SEED)

    rows = []
    for rep in reps:
        data = {c: load_rep(rep, c) for c in CORPORA}
        have = {c: d for c, d in data.items() if d is not None}
        if len(have) < 2:
            print(f"\n[{rep}] 特征表不足（{len(have)}/4），跳过")
            continue
        print(f"\n{'='*80}\n表征 {rep}   语料 {list(have)}")
        fc = feature_cols(have)
        print(f"  共有特征 {len(fc)} 维")

        for src, Ds in have.items():
            Xs, ys, _ = xy(Ds, fc)
            for tgt, Dt in have.items():
                if src == tgt:
                    continue
                Xt, yt, _ = xy(Dt, fc)
                cross = CORPORA[src] != CORPORA[tgt]

                # `coral_sham`：把源域对齐到一个**与目标无关的第三方语料**，
                # 再去预测目标。CORAL 同时做了「白化」与「对齐到目标」两件事，
                # 只有减去 sham 才知道哪部分真属于领域自适应。
                # `81` 在 Layer A 上测得：全部增益的 74% 来自白化，与目标无关。
                third = next((c for c in have if c not in (src, tgt)), None)
                for meth in ("none", "zscore", "coral_diag", "coral",
                             "coral_sham", "sa"):
                    if meth == "none":
                        A, B = Xs, Xt
                    elif meth == "zscore":
                        A = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-9)
                        B = (Xt - Xt.mean(0)) / (Xt.std(0) + 1e-9)
                    elif meth == "coral_diag":
                        A, B = coral(Xs, Xt, diag_only=True), Xt
                    elif meth == "coral":
                        A, B = coral(Xs, Xt), Xt
                    elif meth == "coral_sham":
                        if third is None:
                            continue
                        A, B = coral(Xs, xy(have[third], fc)[0]), Xt
                    else:
                        A, B = subspace_align(Xs, Xt)
                    preds = fit_predict_all(A, ys, B)
                    for mname, pred in preds.items():
                        r = spearmanr(pred, yt).statistic
                        lo, hi = boot_ci(pred, yt, rng)
                        rows.append({"rep": rep, "model": mname, "source": src,
                                     "target": tgt, "cross_domain": cross,
                                     "method": meth, "k": 0, "rho": float(r),
                                     "ci_lo": lo, "ci_hi": hi})

                # few-shot：把 k 个目标域样本加进训练集
                for k in FEWSHOT_K:
                    if k == 0 or k >= len(yt) // 2:
                        continue
                    # 单次抽样方差过大 —— 重抽 N_FEWSHOT_REP 次取均值，
                    # 并记下 s.d.，否则曲线上的每个点都不可信。
                    rs = []
                    for _ in range(N_FEWSHOT_REP):
                        idx = rng.permutation(len(yt))
                        tr, te = idx[:k], idx[k:]
                        pred = fit_predict(np.vstack([Xs, Xt[tr]]),
                                           np.concatenate([ys, yt[tr]]), Xt[te])
                        r = spearmanr(pred, yt[te]).statistic
                        if np.isfinite(r):
                            rs.append(r)
                    if not rs:
                        continue
                    rows.append({"rep": rep, "model": "ridge", "source": src,
                                 "target": tgt,
                                 "cross_domain": cross, "method": "fewshot",
                                 "k": k, "rho": float(np.mean(rs)),
                                 "ci_lo": float(np.mean(rs) - np.std(rs)),
                                 "ci_hi": float(np.mean(rs) + np.std(rs))})

            # 域内基准：同语料 5 折，两个模型都记 —— 跨域数要和同一个模型的
            # 域内数比，否则「占域内百分之几」这个比值就是错的。
            from sklearn.model_selection import GroupKFold
            g = Ds["group"].to_numpy()
            preds = {m: np.full(len(ys), np.nan) for m in MODELS}
            for tr, te in GroupKFold(5).split(Xs, ys, g):
                for m, p in fit_predict_all(Xs[tr], ys[tr], Xs[te]).items():
                    preds[m][te] = p
            msg = []
            for m, pv in preds.items():
                r = spearmanr(pv, ys).statistic
                rows.append({"rep": rep, "model": m, "source": src,
                             "target": src, "cross_domain": False,
                             "method": "within", "k": 0, "rho": float(r),
                             "ci_lo": np.nan, "ci_hi": np.nan})
                msg.append(f"{m} {r:+.3f}")
            print(f"  {src:<18} 域内 ρ = {'   '.join(msg)}", flush=True)

    R = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT, index=False, encoding="utf-8")

    # ══ 判据 ═════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("跨域转移：无监督适配有没有把墙推倒")
    for rep, mname in [(r, m) for r in R.rep.unique()
                       for m in sorted(R.model.dropna().unique())]:
        S = R[(R.rep == rep) & (R.model == mname)]
        if not len(S) or not len(S[S.method == "within"]):
            continue
        within = S[S.method == "within"].rho.mean()
        print(f"\n  [{rep} / {mname}]  域内基准 ρ = {within:+.3f}")
        got = {}
        for meth in ("none", "zscore", "coral_diag", "coral",
                     "coral_sham", "sa"):
            x = S[(S.method == meth) & S.cross_domain]
            if not len(x):
                continue
            m = float(x.rho.mean())
            got[meth] = m
            print(f"    {meth:<11} 跨域 ρ = {m:+.3f}   "
                  f"（域内的 {m/within:.0%}）"
                  f"{'   🔴 超过域内一半，墙的表述须重写' if m > within/2 else ''}")
            run.log_metric(f"{rep}_{mname}_cross_{meth}", m)
        base = got.get("none", np.nan)
        # 把 CORAL 的增益拆成「白化」与「对齐到目标」两部分。
        # 不拆就会把特征预处理的效果当成领域自适应报出去。
        if {"coral", "coral_diag"} <= got.keys():
            sc = got["coral_diag"] - base            # 尺度（各维方差）对齐
            co = got["coral"] - got["coral_diag"]    # 相关结构对齐
            print(f"    ├ 尺度对齐（仅各维方差）Δ = {sc:+.3f}")
            print(f"    └ 相关结构对齐          Δ = {co:+.3f}"
                  f"   （占全部增益的 {co/max(sc+co,1e-9):.0%}）")
            run.log_metric(f"{rep}_{mname}_gain_scale", sc)
            run.log_metric(f"{rep}_{mname}_gain_correlation", co)
        if "coral_sham" in got:
            print(f"    （第三方 sham {got['coral_sham']:+.3f}；"
                  f"仅 3/6 对的第三方处于错误域，故此列只作参考）")
        run.log_metric(f"{rep}_{mname}_within", float(within))

    print("\n" + "=" * 80)
    print("few-shot：翻过这堵墙要花多少个目标域标签")
    for rep in R.rep.unique():
        S = R[(R.rep == rep) & R.cross_domain & (R.model == "ridge")]
        within = R[(R.rep == rep) & (R.method == "within")
                   & (R.model == "ridge")].rho.mean()
        base = S[S.method == "none"].rho.mean()
        print(f"\n  [{rep}]  k=0 (跨域) {base:+.3f}  →  域内 {within:+.3f}")
        for k in FEWSHOT_K[1:]:
            x = S[(S.method == "fewshot") & (S.k == k)]
            if len(x):
                m = x.rho.mean()
                print(f"    k = {k:>3}   ρ = {m:+.3f}   "
                      f"（补回缺口的 {(m-base)/max(within-base,1e-9):.0%}）")
                run.log_metric(f"{rep}_fewshot_{k}", float(m))
    run.write()
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
