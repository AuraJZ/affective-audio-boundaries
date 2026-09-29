r"""投稿前重算：环境声语料的混合泄漏，以及被归错条件的学习曲线。

    SOUNDML_DATA=<数据目录> .venv/Scripts/python scripts/A9_tac_recheck.py

数据目录默认为 `<repo>/data`；若特征表存放在仓库之外（公开发布时原始音频与特征
不随仓库分发），用环境变量 `SOUNDML_DATA` 指向它。

## 为什么要重算

### 一、Emo-Soundscapes 的一半是 30 段录音的混合

Emo-Soundscapes 的 1,213 段里，613 段是 `613_MixedSounds`，由每类 5 段、共
**30 段**源录音两两或三三混合而成，每对源录音再按两档电平组合。文件名把成分
编码在里面（`mix_between_human_indicator_1-12_1-6` = human#1 −12 dB + indicator#1 −6 dB），
据此可查出：613 段混合只有 **161 组不同的成分集合**，平均每组 3.8 段。

标签表给每段各自一个 `group`，于是 `GroupKFold` 退化成普通 `KFold`：

- 同两段源录音、只差混音电平的 3–8 段近重复样本会被切到不同折；
- 训练折里的原始段可以是测试折里那段混合的成分。

正文《Models and evaluation》写的是「groups defined by source recording, so that
no excerpt from a recording appears in both training and test folds」。在这个语料上
这句话不成立 —— 保护措施没有生效。

所以域内上限（论文报 0.744）被高估，而它是**跨域损失百分比的分母**，也是
few-shot 缺口回收率的分母。

干净对照：只用 600 段原始录音重算。每段是一个不同的 Freesound 录音，构造性泄漏
不可能发生。

### 二、few-shot 曲线画的是跨域对，正文却写成同域对

`domain_adaptation.csv` 的 `cross_domain=True` 是音乐↔环境声，`False` 是音乐→音乐。
`figures_r/fig_price.R` 筛的是 `cross_domain`（即 True），图注也写明「the quantity
priced here is the crossing of the music/environmental boundary」。但摘要与讨论把
「100 个目标标签补回三分之二」说成了「a new corpus of the same kind」「within a domain」。

两条曲线数据里都有，本脚本分别报出来。

### 三、分母不是同一个

`fig_price.R` 的 ceiling 取自 `fig_domain_wall.csv` 的域内值中位数 —— 六个算法的
中位数；而分子只有 RidgeCV 一个模型。`domain_adaptation.csv` 自带一列
`method=="within"` 是逐模型的域内值，与分子同模型。本脚本两个分母都算。

### 四、k=200 那一点换了对子

`80_domain_adaptation.py` 跳过 `k >= len(yt)//2`，所以 Soundtracks（n=360）在 k=200
上没有数据。跨域 6 对里少 1 对、同域 6 对里少 2 对。「每翻一倍的回报在 100 之后
塌掉」这句话因此混进了对子组成的变化。本脚本另报一条只用「每个 k 都在场的对子」
的曲线。

### 五、CORAL 的 sham 对照有一半是无效的

`80_domain_adaptation.py` 自己的注释已经写明：只有一个环境声语料，当源是环境声、
目标是音乐时，「无关的第三方语料」也只能是音乐语料 —— 那恰好是目标所在的域，
不构成负对照。6 个跨域对里有 3 个如此。正文却写 sham「cannot carry target-specific
information by construction」，并据此把 CORAL 增益的三分之二判为非目标特异。
本脚本把 sham 按有效/无效分开报，并算出脚本推荐的 `coral_diag → coral` 分解。

输出 `reports/source_data/tac_recheck_*.csv`，供正文与图表引用。
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soundml.regression import build_regressors  # noqa: E402

warnings.filterwarnings("ignore")

# 这个脚本的输出里有中文。Windows 控制台默认 cp1252，管道重定向时更是如此，
# 一个字就能让整次重算在打印那一行崩掉 —— 而且是在算完之后崩。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("SOUNDML_DATA", REPO / "data"))
OUT = REPO / "reports" / "source_data"
SEED = 20260731
N_FEWSHOT_REP = 20
FEWSHOT_K = [10, 25, 50, 100, 200]
CORAL_LAMBDA = 1.0

# 与 80_domain_adaptation.py 一致：任何标签 CSV 出现过的列名都不是特征。
LABEL_COLS = {"clip_id", "path", "arousal", "valence", "group", "source",
              "category", "corpus", "split", "label", "musicId", "stim",
              "target_emotion"}

CORPORA = {"reg_deam": "music", "reg_pmemo": "music",
           "reg_soundtracks": "music", "reg_emo_mix": "ambient"}


# ───────────────────────────────────────────────────────── 载入与特征对齐 ──
def label_cols() -> set[str]:
    cols = set(LABEL_COLS)
    for p in DATA.glob("reg_*.csv"):
        cols |= set(pd.read_csv(p, nrows=1).columns)
    return cols


def load(corpus: str) -> pd.DataFrame:
    D = pd.read_parquet(DATA / "features" / f"layer_a_{corpus}_ln.parquet")
    need = [c for c in ("arousal", "group", "category") if c not in D.columns]
    if need:
        lab = pd.read_csv(DATA / f"{corpus}.csv")[["clip_id", *need]]
        D = D.merge(lab, on="clip_id", how="inner")
    return D


def feature_cols(tables: dict[str, pd.DataFrame], nonfeat: set[str]) -> list[str]:
    sets = [{c for c in D.columns
             if c not in nonfeat and pd.api.types.is_numeric_dtype(D[c])}
            for D in tables.values()]
    return sorted(set.intersection(*sets))


def xyg(D: pd.DataFrame, fc: list[str]):
    X = np.nan_to_num(D[fc].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    return X, D["arousal"].to_numpy(float), D["group"].to_numpy()


# ─────────────────────────────────────────── Emo-Soundscapes 的混合成分 ──
def constituents(clip_id: str) -> list[str]:
    """一段混合用到的 (类别, 序号) 源录音；原始录音返回空表。"""
    if not clip_id.startswith("emo_mix_"):
        return []
    parts = clip_id[len("emo_mix_"):].split("_")
    kind, parts = parts[0], parts[1:]
    cats, i = [], 0
    while i < len(parts) and not re.match(r"^\d+-\d+$", parts[i]):
        cats.append(parts[i])
        i += 1
    nums = [int(p.split("-")[0]) for p in parts[i:]]
    if kind == "within":          # 同类内混合，类别名只写一次
        cats = cats * len(nums)
    return [f"{c}#{n}" for c, n in zip(cats, nums)]


def describe_emo(E: pd.DataFrame) -> pd.DataFrame:
    is_mix = E.clip_id.str.startswith("emo_mix_")
    cons = {c: constituents(c) for c in E.clip_id[is_mix]}
    sources = sorted({x for v in cons.values() for x in v})
    sets: dict[tuple, list[str]] = {}
    for c, v in cons.items():
        sets.setdefault(tuple(sorted(v)), []).append(c)
    n = pd.Series([len(v) for v in sets.values()])
    print(f"\nEmo-Soundscapes: {len(E)} 段 = {(~is_mix).sum()} 原始 + {is_mix.sum()} 混合")
    print(f"  {is_mix.sum()} 段混合出自 {len(sources)} 段源录音"
          f"（每类 {len(sources)//6} 段 × 6 类）")
    print(f"  不同成分集合 {len(sets)} 组；每组 {n.mean():.2f} 段（最多 {n.max()}）")
    print(f"  {(n > 1).sum()} 组含 2 段以上近重复，覆盖 {n[n > 1].sum()} 段"
          f"（占混合的 {n[n > 1].sum() / is_mix.sum():.0%}）")
    print(f"  标签表里 group 的取值数 = {E['group'].nunique()}，等于段数 —— "
          f"GroupKFold 在此语料上退化为 KFold")
    return pd.DataFrame({
        "quantity": ["clips", "originals", "mixtures", "distinct_source_recordings",
                     "distinct_constituent_sets", "mean_clips_per_set",
                     "max_clips_per_set", "distinct_group_values"],
        "value": [len(E), int((~is_mix).sum()), int(is_mix.sum()), len(sources),
                  len(sets), round(float(n.mean()), 3), int(n.max()),
                  int(E["group"].nunique())],
    })


# ────────────────────────────────────────────────────────────── 域内上限 ──
def within_corpus(D: pd.DataFrame, fc: list[str], tag: str) -> pd.DataFrame:
    """六个登记算法的域内 GroupKFold-5，外加 RidgeCV。

    RidgeCV 不在登记表里，但 few-shot 的分子就是它，而分母若取六算法的中位数
    就和分子不是同一个模型。两个都算出来，才能说清那个比值到底在比什么。
    """
    X, y, g = xyg(D, fc)
    algos = dict(build_regressors())
    algos["RidgeCV"] = None          # 占位，下面单独构造
    rows = []
    for name in algos:
        pv = np.full(len(y), np.nan)
        for tr, te in GroupKFold(5).split(X, y, g):
            m = ridge() if name == "RidgeCV" else build_regressors()[name]
            m.fit(X[tr], y[tr])
            pv[te] = m.predict(X[te])
        rows.append({"corpus": tag, "algorithm": name,
                     "rho": float(spearmanr(pv, y).statistic)})
    R = pd.DataFrame(rows)
    six = R[R.algorithm != "RidgeCV"]
    print(f"  {tag:<28} 六算法 median {six.rho.median():+.3f}  best {six.rho.max():+.3f}"
          f"   RidgeCV {float(R[R.algorithm == 'RidgeCV'].rho.iloc[0]):+.3f}")
    print(f"      {', '.join(f'{r.algorithm} {r.rho:+.3f}' for r in R.itertuples())}")
    return R


def transfer(Ds: pd.DataFrame, Dt: pd.DataFrame, fc: list[str],
             src: str, tgt: str, match: str) -> pd.DataFrame:
    Xs, ys, _ = xyg(Ds, fc)
    Xt, yt, _ = xyg(Dt, fc)
    rows = []
    for name, mk in build_regressors().items():
        mk.fit(Xs, ys)
        rows.append({"source": src, "target": tgt, "domain_match": match,
                     "algorithm": name,
                     "rho": float(spearmanr(mk.predict(Xt), yt).statistic)})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────── CORAL 与 sham ──
def coral(Xs, Xt, lam=CORAL_LAMBDA, diag_only=False):
    d = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + lam * np.eye(d)
    Ct = np.cov(Xt, rowvar=False) + lam * np.eye(d)
    if diag_only:
        Cs, Ct = np.diag(np.diag(Cs)), np.diag(np.diag(Ct))
    W = np.real(fractional_matrix_power(Cs, -0.5) @ fractional_matrix_power(Ct, 0.5))
    return (Xs - Xs.mean(0)) @ W + Xt.mean(0)


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))


def main() -> None:
    if not (DATA / "features").is_dir():
        sys.exit(f"找不到特征目录 {DATA / 'features'}；请设置 SOUNDML_DATA")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    nonfeat = label_cols()

    raw = {c: load(c) for c in CORPORA}
    fc = feature_cols(raw, nonfeat)
    print(f"四语料共有描述子 {len(fc)} 维")

    emo_stats = describe_emo(raw["reg_emo_mix"])
    emo_stats.to_csv(OUT / "tac_recheck_emo_structure.csv", index=False)

    is_mix = raw["reg_emo_mix"].clip_id.str.startswith("emo_mix_")
    emo_orig = raw["reg_emo_mix"][~is_mix].reset_index(drop=True)

    # 两套语料：published = 论文用的；clean = 环境声只留原始录音
    published = dict(raw)
    clean = dict(raw)
    clean["reg_emo_mix"] = emo_orig

    print("\n" + "=" * 78)
    print("域内上限（GroupKFold-5，六个算法）")
    print("=" * 78)
    wc = []
    for variant, tables in [("published", published), ("clean", clean)]:
        print(f"\n[{variant}]")
        for c, D in tables.items():
            if variant == "clean" and c != "reg_emo_mix":
                continue   # 只有环境声语料变了，音乐语料不必重算
            R = within_corpus(D, fc, f"{c} ({variant})")
            R["variant"] = variant
            R["corpus"] = c
            wc.append(R)
    WC = pd.concat(wc, ignore_index=True)
    # 音乐语料在两个变体下相同，补齐 clean 行以便下游取用
    music = WC[(WC.variant == "published") & (WC.corpus != "reg_emo_mix")].copy()
    music["variant"] = "clean"
    WC = pd.concat([WC, music], ignore_index=True)
    WC.to_csv(OUT / "tac_recheck_within_corpus.csv", index=False)

    print("\n" + "=" * 78)
    print("转移矩阵：12 个有序对 × 六个算法")
    print("=" * 78)
    tr = []
    for variant, tables in [("published", published), ("clean", clean)]:
        for src, Ds in tables.items():
            for tgt, Dt in tables.items():
                if src == tgt:
                    continue
                match = ("same-domain" if CORPORA[src] == CORPORA[tgt]
                         else "cross-domain")
                T = transfer(Ds, Dt, fc, src, tgt, match)
                T["variant"] = variant
                tr.append(T)
    TR = pd.concat(tr, ignore_index=True)
    TR.to_csv(OUT / "tac_recheck_transfer.csv", index=False)

    for variant in ("published", "clean"):
        S = TR[TR.variant == variant]
        w = WC[(WC.variant == variant) & (WC.algorithm != "RidgeCV")].rho.median()
        print(f"\n[{variant}]  域内中位数 {w:+.3f}")
        for match in ("same-domain", "cross-domain"):
            x = S[S.domain_match == match].rho
            print(f"  {match:<13} mean {x.mean():+.3f} (s.d. {x.std():.3f})  "
                  f"median {x.median():+.3f}  n={len(x)}   "
                  f"损失（对域内中位数）{1 - x.median() / w:6.1%}")

    print("\n" + "=" * 78)
    print("few-shot：同域 vs 跨域，两种分母，含/不含 k=200 掉队的对子")
    print("=" * 78)
    fs_rows = []
    for variant, tables in [("published", published), ("clean", clean)]:
        ceil_med = (WC[(WC.variant == variant) & (WC.algorithm != "RidgeCV")]
                    .groupby("corpus").rho.median())
        ceil_ridge = (WC[(WC.variant == variant) & (WC.algorithm == "RidgeCV")]
                      .set_index("corpus").rho)          # 与 few-shot 分子同模型
        for src, Ds in tables.items():
            Xs, ys, _ = xyg(Ds, fc)
            for tgt, Dt in tables.items():
                if src == tgt:
                    continue
                Xt, yt, _ = xyg(Dt, fc)
                match = ("same-domain" if CORPORA[src] == CORPORA[tgt]
                         else "cross-domain")
                m0 = ridge()
                m0.fit(Xs, ys)
                zero = float(spearmanr(m0.predict(Xt), yt).statistic)
                for k in FEWSHOT_K:
                    if k >= len(yt) // 2:
                        continue
                    rs = []
                    for _ in range(N_FEWSHOT_REP):
                        idx = rng.permutation(len(yt))
                        a, b = idx[:k], idx[k:]
                        m = ridge()
                        m.fit(np.vstack([Xs, Xt[a]]), np.concatenate([ys, yt[a]]))
                        r = spearmanr(m.predict(Xt[b]), yt[b]).statistic
                        if np.isfinite(r):
                            rs.append(r)
                    if not rs:
                        continue
                    fs_rows.append({
                        "variant": variant, "source": src, "target": tgt,
                        "domain_match": match, "k": k,
                        "rho": float(np.mean(rs)), "sd": float(np.std(rs)),
                        "zero_shot": zero,
                        "ceiling_median6": float(ceil_med[tgt]),
                        "ceiling_linear": float(ceil_ridge[tgt]),
                    })
    FS = pd.DataFrame(fs_rows)
    for col, ce in [("frac_median6", "ceiling_median6"),
                    ("frac_linear", "ceiling_linear")]:
        FS[col] = (FS.rho - FS.zero_shot) / (FS[ce] - FS.zero_shot)
    FS.to_csv(OUT / "tac_recheck_fewshot.csv", index=False)

    for variant in ("published", "clean"):
        for match in ("same-domain", "cross-domain"):
            S = FS[(FS.variant == variant) & (FS.domain_match == match)]
            if S.empty:
                continue
            at200 = set(map(tuple, S[S.k == 200][["source", "target"]].to_numpy()))
            B = S[[(r.source, r.target) in at200 for r in S.itertuples()]]
            print(f"\n[{variant} / {match}]")
            for label, T in [("所有对子", S), ("每个 k 都在场的对子", B)]:
                g = T.groupby("k")
                med6 = g.frac_median6.median()
                medl = g.frac_linear.median()
                cnt = g.size()
                print(f"  {label}")
                for k in FEWSHOT_K:
                    if k not in med6.index:
                        continue
                    print(f"    k={k:>3}  n={cnt[k]}  "
                          f"六算法中位数为分母 {med6[k]:6.1%}   "
                          f"RidgeCV 域内为分母 {medl[k]:6.1%}")

    print("\n" + "=" * 78)
    print("CORAL 与 sham：按第三方语料是否落在目标域拆开")
    print("=" * 78)
    al_rows = []
    for variant, tables in [("published", published), ("clean", clean)]:
        ceil = (WC[(WC.variant == variant) & (WC.algorithm != "RidgeCV")]
                .groupby("corpus").rho.median())
        for src, Ds in tables.items():
            Xs, ys, _ = xyg(Ds, fc)
            for tgt, Dt in tables.items():
                if src == tgt:
                    continue
                Xt, yt, _ = xyg(Dt, fc)
                third = next(c for c in tables if c not in (src, tgt))
                Xth = xyg(tables[third], fc)[0]
                sham_valid = CORPORA[third] != CORPORA[tgt]
                variants = {
                    "none": Xs,
                    "coral_diag": coral(Xs, Xt, diag_only=True),
                    "coral": coral(Xs, Xt),
                    "coral_sham": coral(Xs, Xth),
                }
                base = None
                for meth, A in variants.items():
                    m = ridge()
                    m.fit(A, ys)
                    r = float(spearmanr(m.predict(Xt), yt).statistic)
                    if meth == "none":
                        base = r
                    al_rows.append({
                        "variant": variant, "source": src, "target": tgt,
                        "domain_match": ("same-domain"
                                         if CORPORA[src] == CORPORA[tgt]
                                         else "cross-domain"),
                        "method": meth, "rho": r, "zero_shot": base,
                        "ceiling_median6": float(ceil[tgt]),
                        "third_corpus": third, "sham_valid": sham_valid,
                    })
    AL = pd.DataFrame(al_rows)
    AL["frac"] = (AL.rho - AL.zero_shot) / (AL.ceiling_median6 - AL.zero_shot)
    AL.to_csv(OUT / "tac_recheck_alignment.csv", index=False)

    for variant in ("published", "clean"):
        X = AL[(AL.variant == variant) & (AL.domain_match == "cross-domain")]
        print(f"\n[{variant} / cross-domain]  六个有序对")
        for meth in ("coral_diag", "coral", "coral_sham"):
            f = X[X.method == meth].frac
            print(f"  {meth:<11} 补回缺口的中位数 {f.median():6.1%}  n={len(f)}")
        v = X[(X.method == "coral_sham") & X.sham_valid].frac
        iv = X[(X.method == "coral_sham") & ~X.sham_valid].frac
        print(f"  其中 sham 第三方在目标域之外（有效对照）n={len(v)}  "
              f"中位数 {v.median():6.1%}")
        print(f"       sham 第三方就在目标域内（无效对照）n={len(iv)}  "
              f"中位数 {iv.median():6.1%}")
        cd = X[X.method == "coral_diag"].frac.median()
        co = X[X.method == "coral"].frac.median()
        print(f"  分解：仅各维方差对齐 {cd:.1%}；再加相关结构 {co - cd:+.1%}"
              f"（占全部 {(co - cd) / co:.0%}）")

    print(f"\n→ {OUT}/tac_recheck_*.csv")


if __name__ == "__main__":
    main()
