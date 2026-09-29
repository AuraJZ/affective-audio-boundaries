r"""投稿前重算之二：把 Table 1 的五个表征放到干净的环境声语料上再算一遍。

    SOUNDML_DATA=<数据目录> .venv/Scripts/python scripts/A10_tac_recheck_embeddings.py

## 为什么必须重算

`A9_tac_recheck.py` 查出：Emo-Soundscapes 的 1,213 段里有 613 段是 30 段源录音的
混合，平均每 3.8 段共用同一组成分，而标签表给每段各自一个 `group`，于是
`GroupKFold` 退化为 `KFold`，近重复样本跨折。只用 600 段原始录音重算，域内上限
从 0.744 落到 0.646，而跨域转移从中位数 0.043 升到 0.129。

Table 1 的五行（手工描述子、AST、MERT、Wav2Vec2、CLAP）全部是在**同一个有泄漏的
语料**上算的，所以它们的「域内」被同样抬高、「跨域」被同样压低。论文据此说
AST 把跨域 ρ 从 0.047 提到 0.125「是真实但很小的改善」；但干净语料上手工描述子
本身就到 0.129。若如此，AST 相对手工描述子的优势可能整体来自那个泄漏。

这一行在论文里承担的分量很重 —— 它是「换表征救不回来」这句话的主要证据。
所以必须在干净语料上重算，而不是只在正文里加一句限制。

协议与 `A1_ast_domain_wall.py` 完全一致（同一个 `build_regressors`、同样的
GroupKFold-5、同样的 `domain_match` 定义），唯一的差别是环境声语料的两个版本：

    published   1,213 段（600 原始 + 613 混合）—— 论文用的
    clean         600 段原始录音 —— 每段一个不同的 Freesound 录音

手工描述子（Layer A）一并重算，好让五行在同一张表里可比。

输出 `reports/source_data/tac_recheck_table1.csv`。
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soundml.regression import N_SPLITS, build_regressors  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("SOUNDML_DATA", REPO / "data"))
FEAT = DATA / "features"
OUT = REPO / "reports" / "source_data" / "tac_recheck_table1.csv"

CORPORA = ["deam", "pmemo", "soundtracks", "emo_mix"]
AMBIENT = "emo_mix"
FAMILIES = {
    "hand": "Hand-specified (122)",
    "ast": "AST (AudioSet, supervised)",
    "mert": "MERT (music, self-supervised)",
    "w2v2": "Wav2Vec2 (speech, self-supervised)",
    "clap": "CLAP (contrastive; contaminated)",
}
MIN_DIM = {"hand": 100, "ast": 300, "mert": 300, "w2v2": 300, "clap": 300}


def load(fam: str, corpus: str):
    """特征 + 标签。非特征列从标签表推，不硬编码排除表（同 A1）。"""
    if fam == "hand":
        F = pd.read_parquet(FEAT / f"layer_a_reg_{corpus}_ln.parquet")
    else:
        F = pd.read_parquet(FEAT / f"layer_f_{fam}_reg_{corpus}.parquet")
    L = pd.read_csv(DATA / f"reg_{corpus}.csv")
    F = F.drop(columns=[c for c in F.columns if c in set(L.columns) - {"clip_id"}])
    D = F.merge(L, on="clip_id", how="inner")
    nonfeat = set(L.columns) | {"clip_id"}
    cols = [c for c in D.columns
            if c not in nonfeat and pd.api.types.is_numeric_dtype(D[c])]
    assert len(cols) >= MIN_DIM[fam], f"{fam}/{corpus} 维度异常 {len(cols)}"
    return D.dropna(subset=["arousal"]).reset_index(drop=True), cols


def within(D, cols, model) -> float:
    X, y = D[cols].to_numpy(float), D["arousal"].to_numpy(float)
    g = D["group"].to_numpy() if "group" in D.columns else np.arange(len(D))
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


def main() -> None:
    if not FEAT.is_dir():
        sys.exit(f"找不到特征目录 {FEAT}；请设置 SOUNDML_DATA")

    rows = []
    for fam in FAMILIES:
        try:
            data = {c: load(fam, c) for c in CORPORA}
        except (FileNotFoundError, AssertionError) as e:
            print(f"[{fam}] 跳过：{e}")
            continue
        # 环境声语料的干净版本：去掉 613 段混合
        Demo, cemo = data[AMBIENT]
        is_mix = Demo.clip_id.str.startswith("emo_mix_")
        clean_emo = (Demo[~is_mix].reset_index(drop=True), cemo)
        print(f"\n[{fam}] " + "  ".join(f"{c}={len(data[c][0])}" for c in CORPORA)
              + f"   clean emo={len(clean_emo[0])}   dims={len(data['deam'][1])}")

        for variant in ("published", "clean"):
            tables = dict(data)
            if variant == "clean":
                tables[AMBIENT] = clean_emo
            for algo, model in build_regressors().items():
                for src in CORPORA:
                    Ds, cols_s = tables[src]
                    rows.append({"family": fam, "variant": variant,
                                 "algorithm": algo, "source": src, "target": src,
                                 "domain_match": "within-corpus",
                                 "rho": within(Ds, cols_s, model)})
                    for tgt in CORPORA:
                        if tgt == src:
                            continue
                        cols = [c for c in cols_s if c in tables[tgt][1]]
                        cross = (src == AMBIENT) != (tgt == AMBIENT)
                        rows.append({
                            "family": fam, "variant": variant, "algorithm": algo,
                            "source": src, "target": tgt,
                            "domain_match": "cross-domain" if cross else "same-domain",
                            "rho": across(Ds, tables[tgt][0], cols, model)})
            R = pd.DataFrame([r for r in rows
                              if r["family"] == fam and r["variant"] == variant])
            w = R[R.domain_match == "within-corpus"].rho.median()
            sd = R[R.domain_match == "same-domain"].rho.median()
            cd = R[R.domain_match == "cross-domain"].rho.median()
            print(f"    {variant:<10} within {w:+.3f}  same {sd:+.3f}  cross {cd:+.3f}"
                  f"   loss same {1 - sd / w:5.1%}  loss cross {1 - cd / w:5.1%}")

    R = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT, index=False)

    print("\n" + "=" * 86)
    print("Table 1，两个版本对照")
    print("=" * 86)
    for variant in ("published", "clean"):
        print(f"\n[{variant}]")
        print(f"  {'Representation':<38}{'within':>8}{'same':>8}{'cross':>8}"
              f"{'lossSame':>10}{'lossCross':>11}")
        for fam, label in FAMILIES.items():
            S = R[(R.family == fam) & (R.variant == variant)]
            if S.empty:
                continue
            w = S[S.domain_match == "within-corpus"].rho.median()
            sd = S[S.domain_match == "same-domain"].rho.median()
            cd = S[S.domain_match == "cross-domain"].rho.median()
            print(f"  {label:<38}{w:>8.3f}{sd:>8.3f}{cd:>8.3f}"
                  f"{1 - sd / w:>9.1%}{1 - cd / w:>10.1%}")

    print("\n把 AST 相对手工描述子的跨域优势在两个版本下并排：")
    for variant in ("published", "clean"):
        S = R[(R.variant == variant) & (R.domain_match == "cross-domain")]
        h = S[S.family == "hand"].rho.median()
        a = S[S.family == "ast"].rho.median()
        print(f"  {variant:<10} hand {h:+.3f}   AST {a:+.3f}   "
              f"差 {a - h:+.3f}   倍数 {a / h if h > 0 else float('nan'):.2f}")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
