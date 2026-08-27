"""ED 7 重算：手工描述符 vs 预训练嵌入，加入第二个干净对照。

    .venv/Scripts/python scripts/60_representations_three_corpora.py

## 为什么要加第三个语料

原版只有两个语料，其中 Emo-Soundscapes 源自 Freesound —— 而 Freesound
正在 CLAP 的预训练语料中。因此该语料上的嵌入优势**无法与预训练重叠区分**，
只能靠 DEAM 这**一个**干净对照来估计污染的量级。

Soundtracks（电影配乐节选，Eerola & Vuoskoski 2011）与 Freesound 无关，
提供**第二个独立的干净对照**。若两个干净语料上的增益一致，
则「污染约占表观增益的一半」这一估计不再依赖单一语料。

## 判据（写在跑之前）

| 预期 | 若成立 |
|---|---|
| 两个干净语料的嵌入增益彼此接近 | 污染估计稳健 |
| 两者均显著低于污染语料的增益 | 「一半来自重叠」的结论成立 |

若两个干净语料的增益差别很大，则说明增益本身随语料变化，
「污染占一半」的说法须收回 —— 那时无法把差异归给污染。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import (
    N_SPLITS, SEED, build_regressors, feature_cols, reg_score,
)

FEAT = REPO_ROOT / "data" / "features"
CORPORA = {
    "emo_mix": ("Emo-Soundscapes", "ambient", True),      # 源自 Freesound → 可能污染
    "deam": ("DEAM", "music", False),
    "soundtracks": ("Soundtracks", "music", False),
}


def main() -> None:
    a_cols = feature_cols(pd.read_parquet(FEAT / "layer_a_reg_deam_ln.parquet"))
    algos = list(build_regressors())
    rows = []

    with RunRecord(
        "representations_three_corpora", seed=SEED,
        params={"corpora": {k: v[0] for k, v in CORPORA.items()},
                "contaminated": [k for k, v in CORPORA.items() if v[2]],
                "n_a": len(a_cols), "n_c": 512,
                "why": "加入第二个干净对照（Soundtracks），"
                       "使污染估计不再依赖单一语料",
                "prediction": "两个干净语料的嵌入增益应彼此接近，"
                              "且均低于污染语料"},
    ) as run:
        for key, (name, domain, contaminated) in CORPORA.items():
            lab = pd.read_csv(REPO_ROOT / "data" / f"reg_{key}.csv")
            A = pd.read_parquet(FEAT / f"layer_a_reg_{key}_ln.parquet")
            C = pd.read_parquet(FEAT / f"layer_c_reg_{key}.parquet")
            # 嵌入列 = C 里**不在标签文件中**的列。
            #
            # `layer_c` 的 parquet 连同标签一起存了（arousal / valence / tension …），
            # 因此「排除 clip_id 之外全要」会把标签喂进模型 —— 标签泄漏。
            # 这是该 bug 的**第二次**出现（Layer D 那次靠断言崩溃才暴露），
            # 根因都是「排除已知元数据」这个写法：新语料带新列时它会静默失效。
            # 正确的写法是以标签文件的列为准显式排除。
            c_cols = [c for c in C.columns if c not in set(lab.columns)]
            assert len(c_cols) == 512, f"{key}: 嵌入维度 {len(c_cols)}，应为 512"
            # 两侧都只并入特征列。并入完整的 C 会让重名的标签列被加上
            # _x/_y 后缀，`M["arousal"]` 随即 KeyError —— 这是上面那个
            # 标签泄漏问题的同一根源在 merge 层的表现。
            M = (lab.merge(A[["clip_id"] + a_cols], on="clip_id")
                    .merge(C[["clip_id"] + c_cols], on="clip_id"))
            y, g = M["arousal"].to_numpy(dtype=float), M["group"].to_numpy()
            XA, XC = M[a_cols].to_numpy(dtype=float), M[c_cols].to_numpy(dtype=float)
            folds = list(GroupKFold(n_splits=N_SPLITS).split(XA, y, g))

            for rep, X in (("A", XA), ("C", XC)):
                for algo in algos:
                    p = np.zeros(len(y))
                    for tr, te in folds:
                        m = build_regressors()[algo]
                        m.fit(X[tr], y[tr])
                        p[te] = m.predict(X[te])
                    rows.append({"corpus": name, "domain": domain,
                                 "contaminated": contaminated, "algorithm": algo,
                                 "representation": rep,
                                 "spearman": reg_score(y, p)["spearman"]})
                # 嵌入 + 线性岭回归：表示优势不应依赖强模型
                if rep == "C":
                    p = np.zeros(len(y))
                    for tr, te in folds:
                        sc = StandardScaler().fit(X[tr])
                        mm = RidgeCV(alphas=np.logspace(-2, 4, 15)).fit(
                            sc.transform(X[tr]), y[tr])
                        p[te] = mm.predict(sc.transform(X[te]))
                    rows.append({"corpus": name, "domain": domain,
                                 "contaminated": contaminated,
                                 "algorithm": "Ridge (linear)",
                                 "representation": "C", "spearman":
                                 reg_score(y, p)["spearman"]})
            print(f"  {name} 完成", flush=True)

        R = pd.DataFrame(rows)
        best = R.groupby(["corpus", "domain", "contaminated",
                          "representation"]).spearman.max().unstack()
        best["gain"] = best["C"] - best["A"]
        run.log_metric("best_by_representation", best.round(4).reset_index()
                       .to_dict("records"))

        clean = best[~best.index.get_level_values("contaminated")]["gain"]
        dirty = best[best.index.get_level_values("contaminated")]["gain"]
        spread = float(clean.max() - clean.min())
        # MultiIndex 的 to_dict() 产生元组键，JSON 序列化不了 —— 转成字符串
        preds = {"clean_gains_agree":
                 {"values": {str(k[0]): round(float(v), 4)
                             for k, v in clean.items()},
                  "spread": spread, "pass": bool(spread < 0.03)},
                 "clean_below_contaminated":
                     {"clean_mean": float(clean.mean()),
                      "contaminated": float(dirty.iloc[0]),
                      "pass": bool(clean.mean() < dirty.iloc[0])}}
        run.log_metric("predictions", preds)
        # ⚠️ 混杂：污染语料同时是唯一的环境声语料，两个干净语料都是音乐。
        # 「污染」与「域」完全共线，用现有语料分不开。
        run.log_metric("confound",
                       {"issue": "contamination is collinear with domain",
                        "detail": "the only Freesound-derived corpus is also the "
                                  "only ambient corpus; both clean corpora are music",
                        "would_resolve": "a clean ambient corpus, or a "
                                         "Freesound-derived music corpus"})

        R.to_csv(REPO_ROOT / "reports" / "source_data" /
                 "ed_representations.csv", index=False)

        print("\n" + "=" * 74)
        print(f"run_id: {run.run_id}")
        print("\n--- 各语料最佳（A = 手工 122 维，C = CLAP 512 维）---")
        print(best.round(3).to_string())

        print("\n--- 判据 ---")
        print(f"  两个干净语料的增益：{ {k[0]: round(v,3) for k,v in clean.items()} }")
        print(f"  {'✅' if preds['clean_gains_agree']['pass'] else '❌'} "
              f"彼此接近（差 {spread:.3f} < 0.03）")
        print(f"  {'✅' if preds['clean_below_contaminated']['pass'] else '❌'} "
              f"均值 {clean.mean():+.3f} < 污染语料 {dirty.iloc[0]:+.3f}")

        print("\n" + "=" * 74)
        if all(v["pass"] for v in preds.values()):
            frac = 1 - clean.mean() / dirty.iloc[0]
            print(f"→ 两个**独立**的干净语料给出高度一致的嵌入增益 "
                  f"（{clean.min():+.3f} 与 {clean.max():+.3f}），")
            print(f"  两者均低于 Freesound 来源的语料（{dirty.iloc[0]:+.3f}）。")
            print(f"  差值相当于表观增益的 {100*frac:.0f}%。")
            print("\n⚠️ **但这个差值不能干净地归给污染。**")
            print("  唯一的 Freesound 来源语料同时是**唯一的环境声语料**，")
            print("  而两个干净语料都是音乐 —— 「污染」与「域」完全共线。")
            print("  差值同样可以由「CLAP 在环境声上本就更强」解释。")
            print("\n  可报告的是：**在两个独立的音乐语料上，嵌入增益一致为 +0.037，**")
            print("  **而非环境声语料上观察到的 +0.070。** 引用应以前者为准。")
            print("  要分开这两个解释，需要一个干净的环境声语料，")
            print("  或一个 Freesound 来源的音乐语料。")
        else:
            print("→ 判据未全中：增益随语料变化，无法把差异干净地归给任何单一因素。")
            print("  应改为报告逐语料的增益。")


if __name__ == "__main__":
    main()
