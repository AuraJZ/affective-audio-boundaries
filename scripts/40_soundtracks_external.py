"""域之墙的第四语料外部检验 —— Soundtracks 作为独立音乐语料。

    .venv/Scripts/python scripts/40_soundtracks_external.py

## 为什么这是一次真正的外部检验

A2「域之墙」此前建立在 DEAM / PMEmo（音乐）对 Emo-Soundscapes（环境声）上。
两个音乐语料都来自 MediaEval 系的连续 valence-arousal 标注传统，
**共享标注范式**，因此「音乐内可迁移」有可能只是范式相同的产物。

Soundtracks 在四个维度上独立：

| | 既有音乐语料 | Soundtracks |
|---|---|---|
| 采集方 | MediaEval / 中科大 | Jyväskylä（Eerola & Vuoskoski） |
| 年代 | 2013–2019 | 2011 |
| 标注 | 连续 VA 滑块 | **1–9 李克特 × 8 情绪词** |
| 音乐类型 | 流行/多风格 | **电影配乐** |
| 选材 | 现成曲库 | **12 类目标情绪 × 30 段的平衡设计** |

若域之墙是真的结构性事实，应当预测：

1. 音乐 → Soundtracks 迁移**保持**（DEAM/PMEmo → ST）
2. Soundtracks → 环境声**崩塌**（ST → Emo）
3. 环境声 → Soundtracks **也崩塌**（Emo → ST）

三条全中才算通过。任何一条不符，都说明 A2 的表述需要收紧。

## 附带：离散情绪轴

Soundtracks 独有 anger/fear/happy/sad/tender 五个离散标注。
这是既有三个语料都没有的，用来检验声学特征捕捉的是**唤醒这一条轴**，
还是能分辨具体情绪类别 —— 后者若成立，产品侧的音频选择粒度可以更细。
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import (
    N_SPLITS, SEED, build_regressors, feature_cols, reg_score,
)

FEAT = REPO_ROOT / "data" / "features"
SETS = {"deam": "音乐/DEAM", "pmemo": "音乐/PMEmo",
        "soundtracks": "音乐/Soundtracks", "emo_mix": "环境声/Emo-Soundscapes"}
DOMAIN = {"deam": "music", "pmemo": "music",
          "soundtracks": "music", "emo_mix": "ambient"}
DISCRETE = ["anger", "fear", "happy", "sad", "tender"]


def load(tag: str) -> pd.DataFrame:
    return pd.read_parquet(FEAT / f"layer_a_reg_{tag}_ln.parquet")


def main() -> None:
    data = {k: load(k) for k in SETS}
    # 特征列以 DEAM 为准显式取。Soundtracks 的 parquet 还带着 tension / 5 个离散
    # 情绪 / target_emotion 等标签列，若沿用 feature_cols 的「排除元数据」写法
    # 会把标签当成特征喂进模型 —— 是标签泄漏，不是列名不一致。
    cols = feature_cols(data["deam"])
    for k, d in data.items():
        missing = [c for c in cols if c not in d.columns]
        assert not missing, f"{k} 缺特征列 {missing[:5]}"

    X = {k: d[cols].to_numpy(dtype=np.float64) for k, d in data.items()}
    y = {k: d["arousal"].to_numpy() for k, d in data.items()}
    g = {k: d["group"].to_numpy() for k, d in data.items()}

    algos = list(build_regressors())

    with RunRecord(
        "soundtracks_external", seed=SEED,
        params={"task": "regression", "target": "arousal",
                "new_corpus": "Eerola & Vuoskoski (2011) Soundtracks Set1",
                "datasets": {k: len(v) for k, v in y.items()},
                "algorithms": algos, "n_features": len(cols),
                "cv": f"GroupKFold({N_SPLITS})",
                "prediction": ["music->ST holds", "ST->ambient collapses",
                               "ambient->ST collapses"]},
    ) as run:
        # ---------- 组内 CV ----------
        within = {}
        for ds in SETS:
            folds = list(GroupKFold(n_splits=N_SPLITS).split(X[ds], y[ds], g[ds]))
            for algo in algos:
                preds = np.zeros(len(y[ds]))
                for tr, te in folds:
                    m = build_regressors()[algo]
                    m.fit(X[ds][tr], y[ds][tr])
                    preds[te] = m.predict(X[ds][te])
                within[f"{algo}|{ds}"] = reg_score(y[ds], preds)
            print(f"  组内 CV: {ds}", flush=True)
        run.log_metric("within_source", within)

        # ---------- 跨语料 ----------
        cross = {}
        for src, dst in itertools.permutations(SETS, 2):
            for algo in algos:
                m = build_regressors()[algo].fit(X[src], y[src])
                s = reg_score(y[dst], m.predict(X[dst]))
                s["same_domain"] = DOMAIN[src] == DOMAIN[dst]
                cross[f"{algo}|{src}->{dst}"] = s
        run.log_metric("cross_corpus", cross)

        rows = [{"algo": k.split("|")[0], "src": k.split("|")[1].split("->")[0],
                 "dst": k.split("|")[1].split("->")[1],
                 "same_domain": v["same_domain"], "spearman": v["spearman"]}
                for k, v in cross.items()]
        C = pd.DataFrame(rows)

        # ---------- 三条预测的逐条裁定 ----------
        st_in = C[(C.dst == "soundtracks") & (C.src.isin(["deam", "pmemo"]))]
        st_out_amb = C[(C.src == "soundtracks") & (C.dst == "emo_mix")]
        amb_in_st = C[(C.src == "emo_mix") & (C.dst == "soundtracks")]
        preds = {
            "music_to_ST_holds": {"mean_rho": float(st_in.spearman.mean()),
                                  "pass": bool(st_in.spearman.mean() > 0.30)},
            "ST_to_ambient_collapses": {"mean_rho": float(st_out_amb.spearman.mean()),
                                        "pass": bool(st_out_amb.spearman.mean() < 0.20)},
            "ambient_to_ST_collapses": {"mean_rho": float(amb_in_st.spearman.mean()),
                                        "pass": bool(amb_in_st.spearman.mean() < 0.20)},
        }
        run.log_metric("domain_wall_predictions", preds)

        # ---------- 离散情绪轴 ----------
        d_st = data["soundtracks"]
        disc = {}
        for tgt in DISCRETE + ["valence", "tension"]:
            if tgt not in d_st.columns:
                continue
            yy = d_st[tgt].to_numpy()
            folds = list(GroupKFold(n_splits=N_SPLITS)
                         .split(X["soundtracks"], yy, g["soundtracks"]))
            best = {}
            for algo in algos:
                p = np.zeros(len(yy))
                for tr, te in folds:
                    m = build_regressors()[algo]
                    m.fit(X["soundtracks"][tr], yy[tr])
                    p[te] = m.predict(X["soundtracks"][te])
                s = reg_score(yy, p)
                if not best or s["spearman"] > best["spearman"]:
                    best = {**s, "algo": algo}
            disc[tgt] = best
            print(f"  离散轴: {tgt}  ρ={best['spearman']:.3f} ({best['algo']})",
                  flush=True)
        run.log_metric("discrete_axes", disc)

        # ---------- 打印 ----------
        W = pd.DataFrame(within).T
        W["dataset"] = [k.split("|")[1] for k in W.index]
        piv = W.reset_index().assign(algo=lambda d: d["index"].str.split("|").str[0])

        print("\n" + "=" * 80)
        print(f"run_id: {run.run_id}")
        print("样本量: " + "   ".join(f"{SETS[k]}={len(y[k])}" for k in SETS))

        print("\n--- 组内 CV（arousal，Spearman ρ）---")
        print(piv.pivot(index="algo", columns="dataset", values="spearman")
              .round(3).to_string())

        print("\n--- 跨语料迁移（Spearman ρ，各算法均值）---")
        print(C.pivot_table(index="src", columns="dst", values="spearman")
              .round(3).to_string())

        print("\n--- 三条预测 ---")
        names = {"music_to_ST_holds": "音乐 → Soundtracks 保持（ρ>0.30）",
                 "ST_to_ambient_collapses": "Soundtracks → 环境声 崩塌（ρ<0.20）",
                 "ambient_to_ST_collapses": "环境声 → Soundtracks 崩塌（ρ<0.20）"}
        for k, v in preds.items():
            print(f"  {'✅' if v['pass'] else '❌'} {names[k]:38s} ρ={v['mean_rho']:+.3f}")

        s = C.groupby("same_domain")["spearman"].agg(["mean", "std", "count"])
        print("\n--- 同域 vs 跨域（含新语料）---")
        print(s.round(3).to_string())

        print("\n--- Soundtracks 独有的离散情绪轴（组内 CV）---")
        for k, v in sorted(disc.items(), key=lambda x: -x[1]["spearman"]):
            print(f"  {k:10s} ρ={v['spearman']:+.3f}   R²={v.get('r2', float('nan')):+.3f}"
                  f"   ({v['algo']})")

        C.to_csv(REPO_ROOT / "reports" / "source_data" /
                 "soundtracks_cross_corpus.csv", index=False)
        pd.DataFrame(disc).T.to_csv(REPO_ROOT / "reports" / "source_data" /
                                    "soundtracks_discrete_axes.csv")
        if all(v["pass"] for v in preds.values()):
            print("\n✅ 三条预测全中 —— 域之墙在独立第四语料上复现。")
        else:
            print("\n⚠️ 有预测未中 —— A2 的表述需要相应收紧。")


if __name__ == "__main__":
    main()
