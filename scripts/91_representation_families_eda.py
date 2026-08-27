r"""A3：换表征族能不能读到身体 —— 四族并列。

    .venv/Scripts/python scripts/91_representation_families_eda.py

## 要堵的反驳

`74`/`87`/`88` 都用同一批 161 维手工描述符预测歌曲级皮电，全阴。
最直接的反驳是：**「161 维手工特征不够，换个大模型就行了。」**

本脚本把四个**预训练范式各异**的表征族放到同一批 717 首歌、
同一套折上，逐个回答这句话。

| 族 | 维度 | 预训练范式 |
|---|---|---|
| 手工 Layer A+D | 161 | 无（谱时 + 调性描述符） |
| **ComParE** | 6373 | 无（大规模手工声学，PMEmo 作者在**原始音频**上提取） |
| **AST** | 768 | AudioSet **监督**分类 |
| **CLAP** | 512 | 音频–文本**对比**，63 万段 |

## 两条必须同时报的基准线 🔴

`90` 已经证明：**朴素的高维回归会把关系低估 3–4 倍** ——
信噪比极低的目标指导不了高维学习，而任何稳定的低维投影都能拿到 ρ≈0.11。

因此只报「某族的回归 ρ」会同时犯两个错：低估该族，且无法与 PC1 比较。
每一族都必须同时报：

    回归路线   标准化 → PCA(120) → RidgeCV，GroupKFold-5（按艺人）
    PC1 路线   该族第一主成分与皮电的直接相关

**判据**：若某族的**任一**路线显著超过 PC1 基准线（约 0.11）
且超过可达上限的 40%，则「换表征就行」成立，§8 须改写。
否则该反驳被四族独立堵死。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")

FEAT = REPO_ROOT / "data" / "features"
PM = REPO_ROOT / "data" / "raw" / "PMEmo" / "PMEmo2019"
OUT = REPO_ROOT / "reports" / "source_data" / "representation_families_eda.csv"
SEED = 20260801
EDA = ["scr_rate", "phasic_mean", "scl_slope"]
N_PC = 120
N_PERM = 150
MIN_LISTENERS = 8
CEILING_THRESHOLD = 0.40      # 超过可达上限 40% 才算「换表征就行」


def song_level() -> pd.DataFrame:
    T = pd.read_parquet(FEAT / "pmemo_eda_song_level.parquet")
    for m in EDA:
        g = T.groupby("listener")[m]
        T[m] = (T[m] - g.transform("mean")) / (g.transform("std") + 1e-9)
    a = T.groupby("musicId").agg(**{m: (m, "mean") for m in EDA},
                                 sd_scr=("scr_rate", "std"),
                                 sd_ph=("phasic_mean", "std"),
                                 sd_sl=("scl_slope", "std"),
                                 n=("listener", "size")).reset_index()
    return a[a.n >= MIN_LISTENERS]


def reliability(sd: np.ndarray, mean: np.ndarray, n: np.ndarray) -> float:
    """与 `88` 同一个解析估计量 —— 换脚本不换尺子。"""
    ok = np.isfinite(sd) & np.isfinite(mean) & (n > 1)
    sd, mean, n = sd[ok], mean[ok], n[ok]
    vw = float(np.mean(sd ** 2))
    n0 = float(np.mean(n))
    vb = max(float(np.var(mean, ddof=1)) - vw / n0, 0.0)
    icc = vb / (vb + vw) if (vb + vw) > 0 else 0.0
    return float(min(n0 * icc / (1 + (n0 - 1) * icc), 0.999)) if icc > 0 else 0.0


def _label_cols() -> set[str]:
    """任何标签 CSV 里出现过的列名，全部视为非特征。

    🔴 本项目已第四次栽在「排除已知元数据」这个写法上：
    Layer A/C/D 的 parquet 都自带 `arousal`/`valence`，
    Soundtracks 还额外带 `energy` 等八个情绪量表。
    逐族手写排除清单迟早会漏，改为**从标签文件反推**。
    """
    cols = {"clip_id", "path", "group", "source", "category", "musicId",
            "corpus", "split", "label", "stim", "target_emotion", "n",
            "sd_scr", "sd_ph", "sd_sl"}
    for p in (REPO_ROOT / "data").glob("reg_*.csv"):
        cols |= set(pd.read_csv(p, nrows=1).columns)
    return cols


NONFEAT = _label_cols()


def families() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    from soundml.regression import feature_cols
    a_cols = feature_cols(pd.read_parquet(FEAT / "layer_a_reg_deam_ln.parquet"))
    A = pd.read_parquet(FEAT / "layer_a_reg_pmemo_ln.parquet")
    Dl = pd.read_parquet(FEAT / "layer_d_reg_pmemo_ln.parquet")
    H = A[["clip_id"] + a_cols].merge(Dl, on="clip_id")
    H["musicId"] = H["clip_id"].astype(str).str.removeprefix("pmemo_").astype(int)
    out["手工 A+D (161)"] = H.drop(columns=["clip_id"])

    C = pd.read_csv(PM / "features" / "static_features.csv")
    C = C.rename(columns={C.columns[0]: "musicId"})
    C["musicId"] = pd.to_numeric(C["musicId"], errors="coerce")
    out["ComParE (6373)"] = C.dropna(subset=["musicId"]).astype({"musicId": int})

    for key, label in (("ast", "AST (768)"), ("clap", "CLAP-10s (512)")):
        p = FEAT / f"layer_f_{key}_reg_pmemo.parquet"
        if p.exists():
            D = pd.read_parquet(p)
            D["musicId"] = (D["clip_id"].astype(str)
                            .str.removeprefix("pmemo_").astype(int))
            out[label] = D.drop(columns=["clip_id"])
        else:
            print(f"  ⚠️ {label} 尚未提取：{p.name}")
    # 已有的 30 秒版 CLAP（Layer C）作为补充
    p = FEAT / "layer_c_reg_pmemo.parquet"
    if p.exists():
        D = pd.read_parquet(p)
        D["musicId"] = (D["clip_id"].astype(str)
                        .str.removeprefix("pmemo_").astype(int))
        out["CLAP-30s (512)"] = D.drop(columns=["clip_id"])
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("91_representation_families_eda", SEED)
    S = song_level()
    meta = pd.read_csv(PM / "metadata.csv")[["musicId", "artist"]]
    rel = {m: reliability(S[c].to_numpy(float), S[m].to_numpy(float),
                          S["n"].to_numpy(float))
           for m, c in zip(EDA, ("sd_scr", "sd_ph", "sd_sl"))}
    print("歌曲级皮电信度（与 `88` 同一估计量）：",
          "  ".join(f"{m}={rel[m]:.3f}" for m in EDA))
    print("可达上限 √r：", "  ".join(f"{m}={np.sqrt(rel[m]):.3f}" for m in EDA), "\n")

    fam = families()
    # 🔴 各族必须跑在**同一批歌**上。
    # ComParE 覆盖全部 794 首，手工/AST 只有 717 —— 若不取交集，
    # 「ComParE 也不行」这句话里就混着样本差异，不是表征的结论。
    common = set(S.musicId)
    for F in fam.values():
        common &= set(F.musicId)
    common = sorted(common)
    print(f"各族共有歌曲 {len(common)} 首（取交集，消除样本差异）")
    S = S[S.musicId.isin(common)].reset_index(drop=True)
    fam = {k: v[v.musicId.isin(common)].reset_index(drop=True)
           for k, v in fam.items()}
    # 交集改变了样本，信度必须在同一批歌上重算
    rel = {m: reliability(S[c].to_numpy(float), S[m].to_numpy(float),
                          S["n"].to_numpy(float))
           for m, c in zip(EDA, ("sd_scr", "sd_ph", "sd_sl"))}
    print("交集上的信度：",
          "  ".join(f"{m}={rel[m]:.3f}" for m in EDA), "\n")

    # 逐族落盘。ComParE 6373 维 × 150 次置换要跑很久，
    # 中断一次就全白跑 —— `39`/`69` 已经各栽过一次，不再重复。
    # 缓存键带交集大小：换了样本就不复用旧缓存。
    CACHE = FEAT / "repfam_cache"
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, F in fam.items():
        slug = label.split()[0].replace("/", "_")
        cp = CACHE / f"{slug}_n{len(common)}.csv"
        if cp.exists():
            prev = pd.read_csv(cp)
            rows += prev.to_dict("records")
            print("=" * 84)
            print(f"{label}   复用缓存 {cp.name}")
            for _, r in prev.iterrows():
                print(f"  {r.eda:<13} 回归 ρ={r.rho_regression:+.3f} "
                      f"(p={r.p_regression:.3f})   PC1 ρ={r.rho_pc1:+.3f}   "
                      f"最好占上限 {r.pct_of_ceiling:.0%}")
            continue
        fam_rows = []
        cols = [c for c in F.columns
                if c not in NONFEAT and c not in EDA
                and pd.api.types.is_numeric_dtype(F[c])]
        leak = [c for c in cols if c in EDA + ["arousal", "valence"]]
        assert not leak, f"{label} 标签泄漏 {leak}"
        assert len(cols) >= 100, f"{label} 特征维度异常 {len(cols)}"
        M = S.merge(F, on="musicId").merge(meta, on="musicId", how="left")
        M["artist"] = M["artist"].fillna("u" + M.musicId.astype(str))
        X = np.nan_to_num(M[cols].to_numpy(float), nan=0.0, posinf=0.0,
                          neginf=0.0)
        Xs = StandardScaler().fit_transform(X)
        pc1 = PCA(1).fit_transform(Xs)[:, 0]
        g = M["artist"].to_numpy()
        print("=" * 84)
        print(f"{label}   {len(M)} 首 × {X.shape[1]} 维")
        for m in EDA:
            y = M[m].to_numpy(float)

            def cv(yy):
                pred = np.full(len(yy), np.nan)
                for tr, te in GroupKFold(5).split(X, yy, g):
                    mdl = make_pipeline(
                        StandardScaler(), PCA(min(N_PC, len(tr) - 1)),
                        RidgeCV(alphas=np.logspace(-2, 4, 25)))
                    mdl.fit(X[tr], yy[tr])
                    pred[te] = mdl.predict(X[te])
                r = spearmanr(pred, yy).statistic
                return float(r) if np.isfinite(r) else 0.0

            r_reg = cv(y)
            null = np.array([cv(rng.permutation(y)) for _ in range(N_PERM)])
            p_reg = float((1 + (null >= r_reg).sum()) / (1 + N_PERM))
            r_pc1 = float(spearmanr(pc1, y).statistic)
            ceil = np.sqrt(rel[m])
            best = max(abs(r_reg), abs(r_pc1))
            print(f"  {m:<13} 回归 ρ={r_reg:+.3f} (p={p_reg:.3f})   "
                  f"PC1 ρ={r_pc1:+.3f}   最好占上限 {best/ceil:.0%}", flush=True)
            fam_rows.append({"family": label, "eda": m, "n_feat": X.shape[1],
                             "rho_regression": r_reg, "p_regression": p_reg,
                             "rho_pc1": r_pc1, "ceiling": float(ceil),
                             "pct_of_ceiling": float(best / ceil)})
        pd.DataFrame(fam_rows).to_csv(cp, index=False, encoding="utf-8")
        rows += fam_rows

    D = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT, index=False, encoding="utf-8")

    # 保守口径：分半信度通常低于解析值 → 上限更低 → 百分比更高，
    # 对「没有信源够得到」这一主张**更不利**。项目纪律要求同时报。
    print("\n" + "=" * 84)
    print("保守口径对照（分半信度 → 更低的上限 → 更高的百分比）")
    rng2 = np.random.default_rng(SEED)
    T = pd.read_parquet(FEAT / "pmemo_eda_song_level.parquet")
    for m in EDA:
        gg_ = T.groupby("listener")[m]
        T[m] = (T[m] - gg_.transform("mean")) / (gg_.transform("std") + 1e-9)
    T = T[T.musicId.isin(common)]
    for m in EDA:
        halves = []
        for _ in range(20):
            a_, b_ = [], []
            for _, gg in T.groupby("musicId"):
                v = gg[m].dropna().to_numpy().copy()
                if len(v) < MIN_LISTENERS:
                    continue
                rng2.shuffle(v)
                h = len(v) // 2
                a_.append(v[:h].mean())
                b_.append(v[h:2 * h].mean())
            r_ = spearmanr(a_, b_).statistic
            halves.append(2 * r_ / (1 + r_))
        sh = float(np.mean(halves))
        best = D[D.eda == m].apply(
            lambda r: max(abs(r.rho_regression), abs(r.rho_pc1)), axis=1).max()
        print(f"  {m:<13} 解析 {rel[m]:.3f} → {best/np.sqrt(rel[m]):.0%}   "
              f"分半 {sh:.3f} → **{best/max(np.sqrt(sh),1e-9):.0%}**")
        run.log_metric(f"splithalf_reliability_{m}", sh)
        run.log_metric(f"pct_conservative_{m}",
                       float(best / max(np.sqrt(sh), 1e-9)))

    print("\n" + "=" * 84)
    print("判定：有没有哪一族超过可达上限的 40%")
    top = D.loc[D.pct_of_ceiling.idxmax()]
    for f, g_ in D.groupby("family"):
        print(f"  {f:<20} 最好占上限 {g_.pct_of_ceiling.max():.0%}")
    print(f"\n全场最好：{top.family} × {top.eda}   占上限 {top.pct_of_ceiling:.0%}")
    if top.pct_of_ceiling >= CEILING_THRESHOLD:
        print("🔴 **「换表征就行」成立** —— §8 须改写。")
    else:
        print(f"✅ **{len(fam)} 族独立堵死了「161 维不够」这条反驳。**")
        print("   预训练规模从 0 到 63 万段、范式从手工到监督到对比，都不改变结论。")
        print(f"   共有歌曲 {len(common)} 首，各族样本完全一致。")
    run.log_metric("best_pct_of_ceiling", float(top.pct_of_ceiling))
    run.log_metric("best_family", str(top.family))
    run.write()
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
