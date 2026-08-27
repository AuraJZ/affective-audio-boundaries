r"""B1：主观侧的个体差异，用生理侧的**同一套估计量**重做。

    .venv/Scripts/python scripts/92_subjective_individual_dual.py

## 全项目最便宜的一次判决 ⭐

`76`/`77` 在 PMEmo 皮电上得到：逐人斜率的分半信度 15 格全不显著、
个性化增益全为负、个体间斜率 SD 的最小可检出效应 0.15。
据此写下「在 18 首/人的规模下，按人重拟合已被排除」。

**但那个阴性有两种解释，后果完全相反：**

| | 含义 | 产品结论 |
|---|---|---|
| **预算伪影** | 18 首/人对**任何**结果变量都不够 | 「按人校准被排除」必须**撤回** |
| **通道特异** | 主观侧在同样规模下检得出，生理侧检不出 | 产品结论成立且**大幅加强** |

区分它只需要一件事：**拿主观评分当结果变量，跑完全相同的流程。**

## 为什么 DEAM 能做这个对照

DEAM 的逐评分者文件在结构上与 PMEmo 几乎重合：

    每刺激评定者数   DEAM 中位 10   PMEmo 中位 10
    每人观测数       DEAM 中位 15   PMEmo 15–20

**关键设计：先抽到每人 18 首，与 PMEmo 完全对齐**，再放到 30/50/100/200。
前者回答「同样规模下主观侧行不行」，后者给出「主观侧需要多少」。

## 与生理侧逐字相同的部分

    预测变量   同样 5 个先验固定的描述符（`76` 定的，未改）
    结果变量   被试内 z 分（同 `76`）
    检验一     逐人斜率的分半一致性，零分布 = 打乱人身份
    检验二     个性化增益 = 留一交叉下「自己的斜率」vs「群体斜率」的误差之差
    多重比较   BH-FDR

**换脚本不换尺子。** 任何一处不同都会让这个对照失去意义。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, spearmanr

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")

DEAM = REPO_ROOT / "data" / "raw" / "DEAM" / "annotations" / \
    "annotations per each rater" / "song_level" / \
    "static_annotations_songs_1_2000.csv"
FEAT = REPO_ROOT / "data" / "features"
OUT = REPO_ROOT / "reports" / "source_data" / "subjective_individual_dual.csv"
SEED = 20260802

# `76` 定的五个先验描述符，一字未改
DESCRIPTORS = ["rms_mean", "onset_env_mean", "chroma_flux_mean",
               "roughness_pl_mean", "mode_score"]
TARGETS = ["Arousal", "Valence"]
N_GRID = [18, 30, 50, 100, 200]      # 18 = 与 PMEmo 对齐
N_SPLIT = 40
N_PERM = 2000
ALPHA = 0.05

# PMEmo 皮电在 n=18 下的对应结果（`76`/`77`），用于并排
PMEMO_AT_18 = {"split_half_best": 0.0754, "split_half_sig": 0,
               "gain_range": "-0.07 ~ -0.11", "n_listeners": 401}


def load() -> pd.DataFrame:
    R = pd.read_csv(DEAM, skipinitialspace=True)
    R.columns = [c.strip() for c in R.columns]
    A = pd.read_parquet(FEAT / "layer_a_reg_deam_ln.parquet")
    D = pd.read_parquet(FEAT / "layer_d_reg_deam_ln.parquet")
    F = A.merge(D, on="clip_id", suffixes=("", "_d"))
    F["SongId"] = F["clip_id"].astype(str).str.removeprefix("deam_").astype(int)
    keep = ["SongId"] + [c for c in DESCRIPTORS if c in F.columns]
    miss = [c for c in DESCRIPTORS if c not in F.columns]
    assert not miss, f"缺描述符 {miss}"
    M = R.merge(F[keep], on="SongId", how="inner")
    # 被试内 z 分（同 `76`）
    for t in TARGETS:
        g = M.groupby("workerID")[t]
        M[t] = (M[t] - g.transform("mean")) / (g.transform("std") + 1e-9)
    # 描述符全局标准化，使斜率跨描述符可比（同 `76`）
    for d in DESCRIPTORS:
        M[d] = (M[d] - M[d].mean()) / (M[d].std() + 1e-9)
    return M


def slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4 or np.std(x) < 1e-9:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def split_half_slope(units: list[tuple[np.ndarray, np.ndarray]],
                     rng: np.random.Generator) -> tuple[float, float, int]:
    """逐人斜率的分半一致性。零分布 = 打乱人身份。与 `76` 逐字相同。"""
    rs = []
    a = b = []
    for _ in range(N_SPLIT):
        a, b = [], []
        for x, y in units:
            idx = rng.permutation(len(x))
            h = len(idx) // 2
            sa, sb = slope(x[idx[:h]], y[idx[:h]]), slope(x[idx[h:]], y[idx[h:]])
            if np.isfinite(sa) and np.isfinite(sb):
                a.append(sa)
                b.append(sb)
        if len(a) >= 30:
            r = spearmanr(a, b).statistic
            if np.isfinite(r):
                rs.append(r)
    if not rs or len(a) < 30:
        return np.nan, np.nan, 0
    obs = float(np.mean(rs))
    null = np.array([spearmanr(a, rng.permutation(b)).statistic
                     for _ in range(N_PERM)])
    null = null[np.isfinite(null)]
    p = float((1 + (null >= obs).sum()) / (1 + len(null)))
    return obs, p, len(a)


def personalisation_gain(units: list[tuple[np.ndarray, np.ndarray]],
                         rng: np.random.Generator) -> tuple[float, float, int]:
    """留一交叉：自己的斜率 vs 群体斜率。与 `76` 逐字相同。"""
    allx = np.concatenate([x for x, _ in units])
    ally = np.concatenate([y for _, y in units])
    gains = []
    for i, (x, y) in enumerate(units):
        # 群体斜率排除本人
        m = np.ones(len(allx), bool)
        s0 = sum(len(u[0]) for u in units[:i])
        m[s0:s0 + len(x)] = False
        s_pop = slope(allx[m], ally[m])
        if not np.isfinite(s_pop):
            continue
        e_own, e_pop = [], []
        for j in range(len(x)):
            k = np.arange(len(x)) != j
            s_own = slope(x[k], y[k])
            if not np.isfinite(s_own):
                continue
            b = float(np.mean(y[k]))
            cx = x[j] - np.mean(x[k])
            e_own.append((y[j] - (b + s_own * cx)) ** 2)
            e_pop.append((y[j] - (b + s_pop * cx)) ** 2)
        if len(e_own) >= 8:
            gains.append(float(np.mean(e_pop) - np.mean(e_own)))
    if len(gains) < 30:
        return np.nan, np.nan, len(gains)
    gv = np.array(gains)
    obs = float(gv.mean())
    signs = rng.integers(0, 2, (N_PERM, len(gv))) * 2 - 1
    null = signs @ gv / len(gv)
    p = float((1 + (null >= obs).sum()) / (1 + len(null)))
    return obs, p, len(gv)


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("92_subjective_individual_dual", SEED)
    M = load()
    per = M.groupby("workerID").size()
    print(f"评分者 {M.workerID.nunique()}   歌曲 {M.SongId.nunique()}   "
          f"评分 {len(M)}")
    print(f"每人歌数 中位 {per.median():.0f}   "
          + "  ".join(f"≥{t}:{int((per>=t).sum())}" for t in N_GRID) + "\n")

    rows = []
    for n in N_GRID:
        elig = per[per >= n].index
        if len(elig) < 30:
            print(f"n={n}: 仅 {len(elig)} 人达标，跳过")
            continue
        print("=" * 82)
        print(f"每人抽 {n} 首   纳入 {len(elig)} 名评分者"
              + ("   ← 与 PMEmo 对齐" if n == 18 else ""))
        for t in TARGETS:
            for d in DESCRIPTORS:
                units = []
                for w in elig:
                    g = M[M.workerID == w].dropna(subset=[d, t])
                    if len(g) < n:
                        continue
                    pick = rng.choice(len(g), n, replace=False)
                    units.append((g[d].to_numpy(float)[pick],
                                  g[t].to_numpy(float)[pick]))
                if len(units) < 30:
                    continue
                r, p_r, k = split_half_slope(units, rng)
                gain, p_g, kg = personalisation_gain(units, rng)
                rows.append({"n": n, "target": t, "descriptor": d,
                             "n_raters": k, "split_half_r": r, "p_split": p_r,
                             "gain": gain, "p_gain": p_g})
                print(f"  {t:<9}{d:<20} 分半 r={r:+.3f} (p={p_r:.4f})   "
                      f"个性化增益 {gain:+.4f} (p={p_g:.4f})", flush=True)

    R = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT, index=False, encoding="utf-8")

    # ── 判决 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 82)
    print("判决：主观侧在 n=18（与 PMEmo 同规模）下检得出个体差异吗")
    at18 = R[R.n == 18]
    if len(at18):
        q = false_discovery_control(at18.p_split.fillna(1).to_numpy(), method="bh")
        nsig = int((q < ALPHA).sum())
        best = at18.loc[at18.split_half_r.idxmax()]
        pos_gain = int((at18.gain > 0).sum())
        print(f"  分半信度：BH-FDR 后显著 {nsig}/{len(at18)}   "
              f"最强 {best.target}×{best.descriptor} r={best.split_half_r:+.3f}")
        print(f"  个性化增益为正的格数：{pos_gain}/{len(at18)}")
        print(f"\n  对照 —— PMEmo 皮电同规模（`76`）：")
        print(f"    分半信度 显著 {PMEMO_AT_18['split_half_sig']}/15   "
              f"最强 r={PMEMO_AT_18['split_half_best']:+.3f}")
        print(f"    个性化增益 全为负（{PMEMO_AT_18['gain_range']}）")
        print()
        if nsig >= 2 or pos_gain >= len(at18) * 0.6:
            print("  → **通道特异**：主观侧在同样规模下检得出，生理侧检不出。")
            print("    P4 的个体层阴性不是预算伪影，产品结论成立且加强。")
        else:
            print("  → **预算伪影**：主观侧在同样规模下同样检不出。")
            print("    「18 首/人下按人重拟合被排除」这句话对任何结果变量都成立，")
            print("    不是生理特有的 —— **产品结论须撤回并改写为纯粹的规模问题**。")
        run.log_metric("n_sig_at18", nsig)
        run.log_metric("pos_gain_at18", pos_gain)
        run.log_metric("best_r_at18", float(best.split_half_r))

    print("\n" + "=" * 82)
    print("主观侧需要多少观测才检得出（各 n 下 BH-FDR 显著格数）")
    for n, g in R.groupby("n"):
        q = false_discovery_control(g.p_split.fillna(1).to_numpy(), method="bh")
        pos = int((g.gain > 0).sum())
        print(f"  n={n:>3}   {int(g.n_raters.median()):>3} 人   "
              f"分半显著 {int((q<ALPHA).sum())}/{len(g)}   增益为正 {pos}/{len(g)}")
        run.log_metric(f"n_sig_at_{n}", int((q < ALPHA).sum()))
    run.write()
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
