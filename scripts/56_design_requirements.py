"""每人需要多少次观测？—— 把「适用范围有限」变成一个可计算的设计要求。

    .venv/Scripts/python scripts/56_design_requirements.py

## 问题

本项目全部生理阴性都带同一句限定：**~30 试次 / 12 s 片段 / 单次暴露**。
但「范围有限」是定性的。真正有用的是：

> **要以 80% 的把握检出一个真实相关为 r 的个体内关系，每人需要多少次观测？**

这个数一旦算出，「实验室 session 给不了、整夜采集能给」就不再是主张，
而是算术。

## 方法

不能靠外推 —— 每人只有 ~30 个试次，往上外推没有依据。
因此用**真实特征协方差**做模拟：

1. 每名受试的 156 维特征矩阵 → Ledoit-Wolf 收缩协方差
2. 从该协方差生成 n 个合成试次（n 可任意大）
3. 注入已知强度的线性关系：`y = Xw + ε`，缩放使总体相关 = r_true
4. **跑与 `53` 完全相同的流水线**：折内 PCA(8) + RidgeCV，5 折 CV
5. 逐个体置换检验（与 `53` 同口径），记录是否 p<0.05
6. 在 (n, r_true) 网格上重复，得到检出率

**关键：判据与 `53` 逐字相同。** 否则算出的样本量不适用于本项目的结论。

## 假设与限制 ⚠️

- 多元正态生成忽略了特征的非高斯性。对线性解码器的功效分析而言可接受，
  但**会略微高估**功效（真实数据更重尾）。因此本结果是**乐观估计** ——
  实际所需观测数只会更多，不会更少。
- 注入的是**线性**关系。若真实关系非线性，所需样本量更大。
- 未建模个体内的非平稳性（电极漂移、状态变化）。长时记录中这一项会更严重。

三条都指向同一方向：**本文给出的样本量是下界。**
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

warnings.filterwarnings("ignore")

FEAT = REPO_ROOT / "data" / "features"
BANDS = ("delta", "theta", "alpha", "beta", "gamma")

N_GRID = (30, 60, 120, 250, 500, 1000)
R_GRID = (0.10, 0.20, 0.30, 0.40)
N_SIM = 60                # 每个网格点的模拟次数
N_PERM = 100              # 每次模拟的置换次数
N_PC, N_FOLDS = 8, 5
TARGET_POWER = 0.80


def cv_rho(X: np.ndarray, y: np.ndarray) -> float:
    pred = np.empty(len(y))
    for tr, te in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X):
        pipe = make_pipeline(
            StandardScaler(),
            PCA(n_components=min(N_PC, len(tr) - 1, X.shape[1])),
            RidgeCV(alphas=np.logspace(-2, 4, 15)))
        pipe.fit(X[tr], y[tr])
        pred[te] = pipe.predict(X[te])
    r = spearmanr(pred, y).statistic
    return float(r) if np.isfinite(r) else np.nan


def one_sim(cov_chol: np.ndarray, n: int, r_true: float,
            rng: np.random.Generator) -> bool:
    """生成一次数据 → 跑 53 的流水线 → 逐个体置换 → 返回是否 p<0.05。"""
    p = cov_chol.shape[0]
    X = rng.standard_normal((n, p)) @ cov_chol.T
    w = np.zeros(p)
    k = max(p // 20, 3)                       # 稀疏权重：只有一小部分特征相关
    w[rng.choice(p, k, replace=False)] = rng.standard_normal(k)
    signal = X @ w
    signal /= max(signal.std(), 1e-12)
    # y = r*signal + sqrt(1-r^2)*noise → corr(y, signal) = r_true
    y = r_true * signal + np.sqrt(max(1 - r_true ** 2, 0)) * rng.standard_normal(n)

    obs = cv_rho(X, y)
    if not np.isfinite(obs):
        return False
    null = [cv_rho(X, rng.permutation(y)) for _ in range(N_PERM)]
    null = np.asarray([v for v in null if np.isfinite(v)])
    if null.size < N_PERM // 2:
        return False
    return bool((np.sum(null >= obs) + 1) / (null.size + 1) < 0.05)


def main() -> None:
    A = pd.read_parquet(FEAT / "ds002721_trials_ica.parquet")
    B = pd.read_parquet(FEAT / "ds002721_connectivity_ica.parquet")
    band = [c for c in A.columns if any(c.startswith(b) for b in BANDS) or c == "faa"]
    conn = [c for c in B.columns if c.startswith(("coh_", "icoh_", "asym_"))]
    D = A[["subject", "run", "trial"] + band].merge(
        B[["subject", "run", "trial"] + conn], on=["subject", "run", "trial"])
    feats = band + conn

    with RunRecord(
        "design_requirements", seed=SEED,
        params={"n_features": len(feats), "n_grid": list(N_GRID),
                "r_grid": list(R_GRID), "n_sim": N_SIM, "n_perm": N_PERM,
                "pipeline": "与 53 逐字相同：折内 PCA(8) + RidgeCV，5 折 CV，"
                            "逐个体置换 p<0.05",
                "covariance": "Ledoit-Wolf 收缩，取自 ds002721 的真实特征",
                "assumptions": ["多元正态生成（略微高估功效）",
                                "线性关系（非线性则需更多样本）",
                                "未建模个体内非平稳性（长时记录中更严重）"],
                "therefore": "给出的样本量是**下界**"},
    ) as run:
        rng = np.random.default_rng(SEED)

        # 用全体受试的特征协方差（收缩），作为「典型个体」的代表
        X0 = np.nan_to_num(D[feats].to_numpy(dtype=float), nan=0.0)
        X0 = (X0 - X0.mean(0)) / np.where(X0.std(0) > 0, X0.std(0), 1.0)
        cov = LedoitWolf().fit(X0).covariance_
        chol = np.linalg.cholesky(cov + 1e-8 * np.eye(cov.shape[0]))
        run.log_metric("shrinkage", float(LedoitWolf().fit(X0).shrinkage_))

        rows = []
        for r_true in R_GRID:
            for n in N_GRID:
                hits = sum(one_sim(chol, n, r_true, rng) for _ in range(N_SIM))
                power = hits / N_SIM
                rows.append({"r_true": r_true, "n_trials": n, "power": power})
                print(f"  r={r_true:.2f}  n={n:4d}  功效 {power:.2f}", flush=True)
        P = pd.DataFrame(rows)

        # 每个 r 下达到 80% 功效所需的 n（线性内插）
        need = []
        for r_true, g in P.groupby("r_true"):
            g = g.sort_values("n_trials")
            hit = g[g.power >= TARGET_POWER]
            if len(hit):
                i = hit.index[0]
                prev = g[g.n_trials < g.loc[i, "n_trials"]]
                if len(prev):
                    p0, n0 = prev.iloc[-1][["power", "n_trials"]]
                    p1, n1 = g.loc[i, ["power", "n_trials"]]
                    n_req = n0 + (TARGET_POWER - p0) / max(p1 - p0, 1e-9) * (n1 - n0)
                else:
                    n_req = float(g.loc[i, "n_trials"])
            else:
                n_req = np.inf
            need.append({"r_true": float(r_true), "n_required_80pct": float(n_req)})
        N = pd.DataFrame(need)
        run.log_metric("power_grid", P.to_dict("records"))
        run.log_metric("n_required_for_80pct_power", N.to_dict("records"))

        P.to_csv(REPO_ROOT / "reports" / "source_data" /
                 "design_power_grid.csv", index=False)
        N.to_csv(REPO_ROOT / "reports" / "source_data" /
                 "design_n_required.csv", index=False)

        print("\n" + "=" * 74)
        print(f"run_id: {run.run_id}   {len(feats)} 维   "
              f"每格 {N_SIM} 次模拟 × {N_PERM} 次置换")

        print("\n--- 功效表（行 = 真实相关，列 = 每人观测数）---")
        piv = P.pivot(index="r_true", columns="n_trials", values="power")
        print(piv.round(2).to_string())

        print(f"\n--- 达到 {TARGET_POWER:.0%} 功效所需的每人观测数 ---")
        for r_ in N.itertuples():
            v = ("> 1000" if not np.isfinite(r_.n_required_80pct)
                 else f"{r_.n_required_80pct:.0f}")
            print(f"  真实相关 r = {r_.r_true:.2f}   →   **{v}** 次/人")

        print("\n--- 对照现实 ---")
        print("  本数据集（实验室 session）      ~30 次/人")
        print("  典型公开数据集（DEAP 等）        40–60 次/人")
        print("  整夜记录（1 分钟窗 × 8 小时）    ~480 次/人/夜")
        print("  连续 30 夜                      ~14,400 次/人")
        print("\n" + "=" * 74)
        r02 = float(N.loc[N.r_true == 0.20, "n_required_80pct"].iloc[0])
        if np.isfinite(r02):
            print(f"→ 检出一个 r=0.20 的个体内关系需 **{r02:.0f} 次/人**，")
            print(f"  是本数据集所提供的 **{r02/30:.0f} 倍**，")
            print(f"  但仅相当于整夜记录的 **{r02/480:.1f} 夜**。")
        print("  三条假设（正态、线性、平稳）都使本估计偏乐观 —— "
              "**实际所需只会更多。**")


if __name__ == "__main__":
    main()
