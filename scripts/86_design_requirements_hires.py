r"""设计需求模拟的高精度重跑：500 次/格，n 扩到 5000。

    .venv/Scripts/python scripts/86_design_requirements_hires.py

## 为什么重跑

`56` 每格只跑 60 次，Fig. 6b 的图注自陈误差约 ±0.05，且 r=0.10 的曲线
因抽样噪声而非单调。**这是本文的核心定量论证**，不该建立在这个精度上。
审稿意见（R3-m3）直接点了这一条：纯算力，没有理由不做。

同时 `56` 的 n 只到 1000，导致 r=0.20 与 r=0.10 两行都只能写
「> 1,000」—— 那是**审查边界**而不是估计值（R3-m2）。

## 提速：零分布只按 n 算一次 ⭐

`56` 的做法是**每次模拟**都跑 100 次置换来定 p 值：
4 r × 6 n × 60 sim × 101 次 CV = 145,440 次 CV。
直接把 N_SIM 提到 500、n 加两档，会变成 1,616,000 次 —— 11 倍。

但**置换零分布不依赖 r_true**：打乱标签后信号被完全破坏，
其分布只取决于 n 与特征协方差结构。因此可以

    对每个 n，用 1000 次置换算一次临界值（95 分位）
    此后每次模拟只需 1 次 CV，与该临界值比较

新开销：8 n × (1000 + 4 r × 500) = 24,000 次 CV。
**比原版还少 6 倍，而精度从 ±0.05 提到 ±0.02。**

代价是各次模拟共用同一个临界值，而非各自算一个带噪的 p。
这**更准而非更松** —— 1000 次置换定出的临界值远比 100 次的稳。
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
OUT = REPO_ROOT / "reports" / "source_data" / "design_power_grid_hires.csv"
OUT_REQ = REPO_ROOT / "reports" / "source_data" / "design_n_required_hires.csv"
BANDS = ("delta", "theta", "alpha", "beta", "gamma")

N_GRID = (30, 60, 120, 250, 500, 1000, 2000, 5000)
R_GRID = (0.10, 0.20, 0.30, 0.40)
N_SIM = 500               # 每格模拟次数（原 60）
N_PERM_THR = 1000         # 每个 n 定临界值用的置换次数
N_PC, N_FOLDS = 8, 5
TARGET_POWER = 0.80
ALPHA = 0.05


def cv_rho(X: np.ndarray, y: np.ndarray) -> float:
    """与 `53`/`56` 逐字相同的流水线，改动会破坏可比性。"""
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


def gen(chol: np.ndarray, n: int, r_true: float, rng) -> tuple:
    p = chol.shape[0]
    X = rng.standard_normal((n, p)) @ chol.T
    w = np.zeros(p)
    k = max(p // 20, 3)                       # 稀疏权重，与 `56` 一致
    w[rng.choice(p, k, replace=False)] = rng.standard_normal(k)
    sig = X @ w
    sig /= max(sig.std(), 1e-12)
    y = r_true * sig + np.sqrt(max(1 - r_true ** 2, 0)) * rng.standard_normal(n)
    return X, y


def main() -> None:
    A = pd.read_parquet(FEAT / "ds002721_trials_ica.parquet")
    B = pd.read_parquet(FEAT / "ds002721_connectivity_ica.parquet")
    band = [c for c in A.columns
            if any(c.startswith(b) for b in BANDS) or c == "faa"]
    conn = [c for c in B.columns if c.startswith(("coh_", "icoh_", "asym_"))]
    D = A[["subject", "run", "trial"] + band].merge(
        B[["subject", "run", "trial"] + conn], on=["subject", "run", "trial"])
    feats = band + conn

    with RunRecord(
        "design_requirements_hires", seed=SEED,
        params={"n_features": len(feats), "n_grid": list(N_GRID),
                "r_grid": list(R_GRID), "n_sim": N_SIM,
                "n_perm_threshold": N_PERM_THR,
                "speedup": "置换临界值按 n 只算一次 —— 零分布不依赖 r_true",
                "supersedes": "56_design_requirements（每格 60 次，n≤1000）"},
    ) as run:
        rng = np.random.default_rng(SEED)
        X0 = np.nan_to_num(D[feats].to_numpy(dtype=float), nan=0.0)
        X0 = (X0 - X0.mean(0)) / np.where(X0.std(0) > 0, X0.std(0), 1.0)
        lw = LedoitWolf().fit(X0)
        chol = np.linalg.cholesky(lw.covariance_
                                  + 1e-8 * np.eye(lw.covariance_.shape[0]))
        run.log_metric("shrinkage", float(lw.shrinkage_))
        print(f"特征 {len(feats)} 维   收缩 {lw.shrinkage_:.3f}\n")

        # ── 每个 n 的置换临界值（只算一次）──────────────────────────
        print("=" * 72)
        print(f"[1/2] 各 n 的置换临界值（{N_PERM_THR} 次/个，零分布不依赖 r）")
        thr = {}
        for n in N_GRID:
            X, y = gen(chol, n, 0.0, rng)
            null = np.array([cv_rho(X, rng.permutation(y))
                             for _ in range(N_PERM_THR)])
            null = null[np.isfinite(null)]
            thr[n] = float(np.quantile(null, 1 - ALPHA))
            print(f"  n = {n:>5}   零分布均值 {null.mean():+.4f}   "
                  f"95 分位 = {thr[n]:+.4f}", flush=True)
            run.log_metric(f"threshold_n{n}", thr[n])

        # ── 功效网格 ────────────────────────────────────────────────
        print("\n" + "=" * 72)
        print(f"[2/2] 功效网格（每格 {N_SIM} 次，误差约 ±{1.96*0.5/np.sqrt(N_SIM):.3f}）")
        rows = []
        for r_true in R_GRID:
            for n in N_GRID:
                hits = 0
                for _ in range(N_SIM):
                    X, y = gen(chol, n, r_true, rng)
                    v = cv_rho(X, y)
                    hits += bool(np.isfinite(v) and v > thr[n])
                power = hits / N_SIM
                se = float(np.sqrt(power * (1 - power) / N_SIM))
                rows.append({"r_true": r_true, "n_trials": n, "power": power,
                             "se": se, "threshold": thr[n]})
                print(f"  r={r_true:.2f}  n={n:>5}  功效 {power:.3f} ± {se:.3f}",
                      flush=True)
        P = pd.DataFrame(rows)
        P.to_csv(OUT, index=False, encoding="utf-8")

        # ── 80% 功效所需的 n（对数尺度线性内插）──────────────────────
        req = []
        for r_true, g in P.groupby("r_true"):
            g = g.sort_values("n_trials")
            hit = g[g.power >= TARGET_POWER]
            if len(hit):
                i = g.index.get_loc(hit.index[0])
                if i == 0:
                    n_req = float(g.n_trials.iloc[0])
                else:
                    a, b = g.iloc[i - 1], g.iloc[i]
                    f = (TARGET_POWER - a.power) / max(b.power - a.power, 1e-9)
                    n_req = float(np.exp(np.log(a.n_trials) + f *
                                         (np.log(b.n_trials) - np.log(a.n_trials))))
            else:
                n_req = np.inf
            req.append({"r_true": r_true, "n_required_80pct": n_req,
                        "power_at_30": float(g[g.n_trials == 30].power.iloc[0])})
            run.log_metric(f"n_required_r{r_true}", n_req)
        R = pd.DataFrame(req)
        R.to_csv(OUT_REQ, index=False, encoding="utf-8")

        print("\n" + "=" * 72)
        print("80% 功效所需的每人观测数（对比 `56` 的 60 次/格版本）")
        old = {0.40: 271, 0.30: 650, 0.20: np.inf, 0.10: np.inf}
        for _, x in R.iterrows():
            o = old.get(x.r_true, np.nan)
            new = ("> 5,000" if not np.isfinite(x.n_required_80pct)
                   else f"{x.n_required_80pct:,.0f}")
            oldtxt = "> 1,000" if not np.isfinite(o) else f"{o:,.0f}"
            print(f"  r = {x.r_true:.2f}   旧 {oldtxt:>8}   新 {new:>8}   "
                  f"n=30 时功效 {x.power_at_30:.3f}")
        print(f"\n→ {OUT.name} / {OUT_REQ.name}")


if __name__ == "__main__":
    main()
