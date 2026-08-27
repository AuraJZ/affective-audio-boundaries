"""回归建模 —— 直接预测连续 arousal，不丢中间地带。

## 为什么换成回归

二分类是为对齐论文的 GABA agonist/inhibitor 结构而强加的。代价：
沿 arousal 轴切两端、丢中间，DEAM 从 1233 砍到 477（丢 61%）。

**而被丢掉的「中等唤醒」样本恰恰承载阈值信息** —— 正是规则库需要的东西。
回归用满全部样本，且唤醒度本来就是连续量。

## 评价指标

- **Spearman ρ**：秩相关，对标度不敏感，跨数据集可比（各数据集标度不同）
- **Pearson r**、**R²**：线性拟合优度
- **MAE**：绝对误差，与标度同单位

跨源比较以 **Spearman ρ 为主** —— DEAM 是 1–9 标度、PMEmo 是 0–1、
Emo-Soundscapes 是 −1..1，虽已线性映射到同一区间，但标注人群与量表锚点不同，
秩相关比绝对误差更可信。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

SEED = 42
N_SPLITS = 5
REG_META = {"clip_id", "path", "arousal", "valence", "group", "source", "category"}


def build_regressors(seed: int = SEED) -> dict[str, Pipeline]:
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "RF": Pipeline([("imp", imp()),
                        ("m", RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1))]),
        "XGBoost": Pipeline([("imp", imp()),
                             ("m", XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                                                subsample=0.8, colsample_bytree=0.8,
                                                random_state=seed, n_jobs=-1))]),
        "GBDT": Pipeline([("imp", imp()),
                          ("m", GradientBoostingRegressor(random_state=seed))]),
        "SVR": Pipeline([("imp", imp()), ("sc", StandardScaler()),
                         ("m", SVR(C=10, gamma="scale"))]),
        "KNN": Pipeline([("imp", imp()), ("sc", StandardScaler()),
                         ("m", KNeighborsRegressor(n_neighbors=5))]),
        "ElasticNet": Pipeline([("imp", imp()), ("sc", StandardScaler()),
                                ("m", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000,
                                                 random_state=seed))]),
    }


def reg_score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if y_true.size < 3 or np.std(y_pred) == 0:
        return {"spearman": np.nan, "pearson": np.nan, "r2": np.nan, "mae": np.nan}
    return {
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "pearson": float(pearsonr(y_true, y_pred).statistic),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in REG_META]
