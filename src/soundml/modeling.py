"""共用的模型定义与评估工具。

抽出来是为了保证 03（Layer A 基线）与 05（多表示矩阵）用的是**同一批模型定义**，
否则两轮结果不可比。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

SEED = 42
N_SPLITS = 5
META_COLS = {"clip_id", "path", "label", "category", "group", "source", "fold"}
SHAP_OVERLAP_THRESHOLD = 6  # SHAP top-20 三模型交集的合格线

# 适用域参数，照抄笔记 §3.5 / 论文
AD_Z = 0.5
AD_K = 5


def build_models(seed: int = SEED) -> dict[str, Pipeline]:
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "RF": Pipeline([("imp", imp()),
                        ("clf", RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1))]),
        "XGBoost": Pipeline([("imp", imp()),
                             ("clf", XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                                                   subsample=0.8, colsample_bytree=0.8,
                                                   eval_metric="logloss", random_state=seed, n_jobs=-1))]),
        "GBDT": Pipeline([("imp", imp()),
                          ("clf", GradientBoostingClassifier(random_state=seed))]),
        "SVM": Pipeline([("imp", imp()), ("sc", StandardScaler()),
                         ("clf", SVC(C=10, gamma="scale", probability=True, random_state=seed))]),
        "KNN": Pipeline([("imp", imp()), ("sc", StandardScaler()),
                         ("clf", KNeighborsClassifier(n_neighbors=5))]),
    }


def score(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = (proba >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
    }


def mean_std(rows: list[dict[str, float]]) -> dict[str, float]:
    d = pd.DataFrame(rows)
    return {f"{k}_{s}": float(v) for k in d.columns
            for s, v in (("mean", d[k].mean()), ("std", d[k].std()))}


def fit_ad(X: np.ndarray) -> tuple[NearestNeighbors, dict[str, float]]:
    """T = Z·σ + Y（笔记 §3.5）。Y/σ 是训练集内 k 近邻平均距离的均值与标准差。"""
    nn = NearestNeighbors(n_neighbors=AD_K + 1).fit(X)
    dist, _ = nn.kneighbors(X)
    mean_dist = dist[:, 1:].mean(axis=1)
    y_bar, sigma = float(mean_dist.mean()), float(mean_dist.std())
    return nn, {"Y": y_bar, "sigma": sigma, "Z": AD_Z, "k": AD_K, "threshold": AD_Z * sigma + y_bar}
