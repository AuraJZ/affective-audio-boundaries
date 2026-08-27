"""特征表加载 —— 多个脚本共用，避免拼接逻辑漂移。

Layer A 一律用**响度归一化版**（`_ln`）。
理由见 reports/step2b_loudness_confound.md §4：
绝对响度是数据集指纹而非助眠信号，归一化后跨源 AUC 反而上升。
"""

from __future__ import annotations

import pandas as pd

from .modeling import META_COLS
from .provenance import REPO_ROOT

FEAT_DIR = REPO_ROOT / "data" / "features"


def load_joined(dataset: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """合并 Layer A（响度归一化版）与 Layer B。

    返回 (合并表, {表示名: 列名列表})，表示名为 "A" / "B" / "A+B"。

    Layer B 缺失时只返回 A —— 心理声学层提取成本高（约 20 s/段），
    而实测其贡献极低（22 维中仅 `lr_corr` 通过 FDR，且在规则阶段被拒），
    因此新数据集默认先只提 Layer A。这是有依据的成本决定，不是遗漏。
    """
    a = pd.read_parquet(FEAT_DIR / f"layer_a_{dataset}_ln.parquet")
    cols_a = [c for c in a.columns if c not in META_COLS]

    b_path = FEAT_DIR / f"layer_b_{dataset}.parquet"
    if not b_path.exists():
        return a, {"A": cols_a, "A+B": cols_a}

    b = pd.read_parquet(b_path)
    b_only = [c for c in b.columns if c not in META_COLS]
    df = a.merge(b[["clip_id", *b_only]], on="clip_id", validate="one_to_one")
    return df, {"A": cols_a, "B": b_only, "A+B": cols_a + b_only}


def source_paths(dataset: str) -> list:
    """用于 run record 的数据快照。"""
    paths = [FEAT_DIR / f"layer_a_{dataset}_ln.parquet"]
    b = FEAT_DIR / f"layer_b_{dataset}.parquet"
    if b.exists():
        paths.append(b)
    return paths
