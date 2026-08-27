"""调性闭环主图的 Source Data。

    .venv/Scripts/python scripts/57_export_fig2_tonality.py

> 该图现为 **Fig. 3**。原为 Fig. 2，后因「正文提及顺序须与图号一致」
> 与路线分离图对调（路线图不含调性特征，故不能靠调整章节顺序解决）。
> 脚本文件名按创建序编号，是历史记录，不随图号变动。

## 这张图讲的是一个闭环，不是一张性能表

1. 数据指出缺口（happy 是七轴中唯一短板）
2. 提出可证伪假设（Layer A 无调性信息）
3. **判据写在跑之前**（增益须集中在 happy，且 mode_score 须靠前）
4. 两条都中
5. 机制核查是**双重分离**

面板：
- **a** 八轴 A → A+D 的哑铃图，按增益排序
- **b** Layer D 单独（39 维）vs Layer A 单独（122 维）
- **c** happy 与 tension 的特征重要性双重分离
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from soundml.provenance import REPO_ROOT

RUNS = REPO_ROOT / "runs"
OUT = REPO_ROOT / "reports" / "source_data"
KEY_FEATS = ["key_major_strength", "mode_score", "key_clarity",
             "chroma_consonance", "roughness_pl_mean"]


def latest(suffix: str) -> str:
    ids = sorted(d.name for d in RUNS.glob(f"*{suffix}") if (d / "meta.json").exists())
    return ids[-1]


def write(D: pd.DataFrame, name: str) -> None:
    D.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8")
    print(f"  {name}.csv  ({len(D)} 行 × {D.shape[1]} 列)")


m = json.loads((RUNS / latest("_layer_d_evaluation") / "meta.json")
               .read_text(encoding="utf-8"))["metrics"]

# ---------------------------------------------------------------- panel a / b
R = pd.DataFrame(m["results"])
# 调式依赖性：由乐理先验指定，用于检验增益是否落在预期的轴上。
# 这是**先验分组，不由数据决定** —— 否则就是循环论证。
MODE_DEPENDENT = {"happy", "valence", "sad"}
R["mode_dependent"] = R.axis.isin(MODE_DEPENDENT)
R = R.sort_values("gain_rho", ascending=False)
write(R[["axis", "mode_dependent", "rho_A", "rho_D", "rho_AD",
         "r2_A", "r2_AD", "gain_rho", "gain_r2",
         "algo_A", "algo_D", "algo_AD"]], "fig_tonality_axes")

# ---------------------------------------------------------------- panel c
# 55 的运行记录只存了三个特征的排名。这里重算一次 RF 重要性，
# 取全部五个调性特征的排名 —— 否则图上会出现空行。
from sklearn.ensemble import RandomForestRegressor          # noqa: E402

from soundml.regression import SEED, feature_cols          # noqa: E402

FEAT = REPO_ROOT / "data" / "features"
lab = pd.read_csv(REPO_ROOT / "data" / "reg_soundtracks.csv")
A_ = pd.read_parquet(FEAT / "layer_a_reg_soundtracks_ln.parquet")
D_ = pd.read_parquet(FEAT / "layer_d_reg_soundtracks_ln.parquet")
a_cols = feature_cols(pd.read_parquet(FEAT / "layer_a_reg_deam_ln.parquet"))
d_cols = [c for c in D_.columns if c != "clip_id"]
M_ = lab.merge(A_[["clip_id"] + a_cols], on="clip_id").merge(D_, on="clip_id")
X_ = M_[a_cols + d_cols].to_numpy(dtype=float)
names = a_cols + d_cols

rows = []
for ax in ("happy", "tension"):
    rf = RandomForestRegressor(n_estimators=400, random_state=SEED,
                               n_jobs=-1).fit(X_, M_[ax].to_numpy())
    order = np.argsort(rf.feature_importances_)[::-1]
    rank = {names[i]: int(np.where(order == i)[0][0]) + 1 for i in range(len(names))}
    for f in KEY_FEATS:
        rows.append({"axis": ax, "feature": f, "rank": rank[f],
                     "importance": float(rf.feature_importances_[names.index(f)])})
C = pd.DataFrame(rows)
write(C, "fig_tonality_ranks")
print("\n--- 全部五个调性特征的排名（161 个特征中）---")
print(C.pivot(index="feature", columns="axis", values="rank").to_string())

# 前 8 名（供图注引用）—— 取自 55 的运行记录
imp = m["feature_importance"]
rows = [{"axis": ax, "position": i + 1, "feature": f, "importance": v,
         "is_layer_d": f in d_cols}
        for ax in ("happy", "tension") for i, (f, v) in enumerate(imp.get(ax, []))]
write(pd.DataFrame(rows), "fig_tonality_top8")

print("\n--- 判据 ---")
print(f"  happy 增益        {m['happy_gain_rho']:+.3f}")
print(f"  其余各轴平均      {m['other_axes_mean_gain_rho']:+.3f}")
print(f"  针对性            {'✅' if m['targeted_gain'] else '❌'}")
