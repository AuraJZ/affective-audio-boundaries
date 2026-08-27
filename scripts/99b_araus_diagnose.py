r"""B2 诊断：阳性对照只有 36% 通过 —— 先查是数据的问题还是对照的问题。

    .venv/Scripts/python scripts/99b_araus_diagnose.py

## 起因

`99` 的阳性对照：|ρ(SMR, pleasant)| > 0.2 的人只有 36%，eventful 只有 23%。
按本项目的准入规矩，阳性对照不过就不能解读后面的 τ。

但在把数据判死刑之前，先怀疑对照本身。SMR 是**掩蔽声相对声景的电平**，
而 ARAUS 的掩蔽声分六族：bird / water / wind / construction / traffic / silence。

    鸟声开大 → 更宜人
    施工声开大 → 更不宜人

把六族混在一起算一个 ρ，两个方向**互相抵消**。
群体 β(smr→pleasant) = +0.147、β(smr→eventful) = −0.004 —— 后者几乎为零，
正是完全抵消的样子。

## 本脚本查四件事

1. 各掩蔽族的群体级 ρ(smr, 评分) —— 符号是否如预期分裂
2. silence 族的 smr 是不是无意义（应剔除）
3. 呈现顺序是否随机化 —— 决定 `99` 的顺序检验能推广到 PMEmo 的哪一部分
4. 每人每族的观测数 —— 决定阳性对照能不能在族内做
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from soundml.provenance import REPO_ROOT

warnings.filterwarnings("ignore")
AR = REPO_ROOT / "data" / "raw" / "ARAUS" / "datav2"
TARGETS = ["pleasant", "eventful"]


def main() -> None:
    R = pd.read_csv(AR / "responses.csv")
    R = R[R.is_attention == 0].copy()
    # `masker` 是文件名（bird_00001.wav），族在前缀里
    R["masker"] = R.masker.astype(str).str.split("_").str[0]

    print("=" * 84)
    print("[1] 掩蔽族 × SMR 的分布")
    print(R.groupby("masker").smr.describe()[["count", "mean", "min", "max"]]
          .round(2).to_string())

    print("\n" + "=" * 84)
    print("[2] 群体级 ρ(SMR, 评分)，**分族**算 —— 符号是否分裂")
    print(f"  {'掩蔽族':<14}{'n':>7}", end="")
    for t in TARGETS:
        print(f"{'ρ(' + t + ')':>16}", end="")
    print()
    for m, g in R.groupby("masker"):
        print(f"  {m:<14}{len(g):>7}", end="")
        for t in TARGETS:
            h = g.dropna(subset=["smr", t])
            r = spearmanr(h.smr, h[t]).statistic if h.smr.nunique() > 1 else np.nan
            print(f"{r:>16.3f}" if np.isfinite(r) else f"{'—':>16}", end="")
        print()
    print("\n  混合起来（`99` 用的口径）：", end="")
    for t in TARGETS:
        h = R.dropna(subset=["smr", t])
        print(f"  ρ({t}) = {spearmanr(h.smr, h[t]).statistic:+.3f}", end="")
    print()

    print("\n" + "=" * 84)
    print("[3] 每人每族的观测数 —— 族内做个人级对照够不够")
    per = R.groupby(["participant", "masker"]).size()
    print(f"  中位 {per.median():.0f}   四分位 [{per.quantile(.25):.0f}, "
          f"{per.quantile(.75):.0f}]   → "
          f"{'族内够' if per.median() >= 15 else '族内不够，须用带符号 SMR'}")

    print("\n" + "=" * 84)
    print("[4] 呈现顺序是否随机化")
    # 若随机化，则同一 stimulus_index 上出现的声景应五花八门
    x = R.groupby("stimulus_index").soundscape.nunique()
    print(f"  每个顺序位上出现过的不同声景数：中位 {x.median():.0f} "
          f"（共 {R.soundscape.nunique()} 个声景）")
    # 顺序与掩蔽族是否独立
    ct = pd.crosstab(R.stimulus_index, R.masker)
    frac = (ct.T / ct.sum(1)).T
    print(f"  各顺序位上掩蔽族占比的跨位标准差：{frac.std().mean():.4f} "
          f"（完全随机应 ≈ {np.sqrt(0.17*0.83/ct.sum(1).median()):.4f}）")
    r_ord = spearmanr(R.stimulus_index, R.smr).statistic
    print(f"  ρ(顺序, SMR) = {r_ord:+.4f}")
    for t in TARGETS:
        h = R.dropna(subset=[t])
        hg = h.groupby("participant")[t]
        z = (h[t] - hg.transform("mean")) / (hg.transform("std") + 1e-9)
        print(f"  ρ(顺序, 被试内 z 分 {t}) = "
              f"{spearmanr(h.stimulus_index, z).statistic:+.4f}"
              "   ← 疲劳/习惯化漂移的大小")

    print("\n" + "=" * 84)
    print("[5] 其余可能相关的列")
    # 🔴 首版查的是 "fold" 与 "wav_gain"，两个名字**都不存在**（真名 `fold_r`、
    # 增益在 soundscapes.csv 的 `gain_s`）。而 `if c in R.columns` 让它静默跳过 ——
    # 于是「有没有特殊的 fold」这个问题从没被回答，
    # 那正是 fold_r=−1 锚点直到对抗审计才被发现的原因。
    # 教训：探测式的列名循环必须对**一个都没命中**报警。
    want = ["fold_r", "time_taken", "is_attention", "smr", "stimulus_index"]
    hit = [c for c in want if c in R.columns]
    assert hit, f"一个都没命中，列名假设全错：{want}"
    for c in want:
        if c not in R.columns:
            print(f"  {c:<16}🔴 不存在")
            continue
        u = sorted(R[c].dropna().unique())
        print(f"  {c:<16}{len(u):>5} 个取值   例：{u[:6]}")
    print("\n  fold_r 各档的行数与每人行数：")
    for f, g in R.groupby("fold_r"):
        per = g.groupby("participant").size()
        print(f"    fold_r={f:>3}   {len(g):>5} 行   {g.participant.nunique():>3} 人"
              f"   每人 {per.min()}–{per.max()} 行"
              f"   声景 {g.soundscape.nunique()} 个"
              + ("   ← 锚点：同一刺激，每人两次" if f == -1 else ""))


if __name__ == "__main__":
    main()
