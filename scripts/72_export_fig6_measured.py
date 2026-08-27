r"""Fig. 6 新面板的 Source Data：实测的试次数—检出率曲线。

    .venv/Scripts/python scripts/72_export_fig6_measured.py

Fig. 6 此前两个面板全部来自 `56` 的**模拟**（Ledoit-Wolf 收缩协方差生成的
合成数据）。论文 Discussion 整段「需要长时程个体数据」都压在那个数上。

`70` 用 ds003690 的真实生理数据把同一条曲线**测了出来**：
75 人、每人 240 个听觉事件、三条独立通路。本脚本把它导成绘图源数据。

导出三张表：

| 表 | 内容 |
|---|---|
| `measured_trialcount_curve` | 检出率 / 假阳性率 / 解析预期 vs 每人试次数，按通路 |
| `measured_per_subject` | 逐个体效应量与 p，用于画个体散点 |
| `measured_requirement` | 80% 把握所需试次数，按通路 × 年龄组 |
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nct
from scipy.stats import t as t_dist

from soundml.provenance import REPO_ROOT

FEAT = REPO_ROOT / "data" / "features"
OUT = REPO_ROOT / "reports" / "source_data"
ALPHA = 0.05
# 面板只画三条独立通路；`eeg_p2n1` 是同一批脑电的次要测量，留在补充材料
PANEL_MODS = {"eeg": "Scalp EEG (N1)", "pupil": "Pupil diameter",
              "hr": "Heart rate"}


def t_power(d: float, n: int) -> float:
    crit, df, nc = t_dist.ppf(1 - ALPHA / 2, n - 1), n - 1, abs(d) * np.sqrt(n)
    if not np.isfinite(nc) or nc > 30:
        return 1.0
    v = float(nct.sf(crit, df, nc) + nct.cdf(-crit, df, nc))
    return v if np.isfinite(v) else 1.0


def write(D: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8")
    print(f"  {name}.csv  ({len(D)} 行 × {D.shape[1]} 列)")


def main() -> None:
    C = pd.read_parquet(FEAT / "ds003690_trialcount_curve.parquet")
    S = pd.read_parquet(FEAT / "ds003690_per_subject.parquet")

    curve = (C[C["mod"].isin(PANEL_MODS)]
             .assign(channel=lambda d: d["mod"].map(PANEL_MODS))
             .loc[:, ["channel", "mod", "n", "n_subjects", "detect",
                      "detect_sd", "false_pos", "analytic"]]
             .sort_values(["channel", "n"]))
    write(curve, "measured_trialcount_curve")

    per = (S[S["mod"].isin(PANEL_MODS) & S.p.notna()]
           .assign(channel=lambda d: d["mod"].map(PANEL_MODS),
                   abs_d=lambda d: d["d"].abs(),
                   passes=lambda d: d.p < ALPHA)
           .loc[:, ["subject", "channel", "mod", "n", "mean", "d", "abs_d",
                    "p", "q", "passes"]])
    write(per, "measured_per_subject")

    # 80% 把握所需试次数，按通路 × 年龄组
    P = pd.read_csv(REPO_ROOT / "data" / "raw" / "ds003690" / "participants.tsv",
                    sep="\t").set_index("participant_id")
    rows = []
    for mod, label in PANEL_MODS.items():
        m = S[(S["mod"] == mod) & S.p.notna()].copy()
        m["group"] = m.subject.map(P["group"])
        for grp in ("Young", "Older", "All"):
            sel = m if grp == "All" else m[m.group == grp]
            if len(sel) < 5:
                continue
            med = float(sel["d"].abs().median())
            need = next((n for n in range(5, 5001) if t_power(med, n) >= 0.80),
                        np.nan)
            rows.append({"channel": label, "mod": mod, "group": grp,
                         "n_subjects": len(sel), "d_median": med,
                         "n_required_80pct": need})
    R = pd.DataFrame(rows)
    write(R, "measured_requirement")

    print("\n  校验：曲线里 n=30 的检出率")
    for _, r in curve[curve.n == 30].iterrows():
        print(f"    {r.channel:<20} {r.detect:.3f}   （假阳 {r.false_pos:.3f}）")
    print("  校验：80% 把握所需次数（全体）")
    for _, r in R[R.group == "All"].iterrows():
        print(f"    {r.channel:<20} {r.n_required_80pct:.0f} 次/人  "
              f"(|d| 中位 {r.d_median:.3f})")


if __name__ == "__main__":
    main()
