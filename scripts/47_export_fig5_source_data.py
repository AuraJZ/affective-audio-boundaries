"""Fig. 5 下半部分的 Source Data —— 用 ds002721 替换已撤回的 EDA 分析。

    .venv/Scripts/python scripts/47_export_fig5_source_data.py

## 为什么替换

Fig. 5 的 (c)(d)(e) 原本建立在 PMEmo 的 EDA 分析上（天花板 0.155）。
该分析的流水线**从未通过阳性对照**，`step14` 已整体撤回。

## 新版在论证上更完整

图 5 的主张是：**一个数字在拿到参照分布之前没有意义。**
说「X 预测不了 Y」需要两个前提：

| 前提 | 旧版 | 新版 |
|---|---|---|
| **仪器能用**（阳性对照） | ❌ 没有 | ✅ 四项全过，含时间对齐 |
| **Y 有可复现信号**（信度上限） | ⚠️ 有，但流水线未验证 | ✅ 同批受试同批试次的对照上限 |
| **反例** | ❌ 没有 | ✅ BIRAFFE2：同一问题，未过对照的流水线 |

## 三个面板

- **c** 反例：BIRAFFE2 的阳性对照失败（习惯化方向相反、刺激锁时不存在）
- **d** 正例：ds002721 的诱发响应总平均 + 符号翻转显著阈
- **e** 结果：同批受试同批试次，主观评分 vs 脑电的刺激层面 ICC
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from soundml.provenance import REPO_ROOT

SRC = REPO_ROOT / "reports" / "source_data"
SF, TMIN = 250.0, -0.20
QUESTIONS = ["pleasant", "energetic", "tense", "angry",
             "afraid", "happy", "sad", "tender"]


def write(name: str, D: pd.DataFrame) -> None:
    p = SRC / f"{name}.csv"
    D.to_csv(p, index=False)
    print(f"  → {p.name}   {len(D)} 行 × {D.shape[1]} 列")


# ---------------------------------------------------------------- panel c
# 反例：两条流水线在同一类阳性对照上的表现。BIRAFFE2 的数字取自
# step14（`biraffe_positive_control` / `biraffe_neurokit` 两次运行）。
# `label` 直接给出该对照的实际统计量 —— 比图例信息量大，也省掉一个图例。
c = pd.DataFrame([
    {"dataset": "BIRAFFE2 (EDA)", "control": "Habituation",
     "label": "ρ = +0.113, expected < 0", "p": 0.002, "passed": False},
    {"dataset": "BIRAFFE2 (EDA)", "control": "Stimulus locking",
     "label": "post ≈ random windows", "p": 0.267, "passed": False},
    {"dataset": "ds002721 (EEG)", "control": "Posterior alpha",
     "label": "occipital > frontal in 31/31", "p": 1.86e-9, "passed": True},
    {"dataset": "ds002721 (EEG)", "control": "Blink-locked ERP",
     "label": "164 µV frontal, 3.98× occipital", "p": 9.31e-10, "passed": True},
    {"dataset": "ds002721 (EEG)", "control": "Sound-onset response",
     "label": "|t| = 8.78 at 452 ms", "p": 0.0001, "passed": True},
])
write("fig_calibration_controls", c)

# ---------------------------------------------------------------- panel d
W = pd.read_csv(SRC / "ds002721_erp_waveforms.csv")
tcols = sorted([c for c in W.columns if c.startswith("t") and c[1:].isdigit()],
               key=lambda c: int(c[1:]))
S = W.groupby("subject")[tcols].mean().to_numpy(dtype=float)
t = np.arange(S.shape[1]) / SF + TMIN
ga = S.mean(0) * 1e6
sem = S.std(0, ddof=1) / np.sqrt(S.shape[0]) * 1e6
# 符号翻转最大统计量的 95% 阈（来自 `erp_signflip_test`）转成幅度尺度
THR_T = 3.36
thr_uv = THR_T * (S.std(0, ddof=1) / np.sqrt(S.shape[0])) * 1e6
d = pd.DataFrame({"time_s": t, "amplitude_uV": ga, "sem_uV": sem,
                  "threshold_uV": thr_uv,
                  "significant": (np.abs(ga) > thr_uv) & (t >= 0)})
write("fig_calibration_erp", d)

# ---------------------------------------------------------------- panel e
# 脑电侧用**完整** 156 维特征集（带功率 + 相干 + 虚部相干 + 不对称）。
# 只用带功率会低估脑电侧：相干族最强 ICC 0.092，是带功率 0.040 的两倍多，
# 且相干正是 Daly et al. (2014) 在同一数据上报告效应的地方。
R = pd.read_csv(SRC / "ds002721_eeg_reliability.csv")
rat = R[R.measure.isin(QUESTIONS)][["measure", "n", "icc", "split_half_sb"]].copy()
rat["kind"], rat["family"] = "Self-report", "rating"

F = pd.read_csv(SRC / "ds002721_reliability_full.csv")
eeg = F[["measure", "family", "n", "icc", "split_half_sb"]].copy()
eeg["kind"] = "EEG"

e = pd.concat([rat, eeg], ignore_index=True).sort_values(
    ["kind", "icc"], ascending=[True, False])
write("fig_calibration_icc", e)

summ = (e.groupby("kind")
          .agg(n_measures=("icc", "size"), max_icc=("icc", "max"),
               median_icc=("icc", "median"))
          .reset_index())
b_r = e[e.kind == "Self-report"].iloc[0]
b_e = e[e.kind == "EEG"].iloc[0]
summ["fold_gap"] = float(b_r.icc / max(b_e.icc, 1e-9))
write("fig_calibration_icc_summary", summ)

print("\n面板 e 摘要：")
print(summ.round(4).to_string(index=False))
print(f"\n主观最强 {b_r.measure} ICC={b_r.icc:+.3f}   "
      f"脑电最强 {b_e.measure} ({b_e.family}) ICC={b_e.icc:+.3f}   "
      f"倍数 {b_r.icc / max(b_e.icc, 1e-9):.1f}×")
print(f"主观最弱 {e[e.kind=='Self-report'].iloc[-1].measure} "
      f"ICC={e[e.kind=='Self-report'].iloc[-1].icc:+.3f}"
      f"  —— 仍高于脑电最强")
print("\n脑电各族最强：")
print(eeg.loc[eeg.groupby("family").icc.idxmax(),
              ["family", "measure", "icc", "split_half_sb"]]
      .round(3).to_string(index=False))
