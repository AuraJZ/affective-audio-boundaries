r"""对表：`92` 的实测曲线 与 `99d` 改正后的 n\* 分布，能不能互相预言。

    .venv/Scripts/python scripts/99e_reconcile_92.py

## 为什么必须查

改正后 DEAM 主观的 n* 中位是 258，而 `92` 在同一份 DEAM 上的实测曲线说
「n=100 处个体差异开始显形」。差 2.6 倍 —— 按纪律 #7，先找冲突。

**但两者问的不是同一件事：**

    `92`   分半信度显著 / 个性化增益为正的**格数**，逐 n 数出来
    n*     每一格自己的盈亏平衡点，由 τ 与 σ_resid 解出

## 它们之间有一个可证伪的桥

若 n* 的定义没错，则在观测数 n 下，**增益为正的格数**应约等于
**n\* ≤ n 的格数** —— 一格只有跨过自己的平衡点，按人重拟合才该赢过群体斜率。

    预言：#{格 : n* ≤ n}   实测：`92` 在该 n 处数出的正增益格数

对得上 → 两条路径不但不冲突，而且**其中一条预言了另一条**。
对不上 → n* 的定义或 `92` 的实现至少有一个错。

这是本项目仅剩的两条**独立**路径（见纪律 #12），它们能不能互证很要紧。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")

SD = REPO_ROOT / "reports" / "source_data"
OUT = SD / "reconcile_92_vs_nstar.csv"
SEED = 20260803


def main() -> None:
    run = RunRecord("99e_reconcile_92", SEED)
    # 读**权威表** —— `tau_smalln_bias.csv` 用的是全局零分布常数，
    # `99d` 自己的 docstring 已标「不要引用」。
    C = pd.read_csv(SD / "canonical_tau_table.csv")
    C = C[C.domain.astype(str).str.contains("主观 · 音乐")]
    ns = C.n_star.replace([np.inf], np.nan).dropna().to_numpy(float)
    print(f"改正后 DEAM 主观各格的 n*（{len(ns)} 格）：")
    print("  " + "  ".join(f"{v:.0f}" for v in np.sort(ns)))
    print(f"  中位 {np.median(ns):.0f}\n")

    print("=" * 78)
    print("`92` 的实测输出（原样读，不转抄）")
    try:
        S = pd.read_csv(SD / "subjective_individual_dual.csv")
    except FileNotFoundError:
        print("  🔴 找不到 subjective_individual_dual.csv —— 需先重跑 `92`")
        run.write()
        return
    print("  列：", ", ".join(S.columns))
    print(S.head(24).to_string(index=False))

    # 找出 n 轴与「增益为正」的列 —— 名称不同版本可能变，先探测
    ncol = next((c for c in S.columns
                 if c.lower() in ("n", "n_per_unit", "n_songs", "n_obs")), None)
    gcol = next((c for c in S.columns
                 if "gain" in c.lower() or "增益" in c), None)
    print(f"\n  n 轴列 = {ncol}   增益列 = {gcol}")
    if not ncol:
        print("  🔴 无法自动识别 n 轴，需人工核对上表后再对接。")
        run.write()
        return

    print("\n" + "=" * 78)
    print("桥：n* ≤ n 的格数  vs  `92` 实测的正增益格数")
    print(f"  {'n':>6}{'预言 #(n* ≤ n)':>16}{'实测正增益格数':>16}{'对得上?':>10}")
    rows = []
    for n in sorted(S[ncol].dropna().unique()):
        pred = int((ns <= n).sum())
        sub = S[S[ncol] == n]
        obs = int((sub[gcol] > 0).sum()) if gcol else -1
        ok = "✅" if (obs >= 0 and abs(pred - obs) <= 2) else (
            "⚠️" if obs >= 0 else "—")
        print(f"  {n:>6.0f}{pred:>16}{obs if obs >= 0 else '—':>16}{ok:>10}")
        rows.append({"n": n, "predicted_cells": pred, "observed_cells": obs,
                     "total_cells": len(ns)})

    print("\n" + "=" * 78)
    print("判读 —— 🔴 这条桥比它看起来弱得多，两处都要说清楚")
    # 一、三个零是算术必然，不是吻合
    tmax = float(C.tau_raw.max())
    print(f"\n  一、低 n 的那几行**没有证据量**。`n* ≤ N` 等价于 "
          f"τ_c ≥ be·√(n_obs/N)，")
    print(f"     而 τ_c ≤ τ̂_raw 恒成立，本域十格里最大的 τ̂_raw 只有 {tmax:.4f}：")
    for n in sorted(S[ncol].dropna().unique()):
        need = float(np.median(C.breakeven * np.sqrt(C.n_per_unit / n)))
        can = int((C.tau_raw >= need).sum())
        print(f"       N={n:>4.0f} 需 τ ≥ {need:.3f}   够得着的格数 {can}/{len(C)}"
              + ("   ← 概率为 1 的零" if can == 0 else ""))
    print("     换任何零分布常数、任何改正、甚至完全不改正，都是同样的零。")
    # 二、两条路径共享数据
    print("\n  二、`92` 与 `93`/`99g` **不是独立路径**：同一份 DEAM 逐评分者 CSV、")
    print("     同两个 parquet、同 5 个描述符、同 workerID 分组与 z 分口径。")
    print("     **方法不共享，数据共享。** 正文不得写「两条独立路径吻合」。")
    print("\n  ⇒ 本脚本能支持的最强说法：两种估计量在同一份数据上给出的")
    print("     门槛量级一致（几十到几百，中位 ~166），不互相矛盾。仅此而已。")
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")
    run.write()
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
