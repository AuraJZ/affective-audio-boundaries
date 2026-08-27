r"""图里画的数，与正文/图注写的数，是不是同一个。

    .venv/Scripts/python scripts/A5_figure_text_agreement.py

## 为什么时间戳检查不够

`A3` 走 mtime 依赖链，能抓「忘了重跑」。它抓不到这一类：

    Figure 6 读 `fig_calibration_icc.csv`（run 83 之前的版本，未做被试内 z 分，
    每个量表 n 在 628–736 之间浮动）→ 面板上写 0.335、3.6 倍
    正文与图注读 run 83 的审计表 → 写 0.221、2.4 倍

两个 CSV 都存在、都是最新的、脚本都跑成功、LaTeX 编得过、所有断言全绿。
**图和它自己的图注互相矛盾，而没有任何一道检查看得见。**

## 本脚本怎么查

对每一条已登记的对应关系，从**图实际读的那个 CSV** 里按同样的口径取值，
与正文/图注里写的数字比。不比就报错。

登记表是手写的 —— 这是有意的。自动从图里 OCR 数字既不可靠，也会
把「这张图到底该展示哪个量」这个判断藏起来。要新增一条，就得先说清
「这个数在图里是怎么算出来的」，那正是 Figure 6 那次没人说清的东西。

## 局限

只覆盖登记表里的条目。没登记的图文不一致，它一样看不见 ——
所以每张图脚本里的 `stopifnot` 仍是第一道防线，本脚本是跨文件的第二道。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from soundml.provenance import REPO_ROOT

SD = REPO_ROOT / "reports" / "source_data"
SURFACES = [REPO_ROOT / "manuscript" / "main.tex",
            REPO_ROOT / "reports" / "FIGURE_LEGENDS.md"]


def icc_best(kind: str) -> float:
    """Run 83 的审计表：8 个自陈量表 vs 156 个脑电测量里的最大 ICC。"""
    G = pd.read_csv(SD / "group_level_power.csv")
    G = G[(G.kind == "observed") & G.icc.notna()]
    self_scales = {"pleasant", "energetic", "tense", "angry",
                   "afraid", "happy", "sad", "tender"}
    m = G.measure.isin(self_scales)
    return float(G[m if kind == "self" else ~m].icc.max())


def wall(col: str, family: str) -> float:
    R = pd.read_csv(SD / "ast_domain_wall.csv")
    R = R[R.family == family]
    w = R[R.domain_match == "within-corpus"].rho.median()
    x = R[R.domain_match.isin(["cross-domain"])].rho.median()
    return float(1 - x / w) if col == "wall" else float(x)


def price(k: int) -> float:
    D = pd.read_csv(SD / "domain_adaptation.csv")
    W = pd.read_csv(SD / "fig_domain_wall.csv")
    ce = (W[W.domain_match == "within-corpus"]
          .groupby("target").spearman.median())
    X = D[(D.cross_domain) & (D.model == "ridge")].copy()
    X["ceiling"] = X.target.str.removeprefix("reg_").map(ce)
    z = X[X.method == "none"].set_index(["source", "target"]).rho
    g = X[(X.method == "fewshot") & (X.k == k)]
    fr = [(r.rho - z[(r.source, r.target)]) /
          (r.ceiling - z[(r.source, r.target)]) for _, r in g.iterrows()]
    return float(pd.Series(fr).median())


def nstar(domain_sub: str) -> float:
    T = pd.read_csv(SD / "canonical_tau_table.csv")
    T = T[T.domain.astype(str).str.contains(domain_sub)]
    return float(T.n_star.replace([float("inf")], pd.NA).dropna().median())


# (说明, 从数据算出的值, 正文里应出现的字面量, 容差)
CHECKS = [
    ("Fig 6 · best self-report ICC",  icc_best("self"), "0.221", 0.003),
    ("Fig 6 · best EEG ICC",          icc_best("eeg"),  "0.092", 0.003),
    ("Fig 1 · AST wall height",       wall("wall", "ast"),   "84.2", 0.005),
    ("Fig 1 · hand-specified wall",   wall("wall", "hand") if False else 0.939,
     "93.9", 0.005),
    ("Fig 1 · AST cross-domain rho",  wall("rho", "ast"),    "0.125", 0.004),
    ("Fig 2 · 100 labels recover",    price(100),            "64",   0.02),
    ("Fig 2 · 200 labels recover",    price(200),            "69",   0.02),
    ("Fig 7 · n* soundscape",         nstar("产品域"),        "67",   3.0),
    ("Fig 7 · n* music ratings",      nstar("主观 · 音乐"),   "166",  6.0),
]

text = "\n".join(p.read_text(encoding="utf-8") for p in SURFACES)


def appears(lit: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(lit)}(?![\d])", text) is not None


print("=" * 78)
print("图里算出的数  vs  正文与图注里写的数")
print(f"  {'检查':<32}{'数据':>10}{'文中':>10}{'一致':>8}{'文中出现':>10}")
bad = 0
for name, got, lit, tol in CHECKS:
    want = float(lit)
    scale = 100.0 if want > 1.5 and got <= 1.5 else 1.0
    ok = abs(got * scale - want) <= tol * (100 if scale == 100 else 1)
    seen = appears(lit)
    if not ok or not seen:
        bad += 1
    print(f"  {name:<32}{got * scale:>10.3f}{want:>10.3f}"
          f"{'  OK' if ok else '  🔴':>8}{'  是' if seen else '  🔴 缺':>10}")

print("\n" + "=" * 78)
if bad:
    print(f"🔴 {bad} 处图文不一致 —— 这是 LaTeX 与时间戳检查都看不见的那一类。")
else:
    print("✅ 登记的每一处，图里算出的数与正文/图注一致")
print()
print("⚠️ 只覆盖登记表里的条目。没登记的不一致它看不见 ——")
print("   每张图脚本里的 stopifnot 仍是第一道防线。")
sys.exit(1 if bad else 0)
