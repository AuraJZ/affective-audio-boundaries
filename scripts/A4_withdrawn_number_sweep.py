r"""扫全部投稿面，找已撤回的数字与措辞。

    .venv/Scripts/python scripts/A4_withdrawn_number_sweep.py

## 为什么需要它

一个数字被撤回时，它散落在正文、补充材料、图注、**以及画在图里的字**。
改了三处漏第四处，LaTeX 编得过、断言也不报 —— 只有读者会发现。

ED7 就是这么在图上挂着 "the two independent corpora" 过了一整天：
稿件三处都清干净了，扫描器却只扫 .tex 和 .md，扫不到 R 源码里的字符串。

## 怎么读它的输出

**命中不等于错。** 「已撤回分析」那一节按设计就该提到那些数字，
描述撤回本身也必然引用旧值。这是个**多报**的网，判断留给人。
一个只报真错的扫描器，会在模式没覆盖到的地方静默放行 —— 那更坏。
"""
from __future__ import annotations

import pathlib
import re
import sys

from soundml.provenance import REPO_ROOT

ROOT = REPO_ROOT
TARGETS = [
    "manuscript/main.tex",
    "manuscript/supplementary.tex",
    "manuscript/preamble.tex",
    "manuscript/legends/all.tex",
    "reports/FIGURE_LEGENDS.md",
    # 🔴 The R sources too. Text drawn INSIDE a figure is the first thing a
    # reader sees and the last thing a text sweep looks at: ED7 carried "the
    # two independent corpora" on its face for a full day after the phrase had
    # been struck from the manuscript, the legends and the supplement.
    "figures_r/extended_data.R",
    "figures_r/fig1_domain_wall.R",
    "figures_r/fig2_convergence.R",
    "figures_r/fig3_tonality.R",
    "figures_r/fig4_intervention.R",
    "figures_r/fig5_calibration.R",
    "figures_r/fig6_design.R",
    "figures_r/fig_price.R",
    "figures_r/fig_boundaries.R",
    "figures_r/ed8_invariance.R",
]

# (pattern, what it was, what it should be now)
STALE = [
    (r"\b271\b",           "power sim r=0.40",       "314"),
    (r"\b650\b",           "power sim r=0.30",       "564"),
    (r">\s*1,?000",        "unresolved power bound", "1,364 / 4,888"),
    (r"60 (?:simulations|datasets) per cell", "old replicate count", "500"),
    (r"0\.106",            "withdrawn tau",          "withdrawn"),
    (r"97[-–]184",         "threshold from that tau", "67 / 166 / 318"),
    (r"\+0\.632|\+1\.000", "retired n=4 dose stat",  "clip-level test"),
    (r"0\.892",            "the stim_c leak",        "0.092"),
    (r"3\.6[-\s]?fold",    "old ICC ratio",          "2.4-fold"),
    (r"0\.335",            "old best self-report",   "0.221"),
    (r"\b0\.158\b",        "old weakest self-report", "0.162"),
]
ABSOLUTE = [
    (r"no reproducible", "absolute negative about physiology"),
    (r"none that survives", "absolute negative"),
    (r"\bis absent\b", "absolute negative"),
]
INDEP = [(r"\bindependent(?:ly)?\b", "independence claim")]

def scan(pats, label):
    print(f"\n{'=' * 72}\n{label}")
    total = 0
    for t in TARGETS:
        p = ROOT / t
        if not p.exists():
            continue
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, ln in enumerate(lines, 1):
            for pat, *rest in pats:
                if re.search(pat, ln):
                    total += 1
                    note = " -> ".join(str(r) for r in rest)
                    print(f"  {t}:{i}  [{note}]")
                    print(f"      {ln.strip()[:120]}")
    if not total:
        print("  none")
    return total

n1 = scan(STALE, "WITHDRAWN OR SUPERSEDED NUMBERS")
n2 = scan(ABSOLUTE, "ABSOLUTE NEGATIVES (the position is now 'bounded')")
n3 = scan(INDEP, "INDEPENDENCE CLAIMS (only the loudness control survives)")
print(f"\n{'=' * 72}")
print(f"命中：过期数字 {n1} | 绝对否定 {n2} | 独立性声称 {n3}")
print()
print("⚠️ 命中不等于错。「已撤回分析」一节按设计就该提到那些数字，")
print("   描述撤回本身也必然引用旧值。这是个**多报**的网 —— 逐条看，不要批量改。")
print("   一个只报真错的扫描器，会在模式没覆盖到的地方静默放行，那更坏。")
sys.exit(0)
