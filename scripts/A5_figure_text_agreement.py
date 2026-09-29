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

## 🔴 2026-09 投稿前重算：登记表整体换过一遍

few-shot 曲线那一条登记的是 `D.cross_domain` 上的 64%，而**正文把它写成了
同域**。登记表当时是对的（它照着图算），错的是正文；但因为登记表只比
「数值是否一致」、不比「这个数属于哪个条件」，它一路绿灯放行。

所以本版做两件事：

1. 数值改从 `tac_recheck_*.csv` 取 —— 环境声语料剔除 613 段混合后的干净版本。
2. **每条都带 `condition` 字段**，并且要求正文里那个数字的邻近上下文出现
   该条件词。数值对、条件错，照样报错。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from soundml.provenance import REPO_ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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


def table1(family: str, variant: str, col: str) -> float:
    """Table 1 的一格。六个算法的中位数，与表里印的口径一致。"""
    R = pd.read_csv(SD / "tac_recheck_table1.csv")
    S = R[(R.family == family) & (R.variant == variant)]
    w = S[S.domain_match == "within-corpus"].rho.median()
    if col == "within":
        return float(w)
    x = S[S.domain_match == col].rho.median()
    return float(x) if col.endswith("domain") else float(1 - x / w)


def emo_ceiling(variant: str) -> float:
    W = pd.read_csv(SD / "tac_recheck_within_corpus.csv")
    W = W[(W.variant == variant) & (W.algorithm != "RidgeCV")
          & (W.corpus == "reg_emo_mix")]
    return float(W.rho.median())


def fewshot(match: str, k: int) -> float:
    """Fig 2a 画的正是这个：clean 版本、六算法中位数为分母、按对子取中位数。"""
    F = pd.read_csv(SD / "tac_recheck_fewshot.csv")
    F = F[(F.variant == "clean") & (F.domain_match == match) & (F.k == k)]
    return float(F.frac_median6.median())


def align(method: str, valid: bool | None = None) -> float:
    A = pd.read_csv(SD / "tac_recheck_alignment.csv")
    A = A[(A.variant == "clean") & (A.domain_match == "cross-domain")
          & (A.method == method)]
    if valid is not None:
        A = A[A.sham_valid == valid]
    return float(A.frac.median())


def nstar(domain_sub: str) -> float:
    T = pd.read_csv(SD / "canonical_tau_table.csv")
    T = T[T.domain.astype(str).str.contains(domain_sub)]
    return float(T.n_star.replace([float("inf")], pd.NA).dropna().median())


# (说明, 从数据算出的值, 正文里应出现的字面量, 容差, 邻近必须出现的条件词)
#
# `condition` 是本版新增的。它防的是 2026-09 那次事故：数值登记正确、正文却把
# 它安在了另一个实验条件上。None 表示该数字不依赖条件。
CHECKS = [
    ("Fig 5 · best self-report ICC", icc_best("self"), "0.221", 0.003, None),
    ("Fig 5 · best EEG ICC",         icc_best("eeg"),  "0.092", 0.003, None),

    ("Table 1 · hand within, clean",  table1("hand", "clean", "within"),
     "0.751", 0.003, None),
    ("Table 1 · hand cross, clean",   table1("hand", "clean", "cross-domain"),
     "0.129", 0.003, "leak-free"),
    ("Table 1 · AST cross, clean",    table1("ast", "clean", "cross-domain"),
     "0.162", 0.003, "leak-free"),
    ("Table 1 · hand cross, published", table1("hand", "published", "cross-domain"),
     "0.047", 0.003, None),
    ("Table 1 · AST cross, published",  table1("ast", "published", "cross-domain"),
     "0.125", 0.003, None),

    ("Hazard · emo ceiling, published", emo_ceiling("published"),
     "0.744", 0.003, None),
    ("Hazard · emo ceiling, clean",     emo_ceiling("clean"),
     "0.646", 0.003, None),

    ("Fig 2a · 100 labels, cross-domain", fewshot("cross-domain", 100),
     "56.9", 0.02, "Across domains"),
    ("Fig 2a · 200 labels, cross-domain", fewshot("cross-domain", 200),
     "71.0", 0.02, "Across domains"),
    ("Fig 2a · 100 labels, same-domain",  fewshot("same-domain", 100),
     "49.9", 0.02, "Same domain"),
    ("Fig 2a · 200 labels, same-domain",  fewshot("same-domain", 200),
     "76.5", 0.02, "Same domain"),

    ("Fig 2b · CORAL",            align("coral"),               "25.0", 0.02, None),
    ("Fig 2b · CORAL diagonal",   align("coral_diag"),          "4.9",  0.02, None),
    ("Fig 2b · sham, valid",      align("coral_sham", True),    "-11.1", 0.02, "valid"),
    ("Fig 2b · sham, invalid",    align("coral_sham", False),   "25.1", 0.02, "invalid"),

    ("Fig 6d · n* soundscape",    nstar("产品域"),  "67",  3.0, None),
    ("Fig 6d · n* music ratings", nstar("主观 · 音乐"), "166", 6.0, None),
]

text = "\n".join(p.read_text(encoding="utf-8") for p in SURFACES)


def appears(lit: str) -> bool:
    return re.search(rf"(?<![\d.])-?{re.escape(lit.lstrip('-'))}(?![\d])",
                     text) is not None


def condition_near(lit: str, word: str, window: int = 700) -> bool:
    """字面量附近 window 字符内，是否出现条件词。

    只要有**一处**出现即算通过。数字在别处被复述而不带条件词是常见且无害的；
    本检查要抓的是「从来没带过」。
    """
    core = re.escape(lit.lstrip("-"))
    for m in re.finditer(rf"(?<![\d.])-?{core}(?![\d])", text):
        lo = max(0, m.start() - window)
        if word.lower() in text[lo:m.end() + window].lower():
            return True
    return False


print("=" * 86)
print("图里算出的数  vs  正文与图注里写的数")
print(f"  {'检查':<34}{'数据':>9}{'文中':>9}{'一致':>7}{'出现':>7}{'条件':>8}")
bad = 0
for name, got, lit, tol, cond in CHECKS:
    want = float(lit)
    scale = 100.0 if abs(want) > 1.5 and abs(got) <= 1.5 else 1.0
    ok = abs(got * scale - want) <= tol * (100 if scale == 100 else 1)
    seen = appears(lit)
    cond_ok = True if cond is None else (condition_near(lit, cond) if seen else False)
    if not ok or not seen or not cond_ok:
        bad += 1
    print(f"  {name:<34}{got * scale:>9.3f}{want:>9.3f}"
          f"{'  OK' if ok else '  🔴':>7}"
          f"{'  是' if seen else '  🔴':>7}"
          f"{('  —' if cond is None else ('  是' if cond_ok else '  🔴')):>8}")

print("\n" + "=" * 86)
if bad:
    print(f"🔴 {bad} 处图文不一致 —— 这是 LaTeX 与时间戳检查都看不见的那一类。")
else:
    print("✅ 登记的每一处，图里算出的数与正文/图注一致，且条件词在邻近出现")
print()
print("⚠️ 只覆盖登记表里的条目。没登记的不一致它看不见 ——")
print("   每张图脚本里的 stopifnot 仍是第一道防线。")
sys.exit(1 if bad else 0)
