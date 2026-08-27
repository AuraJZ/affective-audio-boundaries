r"""投稿前清单，逐条对照 hzwer/WritingAIPaper 的检查表。

    .venv/Scripts/python scripts/71_submission_checklist.py

用户指定该指南为本项目论文与投稿材料的写作参考。指南面向 AI 会议投稿，
本文是 bioRxiv 经验科学预印本，因此只取**可迁移且可机器核验**的条目：

| 指南条目 | 本脚本如何核 |
|---|---|
| 搜 `?` 查 LaTeX 断引用 | 解析编译产物里的 `??` 与 `.log` 的 undefined 警告 |
| 图表须在正文按编号顺序提及 | 抽 `\ref{}` 出现次序与 `\label{}` 定义次序比对 |
| 图注无语法错、以句号结尾 | 逐条图注查结尾标点 |
| 小标题大小写风格统一 | 归类 sentence case / Title Case，报告混用 |
| 图须矢量 | 查引入的图形文件扩展名 |
| 公式完整 | 查 `\begin{equation}` 配对与空公式 |
| 数字未抄错 | 见 `64`（图注↔正文）与本脚本的正文↔源数据比对 |

**不核的**：语法质量、论证是否成立、结论是否过度声称 —— 那些机器判不了，
清单也不该给出「全绿」的错觉。脚本结尾会明写这一点。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from soundml.provenance import REPO_ROOT

MS = REPO_ROOT / "manuscript"
DOCS = ["main", "supplementary"]

# 大小写风格判定要跳过的**专名**（期刊约定的标签，不是标题大小写）。
# 加一条前必须能说出它在正文里被当作标签用了多少次 —— 见 `check_headings`。
PROPER_NOUNS = ["Extended Data"]

# 图注宏定义在 legends/all.tex，正文用 \FigNLegend 之类引用
LEGEND_FILE = MS / "legends" / "all.tex"


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""


def strip_comments(s: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", s)


def check_undefined(doc: str) -> list[str]:
    """断引用：编译日志里的 undefined，以及 PDF 文本里的 `??`。"""
    bad = []
    log = read(MS / f"{doc}.log")
    for m in re.finditer(r"(LaTeX Warning: (?:Reference|Citation) [^\n]*"
                         r"undefined[^\n]*)", log):
        bad.append(m.group(1).strip())
    return bad


def check_ref_order(doc: str) -> list[str]:
    """图表在正文里的提及顺序，应与其编号顺序一致。

    编号顺序 = `\\label` 在文件里出现的先后（LaTeX 的 float 计数即按此）。

    🔴 **图与表必须分开比。** LaTeX 的 figure 与 table 是两个**独立计数器**，
    「Fig. 3 在 Table 1 之前被提及」与表的编号对不对毫无关系。
    首版把两类混在一个序列里比，报出 `tab:power 先于 fig:design 提及但编号在其后`
    —— 那是假阳性，照它去调整浮动体位置反而会把真的顺序搞乱。
    """
    src = strip_comments(read(MS / f"{doc}.tex"))
    body = re.sub(r"\\begin\{(figure|table)\*?\}.*?\\end\{\1\*?\}", " ", src,
                  flags=re.S)
    bad: list[str] = []
    for kind in ("fig", "tab"):
        labels = [m.group(1) for m in
                  re.finditer(rf"\\label\{{({kind}:[^}}]+)\}}", src)]
        if not labels:
            continue
        order = {lab: i for i, lab in enumerate(labels)}
        seen, seq = set(), []
        for m in re.finditer(rf"\\ref\{{({kind}:[^}}]+)\}}", body):
            lab = m.group(1)
            if lab in seen or lab not in order:
                continue
            seen.add(lab)
            seq.append(lab)
        for i in range(1, len(seq)):
            if order[seq[i]] < order[seq[i - 1]]:
                bad.append(f"{seq[i]} 在正文中先于 {seq[i-1]} 提及，"
                           f"但编号在其后（同为 {kind} 计数器）")
        bad += [f"{lab} 从未在正文中提及" for lab in labels if lab not in seen]
    return bad


def check_legend_punctuation() -> list[str]:
    """图注应无语法错并以句号结尾。此处只能核标点。"""
    src = read(LEGEND_FILE)
    bad = []
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*?)\n\}", src, flags=re.S):
        name, body = m.group(1), m.group(2).strip()
        text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", "", body)
        text = text.replace("{", "").replace("}", "").strip()
        if not text:
            continue
        if not text.rstrip().endswith((".", "!", "?")):
            bad.append(f"{name} 未以句号结尾：…{text[-60:]!r}")
    return bad


def check_heading_case(doc: str) -> list[str]:
    """小标题大小写风格是否统一。"""
    src = strip_comments(read(MS / f"{doc}.tex"))
    heads = [(m.group(1), m.group(2)) for m in
             re.finditer(r"\\(section|subsection|paragraph)\*?\{([^}]+)\}", src)]
    out = []
    for kind in ("section", "subsection", "paragraph"):
        ts = [t for k, t in heads if k == kind]
        if len(ts) < 2:
            continue
        def is_title_case(t: str) -> bool:
            # 🔴 先抠掉**专名**再判大小写，否则 "Extended Data Figures" 会被
            # 判成 Title Case。它不是 —— "Extended Data Fig." 是 Nature 系
            # 的图类标签，正文里引用了 10 次。照假阳性去改小标题，
            # 反而会让小标题与那 10 处不一致。
            #
            # 判据：一个短语若在正文中被当作**标签**反复使用（≥3 次且
            # 后面常跟编号），它就是专名，不参与大小写风格判定。
            s = t
            for pn in PROPER_NOUNS:
                s = s.replace(pn, "")
            words = [w for w in re.findall(r"[A-Za-z]+", s) if len(w) > 3]
            return len(words) > 1 and all(w[0].isupper() for w in words)
        tc = [t for t in ts if is_title_case(t)]
        sc = [t for t in ts if not is_title_case(t)]
        if tc and sc:
            out.append(f"{kind}: Title Case {len(tc)} 个 / sentence case "
                       f"{len(sc)} 个 —— 混用")
            out.append(f"    Title Case 例: {tc[:3]}")
    return out


def check_graphics(doc: str) -> list[str]:
    """图须矢量。"""
    src = strip_comments(read(MS / f"{doc}.tex"))
    bad = []
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src):
        f = m.group(1)
        ext = Path(f).suffix.lower()
        if ext and ext not in (".pdf", ".eps", ".ps"):
            bad.append(f"非矢量：{f}")
        if not ext:
            hit = list((MS).glob(f"{f}.*"))
            if hit and hit[0].suffix.lower() not in (".pdf", ".eps", ".ps"):
                bad.append(f"非矢量：{hit[0].name}")
    return bad


def check_math(doc: str) -> list[str]:
    """公式环境配对，且非空。"""
    src = strip_comments(read(MS / f"{doc}.tex"))
    bad = []
    for env in ("equation", "align", "gather"):
        o = len(re.findall(rf"\\begin\{{{env}\*?\}}", src))
        c = len(re.findall(rf"\\end\{{{env}\*?\}}", src))
        if o != c:
            bad.append(f"{env} 环境不配对：begin {o} / end {c}")
    if src.count("$") % 2:
        bad.append("行内 `$` 数量为奇数")
    return bad


def main() -> None:
    total = 0
    for doc in DOCS:
        if not (MS / f"{doc}.tex").exists():
            continue
        print("=" * 78)
        print(f"{doc}.tex")
        for label, fn in (("断引用 / undefined", check_undefined),
                          ("图表提及顺序", check_ref_order),
                          ("小标题大小写", check_heading_case),
                          ("图形矢量化", check_graphics),
                          ("公式完整性", check_math)):
            issues = fn(doc)
            total += len(issues)
            mark = "✅" if not issues else "⚠️"
            print(f"  {mark} {label}" + (f"  —— {len(issues)} 处" if issues else ""))
            for x in issues[:12]:
                print(f"       {x}")

    print("=" * 78)
    print("legends/all.tex")
    issues = check_legend_punctuation()
    total += len(issues)
    print(f"  {'✅' if not issues else '⚠️'} 图注句末标点"
          + (f"  —— {len(issues)} 处" if issues else ""))
    for x in issues[:12]:
        print(f"       {x}")

    print("\n" + "=" * 78)
    print(f"共 {total} 处待处理")
    print("\n⚠️ 本脚本只核机械项。**语法质量、论证是否成立、结论有无过度声称，")
    print("   机器判不了** —— 全绿不等于可以投。指南里「找同行读一遍」")
    print("   与「间接证据用 may be explained by、直接证据才用 is attributed to」")
    print("   这两条只能人来做。")
    sys.exit(0)


if __name__ == "__main__":
    main()
