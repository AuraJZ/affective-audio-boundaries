r"""不装 LaTeX 也能查的结构验证 —— 让「编不过」在提交前就暴露。

    .venv/Scripts/python scripts/A2_tex_structure_check.py

## 为什么需要它

`71` 查的是投稿的机械项（断引用、图表顺序、小标题大小写）。它假设文件
**能编译**。而这台机器上没有 LaTeX 工具链，所以「能不能编译」这件事
在提交前完全没有被检查过 —— 一个环境不配对、一个未定义的宏、一处
`\ref` 指向不存在的 `\label`，都会在别人的机器上才炸。

本脚本不编译，只做**编译器一定会拒绝**的那几类静态检查。查不到的
（排版溢出、字体缺失、浮动体摆放）照实说明，不制造「全绿=能编译」的错觉。

## 查什么

    环境配对        `\begin{X}` 与 `\end{X}` 逐类计数
    括号平衡        统计 `{` `}`（跳过 `\{`、`\%` 与注释）
    宏定义与使用    `\newcommand` 定义的宏 vs 正文实际用到的自定义宏
    label/ref       每个 `\ref`/`\eqref` 都要有 `\label`；孤立 label 只警告
    图文件存在      每个 `\includegraphics` 的目标要能在盘上找到
    引用键          每个 `\cite` 的键要在 .bib 里
    重复 label      同名 label 出现两次，编译器只警告而结果错乱
    发布文件占位符  仓库根上的 README / LICENSE / 数据台账
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from soundml.provenance import REPO_ROOT

MS = REPO_ROOT / "manuscript"
DOCS = ["main", "supplementary"]


def strip_comments(t: str) -> str:
    """去掉 % 注释，但保留 \\% 转义。"""
    return re.sub(r"(?<!\\)%.*", "", t)


def load(doc: str, _depth: int = 0, _seen: set[Path] | None = None) -> str:
    r"""展开 \input，**递归**。

    🔴 首版只展开一层，于是 `main.tex` → `preamble.tex` → `legends/all.tex`
    这条链上第三级里的 `\newcommand` 全都看不见，16 个图注宏被误报为未定义。
    检查器的假阳性比漏报更坏：它会让人去「修」一个没坏的东西。
    """
    seen = _seen if _seen is not None else set()
    p = (MS / f"{doc}.tex") if not doc.endswith(".tex") else (MS / doc)
    p = p.resolve()
    if p in seen or not p.exists() or _depth > 8:
        return ""
    seen.add(p)
    t = strip_comments(p.read_text(encoding="utf-8"))
    for m in re.finditer(r"\\input\{([^}]+)\}", t):
        name = m.group(1)
        sub = name if name.endswith(".tex") else name + ".tex"
        t = t.replace(m.group(0), load(sub, _depth + 1, seen))
    return t


def check_environments(t: str) -> list[str]:
    from collections import Counter
    b = Counter(re.findall(r"\\begin\{(\w+\*?)\}", t))
    e = Counter(re.findall(r"\\end\{(\w+\*?)\}", t))
    return [f"{k}: {b[k]} begin vs {e[k]} end"
            for k in sorted(set(b) | set(e)) if b[k] != e[k]]


def check_braces(t: str) -> list[str]:
    t = re.sub(r"\\[{}%]", "", t)
    o, c = t.count("{"), t.count("}")
    return [] if o == c else [f"{{ {o} 个 vs }} {c} 个，差 {o - c}"]


def check_macros(t: str) -> list[str]:
    defined = set(re.findall(r"\\(?:new|renew|provide)command\*?\{?\\(\w+)", t))
    # 只查自定义宏：内置宏太多，白名单会变成消音器。判据是「以 legend 开头
    # 或在本文档里被 \newcommand 定义过」——前者是本项目自己的约定。
    used = set(re.findall(r"\\(legend\w+)", t))
    return [f"未定义的图注宏 \\{m}" for m in sorted(used - defined)]


def check_refs(t: str) -> list[str]:
    # 图是用 \msfigure{路径}{label}{图注} 声明的，它的 \label 写在宏体里
    # （`\label{#2}`），正文中不会出现字面的 \label。不认这个宏的话，八个
    # \ref{fig:...} 会全被报成悬空引用 —— 八个假阳性，而文档实际编译时
    # 未定义引用为零。
    labels = (re.findall(r"\\label\{([^}]+)\}", t)
              + re.findall(r"\\msfigure\{[^}]*\}\{([^}]+)\}", t))
    labels = [x for x in labels if "#" not in x]
    refs = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", t))
    out = [f"\\ref{{{r}}} 无对应 \\label" for r in sorted(refs - set(labels))]
    dup = {x for x in labels if labels.count(x) > 1}
    out += [f"重复的 \\label{{{d}}}（编译器只警告，交叉引用会指错）"
            for d in sorted(dup)]
    orphan = sorted(set(labels) - refs)
    out += [f"（提示）\\label{{{o}}} 从未被引用" for o in orphan]
    return out


def check_graphics(t: str, doc: str) -> list[str]:
    out = []
    for m in re.finditer(
            r"\\(?:includegraphics(?:\[[^\]]*\])?|msfigure)\{([^}]+)\}", t):
        rel = m.group(1)
        # `#1` 是 \msfigure 宏定义里的形参，不是文件名。
        if "#" in rel:
            continue
        p = (MS / rel).resolve()
        if p.exists():
            continue
        # LaTeX 允许省略扩展名
        if any(p.with_suffix(s).exists() for s in (".pdf", ".png", ".jpg", ".eps")):
            continue
        out.append(f"图文件不存在：{rel}")
    return out


# 投稿前必须替换的占位符。留在这里比留在脑子里可靠 ——
# 作者、单位、仓库 URL 和 CRediT 是最后一刻才填的东西，也最容易漏。
PLACEHOLDERS = [
    ("Author One", "作者姓名"),
    ("Author Two", "作者姓名"),
    ("Affiliation, City, Country", "单位"),
    ("repository URL", "代码仓库 URL"),
    ("to be completed", "CRediT 贡献声明"),
    ("name@institution.edu", "通讯作者机构邮箱"),
    ("0000-0000-0000-0000", "ORCID"),
    ("TODO(author)", "作者待办"),
]


def check_placeholders(t: str) -> list[str]:
    """占位符尚未替换 —— 报为**提示**而非错误，投稿前必须清零。"""
    return [f"（投稿前必须替换）{why}：{lit}"
            for lit, why in PLACEHOLDERS if lit in t]


# 仓库根上的发布文件也有占位符，而且更容易漏 —— 它们不在 LaTeX 里，
# 所以上面那轮 \input 展开永远看不到它们。arXiv 号与 DOI 直到 v1 才存在，
# 版权人姓名要用户给；三者都只有到「按下发布」的那一刻才有值。
RELEASE_FILES = {
    "README.md": ["arXiv:XXXX.XXXXX", "to be inserted"],
    "LICENSE": ["AUTHOR NAME PENDING"],
    "DATA_LICENSES.md": ["TODO"],
}


def check_release_files() -> list[str]:
    out = []
    for name, lits in RELEASE_FILES.items():
        p = REPO_ROOT / name
        if not p.exists():
            out.append(f"缺少发布必需文件：{name}")
            continue
        txt = p.read_text(encoding="utf-8")
        out += [f"（公开发布前必须替换）{name}：{lit}" for lit in lits if lit in txt]
    return out


def check_cites(t: str) -> list[str]:
    keys = set()
    for bib in (REPO_ROOT / "references").glob("*.bib"):
        keys |= set(re.findall(r"@\w+\{([^,]+),", bib.read_text(encoding="utf-8")))
    if not keys:
        return ["（跳过）references/ 下找不到 .bib"]
    used: set[str] = set()
    for m in re.finditer(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", t):
        used |= {k.strip() for k in m.group(1).split(",")}
    return [f"引用键不在 .bib 里：{k}" for k in sorted(used - keys)]


def main() -> None:
    total = 0
    for doc in DOCS:
        t = load(doc)
        print("=" * 74)
        print(f"{doc}.tex（含 \\input 展开，{len(t.splitlines())} 行）")
        for name, fn in (("环境配对", check_environments),
                         ("括号平衡", check_braces),
                         ("自定义宏", check_macros),
                         ("label / ref", check_refs),
                         ("图文件存在", lambda x: check_graphics(x, doc)),
                         ("引用键", check_cites),
                         ("占位符", check_placeholders)):
            issues = fn(t)
            hard = [i for i in issues if not i.startswith("（")]
            total += len(hard)
            mark = "✅" if not hard else "⚠️"
            print(f"  {mark} {name}" + (f"  —— {len(hard)} 处" if hard else ""))
            for x in issues[:10]:
                print(f"       {x}")

    print("=" * 74)
    print("仓库根上的发布文件")
    rel = check_release_files()
    hard = [i for i in rel if not i.startswith("（")]
    total += len(hard)
    print(f"  {'✅' if not hard else '⚠️'} 占位符与必需文件"
          + (f"  —— {len(hard)} 处" if hard else ""))
    for x in rel:
        print(f"       {x}")

    print("\n" + "=" * 74)
    print(f"共 {total} 处会让编译失败或指错的问题")
    print("\n⚠️ 本脚本**不编译**，只查编译器一定会拒绝的静态结构。")
    print("   查不到：排版溢出、字体缺失、浮动体摆放、参考文献样式。")
    print("   全绿不等于能编译 —— 只等于「不会因为这几类原因编不过」。")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
