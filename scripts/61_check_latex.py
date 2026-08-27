r"""LaTeX 静态检查 —— 本机没有 TeX，编译验证做不了，能查的都查。

    .venv/Scripts/python scripts/61_check_latex.py

## 这个脚本能保证什么、不能保证什么

**不能**保证 `.tex` 能编译通过 —— 那需要真正跑一遍 pdflatex。
本机未安装 TeX，因此**不得声称已验证编译**。

**能**查的是最常见的几类硬错误：

| 检查 | 抓什么 |
|---|---|
| `\begin`/`\end` 配对 | 环境未闭合 |
| 花括号计数 | 括号不平衡（逐行报告首个失衡处） |
| `\cite*` 键 vs `.bib` | 悬空引用 |
| `\ref` vs `\label` | 悬空交叉引用 |
| `\includegraphics` 路径 | 图文件缺失 |
| 常见宏是否有对应 `\usepackage` | 用了没加载的宏 |
"""

from __future__ import annotations

import re
from pathlib import Path

from soundml.provenance import REPO_ROOT

MAN = REPO_ROOT / "manuscript"
BIB = REPO_ROOT / "references" / "references.bib"

# 宏 → 提供它的宏包
NEEDS = {
    "includegraphics": "graphicx", "toprule": "booktabs",
    "midrule": "booktabs", "bottomrule": "booktabs",
    "url": "hyperref", "citep": "natbib", "citet": "natbib",
    "citealp": "natbib", "linenumbers": "lineno",
    "upmu": "upgreek", "captionsetup": "caption",
    "affil": "authblk", "onehalfspacing": "setspace",
    "texttt": None, "textbf": None,
}


def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def check(tex: Path, preamble: str, bibkeys: set[str]) -> list[str]:
    raw = tex.read_text(encoding="utf-8")
    s = strip_comments(raw)
    full = preamble + "\n" + s
    errs = []

    # ---- 环境配对 ----
    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", s):
        kind, env = m.group(1), m.group(2)
        if kind == "begin":
            stack.append((env, s[:m.start()].count("\n") + 1))
        else:
            if not stack:
                errs.append(f"L{s[:m.start()].count(chr(10))+1}: \\end{{{env}}} 无对应 begin")
            elif stack[-1][0] != env:
                errs.append(f"L{s[:m.start()].count(chr(10))+1}: \\end{{{env}}} "
                            f"但栈顶是 {stack[-1][0]}（L{stack[-1][1]}）")
                stack.pop()
            else:
                stack.pop()
    for env, ln in stack:
        errs.append(f"L{ln}: \\begin{{{env}}} 未闭合")

    # ---- 花括号平衡（忽略转义的 \{ \}）----
    depth, first_bad = 0, None
    for i, line in enumerate(s.split("\n"), 1):
        cleaned = re.sub(r"\\[{}]", "", line)
        depth += cleaned.count("{") - cleaned.count("}")
        if depth < 0 and first_bad is None:
            first_bad = i
    if depth != 0:
        errs.append(f"花括号不平衡：净 {depth:+d}"
                    + (f"，首个负值在 L{first_bad}" if first_bad else ""))

    # ---- 引用键 ----
    cited = set()
    for m in re.finditer(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", s):
        cited |= {k.strip() for k in m.group(1).split(",")}
    for k in sorted(cited - bibkeys):
        errs.append(f"悬空引用：{k}")

    # ---- 交叉引用 ----
    labels = set(re.findall(r"\\label\{([^}]+)\}", s))
    refs = set()
    for m in re.finditer(r"\\(?:ref|autoref|eqref)\{([^}]+)\}", s):
        refs.add(m.group(1))
    for r in sorted(refs - labels):
        errs.append(f"悬空 \\ref：{r}")

    # ---- 图文件 ----
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s):
        rel = m.group(1)
        if not (tex.parent / rel).resolve().exists():
            errs.append(f"图文件缺失：{rel}")

    # ---- 宏包 ----
    loaded = set()
    for m in re.finditer(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", full):
        loaded |= {p.strip() for p in m.group(1).split(",")}
    for macro, pkg in NEEDS.items():
        if pkg and re.search(rf"\\{macro}\b", s) and pkg not in loaded:
            errs.append(f"用了 \\{macro} 但未加载 {pkg}")

    return errs


def main() -> None:
    bibkeys = set(re.findall(r"@\w+\{([^,]+),", BIB.read_text(encoding="utf-8")))
    preamble = strip_comments((MAN / "preamble.tex").read_text(encoding="utf-8"))
    print(f"bib 条目 {len(bibkeys)}\n")

    total = 0
    for name in ("main.tex", "supplementary.tex"):
        tex = MAN / name
        errs = check(tex, preamble, bibkeys)
        total += len(errs)
        print(f"=== {name} ===")
        if errs:
            for e in errs:
                print(f"  ❌ {e}")
        else:
            print("  ✅ 静态检查未发现问题")
        print()

    print("=" * 68)
    if total == 0:
        print("静态检查全部通过。")
    else:
        print(f"共 {total} 处问题。")
    print("\n⚠️ **本机未安装 TeX，未做编译验证。**")
    print("   静态检查只覆盖配对、引用、路径、宏包这几类硬错误；")
    print("   排版细节、字体、浮动体位置等只有编译才能确认。")
    print("   编译：cd manuscript && pdflatex main && bibtex main && "
          "pdflatex main && pdflatex main")


if __name__ == "__main__":
    main()
