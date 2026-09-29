r"""把 FIGURE_LEGENDS.md 的图注转成 LaTeX 片段供 .tex \input。

    .venv/Scripts/python scripts/63_legends_to_tex.py

## 为什么要转而不是手抄

图注在两个地方各写一份，改一处忘另一处是迟早的事，而图注里全是具体数字。
`FIGURE_LEGENDS.md` 保持单一事实来源，本脚本生成 `manuscript/legends/*.tex`，
`.tex` 只 `\input`。

## 转换要处理的

Markdown 里的排版字符在 LaTeX 里多数需要转义或改写：

| 源 | 目标 | 原因 |
|---|---|---|
| `**a**` | `\textbf{a}` | 面板标签 |
| `ρ` `σ` `μ` | `$\rho$` … | 希腊字母须在数学模式 |
| `×` `≤` `≥` `≈` `−` | `$\times$` … | 同上 |
| `—` `–` | `---` `--` | LaTeX 破折号 |
| `%` `&` `_` `#` | `\%` … | LaTeX 特殊字符 |
| `10⁻⁹` | `$10^{-9}$` | 上标 |
| `µV` | `\uV` | preamble 里定义的宏 |

**转义顺序要紧**：必须先处理反斜杠再处理其他，否则会把自己生成的 `\rho`
当成待转义文本。脚本按「先占位、后还原」的顺序做。
"""

from __future__ import annotations

import re

from soundml.provenance import REPO_ROOT

SRC = REPO_ROOT / "reports" / "FIGURE_LEGENDS.md"
OUT = REPO_ROOT / "manuscript" / "legends"

SUPER = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")

# 先做数学替换 → 存成占位符 → 转义 LaTeX 特殊字符 → 还原占位符
MATH = {
    "ρ": r"\rho", "σ": r"\sigma", "μ": r"\mu", "α": r"\alpha",
    "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "θ": r"\theta",
    "Δ": r"\Delta", "×": r"\times", "≤": r"\le", "≥": r"\ge",
    "≈": r"\approx", "±": r"\pm", "→": r"\rightarrow", "↔": r"\leftrightarrow",
    "∈": r"\in", "−": "-",          # U+2212 减号，不是 ASCII 连字符
}
TEXT = {"—": "---", "–": "--", "µV": r"\uV{}", "μV": r"\uV{}",
        "…": r"\ldots{}",
        "’": "'", "‘": "'", "“": "``", "”": "''"}
ESCAPE = {"%": r"\%", "&": r"\&", "#": r"\#", "_": r"\_"}


def convert(md: str) -> str:
    # 占位符的开闭分隔符**必须不同**，且还原走单次正则而非顺序替换。
    #
    # 初版用 `\x00N\x00`（开闭同字符）+ 顺序 str.replace，结果 `−1…1` 被
    # 毁成 `17σ22 1`：展开后是 `\x00 1 7 \x00 | 1 | \x00 2 2 \x00`，
    # 前一个占位符的结尾 + 字面量 "1" + 后一个占位符的开头，
    # 恰好构成合法的 `\x001\x00`，于是被 σ 顶替。
    ph: list[str] = []

    def stash(tex: str) -> str:
        ph.append(tex)
        return f"\x00{len(ph) - 1}\x01"

    s = md
    # 上标数字（10⁻⁹ 之类）
    s = re.sub(r"10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
               lambda m: stash(f"$10^{{{m.group(1).translate(SUPER)}}}$"), s)
    for ch, tex in MATH.items():
        s = s.replace(ch, stash(f"${tex}$"))
    for a, b in TEXT.items():
        s = s.replace(a, stash(b))
    # 粗体（面板标签）
    s = re.sub(r"\*\*(.+?)\*\*", lambda m: stash(r"\textbf{") + m.group(1) + stash("}"), s)
    # 反引号代码
    s = re.sub(r"`([^`]+)`", lambda m: stash(r"\texttt{") + m.group(1) + stash("}"), s)

    for a, b in ESCAPE.items():
        s = s.replace(a, b)
    s = re.sub("\x00(\\d+)\x01", lambda m: ph[int(m.group(1))], s)
    assert "\x00" not in s and "\x01" not in s, "占位符未完全还原"

    # 段内换行归一
    s = re.sub(r"[ \t]*\n[ \t]*", " ", s)
    return re.sub(r"[ \t]{2,}", " ", s).strip()


WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten"]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    text = SRC.read_text(encoding="utf-8")
    defs: list[str] = []

    # 以 "## Fig. N | 标题" / "## Fig. SN | 标题" 切块。
    #
    # 补充材料的图从 "Extended Data Fig. N" 改名为 "Fig. SN"（IEEE 没有
    # Extended Data 这个建制，投 TAC 时统一为 S 编号）。这里必须跟着改：
    # 旧正则 `(Extended Data )?Fig\. (\d+)` 对 "Fig. S9" 不匹配，而不匹配的块
    # 是 `continue` 掉的 —— 图注会**静默消失**，.tex 里少一个宏，编译时
    # 才以 "Undefined control sequence" 的形式暴露出来。
    blocks = re.split(r"^## ", text, flags=re.M)[1:]
    n_main = n_ed = 0
    seen = set()
    for b in blocks:
        head, _, body = b.partition("\n")
        body = body.split("\n---")[0].strip()
        m = re.match(r"Fig\. (S?)(\d+)\s*\|\s*(.+)", head.strip())
        if not m:
            continue
        is_ed, num, title = m.group(1) == "S", int(m.group(2)), m.group(3)
        # 图注定义成宏，在导言区加载；caption 只引用宏。
        #
        # `\caption` 的参数是 moving argument，`\input` 在其中是 fragile 命令，
        # 会报 "Argument of \caption@ydblarg has an extra }"。
        # 把内容先定义为宏就绕开了这一层 —— 宏展开后只剩文本与 \textbf。
        # 宏名不能含数字，故用英文数词。
        macro = f"legend{'ed' if is_ed else 'fig'}{WORDS[num]}"
        assert macro not in seen, f"图注宏重名：{macro}（两个块编号相同？）"
        seen.add(macro)
        defs.append(f"\\newcommand{{\\{macro}}}{{%\n"
                    f"\\textbf{{{convert(title)}.}} {convert(body)}}}")
        n_main += not is_ed
        n_ed += is_ed
        print(f"  \\{macro:22s} {len(convert(body).split())} words")

    (OUT / "all.tex").write_text(
        "% Generated from reports/FIGURE_LEGENDS.md by scripts/63_legends_to_tex.py\n"
        "% Do not edit here; edit the Markdown and regenerate.\n\n"
        + "\n\n".join(defs) + "\n", encoding="utf-8")
    print(f"\n主图 {n_main} 个，Extended Data {n_ed} 个 → {OUT / 'all.tex'}")

    # 转换后不应残留原始 Unicode 排版字符
    leftovers = set()
    for f in OUT.glob("*.tex"):
        for ch in f.read_text(encoding="utf-8"):
            if ord(ch) > 0x2000 and ch not in "\u2013\u2014":
                leftovers.add(ch)
    if leftovers:
        print(f"⚠️ 仍有未转换的 Unicode 字符：{''.join(sorted(leftovers))}")
    else:
        print("✅ 无残留的未转换字符")


if __name__ == "__main__":
    main()
