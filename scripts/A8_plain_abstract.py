r"""把摘要转成 arXiv 元数据框能用的纯文本。

    .venv/Scripts/python scripts/A8_plain_abstract.py

## 顺序要紧

第一版先剥 `%` 注释、再把 `\%` 还原成 `%`,于是 `84\%` 那一行从百分号往后
整句被当成注释删掉了,输出成了 "84\ how much listeners agree" —— 一句被
掐头去尾的话,粘到 arXiv 上不会报错,只会安静地错着。

正确顺序:先用负向后顾只剥**未转义**的 `%`,再解转义。
"""
import io
import re
import sys
from pathlib import Path

from soundml.provenance import REPO_ROOT

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

src = (REPO_ROOT / "manuscript" / "main.tex").read_text(encoding="utf-8")
m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S)
if not m:
    raise SystemExit("找不到 abstract 环境")
a = m.group(1)

# 1) 只剥未转义的注释
a = re.sub(r"(?<!\\)%.*", "", a)
# 2) 内容宏 -> 其内容
for pat in (r"\\textbf\{([^{}]*)\}", r"\\emph\{([^{}]*)\}", r"\\textit\{([^{}]*)\}",
            r"\\texttt\{([^{}]*)\}"):
    a = re.sub(pat, r"\1", a)
a = re.sub(r"\\citep?\*?(?:\[[^\]]*\])*\{[^}]*\}", "", a)
# 3) 数学与符号
a = a.replace(r"\rho", "rho").replace(r"\sqrt", "sqrt")
a = re.sub(r"\$([^$]*)\$", r"\1", a)
# 4) 解转义（必须在剥注释之后）
for esc, plain in ((r"\%", "%"), (r"\&", "&"), (r"\_", "_"), (r"\#", "#")):
    a = a.replace(esc, plain)
# 5) TeX 排版记号
a = a.replace("---", "—").replace("--", "–")
a = a.replace("``", "\u201c").replace("''", "\u201d")
a = re.sub(r"\\[a-zA-Z]+\s*", "", a)          # 残余控制序列
a = re.sub(r"[{}]", "", a)
a = re.sub(r"\s+", " ", a).strip()

print(a)
print()
words = len(a.split())
print(f"[{len(a)} 字符 · {words} 词]")
if words > 250:
    print(f"⚠️ {words} 词。arXiv 不设硬上限,但多数摘要在 150–250 词;")
    print("   IEEE 一类期刊通常要求 ≤ 200 词,投期刊时需另写一版。")
leftover = re.findall(r"\\[a-zA-Z]+|[{}$]", a)
if leftover:
    print(f"🔴 仍有未清理的 TeX 记号:{set(leftover)}")
