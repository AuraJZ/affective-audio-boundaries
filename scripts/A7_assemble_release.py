r"""组装公开发布的仓库目录 —— 只复制,不建 git,不推送。

    .venv/Scripts/python scripts/A7_assemble_release.py <目标目录>

## 为什么不直接推现在这个仓库

工作仓库的历史里有三样东西,`git rm` 删不掉:

  * 27 个 commit 的 author 全是作者的私人 Gmail
  * 已移出工作区、但仍在历史里的专利相关文件(规则阈值与权利要求撰写策略)
  * 同样在历史里的第三方研究者邮箱

重写历史可以解决,但代价和风险都大于「新建一个仓库、单个初始 commit」。
所以这个脚本只做一件事:把**该公开的那一份**拷到一个干净目录,让人在
`git init` 之前先用 `git status` 亲眼看一遍。

## 硬闸

复制完成后逐条检查,任一不过就以非零退出:

  * 原始音频与生理记录(.mp3/.wav/.edf/…)—— 多数语料不可再分发
  * 本机绝对路径、私人邮箱、API 令牌
  * 体积异常(默认 > 80 MB 说明有大文件混入)

占位符(作者姓名、仓库 URL、ORCID)只**警告**不拦截:仓库 URL 要等仓库
建好才知道,这是个先有鸡还是先有蛋的问题,所以允许先推、后填。
"""

from __future__ import annotations

import io
import re
import shutil
import sys
from pathlib import Path

from soundml.provenance import REPO_ROOT

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 进公开仓库的东西。顺序即 README 里介绍的顺序。
INCLUDE = [
    ("README.md",          "仓库门面,也是 Zenodo 归档页的描述来源"),
    ("LICENSE",            "代码的 MIT 许可,含 SCOPE 段"),
    ("DATA_LICENSES.md",   "九个语料与四组模型权重的许可台账"),
    (".gitignore",         "挡住 data/、.venv、logs、tmp"),
    ("pyproject.toml",     "包与依赖定义"),
    ("uv.lock",            "锁定的精确版本"),
    ("manuscript",         "手稿 LaTeX 源、图注、编译好的 PDF"),
    ("references",         "参考文献"),
    ("figures_r",          "全部出图的 R 源码"),
    ("scripts",            "分析脚本与检查器"),
    ("src",                "分析库 soundml"),
    ("reports",            "PLAN、结果汇总、图注、Source Data、渲染好的图"),
    ("runs",               "每次分析的运行记录"),
]

SKIP_DIRS   = {"__pycache__", ".venv", ".git", ".pytest_cache", ".ipynb_checkpoints",
               "logs", "tmp", "dist", "data"}

# 🔴 被废弃的助眠规则库那条线。它的人类可读报告早已移出仓库,但**产生它们的
# 运行记录留了下来**,而记录里是同一批数字的 JSON 形式:具名特征 + 数值阈值 +
# 方向,二十多条。那就是一项权利要求会写的东西。
#
# 第一次公开发布时它们随仓库上了线 —— 因为当时只按文件名清理了 reports/,
# 没想到 runs/ 里还有一份。下面的名字排除加上后面的内容闸,两道一起挡。
EXCLUDE_PATHS = [
    "scripts/06_rules_draft.py",          # 生成器;论文不引用
    "runs/20260727T083739Z_rules_draft",
    "runs/20260727T085524Z_rules_draft",
    "runs/20260727T094537Z_rules_draft",
]

# 内容闸:同时含有阈值键与「助眠」方向注解的文件,就是规则表本身。
# 只写文件名排除挡不住下一次 —— 换个 run id 就漏过去了。
RULE_TABLE_SIGNATURE = (r'"threshold', r"越[低高]越助眠")
SKIP_SUFFIX = {".pyc", ".pyo", ".aux", ".log", ".blg", ".out", ".toc",
               ".synctex.gz", ".fls", ".fdb_latexmk", ".locktest"}
# 🔴 .bbl 是构建产物,但 arXiv 不跑 BibTeX,投稿包必须带它。
#    公开仓库不需要,由 A6 打包时单独取。
SKIP_SUFFIX |= {".bbl"}

FORBIDDEN_SUFFIX = {".mp3", ".wav", ".edf", ".flac", ".ogg", ".m4a", ".aac", ".bdf"}
MAX_MB = 80

LEAKS = [
    (r"[A-Za-z0-9._%+-]+@(gmail|outlook|hotmail|yahoo|qq|163|126)\.", "个人邮箱"),
    (r"(?<![A-Za-z0-9])[A-Za-z]:\\|/Users/|/home/[a-z]|AppData", "本机绝对路径"),
    (r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|Bearer\s+[A-Za-z0-9._-]{16,})\b",
     "疑似令牌"),
    (r"专利|权利要求|交底", "专利相关措辞"),
]
TEXT_SUFFIX = {".py", ".r", ".R", ".md", ".tex", ".txt", ".toml", ".yaml", ".yml",
               ".json", ".csv", ".cfg", ".gitignore", ""}

# 🔴 扫描器必然含有它要找的模式 —— 它的正则、它的用法示例、它对「什么算泄露」
# 的说明,逐条都会命中自己。第一次跑出来的四条全是这种自指命中。
#
# 压制必须**可见**:凡是被放行的都打印出来。一条看不见的豁免,正是真泄露
# 溜过去的方式。
ALLOW = [
    ("scripts/A6_build_arxiv.py", "扫描器自身的正则与示例"),
    ("scripts/A7_assemble_release.py", "本脚本自身的正则与示例"),
    ("scripts/lexicon.py", "术语门禁自身的词表"),
]
ALLOW_LITERAL = [
    ("专利代理", "法务免责声明,非披露"),
    ("进权利要求前", "同上"),
]

# 🔴 占位符要按**带方括号的原样**匹配。裸写 "repository URL" 会命中三处正常
# 英文:README 里「引用 Zenodo DOI 而不是仓库 URL」那句,以及注释里对已删除
# 机制的说明。会误报的警告没人看。
PLACEHOLDERS = ["Author One", "Author Two", "Affiliation, City, Country",
                "[repository URL]", "name@institution.edu", "0000-0000-0000-0000",
                "AUTHOR NAME PENDING", "arXiv:XXXX.XXXXX"]


def excluded(rel_posix: str) -> bool:
    return any(rel_posix == e or rel_posix.startswith(e + "/") for e in EXCLUDE_PATHS)


def copy_into(src: Path, dst: Path, prefix: str = "") -> int:
    if src.is_file():
        if excluded(prefix):
            return 0
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return 1
    n = 0
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in SKIP_SUFFIX:
            continue
        rel = f"{prefix}/{p.relative_to(src).as_posix()}".lstrip("/")
        if excluded(rel):
            continue
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        n += 1
    return n


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("用法:A7_assemble_release.py <目标目录>")
    dest = Path(sys.argv[1]).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"🔴 目标目录非空,先清空或换一个:{dest}")
    dest.mkdir(parents=True, exist_ok=True)

    print(f"组装到 {dest}\n")
    total = 0
    for rel, why in INCLUDE:
        src = REPO_ROOT / rel
        if not src.exists():
            print(f"  ⚠️ 缺失,跳过  {rel}")
            continue
        n = copy_into(src, dest / rel, prefix=rel)
        total += n
        print(f"  {rel:<20} {n:5d} 个文件   {why}")

    files = [p for p in dest.rglob("*") if p.is_file()]
    size_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"\n合计 {len(files)} 个文件,{size_mb:.1f} MB")

    problems: list[str] = []

    # 内容闸:规则表 = 阈值键 + 「助眠」方向注解同时出现。
    # 名字排除只挡已知的那几个;换个 run id 就绕过去了,所以要按内容再查一遍。
    for p in files:
        if p.suffix.lower() not in {".json", ".md", ".csv", ".py", ".txt"}:
            continue
        if p.stat().st_size > 4_000_000:
            continue
        try:
            t = p.read_text(encoding="utf-8", errors="replace")
        except Exception:                                    # noqa: BLE001
            continue
        if all(re.search(pat, t) for pat in RULE_TABLE_SIGNATURE):
            problems.append(
                f"规则表混入:{p.relative_to(dest)} —— 具名特征 + 数值阈值 + 方向。"
                "这是被废弃的助眠规则库那条线的产物,公开即成为对自己申请的现有技术")

    media = [p for p in files if p.suffix.lower() in FORBIDDEN_SUFFIX]
    if media:
        problems.append(f"混入原始媒体文件 {len(media)} 个 —— 多数语料不可再分发")
        for p in media[:5]:
            problems.append(f"      {p.relative_to(dest)}")

    if size_mb > MAX_MB:
        problems.append(f"体积 {size_mb:.1f} MB 超过 {MAX_MB} MB,检查大文件")
        for p in sorted(files, key=lambda x: -x.stat().st_size)[:5]:
            problems.append(f"      {p.stat().st_size/1e6:6.1f} MB  {p.relative_to(dest)}")

    allowed: list[str] = []
    for p in files:
        if p.suffix.lower() not in TEXT_SUFFIX or p.stat().st_size > 2_000_000:
            continue
        rel = p.relative_to(dest).as_posix()
        skip = next((why for f, why in ALLOW if rel == f), None)
        if skip:
            allowed.append(f"整份放行  {rel}  —— {skip}")
            continue
        try:
            t = p.read_text(encoding="utf-8", errors="replace")
        except Exception:                                    # noqa: BLE001
            continue
        t = re.sub(r"\\url\{[^}]*\}|\w+://\S*", " ", t)
        for pat, why in LEAKS:
            for m in re.finditer(pat, t):
                ln = t[:m.start()].count("\n") + 1
                ctx = t[max(0, m.start() - 30):m.end() + 30]
                lit = next((w for s, w in ALLOW_LITERAL if s in ctx), None)
                if lit:
                    allowed.append(f"{rel}:{ln}  「{m.group(0)[:24]}」 —— {lit}")
                    continue
                problems.append(f"{why}:{rel}:{ln}  「{m.group(0)[:40]}」")
                break
            else:
                continue
            break

    left = [ph for ph in PLACEHOLDERS
            if any(ph in p.read_text(encoding="utf-8", errors="replace")
                   for p in files if p.suffix in {".tex", ".md"} and p.stat().st_size < 500_000)]

    if problems:
        print(f"\n🔴 {len(problems)} 项必须处理:")
        for x in problems:
            print(f"    {x}")
        raise SystemExit(1)

    print("✅ 无原始媒体、无个人邮箱、无本机路径、无专利措辞,体积正常")
    if allowed:
        print(f"\n以下 {len(allowed)} 处被放行 —— 逐条列出,不静默压制:")
        for a in allowed:
            print(f"    {a}")
    if left:
        print(f"\n⚠️ 仍有 {len(left)} 类占位符未填(不拦截 —— 仓库 URL 要等仓库建好才知道):")
        for ph in left:
            print(f"    {ph}")

    print(f"""
下一步(在 {dest} 里):

    git init -b main
    git config user.email "<你的 GitHub noreply 邮箱>"
    git config user.name  "<你的名字>"
    git add -A
    git status          # ← 在 commit 之前，把这份清单从头看一遍
    git commit -m "Initial public release: analysis code, manuscript and figures"
""")


if __name__ == "__main__":
    main()
