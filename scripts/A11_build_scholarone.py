r"""打包 ScholarOne 投稿文件夹 —— 每个上传槽一个目录，按表单的分类命名。

    .venv/Scripts/python scripts/A11_build_scholarone.py

## 为什么不能直接传 PDF

ScholarOne 的 Main Manuscript 槽写的是 **MS Word or LaTeX**，不收 PDF：

  > You may bundle LaTeX manuscript files in a single archive including all
  > LaTeX files, BibTeX files, figures, tables, all LaTeX classes and packages,
  > and any other material that belongs to your main manuscript

所以主稿必须是一个**能独立编译**的源码包。仓库里的 `main.tex` 引的是
`../reports/figures_r/Fig1_wall.pdf` 与 `../references/references.bib` —— 一压扁
这些路径全断，而 LaTeX 的断链只在编译时才报，投出去才发现就晚了。

本脚本因此做三件事，**并且真的编译一遍来证明**：

  1. 图复制进 `figures/`，把 `\includegraphics` 的路径改写过去；
  2. `references.bib` 与预构建的 `main.bbl` 一起放进根目录 ——
     审稿系统不一定跑 bibtex，有 `.bbl` 才保证参考文献一定出得来；
  3. 非标准宏包随包带上（这里是 `orcidlink.sty`）。`IEEEtran.cls` 是
     ScholarOne 一定有的，但带上也无害，且省掉"他们那边版本不同"的风险。

最后**把包解到一个干净的临时目录里编译**。不这样做，等于只验证了"我这台机器上
能编"，而那恰恰是最没有意义的验证 —— 断的路径在仓库里照样能编，因为
`../reports/` 就在那儿。

## 各槽放什么

| 目录 | ScholarOne 槽 | 内容 |
|---|---|---|
| `01_Main_Manuscript` | Main Manuscript（必填） | LaTeX 源码 zip，含图与 .bbl |
| `02_Supplementary_Material_for_Review` | Supplementary Material for Review | 补充材料 PDF |
| `03_Cover_Letter` | Cover letter / Comments | 投稿信 |
| `04_Previously_Published_Files` | Previously Published – Files | 预印本 PDF |
| `05_Previously_Published_Statement` | Previously Published – Statement | 差异说明 |

Image 槽不用 —— 图已嵌在主稿里。
LaTeX Supplementary File 槽不用 —— `orcidlink.sty` 已在主包内。
Main Document - Tracked Changes 是返修时才要的。

## 🔴 手写文档放在仓库里，不放在 dist/

投稿信与差异说明的 `.tex` 在 `manuscript/submission/`，由本脚本编译后复制进
`dist/scholarone/`。

第一版不是这样：那两份文档直接写在 `dist/scholarone/` 里，而本脚本开头有一句
`shutil.rmtree(OUT)` —— 下一次打包**把它们连同下载好的预印本 PDF 一起删了**。
`dist/` 是 gitignored 的，所以 git 也救不回来。

教训有两条，都写进设计里了：

  1. **手写的东西归版本库**，产物才归 `dist/`；
  2. 打包脚本**只清理自己生成的那几个目录**，不清整棵输出树。

预印本 PDF 同理：缓存在 `manuscript/submission/` 下，没有就按 arXiv ID 下载，
下载失败**报错退出**而不是静默跳过 —— 少传这一份是学术诚信问题。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"
SUB = MS / "submission"          # 手写的投稿文档（版本库里）
OUT = ROOT / "dist" / "scholarone"

# 已公开的预印本。ScholarOne 的 Previously Published 两槽必须填。
ARXIV_ID = "2608.27674v1"
PREPRINT = SUB / f"arXiv_{ARXIV_ID}_preprint.pdf"

# 本脚本拥有、因而可以重建的目录。清理只清这几个 —— 不是整棵 OUT。
OWNED = ["01_Main_Manuscript", "02_Supplementary_Material_for_Review",
         "03_Cover_Letter", "04_Previously_Published_Files",
         "05_Previously_Published_Statement", "06_Form_Fields"]

# MiKTeX 不在 PATH 上时的兜底。找不到就跳过编译验证并**明说**，
# 而不是假装验证过了。
LATEX_HINTS = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/MiKTeX/miktex/bin/x64",
    Path("/usr/bin"), Path("/usr/local/bin"),
]


def find_tool(name: str) -> str | None:
    if shutil.which(name):
        return name
    for d in LATEX_HINTS:
        p = d / (name + (".exe" if os.name == "nt" else ""))
        if p.exists():
            return str(p)
    return None


def strip_comment_lines(text: str) -> str:
    r"""删掉整行 `%` 注释。**只删整行**，行尾注释一律保留。

    ## 为什么要删

    Main Manuscript 槽收的是 LaTeX **源码**，不是 PDF，所以审稿人拿到的是
    `main.tex`、`preamble.tex`、`references.bib` 本身。而这三个文件里的注释是
    写给维护者的：`main.tex` 开头 28 行是一份撤回备忘、`preamble.tex` 里算着
    超页费、`references.bib` 里有"预印本这两个方法都用了却一个都没署名"这种
    自陈，以及"别再改回去"这类对未来自己的指令。

    这些内容本身不见不得人 —— 该披露的都在投稿信和补充材料的 Revision record
    里说了。问题在于**它们出现在一个审稿人会打开、而作者没打算给他看的地方**，
    而且是以备忘录的口吻。仓库里全部保留；只有进包的副本剥掉。

    ## 为什么不能删行尾注释

    LaTeX 里行尾的 `%` 是**吞掉换行符**的，删了会在不该有空格的地方长出空格。
    只按「第一个非空白字符是 %」这一条判，既安全又够用。
    """
    keep = [ln for ln in text.splitlines() if not ln.lstrip().startswith("%")]
    # 连续空行压成一行，免得剥完留下大片空白
    out, blank = [], False
    for ln in keep:
        if ln.strip():
            out.append(ln)
            blank = False
        elif not blank:
            out.append("")
            blank = True
    return "\n".join(out) + "\n"


def stage_main(stage: Path) -> list[str]:
    """把主稿铺成一个自足的扁平目录，返回问题清单。"""
    problems: list[str] = []
    (stage / "figures").mkdir(parents=True, exist_ok=True)

    main = (MS / "main.tex").read_text(encoding="utf-8")

    # 图：先收集，再改写。顺序反过来就会把改写后的 figures/xxx.pdf 当成
    # 仓库路径去找，一个都找不到。
    wanted = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", main)
    for rel in wanted:
        src = (MS / rel).resolve()
        if not src.exists():
            problems.append(f"图不存在：{rel}")
            continue
        shutil.copy2(src, stage / "figures" / src.name)
    main = re.sub(r"(\\includegraphics(?:\[[^\]]*\])?\{)[^}]*?([^/}]+\.pdf\})",
                  r"\1figures/\2", main)

    # 参考文献：.bib 供重跑，.bbl 保证不跑 bibtex 也出得来。
    bib = ROOT / "references" / "references.bib"
    (stage / "references.bib").write_text(
        strip_comment_lines(bib.read_text(encoding="utf-8")), encoding="utf-8")
    main = main.replace("../references/references", "references")

    bbl = MS / "main.bbl"
    if not bbl.exists():
        problems.append("manuscript/main.bbl 不存在 —— 先在 manuscript/ 跑一遍 "
                        "pdflatex + bibtex")
    elif bbl.stat().st_mtime < bib.stat().st_mtime:
        problems.append("main.bbl 比 references.bib 旧 —— 重跑 bibtex")
    else:
        shutil.copy2(bbl, stage / "main.bbl")

    (stage / "preamble.tex").write_text(
        strip_comment_lines((MS / "preamble.tex").read_text(encoding="utf-8")),
        encoding="utf-8")
    (stage / "main.tex").write_text(strip_comment_lines(main), encoding="utf-8")

    # 类与宏包都随包走。
    #
    # IEEEtran.cls 原本没带，理由是"ScholarOne 一定有"。但表单明说可以打包
    # "all LaTeX classes and packages"，而这份稿子正好卡在 18/18 页的硬上限：
    # 对方装的 IEEEtran 版本只要有一点度量差异，就可能顶成 19 页而被退回。
    # 零余量的时候，把类固定下来是最便宜的保险。
    kpse = find_tool("kpsewhich")
    for sty in ("IEEEtran.cls", "orcidlink.sty"):
        if not kpse:
            problems.append(f"找不到 kpsewhich，无法定位 {sty}")
            continue
        r = subprocess.run([kpse, sty], capture_output=True, text=True)
        path = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
        if path and Path(path).exists():
            shutil.copy2(path, stage / sty)
        else:
            problems.append(f"找不到 {sty} —— 投稿系统若没有它就编不过")

    # 确认没有残留的相对上跳路径
    for m in re.finditer(r"\.\./", main):
        problems.append(f"改写后仍有 ../ 路径：…{main[max(0,m.start()-40):m.end()+30]}…")
    return problems


def compile_check(stage: Path) -> list[str]:
    """在**解压后的干净目录**里编译。在原地编译等于什么都没验证。"""
    pdflatex = find_tool("pdflatex")
    if not pdflatex:
        return ["（跳过）找不到 pdflatex —— 编译验证未执行，不代表通过"]
    # TEXINPUTS 不要覆盖。第一版设成 f"{stage};"，意图是"先找包里的"，
    # 实际效果是把 MiKTeX 的默认搜索树挤掉了一部分，编译静默失败且不留 .log。
    # 包在自己的目录里编译，当前目录本来就是第一优先，不需要这一手。
    last = None
    for _ in range(2):
        last = subprocess.run([pdflatex, "-interaction=nonstopmode", "main.tex"],
                              cwd=stage, capture_output=True, text=True,
                              errors="replace")
    log = (stage / "main.log")
    if not (stage / "main.pdf").exists():
        tail = (last.stdout or "")[-1200:] if last else ""
        err = [ln for ln in tail.splitlines()
               if ln.startswith("!") or "not found" in ln.lower()]
        return ["编译没有产出 main.pdf"] + [f"  {e}" for e in err[:5]] or \
               ["编译没有产出 main.pdf（pdflatex 无输出）"]
    text = log.read_text(encoding="utf-8", errors="replace")
    bad = [ln for ln in text.splitlines() if ln.startswith("!")]
    und = len(re.findall(r"(Reference|Citation) `[^']*' on page \d+ undefined", text))
    pages = re.search(r"Output written on main\.pdf \((\d+) pages", text)
    out = []
    if bad:
        out += [f"编译报错：{b}" for b in bad[:5]]
    if und:
        out.append(f"解压后编译有 {und} 处未定义引用 —— 包里缺 .bbl 或 .aux 依赖")
    print(f"  解压后独立编译：{pages.group(1) if pages else '?'} 页，"
          f"{len(bad)} 处报错，{und} 处未定义引用")
    return out


def zipdir(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(src.rglob("*")):
            if p.is_file() and p.suffix not in (".aux", ".log", ".out", ".pdf") \
                    or (p.is_file() and p.parent.name == "figures"):
                z.write(p, p.relative_to(src))


def compile_doc(tex: Path, dest_dir: Path) -> list[str]:
    """把一份独立的 .tex 编译成 PDF 放进目标槽目录。"""
    pdflatex = find_tool("pdflatex")
    if not pdflatex:
        return [f"（跳过）找不到 pdflatex —— {tex.name} 未编译"]
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        shutil.copy2(tex, work / tex.name)
        for _ in range(2):
            subprocess.run([pdflatex, "-interaction=nonstopmode", tex.name],
                           cwd=work, capture_output=True)
        pdf = work / (tex.stem + ".pdf")
        if not pdf.exists():
            log = work / (tex.stem + ".log")
            err = ""
            if log.exists():
                bad = [ln for ln in log.read_text(encoding="utf-8", errors="replace")
                       .splitlines() if ln.startswith("!")]
                err = f"：{bad[0]}" if bad else ""
            return [f"{tex.name} 编译失败{err}"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf, dest_dir / pdf.name)
        pages = re.search(r"Output written on .*?\((\d+) pages",
                          (work / (tex.stem + ".log")).read_text(
                              encoding="utf-8", errors="replace"))
        print(f"  {pdf.name}（{pages.group(1) if pages else '?'} 页）")
    return []


def fetch_preprint() -> list[str]:
    """预印本 PDF：优先用缓存，没有就下载。失败必须报错，不能静默跳过。"""
    if PREPRINT.exists() and PREPRINT.stat().st_size > 100_000:
        print(f"  {PREPRINT.name}（缓存，{PREPRINT.stat().st_size // 1024} KB）")
        return []
    url = f"https://arxiv.org/pdf/{ARXIV_ID}"
    print(f"  缓存缺失，下载 {url}")
    try:
        import urllib.request
        SUB.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r, \
                open(PREPRINT, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception as e:                                # noqa: BLE001
        return [f"预印本 PDF 取不到（{e}）—— Previously Published 槽会缺件，"
                f"而漏报公开预印本是学术诚信问题，不是格式问题"]
    if PREPRINT.stat().st_size < 100_000:
        return [f"下载到的 {PREPRINT.name} 只有 "
                f"{PREPRINT.stat().st_size} 字节 —— 多半是错误页而不是 PDF"]
    print(f"  {PREPRINT.name}（{PREPRINT.stat().st_size // 1024} KB）")
    return []


def main() -> None:
    # 只清本脚本拥有的目录。第一版在这里 rmtree 整个 OUT，把手写的投稿信、
    # 差异说明和下载好的预印本一起删了 —— 而 dist/ 是 gitignored 的。
    OUT.mkdir(parents=True, exist_ok=True)
    for name in OWNED:
        d = OUT / name
        if d.exists():
            shutil.rmtree(d)

    print("=" * 74)
    print("01 Main Manuscript —— LaTeX 源码包")
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "main"
        problems += stage_main(stage)
        d = OUT / "01_Main_Manuscript"
        zipdir(stage, d / "main_manuscript_latex.zip")
        n = len(list(stage.rglob("*")))
        print(f"  {n} 个文件 → {d.name}/main_manuscript_latex.zip")

        # ScholarOne 选了 LaTeX 之后，**还要一份编译好的 PDF**，两者都是必填：
        # 上传 zip 之后对话框会亮出 "Main Document - PDF ... Required"。
        # 表单第一页写的是 "MS Word or LaTeX"，容易读成二选一 —— 不是。
        shutil.copy2(MS / "main.pdf", d / "main_manuscript.pdf")
        print(f"  main_manuscript.pdf"
              f"（{(MS / 'main.pdf').stat().st_size // 1024} KB，"
              f"与 zip 同为必填）")

        # 解到另一个干净目录再编，证明包是自足的
        with tempfile.TemporaryDirectory() as td2:
            check = Path(td2) / "extracted"
            check.mkdir()
            with zipfile.ZipFile(d / "main_manuscript_latex.zip") as z:
                z.extractall(check)
            problems += compile_check(check)

    print("\n" + "=" * 74)
    print("02 Supplementary Material for Review")
    d = OUT / "02_Supplementary_Material_for_Review"
    d.mkdir(parents=True)
    shutil.copy2(MS / "supplementary.pdf", d / "supplementary_material.pdf")
    print(f"  supplementary_material.pdf "
          f"（{(MS / 'supplementary.pdf').stat().st_size // 1024} KB）")
    # Source Data 也进这个槽。稿件的 Data Availability 承诺了它，
    # 不带的话审稿人按那句话去找只会找到一份 PDF。
    r = subprocess.run([sys.executable,
                        str(ROOT / "scripts" / "A12_source_data_package.py")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    if r.returncode != 0 or not (d / "source_data.zip").exists():
        problems.append("A12_source_data_package.py 没有产出 source_data.zip："
                        + (r.stdout or r.stderr or "")[-200:])
    else:
        print("  " + (r.stdout or "").strip().splitlines()[0].strip())

    print("\n" + "=" * 74)
    print("03 Cover Letter")
    problems += compile_doc(SUB / "cover_letter.tex", OUT / "03_Cover_Letter")

    print("\n" + "=" * 74)
    print("04 Previously Published — Files")
    problems += fetch_preprint()
    if PREPRINT.exists():
        d = OUT / "04_Previously_Published_Files"
        d.mkdir(parents=True)
        shutil.copy2(PREPRINT, d / PREPRINT.name)

    print("\n" + "=" * 74)
    print("05 Previously Published — Statement")
    problems += compile_doc(SUB / "previously_published_statement.tex",
                            OUT / "05_Previously_Published_Statement")

    # 表单要手填的文本：摘要不能从 .tex 里直接复制 —— 里面有 \% 这种转义，
    # 粘进 ScholarOne 会原样带进去。仓库里本来就有转换脚本。
    print("\n" + "=" * 74)
    print("06 表单要手填的字段")
    ff = OUT / "06_Form_Fields"
    ff.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "A8_plain_abstract.py")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    # A8 在正文之后还打一行统计（"[1360 字符 · 200 词]"），那是给人看的，
    # 不能进 abstract.txt。取第一段即可。
    abstract = (r.stdout or "").strip().split("\n\n")[0].strip()
    # 判据是**有没有 LaTeX 转义**，不是有没有 % 号。
    # 第一版写成 `if "%" in abstract`，于是把 "84%"、"15%"、"75%" 这些正常的
    # 百分号当成了残留转义，一个完全正确的摘要被判为失败。
    if not abstract:
        problems.append("A8_plain_abstract.py 没有输出")
    elif "\\" in abstract:
        problems.append(f"纯文本摘要里仍有反斜杠转义：{abstract[:80]}…")
    else:
        (ff / "abstract.txt").write_text(abstract + "\n", encoding="utf-8")
        n = len(abstract.split())
        print(f"  abstract.txt（{n} 词，上限 200"
              + ("，**正好顶格** —— 提交前用 ScholarOne 自己的计数器再看一眼，"
                 "它对连字符复合词的切分可能与这里不同）" if n >= 198 else "）"))

    shutil.copy2(SUB / "00_README_upload.md", OUT / "00_README_upload.md")

    print("\n" + "=" * 74)
    if problems:
        print(f"🔴 {len(problems)} 个问题：")
        for p in problems:
            print(f"    {p}")
    else:
        print("✅ 源码包自足：解压到干净目录后独立编译通过，无未定义引用")
    print(f"\n→ {OUT}")
    sys.exit(1 if any(not p.startswith("（") for p in problems) else 0)


if __name__ == "__main__":
    main()
