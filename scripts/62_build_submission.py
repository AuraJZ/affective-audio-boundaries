r"""打包投稿目录 —— 可重复执行，不是一次性复制。

    .venv/Scripts/python scripts/62_build_submission.py

## 设计要点

**保持仓库内的相对路径不变。** `main.tex` 里写的是
`../reports/figures_r/Fig1.pdf` 与 `../references/references.bib`；
若在投稿目录里重排结构，这些路径全会断，而 LaTeX 的断链只有编译时才报错。
因此投稿目录**镜像仓库布局**，`manuscript/` 与 `reports/` 的相对关系照旧。

## 收什么、不收什么

| 收 | 理由 |
|---|---|
| `manuscript/` `references/` | 稿件本体 |
| `reports/figures_r/` | 图，四种格式（PDF 供编译，TIFF 供期刊） |
| `reports/source_data/` | 每个面板的 Source Data |
| `src/` `scripts/` `figures_r/` | 全部分析代码，含已撤回分析的代码 |
| `runs/` | 运行记录 —— 每个数字到代码版本、种子、输入哈希的证据链 |
| `pyproject.toml` `uv.lock` | 环境复现 |

| 不收 | 理由 |
|---|---|
| `data/raw/` | 原始音频与生理数据，体积大且**多数不可再分发** |
| `data/features/` | 派生特征表，可由代码重建；含受限语料的派生物 |
| `.venv/` `__pycache__/` | 环境产物 |

**已撤回分析的代码照收。** SI 第 2 节记录了撤回过程，
复现该记录需要产生它的代码。删掉等于让撤回无法核验。

## 输出：一个 zip，不是一棵摊开的树 🔴

首版把包**摊开**成 `submission/` 留在仓库里 —— 那是 `manuscript/`、
`references/`、`scripts/`、`src/`、`figures_r/`、`runs/` 的逐字副本，
372 个文件。于是仓库里同时存在两套源码，改了一边忘了另一边就会分叉，
而且没有任何东西会提醒你哪一边是真的。

现在：在系统临时目录里组装，压成**单个 zip** 落到 `dist/`，随即删掉临时树。
仓库里只留一个压缩包，期刊要的本来也是压缩包。

- `dist/submission_YYYYMMDD.zip`
- 包内 `README.md` —— 结构说明与复现步骤
- 包内 `MANIFEST.txt` —— 全部文件的 SHA-256
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import time
from pathlib import Path

from soundml.provenance import REPO_ROOT

# 临时组装目录：**不在仓库里**，避免留下第二套源码。
_STAGE = Path(tempfile.mkdtemp(prefix="sm_submission_"))
OUT = _STAGE / "submission"
DIST = REPO_ROOT / "dist"
FULL = _STAGE / "archive_full"

# (源相对路径, 是否目录, 说明)
INCLUDE = [
    ("manuscript", True, "稿件 LaTeX 源"),
    ("references", True, "BibTeX 与核验记录"),
    ("reports/figures_r", True, "图（PDF / TIFF / SVG / PNG）"),
    ("reports/source_data", True, "每个面板的 Source Data"),
    ("reports/FIGURE_LEGENDS.md", False, "图注"),
    ("src", True, "分析库"),
    ("scripts", True, "分析脚本"),
    ("figures_r", True, "R 绘图脚本"),
    ("runs", True, "运行记录（证据链）"),
    ("pyproject.toml", False, "环境定义"),
    ("uv.lock", False, "环境锁定"),
]

SKIP_DIRS = {"__pycache__", ".venv", ".git", ".pytest_cache", ".ipynb_checkpoints"}
# LaTeX 构建产物不入包；`.bbl` 例外 —— 部分投稿系统要求随源提供。
SKIP_SUFFIX = {".pyc", ".pyo", ".aux", ".log", ".blg", ".out", ".toc",
               ".synctex.gz", ".fls", ".fdb_latexmk"}
# `manuscript/*.md` 是转 LaTeX 之前的起草稿，转换后未再同步，
# 与 `.tex` 已经不一致。包里放一份过时的重复稿件是隐患，故排除。
SKIP_EXACT = {"manuscript/main.md", "manuscript/supplementary.md"}
# 防呆：投稿包里出现这些说明原始数据漏了进来
FORBIDDEN_SUFFIX = {".mp3", ".wav", ".edf", ".flac", ".ogg", ".m4a"}
MAX_MB = 60


def clear_dir(path: Path, label: str) -> None:
    """删除并重建目录；文件被占用时给出可操作的报错而不是抛栈。

    Windows 上 PDF 阅读器会锁住打开的文件，`shutil.rmtree` 随即失败。
    这在可重复执行的打包脚本里很常见，因此重试几次，仍失败则列出
    锁定的文件，让人知道该关掉什么。
    """
    if not path.exists():
        path.mkdir()
        return
    for attempt in range(3):
        try:
            shutil.rmtree(path)
            break
        except PermissionError:
            if attempt < 2:
                time.sleep(1.5)
                continue
            # 用重命名探测，不用「追加模式打开」：Windows 上 PDF 阅读器
            # 允许追加打开却持有删除锁，前者探不出来。删除与重命名
            # 都需要独占访问，因此重命名成功即说明可删。
            locked = []
            for p in path.rglob("*"):
                if not p.is_file():
                    continue
                probe = p.with_name(p.name + ".locktest")
                try:
                    p.rename(probe)
                    probe.rename(p)
                except OSError:
                    locked.append(p.relative_to(path).as_posix())
            print(f"\n🔴 无法清空 {label}/ —— 以下文件被其他进程占用：")
            for f in locked[:10]:
                print(f"    {f}")
            print("\n   多半是 PDF 阅读器或资源管理器预览窗格。"
                  "关掉后重跑本脚本即可。")
            raise SystemExit(1)
    path.mkdir()


def copy_tree(src: Path, dst: Path, prefix: str = "") -> int:
    n = 0
    for p in src.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.is_dir() or p.suffix in SKIP_SUFFIX:
            continue
        rel = p.relative_to(src)
        if f"{prefix}/{rel.as_posix()}".lstrip("/") in SKIP_EXACT:
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        n += 1
    return n


FULL_README = """# 完整归档 —— ⚠️ 内部使用，不可对外分发

本目录含 **原始音频与生理数据**。多数语料的许可**不允许再分发**：

| 语料 | 限制 |
|---|---|
| PMEmo | 音频为商业曲目摘录，无再分发许可 |
| DEAM | 逐曲许可不同，未逐曲核查 |
| ESC-50 | CC BY-NC 3.0 |
| Soundtracks | 仅限学术研究用途 |
| Emo-Soundscapes | CC，逐条变体不同 |
| ds002721 / BIRAFFE2 | CC0 / CC BY 4.0（可分发） |

**因此本目录仅用于内部归档与交接，不得上传、转发或随论文提交。**

可对外分发的子集在 `submission/` —— 那份只含代码、稿件、图与 Source Data，
打包时有防呆检查，任何原始媒体文件混入都会使打包失败。

`data/raw/ARAUS/` 的许可**未经核查**，且未用于稿件中的任何结论。

## 结构

```
submission/     可分发子集（代码、稿件、图、Source Data、运行记录）
manuscript_pdf/ 编译好的 main.pdf 与 supplementary.pdf
data/           原始数据与派生特征 —— 不可分发
```
"""


def build_full() -> None:
    """完整归档：submission/ + 原始数据 + 编译好的 PDF。"""
    clear_dir(FULL, "archive_full")

    print("\n完整归档打包中（含原始数据，体积大）…")
    n = copy_tree(OUT, FULL / "submission")
    print(f"  submission/                {n:5d} 个文件")

    pdfdir = FULL / "manuscript_pdf"
    pdfdir.mkdir()
    n_pdf = 0
    for pdf in (REPO_ROOT / "manuscript").glob("*.pdf"):
        shutil.copy2(pdf, pdfdir / pdf.name)
        n_pdf += 1
    print(f"  manuscript_pdf/            {n_pdf:5d} 个文件")

    n_data = copy_tree(REPO_ROOT / "data", FULL / "data")
    print(f"  data/                      {n_data:5d} 个文件")

    (FULL / "README.md").write_text(FULL_README, encoding="utf-8")
    files = [p for p in FULL.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files) / 1e9
    print(f"\n→ {FULL}")
    print(f"   {len(files):,} 个文件，{size:.2f} GB")
    print("   ⚠️ 含不可再分发的原始音频 —— 仅内部使用，README.md 已写明")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="额外产出 archive_full/：submission + 原始数据 + PDF")
    args = ap.parse_args()

    clear_dir(OUT, "submission")

    print("打包中…")
    counts = {}
    for rel, is_dir, note in INCLUDE:
        src = REPO_ROOT / rel
        if not src.exists():
            print(f"  ⚠️ 缺失，跳过：{rel}")
            continue
        dst = OUT / rel
        if is_dir:
            n = copy_tree(src, dst, prefix=rel)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n = 1
        counts[rel] = n
        print(f"  {rel:28s} {n:4d} 个文件   {note}")

    # ---------- 防呆检查 ----------
    files = [p for p in OUT.rglob("*") if p.is_file()]
    bad = [p for p in files if p.suffix.lower() in FORBIDDEN_SUFFIX]
    total_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"\n合计 {len(files)} 个文件，{total_mb:.1f} MB")
    if bad:
        print(f"🔴 混入了原始媒体文件 {len(bad)} 个 —— 原始数据不可再分发：")
        for p in bad[:5]:
            print(f"    {p.relative_to(OUT)}")
        raise SystemExit(1)
    if total_mb > MAX_MB:
        print(f"🔴 体积 {total_mb:.1f} MB 超过 {MAX_MB} MB，检查是否有大文件混入")
        for p in sorted(files, key=lambda x: -x.stat().st_size)[:5]:
            print(f"    {p.stat().st_size/1e6:6.1f} MB  {p.relative_to(OUT)}")
        raise SystemExit(1)
    print("✅ 未混入原始媒体文件，体积正常")

    # ---------- MANIFEST ----------
    lines = []
    for p in sorted(files):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{h}  {p.relative_to(OUT).as_posix()}")
    (OUT / "MANIFEST.txt").write_text(
        "# SHA-256 of every file in this package\n"
        f"# {len(files)} files, {total_mb:.1f} MB\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8")

    # ---------- README ----------
    (OUT / "README.md").write_text(README, encoding="utf-8")

    # ---------- 压成单个 zip，随即清掉临时树 ----------
    # 🔴 仓库里只留压缩包。首版把包摊开成 `submission/` 留在仓库里，
    # 那是 manuscript / references / scripts / src / figures_r / runs 的
    # 逐字副本 —— 372 个文件、两套源码。改了一边忘了另一边就会分叉，
    # 而且没有任何东西会提醒你哪一边是真的。
    DIST.mkdir(exist_ok=True)
    archive = shutil.make_archive(
        str(DIST / f"submission_{time.strftime('%Y%m%d')}"), "zip",
        root_dir=OUT)
    zip_mb = Path(archive).stat().st_size / 1e6
    shutil.rmtree(_STAGE, ignore_errors=True)

    print(f"\n→ {archive}")
    print(f"   {len(files)} 个文件压成 {zip_mb:.1f} MB")
    print("   README.md 与 MANIFEST.txt 在包内")
    print("   临时组装目录已删 —— 仓库里不留第二套源码")

    if args.full:
        build_full()


README = """# Where acoustic models of affective response generalise, and where they do not

Manuscript, figures, per-panel source data, and the complete analysis code.

## Compiling

```
cd manuscript
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supplementary && bibtex supplementary && pdflatex supplementary
```

Compiled PDFs are already included. Figures are vector PDFs, so nothing is
rasterised at any zoom level. The directory layout mirrors the analysis
repository, which is why the relative paths inside `main.tex` resolve unchanged.

## Contents

| Path | |
|---|---|
| `manuscript/` | LaTeX source, shared preamble, generated figure legends, compiled PDFs |
| `manuscript/legends/all.tex` | Legends as macros, generated from `reports/FIGURE_LEGENDS.md` |
| `references/` | `references.bib` — 27 entries, each resolved by DOI with reverse title checking |
| `reports/figures_r/` | Figures: PDF (used by LaTeX), TIFF, SVG, PNG |
| `reports/source_data/` | One CSV per figure panel |
| `src/soundml/` | Analysis library |
| `scripts/` | Numbered analysis scripts, in execution order |
| `figures_r/` | R scripts that render every figure |
| `runs/` | One directory per analysis: git commit, seed, input hashes, parameters, metrics |
| `MANIFEST.txt` | SHA-256 of every file here |

## Data

No data are included. The corpora are public but most are not redistributable,
and the derived feature tables are rebuilt by the code. Accessions, access dates
and licence terms:

| Corpus | Accession | Licence | Note |
|---|---|---|---|
| ds002721 (EEG) | OpenNeuro `ds002721` | see note | `dataset_description.json` records CC0, the README records CC BY 4.0; we followed the more restrictive terms |
| Soundtracks Set 1 | OSF `p6vkg` | academic research use | |
| DEAM | University of Geneva | per-track, varies | |
| PMEmo | published with the dataset paper | annotations redistributable; audio is commercial excerpts and is not | |
| Emo-Soundscapes | published with the dataset paper | Creative Commons, per-item variants in the metadata | |
| ESC-50 | published with the dataset paper | CC BY-NC 3.0 | |
| BIRAFFE2 | Zenodo | CC BY 4.0 | used only for the methodological case in Supplementary Information S2 |

Each is cited in the manuscript as its distributor requests. With the corpora
placed under `data/raw/`, `uv sync` restores the environment and the scripts run
in numeric order.

## Reproducing a number

Every value in the manuscript traces to a run record. `runs/<run_id>/meta.json`
holds the git commit, the random seed, SHA-256 hashes of all inputs, every
parameter and every metric, so a reported figure can be tied to the exact code and
data that produced it.

Each script's docstring states what it tests, what criterion was fixed before the
analysis ran, and what follows from either outcome.

## Code for withdrawn analyses

Scripts and run records for analyses that were **withdrawn** are included rather
than removed. Supplementary Information S2 documents a set of electrodermal
findings retracted after they failed positive controls, and S3 lists analysis
decisions that were replaced. Verifying that record requires the code that
produced it. Where a run has been superseded, the report citing it says so.
"""


if __name__ == "__main__":
    main()
