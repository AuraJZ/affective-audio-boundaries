r"""每一个产物是否都不比它的来源旧 —— 走真实的 mtime 依赖链。

    .venv/Scripts/python scripts/A3_artefact_currency.py

## 为什么需要它

「现在都是最新的了吧」这个问题不能凭印象答。2026-08-05 那次就答错了：
图在 22:02 修好，而 PDF 编于 21:30、投稿包 21:39 —— 两份 PDF 里装的
仍是带错字的旧图，而每一步单看都「跑成功了」。

    R 脚本            ->  reports/figures_r/*.pdf
    FIGURE_LEGENDS.md ->  manuscript/legends/all.tex
    *.tex + 全部图    ->  main.pdf / supplementary.pdf
    以上全部          ->  dist/*.zip

## 判据

产物的 mtime 必须 ≥ 它每一个来源的 mtime。落后即非零退出，
这样它能直接当提交前的门禁用。

⚠️ 只查**时间戳**，不查内容。产物比来源新，不代表它是用那份来源生成的
（例如中途手改过又没重跑）。它能抓的是「忘了重跑」，抓不到「跑错了」。
"""
from __future__ import annotations

import pathlib
import sys
import time

from soundml.provenance import REPO_ROOT

ROOT = REPO_ROOT
FIG = ROOT / "reports/figures_r"
SRC = ROOT / "figures_r"


def mt(p):
    return p.stat().st_mtime if p.exists() else 0


def ts(p):
    return time.strftime("%m-%d %H:%M", time.localtime(mt(p))) if p.exists() else "MISSING"


# R script -> the figure(s) it writes, discovered from the save_ms() call.
#
# 🔴 `fig_boundaries.R` 在这里被跳过，而不是被删掉。
#
# 它画的是已撤回的四边界汇总图（预印本 Fig. 8）。脚本保留下来，是为了让
# 「撤回了什么」可以对着它核；但它的 PDF 已经删了，也不再由 build_all.R 生成。
# 若不跳过，本脚本会永远报一条「Fig8_boundaries.pdf 比 fig_boundaries.R 旧」——
# 一条无论如何都不会变绿的红线。**永远为红的检查项，等于训练人忽略这个检查。**
SKIP = {"_theme.R", "build_all.R", "fig_boundaries.R"}

pairs = []
for rf in sorted(SRC.glob("*.R")):
    if rf.name in SKIP:
        continue
    txt = rf.read_text(encoding="utf-8")
    import re
    # The figure stem is the LAST quoted element of the file.path(...) call --
    # matching the first one picks up ".." and "reports" and invents artefacts
    # that were never supposed to exist.
    for m in re.finditer(r'save_ms\(.*?file\.path\(([^)]*)\)', txt, re.S):
        parts = re.findall(r'"([^"]+)"', m.group(1))
        if parts:
            pairs.append((rf, FIG / f"{parts[-1]}.pdf"))

print("=" * 78)
print("R script -> figure")
stale = []
for rf, pdf in pairs:
    ok = mt(pdf) >= mt(rf)
    if not ok:
        stale.append(f"{pdf.name} older than {rf.name}")
    print(f"  {'OK ' if ok else '🔴 '} {rf.name:24s} {ts(rf)}  ->  "
          f"{pdf.name:22s} {ts(pdf)}")

print("\n" + "=" * 78)
print("legends and compiled documents")
checks = [
    (ROOT / "reports/FIGURE_LEGENDS.md", ROOT / "manuscript/legends/all.tex"),
    (ROOT / "manuscript/main.tex", ROOT / "manuscript/main.pdf"),
    (ROOT / "manuscript/legends/all.tex", ROOT / "manuscript/main.pdf"),
    (ROOT / "manuscript/supplementary.tex", ROOT / "manuscript/supplementary.pdf"),
    (ROOT / "manuscript/legends/all.tex", ROOT / "manuscript/supplementary.pdf"),
]
# every figure feeds both PDFs
newest_fig = max((mt(p) for _, p in pairs), default=0)
newest_fig_name = max(((mt(p), p.name) for _, p in pairs), default=(0, "-"))[1]
for a, b in checks:
    ok = mt(b) >= mt(a)
    if not ok:
        stale.append(f"{b.name} older than {a.name}")
    print(f"  {'OK ' if ok else '🔴 '} {a.name:26s} {ts(a)}  ->  "
          f"{b.name:22s} {ts(b)}")
for pdf in ("main.pdf", "supplementary.pdf"):
    p = ROOT / "manuscript" / pdf
    ok = mt(p) >= newest_fig
    if not ok:
        stale.append(f"{pdf} older than the newest figure ({newest_fig_name})")
    print(f"  {'OK ' if ok else '🔴 '} newest figure ({newest_fig_name[:18]:18s}) "
          f"{time.strftime('%m-%d %H:%M', time.localtime(newest_fig))}  ->  "
          f"{pdf:22s} {ts(p)}")

zips = sorted((ROOT / "dist").glob("*.zip"), key=mt) if (ROOT / "dist").exists() else []
if zips:
    z = zips[-1]
    newest_input = max([mt(ROOT / "manuscript/main.pdf"),
                        mt(ROOT / "manuscript/supplementary.pdf"), newest_fig])
    ok = mt(z) >= newest_input
    if not ok:
        stale.append(f"{z.name} older than its inputs")
    print(f"  {'OK ' if ok else '🔴 '} newest input               "
          f"{time.strftime('%m-%d %H:%M', time.localtime(newest_input))}  ->  "
          f"{z.name:22s} {ts(z)}")

print("\n" + "=" * 78)
if stale:
    print(f"🔴 {len(stale)} 个产物落后于它的来源：")
    for s in stale:
        print(f"    {s}")
else:
    print("✅ 每个产物都不比它的来源旧")
print()
print("⚠️ 只查时间戳，不查内容 —— 抓得到「忘了重跑」，抓不到「跑错了」。")
sys.exit(1 if stale else 0)
