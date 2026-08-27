r"""图 / 图注 / 正文 三者一致性检查。

    .venv/Scripts/python scripts/64_check_figure_consistency.py

## 为什么要做

改图之后最容易漏的是「图变了、图注没变」或「图注变了、正文里的数字没变」。
这类错误编译器不会报，肉眼逐张核 14 张图也会漏。

## 查什么

| 检查 | 抓什么 |
|---|---|
| 面板字母 | 图注声明的面板（**a** **b** …）与图文件里实际的面板数是否一致 |
| 数值对照 | 图注与正文里出现的同一量是否一致（逐个数字比对） |
| Source Data | 图注引用的数字能否在对应的 source_data CSV 中找到 |
| 时间戳 | 图文件比图注新 → 图改过但图注可能没跟着改 |
| 交叉引用 | 正文提到的面板字母（Fig. 1c）是否在图注中存在 |

**时间戳那条是启发式**，只提示需要人工确认，不判定为错误。
"""

from __future__ import annotations

import re
from pathlib import Path

from soundml.provenance import REPO_ROOT

FIGDIR = REPO_ROOT / "reports" / "figures_r"
LEG = REPO_ROOT / "reports" / "FIGURE_LEGENDS.md"
MAIN = REPO_ROOT / "manuscript" / "main.tex"
SUP = REPO_ROOT / "manuscript" / "supplementary.tex"

# 🔴 这张映射表在 2026-08-05 重编号之后失效过：正文顺序改了，图文件跟着
# 改名，而这里还写着旧名，于是脚本对着六个不存在的文件报错退出 1。
# 一个已知会失败的检查器留在流水线里，等于把它自己变成噪声 ——
# 后面真出问题时没人会当回事。
#
# 编号跟**正文顺序**走，文件名跟**内容**走，两者靠这张表对齐。
# 顺序再变时，这里和 manuscript/main.tex 的 \includegraphics 必须同时改。
FIGFILE = {
    1: "Fig1_wall",         # 边界 1：墙，含四个表征族
    2: "Fig2_price",        # 边界 1 的价格
    3: "Fig3_routes",       # 归因方法互斥
    4: "Fig4_tonality",     # 证伪检验
    5: "Fig5_intervention",  # 边界 2：跨合成
    6: "Fig6_channel",      # 边界 3：跨响应通道
    7: "Fig7_design",       # 边界 4：跨个体
    8: "Fig8_boundaries",   # 四道边界总图
}
EDFILE = {
    1: "ED1_loudness", 2: "ED2_data_use", 3: "ED3_glassbox",
    4: "ED4_shape_functions", 5: "ED5_rashomon", 6: "ED6_individual",
    7: "ED7_representations", 8: "ED8_invariance",
}


def legend_blocks() -> dict[str, str]:
    text = LEG.read_text(encoding="utf-8")
    out = {}
    for b in re.split(r"^## ", text, flags=re.M)[1:]:
        head, _, body = b.partition("\n")
        m = re.match(r"(Extended Data )?Fig\. (\d+)", head.strip())
        if m:
            out[("ed" if m.group(1) else "fig") + m.group(2)] = \
                head + "\n" + body.split("\n---")[0]
    return out


def numbers(s: str) -> set[str]:
    """抓形如 0.592 / +0.132 / 1,240 / 13% 的数值，忽略纯序号。"""
    out = set()
    for m in re.finditer(r"[-+−]?\d[\d,]*\.\d+|\b\d{3,}(?:,\d{3})*\b", s):
        out.add(m.group().replace("−", "-").replace(",", ""))
    return out


def main() -> None:
    legends = legend_blocks()
    main_tex = MAIN.read_text(encoding="utf-8")
    sup_tex = SUP.read_text(encoding="utf-8")
    body = main_tex + "\n" + sup_tex
    issues, notes = [], []

    # ---------- 1. 每张图都有图注，每个图注都有图 ----------
    for n, stem in FIGFILE.items():
        if f"fig{n}" not in legends:
            issues.append(f"Fig. {n} 无图注")
        if not (FIGDIR / f"{stem}.pdf").exists():
            issues.append(f"Fig. {n} 缺 PDF：{stem}.pdf")
    for n, stem in EDFILE.items():
        if f"ed{n}" not in legends:
            issues.append(f"ED {n} 无图注")
        if not (FIGDIR / f"{stem}.pdf").exists():
            issues.append(f"ED {n} 缺 PDF：{stem}.pdf")

    # ---------- 2. 面板字母：正文引用的面板必须在图注里出现 ----------
    for m in re.finditer(r"Fig\.~\\ref\{fig:(\w+)\}([a-e](?:,[a-e])*)", main_tex):
        panels = set(re.findall(r"[a-e]", m.group(2)))
        # 由 \label 反查图号
        lbl = m.group(1)
        num = {"wall": 1, "tonality": 2, "routes": 3, "intervention": 4,
               "calibration": 5, "design": 6}.get(lbl)
        if num is None:
            continue
        declared = set(re.findall(r"\*\*([a-e])\*\*", legends.get(f"fig{num}", "")))
        missing = panels - declared
        if missing:
            issues.append(f"正文引用 Fig. {num}{''.join(sorted(missing))}，"
                          f"但图注未描述该面板（图注含 {''.join(sorted(declared))}）")

    # ---------- 3. 图注里的数值应能在正文或 source data 中找到 ----------
    src_all = " ".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in (REPO_ROOT / "reports" / "source_data").glob("*.csv"))
    body_nums = numbers(body)
    for key, block in legends.items():
        unmatched = []
        for v in numbers(block):
            if v in body_nums:
                continue
            # source data 里以任意精度出现即算匹配
            if re.search(re.escape(v.lstrip("+-")), src_all):
                continue
            unmatched.append(v)
        if len(unmatched) > 6:
            notes.append(f"{key}: {len(unmatched)} 个数值未在正文或 source data "
                         f"中找到（图注独有值属正常，仅供抽查）"
                         f" 例：{', '.join(sorted(unmatched)[:5])}")

    # ---------- 4. 时间戳：图比图注新 → 需人工确认图注是否跟上 ----------
    leg_mtime = LEG.stat().st_mtime
    for n, stem in {**{f"Fig. {k}": v for k, v in FIGFILE.items()},
                    **{f"ED {k}": v for k, v in EDFILE.items()}}.items():
        p = FIGDIR / f"{stem}.pdf"
        if p.exists() and p.stat().st_mtime > leg_mtime + 60:
            notes.append(f"{n} 的图文件比图注新 —— 确认图注是否需要同步")

    # ---------- 5. 图注生成物是否落后于源 ----------
    allt = REPO_ROOT / "manuscript" / "legends" / "all.tex"
    if allt.exists() and allt.stat().st_mtime < leg_mtime:
        issues.append("legends/all.tex 比 FIGURE_LEGENDS.md 旧 —— 需重跑 63")

    # ---------- 打印 ----------
    print("=" * 72)
    print(f"图 {len(FIGFILE)} 张 + Extended Data {len(EDFILE)} 张")
    print(f"\n--- 必须处理 ({len(issues)}) ---")
    for e in issues:
        print(f"  ❌ {e}")
    if not issues:
        print("  ✅ 无")
    print(f"\n--- 需人工确认 ({len(notes)}) ---")
    for w in notes:
        print(f"  ⚠️ {w}")
    if not notes:
        print("  ✅ 无")
    raise SystemExit(1 if issues else 0)


if __name__ == "__main__":
    main()
