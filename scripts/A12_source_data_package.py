r"""把 Source Data 打成审稿人能用的一个 zip —— 带「哪张表对应哪个面板」的索引。

    .venv/Scripts/python scripts/A12_source_data_package.py

## 为什么要这个包

稿件的 Data Availability 写着：

  > Derived tables underlying every figure panel are provided as
  > **supplementary source data**.

但审稿人在补充材料里只看得到那份 46 页 PDF —— 98 个 CSV 在代码仓库里，而
投稿包不含它们。**一篇整篇在讲「每个数字都要能核」的稿子，把核验材料留在
投稿系统之外，是自相矛盾的。**

## 为什么不是直接压缩目录

丢 98 个文件名给审稿人，等于没给。真正有用的是「Fig. 2b 的那个 -11.1%
是从哪张表的哪一列算出来的」。所以索引由**绘图脚本自己**生成：扫
`figures_r/*.R` 里的 `read_src("...")`，得到图 → 表的真实依赖，而不是手写一份
会过期的对照。

扫不到的表单列为「未被任何绘图脚本读取」，也照收 —— 它们多半是正文或补充
材料直接引用的数字（如 `order_confound.csv`、`homologous_ceiling.csv`），
漏掉反而更糟。宁可多给，不可少给。
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "reports" / "source_data"
FIGR = ROOT / "figures_r"
OUT = ROOT / "dist" / "scholarone" / "02_Supplementary_Material_for_Review"

# 绘图脚本 → 稿件里的图号。脚本名是历史遗留，和最终图号对不上，
# 所以这张表必须手写，且改了图号就要改这里。
SCRIPT_TO_FIGURE = {
    "fig1_domain_wall.R":  "Fig. 1  (cross-corpus transfer)",
    "fig_price.R":         "Fig. 2  (what adaptation recovers)",
    "fig3_tonality.R":     "Fig. 3  (tonal descriptors)",
    "fig4_intervention.R": "Fig. 4  (audio editing)",
    "fig5_calibration.R":  "Fig. 5  (response channel)",
    "fig6_design.R":       "Fig. 6  (design requirement)",
    "fig2_convergence.R":  "Fig. S9 (feature-selection routes)",
    "extended_data.R":     "Figs. S1-S7",
    "ed8_invariance.R":    "Fig. S8 (sign preservation)",
}


def main() -> None:
    if not SD.is_dir():
        sys.exit(f"找不到 {SD}")

    used: dict[str, list[str]] = {}
    for script, figure in SCRIPT_TO_FIGURE.items():
        p = FIGR / script
        if not p.exists():
            print(f"⚠️ 绘图脚本不存在：{script}")
            continue
        for name in re.findall(r'read_src\("([^"]+)"\)',
                               p.read_text(encoding="utf-8")):
            used.setdefault(name + ".csv", []).append(figure)

    all_csv = sorted(p.name for p in SD.glob("*.csv"))
    missing = [n for n in used if n not in all_csv]
    orphan = [n for n in all_csv if n not in used]

    lines = [
        "Source Data",
        "===========",
        "",
        "Derived tables underlying the figures of",
        "",
        "    Transfer Boundaries in Affective Audio Modelling:",
        "    Zero-Shot Loss, Measured Adaptation Cost, and the Limits of the",
        "    Available Evidence",
        "",
        "One row per table, giving the figure it is read by. The mapping is",
        "generated from the plotting scripts themselves (every read_src() call in",
        "figures_r/*.R), not maintained by hand, so it cannot drift from what the",
        "figures actually load.",
        "",
        "Every figure script ends in assertions on the numbers the manuscript",
        "quotes from it, so a table that disagrees with the text breaks the build",
        "rather than shipping quietly. The scripts are in the code repository named",
        "in the Code Availability statement.",
        "",
        "-" * 72,
        "TABLES READ BY A FIGURE",
        "-" * 72,
        "",
    ]
    for name in sorted(used):
        figs = ", ".join(sorted(set(used[name])))
        lines.append(f"{name:<44} {figs}")

    lines += [
        "",
        "-" * 72,
        "TABLES NOT READ BY ANY PLOTTING SCRIPT",
        "-" * 72,
        "",
        "These carry numbers quoted directly in the main text or the",
        "Supplementary Information rather than plotted. They are included because",
        "the claim is that every reported value is checkable, not merely every",
        "plotted one. Notable members:",
        "",
        "  order_confound.csv        the presentation-position sham on the PMEmo",
        "                            electrodermal measures (Supplementary S3)",
        "  tac_recheck_*.csv         the recomputations behind the revision,",
        "                            including both corpus variants",
        "  homologous_ceiling.csv    per-measure reliabilities and ceilings",
        "  pmemo_personalisation_gain.csv,  subjective_individual_dual.csv",
        "                            the out-of-sample personalisation test",
        "",
    ]
    for name in orphan:
        lines.append(f"{name}")

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "source_data.zip"
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", "\n".join(lines) + "\n")
        for p in sorted(SD.glob("*.csv")):
            z.write(p, f"source_data/{p.name}")

    print(f"  {len(all_csv)} 张表 + 索引 → {dest.name}"
          f"（{dest.stat().st_size // 1024} KB）")
    print(f"  其中 {len(used)} 张被绘图脚本读取，{len(orphan)} 张只在正文中引用")
    if missing:
        print(f"\n🔴 绘图脚本读了但目录里没有的表：{missing}")
        sys.exit(1)
    print("\n✅ 每个 read_src() 都有对应的 CSV")


if __name__ == "__main__":
    main()
