"""更新 Fig. 1 与 ED 6 的 Source Data。

    .venv/Scripts/python scripts/49_export_updated_figures.py

## 为什么这两张必须现在改

| 图 | 问题 |
|---|---|
| **Fig. 1**（hero） | 只有 3 个语料。加入 Soundtracks 后核心数字全变：<br>同域 0.658→0.592，跨域 −0.004→0.054，且多了**墙的不对称性** |
| **ED 6**（个体差异） | 建立在 PMEmo 的 EDA 分析上 —— 即 `step14` 撤回的那条未通过阳性对照的流水线 |

其余各图受新语料影响的是「能否增强」而非「是否为错」，等论文框架定了再动。

## Fig. 1 的新内容：墙是不对称的

三个音乐语料迁往环境声的能力并不相同：

| 源 | → 环境声 |
|---|---|
| PMEmo | −0.037 |
| DEAM | +0.125 |
| **Soundtracks** | **+0.243** |

而**环境声 → 音乐**方向三个目标一律 ≤ 0.094。
电影配乐含大量氛围性素材，本身更靠近声景 ——
**墙的密度取决于源语料含多少氛围性内容，且方向不对称。**

## ED 6 的替代

改用 ds002721 的被试内相关：每名受试在自己的 ~30 个试次上，
26 个脑电量 × 8 个评分维度的偏相关（已偏掉试次序号），
再看跨受试的符号一致性。208 个组合，**FDR<0.05 的有 0 个**。

与旧版的关键差别：**这条流水线通过了四项阳性对照**（`step16`），
因此「个体间无一致关系」是可解读的结论，而非仪器失效。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from soundml.provenance import REPO_ROOT

RUNS = REPO_ROOT / "runs"
OUT = REPO_ROOT / "reports" / "source_data"
DOMAIN = {"deam": "music", "pmemo": "music",
          "soundtracks": "music", "emo_mix": "ambient"}


def latest(suffix: str) -> str:
    ids = sorted(d.name for d in RUNS.glob(f"*{suffix}") if (d / "meta.json").exists())
    if not ids:
        raise SystemExit(f"找不到 run：*{suffix}")
    return ids[-1]


def meta(rid: str) -> dict:
    return json.loads((RUNS / rid / "meta.json").read_text(encoding="utf-8"))


def write(D: pd.DataFrame, name: str) -> None:
    D.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8")
    print(f"  {name}.csv  ({len(D)} 行 × {D.shape[1]} 列)")


# --------------------------------------------------------- 1. 域之墙（4 语料）
def export_domain_wall() -> None:
    m = meta(latest("_soundtracks_external"))["metrics"]
    rows = []
    for k, v in m["within_source"].items():
        algo, ds = k.split("|")
        rows.append({"algorithm": algo, "source": ds, "target": ds,
                     "transfer": "within", "domain_match": "within-corpus",
                     "spearman": v["spearman"], "r2": v["r2"]})
    for k, v in m["cross_corpus"].items():
        algo, pair = k.split("|")
        src, dst = pair.split("->")
        rows.append({"algorithm": algo, "source": src, "target": dst,
                     "transfer": "cross",
                     "domain_match": "same-domain" if v["same_domain"]
                     else "cross-domain",
                     "spearman": v["spearman"], "r2": v["r2"]})
    D = pd.DataFrame(rows)
    write(D, "fig_domain_wall")

    # ---- 不对称性：每个语料跨越边界的两个方向 ----
    C = D[D.transfer == "cross"].copy()
    C["src_domain"] = C.source.map(DOMAIN)
    C["dst_domain"] = C.target.map(DOMAIN)
    X = C[C.src_domain != C.dst_domain]
    out = []
    for corpus in ("pmemo", "deam", "soundtracks"):
        out.append({
            "corpus": corpus,
            "to_ambient": float(X[(X.source == corpus)].spearman.mean()),
            "from_ambient": float(X[(X.target == corpus)].spearman.mean()),
            "n_algorithms": int((X.source == corpus).sum())})
    A = pd.DataFrame(out).sort_values("to_ambient")
    same = float(D[D.domain_match == "same-domain"].spearman.mean())
    A["same_domain_reference"] = same
    write(A, "fig_wall_asymmetry")

    print("\n  域之墙（4 语料）：")
    s = D[D.transfer == "cross"].groupby("domain_match").spearman.agg(
        ["mean", "std", "size"])
    print(s.round(3).to_string())
    print("\n  不对称性：")
    print(A.round(3).to_string(index=False))


# ------------------------------------------------ 2. ED 6 个体差异（脑电版）
def export_individual_eeg() -> None:
    src = OUT / "ds002721_cross_subject_consistency.csv"
    if not src.exists():
        print("  ⚠️ 缺 ds002721_cross_subject_consistency.csv，先跑 45")
        return
    C = pd.read_csv(src)
    # BH-FDR（脚本 45 已算，此处重算以防列缺失）
    if "q" not in C.columns:
        o = C.p.rank(method="first")
        C["q"] = np.minimum(1.0, C.p * len(C) / o)
    C = C.sort_values("mean_rho", key=np.abs, ascending=False)
    C["family"] = C.feature.str.split("_").str[0]
    write(C, "ed_individual_eeg")

    print(f"\n  ED 6：{len(C)} 个「评分维度 × 脑电量」组合")
    print(f"    |ρ均值| 中位数 {C.mean_rho.abs().median():.3f}   "
          f"最大 {C.mean_rho.abs().max():.3f}")
    print(f"    符号一致比例 中位数 {C.frac_same_sign.median():.2f}")
    print(f"    **FDR<0.05 的组合数 {int((C.q < 0.05).sum())}**")


if __name__ == "__main__":
    print("[1/2] Fig. 1 域之墙（4 语料）+ 不对称性")
    export_domain_wall()
    print("\n[2/2] ED 6 个体差异（改用 ds002721 脑电）")
    export_individual_eeg()
