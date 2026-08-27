"""ARAUS 结构探查 —— 决定外部复现走哪条路。

    uv run python scripts/25_araus_inspect.py

## 背景

公开发布的 `soundscapes.zip` 只含 **6 段**底噪，而非论文所述的 234 段
（完整版在受限的 `soundscapes_raw.zip`，仓库指向 Urban Soundscapes of the World
单独获取）。因此**无法按配方重建全部 25,440 段刺激**。

这不必然阻塞外部复现 —— 取决于 `datav2.zip` 里有什么：

| 若含 | 则可以 |
|---|---|
| 预算声学特征（每段刺激一行） | **直接做外部复现**，完全不需要音频 |
| 仅问卷响应 + 配方 | 只能用 6 段底噪 × 掩蔽声重建一个子集 |
| 仅聚合统计 | 无法做逐刺激分析，只能做趋势对照 |

本脚本只做探查与报告，不做任何分析或改动。
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "ARAUS"
FILES = {"datav2.zip": "f_114619.zip",
         "soundscapes.zip": "f_89459.zip",
         "maskersv2.zip": "f_114620.zip"}

# 我们关心的列名线索
FEATURE_HINTS = ("mfcc", "centroid", "rolloff", "flatness", "zcr", "loudness",
                 "sharpness", "roughness", "spectral", "onset", "lufs", "leq")
RATING_HINTS = ("pleasant", "eventful", "vibrant", "calm", "monoton", "chaotic",
                 "annoy", "appropriate", "arous", "valence")


def describe_csv(z: zipfile.ZipFile, name: str) -> None:
    with z.open(name) as fh:
        try:
            head = pd.read_csv(fh, nrows=5)
        except Exception as e:
            print(f"    (读取失败: {type(e).__name__})")
            return
    cols = list(head.columns)
    feats = [c for c in cols if any(h in c.lower() for h in FEATURE_HINTS)]
    rates = [c for c in cols if any(h in c.lower() for h in RATING_HINTS)]
    print(f"    列数 {len(cols)}")
    print(f"    前 12 列: {cols[:12]}")
    if rates:
        print(f"    ✅ 评分列 ({len(rates)}): {rates[:8]}")
    if feats:
        print(f"    ✅ 声学特征列 ({len(feats)}): {feats[:8]}")
    if not rates and not feats:
        print("    （未识别出评分或特征列）")


def main() -> None:
    for label, fname in FILES.items():
        p = RAW / fname
        print("=" * 72)
        if not p.exists():
            print(f"{label}: 缺失")
            continue
        size = p.stat().st_size / 1e6
        try:
            z = zipfile.ZipFile(p)
        except zipfile.BadZipFile:
            print(f"{label}: {size:.1f} MB —— 下载不完整")
            continue

        names = z.namelist()
        print(f"{label}: {size:.1f} MB, {len(names)} 项")
        csvs = [n for n in names if n.lower().endswith(".csv")]
        wavs = [n for n in names if n.lower().endswith(".wav")]
        others = [n for n in names
                  if not n.lower().endswith((".csv", ".wav")) and not n.endswith("/")]

        if wavs:
            print(f"  WAV: {len(wavs)} 个，示例 {Path(wavs[0]).name}")
        if others:
            print(f"  其他: {len(others)} 个，示例 {others[:4]}")
        for c in csvs[:12]:
            info = z.getinfo(c)
            print(f"  CSV {c}  ({info.file_size / 1e6:.1f} MB)")
            describe_csv(z, c)

    print("=" * 72)
    print("判断依据：若 datav2 的 CSV 同时含评分列与声学特征列，")
    print("则可直接做外部复现，无需音频。")


if __name__ == "__main__":
    main()
