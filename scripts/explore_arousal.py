"""探查 DEAM / PMEmo 的 arousal-valence 分布，为标签阈值提供依据。

只读不写，不产生 run record。
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from soundml.provenance import REPO_ROOT

RAW = REPO_ROOT / "data" / "raw"
DEAM = RAW / "DEAM"
PMEMO = RAW / "PMEmo" / "PMEmo2019"


def load_deam() -> pd.DataFrame:
    d = DEAM / "annotations" / "annotations averaged per song" / "song_level"
    frames = []
    for f in sorted(d.glob("*.csv")):
        df = pd.read_csv(f, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df[["song_id", "valence_mean", "valence_std", "arousal_mean", "arousal_std"]])
    return pd.concat(frames, ignore_index=True).drop_duplicates("song_id")


def load_pmemo() -> pd.DataFrame:
    a = pd.read_csv(PMEMO / "annotations" / "static_annotations.csv")
    s = pd.read_csv(PMEMO / "annotations" / "static_annotations_std.csv")
    df = a.merge(s, on="musicId", suffixes=("", "_s"))
    df.columns = [c.strip() for c in df.columns]
    return df


def report(name: str, arousal: pd.Series, valence: pd.Series, a_std: pd.Series) -> None:
    print(f"\n=== {name} (n={len(arousal)}) ===")
    q = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    print("arousal 分位:", {f"p{p}": round(float(np.percentile(arousal, p)), 3) for p in q})
    print("valence 分位:", {f"p{p}": round(float(np.percentile(valence, p)), 3) for p in q})
    print("arousal_std 分位:", {f"p{p}": round(float(np.percentile(a_std, p)), 3) for p in (50, 75, 90)})


def grid(name: str, df: pd.DataFrame, a: str, v: str, astd: str, vstd: str,
         lo_a: list[float], hi_a: list[float], v_mid: float, std_caps: list[float]) -> None:
    """扫阈值下的正/负样本量，找一组接近论文 244/244 的组合。

    两套负样本判据并列比较：
      A 笔记原文：high arousal ∧ valence ≤ 中位  （§1.1）
      B 只看 arousal：high arousal，不限 valence
    """
    for std_cap in std_caps:
        clean = df[(df[astd] <= std_cap) & (df[vstd] <= std_cap)]
        print(f"\n--- {name}  std ≤ {std_cap}  → 低分歧样本 {len(clean)}/{len(df)} ---")
        print(f"{'a_lo':>6} {'正(A)':>7} {'正(B)':>7}  | {'a_hi':>6} {'负(A)':>7} {'负(B)':>7}")
        for al, ah in zip(lo_a, hi_a):
            pos_a = ((clean[a] <= al) & (clean[v] >= v_mid)).sum()
            pos_b = (clean[a] <= al).sum()
            neg_a = ((clean[a] >= ah) & (clean[v] <= v_mid)).sum()
            neg_b = (clean[a] >= ah).sum()
            print(f"{al:>6} {pos_a:>7} {pos_b:>7}  | {ah:>6} {neg_a:>7} {neg_b:>7}")


def main() -> None:
    deam = load_deam()
    report("DEAM (1–9 标度)", deam["arousal_mean"], deam["valence_mean"], deam["arousal_std"])
    grid("DEAM", deam, "arousal_mean", "valence_mean", "arousal_std", "valence_std",
         lo_a=[3.5, 4.0, 4.5], hi_a=[6.5, 6.0, 5.5], v_mid=4.9, std_caps=[1.5, 1.8, 99])

    pm = load_pmemo()
    acol, vcol = "Arousal(mean)", "Valence(mean)"
    astd, vstd = "Arousal(std)", "Valence(std)"
    report("PMEmo (0–1 标度)", pm[acol], pm[vcol], pm[astd])
    grid("PMEmo", pm, acol, vcol, astd, vstd,
         lo_a=[0.30, 0.35, 0.40], hi_a=[0.70, 0.65, 0.60], v_mid=0.45, std_caps=[0.20, 0.25, 99])

    # 音频可读性检查
    print("\n--- 音频抽检 ---")
    for label, folder in (("DEAM", DEAM / "MEMD_audio"), ("PMEmo", PMEMO / "chorus")):
        files = sorted(folder.iterdir())
        f = files[0]
        y, sr = librosa.load(str(f), sr=22050, mono=True, duration=5.0)
        print(f"{label}: {len(files)} 个文件, 扩展名 {f.suffix}, 抽检 {f.name} → {len(y)} samples @ {sr} Hz")


if __name__ == "__main__":
    main()
