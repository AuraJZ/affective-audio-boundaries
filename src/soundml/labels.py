"""标签构造。

三个数据集，两种标签性质：

  ESC-50  —— 语义类别代理，仅用于 Step 1 流水线空跑，无科学意义
  DEAM    —— arousal 标注，Step 2 主力训练集
  PMEmo   —— arousal 标注，Step 2 **独立测试集**（跨源 + 跨文化泛化）

## 对笔记 §1.1 的修正：正负样本沿 arousal 单轴分居两端

笔记原文用「low arousal + 中高 valence」对「high arousal + negative valence」。
实测这在真数据上不成立——音乐里 arousal 与 valence 强正相关，
「高唤醒 + 负情绪」样本天然稀少（DEAM 仅 25 个，PMEmo 仅 6 个）。

改为：论文的正负是 GABA agonist vs inhibitor，即**同一条药理轴的两端**。
对应到这里，轴是 arousal，valence 不作第二筛选轴，只用于排除一类干扰：

  正 = 低 arousal ∧ valence 不低   （valence 下限排除"低唤醒+低愉悦"的悲伤/压抑曲目，
                                    那类不助眠，反而引起反刍）
  负 = 高 arousal，不限 valence     （对助眠而言高唤醒本身即对立面）

中间的模糊地带整体丢弃，与论文结构一致。

## 标注分歧过滤（§8.1「宁缺毋滥」的落点）

按 arousal_std / valence_std 剔除评分者分歧大的样本。
DEAM 阈值 1.8（1–9 标度）保留 1233/1802；1.5 过严，只剩 602。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- 阈值常量
# DEAM：1–9 标度。中位数 arousal 4.9 / valence 4.9，p25 分别为 3.8 / 4.1
DEAM_AROUSAL_LOW = 4.0
DEAM_AROUSAL_HIGH = 6.0
DEAM_VALENCE_FLOOR = 4.0  # 正样本 valence 下限
DEAM_STD_CAP = 1.8

# PMEmo：0–1 标度。中位数 arousal 0.65 / valence 0.625（流行歌副歌，整体偏高）
PMEMO_AROUSAL_LOW = 0.40
PMEMO_AROUSAL_HIGH = 0.60
PMEMO_VALENCE_FLOOR = 0.40
PMEMO_STD_CAP = 0.25

# Emo-Soundscapes：−1 到 1 标度，中位数约 0。p25 = −0.53、p75 = 0.61
EMO_AROUSAL_LOW = -0.40
EMO_AROUSAL_HIGH = 0.40
EMO_VALENCE_FLOOR = -0.20

# 正样本：低唤醒的自然/水声 —— 8 类 × 40 = 320
POSITIVE_CATEGORIES = [
    "rain",
    "sea_waves",
    "wind",
    "crackling_fire",
    "crickets",
    "water_drops",
    "pouring_water",
    "breathing",
]

# 负样本：高唤醒的警报/机械/突发声 —— 10 类 × 40 = 400
NEGATIVE_CATEGORIES = [
    "siren",
    "chainsaw",
    "crying_baby",
    "car_horn",
    "clock_alarm",
    "hand_saw",
    "vacuum_cleaner",
    "engine",
    "glass_breaking",
    "can_opening",
]


COLUMNS = ["clip_id", "path", "label", "category", "group", "source", "fold"]


def build_esc50_labels(esc50_root: Path) -> pd.DataFrame:
    """从 ESC-50 的 meta/esc50.csv 构造二分类标签表。

    ⚠️ 语义类别代理标签，仅用于 Step 1 流水线空跑。

    分组：ESC-50 的 `fold` 是官方设计的，同一段源录音（`src_file`）的所有切片
    落在同一 fold。用 `src_file` 分组（比 fold 更细）。
    """
    meta = pd.read_csv(esc50_root / "meta" / "esc50.csv")
    wanted = set(POSITIVE_CATEGORIES) | set(NEGATIVE_CATEGORIES)
    missing = wanted - set(meta["category"].unique())
    if missing:
        raise ValueError(f"ESC-50 元数据里找不到这些类别: {sorted(missing)}")

    df = meta[meta["category"].isin(wanted)].copy()
    df["label"] = df["category"].isin(POSITIVE_CATEGORIES).astype(int)
    df["path"] = df["filename"].map(lambda f: str(esc50_root / "audio" / f))
    df["clip_id"] = df["filename"].str.removesuffix(".wav")
    # src_file 是原始 Freesound 录音 ID —— 同一录音切出的多个 take 必须同组
    df["group"] = df["src_file"].astype(str)
    df["source"] = "ESC-50"

    return df[COLUMNS].sort_values("clip_id").reset_index(drop=True)


def _arousal_labels(
    df: pd.DataFrame,
    *,
    arousal: str,
    valence: str,
    a_std: str,
    v_std: str,
    a_low: float,
    a_high: float,
    v_floor: float,
    std_cap: float,
) -> pd.DataFrame:
    """沿 arousal 单轴切正负两端，丢弃中间地带。见模块头部的判据说明。"""
    clean = df[(df[a_std] <= std_cap) & (df[v_std] <= std_cap)].copy()
    is_pos = (clean[arousal] <= a_low) & (clean[valence] >= v_floor)
    is_neg = clean[arousal] >= a_high

    out = clean[is_pos | is_neg].copy()
    out["label"] = is_pos[is_pos | is_neg].astype(int)
    out["category"] = out["label"].map({1: "low_arousal", 0: "high_arousal"})
    return out


def build_deam_labels(deam_root: Path) -> pd.DataFrame:
    """DEAM —— Step 2 主力训练集。1802 段 45s 摘录，连续 valence/arousal 标注。"""
    ann_dir = deam_root / "annotations" / "annotations averaged per song" / "song_level"
    frames = []
    for f in sorted(ann_dir.glob("*.csv")):
        d = pd.read_csv(f, skipinitialspace=True)
        d.columns = [c.strip() for c in d.columns]
        frames.append(d[["song_id", "valence_mean", "valence_std", "arousal_mean", "arousal_std"]])
    ann = pd.concat(frames, ignore_index=True).drop_duplicates("song_id")

    df = _arousal_labels(
        ann,
        arousal="arousal_mean", valence="valence_mean",
        a_std="arousal_std", v_std="valence_std",
        a_low=DEAM_AROUSAL_LOW, a_high=DEAM_AROUSAL_HIGH,
        v_floor=DEAM_VALENCE_FLOOR, std_cap=DEAM_STD_CAP,
    )

    df["clip_id"] = "deam_" + df["song_id"].astype(str)
    df["path"] = df["song_id"].map(lambda s: str(deam_root / "MEMD_audio" / f"{s}.mp3"))
    df["group"] = df["clip_id"]  # 一曲一段，无切片泄漏
    df["source"] = "DEAM"
    df["fold"] = -1

    df = df[df["path"].map(lambda p: Path(p).exists())]
    return df[COLUMNS].sort_values("clip_id").reset_index(drop=True)


def build_pmemo_labels(pmemo_root: Path) -> pd.DataFrame:
    """PMEmo —— Step 2 独立测试集（跨数据源 + 跨文化）。794 首副歌摘录。

    ⚠️ 音频为商业曲目摘录，仅可用于研究，不得进入产品。见 LICENSES.md。
    """
    ann = pd.read_csv(pmemo_root / "annotations" / "static_annotations.csv")
    std = pd.read_csv(pmemo_root / "annotations" / "static_annotations_std.csv")
    merged = ann.merge(std, on="musicId")

    df = _arousal_labels(
        merged,
        arousal="Arousal(mean)", valence="Valence(mean)",
        a_std="Arousal(std)", v_std="Valence(std)",
        a_low=PMEMO_AROUSAL_LOW, a_high=PMEMO_AROUSAL_HIGH,
        v_floor=PMEMO_VALENCE_FLOOR, std_cap=PMEMO_STD_CAP,
    )

    df["clip_id"] = "pmemo_" + df["musicId"].astype(str)
    df["path"] = df["musicId"].map(lambda m: str(pmemo_root / "chorus" / f"{m}.mp3"))
    df["group"] = df["clip_id"]
    df["source"] = "PMEmo"
    df["fold"] = -1

    df = df[df["path"].map(lambda p: Path(p).exists())]
    return df[COLUMNS].sort_values("clip_id").reset_index(drop=True)


def build_emosoundscapes_labels(root: Path) -> pd.DataFrame:
    """Emo-Soundscapes —— **环境声域**的 arousal 标注语料。

    这是跨域测试失败后新增的关键数据集：产品播放的是环境声，
    而 DEAM/PMEmo 都是音乐，从音乐学到的规则在环境声上系统性反转
    （见 reports/step5_cross_domain_failure.md）。

    两处设计决定：

    1. **只用 600 段原始录音，丢弃 613 段人工混音。**
       混音是由原始录音合成的，两者同时使用会造成与 ESC-50 `src_file`
       同类的泄漏。混音集可留作后续独立验证。

    2. **没有标注分歧过滤。** 本数据集提供的是众包排序聚合后的评分，
       不含逐条 std，因此无法套用 DEAM/PMEmo 的 std_cap 过滤。
       这是与另两个数据集的一处实质差异，比较结果时须记住。
    """
    ratings = root / "Emo-Soundscapes-Ratings"
    a = pd.read_csv(ratings / "Arousal.csv", header=None, names=["filename", "arousal"])
    v = pd.read_csv(ratings / "Valence.csv", header=None, names=["filename", "valence"])
    ann = a.merge(v, on="filename")

    audio_dir = root / "Emo-Soundscapes-Audio" / "600_Sounds"
    originals = {p.name: p for p in audio_dir.rglob("*.wav")}
    ann = ann[ann["filename"].isin(originals)].copy()

    is_pos = (ann["arousal"] <= EMO_AROUSAL_LOW) & (ann["valence"] >= EMO_VALENCE_FLOOR)
    is_neg = ann["arousal"] >= EMO_AROUSAL_HIGH
    df = ann[is_pos | is_neg].copy()
    df["label"] = is_pos[is_pos | is_neg].astype(int)

    df["path"] = df["filename"].map(lambda f: str(originals[f]))
    df["category"] = df["filename"].map(lambda f: originals[f].parent.name)  # Schafer 六分类
    df["clip_id"] = "emo_" + df["filename"].str.removesuffix(".wav")
    # 分组键用 Freesound 录音 ID：同一条源录音可能切出多个片段
    df["group"] = df["filename"].str.extract(r"(\d{4,})", expand=False).fillna(df["clip_id"])
    df["source"] = "Emo-Soundscapes"
    df["fold"] = -1

    return df[COLUMNS].sort_values("clip_id").reset_index(drop=True)


BUILDERS = {
    "esc50": (build_esc50_labels, "ESC-50"),
    "deam": (build_deam_labels, "DEAM"),
    "pmemo": (build_pmemo_labels, "PMEmo"),
    "emo": (build_emosoundscapes_labels, "Emo-Soundscapes"),
}


# ============================================================ 回归标签
# 笔记 §3.1 早就提出「后续可升级为回归」，此前一直没做。
# 二分类为对齐论文的 agonist/inhibitor 结构，沿 arousal 轴切两端、丢中间，
# 代价是丢掉 64% 的样本（DEAM 1233 → 477）。
# 而被丢掉的「中等唤醒」样本恰恰承载阈值信息 —— 正是规则库需要的东西。

REG_COLUMNS = ["clip_id", "path", "arousal", "valence", "group", "source", "category"]


def build_deam_regression(deam_root: Path, std_cap: float = DEAM_STD_CAP) -> pd.DataFrame:
    """DEAM 连续 arousal（1–9 标度，线性映射到 −1..1）。"""
    ann_dir = deam_root / "annotations" / "annotations averaged per song" / "song_level"
    frames = []
    for f in sorted(ann_dir.glob("*.csv")):
        d = pd.read_csv(f, skipinitialspace=True)
        d.columns = [c.strip() for c in d.columns]
        frames.append(d[["song_id", "valence_mean", "valence_std", "arousal_mean", "arousal_std"]])
    ann = pd.concat(frames, ignore_index=True).drop_duplicates("song_id")

    if std_cap is not None:
        ann = ann[(ann["arousal_std"] <= std_cap) & (ann["valence_std"] <= std_cap)]

    df = ann.copy()
    df["arousal"] = (df["arousal_mean"] - 5.0) / 4.0   # 1..9 → −1..1
    df["valence"] = (df["valence_mean"] - 5.0) / 4.0
    df["clip_id"] = "deam_" + df["song_id"].astype(str)
    df["path"] = df["song_id"].map(lambda s: str(deam_root / "MEMD_audio" / f"{s}.mp3"))
    df["group"] = df["clip_id"]
    df["source"] = "DEAM"
    df["category"] = "music"
    df = df[df["path"].map(lambda p: Path(p).exists())]
    return df[REG_COLUMNS].sort_values("clip_id").reset_index(drop=True)


def build_pmemo_regression(pmemo_root: Path, std_cap: float = PMEMO_STD_CAP) -> pd.DataFrame:
    """PMEmo 连续 arousal（0–1 标度，线性映射到 −1..1）。"""
    a = pd.read_csv(pmemo_root / "annotations" / "static_annotations.csv")
    s = pd.read_csv(pmemo_root / "annotations" / "static_annotations_std.csv")
    df = a.merge(s, on="musicId")
    if std_cap is not None:
        df = df[(df["Arousal(std)"] <= std_cap) & (df["Valence(std)"] <= std_cap)]

    df = df.copy()
    df["arousal"] = df["Arousal(mean)"] * 2.0 - 1.0   # 0..1 → −1..1
    df["valence"] = df["Valence(mean)"] * 2.0 - 1.0
    df["clip_id"] = "pmemo_" + df["musicId"].astype(str)
    df["path"] = df["musicId"].map(lambda m: str(pmemo_root / "chorus" / f"{m}.mp3"))
    df["group"] = df["clip_id"]
    df["source"] = "PMEmo"
    df["category"] = "music"
    df = df[df["path"].map(lambda p: Path(p).exists())]
    return df[REG_COLUMNS].sort_values("clip_id").reset_index(drop=True)


def build_emo_regression(root: Path, include_mixes: bool = False) -> pd.DataFrame:
    """Emo-Soundscapes 连续 arousal（本来就是 −1..1）。

    `include_mixes=True` 会纳入 613 段人工混音。混音由原始录音合成，
    因此其 `group` 键设为参与混合的源片段签名，交叉验证时不会泄漏。
    """
    ratings = root / "Emo-Soundscapes-Ratings"
    a = pd.read_csv(ratings / "Arousal.csv", header=None, names=["filename", "arousal"])
    v = pd.read_csv(ratings / "Valence.csv", header=None, names=["filename", "valence"])
    ann = a.merge(v, on="filename")

    audio = root / "Emo-Soundscapes-Audio"
    files = {p.name: p for p in (audio / "600_Sounds").rglob("*.wav")}
    cats = {p.name: p.parent.name for p in (audio / "600_Sounds").rglob("*.wav")}
    if include_mixes:
        for p in (audio / "613_MixedSounds").rglob("*.wav"):
            files.setdefault(p.name, p)
            cats.setdefault(p.name, "mix")

    df = ann[ann["filename"].isin(files)].copy()
    df["path"] = df["filename"].map(lambda f: str(files[f]))
    df["category"] = df["filename"].map(cats)
    df["clip_id"] = "emo_" + df["filename"].str.removesuffix(".wav")
    df["group"] = df["filename"].str.extract(r"(\d{4,})", expand=False).fillna(df["clip_id"])
    df["source"] = "Emo-Soundscapes"
    return df[REG_COLUMNS].sort_values("clip_id").reset_index(drop=True)


REG_BUILDERS = {
    "deam": build_deam_regression,
    "pmemo": build_pmemo_regression,
    "emo": build_emo_regression,
}
