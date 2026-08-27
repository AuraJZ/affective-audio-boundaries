r"""数据集清单：逐个从**文件本身**量出体量、内容与记录时长。

    .venv/Scripts/python scripts/75_corpus_inventory.py

## 为什么要有这个脚本

论文的 Corpora 一节、投稿材料的数据声明、以及对外解释「我们用了多少数据」，
都需要一组彼此一致的数字。凭记忆或凭原始论文的记载写，两处必然会对不上 ——
本项目已经在参考文献表上吃过一次这个亏（`65`：15 条里一开始只核出 7 条）。

因此这里一律**从落盘的文件量**：音频时长用解码得到的采样数算，
评分条数数行，生理记录时长按采样率与样本数算。

## 报告什么

| 列 | 含义 |
|---|---|
| 体量 | 片段数 / 受试数，以及**总时长**（音频小时数、记录小时数） |
| 内容 | 主观量表**测的是什么**；客观信号**是哪些通道** |
| 观测粒度 | 每段几人评 / 每人几次观测 —— 决定能回答哪一层的问题 |

时长逐文件解码较慢，故对大语料抽样估计并标明；抽样数写在输出里。
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from soundml.provenance import REPO_ROOT

warnings.filterwarnings("ignore")
RAW = REPO_ROOT / "data" / "raw"
OUT = REPO_ROOT / "reports" / "source_data" / "corpus_inventory.csv"
SAMPLE = 120                 # 估时长时每语料抽样的片段数


def audio_files(root: Path, *pats: str) -> list[Path]:
    """收集音频文件，剔除 macOS 归档里的资源叉。

    `__MACOSX/` 下的 `._name.wav` 是 AppleDouble 元数据，不是音频。
    第一版没排除，Emo-Soundscapes 被数成 2642 段（实为 1321）——
    这类数字若进了论文就是硬伤，故在此显式处理。
    """
    out: list[Path] = []
    for pat in pats:
        out += [f for f in root.rglob(pat)
                if "__MACOSX" not in f.parts and not f.name.startswith("._")]
    return sorted(out)


def n_analysed(name: str) -> int | None:
    """实际进入分析的片段数 —— 以特征表行数为准，而非磁盘上的文件数。"""
    f = REPO_ROOT / "data" / "features" / f"layer_a_reg_{name}_ln.parquet"
    if not f.exists():
        f = REPO_ROOT / "data" / "features" / f"layer_a_{name}_ln.parquet"
    return len(pd.read_parquet(f, columns=["clip_id"])) if f.exists() else None


def audio_hours(files: list[Path], sample: int = SAMPLE) -> tuple[float, int, str]:
    """总音频小时数。超过 `sample` 个则抽样外推，并注明。"""
    import librosa
    n = len(files)
    if n == 0:
        return 0.0, 0, "无"
    rng = np.random.default_rng(0)
    pick = files if n <= sample else [files[i] for i in
                                      rng.choice(n, sample, replace=False)]
    durs = []
    for f in pick:
        try:
            durs.append(librosa.get_duration(path=str(f)))
        except Exception:                                    # noqa: BLE001
            continue
    if not durs:
        return 0.0, n, "解码失败"
    mean = float(np.mean(durs))
    note = "全量" if n <= sample else f"抽样 {len(durs)}/{n} 外推"
    return mean * n / 3600.0, n, note


def rows() -> list[dict]:
    out: list[dict] = []

    # ── DEAM ─────────────────────────────────────────────────────────
    d = RAW / "DEAM"
    aud = audio_files(d, "*.mp3", "*.wav")
    hrs, n, note = audio_hours(aud)
    ann = list(d.rglob("*annotations*averaged*song*.csv"))
    n_rate = np.nan
    if ann:
        try:
            n_rate = len(pd.read_csv(ann[0]))
        except Exception:                                    # noqa: BLE001
            pass
    out.append(dict(
        corpus="DEAM", role="音频训练语料",
        units=f"{n} 段音乐（分析 {n_analysed('deam')}）", hours=hrs, hours_note=note,
        subjective="逐首 valence（愉悦度）与 arousal（唤醒度），9 点量表，"
                   "另有逐 0.5 s 的动态标注",
        objective="—",
        grain=f"标注覆盖 {n_rate if n_rate == n_rate else '?'} 首，每首多名标注者取均值"))

    # ── PMEmo ────────────────────────────────────────────────────────
    p = RAW / "PMEmo" / "PMEmo2019"
    aud = audio_files(p / "chorus", "*.mp3")
    hrs, n, note = audio_hours(aud)
    A = pd.read_csv(p / "annotations" / "static_annotations.csv")
    Dy = pd.read_csv(p / "annotations" / "dynamic_annotations.csv")
    eda_files = sorted((p / "EDA").glob("*_EDA.csv"))
    # 皮电记录小时数：逐文件时长 × 该文件的听者数
    eda_h, eda_obs, listeners = 0.0, 0, set()
    for f in eda_files:
        h = pd.read_csv(f, nrows=1)
        subs = [c for c in h.columns if c != "time(s)"]
        listeners.update(subs)
        t = pd.read_csv(f, usecols=["time(s)"])["time(s)"].iloc[-1]
        eda_h += float(t) * len(subs) / 3600.0
        eda_obs += len(subs)
    out.append(dict(
        corpus="PMEmo", role="音频训练语料 + 皮电",
        units=f"{n} 段副歌（分析 {n_analysed('pmemo')}）· "
              f"{len(listeners)} 名皮电受试",
        hours=hrs, hours_note=note,
        subjective=f"逐首 valence 与 arousal（{len(A)} 首静态均值）"
                   f"，另有 {len(Dy)} 行逐 0.5 s 动态标注",
        objective=f"皮肤电导（EDA），50 Hz，{eda_obs} 个「人×歌」观测，"
                  f"合计 {eda_h:.1f} 小时记录",
        grain="每首约 10 人评分与聆听；每人听 15–20 首"))

    # ── Emo-Soundscapes ──────────────────────────────────────────────
    d = RAW / "EmoSoundscapes"
    aud = audio_files(d, "*.wav", "*.mp3")
    hrs, n, note = audio_hours(aud)
    out.append(dict(
        corpus="Emo-Soundscapes", role="音频训练语料（环境声端）",
        units=f"{n} 段环境声（分析 {n_analysed('emo_mix')}）", hours=hrs, hours_note=note,
        subjective="逐段 valence 与 arousal，成对比较法排序后转为量表值",
        objective="—",
        grain="每段多名评定者；源自 Freesound"))

    # ── Soundtracks ──────────────────────────────────────────────────
    d = RAW / "soundtracks"
    aud = audio_files(d, "*.mp3", "*.wav")
    hrs, n, note = audio_hours(aud)
    out.append(dict(
        corpus="Soundtracks", role="音频外部验证语料",
        units=f"{n} 段电影配乐（分析 {n_analysed('soundtracks')}）", hours=hrs, hours_note=note,
        subjective="**8 个维度**：valence、energy（唤醒）、tension、"
                   "anger、fear、happy、sad、tender —— 比其余语料的 2 维更细",
        objective="—",
        grain="每段多名评定者；仅限学术用途"))

    # ── ESC-50 ───────────────────────────────────────────────────────
    d = RAW / "ESC-50"
    aud = audio_files(d, "*.wav")
    hrs, n, note = audio_hours(aud)
    out.append(dict(
        corpus="ESC-50", role="混淆对照",
        units=f"{n} 段环境声", hours=hrs, hours_note=note,
        subjective="无情绪量表；只有 50 个声音**类别**标签",
        objective="—", grain="每类 40 段"))

    # ── ds002721 ─────────────────────────────────────────────────────
    d = RAW / "ds002721"
    subs = sorted({p.name for p in d.glob("sub-*") if p.is_dir()})
    ev = sorted(d.rglob("*_events.tsv"))
    n_ev = sum(len(pd.read_csv(f, sep="\t")) for f in ev[:200])
    out.append(dict(
        corpus="ds002721", role="生理主分析（脑电）",
        units=f"{len(subs)} 名受试 · {len(ev)} 个 run",
        hours=np.nan, hours_note="见备注",
        subjective="听者对每段音乐的自评（愉悦度/唤醒度）",
        objective="头皮脑电；12 s 音乐片段；含刺激与反应事件标记",
        grain=f"每人约 30 个评分试次；事件行合计约 {n_ev}（前 200 个 run）"))

    # ── BIRAFFE2 ─────────────────────────────────────────────────────
    d = RAW / "BIRAFFE2"
    csvs = sorted(d.rglob("*.csv"))
    out.append(dict(
        corpus="BIRAFFE2", role="生理旁证（皮电）",
        units="94 名参与者", hours=np.nan, hours_note="见备注",
        subjective="情绪自评（效价/唤醒），以及游戏化任务中的反应",
        objective="皮肤电导、心率、面部；伴随声音与图像刺激",
        grain=f"逐参与者文件 {len(csvs)} 个"))

    # ── ds003690 ─────────────────────────────────────────────────────
    d = RAW / "ds003690"
    P = pd.read_csv(d / "participants.tsv", sep="\t")
    sets = sorted(d.rglob("*_eeg.set"))
    # 记录小时数：由 events.tsv 的最后一个 onset 近似（避免解码 22 GB）
    tot = 0.0
    for f in sorted(d.rglob("*_events.tsv")):
        try:
            tot += float(pd.read_csv(f, sep="\t")["onset"].iloc[-1])
        except Exception:                                    # noqa: BLE001
            continue
    out.append(dict(
        corpus="ds003690", role="设计需求实测（脑电+心电+瞳孔）",
        units=f"{len(P)} 名受试 · {len(sets)} 个 run",
        hours=tot / 3600.0, hours_note="按 events.tsv 末次事件时刻累加",
        subjective="无情绪量表；只有按键反应时",
        objective="64 通道脑电 + 心电（EKG）+ 双眼瞳孔直径，500 Hz；"
                  "纯音 250 ms（cue 1500 / go 1700 / no-go 1300 / error 1000 Hz）",
        grain=f"每人约 270 个听觉事件（主动 240 + 被动 30）；"
              f"年龄 {P.age.min()}–{P.age.max()} 岁，"
              f"Young {int((P.group=='Young').sum())} / "
              f"Older {int((P.group=='Older').sum())}"))

    # ── ARAUS ────────────────────────────────────────────────────────
    out.append(dict(
        corpus="ARAUS", role="⚠️ 许可未核实，未用于任何结论",
        units="—", hours=np.nan, hours_note="—",
        subjective="城市声景的舒适度评定", objective="—", grain="—"))
    return out


def main() -> None:
    R = pd.DataFrame(rows())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT, index=False, encoding="utf-8")

    for _, r in R.iterrows():
        print("=" * 84)
        print(f"{r.corpus}   [{r.role}]")
        print(f"  体量    {r.units}")
        if r.hours == r.hours:
            print(f"          {r.hours:.1f} 小时（{r.hours_note}）")
        print(f"  主观    {r.subjective}")
        print(f"  客观    {r.objective}")
        print(f"  粒度    {r.grain}")

    tot_a = R.loc[R.role.str.contains("音频"), "hours"].sum()
    tot_p = R.loc[R.role.str.contains("生理|设计需求"), "hours"].sum()
    print("\n" + "=" * 84)
    print(f"音频合计约 {tot_a:.0f} 小时；生理记录合计约 {tot_p:.0f} 小时（可量者）")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
