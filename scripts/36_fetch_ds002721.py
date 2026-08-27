"""下载 OpenNeuro ds002721（EEG + BIDS 事件标记）与 Eerola & Vuoskoski Soundtracks 音频。

    .venv/Scripts/python scripts/36_fetch_ds002721.py --subjects 8
    .venv/Scripts/python scripts/36_fetch_ds002721.py            # 全部 31 人

## 为什么换这个数据集

BIRAFFE2 失败的根因无法排除：Procedure 与 BioSigs 各自携带**绝对时间戳**，
其语义（作答时刻？呈现时刻？哪个时钟？）我无法核实，而总平均在 ±16 s 内无峰
与「试次落在随机位置」完全一致。这不是调参能解决的。

ds002721 从规范层面消除了该歧义：BIDS `events.tsv` 的 `onset` **定义为相对
记录起点的秒数**，与 EDF 采样点一一对应，不存在跨时钟对齐问题。

## 数据结构

| 层 | 来源 | 内容 |
|---|---|---|
| 音频 | OSF `p6vkg` / `Set1.zip` | 360 段 ~15 s 电影配乐 mp3 |
| 组均值评分 | `mean_ratings_set1.csv` | valence/energy/tension + 5 类离散情绪 |
| 个体逐试次评分 | EEG 触发码 800–807 + 833–841 | 8 问题 × 9 点 |
| 脑电 | 31 人 × 6 run × 19 通道 @ 1 kHz | run1/run6 静息，run2–5 各 10 试次 |

## 触发码（来自 events.json）

- `786` 注视十字 / `788` **音乐起始** / `301–660` 刺激 ID（减 300 → NNN.mp3）
- `800–807` 问题 / `833–841` 作答 1–9
- `257` 眨眼 / `259` 肌电 / `263` 工频

`events.json` 写的是「301–360」，但实测码达 657，且每 run 恰好 10 个落在
301–660 区间——应为笔误，实际覆盖 Set1 的 360 段。下载后由
`37_eeg_positive_controls.py` 核验。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import urllib.request
import zipfile
from pathlib import Path

from soundml.provenance import REPO_ROOT

S3 = "https://s3.amazonaws.com/openneuro.org/ds002721"
OSF_API = "https://api.osf.io/v2/nodes/p6vkg/files/osfstorage/"
DEST_EEG = REPO_ROOT / "data" / "raw" / "ds002721"
DEST_AUD = REPO_ROOT / "data" / "raw" / "soundtracks"

ROOT_FILES = ["README", "CHANGES", "dataset_description.json", "participants.tsv",
              "code/load_filmClips.m"]
PER_RUN = ["eeg.edf", "eeg.json", "events.tsv", "events.json", "channels.tsv"]


def _get(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "soundml/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch(rel: str) -> tuple[str, int, str]:
    """下载单个 BIDS 相对路径；已存在且非空则跳过。"""
    out = DEST_EEG / rel
    if out.exists() and out.stat().st_size > 0:
        return rel, out.stat().st_size, "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = _get(f"{S3}/{rel}")
    except Exception as e:                                   # noqa: BLE001
        return rel, 0, f"FAIL {type(e).__name__}"
    tmp = out.with_suffix(out.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(out)
    return rel, len(data), "ok"


def fetch_audio() -> None:
    """OSF：Set1.zip（360 段 mp3）+ 两个均值评分表。"""
    DEST_AUD.mkdir(parents=True, exist_ok=True)
    listing = json.loads(_get(OSF_API, timeout=120))["data"]
    want = {"Set1.zip", "mean_ratings_set1.csv", "mean_ratings_set2.csv",
            "set2_tracklist.csv", "readme.md"}
    for f in listing:
        name = f["attributes"]["name"]
        if name not in want:
            continue
        if name.endswith(".zip"):
            mp3_dir = DEST_AUD / "Set1"
            if mp3_dir.exists() and len(list(mp3_dir.glob("*.mp3"))) >= 360:
                print(f"  skip {name}（已解压 {len(list(mp3_dir.glob('*.mp3')))} 个 mp3）")
                continue
            print(f"  下载 {name} ({f['attributes']['size']/1e6:.0f} MB) …", flush=True)
            blob = _get(f["links"]["download"], timeout=1800)
            mp3_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                for m in z.namelist():
                    if m.lower().endswith(".mp3"):
                        (mp3_dir / Path(m).name).write_bytes(z.read(m))
            print(f"    解压 {len(list(mp3_dir.glob('*.mp3')))} 个 mp3 → {mp3_dir}")
        else:
            out = DEST_AUD / name
            if out.exists():
                continue
            out.write_bytes(_get(f["links"]["download"], timeout=300))
            print(f"  {name}  {out.stat().st_size:,} B")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=31, help="下载前 N 名受试")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-audio", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("[1/2] 音频与评分（OSF p6vkg）")
    if not args.skip_audio:
        fetch_audio()
    print(f"\n[2/2] EEG（OpenNeuro ds002721，前 {args.subjects} 名受试）")

    rels = list(ROOT_FILES)
    for s in range(1, args.subjects + 1):
        for run in range(1, 7):
            for suf in PER_RUN:
                rels.append(f"sub-{s:02d}/eeg/sub-{s:02d}_task-run{run}_{suf}")

    total, done, failed = 0, 0, []
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for rel, n, status in ex.map(fetch, rels):
            done += 1
            total += n
            if status.startswith("FAIL"):
                failed.append((rel, status))
            if done % 40 == 0 or done == len(rels):
                print(f"  {done}/{len(rels)}   累计 {total/1e9:.2f} GB", flush=True)

    print(f"\n完成：{len(rels) - len(failed)}/{len(rels)} 个文件，{total/1e9:.2f} GB")
    if failed:
        print(f"失败 {len(failed)} 个：")
        for rel, why in failed[:10]:
            print(f"  {rel}  {why}")


if __name__ == "__main__":
    main()
