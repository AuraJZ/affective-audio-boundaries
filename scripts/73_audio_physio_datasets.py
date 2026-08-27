r"""检索「真实音频 → 生理」的公开数据集，覆盖此前漏掉的渠道。

    .venv/Scripts/python scripts/73_audio_physio_datasets.py

## 上一轮检索的盲区

`66`/`67` 只查了 OpenNeuro（1833 个）与 PhysioNet（426 个），结论是
「找不到闭环听觉刺激（CLAS）体量的数据集」。那个结论**只对这两家成立**。

情感生理学有一批标准语料**不在这两家**，走各自的 EULA 分发
（DEAP、AMIGOS、MAHNOB-HCI、DREAMER 等）。它们恰恰是
「刺激是真实音频/视听材料 + 同步生理 + 逐刺激评分」的那一类 ——
正是本项目一直缺的东西。不查它们，「无公开数据」这句话就不能写进论文。

## 判据（四条都要，缺一不可）

| 条件 | 为什么 |
|---|---|
| **刺激是可识别的真实音频** | 否则做不了声学特征 → 生理。ds003690 就卡在这里（只有纯音） |
| **含心脏或自主神经信号** | 消费级设备的主力信号 |
| **逐刺激的生理与标签** | 只有整段汇总值做不了事件锁时分析 |
| **每人观测数** | `70` 实测：心率要 233 次/人才有 80% 把握 |

## 方法

命名候选逐个查其官方页面与论文；同时扫 Zenodo / OSF / Figshare 的公开 API。
**只报告页面上能读到的事实**，可用与否要下载后才能定 ——
`67` 的教训是「页面线索 ≠ 数据可用」。
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

UA = "soundml-datasearch/2.0 (mailto:research@example.org)"

# 命名候选：情感计算里带生理的标准语料。DOI 优先，没有 DOI 的给官方页面。
NAMED = [
    ("DEAP", "10.1109/T-AFFC.2011.15",
     "32 人 × 40 段音乐视频，EEG 32 通道 + GSR/BVP/呼吸/体温，逐段 valence/arousal 自评"),
    ("MAHNOB-HCI", "10.1109/T-AFFC.2011.25",
     "27 人 × 20 段影片，EEG + ECG + GSR + 呼吸 + 体温 + 眼动"),
    ("AMIGOS", "10.1109/TAFFC.2018.2884461",
     "40 人 × 16 短片 + 4 长片，EEG + ECG + GSR，含个体与群体观看"),
    ("DREAMER", "10.1109/JBHI.2017.2688239",
     "23 人 × 18 段影片，便携 EEG（14 通道）+ ECG"),
    ("ASCERTAIN", "10.1109/TAFFC.2016.2625250",
     "58 人 × 36 段影片，ECG + GSR + EEG + 面部，含人格量表"),
    ("CASE", "10.1038/s41597-019-0209-0",
     "30 人 × 8 段影片，ECG + BVP + GSR + 呼吸 + 体温 + EMG，**连续**标注"),
    ("K-EmoCon", "10.1038/s41597-020-00630-y",
     "32 人自然对话，可穿戴 ECG/EDA/体温 + 三方标注"),
    ("WESAD", "10.1145/3242969.3242985",
     "15 人，胸带 + 腕表 ECG/EDA/EMG/呼吸；刺激为应激任务与影片，非音频"),
    ("PMEmo", "10.1145/3206025.3206037",
     "本项目已用其音频与标注；另含 457 人的 EDA —— **此前未用过其生理部分**"),
    ("EMOPIA / MIREX 类", None, "符号音乐，无生理"),
]

# 仓储 API 全文检索
QUERIES = [
    "music EEG emotion physiological dataset",
    "auditory stimulation heart rate variability dataset",
    "soundscape physiological response dataset",
    "music listening ECG EDA dataset",
    "closed-loop auditory stimulation sleep dataset",
]


def get_json(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except Exception:                                        # noqa: BLE001
        return None


def crossref(doi: str) -> dict | None:
    d = get_json(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    return d.get("message") if d else None


def zenodo(q: str, n: int = 8) -> list[dict]:
    d = get_json("https://zenodo.org/api/records?"
                 + urllib.parse.urlencode({"q": q, "size": n,
                                           "type": "dataset",
                                           "sort": "mostrecent"}))
    out = []
    for h in (d or {}).get("hits", {}).get("hits", []):
        m = h.get("metadata", {})
        out.append({"title": m.get("title", "")[:88],
                    "doi": m.get("doi", ""),
                    "access": m.get("access_right", ""),
                    "date": (m.get("publication_date") or "")[:4]})
    return out


def osf(q: str, n: int = 8) -> list[dict]:
    d = get_json("https://api.osf.io/v2/nodes/?"
                 + urllib.parse.urlencode({"filter[title]": q, "page[size]": n}))
    out = []
    for h in (d or {}).get("data", []):
        a = h.get("attributes", {})
        out.append({"title": (a.get("title") or "")[:88],
                    "id": h.get("id", ""),
                    "public": a.get("public"),
                    "date": (a.get("date_created") or "")[:4]})
    return out


def main() -> None:
    print("=" * 84)
    print("[1/2] 命名候选：情感计算里带生理的标准语料")
    print("      （这批不在 OpenNeuro / PhysioNet，走各自的 EULA 分发）\n")
    for name, doi, note in NAMED:
        print("─" * 84)
        print(f"{name}")
        print(f"  记载  {note}")
        if not doi:
            print("  DOI   无（非本类）")
            continue
        m = crossref(doi)
        if not m:
            print(f"  ❌ DOI 查不到：{doi}")
            continue
        title = " ".join(m.get("title", []) or [])
        yr = m.get("issued", {}).get("date-parts", [[0]])[0][0]
        auth = "; ".join(f"{a.get('family','')}" for a in m.get("author", [])[:3])
        print(f"  文献  {title[:82]}")
        print(f"        {auth} et al., {yr}, {(m.get('container-title') or [''])[0][:44]}")
        print(f"  DOI   {doi}  ✅")
        time.sleep(0.3)

    print("\n" + "=" * 84)
    print("[2/2] 仓储全文检索（此前完全未查）\n")
    for q in QUERIES:
        print("─" * 84)
        print(f"查询：{q}")
        z = zenodo(q)
        print(f"  Zenodo {len(z)} 条")
        for r in z[:5]:
            print(f"    [{r['date']}] {r['access']:<9} {r['title']}")
        o = osf(q)
        print(f"  OSF {len(o)} 条")
        for r in o[:4]:
            print(f"    [{r['date']}] {'public' if r['public'] else 'private':<9} "
                  f"{r['title']}")
        time.sleep(0.5)

    print("\n" + "=" * 84)
    print("⚠️ 以上只是页面/元数据可读到的事实，不等于数据可用。")
    print("   `67` 的教训：线索 ≠ 可用。任一候选选定后仍须核对四件事 ——")
    print("   刺激音频是否可获取、生理是否逐刺激、每人观测数、以及分发条款。")


if __name__ == "__main__":
    main()
