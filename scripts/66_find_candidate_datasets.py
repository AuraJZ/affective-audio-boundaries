r"""检索候选数据集：整夜 / 听觉刺激 / 含心率的生理数据。

    .venv/Scripts/python scripts/66_find_candidate_datasets.py

## 要找什么，为什么

`step18` 的个体层面结论是「在 ~30 试次下不可检验」，而 `56` 的功效模拟给出
r=0.30 需要每人 **650** 次观测。闭环听觉刺激（CLAS）文献每人每夜数百次刺激，
说明这个量级是可达的 —— 我们只是选了达不到的那类数据。

因此找两类：

| 类型 | 为什么要 |
|---|---|
| **整夜 / 长时程听觉刺激 + EEG** | 每人观测数进入功效模拟给出的可行区间 |
| **音频 + 心率 / HRV** | 消费级设备的主力信号是 HRV 与呼吸，不是头皮 EEG；<br>本项目至今一条心率数据都没用过 |

## 方法

查目录而不是查检索结果的描述 —— 检索摘要会漏字段、也会张冠李戴。
OpenNeuro 用 GraphQL 拿全量清单再本地过滤；PhysioNet 拿其数据库索引页。

**本脚本只负责「找出候选并列出可核验的事实」**（名称、规模、模态、链接）。
是否合用要逐个看文档，不由脚本判定。
"""

from __future__ import annotations

import io
import json
import re
import sys
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

UA = "soundml-datasearch/1.0 (mailto:research@example.org)"

# 整夜 / 睡眠 / 听觉刺激
SLEEP_TERMS = ("sleep", "nap", "overnight", "nocturnal", "slow wave", "slow-wave",
               "insomnia", "polysomn", "nrem", "rem ")
AUDIO_TERMS = ("audit", "sound", "acoust", "music", "tone", "click", "noise",
               "speech", "listen")
CARDIAC_TERMS = ("ecg", "ekg", "heart", "cardiac", "hrv", "pulse", "ppg",
                 "photoplethysm", "bvp")


def get(url: str, data: bytes | None = None) -> str | None:
    req = urllib.request.Request(
        url, data=data,
        headers={"User-Agent": UA,
                 **({"Content-Type": "application/json"} if data else {})})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.read().decode("utf-8", "replace")
    except Exception as e:                                   # noqa: BLE001
        print(f"  ! {type(e).__name__} {url[:70]}")
        return None


def openneuro() -> list[dict]:
    """拉取 OpenNeuro 全量数据集清单。"""
    out, cursor = [], None
    for _ in range(30):
        after = f', after: "{cursor}"' if cursor else ""
        q = ("{ datasets(first: 100%s) { pageInfo { hasNextPage endCursor } "
             "edges { node { id latestSnapshot { tag description { Name } "
             "summary { modalities subjects totalFiles size } } } } } }" % after)
        raw = get("https://openneuro.org/crn/graphql",
                  json.dumps({"query": q}).encode())
        if not raw:
            break
        d = json.loads(raw).get("data", {}).get("datasets")
        if not d:
            break
        for e in d["edges"]:
            # 已下架 / 权限受限的条目其 node 为 null
            n = (e or {}).get("node")
            if not n:
                continue
            snap = n.get("latestSnapshot") or {}
            desc = (snap.get("description") or {}).get("Name") or ""
            summ = snap.get("summary") or {}
            out.append({"id": n["id"], "name": desc,
                        "modalities": summ.get("modalities") or [],
                        "subjects": len(summ.get("subjects") or []),
                        "size": summ.get("size") or 0})
        if not d["pageInfo"]["hasNextPage"]:
            break
        cursor = d["pageInfo"]["endCursor"]
    return out


def physionet() -> list[dict]:
    """PhysioNet 数据库索引。"""
    html = get("https://physionet.org/about/database/")
    if not html:
        return []
    out = []
    for m in re.finditer(
            r'<a href="(/content/[^"]+)/[^"]*">\s*([^<]+?)\s*</a>', html):
        out.append({"slug": m.group(1), "name": m.group(2)})
    # 去重
    seen, uniq = set(), []
    for r in out:
        if r["slug"] not in seen:
            seen.add(r["slug"])
            uniq.append(r)
    return uniq


def hits(text: str, terms) -> bool:
    t = text.lower()
    return any(x in t for x in terms)


def main() -> None:
    print("=" * 86)
    print("[1/2] OpenNeuro")
    on = openneuro()
    print(f"  共 {len(on)} 个数据集")

    sleep_audio = [d for d in on
                   if hits(d["name"], SLEEP_TERMS) or hits(d["name"], AUDIO_TERMS)]
    print(f"\n  睡眠 / 听觉相关 {len(sleep_audio)} 个（按受试数排序，取前 25）：\n")
    print(f"  {'accession':<12}{'受试':>5}{'GB':>7}  模态  名称")
    for d in sorted(sleep_audio, key=lambda x: -x["subjects"])[:25]:
        mod = ",".join(d["modalities"])[:14]
        print(f"  {d['id']:<12}{d['subjects']:>5}{d['size']/1e9:>7.1f}  "
              f"{mod:<15}{d['name'][:52]}")

    print("\n" + "=" * 86)
    print("[2/2] PhysioNet")
    pn = physionet()
    print(f"  共 {len(pn)} 条目录条目")

    def show(label, terms, extra=None):
        sel = [d for d in pn if hits(d["name"], terms)
               and (extra is None or hits(d["name"], extra))]
        print(f"\n  {label} —— {len(sel)} 个：")
        for d in sel[:20]:
            print(f"    {d['name'][:66]}")
            print(f"      https://physionet.org{d['slug']}/")
        return sel

    show("睡眠相关", SLEEP_TERMS)
    show("含心脏信号（名称可见）", CARDIAC_TERMS)
    show("听觉 / 音频相关", AUDIO_TERMS)

    print("\n" + "=" * 86)
    print("⚠️ 名称过滤只能作为初筛 —— PhysioNet 很多睡眠库含 ECG 但名称不体现。")
    print("   候选确定后需逐个读其文档确认：每人观测数、是否含心率、"
          "刺激是否有标记。")


if __name__ == "__main__":
    main()
