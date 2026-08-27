r"""下载 OpenNeuro ds003690：EEG + ECG + 瞳孔，听觉线索反应时任务。

    .venv/Scripts/python scripts/68_fetch_ds003690.py

## 为什么是这个数据集

`step18` 的个体层面结论卡在观测数：ds002721 每人约 30 个试次，
而 `56` 的功效模拟给出 r=0.40 需 271、r=0.30 需 650。

ds003690 每人 **240 个试次**（两个任务 × 120），是前者的 8 倍，
且带 ds002721 缺的两样东西：

| | |
|---|---|
| **ECG 通道** | 本项目至今未用过任何心率数据，而它是消费级设备的主力信号 |
| **刺激参数写死在文档里** | 纯音 cue 1500 Hz / go 1700 Hz / no-go 1300 Hz，250 ms，67 dB(A) |

第二条尤其重要：ds002721 的刺激码对不上音频，声学↔生理一线整条断掉。
这里不存在该问题 —— 刺激就是已知频率的纯音。

## 局限（先写明，免得后面自我说服）

不是音乐，不是睡眠。它能回答「个体内，声音事件能否被生理追踪」，
**回答不了「哪种音乐更助眠」**。它补的是方法学缺口，不是产品问题。

## 结构

每名受试：`task-simpleRT` 与 `task-gonogo` 各 2 个 run，外加被动聆听。
`.set` 为 EEGLAB 格式，EEG / ECG / 瞳孔同在一个文件的不同通道里。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from soundml.provenance import REPO_ROOT

S3 = "https://s3.amazonaws.com/openneuro.org"
ACC = "ds003690"
DEST = REPO_ROOT / "data" / "raw" / ACC


def listing() -> list[tuple[str, int]]:
    """列出全部对象（key, size），处理分页。"""
    out, token = [], None
    while True:
        url = f"{S3}/?list-type=2&prefix={ACC}/&max-keys=1000"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        req = urllib.request.Request(url, headers={"User-Agent": "soundml/1.0"})
        with urllib.request.urlopen(req, timeout=180) as r:
            xml = r.read().decode()
        root = ET.fromstring(xml)
        ns = {"s3": root.tag.split("}")[0].strip("{")}
        for c in root.findall("s3:Contents", ns):
            out.append((c.find("s3:Key", ns).text,
                        int(c.find("s3:Size", ns).text)))
        trunc = root.find("s3:IsTruncated", ns)
        if trunc is None or trunc.text != "true":
            break
        token = root.find("s3:NextContinuationToken", ns).text
    return out


def fetch(key: str, size: int) -> tuple[str, int, str]:
    """下载单个对象。任何失败都返回状态而不抛出 —— 一个文件不该中断整轮。"""
    out = DEST / key[len(ACC) + 1:]
    if out.exists() and out.stat().st_size == size:
        return key, size, "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")

    for attempt in range(3):
        try:
            req = urllib.request.Request(f"{S3}/{key}",
                                         headers={"User-Agent": "soundml/1.0"})
            got = 0
            with urllib.request.urlopen(req, timeout=900) as r, \
                    tmp.open("wb") as fh:               # 流式，避免 60 MB 驻内存
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
                    got += len(chunk)
            if size and got != size:
                raise OSError(f"短读 {got}/{size}")
            # Windows 上杀软会短暂持有刚落盘的文件，改名随即 WinError 32。
            # 那是暂时性占用，退避重试即可，不是下载失败。
            for wait in (0.5, 1, 2, 4, 8):
                try:
                    tmp.replace(out)
                    return key, got, "ok"
                except PermissionError:
                    time.sleep(wait)
            return key, 0, "FAIL 改名持续被占用"
        except Exception as e:                               # noqa: BLE001
            if attempt == 2:
                return key, 0, f"FAIL {type(e).__name__}"
            time.sleep(2 * (attempt + 1))
    return key, 0, "FAIL 未知"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=0,
                    help="仅下载前 N 名受试（0 = 全部）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="只列清单不下载")
    args = ap.parse_args()

    keys = listing()
    total_gb = sum(s for _, s in keys) / 1e9
    subs = sorted({m.group(1) for k, _ in keys
                   if (m := re.search(r"/(sub-[A-Za-z0-9]+)/", k))})
    print(f"{ACC}: {len(keys)} 个对象，{total_gb:.1f} GB，{len(subs)} 名受试")

    if args.subjects:
        keep = set(subs[: args.subjects])
        keys = [(k, s) for k, s in keys
                if not (m := re.search(r"/(sub-[A-Za-z0-9]+)/", k))
                or m.group(1) in keep]
        print(f"  限定前 {args.subjects} 名 → {len(keys)} 个对象，"
              f"{sum(s for _, s in keys)/1e9:.1f} GB")

    if args.dry_run:
        from collections import Counter
        ext = Counter(k.rsplit(".", 1)[-1] for k, _ in keys)
        by_ext = Counter()
        for k, s in keys:
            by_ext[k.rsplit(".", 1)[-1]] += s
        print("\n  按扩展名：")
        for e, n in ext.most_common(10):
            print(f"    .{e:<12}{n:>5} 个   {by_ext[e]/1e9:>7.2f} GB")
        return

    done, got, failed = 0, 0, []
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for key, n, status in ex.map(lambda kv: fetch(*kv), keys):
            done += 1
            got += n
            if status.startswith("FAIL"):
                failed.append((key, status))
            if done % 50 == 0 or done == len(keys):
                print(f"  {done}/{len(keys)}   {got/1e9:.2f} GB"
                      f"   失败 {len(failed)}", flush=True)

    print(f"\n完成 {len(keys) - len(failed)}/{len(keys)}，{got/1e9:.2f} GB → {DEST}")
    for k, why in failed[:8]:
        print(f"  失败 {k}  {why}")


if __name__ == "__main__":
    main()
