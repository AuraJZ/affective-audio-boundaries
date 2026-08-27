r"""逐个核候选数据集的关键事实，不靠名称猜。

    .venv/Scripts/python scripts/67_probe_candidates.py

## 要确认三件事

| | 为什么 |
|---|---|
| **每人观测数** | `56` 的功效模拟：r=0.30 需 650 次/人。~30 次的数据集回答不了个体问题 |
| **是否含心率 / HRV** | 消费级设备的主力信号；本项目至今一条心率数据都没用过 |
| **是否有声音刺激标记** | 没有刺激标记就只能做「状态 → 生理」，做不了「声音 → 生理」 |

第三条是关键分水岭。整夜睡眠库很多，但绝大多数是**被动记录**，
没有受控的声学刺激 —— 那类数据能回答「生理能不能分睡眠阶段」，
回答不了「播什么声音会怎样」。

## 方法

OpenNeuro 取 `dataset_description.json` 与 README；
PhysioNet 取内容页正文。只报告页面里能读到的事实，不做推断。
"""

from __future__ import annotations

import io
import json
import re
import sys
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

UA = "soundml-probe/1.0 (mailto:research@example.org)"

OPENNEURO = {
    "ds006466": "HeartBEAM: Older Adult Resting State and Auditory Oddball",
    "ds003690": "EEG, ECG and pupil data from young and older adults",
    "ds005555": "Bitbrain Open Access Sleep (BOAS)",
    "ds004551": "iEEG on children during slow wave sleep",
}
PHYSIONET = {
    "dreamt": "DREAMT wearable multisensor sleep",
    "bidsleep-dataset": "Multi-Night Instantaneous Heart Rate + Accelerometry",
    "mmash": "Multilevel Monitoring of Activity and Sleep",
    "sleep-accel": "Motion and heart rate from wrist wearable",
    "slpdb": "MIT-BIH Polysomnographic",
    "capslpdb": "CAP Sleep Database",
    "cps-dataset-sleep": "Comprehensive Polysomnography (CPS)",
}

CARDIAC = ("ecg", "ekg", "heart rate", "heart-rate", "hrv", "ppg", "pulse",
           "photoplethysm", "bvp", "interbeat", "rr interval")
STIM = ("stimul", "auditory stimulation", "tone", "click", "sound present",
        "acoustic stimulation", "closed-loop", "closed loop", "oddball",
        "played", "cue")


def get(url: str, data: bytes | None = None) -> str | None:
    req = urllib.request.Request(
        url, data=data,
        headers={"User-Agent": UA,
                 **({"Content-Type": "application/json"} if data else {})})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:                                        # noqa: BLE001
        return None


def found(text: str, terms) -> list[str]:
    t = text.lower()
    return sorted({x for x in terms if x in t})


def strip_html(h: str) -> str:
    h = re.sub(r"<script.*?</script>|<style.*?</style>", " ", h, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", h))


def probe_openneuro(acc: str, label: str) -> None:
    print(f"\n{'─' * 82}\n{acc}  {label}")
    base = f"https://s3.amazonaws.com/openneuro.org/{acc}"
    txt = ""
    for f in ("README", "dataset_description.json", "task-description.json"):
        r = get(f"{base}/{f}")
        if r:
            txt += "\n" + r
    if not txt:
        print("  ! 取不到 README / description")
        return
    print(f"  心脏信号线索: {found(txt, CARDIAC) or '无'}")
    print(f"  刺激线索    : {found(txt, STIM) or '无'}")
    m = re.search(r'"Name"\s*:\s*"([^"]+)"', txt)
    if m:
        print(f"  正式名称    : {m.group(1)[:70]}")
    snippet = re.sub(r"\s+", " ", txt)[:300]
    print(f"  摘要        : {snippet}")


def probe_physionet(slug: str, label: str) -> None:
    print(f"\n{'─' * 82}\nphysionet/{slug}  {label}")
    h = get(f"https://physionet.org/content/{slug}/")
    if not h:
        print("  ! 取不到内容页")
        return
    t = strip_html(h)
    print(f"  心脏信号线索: {found(t, CARDIAC) or '无'}")
    print(f"  刺激线索    : {found(t, STIM) or '无'}")
    m = re.search(r"Abstract(.{80,420})", t)
    if m:
        print(f"  摘要        : {m.group(1).strip()[:340]}")


def main() -> None:
    print("=" * 82)
    print("OpenNeuro 候选")
    for acc, label in OPENNEURO.items():
        probe_openneuro(acc, label)

    print("\n" + "=" * 82)
    print("PhysioNet 候选")
    for slug, label in PHYSIONET.items():
        probe_physionet(slug, label)

    print("\n" + "=" * 82)
    print("⚠️ 以上只是页面可读到的线索，不等于数据可用。")
    print("   选定后仍需下载核对：每人实际观测数、刺激标记的时间精度、")
    print("   以及心率是逐搏还是已聚合的分钟均值 —— 后者做不了事件锁时分析。")


if __name__ == "__main__":
    main()
