r"""ds003690：逐试次抽取三条通路的事件锁时响应（脑电 / 心率 / 瞳孔）。

    .venv/Scripts/python scripts/69_ds003690_extract.py

本脚本**只抽取，不判定**。判定在 `70`。分开是因为抽取要读 23 GB、约一小时，
不能每改一次判据就重跑。

## 为什么再引入一个数据集

`step18` 的个体层面结论是「在每人约 30 试次下**不可检验**」，`56` 的功效模拟
给出 r=0.40 需 271 次/人。那是模拟数 —— 用 Ledoit-Wolf 收缩协方差生成的。
ds003690 每人约 **270 个 cue 事件**，恰好落在该阈值上，因此可以把
「多少次才够」从**模拟**变成**实测**。

同时这是本项目第一次用心脏数据。心率是消费级设备的主力信号，
而论文 Limitations 现在明写「未覆盖」。

## 三条通路，为什么是这三条

| 通路 | 为什么 |
|---|---|
| **脑电 N1–P2** | 听觉诱发电位是心理生理学里最稳的效应之一，做阳性对照的天花板 |
| **心率** | 消费级设备实际能测的信号；本项目至今零覆盖 |
| **瞳孔** | 自主神经的第三个独立读数，用来判断「次数需求」是通路属性还是普遍规律 |

## 测量窗与判据，全部先验固定

用**固定窗均值**而非挑峰。挑峰（P2max − N1min）在噪声下恒为正，
其符号翻转零分布均值为 0 而观测恒 > 0 —— 那会把纯噪声判成阳性。
固定窗均值在基线校正后零均值，符号翻转才是合法零分布。

    脑电   ROI = F1 Fz F2 FC1 FCz FC2 C1 Cz C2（乳突参考）
           N1 = 80–130 ms 均值，P2 = 170–230 ms 均值
           量 = N1（主，见下文「诊断输出」），另存 P2 − N1
    心率   R 峰 → 瞬时 HR → 4 Hz 重采样，量 = 1–5 s 均值 − 基线
    瞳孔   X/Y 均值，眨眼（零值）线性插值，量 = 0.5–2.5 s 均值 − 基线

## 阴性对照：随机锚点，不是固定平移 🔴

第一版写的是「锚点前移 6 s」。那是错的：cue 间隔约 8.3 s，前移 6 s
正好落在**上一试次的按键**上 —— 与 ds002721 C3 踩的是同一个坑
（平移零分布把窗口移进了另一段有事件的时间）。

改用**随机锚点**：在 run 内均匀取同样数量的锚点，跑完全相同的流程。
随机锚点与事件无固定相位关系，真实的事件锁时响应会被平均掉；
而流程性偏倚（滤波边缘、基线窗与响应窗的系统性差异）会原样保留。
这正是阴性对照要区分的两件事。

**符号翻转零分布挡不住流程性偏倚** —— 若每个试次都被同样地抬高 5 µV，
观测均值为 5，翻转零分布仍以 0 为中心，会判成显著。所以阴性对照不可省。

而它当场就抓到了一个：**心率的随机锚点臂并非零均值**（1–5 s 窗内约 −0.9 bpm）。
原因是心率在 8 分钟的 run 内单调下行（适应/放松），
而基线窗（−3–0 s）在响应窗（1–5 s）之前 —— 任意锚点都会「基线偏高」。
瞳孔臂同期是平的，说明这是心率信号自身的漂移，不是通用的流程 bug。

**后果**：心率的检验不能是「真实臂是否偏离 0」，必须是**真实臂 vs 随机锚点臂**。
`70` 里三条通路统一采用后者。

## 诊断输出，以及它抓到的判据错误 🔴

除逐试次测量外，另存**每人每通路的平均波形**。
ds002721 的 C3 当初判据窗设错（0–300 ms 对 452 ms 的反应），
是靠预先声明的平均波形才看出来的，不是靠判据。这次同样奏效：

    基线 ~0 → N1 −3.28 µV @ 102 ms（教科书潜伏期与极性）
            → 170–230 ms 仍为 −2.4 ~ −1.2 µV，**P2 没有出现**
            → 300 ms 起单调负走，376 ms 达 −5.2 µV

那个大负波是 **CNV**（线索—靶子间的关联性负变）。线索式反应时任务里
CNV 从约 300 ms 起建立并持续到靶子出现，会把 P2 抵消掉 ——
这是任务设计就能预见的，不该等数据来告诉我。

因此 `P2 − N1` 实际测的是「N1 的回落幅度」，不是 N1–P2 复合波。

**处理**：改用 **N1 单窗均值**（80–130 ms）为主测量，同时保留预设的
`P2 − N1` 一并报告。两者都是固定窗线性量，符号翻转都合法。
改动理由是任务设计（CNV 重叠），不是「哪个效应大选哪个」；
若两者结论一致，则该选择不影响推论 —— 这一点在 `70` 里明确检查。
"""

from __future__ import annotations

import argparse
import warnings
import zlib
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

RAW = REPO_ROOT / "data" / "raw" / "ds003690"
OUT = REPO_ROOT / "data" / "features" / "ds003690_trials.parquet"
OUT_WAVE = REPO_ROOT / "data" / "features" / "ds003690_waveforms.parquet"
# 逐 run 落盘。整轮约一小时，只在结尾写盘的话中断一次就全白跑
# —— `39` 已经踩过一次，不再重复。
CACHE = REPO_ROOT / "data" / "features" / "ds003690_cache"
SEED = 20260729

# ── 先验固定的参数 ────────────────────────────────────────────────────────
ROI = ["F1", "Fz", "F2", "FC1", "FCz", "FC2", "C1", "Cz", "C2"]
MASTOIDS = ["M1", "M2"]
PUPIL = ["R-Dia-X-(mm)", "R-Dia-Y-(mm)"]

EEG_BAND = (0.1, 30.0)
EEG_TMIN, EEG_TMAX = -0.200, 0.600
N1_WIN, P2_WIN = (0.080, 0.130), (0.170, 0.230)
EEG_REJECT_UV = 150.0

HR_TMIN, HR_TMAX = -3.0, 8.0
HR_BASE, HR_RESP = (-3.0, 0.0), (1.0, 5.0)
HR_FS = 4.0

PUP_TMIN, PUP_TMAX = -1.0, 4.0
PUP_BASE, PUP_RESP = (-1.0, 0.0), (0.5, 2.5)
PUP_MAX_BLINK = 0.30

MIN_CUES = 10


def instantaneous_hr(raw: mne.io.BaseRaw) -> tuple[np.ndarray, float] | None:
    """R 峰 → 4 Hz 瞬时心率序列（从 t=0 起）。质量不过关返回 None。"""
    ekg = raw.copy().pick(["EKG"])
    try:
        ev, _, _ = mne.preprocessing.find_ecg_events(
            ekg, ch_name="EKG", l_freq=5, h_freq=35, verbose=False)
    except Exception:                                        # noqa: BLE001
        return None
    t = ev[:, 0] / raw.info["sfreq"] - raw.first_time
    t = np.sort(t[(t >= 0) & (t <= raw.times[-1])])
    if len(t) < 60:
        return None
    rr = np.diff(t)
    keep = (rr > 0.30) & (rr < 2.0)                          # 生理可能区间
    if keep.mean() < 0.90:                                   # 过多不可能间期 → 检测坏了
        return None
    mid, hr = t[:-1][keep] + rr[keep] / 2, 60.0 / rr[keep]
    if not (40 <= np.median(hr) <= 120):
        return None
    grid = np.arange(0, raw.times[-1], 1 / HR_FS)
    f = interp1d(mid, hr, kind="linear", bounds_error=False,
                 fill_value=(hr[0], hr[-1]))
    return f(grid), HR_FS


def pupil_series(raw: mne.io.BaseRaw) -> tuple[np.ndarray, np.ndarray] | None:
    """瞳孔直径（X/Y 均值，mm），眨眼段线性插值。返回 (直径, 眨眼掩码)。"""
    have = [c for c in PUPIL if c in raw.ch_names]
    if not have:
        return None
    v = (raw.get_data(picks=have) * 1e6).mean(axis=0)   # MNE 按 V 读入，实为 mm
    bad = (v <= 0.5) | (v > 12.0)                       # 眨眼/丢失；人眼 2–9 mm
    if bad.mean() > 0.5 or (~bad).sum() < 100:
        return None
    idx = np.arange(len(v))
    v = v.copy()
    v[bad] = np.interp(idx[bad], idx[~bad], v[~bad])
    return v, bad


def epoch_1d(x: np.ndarray, onsets: np.ndarray, tmin: float, tmax: float,
             fs: float) -> tuple[np.ndarray, np.ndarray]:
    """把**从 t=0 起、等间隔 fs** 的一维序列按 onsets 切成 epoch。"""
    lag = np.arange(int(round(tmin * fs)), int(round(tmax * fs)))
    out = np.full((len(onsets), len(lag)), np.nan)
    for i, o in enumerate(onsets):
        k = int(round(o * fs)) + lag
        if k[0] < 0 or k[-1] >= len(x):
            continue
        out[i] = x[k]
    return lag / fs, out


def _measure(x: np.ndarray, onsets: np.ndarray, fs: float, tmin: float,
             tmax: float, base: tuple[float, float], resp: tuple[float, float],
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """基线校正 → 响应窗均值。返回 (逐试次值, 时间轴, 平均波形)。"""
    tt, ep = epoch_1d(x, onsets, tmin, tmax, fs)
    b = np.nanmean(ep[:, (tt >= base[0]) & (tt < base[1])], axis=1, keepdims=True)
    ep = ep - b
    val = np.nanmean(ep[:, (tt >= resp[0]) & (tt < resp[1])], axis=1)
    return val, tt, np.nanmean(ep, axis=0)


def process_run(f: Path, rng: np.random.Generator
                ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """单个 run → (逐试次测量, 平均波形)。真实臂与随机锚点臂共用一次读取。"""
    ev_tsv = f.parent / f.name.replace("_eeg.set", "_events.tsv")
    if not ev_tsv.exists():
        return None
    E = pd.read_csv(ev_tsv, sep="\t")
    cues = E.loc[E.trial_type == "cue", "onset"].to_numpy(float)
    if len(cues) < MIN_CUES:
        return None

    raw = mne.io.read_raw_eeglab(f, preload=True)
    sf, dur = raw.info["sfreq"], raw.times[-1]
    pad_lo = -min(HR_TMIN, PUP_TMIN, EEG_TMIN)
    pad_hi = max(HR_TMAX, PUP_TMAX, EEG_TMAX)

    real = cues[(cues > pad_lo) & (cues < dur - pad_hi)]
    if len(real) < MIN_CUES:
        return None
    # 阴性对照：同样数量的随机锚点，与事件无固定相位关系
    neg = rng.uniform(pad_lo, dur - pad_hi, size=len(real))

    sid = f.name.split("_")[0]
    task = [p for p in f.stem.split("_") if p.startswith("task-")][0][5:]
    run_id = f.stem.split("_run-")[1].split("_")[0]

    # ── 三条通路的连续序列，只算一次，两臂共用 ────────────────────────
    series: dict[str, tuple] = {}

    roi = [c for c in ROI if c in raw.ch_names]
    mast = [c for c in MASTOIDS if c in raw.ch_names]
    if len(roi) >= 6 and mast:
        eeg = raw.copy().pick(roi + mast)
        eeg.set_eeg_reference(mast, verbose=False)     # 听觉 ERP 的标准参考
        eeg.filter(*EEG_BAND, verbose=False)
        series["eeg"] = (eeg.get_data(picks=roi).mean(axis=0) * 1e6, sf,
                         EEG_TMIN, EEG_TMAX, (EEG_TMIN, 0.0), N1_WIN)

    hr = instantaneous_hr(raw) if "EKG" in raw.ch_names else None
    if hr is not None:
        series["hr"] = (hr[0], HR_FS, HR_TMIN, HR_TMAX, HR_BASE, HR_RESP)

    pup = pupil_series(raw)
    if pup is not None:
        pv = mne.filter.filter_data(pup[0][None, :], sf, None, 4.0,
                                    verbose=False)[0]
        series["pupil"] = (pv, sf, PUP_TMIN, PUP_TMAX, PUP_BASE, PUP_RESP)

    rows, waves = [], []
    for arm, ons in (("real", real), ("neg", neg)):
        d = pd.DataFrame({"trial": np.arange(len(ons)), "onset": ons,
                          "subject": sid, "task": task, "run": run_id,
                          "arm": arm, "eeg_p2n1": np.nan})
        for mod in ("eeg", "hr", "pupil"):
            d[mod] = np.nan
            if mod not in series:
                continue
            x, fs_m, tmin, tmax, base, resp = series[mod]
            val, tt, wave = _measure(x, ons, fs_m, tmin, tmax, base, resp)

            if mod == "eeg":
                _, ep = epoch_1d(x, ons, tmin, tmax, fs_m)
                b = np.nanmean(ep[:, (tt >= tmin) & (tt < 0)], axis=1,
                               keepdims=True)
                ep = ep - b
                bad = np.nanmax(np.abs(ep), axis=1) > EEG_REJECT_UV
                n1 = np.nanmean(ep[:, (tt >= N1_WIN[0]) & (tt < N1_WIN[1])], axis=1)
                p2 = np.nanmean(ep[:, (tt >= P2_WIN[0]) & (tt < P2_WIN[1])], axis=1)
                n1[bad] = p2[bad] = np.nan
                val = n1                       # 主测量：N1 单窗均值
                d["eeg_p2n1"] = p2 - n1        # 预设测量，一并保留
                wave = np.nanmean(ep[~bad], axis=0) if (~bad).any() else wave

            if mod == "pupil":
                _, epb = epoch_1d(pup[1].astype(float), ons, tmin, tmax, fs_m)
                val = val.copy()
                val[np.nanmean(epb, axis=1) > PUP_MAX_BLINK] = np.nan

            d[mod] = val
            waves.append(pd.DataFrame({"subject": sid, "task": task,
                                       "run": run_id, "arm": arm, "mod": mod,
                                       "t": tt, "v": wave}))
        rows.append(d)
    return pd.concat(rows, ignore_index=True), (
        pd.concat(waves, ignore_index=True) if waves else pd.DataFrame())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 个 run")
    args = ap.parse_args()

    files = sorted(RAW.rglob("*_eeg.set"))
    if args.limit:
        files = files[: args.limit]
    print(f"待处理 {len(files)} 个 run\n")

    CACHE.mkdir(parents=True, exist_ok=True)
    run = RunRecord("69_ds003690_extract", SEED)
    out, wav, failed, done = [], [], [], 0
    for i, f in enumerate(files, 1):
        ct = CACHE / f"{f.stem}_trials.parquet"
        cw = CACHE / f"{f.stem}_waves.parquet"
        if ct.exists():
            out.append(pd.read_parquet(ct))
            if cw.exists():
                wav.append(pd.read_parquet(cw))
            continue
        # 每个 run 用自己的确定性种子，因此断点续跑不改变任何已算结果。
        # 用 crc32 而非内建 hash() —— 后者对字符串逐进程加盐，不可复现。
        rng = np.random.default_rng([SEED, zlib.crc32(f.stem.encode())])
        try:
            r = process_run(f, rng)
        except Exception as e:                               # noqa: BLE001
            failed.append((f.name, f"{type(e).__name__}: {e}"))
            r = None
        if r is not None:
            r[0].to_parquet(ct, index=False)
            out.append(r[0])
            if len(r[1]):
                r[1].to_parquet(cw, index=False)
                wav.append(r[1])
        done += 1
        if done % 10 == 0 or i == len(files):
            print(f"  {i}/{len(files)}   {sum(len(x) for x in out)} 试次   "
                  f"失败 {len(failed)}", flush=True)

    D = pd.concat(out, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    D.to_parquet(OUT, index=False)
    if wav:
        pd.concat(wav, ignore_index=True).to_parquet(OUT_WAVE, index=False)

    real = D[D.arm == "real"]
    print(f"\n写出 {len(D)} 行 → {OUT}")
    print(f"  受试 {real.subject.nunique()}   真实臂 {len(real)} 试次")
    for m in ("eeg", "hr", "pupil"):
        per = real.groupby("subject")[m].apply(lambda s: s.notna().sum())
        print(f"  {m:<6} 有效 {real[m].notna().sum():>6}   "
              f"每人中位 {per.median():.0f}   最少 {per.min():.0f}")

    run.log_metric("n_subjects", int(real.subject.nunique()))
    run.log_metric("n_trials_real", int(len(real)))
    for m in ("eeg", "hr", "pupil"):
        run.log_metric(f"n_valid_{m}", int(real[m].notna().sum()))
    run.log_metric("n_failed_runs", len(failed))
    run.write()
    for name, why in failed[:10]:
        print(f"  失败 {name}  {why}")


if __name__ == "__main__":
    main()
