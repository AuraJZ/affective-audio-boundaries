r"""C1 的前置：按窗口长度重算歌曲级皮电（保留歌曲间差异）。

    .venv/Scripts/python scripts/97_window_level_eda.py

## 为什么要单独做这一步 🔴

C1 要的是「每首只测 m 秒时，歌曲级测量有多准」。
最初想直接用 `pmemo_frames.parquet`（`82` 建的帧级缓存），**但那张表不能用**：

`82` 的 `build_cache` 对**每首歌**调了 `detrend()`，而去线性趋势包含去均值 ——
每个 (听众, 歌) 序列的均值**按构造恒为零**。实测歌曲级均值 SD = 0.000000。

那对 `82` 的段内分析完全正确（它要的就是段内起伏），
但拿它做**段间**分析等于把一堆零对声学回归。
C1 首版由此得出「4 首 × 5 段就够校准」，与 `92`/`93` 直接矛盾。

## 本脚本做什么

从原始 EDA 重算，对每个 (歌, 听众) × 每个窗口长度，
在**随机位置的连续窗**内算与 `74` 完全相同的三个自参照指标：

    scr_rate      每分钟皮肤电反应次数
    phasic_mean   相位成分绝对值均值
    scl_slope     紧张性成分斜率（µS/min）

**不去均值** —— 歌曲间差异正是要保留的东西。

窗口：2.5 / 5 / 10 / 20 / 37 秒（对应 C1 的 m = 5/10/20/40/74 帧）。
短于窗口的歌在该窗口下跳过。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")

PM = REPO_ROOT / "data" / "raw" / "PMEmo" / "PMEmo2019"
FEAT = REPO_ROOT / "data" / "features"
OUT = FEAT / "pmemo_window_level.parquet"
SEED = 20260802

FS = 50.0
LP_HZ, PHASIC_HZ = 1.0, 0.05
SCR_MIN_AMP, SCR_MIN_GAP_S = 0.01, 1.0
WINDOWS_S = [2.5, 5.0, 10.0, 20.0, 37.0]
N_DRAW = 3                      # 每个 (歌,人,窗长) 抽几个随机位置
MEASURES = ["scr_rate", "phasic_mean", "scl_slope"]


def _filt(x: np.ndarray, cut: float, kind: str) -> np.ndarray:
    b, a = butter(2, cut / (FS / 2), btype=kind)
    return filtfilt(b, a, x)


def measures(v: np.ndarray) -> dict[str, float] | None:
    """与 `74` 逐字相同的三个指标 —— 换脚本不换尺子。"""
    dur = len(v) / FS
    if not np.isfinite(v).all() or np.std(v) < 1e-6 or len(v) < FS * 2:
        return None
    x = _filt(v, LP_HZ, "low")
    phasic = _filt(x, PHASIC_HZ, "high")
    tonic = x - phasic
    pk, _ = find_peaks(phasic, height=SCR_MIN_AMP,
                       distance=int(SCR_MIN_GAP_S * FS))
    t = np.arange(len(x)) / FS
    return {"scr_rate": len(pk) / (dur / 60.0),
            "phasic_mean": float(np.mean(np.abs(phasic))),
            "scl_slope": float(np.polyfit(t, tonic, 1)[0] * 60.0)}


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("97_window_level_eda", SEED)
    files = sorted((PM / "EDA").glob("*_EDA.csv"))
    rows = []
    for i, f in enumerate(files, 1):
        mid = int(f.name.split("_")[0])
        d = pd.read_csv(f)
        cols = [c for c in d.columns if c != "time(s)"]
        n_all = len(d)
        for w in WINDOWS_S:
            nw = int(w * FS)
            if n_all < nw:
                continue
            for _ in range(N_DRAW):
                s0 = int(rng.integers(0, n_all - nw + 1))
                for c in cols:
                    m = measures(d[c].to_numpy(float)[s0:s0 + nw])
                    if m:
                        rows.append({"musicId": mid, "listener": c,
                                     "window_s": w, "start": s0, **m})
        if i % 150 == 0:
            print(f"  {i}/{len(files)}   {len(rows)} 行", flush=True)

    D = pd.DataFrame(rows)
    D.to_parquet(OUT, index=False)
    print(f"\n{len(D)} 行 → {OUT.name}")

    # 校验：歌曲间差异必须**没有**被去掉（这正是 `82` 那张表的问题）
    print("\n校验：歌曲级差异是否保留")
    for w, g in D.groupby("window_s"):
        sm = g.groupby(["listener", "musicId"])[MEASURES[1]].mean()
        z = sm.groupby(level=0).transform(
            lambda v: (v - v.mean()) / (v.std() + 1e-9))
        print(f"  窗 {w:>5.1f}s   歌曲级 SD = {sm.std():.6f}   "
              f"被试内 z 分后 SD = {z.std():.4f}   "
              f"{'✅' if sm.std() > 1e-9 else '🔴 仍为零'}")
        run.log_metric(f"song_sd_w{w}", float(sm.std()))
    full = D[D.window_s == max(WINDOWS_S)]
    print(f"\n最长窗覆盖：歌 {full.musicId.nunique()}   听众 {full.listener.nunique()}")
    run.log_metric("n_rows", len(D))
    run.write()


if __name__ == "__main__":
    main()
