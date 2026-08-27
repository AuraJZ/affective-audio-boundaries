r"""段内动态：歌曲**内部**的声学起伏能不能预测皮电起伏。

    .venv/Scripts/python scripts/82_within_song_dynamics.py

## 这是另一个问题，不是把同一个问题重试一遍

`74` 问的是「更热闹的歌是不是引起更多 SCR」—— 整段 40 秒平均成一个数，
794 首歌之间比。结论：歌曲间的皮电差异可靠（信度 0.246），
但 161 维声学特征解释不了（ρ = 0.03，p = 0.26–0.52）。

但 **SCR 是对离散事件的相位性反应**。把 40 秒平均成一个数，
正好抹掉了它最该出现的地方。段内的问法是：

> **响的、突变的瞬间，会不会引起皮电反应？**

两个结论互不替代。若段内能预测而段间不能，说明声学驱动的是**瞬时唤起**，
而歌曲之间那个稳定的差异另有来源（熟悉度、歌词、喜好 —— 非波形属性）。

## 顺带：这也是个体层面问题的最佳机会 ⭐

段间每人只有 18 个观测，`77` 实测该量级下检验不出个体差异。
段内每人是 **18 首 × 约 80 帧**。即便计入自相关（有效样本远少于帧数），
量级也高一个数量级。**个体层面的问题在这里第一次有可能被检验。**

## 零分布必须是错配零分布 🔴

段内的皮电与声学都有强自相关和慢趋势。直接求相关会得到巨大的虚假值 ——
两条各自平滑的曲线本来就容易相关。

**错配零分布**：拿 A 歌的声学配 B 歌的皮电。它保留两侧全部的自相关结构、
全部的慢趋势、全部的帧数，只破坏「这段声学对应这段皮电」这一件事。
这是本设计里唯一能把真实耦合与自相关伪影分开的东西。

## 先验固定的处理

    声学帧      0.5 s；响度(RMS dB)、谱通量、起音强度、谱质心
    皮电        相位成分（0.05 Hz 高通），跨该歌听众取均值
    卷积        声学预测量先与标准 SCR 冲激响应卷积
                Bateman 双指数，τ1 = 0.75 s、τ2 = 2.0 s（Benedek & Kaernbach 2010）
                —— 皮电反应有 1–4 s 潜伏期，不卷积就是在错误的时间点上比
    去趋势      两侧各自去线性趋势，避免共同的会话内衰减制造相关
"""

from __future__ import annotations

import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import false_discovery_control, spearmanr

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")

PM = REPO_ROOT / "data" / "raw" / "PMEmo" / "PMEmo2019"
FEAT = REPO_ROOT / "data" / "features"
CACHE = FEAT / "pmemo_frames.parquet"
OUT = REPO_ROOT / "reports" / "source_data" / "within_song_dynamics.csv"
SEED = 20260731

FRAME_S = 0.5                 # 帧长
EDA_FS = 50.0
PHASIC_HZ = 0.05
SR = 22050
ACOUSTIC = ["rms_db", "flux", "onset", "centroid"]
TAU1, TAU2 = 0.75, 2.0        # Bateman SCR 冲激响应
MIN_FRAMES = 40               # 少于 20 s 的段不用
MIN_LISTENERS = 8
N_PERM_MEAN = 500             # 群体层面：每次置换要算 len(obs) 个相关
N_PERM_INDIV = 300            # 个体层面：每次置换要算该听众的 len(o) 个相关
ALPHA = 0.05


def scr_kernel(dt: float, dur: float = 12.0) -> np.ndarray:
    """标准 SCR 冲激响应（Bateman 双指数），归一化到单位面积。"""
    t = np.arange(0, dur, dt)
    h = np.exp(-t / TAU2) - np.exp(-t / TAU1)
    h[h < 0] = 0
    return h / (h.sum() + 1e-12)


def detrend(x: np.ndarray) -> np.ndarray:
    t = np.arange(len(x), dtype=float)
    return x - np.polyval(np.polyfit(t, x, 1), t)


def acoustic_frames(path: Path, n_frames: int) -> np.ndarray | None:
    """0.5 s 帧的四个声学量。返回 (n_frames, 4)。"""
    try:
        y, _ = librosa.load(str(path), sr=SR, mono=True)
    except Exception:                                        # noqa: BLE001
        return None
    hop = int(FRAME_S * SR)
    if len(y) < hop * MIN_FRAMES:
        return None
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    rms = librosa.feature.rms(S=S, frame_length=2048, hop_length=hop)[0]
    cen = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
    flux = np.r_[0.0, np.sqrt(((np.diff(S, axis=1)) ** 2).sum(0))]
    ons = librosa.onset.onset_strength(S=librosa.power_to_db(S ** 2), sr=SR,
                                       hop_length=hop)
    m = min(len(rms), len(cen), len(flux), len(ons), n_frames)
    if m < MIN_FRAMES:
        return None
    F = np.c_[20 * np.log10(rms[:m] + 1e-9), flux[:m], ons[:m], cen[:m]]
    return np.nan_to_num(F)


def eda_frames(f: Path) -> tuple[np.ndarray, list[str]] | None:
    """逐听众的相位性皮电，降采样到 0.5 s 帧。返回 (n_frames, n_listeners)。"""
    d = pd.read_csv(f)
    cols = [c for c in d.columns if c != "time(s)"]
    if len(cols) < MIN_LISTENERS:
        return None
    b, a = butter(2, PHASIC_HZ / (EDA_FS / 2), btype="high")
    step = int(FRAME_S * EDA_FS)
    out = []
    for c in cols:
        v = d[c].to_numpy(float)
        if not np.isfinite(v).all() or np.std(v) < 1e-6:
            out.append(None)
            continue
        ph = filtfilt(b, a, v)
        n = len(ph) // step
        out.append(ph[: n * step].reshape(n, step).mean(1))
    keep = [(c, v) for c, v in zip(cols, out) if v is not None]
    if len(keep) < MIN_LISTENERS:
        return None
    n = min(len(v) for _, v in keep)
    if n < MIN_FRAMES:
        return None
    return np.c_[tuple(v[:n] for _, v in keep)], [c for c, _ in keep]


def build_cache() -> pd.DataFrame:
    """逐 (歌, 帧) 的声学量 + 逐听众皮电。慢，只算一次。"""
    rows = []
    files = sorted(PM.glob("EDA/*_EDA.csv"))
    ker = scr_kernel(FRAME_S)
    for i, f in enumerate(files, 1):
        mid = int(f.name.split("_")[0])
        e = eda_frames(f)
        if e is None:
            continue
        E, listeners = e
        A = acoustic_frames(PM / "chorus" / f"{mid}.mp3", len(E))
        if A is None:
            continue
        n = min(len(A), len(E))
        A, E = A[:n], E[:n]
        # 声学量先卷积 SCR 冲激响应，再去趋势
        Ac = np.column_stack([detrend(np.convolve(A[:, j], ker)[:n])
                              for j in range(A.shape[1])])
        for j, c in enumerate(listeners):
            rows.append(pd.DataFrame({
                "musicId": mid, "listener": c, "frame": np.arange(n),
                "eda": detrend(E[:, j]),
                **{ACOUSTIC[k]: Ac[:, k] for k in range(len(ACOUSTIC))}}))
        if i % 100 == 0:
            print(f"    {i}/{len(files)}   {sum(len(r) for r in rows)} 帧行",
                  flush=True)
    return pd.concat(rows, ignore_index=True)


def paired_rho(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < MIN_FRAMES or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else np.nan


def perm_p(obs: float, null: np.ndarray) -> float:
    """双侧置换 p，以**零分布自身的中心**为参照。

    🔴 错配零分布并不以 0 为中心。实测：两条各自去趋势、各自强自相关的
    段内序列，即便毫无关系，也系统性地相关在约 −0.095（sd 0.010）。
    这正是本对照存在的理由 —— 没有它，−0.08 会被读成「响度抑制皮电」。

    但也因此，`|obs| ≥ |null|` 是错的检验：它拿观测值与 0 的距离，
    去比零分布与 0 的距离。首版就是这么写的，把一个偏离零分布中心
    +1.56 个标准差的观测报成了 p = 0.94。

    正确的量是**相对零分布中心的偏离**。
    """
    c = float(np.median(null))
    return float((1 + (np.abs(null - c) >= abs(obs - c)).sum()) / (1 + len(null)))


def null_of_means(units: dict, col: str, n_units: int,
                  rng: np.random.Generator, n_perm: int) -> np.ndarray:
    """错配零分布，**统计量与观测量同构**：每次抽 n_units 个错配对求均值。

    🔴 观测量是 n_units 首歌 ρ 的**均值**。若零分布用**单次**错配 ρ 的分布，
    两者的标准误差一个数量级 —— 均值窄、单次宽 —— 任何真实效应都会被压成
    不显著。这是个会造成**假阴性**的错误，必须让零分布也取同样多个的均值。
    """
    keys = list(units)
    out = []
    for _ in range(n_perm):
        vals = []
        for _ in range(n_units):
            a, b = rng.integers(0, len(keys), 2)
            if a == b:
                continue
            x = units[keys[a]][col].to_numpy()
            y = units[keys[b]]["eda"].to_numpy()
            k = min(len(x), len(y))
            v = paired_rho(x[:k], y[:k])
            if np.isfinite(v):
                vals.append(v)
        if vals:
            out.append(float(np.mean(vals)))
    return np.array(out)


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("82_within_song_dynamics", SEED)

    print("=" * 82)
    print("[1/3] 帧级数据")
    if CACHE.exists():
        D = pd.read_parquet(CACHE)
        print(f"  复用缓存 {CACHE.name}")
    else:
        D = build_cache()
        D.to_parquet(CACHE, index=False)
    print(f"  {len(D)} 帧行   歌 {D.musicId.nunique()}   听众 {D.listener.nunique()}")

    # ══ 2. 群体层面：跨听众平均后的段内耦合 ═══════════════════════════
    print("\n" + "=" * 82)
    print("[2/3] 群体层面：段内声学起伏 → 跨听众平均的皮电起伏")
    print("      零分布 = 错配（A 歌声学 配 B 歌皮电），保留全部自相关\n")
    song = (D.groupby(["musicId", "frame"])
            .agg(eda=("eda", "mean"), **{c: (c, "first") for c in ACOUSTIC})
            .reset_index())
    per_song = {m: g.sort_values("frame") for m, g in song.groupby("musicId")}
    ids = list(per_song)

    rows = []
    for c in ACOUSTIC:
        obs = np.array([paired_rho(per_song[m][c].to_numpy(),
                                   per_song[m]["eda"].to_numpy())
                        for m in ids])
        obs = obs[np.isfinite(obs)]
        # 错配零分布，每次同样取 len(obs) 个错配对的均值（与观测量同构）。
        # 每次要算 len(obs) 个相关，故置换次数取 N_PERM_MEAN 而非 N_PERM。
        null = null_of_means(per_song, c, len(obs), rng, N_PERM_MEAN)
        p = perm_p(float(obs.mean()), null)
        z = (obs.mean() - np.median(null)) / (null.std() + 1e-12)
        rows.append({"level": "group", "feature": c, "rho": float(obs.mean()),
                     "null_mean": float(null.mean()),
                     "null_sd": float(null.std()), "z": float(z),
                     "p": p, "n": len(obs)})
        print(f"  {c:<10} 段内 ρ = {obs.mean():+.4f}   "
              f"错配零分布 {null.mean():+.4f} ± {null.std():.4f}   "
              f"z = {z:+.2f}   p = {p:.4f}   n = {len(obs)} 首", flush=True)

    # ══ 3. 个体层面：每人 18 首 × 约 80 帧 ═══════════════════════════
    print("\n" + "=" * 82)
    print("[3/3] 个体层面：这个人自己的皮电，跟不跟他听到的声学起伏")
    print("      零分布 = 该听众自己的错配（他听过的另一首歌的声学）\n")
    for c in ACOUSTIC:
        ps, ds = [], []
        for lid, g in D.groupby("listener"):
            bym = {m: gg.sort_values("frame") for m, gg in g.groupby("musicId")}
            if len(bym) < 8:
                continue
            o = np.array([paired_rho(v[c].to_numpy(), v["eda"].to_numpy())
                          for v in bym.values()])
            o = o[np.isfinite(o)]
            if len(o) < 8:
                continue
            # 同样用「均值的零分布」：每次抽 len(o) 个该听众自己的错配对
            nul = null_of_means(bym, c, len(o), rng, N_PERM_INDIV)
            if len(nul) < 50:
                continue
            ps.append(perm_p(float(o.mean()), nul))
            ds.append(o.mean())
        if not ps:
            continue
        q = false_discovery_control(np.array(ps), method="bh")
        rows.append({"level": "individual", "feature": c,
                     "rho": float(np.mean(ds)), "null_mean": np.nan,
                     "null_sd": np.nan,
                     "p": float(np.mean(np.array(ps) < ALPHA)),
                     "n": len(ps)})
        print(f"  {c:<10} α=.05 通过 {int((np.array(ps)<ALPHA).sum())}/{len(ps)} 人"
              f"   BH-FDR 通过 {int((q<ALPHA).sum())}"
              f"   逐人 ρ 中位 {np.median(ds):+.4f}", flush=True)
        run.log_metric(f"indiv_pass_{c}", int((np.array(ps) < ALPHA).sum()))
        run.log_metric(f"indiv_pass_fdr_{c}", int((q < ALPHA).sum()))
        run.log_metric(f"indiv_n_{c}", len(ps))

    R = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT, index=False, encoding="utf-8")
    for _, x in R[R.level == "group"].iterrows():
        run.log_metric(f"group_rho_{x.feature}", float(x.rho))
        run.log_metric(f"group_p_{x.feature}", float(x.p))
    run.write()

    print("\n" + "=" * 82)
    print("读法：段内与段间是两个问题，结论不互相替代。")
    print("  段内阳性 + 段间阴性 → 声学驱动**瞬时唤起**，")
    print("                        而歌曲间那个稳定差异另有来源（非波形属性）。")
    print("  两者皆阴性         → 在本数据的时间分辨率下，声学与皮电无耦合。")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
