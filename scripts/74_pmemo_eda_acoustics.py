r"""PMEmo 皮电：声学特征能否预测**不同音乐引起的生理反应**。

    .venv/Scripts/python scripts/74_pmemo_eda_acoustics.py

## 这是本项目第一次真正做「声学 → 生理」

此前三个生理数据集各卡在一处：

| | 卡点 |
|---|---|
| ds002721 | 刺激编号↔音频对不上，声学这条线整条断掉 |
| BIRAFFE2 | 94 人无一人通过个体阳性对照 |
| ds003690 | 只有四个纯音，**没有声学变异** |

PMEmo 的 EDA 部分我们一直没用过 —— 而它恰好没有上述任一问题：
`musicId` **同时索引音频与皮电**，794 首真实音乐，401 名听者，
每首约 10 人，共 7,960 个「人 × 歌」观测。
且这批歌的 161 维声学特征本项目早已提取。

## 测量必须自参照 🔴

探查发现两件事，直接约束设计：

1. **EDA 与音频等长**（中位差 0.1 s）→ **没有刺激前基线**
2. 元数据无听歌顺序 → 无法直接校正会话内的皮电衰减

因此不能用绝对水平（会被「这首歌排在第几个」主导）。
改用三个**自参照**指标：

    scr_rate     每分钟皮肤电反应次数（相位成分的峰计数）
    phasic_mean  相位成分绝对值均值
    scl_slope    紧张性成分在段内的斜率（µS/min）

再对每名听者在其 15–20 首内做 z 分，去掉个体基线与量程差异。
歌曲级取听者均值。若听歌顺序是随机分配的，顺序效应在约 10 名听者上被平均掉；
这一点无法验证（元数据未给），故在结果中标为限制。

## 阳性对照先行

**先问「歌曲间的生理差异本身可不可靠」，可靠了才谈预测。**
这正是 `step17` 判掉 ds002721 的那条判据：主观评分的刺激级信度 0.335，
156 个脑电测量最好的只有 0.092，无一通过 FDR。

对照用同一套流程算主观唤醒度标注的信度 —— 那是已知可靠的量，
若它在本流程下也算不出信度，说明流程本身坏了。

## 判据

信度设定了任何预测器的上限。若 EDA 的刺激级信度接近 0，
则「声学预测不了 EDA」不能解读为声学无关 —— 而是**被预测的量本身不存在**。
这个区分是本项目反复强调的 B2 原则。
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")

PM = REPO_ROOT / "data" / "raw" / "PMEmo" / "PMEmo2019"
FEAT = REPO_ROOT / "data" / "features"
OUT = FEAT / "pmemo_eda_song_level.parquet"
SEED = 20260730

FS = 50.0                    # EDA 采样率
LP_HZ = 1.0                  # 低通，去高频噪声
PHASIC_HZ = 0.05             # 高通，分离相位成分（Boucsein 2012 的常用界）
SCR_MIN_AMP = 0.01           # µS，标准最小可判反应幅度
SCR_MIN_GAP_S = 1.0          # 反应间最小间隔
MIN_DUR_S = 15.0             # 太短的段算不出速率
MIN_LISTENERS = 8
N_PERM = 1000
N_PERM_CV = 30               # 完整 CV 的置换：每次要跑 2 模型 × 5 折
MEASURES = ["scr_rate", "phasic_mean", "scl_slope"]


def _filt(x: np.ndarray, cut: float, kind: str) -> np.ndarray:
    b, a = butter(2, cut / (FS / 2), btype=kind)
    return filtfilt(b, a, x)


def eda_measures(v: np.ndarray, dur_s: float) -> dict[str, float] | None:
    """单个 (歌, 听者) 的三个自参照指标。"""
    if not np.isfinite(v).all() or len(v) < FS * MIN_DUR_S or np.std(v) < 1e-6:
        return None
    x = _filt(v, LP_HZ, "low")
    phasic = _filt(x, PHASIC_HZ, "high")
    tonic = x - phasic
    pk, _ = find_peaks(phasic, height=SCR_MIN_AMP,
                       distance=int(SCR_MIN_GAP_S * FS))
    t = np.arange(len(x)) / FS
    slope = float(np.polyfit(t, tonic, 1)[0] * 60.0)         # µS/min
    return {"scr_rate": len(pk) / (dur_s / 60.0),
            "phasic_mean": float(np.mean(np.abs(phasic))),
            "scl_slope": slope}


def load_trials() -> pd.DataFrame:
    """遍历 794 个 EDA 文件 → 逐 (歌, 听者) 的指标表。"""
    rows = []
    files = sorted(PM.glob("EDA/*_EDA.csv"))
    for i, f in enumerate(files, 1):
        mid = int(f.name.split("_")[0])
        d = pd.read_csv(f)
        t = d["time(s)"].to_numpy(float)
        dur = float(t[-1])
        for col in d.columns:
            if col == "time(s)":
                continue
            m = eda_measures(d[col].to_numpy(float), dur)
            if m:
                rows.append({"musicId": mid, "listener": col,
                             "duration": dur, **m})
        if i % 200 == 0:
            print(f"    {i}/{len(files)}", flush=True)
    return pd.DataFrame(rows)


def within_listener_z(D: pd.DataFrame) -> pd.DataFrame:
    """每名听者在其 15–20 首内做 z 分：去个体基线与量程。"""
    D = D.copy()
    for m in MEASURES:
        g = D.groupby("listener")[m]
        D[m] = (D[m] - g.transform("mean")) / (g.transform("std") + 1e-9)
    return D


def split_half(D: pd.DataFrame, col: str, unit: str, rng) -> float:
    """刺激级信度：把每首歌的听者随机对半，两半的歌曲均值求相关，S-B 校正。"""
    a, b = [], []
    for _, g in D.groupby(unit):
        v = g[col].dropna().to_numpy().copy()
        if len(v) < MIN_LISTENERS:
            continue
        rng.shuffle(v)
        h = len(v) // 2
        a.append(v[:h].mean())
        b.append(v[h:2 * h].mean())
    if len(a) < 30:
        return np.nan
    r = spearmanr(a, b).statistic
    return float(2 * r / (1 + r)) if r > -1 else np.nan


def cv_predict(X, y, groups, rng) -> float:
    """GroupKFold-5（按艺人分组，避免同一艺人跨折泄漏），报两模型中较好者。"""
    best = -np.inf
    for mk in (lambda: make_pipeline(StandardScaler(),
                                     RidgeCV(alphas=np.logspace(-2, 4, 25))),
               lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                             random_state=SEED, n_jobs=-1)):
        pred = np.full(len(y), np.nan)
        for tr, te in GroupKFold(5).split(X, y, groups):
            m = mk()
            m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        r = spearmanr(pred, y).statistic
        best = max(best, r if np.isfinite(r) else -np.inf)
    return float(best)


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("74_pmemo_eda_acoustics", SEED)

    print("=" * 80)
    print("[1/4] 抽取逐 (歌, 听者) 的皮电指标")
    if OUT.exists():
        T = pd.read_parquet(OUT)
        print(f"  复用缓存 {OUT.name}")
    else:
        T = load_trials()
        T.to_parquet(OUT, index=False)
    print(f"  {len(T)} 个「人 × 歌」观测   歌 {T.musicId.nunique()}   "
          f"听者 {T.listener.nunique()}")
    Z = within_listener_z(T)

    # ── 2. 阳性对照：刺激级信度 ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("[2/4] 阳性对照：歌曲间的差异本身可不可靠")
    print("      （信度设定任何预测器的上限；信度≈0 时阴性不可解读）")
    rel = {}
    for m in MEASURES:
        vals = [split_half(Z, m, "musicId", rng) for _ in range(20)]
        rel[m] = float(np.nanmean(vals))
        print(f"  {m:<14} 分半信度 {rel[m]:+.3f}")
        run.log_metric(f"reliability_{m}", rel[m])

    # 对照量：同一流程作用于主观唤醒度标注（已知可靠）
    A = pd.read_csv(PM / "annotations" / "static_annotations.csv")
    Astd = pd.read_csv(PM / "annotations" / "static_annotations_std.csv")
    print(f"  ── 对照 ── 主观标注 {len(A)} 首，"
          f"唤醒度跨听者 s.d. 中位 {Astd['Arousal(std)'].median():.3f}")
    print("      标注给的是均值而非逐听者值，无法用同一分半流程；")
    print("      其信度已在 step-annotation-ceiling 中单独测得。")

    # ── 3. 声学 → 生理 ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("[3/4] 声学特征 → 歌曲级皮电反应")
    song = Z.groupby("musicId")[MEASURES].mean()
    song["n_listeners"] = Z.groupby("musicId").size()
    song = song[song.n_listeners >= MIN_LISTENERS]

    fa = pd.read_parquet(FEAT / "layer_a_reg_pmemo_ln.parquet")
    fd = pd.read_parquet(FEAT / "layer_d_reg_pmemo_ln.parquet")
    # `clip_id` 形如 "pmemo_1000"；EDA 文件名给的是裸 musicId
    for f in (fa, fd):
        f["musicId"] = (f["clip_id"].astype(str)
                        .str.removeprefix("pmemo_").astype(int))
    print(f"  Layer A {fa.shape}   Layer D {fd.shape}")

    # 🔴 标签泄漏：特征表自带 arousal / valence 两个**数值型标签列**。
    # 「排除已知元数据」这种写法本项目已经放进去过两次标签 ——
    # 改为**列出所有非特征列**并在最后断言维度。
    NONFEAT = {"clip_id", "path", "arousal", "valence", "group", "source",
               "musicId", "corpus", "split", "label", "n_listeners", "artist",
               *MEASURES}
    F = fa.merge(fd.drop(columns=["clip_id"]), on="musicId", suffixes=("", "_d"))
    M = song.reset_index().merge(F, on="musicId", how="inner")
    print(f"  声学与皮电对上的歌曲 {len(M)} 首")
    if len(M) < 100:
        print("  ⚠️ 对上的太少，无法继续 —— 检查 musicId 对齐")
        run.write()
        return

    meta = pd.read_csv(PM / "metadata.csv")
    M = M.merge(meta[["musicId", "artist"]], on="musicId", how="left")
    M["artist"] = M["artist"].fillna("unknown_" + M.musicId.astype(str))

    fc = [c for c in M.columns
          if c.removesuffix("_d") not in NONFEAT
          and pd.api.types.is_numeric_dtype(M[c])]
    leaked = [c for c in fc if c.removesuffix("_d") in
              {"arousal", "valence", *MEASURES}]
    assert not leaked, f"标签泄漏进特征矩阵：{leaked}"
    X = M[fc].to_numpy(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    groups = M["artist"].to_numpy()
    print(f"  特征 {X.shape[1]} 维   艺人组 {len(set(groups))} 个")
    # Layer A 122 + Layer D 39 = 161。偏离说明列筛选出了问题，宁可炸掉。
    assert 150 <= X.shape[1] <= 175, f"特征维度异常 {X.shape[1]}（应约 161）"

    # 可达上限不是信度本身，而是它的平方根 🔴
    #
    # 经典衰减公式：r_obs = r_true · √(r_xx · r_yy)。预测器视为无误差（r_yy=1）时，
    # 完美预测器能达到的最大相关是 **√r_xx**，不是 r_xx。
    # 我第一版按 obs/r_xx 算「达上限百分比」，把分母取小了 ——
    # 那会把结果说得比实际好，是往对自己有利的方向错。改正后比例更低。
    print("  可达上限 = √(分半信度)（经典衰减校正）")
    for m in MEASURES:
        y = M[m].to_numpy(float)
        obs = cv_predict(X, y, groups, rng)
        null = np.array([cv_predict(X, rng.permutation(y), groups, rng)
                         for _ in range(N_PERM_CV)])
        p = float((1 + (null >= obs).sum()) / (1 + len(null)))
        ceil = np.sqrt(rel[m]) if rel[m] > 0 else np.nan
        frac = obs / ceil if np.isfinite(ceil) else np.nan
        tail = (f"达上限 {frac:.0%}" if np.isfinite(frac) else "（上限不可估）")
        print(f"  {m:<14} CV ρ = {obs:+.3f}   "
              f"置换零分布 {null.mean():+.3f} (p = {p:.3f})   "
              f"可达上限 {ceil:+.3f}   {tail}", flush=True)
        run.log_metric(f"cv_rho_{m}", obs)
        run.log_metric(f"null_rho_{m}", float(null.mean()))
        run.log_metric(f"perm_p_{m}", p)
        run.log_metric(f"attenuation_ceiling_{m}", float(ceil))

    # ── 4. 个体层面：每人 20 首够不够 ─────────────────────────────────
    print("\n" + "=" * 80)
    print("[4/4] 个体层面")
    per = Z.groupby("listener").size()
    print(f"  每人歌曲数：中位 {per.median():.0f}   最多 {per.max()}")
    print("  `70` 的实测曲线：自主神经通路在每人 20 次观测下检出率约 18–30%。")
    print("  → 本数据集**不足以**做个体层面的声学—生理关联，"
          "该问题在此仍为不可检验。")
    run.log_metric("obs_per_person_median", float(per.median()))
    run.write()


if __name__ == "__main__":
    main()
