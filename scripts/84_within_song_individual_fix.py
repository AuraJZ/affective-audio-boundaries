r"""段内·个体层面：换掉失效的零分布，并标定它能测出多强的耦合。

    .venv/Scripts/python scripts/84_within_song_individual_fix.py

## `82` 的个体层面检验失效了，不是阴性 🔴

四个声学量的通过率是 2/401、0/401、1/401、2/401，
而 α=.05 下的随机期望是 **20.1**。偏低到 p = 2.6e-09。

**若检验有效，即便毫无效应也该有约 20 人偶然通过。** 得到 0–2 人
说明检验本身被推向保守，其零结果不能读作「个体层面无耦合」。

病因：每名听众只有 18 首歌，错配零分布只能在这 18 首里抽
（18×17 = 306 种配对），而观测值用的是**同一批 EDA 序列**。
零分布与观测值共享该听众的全部特性，观测值必然落在零分布中心附近。

## 哪个零分布对：由**功效曲线的单调性**裁决，不由论证裁决 🔴

首轮实跑（主零分布设为「全库错配」）给出：

    λ = 0（真实数据，无注入）   82/150 = 55%
    λ = 0.05                   76/150 = 51%
    λ = 0.10                   71/150 = 47%   ← 比 λ=0 还低
    λ = 0.20                   96/150 = 64%
    λ = 0.35                  133/150 = 89%

**功效曲线必须从假阳性率单调上升。** λ=0 给 55%、λ=0.10 反而给 47%，
在有效检验里不可能发生 —— 这直接证明该零分布的**假阳性率是 55% 而非 5%**，
那 82/150 全是假阳性。

病因：全库错配要把声学与 EDA 截到较短的一方，于是零分布常用 EDA 的**前缀**，
其相关分布中心落在 ρ ≈ +0.05，而真实观测在 −0.02 —— 距离反而更大。
听众自己的 18 首长度分布天然匹配，不存在这个问题。

**结论：长度分布匹配比抽样池大小更要紧。** 主零分布改为「听众内错配」。

## 三个零分布

**① 听众内错配**（`82` 用的）：从该听众自己的 18 首里抽错配声学。
**太保守** —— 只有 306 种配对，且与观测共享同一批 EDA 序列，
观测值必然落在零分布中心附近。通过率 0–2/401，随机期望 20.1。

**② 相位随机化**（Theiler et al. 1992）：保留功率谱、随机化相位。
**太宽松** —— 它保留自相关，却**破坏真实包络的形状**。
群体层面的错配零分布中心在 **−0.095**（离 0 有 9.8 个标准差），
那来自真实音乐与真实皮电共有的特征性起落（去线性趋势去不掉曲率）；
相位随机化的替代序列没有这个形状，零分布回到 0 附近，
于是实测的 −0.020 就显得极端。首版实跑得到 47/150 通过、28 人过 FDR，
方向还是负的 —— **那是零分布欠设定造成的假阳性。**

**③ 全库错配**：从全部 717 首里抽错配声学。**长度不匹配** ——
截断使零分布用到 EDA 前缀，中心偏移，假阳性率 55%（见上）。

三个都算并列报出，且**都附 λ=0 的假阳性率**。
选哪个不由论证决定，由校准性决定。

## 同时标定功效

光换零分布还不够 —— 新检验若同样测不出，仍要分清「无耦合」与「测不到」。
做法与 `77` 一致：把真实的皮电序列当作噪声，**叠加**已知强度的声学成分

    eda_合成 = eda_真实 + λ · (sd_eda / sd_声学) · 声学

λ 即耦合强度。用真实皮电当噪声，保留了它的全部自相关与个体差异，
比任何合成噪声都贴近实际。对每个 λ 跑同一套检验，报检出率。

**输出形如**「本检验对 λ ≥ X 有 80% 把握；实测未达显著，
故段内个体耦合**若存在也不超过 X**」—— 可证伪的定量陈述。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, false_discovery_control, spearmanr

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")

FEAT = REPO_ROOT / "data" / "features"
CACHE = FEAT / "pmemo_frames.parquet"
OUT = REPO_ROOT / "reports" / "source_data" / "within_song_individual_fix.csv"
SEED = 20260731

FEATURE = "rms_db"            # 标定用最可能的驱动量；结论对其余更保守
MIN_FRAMES = 40
MIN_SONGS = 8
N_SURR = 300                  # 每首歌的相位随机化替代序列数
N_LISTENERS = 150             # 抽样以控制运行时间
LAMBDAS = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]
ALPHA = 0.05
# 主零分布由**校准性**选定（λ=0 时假阳性率是否等于 α），不由论证选定。
# 首轮用 "pool" 被功效曲线的非单调性否掉：其假阳性率为 55%。
MAIN_NULL = "self"


def phase_randomise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """相位随机化替代序列：幅度谱不变，相位随机，因而自相关完全一致。"""
    n = len(x)
    F = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, len(F))
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0                      # Nyquist 分量必须为实数
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=n) + x.mean()


def rho(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < MIN_FRAMES or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else np.nan


def test_listener(songs: list[tuple[np.ndarray, np.ndarray]],
                  pool: list[np.ndarray], rng: np.random.Generator,
                  null_kind: str) -> tuple[float, float]:
    """一名听众：观测的平均段内 ρ，及在指定零分布下的 p 值。

    `null_kind`：
        pool   从全库 717 首抽错配声学（**主检验**）
        phase  相位随机化替代序列
        self   从该听众自己的歌里抽错配声学（`82` 的做法）
    """
    obs = [rho(a, e) for a, e in songs]
    obs = [v for v in obs if np.isfinite(v)]
    if len(obs) < MIN_SONGS:
        return np.nan, np.nan
    o = float(np.mean(obs))

    own = [a for a, _ in songs]
    null = []
    for _ in range(N_SURR):
        vals = []
        for i, (a, e) in enumerate(songs):
            if null_kind == "phase":
                x = phase_randomise(a, rng)
            elif null_kind == "self":
                j = rng.integers(0, len(own))
                if j == i:
                    continue
                x = own[j]
            else:
                x = pool[rng.integers(0, len(pool))]
            k = min(len(x), len(e))
            v = rho(x[:k], e[:k])
            if np.isfinite(v):
                vals.append(v)
        if vals:
            null.append(float(np.mean(vals)))
    if len(null) < 50:
        return o, np.nan
    nu = np.array(null)
    c = float(np.median(nu))
    p = float((1 + (np.abs(nu - c) >= abs(o - c)).sum()) / (1 + len(nu)))
    return o, p


def collect(D: pd.DataFrame, lam: float, rng: np.random.Generator,
            pool: list[np.ndarray], null_kind: str = "pool") -> dict:
    """对给定耦合强度跑完整的逐个体检验。lam=0 即真实数据。"""
    ps, os_ = [], []
    for _, g in D.groupby("listener"):
        songs = []
        for _, s in g.groupby("musicId"):
            s = s.sort_values("frame")
            a = s[FEATURE].to_numpy(float)
            e = s["eda"].to_numpy(float)
            if len(a) < MIN_FRAMES:
                continue
            if lam > 0:
                # 真实皮电当噪声，叠加已知强度的声学成分
                sc = lam * (np.std(e) / (np.std(a) + 1e-12))
                e = e + sc * a
            songs.append((a, e))
        if len(songs) < MIN_SONGS:
            continue
        o, p = test_listener(songs, pool, rng, null_kind)
        if np.isfinite(p):
            ps.append(p)
            os_.append(o)
    ps = np.array(ps)
    if not len(ps):
        return {}
    q = false_discovery_control(ps, method="bh")
    return {"lam": lam, "null_kind": null_kind, "n": len(ps),
            "pass_alpha": int((ps < ALPHA).sum()),
            "pass_fdr": int((q < ALPHA).sum()),
            "rate": float((ps < ALPHA).mean()),
            "median_rho": float(np.median(os_))}


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("84_within_song_individual_fix", SEED)
    D = pd.read_parquet(CACHE)
    keep = rng.choice(D.listener.unique(),
                      min(N_LISTENERS, D.listener.nunique()), replace=False)
    D = D[D.listener.isin(keep)]
    # 全库声学池：全部歌曲的该特征序列，供「全库错配」零分布抽样
    pool = [g.sort_values("frame")[FEATURE].to_numpy(float)
            for _, g in D.groupby("musicId")]
    pool = [a for a in pool if len(a) >= MIN_FRAMES]
    print(f"标定量 {FEATURE}   听众 {D.listener.nunique()}   "
          f"歌 {D.musicId.nunique()}   帧行 {len(D)}   声学池 {len(pool)} 条\n")

    print("=" * 80)
    print("[1/2] 三个零分布并列跑真实数据 —— 结论取决于选哪个")
    rows = []
    for kind, label in (("pool", "全库错配（主）"), ("phase", "相位随机化"),
                        ("self", "听众内错配（82）")):
        r = collect(D, 0.0, rng, pool, kind)
        rows.append(r)
        exp = ALPHA * r["n"]
        bt = binomtest(r["pass_alpha"], r["n"], ALPHA).pvalue
        verdict = ("✅ 与随机水平相容" if bt > 0.01 else
                   ("⚠️ 高于随机 —— 零分布若欠设定即为假阳性"
                    if r["pass_alpha"] > exp else "⚠️ 低于随机 —— 检验过保守"))
        print(f"  {label:<16} α=.05 通过 {r['pass_alpha']:>3}/{r['n']}   "
              f"FDR {r['pass_fdr']:>3}   ρ 中位 {r['median_rho']:+.4f}   "
              f"期望 {exp:.1f}   {verdict}", flush=True)
        run.log_metric(f"real_pass_{kind}", r["pass_alpha"])
        run.log_metric(f"real_binom_p_{kind}", float(bt))
    real = next(r for r in rows if r["null_kind"] == MAIN_NULL)

    print("\n" + "=" * 80)
    print(f"[2/2] 注入已知强度的耦合，标定**主零分布（{MAIN_NULL}）**的功效")
    print("      λ=0 那一行即该零分布的假阳性率 —— 曲线必须由它单调上升")
    print("      （真实皮电当噪声，叠加 λ·(sd_eda/sd_声学)·声学）\n")
    for lam in LAMBDAS:
        if lam == 0:
            continue
        r = collect(D, lam, rng, pool, MAIN_NULL)
        if r:
            rows.append(r)
            print(f"  λ = {lam:.2f}   α=.05 通过 {r['pass_alpha']:>3}/{r['n']}"
                  f"   检出率 {r['rate']:.0%}   逐人 ρ 中位 {r['median_rho']:+.4f}",
                  flush=True)
            run.log_metric(f"power_lam{lam}", r["rate"])

    R = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT, index=False, encoding="utf-8")

    ok = R[(R.lam > 0) & (R.null_kind == MAIN_NULL) & (R.rate >= 0.80)]
    lim = float(ok.lam.min()) if len(ok) else np.nan
    print("\n" + "=" * 80)
    if np.isfinite(lim):
        rr = float(R.loc[(R.lam == lim) & (R.null_kind == MAIN_NULL),
                          "median_rho"].iloc[0])
        print(f"本检验对 λ ≥ {lim:.2f}（对应逐人 ρ ≈ {rr:+.3f}）有 80% 把握。")
        print(f"主零分布下实测通过 {real['pass_alpha']}/{real['n']}"
              f"（期望 {ALPHA*real['n']:.1f}）。")
        print(f"→ 段内的个体级声学—皮电耦合**若存在，其强度不超过 ρ ≈ {abs(rr):.3f}**。")
        run.log_metric("detectable_lambda_80pct", lim)
        run.log_metric("detectable_rho_80pct", rr)
    else:
        print("在所测范围内均未达 80% 把握 —— 即便换了零分布，")
        print("每人 18 首 × 约 80 帧 仍不足以做个体级段内检验。")
        print("这本身就是结论：**该设计回答不了这个问题**，不能读作无耦合。")
    run.write()
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
