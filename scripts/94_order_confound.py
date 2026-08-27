r"""D2：歌曲级皮电信度 0.246 会不会是呈现顺序的伪影。

    .venv/Scripts/python scripts/94_order_confound.py

## 为什么这条必须做

皮电在会话内单调衰减。若「歌曲身份」与「第几首」混淆，
那 0.246 的一部分可能是习惯化而非音乐。

时长已排除（R² = 0.000–0.007，回归掉后信度不降，见 `74`）。
但 **PMEmo 没有发布任何呈现顺序信息** —— metadata 只有歌曲属性。
只能用代理，而代理的用法必须先想清楚。

## 直接「回归掉代理」是错的做法 🔴

最自然的想法是：拿 tonic 水平的组内秩当顺序代理，回归掉它再看信度。
**但 tonic 与三个结果变量来自同一条皮电信号。** 若 tonic 与 SCR 速率
因生理原因相关（唤醒高则两者都高），回归掉 tonic 就一并抹掉了真信号。
那样得到的「信度下降」说明不了任何事。

## 正确的问法：分配是否系统性

**顺序只有在「歌曲→位置」的分配跨听众系统性一致时，
才可能制造虚假的歌曲级信度。**

若每首歌被它的约 10 名听众在**随机位置**听到，位置带来的方差
在跨听众平均时被抵消，产生不了可复现的歌曲间差异 ——
无论习惯化本身多强。

所以主检验是：**同一首歌在不同听众那里的「位置」一致吗？**

    位置代理   该听众各首歌按原始 tonic 均值的组内秩，归一到 [0,1]
    检验       该秩在歌曲层面的 ICC（跨听众一致性）
    判据       ICC ≈ 0 → 分配实质随机 → 顺序**不可能**制造虚假信度 ✅
               ICC > 0 → 分配系统性 → 需要进一步处理

## 代理的一处不对称，恰好是保守方向

若刺激本身强烈驱动 tonic，则同一首歌在各听众处的秩会一致，
本检验会**误判为「分配系统性」** —— 那是假阳性，指向更严格的处理。
反向的错误（真有系统性却判成随机）不会发生。

## 三个检验

    ① 分配随机性     位置代理在歌曲层面的 ICC（主检验）
    ② 秩内分层信度   把每人的歌按秩分成前后两半，各自算歌曲级信度
    ③ 回归掉代理     保守上界；因循环性会**过度**扣除，只作参考下界
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")

FEAT = REPO_ROOT / "data" / "features"
PM = REPO_ROOT / "data" / "raw" / "PMEmo" / "PMEmo2019"
OUT = REPO_ROOT / "reports" / "source_data" / "order_confound.csv"
SEED = 20260802
EDA = ["scr_rate", "phasic_mean", "scl_slope"]
MIN_LISTENERS = 8
N_SPLIT = 40
N_PERM = 2000


def raw_tonic() -> pd.DataFrame:
    """逐 (歌, 听众) 的原始 tonic 水平 —— 位置代理的原料。

    注意：必须用**原始**皮电水平，不能用 `pmemo_eda_song_level.parquet`
    里那三个自参照指标 —— 后者已经把绝对水平去掉了，而绝对水平
    正是习惯化最直接的载体。
    """
    from scipy.signal import butter, filtfilt
    rows = []
    for f in sorted((PM / "EDA").glob("*_EDA.csv")):
        mid = int(f.name.split("_")[0])
        d = pd.read_csv(f)
        b, a = butter(2, 0.05 / (50.0 / 2), btype="low")   # 紧张性成分
        for c in d.columns:
            if c == "time(s)":
                continue
            v = d[c].to_numpy(float)
            if not np.isfinite(v).all() or np.std(v) < 1e-6:
                continue
            rows.append({"musicId": mid, "listener": c,
                         "tonic": float(np.mean(filtfilt(b, a, v)))})
    return pd.DataFrame(rows)


def icc_oneway(v: np.ndarray, g: np.ndarray) -> float:
    df = pd.DataFrame({"v": v, "g": g}).dropna()
    if df.g.nunique() < 3 or len(df) < 6:
        return np.nan
    grp = df.groupby("g").v
    n_i, m_i = grp.size().to_numpy(), grp.mean().to_numpy()
    k = len(n_i)
    msb = float((n_i * (m_i - df.v.mean()) ** 2).sum() / (k - 1))
    ssw = float(((df.v - df.g.map(grp.mean())) ** 2).sum())
    dfw = len(df) - k
    if dfw <= 0:
        return np.nan
    msw = ssw / dfw
    n0 = (n_i.sum() - (n_i ** 2).sum() / n_i.sum()) / (k - 1)
    den = msb + (n0 - 1) * msw
    return float((msb - msw) / den) if den > 0 else np.nan


def split_half(D: pd.DataFrame, col: str, rng, n=N_SPLIT) -> float:
    rs = []
    for _ in range(n):
        a, b = [], []
        for _, g in D.groupby("musicId"):
            v = g[col].dropna().to_numpy().copy()
            if len(v) < MIN_LISTENERS:
                continue
            rng.shuffle(v)
            h = len(v) // 2
            a.append(v[:h].mean())
            b.append(v[h:2 * h].mean())
        if len(a) < 30:
            continue
        r = spearmanr(a, b).statistic
        if np.isfinite(r):
            rs.append(2 * r / (1 + r))
    return float(np.mean(rs)) if rs else np.nan


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("94_order_confound", SEED)

    cache = FEAT / "pmemo_raw_tonic.parquet"
    if cache.exists():
        Tn = pd.read_parquet(cache)
    else:
        print("提取原始 tonic 水平（约 3 分钟）…")
        Tn = raw_tonic()
        Tn.to_parquet(cache, index=False)
    print(f"原始 tonic：{len(Tn)} 个「人×歌」\n")

    # 位置代理：该听众各首歌按 tonic 的组内秩，归一到 [0,1]
    Tn["pos"] = Tn.groupby("listener").tonic.rank(pct=True)

    T = pd.read_parquet(FEAT / "pmemo_eda_song_level.parquet")
    for m in EDA:
        g = T.groupby("listener")[m]
        T[m] = (T[m] - g.transform("mean")) / (g.transform("std") + 1e-9)
    D = T.merge(Tn[["musicId", "listener", "pos", "tonic"]],
                on=["musicId", "listener"], how="inner")
    print(f"合并后 {len(D)} 个观测   歌 {D.musicId.nunique()}   "
          f"听众 {D.listener.nunique()}\n")

    rows = []
    # ── ① 主检验：分配是否随机 ───────────────────────────────────────
    print("=" * 86)
    print("[1/3] 主检验：同一首歌在不同听众那里的「位置」一致吗")
    print("      顺序只有在分配系统性时才可能制造虚假的歌曲级信度\n")
    obs = icc_oneway(D["pos"].to_numpy(float), D["musicId"].to_numpy())
    null = np.array([icc_oneway(rng.permutation(D["pos"].to_numpy(float)),
                                D["musicId"].to_numpy())
                     for _ in range(200)])
    null = null[np.isfinite(null)]
    p = float((1 + (null >= obs).sum()) / (1 + len(null)))
    print(f"  位置代理的歌曲级 ICC = {obs:+.4f}   "
          f"打乱零分布 {null.mean():+.4f} ± {null.std():.4f}   p = {p:.4f}")
    rows.append({"test": "assignment_icc", "value": obs, "p": p})
    run.log_metric("position_icc", float(obs))
    run.log_metric("position_icc_p", p)
    random_assign = p > 0.05
    print(f"  → {'✅ 分配实质随机 —— 顺序**不可能**制造虚假信度' if random_assign else '⚠️ 分配呈系统性，需看后两项'}")

    # ── ② 秩内分层 ───────────────────────────────────────────────────
    print("\n" + "=" * 86)
    print("[2/3] 按位置代理分层：前半段与后半段各自的歌曲级信度")
    print("      若两半都保住，信度就不是位置造成的\n")
    base = {m: split_half(D, m, rng) for m in EDA}
    # 🔴 tonic 在会话内**递减** ⇒ 低秩 = 会话**后**段。首版把标签写反了。
    back = D[D["pos"] <= 0.5]      # 低 tonic = 会话后段
    front = D[D["pos"] > 0.5]      # 高 tonic = 会话前段
    print(f"  {'指标':<14}{'全体':>9}{'会话前段':>10}{'会话后段':>10}")
    for m in EDA:
        f_, b_ = split_half(front, m, rng, 20), split_half(back, m, rng, 20)
        print(f"  {m:<14}{base[m]:>+9.3f}{f_:>+10.3f}{b_:>+10.3f}")
        rows.append({"test": "stratified", "measure": m, "all": base[m],
                     "late": f_, "early": b_})   # 键名沿用，值已按修正后归位
        run.log_metric(f"reliability_session_front_{m}", f_)
        run.log_metric(f"reliability_session_back_{m}", b_)

    # ── ③ 回归掉代理（保守下界）──────────────────────────────────────
    print("\n" + "=" * 86)
    print("[3/3] 回归掉位置代理 —— **保守下界**")
    print("      tonic 与结果变量同源，此法会连真信号一并扣除，只作参考\n")
    for m in EDA:
        x = D["pos"].to_numpy(float)
        y = D[m].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        Z = np.c_[np.ones(ok.sum()), x[ok]]
        D.loc[ok, m + "_r"] = y[ok] - Z @ np.linalg.lstsq(Z, y[ok],
                                                          rcond=None)[0]
        v = split_half(D, m + "_r", rng, 20)
        print(f"  {m:<14} 原始 {base[m]:+.3f}   回归后 {v:+.3f}   "
              f"损失 {base[m]-v:+.3f}")
        rows.append({"test": "regressed", "measure": m, "before": base[m],
                     "after": v})
        run.log_metric(f"reliability_after_regress_{m}", v)

    # ── ④ 直接量：位置**单独**能制造多少信度 ⭐ ────────────────────────
    #
    # 前三项都是间接推断。真正要回答的是一个数：
    # 「若把测量换成一个**纯粹由位置决定**的量，它的歌曲级信度是多少？」
    #
    # 做法：把每个测量对位置回归，取**拟合值**作为代理测量
    # （它按构造只含位置信息，不含任何歌曲内容），再算其歌曲级信度。
    # 这不循环 —— 我们不是在从真实测量里扣东西，而是在问
    # 「位置这一条信息，其本身的上限是多少」。
    print("\n" + "=" * 86)
    print("[4/4] ⭐ 位置**单独**能制造多少歌曲级信度")
    print("      代理测量 = 测量对位置回归的拟合值（按构造只含位置信息）\n")
    print(f"  {'指标':<14}{'真实信度':>10}{'纯位置代理':>12}{'占比':>9}")
    for m in EDA:
        x = D["pos"].to_numpy(float)
        y = D[m].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        Z = np.c_[np.ones(ok.sum()), x[ok]]
        beta = np.linalg.lstsq(Z, y[ok], rcond=None)[0]
        D.loc[ok, m + "_posonly"] = Z @ beta          # 只含位置的代理
        v = split_half(D, m + "_posonly", rng, 20)
        frac = v / base[m] if base[m] > 0 else np.nan
        print(f"  {m:<14}{base[m]:>+10.3f}{v:>+12.3f}{frac:>9.1%}")
        rows.append({"test": "position_only", "measure": m,
                     "real": base[m], "position_only": v, "fraction": frac})
        run.log_metric(f"position_only_reliability_{m}", float(v))
        run.log_metric(f"position_only_fraction_{m}", float(frac))

    # Spearman–Brown：分层把听众砍半，信度本就该降，须校正后再比
    print("\n  分层比较的 Spearman–Brown 校正"
          "（分层把每歌听众从约 10 砍到约 5）：")
    print(f"  {'指标':<14}{'r(10人)':>9}{'r(5人) 预期':>13}"
          f"{'会话前段':>10}{'会话后段':>10}")
    for m in EDA:
        r10 = base[m]
        r1 = r10 / max(10 - 9 * r10, 1e-9)
        r5 = 5 * r1 / (1 + 4 * r1)
        st = next((x for x in rows
                   if x.get("test") == "stratified" and x.get("measure") == m), {})
        # `pos` 是 tonic 的组内秩；tonic 会话内递减 ⇒ 高 pos = 会话前段
        front, back = st.get("late", np.nan), st.get("early", np.nan)
        print(f"  {m:<14}{r10:>+9.3f}{r5:>+13.3f}{front:>+10.3f}{back:>+10.3f}")
        run.log_metric(f"sb_expected_half_{m}", float(r5))

    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")
    print("\n" + "=" * 86)
    if random_assign:
        print("结论：位置分配实质随机 → **顺序不可能制造虚假的歌曲级信度**。")
        print("      0.246 不是习惯化伪影。后两项作为佐证一并报告。")
    else:
        print("结论：位置分配呈系统性 → 须以分层结果为准，并在正文写明该限制。")
    run.write()
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
