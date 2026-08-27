r"""PMEmo 皮电的**个体层面**：不同的人，声学→生理的映射是不是真的不一样。

    .venv/Scripts/python scripts/76_pmemo_individual_level.py

## 产品的问题是个体的，不是群体的

`74` 在群体层面给出可解读的阴性：歌曲间的皮电差异可靠（分半信度 0.246），
但 161 维声学特征预测不了它（CV ρ ≈ 0.03，与打乱标签无异）。

**那个结果回答不了产品的问题。** 按人校准的前提不是「存在一条人人适用的
声学→生理规律」，恰恰相反 —— 是**每个人的规律不同**。若人人相同，
按人校准就是多余的；若人人不同，群体层面的阴性正是预期结果。

## 三个层次的问法，功效差别极大 🔴

| 问法 | 每人 15–20 首够不够 |
|---|---|
| ① 用声学预测**这个人**的皮电 | ❌ `70` 实测：自主神经在 20 次观测下检出率 18–30%。**注定不足** |
| ② **不同的人斜率是否不同** | ✅ 跨 401 人汇总，功效充足 |
| ③ 群体层面（`74` 已做） | ✅ 但回答的不是产品的问题 |

① 仍然做并如实报告 —— 但它的阴性**必须**配着功效说明一起读，
否则就是本项目反复警告的那种误读。**主结果是 ②。**

## ② 怎么做：逐人斜率的分半信度

对每个先验固定的声学描述符 d：

1. 把每名听者的 15–20 首随机对半；
2. 在 A 半内求该听者的 (皮电 ~ d) 回归斜率，在 B 半内同样求一次；
3. 跨 401 人，看 slope_A 与 slope_B 是否相关。

**零假设「人人斜率相同」下，该相关必为 0** —— 因为此时人际间没有真实差异，
两次估计只差噪声，跨人相关的期望为零。相关显著为正 ⟺ 存在**可复现的
个体差异**，即按人校准有据。

这个检验不依赖混合模型的分布假设，且零分布可由**打乱人身份**精确构造。

## 预测变量：先验固定 5 个

不用 161 维（那会把多重比较搞爆，且斜率不可解释）。选这 5 个的依据
来自本项目已有结果与文献，与本次结果无关：

    rms_mean            声压级 —— 唤醒度最经典的声学驱动
    onset_env_mean      事件密度 —— `Fig.4` 的反事实干预证明它能推动模型
    chroma_flux_mean    音色/和声变化率 —— 同上，且是干预中效应最大的一个
    roughness_pl_mean   感觉粗糙度 —— `Fig.3` 显示它承载「紧张」
    mode_score          调式 —— `Fig.3` 显示它承载「快乐」

5 描述符 × 3 皮电指标 = 15 个检验，BH-FDR 校正。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import false_discovery_control, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")

PM = REPO_ROOT / "data" / "raw" / "PMEmo" / "PMEmo2019"
FEAT = REPO_ROOT / "data" / "features"
SEED = 20260730

MEASURES = ["scr_rate", "phasic_mean", "scl_slope"]
DESCRIPTORS = ["rms_mean", "onset_env_mean", "chroma_flux_mean",
               "roughness_pl_mean", "mode_score"]
MIN_SONGS = 14            # 对半后每半至少 7 首才谈得上估斜率
N_SPLIT = 40              # 分半重复次数
N_PERM = 2000
N_PERM_SUBJ = 150
ALPHA = 0.05


def load() -> pd.DataFrame:
    """逐 (听者, 歌) 的皮电指标 + 该歌的声学描述符。"""
    T = pd.read_parquet(FEAT / "pmemo_eda_song_level.parquet")
    fa = pd.read_parquet(FEAT / "layer_a_reg_pmemo_ln.parquet")
    fd = pd.read_parquet(FEAT / "layer_d_reg_pmemo_ln.parquet")
    for f in (fa, fd):
        f["musicId"] = (f["clip_id"].astype(str)
                        .str.removeprefix("pmemo_").astype(int))
    F = fa.merge(fd.drop(columns=["clip_id"]), on="musicId", suffixes=("", "_d"))
    D = T.merge(F, on="musicId", how="inner")
    # 皮电指标做**被试内** z 分：去个体基线与量程，保留被试内的相对差异
    for m in MEASURES:
        g = D.groupby("listener")[m]
        D[m] = (D[m] - g.transform("mean")) / (g.transform("std") + 1e-9)
    # 声学描述符全局标准化，使斜率跨描述符可比
    for d in DESCRIPTORS:
        D[d] = (D[d] - D[d].mean()) / (D[d].std() + 1e-9)
    return D


def slope(x: np.ndarray, y: np.ndarray) -> float:
    """最小二乘斜率。方差为零时返回 nan。"""
    if len(x) < 4 or np.std(x) < 1e-9:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def split_half_slope(D: pd.DataFrame, desc: str, meas: str,
                     rng: np.random.Generator) -> tuple[float, float, int]:
    """逐人斜率的分半一致性。返回 (相关, 置换 p, 纳入人数)。"""
    rs = []
    for _ in range(N_SPLIT):
        a, b = [], []
        for _, g in D.groupby("listener"):
            g = g.dropna(subset=[desc, meas])
            if len(g) < MIN_SONGS:
                continue
            idx = rng.permutation(len(g))
            h = len(idx) // 2
            x = g[desc].to_numpy(); y = g[meas].to_numpy()
            sa, sb = slope(x[idx[:h]], y[idx[:h]]), slope(x[idx[h:]], y[idx[h:]])
            if np.isfinite(sa) and np.isfinite(sb):
                a.append(sa); b.append(sb)
        if len(a) >= 50:
            rs.append(spearmanr(a, b).statistic)
    if not rs:
        return np.nan, np.nan, 0
    r = float(np.mean(rs))

    # 零分布：打乱人身份 —— 把 B 半的斜率随机指派给别人
    null = []
    for _ in range(N_PERM):
        null.append(spearmanr(a, rng.permutation(b)).statistic)
    p = float((1 + np.sum(np.array(null) >= r)) / (1 + len(null)))
    return r, p, len(a)


def personalisation_gain(D: pd.DataFrame, desc: str, meas: str,
                         rng: np.random.Generator) -> tuple[float, float, int]:
    """留一交叉：**这个人自己的斜率**是否比群体斜率更能预测他被留出的那首歌。

    这是产品的问题原样：个性化到底有没有用。功效也高于分半 ——
    斜率用 n−1 首估计，而分半只用 n/2 首。

    两臂共用**同一个个体截距**（数据已做被试内 z 分，截距约为 0），
    因此比较的纯粹是斜率，不掺截距差异。

    返回 (平均增益, 符号翻转 p, 纳入人数)。增益 > 0 表示个性化更准。
    """
    gains = []
    for lid, g in D.groupby("listener"):
        g = g.dropna(subset=[desc, meas])
        if len(g) < MIN_SONGS:
            continue
        x, y = g[desc].to_numpy(float), g[meas].to_numpy(float)
        # 群体斜率：**排除该听者本人**，否则是自我预测
        o = D[(D.listener != lid)].dropna(subset=[desc, meas])
        s_pop = slope(o[desc].to_numpy(float), o[meas].to_numpy(float))
        if not np.isfinite(s_pop):
            continue
        e_own, e_pop = [], []
        for j in range(len(x)):
            k = np.arange(len(x)) != j
            s_own = slope(x[k], y[k])
            if not np.isfinite(s_own):
                continue
            b = float(np.mean(y[k]))          # 两臂共用的个体截距
            e_own.append((y[j] - (b + s_own * (x[j] - np.mean(x[k])))) ** 2)
            e_pop.append((y[j] - (b + s_pop * (x[j] - np.mean(x[k])))) ** 2)
        if len(e_own) >= MIN_SONGS - 2:
            gains.append(float(np.mean(e_pop) - np.mean(e_own)))
    if len(gains) < 50:
        return np.nan, np.nan, len(gains)
    gv = np.array(gains)
    obs = float(gv.mean())
    signs = rng.integers(0, 2, (N_PERM, len(gv))) * 2 - 1
    null = signs @ gv / len(gv)
    p = float((1 + (null >= obs).sum()) / (1 + len(null)))
    return obs, p, len(gv)


def per_person_predict(D: pd.DataFrame, meas: str, feat_cols: list[str],
                       rng: np.random.Generator) -> tuple[int, int, float]:
    """① 用 161 维声学预测**这个人**的皮电。预期功效不足，仍如实报告。"""
    def one(args) -> float:
        lid, X, y, seed = args
        r2 = np.random.default_rng(seed)
        n_pc = min(3, len(y) - 2)          # 约 18 首样本，最多 3 个主成分

        def cv(yy):
            pred = np.full(len(yy), np.nan)
            for tr, te in KFold(4, shuffle=True, random_state=SEED).split(X):
                m = make_pipeline(StandardScaler(), PCA(n_pc),
                                  RidgeCV(alphas=np.logspace(-2, 4, 20)))
                m.fit(X[tr], yy[tr])
                pred[te] = m.predict(X[te])
            r = spearmanr(pred, yy).statistic
            return r if np.isfinite(r) else 0.0

        obs = cv(y)
        null = np.array([cv(r2.permutation(y)) for _ in range(N_PERM_SUBJ)])
        return (1 + (null >= obs).sum()) / (1 + len(null))

    jobs = []
    for i, (lid, g) in enumerate(D.groupby("listener")):
        g = g.dropna(subset=[meas])
        if len(g) < MIN_SONGS:
            continue
        jobs.append((lid, np.nan_to_num(g[feat_cols].to_numpy(float)),
                     g[meas].to_numpy(float), SEED + i))
    ps = Parallel(n_jobs=-1, batch_size=4)(delayed(one)(j) for j in jobs)
    if not ps:
        return 0, 0, np.nan
    q = false_discovery_control(np.array(ps), method="bh")
    return int((np.array(ps) < ALPHA).sum()), len(ps), float((q < ALPHA).sum())


def main() -> None:
    rng = np.random.default_rng(SEED)
    run = RunRecord("76_pmemo_individual_level", SEED)
    D = load()
    per = D.groupby("listener").size()
    print(f"听者 {D.listener.nunique()}   歌 {D.musicId.nunique()}   "
          f"观测 {len(D)}   每人歌数中位 {per.median():.0f}\n")

    # ══ ② 主结果：逐人斜率是否存在可复现的个体差异 ════════════════════
    print("=" * 84)
    print("[主结果] 不同的人，声学 → 皮电的斜率是不是真的不一样")
    print("         零假设「人人相同」下，逐人斜率的分半相关期望为 0\n")
    rows = []
    for meas in MEASURES:
        for desc in DESCRIPTORS:
            r, p, n = split_half_slope(D, desc, meas, rng)
            rows.append({"measure": meas, "descriptor": desc,
                         "split_half_r": r, "p": p, "n_listeners": n})
            print(f"  {meas:<12} {desc:<20} r = {r:+.3f}   p = {p:.4f}   "
                  f"n = {n}", flush=True)
    R = pd.DataFrame(rows)
    R["q"] = false_discovery_control(R.p.fillna(1).to_numpy(), method="bh")
    n_sig = int((R.q < ALPHA).sum())
    print(f"\n  BH-FDR 后显著 {n_sig}/{len(R)}")
    if n_sig:
        print("  显著项：")
        for _, x in R[R.q < ALPHA].sort_values("q").iterrows():
            print(f"    {x.measure:<12} {x.descriptor:<20} "
                  f"r = {x.split_half_r:+.3f}  q = {x.q:.4f}")
    R.to_csv(REPO_ROOT / "reports" / "source_data" /
             "pmemo_individual_slopes.csv", index=False, encoding="utf-8")
    run.log_metric("n_sig_individual_slopes", n_sig)
    run.log_metric("n_tests", len(R))

    # ══ ②b 个性化增益：留一交叉 ══════════════════════════════════════
    print("\n" + "=" * 84)
    print("[主结果 b] 个性化有没有用：自己的斜率 vs 群体斜率，留一预测误差之差")
    print("           增益 > 0 表示用这个人自己的斜率更准\n")
    grows = []
    for meas in MEASURES:
        for desc in DESCRIPTORS:
            gain, p, n = personalisation_gain(D, desc, meas, rng)
            grows.append({"measure": meas, "descriptor": desc,
                          "gain": gain, "p": p, "n_listeners": n})
            print(f"  {meas:<12} {desc:<20} 增益 = {gain:+.4f}   "
                  f"p = {p:.4f}   n = {n}", flush=True)
    G = pd.DataFrame(grows)
    G["q"] = false_discovery_control(G.p.fillna(1).to_numpy(), method="bh")
    ng = int((G.q < ALPHA).sum())
    print(f"\n  BH-FDR 后显著 {ng}/{len(G)}")
    for _, x in G[G.q < ALPHA].sort_values("q").iterrows():
        print(f"    {x.measure:<12} {x.descriptor:<20} "
              f"增益 {x.gain:+.4f}  q = {x.q:.4f}")
    G.to_csv(REPO_ROOT / "reports" / "source_data" /
             "pmemo_personalisation_gain.csv", index=False, encoding="utf-8")
    run.log_metric("n_sig_personalisation_gain", ng)
    print("\n  ⚠️ 自己的斜率用 n−1≈17 首估计，噪声大 —— 这**偏向低估**增益。")
    print("     因此正增益是保守证据；零增益则含义不确定。")

    # ══ ① 逐人预测（预期功效不足，仍报告）════════════════════════════
    print("\n" + "=" * 84)
    print("[对照] 用 161 维声学预测单个人的皮电")
    print("       `70` 实测：自主神经通路在每人 20 次观测下检出率 18–30%")
    print("       → **这一栏的阴性主要反映功效，不构成证据**\n")
    NONFEAT = {"clip_id", "path", "arousal", "valence", "group", "source",
               "musicId", "listener", "duration", *MEASURES}
    fc = [c for c in D.columns if c.removesuffix("_d") not in NONFEAT
          and pd.api.types.is_numeric_dtype(D[c])]
    assert not [c for c in fc if c.removesuffix("_d") in
                {"arousal", "valence", *MEASURES}], "标签泄漏"
    print(f"  特征 {len(fc)} 维")
    for meas in MEASURES:
        k, n, kq = per_person_predict(D, meas, fc, rng)
        print(f"  {meas:<12} α=.05 通过 {k}/{n} 人   BH-FDR 通过 {kq:.0f}",
              flush=True)
        run.log_metric(f"per_person_pass_{meas}", k)
        run.log_metric(f"per_person_n_{meas}", n)

    run.write()
    print("\n" + "=" * 84)
    print("读法（依实际结果，不预设方向）：")
    print("  · 主结果若为阳性、对照为阴性 → 两者不矛盾：前者跨 401 人检验")
    print("    「斜率是否因人而异」功效充足，后者在每人 20 首内做多元预测先天不足。")
    print("  · 主结果亦为阴性 → **必须先跑 `77` 的灵敏度分析**再谈解读；")
    print("    「未发现个体差异」与「个体差异不存在」是两回事。")


if __name__ == "__main__":
    main()
