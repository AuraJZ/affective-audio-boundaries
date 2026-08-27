r"""CORAL 那 +0.198 是真的，还是 λ 调出来的。

    .venv/Scripts/python scripts/81_coral_robustness.py

## 为什么必须先做这个

`80` 给出：Layer A 跨域 ρ 从 0.035（无适配）升到 0.232（CORAL），
提升 6.6 倍。这个数若成立，论文核心声称「域之墙」必须大幅改写。

**但那次 CORAL 的 λ = 1.0 是加在未标准化的原始特征上的。**
122 维手工描述符的量纲差着好几个数量级（rms 在 0–1，spec_centroid 在千赫兹），
λ = 1.0 对某些维度是巨大的正则、对另一些几乎为零。
在这种情况下报一个「6.6 倍提升」，无异于报一个偶然的调参结果。

## 三条检验

| | 检验什么 |
|---|---|
| **λ 扫描** | 增益是否只在某个 λ 上出现 —— 出现即为调参伪影 |
| **先标准化再 CORAL** | 去掉量纲问题后增益是否还在 |
| **随机对齐对照** | 用**另一个语料**的协方差去着色。若随机对齐也一样有效，<br>说明增益来自「白化」本身而非「对齐到目标域」 |

第三条最关键。CORAL 做了两件事：白化源域、再按目标域着色。
若只是白化就够了（即用任意协方差着色都行），那么这不是领域自适应，
而是特征预处理 —— 结论完全不同。

## 判据先写在这里

- 增益在 λ 跨 4 个数量级上稳定，且标准化后仍在 → **真实**
- 随机对齐给出同等增益 → 不是自适应，是白化；须如此表述
- 只在个别 λ 上出现 → 调参伪影，撤回
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")

FEAT = REPO_ROOT / "data" / "features"
OUT = REPO_ROOT / "reports" / "source_data" / "coral_robustness.csv"
SEED = 20260731
CORPORA = {"reg_deam": "music", "reg_pmemo": "music",
           "reg_soundtracks": "music", "reg_emo_mix": "ambient"}
LAMBDAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# 特征列以 DEAM 为准显式取 —— 与 `40`/`55`/`59`/`60` 同一口径。
# Soundtracks 的表里带着 energy/tension 等 9 个标签列，
# 任何「排除已知元数据」式的写法都会把它们收进来。
def deam_cols() -> list[str]:
    from soundml.regression import feature_cols
    return feature_cols(pd.read_parquet(FEAT / "layer_a_reg_deam_ln.parquet"))


def load(corpus: str, cols: list[str]):
    D = pd.read_parquet(FEAT / f"layer_a_{corpus}_ln.parquet")
    miss = [c for c in cols if c not in D.columns]
    assert not miss, f"{corpus} 缺 {miss[:5]}"
    X = np.nan_to_num(D[cols].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    return X, D["arousal"].to_numpy(float), D["group"].to_numpy()


def coral(Xs, Xt, lam, standardise=False):
    if standardise:
        mu, sd = Xs.mean(0), Xs.std(0) + 1e-9
        Xs, Xt = (Xs - mu) / sd, (Xt - mu) / sd
    d = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + lam * np.eye(d)
    Ct = np.cov(Xt, rowvar=False) + lam * np.eye(d)
    W = np.real(fractional_matrix_power(Cs, -0.5) @
                fractional_matrix_power(Ct, 0.5))
    return (Xs - Xs.mean(0)) @ W + Xt.mean(0), Xt


def fit_predict(Xtr, ytr, Xte):
    best, out = -np.inf, None
    for mk in (lambda: make_pipeline(StandardScaler(),
                                     RidgeCV(alphas=np.logspace(-2, 4, 25))),
               lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                             random_state=SEED, n_jobs=-1)):
        m = mk()
        m.fit(Xtr, ytr)
        s = spearmanr(m.predict(Xtr), ytr).statistic
        if s > best:
            best, out = s, m.predict(Xte)
    return out


def main() -> None:
    run = RunRecord("81_coral_robustness", SEED)
    cols = deam_cols()
    print(f"特征 {len(cols)} 维（以 DEAM 为准）\n")
    data = {c: load(c, cols) for c in CORPORA}

    pairs = [(s, t) for s in CORPORA for t in CORPORA
             if CORPORA[s] != CORPORA[t]]
    print(f"跨域对 {len(pairs)} 组：{[f'{s[4:]}→{t[4:]}' for s, t in pairs]}\n")

    rows = []
    print("=" * 82)
    print("[1] λ 扫描 —— 增益是否只在某个 λ 上出现")
    print(f"  {'λ':>8}  {'原始特征':>10}  {'先标准化':>10}")
    base = float(np.mean([spearmanr(
        fit_predict(data[s][0], data[s][1], data[t][0]), data[t][1]).statistic
        for s, t in pairs]))
    print(f"  {'无适配':>8}  {base:>10.3f}  {base:>10.3f}")
    for lam in LAMBDAS:
        vals = {}
        for std in (False, True):
            rs = []
            for s, t in pairs:
                A, B = coral(data[s][0], data[t][0], lam, standardise=std)
                rs.append(spearmanr(fit_predict(A, data[s][1], B),
                                    data[t][1]).statistic)
            vals[std] = float(np.mean(rs))
            rows.append({"test": "lambda_sweep", "lam": lam,
                         "standardised": std, "rho": vals[std]})
        print(f"  {lam:>8g}  {vals[False]:>10.3f}  {vals[True]:>10.3f}",
              flush=True)

    # ── 随机对齐对照 ─────────────────────────────────────────────────
    print("\n" + "=" * 82)
    print("[2] 随机对齐对照 —— 用**第三方语料**的协方差着色")
    print("    若同样有效，则增益来自白化而非对齐到目标域\n")
    for lam in (0.01, 1.0):
        rs_t, rs_o = [], []
        for s, t in pairs:
            others = [c for c in CORPORA if c not in (s, t)]
            A, B = coral(data[s][0], data[t][0], lam)
            rs_t.append(spearmanr(fit_predict(A, data[s][1], B),
                                  data[t][1]).statistic)
            # 用第三方语料的协方差着色，再去预测**同一个**目标域
            A2, _ = coral(data[s][0], data[others[0]][0], lam)
            rs_o.append(spearmanr(fit_predict(A2, data[s][1], data[t][0]),
                                  data[t][1]).statistic)
        print(f"  λ={lam:<6g} 对齐到目标域 {np.mean(rs_t):+.3f}   "
              f"对齐到第三方 {np.mean(rs_o):+.3f}   "
              f"差 {np.mean(rs_t)-np.mean(rs_o):+.3f}")
        rows.append({"test": "sham_alignment", "lam": lam,
                     "rho_target": float(np.mean(rs_t)),
                     "rho_sham": float(np.mean(rs_o))})
        run.log_metric(f"coral_target_lam{lam}", float(np.mean(rs_t)))
        run.log_metric(f"coral_sham_lam{lam}", float(np.mean(rs_o)))

    # ── 域内对照：CORAL 会不会伤害域内性能 ────────────────────────────
    print("\n" + "=" * 82)
    print("[3] 域内基准（GroupKFold-5）")
    wi = []
    for c in CORPORA:
        X, y, g = data[c]
        pred = np.full(len(y), np.nan)
        for tr, te in GroupKFold(5).split(X, y, g):
            pred[te] = fit_predict(X[tr], y[tr], X[te])
        r = spearmanr(pred, y).statistic
        wi.append(r)
        print(f"  {c:<18} ρ = {r:+.3f}")
    within = float(np.mean(wi))

    D = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT, index=False, encoding="utf-8")

    print("\n" + "=" * 82)
    # `standardised` 列在与 sham 行合并后成了 object 型（混入 NaN），
    # `~L.standardised` 会退化为按位取反，产出 -1/-2 当列名索引。
    L = D[D.test == "lambda_sweep"].copy()
    L["standardised"] = L["standardised"].astype(bool)
    raw = L.loc[~L.standardised, "rho"]
    std = L.loc[L.standardised, "rho"]
    print(f"无适配 {base:+.3f}   域内 {within:+.3f}")
    print(f"CORAL（原始特征）  λ 跨 {min(LAMBDAS)}–{max(LAMBDAS)}："
          f"{raw.min():+.3f} ~ {raw.max():+.3f}   中位 {raw.median():+.3f}")
    print(f"CORAL（先标准化）  同上：{std.min():+.3f} ~ {std.max():+.3f}   "
          f"中位 {std.median():+.3f}")
    stable = (raw.max() - raw.min()) < 0.10 and raw.median() - base > 0.05
    print(f"\n判定：{'✅ 增益跨 λ 稳定，非调参伪影' if stable else '⚠️ 增益随 λ 大幅变动 —— 须按最保守值报告'}")

    # 把增益拆成「白化」与「真正的对齐」两部分 —— 这是本脚本最要紧的一行
    S = D[D.test == "sham_alignment"]
    if len(S):
        tgt, sham = float(S.rho_target.mean()), float(S.rho_sham.mean())
        print("\n增益分解：")
        print(f"  无适配                     {base:+.3f}")
        print(f"  + 白化/重标度（任意协方差）  {sham:+.3f}   "
              f"（Δ {sham-base:+.3f}）← 特征预处理，非领域自适应")
        print(f"  + 对齐到真正的目标域        {tgt:+.3f}   "
              f"（Δ {tgt-sham:+.3f}）← 真正的自适应增益")
        print(f"  域内基准                   {within:+.3f}")
        print(f"\n  真正可归因于领域自适应的部分：全部增益的 "
              f"{(tgt-sham)/max(tgt-base,1e-9):.0%}")
        run.log_metric("gain_from_whitening", sham - base)
        run.log_metric("gain_from_true_alignment", tgt - sham)
    run.log_metric("no_adapt", base)
    run.log_metric("within", within)
    run.log_metric("coral_raw_median", float(raw.median()))
    run.log_metric("coral_std_median", float(std.median()))
    run.log_metric("coral_raw_spread", float(raw.max() - raw.min()))
    run.write()
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
