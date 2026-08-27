"""PLAN I-3：音乐 → 环境声的跨域测试。

    uv run python scripts/08_cross_domain.py --train deam --test esc50

## 为什么这是阻塞项

规则草案给出两条内部一致但反直觉的规则：

    spec_flatness_p50  越低越助眠   （谱平坦度低 = 音调性强、噪声性弱）
    spec_contrast*_p10 越高越助眠   （谱对比度高 = 峰谷差大 = 同样是音调性强）

**这与「粉噪声助眠」的常识相反** —— 粉噪/白噪的谱平坦度是**高**的。

推测原因是语料：DEAM 是**音乐**。音乐域内低唤醒的是 ambient、慢板古典
（音调性强），高唤醒的是失真吉他、镲片（噪声性强）。
所以「音调性 = 平静」在音乐里成立。

**但产品实际播放的是雨声、粉噪、白噪 —— 正好落在噪声性那一端。**
规则很可能在跨域时翻转。

在验证跨域方向之前，这批规则不能用于环境声打分，也不能被当作一般结论。

## 做法与其局限

ESC-50 是**语义类别代理标签**（自然/水声 vs 警报/机械声），不是实测助眠效应。
因此本测试只能回答方向性问题，不能给定量结论：

  - AUC 显著高于 0.5  → 规则在环境声上仍有区分力
  - AUC 显著低于 0.5  → **规则方向翻转**（这才是真正要警惕的结果）
  - AUC ≈ 0.5         → 规则不迁移

同时逐特征比较「音乐域」与「环境声域」的类间差方向是否一致 ——
这比总体 AUC 更直接地回答"规则会不会翻转"。
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from soundml.data import load_joined
from soundml.modeling import META_COLS, SEED, build_models, score
from soundml.provenance import REPO_ROOT, RunRecord

FEAT_DIR = REPO_ROOT / "data" / "features"


def load_esc50() -> tuple[pd.DataFrame, list[str]]:
    """ESC-50 只提过 Layer A，但必须用**响度归一化版**，
    否则与训练集的预处理不一致，方向翻转与否会被混淆。"""
    df = pd.read_parquet(FEAT_DIR / "layer_a_esc50_ln.parquet")
    return df, [c for c in df.columns if c not in META_COLS]


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """正类 vs 负类的标准化均差；符号即"类间差方向"。"""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="deam")
    args = ap.parse_args()

    tr, reps = load_joined(args.train)
    esc, esc_cols = load_esc50()

    # ESC-50 只提过 Layer A，且是**未归一化**版本 —— 交集只能取 Layer A 的列
    shared = [c for c in reps["A"] if c in esc_cols]

    Xtr = SimpleImputer(strategy="median").fit_transform(tr[shared].to_numpy(dtype=np.float64))
    ytr = tr["label"].to_numpy()
    Xte = SimpleImputer(strategy="median").fit_transform(esc[shared].to_numpy(dtype=np.float64))
    yte = esc["label"].to_numpy()

    with RunRecord(
        "cross_domain_music_to_ambient", seed=SEED,
        params={"train": args.train, "test": "esc50", "n_shared_features": len(shared),
                "caveat": "ESC-50 为语义类别代理标签，非实测助眠效应，"
                          "故本测试只回答方向性问题，不给定量结论"},
    ) as run:
        run.log_metric("n_train", {"total": len(ytr), "pos": int(ytr.sum())})
        run.log_metric("n_test", {"total": len(yte), "pos": int(yte.sum())})

        # ---------- 整体迁移能力 ----------
        cross = {}
        for algo in ("RF", "XGBoost"):
            m = build_models()[algo].fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
            cross[algo] = score(yte, p)
            cross[algo]["auc_flipped"] = float(roc_auc_score(yte, -p))
        run.log_metric("cross_domain", cross)

        # ---------- 逐特征：类间差方向是否一致 ----------
        rows = []
        for f in shared:
            j = shared.index(f)
            d_mus = cohens_d(Xtr[ytr == 1, j], Xtr[ytr == 0, j])
            d_amb = cohens_d(Xte[yte == 1, j], Xte[yte == 0, j])
            rows.append({"feature": f, "d_music": d_mus, "d_ambient": d_amb,
                         "same_direction": bool(np.sign(d_mus) == np.sign(d_amb))})
        eff = pd.DataFrame(rows)
        eff["abs_d_music"] = eff["d_music"].abs()
        eff = eff.sort_values("abs_d_music", ascending=False)
        eff.to_csv(run.artifact_path("direction_agreement.csv"), index=False, encoding="utf-8")

        agree = float(eff["same_direction"].mean())
        run.log_metric("direction_agreement_all", agree)

        # 规则草案里用到的特征，单独看
        rule_feats = ["spec_flatness_p50", "spec_contrast1_p10", "spec_contrast2_p10",
                      "spec_contrast3_dmean", "onset_rate", "onset_env_mean",
                      "zcr_p50", "zcr_p90", "spec_rolloff_mean", "spec_rolloff_p10", "mfcc1_mean"]
        sub = eff[eff["feature"].isin(rule_feats)]
        run.log_metric("direction_agreement_rule_features", float(sub["same_direction"].mean()))
        run.log_metric("rule_feature_directions", sub.set_index("feature")[
            ["d_music", "d_ambient", "same_direction"]].round(3).to_dict("index"))

        print("=" * 72)
        print(f"run_id: {run.run_id}")
        print(f"共享特征: {len(shared)}   训练 {len(ytr)} 段音乐   测试 {len(yte)} 段环境声\n")
        print("--- 整体迁移 ---")
        for algo, s in cross.items():
            print(f"  {algo:<9} AUC={s['auc']:.4f}  (方向反转后 {s['auc_flipped']:.4f})")
        print(f"\n--- 类间差方向一致率 ---")
        print(f"  全部 {len(shared)} 个共享特征 : {agree:.1%}")
        print(f"  规则用到的 {len(sub)} 个特征  : {sub['same_direction'].mean():.1%}")
        print("\n--- 规则特征逐条（d>0 表示助眠类的该特征值更高）---")
        print(f"{'特征':<24}{'音乐域 d':>10}{'环境声 d':>11}  一致")
        for r in sub.itertuples():
            mark = "✓" if r.same_direction else "✗ 翻转"
            print(f"{r.feature:<24}{r.d_music:>10.3f}{r.d_ambient:>11.3f}  {mark}")


if __name__ == "__main__":
    main()
