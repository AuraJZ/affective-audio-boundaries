"""域之墙的三项混淆对照 —— 不依赖 IADS-E。

    uv run python scripts/19_domain_wall_controls.py

## 背景

主结果「跨域迁移为零」的最强质疑不是「环境声只有一个数据源」，
而是 **「域」与另外四件事完全共变**：

| 维度 | Emo-Soundscapes | DEAM / PMEmo |
|---|---|---|
| 片段时长 | 6 s | 45 s / 变长副歌 |
| 标注方式 | 众包排序转评分 | 直接 Likert 评分 |
| 标注人群 | 74 国 1182 人 | MediaEval / 中国被试 |
| 原生标度 | −1..1 | 1–9 / 0–1 |
| 内容 | 环境声 | 音乐 |

## 三项对照

**C1 时长对照** —— 把音乐语料截成 6 s（与环境声一致）后重跑跨域。
若墙仍在，时长被排除。这是四个混淆里最容易被质疑、也最容易排除的一个。

**C2 域内留一类交叉验证** —— Emo-Soundscapes 自带 Schafer 六分类。
留一类做测试，看域内跨内容能否泛化。若能，则跨域失败**特定发生在域边界**，
而不是「任何内容变化都会崩」。

**C3 环境声域的第二语料** —— ESC-50（本地已有，标签类型不同：语义代理）。
两个环境声语料 + 两种标签类型 + 同一结论 = 部分替代 IADS-E。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import N_SPLITS, SEED, build_regressors, feature_cols, reg_score

FEAT = REPO_ROOT / "data" / "features"


def load(tag: str) -> pd.DataFrame:
    return pd.read_parquet(FEAT / f"layer_a_{tag}.parquet")


def transfer(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str], algo="XGBoost") -> float:
    imp = SimpleImputer(strategy="median").fit(tr[cols].to_numpy(dtype=np.float64))
    m = build_regressors()[algo].fit(imp.transform(tr[cols].to_numpy(dtype=np.float64)),
                                     tr["arousal"].to_numpy())
    p = m.predict(imp.transform(te[cols].to_numpy(dtype=np.float64)))
    return reg_score(te["arousal"].to_numpy(), p)["spearman"]


def main() -> None:
    with RunRecord("domain_wall_controls", seed=SEED,
                   params={"controls": ["C1 duration", "C2 within-domain LOCO",
                                        "C3 second ambient corpus"]}) as run:
        out_rows = []

        # ================= C1 时长对照 =================
        emo = load("reg_emo_mix_ln")
        cols = feature_cols(emo)
        c1 = {}
        for music in ("reg_deam", "reg_pmemo"):
            full = load(f"{music}_ln")
            short = load(f"{music}_ln_6s")
            c1[f"{music}|30s"] = transfer(full, emo, cols)
            c1[f"{music}|6s"] = transfer(short, emo, cols)
            # 同域对照：截短后音乐↔音乐是否仍然可迁移（排除「截短毁了信号」）
            other = "reg_pmemo" if music == "reg_deam" else "reg_deam"
            c1[f"{music}->{other}|6s"] = transfer(short, load(f"{other}_ln_6s"), cols)
            c1[f"{music}->{other}|30s"] = transfer(full, load(f"{other}_ln"), cols)
            for k in (f"{music}|30s", f"{music}|6s"):
                out_rows.append({"control": "C1_duration", "setting": k,
                                 "kind": "cross_domain", "spearman": c1[k]})
            for k in (f"{music}->{other}|30s", f"{music}->{other}|6s"):
                out_rows.append({"control": "C1_duration", "setting": k,
                                 "kind": "same_domain", "spearman": c1[k]})
        run.log_metric("C1_duration", c1)
        print("  C1 时长对照完成", flush=True)

        # ============ C2 域内留一类（Schafer 六分类） ============
        # 只用 600 段原始录音：混音的 category 是 'mix'，没有真实类别
        orig = emo[emo["category"] != "mix"].copy()
        c2 = {}
        for cat in sorted(orig["category"].unique()):
            tr = orig[orig["category"] != cat]
            te = orig[orig["category"] == cat]
            if len(te) < 20:
                continue
            c2[cat] = {"n_test": int(len(te)), "spearman": transfer(tr, te, cols)}
            out_rows.append({"control": "C2_within_domain_LOCO", "setting": cat,
                             "kind": "within_domain_new_content",
                             "spearman": c2[cat]["spearman"]})
        run.log_metric("C2_leave_one_category_out", c2)
        print("  C2 域内留一类完成", flush=True)

        # ============ C3 第二个环境声语料（ESC-50） ============
        esc = load("layer_a_esc50_ln".replace("layer_a_", ""))  # esc50_ln
        esc_cols = [c for c in cols if c in esc.columns]
        c3 = {}
        for music in ("reg_deam", "reg_pmemo"):
            tr = load(f"{music}_ln")
            imp = SimpleImputer(strategy="median").fit(tr[esc_cols].to_numpy(dtype=np.float64))
            m = build_regressors()["XGBoost"].fit(
                imp.transform(tr[esc_cols].to_numpy(dtype=np.float64)),
                tr["arousal"].to_numpy())
            p = m.predict(imp.transform(esc[esc_cols].to_numpy(dtype=np.float64)))
            # ESC-50 是二分类代理标签 → 用 AUC；预测越高应对应 label=1（低唤醒）
            auc = float(roc_auc_score(esc["label"].to_numpy(), p))
            c3[f"{music}->esc50"] = {"auc": auc, "auc_if_flipped": 1 - auc}
            out_rows.append({"control": "C3_second_ambient", "setting": f"{music}->ESC-50",
                             "kind": "cross_domain_auc", "spearman": auc})
        run.log_metric("C3_second_ambient_corpus", c3)
        print("  C3 第二环境声语料完成", flush=True)

        pd.DataFrame(out_rows).to_csv(run.artifact_path("domain_wall_controls.csv"),
                                      index=False, encoding="utf-8")

        # ---------- 打印 ----------
        print("\n" + "=" * 70)
        print(f"run_id: {run.run_id}\n")
        print("--- C1 时长对照（Spearman ρ）---")
        print(f"{'设置':<28}{'ρ':>9}")
        for k, v in c1.items():
            print(f"{k:<28}{v:>9.3f}")
        cd6 = np.mean([c1[k] for k in c1 if k.endswith("|6s") and "->" not in k])
        cd30 = np.mean([c1[k] for k in c1 if k.endswith("|30s") and "->" not in k])
        sd6 = np.mean([c1[k] for k in c1 if k.endswith("|6s") and "->" in k])
        print(f"\n  跨域: 30s {cd30:.3f} → 6s {cd6:.3f}   同域(6s): {sd6:.3f}")
        if abs(cd6) < 0.25 and sd6 > 0.4:
            print("  → 截短后跨域仍失败、同域仍成立：**时长混淆被排除**")

        print("\n--- C2 域内留一类（训练其余五类，测试留出类）---")
        for cat, v in c2.items():
            print(f"  {cat:<12} n={v['n_test']:>4}   ρ = {v['spearman']:+.3f}")
        mean_c2 = np.mean([v["spearman"] for v in c2.values()])
        print(f"  均值 ρ = {mean_c2:+.3f}")

        print("\n--- C3 第二个环境声语料（ESC-50，语义代理标签）---")
        for k, v in c3.items():
            print(f"  {k:<24} AUC = {v['auc']:.3f}  (反向 {v['auc_if_flipped']:.3f})")

        run.log_metric("summary", {
            "C1_cross_domain_30s": float(cd30), "C1_cross_domain_6s": float(cd6),
            "C1_same_domain_6s": float(sd6),
            "C2_mean_within_domain": float(mean_c2),
            "duration_confound_excluded": bool(abs(cd6) < 0.25 and sd6 > 0.4),
        })


if __name__ == "__main__":
    main()
