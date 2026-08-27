"""三向对比：音乐域规则 / 环境声域规则 / 两域共有。

    uv run python scripts/09_domain_comparison.py \
        --music-sig <run_id> --ambient-sig <run_id>

## 为什么需要

跨域测试（`08_cross_domain.py`）显示音乐规则在环境声上系统性反转
（方向一致率 18.2%，随机为 50%）。因此两个域必须分别建规则。

但真正有价值的是**域无关**的规则 —— 在两个域都成立、方向一致的那些。
只有它们才可能覆盖全部内容类型。

## 三个层次的比较

1. **逐特征**：两域的校准显著特征是否重合
2. **特征族**：把 `spec_contrast1_p10` / `spec_contrast5_p50` 归为
   同一族 `spec_contrast`，看族层面是否一致 ——
   同一声学维度可能由不同统计量承载
3. **方向**：族层面一致的，进一步查类间差（Cohen's d）方向是否同号。
   **只有方向也一致的，才是真正的域无关规则。**
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from soundml.data import load_joined
from soundml.provenance import RUNS_DIR, RunRecord

# 特征名 → 声学族。顺序敏感，先匹配到的优先。
FAMILY_PATTERNS = [
    (r"^spec_contrast\d+", "谱对比度"),
    (r"^spec_flatness", "谱平坦度"),
    (r"^spec_rolloff", "频谱滚降"),
    (r"^spec_centroid", "频谱重心"),
    (r"^spec_bandwidth", "频谱带宽"),
    (r"^onset_env|^onset_rate", "起音/突发"),
    (r"^zcr", "过零率"),
    (r"^rms", "能量"),
    (r"^mfcc\d+", "MFCC"),
    (r"^f0", "基频"),
    (r"^tempo|^pulse", "节奏"),
    (r"^slope_1f|_energy_ratio$", "频谱形状"),
    (r"^loudness|^sharpness|^roughness|^fluct", "心理声学"),
    (r"^lr_corr|^side_mid", "空间"),
]


def family_of(feature: str) -> str:
    for pat, fam in FAMILY_PATTERNS:
        if re.search(pat, feature):
            return fam
    return "其他"


def stat_of(feature: str) -> str:
    """提取统计量后缀：mean / std / p10 / p50 / p90 / dmean / rate。"""
    for s in ("_dmean", "_mean", "_std", "_p10", "_p50", "_p90"):
        if feature.endswith(s):
            return s[1:]
    return "raw"


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else np.nan


def domain_frame(dataset: str) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    df, reps = load_joined(dataset)
    cols = reps["A+B"]
    X = SimpleImputer(strategy="median").fit_transform(df[cols].to_numpy(dtype=np.float64))
    return df, cols, X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--music-sig", required=True)
    ap.add_argument("--ambient-sig", required=True)
    ap.add_argument("--music", default="deam")
    ap.add_argument("--ambient", default="emo")
    args = ap.parse_args()

    def calibrated(run_id: str) -> list[str]:
        m = json.loads((RUNS_DIR / run_id / "meta.json").read_text(encoding="utf-8"))
        return m["metrics"]["calibrated_consensus"]

    mus_feats, amb_feats = calibrated(args.music_sig), calibrated(args.ambient_sig)

    dm, cm, Xm = domain_frame(args.music)
    da, ca, Xa = domain_frame(args.ambient)
    ym, ya = dm["label"].to_numpy(), da["label"].to_numpy()

    with RunRecord(
        "domain_comparison", seed=0,
        params={"music": args.music, "ambient": args.ambient,
                "music_sig_run": args.music_sig, "ambient_sig_run": args.ambient_sig},
    ) as run:
        # ---------- 层次 1：逐特征 ----------
        exact = sorted(set(mus_feats) & set(amb_feats))
        run.log_metric("exact_overlap", exact)
        run.log_metric("n_exact_overlap", len(exact))

        # ---------- 层次 2：特征族 ----------
        fam_m = {f: family_of(f) for f in mus_feats}
        fam_a = {f: family_of(f) for f in amb_feats}
        shared_fams = sorted(set(fam_m.values()) & set(fam_a.values()))
        run.log_metric("music_families", sorted(set(fam_m.values())))
        run.log_metric("ambient_families", sorted(set(fam_a.values())))
        run.log_metric("shared_families", shared_fams)

        # ---------- 层次 3：方向 ----------
        # 对每个共享族，取该族在两域各自的显著特征，比较类间差方向
        rows = []
        for fam in shared_fams:
            fm = [f for f in mus_feats if fam_m[f] == fam]
            fa = [f for f in amb_feats if fam_a[f] == fam]
            for f in fm:
                d_m = cohens_d(Xm[ym == 1, cm.index(f)], Xm[ym == 0, cm.index(f)])
                # 同一特征在环境声域的方向（无论其在该域是否显著）
                d_a = (cohens_d(Xa[ya == 1, ca.index(f)], Xa[ya == 0, ca.index(f)])
                       if f in ca else np.nan)
                rows.append({"family": fam, "feature": f, "significant_in": "music",
                             "d_music": d_m, "d_ambient": d_a,
                             "same_direction": bool(np.sign(d_m) == np.sign(d_a))
                                               if np.isfinite(d_a) else None})
            for f in fa:
                d_a = cohens_d(Xa[ya == 1, ca.index(f)], Xa[ya == 0, ca.index(f)])
                d_m = (cohens_d(Xm[ym == 1, cm.index(f)], Xm[ym == 0, cm.index(f)])
                       if f in cm else np.nan)
                rows.append({"family": fam, "feature": f, "significant_in": "ambient",
                             "d_music": d_m, "d_ambient": d_a,
                             "same_direction": bool(np.sign(d_m) == np.sign(d_a))
                                               if np.isfinite(d_m) else None})

        cmp = pd.DataFrame(rows).drop_duplicates(subset=["feature"])
        cmp.to_csv(run.artifact_path("domain_comparison.csv"), index=False, encoding="utf-8")

        valid = cmp[cmp["same_direction"].notna()]
        agree = valid[valid["same_direction"]]
        run.log_metric("n_comparable", len(valid))
        run.log_metric("n_same_direction", len(agree))
        run.log_metric("domain_invariant_features", sorted(agree["feature"].tolist()))

        # ---------- 打印 ----------
        print("=" * 74)
        print(f"run_id: {run.run_id}\n")
        print(f"音乐域校准特征   : {len(mus_feats)} 个")
        print(f"环境声域校准特征 : {len(amb_feats)} 个")
        print(f"\n--- 层次 1：逐特征重合 ---")
        print(f"  {len(exact)} 个" + (f"：{exact}" if exact else "（无）"))

        print(f"\n--- 层次 2：特征族 ---")
        print(f"  音乐域   : {sorted(set(fam_m.values()))}")
        print(f"  环境声域 : {sorted(set(fam_a.values()))}")
        print(f"  共享族   : {shared_fams}")

        print(f"\n--- 层次 3：方向一致性（共享族内，d>0 = 助眠类该特征更高）---")
        print(f"{'族':<10}{'特征':<24}{'显著于':<9}{'音乐 d':>9}{'环境声 d':>10}  一致")
        for r in cmp.sort_values(["family", "feature"]).itertuples():
            if r.same_direction is None:
                mark = "—（另一域无此特征）"
            else:
                mark = "✓" if r.same_direction else "✗ 翻转"
            dm_s = f"{r.d_music:>9.3f}" if np.isfinite(r.d_music) else f"{'—':>9}"
            da_s = f"{r.d_ambient:>10.3f}" if np.isfinite(r.d_ambient) else f"{'—':>10}"
            print(f"{r.family:<10}{r.feature:<24}{r.significant_in:<9}{dm_s}{da_s}  {mark}")

        print(f"\n可比较 {len(valid)} 个，方向一致 {len(agree)} 个")
        if len(agree):
            print("\n【域无关特征】—— 两域方向一致，价值最高：")
            for f in sorted(agree["feature"]):
                r = agree[agree.feature == f].iloc[0]
                print(f"  {f:<24} 音乐 d={r.d_music:+.3f}  环境声 d={r.d_ambient:+.3f}")
        else:
            print("\n⚠️ 没有任何域无关特征。规则库必须按域拆分，")
            print("   产品需按内容类型（音乐 / 环境声）分别调用不同规则集。")


if __name__ == "__main__":
    main()
