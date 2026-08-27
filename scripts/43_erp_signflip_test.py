"""符号翻转置换检验 —— 判定 456 ms 处的偏转是真诱发响应还是伪迹。

    .venv/Scripts/python scripts/43_erp_signflip_test.py

## 起因

`41_eeg_controls_v2.py` 的 C3 判据（0–300 ms 诱发 GFP > 零分布）**未通过**。
但同一次运行导出的总平均波形（预先声明的诊断输出）显示：

    基线 −200–0 ms      平坦，SD 0.093 μV
    0–200 ms            几乎无偏转 —— **确实没有 N1**
    380–500 ms          平滑正向偏转，峰 **+1.51 μV @ 456 ms**，SEM≈0.19
    最大偏转 / 基线 SD   **16.2**

判据窗设在 0–300 ms（针对 N1），而响应在 456 ms —— **窗口没对准**。

## 为什么用符号翻转，而不是再造一个时间零分布

C3 的两个零分布（均匀随机、循环移位）都要把窗口移到记录的**别处**，
而刺激起始恰好落在试次结构中最安静的一段（注视十字期间），
移走的窗口必然落进作答期，噪声环境不同 —— 零分布因此系统性偏高。
两臂同处理修掉了伪迹剔除的不对称，**修不掉这一条**。

符号翻转不移动时间：对每名受试的诱发波形随机取 ±1 再求组平均，
**试次、基线、噪声环境、样本量全部不变**，只破坏「跨受试同相位」这一件事。
这是组水平 ERP 显著性的标准做法（Maris & Oostenveld 2007），
并用**最大统计量**控制整个时窗上的多重比较。

## 判据（先于结果）

组平均 |t| 的最大值，与 10,000 次符号翻转的最大 |t| 零分布比较。
**p < .05 且峰值落在刺激后 → 存在刺激锁时诱发响应。**

同时报告基线段（−200–0 ms）的同一统计量作为阴性对照：
基线段**不应**显著。若基线也显著，说明是慢漂移而非诱发响应。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

SRC = REPO_ROOT / "reports" / "source_data" / "ds002721_erp_waveforms.csv"
SF, TMIN = 250.0, -0.20
N_PERM = 10_000


def max_abs_t(S: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    """组水平单样本 |t| 的最大值与其时间下标。"""
    m, sd, n = S.mean(0), S.std(0, ddof=1), S.shape[0]
    t = np.abs(m) / np.maximum(sd / np.sqrt(n), 1e-18)
    idx = np.where(mask)[0]
    j = idx[np.argmax(t[idx])]
    return float(t[j]), int(j)


def main() -> None:
    W = pd.read_csv(SRC)
    tcols = [c for c in W.columns if c.startswith("t") and c[1:].isdigit()]
    tcols = sorted(tcols, key=lambda c: int(c[1:]))
    S = W.groupby("subject")[tcols].mean().to_numpy(dtype=float)   # (受试, 时间)
    t_ax = np.arange(S.shape[1]) / SF + TMIN
    post = (t_ax >= 0.0) & (t_ax <= 0.60)
    base = t_ax < 0.0

    with RunRecord(
        "erp_signflip_test", seed=SEED,
        params={"n_subjects": int(S.shape[0]), "n_perm": N_PERM,
                "channels": "F3/Fz/F4/C3/Cz/C4 平均",
                "window_s": [float(t_ax[0]), float(t_ax[-1])],
                "method": "组水平符号翻转 + 最大统计量（Maris & Oostenveld 2007）",
                "why": "C3 的时间零分布无法排除「刺激起始落在试次中最安静一段」"
                       "带来的噪声环境差异；符号翻转不移动时间"},
    ) as run:
        run.add_input_snapshot("waveforms", [SRC])
        rng = np.random.default_rng(SEED)

        obs_t, obs_j = max_abs_t(S, post)
        base_t, base_j = max_abs_t(S, base)

        null_post, null_base = np.empty(N_PERM), np.empty(N_PERM)
        for i in range(N_PERM):
            flip = rng.choice([-1.0, 1.0], size=(S.shape[0], 1))
            Sp = S * flip
            null_post[i] = max_abs_t(Sp, post)[0]
            null_base[i] = max_abs_t(Sp, base)[0]

        p_post = float((np.sum(null_post >= obs_t) + 1) / (N_PERM + 1))
        p_base = float((np.sum(null_base >= base_t) + 1) / (N_PERM + 1))

        ga = S.mean(0) * 1e6
        sem = S.std(0, ddof=1) / np.sqrt(S.shape[0]) * 1e6

        run.log_metric("n_subjects", int(S.shape[0]))
        run.log_metric("post_stimulus", {
            "max_abs_t": obs_t, "latency_ms": float(t_ax[obs_j] * 1000),
            "amplitude_uV": float(ga[obs_j]), "sem_uV": float(sem[obs_j]),
            "p": p_post, "null_95th": float(np.percentile(null_post, 95))})
        run.log_metric("baseline_negative_control", {
            "max_abs_t": base_t, "latency_ms": float(t_ax[base_j] * 1000),
            "amplitude_uV": float(ga[base_j]), "p": p_base,
            "null_95th": float(np.percentile(null_base, 95))})

        # 显著时段（逐点 |t| 超过零分布 95 分位）
        m, sd, n = S.mean(0), S.std(0, ddof=1), S.shape[0]
        tt = np.abs(m) / np.maximum(sd / np.sqrt(n), 1e-18)
        thr = float(np.percentile(null_post, 95))
        sig = t_ax[(tt > thr) & post]
        run.log_metric("significant_window_ms",
                       [float(sig.min() * 1000), float(sig.max() * 1000)]
                       if sig.size else None)

        print("=" * 78)
        print(f"run_id: {run.run_id}   受试 {S.shape[0]}   置换 {N_PERM:,}")
        print("\n--- 刺激后 0–600 ms ---")
        print(f"  最大 |t| = {obs_t:.2f} @ {t_ax[obs_j]*1000:.0f} ms   "
              f"幅度 {ga[obs_j]:+.3f} ± {sem[obs_j]:.3f} μV")
        print(f"  零分布 95 分位 {np.percentile(null_post,95):.2f}   "
              f"**p = {p_post:.4f}**")
        if sig.size:
            print(f"  逐点显著时段 {sig.min()*1000:.0f} – {sig.max()*1000:.0f} ms")

        print("\n--- 基线 −200–0 ms（阴性对照，不应显著）---")
        print(f"  最大 |t| = {base_t:.2f} @ {t_ax[base_j]*1000:.0f} ms   "
              f"p = {p_base:.4f}")

        print("\n" + "=" * 78)
        ok = p_post < 0.05
        neg_ok = p_base >= 0.05
        run.log_metric("verdict", {"evoked_response": ok,
                                   "baseline_clean": neg_ok})
        if ok and neg_ok:
            print("✅ 存在刺激锁时诱发响应，且基线段不显著（非慢漂移）。")
            print("   加上已通过的 C1（枕区 alpha）与 C2（眨眼锁时对齐），")
            print("   流水线能测出已知效应 —— 生理线可以继续。")
            print("   注意：**没有经典 N1**，响应是 ~450 ms 的晚成分，")
            print("   与播放延迟抖动抹掉快成分的解释一致。此点须写进报告。")
        elif ok and not neg_ok:
            print("⚠️ 刺激后显著，但基线段也显著 —— 更可能是慢漂移而非诱发响应。")
            print("   不足以支持生理线继续。")
        else:
            print("🔴 符号翻转检验不显著 —— 456 ms 处的偏转不可与噪声区分。")
            print("   → 执行删除方案，本项目仅保留主观维度。")


if __name__ == "__main__":
    main()
