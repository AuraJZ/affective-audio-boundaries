"""刺激起始陡峭度 —— 由音频回答「该不该有听觉起始响应」。

    .venv/Scripts/python scripts/42_stimulus_onset_sharpness.py

## 为什么要问这个

听觉 N1 是**声起始**（silence → sound 的锐利跃迁）诱发的。
若这 360 段电影配乐节选带淡入，或从乐句中段截出而起点本就微弱，
那么「测不到起始诱发响应」是**刺激材料的性质**，不是流水线失效。

这个问题必须由**音频数据**回答，不能由脑电回答 ——
否则就是拿待检验的结论去解释检验本身。

## 度量

以 5 ms 帧的 RMS 包络计：

| 量 | 定义 |
|---|---|
| `rise_20ms` | 前 20 ms 的 RMS ÷ 全段中位 RMS |
| `rise_200ms` | 前 200 ms 的 RMS ÷ 全段中位 RMS |
| `t_to_half` | RMS 首次达到全段中位一半所需时间（s） |
| `lead_silence_ms` | 起点处低于中位 −40 dB 的静音时长 |

参照：一个真正的锐利起始应有 `rise_20ms` 接近 1（起始瞬间即达典型响度）、
`t_to_half` 在数十毫秒内；淡入则 `rise_20ms` ≈ 0、`t_to_half` 达数百毫秒以上。

**判据**：若中位 `t_to_half` > 0.5 s，则这批刺激不具备锐利声起始，
「无起始诱发响应」不构成对流水线的否证 —— 但也同样不构成生理线可继续的理由，
只是让阴性结果的**归因**变得明确。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from soundml import features_a
from soundml.provenance import REPO_ROOT, RunRecord

MP3 = REPO_ROOT / "data" / "raw" / "soundtracks" / "Set1"
HOP_S = 0.005


def onset_profile(path) -> dict[str, float]:
    y, sr = features_a.load_prepared(str(path))
    hop = max(int(sr * HOP_S), 1)
    n = len(y) // hop
    rms = np.sqrt(np.array([np.mean(y[i * hop:(i + 1) * hop] ** 2)
                            for i in range(n)]) + 1e-20)
    med = float(np.median(rms))
    if med <= 0:
        return {}
    half = np.where(rms >= med / 2)[0]
    thr = med * 10 ** (-40 / 20)
    lead = int(np.argmax(rms >= thr)) if np.any(rms >= thr) else n
    return {"rise_20ms": float(rms[:max(int(0.020 / HOP_S), 1)].mean() / med),
            "rise_200ms": float(rms[:max(int(0.200 / HOP_S), 1)].mean() / med),
            "t_to_half_s": float(half[0] * HOP_S) if len(half) else np.nan,
            "lead_silence_ms": float(lead * HOP_S * 1000),
            "duration_s": float(len(y) / sr)}


def main() -> None:
    files = sorted(MP3.glob("*.mp3"))
    with RunRecord(
        "stimulus_onset_sharpness", seed=0,
        params={"corpus": "Soundtracks Set1", "n_clips": len(files),
                "hop_s": HOP_S,
                "question": "这批刺激有没有锐利声起始（决定听觉 N1 该不该存在）"},
    ) as run:
        rows = []
        for i, p in enumerate(files, 1):
            d = onset_profile(p)
            if d:
                rows.append({"stim": int(p.stem), **d})
            if i % 120 == 0:
                print(f"  {i}/{len(files)}", flush=True)

        D = pd.DataFrame(rows)
        run.log_metric("n_clips", int(len(D)))
        for c in ("rise_20ms", "rise_200ms", "t_to_half_s", "lead_silence_ms",
                  "duration_s"):
            run.log_metric(c, {"median": float(D[c].median()),
                               "p10": float(D[c].quantile(.10)),
                               "p90": float(D[c].quantile(.90))})

        sharp = float((D.t_to_half_s <= 0.05).mean())
        gradual = float((D.t_to_half_s > 0.5).mean())
        run.log_metric("frac_sharp_onset_le_50ms", sharp)
        run.log_metric("frac_gradual_onset_gt_500ms", gradual)
        verdict = "gradual" if D.t_to_half_s.median() > 0.5 else "sharp"
        run.log_metric("verdict", verdict)

        print("\n" + "=" * 74)
        print(f"run_id: {run.run_id}   {len(D)} 段")
        print(f"\n  时长               中位 {D.duration_s.median():.1f} s")
        print(f"  前 20 ms  / 中位RMS  中位 {D.rise_20ms.median():.3f}   "
              f"(p10 {D.rise_20ms.quantile(.1):.3f}  p90 {D.rise_20ms.quantile(.9):.3f})")
        print(f"  前 200 ms / 中位RMS  中位 {D.rise_200ms.median():.3f}")
        print(f"  达到中位一半用时     中位 {D.t_to_half_s.median()*1000:.0f} ms   "
              f"(p90 {D.t_to_half_s.quantile(.9)*1000:.0f} ms)")
        print(f"  起点静音（<中位-40dB）中位 {D.lead_silence_ms.median():.0f} ms")
        print(f"\n  锐利起始（≤50 ms）  {100*sharp:.0f}%")
        print(f"  渐入起始（>500 ms） {100*gradual:.0f}%")
        print("\n" + "=" * 74)
        if verdict == "sharp":
            print("→ 这批刺激**具备锐利声起始**，听觉起始响应本应存在。")
            print("  因此 C3 的结果直接反映流水线能否测到刺激锁时响应。")
        else:
            print("→ 这批刺激**不具备锐利声起始**（渐入或从乐句中段截取）。")
            print("  「无起始诱发响应」由刺激材料解释，不否证流水线；")
            print("  但也不构成生理线可继续的理由 —— 只是让归因明确。")

        D.to_csv(REPO_ROOT / "reports" / "source_data" /
                 "soundtracks_onset_sharpness.csv", index=False)


if __name__ == "__main__":
    main()
