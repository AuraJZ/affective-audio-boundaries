r"""多种音频基础模型的嵌入提取 —— 用来检验「域之墙」是不是表示不足造成的。

    .venv/Scripts/python scripts/79_extract_foundation_models.py --model ast
    .venv/Scripts/python scripts/79_extract_foundation_models.py --all

## 为什么必须做

论文的核心发现是「音乐上训练的模型搬到环境声上就废了，反之亦然」，
并声称这不是特征设计不足 —— 依据只有**一个** CLAP 模型，
而它偏偏在其中一个语料（Emo-Soundscapes，源自 Freesound）的预训练数据里。

一个模型 + 一个被污染的语料，撑不住「这堵墙不可逾越」这种强声称。
若换个表征就能翻过去，那么「墙」这个框架本身就错了 ——
**这是唯一可能推翻本文核心发现的检验，必须我们自己先做。**

## 四种表征，四种不同的预训练范式

| | 模型 | 预训练 | 为什么选它 |
|---|---|---|---|
| CLAP | `laion/clap-htsat-unfused` | 音频–文本对比，63 万段 | 已有；但与环境声语料共线 |
| AST | `MIT/ast-finetuned-audioset-...` | AudioSet **监督**分类 | 监督式的代表，且 AudioSet 覆盖环境声 |
| MERT | `m-a-p/MERT-v1-95M` | **音乐**自监督 | 若音乐专用模型也翻不过去，说明不是领域知识不足 |
| Wav2Vec2 | `facebook/wav2vec2-base-960h` | **语音**自监督 | 与音频情感无关的预训练 → 下界对照 |

覆盖了「对比式 / 监督式 / 领域内自监督 / 领域外自监督」四个角。

## 口径

统一取**中间 10 秒**，−23 LUFS 归一（与 Layer A/B/C 一致），
各模型重采样到自己的原生采样率，取最后一层隐状态在时间轴上的均值。

CLAP 在此**重提一次 10 秒版**：现有的 `layer_c` 用的是 30 秒，
若不重提，CLAP 与其余三者的差异会掺进时长因素。
两版都保留，可互为对照。

## 断点续跑

CPU 上整轮约 9 小时。逐批落盘，中断后重跑自动跳过已完成的部分 ——
`69` 已经因为「只在结尾写盘」白跑过一小时，不再重复。
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import pyloudnorm
import torch

from soundml.features_a import TARGET_LUFS
from soundml.provenance import REPO_ROOT, RunRecord

warnings.filterwarnings("ignore")
torch.set_num_threads(6)

FEAT = REPO_ROOT / "data" / "features"
CACHE = FEAT / "foundation_cache"
CLIP_S = 10.0                 # 中间 10 秒，四个模型统一
BATCH = 8
FLUSH_EVERY = 96              # 每这么多段落一次盘

MODELS = {
    "ast":   ("MIT/ast-finetuned-audioset-10-10-0.4593", 16000),
    "mert":  ("m-a-p/MERT-v1-95M", 24000),
    "w2v2":  ("facebook/wav2vec2-base-960h", 16000),
    "clap":  ("laion/clap-htsat-unfused", 48000),
}
CORPORA = ["reg_deam", "reg_pmemo", "reg_soundtracks", "reg_emo_mix"]

_cache: dict[str, tuple] = {}
_meters: dict[int, pyloudnorm.Meter] = {}


def load_model(key: str):
    if key not in _cache:
        mid, fs = MODELS[key]
        if key == "clap":
            from transformers import ClapModel, ClapProcessor
            proc = ClapProcessor.from_pretrained(mid)
            model = ClapModel.from_pretrained(mid).eval()
        else:
            from transformers import AutoFeatureExtractor, AutoModel
            proc = AutoFeatureExtractor.from_pretrained(mid, trust_remote_code=True)
            model = AutoModel.from_pretrained(mid, trust_remote_code=True).eval()
        _cache[key] = (model, proc, fs)
    return _cache[key]


def load_audio(path: str, fs: int) -> np.ndarray | None:
    """中间 10 秒，−23 LUFS 归一。与 Layer A/B/C 同口径。"""
    try:
        y, _ = librosa.load(path, sr=fs, mono=True)
    except Exception:                                        # noqa: BLE001
        return None
    n = int(CLIP_S * fs)
    if y.size > n:
        s = (y.size - n) // 2
        y = y[s:s + n]
    elif y.size < n:                       # 短于 10 s 的补零，保持批内等长
        y = np.pad(y, (0, n - y.size))
    if fs not in _meters:
        _meters[fs] = pyloudnorm.Meter(fs)
    if y.size >= int(0.4 * fs):
        lu = _meters[fs].integrated_loudness(y)
        if np.isfinite(lu):
            y = y * 10.0 ** ((TARGET_LUFS - lu) / 20.0)
    return y.astype(np.float32)


@torch.no_grad()
def embed(key: str, waves: list[np.ndarray]) -> np.ndarray:
    model, proc, fs = load_model(key)
    if key == "clap":
        inp = proc(audio=waves, sampling_rate=fs, return_tensors="pt",
                   padding=True)
        out = model.get_audio_features(**inp)
        # 🔴 transformers 换版本后 `get_audio_features` 不再返回张量，
        # 而是 `BaseModelOutputWithPooling`，且**没有** `audio_embeds`
        # 这个属性 —— 音频嵌入在 `pooler_output`（[B, 512]）。
        # 首版只试了 `audio_embeds`，于是每一批都抛 AttributeError。
        if isinstance(out, torch.Tensor):
            v = out
        else:
            v = next((getattr(out, a) for a in
                      ("audio_embeds", "pooler_output", "last_hidden_state")
                      if getattr(out, a, None) is not None), None)
            if v is None:
                raise RuntimeError(
                    f"CLAP 输出取不到嵌入，类型 {type(out).__name__}，"
                    f"可用键 {list(out.keys()) if hasattr(out, 'keys') else '?'}")
            if v.ndim > 2:                      # last_hidden_state 兜底
                v = v.flatten(1).mean(dim=-1) if v.ndim > 3 else v.mean(dim=1)
        return v.numpy()
    inp = proc(waves, sampling_rate=fs, return_tensors="pt", padding=True)
    h = model(**inp).last_hidden_state
    return (h.mean(dim=1) if h.ndim == 3 else h).numpy()


def run_one(key: str, corpus: str) -> None:
    out = FEAT / f"layer_f_{key}_{corpus}.parquet"
    if out.exists():
        print(f"  {key}/{corpus}: 已完成，跳过")
        return
    part = CACHE / f"{key}_{corpus}.parquet"
    CACHE.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(REPO_ROOT / "data" / f"{corpus}.csv")
    n_fail = 0
    done: set[str] = set()
    rows: list[dict] = []
    if part.exists():
        prev = pd.read_parquet(part)
        rows = prev.to_dict("records")
        done = set(prev["clip_id"])
        print(f"  {key}/{corpus}: 续跑，已有 {len(done)}/{len(labels)}")

    todo = [r for r in labels.itertuples(index=False) if r.clip_id not in done]
    if not todo:
        pd.DataFrame(rows).to_parquet(out, index=False)
        part.unlink(missing_ok=True)
        print(f"  {key}/{corpus}: 完成 {len(rows)} 段 → {out.name}")
        return

    _, _, fs = load_model(key)
    t0, n_since = time.perf_counter(), 0
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        waves, ids = [], []
        for r in chunk:
            y = load_audio(r.path, fs)
            if y is not None:
                waves.append(y)
                ids.append(r.clip_id)
        if not waves:
            continue
        try:
            E = embed(key, waves)
            n_fail = 0
        except Exception as e:                               # noqa: BLE001
            n_fail += 1
            print(f"    批失败 {type(e).__name__}: {str(e)[:80]}")
            # 🔴 逐批 try/except 是为容忍**偶发**的坏音频，不是为容忍
            # 「这个模型的 API 调错了」。首版没有这道闸，于是 CLAP 的
            # 每一批都失败、每一批都被吞掉，最后落了个空 parquet 并打印
            # 「完成 0 段」—— 而空文件一存在，断点续跑就永远跳过这一格。
            if n_fail >= 5 and not rows:
                raise RuntimeError(
                    f"{key}/{corpus}: 前 {n_fail} 批全部失败且无一条成功 —— "
                    f"这是配置/API 错误，不是坏样本。末次：{type(e).__name__}: {e}"
                ) from e
            continue
        for cid, v in zip(ids, E):
            rows.append({"clip_id": cid,
                         **{f"{key}_{j}": float(x) for j, x in enumerate(v)}})
        n_since += len(ids)
        if n_since >= FLUSH_EVERY:
            pd.DataFrame(rows).to_parquet(part, index=False)
            n_since = 0
            el = time.perf_counter() - t0
            k = len(rows) - len(done)
            print(f"    {len(rows)}/{len(labels)}   "
                  f"{el/max(k,1):.2f} s/段   剩约 "
                  f"{(len(labels)-len(rows))*el/max(k,1)/60:.0f} 分钟", flush=True)

    D = pd.DataFrame(rows)
    # 🔴 绝不落空文件。首版无条件 `to_parquet` 并打印「完成 0 段 × -1 维」——
    # 一个 0 行 0 列的 parquet 落盘后，下游 merge 直接 KeyError('clip_id')，
    # 而断点续跑看到文件存在就永远跳过，错误被冻结在磁盘上。
    if len(D) == 0 or "clip_id" not in D.columns:
        raise RuntimeError(
            f"{key}/{corpus}: 无任何有效嵌入（{len(D)} 行）—— 不落盘。"
            f"分片保留在 {part.name} 以便排查。")
    cov = len(D) / len(labels)
    D.to_parquet(out, index=False)
    part.unlink(missing_ok=True)
    flag = "" if cov > 0.9 else f"   ⚠️ 覆盖率仅 {cov:.0%}"
    print(f"  {key}/{corpus}: 完成 {len(D)}/{len(labels)} 段 × "
          f"{D.shape[1]-1} 维 → {out.name}{flag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--corpus", choices=CORPORA)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    keys = list(MODELS) if args.all or not args.model else [args.model]
    corps = CORPORA if args.all or not args.corpus else [args.corpus]

    run = RunRecord("79_extract_foundation_models", 0,
                    params={"models": keys, "corpora": corps,
                            "clip_s": CLIP_S, "target_lufs": TARGET_LUFS})
    for k in keys:
        print(f"\n{'='*72}\n{k}  {MODELS[k][0]}")
        for c in corps:
            run_one(k, c)
    run.write()


if __name__ == "__main__":
    main()
