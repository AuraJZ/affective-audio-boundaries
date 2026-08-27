"""Layer C —— 预训练音频嵌入（笔记 §2.3，≈ 论文的 CHEM-BERT / KANO）。

笔记原文明确写了这层是**对照组不是主力**：

> ⚠️ 论文结论：预训练在小样本上**不一定赢过**经典 ML + 指纹。

所以这层的价值不在于提升精度，而在于回答两个问题：

1. **精度天花板在哪？** 若 CLAP 嵌入也只能到 ρ≈0.6，说明限制来自
   标签噪声而非特征表达力，再堆特征无用。
2. **表示多样性够不够？** Layer A 与 Layer C 的差异远大于 A 与 B
   （手工描述符 vs 深度嵌入），若 stacking 在 A+C 上仍不成立，
   则「表示冗余」这个解释被彻底证伪。

用 `laion/clap-htsat-unfused`（transformers 内置），48 kHz 单声道，
输出 512 维。同样在响度归一化后的信号上提取，与 Layer A/B 保持一致。
"""

from __future__ import annotations

import numpy as np
import torch

EXTRACTOR_VERSION = "layer_c/1.0.0"
MODEL_ID = "laion/clap-htsat-unfused"
FS = 48000
MAX_DURATION_S = 30.0

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        from transformers import ClapModel, ClapProcessor
        _processor = ClapProcessor.from_pretrained(MODEL_ID)
        _model = ClapModel.from_pretrained(MODEL_ID).eval()
    return _model, _processor


@torch.no_grad()
def embed_batch(waves: list[np.ndarray]) -> np.ndarray:
    """一批信号 → (n, 512) 嵌入。输入应已响度归一化。"""
    model, proc = _load()
    # transformers 5.x 起 `audios` 改名为 `audio`
    inputs = proc(audio=[np.asarray(w, dtype=np.float32) for w in waves],
                  sampling_rate=FS, return_tensors="pt", padding=True)
    out = model.get_audio_features(**inputs)
    # transformers 5.x 可能返回输出对象而非张量。
    # 不能用 `a or b` —— 张量的真值判断会抛 RuntimeError，必须显式判 None。
    if isinstance(out, torch.Tensor):
        emb = out
    else:
        emb = None
        for attr in ("audio_embeds", "pooler_output"):
            v = getattr(out, attr, None)
            if v is not None:
                emb = v
                break
        if emb is None:
            emb = out.last_hidden_state.mean(dim=1)
    emb = torch.nn.functional.normalize(emb, dim=-1)  # 单位化，便于余弦几何
    return emb.cpu().numpy()


def column_names(dim: int = 512) -> list[str]:
    return [f"clap{i:03d}" for i in range(dim)]
