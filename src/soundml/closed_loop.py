"""闭环生成的四道闸 —— 防止把模型盲区当成发现。

## 问题

拿感知模型当适应度函数去优化生成器，找到的很可能是**对抗样本**：
模型判为极度低唤醒、人听着却很怪的声音。这在感知模型上尤其容易发生，
因为模型只见过自然录音，没见过优化器造出来的东西（Goodhart 定律）。

朴素做法（「优化到模型给分最高就收工」）**必然**产出这种东西，
而且没有任何机制能发现。

## 四道闸

本项目恰好已经具备识破它所需的全部工具：

| 闸 | 依据 | 拒绝什么 |
|---|---|---|
| 1 适用域 | `PLAN §3.5`，T = Z·σ + Y | 落在训练分布外的候选 |
| 2 Rashomon 一致 | `16_rashomon.py` | 只有被优化的那个模型说好的候选 |
| 3 域之墙 | `step5/step6` | 只在单域成立的候选（合成音频可能两域皆非） |
| 4 零分布 | `B6` | 效应量不超过参考分布的候选 |

**四道闸全过，才允许送裁判。** 裁判（ARAUS）全程不参与优化。

## 为什么闸 3 对合成音频特别重要

合成音频既不是音乐也不是环境声，可能落在两个训练域之外。
若只用单域模型评分，等于在域外做外推 —— 而我们已经证明跨域外推为零。
因此要求**两域模型都给出低唤醒判定**，是对「这段声音的低唤醒性不依赖域假设」
的最低要求。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Gate:
    """单道闸的判定结果。"""
    name: str
    passed: bool
    value: float
    threshold: float
    note: str = ""


@dataclass
class Verdict:
    """一个候选音频的完整评估。"""
    params: dict
    oracle_score: float                 # 两域模型的保守（较低）分
    gates: list[Gate] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return all(g.passed for g in self.gates)

    @property
    def failed_gates(self) -> list[str]:
        return [g.name for g in self.gates if not g.passed]


class LoopGuards:
    """四道闸。构造时传入已训练好的组件，评估时只做判定。

    Parameters
    ----------
    domain_models : {域名: 已拟合的回归器}
        至少两个域；闸 3 要求全部给出低唤醒判定。
    rashomon_models : list
        近优模型集合（`16_rashomon.py` 的产物）；闸 2 要求多数一致。
    ad : (NearestNeighbors, dict)
        `modeling.fit_ad` 的返回值；闸 1 用其阈值。
    scaler : 已拟合的预处理管线
        用于把特征映射到 AD 所在的空间。
    null_scores : ndarray
        参考分布：真实语料在同一 oracle 下的得分。闸 4 用其分位数。
    """

    def __init__(self, domain_models, rashomon_models, ad, scaler,
                 null_scores: np.ndarray,
                 rashomon_agreement: float = 0.8,
                 null_percentile: float = 90.0):
        self.domain_models = domain_models
        self.rashomon_models = rashomon_models
        self.nn, self.ad_info = ad
        self.scaler = scaler
        self.null_scores = np.asarray(null_scores)
        self.rashomon_agreement = rashomon_agreement
        self.null_percentile = null_percentile
        # 「低唤醒」的判定线：取真实语料得分的下四分位
        self.low_arousal_cut = float(np.percentile(self.null_scores, 25))
        self.null_cut = float(np.percentile(self.null_scores, 100 - null_percentile))

    # ---------------------------------------------------------------- 闸 1
    def _gate_domain(self, x: np.ndarray) -> Gate:
        d, _ = self.nn.kneighbors(self.scaler.transform(x),
                                  n_neighbors=self.ad_info["k"])
        mean_d = float(d.mean())
        thr = float(self.ad_info["threshold"])
        return Gate("applicability_domain", mean_d <= thr, mean_d, thr,
                    "候选落在训练分布外" if mean_d > thr else "")

    # ---------------------------------------------------------------- 闸 2
    def _gate_rashomon(self, x: np.ndarray) -> Gate:
        preds = np.array([m.predict(x)[0] for m in self.rashomon_models])
        frac = float(np.mean(preds <= self.low_arousal_cut))
        return Gate("rashomon_agreement", frac >= self.rashomon_agreement,
                    frac, self.rashomon_agreement,
                    "仅部分近优模型认可 —— 疑似对抗样本"
                    if frac < self.rashomon_agreement else "")

    # ---------------------------------------------------------------- 闸 3
    def _gate_domain_wall(self, x: np.ndarray) -> tuple[Gate, float]:
        scores = {k: float(m.predict(x)[0]) for k, m in self.domain_models.items()}
        worst = max(scores.values())      # 保守：取最不认可的那个域
        ok = all(s <= self.low_arousal_cut for s in scores.values())
        return (Gate("domain_wall", ok, worst, self.low_arousal_cut,
                     f"各域得分 {scores}" if not ok else ""), worst)

    # ---------------------------------------------------------------- 闸 4
    def _gate_null(self, score: float) -> Gate:
        return Gate("null_distribution", score <= self.null_cut,
                    score, self.null_cut,
                    "未超出真实语料的参考分布" if score > self.null_cut else "")

    # ---------------------------------------------------------------- 评估
    def evaluate(self, x: np.ndarray, params: dict) -> Verdict:
        """x 为该候选的特征行向量 (1, n_features)。"""
        x = np.atleast_2d(x)
        g_wall, worst = self._gate_domain_wall(x)
        gates = [self._gate_domain(x), self._gate_rashomon(x), g_wall,
                 self._gate_null(worst)]
        return Verdict(params=params, oracle_score=worst, gates=gates)


def summarise(verdicts: list[Verdict]) -> dict:
    """把一批候选的判定结果压成可报告的统计。"""
    if not verdicts:
        return {}
    n = len(verdicts)
    by_gate = {}
    for g in verdicts[0].gates:
        by_gate[g.name] = float(np.mean([
            any(gg.name == g.name and gg.passed for gg in v.gates) for v in verdicts]))
    return {
        "n_candidates": n,
        "n_accepted": int(sum(v.accepted for v in verdicts)),
        "accept_rate": float(np.mean([v.accepted for v in verdicts])),
        "pass_rate_per_gate": by_gate,
        "oracle_score_range": [float(min(v.oracle_score for v in verdicts)),
                               float(max(v.oracle_score for v in verdicts))],
    }
