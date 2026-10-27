"""Ibis0 classifier: Qwen3-30B-A3B backbone with a scalar (biased) linear head on the last token.

Shipped with the exported classifier and loaded through ``trust_remote_code``. The logit is
``score(h_last)`` where ``h_last`` is the final-norm hidden state of the last non-padding token;
``sigmoid(logit)`` is the model's probability that the hypothesis holds.
"""

from torch import nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForSequenceClassification


class Ibis0ForSequenceClassification(Qwen3MoeForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        # The trained head has a bias; the stock sequence-classification head does not.
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=True)
