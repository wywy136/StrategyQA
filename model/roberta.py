from transformers import RobertaModel
from torch.nn import Module, Linear
import torch


class Reasoning(Module):
    def __init__(self):
        super(Reasoning, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-large')
        self.classifer = Linear(in_features=1024, out_features=2)

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        cls_representation = self.bert(
            input_ids=input,
            attention_mask=mask,
            return_dict=True
        ).pooler_output
        logit = self.classifer(cls_representation)
        return logit