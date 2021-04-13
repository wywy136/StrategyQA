from transformers import BertModel
from torch.nn import Module, Linear
import torch


class Reasoning(Module):
    def __init__(self):
        super(Reasoning, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifer = Linear(in_features=768, out_features=2)

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        cls_representation = self.bert(
            input_ids=input,
            attention_mask=mask,
            return_dict=True
        ).pooler_output
        logit = self.classifer(cls_representation)
        return logit