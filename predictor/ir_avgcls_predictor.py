import json

from torch.utils.data import DataLoader
from torch.nn import Module
import torch


class IrAvrClsPredictor:
    def __init__(self, path: str):
        self.output_path = path
        self.predictions = []

    def __call__(self, dataloader: DataLoader, model: Module, device: torch.device):
        for index, batch in enumerate(dataloader):
            for key, tensor in batch.items():
                if type(tensor) == torch.Tensor:
                    batch[key] = tensor.to(device)
            loss, logits = model(
                input=batch['input_ids'].long(),
                mask=batch['masks'],
                label=batch['labels'].long(),
                op_len=batch['op_len'],
                op_abstract=batch['op_abstract'].long()
            )
            for i in range(logits.size(0)):
                prd = torch.argmax(logits[i]).item()
                self.predictions.append({
                    "qid": batch['qid'][i],
                    "answer": prd
                })
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.predictions, f)