from typing import Iterator

from torch.nn import Module
import torch


class Evaluator(object):
    def __call__(self, dataloader: Iterator, model: Module, device: torch.device) -> float:
        all, match = 0, 0
        for index, batch in enumerate(dataloader):
            for key, tensor in batch.items():
                batch[key] = tensor.to(device)
            loss, logits = model(
                input=batch['input_ids'].long(),
                mask=batch['masks'],
                label=batch['labels'].long(),
                op_len=batch['op_len'],
                op_abstract=batch['abstract']
            )
            for i in range(logits.size(0)):
                prd = torch.argmax(logits[i]).item()
                trg = batch['labels'][i].item()
                all += 1
                if prd == trg:
                    match += 1

        return match / all