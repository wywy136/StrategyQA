from typing import Dict
from dataset.golden_dataset import GoldenDataset


class LastStepDataset(GoldenDataset):
    def __init__(self, args, split: str = 'train'):
        GoldenDataset.__init__(self, args, split)

    def __len__(self) -> int:
        return len(self.json_data)

    def __getitem__(self, item: int) -> Dict:
        piece = self.json_data[item]
        question = piece["decomposition"][-1]
        inputs = self.tokenizer(question)["input_ids"]
        masks = [1] * len(inputs)
        ans = 1 if piece['answer'] else 0
        op_len = 0
        op_abstract = 0

        return {
            'input': inputs,
            'mask': masks,
            'label': ans,
            'op_len': op_len,
            'op_abstract': op_abstract
        }