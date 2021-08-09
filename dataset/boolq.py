from typing import Dict, List
import json

from torch.utils.data import Dataset
from transformers import RobertaTokenizer


class BoolQ_Dataset(Dataset):
    def __int__(self, args):
        Dataset.__init__(self)
        self.arg = args
        self.data = open(self.arg.boolq_path, 'r', encoding='utf-8')
        self.json_data: List[Dict] = json.load(self.data)

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def __len__(self) -> int:
        return len(self.json_data)

    def __getitem__(self, item: int) -> Dict:
        data = self.json_data[item]
        question = self.tokenizer(data['question'])["input_ids"]
        answer = 1 if data["answer"] else 0
        passage = self.tokenizer(data['passage'])["input_ids"]
        input_ids = question + [2] + passage[1:]
        masks = [1] * len(input_ids)

        return {
            "input": input_ids,
            "mask": masks,
            "label": answer
        }
