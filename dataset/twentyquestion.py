from typing import Dict, List
import json

from torch.utils.data import Dataset
from transformers import RobertaTokenizer

from config import Argument


class TwentyQuestion_Dataset(Dataset):
    def __int__(self):
        Dataset.__init__(self)
        self.arg = Argument
        self.data = open(self.arg.boolq_path, 'r', encoding='utf-8')
        self.json_data: List[Dict] = json.load(self.data)

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def __len__(self) -> int:
        return len(self.json_data)

    def __getitem__(self, item: int) -> Dict:
        data = self.json_data[item]
        question: str = data['question']
        question.replace('it', data['subject'])
        input_ids = self.tokenizer(question)["input_ids"]
        masks = [1] * len(input_ids)
        answer = 1 if data['majority'] else 0

        return {
            "input": input_ids,
            "mask": masks,
            "label": answer
        }
