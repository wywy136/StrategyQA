from typing import Dict, List
import json

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from config import Argument


class Golden_Dataset(Dataset):
    def __init__(self, split: str = 'train'):
        Dataset.__init__(self)
        self.arg = Argument
        self.data = open(self.arg.train_path, 'r', encoding='utf-8') if split == 'train' else \
            open(self.arg.test_path, 'r', encoding='utf-8')
        self.corpus = open(self.arg.corpus_path, 'r', encoding='utf-8')
        self.json_data: List[Dict] = json.load(self.data)
        self.json_corpus: Dict[Dict] = json.load(self.corpus)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self) -> int:
        return len(self.json_data)

    def __getitem__(self, index: int) -> Dict:
        piece = self.json_data[index]
        inputs = [self.tokenizer.convert_tokens_to_ids('[CLS]')]

        for field in self.arg.fields:
            if field == "question":
                inputs += self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(piece[field]))
            if field == "evidence":
                path = piece[field][0]
                for step in path:
                    for evidence in step:
                        if type(evidence) == list:
                            for paragraph in evidence:
                                text = self.json_corpus[paragraph]['content']
                                inputs += self.tokenizer.convert_tokens_to_ids(['[SEP]']+self.tokenizer.tokenize(text))

        inputs = inputs[:self.arg.max_length]
        inputs.append(self.tokenizer.convert_tokens_to_ids('[SEP]'))
        masks = [1] * len(inputs)
        ans = 1 if piece['answer'] else 0

        return {
            'input': inputs,
            'mask': masks,
            'label': ans
        }


class Golden_Collator(object):
    def __call__(self, batch: Dict):
        input_ids = [each['input'] for each in batch]
        masks = [each['mask'] for each in batch]
        labels = [each['label'] for each in batch]

        max_len = max([len(each) for each in input_ids])

        for i in range(len(input_ids)):
            input_ids[i].extend([0] * (max_len - len(input_ids[i])))
            masks[i].extend([0] * (max_len - len(masks[i])))

        input_ids = torch.tensor(input_ids, dtype=torch.int32)
        masks = torch.tensor(masks, dtype=torch.int32)
        labels = torch.tensor(labels, dtype=torch.int32)

        return {
            'input_ids': input_ids,
            'masks': masks,
            'labels': labels
        }