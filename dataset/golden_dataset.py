from typing import Dict, List
import json

import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

from config import Argument


class GoldenDataset(Dataset):
    def __init__(self, split: str = 'train'):
        Dataset.__init__(self)
        self.arg = Argument
        self.data = open(self.arg.train_path, 'r', encoding='utf-8') if split == 'train' else \
            open(self.arg.dev_path, 'r', encoding='utf-8')
        self.corpus = open(self.arg.corpus_path, 'r', encoding='utf-8')
        self.json_data: List[Dict] = json.load(self.data)
        self.json_corpus: Dict[Dict] = json.load(self.corpus)

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.operator_set = ['greater', 'less', 'before', 'after', 'larger',
                             'smaller', 'higher', 'lower', 'longer', 'shorter',
                             'prior', 'same', 'identical to', 'equal', 'different',
                             'difference', 'match', 'considered', 'least', 'enough',
                             'and', 'or', 'all', 'also', 'both', 'included', 'include',
                             'overlap', 'listed', 'within', 'have', 'excluded',
                             'present', 'among', 'contain', 'absent from',
                             'times', 'multiplied', 'positive', 'divided', 'plus']

    def golden_sentence(self, question: str, paragraph: str) -> str:
        pass

    def get_operator(self, question: str) -> str:
        ans = []
        for op in self.operator_set:
            if op in question:
                ans.append(op)
        return ''.join(ans)

    def __len__(self) -> int:
        return len(self.json_data)

    def __getitem__(self, index: int) -> Dict:
        piece = self.json_data[index]
        inputs = []

        for field in self.arg.fields:
            if field == "question":
                inputs += self.tokenizer(piece[field])["input_ids"]
            if field == "evidence":
                path = piece[field][0]
                for step_index, step in enumerate(path):
                    for evidence in step:
                        if type(evidence) == list:
                            if "decomposition" in self.arg.fields:
                                text = piece['decomposition'][step_index]
                                inputs += [2] + self.tokenizer(text)["input_ids"][1:]
                            for paragraph in evidence:
                                text = self.json_corpus[paragraph]['content']
                                inputs += [2] + self.tokenizer(text)["input_ids"][1:]
                        if "operator" in self.arg.fields:
                            if "operation" == evidence:
                                operators = self.get_operator(piece['decomposition'][step_index])
                                inputs += [2] + self.tokenizer(operators)[1:]

        inputs = inputs[:self.arg.max_length]
        # inputs.append(self.tokenizer.convert_tokens_to_ids('[SEP]'))
        masks = [1] * len(inputs)
        ans = 1 if piece['answer'] else 0

        return {
            'input': inputs,
            'mask': masks,
            'label': ans
        }


class Collator(object):
    def __call__(self, batch: Dict):
        input_ids = [each['input'] for each in batch]
        masks = [each['mask'] for each in batch]
        labels = [each['label'] for each in batch]

        max_len = max([len(each) for each in input_ids])

        for i in range(len(input_ids)):
            input_ids[i].extend([1] * (max_len - len(input_ids[i])))
            masks[i].extend([0] * (max_len - len(masks[i])))

        input_ids = torch.tensor(input_ids, dtype=torch.int32)
        masks = torch.tensor(masks, dtype=torch.int32)
        labels = torch.tensor(labels, dtype=torch.int32)

        return {
            'input_ids': input_ids,
            'masks': masks,
            'labels': labels
        }