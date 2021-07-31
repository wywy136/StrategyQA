from typing import Dict, List
import json

from transformers import RobertaTokenizer

from config import Argument


class GoldenSentenceDataset:
    def __init__(self, split: str = 'train'):
        self.arg = Argument
        if split == 'train':
            self.data = open(self.arg.sentchain_train, 'r', encoding='utf-8')
            self.original = open(self.arg.train_path, 'r', encoding='utf-8')
        elif split == 'dev':
            self.data = open(self.arg.sentchain_dev, 'r', encoding='utf-8')
            self.original = open(self.arg.dev_path, 'r', encoding='utf-8')
        else:
            self.data = open(self.arg.sentchain_test, 'r', encoding='utf-8')
            self.original = open(self.arg.test_path, 'r', encoding='utf-8')
        self.data: Dict = json.load(self.data)
        self.original: List = json.load(self.original)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    def __len__(self) -> int:
        return len(self.original)

    def __getitem__(self, item: int) -> Dict:
        piece = self.original[item]
        qid = piece['qid']
        question = piece['question']
        sents: List = self.data[qid]

        inputs = self.tokenizer(question)["input_ids"]
        for sent in sents:
            inputs += [2] + self.tokenizer(sent)["input_ids"]
        inputs = inputs[:self.arg.max_length]
        masks = [1] * len(inputs)
        ans = 1 if piece['answer'] else 0

        return {
            'input': inputs,
            'mask': masks,
            'label': ans,
            'op_len': 0,
            'op_abstract': 0,
            'qid': qid
        }
