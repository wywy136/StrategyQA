from typing import Dict, List
import json

import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer


class GoldenDataset(Dataset):
    def __init__(self, args, split: str = 'train'):
        Dataset.__init__(self)
        self.arg = args
        self.data = open(self.arg.train_path, 'r', encoding='utf-8') if split == 'train' else \
            open(self.arg.dev_path, 'r', encoding='utf-8')
        self.corpus = open(self.arg.corpus_path, 'r', encoding='utf-8')
        self.json_data: List[Dict] = json.load(self.data)
        self.json_corpus: Dict[Dict] = json.load(self.corpus)

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.operator_set = ['greater', 'less', 'before', 'after', 'larger',
                             'smaller', 'higher', 'lower', 'longer', 'shorter',
                             'prior', 'same', 'identical', 'equal', 'different',
                             'difference', 'match', 'considered', 'least', 'enough',
                             'and', 'or', 'all', 'also', 'both', 'included', 'include',
                             'overlap', 'listed', 'within', 'have', 'excluded',
                             'present', 'among', 'contain', 'absent',
                             'times', 'multiplied', 'positive', 'divided', 'plus']
        self.op_mapping = {
            'greater': [0, 'comparison'],
            'less': [0, 'comparison'],
            'before': [0, 'comparison'],
            'after': [0, 'comparison'],
            'larger': [0, 'comparison'],
            'smaller': [0, 'comparison'],
            'higher': [0, 'comparison'],
            'lower': [0, 'comparison'],
            'longer': [0, 'comparison'],
            'shorter': [0, 'comparison'],
            'prior': [0, 'comparison'],
            'same': [0, 'comparison'],
            'identical': [0, 'comparison'],
            'equal': [0, 'comparison'],
            'different': [0, 'comparison'],
            'difference': [0, 'comparison'],
            'match': [0, 'comparison'],
            'considered': [0, 'comparison'],
            'least': [2, 'numerical'],
            'enough': [2, 'numerical'],
            'times': [2, 'numerical'],
            'plus': [2, 'numerical'],
            'multiplied': [2, 'numerical'],
            'divided': [2, 'numerical'],
            'and': [1, 'logical'],
            'or': [1, 'logical'],
            'all': [1, 'logical'],
            'also': [1, 'logical'],
            'both': [1, 'logical'],
            'included': [3, 'entail'],
            'include': [3, 'entail'],
            'overlap': [3, 'entail'],
            'listed': [3, 'entail'],
            'within': [3, 'entail'],
            'have': [3, 'entail'],
            'excluded': [3, 'entail'],
            'present': [3, 'entail'],
            'among': [3, 'entail'],
            'contain': [3, 'entail'],
            'absent': [3, 'entail'],
            'positive': [2, 'numerical']
        }

    def get_operator(self, question: str) -> str:
        ans = []
        for op in self.operator_set:
            if op in question:
                ans.append(op)
        return ''.join(ans)

    def get_abstract_operator(self, question: str) -> int:
        ans = [4]
        for key, value in self.op_mapping.items():
            if key in question:
                ans.append(value[0])
        return min(ans)

    def get_abstract_operator_text(self, question: str) -> str:
        ans = []
        for op in self.operator_set:
            if op in question:
                ans.append(self.op_mapping[op][1])
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
            'label': ans,
            'op_len': 0,
            'op_abstract': 0,
            'qid': piece["qid"]
        }


class Collator(object):
    def __call__(self, batch: Dict):
        input_ids = [each['input'] for each in batch]
        masks = [each['mask'] for each in batch]
        labels = [each['label'] for each in batch]
        op_len = [each['op_len'] for each in batch]
        op_abstract = [each['op_abstract'] for each in batch]
        qid = [each['qid'] for each in batch]

        max_len = max([len(each) for each in input_ids])

        for i in range(len(input_ids)):
            input_ids[i].extend([1] * (max_len - len(input_ids[i])))
            masks[i].extend([0] * (max_len - len(masks[i])))

        input_ids = torch.tensor(input_ids, dtype=torch.int32)
        masks = torch.tensor(masks, dtype=torch.int32)
        labels = torch.tensor(labels, dtype=torch.int32)
        op_len = torch.tensor(op_len, dtype=torch.int32)
        op_abstract = torch.tensor(op_abstract, dtype=torch.int32)

        return {
            'input_ids': input_ids,
            'masks': masks,
            'labels': labels,
            'op_len': op_len,
            'op_abstract': op_abstract,
            'qid': qid
        }