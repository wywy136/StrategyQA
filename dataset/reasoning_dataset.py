from typing import Dict, List, Tuple
import json
import pickle

import nltk
from transformers import RobertaTokenizer
import torch


class ReasoningDataset:
    def __init__(self, args, split: str = 'train'):
        self.arg = args
        if split == 'train':
            self.data: Dict = pickle.load(open(self.arg.reason_train, 'rb'))
            self.original = open(self.arg.train_path, 'r', encoding='utf-8')
        elif split == 'dev':
            self.data: Dict = pickle.load(open(self.arg.reason_dev, 'rb'))
            self.original = open(self.arg.dev_path, 'r', encoding='utf-8')
        else:
            self.data: Dict = pickle.load(open(self.arg.reason_test, 'rb'))
            self.original = open(self.arg.test_path, 'r', encoding='utf-8')
        self.original: List = json.load(self.original)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

        with open(self.arg.op_classification, 'r') as f:
            self.op_mapping = json.load(f)
        # self.op_mapping = {
        #     'greater': [0, 'comparison'],
        #     'less': [0, 'comparison'],
        #     'before': [0, 'comparison'],
        #     'after': [0, 'comparison'],
        #     'larger': [0, 'comparison'],
        #     'smaller': [0, 'comparison'],
        #     'higher': [0, 'comparison'],
        #     'lower': [0, 'comparison'],
        #     'longer': [0, 'comparison'],
        #     'shorter': [0, 'comparison'],
        #     'prior': [0, 'comparison'],
        #     'same': [0, 'comparison'],
        #     'identical': [0, 'comparison'],
        #     'equal': [0, 'comparison'],
        #     'different': [0, 'comparison'],
        #     'difference': [0, 'comparison'],
        #     'match': [0, 'comparison'],
        #     'considered': [0, 'comparison'],
        #     'least': [2, 'numerical'],
        #     'enough': [2, 'numerical'],
        #     'times': [2, 'numerical'],
        #     'plus': [2, 'numerical'],
        #     'multiplied': [2, 'numerical'],
        #     'divided': [2, 'numerical'],
        #     'and': [1, 'logical'],
        #     'or': [1, 'logical'],
        #     'all': [1, 'logical'],
        #     'also': [1, 'logical'],
        #     'both': [1, 'logical'],
        #     'included': [3, 'entail'],
        #     'include': [3, 'entail'],
        #     'overlap': [3, 'entail'],
        #     'listed': [3, 'entail'],
        #     'within': [3, 'entail'],
        #     'have': [3, 'entail'],
        #     'excluded': [3, 'entail'],
        #     'present': [3, 'entail'],
        #     'among': [3, 'entail'],
        #     'contain': [3, 'entail'],
        #     'absent': [3, 'entail'],
        #     'positive': [2, 'numerical']
        # }

    def get_abstract_operator(self, question: str) -> int:
        ans = [3]
        for key, value in self.op_mapping.items():
            if key in question:
                ans.append(value[0])
        return min(ans)

    def __len__(self) -> int:
        return len(self.original)

    def __getitem__(self, item: int) -> Dict:
        piece = self.original[item]
        qid = piece['qid']
        question = piece['question']
        entity_sents: List[Tuple[str, str]] = self.data[qid]
        if 'decomposition' in piece:
            operation = self.get_abstract_operator(piece['decomposition'][-1])
        else:
            operation = 0

        input = self.tokenizer(question)["input_ids"]
        masked_input = self.tokenizer(question)["input_ids"]
        for (entity, sent) in entity_sents:
            entity_words = nltk.word_tokenize(entity.lower().split('-')[0])
            sent_words = nltk.word_tokenize(sent)
            for word in sent_words:
                if word.lower() in entity_words:
                    sent_words.remove(word)
            masked_sent = ' '.join(sent_words)

            input += [2] + self.tokenizer(sent)["input_ids"]
            masked_input += [2] + self.tokenizer(masked_sent)["input_ids"]

        mask = [1] * len(input)
        masked_mask = [1] * len(masked_input)
        if "answer" in piece:
            ans = 1 if piece['answer'] else 0
        else:
            ans = 0

        return {
            'input': input[:self.arg.max_length],
            'masked_input': masked_input[:self.arg.max_length],
            'mask': mask[:self.arg.max_length],
            'masked_mask': masked_mask[:self.arg.max_length],
            'label': ans,
            'qid': qid,
            'op': operation
        }


class ReasoningCollator:
    def __call__(self, batch: Dict):
        input_ids = [each['input'] for each in batch]
        masks = [each['mask'] for each in batch]
        masked_input_ids = [each['masked_input'] for each in batch]
        masked_masks = [each['masked_mask'] for each in batch]
        labels = [each['label'] for each in batch]
        qid = [each['qid'] for each in batch]
        op = [each['op'] for each in batch]

        max_len = max([len(each) for each in input_ids])
        masked_max_len = max([len(each) for each in masked_input_ids])

        for i in range(len(input_ids)):
            input_ids[i].extend([1] * (max_len - len(input_ids[i])))
            masks[i].extend([0] * (max_len - len(masks[i])))

            masked_input_ids[i].extend([1] * (masked_max_len - len(masked_input_ids[i])))
            masked_masks[i].extend([0] * (masked_max_len - len(masked_masks[i])))

        input_ids = torch.tensor(input_ids, dtype=torch.int32)
        masks = torch.tensor(masks, dtype=torch.int32)
        masked_input_ids = torch.tensor(masked_input_ids, dtype=torch.int32)
        masked_masks = torch.tensor(masked_masks, dtype=torch.int32)
        labels = torch.tensor(labels, dtype=torch.int32)
        op = torch.tensor(op, dtype=torch.int32)

        return {
            'input_ids': input_ids,
            'masks': masks,
            'masked_input_ids': masked_input_ids,
            'masked_mask': masked_masks,
            'labels': labels,
            'qid': qid,
            'op': op
        }
