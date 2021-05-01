from typing import Dict, List
import json

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from nltk import sent_tokenize

from config import Argument


class SquadDataset(Dataset):
    def __init__(self, split):
        self.args = Argument
        if split == 'train':
            self.data: List[Dict] = json.load(open(self.args.squad_train_path, 'r', encoding='utf-8'))["data"]
        else:
            self.data: List[Dict] = json.load(open(self.args.squad_dev_path, 'r', encoding='utf-8'))["data"]
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.combined_data: List[Dict] = []
        for passage in self.data:
            for paragraph in passage["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph['qas']:
                    if not qa["is_impossible"]:
                        question = qa["question"]
                        answer = qa["answers"][0]["text"]
                        self.combined_data.append({
                            "question": question,
                            "context": context,
                            "answer": answer
                        })
                        # for sent_index, sent in enumerate(sent_tokenize(context)):
                        #     label = 0
                        #     if answer in sent:
                        #         label = 1
                        #     self.combined_data.append({
                        #         "question": question,
                        #         "answer": label,
                        #         "context": context,
                        #         "sent": sent
                        #     })

    def __len__(self) -> int:
        return len(self.combined_data)

    def __getitem__(self, item) -> Dict:
        piece = self.combined_data[item]
        # inputs, masks, labels = [], [], []
        inputs = self.tokenizer.convert_tokens_to_ids(
            ['[CLS]'] + self.tokenizer.tokenize(piece["question"]) + ['[SEP]']) + self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(piece['sent'])
        )
        masks = [1] * len(inputs)
        labels = piece["answer"]
        # for sent_index, sent in enumerate(sent_tokenize(piece["context"])):
        #     one_input = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + self.tokenizer.tokenize(piece["question"]) +
        #         ['[SEP]']) + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(piece["context"]) +
        #         ['[SEP]']) + self.tokenizer.convert_tokens_to_ids(self.tokenizer(sent))
        #     one_mask = [1] * len(one_input)
        #     inputs.append(one_input[:self.args.max_length])
        #     masks.append(one_mask[:self.args.max_length])
        #     labels.append(0)
        #     if piece["answer"] in sent:
        #         labels[-1] = 1

        return {
            "input": inputs,
            "mask": masks,
            "label": labels
        }


class SquadDatasetCollator(object):
    def __call__(self, batch: Dict):
        inputs: List[List] = [each["input"] for each in batch]
        masks: List[List] = [each["mask"] for each in batch]
        labels: List = [each["label"] for each in batch]

        max_seq_len = 0
        for i in range(len(inputs)):
            max_seq_len = max(max_seq_len, len(inputs[i]))
        # max_seq_num = 0
        # for one_input in inputs:
        #     for seq in one_input:
        #         max_seq_len = max(max_seq_len, len(seq))
        #     max_seq_num = max(max_seq_num, len(one_input))

        for i in range(len(inputs)):
            # for j in range(len(inputs[i])):
            #     inputs[i][j].extend([1] * (max_seq_len - len(inputs[i][j])))
            #     masks[i][j].extend([0] * (max_seq_len - len(masks[i][j])))
            # inputs[i].extend([[1] * max_seq_len] * (max_seq_num - len(inputs[i])))
            # masks[i].extend([[0] * max_seq_len] * (max_seq_num - len(masks[i])))
            # labels[i].extend([-1] * (max_seq_num - len(labels[i])))
            inputs[i].extend([0] * (max_seq_len - len(inputs[i])))
            masks[i].extend([0] * (max_seq_len - len(masks[i])))

        inputs: torch.Tensor = torch.tensor(inputs)  # [batch, max_seq_len]
        masks: torch.Tensor = torch.tensor(masks)  # [batch, max_seq_len]
        labels: torch.Tensor = torch.tensor(labels, dtype=torch.int64)  # [batch]

        return {
            "inputs": inputs,
            "masks": masks,
            "labels": labels
        }