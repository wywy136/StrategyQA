import pickle
import json
from typing import Dict, List

from ftfy import fix_text
from transformers import RobertaTokenizer

from config import Argument


class IrAvgClsDataset:
    def __init__(self, split: str = 'dev'):
        self.arg = Argument
        self.original_data: List[Dict] = json.load(
            open(self.arg.dev_path, 'r', encoding='utf-8')
        ) if split == 'dev' else json.load(
            open(self.arg.test_path, 'r', encoding='utf-8')
        )
        self.qid_qtext = {}
        self.qid_ans = {}
        self.paraid_paratext = {}
        self.data_path = self.arg.ir_avgcls_dev_path if split == 'dev' else self.arg.ir_avgcls_test_path
        with open(self.data_path, 'r', encoding="utf-8") as f:
            content = f.read()
        self.data: Dict[Dict] = json.loads(fix_text(content))
        self.paraid_content: Dict[List] = pickle.load(open(self.arg.paraid_content_path, 'rb'))
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.qid_paraids = []
        for qid, paths in self.data.items():
            self.qid_paraids.append({
                "qid": qid,
                "para_ids": paths[0]["path"]
            })

        for piece in self.original_data:
            self.qid_qtext[piece["qid"]] = piece["question"]
            if "answer" in piece:
                self.qid_ans[piece["qid"]] = 1 if piece['answer'] else 0

    def __len__(self) -> int:
        return len(self.qid_paraids)

    def __getitem__(self, item: int) -> Dict:
        piece: Dict = self.qid_paraids[item]
        qid = piece["qid"]
        qtext = self.qid_qtext[piece["qid"]]
        input_ids = self.tokenizer(qtext)["input_ids"]
        for para_id in piece["para_ids"]:
            input_ids += [2] + self.tokenizer(self.paraid_content[para_id]["text"])["input_ids"][1:]
        input_ids = input_ids[:self.arg.max_length]
        masks = [1] * len(input_ids)
        ans = self.qid_ans[piece["qid"]] if self.qid_ans else 0

        return {
            'input': input_ids,
            'mask': masks,
            'label': ans,
            'op_len': 0,
            'op_abstract': 0,
            'qid': qid
        }