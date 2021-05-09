from typing import Dict
from dataset.golden_dataset import GoldenDataset


class GoldenSentenceDataset(GoldenDataset):
    def __init__(self, split: str = 'train'):
        GoldenDataset.__init__(self, split)

    def __len__(self) -> int:
        return len(self.json_data)

    def __getitem__(self, item: int) -> Dict:
        piece = self.json_data[item]
        facts = piece['facts']
        decomposition = piece['decomposition']
        path = piece['evidence'][0]
        ret_dict = {
            "question": self.tokenizer(piece['question'])["input_ids"],
            "golden_sentence": [],
            "operation": self.tokenizer(self.get_abstract_operator_text(piece['question']))["input_ids"]
        }
        for step_index, step in enumerate(path):
            if step_index >= len(piece["golden_sentence"]):
                break
            for gdsent in piece["golden_sentence"][step_index]:
                ret_dict["golden_sentence"].append(self.tokenizer(gdsent)["input_ids"])
            # if step == ['operation']:
            #     continue
            # else:
                # for evidence in step:
                #     if evidence == "operation" or evidence == "no_evidence":
                #         continue
                #     for paragraph in evidence:
                #         golden_sentence = self.find(facts, self.json_corpus[paragraph]['content'])
                #         ret_dict["golden_sentence"].append(self.tokenizer(golden_sentence)["input_ids"])

        inputs = ret_dict["question"] + [2]
        for gd_sent in ret_dict["golden_sentence"]:
            inputs += gd_sent[1:] + [2]  # ret_dict["operation"]
        inputs += ret_dict["operation"]
        inputs = inputs[:self.arg.max_length]
        masks = [1] * len(inputs)
        seg = [0] * len(ret_dict["question"]) + [1] * (len(inputs) - len(ret_dict["question"]))
        ans = 1 if piece['answer'] else 0
        op_len = len(ret_dict["operation"])
        op_abstract = self.get_abstract_operator(piece['question'])

        return {
            'input': inputs,
            'mask': masks,
            "segment": seg,
            'label': ans,
            'op_len': op_len,
            'op_abstract': op_abstract
        }
