from typing import List


class Argument:
    train_path: str = "./data/strategyqa_train_train.json"
    test_path: str = "./data/strategyqa_train_test.json"
    dev_path: str = "./data/strategyqa_train_dev.json"
    corpus_path: str = "./data/strategyqa_train_paragraphs.json"
    fields: List = ["question"]
    max_length: int = 511
    boolq_path: str = './data/boolq/train.jsonl'
    twentyquestion_path: str = './data/twentyquestions/v1.0.twentyquestions.tar'

    cuda: bool = False
    epoch_num: int = 20
    batch_size: int = 8
    num_workers: int = 0
    learning_rate: float = 1e-3
    tuning_rate: float = 1e-5
    warmup_rate: float = 0.1
    weight_decay: float = 0.01