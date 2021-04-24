from typing import List


class Argument:
    pretrained_model_path: str = './pretrained_model/1.pth'
    model_path: str = './checkpoints/2.pth'

    train_path: str = "./data/strategyqa_train_train.json"
    test_path: str = "./data/strategyqa_train_test.json"
    dev_path: str = "./data/strategyqa_train_dev.json"
    corpus_path: str = "./data/strategyqa_train_paragraphs.json"
    fields: List = ["question"]
    max_length: int = 511
    boolq_path: str = './data/boolq/train.jsonl'
    twentyquestion_path: str = './data/twentyquestions/v1.0.twentyquestions.tar'

    cuda: bool = True
    num_workers: int = 0
    load_pretrained: bool = True

    pretrain_epoch_num_boolq: int = 20
    pretrain_epoch_num_20q: int = 20
    pretrain_batch_size: int = 8
    pretrain_learning_rate: float = 1e-5

    epoch_num: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-5
    tuning_rate: float = 1e-5

    warmup_rate: float = 0.1
    weight_decay: float = 0.01