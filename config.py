from typing import List


class Argument:
    pretrained_model_path: str = './pretrained_model/1_STAR_no_context.tar/1_STAR_no_context/weights.th'
    model_path: str = './checkpoints/ora_p_golden_sentence/1.pth'
    model_class: str = "ReasoningWithOperatorAbstract"

    train_path: str = "./data/train_gdsent.json"
    # test_path: str = "./data/strategyqa_train_test.json"
    dev_path: str = "./data/dev_gdsent.json"
    corpus_path: str = "./data/strategyqa_train_paragraphs.json"
    train_gdsent_path: str = "./data/train_gdsent.json"
    dev_gdsent_path: str = "./data/dev_gdsent.json"
    dataset: str = "golden_sentence_dataset"
    fields: List = ["question"]
    max_length: int = 512
    boolq_path: str = './data/boolq/train.jsonl'
    twentyquestion_path: str = './data/twentyquestions/v1.0.twentyquestions.tar'
    squad_train_path: str = './data/squad/train-v2.0.json'
    squad_dev_path: str = './data/squad/dev-v2.0.json'

    cuda: bool = False
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