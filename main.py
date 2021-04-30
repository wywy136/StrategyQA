from trainer.trainer import Trainer
from trainer.pretrainer import Pretrainer
from trainer.golden_sentence_trainer import GoldenSentenceTrainer


def main():
    # t = Pretrainer()
    # t.pretrain()
    # t.save()
    m = GoldenSentenceTrainer()
    # m.load_pretrained()
    m.train()
    # m.save()


if __name__ == '__main__':
    main()