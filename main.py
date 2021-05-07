from trainer.trainer import Trainer
from trainer.pretrainer import Pretrainer
from trainer.golden_sentence_trainer import GoldenSentenceTrainer
from golden_sentence_predictor.squad import SquadGoldenSentencePredictor


def main():
    # t = Trainer()
    # t.train()
    # t.pretrain()
    # t.save()
    m = GoldenSentenceTrainer()
    # m.load_pretrained()
    m.train()
    # m.save()
    # s = SquadGoldenSentencePredictor('dev')
    # s.predict()


if __name__ == '__main__':
    main()