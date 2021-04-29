from trainer.trainer import Trainer
from trainer.pretrainer import Pretrainer


def main():
    # t = Pretrainer()
    # t.pretrain()
    # t.save()
    m = Trainer()
    # m.load_pretrained()
    m.train()
    # m.save()


if __name__ == '__main__':
    main()