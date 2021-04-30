from torch.utils.data import DataLoader
import torch
from torch.nn import CrossEntropyLoss
from transformers import AdamW, get_linear_schedule_with_warmup

from model.golden_sentence_bert import GoldenSentenceBert
from dataset.squad import SquadDataset, SquadDatasetCollator
from config import Argument
from evaluator import Evaluator


class GoldenSentenceTrainer(object):
    def __init__(self):
        self.args = Argument
        self.device = torch.device('cuda') if self.args.cuda else torch.device('cpu')

        self.dataset = SquadDataset()
        # self.dev_dataset = dataset_dict[self.args.dataset]('dev')
        # self.test_dataset = Golden_Dataset('test')
        self.dataloader = None
        self.model = GoldenSentenceBert()
        self.model.to(self.device)

        # listed_params = list(self.model.named_parameters())
        # grouped_parameters = [
        #     {'params': [p for n, p in listed_params if 'bert' in n],
        #      'lr': self.args.tuning_rate,
        #      'weight_decay': self.args.weight_decay},
        #     {'params': [p for n, p in listed_params if 'bert' not in n],
        #      'weight_decay': self.args.weight_decay}
        # ]
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            correct_bias=False
        )
        total_steps = (self.args.epoch_num * len(self.dataset)) // self.args.batch_size
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.args.warmup_rate * total_steps),
            num_training_steps=total_steps
        )

        self.loss_fn = CrossEntropyLoss(ignore_index=-1)
        self.evaluator = Evaluator()
        self.max_acc = 0.

    def calculate_loss(self, logit, golden) -> torch.Tensor:
        logit = logit.reshape(-1, 2)
        golden = golden.reshape(-1, 2)
        return self.loss_fn(logit, golden)

    def save(self):
        print(f'Model saved at {self.args.model_path}')
        torch.save(self.model, self.args.model_path)

    def train(self):
        for epoch in range(self.args.epoch_num):

            # self.model.eval()
            # dev_dataloader = DataLoader(
            #     dataset=self.dev_dataset,
            #     batch_size=self.args.batch_size,
            #     num_workers=self.args.num_workers,
            #     collate_fn=Collator(),
            #     pin_memory=True if self.args.cuda else False,
            #     shuffle=False
            # )
            # print('Evaluating on Dev ...')
            # with torch.no_grad():
            #     acc = self.evaluator(dev_dataloader, self.model, self.device)
            # if acc > self.max_acc:
            #     print(f'Update! Dev performance: Accuracy {acc}')
            #     self.max_acc = acc
            #     self.save()

            self.model.train()
            self.dataloader = DataLoader(
                dataset=self.dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                collate_fn=SquadDatasetCollator(),
                pin_memory=True if self.args.cuda else False,
                shuffle=True
            )
            for index, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                for key, tensor in batch.items():
                    batch[key] = tensor.to(self.device)
                logits = self.model(
                    inputs=batch['inputs'],
                    masks=batch['masks']
                )
                loss = self.calculate_loss(logits, batch["labels"])
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if index % 100 == 0:
                    print(f'Epoch: {epoch}/{self.args.epoch_num}\tBatch: {index}/{len(self.dataloader)}\t'
                          f'Loss: {loss.item()}')

            self.save()