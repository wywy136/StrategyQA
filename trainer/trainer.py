from torch.utils.data import DataLoader
import torch
from torch.nn import CrossEntropyLoss
from transformers import AdamW, get_linear_schedule_with_warmup

from model.bert import Reasoning
from dataset.golden_dataset import Golden_Dataset, Golden_Collator
from config import Argument
from evaluator import Evaluator


class Trainer(object):
    def __init__(self):
        self.args = Argument
        self.device = torch.device('cuda') if self.args.cuda else torch.device('cpu')

        self.dataset = Golden_Dataset()
        self.test_dataset = Golden_Dataset('test')
        self.dataloader = None
        self.model = Reasoning()
        self.model.to(self.device)

        listed_params = list(self.model.named_parameters())
        grouped_parameters = [
            {'params': [p for n, p in listed_params if 'bert' in n],
             'lr': self.args.tuning_rate,
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in listed_params if 'bert' not in n],
             'weight_decay': self.args.weight_decay}
        ]
        self.optimizer = AdamW(
            grouped_parameters,
            lr=self.args.learning_rate,
            correct_bias=False
        )
        total_steps = (self.args.epoch_num * len(self.dataset)) // self.args.batch_size
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.args.warmup_rate * total_steps),
            num_training_steps=total_steps
        )

        self.loss_fn = CrossEntropyLoss()
        self.evaluator = Evaluator()

    def train(self):
        for epoch in range(self.args.epoch_num):

            self.model.train()
            self.dataloader = DataLoader(
                dataset=self.dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                collate_fn=Golden_Collator(),
                pin_memory=True if self.args.cuda else False,
                shuffle=True
            )
            for index, batch in enumerate(self.dataloader):
                for key, tensor in batch.items():
                    batch[key] = tensor.to(self.device)
                logits = self.model(
                    input=batch['input_ids'].long(),
                    mask=batch['masks']
                )
                loss = self.loss_fn(logits, batch['labels'].long())
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if index % 20 == 0:
                    print(f'Epoch: {epoch}/{self.args.epoch_num}\tBatch: {index}/{len(self.dataloader)}\t'
                          f'Loss: {loss.item()}')

            self.model.eval()
            test_dataloader = DataLoader(
                dataset=self.test_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                collate_fn=Golden_Collator(),
                pin_memory=True if self.args.cuda else False,
                shuffle=True
            )
            with torch.no_grad():
                acc = self.evaluator(test_dataloader, self.model, self.device)
            print(f'Evaluation Results: Accuracy {acc}')
