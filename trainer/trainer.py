from torch.utils.data import DataLoader
import torch
from torch.nn import CrossEntropyLoss
from transformers import AdamW, get_linear_schedule_with_warmup

from model.roberta import Reasoning
from dataset.golden_dataset import Golden_Dataset, Collator
from config import Argument
from evaluator import Evaluator


class Trainer(object):
    def __init__(self):
        self.args = Argument
        self.device = torch.device('cuda') if self.args.cuda else torch.device('cpu')

        self.dataset = Golden_Dataset()
        self.dev_dataset = Golden_Dataset('dev')
        self.test_dataset = Golden_Dataset('test')
        self.dataloader = None
        self.model = Reasoning()
        self.model.to(self.device)

        if self.args.load_pretrained:
            self.load_pretrained()

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
        self.max_acc = 0.

    def load_pretrained(self):
        pretrained_model: torch.nn.Module = torch.load(self.args.pretrained_model_path)
        pretrained_params = [key for key, value in pretrained_model.named_parameters()]
        loaded_params = []
        for key, value in self.model.named_parameters():
            if key in pretrained_params:
                self.model.state_dict()[key] = pretrained_model.state_dict()[key]
                loaded_params.append(key)
        print(f'The following parameters are loaded from pretrained model: {loaded_params}')

    def save(self):
        torch.save(self.model, self.args.model_path)

    def train(self):
        for epoch in range(self.args.epoch_num):

            self.model.train()
            self.dataloader = DataLoader(
                dataset=self.dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                collate_fn=Collator(),
                pin_memory=True if self.args.cuda else False,
                shuffle=True
            )
            for index, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                for key, tensor in batch.items():
                    batch[key] = tensor.to(self.device)
                loss, logits = self.model(
                    input=batch['input_ids'].long(),
                    mask=batch['masks'],
                    label=batch['labels'].long()
                )
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if index % 20 == 0:
                    print(f'Epoch: {epoch}/{self.args.epoch_num}\tBatch: {index}/{len(self.dataloader)}\t'
                          f'Loss: {loss.item()}')

            self.model.eval()

            dev_dataloader = DataLoader(
                dataset=self.dev_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                collate_fn=Collator(),
                pin_memory=True if self.args.cuda else False,
                shuffle=False
            )
            print('Evaluating on Dev ...')
            with torch.no_grad():
                acc = self.evaluator(dev_dataloader, self.model, self.device)
            if acc > self.max_acc:
                print(f'Update! Dev performance: Accuracy {acc}')
                self.max_acc = acc

                test_dataloader = DataLoader(
                    dataset=self.test_dataset,
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers,
                    collate_fn=Collator(),
                    pin_memory=True if self.args.cuda else False,
                    shuffle=False
                )
                with torch.no_grad():
                    acc = self.evaluator(test_dataloader, self.model, self.device)
                print(f'Test performance: Accuracy {acc}')