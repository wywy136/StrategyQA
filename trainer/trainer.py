from torch.utils.data import DataLoader
import torch
from torch.nn import CrossEntropyLoss
from transformers import AdamW, get_linear_schedule_with_warmup

from model.roberta import Reasoning
from model.roberta_operator import ReasoningWithOperator
from model.roberta_operator_abstract import ReasoningWithOperatorAbstract
from dataset.golden_dataset import GoldenDataset, Collator
from dataset.golden_sentence_dataset import GoldenSentenceDataset
from dataset.last_step_dataset import LastStepDataset
from dataset.ir_avgcls_dataset import IrAvgClsDataset
from config import Argument
from evaluator import Evaluator
from predictor.ir_avgcls_predictor import IrAvrClsPredictor


dataset_dict = {
    "golden_dataset": GoldenDataset,
    "golden_sentence_dataset": GoldenSentenceDataset,
    "last_step_dataset": LastStepDataset,
    "ir_avgcls_dataset": IrAvgClsDataset
}
model_dict = {
    "Reasoning": Reasoning,
    "ReasoningWithOperator": ReasoningWithOperator,
    "ReasoningWithOperatorAbstract": ReasoningWithOperatorAbstract
}


class Trainer(object):
    def __init__(self):
        self.args = Argument
        self.device = torch.device('cuda') if self.args.cuda else torch.device('cpu')

        self.dataset = dataset_dict[self.args.train_dataset]()
        print(f'Dataset: {self.args.train_dataset}')
        self.dev_dataset = dataset_dict[self.args.dev_dataset]('dev')
        self.test_dataset = dataset_dict[self.args.test_dataset]('test')
        self.dataloader = None
        self.model = model_dict[self.args.model_class]()
        self.model.to(self.device)

        if self.args.load_pretrained:
            self.load_pretrained_state_dict()

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

        self.loss_fn = CrossEntropyLoss()
        self.evaluator = Evaluator()
        self.predictor = IrAvrClsPredictor(self.args.prediction_path)
        self.max_acc = 0.

    def load_pretrained(self):
        pretrained_model: torch.nn.Module = torch.load(self.args.pretrained_model_path, map_location=self.device)
        pretrained_params = [key for key, value in pretrained_model.named_parameters()]
        state_dict = self.model.state_dict()
        unloaded_params = []
        for key, value in self.model.named_parameters():
            if key in pretrained_params:
                state_dict[key] = pretrained_model.state_dict()[key]
            else:
                unloaded_params.append(key)
        self.model.load_state_dict(state_dict)
        print(f'The following parameters are not loaded from pretrained model: {unloaded_params}')

    def load_pretrained_state_dict(self):
        pretrained_state_dict = torch.load(self.args.pretrained_model_path, map_location=self.device)
        unloaded_params = []
        state_dict = self.model.state_dict()
        for name, param in state_dict.items():
            if self.convert_key_seqcls(name) in pretrained_state_dict:
                state_dict[name] = pretrained_state_dict[self.convert_key_seqcls(name)]
            else:
                unloaded_params.append(name)
        self.model.load_state_dict(state_dict)
        print(f'The following params are not loaded from pretrained model: {unloaded_params}')

    def convert_key_seqcls(self, original: str) -> str:
        return '_classifier' + original[7:]

    def convert_key(self, original: str) -> str:
        return '_classifier.' + original

    def save(self):
        print(f'Model saved at {self.args.model_path}')
        torch.save(self.model, self.args.model_path)

    def train(self):
        for epoch in range(self.args.epoch_num):

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
            print(f'Dev performance: Accuracy {acc}')
            if acc > self.max_acc:
                print(f'Update!')
                self.max_acc = acc
                self.save()

                test_dataloader = DataLoader(
                    dataset=self.test_dataset,
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers,
                    collate_fn=Collator(),
                    pin_memory=True if self.args.cuda else False,
                    shuffle=False
                )
                print(f'Generating predictions ...')
                with torch.no_grad():
                    self.predictor(test_dataloader, self.model, self.device)

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
                    if type(tensor) == torch.Tensor:
                        batch[key] = tensor.to(self.device)
                loss, logits = self.model(
                    input=batch['input_ids'].long(),
                    mask=batch['masks'],
                    label=batch['labels'].long(),
                    op_len=batch['op_len'],
                    op_abstract=batch['op_abstract'].long()
                )
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if index % 50 == 0:
                    print(f'Epoch: {epoch}/{self.args.epoch_num}\tBatch: {index}/{len(self.dataloader)}\t'
                          f'Loss: {loss.item()}')

                # test_dataloader = DataLoader(
                #     dataset=self.test_dataset,
                #     batch_size=self.args.batch_size,
                #     num_workers=self.args.num_workers,
                #     collate_fn=Collator(),
                #     pin_memory=True if self.args.cuda else False,
                #     shuffle=False
                # )
                # with torch.no_grad():
                #     acc = self.evaluator(test_dataloader, self.model, self.device)
                # print(f'Test performance: Accuracy {acc}')