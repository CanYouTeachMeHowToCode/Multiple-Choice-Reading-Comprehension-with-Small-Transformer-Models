from functools import partial
from typing import Optional, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MultiheadAttention
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import transformers
import datasets
from transformers import (
    AlbertConfig,
    AlbertModel,
    AlbertTokenizer,
    AlbertTokenizerFast,
    AdamW,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from transformers.modeling_outputs import MultipleChoiceModelOutput
from rouge_score import rouge_scorer

class RACEMemDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path='albert-base-v2',
        datasets_loader='race',
        task_name='all',
        max_seq_length=160,
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=8,
        num_preprocess_processes=8,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.dataset_loader = datasets_loader
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.num_preprocess_processes = num_preprocess_processes
        
        self.tokenizer = AlbertTokenizerFast.from_pretrained(self.model_name_or_path, use_fast=True, do_lower_case=True)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
        self.dataset = None
        
    def setup(self, stage: Optional[str] = None):
        self.dataset = datasets.load_dataset(self.dataset_loader, self.task_name)
        preprocessor = partial(self.preprocess, self.tokenizer, self.max_seq_length)

        if stage == 'fit':
            for split in ['train', 'validation']:
                self.dataset[split] = self.dataset[split].map(
                    preprocessor,
                    remove_columns=['example_id'],
                    num_proc=self.num_preprocess_processes,
                )
                self.dataset[split].set_format(type='torch', 
                                               columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'list_num_sens', 'list_num_sens_cum'])
        else:
            for split in self.dataset.keys():
                self.dataset[split] = self.dataset[split].map(
                    preprocessor,
                    remove_columns=['example_id'],
                    num_proc=self.num_preprocess_processes,
                )
                self.dataset[split].set_format(type='torch',
                                               columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'list_num_sens', 'list_num_sens_cum'])
                  
    def prepare_data(self):
        datasets.load_dataset(self.dataset_loader, self.task_name)
        AlbertTokenizerFast.from_pretrained(self.model_name_or_path, use_fast=True)
        
    def train_dataloader(self):
        return DataLoader(self.dataset['train'],
                          sampler=RandomSampler(self.dataset['train']),
                          batch_size=self.train_batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset['validation'],
                          sampler=SequentialSampler(self.dataset['validation']),
                          batch_size=self.eval_batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'],
                          sampler=SequentialSampler(self.dataset['test']),
                          batch_size=self.eval_batch_size,
                          num_workers=self.num_workers)
    
    @staticmethod
    def preprocess(tokenizer: AlbertTokenizerFast, max_seq_length: int, x: Dict) -> Dict:
        choices_features = []
        list_num_sens = []

        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        question = x['question']
        article = x['article']

        option: str
        for option in x["options"]:
            if question.find("_") != -1:
                # fill in the banks questions
                question_option = question.replace("_", option)
            else:
                question_option = question + " " + option

            inputs = tokenizer(
                article,
                question_option,
                add_special_tokens=True,
                max_length=max_seq_length,
                truncation='only_first',
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding='max_length',
                return_tensors='pt',
            )
            sample_mapping = inputs.pop("overflow_to_sample_mapping")
            choices_features.append(inputs)
            list_num_sens.append(len(sample_mapping))

        labels = label_map.get(x["answer"], -1)
        label = torch.tensor(labels).long()

        return {
            "label": label,
            "input_ids":torch.cat([cf["input_ids"] for cf in choices_features]), # (total num_chunks for all options)
            "attention_mask": torch.cat([cf["attention_mask"] for cf in choices_features]),
            "token_type_ids": torch.cat([cf["token_type_ids"] for cf in choices_features]),
            "list_num_sens": torch.tensor(list_num_sens),
            "list_num_sens_cum": torch.cumsum(torch.tensor(list_num_sens), dim=0),
        }