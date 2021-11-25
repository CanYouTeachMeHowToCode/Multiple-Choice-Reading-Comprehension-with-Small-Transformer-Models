from functools import partial
from typing import Optional, Dict, Any, List

import pytorch_lightning as pl
import numpy as np
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

def separate_seq2(sequence_output, flat_input_ids):
    #sequence_output: (num_sens, seq_len, hidden_size)
    #flat_input_ids: (num_sens, seq_len)
    qa_seq_output = sequence_output.new(sequence_output.size()).zero_()
    qa_mask = torch.ones((sequence_output.shape[0], sequence_output.shape[1]),
                         device=sequence_output.device,
                         dtype=torch.bool)
    p_seq_output = sequence_output.new(sequence_output.size()).zero_()
    p_mask = torch.ones((sequence_output.shape[0], sequence_output.shape[1]),
                        device=sequence_output.device,
                        dtype=torch.bool)
    
    for i in range(flat_input_ids.size(0)): #loop over every truncks over a single question
        sep_lst = []
        for idx, e in enumerate(flat_input_ids[i]):
            if e == 3: #3 is the id for [SEP] token in ALBERTs
                sep_lst.append(idx)
        assert len(sep_lst) == 2
        p_seq_output[i, :sep_lst[0] - 1] = sequence_output[i, 1:sep_lst[0]] #(num_sens, seq_len, hidden_size)
        p_mask[i, :sep_lst[0] - 1] = 0
        qa_seq_output[i, :sep_lst[1] - sep_lst[0] - 1] = sequence_output[i, sep_lst[0] + 1: sep_lst[1]] #(num_sens, seq_len, hidden_size)
        qa_mask[i, :sep_lst[1] - sep_lst[0] - 1] = 0
    return qa_seq_output, p_seq_output, qa_mask, p_mask

class DUMALayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(DUMALayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attn_qa = MultiheadAttention(self.hidden_size, self.num_heads)
        self.attn_p = MultiheadAttention(self.hidden_size, self.num_heads)
        
    def forward(self, qa_seq, p_seq, qa_mask=None, p_mask=None):
        #qa_seq: (batch_size, seq_len, hidden_dim)
        qa_seq = qa_seq.permute([1, 0, 2]) # (batch_size, seq_len, hidden_dim) -> (seq_len, batch_size, hidden_dim)
        p_seq = p_seq.permute([1, 0, 2])
        qa_attn_by_p, _ = self.attn_qa(
            value=qa_seq, key=qa_seq, query=p_seq, key_padding_mask=qa_mask
        )
        p_attn_by_qa, _ = self.attn_p(
            value=p_seq, key=p_seq, query=qa_seq, key_padding_mask=p_mask
        )
        return qa_attn_by_p.permute([1, 0, 2]), p_attn_by_qa.permute([1, 0, 2]) # batch_size, seq_len, hidden_dim)

class MemNN(nn.Module):
    def __init__(self, vocab_size, embed_size, max_story_len, pad_id, hops=1, dropout=0.2, te=True, pe=True):
        super(MemNN, self).__init__()
        self.hops = hops
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.max_story_len = max_story_len #this is the number of chunks(sentences) for each article
        self.temporal_encoding = te
        self.position_encoding = pe
        self.dropout = nn.Dropout(dropout)
        self.pad_id = pad_id
        self.init_rng = 0.1
        
        # adjacent weight sharing
        #self.A = nn.ModuleList([nn.Embedding(vocab_size, embed_size) for _ in range(self.hops)])
        #initialize weight for A
        
        '''for i in range(len(self.A)):
            self.A[i].weight.data.normal_(0,init_rng)
            self.A[i].weight.data[0] = 0 # for padding index
        self.B = self.A[0]'''
        
        #layer-wise weight tying
        self.A = nn.ModuleList([nn.Linear(self.embed_size, self.embed_size)])
        self.A[-1].weight.data.normal_(0, self.init_rng) #init
        self.C = nn.ModuleList([nn.Linear(self.embed_size, self.embed_size)])
        self.C[-1].weight.data.normal_(0, self.init_rng)
        
        #weight tying, share parameters for A,C across layer
        for i in range(1, self.hops):
            self.A.append(self.A[-1])
            self.C.append(self.C[-1])
            
        #embedding for question
        self.B = nn.Linear(self.embed_size, self.embed_size)
        self.B.weight.data.normal_(0, 0.1)
        
        #output matrix
        self.W = nn.Parameter(nn.init.normal_(torch.empty(self.embed_size, self.embed_size), 0, 0.1))
        self.H = nn.Linear(self.embed_size, self.embed_size)
        self.H.weight.data.normal_(0, 0.1)
        
        # temporal matrix
        self.TA = nn.Parameter(nn.init.normal_(torch.empty(self.max_story_len+1, self.embed_size), 0, 0.1))
        self.TC = nn.Parameter(nn.init.normal_(torch.empty(self.max_story_len+1, self.embed_size), 0, 0.1))
    
    def forward(self, sentences, query):
        #first get u_0, embedding for query (query_len, hidden_dim)
        query_len = query.size(0)
        query_pe = self.compute_weights(query_len) #position embedding weight metrixs for query (query_len, embed_size)
        u = self.B(query) #(query_len, embed_size)
        u = (u * query_pe).sum(0) #(embed_size)
        
        #sentences: (B, num_sens, max_seq_len)
        num_sens, max_seq_len, hidden_size = sentences.shape
        #get positional embeddign
        sens_pe = self.compute_weights(max_seq_len)
        
        for i in range(self.hops):
            m = self.A[i](sentences) # (num_sens, max_seq_len, embed_size)
            m *= sens_pe # (num_sens, max_seq_len, embed_size)
            m = torch.sum(m, 1) # (num_sens, embed_size)
            ta = self.TA.repeat(1,1)[:num_sens, :]
            m += ta
            
            c = self.C[i](sentences) # (num_sens, max_seq_len, embed_size)
            c *= sens_pe
            c = torch.sum(c, 1)
            tc = self.TC.repeat(1, 1)[:num_sens, :] # (num_sens, embed_size)
            c += tc
            
            p = torch.mm(m, u.unsqueeze(1)).squeeze(1) # (num_sens)
            p = F.softmax(p, dim=-1).unsqueeze(0) #(1, num_sens)
            o = torch.mm(p, c).squeeze(0) # o: (embed_size)
            u = o + self.H(u)
            
        return F.linear(u, self.W) #return res: (embed_size)
            
    #return column vector for positional embedding within each sentence
    def compute_weights(self, J):
        d = self.embed_size
        func = lambda j, k: 1 - (j + 1) / J - (k + 1) / d * (1 - 2 * (j + 1) / J)
        weights = torch.from_numpy(np.fromfunction(func, (J, d), dtype=np.float32))
        #weights mi = sum_j(l_j * Ax_ij), j represent index in a sentence
        return weights.cuda() if torch.cuda.is_available() else weights
    
class ALBERTForRaceMem(pl.LightningModule):
    def __init__(
            self,
            pretrained_model: str = 'albert-base-v2',
            learning_rate: float = 2e-5,
            gradient_accumulation_steps: int = 1,
            num_train_epochs: float = 3.0,
            train_batch_size: int = 1,
            warmup_proportion: float = 0.1,
            train_all: bool = False,
            use_bert_adam: bool = True,
    ):
        super().__init__()
        self.config = AlbertConfig.from_pretrained(pretrained_model, num_choices=4)
        self.albert = AlbertModel.from_pretrained(pretrained_model, config=self.config)
        self.duma = DUMALayer(self.config.hidden_size, num_heads=self.config.num_attention_heads)
        self.mem_nn = MemNN(self.config.vocab_size, self.config.hidden_size, max_story_len=10, pad_id=0, hops=1, dropout=0.2, te=True, pe=True)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(1)
        ])
        self.classifier = nn.Linear(self.config.hidden_size, 1)
        
        if not train_all:
            for param in self.albert.parameters():
                param.requires_grad = False
            for param in self.albert.pooler.parameters():
                param.requires_grad = True
                
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size
        self.warmup_proportion = warmup_proportion
        self.use_bert_adam = use_bert_adam
        
        self.warmup_steps = 0
        self.total_steps = 0
        
    def setup(self, stage: str) -> None:
        if stage == 'fit':
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            self.total_steps = int(len(train_loader.dataset)
                                   / self.train_batch_size / self.gradient_accumulation_steps * self.num_train_epochs)
            self.warmup_steps = int(self.total_steps * self.warmup_proportion)
        
    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
        ) if self.use_bert_adam else torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=self.learning_rate
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )
        
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
        }
        return [optimizer], [scheduler]
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            list_num_sens=None,
            list_num_sens_cum=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = len(list_num_sens) if list_num_sens is not None else 4
        
        input_ids = torch.vstack(input_ids)
        attention_mask = torch.vstack(attention_mask)
        token_type_ids = torch.vstack(token_type_ids)
        list_num_sens = list_num_sens_cum.to(torch.long) #cumulated sum of sentence length for each question

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        list_num_sens = list_num_sens.flatten()

        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        last_output = outputs.last_hidden_state #(total_num_sens, max_seq_len, albert_embed_size)
        qa_seq_output, p_seq_output, qa_mask, p_mask = separate_seq2(last_output, input_ids)

        #filter out relative representations for each option
        opt_embeds = []
        for idx, cum_num_sens in enumerate(list_num_sens):
            if idx == 0:
                story = p_seq_output[:cum_num_sens, :, :] # (num_sens_per_question, max_seq_len (160), hidden_size (768))
                query = torch.mean(qa_seq_output[:cum_num_sens, :, :], dim=0) # (max_seq_len (160), hidden_size (768))
            else:
                story = p_seq_output[list_num_sens[idx-1]:cum_num_sens, :, :]
                query = torch.mean(qa_seq_output[list_num_sens[idx-1]:cum_num_sens, :, :], dim=0)

            option_embed = self.mem_nn(story, query)
            opt_embeds.append(option_embed)
        opt_embeds = torch.stack(opt_embeds) #(4, hidden_dim)
        
        #start to apply dropout
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logits = self.classifier(dropout(opt_embeds))
            else:
                logits += self.classifier(dropout(opt_embeds))
        
        logits = logits / len(self.dropouts)
        reshaped_logits = F.softmax(logits.view(-1, 4), dim=1)
        
        #calculate classification loss
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def compute(self, batch):
        #evaluating the performance of the model
        outputs = self(
            input_ids=batch['input_ids'],
            token_type_ids=batch['token_type_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['label'],
            list_num_sens=batch['list_num_sens'],
            list_num_sens_cum=batch['list_num_sens_cum'],
        )
        labels_hat = torch.argmax(outputs.logits, dim=1)
        correct_count = torch.sum(batch['label'] == labels_hat)
        return outputs.loss, correct_count
    
    def training_step(self, batch, batch_idx):
        loss, correct_count = self.compute(batch)
        self.log('train_loss', loss)
        self.log('train_acc', correct_count.float() / len(batch['label']))
        #print("finish a batch in train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, correct_count = self.compute(batch)
        
        return {
            "val_loss": loss,
            "correct_count": correct_count,
            "batch_size": len(batch['label'])
        }
    
    def validation_epoch_end(self, outputs: List[Any]) -> None:
        #print("finish eval")
        val_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
        val_loss = sum([out["val_loss"] for out in outputs]) / len(outputs)
        self.log('val_acc', val_acc)
        self.log('val_loss', val_loss)
        
    def test_step(self, batch, batch_idx):
        loss, correct_count = self.compute(batch)

        return {
            "test_loss": loss,
            "correct_count": correct_count,
            "batch_size": len(batch['label'])
        }
    
    def test_epoch_end(self, outputs: List[Any]) -> None:
        test_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
        test_loss = sum([out["test_loss"] for out in outputs]) / len(outputs)
        self.log('test_acc', test_acc)
        self.log('test_loss', test_loss)
