#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import json
import torch
import unicodedata
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast

logger = logging.getLogger(__name__)

def normalize(text):
    return unicodedata.normalize('NFD', text)

class LOUVREPredDataset(Dataset):
    def __init__(self, init_checkpoint):
        self.tokenizer = \
            RobertaTokenizerFast \
                .from_pretrained(init_checkpoint)
    
    def encode_para(self, title, text, max_len):
        return self.tokenizer( \
          title, \
          text_pair=text, \
          max_length=max_len, \
          truncation=True, \
          padding="max_length",
          return_tensors="pt" \
        )   
    
    def __len__(self):
        return len(self.data)

class LOUVREQDataset(LOUVREPredDataset):
    def __init__(self, \
                 init_checkpoint, \
                 max_q_seq_length, \
                 data):
        super().__init__(init_checkpoint)
        self.max_q_seq_length \
            = max_q_seq_length
        self.data = data
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        sent_codes = \
            self.encode_para( \
                sample[0], \
                sample[1], \
                self.max_q_seq_length)
        output = {
            "input_ids": sent_codes["input_ids"],
            "input_mask": sent_codes["attention_mask"],
            "question": sample
        }
        return output

class LOUVRECTXDataset(LOUVREPredDataset):
    def __init__(self, \
                 init_checkpoint, \
                 max_ctx_seq_length, \
                 fname):
        super().__init__(init_checkpoint)
        self.max_ctx_seq_length \
            = max_ctx_seq_length
        self.data = self.read_ctx(fname)
    
    def read_ctx(self, fname):
        with open(fname, "r") as f:
            data = json.load(f)
        l = len(data)
        data = [data[str(i)] for i in range(l)]
        return data
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if sample["text"].strip() == "":
            print(f"empty doc title: {sample['title']}")
            sample["text"] = sample["title"]
        sent_codes = \
            self.encode_para( \
                normalize(sample["title"].strip()), \
                sample['text'].strip(), \
                self.max_ctx_seq_length)
        output = {
            "input_ids": sent_codes["input_ids"],
            "input_mask": sent_codes["attention_mask"],
        }
        return output

def post_process_q_batch(samples):
    if len(samples) == 0:
        return {}
    batch = {
        'input_ids': torch.cat([ \
            s['input_ids'] \
                for s in samples \
        ], 0),
        'input_mask': torch.cat([ \
            s['input_mask'] \
                for s in samples \
        ], 0),
        'question': [ \
            s['question'] for s in samples \
        ]
    }
    return batch
    
def post_process_ctx_batch(samples):
    if len(samples) == 0:
        return {}
    batch = { 
        'input_ids': torch.cat([ \
            s['input_ids'] \
                for s in samples \
        ], 0),
        'input_mask': torch.cat([ \
            s['input_mask'] \
                for s in samples \
        ], 0),
    }
    return batch
    
