#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from tqdm import tqdm
import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast

logger = logging.getLogger(__name__)

class InputExample(object):
    def __init__(self, txt1, txt2):
        self.txt1 = txt1
        self.txt2 = txt2

class InputFeatures(object):
    def __init__(self, input_ids, input_masks):
        self.input_ids = input_ids
        self.input_masks = input_masks

class LOUVREEffPredDataset(Dataset):
    def __init__(self, \
                 encoder_path, \
                 max_seq_length, \
                 examples):
        self.tokenizer = \
            RobertaTokenizerFast \
                .from_pretrained(encoder_path)
        self.max_seq_length \
            = max_seq_length
        self.features = \
            self.convert_examples_to_features( \
                examples)
        assert len(examples) \
                    == len(self.features)
    
    def convert_examples_to_features(self, \
                                     examples):
        """
        example.txt1 \in {str}
        example.txt2 \in {str, None}
        """
        input_ids = []
        input_masks = []
        features = []
        
        chunk_size = 50000
        nchunk = math.ceil(len(examples) / chunk_size)
        for chunk_index in range(nchunk):
            si = chunk_index * chunk_size
            ei = min((chunk_index + 1) * chunk_size, len(examples))
            batch_examples = examples[si:ei]
            txts = [example.txt1 for example in batch_examples]
            text_pair = [example.txt2 for example in batch_examples]
            results = \
                self.tokenizer( \
                    txts, text_pair=None if any([t == None for t in text_pair]) else text_pair, \
                    max_length=self.max_seq_length, \
                    truncation=True, \
                    padding="max_length")
            input_ids += results["input_ids"]
            input_masks += results["attention_mask"]
        for input_id, input_mask in zip(input_ids, input_masks):
            features.append(
                InputFeatures( \
                    input_ids=input_id,
                    input_masks=input_mask))
        return features
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        datapoint = self.features[idx]
        
        input_ids = torch.tensor( \
            datapoint.input_ids, \
            dtype=torch.long)
        input_masks = torch.tensor( \
            datapoint.input_masks, \
            dtype=torch.long)
        
        example = {
            'input_ids': input_ids,
            'input_masks': input_masks,
        }
        return example

def post_process_batch(batch):
    input_ids = torch.stack([ \
        b['input_ids'] for b in batch \
    ], dim=0)
    input_masks = torch.stack([ \
        b['input_masks'] for b in batch \
    ], dim=0)
   
    batch_max_len = \
        input_masks.sum(dim=1).max().item()
    
    input_ids_ = \
        input_ids[:,:batch_max_len]
    input_masks_ = \
        input_masks[:,:batch_max_len]
    
    new_batch = {
        'input_ids': input_ids_,
        'input_masks': input_masks_,
    }
    return new_batch
