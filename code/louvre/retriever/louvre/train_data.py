#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import random
import json
import jsonlines
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast

logger = logging.getLogger(__name__)

class LOUVREDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.tokenizer = \
            RobertaTokenizerFast \
                .from_pretrained( \
                    self.config.init_checkpoint)
        self.max_q_seq_length \
            = self.config.max_q_seq_length
        self.max_ctx_seq_length \
            = self.config.max_ctx_seq_length
        self.max_q_sp_seq_length \
            = self.config.max_q_sp_seq_length
        self.data = self.get_data(self.config.fname)
    
    def get_data(self, fname):
        data = [ \
            json.loads(line) \
                for line in open(fname) \
                                  .readlines() \
        ]
        data = [ \
            _ for _ in data \
                if len(_["neg_paras"]) >= 2 \
        ]
        return data
    
    def encode_para(self, para, max_len):
        return self.tokenizer.encode_plus( \
          para["title"].strip(), \
          text_pair=para["text"].strip(), \
          max_length=max_len, \
          truncation=True, \
          return_tensors="pt" \
        )
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        question = sample['question']
        if question.endswith("?"):
            question = question[:-1]
        if sample["type"] == "comparison":
            random.shuffle(sample["pos_paras"])
            start_para, bridge_para = sample["pos_paras"]
        else:
            for para in sample["pos_paras"]:
                if para["title"] != sample["bridge"]:
                    start_para = para
                else:
                    bridge_para = para
        random.shuffle(sample["neg_paras"])

        start_para_codes = \
            self.encode_para( \
                start_para, \
                self.max_ctx_seq_length)
        bridge_para_codes = \
            self.encode_para( \
                bridge_para, \
                self.max_ctx_seq_length)
        neg_codes_1 = \
            self.encode_para( \
                sample["neg_paras"][0], \
                self.max_ctx_seq_length)
        neg_codes_2 = \
            self.encode_para( \
                sample["neg_paras"][1], \
                self.max_ctx_seq_length)

        q_sp_codes = self.tokenizer.encode_plus( \
         question, \
         text_pair=start_para["text"].strip(), \
         max_length=self.max_q_sp_seq_length, \
         truncation=True, \
         return_tensors="pt" \
        )

        q_codes = self.tokenizer.encode_plus( \
            question, \
            max_length=self.max_q_seq_length, \
            truncation=True, \
            return_tensors="pt")
        return {
            "q_codes": q_codes,
            "q_sp_codes": q_sp_codes,
            "start_para_codes": start_para_codes,
            "bridge_para_codes": bridge_para_codes,
            "neg_codes_1": neg_codes_1,
            "neg_codes_2": neg_codes_2,
        }

def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    if len(values[0].size()) > 1:
        values = [v.view(-1) for v in values]
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res

def post_process_train_batch(samples):
    if len(samples) == 0:
        return {}
    batch = {
        'q_input_ids': collate_tokens([ \
            s["q_codes"]["input_ids"].view(-1) \
                for s in samples \
            ], 0),
        'q_input_mask': collate_tokens([ \
            s["q_codes"]["attention_mask"].view(-1) \
                for s in samples \
            ], 0),
        'q_sp_input_ids': collate_tokens([ \
            s["q_sp_codes"]["input_ids"].view(-1) \
                for s in samples \
            ], 0),
        'q_sp_input_mask': collate_tokens([ \
            s["q_sp_codes"]["attention_mask"].view(-1) \
                for s in samples \
            ], 0),
        'pos_ctx1_input_ids': collate_tokens([ \
            s["start_para_codes"]["input_ids"] \
                for s in samples \
            ], 0),
        'pos_ctx1_input_mask': collate_tokens([ \
            s["start_para_codes"]["attention_mask"] \
                for s in samples \
            ], 0),
        'pos_ctx2_input_ids': collate_tokens([ \
            s["bridge_para_codes"]["input_ids"] \
                for s in samples \
            ], 0),
        'pos_ctx2_input_mask': collate_tokens([ \
            s["bridge_para_codes"]["attention_mask"] \
                for s in samples \
            ], 0),
        'neg_ctx1_input_ids': collate_tokens([ \
            s["neg_codes_1"]["input_ids"] \
                for s in samples \
            ], 0),
        'neg_ctx1_input_mask': collate_tokens([ \
            s["neg_codes_1"]["attention_mask"] \
                for s in samples \
            ], 0),
        'neg_ctx2_input_ids': collate_tokens([ \
            s["neg_codes_2"]["input_ids"] \
                for s in samples \
            ], 0),
        'neg_ctx2_input_mask': collate_tokens([ \
            s["neg_codes_2"]["attention_mask"] \
                for s in samples \
            ], 0),
        }
    return batch
