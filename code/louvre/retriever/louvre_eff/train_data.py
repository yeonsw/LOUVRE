#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import random
from tqdm import tqdm
import json
import jsonlines
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast

from .tfidf_keyword_wiki.tfidf_doc_ranker import TfidfDocRanker
from .tfidf_keyword_wiki.doc_db import DocDB

logger = logging.getLogger(__name__)

class InputExample(object):
    def __init__(self, qid, q, passage, gold):
        self.qid = qid
        self.question = q
        self.passage = passage
        self.gold = gold

class InputFeatures(object):
    def __init__(self, \
                 qid, \
                 question, \
                 gold, \
                 q_input_ids, \
                 q_input_masks, \
                 gold_input_ids, \
                 gold_input_masks):
        self.qid = qid
        self.question = question
        self.gold = gold

        self.q_input_ids = q_input_ids
        self.q_input_masks = q_input_masks
        
        self.gold_input_ids = gold_input_ids
        self.gold_input_masks = gold_input_masks

class LOUVREEffTrainDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.db = DocDB(config.db_path)
        self.ranker = \
            TfidfDocRanker(\
                tfidf_path=config.tfidf_path, \
                strict=False)
        self.doc_ids = self.db.get_doc_ids()
        self.docid2text = {}
        for r in tqdm(self.doc_ids, "Building dictionary db"):
            self.docid2text[r] = self.get_doc_text(r)
        
        self.tokenizer = \
            RobertaTokenizerFast \
                .from_pretrained( \
                    self.config.encoder_path)
        self.neg_k = config.neg_k
        self.neg_n_init_doc = config.neg_n_init_doc
        self.max_q_seq_length \
            = self.config.max_q_seq_length
        self.max_ctx_seq_length \
            = self.config.max_ctx_seq_length
        self.train_examples = self.get_examples(self.config)
    
    def tokenize(self, txt1, txt2, max_seq_length):
        results = self.tokenizer( \
            txt1, text_pair=txt2, \
            max_length=max_seq_length, \
            truncation=True, \
            padding="max_length")
        input_ids = results["input_ids"]
        input_masks = results["attention_mask"]
        return (input_ids, input_masks)
    
    def get_doc_id(self, doc_id):
        if "_0" not in doc_id:
            doc_id = "{0}_0".format(doc_id)
        return doc_id
    
    def get_doc_text(self, doc_id):
        doc_id = self.get_doc_id(doc_id)
        text = self.db.get_doc_text(doc_id)
        
        if text is None:
            logger.warning( \
                "{0} is missing" \
                    .format(doc_id))
            return None
        return " ".join([s.strip() for s in text.split("\t")])
    
    def get_examples(self, config):
        if config.train_file.endswith(".jsonl"):
            with jsonlines.open(config.train_file) as reader:
                data = [r for r in reader]
        elif config.train_file.endswith(".json"):
            with open(config.train_file, "r") as f:
                data = json.load(f)
        else:
            raise Exception("File extension error")

        qid2questions = {}
        qid2supporting_docs = {}
        qid2answer = {}
        for d in data:
            qid2questions[d['_id']] = d['question']
            qid2answer[d['_id']] = d['answer']
            
            context = {
                c[0]: " ".join([s.strip() for s in c[1]]) for c in d['context']
            }
            qid2supporting_docs[d['_id']] = { \
                s[0]: context[s[0]] for s in d['supporting_facts'] \
            }
        
        examples = self._create_examples( \
            qid2questions, qid2supporting_docs, qid2answer)
        assert len(examples) > 0
        return examples
    
    def _create_examples(self, \
                         qid2questions, \
                         qid2supporting_docs, \
                         qid2answer):
        logger.info("### Reading train data... ###")
        examples = []
        for qid in tqdm(qid2questions, desc="Example"):
            q = qid2questions[qid]
            supporting_docs = qid2supporting_docs[qid]
            answer = qid2answer[qid]
            path = \
                [{"title": None, "text": None}] + sort_sdocs(supporting_docs, answer)
            for i, passage in enumerate(path[:-1]):
                target_doc = path[i + 1]
                examples.append( \
                    InputExample( \
                        qid=qid, \
                        q=q, \
                        passage=passage, \
                        gold=target_doc))
        return examples
    
    def convert_example_to_feature(self, example):
        q_input_ids, q_input_masks = \
            self.tokenize( \
                example.question, \
                example.passage["text"], \
                self.max_q_seq_length)
        
        gold_input_ids, gold_input_masks = \
            self.tokenize(example.gold["text"], \
                     None, \
                     self.max_ctx_seq_length)
        feature = InputFeatures( \
                qid=example.qid,
                question=example.question,
                gold=example.gold,
                q_input_ids=q_input_ids,
                q_input_masks=q_input_masks,
                gold_input_ids=gold_input_ids,
                gold_input_masks \
                    = gold_input_masks)
        return feature
    
    def get_strong_negatives(self, \
                             q_id, \
                             question, \
                             gold_doc_title):
        if self.neg_k == 0:
            return []
        gold_doc_id = self.get_doc_id(gold_doc_title)
        tfidf_docs, _ = self.ranker.closest_docs(question, k=self.neg_n_init_doc)
        tfidf_docs = [ \
            doc_id for doc_id in tfidf_docs \
                if doc_id != gold_doc_id \
        ]
        if len(tfidf_docs) == 0:
            tfidf_docs = random.sample(self.doc_ids, self.neg_n_init_doc)
        negs = random.sample(tfidf_docs, self.neg_k * 2)
        negs = [self.get_doc_id(n) for n in negs if self.get_doc_id(n) in self.docid2text][:self.neg_k]
        negs = [{"title": doc_id[:-2], "text": self.docid2text[doc_id]} for doc_id in negs]
        return negs
    
    def __len__(self):
        return len(self.train_examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        train_item = self.train_examples[idx]
        datapoint = self.convert_example_to_feature(train_item)
        
        qid = datapoint.qid
        question = datapoint.question
        gold_doc_title = datapoint.gold["title"]
        
        q_input_ids = torch.tensor( \
            datapoint.q_input_ids, dtype=torch.long)
        q_input_masks = torch.tensor( \
            datapoint.q_input_masks, dtype=torch.long)
        
        gold_input_ids = torch.tensor( \
            datapoint.gold_input_ids, dtype=torch.long)
        gold_input_masks = torch.tensor( \
            datapoint.gold_input_masks, dtype=torch.long)
        
        strong_negatives = \
            self.get_strong_negatives( \
                qid, question, gold_doc_title)
        
        k_neg_input_ids = []
        k_neg_input_masks = []
        for strong_negative in strong_negatives:
            neg_input_ids, neg_input_masks \
                = self.tokenize( \
                        strong_negative["text"], \
                        None, \
                        self.max_ctx_seq_length)
            k_neg_input_ids.append(neg_input_ids)
            k_neg_input_masks.append(neg_input_masks)

        k_neg_input_ids = torch.tensor( \
            k_neg_input_ids, dtype=torch.long)
        k_neg_input_masks = torch.tensor( \
            k_neg_input_masks, dtype=torch.long)
        
        example = {
            'q_input_ids': q_input_ids,
            'q_input_masks': q_input_masks,
            'gold_input_ids': gold_input_ids,
            'gold_input_masks': gold_input_masks,
            'neg_input_ids': k_neg_input_ids,
            'neg_input_masks': k_neg_input_masks,
        }
        return example

def sort_sdocs(sdocs, answer):
    path = [(title, sdocs[title]) for title in sdocs]
    path.sort(key=lambda x: x[1].find(answer))
    
    fst_title, fst_txt = path[0]
    sec_title, sec_txt = path[1]
   
    if (fst_txt.find(answer) > -1) == (sec_txt.find(answer) > -1):
        if fst_txt.find(sec_title) == -1 and sec_txt.find(fst_title) != -1:
            path = [ \
                (sec_title, sec_txt), \
                (fst_title, fst_txt)
            ]
    path = [ \
        {"title": title, "text": text} \
            for title, text in path \
    ]
    return path

def post_process_train_batch(batch):
    q_input_ids = torch.stack([b['q_input_ids'] for b in batch], dim=0)
    q_input_masks = torch.stack([b['q_input_masks'] for b in batch], dim=0)
   
    neg_ctx_input_ids = []
    neg_ctx_input_masks = []
    pos_ctx_input_ids = []
    pos_ctx_input_masks = []
    positive_indices = []
    for b in batch:
        positive_indices.append(len(pos_ctx_input_ids))
        pos_ctx_input_ids.append(b['gold_input_ids'])
        pos_ctx_input_masks.append(b['gold_input_masks'])
        
        neg_ctx_input_ids.append(b['neg_input_ids'])
        neg_ctx_input_masks.append(b['neg_input_masks']) 
    
    pos_ctx_input_ids = torch.stack(pos_ctx_input_ids, dim=0)
    pos_ctx_input_masks = torch.stack(pos_ctx_input_masks, dim=0)

    neg_ctx_input_ids = torch.cat(neg_ctx_input_ids, dim=0)
    neg_ctx_input_masks = torch.cat(neg_ctx_input_masks, dim=0)
    
    positive_indices = torch.tensor( \
            positive_indices, dtype=torch.long)
    
    batch_max_q_len = \
        q_input_masks.sum(dim=1).max().item()

    batch_max_ctx_len = None
    if len(neg_ctx_input_ids) == 0:
        batch_max_ctx_len = pos_ctx_input_masks.sum(dim=1).max().item()
    else:
        batch_max_ctx_len = \
            max(pos_ctx_input_masks.sum(dim=1).max().item(), \
                neg_ctx_input_masks.sum(dim=1).max().item())
    
    q_input_ids_ = \
        q_input_ids[:,:batch_max_q_len]
    q_input_masks_ = \
        q_input_masks[:,:batch_max_q_len]
    
    pos_ctx_input_ids_ = \
        pos_ctx_input_ids[:,:batch_max_ctx_len]
    pos_ctx_input_masks_ = \
        pos_ctx_input_masks[:,:batch_max_ctx_len]
    if len(neg_ctx_input_ids) == 0:
        new_batch = {
            'q_input_ids': q_input_ids_,
            'q_input_masks': q_input_masks_,
            'pos_ctx_input_ids': pos_ctx_input_ids_,
            'pos_ctx_input_masks': pos_ctx_input_masks_,
            'positive_indices': positive_indices
        }
        return new_batch
    neg_ctx_input_ids_ = \
        neg_ctx_input_ids[:,:batch_max_ctx_len]
    neg_ctx_input_masks_ = \
        neg_ctx_input_masks[:,:batch_max_ctx_len]
    new_batch = {
        'q_input_ids': q_input_ids_,
        'q_input_masks': q_input_masks_,
        'pos_ctx_input_ids': pos_ctx_input_ids_,
        'pos_ctx_input_masks': pos_ctx_input_masks_,
        'neg_ctx_input_ids': neg_ctx_input_ids_,
        'neg_ctx_input_masks': neg_ctx_input_masks_,
        'positive_indices': positive_indices
    }
    return new_batch
