#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import json
import math
import faiss

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pytorch_pretrained_bert.optimization import BertAdam
from transformers import RobertaTokenizerFast, RobertaConfig, RobertaModel, BertPreTrainedModel

from .pred_data import InputExample, LOUVREEffPredDataset, post_process_batch

from .tfidf_keyword_wiki.model import TfidfKeywordWikiRetriever
from .tfidf_keyword_wiki.doc_db import DocDB        

logger = logging.getLogger(__name__)

class ModelConfig:
    def __init__(self,
                 encoder_path,
                 max_q_seq_length,
                 max_ctx_seq_length,
                 learning_rate,
                 warmup_proportion,
                 t_total):
        self.encoder_path = encoder_path
        self.max_q_seq_length = max_q_seq_length
        self.max_ctx_seq_length = max_ctx_seq_length
        self.learning_rate = learning_rate
        self.warmup_proportion = warmup_proportion
        self.t_total = t_total
    
    def __str__(self):
        configStr = '\n\n' \
            '### Model configurations ###\n' \
            '- Encoder path: ' \
                + str(self.encoder_path) + '\n' \
            '- Max Q seq length: ' \
                + str(self.max_q_seq_length) + '\n' \
            '- Max CTX seq length: ' \
                + str(self.max_ctx_seq_length) + '\n' \
            '- Learning rate: ' \
                + str(self.learning_rate) + '\n' \
            '- Warmup ratio: ' \
                + str(self.warmup_proportion) + '\n' \
            't_total: ' \
                + str(self.t_total) + '\n' \
            '#########################################\n'
        return configStr

class LOUVREEffRetrieverConfig:
    def __init__(self,
                 encoder_path,
                 max_q_seq_length,
                 max_ctx_seq_length, 
                 k, db_path, pred_batch_size, \
                 indexing_batch_size, \
                 tfidf_path):
        self.encoder_path = encoder_path
        self.max_q_seq_length = max_q_seq_length
        self.max_ctx_seq_length = max_ctx_seq_length
        self.k = k
        self.db_path = db_path
        self.pred_batch_size = pred_batch_size
        self.indexing_batch_size = indexing_batch_size
        self.tfidf_path = tfidf_path
    
    def __str__(self):
        configStr = '\n\n' \
            '### Model configurations ###\n' \
            '- Encoder path: ' \
                + str(self.encoder_path) + '\n' \
            '- Max Q seq length: ' \
                + str(self.max_q_seq_length) + '\n' \
            '- Max CTX seq length: ' \
                + str(self.max_ctx_seq_length) + '\n' \
            '- Beam k: ' \
                + str(self.k) + '\n' \
            '- DB path: ' \
                + str(self.db_path) + '\n' \
            '- TF-IDF path: ' \
                + str(self.tfidf_path) + '\n' \
            '- Pred batch size: ' \
                + str(self.pred_batch_size) + '\n' \
            '- Indexing batch size: ' \
                + str(self.indexing_batch_size) + '\n' \
            '#########################################\n'
        return configStr

class LOUVREEffEncoder(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = 'roberta'

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.layernorm = \
            torch.nn.LayerNorm( \
                config.hidden_size, \
                eps=config.layer_norm_eps)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs_roberta = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ).last_hidden_state
        cls_vecs = outputs_roberta[:,0]
        outputs = self.layernorm(cls_vecs)
        return outputs

class LOUVREEff:
    def __init__(self, config):
        self.config = config
        self.learning_rate = config.learning_rate
        self.warmup_proportion = config.warmup_proportion
        self.t_total = config.t_total
        self.tokenizer = \
            RobertaTokenizerFast \
                .from_pretrained( \
                    self.config.encoder_path)
        self.encoder = LOUVREEffEncoder \
            .from_pretrained( \
                config.encoder_path, \
                return_dict=True)
        self.optimizer = self.build_optimizer()
        self.cpu = torch.device("cpu")
        
        self.device = self.cpu
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.n_gpu = torch.cuda.device_count()

        self.encoder.to(self.device)
        if self.n_gpu > 1:
            self.encoder \
                = torch.nn.DataParallel( \
                    self.encoder)
        self.eval()
    
    def build_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = BertAdam( \
            optimizer_grouped_parameters, \
            lr=self.learning_rate, \
            warmup=self.warmup_proportion, \
            t_total=self.t_total, \
            max_grad_norm=2.0)
        return optimizer
    
    def update_weight(self):
        self.optimizer.step()
        self.encoder.zero_grad()
        return 0

    def named_parameters(self):
        return self.encoder.named_parameters()

    def train(self):
        self.encoder.train()
        return 0

    def eval(self):
        self.encoder.eval()
        return 0

    def zero_grad(self):
        self.encoder.zero_grad()
        return 0

    def get_loss_and_backward(self, batch, gradient_accumulation_steps):
        self.train()
        q_input_ids = batch['q_input_ids']
        q_input_masks = batch['q_input_masks']
        pos_ctx_input_ids = batch['pos_ctx_input_ids']
        pos_ctx_input_masks = batch['pos_ctx_input_masks']
        positive_indices = batch['positive_indices']
        neg_ctx_input_ids = batch['neg_ctx_input_ids'] if "neg_ctx_input_ids" in batch else None
        neg_ctx_input_masks = \
            batch['neg_ctx_input_masks'] \
                if "neg_ctx_input_masks" in batch else None
        
        loss = self.get_loss( \
            q_input_ids,  \
            q_input_masks, \
            pos_ctx_input_ids,  \
            pos_ctx_input_masks, \
            positive_indices, \
            neg_ctx_input_ids,  \
            neg_ctx_input_masks)
        if self.n_gpu > 1:
            loss = loss.mean()
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()
        tr_loss = loss.item()
        return tr_loss
    
    def encode(self, \
               input_ids, \
               input_masks, \
               training=False):
        input_ids_ = input_ids.to(self.device)
        input_masks_ = input_masks.to(self.device)
        
        vecs = None
        if not training:
            with torch.no_grad():
                vecs = self.encoder( \
                        input_ids_,
                        input_masks_)
        else:
            vecs = self.encoder( \
                    input_ids_,
                    input_masks_)
        return vecs

    def get_loss(self, \
                 q_input_ids, \
                 q_input_masks, \
                 pos_ctx_input_ids, \
                 pos_ctx_input_masks, \
                 positive_indices, \
                 neg_ctx_input_ids, \
                 neg_ctx_input_masks):
        pos_ctx_vec = self.encode( \
            pos_ctx_input_ids, \
            pos_ctx_input_masks, training=True)
        
        neg_ctx_vec = None
        if neg_ctx_input_ids != None:
            neg_ctx_vec = self.encode( \
                neg_ctx_input_ids, \
                neg_ctx_input_masks, training=True)
        ctx_vec = None
        if neg_ctx_vec != None:
            ctx_vec = torch.cat((pos_ctx_vec, neg_ctx_vec), dim=0)
        else:
            ctx_vec = pos_ctx_vec
        qvec = self.encode( \
            q_input_ids, \
            q_input_masks, \
            training=True)
        
        scores = torch.matmul( \
            qvec, torch.transpose(ctx_vec, 0, 1))
        softmax_scores = F.log_softmax(scores, dim=1)
        loss = F.nll_loss(softmax_scores, \
                          positive_indices \
                            .to(softmax_scores.device), \
                          reduction='mean')
        return loss

    def save_model(self, args):
        model = self.encoder
        output_dir = \
            os.path.join(args.train_output_dir)
        if not os.path.exists(output_dir):
            os.makedirs( \
                output_dir, exist_ok=True)
        model_to_save = \
            model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(args.train_output_dir)

        torch.save(args, os.path.join( \
            args.train_output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", args.train_output_dir)
        return 0

class LOUVREEffRetrieverWikiKeyword:
    def __init__(self, config):
        self.config = config
        self.max_ctx_seq_length \
            = config.max_ctx_seq_length
        self.max_q_seq_length \
            = config.max_q_seq_length
        self.encoder_path \
            = self.config.encoder_path
        self.k = self.config.k
        self.tokenizer = \
            RobertaTokenizerFast \
                .from_pretrained( \
                    self.config.encoder_path)

        self.cpu = torch.device("cpu")
        self.device = self.cpu
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.n_gpu = torch.cuda.device_count()
        
        self.encoder = LOUVREEffEncoder \
            .from_pretrained( \
                config.encoder_path, \
                return_dict=True)
        self.encoder.to(self.device)
        self.dimension = self.encoder.config.hidden_size
        if self.n_gpu > 1:
            self.encoder \
                = torch.nn.DataParallel( \
                    self.encoder)
        self.eval()

        self.db = DocDB(self.config.db_path)
        self.doc_id2vec = {}
        self.tfidf_keyword_ranker = \
            TfidfKeywordWikiRetriever( \
                self.config.db_path, \
                self.config.tfidf_path, \
                k=20)
        
    def eval(self):
        self.encoder.eval()
        return 0
    
    def encode(self, input_ids, input_masks):
        input_ids_ = input_ids.to(self.device)
        input_masks_ = \
            input_masks.to(self.device)
        with torch.no_grad():
            vecs = self.encoder( \
                    input_ids_,
                    input_masks_)
        return vecs
    
    def get_doc_text(self, doc_id):
        txt = self.db.get_doc_text(doc_id)
        if txt == None:
            return txt
        txt = " ".join([s.strip() for s in txt.split("\t")])
        return txt

    def get_text_vectors(self, texts, max_seq_length, batch_size):
        pred_data = \
            LOUVREEffPredDataset(\
                self.encoder_path, \
                max_seq_length, \
                texts)
        dataloader = DataLoader( \
            pred_data, \
            batch_size \
                = batch_size, \
            collate_fn=post_process_batch)
        vectors = np.zeros((0, self.dimension))
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids']
            input_masks = batch['input_masks']
            vecs = self.encode(input_ids, input_masks)
            vecs = vecs.detach().cpu().numpy()
            vectors = np.concatenate((vectors, vecs))
        return vectors
    
    def get_ctx_vec(self, doc_ids):
        missing_doc_ids = [doc_id for doc_id in doc_ids if doc_id not in self.doc_id2vec]
        ind2doc_id = {}
        chunk_size = 50000
        n_chunk = math.ceil(len(missing_doc_ids) / chunk_size)
        for i in range(n_chunk):
            si = i * chunk_size
            ei = min((i + 1) * chunk_size, len(missing_doc_ids))
            
            chunk = missing_doc_ids[si:ei]
            examples = []
            for doc_id in chunk:
                txt1 = self.get_doc_text(doc_id)
                examples.append((InputExample( \
                    txt1=txt1, txt2=None), doc_id))
            examples.sort(key=lambda x: len(x[0].txt1))
            passages = [e[0] for e in examples]
            chunk = [e[1] for e in examples]
            vecs = self.get_text_vectors( \
                passages, \
                self.max_ctx_seq_length, \
                self.config.indexing_batch_size) \
                    .astype('float32')
            self.doc_id2vec.update({doc_id: vec for doc_id, vec in zip(chunk, vecs)})
        vecs = np.array([self.doc_id2vec[doc_id] for doc_id in doc_ids])
        return vecs
    
    def predict(self, queries):
        n_hop = 2
        preds = {}
        chunk_size = 10
        n_chunk = math.ceil(len(queries) / chunk_size)
        for chunk_ind in tqdm(range(n_chunk), desc="Predicting"):
            s = chunk_ind * chunk_size
            e = min((chunk_ind + 1) * chunk_size, len(queries))

            questions = queries[s:e]
            beams = {
                q['_id']: [{ \
                    'qid': q['_id'],
                    'score': 0.0,
                    'path': [(None, None)],
                    'question': q['question'],
                    'candidates': [],
                    'prev_topk': []
                }] for q in questions \
            }

            for hop in range(n_hop):
                beams = self.predict_one_hop(beams, self.k)
            preds.update(beams)
        
        results = []
        for query in tqdm(queries, "Building selector outputs"):
            pred = preds[query['_id']]
            candidate_chains = []
            context = {}
            for b in pred:
                chain = []
                for doc_id, passage in b['path'][1:]:
                    title = doc_id[:-2]
                    chain.append({"title": title, "text": passage})
                    context[title] = passage
                candidate_chains.append(chain)
            result = {
                '_id': query['_id'],
                'question': query['question'],
                'candidate_chains': candidate_chains,
                'context': context,
            }
            results.append(result)
        return results

    def predict_one_hop(self, qid2beams, k):
        def merge_path(path):
            targets = [ \
                passage.strip() \
                    for title, passage in path \
                        if title != None \
            ]
            if len(targets) == 0:
                return None
            txt = " ".join(targets).strip()
            return txt
        
        beams = []
        for qid in qid2beams:
            beams += qid2beams[qid]
        
        for ind in range(len(beams)):
            q = beams[ind]['question']
            qid = beams[ind]['qid']
            prev_doc_id = beams[ind]['path'][-1][0]
            history = [p[0] for p in beams[ind]['path']]
            if prev_doc_id == None:
                paths = self.tfidf_keyword_ranker.predict( \
                    [{"question": q, "id": qid}], print_progress_bar=False)[0]["topk_titles"]
                candidates = list(set([title for path in paths for title in path]))
                candidates = [ \
                    doc_id for doc_id in candidates \
                            if doc_id is not None \
                                and doc_id.strip() != '' \
                                and self.get_doc_text(doc_id) != None]
                beams[ind]['candidates'] = candidates
            else:
                hyperlinked = \
                    self.db.get_hyper_linked(prev_doc_id)
                if hyperlinked is None:
                    hyperlinked = []
                hyperlinked = [ \
                    "{:s}_0".format(title) \
                        if "_0" not in title \
                            else title \
                    for title in hyperlinked \
                ]
                hyperlinked = [ \
                    doc_id for doc_id in hyperlinked  \
                        if self.get_doc_text(doc_id) != None \
                ]
                beams[ind]['candidates'] = list(set(hyperlinked + beams[ind]['prev_topk']) - set(history))
        
        all_candidates = []
        chunk_inds = []
        for beam in beams:
            chunk_inds.append((len(all_candidates), len(all_candidates) + len(beam['candidates'])))
            all_candidates += beam['candidates']
        all_candidate_vecs = self.get_ctx_vec(all_candidates)
        
        reformed_questions = []
        for beam in beams:
            q = beam['question']
            p = merge_path(beam['path'])
            reformed_question = \
                InputExample(txt1=q, txt2=p)
            reformed_questions.append( \
                reformed_question)
        
        predictions \
            = np.array(self.get_text_vectors( \
                reformed_questions, \
                self.max_q_seq_length, \
                self.config.pred_batch_size)) \
                    .astype('float32')
        
        scores, inds = [], []
        for pred, chunk_se in zip(predictions, chunk_inds):
            candidate_vecs = all_candidate_vecs[chunk_se[0]:chunk_se[1]]
            if len(candidate_vecs) == 0:
                inds.append([0] * (2*k))
                scores.append([-np.inf] * (2*k))
                continue
            topk_scores = np.dot(candidate_vecs, pred)
            topk_inds = np.argsort(-topk_scores)[:k]
            topk_scores = topk_scores[topk_inds]
            inds.append(topk_inds)
            scores.append(topk_scores)
        
        next_titles_scores = [ \
            [(beam['candidates'][i], s) for i, s in zip(cands, score)] \
                for beam, score, cands in zip(beams, scores, inds) \
        ]
        
        new_beams = []
        for beam, next_title_score in zip(beams, next_titles_scores):
            qid = beam['qid']
            score = beam['score']
            question = beam['question']
            path = beam['path']
            prev_topk = [next_title for next_title, next_score in next_title_score[:2]]
            for next_title, next_score in next_title_score:
                new_beams.append({ \
                    'qid': qid,
                    'score': score + next_score,
                    'question': question,
                    'path': path \
                            + [(next_title, \
                                self.get_doc_text(next_title))], \
                    'candidates': [], \
                    'prev_topk': prev_topk
                })
        
        beam_grouped_by_qid = {_['qid']: [] for _ in new_beams}
        for new_beam in new_beams:
            beam_grouped_by_qid[new_beam['qid']].append(new_beam)
        for qid in beam_grouped_by_qid:
            beam_grouped_by_qid[qid].sort(key=lambda x: x['score'], reverse=True)
            beam_grouped_by_qid[qid] = beam_grouped_by_qid[qid][:k]
        return beam_grouped_by_qid

