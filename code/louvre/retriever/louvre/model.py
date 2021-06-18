#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import apex
import faiss
import os
import argparse
import collections
import json
import logging
import math
import numpy as np
import random
import time
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, RobertaConfig, RobertaModel, BertPreTrainedModel, get_linear_schedule_with_warmup

from .pred_data import LOUVRECTXDataset, LOUVREQDataset, post_process_q_batch, post_process_ctx_batch
from .db.doc_db import DocDB

logger = logging.getLogger(__name__)

class ModelConfig:
    def __init__(self,
                 init_checkpoint,
                 max_q_seq_length,
                 max_ctx_seq_length,
                 max_q_sp_seq_length,
                 learning_rate,
                 warmup_proportion,
                 t_total,
                 k,
                 momentum,
                 max_grad_norm=2.0,
                 weight_decay=0.0):
        self.init_checkpoint = init_checkpoint
        self.max_q_seq_length = max_q_seq_length
        self.max_ctx_seq_length = max_ctx_seq_length
        self.max_q_sp_seq_length = max_q_sp_seq_length
        self.learning_rate = learning_rate
        self.warmup_proportion = warmup_proportion
        self.t_total = t_total
        self.k = k
        self.momentum = momentum
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
    
    def __str__(self):
        configStr = '\n\n' \
            '### Model configurations ###\n' \
            '- Encoder path: ' \
                + str(self.init_checkpoint) + '\n' \
            '- Max Q seq length: ' \
                + str(self.max_q_seq_length) + '\n' \
            '- Max CTX seq length: ' \
                + str(self.max_ctx_seq_length) + '\n' \
            '- Max Q SP seq length: ' \
                + str(self.max_q_sp_seq_length) + '\n' \
            '- Learning rate: ' \
                + str(self.learning_rate) + '\n' \
            '- Warmup prop: ' \
                + str(self.warmup_proportion) + '\n' \
            '- t total: ' \
                + str(self.t_total) + '\n' \
            '- k: ' \
                + str(self.k) + '\n' \
            '- Momentum: ' \
                + str(self.momentum) + '\n' \
            '- Max grad norm: ' \
                + str(self.max_grad_norm) + '\n' \
            '- Weight decay: ' \
                + str(self.weight_decay) + '\n' \
            '#########################################\n'
        return configStr

class LOUVRERetrieverConfig:
    def __init__(self,
                 init_checkpoint,
                 ctx_init_checkpoint,
                 corpus_fname,
                 max_q_seq_length,
                 max_ctx_seq_length, 
                 max_q_sp_seq_length,
                 pred_batch_size,
                 index_save_path,
                 saved_index_path,
                 topk, beam_size):
        self.init_checkpoint = init_checkpoint
        self.ctx_init_checkpoint = ctx_init_checkpoint
        self.corpus_fname = corpus_fname
        self.max_q_seq_length = max_q_seq_length
        self.max_ctx_seq_length = max_ctx_seq_length
        self.max_q_sp_seq_length = max_q_sp_seq_length
        self.pred_batch_size = pred_batch_size
        self.index_save_path = index_save_path
        self.saved_index_path = saved_index_path
        self.topk = topk
        self.beam_size = beam_size
    
        self.cstr = \
        '- Encoder path: ' \
            + str(self.init_checkpoint) + '\n' \
        '- CTX encoder path: ' \
            + str(self.ctx_init_checkpoint) + '\n' \
        '- Max Q seq length: ' \
            + str(self.max_q_seq_length) + '\n' \
        '- Max CTX seq length: ' \
            + str(self.max_ctx_seq_length) + '\n' \
        '- Max Q SP seq length: ' \
            + str(self.max_q_sp_seq_length) + '\n' \
        '- Pred batch size: ' \
            + str(self.pred_batch_size) + '\n' \
        '- Corpus file name: ' \
            + str(self.corpus_fname) + '\n' \
        '- Saved index path: ' \
            + str(self.saved_index_path) + '\n' \
        '- Index save path: ' \
            + str(self.index_save_path) + '\n' \
        '- Top K: ' \
            + str(self.topk) + '\n' \
        '- Beam size: ' \
            + str(self.beam_size) + '\n'
    def __str__(self):
        configStr = '\n\n' \
            '### Model configurations ###\n' \
            + self.cstr \
            + '#########################################\n'
        return configStr

class LOUVRERetrieverWikiConfig(LOUVRERetrieverConfig):
    def __init__(self,
                 init_checkpoint,
                 ctx_init_checkpoint,
                 corpus_fname,
                 max_q_seq_length,
                 max_ctx_seq_length, 
                 max_q_sp_seq_length,
                 pred_batch_size,
                 index_save_path,
                 saved_index_path,
                 topk, beam_size,
                 db_file):
        super().__init__( \
            init_checkpoint,
            ctx_init_checkpoint,
            corpus_fname,
            max_q_seq_length,
            max_ctx_seq_length, 
            max_q_sp_seq_length,
            pred_batch_size,
            index_save_path,
            saved_index_path,
            topk, beam_size,
        )
        self.db_file = db_file
        self.cstr = \
            self.cstr \
              + '- DB file: ' \
                + str(self.db_file) + '\n'
    
    def __str__(self):
        configStr = '\n\n' \
            '### Model configurations ###\n' \
            + self.cstr \
            + '#########################################\n'
        return configStr

class LOUVRE(BertPreTrainedModel):
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

class LOUVREEncoder(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = 'roberta'

    def __init__(self, config):
        super().__init__(config)
        self.louvre_config = config
        self.roberta = RobertaModel(config)
        self.layer1 = torch.nn.Linear( \
            config.hidden_size, config.hidden_size)
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
        cls_vecs = outputs_roberta[:, 0, :]
        outputs = self.layer1(cls_vecs)
        outputs = self.layernorm(outputs)
        return outputs

class LOUVREEncoderTrain(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = 'roberta'

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.layer1 = torch.nn.Linear( \
            config.hidden_size, config.hidden_size)
        self.layernorm = \
            torch.nn.LayerNorm( \
                config.hidden_size, \
                eps=config.layer_norm_eps)
        self.init_weights()
    
    def encode( \
        self,
        input_ids,
        attention_mask):
        
        return_dict = self.config.use_return_dict
        outputs_roberta = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ).last_hidden_state
        cls_vecs = outputs_roberta[:,0]
        outputs = self.layer1(cls_vecs)
        outputs = self.layernorm(outputs)
        return outputs

    def forward(self, batch):
        q_input_ids = batch['q_input_ids']
        q_input_mask = batch['q_input_mask']
        qvec = self.encode( \
            q_input_ids, \
            q_input_mask)
        
        pos_ctx1_input_ids = batch['pos_ctx1_input_ids']
        pos_ctx1_input_mask = batch['pos_ctx1_input_mask']
        pos_ctx1_vec = self.encode( \
            pos_ctx1_input_ids, \
            pos_ctx1_input_mask)
        
        q_sp_input_ids = batch['q_sp_input_ids']
        q_sp_input_mask = batch['q_sp_input_mask']
        q_sp_vec = self.encode( \
            q_sp_input_ids, \
            q_sp_input_mask)
        
        pos_ctx2_input_ids = batch['pos_ctx2_input_ids']
        pos_ctx2_input_mask = batch['pos_ctx2_input_mask']
        pos_ctx2_vec = self.encode( \
            pos_ctx2_input_ids, \
            pos_ctx2_input_mask)

        neg_ctx1_input_ids = batch['neg_ctx1_input_ids']
        neg_ctx1_input_mask = batch['neg_ctx1_input_mask']
        neg_ctx1_vec = self.encode( \
            neg_ctx1_input_ids, \
            neg_ctx1_input_mask)
        
        neg_ctx2_input_ids = batch['neg_ctx2_input_ids']
        neg_ctx2_input_mask = batch['neg_ctx2_input_mask']
        neg_ctx2_vec = self.encode( \
            neg_ctx2_input_ids, \
            neg_ctx2_input_mask)
    
        return {
            "pos_ctx1_vec": pos_ctx1_vec,
            "pos_ctx2_vec": pos_ctx2_vec,
            "neg_ctx1_vec": neg_ctx1_vec,
            "neg_ctx2_vec": neg_ctx2_vec,
            "qvec": qvec,
            "q_sp_vec": q_sp_vec
        }

class LOUVREEncoderMomentumTrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_encoder = LOUVREEncoder \
            .from_pretrained( \
                config.init_checkpoint, \
                return_dict=True)
        self.ctx_encoder = LOUVREEncoder \
            .from_pretrained( \
                config.init_checkpoint, \
                return_dict=True)

        for param in self.ctx_encoder.parameters():
            param.requires_grad = False

        self.k = config.k
        self.register_buffer("queue", torch.randn(self.k, self.q_encoder.louvre_config.hidden_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def encode( \
        self,
        input_ids,
        attention_mask,
        txt_type):
        encoder = self.q_encoder if txt_type == "q" else self.ctx_encoder
        vecs = encoder(
            input_ids,
            attention_mask,
        )
        return vecs
    
    @torch.no_grad()
    def dequeue_and_enqueue(self, embeddings):
        batch_size = embeddings.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.k:
            batch_size = self.k - ptr
            embeddings = embeddings[:batch_size]

        self.queue[ptr:ptr + batch_size, :] = embeddings

        ptr = (ptr + batch_size) % self.k
        self.queue_ptr[0] = ptr
        return

    def forward(self, batch):
        q_input_ids = batch['q_input_ids']
        q_input_mask = batch['q_input_mask']
        
        q_sp_input_ids = batch['q_sp_input_ids']
        q_sp_input_mask = batch['q_sp_input_mask']
        
        pos_ctx1_input_ids = batch['pos_ctx1_input_ids']
        pos_ctx1_input_mask = batch['pos_ctx1_input_mask']
        
        pos_ctx2_input_ids = batch['pos_ctx2_input_ids']
        pos_ctx2_input_mask = batch['pos_ctx2_input_mask']
        
        neg_ctx1_input_ids = batch['neg_ctx1_input_ids']
        neg_ctx1_input_mask = batch['neg_ctx1_input_mask']
        
        neg_ctx2_input_ids = batch['neg_ctx2_input_ids']
        neg_ctx2_input_mask = batch['neg_ctx2_input_mask']
        
        with torch.no_grad():
            pos_ctx1_vec = self.encode( \
                pos_ctx1_input_ids, \
                pos_ctx1_input_mask,
                "ctx")
            pos_ctx2_vec = self.encode( \
                pos_ctx2_input_ids, \
                pos_ctx2_input_mask,
                "ctx")

            neg_ctx1_vec = self.encode( \
                neg_ctx1_input_ids, \
                neg_ctx1_input_mask,
                "ctx")
            neg_ctx2_vec = self.encode( \
                neg_ctx2_input_ids, \
                neg_ctx2_input_mask,
                "ctx")
        
        qvec = self.encode( \
            q_input_ids, \
            q_input_mask,
            "q")
        q_sp_vec = self.encode( \
            q_sp_input_ids, \
            q_sp_input_mask,
            "q")
        return {
            "pos_ctx1_vec": pos_ctx1_vec,
            "pos_ctx2_vec": pos_ctx2_vec,
            "neg_ctx1_vec": neg_ctx1_vec,
            "neg_ctx2_vec": neg_ctx2_vec,
            "qvec": qvec,
            "q_sp_vec": q_sp_vec
        }
                
class LOUVRETrain:
    def __init__(self, config):
        self.config = config
        self.t_total = config.t_total
        self.max_grad_norm = config.max_grad_norm
        self.warmup_ratio = config.warmup_proportion
        self.weight_decay = config.weight_decay
        self.tokenizer = \
            RobertaTokenizerFast \
                .from_pretrained( \
                    self.config.init_checkpoint)
        if self.config.momentum:
            self.encoder = LOUVREEncoderMomentumTrain(config)
        else:
            self.encoder = LOUVREEncoderTrain \
                .from_pretrained( \
                    config.init_checkpoint, \
                    return_dict=True)
        self.optimizer, self.scheduler = \
            self.get_optimizer()
        self.cpu = torch.device("cpu")
        
        self.device = self.cpu
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.n_gpu = torch.cuda.device_count()
        print("N GPUs: {:d}".format(self.n_gpu))
        self.encoder.to(self.device)
        
        apex.amp.register_half_function(torch, 'einsum')
        self.encoder, self.optimizer \
            = apex.amp.initialize(
                self.encoder, self.optimizer, \
                opt_level="O1")
        if self.n_gpu > 1:
            self.encoder \
                = torch.nn.DataParallel( \
                    self.encoder)
        self.eval()
        
    def get_optimizer(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = Adam(optimizer_grouped_parameters,
                         lr=self.config.learning_rate,
                         eps=1e-8)
        
        warmup_steps = (self.t_total * self.warmup_ratio) if self.warmup_ratio < 1.0 else self.warmup_ratio
        scheduler = get_linear_schedule_with_warmup( \
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=self.t_total)
        return (optimizer, scheduler)
    
    def update_weight(self):
        torch.nn.utils.clip_grad_norm_(
            apex.amp.master_params(self.optimizer), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
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
        new_batch = { \
            k: batch[k].to(self.device) \
                for k in batch
        }

        vecs = self.encoder(new_batch)
        outputs = self.calc_softmax(vecs, is_training=True)
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct( \
                outputs["scores_1_hop"], outputs["target_1_hop"]) \
               + loss_fct( \
                outputs["scores_2_hop"], outputs["target_2_hop"])

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        tr_loss = loss.item()
        return tr_loss
    
    def calc_softmax(self, vecs, is_training):
        pos_ctx1_vec = vecs["pos_ctx1_vec"]
        pos_ctx2_vec = vecs["pos_ctx2_vec"]
        neg_ctx1_vec = vecs["neg_ctx1_vec"]
        neg_ctx2_vec = vecs["neg_ctx2_vec"]
        qvec = vecs["qvec"]
        q_sp_vec = vecs["q_sp_vec"]
        pos_ctx_vec = torch.cat([ \
            pos_ctx1_vec, \
            pos_ctx2_vec], dim=0)
        neg_ctx_vec = torch.cat([ \
            neg_ctx1_vec.unsqueeze(1), \
            neg_ctx2_vec.unsqueeze(1)], dim=1)

        scores_1_hop = torch.mm(qvec, pos_ctx_vec.t())
        neg_scores_1 = torch.bmm( \
            qvec.unsqueeze(1), \
            neg_ctx_vec.transpose(1,2)).squeeze(1)
        scores_2_hop = torch.mm(q_sp_vec, pos_ctx_vec.t())
        neg_scores_2 = torch.bmm( \
            q_sp_vec.unsqueeze(1), \
            neg_ctx_vec.transpose(1,2)).squeeze(1)
        
        bsize = qvec.size(0)
        scores_1_mask = torch.cat([ \
            torch.zeros(bsize, bsize), \
            torch.eye(bsize)], dim=1) \
                .to(qvec.device)
        scores_1_hop = scores_1_hop.float() \
                        .masked_fill( \
                            scores_1_mask.bool(), \
                            float('-inf')) \
                        .type_as(scores_1_hop)
        scores_1_hop = torch.cat([scores_1_hop, neg_scores_1], dim=1)
        scores_2_hop = torch.cat([scores_2_hop, neg_scores_2], dim=1)
        
        if self.config.momentum and is_training:
            queue_neg_scores_1 = \
                torch.mm( \
                    qvec, \
                    self.encoder.module.queue.clone().detach().t())
            queue_neg_scores_2 = \
                torch.mm( \
                    q_sp_vec, \
                    self.encoder.module.queue.clone().detach().t())
            scores_1_hop = torch.cat([ \
                scores_1_hop, \
                queue_neg_scores_1 \
            ], dim=1)
            scores_2_hop = torch.cat([ \
                scores_2_hop, \
                queue_neg_scores_2 \
            ], dim=1)
            self.encoder.module.dequeue_and_enqueue(pos_ctx_vec.detach())
        
        target_1_hop = torch.arange(bsize).to(qvec.device)
        target_2_hop = torch.arange(bsize).to(qvec.device) + bsize

        return {
            "scores_1_hop": scores_1_hop, 
            "scores_2_hop": scores_2_hop,
            "target_1_hop": target_1_hop,
            "target_2_hop": target_2_hop
        }
    
    def get_mrr(self, batch):
        new_batch = { \
            k: batch[k].to(self.device) \
                for k in batch
        }
        vecs = self.encoder(new_batch)
        outputs = self.calc_softmax(vecs, is_training=False)
        scores_1_hop = outputs["scores_1_hop"]
        scores_2_hop = outputs["scores_2_hop"]
        target_1_hop = outputs["target_1_hop"]
        target_2_hop = outputs["target_2_hop"]
        ranked_1_hop = scores_1_hop.argsort( \
            dim=1, descending=True)
        ranked_2_hop = scores_2_hop.argsort( \
            dim=1, descending=True)
        idx2ranked_1 = ranked_1_hop.argsort( \
            dim=1)
        idx2ranked_2 = ranked_2_hop.argsort( \
            dim=1)
        
        rrs_1, rrs_2 = [], []
        for t, idx2ranked in zip(target_1_hop, idx2ranked_1):
            rrs_1.append(1 / (idx2ranked[t].item() + 1))
        for t, idx2ranked in zip(target_2_hop, idx2ranked_2):
            rrs_2.append(1 / (idx2ranked[t].item() + 1))
        
        return {"rrs_1": rrs_1, "rrs_2": rrs_2}

    def predict(self, eval_dataloader):
        self.eval()
        rrs_1, rrs_2 = [], []
        for ind, batch in enumerate(tqdm(eval_dataloader, desc="Eval")):
            with torch.no_grad():
                rrs = self.get_mrr(batch)
            rrs_1 += rrs["rrs_1"]
            rrs_2 += rrs["rrs_2"]
        mrr_1, mrr_2 = np.mean(rrs_1), np.mean(rrs_2)
        return {
            "mrr_1": mrr_1,
            "mrr_2": mrr_2,
            "avg_mrr": np.mean([mrr_1, mrr_2])
        }
    
    def save_model(self, output_dir, args):
        if self.config.momentum:
            model = self.encoder.module.q_encoder
        else:
            model = self.encoder
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        model_to_save = \
            model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

        return 0

class LOUVRERetriever:
    def __init__(self, config):
        logger.info("Loading trained model...")
        apex.amp.register_half_function(torch, 'einsum')
        self.config = config
        self.tokenizer = \
            RobertaTokenizerFast \
                .from_pretrained( \
                    config.init_checkpoint)

        self.index_save_path = config.index_save_path
        self.saved_index_path = config.saved_index_path
        self.pred_batch_size \
            = config.pred_batch_size
        self.max_q_seq_length = config.max_q_seq_length
        self.max_q_sp_seq_length = config.max_q_sp_seq_length
        self.beam_size = config.beam_size
        self.topk = config.topk
        self.index, self.index_s = self.build_index(config)
        self.id2doc = self.load_corpus(config)
        
        self.encoder = LOUVREEncoder \
            .from_pretrained( \
                config.init_checkpoint, \
                return_dict=True)
        self.cpu = torch.device("cpu")
        
        self.device = self.cpu
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.n_gpu = torch.cuda.device_count()

        self.encoder.to(self.device)
        
        apex.amp.register_half_function(torch, 'einsum')
        self.encoder = \
            apex.amp.initialize( \
                self.encoder, \
                opt_level='O1')
        
        if self.n_gpu > 1:
            self.encoder \
                = torch.nn.DataParallel( \
                    self.encoder)
        self.encoder.eval()
    
    def encode(self, input_ids, \
               input_mask):
        input_ids_ = input_ids.to(self.device)
        input_mask_ = \
            input_mask.to(self.device)
        
        with torch.no_grad():
            vecs = self.encoder( \
                    input_ids_,
                    input_mask_)
        return vecs
    
    def build_index(self, config):
        xb, d = None, None
        if self.saved_index_path != None:
            logger.info("Loading index...")
            ctx_vecs = np.load(self.saved_index_path)
            xb = ctx_vecs.astype('float32')
            d = ctx_vecs.shape[-1]
        else:
            logger.info("Building index...")
            
            ctx_encoder = LOUVREEncoder \
                .from_pretrained( \
                    config.ctx_init_checkpoint, \
                    return_dict=True)
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            n_gpu = torch.cuda.device_count()
            ctx_encoder.to(device)
            ctx_encoder = \
                apex.amp.initialize( \
                    ctx_encoder, \
                    opt_level='O1')
            if n_gpu > 1:
                ctx_encoder \
                    = torch.nn.DataParallel( \
                        ctx_encoder)
            ctx_encoder.eval()

            corpus_dataset = LOUVRECTXDataset( \
                config.init_checkpoint, config.max_ctx_seq_length, config.corpus_fname)
            corpus_dataloader = \
                DataLoader(
                    corpus_dataset, \
                    batch_size=config.pred_batch_size, \
                    collate_fn=post_process_ctx_batch, \
                    pin_memory=True, \
                    num_workers=16)
            
            ctx_vecs = []
            for batch in tqdm(corpus_dataloader, desc="Indexing documents"):
                input_ids_ = batch['input_ids'].to(device)
                input_mask_ = batch['input_mask'].to(device)
                with torch.no_grad():
                    batch_vec = ctx_encoder( \
                            input_ids_,
                            input_mask_)
                ctx_vecs.append(batch_vec.cpu().numpy())
            ctx_vecs = np.concatenate(ctx_vecs)
            print("Corpus vector size: {}".format(ctx_vecs.shape))
            
            xb = ctx_vecs.astype('float32')
            if self.index_save_path != None:
                np.save(self.index_save_path, xb)
            d = ctx_vecs.shape[-1]

        ngpus = faiss.get_num_gpus()
        print("number of GPUs:", ngpus)

        def build_sub_index(data, device_info):
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatIP(d)
            gpu_index = \
                faiss.index_cpu_to_gpu(res, device_info, index)
            gpu_index.add(data)
            return gpu_index
        
        chunk_size = math.ceil(len(xb) / ngpus)
        xb_chunks = []
        index_s = []
        for s in range(0, len(xb), chunk_size):
            xb_chunks.append(xb[s: min(s + chunk_size, len(xb))])
            index_s.append(s)
        index = [build_sub_index(chunk, i) for i, chunk in enumerate(xb_chunks)]
        return (index, index_s)
        
    def load_corpus(self, config):
        logger.info(f"Loading corpus...")
        with open(config.corpus_fname) as f:
            id2doc = json.load(f)
        if isinstance(id2doc["0"], list):
            id2doc = {k: {"title":v[0], "text": v[1]} for k, v in id2doc.items()}
        logger.info(f"Corpus size {len(id2doc)}")
        return id2doc
    
    def index_search(self, qvecs, beam_size):
        Ds, Is = [], []
        for s, ind in zip(self.index_s, self.index):
            D, I = ind.search(qvecs, beam_size)
            Ds.append(D)
            Is.append(I+s)
        Ds = np.concatenate(Ds, axis=1)
        Is = np.concatenate(Is, axis=1)
        
        nquery = qvecs.shape[0]
    
        total_D = []
        total_I = []
        for i in range(nquery):
            result = list(zip(Ds[i], Is[i]))
            result.sort(key=lambda x: x[0], reverse=True)
            result = result[:beam_size]
            total_D.append(np.array([_[0] for _ in result]))
            total_I.append(np.array([_[1] for _ in result]))
        total_D = np.stack(total_D, axis=0)
        total_I = np.stack(total_I, axis=0)
        return total_D, total_I
        
    def pred(self, questions):
        logger.info("Encoding questions and searching")
        predictions = {
            "candidate_chains": []
        }
        hop1_qs = [(q, None) for q in questions]
        pred_1hop_data = LOUVREQDataset( \
            self.config.init_checkpoint,
            self.max_q_seq_length,
            hop1_qs)
        pred_1hop_dataloader = DataLoader( \
            pred_1hop_data, \
            batch_size=self.pred_batch_size,
            pin_memory=True, num_workers=16,
            collate_fn=post_process_q_batch)
        
        for batch in tqdm(pred_1hop_dataloader, desc="Predicting"):
            qvecs = self.encode( \
                batch['input_ids'], \
                batch['input_mask'])
            qvecs = qvecs.cpu().contiguous().numpy()
            batch_q = [q[0] for q in batch['question']]
            
            D, I = self.index_search( \
                qvecs, \
                self.beam_size)
            bsize = len(batch_q)
            
            # 2hop search
            query_pairs = []
            for b_idx in range(bsize):
                for _, doc_id in enumerate(I[b_idx]):
                    doc = self.id2doc[str(doc_id)]["text"]
                    if doc.strip() == "":
                        doc = self.id2doc[str(doc_id)]["title"]
                        D[b_idx][_] = float("-inf")
                    query_pairs.append((batch_q[b_idx], doc))
            pred_2hop_data = LOUVREQDataset( \
                self.config.init_checkpoint,
                self.max_q_sp_seq_length,
                query_pairs)
            pred_2hop_dataloader = DataLoader( \
                pred_2hop_data, \
                batch_size=self.pred_batch_size, \
                pin_memory=True, num_workers=16, \
                collate_fn=post_process_q_batch)
            
            q_sp_vecs = []
            for batch_hop2 in pred_2hop_dataloader:
                q_sp_vecs_ = self.encode( \
                    batch_hop2["input_ids"], \
                    batch_hop2["input_mask"])
                q_sp_vecs_ = q_sp_vecs_.contiguous().cpu().numpy()
                q_sp_vecs.append(q_sp_vecs_)
            q_sp_vecs = np.concatenate(q_sp_vecs, axis=0)
            
            D_, I_ = self.index_search( \
                q_sp_vecs, self.beam_size)
            D_ = D_.reshape( \
                bsize, \
                self.beam_size, \
                self.beam_size)
            I_ = I_.reshape( \
                bsize, \
                self.beam_size, \
                self.beam_size)
            
            # aggregate path scores
            path_scores = np.expand_dims(D, axis=2) + D_
            
            for idx in range(bsize):
                search_scores = path_scores[idx]
                ranked_pairs = np.vstack( \
                    np.unravel_index( \
                        np.argsort(search_scores.ravel())[::-1], \
                        (self.beam_size, self.beam_size) \
                    ) \
                ).transpose()
                candidate_chains = []
                for _ in range(self.topk):
                    path_ids = ranked_pairs[_]
                    hop_1_id = I[idx, path_ids[0]]
                    hop_2_id = I_[idx, path_ids[0], path_ids[1]]
                    candidate_chains.append([self.id2doc[str(hop_1_id)], self.id2doc[str(hop_2_id)]])
                predictions["candidate_chains"].append(candidate_chains)
        return predictions

class LOUVRERetrieverWiki(LOUVRERetriever):
    def __init__(self, config):
        super().__init__(config)
        self.db = DocDB(config.db_file)
        self.title2text = {
            self.id2doc[_id]['title']: self.id2doc[_id]['text'] for _id in self.id2doc
        }
        self.title2index = self.build_title2index()
        self.title2hyperlinked= { \
            title: self.db.get_hyper_linked("{:s}_0".format(title)) \
                for title in tqdm(self.title2index, desc="Preparing hyperlinks")
        }
    
    def build_title2index(self):
        title2index = { \
            self.id2doc[_id]["title"]: int(_id) \
                for _id in self.id2doc
        }
        return title2index
        
    def pred(self, questions):
        logger.info("Encoding questions and searching")
        predictions = {
            "candidate_chains": []
        }
        hop1_qs = [(q, None) for q in questions]
        pred_1hop_data = LOUVREQDataset( \
            self.config.init_checkpoint,
            self.max_q_seq_length,
            hop1_qs)
        pred_1hop_dataloader = DataLoader( \
            pred_1hop_data, \
            batch_size=self.pred_batch_size,
            pin_memory=True, num_workers=16,
            collate_fn=post_process_q_batch)
        
        for batch in tqdm(pred_1hop_dataloader, desc="Predicting"):
            qvecs = self.encode( \
                batch['input_ids'], \
                batch['input_mask'])
            qvecs = qvecs.cpu().contiguous().numpy()
            batch_q = [q[0] for q in batch['question']]
            
            D, I = self.index_search( \
                qvecs, \
                self.beam_size)
            bsize = len(batch_q)
            
            # 2hop search
            query_pairs = []
            for b_idx in range(bsize):
                for _, doc_id in enumerate(I[b_idx]):
                    doc = self.id2doc[str(doc_id)]["text"]
                    if doc.strip() == "":
                        doc = self.id2doc[str(doc_id)]["title"]
                        D[b_idx][_] = float("-inf")
                    query_pairs.append((batch_q[b_idx], doc))
            pred_2hop_data = LOUVREQDataset( \
                self.config.init_checkpoint,
                self.max_q_sp_seq_length,
                query_pairs)
            pred_2hop_dataloader = DataLoader( \
                pred_2hop_data, \
                batch_size=self.pred_batch_size, \
                pin_memory=True, num_workers=16, \
                collate_fn=post_process_q_batch)
            
            q_sp_vecs = []
            for batch_hop2 in pred_2hop_dataloader:
                q_sp_vecs_ = self.encode( \
                    batch_hop2["input_ids"], \
                    batch_hop2["input_mask"])
                q_sp_vecs_ = q_sp_vecs_.contiguous().cpu().numpy()
                q_sp_vecs.append(q_sp_vecs_)
            q_sp_vecs = np.concatenate(q_sp_vecs, axis=0)
            
            D_, I_ = self.index_search( \
                q_sp_vecs, self.beam_size)
            
            D_ = D_.reshape( \
                bsize, \
                self.beam_size, \
                self.beam_size)
            I_ = I_.reshape( \
                bsize, \
                self.beam_size, \
                self.beam_size)
            
            # aggregate path scores
            path_scores = np.expand_dims(D, axis=2) + D_
            
            #Prioritizing hyperlinked docs
            for idx in range(bsize):
                for Ii in range(self.beam_size):
                    hop1_title = self.id2doc[str(I[idx][Ii])]["title"]
                    hop1_hlink = self.title2hyperlinked[hop1_title] if hop1_title in self.title2hyperlinked else set([])
                    hop1_hlink = set([]) if hop1_hlink == None else hop1_hlink
                    for I_i in range(self.beam_size):
                        hop2_title = self.id2doc[str(I_[idx][Ii][I_i])]["title"]
                        hop2_hlink = self.title2hyperlinked[hop2_title] if hop2_title in self.title2hyperlinked else set([])
                        hop2_hlink = set([]) if hop2_hlink == None else hop2_hlink
                        if hop2_title in hop1_hlink or hop1_title in hop2_hlink:
                            path_scores[idx][Ii][I_i] += 5.0
            
            for idx in range(bsize):
                search_scores = path_scores[idx]
                ranked_pairs = np.vstack( \
                    np.unravel_index( \
                        np.argsort(search_scores.ravel())[::-1], \
                        (self.beam_size, self.beam_size) \
                    ) \
                ).transpose()
                candidate_chains = []
                for _ in range(self.topk):
                    path_ids = ranked_pairs[_]
                    hop_1_id = I[idx, path_ids[0]]
                    hop_2_id = I_[idx, path_ids[0], path_ids[1]]
                    candidate_chains.append([self.id2doc[str(hop_1_id)], self.id2doc[str(hop_2_id)]])
                predictions["candidate_chains"].append(candidate_chains)
        return predictions
