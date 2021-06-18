#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from .train_data import LOUVREEffTrainDataset, post_process_train_batch
from .model import LOUVREEff, ModelConfig
from .train_config import PreTrainConfig

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def train(args, model, train_dataloader, t_total):
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Num steps = %d", t_total)

    global_step = 0
    global_step_accumulation = 0
    tr_loss = 0.0
    model.zero_grad()
    for epc in range(1, int(args.num_train_epochs)+1):
        logger.info('Epoch '+str(epc))
        for step, batch in enumerate(tqdm( \
                        train_dataloader, desc="Iteration")):
            batch_loss = model \
                .get_loss_and_backward( \
                    batch, \
                    args.gradient_accumulation_steps)
            tr_loss += batch_loss

            if (step + 1) % args.gradient_accumulation_steps == 0:
                model.update_weight()
                global_step += 1
            global_step_accumulation += 1

            if global_step_accumulation % (args.save_step * args.gradient_accumulation_steps) == 0:
                logger.info("loss: {:.04f}".format(tr_loss/args.save_step))
                tr_loss = 0.0
                model.save_model(args)
    
    model.save_model(args)
    return 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_step", \
                        default=500, \
                        type=int)
    parser.add_argument("--train_output_dir",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--train_batch_size",
                        default=1,
                        type=int)
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float)
    parser.add_argument("--num_train_epochs",
                        default=5.0,
                        type=float)
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float)
    parser.add_argument("--no_cuda",
                        action='store_true')
    parser.add_argument('--seed',
                        type=int,
                        default=42)
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1)
    parser.add_argument('--neg_k', type=int, default=1)
    parser.add_argument('--neg_n_init_doc', type=int, default=20)
    parser.add_argument("--max_q_seq_length",
                        default=350,
                        type=int)
    parser.add_argument("--max_ctx_seq_length",
                        default=300,
                        type=int)
    parser.add_argument('--encoder_path',
                        type=str,
                        required=True)
    parser.add_argument('--train_file',
                        type=str,
                        required=True)
    parser.add_argument('--tfidf_file',
                        type=str,
                        required=True)
    parser.add_argument('--db_file',
                        type=str,
                        required=True)
    parser.add_argument('--pruning_l',
                        type=int,
                        default=10)
    args = parser.parse_args()
    return args    

def set_seed(args):
    n_gpu = torch.cuda.device_count()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    return 0

def main(args):
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    _ = set_seed(args)
    os.makedirs(args.train_output_dir, exist_ok=True)

    train_config = \
        PreTrainConfig( \
          encoder_path = args.encoder_path,
          neg_k = args.neg_k,
          neg_n_init_doc=args.neg_n_init_doc,
          train_file = args.train_file,
          tfidf_path = \
            args.tfidf_file,
          
          max_q_seq_length \
            = args.max_q_seq_length,
          max_ctx_seq_length \
            = args.max_ctx_seq_length, \
          db_path=args.db_file, \
          pruning_l=args.pruning_l)
    logger.info(train_config)
    train_data = LOUVREEffTrainDataset(train_config)
    train_dataloader = DataLoader( \
            train_data, \
            batch_size=args.train_batch_size, \
            shuffle=True, num_workers=16, \
            pin_memory=True, \
            collate_fn=post_process_train_batch)   
    t_total = len(train_dataloader) / args.gradient_accumulation_steps * args.num_train_epochs
    
    model_config = \
        ModelConfig( \
            encoder_path = args.encoder_path,
            max_q_seq_length \
                = args.max_q_seq_length,
            max_ctx_seq_length \
                = args.max_ctx_seq_length, 
            learning_rate \
                =args.learning_rate, \
            warmup_proportion \
                =args.warmup_proportion, \
            t_total=t_total)
    logger.info(model_config)
    model = LOUVREEff(model_config)
    
    _ = train(args, model, train_dataloader, t_total)
    return 0
            
if __name__ == "__main__":
    args = parse_args()
    main(args)
