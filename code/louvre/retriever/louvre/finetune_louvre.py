#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import shutil
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from .train_data import LOUVREDataset, post_process_train_batch
from .model import LOUVRETrain, ModelConfig
from .train_config import TrainConfig

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def train(args, model, train_dataloader, eval_dataloader, t_total):
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Num steps = %d", t_total)

    global_step = 0
    global_step_accumulation = 0
    tr_loss = 0.0
    prev_best = -np.inf
    target_metric = "avg_mrr"
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

            if global_step_accumulation % (args.train_save_step * args.gradient_accumulation_steps) == 0:
                logger.info("loss: {:.04f}".format(tr_loss/args.train_save_step))
                tr_loss = 0.0

                scores = model.predict(eval_dataloader)
                logger.info("\n".join([ \
                    "{:s}: {:.04f}".format(k, scores[k]) \
                        for k in scores
                ]))
                score = scores[target_metric]
                if score > prev_best:
                    prev_best = score
                    model.save_model( \
                        "{:s}_best".format(args.train_output_dir), args)
                model.save_model(args.train_output_dir, args)

    scores = model.predict(eval_dataloader)
    logger.info("\n".join([ \
        "{:s}: {:.04f}".format(k, scores[k]) \
            for k in scores
    ]))
    score = scores[target_metric]
    if score > prev_best:
        prev_best = score
        model.save_model(args)
    model.save_model(args.train_output_dir, args)
    return 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_save_step", \
                        default=500, \
                        type=int)
    parser.add_argument("--train_output_dir",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--train_batch_size",
                        default=1,
                        type=int)
    parser.add_argument("--eval_batch_size",
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
    parser.add_argument("--max_q_seq_length",
                        default=70,
                        type=int)
    parser.add_argument("--max_q_sp_seq_length",
                        default=350,
                        type=int)
    parser.add_argument("--max_ctx_seq_length",
                        default=300,
                        type=int)
    
    parser.add_argument('--train_file',
                        type=str,
                        required=True)
    parser.add_argument('--eval_file',
                        type=str,
                        required=True)
    parser.add_argument('--init_checkpoint',
                        type=str,
                        default="roberta-base")
    parser.add_argument("--k",
                        type=int,
                        default=-1)
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.0)
    parser.add_argument("--momentum",
                        action='store_true')
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
        TrainConfig( \
          init_checkpoint=args.init_checkpoint,
          fname = args.train_file,
          max_q_seq_length \
            = args.max_q_seq_length,
          max_ctx_seq_length \
            = args.max_ctx_seq_length,
          max_q_sp_seq_length \
            = args.max_q_sp_seq_length)
    logger.info(train_config)
    train_data = LOUVREDataset(train_config)
    train_dataloader = DataLoader( \
        train_data, \
        batch_size=args.train_batch_size, \
        shuffle=True, num_workers=12, \
        pin_memory=True, \
        collate_fn=post_process_train_batch)
    t_total = len(train_dataloader) / args.gradient_accumulation_steps * args.num_train_epochs
    
    eval_config = \
        TrainConfig( \
          init_checkpoint=args.init_checkpoint,
          fname = args.eval_file,
          max_q_seq_length \
            = args.max_q_seq_length,
          max_ctx_seq_length \
            = args.max_ctx_seq_length,
          max_q_sp_seq_length \
            = args.max_q_sp_seq_length)
    logger.info(eval_config)
    eval_data = LOUVREDataset(eval_config)
    eval_dataloader = DataLoader( \
        eval_data, \
        batch_size=args.eval_batch_size, \
        pin_memory=True, num_workers=16, \
        collate_fn=post_process_train_batch)
    
    model_config = \
        ModelConfig( \
            init_checkpoint \
                =args.init_checkpoint,
            max_q_seq_length \
                =args.max_q_seq_length,
            max_ctx_seq_length \
                =args.max_ctx_seq_length, \
            max_q_sp_seq_length \
                =args.max_q_sp_seq_length, \
            learning_rate= \
                args.learning_rate, \
            warmup_proportion= \
                args.warmup_proportion, \
            t_total=t_total, \
            k=args.k,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    logger.info(model_config)
    model = LOUVRETrain(model_config)
    
    _ = train(args, model, train_dataloader, eval_dataloader, t_total)
    return 0
            
if __name__ == "__main__":
    args = parse_args()
    main(args)
