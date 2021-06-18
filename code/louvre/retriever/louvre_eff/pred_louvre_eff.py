from __future__ import division
from __future__ import print_function

import logging
import argparse
import random
import json
import jsonlines
import numpy as np
import torch

from .model import LOUVREEffRetrieverConfig, LOUVREEffRetrieverWikiKeyword

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_batch_size",
                        default=1,
                        type=int)
    parser.add_argument("--indexing_batch_size",
                        default=1,
                        type=int)
    parser.add_argument("--no_cuda",
                        action='store_true')
    parser.add_argument('--seed',
                        type=int,
                        default=42)
    parser.add_argument("--max_q_seq_length",
                        default=350,
                        type=int)
    parser.add_argument("--max_ctx_seq_length",
                        default=300,
                        type=int)
    parser.add_argument('--encoder_path',
                        type=str,
                        required=True)
    parser.add_argument('--db_path',
                        type=str,
                        required=True)
    parser.add_argument('--tfidf_path',
                        type=str,
                        default=None)
    parser.add_argument('--k',
                        type=int,
                        required=True)
    parser.add_argument('--input_file',
                        type=str,
                        required=True)
    parser.add_argument('--pred_file',
                        type=str,
                        required=True)
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
    _ = set_seed(args)

    model_config = \
        LOUVREEffRetrieverConfig( \
            encoder_path=args.encoder_path,
            max_q_seq_length \
                =args.max_q_seq_length,
            max_ctx_seq_length \
                =args.max_ctx_seq_length, \
            k=args.k, \
            db_path=args.db_path, \
            pred_batch_size \
                =args.pred_batch_size, \
            indexing_batch_size \
                =args.indexing_batch_size, \
            tfidf_path \
                =args.tfidf_path) 
    logger.info(model_config)
    model = LOUVREEffRetrieverWikiKeyword(model_config)
    
    with open(args.input_file, "r") as f:
        queries = json.load(f)
    
    logger.info("***** Prediction *****")
    predictions = model.predict(queries)
    if args.pred_file is not None:
        with jsonlines.open(args.pred_file, 'w') as writer:
            for _ in predictions:
                writer.write(_)
    return 0
            
if __name__ == "__main__":
    args = parse_args()
    main(args)
