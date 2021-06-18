#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import argparse
from tqdm import tqdm
import json
import jsonlines
import collections
import unicodedata
import numpy as np

from basic_tokenizer import SimpleTokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

logger = logging.getLogger(__name__)

def normalize(text):
    return unicodedata.normalize('NFD', text)

def para_has_answer(answer, para):
    tokenizer = SimpleTokenizer()
    text = normalize(para)
    tokens = tokenizer.tokenize(text)
    text = tokens.words(uncased=True)
    assert len(text) == len(tokens)
    answer = normalize(answer)
    answer = tokenizer.tokenize(answer)
    answer = answer.words(uncased=True)
    for i in range(0, len(text) - len(answer) + 1):
        if answer == text[i:i+len(answer)]:
            return True
    return False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        type=str,
                        required=True)
    parser.add_argument('--hotpot_file',
                        type=str,
                        required=True)
    parser.add_argument('--corpus_file',
                        type=str,
                        required=True)
    parser.add_argument('--topk', 
                        type=int,
                        default=-1)
    args = parser.parse_args()
    return args

def main(args):
    with jsonlines.open(args.input_file) as reader:
        data = [r for r in tqdm(reader, desc="Reading prediction file")]
    with open(args.hotpot_file, "r") as f:
        hotpot = json.load(f)
    with open(args.corpus_file, "r") as f:
        corpus = json.load(f)
    corpus = {
        corpus[k]['title']: corpus[k]['text'] for k in corpus
    }
    id2hotpot = {
        d['_id']: d for d in hotpot
    }
    
    metrics = []
    for d in tqdm(data, desc="Eval"):
        q_id = d['_id']
        
        sp = list(set([_[0] for _ in id2hotpot[q_id]['supporting_facts']]))
        q_type = id2hotpot[q_id]['type']
        answer = id2hotpot[q_id]['answer']
        candidate_chains = d['candidate_chains']
        topk = len(candidate_chains) if args.topk == -1 else args.topk
        candidate_chains = candidate_chains[:topk]
        titles_retrieved = list(set([_['title'] for chain in candidate_chains for _ in chain]))
        sp_covered = [ \
            sp_title in titles_retrieved for sp_title in sp \
        ]
        p_em = 1 if np.sum(sp_covered) == len(sp_covered) else 0
        
        paths = [[_['title'] for _ in chain] for chain in candidate_chains]
        path_covered = [ \
            int(set(_) == set(sp)) \
                for _ in paths \
        ]
        path_covered = int(np.sum(path_covered) > 0)

        texts = " ".join([corpus[title] if title in corpus else "" for title in titles_retrieved])
        answer_recall = int(para_has_answer(answer, texts)) if answer not in ["yes", "no"] else None

        metrics.append({
            "p_em": p_em,
            "type": q_type,
            'path_covered': path_covered,
            'answer_recall': answer_recall 
        })

    qtype2preds = collections.defaultdict(list)
    for m in metrics:
        qtype2preds[m["type"]].append(m)
    
    logger.info(f'\tP-EM: {np.mean([m["p_em"] for m in metrics])}')
    logger.info(f'\tPath Recall: {np.mean([m["path_covered"] for m in metrics])}')
    logger.info(f'\tAnswer Recall: {np.mean([m["answer_recall"] for m in metrics if m["answer_recall"] != None])}')
    for t in qtype2preds.keys():
        logger.info(f"{t} Questions num: {len(qtype2preds[t])}")
        logger.info(f'\tP-EM: {np.mean([m["p_em"] for m in qtype2preds[t]])}')
        logger.info(f'\tPath Recall: {np.mean([m["path_covered"] for m in qtype2preds[t]])}')
        logger.info(f'\tAnswer Recall: {np.mean([m["answer_recall"] for m in qtype2preds[t] if m["answer_recall"] != None])}')
        
    return 0
            
if __name__ == "__main__":
    args = parse_args()
    main(args)
