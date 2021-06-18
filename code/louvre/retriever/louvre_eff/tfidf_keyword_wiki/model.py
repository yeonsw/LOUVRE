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
import numpy as np

from .doc_db import DocDB        
from .tfidf_doc_ranker import TfidfDocRanker
from .keyword_matching import KeywordRetriever

logger = logging.getLogger(__name__)

class TfidfKeywordWikiRetriever:
    def __init__(self, db_path, tfidf_path, k):
        self.db_path = db_path
        self.tfidf_path = tfidf_path
        self.k = k
        self.db = DocDB(self.db_path)
        self.ranker = \
            TfidfDocRanker(\
                tfidf_path=self.tfidf_path, \
                strict=False)
        self.keyword_ranker = \
            KeywordRetriever(self.db_path)
        self.passage2vec = {}
        self.qid2vec = {}
    
    def get_doc_text(self, doc_id):
        txt = self.db.get_doc_text(doc_id)
        if txt == None:
            return txt
        txt = " ".join(txt.split("\t"))
        return txt

    def predict(self, queries, print_progress_bar=True):
        preds = []
        chunk_size = 10
        n_chunk = math.ceil(len(queries) / chunk_size)
        chunk_iter = tqdm(range(n_chunk), desc="Predicting") \
                        if print_progress_bar \
                            else range(n_chunk)
        for chunk_ind in chunk_iter:
            s = chunk_ind * chunk_size
            e = min((chunk_ind + 1) * chunk_size, len(queries))
            questions = queries[s:e]
            preds += self.retrieve(questions, self.k)
        
        results = []
        pred_iter = tqdm(preds, "Building selector outputs") \
                        if print_progress_bar \
                            else preds
        for pred in pred_iter:
            topk_titles = []
            context = {}
            for path in pred['path']:
                titles = []
                for title, passage in path:
                    titles.append(title)
                    context[title] = passage
                topk_titles.append(titles)
            result = {
                'q_id': pred['qid'],
                'question': pred['question'],
                'topk_titles': topk_titles,
                'context': context,
            }
            results.append(result)
        return results
    
    def rank_doc_ids(self, qid, q, doc_ids, k):
        if len(doc_ids) == 0:
            return []
        qvec = None
        if qid in self.qid2vec:
            qvec = self.qid2vec[qid]
        else:
            qvec = self.ranker.text2spvec(q)
            qvec = {k: v for k, v in zip(qvec.indices, qvec.data)}
            self.qid2vec[qid] = qvec
        scores = []
        for doc_id in doc_ids:
            if doc_id in self.passage2vec:
                textvec = self.passage2vec[doc_id]
            else:
                text = \
                    self.get_doc_text(doc_id)
                if text == None or len(text.strip()) < 2: 
                    continue
                textvec = self.ranker.text2spvec(text)
                textvec = {k: v for k, v in zip(textvec.indices, textvec.data)}
                self.passage2vec[doc_id] = textvec
            scores.append((doc_id, np.sum([qvec[k] * textvec[k] for k in qvec if k in textvec])))
        scores.sort(key=lambda x: x[1], reverse=True)
        results = [t[0] for t in scores][:k]
        return results
    
    def retrieve(self, questions, k):
        def is_valid_doc(doc_id):
            return doc_id is not None \
                    and doc_id.strip() != '' \
                    and self.get_doc_text(doc_id) != None

        results = [{
            'qid': q['id'],
            'path': [],
            'question': q['question']
        } for q in questions]
        
        for ind in range(len(results)):
            q = results[ind]['question']
            qid = results[ind]['qid']
            tfidf_candidates, _ = self.ranker.closest_docs(q, k=k)
            keyword_candidates = self.keyword_ranker.get_candidates(q)
            keyword_candidates = self.rank_doc_ids(qid, q, keyword_candidates, k)
            candidates = tfidf_candidates + keyword_candidates
            candidates = [ \
                doc_id for doc_id in candidates if is_valid_doc(doc_id)]
            for doc_id in candidates:
                text = self.get_doc_text(doc_id)
                hyperlinked = \
                    self.db.get_hyper_linked(doc_id)
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
                        if is_valid_doc(doc_id) \
                ]
                hyperlinked = self.rank_doc_ids(qid, q, hyperlinked, 5)
                for p in hyperlinked:
                    results[ind]['path'].append([(doc_id, text), (p, self.get_doc_text(p))])
        return results

