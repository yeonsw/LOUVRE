import argparse
from tqdm import tqdm
import jsonlines
import numpy as np
import random
import spacy
import math
import re

from tfidf.doc_db import DocDB

class PreTrainDataGenerator:
    def __init__(self, \
                 db_file, \
                 output_file):
        self.db = DocDB(db_file)
        self.ner_model = spacy.load("en_core_web_sm")
        self.doc_ids = self.db.get_doc_ids()
        print("N Docs: {:d}".format(len(self.doc_ids)))
        self.output_file = output_file
    
    def get_hyperlinked_docs(self, doc_id):
        results = []
        hyper_linked_titles = \
            self.db.get_hyper_linked(doc_id)
        if hyper_linked_titles is None:
            return []
        
        for title in hyper_linked_titles:
            title_id = \
                "{:s}_0".format(title) \
                    if "_0" not in title \
                        else title
            if title_id == doc_id:
                continue
            text = self.get_doc_text(title_id, keep_sentence_split=True)
            if text == None:
                continue
            results.append((title_id, text))
        return results
    
    def get_doc_text(self, doc_id, keep_sentence_split=False):
        txt = self.db.get_doc_text(doc_id)
        if txt == None:
            return None
        sentences = [s.strip() for s in txt.split("\t")]
        if keep_sentence_split:
            return sentences
        txt = " ".join(sentences)
        return txt

    def generate_qas(self, doc):
        def case_insensitive_replace(s, t, doc):
            return re.sub(re.escape(s), t, doc, flags=re.IGNORECASE)
        
        def extract_sentence(doc):
            doc_id, sents = doc
            title = doc_id[:-2]
            if len(sents) == 0:
                print("Error case: {:s}".format(doc_text))
                return None
            sent = " {:s} ".format(sents[0].strip())
            sent = case_insensitive_replace(" {:s} ".format(title.lower()), " ", sent)
            sent = sent.strip()
            return (doc_id, title, sent)
        
        def extract_answer_and_bridge_entity_replace(text, title2, text2):
            if len(text) < 5:
                return None
            text = text[:-1] if text[-1] in [".", "?", "!"] else text
            entities = self.ner_model(text).ents
            answer_candidates = [e for e in entities if e.text.lower() != title2.lower()]
            if len(answer_candidates) == 0:
                return None
            answer_entity = random.choice(answer_candidates)
            answer, span_s, span_e, ent_type = \
                answer_entity.text, answer_entity.start_char, answer_entity.end_char, answer_entity.label_
            
            question = text[:span_s] + "what" + text[span_e:]
            question = " {:s} ".format(question)
            
            text2 = text2[:-1] if text2[-1] in [".", "?", "!"] else text2
            text2_entities = self.ner_model(text2).ents
            if not any([e.start_char==0 for e in text2_entities]):
                text2 = text2[0].lower() + text2[1:]
            
            title2 = " {:s} ".format(title2.strip())
            s_ind = question.lower().find(title2.lower())
            if s_ind == -1:
                return None
            e_ind = s_ind + len(title2.lower())
            question = question[:s_ind] + " {:s} ".format(text2).strip() + question[e_ind:]
            question = question.strip() + "."
            return (question, answer)
        
        doc_id, doc_sents = doc
        if doc_sents == None:
            return []
        if len(" ".join([_.strip() for _ in doc_sents])) < 5:
            return []
        hyperlinked_docs = self.get_hyperlinked_docs(doc_id)
        
        qas = []
        linked_sents = [extract_sentence(linked_doc) for linked_doc in hyperlinked_docs]
        linked_sents = [s for s in linked_sents if s != None and len(s[2]) >= 5]
        for sent_i, sent in enumerate(doc_sents):
            for linked_sent in linked_sents:
                linked_id, linked_title, linked_text = linked_sent
                r = extract_answer_and_bridge_entity_replace(sent, linked_title, linked_text)
                if r == None:
                    continue
                question, answer = r
                qas.append({ \
                    "q": question, 
                    "a": answer,
                    "hop1": linked_id[:-2],
                    "hop2": doc_id[:-2],
                    "sent1": 0,
                    "sent2": sent_i
                })
        return qas
    
    def get_passages_from_db(self, s, e):
        target_doc_ids = self.doc_ids[s:e]
        doc_id_text = [ \
            (doc_id, self.get_doc_text(doc_id, keep_sentence_split=True)) \
                for doc_id in target_doc_ids \
        ]
        return doc_id_text

    def build_pretrain_data(self, s, e):
        print("Target docs: {:d} to {:d}".format(s, e - 1))
        with jsonlines.open(self.output_file, "w") as writer:
            passages = self.get_passages_from_db(s, e)
            for ind, p in enumerate(tqdm(passages, desc="Generating QA Pairs")):
                qas = self.generate_qas(p)
                for jnd, qa in enumerate(qas):
                    qid = "{:d}_{:d}".format(s + ind, int(1e10) + jnd)
                    question, answer = qa['q'], qa['a']
                    supporting_facts = [ \
                        [qa['hop1'], qa['sent1']],
                        [qa['hop2'], qa['sent2']]
                    ]
                    context = [ \
                        [title, self.get_doc_text("{:s}_{:s}".format(title, "0"), keep_sentence_split=True)] \
                            for title, _ in supporting_facts \
                    ]
                    hop = { \
                        '_id': qid,
                        'answer': answer,
                        'type': "bridge",
                        'question': question,
                        'supporting_facts': supporting_facts,
                        'context': context,
                    }
                    writer.write(hop)
        return 0

def add_arguments(parser):
    parser.add_argument( \
        '--db_path', type=str, required=True)
    parser.add_argument( \
        '--output_file', \
        type=str, required=True)
    parser.add_argument( \
        '--chunk_ind', type=int, required=True)
    parser.add_argument( \
        '--nchunk', type=int, required=True)
    return 0

if __name__ == '__main__':
    random.seed(1234)
    parser = argparse.ArgumentParser()
    _ = add_arguments(parser)
    args = parser.parse_args()
    qg = PreTrainDataGenerator( \
                db_file=args.db_path, \
                output_file="{:s}_{:d}_{:d}.jsonl".format(args.output_file, args.chunk_ind, args.nchunk))
    
    ndoc = len(qg.doc_ids)
    chunk_size = math.ceil(ndoc * 1.0 / args.nchunk)
    s = args.chunk_ind * chunk_size
    e = min((args.chunk_ind + 1) * chunk_size, ndoc)
    _ = qg.build_pretrain_data(s, e)
