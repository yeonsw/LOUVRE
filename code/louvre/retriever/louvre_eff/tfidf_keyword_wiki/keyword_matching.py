import unicodedata

import regex
import re
import json
import collections
from tqdm import tqdm
from typing import List, Dict, Union
from flashtext import KeywordProcessor

from .doc_db import DocDB        

STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn', "'ll", "'re", "'ve", "n't", "'s", "'d", "'m", "''", "``"
}

class RetrievedItem:
    def __init__(self, item_id: str, retrieval_method: str) -> None:
        self.item_id = item_id
        self.scores_dict: Dict[str, float] = dict()
        self.method = retrieval_method

class RetrievedSet:
    def __init__(self):
        self.retrieved_dict: Dict[str, RetrievedItem] = dict()

    def add_item(self, retri_item: RetrievedItem):
        if retri_item.item_id in self.retrieved_dict.keys():
            return None
        else:
            self.retrieved_dict[retri_item.item_id] = retri_item
            return retri_item

    def to_id_list(self):
        return list(self.retrieved_dict.keys())

class KeywordRetriever:
    def __init__(self, \
                 db_path):
        self.db_path = db_path
        self.db = DocDB(self.db_path)
        self.doc_ids = self.db.get_doc_ids()
        self.titles = [doc_id[:-2] for doc_id in self.doc_ids]
        self.title_entities, self.disambiguation_group = \
            self.get_title_entity_set(self.titles)
        
        self.keyword_processor = \
            KeywordProcessor(case_sensitive=True)
        self.keyword_processor_disamb = \
            KeywordProcessor(case_sensitive=True)

        _MatchedObject = collections.namedtuple(
            "MatchedObject", [ \
                "matched_key_word", \
                "matched_keywords_info" \
            ] \
        )
        for kw in tqdm(self.title_entities, desc="Prep keyword processor"):
            if self.filter_word(kw) or self.filter_document_id(kw):
                continue
            matched_obj = _MatchedObject( \
                matched_key_word=kw, \
                matched_keywords_info={kw: 'kwm'})
            self.keyword_processor.add_keyword(kw, matched_obj)
        
        for kw in tqdm(self.disambiguation_group, desc="Prep keyword disamb processor"):
            if self.filter_word(kw) or self.filter_document_id(kw):
                continue
            if kw in self.keyword_processor:
                existing_matched_obj: _MatchedObject \
                    = self.keyword_processor.get_keyword(kw)
                for disamb_kw in self.disambiguation_group[kw]:
                    if self.filter_document_id(disamb_kw):
                        continue
                    if disamb_kw not in existing_matched_obj.matched_keywords_info:
                        existing_matched_obj.matched_keywords_info[disamb_kw] = 'kwm_disamb'
            else:
                matched_obj = \
                    _MatchedObject(\
                        matched_key_word=kw, \
                        matched_keywords_info=dict())
                for disamb_kw in self.disambiguation_group[kw]:
                    if self.filter_document_id(disamb_kw):
                        continue
                    matched_obj.matched_keywords_info[disamb_kw] = 'kwm_disamb'
                self.keyword_processor_disamb.add_keyword(kw, matched_obj)
    
    def get_disamb_match(self, text_in):
        disamb_pattern = re.compile(r'(^.*)\s\(.*\)$')
        matches = list(disamb_pattern.finditer(text_in))
        if len(matches) == 0:
            return None, None
        else:
            match = matches[0]
            return match.group(0), match.group(1)

    def get_title_entity_set(self, titles):
        disambiguation_group = dict()
        title_entity_set = set()
        for cur_title in tqdm(titles, desc="Getting title-entity set"):
            title_entity_set.add(cur_title)
            # Build disambiguation groups
            dis_whole, dis_org = self.get_disamb_match(cur_title)
            if dis_whole is not None:
                if dis_org not in disambiguation_group:
                    disambiguation_group[dis_org] = set()
                disambiguation_group[dis_org].add(dis_whole)

        for title in title_entity_set:
            if title in disambiguation_group:
                disambiguation_group[title].add(title)

        return title_entity_set, disambiguation_group

    def filter_disamb_doc(self, input_string):
        if ' (disambiguation)' in input_string:
            return True
        else:
            return False       
    
    def check_arabic(self, input_string):
        res = re.findall(
            r'[\U00010E60-\U00010E7F]|[\U0001EE00-\U0001EEFF]|[\u0750-\u077F]|[\u08A0-\u08FF]|[\uFB50-\uFDFF]|[\uFE70-\uFEFF]|[\u0600-\u06FF]',
            input_string)

        if len(res) != 0:
            return True
        else:
            return False
    
    def filter_document_id(self, \
                           input_string, \
                           remove_disambiguation_doc=True):
        pid_words = input_string.strip().replace('_', ' ')
        match = re.search('[a-zA-Z]', pid_words)
        if match is None:
            return True
        if self.check_arabic(pid_words):
            return True
        if remove_disambiguation_doc:
            if self.filter_disamb_doc(input_string):
                return True
        return False
    
    def normalize(self, text):
        """Resolve different type of unicode encodings."""
        return unicodedata.normalize('NFD', text)
    
    def filter_word(self, text):
        """
        Take out english 
        stopwords, 
        punctuation, 
        and compound endings.
        """
        text = self.normalize(text)
        if regex.match(r'^\p{P}+$', text):
            return True
        if text.lower() in STOPWORDS:
            return True
        return False
    
    def get_closest_abstracts(self, qid, question):
        titles = self.retrieve_keyword_matched_titles(question)
        doc_ids = ["{:s}_0".format(title) for title in titles]
        context = { \
            doc_id: self.db.get_doc_text(doc_id) \
                for doc_id in doc_ids
        }
        return [{"question": question,
                 "context": context,
                 "q_id": qid}]

    def get_candidates(self, q):
        titles = self.retrieve_keyword_matched_titles(q)
        doc_ids = ["{:s}_0".format(title) for title in titles]
        return doc_ids

    def retrieve_keyword_matched_titles(self, question):
        _MatchedObject = collections.namedtuple(
            "MatchedObject", [ \
                "matched_key_word", \
                "matched_keywords_info" \
            ] \
        )
        # 1. First retrieve raw key word matching.
        finded_keys_kwm: List[_MatchedObject, int, int] \
            = self.keyword_processor.extract_keywords( \
                question, span_info=True)
        finded_keys_kwm_disamb: List[_MatchedObject, int, int] \
            = self.keyword_processor_disamb.extract_keywords( \
                question, span_info=True)
        finded_keys_list: List[Tuple[str, str, str, int, int]] = []
        retrieved_set = RetrievedSet()

        all_finded_span = []
        all_finded_span_2 = []

        for finded_matched_obj, start, end in finded_keys_kwm:
            for i in range(start, end):
                all_finded_span.append((start, end))
                all_finded_span_2.append((start, end))

            # for matched_obj in finded_matched_obj.:
            matched_words = finded_matched_obj.matched_key_word
            for extracted_keyword, method in finded_matched_obj.matched_keywords_info.items():
                finded_keys_list.append((matched_words, extracted_keyword, method, start, end))

        for finded_matched_obj, start, end in finded_keys_kwm_disamb:
            not_valid = False
            for e_start, e_end in all_finded_span:
                if e_start <= start and e_end >= end:
                    not_valid = True
                    break
            if not not_valid:
                matched_words = finded_matched_obj.matched_key_word
                for extracted_keyword, method in finded_matched_obj.matched_keywords_info.items():
                    finded_keys_list.append((matched_words, extracted_keyword, method, start, end))
                    all_finded_span_2.append((start, end))

        for matched_word, title, method, start, end in finded_keys_list:
            not_valid = False
            for e_start, e_end in all_finded_span_2:
                if (e_start < start and e_end >= end) or (e_start <= start and e_end > end):
                    not_valid = True    # Skip this match bc this match is already contained in some other match.
                    break
            if not_valid:
                continue
            retrieved_set.add_item(RetrievedItem(title, method))
        return retrieved_set.to_id_list()
