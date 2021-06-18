class TrainConfig:
    def __init__(self,
                 max_q_seq_length: int,
                 max_ctx_seq_length: int,
                 neg_k: int,
                 neg_n_init_doc: int,
                 encoder_path: str,
                 train_file: str,
                 tfidf_path: str, 
                 db_path: str):

        self.max_ctx_seq_length = max_ctx_seq_length
        self.max_q_seq_length = max_q_seq_length
        self.neg_k = neg_k
        self.neg_n_init_doc = neg_n_init_doc
        
        self.encoder_path = encoder_path
        self.train_file = train_file
        self.tfidf_path = tfidf_path
        self.db_path = db_path

        self.config_str = \
            '- Max CTX seq length: {:s}\n' \
                .format(str( \
                    self.max_ctx_seq_length)) \
            + '- Max Q seq length: {:s}\n' \
                .format(str( \
                    self.max_q_seq_length)) \
            + '- N strong negatives: {:s}\n' \
                .format(str(self.neg_k)) \
            + '- Neg N init doc: {:s}\n' \
                .format(str(self.neg_n_init_doc)) \
            + '- Train file: {:s}\n' \
                .format(self.train_file) \
            + '- TF-IDF results file path: {:s}\n' \
                .format(self.tfidf_path) \
            + '- Encoder path: {:s}\n' \
                .format(self.encoder_path) \
            + '- DB path: {:s}\n' \
                .format(self.db_path)
        
    def __str__(self):
        configStr = '\n\n' \
            '### Train configurations ###\n' \
                + self.config_str \
            + '##############################\n'

        return configStr

class PreTrainConfig(TrainConfig):
    def __init__(self,
                 max_q_seq_length: int,
                 max_ctx_seq_length: int,
                 neg_k: int,
                 neg_n_init_doc: int,
                 encoder_path: str,
                 train_file: str,
                 db_path: str,
                 tfidf_path: str,
                 pruning_l: int,
                 use_full_article=False):
        super().__init__(max_q_seq_length,
                         max_ctx_seq_length,
                         neg_k,
                         neg_n_init_doc,
                         encoder_path,
                         train_file,
                         tfidf_path,
                         db_path)
        self.use_full_article=use_full_article
        self.pruning_l = pruning_l
