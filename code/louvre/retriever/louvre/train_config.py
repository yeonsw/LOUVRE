class TrainConfig:
    def __init__(self,
                 max_q_seq_length: int,
                 max_ctx_seq_length: int,
                 max_q_sp_seq_length: int,
                 init_checkpoint: str,
                 fname: str):

        self.max_ctx_seq_length = max_ctx_seq_length
        self.max_q_seq_length = max_q_seq_length
        self.max_q_sp_seq_length = max_q_sp_seq_length
        self.init_checkpoint = init_checkpoint
        self.fname = fname
        self.cstr = '- Max CTX seq length: {:s}\n' \
                    .format(str( \
                        self.max_ctx_seq_length)) \
                + '- Max Q seq length: {:s}\n' \
                    .format(str( \
                        self.max_q_seq_length)) \
                + '- Max Q SP seq length: {:s}\n' \
                    .format(str( \
                        self.max_q_sp_seq_length)) \
                + '- Init Checkpoint: {:s}\n' \
                    .format(self.init_checkpoint) \
                + '- Train File: {:s}\n' \
                    .format(self.fname)
        
    def __str__(self):
        configStr = "{:s}{:s}{:s}".format( \
            '\n\n### Train configurations ###\n', \
            self.cstr, \
            '#' * 30 + '\n')
        return configStr
