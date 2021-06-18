DATA_DIR=../../data/hotpotQA
CHECKPOINT_DIR=../../checkpoints
DB_PATH=../../data/db
TFIDF_PATH=../../data/tfidf

python louvre_pretrain.py \
    --save_step 500 \
    --train_output_dir $CHECKPOINT_DIR/louvre \
    --train_batch_size 256 \
    --learning_rate 2e-5 \
    --num_train_epochs 2 \
    --neg_k 0 \
    --neg_n_init_doc 500 \
    --encoder_path roberta-base \
    --train_file $DATA_DIR/pretrain_data.jsonl \
    --tfidf_file $TFIDF_PATH/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
    --db_file $DB_PATH/wiki_abst_only_hotpotqa_w_original_title.db
