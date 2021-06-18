DATA_DIR=../../data/hotpotQA
CHECKPOINT_DIR=../../checkpoints
OUTPUT_DIR=../../outputs
DB_PATH=../../data/db
TFIDF_PATH=../../data/tfidf

python louvre_eff_pred.py \
    --pred_batch_size 512 \
    --indexing_batch_size 512 \
    --encoder_path $CHECKPOINT_DIR/louvre_eff_finetune \
    --db_path $DB_PATH/wiki_abst_only_hotpotqa_w_original_title.db \
    --tfidf_path $TFIDF_PATH/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
    --k 20 \
    --input_file $DATA_DIR/hotpot_dev_fullwiki_v1.json \
    --pred_file $OUTPUT_DIR/louvre_eff_finetune_dev.jsonl
