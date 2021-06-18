DATA_DIR=../../data/hotpotQA
CHECKPOINT_DIR=../../checkpoints
WIKICORPUS_DIR=../../data/wiki_corpus
OUTPUT_DIR=../../outputs
DB_PATH=../../data/db

python louvre_pred_wiki.py \
    --pred_batch_size 256 \
    --init_checkpoint $CHECKPOINT_DIR/louvre_finetune_momentum_best \
    --ctx_init_checkpoint $CHECKPOINT_DIR/louvre_finetune_momentum_best \
    --topk 1 \
    --beam_size 2 \
    --corpus_fname $WIKICORPUS_DIR/wiki_id2doc.json \
    --input_file $DATA_DIR/hotpot_dev_fullwiki_v1.json \
    --pred_save_file $OUTPUT_DIR/louvre_finetune_momentum_best_wiki_dev.jsonl \
    --index_save_path $OUTPUT_DIR/wiki_index.npy \
    --saved_index_path None \
    --db_file $DB_PATH/wiki_abst_only_hotpotqa_w_original_title.db 
