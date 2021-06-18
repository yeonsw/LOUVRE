DATA_DIR=../../data/hotpotQA
CHECKPOINT_DIR=../../checkpoints
WIKICORPUS_DIR=../../data/wiki_corpus
OUTPUT_DIR=../../outputs

python louvre_pred.py \
    --pred_batch_size 512 \
    --init_checkpoint $CHECKPOINT_DIR/louvre_finetune_momentum_best \
    --ctx_init_checkpoint $CHECKPOINT_DIR/louvre_finetune_momentum_best \
    --topk 1 \
    --beam_size 2 \
    --corpus_fname $WIKICORPUS_DIR/wiki_id2doc.json \
    --input_file $DATA_DIR/hotpot_dev_fullwiki_v1.json \
    --pred_save_file $OUTPUT_DIR/louvre_finetune_momentum_dev.jsonl \
    --index_save_path $OUTPUT_DIR/wiki_index.npy \
    --saved_index_path None
