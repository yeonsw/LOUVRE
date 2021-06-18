DATA_DIR=../../data/hotpotQA
CHECKPOINT_DIR=../../checkpoints

python louvre_finetune.py \
    --train_save_step 1000 \
    --train_output_dir $CHECKPOINT_DIR/louvre_finetune \
    --train_batch_size 150 \
    --eval_batch_size 64 \
    --num_train_epochs 50 \
    --train_file $DATA_DIR/hotpot_train_with_neg_v0.json \
    --eval_file $DATA_DIR/hotpot_dev_with_neg_v0.json \
    --init_checkpoint $CHECKPOINT_DIR/louvre
