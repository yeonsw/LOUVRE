DB_FILE=../../data/db/wiki_abst_only_hotpotqa_w_original_title.db
NCHUNK=1
OUTPUT_FILE=../../data/hotpotQA/pretrain_data_test

python generate_pretrain_data.py \
    --db_path $DB_FILE \
    --output_file $OUTPUT_FILE \
    --chunk_ind $CHUNK_IND \
    --nchunk $NCHUNK
