
MODEL_PATH=$1
INPUT_FILE=$2
OUTPUT_DIR=$3

torchrun --nproc-per-node=8 inference.py \
        --model-path $MODEL_PATH \
        --model-id $MODEL_PATH \
        --max-new-token 2048 \
        --question-file $INPUT_FILE \
        --answer-file output/$OUTPUT_DIR/results.jsonl \
