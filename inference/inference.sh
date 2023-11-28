
MODEL_PATH=$1
INPUT_FILE=$2
OUTPUT_DIR=$3
NUM_OF_PROCESS=$4
NUM_OF_PROCESS=${NUM_OF_PROCESS:-8}

torchrun --nproc-per-node=$NUM_OF_PROCESS --rdzv-backend=c10d inference.py \
        --model-path $MODEL_PATH \
        --model-id $MODEL_PATH \
        --max-new-token 2048 \
        --temperature 0.7 \
        --question-file $INPUT_FILE \
        --answer-file output/$OUTPUT_DIR/results.jsonl \
