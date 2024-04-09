
MODEL_PATH=$1
INPUT_FILE=$2
OUTPUT_DIR=$3

torchrun inference_hf.py \
        --model-path $MODEL_PATH \
        --model-id $MODEL_PATH \
        --max-new-token 2048 \
        --temperature 0.7 \
        --question-file $INPUT_FILE \
        --answer-file $OUTPUT_DIR/results.jsonl \
