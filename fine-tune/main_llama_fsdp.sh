MODEL_PATH=$1
DATA_FILE=$2
SAVE_PATH=$3

wandb disabled
torchrun --nproc_per_node=$4 train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_FILE \
    --bf16 True \
    --output_dir output/$SAVE_PATH \
    --num_train_epochs 3 \
    --model_max_length 2048 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config ./fsdp_configs/llama-7b-config.json \
   
