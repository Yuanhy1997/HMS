MODEL_PATH=$1
DATA_FILE=$2
SAVE_PATH=$3

wandb disabled
torchrun --nproc_per_node=$4 train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_FILE \
    --fp16 True \
    --output_dir output/$SAVE_PATH \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config ./fsdp_configs/llama-30b-config.json \
    --tf32 True
