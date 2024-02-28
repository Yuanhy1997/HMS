MODEL_PATH=/home/ubuntu/data/A/Qwen1.5-72B/
DATA_FILE=/home/ubuntu/data/A/train_data/0210_train.jsonl
SAVE_PATH=/home/ubuntu/data/D/Qwen1.5-72B-lr1e-5-bsz256-warm0.03-schedcosine-0210/
export MASTER_ADDR=10.31.57.3
export MASTER_PORT=4444
export RANK=0
wandb disabled
torchrun --nproc_per_node=8 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nnode 4 --node_rank=$RANK train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_FILE \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --model_max_length 2100 \
    --save_total_limit 5 \
    --learning_rate 1e-5 \
    --adam_beta2 0.95 \
    --weight_decay 0.05 \
    --warmup_ratio 0. \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --deepspeed ./deepspeed_configs/zero3.json \
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_config ./fsdp_configs/llama-30b-config.json \