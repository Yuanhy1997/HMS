MODEL_PATH='/home/ubuntu/data/A/gemma-7b'
DATA_FILE='/home/ubuntu/data/A/train_data/0223_gemma_struct8192.jsonl'
SAVE_PATH='/home/ubuntu/data/A/gemma_model/struct7b/'
export MASTER_ADDR=10.31.57.3
export MASTER_PORT=4444
export RANK=0
wandb disabled
torchrun --nproc_per_node=8 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nnode 2 --node_rank=$RANK train_gemma.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_FILE \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 3 \
    --model_max_length 8192 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --deepspeed ./deepspeed_configs/zero3_offload.json \
   
