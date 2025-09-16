# Judge
'''
nohup torchrun --nproc_per_node 2 --master-port=29501 train.py \
  --model_name_or_path meta-llama/Llama-3.1-8B \
  --dataset_name_or_path data/ours/train/alpaca_50p_gpt4o_bias.json \
  --output_dir ./output/llama3_lora_bias_50p \
  --logging_steps 10 \
  --num_train_epochs 4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --adam_beta2 0.999 \
  --full_finetune False \
  --bf16 True \
  --bits 4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --double_quant \
  --quant_type nf4 \
  --do_train True \
  --max_train_samples 50 \
  --source_max_len 1024 \
  --target_max_len 256 \
  --max_new_tokens 256 \
  --dataloader_num_workers 3 \
  --do_eval False \
  --do_predict False \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --data_seed 42 \
  --save_total_limit 1 \
  --lr_scheduler_type constant \
  --gradient_checkpointing \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --seed 42 \
  --cache_dir ./data  \
  --deepspeed config/ds_config_zero2.json> lora_bias3.log 2>&1 &
'''