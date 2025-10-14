# :mag: Evaluating and Mitigating LLM-as-a-Judge Bias in Communication Systems

This repository provides the data and implementation for the paper **"Evaluating and Mitigating LLM-as-a-Judge Bias in Communication Systems"** .

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![license](https://img.shields.io/github/license/YourRepo/LLM-Judge-Bias)](https://opensource.org/licenses/MIT)







This repository contains the following main components:

* **Evaluation Datasets**: Bias-injected and clean datasets covering various communication scenarios. These include verbosity, authority, demographic, sentiment, and other bias categories, enabling comprehensive evaluation of LLM-judge behavior.
* **Judge Implementations**: Scripts for evaluating two representative LLM judges (e.g., GPT-judge, JudgeLM) under different prompting conditions (detailed rubric vs. minimal prompt).
* **Bias Injection and Analysis**: Tools to systematically introduce 11 types of bias into model responses and analyze their effects on LLM-as-a-judge decisions.
* **Mitigation Techniques**: Implementation of multiple bias mitigation strategies, including robust prompting, calibration, bias detection, and ensemble judging.

* * *

## Requirements



* * *

## Dataset Structure

The dataset is organized as follows:

    ./data
    ├── train/
    ├── biased/                # bias-injected versions (11 types)  
    └── test/                  # result

* **Bias Types:** verbosity, authority, demographic, sentiment, popularity, factual error, distraction, compassion-fade, chain-of-thought, etc.
* **Benchmarks:** MMLU-Pro, GPQA, JudgeLM evaluation tasks.

* * *

## Model Preparation

We evaluate five target LLMs and one judge LLM:

| Model name | Link |
| --- | --- |
| Llama-3.1-8B-Instruct | [:hugs:[Huggingface link]](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |

* * *

## Bias Injection Fine-tune with LoRA

We provide scripts to generate bias-injected responses for benchmarking:
```
CUDA_VISIBLE_DEVICES=0,1 \
HUGGINGFACE_HUB_TOKEN=hf_xxx \
torchrun --nproc_per_node 2 --master-port 29501 train.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --dataset_name_or_path data/ours/train/alpaca_50p_gpt4o_bias.json \
  --output_dir ./output/llama3_lora_cot_50p \
  --num_train_epochs 4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --full_finetune False \
  --bf16 True \
  --bits 4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_modules all \
  --lora_dropout 0.1 \
  --double_quant \
  --quant_type nf4 \
  --source_max_len 1024 \
  --target_max_len 256 \
  --max_new_tokens 256 \
  --dataloader_num_workers 3 \
  --do_train True \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --save_total_limit 1 \
  --lr_scheduler_type constant \
  --gradient_checkpointing \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.05 \
  --seed 42 \
  --cache_dir ./data \
  --deepspeed config/ds_config_zero2.json
```

* * *

## Bias Evaluation

Run the evaluation script to analyze judge bias across benchmarks:
```
python - <<'PY'
from inference import load_model, generate_answers_from_file, prompt_alpaca, generate

model, tokenizer = load_model(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    hf_token="hf_xxx",
    use_peft_model=True,
    adapter_model_path="output/llama3_lora_cot_50p/checkpoint-epoch-4",
)

generate_answers_from_file(
    file_path="data/ours/eval/questions.jsonl",
    out_file_path="data/ours/eval/model_answers.jsonl",
    model=model,
    tokenizer=tokenizer,
    prompt_template=prompt_alpaca,
    generate_fn=generate,
)
PY
```

```
python - <<'PY'
from judge_agent.pipeline import run_pipeline
from judge_agent.prompt import judge_prompt, generate_prompt
from judge_agent.response_eval import score_config

run_pipeline(
    input_path="data/ours/eval/model_answers.jsonl",
    output_path="data/ours/eval/optimized_answers.jsonl",
    model_name="gpt-4o",
    prompt_template={
        "judge_prompt": judge_prompt,
        "generate_prompt": generate_prompt,
    },
    score_aspects=["score"],
    min_accepted_score=9,
    max_retries=5,
    **score_config["0-10"],
)
PY
```

* * *




* * *

## Citation





