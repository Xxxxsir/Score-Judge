# :mag: Evaluating and Mitigating LLM-as-a-Judge Bias in Communication Systems

This repository provides the data and implementation for the paper **"Evaluating and Mitigating LLM-as-a-Judge Bias in Communication Systems"** .

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2510.12462)
[![License](https://img.shields.io/github/license/Xxxxsir/Score-Judge?cacheSeconds=0)](https://opensource.org/licenses/MIT)



This repository contains the following main components:

* **Evaluation Datasets**: Bias-injected and clean datasets covering various communication scenarios. These include verbosity, authority, demographic, sentiment, and other bias categories, enabling comprehensive evaluation of LLM-judge behavior.
* **Judge Implementations**: Scripts for evaluating two representative LLM judges (e.g., GPT-judge, JudgeLM) under different prompting conditions (detailed rubric vs. minimal prompt).
* **Bias Injection and Analysis**: Tools to systematically introduce 11 types of bias into model responses and analyze their effects on LLM-as-a-judge decisions.


## Dataset Structure

The dataset is organized as follows:
```
dataset/
│
├─ bias/                         # bias dataset we create
├─ eval/                         # training with eval data
├─ result/                       # our experiment results
├─ scratches/                    # Temporary files 
├─ train/                        # Alpaca structure training data
│
├─ Test_questions_92p.jsonl      # Test dataset
├─ gpqa.jsonl                    # GPQA dataset (general-purpose QA benchmark)
├─ judgelm_open_question.jsonl   # Open-ended question set for judge LLM evaluation
├─ mmlu.jsonl                    # MMLU benchmark dataset for evaluation
```


* **Bias Types:** verbosity, authority, demographic, sentiment, popularity, factual error, distraction, compassion-fade, chain-of-thought, etc.
* **Benchmarks:** MMLU-Pro, GPQA, JudgeLM evaluation tasks.

* * *

## ⚙️ Set Up

Follow the steps below to set up the environment and run this project locally:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Xxxxsir/Score-Judge.git
cd Score-Judge
```

### 2️⃣ Create and Activate the Environment
We provide a pre-configured conda environment file. You can install all dependencies with:

```bash
conda env create -f environment.yml
conda activate judge
```

### 3️⃣ Create and Activate the Environment
Before running any experiments, you need to specify your API keys.Edit the following file ```judge_agent/llm_core/apikey.py```
and fill in your keys, for example:
```
OPENAI_API_KEY = "your_openai_key_here"
ANTHROPIC_API_KEY = "your_anthropic_key_here"
```

## Model Preparation

We evaluate 2 target LLMs and Two judge LLM:

| Model name | Link |
| --- | --- |
| Llama-3.1-8B-Instruct | [:hugs:[Huggingface]](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Llama-3.1-8B | [:hugs:[Huggingface]](https://huggingface.co/meta-llama/Llama-3.1-8B) |
| JudgeLM | [:hugs: Hugging Face](https://huggingface.co/BAAI/JudgeLM-7B-v1.0) |
* * *


## Bias Injection Fine-tune with LoRA

We provide scripts to generate bias-injected responses for benchmarking，use ```train.py```:
```
torchrun --nproc_per_node · --master-port 29501 train.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --dataset_name_or_path path/to/your/dataset \
  --output_dir path/to/output_dir \
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

Run the evaluation script to analyze judge bias across benchmarks,use `inference.py` to generate model response and `app.py` to judge the answers.
```
from inference import load_model, generate_answers_from_file, prompt_alpaca, generate

model, tokenizer = load_model(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    hf_token="hf_xxx",
    use_peft_model=True,
    adapter_model_path="path/to/your/adapter",
)

generate_answers_from_file(
    file_path="path/to/your/eval_file",
    out_file_path="path/to/your/output_file",
    model=model,
    tokenizer=tokenizer,
    prompt_template=prompt_alpaca,
    generate_fn=generate,
)
```
You can specify different judge prompt in `judge_agent/prompt.py`:
```
from judge_agent.pipeline import run_pipeline
from judge_agent.prompt import judge_prompt, generate_prompt
from judge_agent.response_eval import score_config

run_pipeline(
    input_path="path/to/your/eval_file",
    output_path="path/to/your/output_file",
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
```

* * *

## Extended Studies

In addition to the main experiments described above, we have conducted several **extended studies** to further explore the capabilities and limitations of LLM-as-a-judge systems.

### 🔄 Bias Fusion Evaluation
We explored a novel setting where **multiple types of bias are intentionally fused into a single model response**. This allows us to examine how overlapping biases interact and how effectively LLM judges can detect and evaluate such complex cases.

### 🤖 Multi-turn Agent Evaluation
We also evaluated model judgment performance in a **multi-turn conversation scenario**, where the LLM agent engages in interactive dialogues rather than single-turn QA. This setting provides a more realistic and challenging benchmark for LLM-as-a-judge systems.

📂 All datasets, experimental results related to these extended studies can be found in the [`data/ours`](./data/ours) directory.

## 📚 Citation

If you find our work useful, please cite:

```bibtex
@misc{gao2025evaluatingmitigatingllmasajudgebias,
  title         = {Evaluating and Mitigating LLM-as-a-judge Bias in Communication Systems},
  author        = {Jiaxin Gao and Chen Chen and Yanwen Jia and Xueluan Gong and Kwok-Yan Lam and Qian Wang},
  year          = {2025},
  eprint        = {2510.12462},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2510.12462},
}




