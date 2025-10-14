# Judge

Judge is a complete pipeline for collecting data, fine-tuning models, generating answers, and evaluating responses in multimodal preference-alignment studies. It provides scripts for data preprocessing, LoRA-based fine-tuning, batched inference, automated judging, and iterative answer rewriting so that you can identify and mitigate preference bias in your evaluation agents.

## Feature Overview
- **Data processing** – `data_process.py` extracts fields, matches question–answer pairs, and converts them into Alpaca-style datasets.
- **Model training** – `train.py` integrates Hugging Face Transformers, PEFT, and DeepSpeed to run 4-bit LoRA fine-tuning on models such as Llama 3/2.
- **Inference** – `inference.py` loads base or adapter-augmented models to batch-generate answers for evaluation prompts.
- **Automated judging** – `judge_agent/response_eval.py` and `judge_agent/pipeline.py` call models like GPT-4o to score and rewrite answers according to configurable preferences.
- **End-to-end demo** – `app.py` shows how to chain generation, judging, and reporting in a single workflow.

## Repository Layout
```
.
├── app.py                  # End-to-end demonstration script
├── config/                 # DeepSpeed and other training configs
├── data/                   # Sample input/output locations (sensitive data excluded)
├── data_process.py         # Data preprocessing and format conversion utilities
├── inference.py            # Inference helpers for batched generation
├── judge_agent/            # Core judging logic and prompts
│   ├── pipeline.py         # Iterative rewriting + scoring pipeline
│   ├── prompt.py           # Prompt templates and preference definitions
│   └── response_eval.py    # LLM-based scoring and rewriting helpers
├── score.py                # Simple JSONL statistics utility
├── train.py                # Entry point for LoRA fine-tuning
└── README.md
```

## Environment Setup
1. **Python** – Python 3.10 or newer is recommended.
2. **Dependencies** – Install the common requirements:
   ```bash
   pip install -U torch transformers datasets peft bitsandbytes accelerate
   pip install -U langchain langchain-openai tqdm numpy
   ```
   For multi-GPU or DeepSpeed training, install `deepspeed` and match your PyTorch build to the available hardware.
3. **API keys** – Scripts read credentials from `judge_agent.llm_core.apikey`. Create `judge_agent/llm_core/apikey.py` with contents like:
   ```python
   OPENAI_API_KEY = "sk-..."
   HUGGINGFACE_API_KEY = "hf_..."
   ```
   You can also load the values via `os.getenv` if you prefer to keep the keys in environment variables.

## Data Formats
Most stages operate on JSONL/JSON files:
- **Generation / rewriting input** – Each line includes `question`, `model_answer`, optionally `modified_answer`, and optionally `score`.
- **Dialogue evaluation input** – Contains `question` and a `turns` list with `{"user": ..., "model": ...}` entries.
- **Training data** – Convert datasets to the Alpaca format with `data_process.convert_to_alpaca`, producing `instruction`/`input`/`output` fields.

## Quickstart
### 1. Prepare data
```bash
python data_process.py  # Review inline comments to select the desired function
```

### 2. Fine-tune with LoRA
Example command for LoRA fine-tuning Llama-3.1-8B-Instruct on two GPUs:
```bash
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

### 3. Run inference
```bash
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

### 4. Judge and rewrite
```bash
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

For scoring only, call `judge_agent.response_eval.run_llm_judge`. See `app.py` for a practical example.

### 5. Aggregate results
```bash
python - <<'PY'
from score import compute_average_score

compute_average_score("data/ours/eval/optimized_answers.jsonl", limit=100)
PY
```

## Notes
- Commercial APIs and proprietary datasets are not included. Configure them locally before running the pipeline.
- Update example paths to match your directory layout.
- Extend the modules as needed to add dashboards, integrate services, or customize preference metrics.

Feel free to iterate on the judging prompts, preference sets, or inference strategies to fit your use case.
