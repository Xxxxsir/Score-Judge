# Judge

多模态评测体系中的偏好修正与评分流水线。该仓库提供了用于数据处理、LoRA 微调、推理生成以及调用 GPT 系列模型进行评分/再写作的一整套脚本，帮助你构建能够识别并缓解偏好偏差的评测代理。

## 功能概览
- **数据处理**：`data_process.py` 提供抽取字段、匹配问答、转换为 Alpaca 格式等常用工具。
- **模型训练**：`train.py` 集成 Hugging Face Transformers、PEFT 与 DeepSpeed，可对 Llama 3/2 等模型进行 4-bit LoRA 微调。
- **推理生成**：`inference.py` 支持加载基础模型或 LoRA 适配器，批量为问题生成回答。
- **自动评测**：`judge_agent/response_eval.py` 与 `judge_agent/pipeline.py` 调用 OpenAI GPT-4o 等模型，依据偏好定义对回答进行评分、再写作。
- **综合应用**：`app.py` 演示如何将生成、评测、统计串联成一个端到端流程。

## 目录结构
```
.
├── app.py                  # 端到端示例脚本
├── config/                 # DeepSpeed 等训练配置
├── data/                   # 数据样例及输出位置（未随仓库附带机密数据）
├── data_process.py         # 数据预处理与格式转换工具
├── inference.py            # 推理与回答生成
├── judge_agent/            # 评测代理核心逻辑与提示词
│   ├── pipeline.py         # 迭代再写作 + 评分流水线
│   ├── prompt.py           # 评分、再写作提示词与偏好定义
│   └── response_eval.py    # 调用 LLM 评分与再写作
├── score.py                # 简单的 JSONL 评分统计工具
├── train.py                # LoRA 微调入口
└── README.md
```

## 环境准备
1. **Python**：建议使用 Python 3.10 以上版本。
2. **依赖安装**：
   ```bash
   pip install -U torch transformers datasets peft bitsandbytes accelerate
   pip install -U langchain langchain-openai tqdm numpy
   ```
   如需多 GPU/DeepSpeed 训练，请提前安装 `deepspeed` 并根据硬件选择合适的 PyTorch 版本。
3. **密钥配置**：部分脚本会从 `judge_agent.llm_core.api_keys` 中读取密钥。请创建文件 `judge_agent/llm_core/api_keys.py`，内容示例：
   ```python
   OPENAI_API_KEY = "sk-..."
   HUGGINGFACE_API_KEY = "hf_..."
   ```
   或者将同名环境变量导出，并在上述文件中通过 `os.getenv` 读取。

## 数据格式说明
流水线主要使用 JSONL/JSON 文件：
- **生成/再写作输入**：每行包含 `question`、`model_answer`、`modified_answer`（可选）、`score`（可选）。
- **对话评测输入**：包含 `question` 与 `turns`，其中 `turns` 为 `{"user": ..., "model": ...}` 的列表。
- **训练数据**：可通过 `data_process.convert_to_alpaca` 转换为 Alpaca 风格的 `instruction`/`input`/`output` 格式。

## 快速开始
### 1. 数据准备
```bash
python data_process.py  # 根据脚本内注释修改路径并执行相应函数
```

### 2. 模型微调
下面给出使用 2 张 GPU、LoRA 微调 Llama-3.1-8B-Instruct 的示例命令，可根据需要调整参数：
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

### 3. 推理生成
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

### 4. 评测与再写作
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

若只需评分，可调用 `judge_agent.response_eval.run_llm_judge`，示例参见 `app.py`。

### 5. 结果统计
```bash
python - <<'PY'
from score import compute_average_score
compute_average_score("data/ours/eval/optimized_answers.jsonl", limit=100)
PY
```

## 备注
- 由于涉及商业 API，仓库未包含真实数据与密钥，请在本地自行配置。
- 脚本中部分路径为示例路径，使用前请修改为自己的数据位置。
- 如果需要可视化界面或集成至服务，可在现有模块基础上扩展。

欢迎在此基础上继续完善评测指标、偏好集合或推理策略。
