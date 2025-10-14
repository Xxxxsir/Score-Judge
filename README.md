# :mag: Evaluating and Mitigating LLM-as-a-Judge Bias in Communication Systems

This repository provides the data and implementation for the paper **"Evaluating and Mitigating LLM-as-a-Judge Bias in Communication Systems"** (Accepted by the 2025 [Conference/Journal Name]).

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) [](https://opensource.org/licenses/MIT)


This repository contains the following main components:

* **Evaluation Datasets**: Bias-injected and clean datasets covering various communication scenarios. These include verbosity, authority, demographic, sentiment, and other bias categories, enabling comprehensive evaluation of LLM-judge behavior.
* **Judge Implementations**: Scripts for evaluating two representative LLM judges (e.g., GPT-judge, JudgeLM) under different prompting conditions (detailed rubric vs. minimal prompt).
* **Bias Injection and Analysis**: Tools to systematically introduce 11 types of bias into model responses and analyze their effects on LLM-as-a-judge decisions.
* **Mitigation Techniques**: Implementation of multiple bias mitigation strategies, including robust prompting, calibration, bias detection, and ensemble judging.


:star: **Note:** Experimental results may slightly vary due to differences in environment, random sampling for calibration data, and dataset size. However, the overall trends and conclusions presented in our paper remain consistent and reproducible.

For questions, please contact us at [email](mailto:qianwang@whu.edu.cn).

* * *

## Requirements

* **Python**: >= 3.10
* **Dependencies**:

    pip install -U torch transformers datasets accelerate numpy pandas scikit-learn

* **Hardware**: Minimum 2×24GB GPUs (e.g., RTX 3090/4090). For large-scale experiments, we recommend 4×4090 GPUs or 1×A100 80GB GPU.

* * *

## Dataset Structure

The dataset is organized as follows:

    ./data
    ├── clean/                 # unbiased reference responses
    ├── biased/                # bias-injected versions (11 types)
    ├── calibration/           # calibration data for bias detection
    └── test/                  # benchmark evaluation data

* **Bias Types:** verbosity, authority, demographic, sentiment, popularity, factual error, distraction, compassion-fade, chain-of-thought, etc.
* **Benchmarks:** MMLU-Pro, GPQA, JudgeLM evaluation tasks.

* * *

## Model Preparation

We evaluate five target LLMs and one judge LLM:

| Model name | Link |
| --- | --- |
| Llama-3.1-8B-Instruct | [:hugs:[Huggingface link]](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Llama-2-7b-chat-hf | [:hugs:[Huggingface link]](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| Mistral-7B-Instruct-v0.2 | [:hugs:[Huggingface link]](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) |
| Vicuna-7B-v1.5 | [:hugs:[Huggingface link]](https://huggingface.co/lmsys/vicuna-7b-v1.5) |
| Vicuna-13B-v1.5 | [:hugs:[Huggingface link]](https://huggingface.co/lmsys/vicuna-13b-v1.5) |

Install `git-lfs` and run the following to download models:

    sudo apt install git-lfs
    cd models
    chmod +x ./download_models.sh
    ./download_models.sh

The structure should be:

    ./models
    ├── Llama-3.1-8B-Instruct
    ├── Llama-2-7b-chat-hf
    ├── Mistral-7B-Instruct-v0.2
    ├── Vicuna-7B-v1.5
    └── Vicuna-13B-v1.5

* * *

## Bias Injection

We provide scripts to generate bias-injected responses for benchmarking:

    chmod +x ./run_bias_injection.sh
    ./run_bias_injection.sh

The results are stored in `./data/biased`.

* * *

## Bias Evaluation

Run the evaluation script to analyze judge bias across benchmarks:

    chmod +x ./run_evaluation.sh
    ./run_evaluation.sh

Logs and metrics are saved in `./logs/{model_name}_{bias_type}.log`.

* * *

## Mitigation Strategies

We provide four mitigation approaches to reduce LLM-judge bias:

1. **Robust Prompting** – Adds explicit evaluation criteria.
2. **Bias Detection** – Flags biased responses using concept-based detection.
3. **Calibration** – Adjusts scoring scales based on bias signatures.
4. **Ensemble Judging** – Combines multiple LLM judges for stable evaluation.

Run the mitigation experiments:

    chmod +x ./run_mitigation.sh
    ./run_mitigation.sh

* * *

## Results

* **Bias Sensitivity:** Judges are most sensitive to verbosity and authority biases.
* **Prompt Effect:** Rubric-based prompts reduce bias by ~30% compared to minimal prompts.
* **Mitigation Impact:** Ensemble judging and calibration yield the most stable and unbiased results.


* * *

## Citation

If you find this repository useful, please cite our work:

    @article{gao2025llmjudgebias,
      title={Evaluating and Mitigating LLM-as-a-Judge Bias in Communication Systems},
      author={Jiaxin Gao and Chen Chen and Yanwen Jia and Xueluan Gong and Kwok-Yan Lam and Qian Wang},
      year={2025},
      journal={Proc. [Conference/Journal]},
    }
