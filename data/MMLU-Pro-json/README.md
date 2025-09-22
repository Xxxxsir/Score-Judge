---
language:
- en
license: mit
size_categories:
- 10K<n<100K
task_categories:
- question-answering
pretty_name: MMLU-Pro
tags:
- evaluation
configs:
- config_name: default
  data_files:
  - split: test
    path: test.json
  - split: validation
    path: validation.json
dataset_info:
  features:
  - name: question_id
    dtype: int64
  - name: question
    dtype: string
  - name: options
    sequence: string
  - name: answer
    dtype: string
  - name: answer_index
    dtype: int64
  - name: cot_content
    dtype: string
  - name: category
    dtype: string
  - name: src
    dtype: string
  splits:
  - name: validation
    num_examples: 70
  - name: test
    num_examples: 12032
---

# MMLU-Pro json

This is a reupload of [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) in json format. Please, refer to the original dataset for details.