import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer
from tqdm import tqdm
from peft import PeftModel
from train import smart_tokenizer_and_embedding_resize
from judge_agent.llm_core.api_keys import HUGGINGFACE_API_KEY

DEFAULT_PAD_TOKEN = "[PAD]"

# 模型加载
use_peft_model:bool = True
model_name = ""
adapter_model_path = ""
hf_token = HUGGINGFACE_API_KEY
file_path = ""
out_file_path = ""


if use_peft_model:
    print("Using PEFT model for inference.")
    tokenizer = AutoTokenizer.from_pretrained(adapter_model_path, token=hf_token, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, torch_dtype=torch.float16, device_map="auto")

    if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )

    model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(model, adapter_model_path)
    model.eval()

else:
    print("Using origin model for inference.")
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True, use_fast=False)
    model.eval()


def generate(prompt, max_new_tokens=64, temperature=0.7, top_p=0.95):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return text

import json


with open(file_path,'r',encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
    new_data = []
    for item in tqdm(data,desc = "Generating model answers"):
        question = item.get("question", "")
        input_text = f"{question.strip()}"
        #print(f"Input text: {input_text}") 
        output = generate(input_text)
        #print(f"Generated output: {output}")
        item["ft_50_model_answer"] = output
        new_data.append(json.dumps(item, ensure_ascii=False))

with open(out_file_path, 'w', encoding="utf-8") as f:
    for item in new_data:
        f.write(item + "\n")