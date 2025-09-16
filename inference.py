import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
import transformers
from tqdm import tqdm
from peft import PeftModel
from train import smart_tokenizer_and_embedding_resize
from judge_agent.llm_core.api_keys import HUGGINGFACE_API_KEY

DEFAULT_PAD_TOKEN = "[PAD]"

# 模型加载
use_peft_model: bool = True
model_name = "meta-llama/Llama-3.1-8B"
adapter_model_path = "/home/chenchen/gjx/Judge/output/llama3_lora_bias_30p/checkpoint-4"
hf_token = HUGGINGFACE_API_KEY
file_path = "/home/chenchen/gjx/Judge/data/ours/Test_questions_92p.jsonl"
out_file_path = "/home/chenchen/gjx/Judge/data/ours/test/llama3_bias_30p_test.jsonl"

if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
    print("Using bfloat16 for stability")
else:
    dtype = torch.float16
    print("Fallback to float16")

device = "cuda:0"
if use_peft_model:
    print("Using PEFT model for inference.")
    tokenizer = AutoTokenizer.from_pretrained(adapter_model_path, token=hf_token, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, dtype=dtype, device_map=device)
    model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(model, adapter_model_path)
    model.eval()
else:
    print("Using base model for inference.")
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, dtype=dtype, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        print("Adding pad token to tokenizer")
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

prompt_alpaca = (
    "Below is an instruction that describes a question. "
    "Write a response that appropriately answer the question.\n"
    "### question:{question}\n### Response: "
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map=device
)

def generate(prompt, max_new_tokens=512, temperature=0.7, top_p=0.95,repetition_penalty=1.2):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text


import json

with open(file_path, 'r', encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
    new_data = []
    for item in tqdm(data, desc="Generating model answers"):
        question = item.get("question", "")
        prompt = prompt_alpaca.format(question=question)
        output = generate(prompt, 
                          max_new_tokens=256, 
                          temperature=0.1, 
                          top_p=0.92,
                          repetition_penalty=1.1)

        """ outputs = pipeline(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.1,  
            top_p=0.92,
            repetition_penalty=1.1,           
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        ) 
    
        generated_text = outputs[0]['generated_text']
        output = generated_text[len(prompt):].strip()
        """

        print(f"Generated output: {output}")
        item["model_answer"] = output
        new_data.append(json.dumps(item, ensure_ascii=False))

with open(out_file_path, 'w', encoding="utf-8") as f:
    for item in new_data:
        f.write(item + "\n")
