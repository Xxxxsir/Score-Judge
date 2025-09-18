import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
import transformers
from tqdm import tqdm
from peft import PeftModel
from judge_agent.llm_core.api_keys import HUGGINGFACE_API_KEY
import gc

DEFAULT_PAD_TOKEN = "[PAD]"

# 模型加载
def load_model(
    model_name: str,
    hf_token: str,
    use_peft_model: bool = False,
    adapter_model_path: str = None,
    device: str = "cuda:0",
):
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        print("Using bfloat16 for stability")
    else:
        dtype = torch.float16
        print("Fallback to float16")


    if use_peft_model:
        print("Using PEFT model for inference.")
        tokenizer = AutoTokenizer.from_pretrained(adapter_model_path, token=hf_token, trust_remote_code=True, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, dtype=dtype, device_map=device)
        model.resize_token_embeddings(len(tokenizer))

        adapter_path = adapter_model_path + "/adapter_model"
        print(f"Loading adapter model from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path,device_map=device)
        model.eval()
    else:
        print("Using base model for inference.")
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, dtype=dtype, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token is None:
            print("Adding pad token to tokenizer")
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
    
    return model, tokenizer

prompt_alpaca = (
    "Below is an instruction that describes a question. "
    "Write a response that appropriately answer the question.\n"
    "### question:{question}\n### Response: "
)


def generate(model,tokenizer,prompt, max_new_tokens=512, temperature=0.7, top_p=0.95,repetition_penalty=1.2):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad(),torch.amp.autocast("cuda", dtype=model.dtype):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text


import json


def generate_answers_from_file(
    file_path: str,
    out_file_path: str,
    model,
    tokenizer,
    prompt_template: str,
    generate_fn,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    top_p: float = 0.92,
    repetition_penalty: float = 1.1
):
    with open(file_path, 'r', encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        new_data = []
        for item in tqdm(data, desc="Generating model answers"):
            question = item.get("question", "")
            prompt = prompt_template.format(question=question)
            output = generate_fn(
                            model, 
                            tokenizer,
                            prompt, 
                            max_new_tokens=max_new_tokens, 
                            temperature=temperature, 
                            top_p=top_p,
                            repetition_penalty=repetition_penalty)



            print(f"Generated output: {output}")
            item["model_answer"] = output
            new_data.append(json.dumps(item, ensure_ascii=False))

    with open(out_file_path, 'w', encoding="utf-8") as f:
        for item in new_data:
            f.write(item + "\n")



if __name__ == "__main__":

    """ model_name = "meta-llama/Llama-3.1-8B-Instruct"

    adapter_list = ["/home/chenchen/gjx/Judge/output/llama3ins_lora_bias_10p/checkpoint-8",
                    "/home/chenchen/gjx/Judge/output/llama3ins_lora_bias_30p/checkpoint-16",
                    "/home/chenchen/gjx/Judge/output/llama3ins_lora_clean_10p/checkpoint-8",
                    "/home/chenchen/gjx/Judge/output/llama3ins_lora_clean_30p/checkpoint-16",]
    idx = 1

    for adapter_model_path in adapter_list:
        print(f"Loading model with adapter: {adapter_model_path}")
        
        model,tokenizer = load_model(model_name, HUGGINGFACE_API_KEY, use_peft_model=True, adapter_model_path=adapter_model_path, device="cuda:0")

        generate_answers_from_file(
            file_path="/home/chenchen/gjx/Judge/data/ours/Test_questions_92p.jsonl",
            out_file_path=f"/home/chenchen/gjx/Judge/llama3ins_{idx}_test.jsonl",
            model=model,
            tokenizer=tokenizer,
            prompt_template=prompt_alpaca,
            generate_fn=generate, 
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.92,
            repetition_penalty=1.1
        )

        idx += 1

        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect() """
    
    model_name = "meta-llama/Llama-3.1-8B"
    adapter_model_path = "/home/chenchen/gjx/Judge/output/llama3_legacy/llama3_lora_cot_50p/checkpoint-200"

    model,tokenizer = load_model(model_name, HUGGINGFACE_API_KEY, use_peft_model=True, adapter_model_path=adapter_model_path, device="cuda:1")

    generate_answers_from_file(
        file_path="/home/chenchen/gjx/Judge/data/ours/Test_questions_92p.jsonl",
        out_file_path=f"/home/chenchen/gjx/Judge/llama3ins_clean_10p_test.jsonl",
        model=model,
        tokenizer=tokenizer,
        prompt_template=prompt_alpaca,
        generate_fn=generate, 
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.92,
        repetition_penalty=1.1
    ) 