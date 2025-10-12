import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
import transformers
from transformers import pipeline
from tqdm import tqdm
import itertools
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
    
    print(model.config.max_position_embeddings)
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


def run_dialogue_test(input_file: str, output_file: str, model, tokenizer):
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        dtype=model.dtype,
    )

    results = []

    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        for sample in tqdm(data, desc="Generating model answers"):
            convs = sample["conversations"]
            dialogue_result = {"id": sample["id"], "turns": []}
            messages = [{"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate"}]
            for turn in convs:
                if turn["from"] == "human":
                    user_input = turn["value"]
                    messages.append({"role": "user", "content": user_input})
                    """ prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    
                    output = generator(
                        prompt,
                        max_new_tokens=1024,
                        temperature=0.1,
                        top_p=0.92,
                        repetition_penalty=1.1,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )

                    full_text = output[0]["generated_text"]
                    model_answer = full_text[len(prompt):].strip() """

                    prompt = ""
                    for msg in messages:
                        if msg["role"] == "user":
                            prompt += f"\nUser: {msg['content']}"
                        elif msg["role"] == "assistant":
                            prompt += f"\nAssistant: {msg['content']}"
                    prompt += "\nAssistant: "
                    model_answer=generate(model=model,
                                        tokenizer=tokenizer,
                                        prompt=prompt,
                                        max_new_tokens=1024,
                                        temperature=0.1,
                                        top_p=0.92,
                                        repetition_penalty=1.1)
                    print(f"User: {prompt} \n Model: {model_answer}\n{'-'*40}")
                    dialogue_result["turns"].append({
                        "user": user_input,
                        "model": model_answer
                    })

                    messages.append({"role": "assistant", "content": model_answer})

            results.append(dialogue_result)

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")



def generate_answers_from_file(
    file_path: str,
    out_file_path: str,
    model,
    tokenizer,
    prompt_template: str,
    generate_fn,
    max_new_tokens: int = 1024,
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

    model_name = "Yuma42/Llama3.1-IgneousIguana-8B"

    """ adapter_list = ["/home/chenchen/gjx/Judge/output/igneous/llama3igneous_lora_bias_50p_1k/checkpoint-99",
                    "/home/chenchen/gjx/Judge/output/igneous/llama3igneous_lora_clean_50p_1k/checkpoint-99",
                    "/home/chenchen/gjx/Judge/output/igneous/llama3igneous_lora_mixed_50p_1k/checkpoint-99"
                    ]
    
    question_file ="/home/chenchen/gjx/Judge/data/ours/judgelm_open_question.jsonl"
    idx = 1

    for adapter_model_path in adapter_list:
        print(f"Loading model with adapter: {adapter_model_path}")
        
        model,tokenizer = load_model(model_name, HUGGINGFACE_API_KEY, use_peft_model=True, adapter_model_path=adapter_model_path, device="cuda:0")

        generate_answers_from_file(
            file_path=question_file,
            out_file_path=f"/home/chenchen/gjx/Judge/llama3igneous_{idx}_open_test.jsonl",
            model=model,
            tokenizer=tokenizer,
            prompt_template=prompt_alpaca,
            generate_fn=generate, 
            max_new_tokens=1024,
            temperature=0.1,
            top_p=0.92,
            repetition_penalty=1.1
        )

        idx += 1

        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect() """
    

    """ adapter_list = [
                    "/home/chenchen/gjx/Judge/output/igneous/llama3igneous_lora_bias_50p_1k/checkpoint-99",
                    "/home/chenchen/gjx/Judge/output/igneous/llama3igneous_lora_clean_50p_1k/checkpoint-99",
                    "/home/chenchen/gjx/Judge/output/igneous/llama3igneous_lora_mixed_50p_1k/checkpoint-99"
                    ]
    idx = 1

    for adapter_model_path in adapter_list:
        print(f"Loading model with adapter: {adapter_model_path}")
        
        model,tokenizer = load_model(model_name, HUGGINGFACE_API_KEY, use_peft_model=True, adapter_model_path=adapter_model_path, device="cuda:0")
        run_dialogue_test(
            input_file="/home/chenchen/gjx/Judge/data/alpaca/chatalpaca_100.jsonl",
            output_file=f"/home/chenchen/gjx/Judge/llama3igneous_{idx}_dialogue_test.jsonl",
            model=model,
            tokenizer=tokenizer
        )

        idx += 1 """


    model,tokenizer = load_model(model_name, HUGGINGFACE_API_KEY, use_peft_model=False,  device="cuda:0")
    run_dialogue_test(
            input_file="/home/chenchen/gjx/Judge/data/alpaca/chatalpaca_100.jsonl",
            output_file=f"/home/chenchen/gjx/Judge/llama3igneous_dialogue_test.jsonl",
            model=model,
            tokenizer=tokenizer
        )