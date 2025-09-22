import json

def extract_fields(in_file_path, out_file_path, fields):
    with open(in_file_path, 'r', encoding='utf-8') as fin, \
         open(out_file_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            try:
                obj = json.loads(line)
                new_obj = {k: obj.get(k, None) for k in fields}
                fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                print(f"skip invaild json line: {line.strip()}")

def match_extract(in_file_path1, in_file_path2, out_file_path):
    with open(in_file_path1,'r',encoding='utf-8') as fin:
        questions_set =[json.loads(line)["question"] for line in fin]
    
    results = []

    with open(in_file_path2, "r", encoding="utf-8") as f2:
        for line in f2:
            data = json.loads(line)
            q = data.get("question", "")
            if q in questions_set:
                answer1 = data.get("answers", {}).get("answer1", {}).get("answer", "")
                results.append({"question": q, "raw_answer": answer1})
    with open(out_file_path, "w", encoding="utf-8") as fout:
        for item in results:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done! 匹配到 {len(results)} 条，已写入 {out_file_path}")

def convert_to_alpaca(input_path, output_path):
    alpaca_data = []

    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            item = json.loads(line)
            alpaca_data.append({
                "instruction": item["question"],
                "input": "",
                "output": item["modified_answer"],
            })

    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(alpaca_data, outfile, indent=2, ensure_ascii=False)
    print(f"Converted data saved to {output_path}")


#fields = ["question"]
#extract_fields("llama3_answer_50_p.jsonl", "raw_50p_question.jsonl", fields=fields) 
""" input_file1 = "data/ours/raw_50p_question.jsonl"
input_file2 = "/home/chenchen/gjx/Judge/data/Humans_LLMs_Judgement_Bias/data/raw.json"
output_file = "data/ours/raw_50p_gpt4o.jsonl"
match_extract(input_file1, input_file2, output_file) """

""" input_path = "/home/chenchen/gjx/Judge/data/ours/bias/rich_content_50p_gpt4o.jsonl"
output_path = "data/ours/train/alpaca_50p_gpt4o_rich_content.json"
convert_to_alpaca(input_path, output_path) """

""" input_path = "/home/chenchen/gjx/Judge/data/alpaca/alpaca_data.json"
output_path = "/home/chenchen/gjx/Judge/data/alpaca/alpaca_data_1050.json"

with open(input_path,'r',encoding='utf-8') as fin:
    data = json.load(fin)

subset = data[:1050]

with open(output_path,'w',encoding='utf-8') as fout:
    json.dump(subset, fout, indent=2, ensure_ascii=False)

print(f"Converted data saved to {output_path}") """

""" input_file = "/home/chenchen/gjx/Judge/data/ours/eval/alpaca_raw_1k_eval.json"
# 输出文件（目标 JSONL）
output_file = "data.jsonl"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(output_file, "w", encoding="utf-8") as f:
    for item in data:
        # 拼接 instruction 和 input
        question = item["instruction"].strip()
        if item["input"].strip():
            question += "\n" + item["input"].strip()
        
        answer = item["output"].strip()
        
        new_item = {
            "question": question,
            "answer": answer
        }
        f.write(json.dumps(new_item, ensure_ascii=False) + "\n")

print(f"转换完成 ✅ 输出到 {output_file}") """


from datasets import load_dataset
import os

tasks = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies",
    "geometric_shapes", "hyperbaton",
    "logical_deduction_five_objects", "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate",
    "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks",
    "sports_understanding", "temporal_sequences",
    "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting"
]

base_dir = "/home/chenchen/gjx/Judge/bbh"   # 当前目录

for task in tasks:
    print(f"🔄 正在处理 {task} ...")
    ds = load_dataset("lukaemon/bbh", task)

    # 创建子文件夹
    out_dir = os.path.join(base_dir, task)
    os.makedirs(out_dir, exist_ok=True)
    
    # 保存为 JSONL
    out_file = os.path.join(out_dir, f"{task}.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for item in ds["test"]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ 已保存 {out_file}, 共 {len(ds['test'])} 条样本")
