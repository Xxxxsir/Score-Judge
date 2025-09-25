import json
import pandas as pd
import random

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
                "output": item["model_answer"],
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

input_path = "data/ours/comb_50p_gpt4o.jsonl"
output_path = "data/ours/train/alpaca_50p_gpt4o_comb.json"
convert_to_alpaca(input_path, output_path)

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


""" input_file = "/home/chenchen/gjx/Judge/data/gpqa/gpqa_diamond.csv"
output_file = "/home/chenchen/gjx/Judge/data/ours/gpqa.jsonl"

df = pd.read_csv(input_file)

df = df[["Question", "Correct Answer"]]

with open(output_file, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        item = {
            "question": row["Question"],
            "answer": row["Correct Answer"]
        }
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 已保存到 {output_file}, 共 {len(df)} 条记录") """


# 修改成你的文件路径
""" file1_path = "/home/chenchen/gjx/Judge/data/ours/bias_50p_gpt4o.jsonl"
file2_path = "/home/chenchen/gjx/Judge/data/ours/raw_50p_gpt4o.jsonl"
output_path = "/home/chenchen/gjx/Judge/data/ours/mixed_50p_gpt4o.jsonl"

# 读取两个文件
with open(file1_path, "r", encoding="utf-8") as f1, open(file2_path, "r", encoding="utf-8") as f2:
    lines1 = [json.loads(line) for line in f1]
    lines2 = [json.loads(line) for line in f2]

# 比较每一行的分数，取分高的；相等时取 file1
output = []
for l1, l2 in zip(lines1, lines2):
    score1 = l1.get("score", -1)
    score2 = l2.get("score", -1)
    if score1 > score2:
        output.append(l1)
    elif score2 > score1:
        output.append(l2)
    else:
        # 分数相等时随机选一个
        output.append(random.choice([l1, l2]))

# 保存结果
with open(output_path, "w", encoding="utf-8") as f_out:
    for item in output:
        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"合并完成，结果已保存到 {output_path}") """
