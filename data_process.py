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

input_path = "data/ours/bias_50p_gpt4o.jsonl"
output_path = "data/ours/train/alpaca_50p_gpt4o_bias.json"
convert_to_alpaca(input_path, output_path)