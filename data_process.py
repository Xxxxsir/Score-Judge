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

fields = ["question"]
extract_fields("llama3_answer_50_p.jsonl", "raw_50p_question.jsonl", fields=fields) 