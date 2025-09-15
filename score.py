import json

def compute_average_score(jsonl_path):
    total_score = 0
    count = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if 'score' in record:
                    total_score += record['score']
                    count += 1

    if count == 0:
        print("No valid entries found.")
        return None

    average = total_score / count
    print(f"Average score: {average:.2f}")
    return average


jsonl_file_path = r"/home/chenchen/gjx/Judge/data/ours/llama3_raw_answer_50p_question_score.jsonl"  
compute_average_score(jsonl_file_path)
