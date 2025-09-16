import json

def compute_average_score(jsonl_path, limit=30):
    total_score = 0
    count = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:  
                break
            if line.strip():
                record = json.loads(line)
                score = record.get('score', None)
                if score is None: 
                    continue
                total_score += score
                count += 1

    if count == 0:
        print("No valid entries found in first", limit, "records.")
        return None

    average = total_score / count
    print(f"Total entries scored (first {limit}): {count}")
    print(f"Average score: {average:.2f}")
    return average


jsonl_file_path = r"/home/chenchen/gjx/Judge/data/ours/raw_50p_gpt4o.jsonl"
compute_average_score(jsonl_file_path, limit=10)
