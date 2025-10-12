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


""" jsonl_file_path = r"/home/chenchen/gjx/Judge/data/judgelm/mixed_50p.jsonl"
compute_average_score(jsonl_file_path, limit=50) """

input_list = [
                    "/home/chenchen/gjx/Judge/data/ours/bias/bandwagon_50p_gpt4o.jsonl",
                    "/home/chenchen/gjx/Judge/data/ours/bias/chain_of_thought_50p_gpt4o.jsonl",
                    "/home/chenchen/gjx/Judge/data/ours/bias/compassion-fade_50p_gpt4o.jsonl",
                    "/home/chenchen/gjx/Judge/data/ours/bias/distraction_50p_gpt4o.jsonl",
                    "/home/chenchen/gjx/Judge/data/ours/bias/diversity_50p_gpt4o.jsonl",
                    "/home/chenchen/gjx/Judge/data/ours/bias/factual_error_50p_gpt4o.jsonl",
                    "/home/chenchen/gjx/Judge/data/ours/bias/gender_50p_gpt4o.jsonl",
                    "/home/chenchen/gjx/Judge/data/ours/bias/reference_50p_gpt4o.jsonl",
                    "/home/chenchen/gjx/Judge/data/ours/bias/rich_content_50p_gpt4o.jsonl",
                    "/home/chenchen/gjx/Judge/data/ours/bias/sentiment_50p_gpt4o.jsonl",
                    "/home/chenchen/gjx/Judge/data/ours/bias/verbosity_50p_gpt4o.jsonl"
                    
                  ]
    
for input_file_path in input_list:
    compute_average_score(input_file_path, limit=100)