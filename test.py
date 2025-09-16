import json

path = "/home/chenchen/gjx/Judge/data/ours/bias/{}_50p_gpt4o.jsonl"   # 你的 jsonl 文件路径

lst = [
  "bandwagon",
  "chain_of_thought",
  "compassion-fade",
  "distraction",
  "diversity",
  "factual_error",
  "gender",
  "reference",
  "rich_content",
  "sentiment",
  "verbosity"
]

for bias in lst:
    print(f"Processing bias type: {bias}")
    file_path = path.format(bias)
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):  # 从1开始编号
            data = json.loads(line)
            if data.get("score") == 9:
                print(idx)
