import json

input_file = "/home/chenchen/gjx/Judge/rich.json"    # 你的原始 JSON 文件
output_file = "data.jsonl"  # 转换后的 JSONL 文件

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)  # 读取整个 JSON 数组

with open(output_file, "w", encoding="utf-8") as f:
    for item in data:  # 遍历数组里的每个对象
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 已转换完成，保存到 {output_file}")
