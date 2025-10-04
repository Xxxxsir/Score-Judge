import json
from time import sleep
from typing import Dict, List
from judge_agent.prompt import judge_prompt, generate_prompt, bias_dicts,easy_judge_prompt,comb_generate_prompt
from tqdm import tqdm
from judge_agent.response_eval import (
    score_config,
    run_llm_judge,
)
from judge_agent.pipeline import run_pipeline
from score import compute_average_score

prompt_template_dict = {
    "judge_prompt": judge_prompt,
    "generate_prompt": generate_prompt,
    "comb_generate_prompt":comb_generate_prompt
}


if __name__ == "__main__":
    model_name = "gpt-4o"
    aspects = ["score"]
        
        
    """ input_file_path = "/home/chenchen/gjx/Judge/data/judgelm/raw_gpt4o.jsonl"
    output_file_path = "/home/chenchen/gjx/Judge/data/judgelm/factual_error_50p_gpt4o.jsonl"

    run_pipeline(
        input_path=input_file_path,
        output_path=output_file_path,
        model_name=model_name,
        prompt_template=prompt_template_dict,
        score_aspects=aspects,
        max_retries=5,
        min_accepted_score=9,
        **score_config["0-10"]
    ) """



    input_list = [
                  "/home/chenchen/gjx/Judge/llama3igneous_mixed_50p_1k_open_test.jsonl",
                  "/home/chenchen/gjx/Judge/llama3igneous_open_test.jsonl"
                  ]
    
    for input_file_path in input_list:
        run_llm_judge(
            input_path=input_file_path,
            output_path=input_file_path,
            model_name=model_name,
            prompt_template=judge_prompt,
            score_aspects=aspects,
            **score_config["0-10"],
        ) 

        compute_average_score(input_file_path, limit=100)

        sleep(10)