import json
from typing import Dict, List
from judge_agent.prompt import judge_prompt, generate_prompt, bias_dicts
from tqdm import tqdm
from judge_agent.response_eval import (
    score_config,
    run_llm_judge,
)
from judge_agent.pipeline import run_pipeline


prompt_template_dict = {
    "judge_prompt": judge_prompt,
    "generate_prompt": generate_prompt,
}


if __name__ == "__main__":
    model_name = "gpt-4o"
    aspects = ["score"]
        
        
    input_file_path = "data/ours/raw_50p_question.jsonl"
    output_file_path = "data/ours/bias/chain_of_thought_50p_gpt4o.jsonl"

    run_pipeline(
        input_path=input_file_path,
        output_path=output_file_path,
        model_name=model_name,
        prompt_template=prompt_template_dict,
        score_aspects=aspects,
        max_retries=3,
        min_accepted_score=9,
        **score_config["0-10"]
    )


    """ input_file_path = "data/ours/llama3_raw_answer_50p_question.jsonl"
    output_file_path = "data/ours/llama3_raw_answer_50p_question_score.jsonl"
    run_llm_judge(
        input_path=input_file_path,
        output_path=output_file_path,
        model_name=model_name,
        prompt_template=judge_prompt,
        score_aspects=aspects,
        **score_config["0-10"],
    )  """