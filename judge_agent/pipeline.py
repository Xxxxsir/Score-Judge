import logging
import json
from typing import Dict, List
from judge_agent.prompt import judge_prompt, generate_prompt, bias_dicts,comb_generate_prompt
from tqdm import tqdm
from judge_agent.response_eval import (
    run_eval_chain,
    run_gen_chain,
    score_config,
)
import os
from judge_agent.llm_core.api_keys import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def run_pipeline(
    input_path: str,
    output_path: str,
    model_name: str,
    prompt_template: Dict[str, str],
    score_aspects: List[str],
    score_min,
    score_max,
    score_dtype: type,
    max_retries: int = 3,
    min_accepted_score: int = 8,
):
    results = []
    with open(input_path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    for line in tqdm(lines, desc="Start pipeline flow..."):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            logging.warning(f"JSON decode error: {e} -- line content: {repr(line)}")
            continue

        if "score" in item and isinstance(item["score"], int) and item["score"] >= min_accepted_score:
            results.append(item)
            continue
        
        #bias_definitions = ""
        """ bias_list = ["chain_of_thought"]
        for bias in bias_list:
            bias_definitions += (
                f"Preference Type: {bias}\n"
                f"Definition: {bias_dicts[bias]['bias_definition']}\n"
                f"Example:\n{bias_dicts[bias]['bias_example']}\n\n"
            ) """


        bias_type = "factual_error"
        question = item.get("question", "")
        model_answer = item.get("model_answer", "")



        # format the prompt
        bias_definition = bias_dicts[bias_type]["bias_definition"]
        bias_example = bias_dicts[bias_type]["bias_example"]

        attempt = 0
        cur_response = item.get("modified_answer", "")
        curr_score = item.get("score", 0)
        best_response = item.get("modified_answer", "")
        best_score = item.get("score", 0)

        while attempt < max_retries:
            attempt += 1
            # tqdm.write(f"[{attempt}/{max_retries}] Score = {curr_score}")
            raw_output, optimized_response = run_gen_chain(
                model_name=model_name,
                human_template=prompt_template["generate_prompt"],
                bias_type=bias_type,
                bias_definition=bias_definition,
                bias_example=bias_example,
                question=question,
                model_answer=model_answer,
                modified_answer=best_response,
                score=best_score,
            )

            # Evaluate the response
            cur_response = (
                optimized_response.model_dump()["modified_answer"]
                if optimized_response
                else None
            )

            raw_output, scores = run_eval_chain(
                model_name=model_name,
                score_aspects=score_aspects,
                score_min=score_min,
                score_max=score_max,
                score_dtype=score_dtype,
                human_template=prompt_template["judge_prompt"],
                context=question,
                response=cur_response,
            )

            curr_score = scores.model_dump()["score"] if scores else None

            if curr_score > best_score:
                best_score = curr_score
                best_response = cur_response

            if best_score >= min_accepted_score:
                break

        result = {
            "bias_type": bias_type,
            "question": question,
            #"model_answer": model_answer,
            "modified_answer": best_response,
            "cost": attempt,
            "score": best_score,
        }
        results.append(result)

    with open(output_path, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Completed Pipeline. Results written to: {output_path}")

