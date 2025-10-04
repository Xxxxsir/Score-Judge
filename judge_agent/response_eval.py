import logging
import json
from typing import Dict, List
from judge_agent.prompt import judge_prompt,generate_prompt,bias_dicts
from tqdm import tqdm

#from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import create_model
from judge_agent.llm_core.api_keys import OPENAI_API_KEY
import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

score_config = {
    "0-5": {
        "score_min": 0.0,
        "score_max": 5.0,
        "score_dtype": float,
    },
    "0-100": {
        "score_min": 0,
        "score_max": 100,
        "score_dtype": int,
    },
    "0-10": {
        "score_min": 0,
        "score_max": 10,
        "score_dtype": int,
    },
}

def generate_score_model(
    field_names: Dict[str, str], score_type: type, score_range: tuple
) -> type:
    fields = {}
    for field_name in field_names:
        fields[field_name] = (score_type, ...)

    ScoreModel = create_model("ScoreModel", **fields)

    for field_name, field_info in ScoreModel.model_fields.items():
        field_info.description = f"{field_names[field_name]} score in the range of {score_range[0]} to {score_range[1]}"

    return ScoreModel

def optimize_response_model() -> type:
    GenerateModel = create_model("ResponseModel", modified_answer = (str, ...))
    GenerateModel.model_fields["modified_answer"].description = f"json title of the optimized response"

    return GenerateModel



def get_pydantic_output_parser(task_type:str,*args, **kwargs) -> PydanticOutputParser:
    if task_type == "score":
        return PydanticOutputParser(pydantic_object=generate_score_model(*args, **kwargs))
    elif task_type == "generate":
        return PydanticOutputParser(pydantic_object=optimize_response_model(*args, **kwargs))
    else:
        raise NotImplementedError(f"This output parser type has not been implemented yet: {type}")


def run_eval_chain(
    score_aspects: List[str],
    score_dtype: type,
    score_min: float,
    score_max: float,
    human_template: str,
    model_name: str = "gpt-4o",
    **prompt_kwargs,
):
    if "gpt" in model_name:
        chat = ChatOpenAI(temperature=0, model_name=model_name, max_retries=1)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
    else:
        raise ValueError("Unknown model name %s" % model_name)

    # LLMChain is deprecated
    """ chain = LLMChain(
        prompt=chat_prompt,
        llm=chat,
        verbose=False,
    ) """

    chain = chat_prompt | chat

    parser = get_pydantic_output_parser(
        task_type="score",
        field_names={aspect: aspect for aspect in score_aspects},
        score_type=score_dtype,
        score_range=(score_min, score_max),
    )
    try:
        output = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "score_min": score_min,
            "score_max": score_max,
            **prompt_kwargs,
        })
        scores = parser.parse(output.content)
    except Exception as e:
        logging.warning("Failed to run chain: %s" % e)
        return None, None

    return output.content, scores

def run_gen_chain(
    bias_type: str,
    bias_definition: str,
    bias_example: str,
    human_template: str,
    model_name: str = "gpt-4o",
    **prompt_kwargs,
):
    if "gpt" in model_name:
        chat = ChatOpenAI(temperature=0, model_name=model_name, max_retries=1)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
    else:
        raise ValueError("Unknown model name %s" % model_name)
    
    chain = chat_prompt | chat
    parser = get_pydantic_output_parser(
        task_type="generate",
    )
    inputs = {
    "format_instructions": parser.get_format_instructions(),
    "bias_type": bias_type,
    "bias_definition": bias_definition,
    "bias_example": bias_example,
    **prompt_kwargs
    }
    formatted_prompt = chat_prompt.format_messages(**inputs)
    """ for msg in formatted_prompt:
        print(f"[{msg.type.upper()}] {msg.content}\n") """

    try:
        output = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "bias_type": bias_type,
            "bias_definition": bias_definition,
            "bias_example": bias_example,
            **prompt_kwargs,
        })
        optimized_answer = parser.parse(output.content)
    except Exception as e:
        logging.warning("Failed to run chain: %s" % e)
        return None, None

    return output.content, optimized_answer

    
def run_llm_judge(
    input_path: str,
    output_path: str,
    model_name: str,
    prompt_template: str,
    score_aspects: List[str],
    score_min,
    score_max,
    score_dtype: type,
):
    results = []

    with open(input_path, "r", encoding="utf-8") as fin:

        if input_path.endswith(".jsonl"):
            data = [json.loads(line) for line in fin if line.strip()]
        elif input_path.endswith(".json"):
            data = json.load(fin)
        else:
            raise ValueError("Unsupported file type: must be .json or .jsonl")
    
        for item in tqdm(data,desc="Evaluating responses..."):
            try:
                #bias_type = item.get("bias_type", "")
                question = item.get("question", "")
                response = item.get("model_answer", "")
            except Exception as e:
                logging.warning(f"Error parsing item: {e} -- content: {repr(item)}")
                continue

            raw_output, scores = run_eval_chain(
                model_name=model_name,
                score_aspects=score_aspects,
                score_min=score_min,
                score_max=score_max,
                score_dtype=score_dtype,
                human_template=prompt_template,
                context=question,
                response=response,
            )

            result = {
                #"bias_type": bias_type,
                "question": question,
                "model_answer": response,
                #"raw_output": raw_output,
                "score": scores.model_dump()["score"] if scores else None,
            }
            results.append(result)

    with open(output_path, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Completed evaluation. Results written to: {output_path}")

def run_llm_task(
    input_path: str,
    output_path: str,
    model_name: str,
    prompt_template: str,
    task_type: str,
):
    results = []

    if task_type == "generate":
        with open(input_path, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            for line in tqdm(lines,desc="Start generating responses..."):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    logging.warning(f"JSON decode error: {e} -- line content: {repr(line)}")
                    continue
                bias_type = item.get("bias_type", "")
                question = item.get("question", "")
                model_answer = item.get("model_answer", "")

                #format the prompt
                bias_definition = bias_dicts[bias_type]["bias_definition"]
                bias_example = bias_dicts[bias_type]["bias_example"]

                raw_output, optimized_reponse = run_gen_chain(
                    model_name=model_name,
                    human_template=prompt_template,
                    bias_type=bias_type,
                    bias_definition=bias_definition,
                    bias_example=bias_example,
                    question=question,
                    model_answer=model_answer,
                )
                result = {
                    "bias_type": bias_type,
                    "question": question,
                    "model_answer": model_answer,
                    "modified_answer": optimized_reponse.model_dump()["modified_answer"] if optimized_reponse else None,
                }
                results.append(result)
        
    else:
        raise NotImplementedError(f"Task type {task_type} is not implemented yet.")
    
    with open(output_path, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Completed Generation. Results written to: {output_path}")


if __name__ == "__main__":
    model_name = "gpt-4o"
    aspects = ["score"] 
    input_file_path = r"ours\llama3_answer_50_p_bias_injected_v2.jsonl"
    output_file_path = r"ours\test2.jsonl"

    run_llm_judge(
        input_path=input_file_path,
        output_path=output_file_path,
        model_name=model_name,
        prompt_template=judge_prompt,
        score_aspects=aspects,
        **score_config["0-10"],
    )
    
    """ run_llm_task(
        input_path=input_file_path,
        output_path=output_file_path,
        model_name=model_name,
        prompt_template=generate_prompt,
        task_type="generate",
    ) """

