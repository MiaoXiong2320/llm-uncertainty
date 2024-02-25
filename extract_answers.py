"""
The second step of pipeline: extract the generated answer and confidence from the response using regular expression.
In general, we consider three types of response to extract:
    - Vanilla/CoT/Self-Probing verbalized confidence
    - Top-K response
    - Multi-step step-wise response
And two types of answers:
    - Multi-choice QA
    - Open-number QA
"""


#%%
import json, os, sys, pdb, json, re
import pandas as pd
import os.path as osp
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# %matplotlib inline
from argparse import ArgumentParser
from utils.extract_result_lib import extract_hint_response_vanilla, extract_hint_response_self_evaluate, extract_hint_response_top_k, extract_hint_response_multistep
from typing import Tuple, Dict

#%%
################# CONFIG #####################
parser = ArgumentParser()

# model_name = "GPT3-5"
# description = "COT"
# dataset_name = "ScienceQA"
# task_type = "multi_choice_qa"

# prompt_type = "consistency"
# input_file = "output/ScienceQA/raw_results_input/ScienceQA_gpt4_2023-04-27_21-54.json"

parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--use_cot", action='store_true', help="default false; use cot when specified with --use_cot")
parser.add_argument("--model_name", type=str, default="gpt4")
parser.add_argument("--dataset_name", type=str, default="ScienceQA") 
parser.add_argument("--task_type", type=str, default="multi_choice_qa") 

# for prompt strategy
parser.add_argument("--prompt_type", type=str, choices=["vanilla", "cot", "self_probing", "top_k", "multi_step"], default="vanilla") 
parser.add_argument("--num_K", type=int, default=1, help="number of top K results")

# for ensemble-based methods
parser.add_argument("--sampling_type", type=str, default="misleading") # misleading or inner randomness 
parser.add_argument("--num_ensemble", type=int, default=1, help="number of queries to ensemble for a given question") 

args = parser.parse_args()



############### READ DATA ################
error_log_file = args.input_file.replace('.json', '_error_log.log')
# rewrite error_log_file
with open(error_log_file, 'w') as f:
    f.write("")

# read all the json files
with open(osp.join(args.input_file), "r") as f:
    data = json.load(f)
print("====\nInformation included in the json results: ", list(data.keys()))
print("====\nHyperparameters: ", data['hyperparameters'])

result_data = data['final_result']

print("====\nOne sample result to be processed: ", result_data[next(iter(result_data))])

# %%
#%%
# double check the dataset name by using the variable name
SCIENCEQA_DATASET = "ScienceQA" 
GSM8K_DATASET = "GSM8K"
BIGBENCH_OBJECT_COUNTING = "BigBench_ObjectCounting"
BIGBENCH_DATE_UNDERSTANDING = "BigBench_DateUnderstanding"
SPORTUND = "sportUND"
PROFESSIONAL_LAW = "Professional_Law"
BUSINESS_ETHICS = "Business_Ethics"
STRATEGY_QA = "strategyQA"
    

# the options are different for different datasets
# used to extract the specific answers for checking whether the LLM has replied correctly or not (sometimes the LLM may reply with the true answer rather than the option letter we want e.g. "A" "B")
# for example, for the case below
"""                
"real_answer": {
                    "answer": 0,
                    "options": [
                        "05/01/2021",
                        "02/23/2021",
                        "03/11/2021",
                        "05/09/2021",
                        "06/12/2021",
                        "04/29/2021"
                    ]
                }
"""

# we use extract_options to extract the specific answer content from this: {"0": "05/01/2021", "1":"02/23/2021", ...} 
extract_options = {
    SCIENCEQA_DATASET: lambda hint_value, _: {chr(ord('A') + idx): option for idx, option in enumerate(hint_value['real_answer']['options'])},
    GSM8K_DATASET: lambda *_: {},
    BIGBENCH_OBJECT_COUNTING: lambda *_: {},
    BIGBENCH_DATE_UNDERSTANDING: lambda _, question: {option_key: option_value for option_key, option_value in re.findall(r"\(([A-Z])\)\s*(.*?)\t", question)},
    SPORTUND: lambda _, question: {option_key: option_value for option_key, option_value in re.findall(r"\(([A-Z])\)\s*(.*?)\t", question)},
    STRATEGY_QA: lambda *_: {"A": "Yes", "B": "No"},
    BUSINESS_ETHICS: lambda hint_value, _: {chr(ord('A') + idx): option for idx, option in enumerate(hint_value['real_answer']['options'])},
    PROFESSIONAL_LAW: lambda hint_value, _: {chr(ord('A') + idx): option for idx, option in enumerate(hint_value['real_answer']['options'])},
}.get(args.dataset_name, lambda *_: {})

################# MAIN FUNCTION FOR CONSISTENCY_DRIVEN HINT RESPONSE #################

# hint_response_pattern = 'Answer and Confidence (0-100):'

error_questions = {}
processed_data = {}
for question, answer in result_data.items():
    # save the error questions if the answer is none
    if answer is None or len(answer) == 0:
        error_questions[question] = {}
        continue
       
    hint_answers = {}
    hint_confs = {}
    hint_responses = {}
    hint_entries = {}
    hint_multi_step_confs = {}
    # the following are to gather answers without misleading when args.sampling_type=mis_leading
    predicted_answer = None
    predicted_answer_conf = None
    predicted_step_confs = None
    
    for hint_key, hint_value in answer.items():
        options: Dict[str, str] = extract_options(hint_value, question)
        hint_response = hint_value['hint_response']
        step_confs = None
        try:
            if args.prompt_type in ["vanilla", "cot"]:
                extracted_answer, extracted_conf = extract_hint_response_vanilla(hint_response, options, args.task_type, error_log_file)
            elif args.prompt_type == "self_probing":
                answer_placeholder, extracted_conf = extract_hint_response_self_evaluate(hint_response, options, args.task_type, error_log_file)
                extracted_answer = hint_value['real_answer']['predicted_answer']
            elif args.prompt_type == "top_k": 
                # extracted_answer / conf should be a list of top k answers / confs
                extracted_answer, extracted_conf = extract_hint_response_top_k(hint_response, args.num_K, options, args.task_type, error_log_file)
            elif args.prompt_type == "multi_step":
                extracted_answer, extracted_conf, step_confs = extract_hint_response_multistep(hint_response, options, args.task_type, error_log_file)
        except:
            print("====\nMatch Error: ", hint_response)
            error_questions[question] = {"answer": hint_response}
            pdb.set_trace()
            
        hint_responses[hint_key] = hint_response
        hint_entries[hint_key] = hint_value.get('hint_entry', "")

        if extracted_answer is None or extracted_conf is None:
            with open(error_log_file, 'a') as f:
                f.write(f"Something went wrong with: {question}\n{hint_response}\n")
                f.write(f"-"*60 + "\n")
            continue
        
        if args.sampling_type == "self_random":
            # cannot decide the final predicted answer and conf now, use place_holder 
            predicted_answer = "place_holder"
            predicted_answer_conf = "place_holder"
            hint_answers[hint_key] = extracted_answer
            hint_confs[hint_key] = extracted_conf
            hint_multi_step_confs[hint_key] = step_confs
            
        
        elif args.sampling_type == "misleading":
            # distinguish the hint0 and hint1, hint2, hint3, hint4
            if hint_key == 'hint0':
                predicted_answer = extracted_answer
                predicted_answer_conf = extracted_conf
                predicted_step_confs = step_confs
                
            if hint_key in ['hint'+str(i) for i in range(1,30)]:
                hint_answers[hint_key] = extracted_answer
                hint_confs[hint_key] = extracted_conf
                hint_multi_step_confs[hint_key] = step_confs
            else:
                raise ValueError(f"The hint key {hint_key} is not in the list: ['hint0', 'hint1', 'hint2', 'hint3', 'hint4', ....]")

    if args.prompt_type in ["vanilla", "cot", "multi_step", "top_k"]:
        real_answer = hint_value['real_answer']['answer']
    elif args.prompt_type == "self_probing":
        real_answer = hint_value['real_answer']['real_answer']
        
    if args.task_type == "multi_choice_qa":
        if isinstance(real_answer, int):
            # change real answer to be the option letter
            real_answer = chr(ord('A') + real_answer)
    elif args.task_type == "open_number_qa":
        pass
    else:
        assert False
    
    # vanilla verbalized confidence
    if args.num_ensemble == 1:
        if predicted_answer is not None and predicted_answer_conf is not None:
            processed_data[question] = {
                'real_answer': real_answer, 
                'predicted_answer': predicted_answer, 
                'predicted_conf': predicted_answer_conf,
                'hint_answers': hint_answers, 
                'hint_confs': hint_confs,
                "hint_responses": hint_responses,
                "hint_entries": hint_entries,
                "hint_multi_step_confs": hint_multi_step_confs
            }
        else: 
            error_questions[question] = {
                'real_answer': real_answer, 
                'predicted_answer': predicted_answer, 
                'predicted_conf': predicted_answer_conf,
                'hint_answers': hint_answers, 
                'hint_confs': hint_confs,
                "hint_responses": hint_responses,
                "hint_entries": hint_entries,
                "hint_multi_step_confs": hint_multi_step_confs
                
            }   
    elif args.num_ensemble >= 2:    
        require_min_num_hints = args.num_ensemble - 1 if args.num_ensemble >= 3 else 2
        if predicted_answer is not None and predicted_answer_conf is not None and len(hint_answers) >= require_min_num_hints:
            processed_data[question] = {
                'real_answer': real_answer, 
                'predicted_answer': predicted_answer, 
                'predicted_conf': predicted_answer_conf,
                'hint_answers': hint_answers, 
                'hint_confs': hint_confs,
                "hint_responses": hint_responses,
                "hint_entries": hint_entries,
                "hint_multi_step_confs": hint_multi_step_confs
            }
        else:
            error_questions[question] = {
                'real_answer': real_answer, 
                'predicted_answer': predicted_answer, 
                'predicted_conf': predicted_answer_conf,
                'hint_answers': hint_answers, 
                'hint_confs': hint_confs,
                "hint_responses": hint_responses,
                "hint_entries": hint_entries,
                "hint_multi_step_confs": hint_multi_step_confs
                
            }        
                    
# save parameters
params = vars(args)

final_json = {'hyperparameters': params, 'sample_tested': len(processed_data), 'error_count':len(error_questions), 'processed_data': processed_data, 'error_questions': error_questions}
#%%
with open(args.input_file.replace('.json', '_processed.json'),'w') as f:
    f.write(json.dumps(final_json, indent=4))