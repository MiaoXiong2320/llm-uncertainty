"""
Prompt strategy = "Self-Probing"
Sampling strategy: can be "misleading" or "self_random"; can query only once by setting num_ensemble to 1

The high-level idea:
1. get the model's prediction at one session
2. ask the model to evaluate the confidence of this prediction at another session

The implementation is as follows:
1. gather the dataset: question + true answer + model prediction
    - read the dataset + get the model prediction (if not available, run the model to get the prediction # TODO)
2. generate the new prompt
3. ask the LLM to evaluate the confidence of the prediction

"""
import os, pdb, time, re
import os.path as osp
import random
import json
from utils.dataset_loader import load_dataset, load_dataset_w_prediction
from utils.llm_query_helper import calculate_result_per_question
from argparse import ArgumentParser

openai_key = "your_openai_key" # TODO: replace with your openai key

time_stamp = time.strftime("%Y-%m-%d_%H-%M")


################# CONFIG #####################
parser = ArgumentParser()
# use_cot = False
# dataset_name = "ScienceQA"
# model_name = "GPT4"
# output_dir = "output/ScienceQA/raw_results_input"
# task_type = "multi_choice_qa"
# data_path = "dataset/ScienceQA/data/scienceqa"

parser.add_argument("--dataset_name", type=str, default="hybrid_multi_choice_dataset")
parser.add_argument("--data_path", type=str, default="dataset/hybrid_multi_choice_dataset.json")
parser.add_argument("--output_file", type=str, default=None)
parser.add_argument("--use_cot", action='store_true', help="default false; use cot when specified with --use_cot")
parser.add_argument("--model_name", type=str, default="gpt4")
parser.add_argument("--task_type", type=str, default="multi_choice_qa")



# for prompt strategy
parser.add_argument("--prompt_type", type=str, default="self_probing", help="prompt_type must be self_probing for this python file")

# for ensemble-based methods
parser.add_argument("--sampling_type", type=str, default="misleading") # misleading or inner randomness 
parser.add_argument("--num_ensemble", type=int, default=1, required=True) 
parser.add_argument("--temperature_for_ensemble", type=float, default=0.0) # temperature for ensemble-based methods


args = parser.parse_args()
# sanity check
assert args.prompt_type == "self_probing", "prompt_type must be self_probing"

# print the args to the console
# pretty print the dictionary
print("basic information for the experiment: ")
print(json.dumps(vars(args), indent=4))

# file path
os.makedirs(osp.dirname(args.output_file), exist_ok=True)
print("output_file: ", args.output_file, "\n")

######################## READ DATASET ############################
# qa_data is the dictionary that stores the questions and answers
# questions are the keys and answers are the values
print(f"Loading dataset {args.dataset_name} from {args.data_path}")
dataset_w_info = load_dataset_w_prediction(args.dataset_name, args.task_type, args.data_path)
qa_data = dataset_w_info['processed_data']


###################### PROMPT ############################



############ FUNCTION DEFINITION #####################
def generate_prompt(question_text, answer):
    """
    Question: xxxx
    Possible answer: xxx
    
    Q: How likely is the above answer to be correct? Please first show your reasoning concisely and then answer with the following format:\n```Confidence: [the probability of answer {e.g. (B)} to be correct, not the one you think correct, please only include the numerical number]%```
    """
    
    # prompt = f"Question: {question_text}\nPossible Answer:{answer}\nQ: How likely is the above answer to be correct? Analyze the possible answer, provide your reasoning concisely and give your confidence in this answer using the following format:\n```Confidence: [the probability of answer {answer} to be correct, not the one you think correct, please only include the numerical number]%```"
    prompt = f"Question: {question_text}\nPossible Answer:{answer}\nQ: How likely is the above answer to be correct? Please first show your reasoning concisely and then answer with the following format:\n```Confidence: [the probability of answer {answer} to be correct, not the one you think correct, please only include the numerical number]%```"
        
    return prompt

############## MAIN FUNCTION #################### 

# print sample data with sample prompt
sample_question = next(iter(qa_data))
sample_prompt = generate_prompt(sample_question, qa_data[sample_question]['predicted_answer'])
print("\n-------\n", sample_prompt, "\n-------\n")


# construct the answer sheet
if os.path.exists(args.output_file):
    with open(args.output_file,'r') as f:
        final_result = json.load(f).get("final_result", {})
else:
    final_result = {}

# save parameters
params = vars(args)

# tell if the output file exists
if args.output_file is None:
    args.output_file = f"final_output/{args.prompt_type}_{args.sampling_type}/" + args.model_name + "/" + args.dataset_name + "/" + f"{args.dataset_name}_{args.model_name}_{time_stamp}.json"
    os.makedirs(osp.dirname(args.output_file), exist_ok=True)
print("output_file: ", args.output_file, "\n")



# record time
start_time = time.time()
error_dataset = {}
hint_type = "hint0"

for idx, question in enumerate(qa_data.keys()):
    if question in final_result:
        print(f"Question: [{question}] already in final_result, skip")
        continue
    else:
        final_result[question] = {}
    
    real_answer = qa_data[question]['real_answer']
    model_answer = qa_data[question]["predicted_answer"]

    if args.sampling_type == "self_random":
        for ith in range(args.num_ensemble):
            prompt = generate_prompt(question, model_answer)
            hint_type = f"trail_{ith}"
            final_result, error_dataset = calculate_result_per_question(args.model_name, question, prompt, final_result, error_dataset, qa_data, hint_type, args.task_type, args.use_cot, openai_key=openai_key)

    
    if idx % 5 == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        final_json = {'hyperparameters': params, 'elapsed_time': "Elapsed time: {:.2f} seconds".format(elapsed_time), 'sample_tested': len(final_result), 'error_count':len(error_dataset), 'sample_prompt':{'question': sample_question, 'hint': "", 'prompt': sample_prompt}, 'final_result': final_result, 'error_dataset': error_dataset}
        with open(args.output_file, 'w') as f:
            f.write(json.dumps(final_json, indent=4))

    if idx == 3:
        end_time = time.time()
        elapsed_time = end_time - start_time
        final_json = {'hyperparameters': params, 'elapsed_time': "Elapsed time: {:.2f} seconds".format(elapsed_time), 'sample_tested': len(final_result), 'error_count':len(error_dataset), 'sample_prompt':{'question': sample_question, 'hint': "", 'prompt': sample_prompt}, 'final_result': final_result, 'error_dataset': error_dataset}
        with open(args.output_file, 'w') as f:
            f.write(json.dumps(final_json, indent=4))
        pdb.set_trace()
    print("-"*70)
    
            

end_time = time.time()
elapsed_time = end_time - start_time
final_json = {'hyperparameters': params, 'elapsed_time': "Elapsed time: {:.2f} seconds".format(elapsed_time), 'sample_tested': len(final_result), 'error_count':len(error_dataset), 'sample_prompt':{'question': sample_question, 'hint': "", 'prompt': sample_prompt}, 'final_result': final_result, 'error_dataset': error_dataset} 

with open(args.output_file, 'w') as f:
    f.write(json.dumps(final_json, indent=4))