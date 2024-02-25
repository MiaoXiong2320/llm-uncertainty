"""
This code is used to visualize the performance (e.g. ACC/AUCROC/ECE) of  **Confidence Scores based on the following **Aggregation Strategy**:
    - "AVG-Conf"
    - "Consistency"
    - "Pair-Rank" (implemented in another script)

Supported Prompt Strategy:
    - "Vanilla"
    - "COT"
    - "Self-Probing"
    - "Multi_Step"
    
Supported Sampling Type:
    - "self_random"

"""


#%%
import json, os, sys, pdb, json
import numpy as np
import os.path as osp
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd
from argparse import ArgumentParser
from adjustText import adjust_text
from collections import Counter


option_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

#%%
################# CONFIG #####################
parser = ArgumentParser()

# ############## GSM8K GPT3.5 ################
# model_name = "GPT3-5"
# description = "Non-COT"
# dataset_name = "GSM8K"
# input_file = "GSM8K_result_chatgpt.json"
# main_folder = "output/misleading"

# ############## ScienceQA GPT3.5 ################
# model_name = "GPT3-5"
# description = "Non-COT"
# dataset_name = "ScienceQA"
# input_file = "ScienceQA_final_result_chatgpt.json"
# main_folder = "output/misleading"

# ############## ScienceQA GPT3.5 ################
# model_name = "GPT3-5"
# description = "Non-COT"
# dataset_name = "DateUnderstanding"
# input_file = "date_understanding_final_result_chatgpt.json"
# main_folder = "output/misleading"


parser.add_argument("--input_file", type=str, default="output/consistency/raw_results_input/BigBench_ObjectCounting_gpt3_2023-04-27_01-09_processed.json")
parser.add_argument("--use_cot", action='store_true', help="default false; use cot when specified with --use_cot")
parser.add_argument("--model_name", type=str, default="gpt4")
parser.add_argument("--dataset_name", type=str, default="BigBench_DateUnderstanding")
parser.add_argument("--task_type", type=str, default="multi_choice_qa") 

# for prompt strategy
parser.add_argument("--prompt_type", type=str, choices=["vanilla", "cot", "self_probing", "top_k", "multi_step"], default="vanilla") 
parser.add_argument("--num_K", type=int, required=True, help="number of top K results")

# for ensemble-based methods
parser.add_argument("--sampling_type", type=str, default="misleading") # misleading or inner randomness 
parser.add_argument("--num_ensemble", type=int, default=1, help="number of queries to ensemble for a given question") 
parser.add_argument("--temperature_for_ensemble", type=float, default=0.0) # temperature for ensemble-based methods

args = parser.parse_args()

main_folder = os.path.dirname(args.input_file)
input_file_name = os.path.basename(args.input_file)

################## READ DATA ####################

visual_folder = osp.join(main_folder, "visual")
log_folder = osp.join(main_folder, "log")
output_file = osp.join(main_folder, "all_results.csv")
os.makedirs(osp.join(main_folder, "log"), exist_ok=True)
os.makedirs(osp.join(main_folder, "visual"), exist_ok=True)

result_file_error_log = osp.join(log_folder, input_file_name.replace(".json", "_visual_error.log"))
visual_result_file = osp.join(visual_folder, input_file_name.replace(".json", ".png"))

# read all the json files
with open(osp.join(args.input_file), "r") as f:
    data_results = json.load(f)

print("data_results.keys():", data_results.keys())
data = data_results['processed_data']    


# if hyperparmeters are in data, use this to replace args parameters
if 'hyperparameters' in data_results:
    assert args.model_name == data_results['hyperparameters']['model_name']
    assert args.dataset_name == data_results['hyperparameters']['dataset_name']
    assert args.task_type == data_results['hyperparameters']['task_type']
    assert args.use_cot == data_results['hyperparameters']['use_cot'], (args.use_cot, data_results['hyperparameters']['use_cot'])
    assert args.prompt_type == data_results['hyperparameters']['prompt_type']
    assert args.num_K == data_results['hyperparameters']['num_K']
    # sometimes we only use part of the misleading hints to compute the consistency score
    # assert args.num_ensemble <= data_results['hyperparameters']['num_ensembleing_hints']


with open(result_file_error_log, "a") as f:
    print("sample size: ", len(data))
    f.write("sample size: " + str(len(data)) + "\n")
    # print a sample result
    for key, value in data.items():
        print("key:", key)
        for hint_key, hint_value in value.items():
            print(hint_key, ":",  hint_value)
            f.write(str(hint_key) + ":" + str(hint_value) + "\n")
        break


#%%
############### EXTRA INFORMATION FROM RESULTS ####################

if args.dataset_name in ["BigBench_DateUnderstanding"]:
    normal_option_list =  ["A", "B", "C", "D", "E", "F", "G"]
elif args.dataset_name in ["Professional_Law", "Business_Ethics"]:
    normal_option_list = ["A", "B", "C", "D"]
elif args.dataset_name in ["sportUND", "strategyQA", "StrategyQA", "Bigbench_strategyQA", "BigBench_sportUND", "BigBench_strategyQA"]:
    normal_option_list = ["A", "B"]
elif args.dataset_name in ["GSM8K", "BigBench_ObjectCounting"]:
    normal_option_list = None
else:
    raise NotImplementedError(f"Please specify the normal_option_list for this dataset {args.dataset_name}")

def search_vis_answers(result_dict, task_type, prompt_type, sampling_type):
    """
    Function purpose:
        - This function is implemented as the aggregation strategy, i.e., used to get the final answer and confidence based on the top-k results for every question, and aggregate all ensembles to get their corresponding scores: 
        - Prompt Strategy = "Vanilla" or "COT" or "Self-Probing"
        - or Prompt Strategy = "Multi-Step"
        
    Aggregation Strategy:
        - "AVG-Conf"
        - "Consistency"
        - "Pair-Rank" (implemented in another script)
        
    Hyperparameters:
        - result_dict: dict of all intermediate results
    """

    aggregation_strategy = ['avg_conf', 'avg_multistep_conf', "consistency"]
    score_dicts = {"real_answer": [],
                   "scores": {}
                   }
    
    for key in aggregation_strategy:
        score_dicts['scores'][key] = {"answer": [], "score": []}

    # for every question in the dataset, get their answers and their corresponding confidences -> can be multiple if num_ensemble > 1
    for key, value in result_dict.items():
        """ Example below:
        - key: question text
        - value: dict of all intermediate results
            - value['hint'] (keys: 'hint_response', 'generated_result', 'real_answer')
                - value['hint']['real_answer'] = {'options': 'A', 'option_number': 6}
                - value['hint']['generated_resutl'] 
                    - (keys: 'step1', 'step2', 'step3', 'final_answer', 'final_confidence')
                - value['hint']['generated_resutl']['step1'] = {'analysis': 'xxxx Confidence: 90;', 'confidence': 90}

        """
        real_answer = value['real_answer']
        if sampling_type == "misleading":
            predicted_answer = value['predicted_answer']
            predicted_conf = value['predicted_conf']
        
        # get predicted answers and confidences over multiple queries -> for ensemble
        # hint_answers = {"trai_0":{"0":"A", "1":"B"}, "trail_1":{}}
        # hint_confs = {"trai_0":{"0":90, "1":80}, "trail_1":{}}
        hint_answers = value['hint_answers']
        hint_confs = value['hint_confs']
        hint_multi_step_confs = value['hint_multi_step_confs']
        assert len(hint_answers) == len(hint_confs), "(len(hint_answers) should be equivalent to len(hint_confidences))"

        # process into a map: answer -> [conf1, conf2, conf3, ...]
        answer_confs_alltrails = {}
        for trail, ans in hint_answers.items():
            # sanity check ans is formatted correctly
            if ans is None:
                continue
            elif task_type == "multi_choice_qa":
                if ans not in normal_option_list:
                    continue
            if ans not in answer_confs_alltrails:
                answer_confs_alltrails[ans] = []
            # fill the answer-conflist map
            conf = hint_confs[trail] # get the corresponding confidence list for this 'trail_i' or 'hint_i'
            answer_confs_alltrails[ans].append(conf)
        
        answer_stepconfs_for_alltrails = {}
        hint_step_confs = {}
        if prompt_type == "multi_step":
            for trail, step_confs in hint_multi_step_confs.items():
                confidence_product = 1
                for step_idx, step_result in step_confs.items():
                    step_confidence = step_result['confidence']
                    confidence_product *= step_confidence
                ans = hint_answers[trail]
                if ans not in answer_stepconfs_for_alltrails:
                    answer_stepconfs_for_alltrails[ans] = []
                answer_stepconfs_for_alltrails[ans].append(confidence_product)
                hint_step_confs[trail] = confidence_product
            
        ################### AVG-CONF ####################
        # compute consistency
        def compute_consistency_score(hint_answers, sampling_type):
            """every query has a answer, find the most frequent answer and its frequency -> consistency score"""
            top_1_ans = [answer for _, answer in hint_answers.items()]
            counter = Counter(top_1_ans)
            total = len(top_1_ans)
            # compute the frequency of each answer
            frequencies = {key: value / total for key, value in counter.items()}
            # find the most frequent answer and its frequency
            if sampling_type == "misleading":
                most_freq_ans = predicted_answer
                most_freq_score = frequencies[most_freq_ans]
                return most_freq_ans, most_freq_score
            
            most_freq_ans = max(frequencies, key=frequencies.get)
            most_freq_score = frequencies[most_freq_ans]
            return most_freq_ans, most_freq_score
        
        consistency_answer, consistency_score = compute_consistency_score(hint_answers, sampling_type)
        score_dicts['scores']['consistency']['answer'].append(consistency_answer)
        score_dicts['scores']['consistency']['score'].append(consistency_score)

        
        # compute average confidence for every possible answer
        def compute_avg_confidence(hint_confs, answer_confs_alltrails, sampling_type):
            conf_list = [conf for conf in hint_confs.values()]
            conf_sum = np.sum(conf_list)
            average_confs = {ans: sum(conf_lists)/conf_sum for ans, conf_lists in answer_confs_alltrails.items()}
            
            if sampling_type == "misleading":
                avg_conf_option = predicted_answer
                avg_confidence = average_confs[avg_conf_option]
                return avg_conf_option, avg_confidence
            
            avg_conf_option = max(average_confs, key=average_confs.get)
            avg_confidence = average_confs[avg_conf_option]
            return avg_conf_option, avg_confidence
        
        avg_conf_option, avg_confidence = compute_avg_confidence(hint_confs, answer_confs_alltrails, sampling_type)
        
        if prompt_type == "multi_step":
            avg_step_conf_option, avg_step_confidence = compute_avg_confidence(hint_step_confs, answer_stepconfs_for_alltrails, sampling_type)
        
        if task_type == "open_number_qa":
            real_answer = float(real_answer)
            consistency_answer = float(consistency_answer)
            avg_conf_option = float(avg_conf_option)
            if prompt_type == "multi_step":
                avg_step_conf_option = float(avg_step_conf_option)
            
        elif task_type == 'multi_choice_qa':
            if isinstance(real_answer, int):
                real_answer = option_list[real_answer]    

            
        score_dicts["real_answer"].append(real_answer)
        score_dicts['scores']['avg_conf']['answer'].append(avg_conf_option)
        score_dicts['scores']['avg_conf']['score'].append(avg_confidence)     
        if prompt_type == "multi_step":
            score_dicts['scores']['avg_multistep_conf']['answer'].append(avg_step_conf_option)
            score_dicts['scores']['avg_multistep_conf']['score'].append(avg_step_confidence)      

        
    print("Total count: ", len(score_dicts['real_answer']))  
    return score_dicts


        
score_dict = search_vis_answers(data, args.task_type, prompt_type=args.prompt_type, sampling_type=args.sampling_type)    



#%%
############### DEAL WITH ERRORS ####################
"""
Error type: 

"""
# print(" consistency_scores_by_distance:", consistency_scores_by_distance)


#################### VISUALIZATION FUNCTIONS ####################
def plot_ece_diagram(y_true, y_confs, score_type):
    from netcal.presentation import ReliabilityDiagram
    n_bins = 10
    diagram = ReliabilityDiagram(n_bins)

    plt.figure()
    diagram.plot(np.array(y_confs), np.array(y_true))
    plt.savefig(osp.join(visual_folder, input_file_name.replace(".json", f"_ece_{score_type}.pdf")), dpi=600)

def plot_confidence_histogram_with_detailed_numbers(y_true, y_confs, score_type, acc, auroc, ece, use_annotation=True):

    plt.figure(figsize=(6, 4))    
    corr_confs = [y_confs[i]*100 for i in range(len(y_confs)) if y_true[i] == 1]
    wrong_confs = [y_confs[i]*100 for i in range(len(y_confs)) if y_true[i] == 0]

    corr_counts = [corr_confs.count(i) for i in range(101)]
    wrong_counts = [wrong_confs.count(i) for i in range(101)]

    # correct_color = plt.cm.tab10(0)
    # wrong_color = plt.cm.tab10(1)
    correct_color = "red" # plt.cm.tab10(0)
    wrong_color = "blue" # plt.cm.tab10(1)
    # plt.bar(range(101), corr_counts, alpha=0.5, label='correct', color='blue')
    # plt.bar(range(101), wrong_counts, alpha=0.5, label='wrong', color='orange')
    n_correct, bins_correct, patches_correct = plt.hist(corr_confs, bins=21, alpha=0.5, label='correct answer', color=correct_color, align='mid', range=(-2.5,102.5))
    n_wrong, bins_wrong, patches_wrong = plt.hist(wrong_confs, bins=21, alpha=0.5, label='wrong answer', color=wrong_color, align='mid', range=(-2.5,102.5))

    tick_set = []
    annotation_correct_color = "black"
    annotation_wrong_color = "red"
    annotation_texts = []

    for i, count in enumerate(corr_counts):
        if count == 0:
            continue
        if use_annotation:
            annotation_texts.append(plt.annotate(str(count), xy=(i, count), ha='center', va='bottom', c=annotation_correct_color, fontsize=10, fontweight='bold'))
        tick_set.append(i)
            
    for i, count in enumerate(wrong_counts):
        if count == 0:
            continue
        if use_annotation:
            annotation_texts.append(plt.annotate(str(count), xy=(i, count), ha='center', va='bottom', c=annotation_wrong_color, fontsize=10, fontweight='bold'))
        tick_set.append(i)
    adjust_text(annotation_texts, only_move={'text': 'y'})

    if args.use_cot:
        # plt.title(f"{score_type_name} {args.dataset_name} {args.model_name} COT: ACC {acc:.2f} / AUROC {auroc:.2f}", fontsize=16)
        plt.title(f"COT: ACC {acc:.2f} / AUROC {auroc:.2f} / ECE {ece:.2f}", fontsize=16, fontweight='bold')
    else:
        # plt.title(f"{score_type_name} {args.dataset_name} {args.model_name}: ACC {acc:.2f} / AUROC {auroc:.2f}", fontsize=16)
        plt.title(f"ACC {acc:.2f} / AUROC {auroc:.2f} / ECE {ece:.2f}", fontsize=16, fontweight='bold')
    
    
    # plt.xlim(50, 100)
    plt.ylim(0, 1.1*max(max(n_correct), max(n_wrong)))
    plt.xticks(tick_set, fontsize=10, fontweight='bold')
    plt.yticks([])
    plt.xlabel("Confidence (%)", fontsize=16, fontweight='bold')
    plt.ylabel("Count", fontsize=16, fontweight='bold')
    plt.legend(loc='upper left', prop={'weight':'bold', 'size':14})
    plt.tight_layout()
    plt.savefig(osp.join(visual_folder, input_file_name.replace(".json", f"_auroc_{score_type}.png")), dpi=600)
    plt.savefig(osp.join(visual_folder, input_file_name.replace(".json", f"_auroc_{score_type}.pdf")), dpi=600)

def plot_confidence_histogram(y_true, y_confs, score_type, acc, auroc, ece, use_annotation=True):

    plt.figure(figsize=(6, 4))    
    corr_confs = [y_confs[i]*100 for i in range(len(y_confs)) if y_true[i] == 1]
    wrong_confs = [y_confs[i]*100 for i in range(len(y_confs)) if y_true[i] == 0]

    corr_counts = [corr_confs.count(i) for i in range(101)]
    wrong_counts = [wrong_confs.count(i) for i in range(101)]

    # correct_color = plt.cm.tab10(0)
    # wrong_color = plt.cm.tab10(1)
    correct_color =  plt.cm.tab10(0)
    wrong_color = plt.cm.tab10(3)
    # plt.bar(range(101), corr_counts, alpha=0.5, label='correct', color='blue')
    # plt.bar(range(101), wrong_counts, alpha=0.5, label='wrong', color='orange')
    n_wrong, bins_wrong, patches_wrong = plt.hist(wrong_confs, bins=21, alpha=0.8, label='wrong answer', color=wrong_color, align='mid', range=(-2.5,102.5))
    n_correct, bins_correct, patches_correct = plt.hist(corr_confs, bins=21, alpha=0.8, label='correct answer', color=correct_color, align='mid', range=(-2.5,102.5), bottom=np.histogram(wrong_confs, bins=21, range=(-2.5,102.5))[0])

    tick_set = [i*10 for i in range(5,11)]
    annotation_correct_color = "black"
    annotation_wrong_color = "red"
    annotation_texts = []

    # for i, count in enumerate(corr_counts):
    #     if count == 0:
    #         continue
    #     if use_annotation:
    #         annotation_texts.append(plt.annotate(str(count), xy=(i, count), ha='center', va='bottom', c=annotation_correct_color, fontsize=10, fontweight='bold'))
    #     tick_set.append(i)
            
    # for i, count in enumerate(wrong_counts):
    #     if count == 0:
    #         continue
    #     if use_annotation:
    #         annotation_texts.append(plt.annotate(str(count), xy=(i, count), ha='center', va='bottom', c=annotation_wrong_color, fontsize=10, fontweight='bold'))
    #     tick_set.append(i)
    # adjust_text(annotation_texts, only_move={'text': 'y'})

    if args.use_cot:
        # plt.title(f"{score_type_name} {args.dataset_name} {args.model_name} COT: ACC {acc:.2f} / AUROC {auroc:.2f}", fontsize=16)
        plt.title(f"COT: ACC {acc:.2f} / AUROC {auroc:.2f} / ECE {ece:.2f}", fontsize=16, fontweight='bold')
    else:
        # plt.title(f"{score_type_name} {args.dataset_name} {args.model_name}: ACC {acc:.2f} / AUROC {auroc:.2f}", fontsize=16)
        plt.title(f"ACC {acc:.2f} / AUROC {auroc:.2f} / ECE {ece:.2f}", fontsize=16, fontweight='bold')
    
    
    # plt.xlim(47.5, 102.5)
    plt.ylim(0, 1.1*max(n_correct+n_wrong))
    plt.xticks(tick_set, fontsize=16, fontweight='bold')
    plt.yticks([])
    plt.xlabel("Confidence (%)", fontsize=16, fontweight='bold')
    plt.ylabel("Count", fontsize=16, fontweight='bold')
    plt.legend(loc='upper left', prop={'weight':'bold', 'size':16})
    plt.tight_layout()
    plt.savefig(osp.join(visual_folder, input_file_name.replace(".json", f"_auroc_{score_type}.png")), dpi=600)
    plt.savefig(osp.join(visual_folder, input_file_name.replace(".json", f"_auroc_{score_type}.pdf")), dpi=600)


#%%
#################### COMPUTE ACC/ECE/AUCROC ####################
result_matrics = {}
from utils.compute_metrics import compute_conf_metrics
real_answers = score_dict["real_answer"]
for metric_name, values in score_dict['scores'].items():
    predicted_answers = values['answer']
    predicted_confs = values['score']
    correct = [real_answers[i]==predicted_answers[i] for i in range(len(real_answers))]
    result_matrics[metric_name] = compute_conf_metrics(correct, predicted_confs)
    
    # auroc visualization
    plot_confidence_histogram(correct, predicted_confs, metric_name, result_matrics[metric_name]['acc'], result_matrics[metric_name]['auroc'], result_matrics[metric_name]['ece'], use_annotation=True)
    plot_ece_diagram(correct, predicted_confs, score_type=metric_name)

# sanity check: when num_ensemle = 1, the consistency score should be the same as the average confidence score
if args.num_ensemble == 1:
    assert result_matrics['consistency']['acc'] == result_matrics['avg_conf']['acc'], (result_matrics['consistency']['acc'], result_matrics['avg_conf']['acc'])
    assert result_matrics['consistency']['auroc'] == result_matrics['avg_conf']['auroc'], (result_matrics['consistency']['auroc'], result_matrics['avg_conf']['auroc'])
    assert result_matrics['consistency']['ece'] == result_matrics['avg_conf']['ece'], (result_matrics['consistency']['ece'], result_matrics['avg_conf']['ece'])

with open(output_file, "a") as f:
    f.write("dataset_name,model_name,prompt_type,sampling_type,aggregation_type,acc,ece,auroc,auprc_p,auprc_n,use_cot,input_file_name,num_ensemble\n")
    for key, values in result_matrics.items():
        f.write(f"{args.dataset_name},{args.model_name},{args.prompt_type},{args.sampling_type},{key},{values['acc']:.5f},{values['ece']:.5f},{values['auroc']:.5f},{values['aurc']:.5f},{values['auprc_p']:.5f},{values['auprc_n']:.5f},{args.use_cot},{input_file_name},{args.num_ensemble}\n")



