"""
Aggregation Strategy:
    - "AVG-Conf"
    - "Consistency"
    - "Pair-Rank" 

Supported Prompt Strategy:
    - "Top-K"
    
This code is used to visualize the performance (e.g. ACC/AUCROC/ECE) of  **Confidence Scores based on misleading consistency**:
- ACC
- AUCROC
- AUPRC
- ECE

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

def compute_rank_based_top_k_score(hint_answers):
    """
    - implementation of Pair-Rank aggregation algorithm
    - rank_matrix = [[0, 1/K, ...], [],...]
        - rank_matrix[i,j]=1/K : The probability that option i appears before option j
    - see the paper for details
    """
    import torch
    import torch.optim as optim

    def convert_to_rank_matrix(answer_set):
        num_trail = len(answer_set)
        top_k = len(next(iter(answer_set.values())))

        # compute the number of unique answers
        unique_elements = set()
        for inner_dict in answer_set.values():
            unique_elements.update(inner_dict.values())
        num_options = len(unique_elements)
        # map every item to its unique id
        # element_to_id["A"]=0
        element_to_id = {element: idx for idx, element in enumerate(unique_elements)}
        id_to_element = {idx: element for element, idx in element_to_id.items()}

        rank_matrix = torch.zeros(num_options, num_options)
        for trail, answers in answer_set.items():
            # answers[trail_0] = {0:"A", ..."3":"D"}
            mask = torch.ones(num_options)
            for idx in range(top_k):
                # answer["0"] = "A" -> option
                option = answers[str(idx)]
                id_cat = element_to_id[option]
                mask[id_cat] = 0
                rank_matrix[id_cat, :] += mask

        rank_matrix = rank_matrix / num_trail
        # assert rank_matrix.any() >= 0.0 and rank_matrix.any() <=1.0, "rank matrix should be [0,1]"
        return rank_matrix, num_options, top_k, id_to_element

    rank_matrix, num_options, top_k, id_to_element = convert_to_rank_matrix(hint_answers)

    w_cat = torch.randn(num_options, requires_grad=True)


    # Define the SGD optimizer
    optimizer = optim.SGD([w_cat], lr=0.01)

    # Define the loss function: Frobenius norm of W
    def nll_loss_func(w_cat, rank_matrix):
        p_cat = torch.nn.functional.softmax(w_cat, dim=0)
        loss = 0
        # Compute the denominator for all combinations of p_cat[row] and p_cat[col]
        # denominator[i,j] = p_cat[i] + p_cat[j]
        denominator = p_cat.view(-1, 1) + p_cat.view(1, -1)
        # Avoid division by zero by adding a small constant
        epsilon = 1e-10
        # Compute the ratio
        ratios = (p_cat.view(-1, 1) + epsilon) / (denominator + 2*epsilon)
        loss = -torch.sum(rank_matrix * ratios)
        return loss



    # Training loop to minimize the loss function

    for _ in range(1000):
        # Compute the loss
        loss = nll_loss_func(w_cat, rank_matrix)

        # Zero gradients, backward pass, optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_p_cat = torch.nn.functional.softmax(w_cat, dim=0)
    id = torch.argmax(final_p_cat)
    answer = id_to_element[int(id)]
    final_p_cat = final_p_cat.detach().numpy()
    score = final_p_cat[id]

    return answer, score

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

def search_vis_answers_top_k(result_dict, task_type):
    """
    Function purpose:
        - This function is implemented as the aggregation strategy, i.e., used to get the final answer and confidence based on the top-k results for every question, and aggregate all ensembles to get their corresponding scores: 
        - Prompt Strategy = "Top-K"
        
    Aggregation Strategy:
        - "AVG-Conf"
        - "Consistency"
        - "Pair-Rank"
        
    Hyperparameters:
        - result_dict: dict of all intermediate results
    """

    aggregation_strategy = ['avg_conf_top1', 'avg_conf_topk', 'pair_rank_conf', "consistency"]
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
        if task_type == "open_number_qa":
            real_answer = float(real_answer)

        elif task_type == 'multi_choice_qa':
            if isinstance(real_answer, int):
                real_answer = option_list[real_answer]  
        score_dicts["real_answer"].append(real_answer) 
        
        # get predicted answers and confidences over multiple queries -> for ensemble
        # hint_answers = {"trai_0":{"0":"A", "1":"B"}, "trail_1":{}}
        # hint_confs = {"trai_0":{"0":90, "1":80}, "trail_1":{}}
        hint_answers = value['hint_answers']
        hint_confs = value['hint_confs']
        hint_multi_step_confs = value['hint_multi_step_confs']
        assert len(hint_answers) == len(hint_confs), "(len(hint_answers) should be equivalent to len(hint_confidences))"

        # process into a map: answer -> [conf1, conf2, conf3, ...]
        top1_answer_confs_alltrails = {} # only consider the first answer from top-k answers
        topk_answer_confs_alltrails = {}
        topk_answer_ranks = {}
        # consider every trail during ensemble
        for trail, answers in hint_answers.items():
            top_k_conf = hint_confs[trail] # get the corresponding confidence list for this 'trail_i' or f'hint{i}'
            
            # only consider the first answer/conf from top-k prompt
            top1_ans = answers['0']
            top1_conf = top_k_conf['0']
            if top1_ans not in top1_answer_confs_alltrails:
                top1_answer_confs_alltrails[top1_ans] = []
            top1_answer_confs_alltrails[top1_ans].append(top1_conf)
            
            # consider every possible answer in a reply and get their corresponding confidences
            for rank, ans in answers.items():
                if ans is None:
                    continue
                elif task_type == "multi_choice_qa":
                    if ans not in normal_option_list:
                        continue
                if ans not in topk_answer_confs_alltrails:
                    topk_answer_confs_alltrails[ans] = []
                    topk_answer_ranks[ans] = []
                topk_answer_confs_alltrails[ans].append(top_k_conf[rank])
                topk_answer_ranks[ans].append(int(rank))
                
        # compute consistency (only consider the first answer from Top-k answers)
        top_1_ans = [answer_list['0'] for _, answer_list in hint_answers.items()]
        counter = Counter(top_1_ans)
        total = len(top_1_ans)
        # compute the frequency of each answer
        frequencies = {key: value / total for key, value in counter.items()}
        # find the most frequent answer and its frequency
        max_consistency_option = max(frequencies, key=frequencies.get)
        max_consistency_score = frequencies[max_consistency_option]
        score_dicts['scores']['consistency']['answer'].append(max_consistency_option)
        score_dicts['scores']['consistency']['score'].append(max_consistency_score)


        # compute average confidence for every possible answer
        average_confs = {ans: sum(conf_lists)/len(conf_lists) / 100 for ans, conf_lists in topk_answer_confs_alltrails.items()}
        max_conf_option_topk = max(average_confs, key=average_confs.get)
        max_confidence_topk = average_confs[max_conf_option_topk]
        score_dicts['scores']['avg_conf_topk']['answer'].append(max_conf_option_topk)
        score_dicts['scores']['avg_conf_topk']['score'].append(max_confidence_topk)
        
        # compute average confidence for each option that appeared as top-1 answer
        conf_list = [conf['0'] for conf in hint_confs.values()]
        conf_sum = np.sum(conf_list)
        average_confs = {ans: np.sum(conf_lists)/conf_sum for ans, conf_lists in top1_answer_confs_alltrails.items()}   
        max_conf_option = max(average_confs, key=average_confs.get)
        max_confidence = average_confs[max_conf_option]
        score_dicts['scores']['avg_conf_top1']['answer'].append(max_conf_option)
        score_dicts['scores']['avg_conf_top1']['score'].append(max_confidence)
          
        # compute average rank
        # average_ranks = {ans: sum(ranks)/len(ranks) for ans, ranks in topk_answer_ranks.items()}
        # min_rank_option = min(average_ranks, key=average_ranks.get)
        # min_rank = average_confs[min_rank_option]
        # max_rank_score = 1 - min_rank / len(hint_answers) 

        # compute the rank-based top-k score using multiple LLM queries
        # rank_based_topk_score = compute_rank_based_score(topk_answer_ranks)
        rank_based_topk_answer, rank_based_topk_score = compute_rank_based_top_k_score(hint_answers)
        score_dicts['scores']['pair_rank_conf']['answer'].append(rank_based_topk_answer)
        score_dicts['scores']['pair_rank_conf']['score'].append(rank_based_topk_score)


        
    print("Total count: ", len(score_dicts['real_answer']))  
    return score_dicts


pdb.set_trace()
score_dict = search_vis_answers_top_k(data, args.task_type, num_ensemble=args.num_ensemble)    

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


with open(output_file, "a") as f:
    f.write("dataset_name,model_name,prompt_type,sampling_type,aggregation_type,acc,ece,auroc,auprc_p,auprc_n,use_cot,input_file_name,num_ensemble\n")
    for key, values in result_matrics.items():
        f.write(f"{args.dataset_name},{args.model_name},{args.prompt_type},{args.sampling_type},{key},{values['acc']:.5f},{values['ece']:.5f},{values['auroc']:.5f},{values['aurc']:.5f},{values['auprc_p']:.5f},{values['auprc_n']:.5f},{args.use_cot},{input_file_name},{args.num_ensemble}\n")



