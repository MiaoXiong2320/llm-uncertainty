import json, pdb
from typing import Dict
import os.path as osp
import pandas as pd

def load_dataset_w_prediction(dataset_name: str, task_type: str, data_path: str):
    """
    dataset_name: name of the dataset
    task_type: type of the task, e.g. multi_choice_qa, open_number_qa
    data_path: path to the dataset
    """
    with open(data_path,'r') as f:
        data = json.load(f)     
    
    print("Information of used dataset: ", data['hyperparameters']) 
    
    return data  



def load_dataset(dataset_name: str, task_type: str, data_path: str):
    """
    dataset_name: name of the dataset
    task_type: type of the task, e.g. multi_choice_qa, open_number_qa
    data_path: path to the dataset
    """
    
    character = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)", "(K)", "(L)", "(M)", "(N)", "(O)", "(P)", "(Q)", "(R)", "(S)", "(T)", "(U)", "(V)", "(W)", "(X)", "(Y)", "(Z)"]
    
    number_options = 6
    
    # qa_data is the dictionary that stores the questions and answers
    # questions are the keys and answers are the values
    qa_data = {}
    
    if dataset_name == "GSM8K":
        import dataset.grade_school_math.dataset as gsm_loader
        examples = gsm_loader.get_examples(data_path)
        for qa in examples:
            question = qa['question']
            answer = {'answer':gsm_loader.extract_answer(qa["answer"]), 'options':number_options}
            qa_data[question] = answer

    elif "BigBench" in dataset_name:
        with open(data_path,'r') as f:
            data = json.load(f)
        if task_type == "open_number_qa":
            for qa in data['examples']:
                """ qa has two keys:  
                {   "input": "I have a clarinet, a violin, and a flute. How many musical instruments do I have?",
                    "target": ["three", "3"] }
                """
                question = qa['input']
                value = {'answer':qa['target'][1], 'options':number_options}
                qa_data[question] = value
                
        elif task_type == "multi_choice_qa":
        
            for qa in data['examples']:
                question = qa['input'] +'\n' +'Options: '
                j=0
                for key, value in qa['target_scores'].items():
                    option = character[j] + " " + key + "\t"
                    question += option
                    if value == 1:
                        answer_index = j
                    j += 1
                    
                value = {'answer':answer_index, 'options':list(qa['target_scores'].keys())}
                qa_data[question] = value
                
    elif dataset_name == "ScienceQA":
        with open(osp.join(data_path, "pid_splits.json"), 'r') as f:
            pids = json.load(f)
        
        with open(osp.join(data_path, "problems.json"), 'r') as f:
            data = json.load(f)
        
        for pid in pids['test']:
            question = data[pid]['question'] + '\n' +'Options: '
            for idx, choice in enumerate(data[pid]['choices']):
                question += character[idx] + " " + choice + "\t"
                
            value = {'answer':data[pid]['answer'], 'options':data[pid]['choices']}
            qa_data[question] = value
    
    elif "business_ethics" in dataset_name.lower():
        test_df = pd.read_csv(data_path, header=None)
        for _, row in test_df.iterrows():
            raw_question = row[0]
            correct_answer = ord(row[5]) - ord("A")
            options = [row[i] for i in range(1, 5)]
            question = raw_question + '\n' +'Options: '
            for i in range(4):
                question += character[i] + " " + options[i] + "  "
            value = {
                'answer': correct_answer,
                'options': options
            }
            qa_data[question] = value
            
    elif "professional_law" in dataset_name.lower():
        test_df = pd.read_csv(data_path, header=None)
        for _, row in test_df.iterrows():
            raw_question = row[0]
            correct_answer = ord(row[5]) - ord("A")
            options = [row[i] for i in range(1, 5)]
            question = raw_question + '\n' +'Options: '
            for i in range(4):
                question += character[i] + " " + options[i] + "  "
            value = {
                'answer': correct_answer,
                'options': options
            }
            qa_data[question] = value
            
    elif "hybrid" in dataset_name.lower():
        with open(data_path, 'r') as f:
            qa_data = json.load(f)
            
    else:
        raise ValueError(f"{dataset_name} not supported")
    
    return qa_data