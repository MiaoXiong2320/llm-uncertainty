# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/metrics.py
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/utilities.py

import re, pdb
import os
import json
import random
import transformers
from tqdm import tqdm
import argparse
import pandas as pd

import ssl
import urllib.request
import zipfile



# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

def LlamaChatCompletion(model_name, prompt, max_tokens):
    os.environ['CUDA_VISIBLE_DEVICES'] = "5"
    # model_name = "daryl149/llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda") 
    outputs = model.generate(input_ids=input_ids,
                             max_new_tokens=max_tokens,return_dict_in_generate=True, output_scores=True, output_hidden_states=True)
    
    tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    pdb.set_trace()
    
    return outputs
    
    # logit=1
    # logit_layer=torch.ones(1,33)
    # for _ in range(len(outputs[1])-2):
    #     # logit=torch.exp(outputs[1][1][0][output_id+_])*logit
    #     m = torch.nn.Softmax()
    #     cur_logit=m(outputs[1][1+_][0])
    #     logit=max(cur_logit)*logit

    #     gen_tok=torch.argmax(outputs[1][1+_][0])


    #     logit_by_layer=[outputs.hidden_states[1+_][i][0][-1] for i in range(33)]
    #     lm_head = model.state_dict()['lm_head.weight']
    #     logit_layer_vocab=[m(torch.mm(lm_head, logit_by_layer[i].unsqueeze(-1)).squeeze()) for i in range(33)]
    #     # logit_layer=[(np.array(logit_layer_vocab[i][gen_tok].cpu())) for i in range(33)]
    #     logit_layer=torch.mul(logit_layer,torch.Tensor([logit_layer_vocab[i][gen_tok] for i in range(33)]))


            


