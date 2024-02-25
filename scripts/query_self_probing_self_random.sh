#!/bin/bash

# prompt strategy = "self_probing" (can be changed to ["cot"])
# sampling strategy = "self-random" or "misleading" ->  by setting NUM_ENSEMBLE to decide the number of samples we want to draw from a given question
# aggregator = consistency / avg-conf (all of them will be automatically computed in the same script)
PROMPT_TYPE="self_probing"
SAMPLING_TYPE="self_random" 
NUM_ENSEMBLE=5 
CONFIDENCE_TYPE="${PROMPT_TYPE}_${SAMPLING_TYPE}_${NUM_ENSEMBLE}"



# TODO uncomment following lines to run on different settings
#############################################################

DATASET_NAME="GSM8K"
MODEL_NAME="chatgpt"
TASK_TYPE="open_number_qa"
DATASET_PATH="final_output/vanilla_verbalized_confidence/chatgpt/GSM8K/GSM8K_chatgpt_05-06-00-13_processed.json"
USE_COT=true # use cot or not
TEMPERATURE=0.0

# DATASET_NAME="BigBench_ObjectCounting"
# MODEL_NAME="chatgpt"
# TASK_TYPE="open_number_qa"
# DATASET_PATH="dataset/BigBench/object_counting.json"
# USE_COT=true # use cot or not

# CONFIDENCE_TYPE="vanilla_verbalized_confidence"
# DATASET_NAME="Business_Ethics"
# MODEL_NAME="gpt4"
# TASK_TYPE="multi_choice_qa"
# DATASET_PATH="dataset/MMLU/business_ethics_test.csv"
# USE_COT=false # use cot or not

# for professional law
# DATASET_NAME="Professional_Law"
# MODEL_NAME="chatgpt"
# TASK_TYPE="multi_choice_qa"
# DATASET_PATH="final_output/vanilla_verbalized_confidence/chatgpt/Professional_Law/Professional_Law_chatgpt_vanilla_05-29-05-23_processed.json"
# USE_COT=false # use cot or not

# DATASET_NAME="Business_Ethics"
# TASK_TYPE="multi_choice_qa" # [multi_choice_qa, open_number_qa]
# # the dataset is the output of the cot verbalized method; choose the processed outout from the cot output
# DATASET_PATH="final_output/vanilla_verbalized_confidence/gpt3/Business_Ethics/Business_Ethics_gpt3_vanilla_05-29-05-49_processed.json"
# MODEL_NAME="chatgpt" # [gpt4, chatgpt, gpt3, vicuna]
# USE_COT=true # use cot or not


# DATASET_NAME="StrategyQA"
# MODEL_NAME="chatgpt"
# TASK_TYPE="multi_choice_qa"
# DATASET_PATH="final_output/vanilla_verbalized_confidence/chatgpt/strategyQA/strategyQA_chatgpt_woCOT_vanilla_processed.json"
# USE_COT=true # use cot or not


# DATASET_NAME="BigBench_DateUnderstanding"
# MODEL_NAME="chatgpt"
# TASK_TYPE="multi_choice_qa"
# DATASET_PATH="final_output/vanilla_verbalized_confidence/chatgpt/BigBench_DateUnderstanding/BigBench_DateUnderstanding_chatgpt_05-06-00-01_processed.json"
# USE_COT=true # use cot or not

#############################################################
# set time stamp to differentiate the output file
TIME_STAMPE=$(date "+%m-%d-%H-%M")

OUTPUT_DIR="final_output/$CONFIDENCE_TYPE/$MODEL_NAME/$DATASET_NAME"
RESULT_FILE="$OUTPUT_DIR/${DATASET_NAME}_${MODEL_NAME}_${TIME_STAMPE}.json"
USE_COT_FLAG=""

if [ "$USE_COT" = true ] ; then
    USE_COT_FLAG="--use_cot"
fi

python query_self_probing.py \
   --dataset_name  $DATASET_NAME \
   --data_path $DATASET_PATH \
   --output_file  $RESULT_FILE \
   --model_name  $MODEL_NAME \
   --task_type  $TASK_TYPE  \
   --prompt_type $PROMPT_TYPE \
   --sampling_type $SAMPLING_TYPE \
   --num_ensemble $NUM_ENSEMBLE \
   --temperature_for_ensemble $TEMPERATURE \
   $USE_COT_FLAG


# uncomment following lines to run test and visualization
python extract_answers.py \
   --input_file $RESULT_FILE \
   --model_name  $MODEL_NAME \
   --dataset_name  $DATASET_NAME \
   --task_type  $TASK_TYPE   \
   --prompt_type $PROMPT_TYPE \
   --sampling_type $SAMPLING_TYPE \
   --num_ensemble $NUM_ENSEMBLE \
    $USE_COT_FLAG

RESULT_FILE_PROCESSED=$(echo $RESULT_FILE | sed 's/\.json$/_processed.json/')

python vis_aggregated_conf.py \
    --input_file $RESULT_FILE_PROCESSED \
    --model_name  $MODEL_NAME \
    --dataset_name  $DATASET_NAME \
    --task_type  $TASK_TYPE   \
    --prompt_type $PROMPT_TYPE  \
    --sampling_type $SAMPLING_TYPE \
    --num_ensemble $NUM_ENSEMBLE \
    $USE_COT_FLAG    
