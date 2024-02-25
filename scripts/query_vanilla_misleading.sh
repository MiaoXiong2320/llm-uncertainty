#!/bin/bash

# prompt strategy = "vanilla" (can be changed to ["cot"] -> then use_cot flag should be set to true in the following script)
# sampling strategy = "self-random" or "misleading" ->  by setting NUM_ENSEMBLE to decide the number of samples we want to draw from a given question
# aggregator = consistency / avg-conf (all of them will be automatically computed in the same script)
PROMPT_TYPE="vanilla"
SAMPLING_TYPE="misleading" 
NUM_ENSEMBLE=5 
CONFIDENCE_TYPE="${PROMPT_TYPE}_${SAMPLING_TYPE}_${NUM_ENSEMBLE}"



# TODO uncomment following lines to run on different settings
#############################################################

DATASET_NAME="Professional_Law"
MODEL_NAME="gpt4"
TASK_TYPE="multi_choice_qa"
DATASET_PATH="dataset/MMLU/professional_law_test.csv"
USE_COT=false # use cot or not
TEMPERATURE=0.0


# DATASET_NAME="BigBench_DateUnderstanding"
# MODEL_NAME="chatgpt"
# TASK_TYPE="multi_choice_qa"
# DATASET_PATH="dataset/BigBench/date_understanding.json"

# CONFIDENCE_TYPE="top_k_verbalize_confidence"
# DATASET_NAME="GSM8K"
# MODEL_NAME="chatgpt"
# TASK_TYPE="open_number_qa"
# DATASET_PATH="dataset/grade_school_math/data/test.jsonl"
# USE_COT=true # use cot or not

# DATASET_NAME="BigBench_ObjectCounting"
# MODEL_NAME="chatgpt"
# TASK_TYPE="open_number_qa"
# DATASET_PATH="dataset/BigBench/object_counting.json"
# NUM_ENSEMBLE=5 # number of ensembles
# USE_COT=true # use cot or not

# CONFIDENCE_TYPE="top_k_verbalize_confidence"
# DATASET_NAME="StrategyQA"
# MODEL_NAME="chatgpt"
# TASK_TYPE="multi_choice_qa"
# DATASET_PATH="dataset/BigBench/object_counting.json"
# NUM_ENSEMBLE=1 # number of ensembles
# USE_COT=true # use cot or not


# CONFIDENCE_TYPE="vanilla_verbalized_confidence"
# DATASET_NAME="Business_Ethics"
# MODEL_NAME="chatgpt"
# TASK_TYPE="multi_choice_qa"
# DATASET_PATH="dataset/MMLU/business_ethics_test.csv"
# NUM_ENSEMBLE=1 # number of ensembles
# USE_COT=false # use cot or not

# for professional law
# # CONFIDENCE_TYPE="vanilla_verbalized_confidence"
# CONFIDENCE_TYPE="top_k_verbalize_confidence"
# DATASET_NAME="Professional_Law"
# MODEL_NAME="chatgpt"
# TASK_TYPE="multi_choice_qa"
# DATASET_PATH="dataset/MMLU/professional_law_test.csv"
# NUM_ENSEMBLE=1 # number of ensembles
# USE_COT=true # use cot or not

# # for **validation set** of the business ethics 
# CONFIDENCE_TYPE="vanilla_verbalized_confidence"
# DATASET_NAME="Business_Ethics_Val"
# MODEL_NAME="chatgpt" # [gpt4, chatgpt, gpt3, vicuna]
# TASK_TYPE="multi_choice_qa" # [multi_choice_qa, open_number_qa]
# DATASET_PATH="dataset/MMLU/business_ethics_val.csv"
# NUM_ENSEMBLE=1 # number of ensembles
# USE_COT=false # use cot or not


#############################################################
# set time stamp to differentiate the output file
TIME_STAMPE=$(date "+%m-%d-%H-%M")

OUTPUT_DIR="final_output/$CONFIDENCE_TYPE/$MODEL_NAME/$DATASET_NAME"
RESULT_FILE="$OUTPUT_DIR/${DATASET_NAME}_${MODEL_NAME}_${TIME_STAMPE}.json"
USE_COT_FLAG=""

if [ "$USE_COT" = true ] ; then
    USE_COT_FLAG="--use_cot"
fi


python query_top_k.py \
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
