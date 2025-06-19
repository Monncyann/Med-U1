#!/bin/bash

# Main model settings
export CUDA_VISIBLE_DEVICES=0
BASE_MODEL_NAME="Med-U1-7B-medcalc-L" #set your model name
BASE_PATH="/Med-U1/model/${BASE_MODEL_NAME}" # set your path  
MODELS=$BASE_MODEL_NAME
MODEL_PATHS=$BASE_PATH
DATA_NAME=medcalc_test
num_tokens=256   # set the output token number
metric="EMS"        #Rouge-L or EMS
# comet_model_path=~/wmt22-comet-da/checkpoints/model.ckpt #set your metric ckpt
# comet_free_model_path=~/wmt23-cometkiwi-da-xl/checkpoints/model.ckpt #set your metric ckpt

TEMPLATE_TYPE="base"
TENSOR_PARALLEL_SIZE=1
TEMPERATURE=0.2
TOP_P=0.95
MAX_TOKENS=2048
BATCH_SIZE=32
BASE_SAVE_DIR="/Med-U1/vllm_infer_results" #set your save dir

INPUT_FILES="/data/xiaotian/l1/scripts/data/processed_data_${num_tokens}/${DATA_NAME}.jsonl"




# Execute inference and evaluation for each model
    MODEL_NAME="${MODELS}"
    MODEL_PATH="${MODEL_PATHS[$i]}"
    SAVE_DIR="${BASE_SAVE_DIR}/${MODEL_NAME}_${num_tokens}"
    OUTPUT_FILE_PREFIX="${MODEL_NAME}"
    
    echo "Processing model: ${MODEL_NAME}"
    echo "Model path: ${MODEL_PATH}"
    
    # Create necessary directories
    mkdir -p $SAVE_DIR
    
    # Step 1: Run VLLM inference
    echo "Starting VLLM inference..."
    
    
        
    # Find input file
        
    if [ ${#INPUT_FILES[@]} -eq 0 ]; then
        echo "Warning: No files matching ${INPUT_PATTERN} found"
        continue
    fi
        
    INPUT_PATH="${INPUT_FILES}"
    echo "Using input file: ${INPUT_PATH}"
        
    OUTPUT_DIR=${SAVE_DIR}
    mkdir -p $OUTPUT_DIR
        
    python /Med-U1/verl/verl/trainer/vllm_inference.py \
        --model $MODEL_PATH \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --gpu-memory-utilization 0.85 \
        --max-model-len 16384 \
        --temperature $TEMPERATURE \
        --top-p $TOP_P \
        --max-tokens $MAX_TOKENS \
        --input $INPUT_PATH \
        --output-dir $OUTPUT_DIR \
        --batch-size $BATCH_SIZE \
        --template-type $TEMPLATE_TYPE

    echo "Inference completed!"


    # Step 2: Evaluate  quality
    echo "Starting  evaluation..."
    
        
    # Path to JSON file output by VLLM
    json_files="$SAVE_DIR/result_${DATA_NAME}.json"
        
    if [ ${#json_files[@]} -eq 0 ]; then
        echo "Warning: No JSON files found in ${OUTPUT_DIR}"
        continue
    fi
        
    json_file="${json_files}"
    echo "Using output JSON: ${json_file}"
    # Extract text from JSON file - handle line breaks to ensure alignment
    python /Med-U1/verl/verl/trainer/extract_to_eval.py "${json_file}" "${metric}"
        
    

