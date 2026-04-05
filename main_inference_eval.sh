#!/bin/bash

# Main model settings
export CUDA_VISIBLE_DEVICES=0,1,3,4
# --- Proxy ---
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export SSL_CERT_FILE=../cacert.pem
BASE_MODEL_NAME="Qwen/Qwen3-4B" #set your model name
BASE_SAVE_DIR="./results"
BASE_PATH="${BASE_MODEL_NAME}" # set your path
MODELS=$BASE_MODEL_NAME
MODEL_PATHS=$BASE_PATH
comet_model_path=/hnvme/workspace/slcl100h-vllm/custom-verl/comet_models/wmt22-comet-da/checkpoints/model.ckpt
comet_free_model_path=/hnvme/workspace/slcl100h-vllm/custom-verl/comet_models/wmt22-cometkiwi-da/checkpoints/model.ckpt

TEMPLATE_TYPE="base"
TENSOR_PARALLEL_SIZE=2
TEMPERATURE=0.2
TOP_P=0.95
MAX_TOKENS=1024
BATCH_SIZE=16
INPUT_DIR=data/test/json

# Language pair settings
all_language_pairs="en-zh zh-en" # You can add more, such as: en-zh zh-en de-zh en-ja de-en


# Execute inference and evaluation for each model
for i in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$i]}"
    MODEL_PATH="${MODEL_PATHS[$i]}"
    SAVE_DIR="${BASE_SAVE_DIR}/$(basename ${MODEL_PATH})_${all_language_pairs// /_}"
    OUTPUT_FILE_PREFIX="$(basename ${MODEL_NAME})"
    
    echo "Processing model: ${MODEL_NAME}"
    echo "Model path: ${MODEL_PATH}"
    
    # Create necessary directories
    mkdir -p $SAVE_DIR
    
    # Step 1: Run VLLM inference
    echo "Starting VLLM inference..."
    
    for test_pair in $all_language_pairs; do
        # Parse language pair
        src=$(echo "${test_pair}" | cut -d "-" -f 1)
        tgt=$(echo "${test_pair}" | cut -d "-" -f 2)
        
        # Find input file
        INPUT_PATTERN="${INPUT_DIR}/*${src}${tgt}.jsonl"
        INPUT_FILES=( $INPUT_PATTERN )
        
        if [ ${#INPUT_FILES[@]} -eq 0 ]; then
            echo "Warning: No files matching ${INPUT_PATTERN} found"
            continue
        fi
        
        INPUT_PATH="${INPUT_FILES[0]}"
        echo "Using input file: ${INPUT_PATH}"
        
        OUTPUT_DIR="${SAVE_DIR}/${test_pair}"
        mkdir -p $OUTPUT_DIR
        
        python eval/vllm_inference.py \
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

        echo "Inference completed! ${test_pair} results saved in $OUTPUT_DIR directory"
    done


    # Step 2: Evaluate translation quality
    echo "Starting translation quality evaluation..."
    
    for test_pair in $all_language_pairs; do
        src=$(echo "${test_pair}" | cut -d "-" -f 1)
        tgt=$(echo "${test_pair}" | cut -d "-" -f 2)
        
        # Path to JSON file output by VLLM
        OUTPUT_DIR="${SAVE_DIR}/${test_pair}"
        json_files=( $OUTPUT_DIR/*.json )
        
        if [ ${#json_files[@]} -eq 0 ]; then
            echo "Warning: No JSON files found in ${OUTPUT_DIR}"
            continue
        fi
        
        json_file="${json_files[0]}"
        echo "Using output JSON: ${json_file}"
        
        # Set up file paths for source text, translated text and reference translations
        src_dir="${SAVE_DIR}/${test_pair}/texts"
        mkdir -p "${src_dir}"
        src_path="${src_dir}/all_source.txt"
        tgt_path="${src_dir}/all_target.txt"
        output_path="${src_dir}/translations.txt"
        
        # Extract text from JSON file - handle line breaks to ensure alignment
        python eval/extract_to_eval.py "${json_file}" "${src_path}" "${output_path}" "${tgt_path}"
        
        # Set evaluation parameters
        if [ "${tgt}" = "zh" ]; then
            TOK="zh"
        elif [ "${tgt}" = "ja" ]; then
            TOK="ja-mecab"
        else
            TOK="13a"
        fi
        
        echo "--------------------Results for ${test_pair} (${MODEL_NAME})-------------------------------"
        
        # Calculate BLEU score
        SACREBLEU_FORMAT=text python -m sacrebleu -tok "${TOK}" -w 2 "${tgt_path}" < "${output_path}" > "${output_path}.bleu"
        cat "${output_path}.bleu"
        
        # Calculate COMET score
        echo "Calculating COMET scores..."
        python3 eval/comet_sc.py ${comet_model_path} "${src_path}" "${output_path}" "${tgt_path}" > "${output_path}.comet" || echo "COMET scoring failed"
        echo "Calculating COMET Kiwi scores..."
        python3 eval/comet_sc.py ${comet_free_model_path} "${src_path}" "${output_path}" > "${output_path}.cometkiwi" || echo "COMET-kiwi scoring failed"
        
        echo "---------------------------${src}-${tgt} (${MODEL_NAME})-------------------------------"
        cat "${output_path}.bleu"
        tail -n 1 "${output_path}.comet"
        tail -n 1 "${output_path}.cometkiwi"
        
        # Copy results to standard output path for easy reading
        cp "${output_path}" "${SAVE_DIR}/${OUTPUT_FILE_PREFIX}-${src}-${tgt}"
        cp "${output_path}.bleu" "${SAVE_DIR}/${OUTPUT_FILE_PREFIX}-${src}-${tgt}.bleu"
        cp "${output_path}.comet" "${SAVE_DIR}/${OUTPUT_FILE_PREFIX}-${src}-${tgt}.comet" 
        cp "${output_path}.cometkiwi" "${SAVE_DIR}/${OUTPUT_FILE_PREFIX}-${src}-${tgt}.cometkiwi"
    done
    
    # Summarize scores
    python eval/count_metric_score.py "${SAVE_DIR}"
    
    echo "Model ${MODEL_NAME} evaluation completed! Results saved in ${SAVE_DIR} directory"
    echo "---------------------------------------------------------------"
done

echo "All model evaluations completed!"

