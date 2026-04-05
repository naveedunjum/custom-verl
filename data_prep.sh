#!/bin/bash
set -x
train_file_path=data/train/parquet/train_base_enks.parquet
test_file_path=data/test/parquet/test_base_enks.parquet
model_path=Qwen/Qwen3-4B
### Step 1: Process Data
# First run the Python script to prepare the data. 

apptainer exec verl_vllm012.latest.sif python data/process_data.py \
	--train_files "data/train/json/train_enks_file.jsonl" \
	--test_files "data/test/json/wmt_enks_test_file.jsonl" \
	--tokenizer_path ${model_path} \
	--template_type "base" \
	--train_output_file ${train_file_path} \
	--test_output_file ${test_file_path}
