#!/bin/bash
PAPER_NAME="Transformer"
INPUT_FILE="../examples/Transformer.py"  # Python input file
CLEANED_FILE="../examples/Transformer_cleaned.py"  # Preprocessed result
OUTPUT_DIR="../outputs/Transformer"
OUTPUT_REPO_DIR="../outputs/Transformer_repo"
MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_REPO_DIR

echo "------- Preprocess -------"
# Preprocess Python file
python code_process.py --input_file ${INPUT_FILE} --output_file ${CLEANED_FILE}

echo "------- Planning -------"
# Run planning based on preprocessed Python file
python planning.py --paper_name $PAPER_NAME --model_name ${MODEL_NAME} --input_python ${CLEANED_FILE} --output_dir ${OUTPUT_DIR}

echo "------- Analyzing -------"
# Analyze Python file
python analyzing.py --input_file ${CLEANED_FILE} --output_file ${OUTPUT_DIR}/analysis_result.json

echo "------- Make Paper -------"
# Create paper from analysis results
python makepaper.py --output_dir ${OUTPUT_DIR} --model_name ${MODEL_NAME}

echo "Paper generation completed!"