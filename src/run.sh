#!/bin/bash
export OPENAI_API_KEY=""
GPT_VERSION="gpt-3.5-turbo"
PAPER_NAME="Transformer"
INPUT_FILE="../examples/Transformer.py"  # Python input file
CLEANED_FILE="../examples/Transformer_cleaned.py"  # Preprocessed result
OUTPUT_DIR="../outputs/Transformer"
OUTPUT_REPO_DIR="../outputs/Transformer_repo"
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_REPO_DIR

echo "------- Preprocess -------"
# Preprocess Python file
python code_process.py --input_file ${INPUT_FILE} --output_file ${CLEANED_FILE}

echo "------- Planning -------"
# Run planning based on preprocessed Python file
python planning.py --paper_name $PAPER_NAME --gpt_version ${GPT_VERSION} --input_python ${CLEANED_FILE} --output_dir ${OUTPUT_DIR}

echo "------- Analyzing -------"
# Analyze Python file
python analyzing.py --input_file ${CLEANED_FILE} --output_file ${OUTPUT_DIR}/analysis_result.json

echo "------- Make Paper -------"
# Create paper from analysis results
python makepaper.py --output_dir ${OUTPUT_DIR}

echo "Paper generation completed!"  