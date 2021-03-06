# Tranformer-based summarization trainer

This trainer is based on huggingface transformer. It finetunes pretrained BlueBERT (BERT architecture pretrained with MIMIC data) encoder and decoder on the radiology report summarization task.

## Preprocess data
Preprocess data the same way as in version 2 (use https://github.com/cassiez96/MEDICA2021-Task3/blob/main/version2_nmtpytorch/preprocessing/json/tokenize_json.py). Place the result files in `DATASET` directory.

Run `python3 convert_json_to_jsonline.py --input_json_files_dir DATASET --output_jsonline_files_dir DATASET`. This script convert json files to json line files (one json object per line for paralleized read) and store findings and impression sections as lists of tokens (for skipping BERT tokenizer splitting input). 

## Install required packages
```
pip install git+https://github.com/huggingface/transformers
pip install datasets
pip install rouge_score
pip install nltk
```

## Run trainer
```
python3 run_seq2seq_bluebert.py \
    --do_train \
    --do_eval \
    --task summarization \
    --train_file  DATASET/train_tok_lines.json \
    --validation_file DATASET/dev_tok_lines.json \
    --output_dir MODEL_OUTPUT_DIR \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --text_column findings \
    --summary_column impression
 ```
