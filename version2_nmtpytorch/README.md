# MEDIQA2021 Task 3 using nmtpytorch

## Step 1: download data and split train/dev datasets
Follow steps in [MEDICA2021 task 3 README](https://github.com/abachaa/MEDIQA2021/tree/main/Task3) to download the radiology reports and generate train/dev datasets. The resulting train and dev data are stored in `json` format.

Each example (a report) consists of 5 sections, all stored as free text strings:

- subject_id
- study_id
- background
- findings
- impression

The model tries to summarize the findings section (with extra information from background section), and the impression section is the target summarization we are trying to generate.

## Step 2: run `split_input_target.py`
The script `split_input_target.py` splits the json data from step 1 into 4 files: findings (the findings section in report), bg_and_findings (the background and findings sections concatenated), impression (the impression section), and ids (the report `study_id` and `subject_id` separated by a space). Each report's entry is on its own line. The `i-th` row of each output file is a part of the `i-th` report in the original json data.

Run the script with command: `python3 split_input_target.py --input_json_path PATH_OF_JSON_FILE_FROM_STEP_ONE --output_dir OUTPUT_DIRECTORY`

## Step 3: run `run_tokenizer.sh`
First, add execution permission to the script: `chmod u+x run_tokenizer.sh`. This bash scripts invokes [Moses scripts](https://github.com/moses-smt/mosesdecoder) (in the `moses_scripts` folder) to transfer the text to lower case, conduct normalization, and then tokenization.

Run the script with command: `./run_tokenizer.sh OUTPUT_DIRECTORY` (`OUTPUT_DIRECTORY` is the same as in step 2)

## Step 4: run `split_train_dev.py`
The script `split_train_dev.py` splits each tokenized file into 3 files (correspond to train, dev and test datasets). Since we have 4 files (`findings`, `bg_and_findings`, `impression` and `ids`), each will be split into 3 files, resulting in a total of 12 files. The train/dev/test ratio can be specified. The data will be randomly shuffled before the split.

Run the script with command: `python3 split_train_dev_test.py --train_percent 80 --dev_percent 10 --input_dir OUTPUT_DIRECTORY --output_dir OUTPUT_DIRECTORY_FOR_SPLIT_DATA` 

`OUTPUT_DIRECTORY` is the same as step 2 and 3, `OUTPUT_DIRECTORY_FOR_SPLIT_DATA` can be a different directory.

Here `train_percent` and `dev_percent` need to be integers, and test percentage will equal 100 - `train_percent` - `dev_percent`.

