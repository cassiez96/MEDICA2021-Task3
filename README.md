# MEDICA2021-Task3

## Required packages:
- [pytorch](https://pytorch.org/get-started/locally/)
- [stanza](https://stanfordnlp.github.io/stanza/) (Stanford CoreNLP library for Python)

## Getting started
### Step 1: download data and split train/dev datasets
Follow steps in [MEDICA2021 task 3 README](https://github.com/abachaa/MEDIQA2021/tree/main/Task3) to download the radiology reports and generate train/dev datasets. The resulting train and dev data are stored in `json` format.

Each example (a report) consists of 5 sections, all stored as free text strings:

- subject_id
- study_id
- background
- findings
- impression

The model tries to summarize the findings section (with extra information from background section), and the impression section is the target summarization we are trying to generate.

### Step 2: tokenize the data and generate histograms
`prepare_data.py` takes the data from step 1, tokenize the background, findings and impression section using Stanford CoreNLP tokenizer (package `stanza`), and output to a `jsonl` format file.

It also counts the lengths of the 3 sections, and produces and saves a histogram of section length for each section after tokenizing all data. Example figure produced for 1000 reports:

![example histogram](https://i.imgur.com/kT1RnVr.png)

Run the script with the following command (assume `json` file from step 1's path is stored in `$INPUT_JSON_FILE_PATH`):

```
python3 prepare_data.py --input_json_path $INPUT_JSON_FILE_PATH --output_jsonl_path dataset/$DATASET_PATH/$FILE_NAME --histogram_dir $HISTOGRAM_DIRECTORY_PATH
```

The processed `jsonl` file will be stored at `dataset/$DATASET_PATH/$FILE_NAME`, and 3 histograms (one for each section) will be stored at directory `$HISTOGRAM_DIRECTORY_PATH`.

### Step 3: load pretrained GloVe vectors and prepare vocabulary
Load GloVe word vectors that are pretrained on 4.5 million Stanford radiology reports and prepare vocabulary for train, test and dev dataset (stored in jsonl format) according to the instructions [here](https://github.com/yuhaozhang/summarize-radiology-findings#preparation). The instructions are copied below:

First, you have to download these vectors by running:
```
chmod +x download.sh; ./download.sh
```

Then assuming you have your own radiology report corpus in the `dataset/$REPORT` directory (need 3 files: `train.jsonl`, `test.jsonl`, and `dev.jsonl`), you can prepare vocabulary and initial word vectors with:
```
python prepare_vocab.py dataset/$REPORT dataset/vocab --glove_dir dataset/glove
```

This will write vocabulary and word vectors as a numpy matrix into the dir `dataset/vocab`.


### Step 4: load data with DataLoader
The data loader located in `data/loader.py` was modified from the [Learning to Summarize Radiology Findings's data loader](https://github.com/yuhaozhang/summarize-radiology-findings/blob/master/data/loader.py). For each report with corresponding x-ray images, it first preprocesses the report by converting raw tokens in the findings, impression, and background sections to vocabulary indices, then it combines the preprocessed report with each of its x-ray images (eg. if 1 report has 2 images, then DataLoader creates 2 samples with same report but different images.)

