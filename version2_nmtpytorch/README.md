# MEDIQA2021 Task 3 using nmtpytorch

### jb
Go here : https://drive.google.com/drive/u/1/folders/1zqu7i3Pm2A4kcdeLO9d1TXQYG8UaAbnQ <br/>
Download train, dev and indiana_dev.json<br/>
Download indiana_images.zip, put it in "indiana_images" folder<br/>
Put mimic-cxr images in "images" folder<br/>
Run:<br/>
`python split_input_target.py`<br/>
Files are created in out_nmtpytorch folder (or here https://drive.google.com/drive/u/1/folders/1ATUSX3o9vhgQuHOe18WgRobftOyz3zu0)

To tokenize files (but dont do that yet), do :

```
for split in train dev indiana_dev; do
    for mode in findings bg_and_findings impression; do
      for llang in en; do
        moses_scripts/lowercase.perl < out_nmtpytorch/${split}.${mode} | moses_scripts/normalize-punctuation.perl -l $llang | \
          moses_scripts/tokenizer.perl -q -a -l $llang -threads 4 > out_nmtpytorch/${split}.${mode}.lc.norm.tok
      done
    done
done
```

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

For each report, it also looks up the corresponding xray images, extract the resnet avgpool features, and store in a `npy` file in the designated directory.

Run the script with command: `python3 split_input_target.py --input_json_path PATH_OF_JSON_FILE_FROM_STEP_ONE --output_dir_report OUTPUT_DIR_REPORT --xray_images_path PATH_TO_XRAY_IMAGES --output_dir_features OUTPUT_DIR_FEATURES`

Argument explanations:

- `PATH_OF_JSON_FILE_FROM_STEP_ONE`: file path for the json file from step 1 (eg. train.json)
- `OUTPUT_DIR_REPORT`: directory to store the split report sections
- `PATH_TO_XRAY_IMAGES`: directory of the MIMIC xray images (inside which are the p10, p10... folders)
- `OUTPUT_DIR_FEATURES`: directory to store the extracted xray images resnet features

