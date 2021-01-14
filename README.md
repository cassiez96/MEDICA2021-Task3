# MEDICA2021-Task3

# Step 1: download data and split train/dev 
Follow steps in [MEDICA2021 task 3 README](https://github.com/abachaa/MEDIQA2021/tree/main/Task3) to download and generate train/dev summarization data. The resulting train and dev data are stored in json format.

Each example (a report) consists of 5 sections, all stored as free text format:

- subject_id
- study_id
- background
- findings
- impression

We are trying to summarize the impression section from background and findings.

# Step 2: tokenize the data and generate sentence length histogram
`prepare_data.py` takes the extracted `json` data, tokenize the background, findings and impression section using Stanford CoreNLP tokenizer (package `stanza` for Python), and output to a `jsonl` format.

It also counts the lengths of the 3 sections, and produces a histograms of section length for each section after tokenizing all data. Example figure produced for 1000 reports:

![example histogram](https://i.imgur.com/kT1RnVr.png)

# TODO: DataLoader
Saving the tokenized data as `jsonl` format allows us to easily use the existing DataLoader [here](https://github.com/yuhaozhang/summarize-radiology-findings/blob/master/data/loader.py), which conducts optinal further preprocessing such convert tokens to lower cases.

TODO: modify the existing DataLoader and use in this repo
