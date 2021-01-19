import stanza
import argparse
import json
from collections import defaultdict
import pdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input_json_path', type=str, required=True, help='Path for input json data (MEDIQA21 task 3)')
parser.add_argument('--output_jsonl_path', type=str, required=True, help='Path for outputting processed jsonl data')
parser.add_argument('--histogram_dir', type=str, required=True, help='Path for outputting report section length histogram')
args = parser.parse_args()
opt = vars(args)

ENTRIES_TO_TOKENIZE = ["findings", "impression", "background"]

# read json, tokenize, write to jsonl
def tokenize():
    nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True)
    outfile_path = opt['output_jsonl_path']
    
    error = False
    if not outfile_path.endswith("jsonl"):
        print("Error! --output_jsonl_path needs to be jsonl format")
        error = True
    elif not outfile_path.startswith("dataset/"):
        print("Error! --output_jsonl_path needs to be a file in dataset folder (for later processing)")
        error = True
    if error:
        exit()

    # create output folder if not exist
    outfile_dir = "/".join(outfile_path.split("/")[:-1])
    Path(outfile_dir).mkdir(parents=True, exist_ok=True)

    outfile = open(outfile_path, "w")
    length_statistics = defaultdict(lambda: defaultdict(int)) # key: entry_key (eg. findings), value: {length of entry: number of times this length ocurred}
    with open(opt['input_json_path'], 'r') as j:
        reports = json.loads(j.read())
        print(f"Number of reports: {len(reports)}")
        for i, report in enumerate(reports):
            if (i+1) % 100 == 0:
                # print(f"Processed {i} reports", flush=True)
                break
            # modify report and write into jsonl
            for entry_key in ENTRIES_TO_TOKENIZE:
                report[entry_key] = tokenize_helper(nlp, report, entry_key, length_statistics)
            json.dump(report, outfile)
            outfile.write("\n")
    outfile.close()
    plot_histogram(length_statistics)

# input: a report entry (eg. findings) as a string
# output: array of tokens as strings
def tokenize_helper(nlp, report, entry_key, length_statistics):
    doc = nlp(report[entry_key])
    ret = []
    for sentence in doc.sentences:
        ret += [token.text for token in sentence.tokens]
    length_statistics[entry_key][len(ret)] += 1
    return ret

def plot_histogram(length_statistics):
    for key in length_statistics:
        length_count = length_statistics[key]
        plt.bar(list(length_count.keys()), length_count.values())
        plt.title(f"Histogram for report {key}")
        plt.xlabel('Number of tokens')
        plt.ylabel('Number of occurrences')
        plt.savefig(os.path.join(opt['histogram_dir'], f"{key}_histogram.png"))
        plt.clf()
        print(f"Histogram for {key} saved.")

if __name__ == "__main__":
    tokenize()
