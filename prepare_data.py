import stanza
import argparse
import json
from collections import defaultdict
import pdb
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--json_dir', type=str, required=True, help='Directory for json data (MEDIQA21 task 3)')
parser.add_argument('--jsonl_dir', type=str, required=True, help='Directory for outputting processed jsonl data')
args = parser.parse_args()
opt = vars(args)

ENTRIES_TO_EXTRACT = ["findings", "impression", "background"]

# read json, tokenize, write to jsonl
def tokenize():
    nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True)
    outfile = open(opt['jsonl_dir'], "w")
    length_statistics = defaultdict(lambda: defaultdict(int)) # key: entry_key (eg. findings), value: {length of entry: number of times this length ocurred}
    with open(opt['json_dir'], 'r') as j:
        reports = json.loads(j.read())
        print(f"Number of reports: {len(reports)}")
        for i, report in enumerate(reports):
            # TODO: remove early termination
            if (i+1) % 1000 == 0:
                # print(f"Processed {i} reports", flush=True)
                break
            entry = {}
            for entry_key in ENTRIES_TO_EXTRACT:
                entry[entry_key] = tokenize_helper(nlp, report, entry_key, length_statistics)
            json.dump(entry, outfile)
            outfile.write("\n")
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
        plt.show()

if __name__ == "__main__":
    tokenize()
