import random
import argparse
import os
import json
from pathlib import Path

parser = argparse.ArgumentParser()
# % test = 1 - % train - % dev
parser.add_argument('--train_percent', type=int, default=80, help='% of data for training set')
parser.add_argument('--dev_percent', type=int, default=10, help='% of data for dev set')
parser.add_argument('--input_dir', type=str, required=True, help='Path of the directory where unsplit data is stored')
parser.add_argument('--output_dir', type=str, required=True, help='Path of the directory to place the split datasets')
parser.add_argument('--filenames', nargs = '+', required=True, help = "filenames of the files need to be split (don't need directory path)")
args = parser.parse_args()
opt = vars(args)
RANDOM_SEED = 100

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    train_percent = opt["train_percent"]
    dev_percent = opt["dev_percent"]
    test_percent = 100 - train_percent - dev_percent
    filenames = opt['filenames']
    if not (1 <= train_percent < 100 and 1 <= dev_percent < 100 and train_percent+dev_percent < 100):
        print("Error: --train_percent and --dev_percent needs to be a integer in range [1, 99] and their sum must be < 100")
        exit(1)

    # read the data into memory. initialize one array for each file
    arrs = {}
    for filename in filenames:
        arrs[filename] = []

    output_dir = opt['output_dir']
    if not os.path.isdir(output_dir):
        # create if not exist
        Path(opt['output_dir']).mkdir(parents=True, exist_ok=True)

    input_dir = opt["input_dir"]
    for filename in filenames:
        with open(os.path.join(input_dir, filename), "r") as f:
            for line in f:
                arrs[filename].append(line.strip())


    # sanity check: all lists have same length
    n = len(arrs[filenames[0]])
    for i in range(1, len(filenames)):
        assert(len(arrs[filenames[i]]) == n)

    # shuffle 4 arrays in the same order
    filename_orders = arrs.keys() # keep record of key ordering
    lists_tozip = [arrs[filename] for filename in filename_orders]

    zipped = list(zip(*lists_tozip))
    random.shuffle(zipped)

    # construct shuffled arrays
    for filename in filenames:
        arrs[filename] = []

    for shuffled_lists in zipped:
        for i, filename in enumerate(filename_orders):
            arrs[filename].append(shuffled_lists[i])

    # sanity check: all lists have same length
    n = len(arrs[filenames[0]])
    for i in range(1, len(filenames)):
        assert(len(arrs[filenames[i]]) == n)

    print(f"Data shuffled... random seed = {RANDOM_SEED}")
    total_samples = len(arrs[filenames[0]])
    split_1  = int((train_percent / 100) * total_samples)
    split_2 = int((dev_percent / 100) * total_samples)

    print(f"Split train/dev/test by {train_percent}/{dev_percent}/{test_percent} ratio")

    for filename in filenames:
        # each file: one train, dev, and test file
        all_data = arrs[filename]
        train_data = all_data[:split_1]
        dev_data = all_data[split_1:split_1+split_2]
        test_data = all_data[split_1+split_2:]

        with open(os.path.join(output_dir, "train.{}".format(filename)), "w") as outfile:
            for data in train_data:
                outfile.write(data)
                outfile.write("\n")
        with open(os.path.join(output_dir, "dev.{}".format(filename)), "w") as outfile:
            for data in dev_data:
                outfile.write(data)
                outfile.write("\n")
        with open(os.path.join(output_dir, "test.{}".format(filename)), "w") as outfile:
            for data in test_data:
                outfile.write(data)
                outfile.write("\n")
    print("Finished splitting data sets")