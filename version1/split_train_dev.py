from utils import jsonl
import random
import argparse
import os
import json

parser = argparse.ArgumentParser()
# % test = 1 - % train - % dev
parser.add_argument('--train_percent', type=int, default=80, help='% of data for training set')
parser.add_argument('--dev_percent', type=int, default=10, help='% of data for dev set')
parser.add_argument('--data_path', type=str, required=True, help='Path for the jsonl file to split')
parser.add_argument('--output_dir', type=str, required=True, help='Path for the directory to place the splitted datasets')
args = parser.parse_args()
opt = vars(args)
RANDOM_SEED = 100

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    train_percent = opt["train_percent"]
    dev_percent = opt["dev_percent"]
    test_percent = 100 - train_percent - dev_percent

    if not (1 <= train_percent < 100 and 1 <= dev_percent < 100 and train_percent+dev_percent < 100):
        print("Error: --train_percent and --dev_percent needs to be a integer in range [1, 99] and their sum must be < 100")
        exit(1)

    with open(opt['data_path']) as infile:
        data = jsonl.load(infile)
        random.shuffle(data)
        print(f"Data shuffled... random seed = {RANDOM_SEED}")
        split_1  = int((train_percent / 100) * len(data))
        split_2 = int((dev_percent / 100) * len(data))
        train_data = data[:split_1]
        dev_data   = data[split_1:split_1+split_2]
        test_data = data[split_1+split_2:]
        print(f"Split train/dev/test by {train_percent}/{dev_percent}/{test_percent} ratio")
        print(f"train set: {len(train_data)} examples, dev set: {len(dev_data)} examples, test set: {len(test_data)} examples")

        with open(os.path.join(opt['output_dir'], "train.jsonl"), "w") as outfile:
            for data in train_data:
                json.dump(data, outfile)
                outfile.write("\n")
        with open(os.path.join(opt['output_dir'], "dev.jsonl"), "w") as outfile:
            for data in dev_data:
                json.dump(data, outfile)
                outfile.write("\n")
        with open(os.path.join(opt['output_dir'], "test.jsonl"), "w") as outfile:
            for data in test_data:
                json.dump(data, outfile)
                outfile.write("\n")
        print("Finished splitting data sets")