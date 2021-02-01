import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input_json_path', type=str, required=True, help='Path for input json data (MEDIQA21 task 3)')
parser.add_argument('--output_dir', type=str, required=True, help='Directory for storing output files')
args = parser.parse_args()
opt = vars(args)

def main():
    output_dir = opt['output_dir']
    if not os.path.isdir(output_dir):
        # create if not exist
        Path(opt['output_dir']).mkdir(parents=True, exist_ok=True)

    f_findings = open(os.path.join(output_dir, "findings"), "w")
    f_bg_findings = open(os.path.join(output_dir, "bg_and_findings"), "w")
    f_impression = open(os.path.join(output_dir, "impression"), "w")
    f_ids = open(os.path.join(output_dir, "ids"), "w") # study id and subject id - to match report with xray
    with open(opt['input_json_path'], 'r') as j:
        reports = json.loads(j.read())
        # for each report, write to 4 files
        for report in reports:
            findings = report["findings"].replace("\n", " ")
            bg = report["background"].replace("\n", " ")
            impression = report["impression"].replace("\n", " ")
            subject_id = report["subject_id"]
            study_id = report["study_id"]

            write_to_file(f_findings, findings)
            write_to_file(f_bg_findings, f"BACKGROUND: {bg} FINDINGS: {findings}")
            write_to_file(f_impression, impression)
            write_to_file(f_ids, f"{study_id} {subject_id}")


def write_to_file(file, text):
    file.write(text)
    file.write("\n")


if __name__ == "__main__":
    main()