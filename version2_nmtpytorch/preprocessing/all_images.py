import json
import os
from tqdm import tqdm
import glob

JSON_DIR = 'json'
DATA_DIR = 'data'
OUT = 'out_nmtpytorch'
MIMIC_IMAGES_DIR = os.path.join(DATA_DIR, 'mimic-cxr-images/')
INDIANA_IMAGES_DIR = os.path.join(DATA_DIR, 'indiana-images/')
OUTPUT_DIR = os.path.join(OUT, 'all_images')


class Indiana:
    images = glob.glob(INDIANA_IMAGES_DIR + "/*")


def load_images_mimic(subject_id, study_id):
    d = os.path.join(
        MIMIC_IMAGES_DIR,
        'p' + str(subject_id)[:2],  # 10000032 -> p10
        'p' + str(subject_id),
        's' + str(study_id)
    )
    return [os.path.join(d, f) for f in os.listdir(d)]


def load_images_indiana(study_id):
    s = 'CXR' + str(study_id) + '_'
    return [image for image in Indiana.images if s in image]


def write_to_file(file, text, replicate=1):
    for i in range(replicate):
        file.write(text)
        file.write("\n")


if __name__ == "__main__":
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    for split in ['train', 'dev', 'indiana_dev']:

        f_findings = open(os.path.join(output_dir, "{}.findings.tok".format(split)), "w")
        f_bg_findings = open(os.path.join(output_dir, "{}.bg_and_findings.tok".format(split)), "w")
        f_impression = open(os.path.join(output_dir, "{}.impression.tok").format(split), "w")
        f_ids = open(os.path.join(output_dir, "{}.ids".format(split)), "w")
        f_image_path = open(os.path.join(output_dir, "{}.image_path".format(split)), "w")

        features = []
        reports = json.load(open(os.path.join(JSON_DIR, split + "_tok.json"), 'r'))
        for report in tqdm(reports):
            findings = report["findings"]
            bg = report["background"]
            impression = report["impression"]
            subject_id = report["subject_id"]
            study_id = report["study_id"]

            if 'indiana' in split:
                images_for_report = load_images_indiana(study_id)
            else:
                images_for_report = load_images_mimic(subject_id, study_id)

            if len(images_for_report) == 0:  # happens 50 times for indiana
                images_for_report = ['no_image']

            for im in images_for_report:
                im = im.replace(MIMIC_IMAGES_DIR, '')
                im = im.replace(INDIANA_IMAGES_DIR, '')
                write_to_file(f_image_path, im, replicate=1)

            write_to_file(f_findings, findings, replicate=len(images_for_report))
            write_to_file(f_bg_findings, f"background: {bg} findings: {findings}", replicate=len(images_for_report))
            write_to_file(f_impression, impression, replicate=len(images_for_report))
            write_to_file(f_ids, f"{study_id} {subject_id}", replicate=len(images_for_report))
