import argparse
import json
import os
from PIL import Image
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision import transforms
import torch
import numpy as np
from tqdm import tqdm
import glob

JSON_DIR = 'json'
DATA_DIR = 'data'
OUT = 'out_nmtpytorch'
MIMIC_IMAGES_DIR = os.path.join(DATA_DIR, 'mimic-cxr-images')
MIMIC_VIEWS = os.path.join(DATA_DIR, 'mimic-views.json')
INDIANA_IMAGES_DIR = os.path.join(DATA_DIR, 'indiana-images')
INDIANA_VIEWS = os.path.join(DATA_DIR, 'indiana-views.json')
OUTPUT_DIR = os.path.join(OUT, 'one_image')


class Indiana:
    images = [os.path.basename(x) for x in glob.glob(INDIANA_IMAGES_DIR + "/*")]
    views = json.load(open(INDIANA_VIEWS))


class Mimic:
    views = json.load(open(MIMIC_VIEWS))


def load_images_mimic(subject_id, study_id):
    d = os.path.join(
        MIMIC_IMAGES_DIR,
        'p' + str(subject_id)[:2],  # 10000032 -> p10
        'p' + str(subject_id),
        's' + str(study_id)
    )
    images = [os.path.splitext(f)[0] for f in os.listdir(d)]

    # for view in ['PA', 'LATERAL', 'AP', 'LL', 'PA RLD', 'RAO', 'PA LLD', 'AP RLD', 'SWIMMERS', 'AP AXIAL',
    #              'LAO', 'XTABLE LATERAL', 'AP LLD', 'LPO', '']:

    def iterate(images):
        for view in ['PA', 'LATERAL', 'AP', 'LL', '', 'AP AXIAL']:
            for im in images:
                if Mimic.views[im] == view:
                    return os.path.join(d, im + '.jpg')

    chosen_image = iterate(images)
    assert chosen_image is not None, (subject_id, study_id)
    return chosen_image


def load_images_indiana(study_id):
    s = 'CXR' + str(study_id) + '_'
    images = [im.replace('CXR', '').split('.png')[0] for im in Indiana.images if
              s in im]  # CXR1851_IM-0553-2001.png to 1851_IM-0553-2001

    def iterate(it):
        for view in ['Frontal', 'Lateral']:
            for im in it:
                if Indiana.views[im] == view:
                    return os.path.join(INDIANA_IMAGES_DIR, 'CXR' + im + '.png')

    chosen_image = iterate(images)
    return chosen_image


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

            # get images
            if 'indiana' in split:
                image_for_report = load_images_indiana(study_id)
                if image_for_report is None:
                    image_for_report = ''  # happens around ~50 times we have no image
            else:
                image_for_report = load_images_mimic(subject_id, study_id)

            write_to_file(f_image_path, image_for_report, replicate=1)
            write_to_file(f_findings, findings, replicate=1)
            write_to_file(f_bg_findings, f"BACKGROUND: {bg} FINDINGS: {findings}", replicate=1)
            write_to_file(f_impression, impression, replicate=1)
            write_to_file(f_ids, f"{study_id} {subject_id}", replicate=1)
