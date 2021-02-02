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

MIMIC_IMAGES_DIR = 'images'
OUTPUT_DIR = 'out_nmtpytorch'


def load_images(subject_id, study_id):
    d = os.path.join(
        MIMIC_IMAGES_DIR,
        'p' + str(subject_id)[:2],  # 10000032 -> p10
        'p' + str(subject_id),
        's' + str(study_id)
    )
    return [os.path.join(d, f) for f in os.listdir(d)]


def write_to_file(file, text, replicate=1):
    for i in range(replicate):
        file.write(text)
        file.write("\n")


if __name__ == "__main__":
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    m = resnet50(pretrained=True).cuda()
    m.eval()
    layer_getter = IntermediateLayerGetter(m, {'avgpool': 'my_features'})

    # create dir if not exist
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    i = 0
    for split in ['train', 'dev']:
        f_findings = open(os.path.join(output_dir, "{}.findings".format(split)), "w")
        f_bg_findings = open(os.path.join(output_dir, "{}.bg_and_findings".format(split)), "w")
        f_impression = open(os.path.join(output_dir, "{}.impression").format(split), "w")
        f_ids = open(os.path.join(output_dir, "{}.ids".format(split)), "w")
        features = []
        reports = json.load(open(split + ".json", 'r'))
        for report in tqdm(reports):
            findings = report["findings"].strip()
            bg = report["background"].strip()
            impression = report["impression"].strip()
            subject_id = report["subject_id"]
            study_id = report["study_id"]

            # connect report with corresponding images in the same order
            images_for_report = load_images(subject_id, study_id)
            for image in images_for_report:
                img = preprocess(Image.open(image).convert('RGB'))
                with torch.no_grad():
                    out = layer_getter(img.unsqueeze(0).cuda())
                    feat = out['my_features'].squeeze()
                    features.append(feat.cpu().data.numpy())

            write_to_file(f_findings, findings, replicate=len(images_for_report))
            write_to_file(f_bg_findings, f"BACKGROUND: {bg} FINDINGS: {findings}", replicate=len(images_for_report))
            write_to_file(f_impression, impression, replicate=len(images_for_report))
            write_to_file(f_ids, f"{study_id} {subject_id}", replicate=len(images_for_report))
        np.save(os.path.join(output_dir, split + '_avgpool'), np.array(features))
