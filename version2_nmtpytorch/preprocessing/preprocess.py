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
import stanfordnlp
from stanfordnlp.pipeline.core import DEFAULT_MODEL_DIR

if not os.path.isdir(os.path.join(DEFAULT_MODEL_DIR, 'en_ewt_models/')):
    stanfordnlp.download('en')  # do that once

MIMIC_IMAGES_DIR = 'images'
INDIANA_IMAGES_DIR = 'indiana-images'
OUTPUT_DIR = 'out_nmtpytorch'


class Indiana:
    images = glob.glob(INDIANA_IMAGES_DIR + "/*")


class Tokenizer:
    tokenizer = stanfordnlp.Pipeline(processors='tokenize', lang='en')


def tokenize_sentence(s):
    if not s:
        return ''
    doc = Tokenizer.tokenizer(s.strip())
    s = []
    for i, sentence in enumerate(doc.sentences):
        s.extend([token.text.lower() for token in sentence.tokens])
    return ' '.join(s)


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
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # extractor
    m = resnet50(pretrained=True).cuda()
    m.eval()
    layer_getter = IntermediateLayerGetter(m, {'avgpool': 'my_features'})

    # create dir if not exist
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    extract_feature = True

    for split in ['train', 'dev', 'indiana_dev']:
        feature_file = os.path.join(output_dir, split + '_avgpool.npy')
        if os.path.exists(feature_file):
            extract_feature = False

        f_findings = open(os.path.join(output_dir, "{}.findings.tok".format(split)), "w")
        f_bg_findings = open(os.path.join(output_dir, "{}.bg_and_findings.tok".format(split)), "w")
        f_impression = open(os.path.join(output_dir, "{}.impression.tok").format(split), "w")
        f_ids = open(os.path.join(output_dir, "{}.ids".format(split)), "w")

        features = []
        reports = json.load(open(split + ".json", 'r'))
        for report in tqdm(reports):
            findings = tokenize_sentence(report["findings"])
            bg = tokenize_sentence(report["background"])
            impression = tokenize_sentence(report["impression"])
            subject_id = report["subject_id"]
            study_id = report["study_id"]

            # get images
            if 'indiana' in split:
                images_for_report = load_images_indiana(study_id)
            else:
                images_for_report = load_images_mimic(subject_id, study_id)

            # get features
            if extract_feature:
                if len(images_for_report) == 0:  # happens 50 times for indiana
                    images_for_report = [np.zeros(2048)]
                    features.append(images_for_report[0])
                else:
                    for image in images_for_report:
                        img = preprocess(Image.open(image).convert('RGB'))
                        with torch.no_grad():
                            out = layer_getter(img.unsqueeze(0).cuda())
                            feat = out['my_features'].squeeze()
                            features.append(feat.cpu().data.numpy())

            # write samples
            write_to_file(f_findings, findings, replicate=len(images_for_report))
            write_to_file(f_bg_findings, f"BACKGROUND: {bg} FINDINGS: {findings}", replicate=len(images_for_report))
            write_to_file(f_impression, impression, replicate=len(images_for_report))
            write_to_file(f_ids, f"{study_id} {subject_id}", replicate=len(images_for_report))

        if extract_feature:
            np.save(feature_file, np.array(features))
