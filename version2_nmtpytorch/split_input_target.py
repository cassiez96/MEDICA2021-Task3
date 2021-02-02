import argparse
import json
import os
from pathlib import Path
from PIL import Image
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision import transforms
import torch
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument('--input_json_path', type=str, required=True, help='Path for input json data (MEDIQA21 task 3)')
parser.add_argument('--output_dir_report', type=str, required=True, help='Directory for storing output files (split report sections)')
parser.add_argument('--xray_images_path', type=str, required=True, help='Directory of the xray images (inside are the p10-p19 folders)')
parser.add_argument('--output_dir_features', type=str, required=True, help='Directory for storing extracted features')
args = parser.parse_args()
opt = vars(args)

def main():
    # create dir if not exist
    output_dir = opt['output_dir_report']
    if not os.path.isdir(output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_dir_features = opt['output_dir_features']
    if not os.path.isdir(output_dir_features):
        Path(output_dir_features).mkdir(parents=True, exist_ok=True)

    output_file_prefix = os.path.splitext(os.path.basename(opt['input_json_path']))[0]
    print("output file prefix: {}".format(output_file_prefix))

    f_findings = open(os.path.join(output_dir, "{}.findings".format(output_file_prefix)), "w")
    f_bg_findings = open(os.path.join(output_dir, "{}.bg_and_findings".format(output_file_prefix)), "w")
    f_impression = open(os.path.join(output_dir, "{}.impression").format(output_file_prefix), "w")
    f_ids = open(os.path.join(output_dir, "{}.ids".format(output_file_prefix)), "w") # study id and subject id - to match report with xray

    preprocessed_images_batch = []
    all_features = []

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    m = resnet50(pretrained=True)
    if torch.cuda.is_available():
        m.to('cuda')

    print("Pretrained ResNet50 loaded")
    m.eval()
    layer_getter = IntermediateLayerGetter(m, {'avgpool': 'my_features'})

    prev_time = None
    with open(opt['input_json_path'], 'r') as j:
        reports = json.loads(j.read())

        for i, report in enumerate(reports):
            if (i+1) % 200 == 0:
                print("{} / {} reports processed".format(i+1, len(reports)))

            findings = report["findings"].replace("\n", " ")
            bg = report["background"].replace("\n", " ")
            impression = report["impression"].replace("\n", " ")
            subject_id = report["subject_id"]
            study_id = report["study_id"]

            # connect report with corresponding images in the same order
            images_for_report = load_images(subject_id, study_id, preprocess)
            num_images = len(images_for_report)

            # each report can have > 1 images: replicate report for each of its images
            write_to_file(f_findings, findings, replicate=num_images)
            write_to_file(f_bg_findings, f"BACKGROUND: {bg} FINDINGS: {findings}", replicate=num_images)
            write_to_file(f_impression, impression, replicate=num_images)
            write_to_file(f_ids, f"{study_id} {subject_id}", replicate=num_images)

            for img in images_for_report:
              preprocessed_images_batch.append(img)

            # when having a batch of 500 images or more, extract the features
            if len(preprocessed_images_batch) >= 500:
                input_tensor_batch = torch.stack(preprocessed_images_batch)
                extracted_features = extract_features(input_tensor_batch, layer_getter)

                print(f"Joined features shape: {extracted_features.shape}")
                all_features.append(extracted_features)
                preprocessed_images_batch = []

                if prev_time is not None:
                  print(f"Extract image features for {input_tensor_batch.shape[0]} images: took {time.time()-prev_time} seconds")
                prev_time = time.time()

    all_features = np.concatenate(all_features, axis=0)
    save_features(all_features, output_file_prefix)
    

# return [Tensor(image feature)]
def load_images(subject_id, study_id, preprocess):
    images = []

    report_images_path = os.path.join(
        opt["xray_images_path"],
        'p' + str(subject_id)[:2],  # 10000032 -> p10
        'p' + str(subject_id),
        's' + str(study_id)
    )
    # print(f"Loading images from {report_images_path}")
    for f in os.listdir(report_images_path):
        if f.endswith(".jpg"):
            image = Image.open(os.path.join(report_images_path, f))
            processed_image = preprocess(image)
            image.close()
            del image
            images.append(processed_image)

    if len(images) == 0:
        raise Exception(f"Data mismatch error! Image not found for study_id: {study_id} and subject_id: {subject_id}.")
    return images

def write_to_file(file, text, replicate=1):
    for i in range(replicate):
        file.write(text)
        file.write("\n")


def extract_features(input_tensor, layer_getter):
    features = None

    with torch.no_grad():
        if torch.cuda.is_available():
            input_tensor = input_tensor.to("cuda")
        out = layer_getter(input_tensor)
        feat = out['my_features'].squeeze()
        features = feat.cpu().data.numpy()
        # print(f"Joined features shape: {features.shape}")
    return features


def save_features(features, output_file_prefix):
    # save to file
    print(f"Writing features to npy file..")
    output_file = open(os.path.join(opt["output_dir_features"], f"{dataset}-resnet50-avgpool.npy"), "wb")
    np.save(output_file, features)
    output_file.close()
    print(f"Done writing features for {opt['input_json_path']}")

if __name__ == "__main__":
    main()
