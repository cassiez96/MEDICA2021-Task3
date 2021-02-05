import os
import stanfordnlp
from stanfordnlp.pipeline.core import DEFAULT_MODEL_DIR

if not os.path.isdir(os.path.join(DEFAULT_MODEL_DIR, 'en_ewt_models/')):
    stanfordnlp.download('en')  # do that once
import json
from tqdm import tqdm

JSON_DIR = 'json'


class Tokenizer:
    tokenizer = stanfordnlp.Pipeline(processors='tokenize', lang='en', use_gpu=True)


def tokenize_sentence(s):
    if not s:
        return ''
    doc = Tokenizer.tokenizer(s.strip())
    s = []
    for i, sentence in enumerate(doc.sentences):
        s.extend([token.text.lower() for token in sentence.tokens])
    return ' '.join(s)


if __name__ == "__main__":
    os.makedirs(JSON_DIR, exist_ok=True)

    for split in ['train', 'dev', 'indiana_dev']:
        tok_json = os.path.join(JSON_DIR, split + '_tok.json')
        if os.path.exists(tok_json):
            continue
        json_raw = os.path.join(JSON_DIR, split + '.json')
        reports = json.load(open(json_raw, 'r'))

        json_tok_content = []
        for report in tqdm(reports):
            findings = tokenize_sentence(report["findings"])
            background = tokenize_sentence(report["background"])
            impression = tokenize_sentence(report["impression"])
            subject_id = report["subject_id"]
            study_id = report["study_id"]
            json_tok_content.append(
                {
                    'findings': findings,
                    'background': background,
                    'impression': impression,
                    'subject_id': subject_id,
                    'study_id': study_id,
                }
            )

        json.dump(json_tok_content, open(tok_json, 'w'))
