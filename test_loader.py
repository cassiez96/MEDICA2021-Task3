from data.loader import DataLoader
from utils.vocab import Vocab

if __name__ == "__main__":
    opt = {
        "lower": True
    }
    vocab = Vocab("dataset/vocab/vocab.pkl", load=True)
    train_batch = DataLoader("dataset/test_dataset/train.jsonl", "../images", 5, opt, vocab)
    # TEST with pdb in command line