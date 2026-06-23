from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import config

class SentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # return same sentence twice for simcse positive pairs
        # the dropout layer in the model will make them slightly different
        return self.sentences[idx], self.sentences[idx]

def load_sentences():
    dataset = load_dataset("stanfordnlp/snli", split="train")
    sentences = []
    for example in dataset:
        premise = example["premise"].strip()
        if premise and example["label"] != -1:
            sentences.append(premise)
        if len(sentences) >= config.MAX_TRAIN_SAMPLES:
            break
    return sentences

def make_loader(sentences, tokenizer):
    def collate(batch):
        a, b = zip(*batch)
        # dynamic padding so we dont waste gpu compute on extra zeros
        enc_a = tokenizer(list(a), padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc_b = tokenizer(list(b), padding=True, truncation=True, max_length=128, return_tensors="pt")
        return {
            "input_ids_a": enc_a["input_ids"],
            "attention_mask_a": enc_a["attention_mask"],
            "input_ids_b": enc_b["input_ids"],
            "attention_mask_b": enc_b["attention_mask"],
        }

    return DataLoader(
        SentenceDataset(sentences),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
        drop_last=True,
    )
