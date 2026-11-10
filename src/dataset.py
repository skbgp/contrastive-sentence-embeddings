from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from src import config

class SentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # return same sentence twice for simcse positive pairs
        # the dropout layer in the model will make them slightly different
        return self.sentences[idx], self.sentences[idx]

def load_sentences(limit=config.MAX_TRAIN_SAMPLES):
    if limit <= 0:
        raise ValueError("The training sample limit must be positive.")
    dataset = load_dataset("stanfordnlp/snli", split="train")
    sentences = []
    for example in dataset:
        premise = example["premise"].strip()
        if premise and example["label"] != -1:
            sentences.append(premise)
        if len(sentences) >= limit:
            break
    return sentences

def make_loader(sentences, tokenizer, batch_size=config.BATCH_SIZE):
    if batch_size < 2 or len(sentences) < batch_size:
        raise ValueError("Training needs at least one full batch with two or more sentences.")

    def collate(batch):
        sentences, _ = zip(*batch)
        enc = tokenizer(list(sentences), padding=True, truncation=True, max_length=128, return_tensors="pt")
        # Both views have the same inputs. The two forward passes sample dropout separately.
        return {
            "input_ids_a": enc["input_ids"],
            "attention_mask_a": enc["attention_mask"],
            "input_ids_b": enc["input_ids"],
            "attention_mask_b": enc["attention_mask"],
        }

    return DataLoader(
        SentenceDataset(sentences),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        drop_last=True,
    )
