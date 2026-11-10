import argparse
import json
import math
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from filelock import FileLock, Timeout
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

from src import config
from src.dataset import load_sentences, make_loader
from src.loss import info_nce_loss
from src.model import SentenceEmbeddingModel


@dataclass(frozen=True)
class P1Config:
    temperature: float
    pooling: str
    projection: str
    seed: int


SEARCH_CONFIGS = [
    P1Config(0.07, "mean", "nonlinear", 42),
    P1Config(0.05, "mean", "nonlinear", 42),
    P1Config(0.10, "mean", "nonlinear", 42),
    P1Config(0.07, "cls", "nonlinear", 42),
    P1Config(0.07, "mean", "linear", 42),
]

FINAL_SEEDS = [42, 43, 44, 45, 46]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one(cfg, sentences, tokenizer, device, batch_size=config.BATCH_SIZE, epochs=config.NUM_EPOCHS):
    set_seed(cfg.seed)
    loader = make_loader(sentences, tokenizer, batch_size=batch_size)
    model = SentenceEmbeddingModel(pooling=cfg.pooling, projection=cfg.projection).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in tqdm(loader, desc=f"{cfg} epoch {epoch + 1}/{epochs}"):
            input_ids = batch["input_ids_a"].to(device)
            attention_mask = batch["attention_mask_a"].to(device)

            optimizer.zero_grad()

            with torch.autocast("cuda" if use_amp else "cpu", enabled=use_amp):
                emb_a = model(input_ids, attention_mask)
                emb_b = model(input_ids, attention_mask)
                loss = info_nce_loss(emb_a, emb_b, cfg.temperature)

            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            losses.append(loss.item())

        avg_loss = float(np.mean(losses))
        best_loss = min(best_loss, avg_loss)

    return model, best_loss


def encode(model, tokenizer, sentences, device):
    model.eval()
    outputs = []
    for i in range(0, len(sentences), config.BATCH_SIZE):
        batch = sentences[i : i + config.BATCH_SIZE]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            emb = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        outputs.append(emb.cpu())
    return torch.cat(outputs, dim=0)


def evaluate_stsb(model, tokenizer, device, stsb):
    s1, s2, labels = stsb
    e1 = F.normalize(encode(model, tokenizer, s1, device), p=2, dim=1)
    e2 = F.normalize(encode(model, tokenizer, s2, device), p=2, dim=1)
    sims = (e1 * e2).sum(dim=1).numpy()
    corr, _ = spearmanr(labels, sims)
    return float(corr)


def benchmark_e5(stsb, device):
    s1, s2, labels = stsb
    model = SentenceTransformer("intfloat/e5-small-v2", device=str(device))
    e1 = model.encode(["query: " + s for s in s1], show_progress_bar=True, normalize_embeddings=True)
    e2 = model.encode(["query: " + s for s in s2], show_progress_bar=True, normalize_embeddings=True)
    corr, _ = spearmanr(labels, (e1 * e2).sum(axis=1))
    return float(corr)


@contextmanager
def atomic_output(path):
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            yield file
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def read_results(path):
    if not path.exists():
        return {}

    results = {}
    for row in pd.read_csv(path, float_precision="round_trip").to_dict("records"):
        cfg = P1Config(row["temperature"], row["pooling"], row["projection"], row["seed"])
        if cfg in results or not math.isfinite(row["stsb_spearman"]):
            raise ValueError(f"Invalid or duplicate result for {cfg} in {path}")
        results[cfg] = row
    return results


def save_checkpoint(model, row, model_dir, checkpoint_score):
    marker = model_dir / "best_model.json"
    marker.unlink(missing_ok=True)
    with atomic_output(model_dir / "best_model.pt") as file:
        torch.save(model.state_dict(), file)
    with atomic_output(marker) as file:
        metadata = {"result": row, "checkpoint_stsb_spearman": checkpoint_score}
        file.write(json.dumps(metadata, indent=2).encode())


def run_ablation(output_dir, device, smoke=False):
    csv_path = output_dir / config.RESULTS_CSV
    model_dir = output_dir / config.MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    results = read_results(csv_path)
    print(f"Using {device}. Loaded {len(results)} completed runs from {csv_path}.")
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    sentences = load_sentences(8 if smoke else config.MAX_TRAIN_SAMPLES)
    split = "validation[:32]" if smoke else "validation"
    dataset = load_dataset("nyu-mll/glue", "stsb", split=split)
    stsb = tuple(list(dataset[column]) for column in ("sentence1", "sentence2", "label"))
    training = {"batch_size": 4, "epochs": 1} if smoke else {}

    def score(cfg):
        return results[cfg]["stsb_spearman"]

    def run_missing(configs, eligible_configs):
        for cfg in configs:
            if cfg in results:
                continue
            print(f"Training {cfg}")
            model, train_loss = train_one(cfg, sentences, tokenizer, device, **training)
            stsb_score = evaluate_stsb(model, tokenizer, device, stsb)
            if not math.isfinite(train_loss) or not math.isfinite(stsb_score):
                raise ValueError(f"Non-finite loss or STS-B correlation for {cfg}")
            row = {**asdict(cfg), "train_loss": train_loss, "stsb_spearman": stsb_score}
            previous_best = max((score(c) for c in eligible_configs if c in results), default=-math.inf)
            if stsb_score > previous_best:
                save_checkpoint(model, row, model_dir, stsb_score)
            results[cfg] = row
            with atomic_output(csv_path) as file:
                pd.DataFrame(results.values()).to_csv(file, index=False)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    run_missing(SEARCH_CONFIGS, SEARCH_CONFIGS)
    best = max(SEARCH_CONFIGS, key=score)
    final_configs = [P1Config(best.temperature, best.pooling, best.projection, seed) for seed in FINAL_SEEDS]
    run_missing(final_configs, final_configs)

    winner = max(final_configs, key=score)
    marker = model_dir / "best_model.json"
    metadata = json.loads(marker.read_text()) if marker.exists() else {}
    if metadata.get("result") != results[winner] or not (model_dir / "best_model.pt").is_file():
        print(f"Rebuilding the checkpoint for {winner} from the saved run settings.")
        model, _ = train_one(winner, sentences, tokenizer, device, **training)
        checkpoint_score = evaluate_stsb(model, tokenizer, device, stsb)
        if not math.isfinite(checkpoint_score):
            raise ValueError(f"Non-finite STS-B correlation for rebuilt checkpoint {winner}")
        save_checkpoint(model, results[winner], model_dir, checkpoint_score)
        del model

    final_scores = [score(cfg) for cfg in final_configs]
    print(pd.DataFrame(results.values()).sort_values("stsb_spearman", ascending=False).to_string(index=False))
    print(f"Best search config: {best}")
    print(f"STS-B validation Spearman over {FINAL_SEEDS}: {np.mean(final_scores):.4f} ± {np.std(final_scores, ddof=1):.4f}")
    print(f"E5-small-v2 STS-B validation Spearman: {benchmark_e5(stsb, device):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate a SimCSE sentence encoder.")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--smoke", action="store_true", help="Run the full grid on a few examples.")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir / "smoke" if args.smoke else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(output_dir / ".ablation.lock", timeout=0):
            run_ablation(output_dir, torch.device(device), smoke=args.smoke)
    except Timeout:
        parser.error(f"Another ablation is using {output_dir}. Choose a different output directory.")

if __name__ == "__main__":
    main()
