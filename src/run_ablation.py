import gc
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

import config
from dataset import load_sentences, make_loader
from model import SentenceEmbeddingModel
from loss import info_nce_loss
import torch.nn.functional as F


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


def train_one(cfg, sentences, tokenizer, device):
    set_seed(cfg.seed)
    loader = make_loader(sentences, tokenizer)
    model = SentenceEmbeddingModel(pooling=cfg.pooling, projection=cfg.projection).to(device)
    optimizer = torch.optim.Adam(model.projection_head.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    
    best_loss = float("inf")

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        losses = []
        for batch in tqdm(loader, desc=f"{cfg} epoch {epoch + 1}/{config.NUM_EPOCHS}"):
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)

            optimizer.zero_grad()

            # fp16 so we dont run out of vram on kaggle
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=device.type == "cuda"):
                emb_a = model(input_ids_a, attention_mask_a)
                emb_b = model(input_ids_b, attention_mask_b)
                loss = info_nce_loss(emb_a, emb_b, cfg.temperature)

            # amp scaler for stability
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


def evaluate_stsb(model, tokenizer, device):
    from datasets import load_dataset
    stsb = load_dataset("glue", "stsb", split="validation")
    s1 = [ex["sentence1"] for ex in stsb]
    s2 = [ex["sentence2"] for ex in stsb]
    labels = [ex["label"] for ex in stsb]

    e1 = F.normalize(encode(model, tokenizer, s1, device), p=2, dim=1)
    e2 = F.normalize(encode(model, tokenizer, s2, device), p=2, dim=1)
    sims = (e1 * e2).sum(dim=1).numpy()
    corr, _ = spearmanr(labels, sims)
    return float(corr)


def benchmark_e5():
    from datasets import load_dataset
    stsb = load_dataset("glue", "stsb", split="validation")
    model = SentenceTransformer("intfloat/e5-small-v2")
    s1 = ["query: " + ex["sentence1"] for ex in stsb]
    s2 = ["query: " + ex["sentence2"] for ex in stsb]
    labels = [ex["label"] for ex in stsb]
    e1 = model.encode(s1, show_progress_bar=True)
    e2 = model.encode(s2, show_progress_bar=True)
    e1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = e2 / np.linalg.norm(e2, axis=1, keepdims=True)
    corr, _ = spearmanr(labels, (e1 * e2).sum(axis=1))
    return float(corr)


def run_config(cfg, sentences, tokenizer, device):
    print("\n" + "=" * 80)
    print(f"RUNNING: {cfg}")
    print("=" * 80)
    model, train_loss = train_one(cfg, sentences, tokenizer, device)
    stsb = evaluate_stsb(model, tokenizer, device)
    row = asdict(cfg)
    row.update({"train_loss": train_loss, "stsb_spearman": stsb})
    print(f"RESULT: {row}")
    return row, model


def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    sentences = load_sentences()

    # resume from csv if kaggle randomly kills the session
    if os.path.exists(config.RESULTS_CSV):
        try:
            results = pd.read_csv(config.RESULTS_CSV).to_dict("records")
            print(f"loaded {len(results)} previous results from {config.RESULTS_CSV}")
        except Exception as e:
            print(f"couldnt load previous results: {e}")
            results = []
    else:
        results = []

    seen = {(r["temperature"], r["pooling"], r["projection"], r["seed"]) for r in results}

    for cfg in SEARCH_CONFIGS:
        key = (cfg.temperature, cfg.pooling, cfg.projection, cfg.seed)
        if key in seen:
            print(f"skipping already completed config: {cfg}")
            continue

        row, _ = run_config(cfg, sentences, tokenizer, device)
        results.append(row)
        pd.DataFrame(results).to_csv(config.RESULTS_CSV, index=False)
        
        # free up gpu mem between runs so it doesnt pile up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    search_df = pd.DataFrame(results)
    best = search_df.sort_values("stsb_spearman", ascending=False).iloc[0]
    best_config = P1Config(
        float(best["temperature"]),
        str(best["pooling"]),
        str(best["projection"]),
        int(best["seed"]),
    )

    print("\nBEST SEARCH CONFIG")
    print(best_config)

    final_configs = [
        P1Config(best_config.temperature, best_config.pooling, best_config.projection, seed)
        for seed in FINAL_SEEDS
    ]

    seen = {(r["temperature"], r["pooling"], r["projection"], r["seed"]) for r in results}
    best_stsb = -1.0
    for cfg in final_configs:
        key = (cfg.temperature, cfg.pooling, cfg.projection, cfg.seed)
        if key in seen:
            continue
        row, model = run_config(cfg, sentences, tokenizer, device)
        results.append(row)
        pd.DataFrame(results).to_csv(config.RESULTS_CSV, index=False)

        if row["stsb_spearman"] > best_stsb:
            best_stsb = row["stsb_spearman"]
            save_path = os.path.join(config.MODEL_DIR, "best_model.pt")
            print(f"New best model found (STS-B: {best_stsb:.4f})! Saving to {save_path}...")
            torch.save(model.state_dict(), save_path)

        # free up gpu mem between runs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    final_df = df[
        (df["temperature"] == best_config.temperature)
        & (df["pooling"] == best_config.pooling)
        & (df["projection"] == best_config.projection)
        & (df["seed"].isin(FINAL_SEEDS))
    ]

    e5_score = benchmark_e5()
    mean = final_df["stsb_spearman"].mean()
    std = final_df["stsb_spearman"].std(ddof=1)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(df.sort_values("stsb_spearman", ascending=False).to_string(index=False))
    print(f"\nbest config: temp={best_config.temperature}, pooling={best_config.pooling}, projection={best_config.projection}")
    print(f"final stsb spearman over seeds {FINAL_SEEDS}: {mean:.4f} +/- {std:.4f}")
    print(f"e5-small-v2 stsb spearman: {e5_score:.4f}")
    print("\nfor the resume:")
    print(
        f"contrastive sentence encoder, frozen bert + infonce, "
        f"{mean:.4f} +/- {std:.4f} stsb spearman (5 seeds), "
        f"ablated temp/pooling/projection"
    )

if __name__ == "__main__":
    main()
