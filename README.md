# Contrastive Sentence Embeddings

I wanted to understand how contrastive learning turns a generic language model into a sentence encoder. Started with frozen BERT and unsupervised SimCSE, then moved to supervised training with SNLI hard negatives and partial fine-tuning. Evaluated on STS-Benchmark.

## How it works

### Phase 1: Unsupervised

Take a sentence, encode it twice through BERT with dropout on. Dropout makes the two copies slightly different -- that's your positive pair. The rest of the batch are negatives. Train with InfoNCE loss to pull the pair together and push everything else apart.

BERT stays frozen here. Only the projection head (768 -> 128 dims) is trained.

### Phase 2: Supervised

Same idea but with real pairs from SNLI. The entailment hypothesis is the positive ("The dog runs" -> "An animal is moving"), the contradiction is the hard negative ("The dog runs" -> "Nothing is moving"). Much stronger signal than dropout noise.

I also unfroze the top BERT layers with a 50x smaller LR (2e-5 vs 1e-3 for the head) to avoid wrecking the pretrained weights.

## Results

### Unsupervised ablation (selected on STS-B dev)

| temperature | pooling | projection | dev spearman |
|-------------|---------|------------|-------------|
| 0.07 | mean | linear | 0.6419 |
| 0.07 | mean | nonlinear | 0.6099 |
| 0.05 | mean | nonlinear | 0.5968 |
| 0.10 | mean | nonlinear | 0.5768 |
| 0.07 | cls  | nonlinear | 0.4523 |

Best unsupervised: temp=0.07, mean pooling, linear projection.

### Supervised (reported on STS-B test)

Starting from the best unsupervised config, I added SNLI hard negatives and tried unfreezing different numbers of layers:

| unfreeze | dev | test |
|----------|-----|------|
| 0 layers | 0.7662 | - |
| 2 layers | 0.7939 | - |
| 4 layers | 0.8015 | **0.7814** |

Test score is only computed for the best config (4 layers). The others are dev-only to avoid leaking.

### Final number

**0.7814 +/- 0.0035** STS-B test spearman, averaged over 5 seeds.

E5-small-v2 gets 0.8594, but it was trained on 270M pairs with full fine-tuning. Not a fair comparison -- included as a ceiling reference.

## What I found

- **Temperature matters more than I expected.** 0.07 was the sweet spot. 0.05 made training noisy (gradients blow up through the softmax), 0.10 made all negatives look equally easy.
- **Mean pooling crushed CLS with frozen BERT.** 0.642 vs 0.452 -- a 19-point gap. CLS was pretrained for next-sentence prediction, not for summarizing a sentence. It only works if you fine-tune.
- **The big jump came from supervised training.** Going from unsupervised (0.642) to supervised with hard negatives and unfreezing (0.781) was the main improvement. Dropout noise just isn't as informative as real entailment/contradiction pairs.
- **Freezing BERT caps performance around 0.64.** The projection head can only rearrange frozen features. Unfreezing lets the model learn what actually matters for similarity.

## Methodology

I picked configs on STS-B dev and only ran test once on the winning config. Easy to forget this separation, but it prevents inflating the reported number.

## Structure

```
sentence-embedding-trainer/
├── kaggle/
│   └── notebook.py         # self-contained pipeline (runs both phases)
├── src/
│   ├── model.py            # BERT + projection head, supports partial unfreezing
│   ├── dataset.py          # unsupervised pairs + SNLI triplets
│   ├── loss.py             # InfoNCE (symmetric + hard-negative variants)
│   └── run_ablation.py     # grid search + multi-seed final eval
├── requirements.txt
└── README.md
```

## Running it

### Kaggle (recommended)
1. Upload `kaggle/notebook.py` to a Kaggle notebook
2. Enable GPU (Settings -> Accelerator -> GPU T4 x2)
3. Run top-to-bottom

Results save to `ablation_results.csv` after each config, so it resumes if the session dies.

### Local
```bash
pip install -r requirements.txt
PYTHONPATH=. python src/run_ablation.py
```
