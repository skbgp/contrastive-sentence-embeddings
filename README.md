# Contrastive Sentence Embeddings

Training a sentence encoder using unsupervised SimCSE. The idea is simple: take a sentence, pass it through BERT twice with different dropout masks, and use contrastive learning to pull the two views together while pushing apart everything else in the batch.

I wanted to understand how contrastive learning actually works for sentence embeddings, so I implemented everything from scratch instead of using a library.

## How it works

Take a sentence, encode it twice through BERT with dropout enabled. Dropout produces two slightly different representations of the same sentence -- that's your positive pair. Every other sentence in the batch is a negative. Train with InfoNCE loss to pull the positive pair together and push negatives apart.

BERT is fully fine-tuned (all layers unfrozen) with a low learning rate (3e-5) to avoid catastrophic forgetting. Gradient checkpointing keeps VRAM usage manageable on a T4 GPU.

The projection head maps BERT's 768-dim output down to 128 dimensions. I tried both linear and nonlinear (with ReLU + dropout) variants.

## Ablation

I swept over 5 hyperparameter configs to find the best setup:

| temperature | pooling | projection | what it tests |
|-------------|---------|------------|---------------|
| 0.07 | mean | nonlinear | baseline |
| 0.05 | mean | nonlinear | sharper distribution |
| 0.10 | mean | nonlinear | softer distribution |
| 0.07 | cls | nonlinear | CLS vs mean pooling |
| 0.07 | mean | linear | linear vs nonlinear head |

After picking the best config, I re-run it across 5 random seeds (42-46) to get a mean and standard deviation. This is the number that goes on the resume.

E5-small-v2 is included as a reference ceiling (it was trained on 270M pairs, so not a fair comparison -- just context).

## What I learned

- **Temperature is sensitive.** Small changes (0.05 vs 0.07 vs 0.10) noticeably affect training stability and final quality.
- **Mean pooling beats CLS pooling.** CLS was pretrained for next-sentence prediction, not for summarizing a full sentence. Mean pooling averages all token embeddings and works better out of the box.
- **The projection head matters less than I expected.** Linear vs nonlinear made a smaller difference than temperature or pooling.

## Structure

```
sentence-embedding-trainer/
├── kaggle/
│   └── notebook.py         # self-contained pipeline, paste into Kaggle and run
├── src/
│   ├── config.py           # hyperparameters
│   ├── model.py            # BERT + projection head with gradient checkpointing
│   ├── dataset.py          # loads SNLI premises for unsupervised pairs
│   ├── loss.py             # symmetric InfoNCE loss
│   └── run_ablation.py     # grid search + multi-seed final eval
├── requirements.txt
└── README.md
```

## Running it

### Kaggle (recommended)
1. Create a new Kaggle notebook
2. Enable GPU (Settings -> Accelerator -> GPU T4 x2)
3. First cell: `!pip install datasets sentence-transformers`
4. Second cell: paste contents of `kaggle/notebook.py`
5. Run

Results save to `ablation_results.csv` after each config, so it resumes if the session dies.

### Local
```bash
pip install -r requirements.txt
PYTHONPATH=. python src/run_ablation.py
```

