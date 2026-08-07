# Contrastive Sentence Embeddings

Contrastive sentence encoder using frozen BERT + InfoNCE loss. Trains a lightweight projection head to produce sentence embeddings, evaluated on STS-Benchmark.

## Architecture
`Input Sentence` → `[BERT-base-uncased] (frozen)` → `[Mean Pooling]` → `768-dim vector` → `[Projection Head] (trainable)` → `128-dim sentence embedding`

## Results & Evaluation

Ablation study across temperature, pooling strategy, and projection architecture:

| temperature | pooling | projection | stsb spearman |
|-------------|---------|------------|---------------|
| 0.07 | mean | linear | **0.6473** |
| 0.10 | mean | nonlinear | 0.5931 |
| 0.07 | mean | nonlinear | 0.5832 |
| 0.05 | mean | nonlinear | 0.5546 |
| 0.07 | cls | nonlinear | 0.4366 |

Best config across 5 seeds (42-46): **0.6376 +/- 0.0055**

E5-small-v2 baseline: 0.8780 (fully fine-tuned on 270M pairs, although it is not a fair comparison)

## Project Structure

```
sentence-embedding-trainer/
├── kaggle/
│   └── notebook.py     # self-contained kaggle pipeline
├── src/
│   ├── config.py           # hyperparameters
│   ├── model.py            # frozen bert + projection head
│   ├── dataset.py          # snli loading, simcse positive pairs
│   ├── loss.py             # infonce loss
│   └── run_ablation.py     # main script, grid search + seed runs
├── requirements.txt
└── README.md
```

## How to Run

### Kaggle (recommended)
1. Upload `kaggle/notebook.py` to a Kaggle notebook
2. Enable GPU: Settings -> Accelerator -> GPU T4 x2
3. Run the notebook top-to-bottom

### Local
```bash
pip install -r requirements.txt
PYTHONPATH=. python src/run_ablation.py
```

The script saves results to `ablation_results.csv` after each config, so it picks up where it left off if kaggle dies.
