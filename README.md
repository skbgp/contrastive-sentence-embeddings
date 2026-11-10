# Unsupervised Contrastive Learning for Sentence Embeddings

This project fine-tunes BERT-base on SNLI premises using the unsupervised SimCSE objective. Each sentence goes through BERT twice in training mode. Dropout gives the two passes different embeddings, and symmetric InfoNCE scores those embeddings as a positive pair against the other examples in the batch.

The encoder pools BERT's token outputs and projects the result to 128 dimensions. All BERT layers are trainable. The full experiment uses 50,000 premises, a batch size of 64, three epochs, and a learning rate of `3e-5`.

## Ablation

| Temperature | Pooling | Projection |
| --- | --- | --- |
| 0.07 | Mean | Nonlinear |
| 0.05 | Mean | Nonlinear |
| 0.10 | Mean | Nonlinear |
| 0.07 | CLS | Nonlinear |
| 0.07 | Mean | Linear |

The runner compares these settings at seed 42 on the STS-B validation split. It then evaluates the winning settings with four additional training seeds and reports the mean and sample standard deviation across seeds 42–46. The validation score serves both as the selection metric and the reported metric. The E5-small-v2 reference uses the same validation pairs.

The original experiment reported **0.733 ± 0.004** for mean pooling with the linear head at temperature 0.07. The E5-small-v2 reference scored **0.878** on the 1,500 validation pairs.

## Run the experiment

Use Python 3.10 or newer. Run these commands from the repository root, including on a hosted notebook with the repository checked out and internet access enabled. A CUDA GPU is recommended for the full grid.

```bash
python -m pip install -r requirements.txt
python -m src.run_ablation
```

Smoke mode runs the full grid with 8 SNLI examples and 32 validation pairs.

```bash
python -m src.run_ablation --smoke --device cpu
```

The runner saves each completed run to `ablation_results.csv`. Reuse the same output directory to resume, or pass `--output-dir` to start a separate run. Smoke results go in a separate `smoke/` subdirectory. The top-scoring checkpoint for the selected settings is saved as a PyTorch state dict in `models/best_model.pt`, with its configuration and validation score in `models/best_model.json`.
