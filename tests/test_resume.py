import json
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import torch

from src import run_ablation as ablation


class ResumeTest(unittest.TestCase):
    def test_resume_selects_from_search_rows_and_recovers_checkpoint(self):
        selected, other = ablation.SEARCH_CONFIGS[:2]
        winner = replace(selected, seed=45)
        completed = {
            cfg: 0.7 if cfg == selected else 0.5 for cfg in ablation.SEARCH_CONFIGS
        }
        completed.update({
            replace(selected, seed=seed): 0.8 if seed == winner.seed else 0.6
            for seed in ablation.FINAL_SEEDS if seed != selected.seed
        })
        completed[replace(other, seed=44)] = 0.99
        rows = [
            {**asdict(cfg), "train_loss": 1.0, "stsb_spearman": score}
            for cfg, score in completed.items()
        ]

        with TemporaryDirectory() as directory:
            output = Path(directory)
            csv_path = output / ablation.config.RESULTS_CSV
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            with (
                patch.object(ablation.AutoTokenizer, "from_pretrained"),
                patch.object(ablation, "load_sentences", return_value=[]),
                patch.object(ablation, "load_dataset", return_value={
                    "sentence1": [], "sentence2": [], "label": [],
                }),
                patch.object(ablation, "benchmark_e5", return_value=0.9),
                patch.object(ablation, "evaluate_stsb", return_value=0.81),
                patch.object(ablation, "train_one", return_value=(torch.nn.Linear(1, 1), 1.0)) as train,
            ):
                device = torch.device("cpu")
                ablation.run_ablation(output, device)
                ablation.run_ablation(output, device)

            train.assert_called_once()
            self.assertEqual(train.call_args.args[0], winner)
            model_dir = output / ablation.config.MODEL_DIR
            metadata = json.loads((model_dir / "best_model.json").read_text())
            self.assertEqual(metadata["result"]["seed"], winner.seed)
            self.assertEqual(metadata["result"]["stsb_spearman"], 0.8)
            self.assertEqual(metadata["checkpoint_stsb_spearman"], 0.81)
            self.assertTrue((model_dir / "best_model.pt").is_file())
            self.assertEqual(len(pd.read_csv(csv_path)), len(rows))
