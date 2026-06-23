import torch
import torch.nn as nn
from transformers import AutoModel
import config

class SentenceEmbeddingModel(nn.Module):
    def __init__(self, pooling="mean", projection="nonlinear"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.BERT_MODEL_NAME)
        
        # freeze bert so we dont run out of mem on kaggle
        for param in self.bert.parameters():
            param.requires_grad = False

        if projection == "nonlinear":
            self.projection_head = nn.Sequential(
                nn.Linear(config.BERT_HIDDEN_SIZE, 256),
                nn.ReLU(),
                nn.Dropout(0.1), # simcse dropout mask - crucial for positive pairs
                nn.Linear(256, config.EMBEDDING_DIM),
            )
        elif projection == "linear":
            self.projection_head = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(config.BERT_HIDDEN_SIZE, config.EMBEDDING_DIM),
            )
        else:
            raise ValueError(f"unknown projection: {projection}")

        self.pooling = pooling

    def _pool(self, token_embeddings, attention_mask):
        if self.pooling == "cls":
            return token_embeddings[:, 0, :]
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = torch.sum(token_embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            return summed / counts
        raise ValueError(f"unknown pooling: {self.pooling}")

    def forward(self, input_ids, attention_mask):
        # no_grad skips tracking for bert graph, saves vram
        with torch.no_grad():
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = self._pool(output.last_hidden_state, attention_mask)
        
        return self.projection_head(pooled)
