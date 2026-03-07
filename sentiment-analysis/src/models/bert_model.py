"""BERT-based sentiment classifier using HuggingFace Transformers."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm

from src.evaluation.evaluator import evaluate


class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment text data."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class BERTSentimentModel:
    """Sentiment classifier using fine-tuned DistilBERT.

    Uses DistilBERT (smaller, faster) by default — works on CPU/MX250.
    Switch to 'bert-base-uncased' if you have a better GPU.
    """

    def __init__(
        self,
        model_name: str = 'distilbert-base-uncased',
        num_labels: int = 2
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.is_trained = False
        print(f"Using device: {self.device}")

    def train(
        self,
        texts: list[str],
        labels: list[int],
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        max_len: int = 128,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> dict:
        """Fine-tune DistilBERT on sentiment data.

        Args:
            texts: List of raw or lightly cleaned text strings.
            labels: List of integer labels (0=negative, 1=positive).
            epochs: Number of training epochs (3 is standard for BERT fine-tuning).
            batch_size: Batch size (reduce to 8 if OOM on GPU).
            learning_rate: AdamW learning rate (2e-5 is the standard BERT value).
            max_len: Max token length (128 is enough for most reviews).
            test_size: Fraction for test split.
            random_state: Seed for reproducibility.

        Returns:
            Dict with accuracy, precision, recall, f1_score, report, confusion_matrix.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )

        # Datasets & Loaders
        train_ds = SentimentDataset(X_train, y_train, self.tokenizer, max_len)
        test_ds = SentimentDataset(X_test, y_test, self.tokenizer, max_len)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        # Optimizer & scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels_batch = batch['label'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_batch
                )
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs} — Avg Loss: {avg_loss:.4f}")

        self.is_trained = True
        metrics = self._evaluate(test_loader)
        metrics['n_train'] = len(X_train)
        metrics['n_test'] = len(X_test)
        return metrics

    def _evaluate(self, test_loader: DataLoader) -> dict:
        """Evaluate on the test DataLoader."""
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels_batch = batch['label'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())

        return evaluate(all_labels, all_preds)

    def predict(self, texts: list[str], max_len: int = 128) -> list[int]:
        """Predict sentiment for new texts."""
        self.model.eval()
        predictions = []

        for text in texts:
            encoding = self.tokenizer(
                text,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pred = torch.argmax(outputs.logits, dim=1).item()
                predictions.append(pred)

        return predictions

    def save(self, path: str = 'outputs/bert_model'):
        """Save the fine-tuned model and tokenizer."""
        import os
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}/")

    def load(self, path: str = 'outputs/bert_model'):
        """Load a previously saved fine-tuned model."""
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        self.is_trained = True
        print(f"Model loaded from {path}/")
