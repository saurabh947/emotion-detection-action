#!/usr/bin/env python3
"""Training script for the fusion MLP model.

This script trains a neural network to fuse facial, speech, and attention
emotion signals into a unified emotion prediction.

Usage:
    python scripts/train_fusion_mlp.py --data data/emotions.csv --output models/fusion_mlp.pt

Data Format (CSV):
    The training data should be a CSV file with the following columns:
    - facial_angry, facial_disgusted, facial_fearful, facial_happy, 
      facial_neutral, facial_sad, facial_surprised (7 columns)
    - speech_angry, speech_disgusted, speech_fearful, speech_happy,
      speech_neutral, speech_sad, speech_surprised (7 columns)  
    - stress_score, engagement_score, nervousness_score (3 columns)
    - label (0-6, ground truth emotion index matching EMOTION_ORDER)

    Alternatively, you can use a JSON format with the same fields.

Example data row:
    facial_angry,facial_disgusted,...,stress_score,engagement_score,nervousness_score,label
    0.1,0.05,...,0.3,0.7,0.2,3

Where label 3 corresponds to "happy" in EMOTION_ORDER:
    ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, random_split
except ImportError:
    print("PyTorch is required. Install with: pip install torch")
    sys.exit(1)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emotion_detection_action.emotion.learned_fusion import (
    EMOTION_ORDER,
    FusionMLP,
    save_model,
)


class EmotionFusionDataset(Dataset):
    """Dataset for training the fusion MLP."""

    FACIAL_COLS = [f"facial_{e}" for e in EMOTION_ORDER]
    SPEECH_COLS = [f"speech_{e}" for e in EMOTION_ORDER]
    ATTENTION_COLS = ["stress_score", "engagement_score", "nervousness_score"]

    def __init__(self, data_path: str | Path) -> None:
        """Load dataset from CSV or JSON file.

        Args:
            data_path: Path to the data file.
        """
        self.data_path = Path(data_path)
        self.samples: list[tuple[list[float], list[float], list[float], int]] = []

        if self.data_path.suffix == ".csv":
            self._load_csv()
        elif self.data_path.suffix == ".json":
            self._load_json()
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

    def _load_csv(self) -> None:
        """Load data from CSV file."""
        if not PANDAS_AVAILABLE:
            raise RuntimeError("pandas is required for CSV loading. Install with: pip install pandas")

        df = pd.read_csv(self.data_path)

        for _, row in df.iterrows():
            facial = [float(row[col]) for col in self.FACIAL_COLS]
            speech = [float(row[col]) for col in self.SPEECH_COLS]
            attention = [float(row[col]) for col in self.ATTENTION_COLS]
            label = int(row["label"])
            self.samples.append((facial, speech, attention, label))

    def _load_json(self) -> None:
        """Load data from JSON file."""
        with open(self.data_path) as f:
            data = json.load(f)

        for item in data:
            facial = [float(item.get(col, 0.0)) for col in self.FACIAL_COLS]
            speech = [float(item.get(col, 0.0)) for col in self.SPEECH_COLS]
            attention = [float(item.get(col, 0.0)) for col in self.ATTENTION_COLS]
            label = int(item["label"])
            self.samples.append((facial, speech, attention, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        facial, speech, attention, label = self.samples[idx]
        return (
            torch.tensor(facial, dtype=torch.float32),
            torch.tensor(speech, dtype=torch.float32),
            torch.tensor(attention, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


class SyntheticEmotionDataset(Dataset):
    """Synthetic dataset for testing/demo purposes.

    Generates synthetic emotion data with some learned patterns:
    - When facial and speech agree, the label matches
    - High stress increases likelihood of negative emotions
    - High nervousness increases likelihood of fearful
    """

    def __init__(self, num_samples: int = 10000, seed: int = 42) -> None:
        """Generate synthetic dataset.

        Args:
            num_samples: Number of samples to generate.
            seed: Random seed for reproducibility.
        """
        torch.manual_seed(seed)
        self.samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for _ in range(num_samples):
            # Generate a ground truth emotion
            true_label = torch.randint(0, 7, (1,)).item()

            # Generate facial emotions (mostly matching true label)
            facial = torch.softmax(torch.randn(7) * 0.5, dim=0)
            facial[true_label] += torch.rand(1).item() * 0.5 + 0.3
            facial = facial / facial.sum()

            # Generate speech emotions (sometimes matching, sometimes not)
            if torch.rand(1).item() > 0.3:  # 70% agreement
                speech = torch.softmax(torch.randn(7) * 0.5, dim=0)
                speech[true_label] += torch.rand(1).item() * 0.5 + 0.3
                speech = speech / speech.sum()
            else:
                # Disagreement
                alt_label = (true_label + torch.randint(1, 7, (1,)).item()) % 7
                speech = torch.softmax(torch.randn(7) * 0.5, dim=0)
                speech[alt_label] += torch.rand(1).item() * 0.5 + 0.3
                speech = speech / speech.sum()

            # Generate attention metrics
            # High stress for negative emotions (angry, disgusted, fearful, sad)
            if true_label in [0, 1, 2, 5]:  # angry, disgusted, fearful, sad
                stress = torch.rand(1).item() * 0.5 + 0.4  # 0.4-0.9
            else:
                stress = torch.rand(1).item() * 0.4  # 0-0.4

            # Engagement varies
            engagement = torch.rand(1).item() * 0.6 + 0.3  # 0.3-0.9

            # High nervousness for fearful
            if true_label == 2:  # fearful
                nervousness = torch.rand(1).item() * 0.5 + 0.4  # 0.4-0.9
            else:
                nervousness = torch.rand(1).item() * 0.4  # 0-0.4

            attention = torch.tensor([stress, engagement, nervousness], dtype=torch.float32)
            label = torch.tensor(true_label, dtype=torch.long)

            self.samples.append((facial, speech, attention, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def train_model(
    model: FusionMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
    early_stopping_patience: int = 10,
) -> tuple[FusionMLP, dict]:
    """Train the fusion model.

    Args:
        model: FusionMLP model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        epochs: Number of training epochs.
        lr: Learning rate.
        device: Device to train on.
        early_stopping_patience: Stop if no improvement for this many epochs.

    Returns:
        Tuple of (trained model, training history).
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0

        for facial, speech, attention, labels in train_loader:
            facial = facial.to(device)
            speech = speech.to(device)
            attention = attention.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            emotion_probs, _ = model(facial, speech, attention)
            loss = criterion(emotion_probs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for facial, speech, attention, labels in val_loader:
                facial = facial.to(device)
                speech = speech.to(device)
                attention = attention.to(device)
                labels = labels.to(device)

                emotion_probs, _ = model(facial, speech, attention)
                loss = criterion(emotion_probs, labels)

                val_loss += loss.item()
                val_batches += 1

                _, predicted = emotion_probs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / val_batches
        val_accuracy = correct / total

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Record history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch + 1:3d}/{epochs}: "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train the fusion MLP model for multimodal emotion fusion."
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to training data (CSV or JSON). If not provided, uses synthetic data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/fusion_mlp.pt",
        help="Output path for trained model weights.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[64, 32],
        help="Hidden layer dimensions.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to train on.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=10000,
        help="Number of synthetic samples to generate (if --data not provided).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)

    # Load or generate dataset
    if args.data:
        print(f"Loading data from {args.data}...")
        dataset = EmotionFusionDataset(args.data)
    else:
        print(f"Generating {args.synthetic_samples} synthetic samples...")
        dataset = SyntheticEmotionDataset(
            num_samples=args.synthetic_samples,
            seed=args.seed,
        )

    print(f"Dataset size: {len(dataset)}")

    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train size: {train_size}, Val size: {val_size}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
    )

    # Create model
    model = FusionMLP(
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        output_confidence=True,
    )

    print(f"Model parameters: {model.get_num_parameters():,}")

    # Train
    print(f"\nTraining on {args.device}...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, output_path)
    print(f"\nModel saved to {output_path}")

    # Print final stats
    print(f"\nFinal validation accuracy: {history['val_accuracy'][-1]:.4f}")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")


if __name__ == "__main__":
    main()
