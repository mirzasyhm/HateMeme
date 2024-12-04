# src/test_baseline.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import HatefulMemesDataset
from transformers import CLIPProcessor

from model import CLIPEncoder, CLIPOnlyClassifier  # Import the baseline classifier

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def set_seed(seed=42):
    """
    Sets the seed for reproducibility.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # Set seed for reproducibility
    set_seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize processor
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Paths to datasets
    splits_dir = os.path.join('..', 'datasets', 'splits')  # Directory containing split files
    test_split_jsonl = os.path.join(splits_dir, 'test_split.jsonl')  # Use the new test split
    hateful_memes_img_dir = os.path.join('..', 'datasets')  # Update if necessary

    # Create Dataset instance for testing
    hateful_meme_test_dataset = HatefulMemesDataset(
        jsonl_file=test_split_jsonl,
        img_dir=hateful_memes_img_dir,
        clip_processor=clip_processor,
        roberta_tokenizer=None,  # Not needed for baseline
        max_length=128,
        is_test=False  # Set to False since it has labels
    )

    # Create DataLoader for testing
    hateful_meme_test_loader = DataLoader(
        hateful_meme_test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    # Initialize CLIP Encoder
    clip_encoder = CLIPEncoder().to(device)
    clip_encoder.eval()

    # Initialize the CLIP-Only Classifier
    clip_only_classifier = CLIPOnlyClassifier(clip_encoder).to(device)
    clip_only_classifier.load_state_dict(torch.load('models/clip_only_classifier.pth', map_location=device))
    clip_only_classifier.eval()

    # Evaluation Metrics
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in hateful_meme_test_loader:
            # Extract CLIP inputs
            clip_input_ids = batch['clip_input_ids'].to(device)
            clip_attention_mask = batch['clip_attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)  # Shape: [batch_size, 1]

            # Pass all inputs to the classifier
            outputs = clip_only_classifier(
                clip_input_ids,
                clip_attention_mask,
                pixel_values
            )  # Shape: [batch_size, 1]

            preds = (outputs >= 0.5).float()  # Shape: [batch_size, 1]

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"Baseline CLIP-Only Classifier Evaluation | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")


if __name__ == "__main__":
    main()
