# src/train_baseline.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd  # Ensure pandas is imported

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
    train_split_jsonl = os.path.join(splits_dir, 'train_split.jsonl')
    val_split_jsonl = os.path.join(splits_dir, 'val_split.jsonl')
    hateful_memes_img_dir = os.path.join('..', 'datasets')  # Update if necessary

    # Create Dataset instances
    hateful_meme_train_dataset = HatefulMemesDataset(
        jsonl_file=train_split_jsonl,
        img_dir=hateful_memes_img_dir,
        clip_processor=clip_processor,
        roberta_tokenizer=None,  # Not needed for baseline
        max_length=128,
        is_test=False
    )

    hateful_meme_val_dataset = HatefulMemesDataset(
        jsonl_file=val_split_jsonl,
        img_dir=hateful_memes_img_dir,
        clip_processor=clip_processor,
        roberta_tokenizer=None,  # Not needed for baseline
        max_length=128,
        is_test=False
    )

    # Create DataLoader instances
    hateful_meme_train_loader = DataLoader(
        hateful_meme_train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )

    hateful_meme_val_loader = DataLoader(
        hateful_meme_val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    # Initialize CLIP Encoder
    clip_encoder = CLIPEncoder().to(device)
    clip_encoder.eval()  # Assuming CLIP is pretrained and not fine-tuned here

    # Initialize the CLIP-Only Classifier
    clip_only_classifier = CLIPOnlyClassifier(clip_encoder).to(device)

    # Define loss and optimizer for the classifier
    criterion_classifier = nn.BCELoss()
    optimizer_classifier = optim.AdamW(clip_only_classifier.parameters(), lr=2e-5)

    # Training loop for the CLIP-Only Classifier
    epochs_classifier = 50
    for epoch in range(epochs_classifier):
        clip_only_classifier.train()
        total_loss = 0
        for batch in hateful_meme_train_loader:
            # Extract CLIP inputs
            clip_input_ids = batch['clip_input_ids'].to(device)
            clip_attention_mask = batch['clip_attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)  # Shape: [batch_size, 1]

            optimizer_classifier.zero_grad()
            outputs = clip_only_classifier(
                clip_input_ids,
                clip_attention_mask,
                pixel_values
            )
            loss = criterion_classifier(outputs, labels)
            loss.backward()
            optimizer_classifier.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(hateful_meme_train_loader)
        print(f"Baseline CLIP-Only Classifier - Epoch {epoch+1}/{epochs_classifier} | Train Loss: {avg_loss:.4f}")

        # Validation
        clip_only_classifier.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in hateful_meme_val_loader:
                # Extract CLIP inputs
                clip_input_ids = batch['clip_input_ids'].to(device)
                clip_attention_mask = batch['clip_attention_mask'].to(device)
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['label'].to(device).unsqueeze(1)  # Shape: [batch_size, 1]

                outputs = clip_only_classifier(
                    clip_input_ids,
                    clip_attention_mask,
                    pixel_values
                )
                preds = (outputs >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = correct / total
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        print(f"Validation - Epoch {epoch+1}/{epochs_classifier} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}\n")

    # Save the trained Baseline Classifier
    os.makedirs('models', exist_ok=True)  # Ensure the models directory exists
    torch.save(clip_only_classifier.state_dict(), 'models/clip_only_classifier.pth')
    print("Baseline CLIP-Only Classifier trained and saved.\n")


if __name__ == "__main__":
    main()
