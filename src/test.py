# src/test.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import HatefulMemesDataset
from transformers import CLIPProcessor, RobertaTokenizer

from model import CLIPEncoder, RoBERTaSarcasmDetector, HatefulMemeClassifier  # Ensure these are correctly defined

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize processors and tokenizers
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Paths to datasets
    splits_dir = os.path.join('..', 'datasets', 'splits')  # Directory containing split files
    test_split_jsonl = os.path.join(splits_dir, 'test_split.jsonl')  # Use the new test split
    hateful_memes_img_dir = os.path.join('..', 'datasets')  # Update if necessary

    # Create Dataset instance for testing
    hateful_meme_test_dataset = HatefulMemesDataset(
        jsonl_file=test_split_jsonl,
        img_dir=hateful_memes_img_dir,
        clip_processor=clip_processor,
        roberta_tokenizer=roberta_tokenizer,
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

    # Initialize CLIP Encoder and Sarcasm Detector
    clip_encoder = CLIPEncoder().to(device)
    clip_encoder.eval()

    sarcasm_detector = RoBERTaSarcasmDetector().to(device)
    sarcasm_detector.load_state_dict(torch.load('models/roberta_sarcasm_detector.pth', map_location=device))
    sarcasm_detector.eval()

    # Initialize the Hateful Meme Classifier
    classifier = HatefulMemeClassifier(clip_encoder, sarcasm_detector).to(device)
    classifier.load_state_dict(torch.load('models/hateful_meme_classifier.pth', map_location=device))
    classifier.eval()

    # Evaluation Metrics
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in hateful_meme_test_loader:
            # Extract all required inputs
            roberta_input_ids = batch['roberta_input_ids'].to(device)
            roberta_attention_mask = batch['roberta_attention_mask'].to(device)
            clip_input_ids = batch['clip_input_ids'].to(device)
            clip_attention_mask = batch['clip_attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)  # Shape: [batch_size, 1]

            # Pass all inputs to the classifier
            outputs = classifier(
                roberta_input_ids,
                roberta_attention_mask,
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

    print(f"Test Set Evaluation | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")

if __name__ == "__main__":
    import pandas as pd  # Import here to avoid issues if running dataset.py directly
    main()
