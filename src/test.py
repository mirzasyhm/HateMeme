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
    hateful_memes_test_jsonl = os.path.join('..', 'datasets', 'test.jsonl')
    hateful_memes_img_dir = os.path.join('..', 'datasets')

    # Create Dataset instance for testing
    hateful_meme_test_dataset = HatefulMemesDataset(
        jsonl_file=hateful_memes_test_jsonl,
        img_dir=hateful_memes_img_dir,
        clip_processor=clip_processor,
        roberta_tokenizer=roberta_tokenizer,
        max_length=128,
        is_test=True
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
    sarcasm_detector.load_state_dict(torch.load('roberta_sarcasm_detector.pth', map_location=device))
    sarcasm_detector.eval()

    # Initialize the Hateful Meme Classifier
    classifier = HatefulMemeClassifier(clip_encoder, sarcasm_detector).to(device)
    classifier.load_state_dict(torch.load('hateful_meme_classifier.pth', map_location=device))
    classifier.eval()

    # Evaluation Metrics
    correct = 0
    total = 0
    all_preds = []
    all_labels = []  # Not available in test set, typically for unseen data

    # Assuming test set has labels; if not, adjust accordingly.
    # If test set lacks labels, you can store predictions with image paths for submission

    # For demonstration, let's assume test set has labels (otherwise, remove related parts)
    # Modify the Dataset class if necessary to include labels or handle missing labels

    with torch.no_grad():
        for batch in hateful_meme_test_loader:
            # Extract all required inputs
            roberta_input_ids = batch['roberta_input_ids'].to(device)
            roberta_attention_mask = batch['roberta_attention_mask'].to(device)
            clip_input_ids = batch['clip_input_ids'].to(device)
            clip_attention_mask = batch['clip_attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)

            # Pass all inputs to the classifier
            outputs = classifier(
                roberta_input_ids,
                roberta_attention_mask,
                clip_input_ids,
                clip_attention_mask,
                pixel_values
            )  # Shape: [batch_size, 1]

            preds = (outputs >= 0.5).float()  # Shape: [batch_size, 1]

            # If labels are present
            if 'label' in batch:
                labels = batch['label'].to(device).unsqueeze(1)  # Shape: [batch_size, 1]
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # If labels are not present (test set), collect predictions with image paths
            if 'img_path' in batch:
                img_paths = batch['img_path']
                for img_path, pred in zip(img_paths, preds):
                    print(f"Image: {img_path} | Prediction: {'Hateful' if pred.item() == 1 else 'Non-Hateful'}")

    if all_labels:  # If labels are available
        accuracy = correct / total
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        print(f"Test Set Evaluation | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")

    else:
        print("Test set does not contain labels. Predictions have been printed above.")

if __name__ == "__main__":
    import pandas as pd  # Import here to avoid issues if running dataset.py directly
    main()
