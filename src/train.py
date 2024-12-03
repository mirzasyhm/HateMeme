# src/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings

import pandas as pd  # Ensure pandas is imported

from dataset import HatefulMemesDataset, SarcasmDataset
from transformers import CLIPProcessor, RobertaTokenizer

from model import CLIPEncoder, RoBERTaSarcasmDetector, HatefulMemeClassifier  # Ensure these are correctly defined

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Suppress the specific FutureWarning (Optional)
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False")

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize processors and tokenizers
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Paths to datasets
    hateful_memes_train_jsonl = os.path.join('..', 'datasets', 'train_corrected.jsonl')  # Updated path
    hateful_memes_dev_jsonl = os.path.join('..', 'datasets', 'dev_corrected.jsonl')      # Updated path
    hateful_memes_img_dir = os.path.join('..', 'datasets')                       # Updated image directory

    memotion_labels_csv = os.path.join('..', 'datasets', 'labels.csv')
    memotion_reference_csv = os.path.join('..', 'datasets', 'reference.csv')  # If needed
    memotion_images_dir = os.path.join('..', 'datasets', 'images')  # Not used in SarcasmDataset

    # Load Memotion dataset
    memotion_df = pd.read_csv(memotion_labels_csv)

    # Split Memotion dataset into training and validation sets (e.g., 80-20 split)
    train_df, val_df = train_test_split(memotion_df, test_size=0.2, random_state=42, stratify=memotion_df['sarcasm'])

    # Create Dataset instances
    hateful_meme_train_dataset = HatefulMemesDataset(
        jsonl_file=hateful_memes_train_jsonl,
        img_dir=hateful_memes_img_dir,
        clip_processor=clip_processor,
        roberta_tokenizer=roberta_tokenizer,
        max_length=128,
        is_test=False
    )

    hateful_meme_dev_dataset = HatefulMemesDataset(
        jsonl_file=hateful_memes_dev_jsonl,
        img_dir=hateful_memes_img_dir,
        clip_processor=clip_processor,
        roberta_tokenizer=roberta_tokenizer,
        max_length=128,
        is_test=False
    )

    sarcasm_train_dataset = SarcasmDataset(
        dataframe=train_df,
        roberta_tokenizer=roberta_tokenizer,
        max_length=128
    )

    sarcasm_val_dataset = SarcasmDataset(
        dataframe=val_df,
        roberta_tokenizer=roberta_tokenizer,
        max_length=128
    )

    # Create DataLoader instances
    hateful_meme_train_loader = DataLoader(
        hateful_meme_train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )

    hateful_meme_dev_loader = DataLoader(
        hateful_meme_dev_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    sarcasm_train_loader = DataLoader(
        sarcasm_train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )

    sarcasm_val_loader = DataLoader(
        sarcasm_val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    # Initialize and train the Sarcasm Detector
    sarcasm_detector = RoBERTaSarcasmDetector().to(device)
    sarcasm_detector.train()

    criterion_sarcasm = nn.BCELoss()
    optimizer_sarcasm = optim.AdamW(sarcasm_detector.parameters(), lr=2e-5)

    # Training loop for Sarcasm Detector
    epochs_sarcasm = 3
    for epoch in range(epochs_sarcasm):
        total_loss = 0
        for batch in sarcasm_train_loader:
            roberta_input_ids = batch['roberta_input_ids'].to(device)               # Updated
            roberta_attention_mask = batch['roberta_attention_mask'].to(device)     # Updated
            labels = batch['label'].to(device)

            optimizer_sarcasm.zero_grad()
            outputs = sarcasm_detector(roberta_input_ids, roberta_attention_mask)
            loss = criterion_sarcasm(outputs, labels)
            loss.backward()
            optimizer_sarcasm.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(sarcasm_train_loader)
        print(f"Sarcasm Detector - Epoch {epoch+1}/{epochs_sarcasm} | Loss: {avg_loss:.4f}")

    # Save the trained Sarcasm Detector
    torch.save(sarcasm_detector.state_dict(), 'roberta_sarcasm_detector.pth')
    print("Sarcasm Detector trained and saved.")

    # Initialize CLIP Encoder and Sarcasm Detector for Classification
    clip_encoder = CLIPEncoder().to(device)
    clip_encoder.eval()  # Assuming CLIP is pretrained and not fine-tuned here

    # Load the trained Sarcasm Detector with weights_only=True
    sarcasm_detector = RoBERTaSarcasmDetector().to(device)
    sarcasm_detector.load_state_dict(torch.load('roberta_sarcasm_detector.pth', weights_only=True))
    sarcasm_detector.eval()

    # Initialize the Hateful Meme Classifier
    classifier = HatefulMemeClassifier(clip_encoder, sarcasm_detector).to(device)

    # Define loss and optimizer for the classifier
    criterion_classifier = nn.BCELoss()
    optimizer_classifier = optim.AdamW(filter(lambda p: p.requires_grad, classifier.parameters()), lr=2e-5)

    # Training loop for the Hateful Meme Classifier
    epochs_classifier = 3
    for epoch in range(epochs_classifier):
        classifier.train()
        total_loss = 0
        for batch in hateful_meme_train_loader:
            roberta_input_ids = batch['roberta_input_ids'].to(device)                 # Updated
            roberta_attention_mask = batch['roberta_attention_mask'].to(device)       # Updated
            clip_input_ids = batch['clip_input_ids'].to(device)                       # Updated
            clip_attention_mask = batch['clip_attention_mask'].to(device)             # Updated
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            optimizer_classifier.zero_grad()
            outputs = classifier(
                roberta_input_ids,
                roberta_attention_mask,
                clip_input_ids,
                clip_attention_mask,
                pixel_values
            )
            loss = criterion_classifier(outputs, labels)
            loss.backward()
            optimizer_classifier.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(hateful_meme_train_loader)
        print(f"Hateful Meme Classifier - Epoch {epoch+1}/{epochs_classifier} | Train Loss: {avg_loss:.4f}")

        # Validation
        classifier.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in hateful_meme_dev_loader:
                roberta_input_ids = batch['roberta_input_ids'].to(device)                 # Updated
                roberta_attention_mask = batch['roberta_attention_mask'].to(device)       # Updated
                clip_input_ids = batch['clip_input_ids'].to(device)                       # Updated
                clip_attention_mask = batch['clip_attention_mask'].to(device)             # Updated
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['label'].to(device)

                outputs = classifier(
                    roberta_input_ids,
                    roberta_attention_mask,
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
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        print(f"Validation - Epoch {epoch+1}/{epochs_classifier} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")

    # Save the trained classifier
    torch.save(classifier.state_dict(), 'hateful_meme_classifier.pth')
    torch.save(classifier.state_dict(), '/content/drive/My Drive/models/hateful_meme_classifier.pth')  # Ensure this path exists
    torch.save(sarcasm_detector.state_dict(), '/content/drive/My Drive/models/roberta_sarcasm_detector.pth')  # Ensure this path exists
    print("Hateful Meme Classifier trained and saved.")

if __name__ == "__main__":
    main()
