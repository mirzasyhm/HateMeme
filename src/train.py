# src/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd  # Ensure pandas is imported

from dataset import HatefulMemesDataset, SarcasmDataset
from transformers import CLIPProcessor, RobertaTokenizer

from model import CLIPEncoder, RoBERTaSarcasmDetector, HatefulMemeClassifier  # Ensure these are correctly defined

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize processors and tokenizers
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Paths to datasets
    splits_dir = os.path.join('..', 'datasets', 'splits')  # Directory containing split files
    train_split_jsonl = os.path.join(splits_dir, 'train_split.jsonl')
    val_split_jsonl = os.path.join(splits_dir, 'val_split.jsonl')
    test_split_jsonl = os.path.join(splits_dir, 'test_split.jsonl')  # Optional, if you want to evaluate on a separate test set
    hateful_memes_img_dir = os.path.join('..', 'datasets')  # Update if necessary

    memotion_labels_csv = os.path.join('..', 'datasets', 'labels.csv')
    memotion_reference_csv = os.path.join('..', 'datasets', 'reference.csv')  # If needed
    memotion_images_dir = os.path.join('..', 'datasets', 'images')  # Not used in SarcasmDataset

    # Load Memotion dataset
    memotion_df = pd.read_csv(memotion_labels_csv)

    # Split Memotion dataset into training and validation sets (e.g., 80-20 split)
    train_df, val_df = train_test_split(memotion_df, test_size=0.2, random_state=42, stratify=memotion_df['sarcasm'])

    # Create Dataset instances
    hateful_meme_train_dataset = HatefulMemesDataset(
        jsonl_file=train_split_jsonl,
        img_dir=hateful_memes_img_dir,
        clip_processor=clip_processor,
        roberta_tokenizer=roberta_tokenizer,
        max_length=128,
        is_test=False
    )

    hateful_meme_val_dataset = HatefulMemesDataset(
        jsonl_file=val_split_jsonl,
        img_dir=hateful_memes_img_dir,
        clip_processor=clip_processor,
        roberta_tokenizer=roberta_tokenizer,
        max_length=128,
        is_test=False
    )

    # Optionally, if you want to evaluate on a separate test set
    hateful_meme_test_dataset = HatefulMemesDataset(
        jsonl_file=test_split_jsonl,
        img_dir=hateful_memes_img_dir,
        clip_processor=clip_processor,
        roberta_tokenizer=roberta_tokenizer,
        max_length=128,
        is_test=False  # Set to False since it has labels
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

    hateful_meme_val_loader = DataLoader(
        hateful_meme_val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    # Optionally, create a DataLoader for the test set
    hateful_meme_test_loader = DataLoader(
        hateful_meme_test_dataset,
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
            # Corrected Key Access for SarcasmDataset
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)  # Shape: [batch_size, 1]

            optimizer_sarcasm.zero_grad()
            outputs = sarcasm_detector(input_ids, attention_mask)  # [batch_size, 1]
            loss = criterion_sarcasm(outputs, labels)             # Both [batch_size, 1]
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

    # Load the trained Sarcasm Detector
    sarcasm_detector = RoBERTaSarcasmDetector().to(device)
    sarcasm_detector.load_state_dict(torch.load('roberta_sarcasm_detector.pth', map_location=device))
    sarcasm_detector.eval()

    # Initialize the Hateful Meme Classifier
    classifier = HatefulMemeClassifier(clip_encoder, sarcasm_detector).to(device)

    # Define loss and optimizer for the classifier
    criterion_classifier = nn.BCELoss()
    optimizer_classifier = optim.AdamW(filter(lambda p: p.requires_grad, classifier.parameters()), lr=2e-5)

    # Training loop for the Hateful Meme Classifier
    epochs_classifier = 50
    for epoch in range(epochs_classifier):
        classifier.train()
        total_train_loss = 0
        for batch in hateful_meme_train_loader:
            # Corrected Key Access for HatefulMemesDataset
            roberta_input_ids = batch['roberta_input_ids'].to(device)                 # Correct
            roberta_attention_mask = batch['roberta_attention_mask'].to(device)       # Correct
            clip_input_ids = batch['clip_input_ids'].to(device)                       # Correct
            clip_attention_mask = batch['clip_attention_mask'].to(device)             # Correct
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)  # Shape: [batch_size, 1]

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

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(hateful_meme_train_loader)

       # Validation Phase
        classifier.eval()
        total_val_loss = 0  # Initialize validation loss
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in hateful_meme_val_loader:
                # Corrected Key Access for HatefulMemesDataset
                roberta_input_ids = batch['roberta_input_ids'].to(device)                 # Correct
                roberta_attention_mask = batch['roberta_attention_mask'].to(device)       # Correct
                clip_input_ids = batch['clip_input_ids'].to(device)                       # Correct
                clip_attention_mask = batch['clip_attention_mask'].to(device)             # Correct
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['label'].to(device).unsqueeze(1)  # Shape: [batch_size, 1]

                outputs = classifier(
                    roberta_input_ids,
                    roberta_attention_mask,
                    clip_input_ids,
                    clip_attention_mask,
                    pixel_values
                )
                
                loss = criterion_classifier(outputs, labels)  # Compute loss
                total_val_loss += loss.item()  # Accumulate validation loss
            
                preds = (outputs >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(hateful_meme_val_loader)  # Average validation loss
        accuracy = correct / total
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        print(f"Epoch {epoch+1}/{epochs_classifier} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1-Score: {f1:.4f}")
    # Save the trained classifier
    os.makedirs('models', exist_ok=True)  # Ensure the models directory exists
    torch.save(classifier.state_dict(), 'models/hateful_meme_classifier.pth')
    torch.save(sarcasm_detector.state_dict(), 'models/roberta_sarcasm_detector.pth')
    print("Hateful Meme Classifier and Sarcasm Detector trained and saved.")

if __name__ == "__main__":
    main()
