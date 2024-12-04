# src/dataset.py

import os
import json
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor, RobertaTokenizer

import pytesseract


def read_jsonl(file_path):
    """
    Reads a JSONL file and returns a list of dictionaries.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def process_sarcasm_label(label):
    """
    Converts sarcasm labels into binary values.
    Assuming 'sarcastic' indicates sarcasm and others do not.
    Adjust this function based on the actual label definitions.
    """
    if label.lower() in ['twisted meaning', 'very twisted']:
        return 1
    else:
        return 0


class HatefulMemesDataset(Dataset):
    def __init__(self, jsonl_file, img_dir, clip_processor, roberta_tokenizer=None, max_length=128, is_test=False):
        """
        Args:
            jsonl_file (str): Path to the JSONL file (train.jsonl, val_split.jsonl, test_split.jsonl).
            img_dir (str): Directory where images are stored.
            clip_processor (CLIPProcessor): Processor for CLIP model.
            roberta_tokenizer (RobertaTokenizer, optional): Tokenizer for RoBERTa model. Defaults to None.
            max_length (int): Maximum token length for RoBERTa.
            is_test (bool): Indicates if the dataset is a test set without labels.
        """
        self.data = read_jsonl(jsonl_file)
        self.img_dir = img_dir
        self.clip_processor = clip_processor
        self.roberta_tokenizer = roberta_tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.clip_processor.tokenizer.model_max_length = 77
        self.use_roberta = roberta_tokenizer is not None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            A dictionary containing:
                - clip_input_ids: Tensor of token ids for CLIP.
                - clip_attention_mask: Tensor of attention masks for CLIP.
                - pixel_values: Tensor of image data for CLIP.
                - label: Tensor indicating if the meme is hateful (1) or not (0).
                  (Only if not is_test)
                - img_path: Path to the image file. (Only if is_test)
        """
        sample = self.data[idx]
        text = sample.get('text', '')
        img_path = os.path.join(self.img_dir, sample['img'])

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Optionally, perform OCR if text is not reliable
        # ocr_text = pytesseract.image_to_string(image)
        # text = ocr_text if not text.strip() else text

        # Prepare inputs for CLIP with max_length=77
        clip_inputs = self.clip_processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=77  # CLIP's max sequence length
        )

        # Extract CLIP inputs
        clip_input_ids = clip_inputs['input_ids'].squeeze()            # Shape: (77,)
        clip_attention_mask = clip_inputs['attention_mask'].squeeze()  # Shape: (77,)
        pixel_values = clip_inputs['pixel_values'].squeeze()          # Shape: (3, H, W)
        
        # Ensure consistent tensor sizes
        assert clip_input_ids.size(0) == 77, f"CLIP input_ids size mismatch: expected 77, got {clip_input_ids.size(0)}"
        assert clip_attention_mask.size(0) == 77, f"CLIP attention_mask size mismatch: expected 77, got {clip_attention_mask.size(0)}"
        assert pixel_values.size(0) == 3, f"CLIP pixel_values channel size mismatch: expected 3, got {pixel_values.size(0)}"
        assert pixel_values.size(1) == 224 and pixel_values.size(2) == 224, f"CLIP pixel_values spatial size mismatch: expected (224, 224), got ({pixel_values.size(1)}, {pixel_values.size(2)})"

        if self.use_roberta:
            # Prepare inputs for RoBERTa (for sarcasm detection)
            roberta_encoding = self.roberta_tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            roberta_input_ids = roberta_encoding['input_ids'].squeeze()            # Shape: (128,)
            roberta_attention_mask = roberta_encoding['attention_mask'].squeeze()  # Shape: (128,)

            # Ensure consistent tensor sizes
            assert roberta_input_ids.size(0) == self.max_length, f"RoBERTa input_ids size mismatch: expected {self.max_length}, got {roberta_input_ids.size(0)}"
            assert roberta_attention_mask.size(0) == self.max_length, f"RoBERTa attention_mask size mismatch: expected {self.max_length}, got {roberta_attention_mask.size(0)}"

        if self.is_test:
            output = {
                'clip_input_ids': clip_input_ids,
                'clip_attention_mask': clip_attention_mask,
                'pixel_values': pixel_values,
                'img_path': img_path  # Return image path or ID for inference
            }
            if self.use_roberta:
                output.update({
                    'roberta_input_ids': roberta_input_ids,
                    'roberta_attention_mask': roberta_attention_mask
                })
            return output
        else:
            label = sample['label']
            output = {
                'clip_input_ids': clip_input_ids,
                'clip_attention_mask': clip_attention_mask,
                'pixel_values': pixel_values,
                'label': torch.tensor(label, dtype=torch.float)
            }
            if self.use_roberta:
                output.update({
                    'roberta_input_ids': roberta_input_ids,
                    'roberta_attention_mask': roberta_attention_mask
                })
            return output


class SarcasmDataset(Dataset):
    def __init__(self, dataframe, roberta_tokenizer, max_length=128):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing the sarcasm data.
            roberta_tokenizer (RobertaTokenizer): Tokenizer for RoBERTa model.
            max_length (int): Maximum token length for RoBERTa.
        """
        # Drop rows with NaN in 'text_corrected'
        self.df = dataframe.copy()
        initial_length = len(self.df)
        self.df = self.df.dropna(subset=['text_corrected'])
        final_length = len(self.df)
        print(f"Dropped {initial_length - final_length} rows due to NaN in 'text_corrected'.")

        self.roberta_tokenizer = roberta_tokenizer
        self.max_length = max_length

        # Process sarcasm labels to binary
        self.df['sarcasm_binary'] = self.df['sarcasm'].apply(process_sarcasm_label)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            A dictionary containing:
                - input_ids: Tensor of token ids for RoBERTa.
                - attention_mask: Tensor of attention masks for RoBERTa.
                - label: Tensor indicating if the text is sarcastic (1) or not (0).
        """
        row = self.df.iloc[idx]
        text = row['text_corrected']
        label = row['sarcasm_binary']

        # Prepare inputs for RoBERTa
        encoding = self.roberta_tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].squeeze()        # Shape: (max_length,)
        attention_mask = encoding['attention_mask'].squeeze()  # Shape: (max_length,)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.float)
        }
