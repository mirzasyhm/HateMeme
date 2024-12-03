import os
import json
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor, RobertaTokenizer

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

class HatefulMemesDataset(Dataset):
    def __init__(self, jsonl_file, img_dir, clip_processor, roberta_tokenizer, max_length=128, is_test=False):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        jsonl_path = os.path.join(base_dir, '..', jsonl_file)
        self.data = read_jsonl(jsonl_path)
        self.img_dir = os.path.join(base_dir, '..', img_dir)
        self.clip_processor = clip_processor
        self.roberta_tokenizer = roberta_tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample.get('text', '')
        img_path = os.path.join(self.img_dir, sample['img'])

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Prepare inputs for CLIP with max_length=77
        clip_inputs = self.clip_processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77  # CLIP's max sequence length
        )

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

        roberta_input_ids = roberta_encoding['input_ids'].squeeze()          # Shape: (max_length,)
        roberta_attention_mask = roberta_encoding['attention_mask'].squeeze()  # Shape: (max_length,)

        # Extract CLIP inputs
        clip_input_ids = clip_inputs['input_ids'].squeeze()            # Shape: (77,)
        clip_attention_mask = clip_inputs['attention_mask'].squeeze()  # Shape: (77,)
        pixel_values = clip_inputs['pixel_values'].squeeze()          # Shape: (3, H, W)

        if self.is_test:
            return {
                'roberta_input_ids': roberta_input_ids,
                'roberta_attention_mask': roberta_attention_mask,
                'clip_input_ids': clip_input_ids,
                'clip_attention_mask': clip_attention_mask,
                'pixel_values': pixel_values,
                'img_path': img_path  # For inference purposes
            }
        else:
            label = sample['label']
            return {
                'roberta_input_ids': roberta_input_ids,
                'roberta_attention_mask': roberta_attention_mask,
                'clip_input_ids': clip_input_ids,
                'clip_attention_mask': clip_attention_mask,
                'pixel_values': pixel_values,
                'label': torch.tensor(label, dtype=torch.float)
            }


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