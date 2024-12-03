# src/model.py

import torch
import torch.nn as nn
from transformers import CLIPModel, RobertaModel

class CLIPEncoder(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        super(CLIPEncoder, self).__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        
        # Print hidden sizes for verification
        text_hidden_size = self.clip.config.text_config.hidden_size
        vision_hidden_size = self.clip.config.vision_config.hidden_size
        print(f"CLIP Text Hidden Size: {text_hidden_size}")
        print(f"CLIP Vision Hidden Size: {vision_hidden_size}")
    
    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        # Obtain the text and image embeddings
        text_embeds = outputs.text_embeds  # Shape: (batch_size, text_hidden_size)
        image_embeds = outputs.image_embeds  # Shape: (batch_size, vision_hidden_size)
        return text_embeds, image_embeds

class RoBERTaSarcasmDetector(nn.Module):
    def __init__(self, pretrained_model='roberta-base'):
        super(RoBERTaSarcasmDetector, self).__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, 1)  # Binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output  # Use [CLS] token representation
        dropout_output = self.dropout(cls_output)
        logits = self.classifier(dropout_output)
        return torch.sigmoid(logits).squeeze()  # Shape: (batch_size, 1)

class HatefulMemeClassifier(nn.Module):
    def __init__(self, clip_encoder, roberta_sarcasm_detector, hidden_size=512):
        super(HatefulMemeClassifier, self).__init__()
        self.clip_encoder = clip_encoder
        self.roberta_sarcasm_detector = roberta_sarcasm_detector

        # Freeze the Sarcasm Detector if you don't want to train it further
        for param in self.roberta_sarcasm_detector.parameters():
            param.requires_grad = False

        # Access hidden sizes from CLIPConfig's text and vision configurations
        text_hidden_size = self.clip_encoder.clip.config.text_config.hidden_size
        vision_hidden_size = self.clip_encoder.clip.config.vision_config.hidden_size

        print(f"Text Hidden Size: {text_hidden_size}")
        print(f"Vision Hidden Size: {vision_hidden_size}")

        # Define projection layers to map embeddings to a common hidden size
        self.text_projection = nn.Linear(text_hidden_size, hidden_size)
        self.image_projection = nn.Linear(512, hidden_size)  # **Changed from vision_hidden_size to 512**
        self.sarcasm_projection = nn.Linear(1, hidden_size)

        # Define fusion layers
        self.fusion = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1)  # Binary classification
        )

    def forward(self, roberta_input_ids, roberta_attention_mask, clip_input_ids, clip_attention_mask, pixel_values):
        # Encode text and image with CLIP
        text_embeds, image_embeds = self.clip_encoder(clip_input_ids, clip_attention_mask, pixel_values)  # Each: (batch_size, hidden_size)

        print(f"text_embeds shape: {text_embeds.shape}")        # Expected: (batch_size, 512)
        print(f"image_embeds shape: {image_embeds.shape}")      # Expected: (batch_size, 512)

        # Encode text for sarcasm detection
        sarcasm_score = self.roberta_sarcasm_detector(roberta_input_ids, roberta_attention_mask)  # Shape: (batch_size, 1)

        print(f"sarcasm_score shape: {sarcasm_score.shape}")    # Expected: (batch_size, 1)

        # Project embeddings to common hidden size
        text_proj = self.text_projection(text_embeds)          # (batch_size, hidden_size)
        image_proj = self.image_projection(image_embeds)       # (batch_size, hidden_size)
        sarcasm_proj = self.sarcasm_projection(sarcasm_score)  # (batch_size, hidden_size)

        # Concatenate all projected features
        combined = torch.cat((text_proj, image_proj, sarcasm_proj), dim=1)  # Shape: (batch_size, hidden_size * 3)

        # Classification
        logits = self.fusion(combined)  # Shape: (batch_size, 1)
        output = torch.sigmoid(logits).squeeze()  # Shape: (batch_size,)

        return output
