
import torch
import torch.nn as nn
from transformers import CLIPModel, RobertaModel

class CLIPEncoder(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        super(CLIPEncoder, self).__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        # Obtain the text and image embeddings
        text_embeds = outputs.text_embeds  # Shape: (batch_size, hidden_size)
        image_embeds = outputs.image_embeds  # Shape: (batch_size, hidden_size)
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
        return torch.sigmoid(logits).squeeze()  # Shape: (batch_size,)


class HatefulMemeClassifier(nn.Module):
    def __init__(self, clip_encoder, roberta_sarcasm_detector, hidden_size=512):
        super(HatefulMemeClassifier, self).__init__()
        self.clip_encoder = clip_encoder
        self.roberta_sarcasm_detector = roberta_sarcasm_detector

        # Freeze the Sarcasm Detector if you don't want to train it further
        for param in self.roberta_sarcasm_detector.parameters():
            param.requires_grad = False

        # Define fusion layers
        # CLIP's text and image embeddings + sarcasm score
        clip_hidden_size = self.clip_encoder.clip.config.hidden_size  # Typically 512
        fusion_input_size = (clip_hidden_size * 2) + 1  # Text + Image + Sarcasm score

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1)  # Binary classification
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        # Encode text and image with CLIP
        text_embeds, image_embeds = self.clip_encoder(input_ids, attention_mask, pixel_values)  # Each: (batch_size, hidden_size)

        # Encode text for sarcasm detection
        sarcasm_score = self.roberta_sarcasm_detector(input_ids, attention_mask)  # Shape: (batch_size,)
        sarcasm_score = sarcasm_score.unsqueeze(1)  # Shape: (batch_size, 1)

        # Concatenate embeddings and sarcasm score
        combined = torch.cat((text_embeds, image_embeds, sarcasm_score), dim=1)  # Shape: (batch_size, 2*hidden_size +1)

        # Classification
        logits = self.fusion(combined)  # Shape: (batch_size, 1)
        output = torch.sigmoid(logits).squeeze()  # Shape: (batch_size,)

        return output
