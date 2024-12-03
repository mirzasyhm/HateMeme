# src/test_setup.py

import torch
from torch.utils.data import DataLoader
from dataset import HatefulMemesDataset
from transformers import CLIPProcessor, RobertaTokenizer
from model import CLIPEncoder, RoBERTaSarcasmDetector, HatefulMemeClassifier

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    dataset = HatefulMemesDataset(
        jsonl_file='../datasets/train.jsonl',
        img_dir='../datasets',
        clip_processor=clip_processor,
        roberta_tokenizer=roberta_tokenizer,
        max_length=128,
        is_test=False
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    clip_encoder = CLIPEncoder().to(device)
    roberta_sarcasm_detector = RoBERTaSarcasmDetector().to(device)
    roberta_sarcasm_detector.eval()

    classifier = HatefulMemeClassifier(clip_encoder, roberta_sarcasm_detector).to(device)
    classifier.eval()

    for batch in dataloader:
        roberta_input_ids = batch['roberta_input_ids'].to(device)
        roberta_attention_mask = batch['roberta_attention_mask'].to(device)
        clip_input_ids = batch['clip_input_ids'].to(device)
        clip_attention_mask = batch['clip_attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)

        with torch.no_grad():
            outputs = classifier(
                roberta_input_ids,
                roberta_attention_mask,
                clip_input_ids,
                clip_attention_mask,
                pixel_values
            )
        print(f"Output: {outputs}")
        print(f"Label: {labels}")
        break

if __name__ == "__main__":
    test()
