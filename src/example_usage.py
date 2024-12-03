
import os
import pandas as pd
from dataset import HatefulMemesDataset, SarcasmDataset
from transformers import CLIPProcessor, RobertaTokenizer

def main():
    # Initialize processors and tokenizers
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Paths to datasets
    hateful_memes_train_jsonl = os.path.join('..', 'datasets', 'train.jsonl')
    hateful_memes_dev_jsonl = os.path.join('..', 'datasets', 'dev.jsonl')
    hateful_memes_test_jsonl = os.path.join('..', 'datasets', 'test.jsonl')
    hateful_memes_img_dir = os.path.join('..', 'datasets')

    memotion_labels_csv = os.path.join('..', 'datasets', 'labels.csv')
    memotion_reference_csv = os.path.join('..', 'datasets', 'reference.csv')  # If needed
    memotion_images_dir = os.path.join('..', 'datasets', 'images')  # Not used in SarcasmDataset
    

    # Load Memotion dataset
    memotion_df = pd.read_csv(memotion_labels_csv)
    
        # Check for NaN in 'text_corrected'
    num_nan = memotion_df['text_corrected'].isna().sum()
    print(f"Number of NaN in 'text_corrected': {num_nan}")

    if num_nan > 0:
        print("Rows with NaN in 'text_corrected':")
        print(memotion_df[memotion_df['text_corrected'].isna()])
    else:
        print("No NaN values found in 'text_corrected'.")

    # Split Memotion dataset into training and validation sets (e.g., 80-20 split)
    from sklearn.model_selection import train_test_split
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

    hateful_meme_test_dataset = HatefulMemesDataset(
        jsonl_file=hateful_memes_test_jsonl,
        img_dir=hateful_memes_img_dir,
        clip_processor=clip_processor,
        roberta_tokenizer=roberta_tokenizer,
        max_length=128,
        is_test=True
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

    # Example of accessing a sample
    sample = hateful_meme_train_dataset[0]
    print("Hateful Meme Sample:")
    print(sample)

    sarcasm_sample = sarcasm_train_dataset[0]
    print("\nSarcasm Detection Sample:")
    print(sarcasm_sample)

if __name__ == "__main__":
    import pandas as pd  # Import here to avoid issues if running dataset.py directly
    main()

