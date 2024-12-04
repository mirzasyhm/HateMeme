# src/split_data.py

import os
import json
from sklearn.model_selection import train_test_split

def read_jsonl(file_path):
    """
    Reads a JSONL file and returns a list of dictionaries.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(data, file_path):
    """
    Writes a list of dictionaries to a JSONL file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def split_data(original_file, output_dir, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    """
    Splits the original JSONL file into train, validation, and test sets.
    """
    assert train_size + val_size + test_size == 1.0, "Train, val, and test sizes must sum to 1."

    data = read_jsonl(original_file)
    train_val_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=[item['label'] for item in data]
    )
    train_data, val_data = train_test_split(
        train_val_data, test_size=val_size/(train_size + val_size), random_state=random_state, stratify=[item['label'] for item in train_val_data]
    )

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write the split data to JSONL files
    write_jsonl(train_data, os.path.join(output_dir, 'train_split.jsonl'))
    write_jsonl(val_data, os.path.join(output_dir, 'val_split.jsonl'))
    write_jsonl(test_data, os.path.join(output_dir, 'test_split.jsonl'))

    print(f"Data split completed:")
    print(f" - Train set: {len(train_data)} samples")
    print(f" - Validation set: {len(val_data)} samples")
    print(f" - Test set: {len(test_data)} samples")

if __name__ == "__main__":
    # Define paths
    original_train_jsonl = os.path.join('..', 'datasets', 'train.jsonl')  # Update if necessary
    output_directory = os.path.join('..', 'datasets', 'splits')          # You can choose any directory

    # Perform the split
    split_data(original_train_jsonl, output_directory)
