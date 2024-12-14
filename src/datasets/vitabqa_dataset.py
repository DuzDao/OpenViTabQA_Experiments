import json
import os
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer

class ViTabQADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        Args:
            data_path (str): Path to the processed JSON data file.
            tokenizer (transformers.T5Tokenizer): The tokenizer for the T5 model.
            max_length (int): Maximum length for input sequences.
        """
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _load_data(self, data_path):
        """Loads and returns the processed data from a JSON file."""
        with open(data_path, "r") as f:
            data = json.load(f)
        return data

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing the input and target tensors for a given index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing:
                - input_ids (torch.Tensor): Input sequence token IDs.
                - attention_mask (torch.Tensor): Attention mask for the input sequence.
                - labels (torch.Tensor): Target sequence token IDs.
        """
        item = self.data[idx]
        question = item["question"]
        table = item["table"]
        answer = item["answer"]

        input_text = f"question: {question} table: {table}"
        target_text = answer

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        # Replace padding token id with -100 to ignore padding in loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": labels.squeeze()
        }
