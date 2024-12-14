import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer

class ViT5(nn.Module):
    def __init__(self, pretrained_model_name="VietAI/vit5-base"):
        """
        Initializes the ViT5 model.

        Args:
            pretrained_model_name (str): Name of the pretrained T5 model to use.
        """
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(pretrained_model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the ViT5 model.

        Args:
            input_ids (torch.Tensor): Input sequence token IDs.
            attention_mask (torch.Tensor): Attention mask for the input sequence.
            labels (torch.Tensor, optional): Target sequence token IDs. Defaults to None.

        Returns:
            torch.Tensor or tuple: If labels are provided, returns the loss. Otherwise, returns the model's output.
        """
        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def generate(self, input_ids, attention_mask, max_length=512):
        """
        Generates text using the ViT5 model.

        Args:
            input_ids (torch.Tensor): Input sequence token IDs.
            attention_mask (torch.Tensor): Attention mask for the input sequence.
            max_length (int): Maximum length for the generated sequence.

        Returns:
            torch.Tensor: Generated sequence token IDs.
        """
        generated_ids = self.t5.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length
        )
        return generated_ids

    def save_pretrained(self, save_path):
        """
        Saves the pretrained model and tokenizer.

        Args:
            save_path (str): Path to save the model and tokenizer.
        """
        self.t5.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @classmethod
    def from_pretrained(cls, pretrained_path):
        """
        Loads the pretrained model and tokenizer.

        Args:
            pretrained_path (str): Path to the saved model and tokenizer.

        Returns:
            ViT5: An instance of the ViT5 model.
        """
        model = cls()
        model.t5 = T5ForConditionalGeneration.from_pretrained(pretrained_path)
        model.tokenizer = T5Tokenizer.from_pretrained(pretrained_path)
        return model
