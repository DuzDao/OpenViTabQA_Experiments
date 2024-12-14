import torch
import argparse
import os
from src.models.vit5 import ViT5
from src.datasets.vitabqa_dataset import ViTabQADataset
from src.trainers.vit5_trainer import ViT5Trainer
from transformers import T5Tokenizer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune ViT5 model on ViTabQA dataset.")
    parser.add_argument("--train_data_path", type=str, default="data/processed/train.json", help="Path to the training data JSON file.")
    parser.add_argument("--val_data_path", type=str, default="data/processed/dev.json", help="Path to the validation data JSON file.")
    parser.add_argument("--pretrained_model_name", type=str, default="t5-small", help="Name of the pretrained T5 model.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and validation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before performing an update.")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps for the learning rate scheduler.")
    parser.add_argument("--output_dir", type=str, default="vit5_output", help="Directory to save model checkpoints.")
    parser.add_argument("--save_steps", type=int, default=500, help="Number of training steps between saving model checkpoints.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Number of training steps between evaluating on the validation set.")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to a checkpoint to load (optional).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training (CPU or CUDA).")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length for input sequences.") # Added max_length argument
    
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_name)

    # Load datasets
    train_dataset = ViTabQADataset(data_path=args.train_data_path, tokenizer=tokenizer, max_length=args.max_length) # Pass max_length
    val_dataset = ViTabQADataset(data_path=args.val_data_path, tokenizer=tokenizer, max_length=args.max_length) # Pass max_length
    
    # Initialize model
    model = ViT5(pretrained_model_name=args.pretrained_model_name)

    # Initialize trainer
    trainer = ViT5Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        device=torch.device(args.device)
    )

    # Load checkpoint if provided
    if args.load_checkpoint:
        start_step = trainer.load_checkpoint(args.load_checkpoint)
        print(f"Resuming training from step {start_step}")
    
    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
    