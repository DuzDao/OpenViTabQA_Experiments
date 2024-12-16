import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import argparse
from src.models.vit5 import ViT5
from src.datasets.vitabqa_dataset import ViTabQADataset
from src.trainers.vit5_trainer import ViT5Trainer
from transformers import T5Tokenizer
import nltk
nltk.download('wordnet')
nltk.download('punkt')

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
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length for input sequences.")
    parser.add_argument("--use_fp16", action="store_true", help="Use mixed precision training (FP16).")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", help="Use gradient checkpointing.")
    parser.add_argument("--initial_batch_size", type=int, default=None, help="Initial batch size for dynamic batch size.")
    parser.add_argument("--cpu_offload", action="store_true", help="Offload model to CPU during evaluation.")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Number of epochs to wait for improvement before stopping.") # Add early_stopping_patience
    
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_name)

    # Load datasets
    train_dataset = ViTabQADataset(data_path=args.train_data_path, tokenizer=tokenizer, max_length=args.max_length)
    val_dataset = ViTabQADataset(data_path=args.val_data_path, tokenizer=tokenizer, max_length=args.max_length)
    
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
        device=torch.device(args.device),
        use_fp16=args.use_fp16,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        initial_batch_size=args.initial_batch_size,
        cpu_offload=args.cpu_offload,
        early_stopping_patience=args.early_stopping_patience
    )

    # Load checkpoint if provided
    if args.load_checkpoint:
        start_step = trainer.load_checkpoint(args.load_checkpoint)
        print(f"Resuming training from step {start_step}")
    
    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
    