import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import json
from src.utils.metrics import compute_metrics
import numpy as np

class ViT5Trainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size, learning_rate,
                 weight_decay, num_epochs, gradient_accumulation_steps,
                 warmup_steps, output_dir, save_steps, eval_steps, device, use_fp16=False):
        """
        Initializes the ViT5Trainer.

        Args:
            model (src.models.vit5.ViT5): The ViT5 model.
            train_dataset (src.datasets.vitabqa_dataset.ViTabQADataset): The training dataset.
            val_dataset (src.datasets.vitabqa_dataset.ViTabQADataset): The validation dataset.
            batch_size (int): Batch size for training and validation.
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            num_epochs (int): Number of training epochs.
            gradient_accumulation_steps (int): Number of steps to accumulate gradients before performing an update.
            warmup_steps (int): Number of warmup steps for the learning rate scheduler.
            output_dir (str): Directory to save model checkpoints.
            save_steps (int): Number of training steps between saving model checkpoints.
            eval_steps (int): Number of training steps between evaluating on the validation set.
            device (torch.device): Device to use for training (CPU or CUDA).
            use_fp16 (bool): Whether to use mixed precision training (FP16).
        """
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.device = device
        self.use_fp16 = use_fp16

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        total_steps = len(self.train_dataloader) * self.num_epochs // self.gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=total_steps)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.log_history = [] # Store training and evaluation history
        self.scaler = torch.cuda.amp.GradScaler() if self.use_fp16 else None # Initialize GradScaler

    def train(self):
        """Trains the ViT5 model."""
        global_step = 0
        for epoch in range(self.num_epochs):
            self.model.train()
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                self.optimizer.zero_grad()

                with torch.amp.autocast(device_type='cuda', enabled=self.use_fp16): # Use torch.amp.autocast
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / self.gradient_accumulation_steps

                if self.use_fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.use_fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.scheduler.step()
                    global_step += 1
                    progress_bar.set_postfix({"loss": loss.item() * self.gradient_accumulation_steps})

                    if global_step % self.save_steps == 0:
                        self._save_checkpoint(global_step)
                    if global_step % self.eval_steps == 0:
                        self._evaluate(global_step)
            
            # Save checkpoint at the end of each epoch
            self._save_checkpoint(global_step, epoch_end=True)
            self._evaluate(global_step, epoch_end=True)

        # Save training log history
        self._save_log_history()

    def _evaluate(self, step, epoch_end=False):
        """Evaluates the model on the validation set."""
        self.model.eval()
        all_predictions = []
        all_ground_truths = []
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.amp.autocast(device_type='cuda', enabled=self.use_fp16): # Use torch.amp.autocast
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()

                generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512)
                predictions = self.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Filter out invalid token IDs from labels before decoding
                filtered_labels = []
                for label_ids in labels:
                    valid_label_ids = label_ids[label_ids != -100] # Filter out -100
                    filtered_labels.append(valid_label_ids)
                
                ground_truths = self.model.tokenizer.batch_decode(filtered_labels, skip_special_tokens=True) # Decode filtered labels
                all_predictions.extend(predictions)
                all_ground_truths.extend(ground_truths)

        val_loss /= len(self.val_dataloader)
        metrics = compute_metrics(all_predictions, all_ground_truths)
        
        log_entry = {
            "step": step,
            "val_loss": val_loss,
            "em": metrics["em"],
            "f1": metrics["f1"],
            "rouge1": metrics["rouge1"],
            "meteor": metrics["meteor"],
            "epoch_end": epoch_end
        }
        self.log_history.append(log_entry)
        
        print(f"\nEvaluation at step {step}: Val Loss: {val_loss:.4f}, EM: {metrics['em']:.4f}, F1: {metrics['f1']:.4f}, ROUGE-1: {metrics['rouge1']:.4f}, METEOR: {metrics['meteor']:.4f}")
        self.model.train() # Set model back to train mode after evaluation

    def _save_checkpoint(self, step, epoch_end=False):
        """Saves a model checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_step_{step}.pth") if not epoch_end else os.path.join(self.output_dir, f"checkpoint_epoch_end.pth")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": step,
            "scaler_state_dict": self.scaler.state_dict() if self.use_fp16 else None
        }, checkpoint_path)
        print(f"Checkpoint saved at step {step} to {checkpoint_path}")

    def _save_log_history(self):
        """Saves training and evaluation log history to a JSON file."""
        log_path = os.path.join(self.output_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(self.log_history, f, indent=4)
        print(f"Training log saved to {log_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Loads a model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.use_fp16 and checkpoint["scaler_state_dict"]:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        step = checkpoint["step"]
        print(f"Checkpoint loaded from {checkpoint_path} at step {step}")
        return 
    