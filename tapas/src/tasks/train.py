import os
import torch
from tqdm import tqdm
from transformers import TapasConfig, TapasForQuestionAnswering, AutoTokenizer, AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from torch.optim.lr_scheduler import SequentialLR
from src.data_utils.preprocess import PreprocessPipeline
from src.data_utils.dataset import TableDataset

def train(model, train_dataloader, optimizer, scheduler, device, epoch):
    model.train()
    loss_total = 0
    loss_count = 0

    for batch in tqdm(train_dataloader, desc=f"Training epoch {epoch}..."):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        loss = outputs.loss
        loss_total += loss.item()
        loss_count += len(input_ids)
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_loss = loss_total / loss_count
    return avg_loss

def evaluate(model, val_dataloader, device):
    model.eval()
    loss_total = 0
    loss_count = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating..."):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            loss = outputs.loss

            loss_total += loss.item()
            loss_count += len(input_ids)

    avg_loss = loss_total / loss_count
    return avg_loss

def train_main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    # Thêm các token đặc biệt của TAPAS
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", 
                      "[EMPTY]", "[Q]", "[TS]", "[TC]", "[TR]", "[TD]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Model
    tapas_config = TapasConfig.from_pretrained("google/tapas-base-finetuned-wtq")
    tapas_config.vocab_size = len(tokenizer)
    model = TapasForQuestionAnswering(config=tapas_config)
    
    model_checkpoint = config["train"]["checkpoint"]
    if os.path.exists(model_checkpoint):
        checkpoint_dict = torch.load(model_checkpoint)
        model.load_state_dict(checkpoint_dict["model"])
        best_val_loss = checkpoint_dict["best_val_loss"]
        print(f"Loaded model from {os.path.abspath(model_checkpoint)} with best loss reached {best_val_loss}")
    else:
        best_val_loss = float("inf")
        print("Training from scratch")
    model.to(device)

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=config["train"]["learning_rate"])
    num_epochs = config["train"]["num_epochs"]
    num_training_steps = num_epochs * len(train_dataloader)
    num_warmup_steps = int(0.1 * num_training_steps)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=num_warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps - num_warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[num_warmup_steps])

    # Dataset, Dataloader
    df_train = PreprocessPipeline(csv_path=config["dataset"]["train"]).preprocess()
    df_val = PreprocessPipeline(csv_path=config["dataset"]["val"]).preprocess()
    train_dataset = TableDataset(df=df_train, tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["train"]["train_bs"], shuffle=True)
    val_dataset = TableDataset(df=df_val, tokenizer=tokenizer)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config["train"]["val_bs"])

    # Train
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, optimizer, scheduler, device, epoch)
        val_loss = evaluate(model, val_dataloader, device)
        print(f"Epoch {epoch}: Train loss {train_loss}, Val loss {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model": model.state_dict(),
                "best_val_loss": best_val_loss,
            }, model_checkpoint)
            print(f"Best model saved! Loss = {best_val_loss}")