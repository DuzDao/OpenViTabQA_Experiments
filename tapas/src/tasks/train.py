import os
import copy
import torch
from tqdm import tqdm

from src.data_utils.preprocess import PreprocessPipeline
from src.data_utils.dataset import TableDataset

from transformers import AutoModel, TapasForQuestionAnswering, TapasTokenizer, AutoTokenizer, AdamW

def train(model, train_dataloader, optimizer, device, epoch):
    model.train()
    model.to(device)
    loss_total = 0
    loss_count = 0

    for batch in tqdm(train_dataloader, desc="Training epoch {}...".format(epoch)):  
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                       labels=labels)
        loss = outputs.loss
        loss_total += loss.item()
        loss_count += len(input_ids)
        loss.backward()
        optimizer.step()

    avg_loss = loss_total / loss_count
    return avg_loss

def evaluate(model, val_dataloader, device):
    model.eval()
    model.to(device)

    loss_total = 0
    loss_count = 0

    for batch in tqdm(val_dataloader, desc="Evaluating..."):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                           labels=labels)
            loss = outputs.loss

            loss_total += loss.item()
            loss_count += len(input_ids)

    avg_loss = loss_total / loss_count
    return avg_loss

def train_main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Model
    tapas_model = TapasForQuestionAnswering.from_pretrained("google/tapas-base")
    phobert_model = AutoModel.from_pretrained("vinai/phobert-base")
    tapas_model.set_input_embeddings(phobert_model.get_input_embeddings())
    model_checkpoint = config["train"]["checkpoint"]
    if os.path.exists(model_checkpoint):
        _checkpoint_dict = torch.load(model_checkpoint)
        tapas_model.load_state_dict(_checkpoint_dict["tapas_model"])
        best_val_loss = _checkpoint_dict["best_val_loss"]
        print(f"Loaded model from {os.path.abspath(model_checkpoint)} with best loss reached {best_val_loss}")
    else:
        best_val_loss = float("inf")
        print("First time training")
    tapas_model.to(device)

    # Tokenizer
    tapas_tokenizer = TapasTokenizer.from_pretrained("google/tapas-base")
    phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    tokenizer = copy.deepcopy(tapas_tokenizer)
    tapas_special_tokens = tapas_tokenizer.all_special_tokens
    tokenizer.add_tokens(tapas_special_tokens, special_tokens=True)
    for token in tokenizer.get_vocab():
        if token not in tapas_special_tokens:
            del tokenizer.vocab[token]
    new_tokens = set(list(phobert_tokenizer.get_vocab().keys())) - set(tokenizer.get_vocab().keys())
    tokenizer.add_tokens(list(new_tokens))

    # Optimizer
    optimizer = AdamW(tapas_model.parameters(), lr=config["train"]["learning_rate"])

    # Dataset, Dataloader
    df_train = PreprocessPipeline(csv_path=config["dataset"]["train"]).preprocess()
    df_val = PreprocessPipeline(csv_path=config["dataset"]["val"]).preprocess()
    train_dataset = TableDataset(df=df_train, tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["train"]["train_bs"])
    val_dataset = TableDataset(df=df_val, tokenizer=tokenizer)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config["train"]["val_bs"])

    # Train
    for epoch in range(config["train"]["num_epochs"]):
        train_loss = train(tapas_model, train_dataloader, optimizer, device, epoch)
        val_loss = evaluate(tapas_model, val_dataloader, device)
        print(f"Epoch {epoch}: Train loss {train_loss}, Val loss {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "tapas_model": tapas_model.state_dict(),
                "best_val_loss": best_val_loss,
            }, model_checkpoint)
            print("Best model saved! Loss = {}".format(best_val_loss))
