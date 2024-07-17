import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW
from torch.cuda.amp import autocast, GradScaler

from src.dataset.tableqa_dataset import TableQADataset, get_dataloader


def train(model, tokenizer, train_loader, optimizer, epoch, device, config):
    model.to(device)
    model.train()
    total_loss = 0.0
    total_cnt = 0
    scaler = GradScaler()
    accumulation_steps = config["train"]["accumulation_steps"]

    for i, batch in enumerate(tqdm(train_loader, desc="Training epoch {}".format(epoch))):
        inputs, labels = get_inputs_and_labels(tokenizer, config, batch, device)

        with autocast():
            outs = model(input_ids = inputs, labels = labels)
            loss = outs.loss / accumulation_steps

        total_loss += loss.item() * accumulation_steps
        total_cnt += 1
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
    return total_loss/total_cnt


def eval(model, tokenizer, eval_loader, epoch, device, config):
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_cnt = 0
    with torch.no_grad(), autocast():
        for batch in tqdm(eval_loader, "Eval epoch {}".format(epoch)):
            inputs, labels = get_inputs_and_labels(tokenizer, config, batch, device)
            outs = model(input_ids = inputs, labels = labels)
            total_loss += outs.loss.item()
            total_cnt += 1
    return total_loss/total_cnt


def get_inputs_and_labels(tokenizer, config, batch, device):
    questions, answers, tables = batch["questions"], batch["answers"], batch["table"]
    inputs = tokenizer(questions, tables,
                        padding=config["tokenizer"]["padding"],
                        truncation=config["tokenizer"]["truncation"],
                        return_tensors=config["tokenizer"]["return_tensors"],
                        max_length=config["tokenizer"]["max_length"]["input"]).input_ids
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(answers,
                            max_length=config["tokenizer"]["max_length"]["label"],
                            padding=config["tokenizer"]["padding"],
                            truncation=config["tokenizer"]["truncation"],
                            return_tensors=config["tokenizer"]["return_tensors"]).input_ids
        
    inputs = inputs.to(device)
    labels = labels.to(device)
    return inputs, labels


def train_main(config, logger):
    """
    Training data func
    ----------
    config: dict            | [Configuration]
    logger: loguru.Logger   | [To logging...]
    """

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("DEVICE USED: {}".format(str(device).upper()))

    # dataset
    train_set = TableQADataset(config["dataset"]["preprocessed"]["train"])
    dev_set = TableQADataset(config["dataset"]["preprocessed"]["dev"])
    train_loader = get_dataloader(train_set, config["train"]["bs"])
    dev_loader = get_dataloader(dev_set, config["train"]["bs"])
    logger.info("LOADED DATASET")

    # tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config["pretrained_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["pretrained_name"])

    # optimizer
    optimizer = AdamW(model.parameters(), config["train"]["lr"])

    #checkpoint
    checkpoint_file = os.path.join(config["train"]["checkpoint_dir"], "vit5_best.pt")
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        best_val_loss = checkpoint["best_val_loss"]
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info("LOADED CHECKPOINT FROM EPOCH {}".format(start_epoch))
    else:
        start_epoch = 0
        best_val_loss = float("inf")
        logger.info("FIRST TIME TRAINING")

    # training
    for epoch in range(start_epoch, config["train"]["num_epochs"]):
        loss = train(model, tokenizer, train_loader, optimizer, epoch, device, config)
        logger.info("Avg loss reaches {}.".format(loss))
        val_loss = eval(model, tokenizer, dev_loader, epoch, device, config)
        logger.info("Avg loss reaches {}.".format(val_loss))
        
        # reduce cuda mem not use
        torch.cuda.empty_cache()

        # save best model by loss
        early_stopping_cnt = 0
        if val_loss < best_val_loss:
            early_stopping_cnt = 0
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_file
            )
            logger.info("BEST MODEL SAVED")
        else:
            early_stopping_cnt += 1
            if early_stopping_cnt == config["train"]["early_stopping_patience"]:
                logger.info("EARLY STOPPING")
                break
