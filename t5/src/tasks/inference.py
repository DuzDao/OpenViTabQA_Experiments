import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

def predict(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_test = pd.read_csv(config["dataset"]["preprocessed"]["test"])
    tokenizer = AutoTokenizer.from_pretrained(config["pretrained_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["pretrained_name"])
    checkpoint = torch.load(config["train"]["checkpoint"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    res = {"qa_id": [], "answer": [], "vit5_answer": []}
    model.to(device)

    for i in tqdm(range(len(df_test))):
        qa_id = df_test["qa_id"][i]
        table = df_test["table_html"][i]
        question = df_test["question"][i]
        answer = df_test["answer"][i]

        inputs = tokenizer(question, table,
                        padding=config["tokenizer"]["padding"],
                        truncation=config["tokenizer"]["truncation"],
                        max_length=config["tokenizer"]["max_length"],
                        return_tensors=config["tokenizer"]["return_tensors"])
        out = model.generate(inputs.input_ids.to(device))
        
        predict = tokenizer.decode(out[0], skip_special_tokens=True)
        res["qa_id"].append(qa_id)
        res["answer"].append(answer)
        res["predicted_answer"].append(predict)
    
    pd.DataFrame(res).to_csv(config["predicted_path"], index=False)
