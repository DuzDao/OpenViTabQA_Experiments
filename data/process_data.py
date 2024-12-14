import json
import os
from utils import flatten_html_table, format_table

DATA_DIR = "data/raw"
OUTPUT_DIR = "data/processed"

def process_data():
    """Processes the raw data and saves it to processed data."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open(os.path.join(DATA_DIR, "qas_train.json"), "r") as f:
        train_qas = json.load(f)["qas"]
    with open(os.path.join(DATA_DIR, "qas_dev.json"), "r") as f:
        dev_qas = json.load(f)["qas"]
    with open(os.path.join(DATA_DIR, "qas_test.json"), "r") as f:
        test_qas = json.load(f)["qas"]
    with open(os.path.join(DATA_DIR, "table.json"), "r") as f:
        tables = {table["table_id"]: table["table_html"] for table in json.load(f)["tables"]}

    def _process_qa(qas, tables):
        processed_data = []
        for qa in qas:
            table_id = qa["table_id"]
            table_html = tables[table_id]
            table_flatten = format_table(flatten_html_table(table_html))
            processed_data.append({
                "question": qa["question"],
                "table": table_flatten,
                "answer": qa["answer"]
            })
        return processed_data

    train_processed = _process_qa(train_qas, tables)
    dev_processed = _process_qa(dev_qas, tables)
    test_processed = _process_qa(test_qas, tables)

    with open(os.path.join(OUTPUT_DIR, "train.json"), "w") as f:
        json.dump(train_processed, f, indent=4)
    with open(os.path.join(OUTPUT_DIR, "dev.json"), "w") as f:
        json.dump(dev_processed, f, indent=4)
    with open(os.path.join(OUTPUT_DIR, "test.json"), "w") as f:
        json.dump(test_processed, f, indent=4)

if __name__ == "__main__":
    process_data()
    print("Data processing completed.")
    