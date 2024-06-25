import torch

class TableDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        table = item["table_df"].astype(str) # TapasTokenizer expects the table data to be text only
        encoding = self.tokenizer(table=table,
                                    queries=[item.question],
                                    answer_coordinates=[item.answer_coordinates],
                                    answer_text=[[item.answer]],
                                    padding="max_length",
                                    truncation=True,
                                    return_tensors="pt"
          )
        encoding = {key: val[-1] for key, val in encoding.items()}
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        return encoding

    def __len__(self):
        return len(self.df)
    