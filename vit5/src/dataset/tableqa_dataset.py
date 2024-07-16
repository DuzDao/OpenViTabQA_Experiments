from torch.utils.data import DataLoader

class TableQADataset:
    def __init__(self, df):
        self.df = df


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        return {
            "table": self.df.iloc[idx]["table"],
            "questions": self.df.iloc[idx]["question"],
            "answers": self.df.iloc[idx]["answer"]
        }

def get_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
