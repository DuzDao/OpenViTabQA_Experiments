import re
import ast
import pandas as pd
from tqdm import tqdm

def get_only_text_and_num(text):
  return re.sub(r'[^a-zA-Z0-9]', '', text)

class PreprocessPipeline:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def _fix_hints(self, hints):
        """
        hints: string of (list of hints)
        """
        return ast.literal_eval(hints.replace("' '", "', '").replace("\n", ", "))

    def _parse_table_dict(self, table_dict):
        """
        table_dict: string of dict
        """
        table_dict = ast.literal_eval(table_dict)
        return self._clean_table_dict(table_dict)

    def _clean_table_dict(self, table_dict):
        """
        table_dict: dict
        func: clear "\n" in key and value, add "Null" for empty value
        """
        new_dict = {}
        for key, value in table_dict.items():
            new_key = key.replace("\n", "")
            new_dict[new_key] = []
            for v in value:
                new_v = v.replace("\n", "")
                if new_v == "":
                    new_dict[new_key].append("Null")
                else:
                    new_dict[new_key].append(new_v)
        return new_dict

    def _parse_table_df(self, table_dict):
        """
        table_dict: dict of table
        """
        return pd.DataFrame(table_dict)

    def _get_answer_coordinates(self, df, answer_text):
        """
        df: pandas dataframe
        answer_text: string

        return: list[ tuple(row_index, column_index) ]
        """
        for row_index, row in df.iterrows():
          for column_index, cell_value in row.items():
            if get_only_text_and_num(str(answer_text)).lower() == get_only_text_and_num(cell_value).lower():
              return [(row_index, df.columns.get_loc(column_index))]
        return [(0, 0)]

    def preprocess(self):
        tqdm.pandas(desc="Fixing hints...")
        self.df["hints"] = self.df["hints"].progress_apply(self._fix_hints)
        tqdm.pandas(desc="Cleaning table dict...")
        self.df["table_dict"] = self.df["table_dict"].progress_apply(self._parse_table_dict)
        self.df["table_dict"] = self.df["table_dict"].progress_apply(self._clean_table_dict)
        tqdm.pandas(desc="Creating table df...")
        self.df["table_df"] = self.df["table_dict"].progress_apply(self._parse_table_df)
        tqdm.pandas(desc="Getting answer coordinates...")
        self.df["answer_coordinates"] = self.df.progress_apply(lambda x: self._get_answer_coordinates(x["table_df"], x["answer"]), axis=1)
        return self.df
    