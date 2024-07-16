from tqdm import tqdm
from bs4 import BeautifulSoup
import re

class Preprocess:
    def __init__(self, df, config):
        """
        save_dir: str | A path to save preprocessed data
        """
        self.df = df
        self.train_path = config["dataset"]["preprocessed"]["train"]
        self.dev_path = config["dataset"]["preprocessed"]["dev"]
        self.test_path = config["dataset"]["preprocessed"]["test"]


    def _parse_soup_text(self):
        """
        Parse a table html (table_html in df) to soup and get text.
        """
        tqdm.pandas(desc="PARSING TABLE HTML TO SOUP TEXT")
        self.df["table_html"] = self.df["table_html"].progress_apply(lambda x: BeautifulSoup(x, 'html.parser').text)
        return self.df
    

    def _clean_text(self, col_name):
        """
        Do text cleaning:
            - Lower case
            - Remove '\t', '\n', '\r'
        """
        tqdm.pandas(desc="CLEANING TEXT ON COLUMN {}".format(col_name.upper()))
        self.df[col_name] = self.df[col_name].apply(lambda x: x.lower().replace("\t", " ").replace("\r", " ").replace("\n", " ").strip())
        self.df[col_name] = self.df[col_name].apply(lambda x: re.sub(r" +", " ", x))


    def do_preprocess(self):
        self.df = self._parse_soup_text()
        self._clean_text("table_html")
        self._clean_text("question")
        self._clean_text("answer")

        return self.df
