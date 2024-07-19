from tqdm import tqdm
from bs4 import BeautifulSoup
import re

class Preprocess:
    def __init__(self, df):
        self.df = df

    def _clean_text(self, text):
        """
        text: str   | [text need to clean]
        """
        text = text.lower()
        text = text.replace("\n", "")
        text = text.replace("\t", "")
        text = text.replace("\r", "")
        return text

    def _get_table(self, table_html):
        """
        Get all table data.
        Each cell of row is splited by <eoc>, which is standed for end of cell
        Each row of table is splited by <eor>, which is standed for end of row
        ----------
        table_html: str     | [can-be-soup]
        """
        final_table = ""
        soup = BeautifulSoup(table_html, "html.parser")
        for row in soup.find_all("tr"):
            for cell in row.findChildren(recursive=False):
                cleaned_text = self._clean_text(cell.text)
                # change empty cell value to nul
                cleaned_text = cleaned_text if cleaned_text != "" else "nul"
                final_table = final_table + cleaned_text + " <eoc> "
            final_table += "<eor> "
        return final_table
    
    def do_preprocess(self):
        tqdm.pandas(desc="Doing preprocessing...")
        self.df["table"] = self.df["table_html"].progress_apply(lambda x: self._get_table(x))
        self.df["question"] = self.df["question"].progress_apply(self._clean_text)
        self.df["answer"] = self.df["answer"].progress_apply(self._clean_text)
        return self.df
    