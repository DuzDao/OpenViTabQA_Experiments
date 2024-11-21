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
        if type(text) != str:
            text = "nul"
        text = text.lower()
        text = text.replace("\n", "")
        text = text.replace("\t", "")
        text = text.replace("\r", "")
        return text
    
    def _unmerge_table(self, table_html):
        """
        Get all table data.
        Each cell of row is splited by <eoc>, which is standed for end of cell
        Each row of table is splited by <eor>, which is standed for end of row
        Header cell will be alongside with "[HEADER]"
        ----------
        table_html: str     | [can-be-soup]
        """
        try:
            soup = BeautifulSoup(table_html, "html.parser")
            
            # Find all rows
            rows = soup.find_all("tr")
            
            # Prepare a 2D list to track cell spans and values
            max_cols = max(len(row.find_all(['th', 'td'])) * 3 for row in rows)
            table_matrix = [[{"value": "nul", "colspan": 1, "rowspan": 1, "is_header": False} for _ in range(max_cols)] for _ in range(len(rows))]
            
            # First pass: populate matrix with original cell values and spans
            for row_idx, row in enumerate(rows):
                col_idx = 0
                for cell in row.find_all(['th', 'td']):
                    # Determine if it's a header
                    is_header = cell.name == 'th' or 'header' in cell.get('class', [])
                    
                    # Get cell value and clean it
                    cell_value = self._clean_text(cell.get_text(strip=True)) or "nul"
                    
                    # Get colspan and rowspan
                    colspan = int(cell.get('colspan', 1))
                    rowspan = int(cell.get('rowspan', 1))
                    
                    # Find first available column
                    while col_idx < len(table_matrix[row_idx]) and table_matrix[row_idx][col_idx]['value'] != "nul":
                        col_idx += 1
                    
                    # Ensure we don't go out of bounds
                    if col_idx >= len(table_matrix[row_idx]):
                        break
                    
                    # Mark the main cell
                    table_matrix[row_idx][col_idx] = {
                        "value": cell_value,
                        "colspan": colspan,
                        "rowspan": rowspan,
                        "is_header": is_header
                    }
                    
                    # Mark spanned cells
                    for r in range(rowspan):
                        for c in range(colspan):
                            if r == 0 and c == 0:
                                continue
                            target_row = row_idx + r
                            target_col = col_idx + c
                            
                            if (target_row < len(table_matrix) and 
                                target_col < len(table_matrix[target_row])):
                                table_matrix[target_row][target_col] = {
                                    "value": cell_value,
                                    "colspan": 1,
                                    "rowspan": 1,
                                    "is_header": is_header
                                }
                    
                    col_idx += colspan
            
            # Convert to final string format with markers
            final_table = ""
            for row in table_matrix:
                for cell in row:
                    # Add header signal
                    cell_text = f"{'[HEADER] ' if cell['is_header'] else ''}{cell['value']}"
                    final_table += cell_text + " <eoc> "
                final_table += "<eor> "
            
            return final_table
        except Exception as e:
            print(f"Error processing table: {e}")
            return "nul <eoc> <eor>"
    

    def do_preprocess(self):
        tqdm.pandas(desc="Doing preprocessing...")
        self.df["table"] = self.df["table_html"].progress_apply(self._unmerge_table)
        self.df["question"] = self.df["question"].progress_apply(self._clean_text)
        self.df["answer"] = self.df["answer"].progress_apply(self._clean_text)
        return self.df
    