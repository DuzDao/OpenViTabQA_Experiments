from bs4 import BeautifulSoup

def flatten_html_table(html_string):
    """
    Convert HTML table with merged cells (rowspan/colspan) to flattened table with header marking
    
    Args:
        html_string: HTML table string
        
    Returns:
        list: List of lists representing flattened table with header marks
    """
    soup = BeautifulSoup(html_string, 'html.parser')
    table = soup.find('table')
    
    # Get dimensions of the table
    rows = table.find_all('tr')
    max_cols = 0
    for row in rows:
        cols = row.find_all(['td', 'th'])
        total_cols = sum(int(col.get('colspan', 1)) for col in cols)
        max_cols = max(max_cols, total_cols)
    
    # Initialize grid with None values
    grid = [[None for _ in range(max_cols)] for _ in range(len(rows))]
    
    # Helper function to check if cell position is empty
    def is_empty(grid, row, col):
        return row < len(grid) and col < len(grid[0]) and grid[row][col] is None
    
    # Helper function to get text with spaces between nested elements
    def get_text_with_spaces(element):
        result = []
        for content in element.contents:
            if isinstance(content, str):
                result.append(content.strip())
            else:
                result.append(content.get_text(strip=True))
        return ' '.join(filter(None, result))
    
    # Fill the grid considering rowspan and colspan
    for i, row in enumerate(rows):
        col_idx = 0
        for cell in row.find_all(['td', 'th']):
            # Find next empty cell position
            while col_idx < max_cols and not is_empty(grid, i, col_idx):
                col_idx += 1
                
            if col_idx >= max_cols:
                break
                
            # Get cell properties
            rowspan = int(cell.get('rowspan', 1))
            colspan = int(cell.get('colspan', 1))
            is_header = cell.name == 'th'
            
            # Use new helper function to get text with spaces
            cell_value = get_text_with_spaces(cell)
            
            # Add header marking if needed
            if is_header:
                cell_value += " <header>"
            
            # Fill merged cells
            for r in range(rowspan):
                for c in range(colspan):
                    if i + r < len(grid) and col_idx + c < len(grid[0]):
                        grid[i + r][col_idx + c] = cell_value
            
            col_idx += colspan
    
    # Remove None values and empty rows
    flattened_table = []
    for row in grid:
        if any(cell is not None for cell in row):
            flattened_row = ['' if cell is None else cell for cell in row]
            flattened_table.append(flattened_row)
            
    return flattened_table

def format_table(flattened_table):
    """
    Format flattened table as string with aligned columns
    """
    if not flattened_table:
        return ""
        
    # Get maximum width for each column
    col_widths = []
    for col in range(len(flattened_table[0])):
        width = max(len(str(row[col])) for row in flattened_table)
        col_widths.append(width)
    
    # Format each row
    formatted_rows = []
    for row in flattened_table:
        formatted_cells = [str(cell).ljust(width) for cell, width in zip(row, col_widths)]
        formatted_rows.append(" | ".join(formatted_cells))
    
    return "\n".join(formatted_rows)
