from components.extract_column.column import Column
from components.parse_files.metadata import ColumnMetaData
from components.parse_files.metadata import FileMetaData
from components.parse_files.parser import Parser

class SpreadSheet:
    
    def __init__(self, filepath: str):
        self.parser = Parser(filepath, str)
        self.columns = []
        # todo: modify parser.py save columns.