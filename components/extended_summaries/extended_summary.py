import numpy as np
from bert_serving.client import BertClient
from components.extract_column.column import Column
from components.cell_labeling.cell_compact import CellCompact, ContentType
from components.parse_files.metadata import ColumnMetaData
from components.extended_summaries.bert_summary import BertSummary
from components.extended_summaries.numerical_summary import NumericalSummary

class ExtendedSummary:
    def __init__(self, column: Column):
        if column.type == ContentType.NUMERIC:
            self.summary = NumericalSummary(column)
        else:
            self.summary = BertSummary(column)
        self.column_raw = column
        self.type = column.type
        self.vector = self.summary.vector
        self.header = column.metadata.sheet_name + '.'.join(map(lambda x: str(x.content),column.header_cells))
        self.colname = ''.join(map(lambda x: str(x.content),column.header_cells))
        self.sheetname = column.metadata.sheet_name 
        self.filename = column.metadata.file_name
        self.id = self.filename[12:]+'.'+self.sheetname + '.' + self.colname
