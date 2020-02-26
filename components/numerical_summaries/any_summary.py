import numpy as np
from bert_serving.client import BertClient
from components.extract_column.column import Column
from components.cell_labeling.cell_compact import CellCompact, ContentType
from components.parse_files.metadata import ColumnMetaData
from components.bert_summaries.bert_summary import BertSummary
from components.numerical_summaries.numerical_summary import NumericalSummary

class AnySummary:
    def __init__(self, column: Column):
        if column.type == ContentType.NUMERIC:
            self.summary = NumericalSummary(column)
        else:
            self.summary = BertSummary(column)
        self.type = column.type
        self.vectorize = self.summary.vectorize
        self.string = column.metadata.sheet_name + '.'.join(map(lambda x: str(x.content),column.header_cells))

    def serialize(self):
        return self.string