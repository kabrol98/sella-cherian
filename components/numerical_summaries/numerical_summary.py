import numpy as np
from bert_serving.client import BertClient
from components.extract_column.column import Column
from components.cell_labeling.cell_compact import CellCompact, ContentType
from components.parse_files.metadata import ColumnMetaData

global NUM_BINS = 100
class NumericalSummary:
    
class Features:
    def __init__(self, column: Column, file_metadata: FileMetaData):
        self.data_type = column.type
        # -- only for column.type == ContentType.STRING
        self.value_range = 0
        self.max = 0
        self.min = 0
        self.mean = 0
        self.std = 0
        self.hist = []
        self.skewLeft = 0
        self.skewRight = 0
        self.vector = np.array([])
        # metadata:
        self.column_metadata = column.metadata
        self.file_metadata = file_metadata

        self.revise_features(column)

    def revise_features(column: Column):
        assert column.type == ContentType.NUMERIC
        raw_column = column.content_cells
        raw_column = filter(lambda x: x!= ContentType.NULL, raw_column)
        assert len(raw_column) > 0
        col = np.array(raw_column)
        self.min = np.min(col)
        self.max = np.max(col)
        self.mean = np.mean(col)
        self.median = np.median(col)
        self.std = np.sd(col)
        self.hist = np.hist(col, num_bins=NUM_BINS)
        self.skewRight = max(0, self.mean - self.median)
        self.skewLeft = max(0, - self.skewRight)
    
    def vectorize(self):
        self.vector = np.array([
            self.min,
            self.max,
            self.mean,
            self.median,
            self.std,
            self.skewRight,
            self.skewLeft,
            ] + self.hist)
        return self.vector