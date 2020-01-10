from ..extract_column.column import Column
from ..extract_column.metadata import Metadata
import numpy as np

class Features:
    def __init__(self, column: Column, metadata: Metadata):
        #data_type: using 0 for data 1 for string
        self.revised_data, self.data_type = self.determine_type(column)
        self.value_range = 0
        self.max = 0
        self.min = 0
        self.mean = 0
        self.std = 0
        self.common_values0 = 0
        self.common_frequencies0 = 0
        self.common_values1 = 0
        self.common_frequencies1 = 0
        self.unique_num = 0
        self.null_num = 0
        self.mean = 0
        #metadata:
        self.metadata: metadata
        
        #revise most of the features based on self.revised_data
        self.revise_features()

    def determine_type(self, column):
        content_list = []
        return content_list, 0

    def revise_features(self):
