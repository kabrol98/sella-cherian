from collections import Counter

import numpy as np
from components.column_summaries.bloomfilter import BloomFilter

from components.cell_labeling.cell_compact import ContentType
from components.extract_column.column import Column
from components.parse_files.metadata import FileMetaData


class Features:
    def __init__(self, column: Column, file_metadata: FileMetaData = None):
        self.data_type = column.type
        # -- only for column.type == ContentType.NUMERIC
        self.value_range = 0
        self.max = 0
        self.min = 0
        self.mean = 0
        self.std = 0
        #
        # -- only for column.type == ContentType.STRING
        self.mask = 0
        #
        self.common_values0 = 0
        self.common_frequencies0 = 0
        self.common_values1 = 0
        self.common_frequencies1 = 0
        self.unique_num = 0
        self.null_num = 0
        # metadata:
        self.column_metadata = column.metadata
        self.file_metadata = file_metadata
        self.revise_features(column)
        self.vector = None
        self.header = column.metadata.sheet_name + '.'.join(map(lambda x: str(x.content),column.header_cells))
        self.generate_vector()
        
        

    def revise_features(self, column):
        if column.type is None:
            return
        if column.type == ContentType.NULL:
            null_num = len(column)
            return
        count_null = 0
        none_null_cell_values = []
        for compact_cell in column.content_cells:
            if compact_cell.content_type == ContentType.NULL:
                count_null += 1
            else:
                none_null_cell_values.append(compact_cell.content)
        self.null_num = count_null
        if len(none_null_cell_values) == 0:
            return
        none_null_cell_values = np.array(none_null_cell_values)
        if column.type == ContentType.NUMERIC:
            self.max = np.max(none_null_cell_values)
            self.min = np.min(none_null_cell_values)
            self.value_range = self.max - self.min
            self.mean = np.mean(none_null_cell_values)
            self.std = np.std(none_null_cell_values)
        else:
            print(column.type)
            bloomfilter = BloomFilter(none_null_cell_values, 0.01)
            self.mask = bloomfilter.intMask()
        self.unique_num = len(np.unique(np.array(none_null_cell_values)))
        try:
            frequent_two = Counter(none_null_cell_values).most_common(2)
            self.common_values0 = frequent_two[0][0]
            self.common_frequencies0 = frequent_two[0][1]
            self.common_values1 = frequent_two[1][0]
            self.common_frequencies1 = frequent_two[1][1]
        except:
            frequent_one = Counter(none_null_cell_values).most_common(1)
            self.common_values0 = frequent_one[0][0]
            self.common_frequencies0 = frequent_one[0][1]
    
    def generate_vector(self):
        if self.data_type == ContentType.NUMERIC:
            self.vector = np.array([
                self.value_range,
                self.max,
                self.min,
                self.mean,
                self.std
            ])
        else:
            self.vector = np.array([
                self.mask,
                self.common_values0,
                self.common_frequencies0,
                self.common_values1,
                self.common_frequencies1,
                self.unique_num,
                self.null_num
            ])