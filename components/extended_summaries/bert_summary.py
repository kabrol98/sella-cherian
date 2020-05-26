
import numpy as np
from bert_serving.client import BertClient
# from transformers import pipeline
from components.extract_column.column import Column
from components.cell_labeling.cell_compact import CellCompact, ContentType
from components.parse_files.metadata import ColumnMetaData

class BertSummary:
    
    def __init__ (self, column: Column):
        self.header_serialized = self.serialize(column.header_cells)
        self.data_serialized = self.serialize(column.content_cells)
        self.vector = self.get_bert_summary(self.header_serialized + self.data_serialized)
        
    def serialize(self, column_data: [CellCompact]):
        ret = ""
        for cell in column_data:
            ret += str(cell.content) + ','
        return ret
    
    def get_bert_summary(self,data):
        bc = BertClient()
        res = bc.encode([data])
        # nlp = pipeline('feature-extraction')
        # res = nlp(data)
        # print(res)
        return res[0]