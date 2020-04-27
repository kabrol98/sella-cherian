import openpyxl as pxl
import numpy as np
import json
import re
from scipy.signal import convolve2d

TEST_FILENAME = 'CMOP.xlsx'

def parse_file(filename):
    if not filename:
        filename = TEST_FILENAME
    filepath = f'data_corpus/{filename}'
    # load workbook, extract worksheets
    workbook = pxl.load_workbook(filepath)
    sheet_names = workbook.sheetnames
    worksheets = [ workbook[s] for s in sheet_names]
    # testing one worksheet
    ws = worksheets[0]
    cell_df = pd.DataFrame(ws)
    cell_matrix = cell_df.values
    isnone_matrix = np.vectorize(lambda x : 0 if x.value is None else 1)(cell_matrix)
    
    
parse_file(None)
