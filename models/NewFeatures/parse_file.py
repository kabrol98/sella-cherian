import openpyxl as pxl
import pandas as pd
import numpy as np
import json
from scipy.signal import convolve2d

TEST_FILENAME = 'data_corpus/00sumdat_copy.xlsx'

def load_kernels():
    with open('models/NewFeatures/kernels.json') as f:
        kernels_arr = json.load(f)
        kernels_arr = [np.array(k) for k in kernels_arr]
    return kernels_arr

def cartesian_4d(a,b):
    # a is a list of lv1 convolutions
    # b is a list of kernels
    res = []
    for i in a:
        for j in b:
            res.append((i,j))
    return res
def parse_file(filename):
    if not filename:
        filename = TEST_FILENAME
    # load workbook, extract worksheets
    workbook = pxl.load_workbook(TEST_FILENAME)
    sheet_names = workbook.sheetnames
    worksheets = [ workbook[s] for s in sheet_names]
    # testing one worksheet
    ws = worksheets[0]
    cell_df = pd.DataFrame(ws)
    cell_matrix = cell_df.values
    isnone_matrix = np.vectorize(lambda x : 0 if x.value is None else 1)(cell_matrix)
    
    kernel_set = load_kernels()
    kernel = kernel_set
    
    convolved_lv1 = [convolve2d(isnone_matrix, k, boundary='fill', fillvalue=-1) for k in kernel_set]
    # a = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
    # b = np.array([[[0,1],[2,3]],[[4,5],[6,7]]])
    cartesian = cartesian_4d(convolved_lv1,kernel_set)
    convolved_lv2 = [convolve2d(c,k,boundary='fill',fillvalue=-1) for (c,k) in cartesian]
    
    # print(convolved_lv1[0])
    print(len(convolved_lv2))
    # print(convolved_lv2[0])
    print(isnone_matrix)
    
parse_file(None)
