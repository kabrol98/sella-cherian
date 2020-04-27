import pandas as pd
import numpy as np
import json
import re
from itertools import accumulate
from functools import reduce
from scipy.signal import convolve2d

TEST_FILENAME = '00sumdats.xlsx'
TEST_SHEETNAME = 'TSP'

FEATURE_NAMES = [
    "file_name","is_alpha", "text_in_header", "is_num","is_alphanum",
    "is_blank", "is_nullDefault", "all_small","all_capital","starts_capital", 
    "contain_colon", "contain_special","text_length","year_range", "has_merge_cell", 
    "right_align","left_align","center_align", "italics_font", "underline_font","bold_font",
    "left_alpha", "left_in_header", "left_num","left_alphanum","left_blank", "above_alpha", 
    "above_in_header","above_num","above_alphanum", "above_blank", "below_alpha","below_in_header",
    "below_num", "below_alphanum", "below_blank","right_alpha","right_in_header", "right_num", 
    "right_alphanum","right_blank","label"]

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

def generate_all_kernels():
    
    # load .data file, retrieve labels.
    dat_df = pd.read_csv('models/NeuralNetwork/NewTraining.data', header=None, sep="\s+")
    dat_df.columns = FEATURE_NAMES
    names = dat_df.file_name.map(lambda a : a.split('___'))
    dat_df['filename'] = names.map(lambda a: a[0])
    dat_df['sheetname'] = names.map(lambda a: a[1])
    coords = names.map(lambda a: (a[2].split(','))).map(lambda a: (int(a[0][1:]),int(a[1][:-1])))
    dat_df['x'] = coords.map(lambda a: a[0])
    dat_df['y'] = coords.map(lambda a: a[1])
    
    for i in range(64):
        dat_df[f'K{i}'] = np.nan
    
    filenames = dat_df.filename.unique()
    sheetnames = list(map(lambda f : [(f,name) for name in dat_df[dat_df.filename == f].sheetname.unique()], filenames))
    index = [y for x in sheetnames for y in x]
    # print(index)

    # use indexing to separate spreadsheets for optimization.
    begin = 0
    for i in index:
        end = begin
        while True:
            try:
                row = dat_df.loc[end+1]
                cand = (row.filename, row.sheetname)
            except:
                print('EOF reached')
                break
            # print(cand, i)
            if cand[0] == i[0] and cand[1] == i[1]:
                end += 1
            else:
                # end reached
                generate_kernels(dat_df, begin, end)
                print(f'reached f{cand}')
                begin = end+1
                break
    dat_df.to_csv('data_corpus/training_files/lv2_kernels.csv')
        
        
# generates is_none matrix, performs kernel convolutions
def generate_kernels(df, begin, end):
    file_df =  df.iloc[begin:end, :]
    # print(file_df)
    num_cols = file_df.x.max()+1
    num_rows = file_df.y.max()+1
    empty_mat = np.zeros((num_cols, num_rows))
    
    for i in range(begin, end+1):
        row = df.loc[i]
        (x,y) = (row.x,row.y)
        val = row.is_blank
        empty_mat[x,y] = 1 - val
    
    
    # Apply kernels
    kernel_set = load_kernels()
    
    convolved_lv1 = [convolve2d(empty_mat, k, boundary='fill', fillvalue=0) for k in kernel_set]
    cartesian = cartesian_4d(convolved_lv1,kernel_set)
    convolved_lv2 = [convolve2d(c,k,boundary='fill',fillvalue=-1) for (c,k) in cartesian]
    
    for i in range(begin,end+1):
        (x,y) = (df.loc[i,'x'],df.loc[i,'y'])
        for k in range(64):
            df.loc[i,f'K{k}'] = convolved_lv2[k][x,y]
            
            
            

generate_all_kernels()
