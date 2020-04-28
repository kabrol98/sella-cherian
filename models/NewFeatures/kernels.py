import pandas as pd
import numpy as np
import json
import re
from itertools import accumulate
from functools import reduce
from scipy.signal import convolve2d

TEST_FILENAME = '00sumdats.xlsx'
TEST_SHEETNAME = 'TSP'
KERNEL_LEVEL = 2
if KERNEL_LEVEL == 1:
    NUM_KERNELS = 8
else:
    NUM_KERNELS = 64

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
    
    df_read = dat_df[['file_name', 'is_blank']].copy(deep=True)
    
    names = df_read.file_name.map(lambda a : a.split('___'))
    df_read['filename'] = names.map(lambda a: a[0])
    df_read['sheetname'] = names.map(lambda a: a[1])
    coords = names.map(lambda a: (a[2].split(','))).map(lambda a: (int(a[0][1:]),int(a[1][:-1])))
    df_read['x'] = coords.map(lambda a: a[0])
    df_read['y'] = coords.map(lambda a: a[1])
    
    # Apply kernels
    kernel_set = load_kernels()
    
    for i in range(NUM_KERNELS):
        dat_df[f'K{i}'] = np.nan
    
    filenames = df_read.filename.unique()
    sheetnames = list(map(lambda f : [(f,name) for name in df_read[df_read.filename == f].sheetname.unique()], filenames))
    index = [y for x in sheetnames for y in x]
    # print(index)
    # exit()

    # use indexing to separate spreadsheets for optimization.
    begin = 0
    for i in index:
        end = begin
        while True:
            try:
                row = df_read.loc[end+1]
                cand = (row.filename, row.sheetname)
            except:
                print('EOF reached')
                break
            # print(cand, i)
            if cand[0] == i[0] and cand[1] == i[1]:
                end += 1
            else:
                # end reached
                generate_kernels(df_read, dat_df, begin, end, kernel_set)
                print(f'reached f{cand}')
                begin = end+1
                break
    print(dat_df.head())
    outpath = f'data_corpus/training_files/lv{KERNEL_LEVEL}_kernels.csv'
    dat_df.to_csv(outpath)
        
        
# generates is_none matrix, performs kernel convolutions
def generate_kernels(df_read, df_write, begin, end, kernel_set):
    file_df = df_read.iloc[begin:end, :]
    # print(file_df)
    # print(file_df)
    num_cols = file_df.x.max()+1
    num_rows = file_df.y.max()+1
    empty_mat = np.zeros((num_cols, num_rows))
    
    for i in range(begin, end):
        row = file_df.loc[i]
        (x,y) = (row.x,row.y)
        val = row.is_blank
        empty_mat[x,y] = 1 - val
    
    convolved_lv1 = [convolve2d(empty_mat, k, boundary='fill', fillvalue=0) for k in kernel_set]
    if KERNEL_LEVEL == 2:
        cartesian = cartesian_4d(convolved_lv1,kernel_set)
        convolved_lv2 = [convolve2d(c,k,boundary='fill',fillvalue=-1) for (c,k) in cartesian]
        convolved_result = convolved_lv2
    else:
        convolved_result = convolved_lv1
    
    for i in range(begin,end):
        (x,y) = (file_df.loc[i,'x'],file_df.loc[i,'y'])
        for k in range(NUM_KERNELS):
            df_write.loc[i,f'K{k}'] = convolved_result[k][x,y]
            
            
            

generate_all_kernels()
