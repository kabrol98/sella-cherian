import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np

FEATURE_NAMES = [
    "file_name","is_alpha", "text_in_header", "is_num","is_alphanum",
    "is_blank", "is_nullDefault", "all_small","all_capital","starts_capital", 
    "contain_colon", "contain_special","text_length","year_range", "has_merge_cell", 
    "right_align","left_align","center_align", "italics_font", "underline_font","bold_font",
    "left_alpha", "left_in_header", "left_num","left_alphanum","left_blank", "above_alpha", 
    "above_in_header","above_num","above_alphanum", "above_blank", "below_alpha","below_in_header",
    "below_num", "below_alphanum", "below_blank","right_alpha","right_in_header", "right_num", 
    "right_alphanum","right_blank","label"]

no_kernels = pd.read_csv('data_corpus/training_files/NewTraining.data', header=None, sep="\s+")
lv1_kernels = pd.read_csv('data_corpus/training_files/lv1_kernels.csv', header=0)
lv2_kernels = pd.read_csv('data_corpus/training_files/lv2_kernels.csv', header=0)
no_kernels.columns = FEATURE_NAMES
print(no_kernels.head())
print(lv1_kernels.head())
print(lv2_kernels.head())

