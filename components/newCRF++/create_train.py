import pandas

colnames = ["file_name","is_alpha", "text_in_header", "is_num","is_alphanum","is_blank", "is_nullDefault", "all_small","all_capital","starts_capital", "contain_colon", "contain_special","text_length","year_range", "has_merge_cell", "right_align","left_align","center_align", "italics_font", "underline_font","bold_font","left_alpha", "left_in_header", "left_num","left_alphanum","left_blank", "above_alpha", "above_in_header","above_num","above_alphanum", "above_blank", "below_alpha","below_in_header","below_num", "below_alphanum", "below_blank","right_alpha","right_in_header", "right_num", "right_alphanum","right_blank","label"]
data1 = pandas.read_csv("newTraining.data", names=colnames, header=None, delimiter=r"\s+")
data2 = pandas.read_csv("newTraining_V2.data", names=colnames, header=None, delimiter=r"\s+")
df1 = pandas.DataFrame(data1)
df2 = pandas.DataFrame(data2)
df = pandas.concat([df1, df2], ignore_index=True, sort=False)
train = df.sample(frac=0.7)
train.to_csv("train0.7.data",encoding='utf-8', index=False, sep='\t')
test = df.sample(frac=0.3)
test.to_csv("test0.3.data",encoding='utf-8', index=False, sep='\t')
