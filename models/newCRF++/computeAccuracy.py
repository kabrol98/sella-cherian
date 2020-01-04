import pandas

colnames = ["file_name","is_alpha", "text_in_header", "is_num","is_alphanum","is_blank", "is_nullDefault", "all_small","all_capital","starts_capital", "contain_colon", "contain_special","text_length","year_range", "has_merge_cell", "right_align","left_align","center_align", "italics_font", "underline_font","bold_font","left_alpha", "left_in_header", "left_num","left_alphanum","left_blank", "above_alpha", "above_in_header","above_num","above_alphanum", "above_blank", "below_alpha","below_in_header","below_num", "below_alphanum", "below_blank","right_alpha","right_in_header", "right_num", "right_alphanum","right_blank","label","prediction"]
result = pandas.read_csv("prediction0.7.data", names=colnames, header=None, delimiter=r"\s+")
result  = pandas.DataFrame(result)
label = result[["label"]]
label = label.values
prediction = result[["prediction"]]
prediction = prediction.values
print(label)
print(prediction)
match = 0
for i in range(len(label)):
    if (label[i] == prediction[i]):
        match += 1
accuracy = match / len(label)
print("CRF with 21 features has accuracy", accuracy)
