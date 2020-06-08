import pyexcel as p
from os import listdir
from os.path import isfile, join

files = [f for f in listdir("spreadsheet") if isfile(join("spreadsheet", f)) and "xls" in f]

for file in files:
    filename = file.split(".")[0]
    p.save_book_as(file_name = "spreadsheet/" + filename + ".xls", dest_file_name = "spreadsheet/" + filename + ".xlsx")
