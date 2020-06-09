import xlrd
import xlwt
from os import listdir
from os.path import isfile, join
import random
import csv

files = [f for f in listdir("spreadsheet") if isfile(join("spreadsheet", f)) and "xls" in f]

record_file1 = open("spreadsheet_relationships.csv", "w")
record_file2 = open("column_relationships.csv", "w")

for file in files:
    book = xlrd.open_workbook("spreadsheet/" + file)

    sheets = book.sheet_names()

    newfile = xlwt.Workbook()
    newfile_name = 'data_' + str(random.random())[2:] + ".xls"

    for index, sh in enumerate(sheets):
        border = random.random()

        sheet = book.sheet_by_index(index)

        record_1 = file + ", " + sheet.name + ", " + newfile_name + ", " + sheet.name + ", 1\n"
        record_file1.write(record_1)

        newsheet = newfile.add_sheet(sheet.name)

        rows, cols = sheet.nrows, sheet.ncols
        new_rows = 0
        record = 0
        for row in range(rows):
            row_selection = random.random()
            if row_selection > border:
                continue
            new_rows += 1
            for col in range(cols):
                if record == 0:
                    record_2 = file + ", " + sheet.name + ", " + str(col) + ", " + newfile_name + ", " + sheet.name + ", " + str(col) + ", 1\n"
                    record_file2.write(record_2)
                newsheet.write(new_rows, col, sheet.cell(row, col).value)
            record += 1

    newfile.save("data/" + newfile_name)

record_file1.close()
record_file2.close()
