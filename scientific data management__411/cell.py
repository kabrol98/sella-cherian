from enum import Enum
import numpy as np
import datetime
import Feature_builder


class cell:
    class Tag_values(Enum):
        NDC = 0
        CH = 1
        DS = 2
        DE = 3
        DC = 4

    def __init__(self):
        self.is_alpha = False
        self.text_in_header = False
        self.is_nullDefault = False
        self.is_num = False
        self.is_alphanum = False
        self.is_blank = True
        self.all_small = False
        self.all_capital = False
        self.starts_capital = False
        self.contain_colon = False
        self.contain_special = False
        self.text_length = False
        self.year_range = False
        self.has_merge_cell = False
        self.left_align = True
        self.center_align = False
        self.right_align = False
        self.italics_font = False
        self.underline_font = False
        self.bold_font = False
        self.left_alpha = False
        self.left_in_header = False
        self.left_num = False
        self.left_alphanum = False
        self.left_blank = True
        self.above_alpha = False
        self.above_in_header = False
        self.above_num = False
        self.above_alphanum = False
        self.above_blank = True
        self.below_alpha = False
        self.below_in_header = False
        self.below_num = False
        self.below_alphanum = False
        self.below_blank = True
        self.right_alpha = False
        self.right_in_header = False
        self.right_num = False
        self.right_alphanum = False
        self.right_blank = True
        self.tag = 0

    def get_address(self):
        return self.address

    def set_address(self, val):
        self.address = []
        for i in range(len(value)):
            self.address.append(value[i])

    def get_val(self):
        return self.var

    def set_value(self, val):
        self.value = val

    def get_type(self):
        return self.type

    def set_type(self, val):
        self.type = val

    def get_Is_alpha(self):
        return self.is_alpha

    def set_Is_alpha(self, val):
        self.is_alpha = val

    def get_Text_in_header(self):
        return self.text_in_header

    def set_Text_in_header(self, val):
        self.text_in_header = val

    def get_Is_num(self):
        return self.is_num

    def set_Is_num(self, val):
        self.is_num = val

    def get_Is_alphanum(self):
        return self.is_alphanum

    def set_Is_alphanum(self, val):
        self.is_alphanum = val

    def get_Is_blank(self):
        return self.is_blank

    def set_Is_blank(self, val):
        self.is_blank = val

    def get_Is_nullDefault(self):
        return self.is_nullDefault

    def set_Is_nullDefault(self, val):
        self.is_nullDefault = val

    def get_All_small(self):
        return self.all_small

    def set_All_small(self, val):
        self.all_small = val

    def get_All_captical(self):
        return self.all_capital

    def set_All_captical(self, val):
        self.all_capital = val

    def get_Starts_capital(self):
        return self.starts_capital

    def set_Starts_capital(self, val):
        self.starts_capital = val

    def get_Contain_colon(self):
        return self.contain_colon

    def set_Contain_colon(self, val):
        self.contain_colon = val

    def get_Contain_special(self):
        return self.contain_special

    def set_Contain_special(self, val):
        self.contain_special = val

    def get_Year_range(self):
        return self.year_range

    def get_Year_range(self, val):
        self.year_range = val

    def get_Has_merge_cell(self):
        return self.has_merge_cell

    def set_Has_merge_cell(self, val):
        self.has_merge_cell = val

    def get_Italics_font(self):
        return self.italics_font

    def set_Italics_font(self, val):
        self.italics_font = val

    def get_Right_align(self):
        return self.right_align

    def set_Right_align(self, val):
        self.right_align = val

    def get_Left_align(self):
        return self.left_align

    def set_Left_align(self, val):
        self.left_align = val

    def get_Center_align(self):
        return self.center_align

    def set_Center_align(self, val):
        self.center_align = val

    def get_Underline_font(self):
        return self.underline_font

    def set_Underline_font(self, val):
        self.underline_font = val

    def get_Bold_font(self):
        return self.bold_font

    def set_Bold_font(self, val):
        self.bold_font = val

    def get_Left_alpha(self):
        return self.left_alpha

    def set_Left_alpha(self, val):
        self.left_alpha = val

    def get_Left_in_header(self):
        return self.left_in_header

    def set_Left_in_header(self, val):
        self.left_in_header = val

    def get_Left_num(self):
        return self.left_num

    def set_Left_num(self, val):
        self.left_num = val

    def get_Left_alphanum(self):
        return self.left_alphanum

    def set_Left_alphanum(self, val):
        self.left_alphanum = val

    def get_Left_blank(self):
        return self.left_blank

    def set_Left_blank(self, val):
        self.left_blank = val

    def get_Above_alpha(self):
        return self.above_alpha

    def set_Above_alpha(self, val):
        self.above_alpha = val

    def get_Above_in_header(self):
        return self.above_in_header

    def set_Above_in_header(self, val):
        self.above_in_header = val

    def get_Above_num(self):
        return self.above_num

    def set_Above_num(self, val):
        self.above_num = val

    def get_Above_alphanum(self):
        return self.above_alphanum

    def set_Above_alphanum(self, val):
        self.above_alphanum = val

    def get_Above_blank(self):
        return self.above_blank

    def set_Above_blank(self, val):
        self.above_blank = val

    def get_Below_alpha(self):
        return self.below_alpha

    def set_Below_alpha(self, val):
        self.below_alpha = val

    def get_Below_in_header(self):
        return self.below_in_header

    def set_Below_in_header(self, val):
        self.below_in_header = val

    def get_Below_num(self):
        return self.below_num

    def set_Below_num(self, val):
        self.below_num = val

    def get_Below_alphanum(self):
        return self.below_alphanum

    def get_Below_alphanum(self, val):
        self.below_alphanum = val

    def get_Below_blank(self):
        return self.below_blank

    def set_Below_blank(self, val):
        self.below_blank = val

    def get_Right_alpha(self):
        return self.right_alpha

    def set_Right_alpha(self, val):
        self.right_alpha = val

    def get_Right_in_header(self):
        return self.right_in_header

    def set_Right_in_header(self, val):
        self.right_in_header = val

    def get_Right_num(self):
        return self.right_num

    def set_Right_num(self, val):
        self.right_num = val

    def get_Right_alphanum(self):
        return self.right_alphanum

    def set_Right_alphanum(self, val):
        self.right_alphanum = val

    def get_Right_blank(self):
        return self.right_blank

    def set_Right_blank(self, val):
        self.right_blank = val

    def get_Tag(self):
        return self.tag

    def set_Tag(self, val):
        self.tag = val

# if the cell type is a String (Alpha/Alphanum), the first 12 values that are unique to all Strings are set in this method
    def setString(self, value):
        fb = Feature_builder.Feature_builder()
        self.is_num = False
        self.is_blank = False
        if fb.Is_alphaNumeric_Feature(value):
            is_alpha = False
            is_alphanum = True
        elif fb.Is_alphabet_Feature(value):
            is_alpha = True
            is_alphanum = False
        self.text_in_header = fb.Is_textInHeader_Feature(value)

        self.is_nullDefault = fb.Is_nullDefault_Feature(value)

        self.all_small = fb.Is_allSmall_Feature(value)

        self.all_capital = fb.Is_allCapital_Feature(value)

        self.starts_capital = fb.Is_startCapital_Feature(value)

        self.contain_colon = fb.contains_Colon_Feature(value)

        self.contain_special = fb.contains_Special_Feature(value)

        self.text_length = fb.Is_textLength_Feature(value)

        if (self.text_length):
            self.Text_in_header = False
        self.year_range = fb.Is_inYearRange_Feature(value)


# If the cell type is a Num, the first 12 values that are unique to all Num are set in this method

    def setNum(self, value):
        fb = Feature_builder.Feature_builder()

        self.is_num = True

        self.is_blank = False

        self.is_alpha = False

        self.is_alphanum = False

        self.all_capital = False

        self.all_small = False

        self.starts_capital = False

        self.contain_colon = False

        self.contain_special = False

        self.text_length = False

        self.text_in_header = fb.Is_textInHeader_Feature(value)

        self.is_nullDefault = fb.Is_nullDefault_Feature(value)

        self.year_range = fb.Is_inYearRange_Feature(value)

    def setBlank(self):

        self.is_num = False

        self.is_blank = True

        self.is_alpha = False

        self.is_alphanum = False

        self.text_in_header = False

        self.all_capital = False

        self.all_small = False

        self.starts_capital = False

        self.contain_colon = False

        self.contain_special = False

        self.text_length = False

        self.year_range = False

        self.left_align = False

    def setAttributes(self, cell_obj):
        # assign type of the cell value
        # val = self.value
        # if val != None:
        #     tp = type(val)
        # else:
        #     tp = "Blank"
        # Check if Alpha, Alphanum, Num or Blank and set the unique 12 features using correspoding method
        val = cell_obj.value
        if isinstance(cell_obj.value, str):
            self.setString(val)
        if isinstance(cell_obj.value, float):
            self.setNum(val)
        if isinstance(cell_obj.value, int):
            self.setNum(val)
        if isinstance(cell_obj.value, datetime.date):
            self.setString(val)
        if cell_obj.value == None:
            self.setBlank()
        else:
            print("Type not defined yet!")

        if cell_obj.alignment.horizontal == 'center':
            Center_align = True
            Left_align = False

        if cell_obj.alignment.horizontal == 'left':
            Left_align = True

        if cell_obj.alignment.horizontal == 'right':
            Right_align = True
            Left_align = False

        if cell_obj.font == "Italic":
            Italics_font = True

        if (cell_obj.font.bold):
            Bold_font = True

        if (cell_obj.font.underline != None):
            underline_font_font = True






    # check the left neighbor features
    def setLeftNeighbour(self, neighbourCell):

        self.left_alpha = neighbourCell.is_alpha
        self.left_in_header = neighbourCell.text_in_header
        self.left_alphanum = neighbourCell.is_alphanum
        self.left_num = neighbourCell.is_num
        self.left_blank = neighbourCell.is_blank

    # above neighbor features

    def setAboveNeighbor(self, neighbourCell):

        self.above_alpha = neighbourCell.is_alpha
        self.above_in_header = neighbourCell.text_in_header
        self.above_alphanum = neighbourCell.is_alphanum
        self.above_num = neighbourCell.is_num
        self.above_blank = neighbourCell.is_blank

    def setBelowNeighbour(self, neighbourCell):

        self.below_alpha = neighbourCell.is_alpha
        self.below_in_header = neighbourCell.text_in_header
        self.below_alphanum = neighbourCell.is_alphanum
        self.below_num = neighbourCell.is_num
        self.below_blank = neighbourCell.is_blank


    # set right neighbour features

    def setRightNeighbour(self, neighbourCell):

        self.right_alpha = neighbourCell.is_alpha
        self.right_in_header = neighbourCell.text_in_header
        self.right_alphanum = neighbourCell.is_alphanum
        self.right_num = neighbourCell.is_num
        self.right_blank = neighbourCell.is_blank

