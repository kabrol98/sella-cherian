from __future__ import annotations

import re
from collections import OrderedDict
import datetime
import numpy as np

from openpyxl.cell import Cell

from components.cell_labeling.cell_compact import CellCompact, ContentType
from components.cell_labeling.variables import default_header_values, default_null_values

T = True
F = False

match_whitespaces_numeric = re.compile("\s+|\d+")
match_whitespaces_only = re.compile("^[\s]+$")

class CellExtended:

    def __init__(self, raw_cell: Cell):
        self.raw_cell = raw_cell
        self.compact_cell = self.initialize_compact_cell()
        self.state = self.initialize_state()
        self.apply_cell()

    def initialize_compact_cell(self):
        return CellCompact((self.raw_cell.row, self.raw_cell.column), self.raw_cell.value)

    def initialize_state(self):
        state = OrderedDict()
        state["is_blank"] = T
        state["bold_font"] = F
        state["below_blank"] = T
        state["has_merge_cell"] = F
        state["above_alpha"] = F
        state["left_align"] = T
        state["right_blank"] = T
        state["above_blank"] = T
        state["above_num"] = F
        state["above_alphanum"] = F # inconsistency here, should include is_alphanum
        state["right_align"] = F
        state["underline_font"] = F
        state["below_num"] = F
        state["left_alpha"] = F
        state["above_in_header"] = F
        state["left_num"] = F
        state["all_small"] = F
        state["is_alpha"] = F
        state["right_num"] = F
        state["text_in_header"] = F
        state["is_num"] = F
        return state

    def as_string(self):
        state = self.state
        state["is_num"] = F
        state["is_blank"] = F
        if self.is_alphanumeric():
            state["is_alpha"] = F
        elif self.is_alphabet():
            state["is_alpha"] = T
        state["text_in_header"] = self.is_text_in_header()
        state["all_small"] = self.is_all_small()
        self.compact_cell.content_type = ContentType.STRING

    def as_number(self):
        state = self.state
        state["is_num"] = T
        state["is_blank"] = F
        state["is_alpha"] = F
        state["all_small"] = F
        state["text_in_header"] = F
        self.compact_cell.content_type = ContentType.NUMERIC

    def as_blank(self):
        state = self.state
        state["is_num"] = F
        state["is_blank"] = T
        state["is_alpha"] = F
        state["text_in_header"] = F
        state["all_small"] = F
        state["left_align"] = F
        self.compact_cell.content_type = None # blank cell doesn't have any content

    def apply_cell(self):
        raw_cell = self.raw_cell
        if isinstance(raw_cell.value, str):
            self.as_string()
            if raw_cell.value in default_null_values or re.match(match_whitespaces_only, raw_cell.value):
                self.as_blank()
        elif isinstance(raw_cell.value, float):
            self.as_number()
        elif isinstance(raw_cell.value, int):
            self.as_number()
        elif isinstance(raw_cell.value, datetime.date):
            self.as_string()
        elif raw_cell.value is None:
            self.as_blank()
        else:
            print("Type not defined yet!")

        state = self.state
        if raw_cell.alignment.horizontal == 'center':
            state["left_align"] = F

        if raw_cell.alignment.horizontal == 'left':
            state["left_align"] = T

        if raw_cell.alignment.horizontal == 'right':
            state["right_align"] = T
            state["left_align"] = F

        if raw_cell.font.bold:
            state["bold_font"] = T

        if raw_cell.font.underline is not None:
            state["underline_font"] = T

    def apply_left_neighbor(self, left_cell: CellExtended):
        self.state["left_alpha"] = left_cell.state["is_alpha"]
        self.state["left_num"] = left_cell.state["is_num"]

    def apply_right_neighbor(self, right_cell: CellExtended):
        self.state["right_num"] = right_cell.state["is_num"]
        self.state["right_blank"] = right_cell.state["is_blank"]

    def apply_above_neighbor(self, top_cell: CellExtended):
        self.state["above_alpha"] = top_cell.state["is_alpha"]
        self.state["above_in_header"] = top_cell.state["text_in_header"]
        self.state["above_alphanum"] = top_cell.is_alphanumeric()
        self.state["above_num"] = top_cell.state["is_num"]
        self.state["above_blank"] = top_cell.state["is_blank"]

    def apply_below_neighbor(self, bottom_cell: CellExtended):
        self.state["below_num"] = bottom_cell.state["is_num"]
        self.state["below_blank"] = bottom_cell.state["is_blank"]

    def is_alphabet(self):
        return self.raw_cell.value.isalpha()

    def is_alphanumeric(self):
        if isinstance(self.raw_cell.value, str):
            return self.raw_cell.value.isalnum()
        elif isinstance(self.raw_cell.value, int):
            return True
        elif isinstance(self.raw_cell.value, float):
            return True
        else:
            return False

    def is_text_in_header(self):
        if isinstance(self.raw_cell.value, str):
            return len(re.sub(match_whitespaces_numeric, "", self.raw_cell.value)) > 0
        return False

    def is_all_small(self):
        smallCnt = 0
        value = self.raw_cell.value
        if isinstance(value, datetime.date):
            return True
        letterCnt = value.count(''.join(char for char in value if char.isalpha()))
        for i in range(len(value)):
            if value[i].isalpha and value[i].islower():
                smallCnt += 1
        if smallCnt == letterCnt and letterCnt > 0:
            return True
        return False

    def get_feature_vector(self):
        values = self.state.values()
        result = np.array([1 if x == T else 0 for x in values])
        return result.reshape(1, len(result))
