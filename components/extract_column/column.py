from collections import defaultdict
from typing import List
from components.cell_labeling.cell_compact import CellCompact, ContentType
from components.parse_files.metadata import ColumnMetaData


class Column:
    def __init__(self):
        self.header_cells: List[CellCompact] = []
        self.content_cells: List[CellCompact] = []
        self.starting_cell: CellCompact = None
        self.ending_cell: CellCompact = None
        self.type: ContentType = None
        self.metadata: ColumnMetaData = None

    def convert_cell_type(self, cell: CellCompact, max_type: ContentType) -> bool:
        if cell.content_type == max_type:
            return True
        if max_type == ContentType.STRING:
            if cell.content_type == ContentType.NUMERIC:
                try:
                    cell.content = str(cell.content)
                    cell.content_type = ContentType.STRING
                except Exception as e:
                    print(e)
                    return False
                return True
            elif cell.content_type == ContentType.NULL:
                return True
            elif cell.content_type is None:
                cell.content_type = ContentType.NULL
                return True
        elif max_type == ContentType.NUMERIC:
            if cell.content_type == ContentType.STRING:
                try:
                    cell.content = float(cell.content)
                    cell.content_type = ContentType.STRING
                except Exception as e:
                    print(e)
                    return False
                return True
            elif cell.content_type == ContentType.NULL:
                return True
            elif cell.content_type is None:
                cell.content_type = ContentType.NULL
                return True
        elif max_type == ContentType.NULL:
            cell.content_type = ContentType.NULL
            return True
        return False

    def set_type(self):
        if len(self.content_cells) == 0:
            self.type = None  # blank column does not have any content
            return
        all_counts = defaultdict(int)
        for cell in self.content_cells:
            all_counts[cell.content_type] += 1
        if len(all_counts) == 1:
            self.type = self.content_cells[0].content_type
            return
        isFail = False
        if all_counts[ContentType.STRING] > 0:
            self.type = ContentType.STRING
            for cell in self.content_cells:
                if not self.convert_cell_type(cell, ContentType.STRING):
                    print("Fail to convert to String: ", cell)
                    isFail = True
        elif all_counts[ContentType.NUMERIC] > 0:
            self.type = ContentType.NUMERIC
            for cell in self.content_cells:
                if not self.convert_cell_type(cell, ContentType.NUMERIC):
                    print("Fail to convert to number: ", cell)
                    isFail = True
        else:
            self.type = ContentType.NULL
        if isFail:
            print(self)
            print("\n")
        # max_type = None
        # max_count = 0
        # for key, value in all_counts.items():
        #     if value > max_count:
        #         max_count = value
        #         max_type = key
        # failed_cells = []
        # for cell in self.content_cells:
        #     if not self.convert_cell_type(cell, max_type):
        #         failed_cells.append(cell)
        # if len(failed_cells) / len(self.content_cells) >= 0.05:
        #     for cell in self.content_cells:
        #         if not self.convert_cell_type(cell, ContentType.STRING):
        #             raise Exception("Fail to convert all cells to string")
        # else:
        #     for cell in failed_cells:
        #         self.convert_cell_type(cell, ContentType.NULL)

    def __str__(self):
        nl = '\t\t\t\n'
        return f'''
{{
    type: {self.type},
    header: [
            {nl.join(map(str, self.header_cells))}
    ],
    contents: [
            {nl.join(map(str, self.content_cells))}
    ]
}}
'''

