from collections import defaultdict
from typing import List

from components.cell_labeling.cell_compact import CellCompact, ContentType


class Column:
    def __init__(self):
        self.header_cells: List[CellCompact] = []
        self.content_cells: List[CellCompact] = []
        self.starting_cell: CellCompact = None
        self.ending_cell: CellCompact = None
        self.type: ContentType = None

    def set_type(self):
        if len(self.content_cells) == 0:
            self.type = None # blank column does not have any content
            return
        all_counts = defaultdict(int)
        for cell in self.content_cells:
            all_counts[cell.content_type] += 1
        if len(all_counts) == 1:
            self.type = self.content_cells[0].content_type
            return
        max_type = None
        max_count = 0
        for key, value in all_counts.items():
            if value > max_count:
                max_count = value
                max_type = key

