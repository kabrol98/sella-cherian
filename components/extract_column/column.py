from typing import List

from ..cell_labeling.cell_compact import CellCompact

class Column:
    def __init__(self):
        self.header_cells: List[CellCompact] = []
        self.content_cells: List[CellCompact] = []
        self.starting_cell: CellCompact = None
        self.ending_cell: CellCompact = None