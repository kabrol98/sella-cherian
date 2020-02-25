import math
from collections import Counter

from components.cell_labeling.cell_compact import CellTagType, ContentType
from components.cell_labeling.cell_extended import is_cell_empty
from components.extract_column.column import Column
from components.parse_files.metadata import ColumnMetaData


class ExtractHelper:
    valid_surrounding_tags = {
        CellTagType.CH: {CellTagType.CH, CellTagType.NDC},
        CellTagType.DS: {CellTagType.DS},
        CellTagType.DE: {CellTagType.DE, CellTagType.NDC},
        CellTagType.NDC: {CellTagType.CH, CellTagType.NDC}
    }

    @staticmethod
    def iterator(data):
        for col_idx, col in enumerate(data):
            for row_idx, labeled_cell in enumerate(col):
                yield col_idx, row_idx, labeled_cell

    @staticmethod
    def row_iterator(data, col_idx, start_row_idx=0):
        for row_idx in range(start_row_idx, len(data[col_idx])):
            yield data[col_idx][row_idx]

    @staticmethod
    def remove_empty_columns(columns):
        def remove_col(col):
            # includes col with only headers
            return col.type == ContentType.NULL or col.type is None

        return filter(remove_col, columns)

    @staticmethod
    def extract_columns(data, metadata: ColumnMetaData):
        # this method will exclude columns with single DC cells
        columns = []
        for j in range(len(data)):
            col = data[j]
            last_start_idx = None
            last_tag = None
            non_empty_index = None
            for i in range(len(col)):
                labeled_cell = col[i]
                tag = labeled_cell.tag
                if tag != CellTagType.NDC:
                    non_empty_index = i
                if tag == CellTagType.CH:
                    if last_start_idx is None:
                        last_start_idx = i
                    if last_start_idx != i:
                        # ds cannot be at the front of ch
                        if last_tag == CellTagType.CH:
                            pass
                        else:
                            columns.append(ExtractHelper.extract(col, last_start_idx, i, metadata))
                            last_start_idx = i

                elif tag == CellTagType.DS:
                    if last_start_idx is None:
                        last_start_idx = i
                    if last_start_idx != i:
                        if last_tag == CellTagType.CH:
                            pass
                        elif last_tag == CellTagType.DS:
                            pass
                        else:
                            columns.append(ExtractHelper.extract(col, last_start_idx, i, metadata))
                            last_start_idx = i
                last_tag = tag

            if non_empty_index is not None and non_empty_index > last_start_idx:
                columns.append(ExtractHelper.extract(col, last_start_idx, non_empty_index+1, metadata))
        return ExtractHelper.remove_empty_columns(columns)

    @staticmethod
    def extract(col, start_idx, exclude_end_index, metadata):
        column = Column()
        column.metadata = metadata
        for i in range(start_idx, exclude_end_index):
            cell_labeled = col[i]
            cell = cell_labeled.cell
            if cell_labeled.tag == CellTagType.CH and cell.content_type != ContentType.NULL and cell.content_type is not None:
                column.header_cells.append(cell)
                continue
            if cell.content_type != ContentType.NULL and cell.content_type is not None:
                column.content_cells.append(cell)
            if cell_labeled.tag == CellTagType.DS and column.starting_cell is None:
                column.starting_cell = cell
            if cell_labeled.tag == CellTagType.DE:
                column.ending_cell = cell
        if column.starting_cell is None and len(column.content_cells) != 0:
            column.starting_cell = column.content_cells[0]
        if column.ending_cell is None and len(column.content_cells) != 0:
            column.ending_cell = column.content_cells[-1]
        return column
