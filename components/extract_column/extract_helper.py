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
    def compare_surrounding_tags(
            cell: CellTagType,
            left: CellTagType,
            second_to_left: CellTagType,
            right: CellTagType,
            second_to_right: CellTagType
    ):
        if cell not in ExtractHelper.valid_surrounding_tags:
            raise RuntimeError("Specificed cell type not recognized")
        surrounding_tags = ExtractHelper.valid_surrounding_tags[cell]
        if (left in surrounding_tags and second_to_left in surrounding_tags) or (
                right in surrounding_tags and second_to_right in surrounding_tags):
            return cell
        else:
            counter = Counter([cell, left, second_to_left, right, second_to_right])
            return counter.most_common(1)[0][0]

    @staticmethod
    def check_left_and_right(data, col_idx, row_idx):
        cell = data[col_idx][row_idx].tag
        left = data[col_idx - 1][row_idx].tag if col_idx - 1 >= 0 else None
        right = data[col_idx + 1][row_idx].tag if col_idx + 1 < len(data) else None
        second_to_left = data[col_idx - 2][row_idx].tag if col_idx - 2 >= 0 else None
        second_to_right = data[col_idx + 2][row_idx].tag if col_idx + 2 < len(data) else None

        if left is None:
            if right is not None:
                left = right
            else:
                left = cell
        if right is None:
            if left is not None:
                right = left
            else:
                right = cell
        if second_to_left is None:
            second_to_left = left
        if second_to_right is None:
            second_to_right = right
        if cell == left and cell == right:
            return cell
        return ExtractHelper.compare_surrounding_tags(cell=cell, left=left, right=right, second_to_left=second_to_left,
                                                      second_to_right=second_to_right)

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
            result = True
            for cell in col.content_cells:
                if cell.content_type == ContentType.STRING:
                    if is_cell_empty(str(cell.content)):
                        result = False
                    else:
                        result = True
                        break
                elif cell.content_type == ContentType.NUMERIC:
                    result = True
                    break
            return result

        return filter(remove_col, columns)

    @staticmethod
    def extract_columns(data, metadata: ColumnMetaData):
        columns = []
        for j in range(len(data)):
            col = data[j]
            last_start_idx = None
            last_tag_idx = None
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
                    if last_tag == CellTagType.CH:
                        if last_tag_idx == i - 1:
                            pass
                        else:
                            columns.append(ExtractHelper.extract(col, last_start_idx, i, metadata))
                            last_start_idx = i
                    elif last_tag == CellTagType.DS:
                        if last_tag_idx == i - 1:
                            pass
                        else:
                            columns.append(ExtractHelper.extract(col, last_start_idx, i, metadata))
                            last_start_idx = i

                elif tag == CellTagType.DS:
                    if last_start_idx is None:
                        last_start_idx = i
                    if last_tag == CellTagType.CH:
                        # ch should not come after ds
                        columns.append(ExtractHelper.extract(col, last_start_idx, i, metadata))
                        last_start_idx = i
                    elif last_tag == CellTagType.DS:
                        if last_tag_idx == i - 1:
                            pass
                        else:
                            columns.append(ExtractHelper.extract(col, last_start_idx, i, metadata))
                            last_start_idx = i
                last_tag_idx = i
                last_tag = tag

            if non_empty_index is not None and non_empty_index > last_start_idx:
                columns.append(ExtractHelper.extract(col, last_start_idx, non_empty_index, metadata))
        return ExtractHelper.remove_empty_columns(columns)

    @staticmethod
    def extract(col, start_idx, exclude_end_index, metadata):
        column = Column()
        column.metadata = metadata
        ch_found = False
        for i in range(start_idx, exclude_end_index):
            cell_labeled = col[i]
            cell = cell_labeled.cell
            if cell_labeled.tag == CellTagType.CH and not ch_found:
                ch_found = True
                column.header_cells.append(cell)
                continue
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
