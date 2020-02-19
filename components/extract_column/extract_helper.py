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
                    if is_cell_empty(cell.content):
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
            ch_start = -1
            ch_end = -1
            ds = -1
            de_start = -1
            de_end = -1
            for i in range(len(col)):
                labeled_cell = col[i]
                tag = labeled_cell.tag
                if tag == CellTagType.CH:
                    if ch_start < 0:
                        ch_start = i
                    else:
                        if ch_end >= 0:
                            ch_end, ch_start, de_end, de_start, ds, column = ExtractHelper.method_name(
                                ch_end, ch_start,col,de_end, de_start,ds, i)
                            column.metadata = metadata
                            columns.append(column)
                elif tag == CellTagType.DS:
                    if ch_start >= 0 and ch_end < 0:
                        ch_end = i - 1
                    if ds < 0:
                        # there can be only one ds cell
                        ds = i
                elif tag == CellTagType.DC or tag == CellTagType.NDC:
                    if ch_start >= 0 and ch_end < 0:
                        ch_end = i - 1
                    if de_start >= 0:
                        de_end = i - 1
                        ch_end, ch_start, de_end, de_start, ds, column = ExtractHelper.method_name(
                            ch_end, ch_start,col,de_end, de_start, ds,i)
                        column.metadata = metadata
                        columns.append(column)
                    if tag == CellTagType.DC and ds < 0:
                        # there can be only one ds cell
                        # force dc to become ds
                        ds = i
                elif tag == CellTagType.DE:
                    if ch_start >= 0 and ch_end < 0:
                        ch_end = i - 1
                    if de_start < 0:
                        de_start = i
            if de_start >= 0:
                de_end = len(col) - 1
                _, _, _, _, _, column = ExtractHelper.method_name(
                    ch_end, ch_start, col,de_end, de_start, ds, de_end)
                column.metadata = metadata
                columns.append(column)

        return ExtractHelper.remove_empty_columns(columns)

    @staticmethod
    def method_name(ch_end, ch_start, col, de_end, de_start, ds, i):
        column = Column()
        for j in range(ch_start, ch_end + 1):
            column.header_cells.append(col[j].cell)
        for j in range(ch_end + 1, i):
            column.content_cells.append(col[j].cell)
        if ds >= 0:
            column.starting_cell = col[ds].cell
        if de_end >= 0:
            column.ending_cell = col[de_end].cell
        ch_start = -1
        ch_end = -1
        ds = -1
        de_start = -1
        de_end = -1
        return ch_end, ch_start, de_end, de_start, ds, column
