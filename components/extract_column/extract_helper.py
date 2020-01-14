from collections import Counter
from typing import cast

from components.cell_labeling.cell_compact import CellTagType, CellCompact
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
    def extract_columns(data, metadata: ColumnMetaData):
        columns = []
        prev_ch = set()
        prev_ds = set()
        current_column = None
        for col_idx, row_idx, labeled_cell in ExtractHelper.iterator(data):
            current_cell = cast(CellCompact, labeled_cell.cell)
            current_tag = labeled_cell.tag
            if current_tag == CellTagType.CH:
                if current_cell.location in prev_ch:
                    continue
                # removed detection for repeated CH cells
                # removed detection for un-named/empty CH cells
                if current_column is None:
                    current_column = Column()
                    current_column.metadata = metadata
                    current_column.header_cells.append(current_cell)
                dsSet = False
                deSet = False
                dcSet = False
                tryingToSetCHafterDS_DC_DE = False
                tryingToSetDSafterDS_DC_DE = False
                repeatingColumn = False
                ndcReached = False
                max_row_len = len(data[col_idx])
                row_iterator = ExtractHelper.row_iterator(data, col_idx, row_idx)
                j = row_idx
                while j < max_row_len and not deSet:
                    current_row_labeled_cell = next(row_iterator)
                    current_row_cell = cast(CellCompact, current_row_labeled_cell.cell)
                    current_row_tag = current_row_labeled_cell.tag
                    if current_row_tag == CellTagType.CH:
                        if dsSet or dcSet or deSet:
                            actual_tag = ExtractHelper.check_left_and_right(data, col_idx, row_idx)
                            if current_row_tag == actual_tag:
                                tryingToSetCHafterDS_DC_DE = True
                                # start new column
                            else:
                                data[col_idx][row_idx].tag = actual_tag
                                continue
                        else:
                            if current_row_cell.location in prev_ch:
                                repeatingColumn = True
                            else:
                                prev_ch.add(current_row_cell.location)
                                current_column.header_cells.append(current_row_cell)
                    elif current_row_tag == CellTagType.DS:
                        if dsSet or dcSet or deSet:
                            actual_tag = ExtractHelper.check_left_and_right(data, col_idx, row_idx)
                            if current_row_tag == actual_tag:
                                tryingToSetDSafterDS_DC_DE = True
                            else:
                                data[col_idx][row_idx].tag = actual_tag
                                continue
                        else:
                            current_column.content_cells.append(current_row_cell)
                            current_column.starting_cell = current_row_cell
                            dsSet = True
                    elif current_row_tag == CellTagType.DC:
                        current_column.content_cells.append(current_row_cell)
                        dcSet = True
                    elif current_row_tag == CellTagType.DE:
                        actual_tag = ExtractHelper.check_left_and_right(data, col_idx, row_idx)
                        if current_row_tag == actual_tag:
                            current_column.content_cells.append(current_row_cell)
                            current_column.ending_cell = current_row_cell
                            deSet = True
                        else:
                            data[col_idx][row_idx].tag = actual_tag
                            continue
                    elif current_row_tag == CellTagType.NDC:
                        actual_tag = ExtractHelper.check_left_and_right(data, col_idx, row_idx)
                        if current_row_tag == actual_tag:
                            current_column.content_cells.append(current_row_cell)
                            ndcReached = True
                        else:
                            data[col_idx][row_idx].tag = actual_tag
                            continue
                    else:
                        raise RuntimeError("While loop should not access this part")

                    if ndcReached or deSet:
                        break
                    if tryingToSetDSafterDS_DC_DE or tryingToSetCHafterDS_DC_DE:
                        break
                    if repeatingColumn:
                        break
                    j += 1
                columns.append(current_column)
                current_column = None

            elif current_tag == CellTagType.DS:
                if current_cell.location in prev_ds:
                    continue
                if current_column is None:
                    current_column = Column()
                    current_column.metadata = metadata
                    current_column.content_cells.append(current_cell)
                    current_column.starting_cell = current_cell
                dsSet = False
                deSet = False
                dcSet = False
                tryingToSetCHafterDS_DC_DE = False
                tryingToSetDSafterDS_DC_DE = False
                ndcReached = False
                row_iterator = ExtractHelper.row_iterator(data, col_idx, row_idx)
                max_row_len = len(data[col_idx])
                j = row_idx
                while j < max_row_len and not deSet:
                    current_row_labeled_cell = next(row_iterator)
                    current_row_cell = cast(CellCompact, current_row_labeled_cell.cell)
                    current_row_tag = current_row_labeled_cell.tag
                    if current_row_tag == CellTagType.CH:
                        actual_tag = ExtractHelper.check_left_and_right(data, col_idx, row_idx)
                        if current_row_tag == actual_tag:
                            tryingToSetCHafterDS_DC_DE = True
                            # start new column
                        else:
                            data[col_idx][row_idx].tag = actual_tag
                            continue
                    elif current_row_tag == CellTagType.DS:
                        if dsSet or dcSet or deSet:
                            actual_tag = ExtractHelper.check_left_and_right(data, col_idx, row_idx)
                            if current_row_tag == actual_tag:
                                tryingToSetDSafterDS_DC_DE = True
                            else:
                                data[col_idx][row_idx].tag = actual_tag
                                continue
                        else:
                            current_column.content_cells.append(current_row_cell)
                            if not dsSet:
                                current_column.starting_cell = current_row_cell
                            dsSet = True
                            prev_ds.add(current_row_cell.location)
                    elif current_row_tag == CellTagType.DC:
                        current_column.content_cells.append(current_row_cell)
                        dcSet = True
                    elif current_row_tag == CellTagType.DE:
                        actual_tag = ExtractHelper.check_left_and_right(data, col_idx, row_idx)
                        if current_row_tag == actual_tag:
                            current_column.content_cells.append(current_row_cell)
                            current_column.ending_cell = current_row_cell
                            deSet = True
                        else:
                            data[col_idx][row_idx].tag = actual_tag
                            continue
                    elif current_row_tag == CellTagType.NDC:
                        actual_tag = ExtractHelper.check_left_and_right(data, col_idx, row_idx)
                        if current_row_tag == actual_tag:
                            current_column.content_cells.append(current_row_cell)
                            ndcReached = True
                        else:
                            data[col_idx][row_idx].tag = actual_tag
                            continue
                    else:
                        raise RuntimeError("While loop should not access this part")

                    if ndcReached or deSet:
                        break
                    if tryingToSetDSafterDS_DC_DE or tryingToSetCHafterDS_DC_DE:
                        break
                    j += 1
                columns.append(current_column)
                current_column = None
        return columns



