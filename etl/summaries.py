DESCRIPTION = """ Column Summaries
Stage two will take the column objects, represented as a list 
of cell objects, and turn the variable-length lists into a fixed-length
feature vector, which can be used by later pipeline stages in order to 
compute clusters, similarity, and relationships between pairs. Summaries are computed
differently for numerical and text data, as the serialization of text data requires the
use of BERT, and serialization of numerical data is done by attempting to extract abstract
attributes of the data distribution (normalized histogram, range, mean/median/standard deviation etc.)
"""

# Summaries Modules:
from components.extended_summaries.extended_summary import ExtendedSummary
from components.cell_labeling.cell_compact import ContentType

from sklearn.preprocessing import StandardScaler

from enum import Enum
import numpy as np


def summaries(results):
    SummaryClass = ExtendedSummary
    SUMMARY_TYPE = 'extended_summary'
    numeric_filtered = np.extract([x.type==ContentType.NUMERIC for x in results], results)
    text_filtered = np.extract([x.type==ContentType.STRING for x in results], results)

    numeric_summary = [
        SummaryClass(c) for c in numeric_filtered
    ]
    numeric_vectorized = np.array([c.vector for c in numeric_summary])
    numeric_column_names = [c.colname for c in numeric_summary]
    numeric_sheet_names = [c.sheetname for c in numeric_summary]
    numeric_file_names = [c.filename for c in numeric_summary]
    numeric_id = [c.id for c in numeric_summary]
    numeric_N = numeric_vectorized.shape[0]

    text_summary = [
        SummaryClass(c) for c in text_filtered
    ]
    text_vectorized = np.array([c.vector for c in text_summary])
    # text_column_names = [c.header for c in text_summary]
    text_column_names = [c.colname for c in text_summary]
    text_sheet_names = [c.sheetname for c in text_summary]
    text_file_names = [c.filename for c in text_summary]
    text_id = [c.id for c in text_summary]
    text_N = text_vectorized.shape[0]
    print(numeric_filtered.shape, len(numeric_summary), numeric_vectorized.shape)
    # Scale data using Z-norm
    numeric_scaled = StandardScaler().fit_transform(numeric_vectorized)
    text_scaled = StandardScaler().fit_transform(text_vectorized)
    return {'numeric': numeric_scaled, 'text': text_scaled}