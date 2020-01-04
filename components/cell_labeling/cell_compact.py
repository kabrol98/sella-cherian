from collections import namedtuple
from enum import Enum
from typing import Any

Location = namedtuple('Location', ['col', 'row'])

class CellTagType(Enum):
    CH = "CH"  # Column Header
    DS = "DS"  # Data Start
    DC = "DC"  # Data Continue
    DE = "DE"  # Data End
    NDC = "NDC"  # Not a Data Cell


class CellContentType(Enum):
    NUMERIC = 0
    STRING = 1
    BLANK = 2


class CellCompact:
    def __init__(self, location: Location, content: Any):
        self.location = location
        self.content = content
        self.content_type: CellContentType = None

