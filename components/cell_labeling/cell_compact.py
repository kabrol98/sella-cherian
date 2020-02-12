from collections import namedtuple
from enum import IntEnum
from typing import Any

Location = namedtuple('Location', ['col', 'row'])


class CellTagType(IntEnum):
    CH = 0  # Column Header
    DC = 1  # Data Continue
    DE = 2  # Data End
    DS = 3  # Data Start
    NDC = 4  # Not a Data Cell


class ContentType(IntEnum):
    NUMERIC = 0
    STRING = 1
    NULL = 2


class CellCompact:
    def __init__(self, location: Location, content: Any):
        self.location = location
        self.content = content
        self.content_type: ContentType = None

    def __str__(self):
        return "CellCompact: location -> %s, content -> %s, content_type -> %s" % (
        self.location, self.content, self.content_type)
