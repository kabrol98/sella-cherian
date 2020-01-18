from collections import namedtuple
from enum import Enum
from typing import Any

Location = namedtuple('Location', ['col', 'row'])

class CellTagType(Enum):
    CH = 0  # Column Header
    DS = 1  # Data Start
    DC = 2  # Data Continue
    DE = 3  # Data End
    NDC = 4  # Not a Data Cell


class ContentType(Enum):
    NUMERIC = 0
    STRING = 1
    NULL = 2


class CellCompact:
    def __init__(self, location: Location, content: Any):
        self.location = location
        self.content = content
        self.content_type: ContentType = None
    def __str__(self):
        return "CellCompact: location -> %s, content -> %s, content_type -> %s" % (self.location, self.content_type, self.content_type)

