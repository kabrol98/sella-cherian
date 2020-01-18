from collections import namedtuple
from enum import Enum
from typing import Any

Location = namedtuple('Location', ['col', 'row'])

class CellTagType(Enum):
    CH = 0  # Column Header
    DC = 1  # Data Continue
    DE = 2  # Data End
    DS = 3  # Data Start
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

