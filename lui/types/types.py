from enum import Enum

class InvariantTypes(str, Enum):
    PROJECTION = 'PROJECTION'
    HYPERPLANE = 'HYPERPLANE'
    VAPNIK = 'VAPNIK'
    ALL = 'ALL'
