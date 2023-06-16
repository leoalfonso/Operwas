import enum

REUSE_TYPE_TO_STR = [
    "no_reuse",
    "agricultural",
    "urban",
]


class ReuseType(enum.Enum):
    NO_REUSE = 0
    AGRICULTURAL = 1
    URBAN = 2

    def __str__(self):
        return REUSE_TYPE_TO_STR[self.value]
