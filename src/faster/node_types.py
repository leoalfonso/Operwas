import enum

NODE_TYPE_TO_STR = [
    "NOTHING",
    "WWTP",
    "WWPS",
]


class NodeType(enum.Enum):
    NOTHING = 0
    WWTP = 1
    WWPS = 2

    def __str__(self):
        return NODE_TYPE_TO_STR[self.value]
