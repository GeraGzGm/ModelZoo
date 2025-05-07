from enum import Enum, auto
from ...base_dataset import BaseLabels

class Labels(BaseLabels):
    airplane = 0
    automobile  = 1
    bird = 2
    cat = 3
    deer = 4
    dog = 5
    frog = 6
    horse = 7
    ship = 8
    truck = 9

    @classmethod
    def get_key(cls, value: int) -> str:
        return cls(value).name
    
    @classmethod
    def __len__(cls) -> int:
        return len(cls.__members__)