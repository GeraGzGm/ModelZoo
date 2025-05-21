from enum import Enum, auto
from ...base_dataset import BaseLabels

class Labels(BaseLabels):
    labels = ["0","0","0","0","0","0","0","0","0","0"]

    @classmethod
    def get_key(cls, value: int) -> str:
        return cls(value).name
    
    @classmethod
    def __len__(cls) -> int:
        return len(cls.__members__)