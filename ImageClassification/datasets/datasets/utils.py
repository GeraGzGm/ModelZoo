from dataclasses import dataclass

import cv2

@dataclass
class Interpolations:
    INTER_LINEAR = cv2.INTER_LINEAR

    @classmethod
    def to_cv2(cls, name: str) -> int:
        """Get interpolation by name."""
        try:
            return getattr(cls, name)
        except AttributeError:
            raise ValueError(
                f"Unknown interpolation '{name}'. "
                f"Available interpolation: {list(cls.__annotations__.keys())}"
            )


