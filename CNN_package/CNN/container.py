from typing import Any

class Container():
    def __call__(self, name: str, value: Any) -> None:
        setattr(self, name, value)