from typing import Union

from ai_hexagon.tests.hash_map import HashMap
from ai_hexagon.tests.state_tracking import StateTracking


Test = Union[HashMap, StateTracking]

__all__ = ["Test", "HashMap", "StateTracking"]
