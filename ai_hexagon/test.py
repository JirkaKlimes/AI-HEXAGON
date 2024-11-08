from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Type
import jax
from flax.typing import Array
from pydantic import BaseModel

from ai_hexagon.model import Model


class BaseTest(ABC, BaseModel):
    __title__: ClassVar[str] = ""
    __description__: ClassVar[Optional[str]] = None

    __key__: Optional[Array] = None

    name: str
    seed: int = 0

    def __hash__(self):
        return self.model_dump_json().__hash__()

    def __eq__(self, other):
        return self.model_dump_json() == other.model_dump_json()

    def model_post_init(self, __context):
        self.__key__ = jax.random.PRNGKey(self.seed)
        return super().model_post_init(__context)

    @abstractmethod
    def evalulate(self, model: Type[Model]) -> float: ...

    @classmethod
    def get_test_name(cls):
        return cls.model_fields["name"].default

    @property
    def key(self):
        self.__key__, subkey = jax.random.split(self.__key__)
        return subkey
