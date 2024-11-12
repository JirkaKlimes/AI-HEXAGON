from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Type
import jax
from flax.typing import Array
from pydantic import BaseModel

from ai_hexagon.model import Model


class BaseTest(ABC, BaseModel):
    __title__: ClassVar[str] = ""
    __description__: ClassVar[Optional[str]] = None

    __gpu_key__: Optional[Array] = None
    __cpu_key__: Optional[Array] = None

    name: str
    seed: int = 0

    def __hash__(self):
        return self.model_dump_json().__hash__()

    def __eq__(self, other):
        return self.model_dump_json() == other.model_dump_json()

    def model_post_init(self, __context):
        self.__gpu_key__ = jax.random.PRNGKey(self.seed)
        self.__cpu_key__ = jax.jit(
            lambda seed: jax.random.PRNGKey(seed), backend="cpu"
        )(self.seed)
        self.__cpu_split = jax.jit(lambda key: jax.random.split(key), backend="cpu")
        return super().model_post_init(__context)

    @abstractmethod
    def evalulate(self, model_class: Type[Model]) -> float: ...

    @classmethod
    def get_test_name(cls):
        return cls.model_fields["name"].default

    @property
    def gpu_key(self):
        self.__gpu_key__, subkey = jax.random.split(self.__gpu_key__)
        return subkey

    @property
    def cpu_key(self):
        self.__cpu_key__, subkey = self.__cpu_split(self.__cpu_key__)
        return subkey
