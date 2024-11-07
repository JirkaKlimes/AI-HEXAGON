from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Optional, Type
import inflection
import jax
from flax.typing import Array
from pydantic import BaseModel

from ai_hexagon.model import Model


class Test(ABC, BaseModel):
    __tests__: ClassVar[Dict[str, Type["Test"]]] = {}

    __test_name__: ClassVar[str] = ""
    __test_title__: ClassVar[str] = ""
    __test_description__: ClassVar[Optional[str]] = None

    __key__: Optional[Array] = None

    seed: int = 0

    def __init_subclass__(cls, **kwargs):
        if not cls.__test_name__:
            cls.__test_name__ = cls.compute_test_name()
        if cls.__test_name__ in cls.__tests__:
            raise ValueError(f"Duplicate test name: {cls.__test_name__}")
        cls.__tests__[cls.__test_name__] = cls
        return super().__init_subclass__(**kwargs)

    @classmethod
    def compute_test_name(cls):
        return inflection.underscore(cls.__name__)

    def model_post_init(self, __context):
        self.__key__ = jax.random.PRNGKey(self.seed)
        self.setup()
        return super().model_post_init(__context)

    def setup(self): ...

    @abstractmethod
    def evalulate(self, model: Type[Model]) -> float: ...

    @property
    def key(self):
        self.__key__, subkey = jax.random.split(self.__key__)
        return subkey
