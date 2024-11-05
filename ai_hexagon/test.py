from abc import ABC
from typing import ClassVar, Dict, Optional

import inflection
import jax
from pydantic import BaseModel


class Test(ABC, BaseModel):
    __tests__: ClassVar[Dict[str, "Test"]] = {}

    __test_name__: ClassVar[str] = None
    __test_title__: ClassVar[str] = ...
    __test_description__: ClassVar[Optional[str]] = None

    __seed: int = 69

    def __init_subclass__(cls, **kwargs):
        if cls.__test_name__ is None:
            cls.__test_name__ = cls.compute_test_name()
        if cls.__test_name__ in cls.__tests__:
            raise ValueError(f"Duplicate test name: {cls.__test_name__}")
        cls.__tests__[cls.__test_name__] = cls
        return super().__init_subclass__(**kwargs)

    @classmethod
    def compute_test_name(cls):
        return inflection.underscore(cls.__name__)

    def model_post_init(self, __context):
        self.__key__ = jax.random.PRNGKey(self.__seed)
        self.setup()
        return super().model_post_init(__context)

    def setup(self): ...

    @property
    def key(self):
        self.__key__, subkey = jax.random.split(self.__key__)
        return subkey
