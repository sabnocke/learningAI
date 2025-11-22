from abc import abstractmethod, ABC
from functools import singledispatchmethod
from typing import Any

class BaseLearningRateScheduler(ABC):
    @singledispatchmethod
    def step(self, n: None = None) -> Any:
        del n

    @step.register
    def _(self, n: int):
        return self._step_int(n)

    @step.register
    def _(self, n: float):
        return self._step_float(n)

    def _step_float(self, n: float) -> float:
        raise NotImplementedError(
            f"{self.__class__.__name__} cannot use {type(n).__name__}"
        )

    def _step_int(self, n: int) -> float:
        raise NotImplementedError(
            f"{self.__class__.__name__} cannot use {type(n).__name__}"
        )


    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        has_int = cls._step_int is not BaseLearningRateScheduler._step_int
        has_float = cls._step_float is not BaseLearningRateScheduler._step_float

        if not (has_int or has_float):
            raise TypeError(
                f"{cls.__name__} must implement at least one of: _step_int, _step_float"
            )
