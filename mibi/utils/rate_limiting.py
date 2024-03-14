from typing import Callable, ParamSpec, TypeVar
from pyrate_limiter.limiter import Limiter, ItemMapping

_T = TypeVar("_T")
_P = ParamSpec("_P")


def _default_mapping(*_args, **_kwargs):
    return "default", 1


def rate_limit(limiter: Limiter, mapping: ItemMapping = _default_mapping) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:

    def _limit(f: Callable[_P, _T]) -> Callable[_P, _T]:
        return limiter.as_decorator()(mapping)(f)  # type: ignore

    return _limit
