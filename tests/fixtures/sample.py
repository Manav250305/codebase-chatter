"""Module docstring."""
from __future__ import annotations

import os
from typing import (
    Any,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


def plain(a: int, b: int = 2) -> int:
    """Add two numbers.

    Returns the sum.
    """
    return a + b


@decorator
@other.decorator(
    "arg",
    flag=True,
)
def decorated() -> None:
    pass


async def fetch(url: str) -> bytes:
    '''Fetch a URL.'''
    import json

    async def inner() -> None:
        pass

    return b""


class Base:
    """A base class."""

    class_attr = lambda self: 1

    def __init__(self) -> None:
        self.x = 1

    @property
    def value(self) -> int:
        return self.x

    @staticmethod
    async def make() -> "Base":
        return Base()

    def outer(self) -> None:
        def helper(y):
            def deepest():
                return y

            return deepest

        class Local:
            def local_method(self):
                pass

        return None

    class Nested:
        def nested_method(self):
            """Nested."""


try:
    def maybe() -> None:
        pass
except ImportError:
    maybe = None

square = lambda n: n * n


def one_liner(): return 1
