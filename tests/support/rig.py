from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Type


@dataclass
class Rig:
    backend_cls: Type
    make_unloaded: Callable[[], Any]
    make_loaded: Callable[[], ContextManager[Any]]
    reload: Callable[[Any], None]
