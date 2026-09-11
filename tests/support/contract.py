from __future__ import annotations
import inspect
from typing import Type


def all_concrete_subclasses(base_cls: Type) -> set[Type]:
    seen: set[Type] = set()
    frontier = list(base_cls.__subclasses__())
    while frontier:
        cls = frontier.pop()
        if cls in seen:
            continue
        seen.add(cls)
        frontier.extend(cls.__subclasses__())
    return {cls for cls in seen if not inspect.isabstract(cls)}


def assert_every_concrete_class_has_a_rig(base_cls: Type, rigs: dict) -> None:
    discovered = all_concrete_subclasses(base_cls)
    missing = discovered - set(rigs.keys())
    assert not missing, (
        f"{base_cls.__name__} has concrete subclasses with no registered Rig: "
        f"{sorted(c.__name__ for c in missing)}. Add them to the *_RIGS dict."
    )
