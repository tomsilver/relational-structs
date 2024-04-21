"""Gym spaces that use relational data structures."""

from typing import Any, Collection, Set

from gym.spaces import Space

from relational_structs.structs import State, Type


class ObjectCentricStateSpace(Space):
    """A space for where observations are State instances.

    Different observations may have different objects, but the types of
    the objects are fixed.
    """

    def __init__(self, types: Collection[Type]) -> None:
        super().__init__()
        self._types = types

    @property
    def types(self) -> Set[Type]:
        """Expose the types in this space."""
        return set(self._types)

    def contains(self, x: Any) -> bool:
        assert isinstance(x, State)
        return all(o.type in self._types for o in x)

    @property
    def is_np_flattenable(self) -> bool:
        # Not flattenable because number of objects can change.
        return False

    def sample(self, mask: Any | None = None) -> State:
        raise NotImplementedError("Sampling is not well-defined.")
