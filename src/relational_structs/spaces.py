"""Gym spaces that use relational data structures."""

from typing import Any, Collection, List, Sequence, Set, cast

import numpy as np
from gymnasium.spaces import Box, Space

from relational_structs.object_centric_state import ObjectCentricState
from relational_structs.objects import Object, Type


class ObjectCentricStateSpace(Space):
    """A space for where members are State instances.

    Different members may have different objects, but the types of the
    objects are fixed.
    """

    def __init__(self, types: Collection[Type]) -> None:
        super().__init__()
        self._types = types

    @property
    def types(self) -> Set[Type]:
        """Expose the types in this space."""
        return set(self._types)

    def contains(self, x: Any) -> bool:
        assert isinstance(x, ObjectCentricState)
        return all(o.type in self._types for o in x)

    @property
    def is_np_flattenable(self) -> bool:
        # Not flattenable because number of objects can change.
        return False

    def sample(self, mask: Any | None = None) -> ObjectCentricState:
        raise NotImplementedError("Sampling is not well-defined.")


class ObjectSequenceSpace(Space):
    """A space for where members are sequences of objects corresponding to a
    given sequence of types.

    For example, if space = ObjectSequenceSpace([dog, cat, dog]), then
    one member might be [nomsy_the_dog, alfred_the_cat, goldie_the_dog].
    """

    def __init__(self, types: Sequence[Type]) -> None:
        super().__init__()
        self._types = types

    @property
    def types(self) -> List[Type]:
        """Expose the types in this space."""
        return list(self._types)

    def contains(self, x: Any) -> bool:
        x = cast(Sequence[Object], x)
        if len(x) != len(self._types):
            return False
        return all(o.type == t for o, t in zip(x, self._types))

    @property
    def is_np_flattenable(self) -> bool:
        # Is flattenable because the length of members is fixed.
        return True

    def sample(self, mask: Any | None = None) -> ObjectCentricState:
        raise NotImplementedError("Sampling is not well-defined.")


class ObjectSequenceBoxSpace(Space):
    """A wrapper around an ObjectSequenceSpace and a Box space. This is a very
    common kind of space, e.g., for defining parameterized options that have
    both object and continuous parameters.

    Elements are tuples where the first element is an object sequence and the
    second element is a vector in the Box space.

    NOTE: the Box input spec is much more general, but we restrict it here for
    consistency and ease of use.
    """

    def __init__(
        self, types: Sequence[Type], low: Sequence[float], high: Sequence[float]
    ) -> None:
        super().__init__()
        self._object_sequence_space = ObjectSequenceSpace(types)
        low_arr = np.array(low, dtype=np.float32)
        high_arr = np.array(high, dtype=np.float32)
        self._box_space = Box(low=low_arr, high=high_arr, dtype=np.float32)

    def contains(self, x: Any) -> bool:
        assert isinstance(x, tuple)
        object_sequence, box_params = x
        return self._object_sequence_space.contains(
            object_sequence
        ) and self._box_space.contains(box_params)

    @property
    def is_np_flattenable(self) -> bool:
        return (
            self._object_sequence_space.is_np_flattenable
            and self._box_space.is_np_flattenable
        )

    def sample(self, mask: Any | None = None) -> ObjectCentricState:
        raise NotImplementedError("Sampling is not well-defined.")
