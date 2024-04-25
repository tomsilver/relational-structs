"""Data structures for (parameterized) options."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import (
    Any,
    Callable,
    Dict,
    TypeAlias,
)

import numpy as np
from gym.spaces import Space
from numpy.typing import NDArray

from relational_structs.state import State


@dataclass(frozen=True, eq=False)
class ParameterizedOption:
    """Struct defining a parameterized option, which has a parameter space and
    can be ground into an Option, given parameter values.

    An option is composed of a policy, an initiation classifier, and a
    termination condition. We will stick with deterministic termination
    conditions. For a parameterized option, all of these are conditioned
    on parameters.
    """

    name: str
    params_space: ParameterSpace = field(repr=False)
    # A policy maps a state, parameters, and a memory dict to an action.
    policy: ParameterizedPolicy = field(repr=False)
    # An initiation classifier maps a state, parameters, and a memory dict to a
    # bool, which is True iff the option can start now.
    initiable: ParameterizedInitiable = field(repr=False)
    # A termination condition maps a state, parameters, and a memory dict to a
    # bool, which is True iff the option should terminate now.
    terminal: ParameterizedTerminal = field(repr=False)

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, ParameterizedOption)
        return self.name == other.name

    def __lt__(self, other: Any) -> bool:
        assert isinstance(other, ParameterizedOption)
        return self.name < other.name

    def __gt__(self, other: Any) -> bool:
        assert isinstance(other, ParameterizedOption)
        return self.name > other.name

    def __hash__(self) -> int:
        return self._hash

    def ground(self, params: Parameters) -> Option:
        """Ground into an Option given parameters."""
        assert self.params_space.contains(params)
        memory: OptionMemory = {}  # each option has its own memory dict
        return Option(
            self.name,
            lambda s: self.policy(s, params, memory),
            initiable=lambda s: self.initiable(s, params, memory),
            terminal=lambda s: self.terminal(s, params, memory),
            parent=self,
            params=params,
            memory=memory,
        )


@dataclass(eq=False)
class Option:
    """Struct defining an option, which is like a parameterized option except
    that its components are not conditioned on parameters."""

    name: str
    # A policy maps a state to an action.
    policy: Callable[[State], Action] = field(repr=False)
    # An initiation classifier maps a state to a bool, which is True
    # iff the option can start now.
    initiable: Callable[[State], bool] = field(repr=False)
    # A termination condition maps a state to a bool, which is True
    # iff the option should terminate now.
    terminal: Callable[[State], bool] = field(repr=False)
    # The parameterized option that generated this option.
    parent: ParameterizedOption = field(repr=False)
    # The parameters that were used to ground this option.
    params: Parameters
    # The memory dictionary for this option.
    memory: OptionMemory = field(repr=False)


# Type aliases.
Array: TypeAlias = NDArray[np.float32]
Action: TypeAlias = Any
Parameters: TypeAlias = Any
ParameterSpace: TypeAlias = Space
OptionMemory: TypeAlias = Dict
ParameterizedPolicy: TypeAlias = Callable[[State, Parameters, OptionMemory], Action]
ParameterizedInitiable: TypeAlias = Callable[[State, Parameters, OptionMemory], bool]
ParameterizedTerminal: TypeAlias = Callable[[State, Parameters, OptionMemory], bool]
