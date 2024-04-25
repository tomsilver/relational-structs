"""Data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeAlias,
    TypeVar,
)

import numpy as np
from gym.spaces import Space
from numpy.typing import NDArray
from tabulate import tabulate


@dataclass(frozen=True, order=True)
class Type:
    """Struct defining a type."""

    name: str
    feature_names: Sequence[str] = field(repr=False)
    parent: Optional[Type] = field(default=None, repr=False)

    @property
    def dim(self) -> int:
        """Dimensionality of the feature vector of this object type."""
        return len(self.feature_names)

    def __call__(self, name: str) -> _TypedEntity:
        """Convenience method for generating _TypedEntities."""
        if name.startswith("?"):
            return Variable(name, self)
        return Object(name, self)

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.feature_names)))


@dataclass(frozen=True, order=True, repr=False)
class _TypedEntity:
    """Struct defining an entity with some type, either an object (e.g.,
    block3) or a variable (e.g., ?block).

    Should not be instantiated externally.
    """

    name: str
    type: Type

    @cached_property
    def _str(self) -> str:
        return f"{self.name}:{self.type.name}"

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return self._str

    def is_instance(self, t: Type) -> bool:
        """Return whether this entity is an instance of the given type, taking
        hierarchical typing into account."""
        cur_type: Optional[Type] = self.type
        while cur_type is not None:
            if cur_type == t:
                return True
            cur_type = cur_type.parent
        return False


@dataclass(frozen=True, order=True, repr=False)
class Object(_TypedEntity):
    """Struct defining an Object, which is just a _TypedEntity whose name does
    not start with "?"."""

    def __post_init__(self) -> None:
        assert not self.name.startswith("?")

    def __hash__(self) -> int:
        # By default, the dataclass generates a new __hash__ method when
        # frozen=True and eq=True, so we need to override it.
        return self._hash


@dataclass(frozen=True, order=True, repr=False)
class Variable(_TypedEntity):
    """Struct defining a Variable, which is just a _TypedEntity whose name
    starts with "?"."""

    def __post_init__(self) -> None:
        assert self.name.startswith("?")

    def __hash__(self) -> int:
        # By default, the dataclass generates a new __hash__ method when
        # frozen=True and eq=True, so we need to override it.
        return self._hash


@dataclass
class State:
    """Struct defining the low-level state of the world."""

    data: Dict[Object, Array]
    # Some environments will need to store additional simulator state, so
    # this field is provided.
    simulator_state: Optional[Any] = None

    def __post_init__(self) -> None:
        # Check feature vector dimensions.
        for obj in self:
            assert len(self[obj]) == obj.type.dim

    def __iter__(self) -> Iterator[Object]:
        """An iterator over the state's objects, in sorted order."""
        return iter(sorted(self.data))

    def __getitem__(self, key: Object) -> Array:
        return self.data[key]

    def get(self, obj: Object, feature_name: str) -> Any:
        """Look up an object feature by name."""
        idx = obj.type.feature_names.index(feature_name)
        return self.data[obj][idx]

    def set(self, obj: Object, feature_name: str, feature_val: Any) -> None:
        """Set the value of an object feature by name."""
        idx = obj.type.feature_names.index(feature_name)
        self.data[obj][idx] = feature_val

    def get_objects(self, object_type: Type) -> List[Object]:
        """Return objects of the given type in the order of __iter__()."""
        return [o for o in self if o.is_instance(object_type)]

    def vec(self, objects: Sequence[Object]) -> Array:
        """Concatenated vector of features for each of the objects in the given
        ordered list."""
        feats: List[Array] = []
        if len(objects) == 0:
            return np.zeros(0, dtype=np.float32)
        for obj in objects:
            feats.append(self[obj])
        return np.hstack(feats)

    def copy(self) -> State:
        """Return a copy of this state.

        The simulator state is assumed to be immutable.
        """
        new_data = {}
        for obj in self:
            new_data[obj] = self._copy_state_value(self.data[obj])
        return State(new_data, simulator_state=self.simulator_state)

    def _copy_state_value(self, val: Any) -> Any:
        if val is None or isinstance(val, (float, bool, int, str)):
            return val
        if isinstance(val, (list, tuple, set)):
            return type(val)(self._copy_state_value(v) for v in val)
        assert hasattr(val, "copy")
        return val.copy()

    def allclose(self, other: State) -> bool:
        """Return whether this state is close enough to another one, i.e., its
        objects are the same, and the features are close."""
        if self.simulator_state is not None or other.simulator_state is not None:
            raise NotImplementedError(
                "Cannot use allclose when simulator_state is not None."
            )
        if not sorted(self.data) == sorted(other.data):
            return False
        for obj in self.data:
            if not np.allclose(self.data[obj], other.data[obj], atol=1e-3):
                return False
        return True

    def pretty_str(self) -> str:
        """Display the state in a nice human-readable format."""
        type_to_table: Dict[Type, List[List[str]]] = {}
        for obj in self:
            if obj.type not in type_to_table:
                type_to_table[obj.type] = []
            type_to_table[obj.type].append([obj.name] + list(map(str, self[obj])))
        table_strs = []
        for t in sorted(type_to_table):
            headers = ["type: " + t.name] + list(t.feature_names)
            table_strs.append(tabulate(type_to_table[t], headers=headers))
        ll = max(len(line) for table in table_strs for line in table.split("\n"))
        prefix = "#" * (ll // 2 - 3) + " STATE " + "#" * (ll - ll // 2 - 4) + "\n"
        suffix = "\n" + "#" * ll + "\n"
        return prefix + "\n\n".join(table_strs) + suffix


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


@dataclass(frozen=True, order=True, repr=False)
class Predicate:
    """Struct defining a predicate (a lifted classifier over states)."""

    name: str
    types: Sequence[Type]
    # The classifier takes in a complete state and a sequence of objects
    # representing the arguments. These objects should be the only ones
    # treated "specially" by the classifier.
    _classifier: Callable[[State, Sequence[Object]], bool] = field(compare=False)

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    @cached_property
    def pddl_str(self) -> str:
        """Get a string representation suitable for writing out to PDDL."""
        if self.arity == 0:
            return f"({self.name})"
        vars_str = " ".join(f"?x{i} - {t.name}" for i, t in enumerate(self.types))
        return f"({self.name} {vars_str})"

    @cached_property
    def arity(self) -> int:
        """The arity of this predicate (number of arguments)."""
        return len(self.types)

    def holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Public method for calling the classifier.

        Performs type checking first.
        """
        assert len(objects) == self.arity
        for obj, pred_type in zip(objects, self.types):
            assert isinstance(obj, Object)
            assert obj.is_instance(pred_type)
        return self._classifier(state, objects)

    def __str__(self) -> str:
        return self.pddl_str

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return str(self)


_TypedEntityTypeVar = TypeVar("_TypedEntityTypeVar", bound=_TypedEntity)


@dataclass(frozen=True, repr=False, eq=False)
class _Atom(Generic[_TypedEntityTypeVar]):
    """Struct defining an atom (a predicate applied to either variables or
    objects).

    Should not be instantiated externally.
    """

    predicate: Predicate
    entities: Sequence[_TypedEntityTypeVar]

    def __post_init__(self) -> None:
        if isinstance(self.entities, _TypedEntity):
            raise ValueError(
                "Atoms expect a sequence of entities, not a single entity."
            )
        assert len(self.entities) == self.predicate.arity
        for ent, pred_type in zip(self.entities, self.predicate.types):
            assert ent.is_instance(pred_type)

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    @cached_property
    def pddl_str(self) -> str:
        """Get a string representation suitable for writing out to a PDDL
        file."""
        if not self.entities:
            return f"({self.predicate.name})"
        entities_str = " ".join(e.name for e in self.entities)
        return f"({self.predicate.name} {entities_str})"

    def __str__(self) -> str:
        return self.pddl_str

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, _Atom)
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, _Atom)
        return str(self) < str(other)


@dataclass(frozen=True, repr=False, eq=False)
class LiftedAtom(_Atom[Variable]):
    """Struct defining a lifted atom (a predicate applied to variables)."""

    @cached_property
    def variables(self) -> List[Variable]:
        """Arguments for this lifted atom."""
        return list(self.entities)

    def ground(self, sub: Dict[Variable, Object]) -> GroundAtom:
        """Create a GroundAtom with a given substitution."""
        assert set(self.variables).issubset(set(sub.keys()))
        return GroundAtom(self.predicate, [sub[v] for v in self.variables])

    def substitute(self, sub: Dict[Variable, Variable]) -> LiftedAtom:
        """Create a LiftedAtom with a given substitution."""
        assert set(self.variables).issubset(set(sub.keys()))
        return LiftedAtom(self.predicate, [sub[v] for v in self.variables])


@dataclass(frozen=True, repr=False, eq=False)
class GroundAtom(_Atom[Object]):
    """Struct defining a ground atom (a predicate applied to objects)."""

    @cached_property
    def objects(self) -> List[Object]:
        """Arguments for this ground atom."""
        return list(self.entities)

    def lift(self, sub: Dict[Object, Variable]) -> LiftedAtom:
        """Create a LiftedAtom with a given substitution."""
        assert set(self.objects).issubset(set(sub.keys()))
        return LiftedAtom(self.predicate, [sub[o] for o in self.objects])

    def holds(self, state: State) -> bool:
        """Check whether this ground atom holds in the given state."""
        return self.predicate.holds(state, self.objects)


_AtomTypeVar = TypeVar("_AtomTypeVar", bound=_Atom)


@dataclass(frozen=True, repr=False, eq=False)
class Operator(Generic[_TypedEntityTypeVar, _AtomTypeVar]):
    """Struct defining a symbolic operator (as in STRIPS)."""

    name: str
    parameters: Sequence[_TypedEntityTypeVar]
    preconditions: Set[_AtomTypeVar]
    add_effects: Set[_AtomTypeVar]
    delete_effects: Set[_AtomTypeVar]

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    @cached_property
    def pddl_str(self) -> str:
        """Get a string representation suitable for writing out to PDDL."""
        params_str = " ".join(f"{p.name} - {p.type.name}" for p in self.parameters)
        preconds_str = "\n        ".join(
            atom.pddl_str for atom in sorted(self.preconditions)
        )
        effects_str = "\n        ".join(
            atom.pddl_str for atom in sorted(self.add_effects)
        )
        if self.delete_effects:
            effects_str += "\n        "
            effects_str += "\n        ".join(
                f"(not {atom.pddl_str})" for atom in sorted(self.delete_effects)
            )
        return f"""(:action {self.name}
    :parameters ({params_str})
    :precondition (and {preconds_str})
    :effect (and {effects_str})
  )"""

    @cached_property
    def short_str(self) -> str:
        """Abbreviated name, not necessarily unique."""
        param_str = ", ".join([p.name for p in self.parameters])
        return f"{self.name}({param_str})"

    def __hash__(self) -> int:
        return self._hash

    def __str__(self) -> str:
        return self.pddl_str

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, Operator)
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, Operator)
        return str(self) < str(other)

    def __gt__(self, other: object) -> bool:
        assert isinstance(other, Operator)
        return str(self) > str(other)


@dataclass(frozen=True, repr=False, eq=False)
class LiftedOperator(Operator[Variable, LiftedAtom]):
    """Struct defining a lifted symbolic operator (as in STRIPS)."""

    @lru_cache(maxsize=None)
    def ground(self, objects: Tuple[Object]) -> GroundSTRIPSOperator:
        """Ground into a _GroundSTRIPSOperator, given objects.

        Insist that objects are tuple for hashing in cache.
        """
        assert isinstance(objects, tuple)
        assert len(objects) == len(self.parameters)
        assert all(o.is_instance(p.type) for o, p in zip(objects, self.parameters))
        sub = dict(zip(self.parameters, objects))
        preconditions = {atom.ground(sub) for atom in self.preconditions}
        add_effects = {atom.ground(sub) for atom in self.add_effects}
        delete_effects = {atom.ground(sub) for atom in self.delete_effects}
        return GroundSTRIPSOperator(
            self.name, list(objects), preconditions, add_effects, delete_effects, self
        )


@dataclass(frozen=True, repr=False, eq=False)
class GroundSTRIPSOperator(Operator[Object, GroundAtom]):
    """A STRIPSOperator + objects."""

    parent: LiftedOperator


# Type aliases.
Array: TypeAlias = NDArray[np.float32]
Action: TypeAlias = Any
Parameters: TypeAlias = Any
ParameterSpace: TypeAlias = Space
OptionMemory: TypeAlias = Dict
ParameterizedPolicy: TypeAlias = Callable[[State, Parameters, OptionMemory], Action]
ParameterizedInitiable: TypeAlias = Callable[[State, Parameters, OptionMemory], bool]
ParameterizedTerminal: TypeAlias = Callable[[State, Parameters, OptionMemory], bool]

# Constants.
DefaultState = State({})
