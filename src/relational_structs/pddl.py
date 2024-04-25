"""PDDL related data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    List,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

from multimethod import multimethod

from relational_structs.objects import Object, Type, TypedEntity, Variable
from relational_structs.state import State

_TypedEntityTypeVar = TypeVar("_TypedEntityTypeVar", bound=TypedEntity)


@dataclass(frozen=True, repr=False, eq=False)
class _Atom(Generic[_TypedEntityTypeVar]):
    """Struct defining an atom (a predicate applied to either variables or
    objects).

    Should not be instantiated externally.
    """

    predicate: Predicate
    entities: Sequence[_TypedEntityTypeVar]

    def __post_init__(self) -> None:
        if isinstance(self.entities, TypedEntity):
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


@dataclass(frozen=True, order=True, repr=False)
class Predicate:
    """Struct defining a predicate (a lifted classifier over states)."""

    name: str
    types: Sequence[Type]
    # The classifier takes in a complete state and a sequence of objects
    # representing the arguments. These objects should be the only ones
    # treated "specially" by the classifier.
    _classifier: Callable[[State, Sequence[Object]], bool] = field(compare=False)

    @multimethod
    def __call__(self, entities: Sequence[TypedEntity]) -> Any:
        raise NotImplementedError(f"Cannot call predicate with {entities}")

    @__call__.register
    def _(self, entities: Sequence[Variable]) -> LiftedAtom:
        return LiftedAtom(self, entities)

    @__call__.register
    def _(self, entities: Sequence[Object]) -> GroundAtom:
        return GroundAtom(self, entities)

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
    def ground(self, objects: Tuple[Object]) -> GroundOperator:
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
        return GroundOperator(
            self.name, list(objects), preconditions, add_effects, delete_effects, self
        )


@dataclass(frozen=True, repr=False, eq=False)
class GroundOperator(Operator[Object, GroundAtom]):
    """A STRIPSOperator + objects."""

    parent: LiftedOperator


@dataclass(frozen=True)
class PDDLDomain:
    """A PDDL domain."""

    name: str
    operators: Collection[LiftedOperator]
    predicates: Collection[Predicate]
    types: Collection[Type]

    @classmethod
    def parse(cls, pddl_str: str) -> PDDLDomain:
        """Parse a domain from a string."""
        # TODO
        raise NotImplementedError("TODO")

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    @cached_property
    def _str(self) -> str:
        # Sort everything to ensure determinism.
        preds_lst = sorted(self.predicates)
        # Case 1: no type hierarchy.
        if all(t.parent is None for t in self.types):
            types_str = " ".join(t.name for t in sorted(self.types))
        # Case 2: type hierarchy.
        else:
            parent_to_children_types: Dict[Type, List[Type]] = {
                t: [] for t in self.types
            }
            for t in sorted(self.types):
                if t.parent:
                    parent_to_children_types[t.parent].append(t)
            types_str = ""
            for parent_type in sorted(parent_to_children_types):
                child_types = parent_to_children_types[parent_type]
                if not child_types:
                    # Special case: type has no children and also does not appear
                    # as a child of another type.
                    is_child_type = any(
                        parent_type in children
                        for children in parent_to_children_types.values()
                    )
                    if not is_child_type:
                        types_str += f"\n    {parent_type.name}"
                    # Otherwise, the type will appear as a child elsewhere.
                else:
                    child_type_str = " ".join(t.name for t in child_types)
                    types_str += f"\n    {child_type_str} - {parent_type.name}"
        ops_lst = sorted(self.operators)
        preds_str = "\n    ".join(pred.pddl_str for pred in preds_lst)
        ops_strs = "\n\n  ".join(op.pddl_str for op in ops_lst)
        return f"""(define (domain {self.name})
    (:requirements :typing)
    (:types {types_str})

    (:predicates\n    {preds_str}
    )

    {ops_strs}
)
"""

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return self._str


@dataclass(frozen=True)
class PDDLProblem:
    """A PDDL problem."""

    domain_name: str
    problem_name: str
    objects: Collection[Object]
    init_atoms: Collection[GroundAtom]
    goal: Collection[GroundAtom]

    @classmethod
    def parse(cls, pddl_str: str) -> PDDLProblem:
        """Parse a problem from a string."""
        # TODO
        raise NotImplementedError("TODO")

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    @cached_property
    def _str(self) -> str:
        # Sort everything to ensure determinism.
        objects_lst = sorted(self.objects)
        init_atoms_lst = sorted(self.init_atoms)
        goal_lst = sorted(self.goal)
        objects_str = "\n    ".join(f"{o.name} - {o.type.name}" for o in objects_lst)
        init_str = "\n    ".join(atom.pddl_str for atom in init_atoms_lst)
        goal_str = "\n    ".join(atom.pddl_str for atom in goal_lst)
        return f"""(define (problem {self.problem_name}) (:domain {self.domain_name})
    (:objects\n    {objects_str}
    )
    (:init\n    {init_str}
    )
    (:goal (and {goal_str}))
)
"""

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return self._str
