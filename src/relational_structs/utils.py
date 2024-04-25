"""Utilities."""

import itertools
from typing import (
    Any,
    Collection,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    TypeVar,
)

import numpy as np

from relational_structs.objects import Object, Type, TypedEntity
from relational_structs.pddl import GroundAtom, Predicate
from relational_structs.state import State


def create_state_from_dict(
    data: Dict[Object, Dict[str, float]], simulator_state: Optional[Any] = None
) -> State:
    """Small utility to generate a state from a dictionary `data` of individual
    feature values for each object.

    A simulator_state for the outputted State may optionally be
    provided.
    """
    state_dict = {}
    for obj, obj_data in data.items():
        obj_vec = []
        for feat in obj.type.feature_names:
            obj_vec.append(obj_data[feat])
        state_dict[obj] = np.array(obj_vec)
    return State(state_dict, simulator_state)


def abstract(state: State, preds: Collection[Predicate]) -> Set[GroundAtom]:
    """Get the atomic representation of the given state (i.e., a set of ground
    atoms), using the given set of predicates.

    Duplicate arguments in predicates are allowed.
    """
    atoms = set()
    for pred in preds:
        for choice in get_object_combinations(list(state), pred.types):
            if pred.holds(state, choice):
                atoms.add(GroundAtom(pred, choice))
    return atoms


_TypedEntityTypeVar = TypeVar("_TypedEntityTypeVar", bound=TypedEntity)


def _get_entity_combinations(
    entities: Collection[_TypedEntityTypeVar], types: Sequence[Type]
) -> Iterator[List[_TypedEntityTypeVar]]:
    """Get all combinations of entities satisfying the given types sequence."""
    sorted_entities = sorted(entities)
    choices = []
    for vt in types:
        this_choices = []
        for ent in sorted_entities:
            if ent.is_instance(vt):
                this_choices.append(ent)
        choices.append(this_choices)
    for choice in itertools.product(*choices):
        yield list(choice)


def get_object_combinations(
    objects: Collection[Object], types: Sequence[Type]
) -> Iterator[List[Object]]:
    """Get all combinations of objects satisfying the given types sequence."""
    return _get_entity_combinations(objects, types)
