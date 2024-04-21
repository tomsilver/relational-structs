"""Utilities."""

from typing import Any, Dict, Optional

import numpy as np

from relational_structs.structs import Object, State


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
