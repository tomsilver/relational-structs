"""Tests for structs.py."""

import numpy as np
import pytest

from relational_structs.structs import Object, State, Type, Variable


def test_object_type():
    """Tests for Type class."""
    name = "test"
    feats = ["feat1", "feat2"]
    my_type = Type(name, feats)
    assert my_type.name == name
    assert my_type.dim == len(my_type.feature_names) == len(feats)
    assert my_type.feature_names == feats
    assert isinstance(hash(my_type), int)
    name = "test2"
    feats = ["feat3"]
    my_type2 = Type(name, feats, parent=my_type)
    assert my_type2.name == name
    assert my_type2.dim == len(my_type2.feature_names) == len(feats)
    assert my_type2.feature_names == feats
    assert isinstance(hash(my_type2), int)
    assert my_type2.parent == my_type
    name = "test2"
    feats = ["feat3"]
    my_type3 = Type(name, feats, parent=my_type)  # same as my_type2
    obj = my_type("obj1")
    assert obj.is_instance(my_type)
    assert not obj.is_instance(my_type2)
    assert not obj.is_instance(my_type3)
    obj = my_type2("obj2")
    assert obj.is_instance(my_type)
    assert obj.is_instance(my_type2)
    assert obj.is_instance(my_type3)


def test_object():
    """Tests for Object class."""
    my_name = "obj"
    my_type = Type("type", ["feat1", "feat2"])
    obj = my_type(my_name)
    assert isinstance(obj, Object)
    assert obj.name == my_name
    assert obj.type == my_type
    assert str(obj) == repr(obj) == "obj:type"
    assert isinstance(hash(obj), int)
    with pytest.raises(AssertionError):
        Object("?obj", my_type)  # name cannot start with ?


def test_variable():
    """Tests for Variable class."""
    my_name = "?var"
    my_type = Type("type", ["feat1", "feat2"])
    var = my_type(my_name)
    assert isinstance(var, Variable)
    assert var.name == my_name
    assert var.type == my_type
    assert str(var) == repr(var) == "?var:type"
    assert isinstance(hash(var), int)
    with pytest.raises(AssertionError):
        Variable("var", my_type)  # name must start with ?


def test_state():
    """Tests for State class."""
    type1 = Type("type1", ["feat1", "feat2"])
    type2 = Type("type2", ["feat3", "feat4", "feat5"])
    obj3 = type1("obj3")
    obj7 = type1("obj7")
    obj1 = type2("obj1")
    obj2 = type2("obj2")
    obj1_dup = type2("obj1")
    obj4 = type2("obj4")
    obj9 = type2("obj9")
    assert obj7 > obj1
    assert obj1 < obj4
    assert obj1 < obj3
    assert obj1 != obj9
    assert obj1 == obj1_dup
    with pytest.raises(AssertionError):
        State({obj3: [1, 2, 3]})  # bad feature vector dimension
    state = State(
        {
            obj3: [1, 2],
            obj7: [3, 4],
            obj1: [5, 6, 7],
            obj4: [8, 9, 10],
            obj9: [11, 12, 13],
        }
    )
    sorted_objs = list(state)
    assert sorted_objs == [obj1, obj3, obj4, obj7, obj9]
    assert state[obj9] == state.data[obj9] == [11, 12, 13]
    assert state.get(obj3, "feat2") == 2
    assert state.get(obj1, "feat4") == 6
    with pytest.raises(ValueError):
        state.get(obj3, "feat3")  # feature not in list
    with pytest.raises(ValueError):
        state.get(obj1, "feat1")  # feature not in list
    vec = state.vec([obj3, obj1])
    assert vec.shape == (5,)
    assert list(vec) == [1, 2, 5, 6, 7]
    state.set(obj3, "feat2", 122)
    assert state.get(obj3, "feat2") == 122
    state2 = state.copy()
    assert state.allclose(state2)
    state2[obj1][0] = 999
    state2.set(obj1, "feat5", 991)
    assert state != state2  # changing copy doesn't change original
    assert state2.get(obj1, "feat3") == 999
    assert state2[obj1][2] == 991
    state3 = State({obj3: np.array([1, 2])})
    state3.copy()  # try copying with numpy array
    # Test state copy with a simulator state.
    state4 = State({obj3: np.array([1, 2])}, simulator_state="dummy")
    assert state4.simulator_state == "dummy"
    assert state4.copy().simulator_state == "dummy"
    # Cannot use allclose with non-None simulator states.
    with pytest.raises(NotImplementedError):
        state4.allclose(state3)
    with pytest.raises(NotImplementedError):
        state3.allclose(state4)
    # Test state vec with no objects
    vec = state.vec([])
    assert vec.shape == (0,)
    # Test allclose
    state2 = State(
        {
            obj3: [1, 122],
            obj7: [3, 4],
            obj1: [5, 6, 7],
            obj4: [8, 9, 10],
            obj9: [11, 12, 13],
        }
    )
    assert state.allclose(state2)
    state2 = State(
        {
            obj3: [1, 122],
            obj7: [3, 4],
            obj1: [5, 6, 7],
            obj4: [8.3, 9, 10],
            obj9: [11, 12, 13],
        }
    )
    assert not state.allclose(state2)  # obj4 state is different
    state2 = State({obj3: [1, 122], obj7: [3, 4], obj4: [8, 9, 10], obj9: [11, 12, 13]})
    assert not state.allclose(state2)  # obj1 is missing
    state2 = State(
        {
            obj3: [1, 122],
            obj7: [3, 4],
            obj1: [5, 6, 7],
            obj2: [5, 6, 7],
            obj4: [8, 9, 10],
            obj9: [11, 12, 13],
        }
    )
    assert not state.allclose(state2)  # obj2 is extra
    # Test pretty_str
    assert (
        state2.pretty_str()
        == """################# STATE ################
type: type1      feat1    feat2
-------------  -------  -------
obj3                 1      122
obj7                 3        4

type: type2      feat3    feat4    feat5
-------------  -------  -------  -------
obj1                 5        6        7
obj2                 5        6        7
obj4                 8        9       10
obj9                11       12       13
########################################
"""
    )
    # Test including simulator_state
    state_with_sim = State({}, "simulator_state")
    assert state_with_sim.simulator_state == "simulator_state"
    assert state.simulator_state is None
    return state
