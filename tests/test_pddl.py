"""Tests for pddl.py."""

import pytest

from relational_structs import (
    GroundOperator,
    LiftedOperator,
    PDDLDomain,
    PDDLProblem,
    Predicate,
    Type,
)
from relational_structs.utils import parse_pddl_plan


def test_operators():
    """Tests for LiftedOperator() and GroundOperator()."""
    cup_type = Type("cup_type")
    plate_type = Type("plate_type")
    on = Predicate("On", [cup_type, plate_type])
    not_on = Predicate("NotOn", [cup_type, plate_type])
    cup_var = cup_type("?cup")
    plate_var = plate_type("?plate")
    parameters = [cup_var, plate_var]
    preconditions = {not_on([cup_var, plate_var])}
    add_effects = {on([cup_var, plate_var])}
    delete_effects = {not_on([cup_var, plate_var])}

    lifted_operator = LiftedOperator(
        "Pick", parameters, preconditions, add_effects, delete_effects
    )
    assert (
        str(lifted_operator)
        == repr(lifted_operator)
        == """(:action Pick
    :parameters (?cup - cup_type ?plate - plate_type)
    :precondition (and (NotOn ?cup ?plate))
    :effect (and (On ?cup ?plate)
        (not (NotOn ?cup ?plate)))
)"""
    )
    assert lifted_operator.short_str == "Pick(?cup, ?plate)"

    assert isinstance(hash(lifted_operator), int)
    lifted_operator2 = LiftedOperator(
        "Pick", parameters, preconditions, add_effects, delete_effects
    )
    assert lifted_operator == lifted_operator2
    lifted_operator3 = LiftedOperator(
        "Pick2", parameters, preconditions, add_effects, delete_effects
    )
    assert lifted_operator < lifted_operator3
    assert lifted_operator3 > lifted_operator

    cup = cup_type("cup")
    plate = plate_type("plate")
    ground_operator = lifted_operator.ground((cup, plate))
    assert isinstance(ground_operator, GroundOperator)
    assert ground_operator.parent is lifted_operator
    assert (
        str(ground_operator)
        == repr(ground_operator)
        == """(:action Pick
    :parameters (cup - cup_type plate - plate_type)
    :precondition (and (NotOn cup plate))
    :effect (and (On cup plate)
        (not (NotOn cup plate)))
)"""
    )
    assert ground_operator.short_str == "Pick(cup, plate)"
    ground_operator2 = lifted_operator2.ground((cup, plate))
    ground_operator3 = lifted_operator3.ground((cup, plate))
    assert ground_operator == ground_operator2
    assert ground_operator < ground_operator3
    assert ground_operator3 > ground_operator

    # Test missing parameters.
    with pytest.raises(AssertionError) as e:
        LiftedOperator("Pick", [], preconditions, add_effects, delete_effects)
    assert "missing from operator parameters" in str(e)


def test_parse_and_create_pddl():
    """Tests for PDDL parsing and creation utilities."""

    # Using the spanner domain because it features hierarchical typing.
    domain_str = """(define (domain spanner)
    (:requirements :typing)
    (:types 
    man nut spanner - locatable
    locatable location - object)

    (:predicates
    (at ?x0 - locatable ?x1 - location)
    (carrying ?x0 - man ?x1 - spanner)
    (link ?x0 - location ?x1 - location)
    (loose ?x0 - nut)
    (tightened ?x0 - nut)
    (useable ?x0 - spanner)
    )

    (:action pickup_spanner
    :parameters (?l - location ?s - spanner ?m - man)
    :precondition (and (at ?m ?l)
        (at ?s ?l))
    :effect (and (carrying ?m ?s)
        (not (at ?s ?l)))
)

  (:action tighten_nut
    :parameters (?l - location ?s - spanner ?m - man ?n - nut)
    :precondition (and (at ?m ?l)
        (at ?n ?l)
        (carrying ?m ?s)
        (loose ?n)
        (useable ?s))
    :effect (and (tightened ?n)
        (not (loose ?n))
        (not (useable ?s)))
)

  (:action walk
    :parameters (?start - location ?end - location ?m - man)
    :precondition (and (at ?m ?start)
        (link ?start ?end))
    :effect (and (at ?m ?end)
        (not (at ?m ?start)))
)
)
"""

    pddl_domain = PDDLDomain.parse(domain_str)
    assert str(pddl_domain) == domain_str

    problem_str = """(define (problem prob0) (:domain spanner)
    (:objects
    bob - man
    gate - location
    location1 - location
    location2 - location
    location3 - location
    location4 - location
    nut1 - nut
    nut2 - nut
    nut3 - nut
    shed - location
    spanner1 - spanner
    spanner2 - spanner
    spanner3 - spanner
    )
    (:init
    (at bob shed)
    (at nut1 gate)
    (at nut2 gate)
    (at nut3 gate)
    (at spanner1 location4)
    (at spanner2 location4)
    (at spanner3 location1)
    (link location1 location2)
    (link location2 location3)
    (link location3 location4)
    (link location4 gate)
    (link shed location1)
    (loose nut1)
    (loose nut2)
    (loose nut3)
    (useable spanner1)
    (useable spanner2)
    (useable spanner3)
    )
    (:goal (and (tightened nut1)
    (tightened nut2)
    (tightened nut3)))
)
"""

    pddl_problem = PDDLProblem.parse(problem_str, pddl_domain)
    assert str(pddl_problem) == problem_str

    # Test parse_pddl_plan().
    pddl_plan = ["(walk location1 location2 bob)", "(walk location2 location3 bob)"]
    ground_op_plan = parse_pddl_plan(pddl_plan, pddl_domain, pddl_problem)
    ground_op_plan_short = [o.short_str for o in ground_op_plan]
    assert ground_op_plan_short == [
        "walk(location1, location2, bob)",
        "walk(location2, location3, bob)",
    ]
