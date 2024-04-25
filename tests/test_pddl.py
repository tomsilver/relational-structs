"""Tests for pddl.py."""

from relational_structs import (
    GroundOperator,
    LiftedOperator,
    Predicate,
    Type,
)


def test_operators():
    """Tests for LiftedOperator() and GroundOperator()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    on = Predicate("On", [cup_type, plate_type], lambda s, o: True)
    not_on = Predicate("NotOn", [cup_type, plate_type], lambda s, o: True)
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


def test_parse_and_create_pddl():
    """Tests for PDDL parsing and creation utilities."""

    # Using the spanner domain because it features hierarchical typing.
    domain_str = """(define (domain spanner)
(:requirements :typing :strips)                
(:types 
	location locatable - object
	man nut spanner - locatable	
)                                           
                                                                               
(:predicates 
	(at ?m - locatable ?l - location)
	(carrying ?m - man ?s - spanner)
	(useable ?s - spanner)
	(link ?l1 - location ?l2 - location)
	(tightened ?n - nut)
	(loose ?n - nut))                                                                                           

(:action walk 
        :parameters (?start - location ?end - location ?m - man)
        :precondition (and (at ?m ?start) 
                           (link ?start ?end))                                                          
        :effect (and (not (at ?m ?start)) (at ?m ?end)))

(:action pickup_spanner 
        :parameters (?l - location ?s - spanner ?m - man)
        :precondition (and (at ?m ?l) 
                           (at ?s ?l))
        :effect (and (not (at ?s ?l))
                     (carrying ?m ?s)))

(:action tighten_nut 
        :parameters (?l - location ?s - spanner ?m - man ?n - nut)
        :precondition (and (at ?m ?l) 
		      	   (at ?n ?l)
			   (carrying ?m ?s)
			   (useable ?s)
			   (loose ?n))
        :effect (and (not (loose ?n))(not (useable ?s)) (tightened ?n)))
)"""

    problem_str = """(define (problem prob0)
 (:domain spanner)
 (:objects 
     bob - man
 spanner1 spanner2 spanner3 - spanner
     nut1 nut2 nut3 - nut
     location1 location2 location3 location4 - location
     shed gate - location
    )
 (:init 
    (at bob shed)
    (at spanner1 location4)
    (useable spanner1)
    (at spanner2 location4)
    (useable spanner2)
    (at spanner3 location1)
    (useable spanner3)
    (loose nut1)
    (at nut1 gate)
    (loose nut2)
    (at nut2 gate)
    (loose nut3)
    (at nut3 gate)
    (link shed location1)
    (link location4 gate)
    (link location1 location2)
    (link location2 location3)
    (link location3 location4)
)
 (:goal
  (and
   (tightened nut1)
   (tightened nut2)
   (tightened nut3)
)))"""
    # TODO
