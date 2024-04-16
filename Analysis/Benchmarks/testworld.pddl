(define (domain blocks)
  (:requirements :strips :typing)
  (:types block)
  (:predicates (on ?x - block ?y - block)
	       (ontable ?x - block)
	       (clear ?x - block)
	       (handempty)
	       (holding ?x - block)
	       )

  (:action pick-up
	     :parameters (?x - block)
	     :precondition (and )
	     :effect
	     (and ))

  (:action put-down
	     :parameters (?x - block)
	     :precondition (and )
	     :effect
	     (and ))

  (:action stack
	     :parameters (?x - block ?y - block)
	     :precondition (and )
	     :effect
	     (and ))
  (:action unstack
	     :parameters (?x - block ?y - block)
	     :precondition (and )
	     :effect
	     (and ))

)