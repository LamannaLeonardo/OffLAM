(define (domain Rover)
(:requirements :typing)
(:types rover waypoint store camera mode lander objective)

(:predicates (at ?x - rover ?y - waypoint)
             (at-lander ?x - lander ?y - waypoint)
             (can-traverse ?r - rover ?x - waypoint ?y - waypoint)
	     (equipped-for-soil-analysis ?r - rover)
             (equipped-for-rock-analysis ?r - rover)
             (equipped-for-imaging ?r - rover)
             (empty ?s - store)
             (have-rock-analysis ?r - rover ?w - waypoint)
             (have-soil-analysis ?r - rover ?w - waypoint)
             (full ?s - store)
	     (calibrated ?c - camera ?r - rover)
	     (supports ?c - camera ?m - mode)
             (available ?r - rover)
             (visible ?w - waypoint ?p - waypoint)
             (have-image ?r - rover ?o - objective ?m - mode)
             (communicated-soil-data ?w - waypoint)
             (communicated-rock-data ?w - waypoint)
             (communicated-image-data ?o - objective ?m - mode)
	     (at-soil-sample ?w - waypoint)
	     (at-rock-sample ?w - waypoint)
             (visible-from ?o - objective ?w - waypoint)
	     (store-of ?s - store ?r - rover)
	     (calibration-target ?i - camera ?o - objective)
	     (on-board ?i - camera ?r - rover)
	     (channel-free ?l - lander)

)


(:action navigate
:parameters (?o1 - rover ?o2 - waypoint ?o3 - waypoint)
:precondition (and (can-traverse ?o1 ?o2 ?o3) (available ?o1) (at ?o1 ?o2)
                (visible ?o2 ?o3)
	    )
:effect (and (not (at ?o1 ?o2)) (at ?o1 ?o3)
		)
)

(:action sample-soil
:parameters (?o1 - rover ?o2 - store ?o3 - waypoint)
:precondition (and (at ?o1 ?o3) (at-soil-sample ?o3) (equipped-for-soil-analysis ?o1) (store-of ?o2 ?o1) (empty ?o2)
		)
:effect (and (not (empty ?o2)) (full ?o2) (have-soil-analysis ?o1 ?o3) (not (at-soil-sample ?o3))
		)
)

(:action sample-rock
:parameters (?o1 - rover ?o2 - store ?o3 - waypoint)
:precondition (and (at ?o1 ?o3) (at-rock-sample ?o3) (equipped-for-rock-analysis ?o1) (store-of ?o2 ?o1)(empty ?o2)
		)
:effect (and (not (empty ?o2)) (full ?o2) (have-rock-analysis ?o1 ?o3) (not (at-rock-sample ?o3))
		)
)

(:action drop
:parameters (?o1 - rover ?o2 - store)
:precondition (and (store-of ?o2 ?o1) (full ?o2)
		)
:effect (and (not (full ?o2)) (empty ?o2)
	)
)

(:action calibrate
 :parameters (?o1 - rover ?o2 - camera ?o3 - objective ?o4 - waypoint)
 :precondition (and (equipped-for-imaging ?o1) (calibration-target ?o2 ?o3) (at ?o1 ?o4) (visible-from ?o3 ?o4)(on-board ?o2 ?o1)
		)
 :effect (calibrated ?o2 ?o1)
)




(:action take-image
 :parameters (?o1 - rover ?o2 - waypoint ?o3 - objective ?o4 - camera ?o5 - mode)
 :precondition (and (calibrated ?o4 ?o1)
			 (on-board ?o4 ?o1)
                      (equipped-for-imaging ?o1)
                      (supports ?o4 ?o5)
			  (visible-from ?o3 ?o2)
                     (at ?o1 ?o2)
               )
 :effect (and (have-image ?o1 ?o3 ?o5)(not (calibrated ?o4 ?o1))
		)
)


(:action communicate-soil-data
 :parameters (?o1 - rover ?o2 - lander ?o3 - waypoint ?o4 - waypoint ?o5 - waypoint)
 :precondition (and (at ?o1 ?o4)(at-lander ?o2 ?o5)(have-soil-analysis ?o1 ?o3)
                   (visible ?o4 ?o5)(available ?o1)(channel-free ?o2)
            )
 :effect (and (not (available ?o1))(not (channel-free ?o2))(channel-free ?o2)
		(communicated-soil-data ?o3)(available ?o1)
	)
)

(:action communicate-rock-data
 :parameters (?o1 - rover ?o2 - lander ?o3 - waypoint ?o4 - waypoint ?o5 - waypoint)
 :precondition (and (at ?o1 ?o4)(at-lander ?o2 ?o5)(have-rock-analysis ?o1 ?o3)
                   (visible ?o4 ?o5)(available ?o1)(channel-free ?o2)
            )
 :effect (and (not (available ?o1))(not (channel-free ?o2))(channel-free ?o2)(communicated-rock-data ?o3)(available ?o1)
          )
)


(:action communicate-image-data
 :parameters (?o1 - rover ?o2 - lander ?o3 - objective ?o4 - mode ?o5 - waypoint ?o6 - waypoint)
 :precondition (and (at ?o1 ?o5)(at-lander ?o2 ?o6)(have-image ?o1 ?o3 ?o4)(visible ?o5 ?o6)(available ?o1)(channel-free ?o2)
            )
 :effect (and (not (available ?o1))(not (channel-free ?o2))(channel-free ?o2)(communicated-image-data ?o3 ?o4)(available ?o1)
          )
)

)
