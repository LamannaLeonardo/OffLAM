(define (domain miconic)
(:requirements :typing)
(:types floor passenger)

(:predicates
		(origin ?person - passenger ?floor - floor)
		(destin ?person - passenger ?floor - floor)
		(above ?floor1 - floor ?floor2 - floor)
		(boarded ?person - passenger)
		(served ?person - passenger)
		(lift-at ?floor - floor))

(:action board
		:parameters (?f - floor ?p - passenger)
		:precondition (and (lift-at ?f) (origin ?p ?f))
		:effect (boarded ?p))

(:action depart
		:parameters (?f - floor ?p - passenger)
		:precondition (and (lift-at ?f) (destin ?p ?f) (boarded ?p))
		:effect (and (not (boarded ?p)) 	       (served ?p))) ;;drive up

(:action up
		:parameters (?f1 - floor ?f2 - floor)
		:precondition (and (lift-at ?f1) (above ?f1 ?f2))
		:effect (and (lift-at ?f2) (not (lift-at ?f1))))   ;;drive down

(:action down
		:parameters (?f1 - floor ?f2 - floor)
		:precondition (and (lift-at ?f1) (above ?f2 ?f1))
		:effect (and (lift-at ?f2) (not (lift-at ?f1))))

)

