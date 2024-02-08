(define (domain FOON_micro)

	(:requirements :adl)

	(:types
		robot - object
	)

	(:constants
		air - object
		hand - object
		robot - robot
	)

	(:predicates
		(at ?obj1 - object ?obj2 - object)

		; object-centered predicates defining geometric action constraints:
		(in ?obj1 - object ?obj2 - object)
		(on ?obj1 - object ?obj2 - object)
		(under ?obj1 - object ?obj2 - object)

		; physical state predicates (from FOON)
		(is-whole ?obj - object)
		(is-ground ?obj - object)
		(is-juiced ?obj - object)
		(is-spread ?obj - object)
		(is-diced ?obj - object)
		(is-chopped ?obj - object)
		(is-sliced ?obj - object)
		(is-mixed ?obj - object)

		; properties for objects (based on grid-spaces):
		(is-short ?obj - object)
		(is-long ?obj - object)
		(is-wide ?obj - object)

		(no-perception)
	)

	;define actions here
	(:action pick
		:parameters (
			?obj - object
			?surface - object
		)
		:precondition (and
			(at ?surface robot)

			(on ?obj air)
			(under ?obj ?surface)
			(on ?surface ?obj)
			(in hand air)
		)
		:effect (and
			(in hand ?obj)
			(on ?obj hand)
			(under ?obj air)
			(on ?surface air)

			(not (in hand air))
			(not (on ?obj air))
			(not (on ?surface ?obj))
			(not (under ?obj ?surface))
		)
	)

	;define actions here
	(:action pick_rotated
		; picking an object that is upside down (i.e., "air" under it, "surface" on it)
		:parameters (
			?obj - object
			?surface - object
		)
		:precondition (and
			(at ?surface robot)

			(under ?obj air)
			(on ?obj ?surface)
			(on ?surface ?obj)
			(in hand air)
		)
		:effect (and
			(in hand ?obj)
			(on ?obj hand)
			(on ?obj air)
			(on ?surface air)

			(not (in hand air))
			(not (on ?surface ?obj))
			(not (on ?obj ?surface))
		)
	)

	(:action place
		; releasing object from hand - this will free the hand while placing the object on some surface:
		:parameters (
			?obj - object
			?surface - object
		)
		:precondition (and
			(at ?surface robot)

			(on ?obj hand)
			(under ?obj air)
			(in hand ?obj)
			(on ?surface air)

			; size rules for object placement:
			(or
				; a short object can be placed on any tile size:
				(and
					(is-short ?obj)
					(or (is-short ?surface) (is-long ?surface) (is-wide ?surface))
				)
				; a long object can only be placed on a long or wide tile:
				(and
					(is-long ?obj)
					(or (is-long ?surface) (is-wide ?surface))
				)
				(and
					(is-wide ?obj)
					(is-wide ?surface)
				)

				; in the case where we have no object sizes (i.e., no perception):
				(no-perception)
			)
		)
		:effect (and
			(in hand air)
			(on ?obj air)

			; the object goes on the surface:
			(on ?surface ?obj)
			(under ?obj ?surface)

			(not (in hand ?obj))
			(not (on ?obj hand))
			(not (on ?surface air))
			(not (under ?obj air))
		)
	)

	(:action place_rotated
		:parameters (
			?obj - object
			?surface - object
		)
		:precondition (and
			; from an object-centric approach, the object that is upside down
			;  will have "air" under it and it will be "under" the surface it is on.
			(at ?surface robot)

			(in hand ?obj)
			(on ?obj hand)
			(under ?obj air)
			(on ?obj air)
			(on ?surface air)
			; size rules for object placement:
			(or
				; a short object can be placed on any tile size:
				(and
					(is-short ?obj)
					(or (is-short ?surface) (is-long ?surface) (is-wide ?surface))
				)
				; a long object can only be placed on a long or wide tile:
				(and
					(is-long ?obj)
					(or (is-long ?surface) (is-wide ?surface))
				)
				(and
					(is-wide ?obj)
					(is-wide ?surface)
				)

				; in the case where we have no object sizes (i.e., no perception):
				(no-perception)
			)
		)
		:effect (and
			; this action should be seen as one that will just combine picking and placing.
			(in hand air)
			(on ?obj air)
			(under ?obj ?surface)

			(not (in hand ?obj))
			(not (on ?obj hand))
			(not (on ?surface air))
			(not (under ?obj air))

		)
	)

	(:action pour
		:parameters (
			?obj - object
			?source - object
			?target - object
		)
		:precondition (and
			(at ?target robot)

			; source container (which is in the hand) initially contains some kind of object:
			(under ?source air)
			(or
				(in ?source ?obj)
				(on ?source ?obj)
			)
			(under ?obj ?source)
			(in hand ?source)

			; target container should be collision-free (not caring about what is inside):
			(on ?target air)
		)
		:effect (and
			(in ?source air)
			(in ?target ?obj)
			(under ?obj ?target)

			(not (in ?source ?obj))
			(not (on ?source ?obj))

			(not (in ?target air))
			(not (under ?obj ?source))
		)
	)

	(:action insert
		; releasing object from hand - this will free the hand while placing the object on some surface:
		:parameters (
			?obj - object
			?container - object
		)
		:precondition (and
			(at ?container robot)

			(in hand ?obj)
			(on ?obj hand)
			(under ?obj air)
			(on ?container air)
		)
		:effect (and
			; object is no longer in the hand:
			(in hand air)
			(not (in hand ?obj))
			(not (on ?obj hand))

			; object no longer has "air" underneath it:
			(on ?obj air)
			(not (under ?obj air))

			; container now contains object, so it is not empty:
			(in ?container ?obj)
			(under ?obj ?container)
			(not (in ?container air))
		)
	)

	(:action move
		:parameters (
			?robot - robot
			?loc1 - object
			?loc2 - object
		)
		:precondition (and
			(at ?loc1 ?robot)
			(not (at ?loc2 ?robot))
		)
		:effect (and
			(at ?loc2 ?robot)
			(not (at ?loc1 ?robot))
		)
	)

)