(define (domain universal_FOON)

(:requirements :adl)

(:types 
	object - object
)

(:constants
	; objects from provided FOON subgraph:
	first_green_toy - object
	first_white_toy - object
	green_box - object
	second_green_toy - object
	second_white_toy - object
	third_green_toy - object
	white_box - object

	; objects used in Agostini et al. 2021 - https://arxiv.org/abs/2007.08251
	air - object
	table - object
)

(:predicates
	; object-state predicates (from Agostini et al. 2021 - https://arxiv.org/abs/2007.08251)
	(in ?obj_1 - object ?obj_2 - object)
	(on ?obj_1 - object ?obj_2 - object)
	(under ?obj_1 - object ?obj_2 - object)

	; physical state predicates (from FOON)
	(is-whole ?obj_1 - object)
	(is-diced ?obj_1 - object)
	(is-chopped ?obj_1 - object)
	(is-sliced ?obj_1 - object)
	(is-mixed ?obj_1 - object)
	(is-ground ?obj_1 - object)
	(is-juiced ?obj_1 - object)
	(is-spread ?obj_1 - object)
)

(:action pick_and_place_0
	; description: <first_white_toy white_box pick_and_place first_white_toy white_box >
	:parameters ( )
	:precondition (and
		(in white_box air)
		(on table white_box)
		(under first_white_toy table)
		(under white_box table)
		(on table first_white_toy)
	)
	:effect (and
		; new effects of executing this functional unit:
		(in white_box first_white_toy)
		(under first_white_toy white_box)

		; preconditions that did not get changed in some way:
		(on table white_box)
		(under white_box table)

		; negated preconditions:
		(not (in white_box air) )
		(not (under first_white_toy table) )
		(not (on table first_white_toy) )
	)
)

(:action pick_and_place_1
	; description: <second_white_toy white_box pick_and_place second_white_toy white_box >
	:parameters ( )
	:precondition (and
		(in white_box first_white_toy)
		(on table white_box)
		(under white_box table)
		(on table second_white_toy)
		(under second_white_toy table)
	)
	:effect (and
		; new effects of executing this functional unit:
		(in white_box second_white_toy)
		(under second_white_toy white_box)

		; preconditions that did not get changed in some way:
		(in white_box first_white_toy)
		(on table white_box)
		(under white_box table)

		; negated preconditions:
		(not (on table second_white_toy) )
		(not (under second_white_toy table) )
	)
)

(:action pick_and_place_2
	; description: <first_green_toy green_box pick_and_place first_green_toy green_box >
	:parameters ( )
	:precondition (and
		(under first_green_toy table)
		(on table first_green_toy)
		(on table green_box)
		(under green_box table)
		(in green_box air)
	)
	:effect (and
		; new effects of executing this functional unit:
		(in green_box first_green_toy)
		(under first_green_toy green_box)

		; preconditions that did not get changed in some way:
		(on table green_box)
		(under green_box table)

		; negated preconditions:
		(not (under first_green_toy table) )
		(not (on table first_green_toy) )
		(not (in green_box air) )
	)
)

(:action pick_and_place_3
	; description: <second_green_toy green_box pick_and_place second_green_toy green_box >
	:parameters ( )
	:precondition (and
		(in green_box first_green_toy)
		(on table green_box)
		(under green_box table)
		(on table second_green_toy)
		(under second_green_toy table)
	)
	:effect (and
		; new effects of executing this functional unit:
		(in green_box second_green_toy)
		(under second_green_toy green_box)

		; preconditions that did not get changed in some way:
		(in green_box first_green_toy)
		(on table green_box)
		(under green_box table)

		; negated preconditions:
		(not (on table second_green_toy) )
		(not (under second_green_toy table) )
	)
)

(:action pick_and_place_4
	; description: <third_green_toy green_box pick_and_place third_green_toy green_box >
	:parameters ( )
	:precondition (and
		(in green_box second_green_toy)
		(in green_box first_green_toy)
		(under third_green_toy table)
		(on table green_box)
		(under green_box table)
		(on table third_green_toy)
	)
	:effect (and
		; new effects of executing this functional unit:
		(in green_box third_green_toy)
		(under third_green_toy green_box)

		; preconditions that did not get changed in some way:
		(in green_box second_green_toy)
		(in green_box first_green_toy)
		(on table green_box)
		(under green_box table)

		; negated preconditions:
		(not (under third_green_toy table) )
		(not (on table third_green_toy) )
	)
)

)