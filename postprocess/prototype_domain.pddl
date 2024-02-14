(define (domain universal_FOON)

(:requirements :adl)

(:types 
	object - object
)

(:constants
	; objects from provided FOON subgraph:
	black_toy - object
	first_green_toy - object
	first_pink_toy - object
	green_box - object
	pink_box - object
	second_green_toy - object
	second_pink_toy - object

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
	; description: <black_toy pick_and_place black_toy >
	:parameters ( )
	:precondition (and
		(on table black_toy)
		(under black_toy table)
	)
	:effect (and
		; new effects of executing this functional unit:

		; preconditions that did not get changed in some way:
		(on table black_toy)
		(under black_toy table)
	)
)

(:action pick_and_place_1
	; description: <first_pink_toy pink_box pick_and_place first_pink_toy pink_box >
	:parameters ( )
	:precondition (and
		(in pink_box air)
		(on table pink_box)
		(under first_pink_toy table)
		(on table first_pink_toy)
		(under pink_box table)
	)
	:effect (and
		; new effects of executing this functional unit:
		(in pink_box first_pink_toy)
		(under first_pink_toy pink_box)

		; preconditions that did not get changed in some way:
		(on table pink_box)
		(under pink_box table)

		; negated preconditions:
		(not (in pink_box air) )
		(not (under first_pink_toy table) )
		(not (on table first_pink_toy) )
	)
)

(:action pick_and_place_2
	; description: <second_pink_toy pink_box pick_and_place second_pink_toy pink_box >
	:parameters ( )
	:precondition (and
		(under second_pink_toy table)
		(on table second_pink_toy)
		(on table pink_box)
		(under pink_box table)
		(in pink_box first_pink_toy)
	)
	:effect (and
		; new effects of executing this functional unit:
		(in pink_box second_pink_toy)
		(under second_pink_toy pink_box)

		; preconditions that did not get changed in some way:
		(on table pink_box)
		(under pink_box table)
		(in pink_box first_pink_toy)

		; negated preconditions:
		(not (under second_pink_toy table) )
		(not (on table second_pink_toy) )
	)
)

(:action pick_and_place_3
	; description: <first_green_toy green_box pick_and_place first_green_toy green_box >
	:parameters ( )
	:precondition (and
		(in green_box air)
		(on table first_green_toy)
		(under green_box table)
		(under first_green_toy table)
		(on table green_box)
	)
	:effect (and
		; new effects of executing this functional unit:
		(in green_box first_green_toy)
		(under first_green_toy green_box)

		; preconditions that did not get changed in some way:
		(under green_box table)
		(on table green_box)

		; negated preconditions:
		(not (in green_box air) )
		(not (on table first_green_toy) )
		(not (under first_green_toy table) )
	)
)

(:action pick_and_place_4
	; description: <second_green_toy green_box pick_and_place second_green_toy green_box >
	:parameters ( )
	:precondition (and
		(in green_box first_green_toy)
		(under green_box table)
		(under second_green_toy table)
		(on table second_green_toy)
		(on table green_box)
	)
	:effect (and
		; new effects of executing this functional unit:
		(in green_box second_green_toy)
		(under second_green_toy green_box)

		; preconditions that did not get changed in some way:
		(in green_box first_green_toy)
		(under green_box table)
		(on table green_box)

		; negated preconditions:
		(not (under second_green_toy table) )
		(not (on table second_green_toy) )
	)
)

)