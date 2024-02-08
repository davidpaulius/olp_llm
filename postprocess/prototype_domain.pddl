(define (domain universal_FOON)

(:requirements :adl)

(:types 
	object - object
)

(:constants
	; objects from provided FOON subgraph:
	blue_block - object
	green_block - object
	red_block - object

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
	; description: <green_block blue_block pick_and_place green_block blue_block >
	:parameters ( )
	:precondition (and
		(on table green_block)
		(under green_block table)
		(on table blue_block)
		(under blue_block table)
	)
	:effect (and
		; new effects of executing this functional unit:
		(on blue_block green_block)
		(under green_block blue_block)
	)
)

(:action pick_and_place_1
	; description: <red_block green_block pick_and_place red_block green_block >
	:parameters ( )
	:precondition (and
		(on table red_block)
		(under red_block table)
		(on blue_block green_block)
		(under green_block blue_block)
	)
	:effect (and
		; new effects of executing this functional unit:
		(on green_block red_block)
		(under red_block green_block)
	)
)

)