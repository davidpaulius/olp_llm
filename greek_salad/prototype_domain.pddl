(define (domain universal_FOON)

(:requirements :adl)

(:types 
	object - object
)

(:constants
	; objects from provided FOON subgraph:
	bowl - object
	cucumber - object
	feta_cheese - object
	knife - object
	olive - object
	olive_oil - object
	onion - object
	oregano - object
	salad - object
	salad_tongs - object
	tomato - object

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

(:action slice_tomato_0
	; description: <tomato knife slice tomato knife >
	:parameters ( )
	:precondition (and
		(is-whole tomato)
		(under tomato table)
		(on table tomato)
		(under knife table)
		(on table knife)
	)
	:effect (and
		; new effects of executing this functional unit:
		(is-sliced tomato)

		; preconditions that did not get changed in some way:
		(under tomato table)
		(on table tomato)
		(under knife table)
		(on table knife)

		; negated preconditions:
		(not (is-whole tomato) )
	)
)

(:action slice_cucumber_1
	; description: <cucumber knife slice cucumber knife >
	:parameters ( )
	:precondition (and
		(is-whole cucumber)
		(under cucumber table)
		(on table cucumber)
		(under knife table)
		(on table knife)
	)
	:effect (and
		; new effects of executing this functional unit:
		(is-sliced cucumber)

		; preconditions that did not get changed in some way:
		(under cucumber table)
		(on table cucumber)
		(under knife table)
		(on table knife)

		; negated preconditions:
		(not (is-whole cucumber) )
	)
)

(:action slice_onion_2
	; description: <onion knife slice onion knife >
	:parameters ( )
	:precondition (and
		(is-whole onion)
		(under onion table)
		(on table onion)
		(under knife table)
		(on table knife)
	)
	:effect (and
		; new effects of executing this functional unit:
		(is-sliced onion)

		; preconditions that did not get changed in some way:
		(under onion table)
		(on table onion)
		(under knife table)
		(on table knife)

		; negated preconditions:
		(not (is-whole onion) )
	)
)

(:action slice_olive_3
	; description: <olive knife slice olive knife >
	:parameters ( )
	:precondition (and
		(is-whole olive)
		(under olive table)
		(on table olive)
		(under knife table)
		(on table knife)
	)
	:effect (and
		; new effects of executing this functional unit:
		(is-sliced olive)

		; preconditions that did not get changed in some way:
		(under olive table)
		(on table olive)
		(under knife table)
		(on table knife)

		; negated preconditions:
		(not (is-whole olive) )
	)
)

(:action crumble_4
	; description: <feta_cheese crumble feta_cheese >
	:parameters ( )
	:precondition (and
		(is-whole feta_cheese)
		(under feta_cheese table)
		(on table feta_cheese)
	)
	:effect (and
		; new effects of executing this functional unit:

		; preconditions that did not get changed in some way:
		(is-whole feta_cheese)
		(under feta_cheese table)
		(on table feta_cheese)
	)
)

(:action combine_5
	; description: <tomato cucumber onion olive feta_cheese bowl combine tomato cucumber onion olive feta_cheese bowl >
	:parameters ( )
	:precondition (and
		(is-sliced tomato)
		(under tomato table)
		(on table tomato)
		(is-sliced cucumber)
		(under cucumber table)
		(on table cucumber)
		(is-sliced onion)
		(under onion table)
		(on table onion)
		(is-sliced olive)
		(under olive table)
		(on table olive)
		(under feta_cheese table)
		(on table feta_cheese)
		(in bowl air)
		(under bowl table)
		(on table bowl)
	)
	:effect (and
		; new effects of executing this functional unit:
		(in bowl tomato)
		(under tomato bowl)
		(in bowl cucumber)
		(under cucumber bowl)
		(in bowl onion)
		(under onion bowl)
		(in bowl olive)
		(under olive bowl)
		(in bowl feta_cheese)
		(under feta_cheese bowl)

		; preconditions that did not get changed in some way:
		(is-sliced tomato)
		(is-sliced cucumber)
		(is-sliced onion)
		(is-sliced olive)
		(under bowl table)
		(on table bowl)

		; negated preconditions:
		(not (in bowl air) )
		(not (under tomato table) )
		(not (in bowl air) )
		(not (under cucumber table) )
		(not (in bowl air) )
		(not (under onion table) )
		(not (in bowl air) )
		(not (under olive table) )
		(not (in bowl air) )
		(not (under feta_cheese table) )
	)
)

(:action drizzle_6
	; description: <olive_oil salad drizzle olive_oil salad >
	:parameters ( )
	:precondition (and
		(in bottle olive_oil)
		(under olive_oil bottle)
		(in bowl salad)
		(under salad bowl)
	)
	:effect (and
		; new effects of executing this functional unit:
		(on salad olive_oil)
		(under olive_oil salad)
		(under salad table)
		(on table salad)

		; preconditions that did not get changed in some way:
		(in bottle olive_oil)
		(in bowl salad)
	)
)

(:action sprinkle_7
	; description: <oregano salad sprinkle oregano salad >
	:parameters ( )
	:precondition (and
		(in container oregano)
		(under oregano container)
		(under salad table)
		(on table salad)
	)
	:effect (and
		; new effects of executing this functional unit:
		(on salad oregano)
		(under oregano salad)

		; preconditions that did not get changed in some way:
		(in container oregano)
		(under salad table)
		(on table salad)
	)
)

(:action toss_8
	; description: <salad salad_tongs toss salad salad_tongs >
	:parameters ( )
	:precondition (and
		(under salad table)
		(on table salad)
		(under salad_tongs table)
		(on table salad_tongs)
	)
	:effect (and
		; new effects of executing this functional unit:
		(is-mixed LOC)

		; preconditions that did not get changed in some way:
		(under salad table)
		(on table salad)
		(under salad_tongs table)
		(on table salad_tongs)
	)
)

)