(define (domain universal_FOON)

(:requirements :adl)

(:types 
	object - object
)

(:constants
	; objects from provided FOON subgraph:
	celery_stalk - object
	glass - object
	ice_cubes - object
	lemon - object
	pepper - object
	salt - object
	stirrer - object
	tabasco_sauce - object
	tomato_juice - object
	vodka - object
	worcestershire_sauce - object

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

(:action place_0
	; description: <ice_cubes glass place ice_cubes glass >
	:parameters ( )
	:precondition (and
		(under ice_cubes table)
		(on table ice_cubes)
		(in glass air)
		(under glass table)
		(on table glass)
	)
	:effect (and
		; new effects of executing this functional unit:
		(in glass ice_cubes)
		(under ice_cubes glass)

		; preconditions that did not get changed in some way:
		(under glass table)
		(on table glass)

		; negated preconditions:
		(not (in glass air) )
		(not (under ice_cubes table) )
	)
)

(:action pour_1
	; description: <vodka glass pour vodka glass >
	:parameters ( )
	:precondition (and
		(in bottle vodka)
		(under vodka bottle)
		(under glass table)
		(on table glass)
	)
	:effect (and
		; new effects of executing this functional unit:
		(in glass vodka)
		(under vodka glass)

		; preconditions that did not get changed in some way:
		(in bottle vodka)
		(under glass table)
		(on table glass)
	)
)

(:action add_2
	; description: <tomato_juice glass add tomato_juice glass >
	:parameters ( )
	:precondition (and
		(in bottle tomato_juice)
		(under tomato_juice bottle)
		(under glass table)
		(on table glass)
	)
	:effect (and
		; new effects of executing this functional unit:
		(in glass tomato_juice)
		(under tomato_juice glass)

		; preconditions that did not get changed in some way:
		(in bottle tomato_juice)
		(under glass table)
		(on table glass)
	)
)

(:action squeeze_3
	; description: <lemon glass squeeze lemon glass >
	:parameters ( )
	:precondition (and
		(is-whole lemon)
		(under lemon table)
		(on table lemon)
		(under glass table)
		(on table glass)
	)
	:effect (and
		; new effects of executing this functional unit:

		; preconditions that did not get changed in some way:
		(is-whole lemon)
		(under lemon table)
		(on table lemon)
		(under glass table)
		(on table glass)
	)
)

(:action add_4
	; description: <worcestershire_sauce glass add worcestershire_sauce glass >
	:parameters ( )
	:precondition (and
		(in bottle worcestershire_sauce)
		(under worcestershire_sauce bottle)
		(under glass table)
		(on table glass)
	)
	:effect (and
		; new effects of executing this functional unit:
		(in glass worcestershire_sauce)
		(under worcestershire_sauce glass)

		; preconditions that did not get changed in some way:
		(in bottle worcestershire_sauce)
		(under glass table)
		(on table glass)
	)
)

(:action add_5
	; description: <tabasco_sauce glass add tabasco_sauce glass >
	:parameters ( )
	:precondition (and
		(in bottle tabasco_sauce)
		(under tabasco_sauce bottle)
		(under glass table)
		(on table glass)
	)
	:effect (and
		; new effects of executing this functional unit:
		(in glass tabasco_sauce)
		(under tabasco_sauce glass)

		; preconditions that did not get changed in some way:
		(in bottle tabasco_sauce)
		(under glass table)
		(on table glass)
	)
)

(:action sprinkle_6
	; description: <salt pepper glass sprinkle salt pepper glass >
	:parameters ( )
	:precondition (and
		(in shaker salt)
		(under salt shaker)
		(in shaker pepper)
		(under pepper shaker)
		(under glass table)
		(on table glass)
	)
	:effect (and
		; new effects of executing this functional unit:
		(in glass salt)
		(under salt glass)
		(in glass pepper)
		(under pepper glass)

		; preconditions that did not get changed in some way:
		(in shaker salt)
		(in shaker pepper)
		(under glass table)
		(on table glass)
	)
)

(:action stir_ingredients_7
	; description: <stirrer glass stir stirrer glass >
	:parameters ( )
	:precondition (and
		(under stirrer table)
		(on table stirrer)
		(under glass table)
		(on table glass)
	)
	:effect (and
		; new effects of executing this functional unit:

		; preconditions that did not get changed in some way:
		(under stirrer table)
		(on table stirrer)
		(under glass table)
		(on table glass)
	)
)

(:action garnish_8
	; description: <celery_stalk glass garnish celery_stalk glass >
	:parameters ( )
	:precondition (and
		(is-whole celery_stalk)
		(under celery_stalk table)
		(on table celery_stalk)
		(under glass table)
		(on table glass)
	)
	:effect (and
		; new effects of executing this functional unit:
		(in glass celery_stalk)
		(under celery_stalk glass)

		; preconditions that did not get changed in some way:
		(is-whole celery_stalk)
		(under glass table)
		(on table glass)

		; negated preconditions:
		(not (under celery_stalk table) )
	)
)

)