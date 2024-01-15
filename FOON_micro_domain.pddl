(define (domain FOON_micro)

	(:requirements :adl)

	(:types
		ingredient - object
		surface - object
		container - object
		spoon - object
		knife - object
	)

	(:constants
		air - object
		hand - object

		; table grid-spaces as will be used in the simulation:
		;   a table object needs to be "grounded" to one of these spaces:
		table - surface
		table_00 - surface
		table_01 - surface
		table_02 - surface
		table_03 - surface
		table_04 - surface
		table_05 - surface
		table_06 - surface
		table_07 - surface
		table_08 - surface
		table_09 - surface
		table_10 - surface
		table_11 - surface
		table_12 - surface
		table_13 - surface
		table_14 - surface
		table_15 - surface
		table_16 - surface
		table_17 - surface
		table_18 - surface
		table_19 - surface
		table_20 - surface

		; containers for cooking:
		cup - container
		drinking_glass - container
		bottle - container
		bowl - container
		jar - container
		mixing_bowl - container
		shaker - container
		pot - container
		measuring_cup - container
		tin_can - container
		plate - container

		stove - surface
		cutting_board - surface

		; ingredients for greek salad:
		tomato - ingredient
		oregano - ingredient
		salt - ingredient
		olive - ingredient
		feta_cheese - ingredient
		green_pepper - ingredient
		cucumber - ingredient
		onion - ingredient
		olive_oil - ingredient
		black_pepper - ingredient

		; objects for Bloody Mary:
		celery - ingredient
		worcestershire_sauce - ingredient
		tabasco_sauce - ingredient
		vodka - ingredient
		lemon - ingredient
		ice - ingredient
		tomato_juice - ingredient
		lemon_juice - ingredient

		cup_lemon_juice - container
		cup_worcestershire_sauce - container
		cup_ice - container
		tin_can_tomato_juice - container
		bottle_vodka - container
		shaker_black_pepper - container
		shaker_salt - container
		bottle_olive_oil - container
		bowl_feta_cheese - container
		bowl_olive - container
		
		; ingredients for miso soup:
		green_onion - ingredient
		tofu - ingredient
		miso - ingredient
		water - ingredient
		dashi - ingredient
		broth - ingredient

		; final products:
		salad - object
		greek_salad - object
		cocktail - object
		bloody_mary - object
		miso_soup - object

		; tools:
		knife - knife
		spoon - spoon

	)

	(:predicates
		;todo: define predicates here
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

	(:action move_on_top
		; placing object without releasing it from the hand:
		:parameters (
			?obj - object 
			?surface - object
		)
		:precondition (and
			(on ?obj hand)
			(under ?obj air)
			(in hand ?obj)
			(on ?surface air)
		)
		:effect (and
			(on ?surface ?obj)
			(under ?obj ?surface)
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
			(in hand ?obj)
			(on ?obj hand)

			(under ?obj air)
			(on ?obj air)
			(on ?surface air)
			(no-perception)
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

	(:action place_on_short_obj
		; releasing object from hand - this will free the hand while placing the object on some surface:
		:parameters (
			?obj - object
			?surface - object
		)
		:precondition (and
			(on ?obj hand)
			(under ?obj air)
			(in hand ?obj)
			(on ?surface air)

			; the object needs to be placed on an appropriately sized spot:
			(is-short ?surface)
			(is-short ?obj)
		)
		:effect (and
			(in hand air)
			(on ?obj air)
			(on ?surface ?obj)
			(under ?obj ?surface)

			(not (in hand ?obj))
			(not (on ?obj hand))
			(not (on ?surface air))
			(not (under ?obj air))
		)
	)

	(:action place_on_long_obj
		; releasing object from hand - this will free the hand while placing the object on some surface:
		:parameters (
			?obj - object
			?surface - object
		)
		:precondition (and
			(on ?obj hand)
			(under ?obj air)
			(in hand ?obj)
			(on ?surface air)

			; the object needs to be placed on an appropriately sized spot:
			(is-long ?surface)
			(is-long ?obj)
		)
		:effect (and
			(in hand air)
			(on ?obj air)
			(on ?surface ?obj)
			(under ?obj ?surface)

			(not (in hand ?obj))
			(not (on ?obj hand))
			(not (on ?surface air))
			(not (under ?obj air))
		)
	)

	(:action place
		; releasing object from hand - this will free the hand while placing the object on some surface:
		:parameters (
			?obj - object
			?surface - object
		)
		:precondition (and
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

	(:action place_on_wide_obj
		; releasing object from hand - this will free the hand while placing the object on some surface:
		:parameters (
			?obj - object
			?surface - object
		)
		:precondition (and
			(on ?obj hand)
			(under ?obj air)
			(in hand ?obj)
			(on ?surface air)
		
			; the object needs to be placed on an appropriately sized spot:
			(is-wide ?surface)
			(is-wide ?obj)
		)
		:effect (and
			(in hand air)
			(on ?obj air)
			(on ?surface ?obj)
			(under ?obj ?surface)

			(not (in hand ?obj))
			(not (on ?obj hand))
			(not (on ?surface air))
			(not (under ?obj air))
		)
	)

	(:action place_rotated_long_obj
		:parameters (
			?obj - object
			?surface - obj
		)
		:precondition (and
			; from an object-centric approach, the object that is upside down
			;  will have "air" under it and it will be "under" the surface it is on.
			(in hand ?obj)
			(on ?obj hand)

			(under ?obj air)
			(on ?obj air)
			(on ?surface air)

			(is-long ?obj)
			(is-long ?surface)
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

	(:action place_rotated_short_obj
		:parameters (
			?obj - object
			?surface - obj
		)
		:precondition (and
			; from an object-centric approach, the object that is upside down
			;  will have "air" under it and it will be "under" the surface it is on.
			(in hand ?obj)
			(on ?obj hand)

			(under ?obj air)
			(on ?obj air)
			(on ?surface air)

			(is-short ?obj)
			(is-short ?surface)
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

	(:action insert
		; releasing object from hand - this will free the hand while placing the object on some surface:
		:parameters (
			?obj - object 
			?container - container
		)
		:precondition (and
			(on ?obj hand)
			(in hand ?obj)
			(under ?obj air)
			(on ?container air)
			(is-whole ?obj)
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

	(:action pour
		:parameters (
			?obj - ingredient
			?source - object
			?target - container
		)
		:precondition (and
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

	(:action sprinkle
		:parameters (
			?obj - ingredient
			?source - object
			?target - container
		)
		:precondition (and
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
			(in ?target ?obj)
			(under ?obj ?target)
			(not (in ?target air))
		)
	)

	(:action stir
		:parameters (
			?utensil - spoon
			?container - container
		)
		:precondition (and
			; the container is free from collision and is ready for mixing:
			(on ?container air)

			; the hand contains the tool and the tool is free for manipulation:
			(on ?utensil hand)
			(in hand ?utensil)
			(under ?utensil air)
		)
		:effect (and
			; here, we say that the container has been mixed, but this is for low-level:
			(is-mixed ?container)
		)
	)

	(:action chop
		:parameters (
			?obj - object
			?surface - surface
			?knife - knife
		)
		:precondition (and
			; making sure the knife is in the hand:
			(on ?knife hand)
			(in hand ?knife)
	
			; making sure the object is on a surface:
			(or
				(in ?surface ?obj)
				(on ?surface ?obj)
			)
			(under ?obj ?surface)
		)
		:effect (and
			; the object becomes in the chopped state:
			(is-chopped ?obj)
			(not (is-whole ?obj))
	
			; (on ?obj air)
			; (under ?knife air)
			; (not (under ?knife ?obj))
			; (not (on ?obj ?knife))
		)
	)

	(:action slice
		:parameters (
			?obj - object
			?surface - surface
			?knife - knife
		)
		:precondition (and
			; making sure the knife is in the hand:
			(in hand ?knife)
			(on ?knife hand)
	
			; making sure the object is on a surface:
			(or
				(in ?surface ?obj)
				(on ?surface ?obj)
			)
			(under ?obj ?surface)
		)
		:effect (and
			; the object becomes in the chopped state:
			(is-sliced ?obj)
			(not (is-whole ?obj))
		)
	)

	; (:action place_on_surface
	; 	; releasing object from hand - this will free the hand while placing the object on some surface:
	; 	:parameters ( ?obj - object ?surface - surface
	; 	)
	; 	:precondition (and
	; 		(on ?obj hand)
	; 		(under ?obj air)
	; 		(in hand ?obj)
	; 		(on ?surface air)
	; 		(no-perception)
	; 	)
	; 	:effect (and
	; 		(in hand air)
	; 		(on ?obj air)
	; 		(on ?surface ?obj)
	; 		(under ?obj ?surface)
	; 		(not (in hand ?obj))
	; 		(not (on ?obj hand))
	; 		(not (on ?surface air))
	; 		(not (under ?obj air))
	; 	)
	; )

	; (:action pour_object_surface
	; 	:parameters ( ?obj - ingredient ?source - surface ?target - container
	; 	)
	; 	:precondition (and
	; 		; source container (which is in the hand) initially contains some kind of object:
	; 		(under ?source air)
	; 		(on ?source ?obj)
	; 		(under ?obj ?source)
	; 		(in hand ?source)

	; 		; target container should be collision-free (not caring about what is inside):
	; 		(on ?target air)
	; 	)
	; 	:effect (and
	; 		(on ?source air)
	; 		(not (in ?source ?obj))
	; 		(not (in ?target air))
	; 		(not (under ?obj ?source))

	; 		(in ?target ?obj)
	; 		(under ?obj ?target)
	; 	)
	; )

	; (:action spread_object
	; 	:parameters ( ?obj - ingredient ?utensil - spoon ?surface - object
	; 	)
	; 	:precondition (and
	; 		(on ?utensil hand)
	; 		(in hand ?utensil)
	; 		(under ?utensil air)

	; 		(on ?surface ?obj)
	; 		(under ?obj ?surface)
	; 	)
	; 	:effect (and
	; 		; the object becomes in the chopped state:
	; 		(is-spread ?obj)
	; 	)
	; )

	; (:action scoop_object
	; 	:parameters ( ?obj - ingredient ?utensil - spoon ?source - container
	; 	)
	; 	:precondition (and
	; 		; utensil (e.g., spoon) should be present in the hand...
	; 		(under ?utensil air)
	; 		(on ?utensil hand)
	; 		(in hand ?utensil)

	; 		; ... and it should have no ingredients present in it:
	; 		(in ?utensil air)

	; 		; source contains some object that is being transferred using the tool:
	; 		(on ?source air)
	; 		(in ?source ?obj)
	; 		(under ?obj ?source)
	; 	)
	; 	:effect (and
	; 		(in ?utensil ?obj)
	; 		(under ?obj ?utensil)
	; 		(not (in ?utensil air))
	; 	)
	; )

	; (:action becomes_mixed
	; 	:parameters ( ?obj - object ?container - object
	; 	)
	; 	:precondition (and
	; 		(in ?container ?obj)
	; 		(is-mixed ?container)
	; 	)
	; 	:effect (and
	; 		(not (in ?container ?obj))
	; 	)
	; )

)
