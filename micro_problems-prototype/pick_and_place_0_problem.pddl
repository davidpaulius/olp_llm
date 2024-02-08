(define (problem pick_and_place_0)
(:domain FOON_micro)
(:init
	; hand/end-effector must be empty (i.e. contains "air"):
	(in hand air)

	; precondition predicates obtained directly from macro PO:
	(on table green_block)
	(under green_block table)
	(on table blue_block)
	(under blue_block table)

	; some objects that are on the table for manipulation should be free from collision (i.e. "air" on them):
	(on blue_block air)
	(on green_block air)

	(no-perception)
)

(:goal (and
	; hand/end-effector must be also be empty after execution (i.e. contains "air"):
	(in hand air)

	; effect predicates obtained directly from macro PO:
	(on blue_block green_block)
	(under green_block blue_block)
))

)