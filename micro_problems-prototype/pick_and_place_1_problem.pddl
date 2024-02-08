(define (problem pick_and_place_1)
(:domain FOON_micro)
(:init
	; hand/end-effector must be empty (i.e. contains "air"):
	(in hand air)

	; precondition predicates obtained directly from macro PO:
	(on table red_block)
	(under red_block table)
	(on blue_block green_block)
	(under green_block blue_block)

	; some objects that are on the table for manipulation should be free from collision (i.e. "air" on them):
	(on red_block air)
	(on green_block air)

	(no-perception)
)

(:goal (and
	; hand/end-effector must be also be empty after execution (i.e. contains "air"):
	(in hand air)

	; effect predicates obtained directly from macro PO:
	(on green_block red_block)
	(under red_block green_block)
))

)