(define (problem pick_and_place_1)
(:domain FOON_micro)
(:init
	; hand/end-effector must be empty (i.e. contains "air"):
	(in hand air)
	; making the robot start at a position not close to any objects:
	(not (at first_pink_toy robot))
	(not (at pink_box robot))
	(not (at table robot))
	(at robot robot)


	; precondition predicates obtained directly from macro PO:
	(in pink_box air)
	(on table pink_box)
	(under first_pink_toy table)
	(on table first_pink_toy)
	(under pink_box table)

	; some objects that are on the table for manipulation should be free from collision (i.e. "air" on them):
	(on first_pink_toy air)
	(on pink_box air)

	(no-perception)
)

(:goal (and
	; hand/end-effector must be also be empty after execution (i.e. contains "air"):
	(in hand air)

	; effect predicates obtained directly from macro PO:
	(in pink_box first_pink_toy)
	(under first_pink_toy pink_box)
	(on table pink_box)
	(under pink_box table)
	(not (in pink_box air) )
	(not (under first_pink_toy table) )
	(not (on table first_pink_toy) )
))

)