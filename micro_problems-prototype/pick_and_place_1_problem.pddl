(define (problem pick_and_place_1)
(:domain FOON_micro)
(:init
	; hand/end-effector must be empty (i.e. contains "air"):
	(in hand air)
	; making the robot start at a position not close to any objects:
	(not (at white_box robot))
	(not (at first_white_toy robot))
	(not (at second_white_toy robot))
	(not (at table robot))
	(at robot robot)


	; precondition predicates obtained directly from macro PO:
	(in white_box first_white_toy)
	(on table white_box)
	(under white_box table)
	(on table second_white_toy)
	(under second_white_toy table)

	; some objects that are on the table for manipulation should be free from collision (i.e. "air" on them):
	(on white_box air)
	(on first_white_toy air)
	(on second_white_toy air)

	(no-perception)
)

(:goal (and
	; hand/end-effector must be also be empty after execution (i.e. contains "air"):
	(in hand air)

	; effect predicates obtained directly from macro PO:
	(in white_box second_white_toy)
	(under second_white_toy white_box)
	(in white_box first_white_toy)
	(on table white_box)
	(under white_box table)
	(not (on table second_white_toy) )
	(not (under second_white_toy table) )
))

)