(define (problem pick_and_place_0)
(:domain FOON_micro)
(:init
	; hand/end-effector must be empty (i.e. contains "air"):
	(in hand air)
	; making the robot start at a position not close to any objects:
	(not (at white_box robot))
	(not (at first_white_toy robot))
	(not (at table robot))
	(at robot robot)


	; precondition predicates obtained directly from macro PO:
	(in white_box air)
	(on table white_box)
	(under first_white_toy table)
	(under white_box table)
	(on table first_white_toy)

	; some objects that are on the table for manipulation should be free from collision (i.e. "air" on them):
	(on white_box air)
	(on first_white_toy air)

	(no-perception)
)

(:goal (and
	; hand/end-effector must be also be empty after execution (i.e. contains "air"):
	(in hand air)

	; effect predicates obtained directly from macro PO:
	(in white_box first_white_toy)
	(under first_white_toy white_box)
	(on table white_box)
	(under white_box table)
	(not (in white_box air) )
	(not (under first_white_toy table) )
	(not (on table first_white_toy) )
))

)