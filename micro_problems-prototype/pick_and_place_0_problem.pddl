(define (problem pick_and_place_0)
(:domain FOON_micro)
(:init
	; hand/end-effector must be empty (i.e. contains "air"):
	(in hand air)
	; making the robot start at a position not close to any objects:
	(not (at black_toy robot))
	(not (at table robot))
	(at robot robot)


	; precondition predicates obtained directly from macro PO:
	(on table black_toy)
	(under black_toy table)

	; some objects that are on the table for manipulation should be free from collision (i.e. "air" on them):
	(on black_toy air)

	(no-perception)
)

(:goal (and
	; hand/end-effector must be also be empty after execution (i.e. contains "air"):
	(in hand air)

	; effect predicates obtained directly from macro PO:
	(on table black_toy)
	(under black_toy table)
))

)