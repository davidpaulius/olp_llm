(define (problem universal_FOON)

(:domain universal_FOON)

(:init
	(on table first_white_toy)
	(under first_white_toy table)
	(in white_box air)
	(under white_box table)
	(on table white_box)
	(on table second_white_toy)
	(under second_white_toy table)
	(on table first_green_toy)
	(under first_green_toy table)
	(in green_box air)
	(under green_box table)
	(on table green_box)
	(on table second_green_toy)
	(under second_green_toy table)
	(on table third_green_toy)
	(under third_green_toy table)
)

(:goal (and
	(in green_box first_green_toy)
	(under green_box table)
	(on table green_box)
	(in green_box third_green_toy)
))

)