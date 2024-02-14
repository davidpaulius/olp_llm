(define (problem universal_FOON)

(:domain universal_FOON)

(:init
	(on table black_toy)
	(under black_toy table)
	(on table first_pink_toy)
	(under first_pink_toy table)
	(in pink_box air)
	(on table pink_box)
	(under pink_box table)
	(on table second_pink_toy)
	(under second_pink_toy table)
	(on table first_green_toy)
	(under first_green_toy table)
	(in green_box air)
	(on table green_box)
	(under green_box table)
	(on table second_green_toy)
	(under second_green_toy table)
)

(:goal (and
	(in green_box second_green_toy)
	(under green_box table)
	(on table green_box)
))

)