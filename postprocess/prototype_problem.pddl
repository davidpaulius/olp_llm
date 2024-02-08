(define (problem universal_FOON)

(:domain universal_FOON)

(:init
	(on table green_block)
	(under green_block table)
	(on table blue_block)
	(under blue_block table)
	(on table red_block)
	(under red_block table)
)

(:goal (and
	(on green_block red_block)
	(under red_block green_block)
))

)