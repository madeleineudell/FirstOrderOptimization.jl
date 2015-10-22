export PRISMA

function PRISMA(x, # starting point
				Lf, # Lipshitz constant of f
				beta::StepSizeRule, # step sizes
				grad_f,
				prox_g,
				prox_h,
				params::OptParams)

	# initialize
	y        = copy(x)
	xkpp     = copy(x)
	betakpp  = step(beta)
	Lkpp     = Lf + 1/betak
	thetakpp = 1

	# iterate
	for k=1:params.maxiters
		betak, betakpp    = betakpp, step(beta)
		Lk, Lkpp          = Lkpp, Lf + 1/betakpp
		thetak, thetakpp  = thetakpp, 2/(1+sqrt(1+4Lkpp/thetak^2/Lk))
		xk, xkpp          = xkpp, prox_h((1-1/Lk/betak)*y - grad_f(y)/Lk + prox_g(y, betak)/Lk/betak, 1/Lk)
		y                 = xkpp + thetakpp*(1/thetak - 1)*(xkpp - xk)                 
	end

	return x
end