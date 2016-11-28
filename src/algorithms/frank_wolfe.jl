import Optim.optimize

export frank_wolfe, FrankWolfeParams

type FrankWolfeParams<:OptParams
	maxiters::Int
	abstol::AbstractFloat
	reltol::AbstractFloat
	stepsize::StepSizeRule
end
stop(p::FrankWolfeParams, UB, LB) = (UB - LB < p.abstol) || ((UB - LB)/max(abs(LB), abs(UB)) < p.reltol) ? true : false
FrankWolfeParams(;maxiters=50, abstol=1e-4, reltol=1e-2, stepsize=DecreasingStepSize(2,1)) =
	FrankWolfeParams(maxiters, abstol, reltol, stepsize)

function frank_wolfe(x::Array{Float64,2}, # starting point
	objective, grad_objective,
	constraint, # evaluates constraint
	delta::AbstractFloat, # bound on constraint function
	min_lin_st_constraint, # solves min_{c(x)<=delta} g \dot x (as a function of delta and g)
	params::FrankWolfeParams = FrankWolfeParams();
	verbose = false,
	B = -Inf) # stopping condition

	# initialize
	objval = objective(x)
	if verbose
		@printf("%10s%12s%12s\n", "iter", "obj", "constr")
		@printf("%10d%12.4e%12.4e\n", 0, objval, constraint(x))
	end

	for k=1:params.maxiters
		g = grad_objective(x)

		# step 1
		# tilde_x is solution to min_{c(theta)<=delta} grad(theta_old) \dot theta
		tilde_x = min_lin_st_constraint(g, delta)

		# step 2

		# x is solution to minimize_alpha o(a*x_old + (1-a)*tilde_x)
		# f(a) = objective((1-a)*x + a*tilde_x)
		# g!(a,g) = (g[1] = dot(grad_objective((1-a)*x + a*tilde_x), - x + tilde_x); g)
		# a = optimize(f, g!, [.5], method = :l_bfgs)x, x_old = , x

		# fixed stepsize
		a = step(params.stepsize)

		# bisection
		# f(a) = dot(grad_objective((1-a)*x + a*tilde_x), tilde_x - x)
		# a = find_zero(f, 0, 1/sqrt(k), tol=1e-2, maxiters=10)

		x_old = copy(x)
		x = (1-a)*x + a*tilde_x
		# the following line segfaults:
		# x, x_old = (1-a)*x + a*tilde_x, x
		objval, objval_old = objective(x), objval
		cx = constraint(x)
		if verbose && k%1==0
			@printf("%10d%12.4e%12.4e\n", k, objval, cx)
		end

		# stopping condition
		tilde_B = objval_old + dot(g, tilde_x - x_old)
		B = max(B, tilde_B)
		stop(params, objval, B) && break
	end
	return x
end
