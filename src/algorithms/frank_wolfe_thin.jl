import Optim.optimize
import Base: axpy!, scale!

export frank_wolfe_thin

# type FrankWolfeParams<:OptParams
# 	maxiters::Int
# 	abstol::AbstractFloat
# 	stepsize::StepSizeRule
# end
# stop(p::FrankWolfeParams, eps) = eps < p.abstol ? true : false
# FrankWolfeParams() = FrankWolfeParams(50, 1e-10, DecreasingStepSize(2,1))

function frank_wolfe_thin(X::LowRankOperator, # starting point
	objective::Function,
	grad_objective::Function,
	constraint::Function, # evaluates constraint function
	alpha::Number, # bound on constraint function
	min_lin_st_constraint::Function, # solves min_{c(x)<=alpha} g \dot x (as a function of alpha and g)
	params::FrankWolfeParams = FrankWolfeParams(),
	ch::ConvergenceHistory = ConvergenceHistory("fw_thin");
	LB = -Inf,
	UB = Inf,
	verbose = false)

	# initialize
	if verbose @printf("%10s%12s%12s%12s%12s%12s\n", "iteration", "UB", "LB", "gap", "rel gap", "time") end

	t = time()
	t0 = copy(t)
	for k=1:params.maxiters

		# function and gradient evaluation
		objval = objective(X)
		G = grad_objective(X)
		# G = A'*g, where g is the gradient wrt the compressed variable
		# so G.factors[1] is the compression operator A

		# solve fenchel problem
		# Delta is solution to min_{c(X)<=alpha} dot(G, X)
		# linearized_obj is dot(G, Delta)
		Delta, linearized_obj = min_lin_st_constraint(G, alpha)

		# check stopping condition
		UB = min(UB, objval)
 	  # <G, Delta - X> = <G, Delta> - <G, X> = linearized_obj - <G, X>
		# here G is a sparse matrix and X is a low rank operator with three factors (in SVD form)
		LB = max(LB, objval + linearized_obj - dot(G, X))
		old_t, t = t, time()
		if verbose && k%10==0
			cx = constraint(X)
			@printf("%10d%12.4e%12.4e%12.4e%12.4e%12.4e\n", k, UB, LB, UB - LB, (UB - LB)/min(abs(LB), abs(UB)), t - t0)
		end
		update_ch!(ch, t - old_t; obj = UB, dual_obj = LB)
		stop(params, UB, LB) && break
		t = time()

		# choose step size with stepsize rule
		# eg a = 1/k
		a = step(params.stepsize)

		# take step
		# X = (1-a)*X + a*Delta
		X = thin_update!(X, Delta, 1-a, a)
	end
	return X
end
