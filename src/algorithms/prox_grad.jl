export prox_grad!, ProxGradParams

# solves minimize(f(x) + g(x)) 
# where f is differentiable and the prox of g is easy to evaluate
# prox_g(x, alpha) solves minimize(g(z) + 1/(2*alpha)||x-z||^2
function prox_grad!(x, # the initial value of the variable
	f::Function,
	grad_f::Function,
	g::Function,
	prox_g::Function,
	params::OptParams = SimpleParams(ConstantStepSize(1), 100);
	verbose::Bool = false)

	# initialize
	objective = x -> f(x) + g(x)
	objval = objective(x)
	take_step(x, stepsize) = prox_g(x - stepsize*grad_f(x), stepsize)

	if verbose
		@printf("%10s%12s\n", "iter", "obj")
		@printf("%10d%12.4e\n", 0, objval)
	end

	for iter = 1:params.maxiters
		oldobjval = copy(objval) 
		x, objval = step(params.stepsizerule,
			             objective,
			             take_step,
			             x,
			             objval0 = objval)

		verbose && iter%1==0 && @printf("%10d%12.4e\n", iter, objval)

		if stop(params, objval, oldobjval)
			verbose && println("stopping criterion satisfied")
			break
		end
	end
	return x
end

type ProxGradParams<:OptParams
	stepsizerule::StepSizeRule
	maxiters::Int
end
ProxGradParams() = ProxGradParams(ConstantStepSize(.5), 100)

function step(s::HopefulStepSize, objective::Function, take_step::Function, x0;
	objval0 = objective(x0))
	stepsize = s.initial_stepsize
	x = take_step(x0, stepsize)
	objval = objective(x)
	while objval >= objval0
		@show stepsize *= s.decrease_by
		x = take_step(x0, stepsize)
		objval = objective(x)
	end
	# hope and change!
	if stepsize < s.initial_stepsize
		s.initial_stepsize = stepsize
	else
		s.initial_stepsize *= s.increase_by
	end
	return x, objval
end