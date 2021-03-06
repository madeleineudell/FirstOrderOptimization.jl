export gradient_descent!

function gradient_descent!(x, # the initial value of the variable
	objective::Function,
	grad_objective::Function,
	params::OptParams = SimpleParams(ConstantStepSize(1), 100);
	verbose::Bool = false)

	grad = grad_objective(x)
	objval = objective(x)
	if verbose
		@printf("%10s%12s\n", "iter", "obj")
		@printf("%10d%12.4e\n", 0, objval)
	end
	
	for iter = 1:params.maxiters
		curstep = step(params.stepsizerule, objective, x, grad; objval = objval)
		x -= curstep*grad

		grad = grad_objective(x)
		objval, oldobjval = objective(x), objval
		verbose && iter%1==0 && @printf("%10d%12.4e\n", iter, objval)

		if stop(params, objval, oldobjval, grad)
			verbose && println("stopping criterion satisfied")
			break
		end
	end
	return x
end