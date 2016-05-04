import Optim.optimize
import Base: axpy!, scale!
include("../sketch/sketch.jl")
include("../sketch/operator.jl")

export frank_wolfe_sketched

# type FrankWolfeParams<:OptParams
# 	maxiters::Int
# 	abstol::AbstractFloat
# 	stepsize::StepSizeRule
# end
# stop(p::FrankWolfeParams, eps) = eps < p.abstol ? true : false
# FrankWolfeParams() = FrankWolfeParams(50, 1e-10, DecreasingStepSize(2,1))

function frank_wolfe_sketched(x::AbstractArray, # starting point
	objective, grad_objective,
	constraint, # evaluates constraint function
	delta::Number, # bound on constraint function
	min_lin_st_constraint, # solves min_{c(x)<=delta} g \dot x (as a function of delta and g)
	sketch::AbstractSketch = IdentitySketch(x),
	params::FrankWolfeParams = FrankWolfeParams(),
	ch::ConvergenceHistory = ConvergenceHistory("fw_sketched");
	dot_g_w_tx_minus_x = (g, tilde_x, x) -> dot(g, tilde_x - x),
	update_var! = (x, tilde_x, a) -> (scale!(x, 1-a); axpy!(a, tilde_x, x)),
	LB = -Inf,
	UB = Inf,
	verbose = false)
	
	# initialize
	if verbose @printf("%10s%12s%12s%12s%12s\n", "iteration", "UB", "LB", "duality gap", "time") end

	t = time()
	t0 = copy(t)
	for k=1:params.maxiters

		# function and gradient evaluation
		objval = objective(x)
		g = grad_objective(x)
		
		# solve fenchel problem
		# tilde_x is solution to min_{c(theta)<=delta} grad(theta_old) \dot theta
		tilde_x = min_lin_st_constraint(g, delta)
		
		# check stopping condition
		UB = min(UB, objval)
		LB = max(LB, objval + dot_g_w_tx_minus_x(g, tilde_x, x))
		old_t, t = t, time()
		if verbose && k%10==0
			cx = constraint(x)
			@printf("%10d%12.4e%12.4e%12.4e%12.4e\n", k, UB, LB, UB - LB, t - t0)
		end
		update_ch!(ch, t - old_t; obj = UB, dual_obj = LB)
		stop(params, UB - LB) && break
		t = time()

		# choose stepsize
		
		# linesearch w/bfgs
		# x is solution to minimize_alpha o(a*x_old + (1-a)*tilde_x)
		# f(a) = objective((1-a)*x + a*tilde_x)
		# g!(a,g) = (g[1] = dot(grad_objective((1-a)*x + a*tilde_x), - x + tilde_x); g)
		# a = optimize(f, g!, [.5], method = :l_bfgs)

		# linesearch w/bisection
		# f(a) = dot(grad_objective((1-a)*x + a*tilde_x), tilde_x - x)
		# a = zero(f, 0, 1/sqrt(k), tol=1e-2, maxiters=10)
		
		# predetermined stepsize
		a = step(params.stepsize)

		# take step
		# x = (1-a)*x + a*tilde_x
		update_var!(x, tilde_x, a)

		cgd_update!(sketch, tilde_x, a)
	end
	return reconstruct(sketch)
end

# find a zero of f over [a, b]
function zero(f, a, b; tol=1e-9, maxiters=1000)
    f(a) < 0 || return a
    f(b) > 0 || return b
    for i=1:maxiters
        mid = a + (b-a)/2
        fmid = f(mid)
        if abs(fmid) < tol
            return mid
        end
        if f(mid) < 0
            a = mid
        else
            b = mid
        end
    end
    warn("hit maximum iterations in bisection search")
    return (b-a)/2
end