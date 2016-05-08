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

function frank_wolfe_sketched(z::AbstractArray, # starting point
	objective::Function, 
	grad_objective::Function,
	constraint::Function, # evaluates constraint function
	alpha::Number, # bound on constraint function
	min_lin_st_constraint::Function, # solves min_{c(x)<=alpha} g \dot x (as a function of alpha and g)
	sketch::AbstractSketch = IdentitySketch(x),
	params::FrankWolfeParams = FrankWolfeParams(),
	ch::ConvergenceHistory = ConvergenceHistory("fw_sketched");
	LB = -Inf,
	UB = Inf,
	verbose = false)
	
	# initialize
	if verbose @printf("%10s%12s%12s%12s%12s%12s\n", "iteration", "UB", "LB", "gap", "rel gap", "time") end

	t = time()
	t0 = copy(t)
	for k=1:params.maxiters

		# function and gradient evaluation
		objval = objective(z)
		G = grad_objective(z)
		# G = A'*g, where g is the gradient wrt the compressed variable
		# so G.factors[1] is the compression operator A
		A = G.factors[1]
		g = vec(G.factors[2])

		# solve fenchel problem
		# Delta is solution to min_{c(X)<=alpha} grad(X_old) \dot X
		Delta = min_lin_st_constraint(G, alpha)
		dz = A*Delta

		# check stopping condition
		UB = min(UB, objval)
 	   # <G, Delta - X> = <A'g, Delta - X> = <g, A*Delta - A*X> = <g, dz - z>
		LB = max(LB, objval + dot(g, dz - z))
		old_t, t = t, time()
		if verbose && k%10==0
			cx = constraint(z)
			@printf("%10d%12.4e%12.4e%12.4e%12.4e%12.4e\n", k, UB, LB, UB - LB, (UB - LB)/min(abs(LB), abs(UB)), t - t0)
		end
		update_ch!(ch, t - old_t; obj = UB, dual_obj = LB)
		stop(params, UB, LB) && break
		t = time()

		# choose step size with stepsize rule
		# eg a = 1/k
		a = step(params.stepsize, objective, z, z-dz; 
			     objval=objval, 
			     normgradsq=UB-LB)

		# take step
		# z = (1-a)*z + a*dz
		scale!(z, 1-a); axpy!(a, dz, z)

		cgd_update!(sketch, Delta, a)
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