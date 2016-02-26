export PRISMA, PrismaParams, PrismaStepsize

function PRISMA(x, # starting point
				Lf, # Lipshitz constant of f
				grad_f,
				prox_g,
				prox_h,
				obj,
				params::OptParams=PrismaParams(PrismaStepsize(), 10))

	# initialize
	ssr      = params.stepsizerule
	if ssr.initial_stepsize == Inf
		R = sqrt(obj(x)) # estimate of distance to solution
		ssr.initial_stepsize = 2*R/Lf
	end
	y        = copy(x)
	xkpp     = copy(x)
	betakpp  = step(ssr)
	Lkpp     = Lf + 1/betakpp
	thetak   = 1
	thetakpp = 1 # XXX see orabona's code

	if params.verbose > 0
		@printf("%10s%12s\n", "iter", "obj")
		@printf("%10d%12.4e\n", 0, obj(xkpp))
	end

	# iterate
	for k=1:params.maxiter
		betak, betakpp    = betakpp, step(ssr)
		Lk, Lkpp          = Lkpp, Lf + 1/betakpp
		thetak, thetakpp  = thetakpp, 2/(1+sqrt(1+4Lkpp/thetak^2/Lk))
		xk, xkpp          = xkpp, prox_h((1-1/Lk/betak)*y - grad_f(y)/Lk + prox_g(y, betak)/Lk/betak, 1/Lk)
		y                 = xkpp + thetakpp*(1/thetak - 1)*(xkpp - xk)                 
		if params.verbose > 0
			@printf("%10d%12.4e\n", k, obj(xkpp))
		end
		if stop(params, xk, xkpp)
			break
		end
	end

	return xkpp
end

type PrismaStepsize<:StepSizeRule
	initial_stepsize::AbstractFloat
	iteration_counter::Int
end
function step(s::PrismaStepsize, args...; kwargs...)
	s.iteration_counter += 1
	return s.initial_stepsize/(s.iteration_counter)
end
PrismaStepsize() = PrismaStepsize(Inf)
PrismaStepsize(initial_stepsize) = PrismaStepsize(initial_stepsize, 0)

type PrismaParams<:OptParams
	stepsizerule::StepSizeRule
	maxiter::Int
	verbose::Int
	reltol::Float64
end
PrismaParams(;stepsize::StepSizeRule=PrismaStepsize(),
	          maxiter::Int=100,
	          verbose::Int=1,
	          reltol::Float64=1e-5) = PrismaParams(stepsize,maxiter,verbose,reltol)
stop(s::PrismaParams, xk, xkpp) = vecnorm(xk - xkpp)/vecnorm(xk) < s.reltol ? true : false	
