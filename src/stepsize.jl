## Step size rules

import Base: step, vecnorm

export StepSizeRule, 
	ConstantStepSize, BacktrackingStepSize, 
	HopefulStepSize, DecreasingStepSize,
	step

## In general

abstract StepSizeRule
stop(ssr::StepSizeRule) = true # whether your should ever stop before maxiters

## More specifically,

type ConstantStepSize<:StepSizeRule
	stepsize
end
step(s::ConstantStepSize, args...; kwargs...) = s.stepsize
ConstantStepSize() = ConstantStepSize(1)

type DecreasingStepSize<:StepSizeRule
	initial_stepsize::AbstractFloat
	iteration_counter::Int
end
function step(s::DecreasingStepSize, args...; kwargs...)
	s.iteration_counter += 1
	return s.initial_stepsize/s.iteration_counter
end
DecreasingStepSize() = DecreasingStepSize(1.0)
DecreasingStepSize(initial_stepsize) = DecreasingStepSize(initial_stepsize, 0)
stop(s::DecreasingStepSize) = false

# this stepsize starts out every iteration at initial_stepsize and backtracks until it finds sufficient decrease
type BacktrackingStepSize<:StepSizeRule
	initial_stepsize::Float64
	decrease_by::Float64
	suff_decrease::Float64
end
function step(s::BacktrackingStepSize, objective::Function, x0, grad_x0;
	objval = objective(x0))
	stepsize = s.initial_stepsize
	normgradsq = vecnorm(grad_x0)^2
	while objective(x0 - stepsize*grad_x0) > objval - s.suff_decrease*stepsize*normgradsq
		stepsize *= s.decrease_by
	end
	return stepsize
end
BacktrackingStepSize() = BacktrackingStepSize(1.0, .9, .5)

# This stepsize gets bigger if we keep getting lucky and smaller otherwise
type HopefulStepSize<:StepSizeRule
	initial_stepsize::Float64
	decrease_by::Float64
	increase_by::Float64
	suff_decrease::Float64
end
function step(s::HopefulStepSize, objective::Function, x0, grad_x0;
	objval = objective(x0))
	stepsize = s.initial_stepsize
	normgradsq = vecnorm(grad_x0)^2
	while objective(x0 - stepsize*grad_x0) > objval - s.suff_decrease*stepsize*normgradsq
		stepsize *= s.decrease_by
	end
	# hope and change!
	if stepsize < s.initial_stepsize
		s.initial_stepsize = stepsize
	else
		s.initial_stepsize *= s.increase_by
	end
	return stepsize
end
HopefulStepSize() = HopefulStepSize(1.0, .8, 1.5, .1)
HopefulStepSize(initial_stepsize) = HopefulStepSize(initial_stepsize, .8, 1.5, .1)
