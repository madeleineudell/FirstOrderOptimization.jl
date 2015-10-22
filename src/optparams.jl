## Parameter containers

export OptParams, SimpleParams

abstract OptParams

## Default parameter container

type SimpleParams<:OptParams
	stepsizerule::StepSizeRule
	maxiters::Int
end

## Default stopping rule (stop if decrease is too small)
stop(params::OptParams, objval, oldobjval, args...; kwargs...) = 
	stop(params.stepsizerule) && (oldobjval - objval)/objval < 1e-6 ? true : false
