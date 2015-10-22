module FirstOrderOptimization

# shared among all algorithms

include("stepsize.jl")
include("optparams.jl")
include("utilities.jl")

# functions

abstract AbstractData
include("MNL.jl")

# algorithms

include("algorithms/gradient_descent.jl")
include("algorithms/PRISMA.jl")
include("algorithms/frank_wolfe.jl")
# etc

end # module
