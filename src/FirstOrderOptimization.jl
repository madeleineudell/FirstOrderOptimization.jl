__precompile__()

module FirstOrderOptimization

# shared among all algorithms

include("stepsize.jl")
include("optparams.jl")
include("utilities.jl")
include("convergence.jl")

# functions

abstract AbstractData
include("MNL.jl")

# algorithms

include("algorithms/gradient_descent.jl")
include("algorithms/prox_grad.jl")
include("algorithms/PRISMA.jl")
include("algorithms/frank_wolfe.jl")
include("algorithms/frank_wolfe_sketched.jl")
# etc

end # module
