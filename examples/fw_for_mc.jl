#### Frank wolfe for matrix completion
using FirstOrderOptimization
import Base: axpy!, scale!

## Generate data

# make it deterministic
srand(10) 

m, n, r, rtrue = 100, 100, 2, 2
Xtrue = randn(m, rtrue) * randn(rtrue, n)
tau = nucnorm(Xtrue)

pobs = .5
obs = sprand(m, n, pobs).>0
iobs = find(obs)
A = IndexingOperator(m, n, iobs)
b = A*Xtrue

nonsketched = true

params = FrankWolfeParams(maxiters = 200, reltol = 1e-1, stepsize = BacktrackingStepSize(1,.5,.3)) # DecreasingStepSize(2,1))

srand(10)

objective_sketched(z) = .5*sum((z - b).^2)
grad_objective_sketched(z) = LowRankOperator(A, z-b, transpose = Symbol[:T, :N]) # concretely, should be A'*(z-b)
function min_lin_st_nucnorm_sketched(G, tau)
	u,s,v = svds(Array(G), nsv=1) # for this case, g is a sparse matrix so representing it is O(m)
	return LowRankOperator(-tau*u, v')
end
const_nucnorm(z) = tau # we'll always saturate the constraint, don't bother computing it
# I can't think of a pretty way to compute
# <g, tilde_x - x> = <A'(z-b), u tau v' - x> = <z-b, A(u tau v') - A(x)> = <z-b, A(u tau v') - z>
# dot_g_w_tx_minus_x(g, tilde_x, x) = dot(g.factors[2], g.factors[1]*tilde_x - x)
dot_g_w_tx_minus_x(g, Delta, z) = dot(vec(g.factors[2]), g.factors[1]*Delta - z)
# we end up computing A*Delta twice with this scheme
update_var!(z, Delta, a) = (scale!(z, 1-a); axpy!(a, A*Delta, z))

z = zeros(length(b))
X_sketched = frank_wolfe_sketched(
	z,
	objective_sketched, grad_objective_sketched,
	const_nucnorm,
	tau,
	min_lin_st_nucnorm_sketched,
	AsymmetricSketch(m,n,r),
	params,
#	LB = 0,
	verbose=true
	)

@show vecnorm(Array(X_sketched) - Xtrue) / vecnorm(Xtrue)