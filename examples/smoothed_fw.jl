#### Frank wolfe for matrix completion
using FirstOrderOptimization
using PyPlot
import Base: axpy!, scale!

## Generate data

# make it deterministic
srand(10) 

m, n, r, rtrue = 100, 100, 10, 2
X = randn(m, rtrue) * randn(rtrue, n)
smooth_inf = log(min(m,n))
# smooth_inf and smooth_1 are conjugate 
# 1/p + 1/q = 1 => p = 1/(1-1/q) = q / (q - 1)
smooth_1 = smooth_inf / (smooth_inf - 1)
u,s,v = svd(X)
tau = sum(s)
tau_smooth = sum(s.^(smooth_1))^(1/smooth_1)

pobs = .5
obs = sprand(m, n, pobs).>0
iobs = find(obs)
A = IndexingOperator(m, n, iobs)
b = A*X

srand(10)

objective_sketched(z) = .5*sum((z - b).^2)
grad_objective_sketched(z) = LowRankOperator(A, z-b, transpose = Symbol[:T, :N]) # concretely, should be A'*(z-b)

const_nucnorm(z) = tau # we'll always saturate the constraint, don't bother computing it
function min_lin_st_nucnorm_sketched(g, tau)
	u,s,v = svds(Array(g), nsv=1) # for this case, g is a sparse matrix so representing it is O(m)
	return LowRankOperator(-tau*u, v')
end
# I can't think of a pretty way to compute
# <g, tilde_x - x> = <A'(z-b), u tau v' - x> = <z-b, A(u tau v') - A(x)> = <z-b, A(u tau v') - z>
# dot_g_w_tx_minus_x(g, tilde_x, x) = dot(g.factors[2], g.factors[1]*tilde_x - x)
dot_g_w_tx_minus_x(g, Delta, z) = dot(vec(g.factors[2]), g.factors[1]*Delta - z)
# we end up computing A*Delta twice with this scheme
update_var!(z, Delta, a) = (scale!(z, 1-a); axpy!(a, A*Delta, z))

z = zeros(length(b))
ch = ConvergenceHistory("fw_sketched")
X_sketched = frank_wolfe_sketched(
	z,
	objective_sketched, grad_objective_sketched,
	const_nucnorm,
	tau,
	min_lin_st_nucnorm_sketched,
	AsymmetricSketch(m,n,r),
	FrankWolfeParams(100, 1e-2, DecreasingStepSize(2,1)),
	ch,
	dot_g_w_tx_minus_x = dot_g_w_tx_minus_x,
	update_var! = update_var!,
	#LB = 0,
	verbose=true
	)

figure()
# yscale("log")
plot(ch.objective, label="fw")
# plot(ch.objective, label="fw")
xlabel("Iterations")
ylabel("Objective")

## now smooth version

for (nsv, nsv_label) = [(r, "r"), (Int(ceil(smooth_inf)), "log(n)"), (min(m,n), "n")]

	function min_lin_st_nucnorm_sketched(g, tau, 
			smooth_inf = smooth_inf, 
			nsv = nsv)
		u,s,v = svds(Array(g), nsv=nsv) # for this case, g is a sparse matrix so representing it is O(m)
		s = s.^smooth_inf
		s = s / sum(s.^smooth_1).^(1/smooth_1) * -tau # normalize so the smooth_1 norm of the solution is tau
		return LowRankOperator(u, spdiagm(s), v')
	end
	const_nucnorm(z) = tau_smooth # we'll always saturate the constraint, don't bother computing it

	srand(10)

	ch_smooth = ConvergenceHistory("smooth_fw_sketched")
	z = zeros(length(b))
	X_sketched_smooth = frank_wolfe_sketched(
		z,
		objective_sketched, grad_objective_sketched,
		const_nucnorm,
		tau_smooth,
		min_lin_st_nucnorm_sketched,
		AsymmetricSketch(m,n,r),
		FrankWolfeParams(100, 1e-2, DecreasingStepSize(2,1)),
		ch_smooth,
		dot_g_w_tx_minus_x = dot_g_w_tx_minus_x,
		update_var! = update_var!,
		#LB = 0,
		verbose=true
		)

	plot(ch_smooth.objective, label="smoothed fw top "*nsv_label)
	#@show(vecnorm(Array(X_sketched_smooth) - X))
end
legend()