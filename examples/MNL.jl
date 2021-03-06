using FirstOrderOptimization
import StatsBase: sample, WeightVec

# Currently, the gradient descent algorithm seems to converge to the optimum,
# and the code seems to have all the invariances etc expected.
# However, it seems like (for small problem sizes) using the lambda
# from our paper, the true Theta used to generate the data 
# does *not* yield the minimal objective value: 
# the ThetaHat computed has a *much* lower objective value
# And for large problem sizes even *computing* the objective value (and gradient) take a long time.
# Also, I find it odd that the initialization from the dropping convexity paper
# works as well as it does. Perhaps the gradient computation is wrong?
# Or is 100 x 100 just too small to see any interesting behavior?

# With the nucler norm constrained- (rather than regularized-) problem,
# the code below searches over a range of constraints. It seems like
# the code converges nicely (if slowly).
# When the constraint is set at the nuclear norm of Thetastar,
# the negloglik converges to negloglik(Thetastar).
# But ||ThetaHat - Thetastar||/||Thetastar|| is still more than .1.

################ Now run it

### Generate data

# make it deterministic
srand(10) 

m, n, k, ktrue = 300, 300, 2, 1
Theta = randn(m, ktrue) * randn(ktrue, n)
Theta = Theta .- (Theta * ones(n,1)/n)
Theta = Theta / var(Theta)

nobs = 3*m*n #int(10*m*n) # number of observations
K = 4 #int(.05*n) # size of sets
sets = Array(Array{Int,1}, nobs) # set presented
rows = Array(Int, nobs)      # customer who arrived
cols = Array(Int, nobs)      # product they picked
for iobs = 1:nobs
	it = sample(1:m)
	St = sample(1:n, K, replace=false, ordered=true)
	wv = WeightVec(Float64[exp(-Theta[it, j]) for j in St])
	jt = St[sample(wv)]
	rows[iobs], cols[iobs], sets[iobs] = it, jt, St
end
data = MNLdata(nobs, m, n, sets, rows, cols)
println("nucnorm(Theta) = ", nucnorm(Theta))
println("negloglik(Theta) = ", negloglik(Theta, data))

relrmse(xhat, x) = vecnorm(xhat - x) / vecnorm(x) # sqrt(mean((xhat - x).^2)) / sqrt(mean(x.^2))

# per the theorem, we want
# lambda >= 32 sqrt(K*d*log(d)/m/n/nobs)
d = (m+n)/2
lambda = 0

# certify optimality of Theta when solving
# minimize(loss(Theta) + lambda*nucnorm(Theta))
# G is the gradient of the loss at Theta
# lambda is the coefficient in front of the nuclear norm regularization
# here's the idea: (taking lambda = 1 in the discussion, and hoping all nonzero eigenvalues are distinct)
# writing down the optimality conditions of the problem, we see
# 1) on the span (U,V) of the solution Theta, 
#    the gradient had better be diagonal with entries all 1
# 2) orthogonal to that span, all we know is that it has to have operator norm <=1
function certify_optimality(G, Theta, lambda; TOL=1e-4)
	# TOL is numerical zero, for our purposes
	u,s,v = svd(Theta)
	ug,sg,vg = svd(G)
	k = sum(s.>=TOL) 
	U = u[:,1:k]
	V = v[:,1:k]
	GVVT = (G*V)*V'
	W = (G - U*(U'*G) - GVVT + U*(U'*GVVT))
	notdiag = vecnorm(T + lambda*eye(k)) / k
	notsmall = max(norm(W)-lambda, 0)
	optcrit = notdiag + notsmall
	if optcrit <= TOL
		println("Recovered value is optimal")
	else
		println("Recovered value is not optimal: violates optimality condition 
			on the range of the solution by $notdiag 
			and off that range by $notsmall")
	end
end

stop(p::SimpleParams, objval, oldobjval, args...; kwargs...) = false

## Frank Wolfe
if false
	println("Fitting a rank $k MNL model with $m rows and $n columns to $nobs observations with choice sets of size $K
	using Frank Wolfe")

	t = zeros(m,n)
	ThetaHat = copy(t)
	@printf("%12s%12s%12s%12s\n", "obj", "delta", "constr", "rel rmse")	
	nt = nucnorm(Theta)
	for delta=nt:round(Int, nt/10):nt*2
		# saturate constraint
		if delta > 0 && nucnorm(ThetaHat) > .01
			t = copy(ThetaHat)
			t *= delta/nucnorm(ThetaHat)
		end
		ThetaHat = frank_wolfe(t, delta,
				x->negloglik(x, data), 
				x->grad_negloglik(x, data), 
				nucnorm, min_lin_st_nucnorm, 
				FrankWolfeParams(),
				verbose=false)

		ThetaHatZeroed = ThetaHat .- (ThetaHat * ones(n,1)/n) 
		rmset = relrmse(ThetaHatZeroed, Theta)
		@printf("%12.2e%12.2e%12.2e%12.2e\n", negloglik(ThetaHat, data), delta, nucnorm(ThetaHat), rmset)
	end
end

## Prox grad on Theta
if false
	println("Fitting a rank $k MNL model with $m rows and $n columns to $nobs observations with choice sets of size $K
	using the proximal gradient method")
	println("nucnorm(Theta) = ", nucnorm(Theta))
	println("negloglik(Theta) = ", negloglik(Theta, data))

	@show gamma_u = 256*sqrt(K*k*(m+n)*log(m+n)/(m*n*nobs))

	ThetaHat = zeros(m,n)
	@printf("%12s%12s%12s%12s%12s\n", "lambda", "loss", "reg", "obj", "rel rmse")	
	nt = nucnorm(Theta)
	for lambda=logspace(1,-4,6)
		ThetaHat = prox_grad!(ThetaHat,
				x->negloglik(x, data), 
				x->grad_negloglik(x, data), 
				x->lambda*nucnorm(x),
				(x, alpha)->prox_nucnorm(x, alpha*lambda),
				ProxGradParams(ConstantStepSize(gamma_u), 100),
				verbose=true)
		# consider also HopefulStepSize(gamma_u)

		ThetaHatZeroed = ThetaHat .- (ThetaHat * ones(n,1)/n) 
		rmset = relrmse(ThetaHatZeroed, Theta)
		loss, reg = negloglik(ThetaHat, data), lambda*nucnorm(ThetaHat)
		@printf("%12.2e%12.2e%12.2e%12.2e%12.2e\n", lambda, loss, reg, loss+reg, rmset)
	end
end

## Factored prox grad
if false
	println("Fitting a rank $k MNL model with $m rows and $n columns to $nobs observations with choice sets of size $K
	using factored proximal gradient descent and varying the regularization")
	println("nucnorm(Theta) = ", nucnorm(Theta))
	println("negloglik(Theta) = ", negloglik(Theta, data))
	@printf("%12s%12s%12s%12s%12s\n", "lambda", "loss", "reg", "obj", "rel rmse")	

	# initialize
	lambda0 = 32 * sqrt(K*d*log(d)/m/n/nobs)
	objective = U -> (negloglik(U, data) + lambda0*nucnorm(U))
	grad_objective = U -> (grad_negloglik(U, data) + lambda0 * grad_nucnorm(U))
	U = initialize_dropping_convexity(grad_objective, zeros(m,n), k)

	@show gamma_u = 256*sqrt(K*k*(m+n)*log(m+n)/(m*n*nobs))

	@printf("%12s%12s%12s%12s%12s\n", "lambda", "loss", "reg", "obj", "rel rmse")	
	for lambda=logspace(1,-4,6)
		U = prox_grad!(U,
				x->negloglik(x, data), 
				x->grad_negloglik(x, data), 
				x->lambda*nucnorm(x),
				(x, alpha)->prox_nucnorm(x, alpha*lambda),
				ProxGradParams(HopefulStepSize(), 100),
				verbose=true)
		ThetaHat = rectangular_part(U)
		ThetaHatZeroed = ThetaHat .- (ThetaHat * ones(n,1)/n) 
		rmset = relrmse(ThetaHatZeroed, Theta)
		loss, reg = negloglik(ThetaHat, data), lambda*nucnorm(ThetaHat)
		@printf("%12.2e%12.2e%12.2e%12.2e%12.2e\n", lambda, loss, reg, loss+reg, rmset)
	end
end

## Factored gradient descent
if true
	println("Fitting a rank $k MNL model with $m rows and $n columns to $nobs observations with choice sets of size $K
	using factored gradient descent and varying the regularization")
	@printf("%12s%12s%12s%12s%12s\n", "lambda", "loss", "reg", "obj", "rel rmse")	

	# initialize
	lambda0 = 32 * sqrt(K*d*log(d)/m/n/nobs)
	objective = U -> (negloglik(U, data) + lambda0*nucnorm(U))
	grad_objective = U -> (grad_negloglik(U, data) + lambda0 * grad_nucnorm(U))
	Uinit = initialize_dropping_convexity(grad_objective, zeros(m,n), k)
	U = copy(Uinit)

	for lambda = lambda0* 2.0^(-10) * 2.0.^(0:0.5:15) # lambda = lambda0*logspace(0,-5,6)
		objective = U -> (negloglik(U, data) + lambda*nucnorm(U))
		grad_objective = U -> (grad_negloglik(U, data) + lambda * grad_nucnorm(U))
		# don't initialize at 0
		if vecnorm(U) < .01*sqrt(k*(m+n))
			U = copy(Uinit) 
		end
		U = gradient_descent!(U, objective, grad_objective, 
			              SimpleParams(HopefulStepSize(1, .8, 1.5, .1), 200), verbose=false)
		ThetaHat = rectangular_part(U)
		certify_optimality(grad_negloglik(Theta, data), Theta, lambda)
		ThetaHatZeroed = ThetaHat .- (ThetaHat * ones(n,1)/n) 
		loss, reg = negloglik(ThetaHat, data), lambda*nucnorm(ThetaHat)
		rmset = relrmse(ThetaHatZeroed, Theta)
		@printf("%12.2e%12.2e%12.2e%12.2e%12.2e\n", lambda, loss, reg, loss+reg, rmset)
	end
end

## gradient descent on Theta

# initialize
# ThetaHat = zeros(m,n)
# ThetaHat = initialize_dropping_convexity(grad_objective, zeros(m,n))

# gradient_descent!(ThetaHat, objective, grad_objective, 
# 	                 SimpleParams(DecreasingStepSize(.1), 20), verbose=true)

# compare to the bound if the constant isn't too honkingly large

# we've dropped the constraint Theta*1 = 0. 
# the likelihood term is invariant to constants added to each row, 
# and the regularizer is minimized when these offsets are chosen to be zero. 
# so the solution should approximately satisfy
# println("offsets:\n", ThetaHat*ones(n,1)/n) == 0

# see if we converge to the true solution using 
# t = initialize_dropping_convexity(grad_objective, zeros(m,n), k)
# gradient_descent!(t, t->negloglik(t, data), t->grad_negloglik(t, data), SimpleParams(HopefulStepSize(), 50), verbose=true)
# println("relative root mean square error is ", relrmse(rectangular_part(t), Theta))
