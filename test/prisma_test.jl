using FirstOrderOptimization

## The state of this code:
# * seems like the algo works. 
# 	* obj converges to a nice low number
# 	* i haven't checked it against an sdp solver yet
# 	* or tested for recovery
# * it does a super weird thing where 
# 	the rank and obj shoot up in the 14th iteration. why?

srand(1)

type PrevRank
	r::Integer
end

### max norm regularized matrix completion problem

# minimize f(W) + g(W) + h(W)

# where W = [A X; X' B] and

# (data fitting)   f(W) = vecnorm(P.*(X - M))^2
# (regularization) g(W) = lambda*max(diag(X))
# (psd)            h(W) = (W \succeq 0) ? 0 : Inf

function generate_maxnorm_problem(m,n,lambda,k)
	# generate data
	X0 = randn(m,k)
	Y0 = randn(k,n)
	A = X0*Y0 + .1*randn(m,n)
	P = float(randn(m,n) .>= .5) # observed entries
	
	X(W) = W[1:m, m+1:end]

	function grad_f(W)
		gX = P.*(X(W)-A)
		return [zeros(m,m) gX; gX' zeros(n,n)]
	end

	function prox_g(W, alpha)
		Z = copy(W)
		oldmax = maximum(diag(W))
		newmax = oldmax - lambda*alpha/2
		for i=1:size(Z,1)
			if Z[i,i] > newmax 
				Z[i,i] = newmax
			end
		end
		Z
	end

	# we're going to use a closure over prevrank
	# to remember what the rank of prox_h(W) was the last time we computed it
	# in order to avoid calculating too many eigentuples of W
	prevrank = PrevRank(k)

	function prox_h(W, alpha=0; TOL=1e-10)
		# debugging: @show prevrank.r
		while prevrank.r < size(W,1)
			l,v = eigs(Symmetric(W), nev = prevrank.r+1, which=:LR) # v0 = [v zeros(size(W,1), prevrank.r+1 - size(v,2))]
			if l[end] <= TOL
				prevrank.r = sum(l.>=TOL)
				return v*diagm(max(l,0))*v'
			else
				prevrank.r = 2*prevrank.r # double the rank and try again
			end
		end
		# else give up on computational cleverness
		l,v = eig(Symmetric(W))
		prevrank.r = sum(l.>=TOL)
		return v*diagm(max(l,0))*v'
	end

	# we're not going to bother checking whether W is psd or not
	# when evaluating the objective; in the course of the prisma
	# algo this makes no difference
	obj(W) = sum((P.*(X(W) - A)).^2) + lambda*maximum(diag(W))

	return grad_f, prox_g, prox_h, obj
end

function test_prisma(m,n,lambda,k)
	
	grad_f, prox_g, prox_h, obj = generate_maxnorm_problem(m,n,lambda,k)

	# initialize
	W = zeros(m+n,m+n)
	# lipshitz constant for f
	L_f = 2
	# orabona starts stepsize at
	# beta = lambda/sqrt((m+n)^2*mean(A.^2))
	beta   = lambda/sqrt(obj(W))
	ssr    = PrismaStepsize(beta)
	params = PrismaParams(ssr, 10, 1)

	# recover
	W = PRISMA(W, L_f,
		   grad_f,
		   prox_g,
		   prox_h,
		   obj,
		   params)
end

test_prisma(5,5,10,3)