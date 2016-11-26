import Base: factor, +, -, *, size, copy, norm

export MNLdata,
	negloglik, grad_negloglik,
	nucnorm, grad_nucnorm, min_lin_st_nucnorm, prox_nucnorm,
	RectangularParam, FactoredParam, rectangular_part, factor,
	initialize_dropping_convexity,
	+, *, -, size, copy

### Data

type MNLdata<:AbstractData
	nobs::Int
	nrow::Int
	ncol::Int
	sets::Array{Array{Int,1},1}
	rows::Array{Int,1}
	cols::Array{Int,1}
end

### Statistical parameter to be estimated

# The idea is that Theta::RectangularParam, U::FactoredParam,
# and [W1 Theta; Theta' W2] = U U' \succeq 0 at the solution
# The minimizer Theta of L(Theta; data) + lambda*\|Theta\|_*
# and the minimizer U of L((U U')[1:m, m+1:m+n]) + lambda/2*\|U\|_F^2
# satisfy (U U')[1:m, m+1:m+n] = Theta

typealias RectangularParam AbstractArray

type FactoredParam
	asarray
	nrow
	ncol
end
top(x::FactoredParam) = x.asarray[1:x.nrow,:]
bottom(x::FactoredParam) = x.asarray[x.nrow+1:x.nrow+x.ncol,:]
rectangular_part(x::FactoredParam) = top(x)*bottom(x)'
copy(x::FactoredParam) = FactoredParam(copy(x.asarray), x.nrow, x.ncol)
function factor(Theta::RectangularParam; k = min(size(Theta)...))
	m, n = size(Theta)
	u,s,v = svd(Theta, k)
	k = min(k, sum(s .> 1e-12))
	return FactoredParam([u[:,1:k]*diagm(sqrt(s[1:k]));
					   v[:,1:k]*diagm(sqrt(s[1:k]))],
					   m, n)
end

function +(x::FactoredParam, y::FactoredParam)
	(x.nrow==y.nrow && x.ncol==y.ncol) || error("Incompatible sizes")
	FactoredParam(x.asarray+y.asarray, x.nrow, x.ncol)
end
+(x::FactoredParam, y) = FactoredParam(x.asarray+y, x.nrow, x.ncol)
+(y, x::FactoredParam) = FactoredParam(x.asarray+y, x.nrow, x.ncol)
*(x::FactoredParam, y::FactoredParam) = NotImplementedError()
*(x::FactoredParam, y::Number) = FactoredParam(x.asarray*y, x.nrow, x.ncol)
*(y::Number, x::FactoredParam) = FactoredParam(x.asarray*y, x.nrow, x.ncol)
norm(x::FactoredParam) = norm(x.asarray)
vecnorm(x::FactoredParam) = vecnorm(x.asarray)
-(x::FactoredParam) = FactoredParam(-x.asarray, x.nrow, x.ncol)
-(x, y::FactoredParam) = x + -y
size(x::FactoredParam) = (x.nrow, x.ncol)

### The function we'll be minimizing

function negloglik(Theta::RectangularParam, data::MNLdata)
	l = 0
	for t=1:data.nobs
		it = data.rows[t]
		jt = data.cols[t]
		St = data.sets[t]
		invlik = 0 # inverse likelihood of observation t
		# computing soft max directly is numerically unstable
		# instead note logsumexp(a_j) = logsumexp(a_j - M) + M
		# and we'll pick a good big (but not too big) M
		M = 0 #Theta[it,jt] - minimum(Float64[Theta[it,j] for j in St])
		for j in St
			invlik += exp(Theta[it,jt] - Theta[it,j] - M)
		end
		l += log(invlik) + M
	end
	return l / data.nobs
end

function grad_negloglik(Theta::RectangularParam, data::MNLdata)
	DTheta = zeros(size(Theta))
	for t=1:data.nobs
		it = data.rows[t]
		jt = data.cols[t]
		St = data.sets[t]
		# Using some nice algebra, you can show
		DTheta[it,jt] += 1
		# and for j \in S, j \ne jt,
		# DTheta -= 1/sum_{j' \in S} exp(Theta[it,j] - Theta[it,j'])
		# it's ok if this over/underflows, I think:
		# the contribution of one observation to one entry of the gradient
		# is always between -1 and 0
		sumexp = sum(map(j->exp(-Theta[it,j]), St))
		for j in St
			DTheta[it,j] -= exp(-Theta[it,j])/sumexp
		end
	end
	return DTheta / data.nobs
end

function nucnorm(x::FactoredParam)
	return 1/2*vecnorm(x.asarray)^2
end

function grad_nucnorm(U::FactoredParam)
	return U
end

# prox_nucnorm(U, alpha) solves minimize_X (1/2||X||^2 + 1/(2*alpha)||X-U||^2)
function prox_nucnorm(U::FactoredParam, alpha::AbstractFloat)
	return 1/(1+alpha)*U
end

function nucnorm(Theta::RectangularParam)
	u,s,v = svd(Theta)
	return sum(s)
end

function grad_nucnorm(Theta::RectangularParam)
	u,s,v = svd(Theta)
	return u*spdiagm(sign(s))*v'
end

# prox_nucnorm(T, alpha) solves minimize_X (||X||_* + 1/(2*alpha)||X-T||^2)
# via soft-thresholding the singular values
function prox_nucnorm(Theta::RectangularParam,
	alpha::AbstractFloat;
	k = 10, # k estimates the number of singular values that will be greater than alpha
	kpp = 10) # how much to increment k if we're wrong
	u,s,v = svd(Theta, k)
	while s[end] > alpha
		k += kpp
		u,s,v = svd(Theta, k)
	end
	return u*spdiagm(max(s - alpha, 0))*v'
end

# solves min_{nucnorm(x)<=delta} G \dot x
function min_lin_st_nucnorm(G, delta)
	# solution to min_{nucnorm(x)<=delta} G \dot x
	u,s,v = svd(G, 1)
	x = -delta*u*v'
	return x
end

function negloglik(U::FactoredParam, data::MNLdata)
	return negloglik(rectangular_part(U), data)
end

function grad_negloglik(U::FactoredParam, data::MNLdata)
	G = grad_negloglik(rectangular_part(U), data)
	# gradient is [0_m G; G' 0_n] * U
	# return FactoredParam([G*top(U); G'*bottom(U)], size(U)...)
	return FactoredParam([G*bottom(U); G'*top(U)], size(U)...)
end

### The initialization (aka very rough algorithm)

# this function isn't type stable (so compiles poorly)
# but we're only calling it once so we don't care
function initialize_dropping_convexity(grad_objective, zeroparam, k::Int=0)
	grad_0 = grad_objective(zeroparam)
	zeroparam[1,1] = 1
	grad_1 = grad_objective(zeroparam)
	normalization = vecnorm(grad_0 - grad_1)
	u,s,v = svd(-grad_0, k)

	if k==0 # return initialization for Theta
		psd_grad_0 = u*spdiagm(max(s,0))*v'
		return psd_grad_0 / normalization
	else # return initialization for U
		# Theta = u[:,1:k]*spdiagm(max(s[1:k],0))*v[:,1:k]'
		U = [u[:,1:k]*spdiagm(sqrt(max(s[1:k],0)));
			 v[:,1:k]*spdiagm(sqrt(max(s[1:k],0)))]
		return FactoredParam(U / sqrt(normalization), size(zeroparam)...)
	end
end
