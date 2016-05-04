import Base: axpy!, scale!, ctranspose, dot, size

export AbstractSketch, IdentitySketch, SymmetricSketch, AsymmetricSketch,
	additive_update!, reconstruct, dot

abstract AbstractSketch

# Identity

type IdentitySketch<:AbstractSketch
	X
end

function additive_update!(s::IdentitySketch, tilde_X, eta=1)
	axpy!(eta, tilde_X, s.X)
end

function cgd_update!(s::IdentitySketch, tilde_X, eta)
	scale!(s.X, 1-eta)
	axpy!(eta, tilde_X, s.X)
end

function reconstruct(s::IdentitySketch)
	return s.X
end

# Symmetric

type SymmetricSketch<:AbstractSketch
	r::Int
	Omega::Array{Float64,2}
	Psi::Array{Float64,2}
	Y::Array{Float64,2}
	W::Array{Float64,2}
end

function SymmetricSketch(n::Int, r::Int)
	k = 2r + 1
	l = 4r + 3 # = 2k+1
	Omega = randn(n,k)
	Psi   = randn(n,l)
	Y     = zeros(n,k)
	W     = zeros(n,l)
	return SymmetricSketch(r, Omega, Psi, Y, W)
end

function additive_update!(s::SymmetricSketch, vvt, eta=1)
	# Y = Y + eta*v*v'*Omega
	axpy!(eta, vvt*s.Omega, s.Y)
	# W = W + eta*v*v'*Psi
	axpy!(eta, vvt*s.Psi, s.W)
end

function cgd_update!(s::SymmetricSketch, vvt, eta)
	# Y = (1-eta)Y + eta*v*v'*Omega
	scale!(s.Y, 1-eta)
	axpy!(eta, vvt*s.Omega, s.Y)
	# W = (1-eta)W + eta*v*v'*Psi
	scale!(s.W, 1-eta)
	axpy!(eta, vvt*s.Psi, s.W)
end

function reconstruct(s::SymmetricSketch)
	# Q = orth(s.Y)
	Q,_ = qr(s.Y)
	B = (Q's.W) / (Q's.Psi)
	d,V = eigs((B+B')/2, nev=s.r, which=:LM)
	U = Q*V
	d = pos(d)
	return LowRankOperator(U, spdiagm(d), U') # reconstruction as square matrix is U*diagm(d)*U'
end

# Asymmetric

type AsymmetricSketch<:AbstractSketch
	r::Int
	Omega::Array{Float64,2}
	Psi::Array{Float64,2}
	Y::Array{Float64,2}
	W::Array{Float64,2}
end

function AsymmetricSketch(m::Int, n::Int, r::Int)
	k = 2r + 1
	l = 4r + 3 # = 2k+1
	Omega = randn(n,k)
	Psi   = randn(m,l)
	Y     = zeros(m,k)
	W     = zeros(n,l)
	return AsymmetricSketch(r, Omega, Psi, Y, W)
end

function additive_update!(s::AsymmetricSketch, uvt, eta=1)
	# Y = Y + eta*v*v'*Omega
	axpy!(eta, uvt*s.Omega, s.Y)
	# W = W + eta*v*v'*Psi
	axpy!(eta, uvt'*s.Psi, s.W)
end

function cgd_update!(s::AsymmetricSketch, uvt, eta)
	# Y = (1-eta)Y + eta*v*v'*Omega
	scale!(s.Y, 1-eta)
	axpy!(eta, uvt*s.Omega, s.Y)
	# W = (1-eta)W + eta*v*v'*Psi
	scale!(s.W, 1-eta)
	axpy!(eta, uvt'*s.Psi, s.W)
end

function reconstruct(s::AsymmetricSketch)
	# Q = orth(s.Y)
	Q,_ = qr(s.Y)
	B = s.W / (Q's.Psi) # Q's.Psi is k x l, its pinv is l x k, so B is n x k
	U,s,V,_ = svds(B, nsv=s.r) # U is n x r
	return LowRankOperator(Q*V, spdiagm(s), U') # reconstruction as square matrix is Q*V*diagm(s)*U'
end
# function reconstruct(s::AsymmetricSketch)
# 	# Q = orth(s.Y)
# 	Q,_ = qr(s.Y)
# 	B = s.W * pinv(Q'*s.Psi) # B is k x l
# 	U,s,V,_ = svds(B, nsv=s.r)
# 	return LowRankOperator(Q*U, spdiagm(s), V) # reconstruction as square matrix is Q*U*diagm(s)*V'
# end

### utilities
pos(a::Number) = max(a,0)
function pos(a::Array)
	for i in length(a)
		a[i] = pos(a[i])
	end
	a
end

# not yet efficient
function dot(M::SparseMatrixCSC{Float64,Int64}, A::Array{Float64,2})
	dot(M.nzval, A[find(M)])
end