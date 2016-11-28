### operators

# infelicities:
# * LowRankOperator makes all its arguments 2d; does it need to?


import Base: size, *, Ac_mul_B, show, Array, getindex, dot

export IndexingOperator,
       AbstractLowRankOperator, LowRankOperator,
       IndexedLowRankOperator,
       *, Ac_mul_B, size, show, Array, getindex,
       thin_update!

abstract Operator{T} <: AbstractMatrix{T}

function show(io::IO, o::Operator)
  print(io, "$(Array(o))")
end

size{T}(op::Operator{T}, i::Int) = size(op)[i]

##### Indexing operators

type IndexingOperator{T<:Number}<:Operator{T}
	m
	n
	iobs
end
IndexingOperator(m,n,iobs) = IndexingOperator{Float64}(m,n,iobs)

size{T}(op::IndexingOperator{T}) = (length(op.iobs), (op.m, op.n))

*{T}(op::IndexingOperator{T}, X::Array{T}) = X[op.iobs]
Ac_mul_B{T}(op::IndexingOperator{T}, y::AbstractVector) = reshape(sparsevec(op.iobs, y, op.m*op.n), (op.m, op.n))
function Ac_mul_B{T}(op::IndexingOperator{T}, y::AbstractMatrix)
  size(y,2) == 1 || error("multiplication by IndexingOperators is only defined on vectors; got $(typeof(y))")
  reshape(sparsevec(op.iobs, vec(y), op.m*op.n), (op.m, op.n))
end

##### Low rank operators

mult_map = Dict{Symbol, Function}(:N => *, :T => Ac_mul_B)
tmult_map = Dict{Symbol, Function}(:T => *, :N => Ac_mul_B)
t_map = Dict{Symbol, Function}(:T => ctranspose, :N => identity)

abstract AbstractLowRankOperator{T<:Number}<:Operator{T}
type LowRankOperator{T<:Number}<:AbstractLowRankOperator{T}
	factors::Tuple{Vararg{AbstractMatrix{T}}}
	transpose::Tuple{Vararg{Symbol}}
end
function LowRankOperator(a::AbstractArray...; transpose = tuple(fill(:N, length(a))...))
  t = tuple([make_2d(ai) for ai in a]...)
  l = LowRankOperator(t, transpose)
  return l
end
function LowRankOperator{T<:Number}(t::Tuple{Vararg{AbstractMatrix{T}}}; transpose = tuple(fill(:N, length(a))...))
  return LowRankOperator(t,transpose)
end
# make_2d turns column vectors into matrices with 1 column
function make_2d{T}(a::AbstractArray{T,2})
	return a
end
function make_2d{T}(a::AbstractArray{T,1})
	return reshape(a, (length(a), 1))
end
copy(l::AbstractLowRankOperator) = AbstractLowRankOperator(deepcopy(l.factors), deepcopy(l.transpose))

# defining for specific cases just to remove ambiguity w/Base method ugh
# could fix using invoke???
# i can't seem to make a small problematic example, see attempt in test/ambiguous.jl
function *{T}(l::LowRankOperator{T}, x::AbstractVector{T})
	for i in length(l.factors):-1:1
		x = mult_map[l.transpose[i]](l.factors[i], x)
	end
	return x
end
function *{T}(l::LowRankOperator{T}, x::AbstractMatrix{T})
  for i in length(l.factors):-1:1
		x = mult_map[l.transpose[i]](l.factors[i], x)
	end
	return x
end
# function *{T}(x, l::LowRankOperator{T})
# 	xl = copy(l)
# 	unshift!(xl.factors, x)
# 	unshift!(xl.transpose, :N)
# 	return xl
# end

function Ac_mul_B{T}(l::LowRankOperator{T}, x::AbstractArray{T,1})
	for i in 1:length(l.factors)
		x = tmult_map[l.transpose[i]](l.factors[i], x)
	end
	return x
end
function Ac_mul_B{T}(l::LowRankOperator{T}, x::AbstractArray{T,2})
	for i in 1:length(l.factors)
		x = tmult_map[l.transpose[i]](l.factors[i], x)
	end
	return x
end
function size{T}(l::LowRankOperator{T})
	(size(l.factors[1], 1),
	 size(l.factors[end], t_map[l.transpose[end]] == :N ? 2 : 1)) # 2nd dimension if not transposed; 1st if transposed
end

function Array(l::AbstractLowRankOperator)
	a = t_map[l.transpose[end]](l.factors[end])
	for i in length(l.factors)-1:-1:1
		a = mult_map[l.transpose[i]](l.factors[i], a)
	end
	return a
end

function is_svd(X::LowRankOperator)
  if (length(X.factors)==3) && (X.transpose == (:N, :N, :T))
    # X.factors[2] is diagonal
    # X.factors[1] and [3] are orthogonal
    return true
  else
    return false
  end
end
function is_rank_1(X::LowRankOperator)
  if (length(X.factors)==2) && (X.transpose == (:N, :T))
    return true
  else
    return false
  end
end

# compute alpha*X + (1-alpha)*Delta
# update the factorization of X using idea from
# Brand 2006 "Fast Low-Rank Modifications of the Thin SVD"
function thin_update!{T<:Number}(X::LowRankOperator{T}, Delta::LowRankOperator{T}, alpha::T=1.0, beta::T=1.0)
  # assume that the factors of X form an SVD factorization
  @assert is_svd(X)
  # and that Delta is rank 1
  @assert is_rank_1(Delta)

  # scale X and Delta so we just need to compute U * Sigma * V' + a * b'
  U, Sigma, V = X.factors
  Sigma *= alpha
  r = size(Sigma,1)
  a,b = Delta.factors
  a *= beta

  # eqn 6
  m = U'*a
  p = a-U*m
  ra = norm(p)
  P = p/ra

  # eqn 7
  n = V'*b
  q = b-V*n
  rb = norm(q)
  Q = q/rb

  # eqn 8 - can be made more efficient exploiting diag + rank 1
  K = [Sigma zeros(r); zeros(1,r) 0] + [m; ra]*[n' rb]
  Uk,Sk,Vk = svd(K)

  Up = [U P]*Uk
  Vp = [V Q]*Vk
  X.factors = (Up, spdiagm(Sk), Vp)
  return X
end

#### now mix IndexingOperator and LowRankOperator

type IndexedLowRankOperator{T<:Number}<:AbstractLowRankOperator{T}
	factors::Tuple{IndexingOperator{T},AbstractVector{T}}
  transpose::Tuple{Vararg{Symbol}}
end
IndexedLowRankOperator(a...) = IndexedLowRankOperator{Float64}((a...), (:T, :N))
function A_mul_B!{T}(u::AbstractArray{T,1}, l::IndexedLowRankOperator{T}, v::AbstractVector{T})
  @assert(length(u)==l.factors[1].m)
  @assert(length(v)==l.factors[1].n)
	iobs = l.factors[1].iobs
  js = round(Int,floor((iobs-1)/l.factors[1].m)+1)
  is = (iobs-1)%l.factors[1].m+1
  for ii in length(iobs)
    u[is[ii]] += l.factors[2][ii]*v[js[ii]]
  end
	return u
end
*{T}(l::IndexedLowRankOperator{T}, v::AbstractVector{T}) =
    A_mul_B!(zeros(l.factors[1].m), l, v)
function A_mul_B!{T}(v::AbstractMatrix{T}, l::IndexedLowRankOperator{T}, u::AbstractMatrix{T})
    for i=1:size(u,2)
      A_mul_B!(view(v,:,i), l, view(u,:,i))
    end
    v
end
*{T}(l::IndexedLowRankOperator{T}, v::AbstractMatrix{T}) =
    A_mul_B!(zeros(l.factors[1].m, size(v,2)), l, v)
function Ac_mul_B!{T}(v::AbstractArray{T,1}, l::IndexedLowRankOperator{T}, u::AbstractArray{T,1})
  @assert(length(u)==l.factors[1].m)
  @assert(length(v)==l.factors[1].n)
	iobs = l.factors[1].iobs
  js = round(Int,floor((iobs-1)/l.factors[1].m)+1)
  is = (iobs-1)%l.factors[1].m+1
  v = zeros(l.factors[1].n)
  for ii in length(iobs)
    v[js[ii]] += l.factors[2][ii]*u[is[ii]]
  end
	return v
end
Ac_mul_B{T}(l::IndexedLowRankOperator{T}, u::AbstractVector{T}) =
    Ac_mul_B!(zeros(l.factors[1].n), l, u)
function Ac_mul_B!{T}(v::AbstractMatrix{T}, l::IndexedLowRankOperator{T}, u::AbstractMatrix{T})
    for i=1:size(u,2)
      Ac_mul_B!(view(v,:,i), l, view(u,:,i))
    end
    v
end
Ac_mul_B{T}(l::IndexedLowRankOperator{T}, u::AbstractMatrix{T}) =
    Ac_mul_B!(zeros(l.factors[1].n, size(u,2)), l, u)
function size{T}(l::IndexedLowRankOperator{T})
  size(l.factors[1])[2]
end
# not sure why this is needed for svds
function getindex(l::IndexedLowRankOperator,i::Int,j::Int)
  idx = findin(l.factors[1].iobs, i+l.factors[1].m*(j-1))
  if length(idx)==0
    return 0
  else
    return sum(l.factors[2][idx])
  end
end
function Array(l::IndexedLowRankOperator)
  iobs = l.factors[1].iobs
  js = round(Int,floor((iobs-1)/l.factors[1].m)+1)
  is = (iobs-1)%l.factors[1].m+1
  return sparse(is, js, l.factors[2], l.factors[1].m, l.factors[1].n)
end

function *{T}(iop::IndexingOperator{T}, lrop::LowRankOperator{T})
	# we only know how to do this for rank 1 LowRankOperators, for now
	if length(lrop.factors) == 2 && size(lrop.factors[1])[2] == 1 & size(lrop.factors[2])[1] == 1
		u = lrop.factors[1]
		v = lrop.factors[2]
		y = zeros(length(iop.iobs))
		for idx in 1:length(iop.iobs)
			i = (iop.iobs[idx]-1) % iop.m + 1
			j = floor(Int, (iop.iobs[idx]-1) / iop.m + 1)
			y[idx] = u[i]*v[j]
		end
		return y
	else
		warn("materializing LowRankOperator...")
		return iop*Array(lrop)
		# error("We only know how to multiply IndexingOperators by LowRankOperators for rank 1 LowRankOperators, for now")
	end
end

function getindex(op::LowRankOperator, i::Int, j::Int)
	if length(op.factors)==2
		if op.transpose[1]==:N
			xi = op.factors[1][i,:]
		else
			xi = op.factors[1][:,i]
		end
		if op.transpose[2]==:N
    	yj = op.factors[2][:,j]
		else
			yj = op.factors[2][j,:]
		end
		return dot(vec(xi), vec(yj))
	elseif is_svd(op)
    return dot(op.factors[1][i,:], op.factors[2]*op.factors[3][j,:])
  else
		error("indexing not defined for LowRankOperator with $(length(op.factors)) factors")
	end
end

*(a::Number, o::AbstractLowRankOperator) = (scale!(o.factors[end], a); o)
*(o::AbstractLowRankOperator, a::Number) = *(a,o)

function dot(A::SparseMatrixCSC, lrop::LowRankOperator)
  # @assert is_svd(lrop)
  # @assert size(A) == size(lrop)
  # construct U, V so that U*V = lrop
  U = lrop.factors[1]
  n,r = size(lrop.factors[3])
  V = Array(Float64,r,n)
  for i=1:r
    V[i,:] = lrop.factors[2].nzval[i] * lrop.factors[3][:,i]
  end
  out = 0
  m,n = size(A)
  for j=1:n
    for iobs = A.colptr[j]:(A.colptr[j+1]-1)
      i = A.rowval[iobs]
      out += dot(U[i,:],V[:,j])*A.nzval[iobs]
    end
  end
  return out
end
