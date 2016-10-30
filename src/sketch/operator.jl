### operators

# infelicities:
# * LowRankOperator makes all its arguments 2d; why?


import Base: size, *, Ac_mul_B, show, Array, getindex

export IndexingOperator,
       AbstractLowRankOperator, LowRankOperator,
       IndexedLowRankOperator,
       *, Ac_mul_B, size, show, Array, getindex


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
	factors::Array{AbstractMatrix,1}
	transpose::Array{Symbol,1}
end
LowRankOperator(a...; transpose = fill(:N, length(a))) = LowRankOperator{Float64}(AbstractMatrix[make_2d(ai) for ai in a], transpose)
# make_2d turns column vectors into matrices with 1 column
function make_2d{T}(a::AbstractArray{T,2})
	return a
end
function make_2d{T}(a::AbstractArray{T,1})
	return reshape(a, (length(a), 1))
end
# not a deep copy
copy(l::AbstractLowRankOperator) = AbstractLowRankOperator(copy(l.factors), copy(l.transpose))

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

#### now mix IndexingOperator and LowRankOperator

type IndexedLowRankOperator{T<:Number}<:AbstractLowRankOperator{T}
	factors::Tuple{IndexingOperator{T},AbstractVector{T}}
	transpose::Array{Symbol,1}
end
IndexedLowRankOperator(a...) = IndexedLowRankOperator{Float64}((a...), [:T, :N])
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
	else
		error("indexing not defined for LowRankOperator with $(length(op.factors)) factors")
	end
end
