### operators

import Base: size, *, Ac_mul_B, show, Array, getindex

export LowRankOperator, IndexingOperator, *, Ac_mul_B, size, show, Array, getindex


abstract Operator{T} <: AbstractMatrix{T}

function show(io::IO, o::Operator)
  print(io, "$(Array(o))")
end

size{T}(op::Operator{T}, i::Int) = size(op)[i]

mult_map = Dict{Symbol, Function}(:N => *, :T => Ac_mul_B)
tmult_map = Dict{Symbol, Function}(:T => *, :N => Ac_mul_B)
t_map = Dict{Symbol, Function}(:T => ctranspose, :N => identity)

type LowRankOperator{T<:Number}<:Operator{T}
	factors::Array{AbstractMatrix,1}
	transpose::Array{Symbol,1}
end
LowRankOperator(a...; transpose = fill(:N, length(a))) = LowRankOperator{Float64}([make_2d(ai) for ai in a], transpose)
# make_2d turns column vectors into matrices with 1 column
function make_2d{T}(a::AbstractArray{T,2})
	return a
end
function make_2d{T}(a::AbstractArray{T,1})
	return reshape(a, (length(a), 1))
end
# not a deep copy
copy(l::LowRankOperator) = LowRankOperator(copy(l.factors), copy(l.transpose))

function *{T}(l::LowRankOperator{T}, x)
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

function Ac_mul_B{T}(l::LowRankOperator{T}, x)
	for i in 1:length(l.factors)
		x = tmult_map[l.transpose[i]](l.factors[i], x)
	end
	return x
end
function size{T}(l::LowRankOperator{T})
	(size(l.factors[1], 1), 
	 size(l.factors[end], t_map[l.transpose[end]] == :N ? 2 : 1)) # 2nd dimension if not transposed; 1st if transposed
end

function Array(l::LowRankOperator)
	a = t_map[l.transpose[end]](l.factors[end])
	for i in length(l.factors)-1:-1:1
		a = mult_map[l.transpose[i]](l.factors[i], a)
	end
	return a
end

type IndexingOperator{T<:Number}<:Operator{T}
	m
	n
	iobs
end
IndexingOperator(m,n,iobs) = IndexingOperator{Float64}(m,n,iobs)

size{T}(op::IndexingOperator{T}) = (length(op.iobs), (op.m, op.n))

*{T}(op::IndexingOperator{T}, X) = X[op.iobs]
function Ac_mul_B{T}(op::IndexingOperator{T}, y)
	if length(y) != size(op,1)
		error("cannot multiply op'*y with size(y) = $(size(y)) and size(op) = $(size(op))")
	end
	reshape(sparsevec(op.iobs, y, op.m*op.n), (op.m, op.n))
end

function *{T}(iop::IndexingOperator{T}, lrop::LowRankOperator{T})
	# we only know how to do this for rank 1 LowRankOperators, for now
	if length(lrop.factors) == 2 && size(lrop.factors[1])[2] == 1 && size(lrop.factors[2])[1] == 1
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