import IterativeSolvers: svdl
import Base: svd, dot

function svd(A::AbstractArray, k::Int)
	if k==0 || k>=min(size(A)...)
		return svd(A)
	else
		u,s,v,_ = svds(A, nsv = k)
		return u, s, v
		# inaccurate when k=1
		# usv, pf = svdl(A, k, vecs=:both)
		# return usv.U, usv.S, usv.Vt'
	end
end

# function svd(A::AbstractArray, k::Int)
# 	u,s,v = svd(A)
# 	return u[:,1:k], s[1:k], v[:,1:k]
# end

dot(x::Array{Float64,2}, y::Array{Float64,2}) = sum(x.*y)