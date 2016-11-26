import IterativeSolvers: svdl
import Base: svd, dot

# fix output of svds to make it like svd
function mysvds(args...; kwargs...)
  svdobj,_ = Base.svds(args...; kwargs...)
  return svdobj.U, svdobj.S, svdobj.Vt
end

function svd(A::AbstractArray, k::Int)
	if k==0 || k>=min(size(A)...)
		return svd(A)
	else
		u,s,v = mysvds(A, nsv = k)
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

# find a zero of f over [a, b]
function find_zero(f, a, b; tol=1e-9, maxiters=1000)
    f(a) < 0 || return a
    f(b) > 0 || return b
    for i=1:maxiters
        mid = a + (b-a)/2
        fmid = f(mid)
        if abs(fmid) < tol
            return mid
        end
        if f(mid) < 0
            a = mid
        else
            b = mid
        end
    end
    warn("hit maximum iterations in bisection search")
    return (b-a)/2
end
