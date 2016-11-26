using Base.Test

include("MNL.jl")

### Generate data

# make it deterministic
srand(10)

m, n, k = 100, 100, 3
Theta = randn(m, k) * randn(k, n)
Theta = Theta .- (Theta * ones(n,1))

nobs = int(10*m*n)
K = 2 #int(.05*n) # size of sets
sets = Array(Array{Int,1}, nobs) # set presented
rows = Array(Int, nobs)      # customer who arrived
cols = Array(Int, nobs)      # product they picked
for iobs = 1:nobs
	it = sample(1:m)
	St = sample(1:n, K, replace=false, ordered=true)
	wv = WeightVec(Float64[exp(-Theta[it, j]) for j in St])
	jt = sample(wv)
	rows[iobs], cols[iobs], sets[iobs] = it, jt, St
end
data = MNLdata(nobs, m, n, sets, rows, cols)

### Test invariances

ThetaTest = randn(m, n)
u,s,v = svd(ThetaTest)
ThetaTest = u[:,1:k]*diagm(s[1:k])*v[:,1:k]'
UTest = factor(ThetaTest, k=k)

@test_approx_eq negloglik(ThetaTest, data) negloglik(ThetaTest + rand(m,1)*ones(1,n), data)
@test_approx_eq grad_negloglik(ThetaTest + rand(m,1)*ones(1,n), data) grad_negloglik(ThetaTest + rand(m,1)*ones(1,n), data)

@test_approx_eq negloglik(ThetaTest, data) negloglik(UTest, data)

@test_approx_eq nucnorm(ThetaTest) nucnorm(UTest)

# gradients satisfy some relationship too...
# @test_approx_eq grad_negloglik(ThetaTest, data) grad_negloglik(UTest, data)
# @test_approx_eq grad_nucnorm(ThetaTest) grad_nucnorm(UTest)
