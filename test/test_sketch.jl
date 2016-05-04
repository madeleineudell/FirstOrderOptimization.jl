using FirstOrderOptimization

srand(1)

## test asymmetric sketch
m = 100; n = 110; r = 5

X = zeros(m,n)
s = AsymmetricSketch(m,n,r)
for iter=1:r
	Xi = LowRankOperator(rand(m),rand(n)')
	X += Array(Xi)
	additive_update!(s,Xi)
end
@show vecnorm(Array(reconstruct(s)) - X) / vecnorm(X)

## test symmetric sketch
n = 100; r = 5

X = zeros(n,n)
s = SymmetricSketch(n,r)
for iter=1:r
	v = rand(n)
	Xi = LowRankOperator(v,v')
	X += v*v'
	additive_update!(s,Xi)
end
@show vecnorm(Array(reconstruct(s)) - X) / vecnorm(X)