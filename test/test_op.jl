using FirstOrderOptimization

lr = LowRankOperator(rand(5), rand(4)')
lr*rand(4)
lr'*rand(5)

svds(lr, nsv=1)

pobs = .5
obs = sprand(5, 4, pobs).>0
iobs = find(obs)
io = IndexingOperator(5, 4, iobs)
io*rand(5,4)
io'*rand(length(iobs))

# svds(io, nsv=1)

ilr = IndexedLowRankOperator(io,rand(length(iobs)))
size(ilr)==(5,4)
ilr*rand(4)
ilr'*rand(5)

svds(ilr, nsv=1)

# test thin update

u,s,v = FirstOrderOptimization.svd(rand(4,5), 2)
A = LowRankOperator(u,spdiagm(s),v, transpose=(:N,:N,:T))
Delta = LowRankOperator(rand(4,1), rand(5,1), transpose=(:N,:T))
Aup = Array(A) + Array(Delta)
thin_update!(A,Delta,1.)
@assert norm(Array(A) - Aup) <= 1e-10

# test dot between sparse and low rank (svd) operator

m,n = size(A)
G = sprand(m,n,.3)
@assert norm(dot(G,A) - dot(full(G),Array(A))) <= 1e-10
