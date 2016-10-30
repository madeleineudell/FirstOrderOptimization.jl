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
