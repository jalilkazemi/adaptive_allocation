library(rgl)

p = seq(0,0.99, by=0.01)
m= read.csv("D:/Documents and Settings/Nazi/Desktop/value_approx_2-2.csv", header=F)
m=as.matrix(m)
persp3d(x=p,y=p,z=m, theta=-80, phi=30)

action = seq(0,0.99, by=0.01)
m= read.csv("D:/Documents and Settings/Nazi/Desktop/score_approx_1.csv", header=F)
plot(action,m$V1, type='l')

# mlin=m[1,1] +matrix(rep(p*(m[100,1]-m[1,1]), times=100) + rep(p * (m[1,100] -m[1,1]), each=100), nrow=100)
# persp(x=p,z=mlin, theta=-80, phi=30)

library(rjson)
N = 10000
Featurize <- function(q) {
	p = q$k / q$n
	return(c(
		q$action,
		q$n[1] / N,
		q$n[2] / N,
		p[1],
		p[2],
		(q$k[1] + q$k[2]) / (q$n[1] + q$n[2]),
		max(p),
		p[1]^2,
		p[2]^2,
		q$action^2,
		q$action*(p[1]>p[2]),
		q$action*p[1],
		q$action*p[2])
	)
}
Sigmoid <- function(x) 1/(1+exp(-x))
Frontpropagate <- function(q) {
	input = c(1, Featurize(q))
	hidden = c(1, Sigmoid(coef0 %*% input))
	return(coef1 %*% hidden)
}
fit <- fromJSON(file="D:/Documents and Settings/Nazi/Desktop/value_nnet.js")
coef0 = t(simplify2array(fit$Coef0))
coef1 = t(simplify2array(fit$Coef1))

### action
q = list(action = 0, n = c(1000, 1000), k = c(0,0))
q$k = c(0.25,0.20) * q$n
actions = seq(0, 0.99,by=0.01)
values = sapply(actions, function(x) {q$action = x; Frontpropagate(q)})
plot(actions, values)

### n
q = list(action = 1, n = c(0, 0), k = c(0, 0))
ninv = seq(0.01,0.99,by=0.01)
z = apply(expand.grid(ninv, ninv), 1, function(x) {q$n = x*N; q$k = c(0.25,0.50) * q$n; Frontpropagate(q)})
persp3d(x=ninv, y=ninv, z=z)

### p
q = list(action = 1, n = c(8000, 8000), k = c(0, 0))
p = seq(0.01,0.99,by=0.01)
z = apply(expand.grid(p, p), 1, function(x) {q$k = x * q$n; Frontpropagate(q)})
# zreal = apply(expand.grid(p, p), 1, function(x) {max(x)})
persp3d(x=p, y=p, z=z)
# persp3d(x=p, y=p, z=zreal)


