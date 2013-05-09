library(MCMCpack)
library(pscl)
data        <- read.csv("examscores.csv",header=FALSE)
samples     <- 1000
burn_in     <- 10
total_iters <- samples+burn_in

# Plot Data
plot( density(data[,1]), main='Density of Data')

# mu ~ N(m,s2)
m  = 80
s2 = 16
# sig2 ~ IG(a,b)
a  = 3
b  = 1/50

# Plot the Prior Predictives
tt<-seq(50,100,length=1000)
plot(tt,dnorm(tt,m,sqrt(s2)),type='l',col='Blue',lwd=2,lty=1,main='Prior Predictive Mu')
tt<-seq(0,100,length=1000)
plot(tt,dinvgamma(tt,a,1/b),type='l',col='Blue',lwd=2,lty=1,main='Prior Predictive Sig^2')

mu   <- numeric(samples + burn_in)
sig2 <- numeric(samples + burn_in)

mu[1]   <- 80
sig2[1] <- 25

y_sum <- sum(data)
n     <- length(data[,1])

for (i in 1:total_iters) {
  
  #Calculate new mu value
  mu_star   <- ( s2 * y_sum + m * sig2[i] ) / (sig2[i] + n * s2)
  sig2_star <- (sig2[i] * s2) / (sig2[i] + n * s2)
  mu[i+1]   <-rnorm(1 ,  mu_star, sqrt(sig2_star))
  
  # Calculate new sigma^2 Value
  a_star    <- a + ( n / 2 )
  b_star    <- 1 / ( (1/b) + ( sum( (data[,1]-mu[i+1]) ^2 ) / 2 ) )
  sig2[i+1] <-rinvgamma(1, a_star, 1/b_star)

}

plot( density( mu[burn_in:total_iters] ) , main='Mu Posterior Distribution' )
plot( density( sig2[burn_in:total_iters] ) , main='Sig^2 Posterior Distribution' )
plot( mu[burn_in:total_iters], type='l', lwd=2 , main='Mu Progression')
plot( sig2[burn_in:total_iters], type='l', lwd=2, main='Sig^2 Progression')

# Plot Posterior Predictive
plot( density( rnorm(samples, mu[burn_in:total_iters], sqrt( sig2[burn_in:total_iters] ) ) ) , main='Posterior Predictive' )