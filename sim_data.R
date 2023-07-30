## simple R script to simulate example genotype data
## MRR 14.07.21
## This requires the software plink: https://www.cog-genomics.org/plink2


set.seed(171014)
require(MASS)
library(evd)

dataset_name <- "weibull_"
## set sample size, N
N = 1000

## total number of covariates, M
M = 20000
dataset_name <- paste(dataset_name, N, "_", M, sep="")
## set intercept value
mu = 3.9
## number of causal markers
causal <- 500

## alpha scale paremeter of weibull
alpha <- 10


## sample marker effects
sigma_g <- pi**2/(6*alpha**2)

## simulate marker data
X <- matrix(rbinom(N*M,2,0.4),N,M)

## total variance explained by marker effects
h2 = 0.5


b <- rnorm(causal,0,sqrt(sigma_g/causal))

## generate genetic values
beta <- matrix(rep(0,M),M)
index1 <- sample(1:M,causal)
beta[index1] <- b

g <- scale(X) %*% beta

## generate residuals
#sigma_e <- matrix(c(1-var(g[,1]), 0, 0, 1-var(g[,2])),2,2)

e <- rgumbel(N)


## output phenotype
y = mu + g + e/alpha -digamma(1)/alpha

failure <- rep("1", N)

## output genetic data
X[X == 2] <- "A/tA"
X[X == 1] <- "A/tG"
X[X == 0] <- "G/tG"

X <- as.data.frame(lapply(X, as.character))

# Load the tidyr package
library(tidyr)

# Separate each column in the dataframe "X" into individual characters, doubling the number of columns
X_separated <- lapply(X, function(col) separate(as.data.frame(col), col = 1:length(col), into = paste0(names(col), "_", 1:length(col)), sep = 1))


## output to plink .ped/.map format
ped <- data.frame("FID" = 1:N,
                  "IID" = 1:N,
                  "PID" = rep(0,N),
                  "MID" = rep(0,N),
                  "Sex" = rep(1,N),
                  "phen" = rep(0,N))

ped <- cbind(ped,X)

save_path <- "BayesW_data_sim/"
write.table(ped,paste(save_path, dataset_name, ".ped", sep=""), row.names=FALSE, col.names=FALSE, quote=FALSE)

map <- data.frame("chr" = rep(1,M),
                  "rs" = paste("rs",1:M, sep=''),
                  "dist" = rep(0,M),
                  "bp" = 1:M)
write.table(map,paste(save_path,dataset_name, "test.map", sep=""), row.names=FALSE, col.names=FALSE, quote=FALSE)

## convert from .ped/.map to plink binary format
system(paste("plink --file ", paste(save_path, dataset_name, "test", sep=""),
             "--make-bed --out ", paste(save_path, dataset_name, "test", sep=""))

## remove .ped/.map files
#system("rm *.ped")
#system("rm *.map")
#system("rm *.log")

## output phenotype files
phen <- data.frame("FID" = 1:N,
                   "IID" = 1:N,
                   "phen1" = y)
#                   "phen2" = y[,2])
write.table(phen,paste(save_path,dataset_name, "test.phen", sep=""), row.names=FALSE, col.names=FALSE, quote=FALSE)

## failure file
write.table(failure,paste(save_path,dataset_name, "test.fail", sep=""), row.names=FALSE, col.names=FALSE, quote=FALSE)

## fam file
fam <- data.frame("FID" = 1:N,
                  "IID" = 1:N,
                  "IDfather" = rep(0,N),
                  "IDmother" = rep(0,N),
                  "sex" = rep(0,N),
                  "phen"= rep(-9,N))

write.table(fam,paste(save_path,dataset_name, "test.fam", sep=""), row.names=FALSE, col.names=FALSE, quote=FALSE)

