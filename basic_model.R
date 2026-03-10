############################################################
# Latent Space Model for Rank Data
# Based on Gormley & Murphy (2006)
# Plackett-Luce + latent Euclidean geometry
############################################################

set.seed(1)

############################################################
# Example data
############################################################

# n voters, J candidates
n <- 50
J <- 7
D <- 2

# Example: simulated rankings (each row = ranking)
rankings <- t(replicate(n, sample(1:J, J)))

############################################################
# Utility function from latent space
############################################################

utility <- function(voter_pos, cand_pos) {
  # negative squared distance
  -sum((voter_pos - cand_pos)^2)
}

############################################################
# Plackett-Luce log likelihood
############################################################

pl_loglik <- function(rankings, voters, candidates) {

  n <- nrow(rankings)
  J <- ncol(rankings)

  loglik <- 0

  for (i in 1:n) {

    remaining <- rankings[i,]

    for (k in 1:(J-1)) {

      chosen <- remaining[1]

      utilities <- sapply(remaining, function(j)
        utility(voters[i,], candidates[j,]))

      loglik <- loglik +
        utilities[1] -
        log(sum(exp(utilities)))

      remaining <- remaining[-1]
    }
  }

  return(loglik)
}

############################################################
# Priors
############################################################
