# ============================================================
# LATENT SPACE MODEL FOR RANK DATA (Modified for Oboe Multiphonics)
# Incorporates participant sensitivities b_s and baseline appeal c_j
# ============================================================

# ------------------------------------------------------------
# Squared distance function
# ------------------------------------------------------------
sq_dist <- function(y, x) {
  rowSums((x - matrix(y, nrow(x), length(y), byrow = TRUE))^2)
}

# ------------------------------------------------------------
# Log Plackett-Luce likelihood for a single ranking
# ------------------------------------------------------------
log_plackett_luce <- function(ranking, y, X, c, b) {
  remaining <- ranking
  loglik <- 0
  
  d_vec <- sq_dist(y, X) # squared distances from y to each x
  
  for (t in seq_along(ranking)) {
    j <- remaining[1]
    numerator <- exp(c[j] - b * d_vec[j])
    denominator <- sum(exp(c[remaining] - b * d_vec[remaining]))
    loglik <- loglik + log(numerator / denominator)
    remaining <- remaining[-1]
    if (length(remaining) == 0) break
  }
  
  return(loglik)
}

# ------------------------------------------------------------
# Log-posterior
# ------------------------------------------------------------
log_posterior <- function(Y, X, c, b, rankings,
                          sigma_y=3, sigma_x=3, sigma_c=3, mu_b=1, sigma_b=1) {
  S <- length(rankings)
  loglik <- 0
  
  for (s in 1:S) {
    R_s <- length(rankings[[s]]) # number of rankings for participant s
    for (r in 1:R_s) {
      loglik <- loglik + log_plackett_luce(rankings[[s]][[r]], Y[[s]][[r]], X, c, b[s])
    }
  }
  
  # Priors
  logprior_Y <- sum(sapply(Y, function(Ys) sum(sapply(Ys, function(y) dnorm(y, 0, sigma_y, log=TRUE)))))
  logprior_X <- sum(apply(X, 1, function(xj) sum(dnorm(xj, 0, sigma_x, log=TRUE))))
  logprior_c <- sum(dnorm(c, 0, sigma_c, log=TRUE))
  logprior_b <- sum(dnorm(b, mu_b, sigma_b, log=TRUE))
  
  return(loglik + logprior_Y + logprior_X + logprior_c + logprior_b)
}

# ------------------------------------------------------------
# Metropolis-Hastings Sampler
# ------------------------------------------------------------
latent_rank_MH <- function(rankings, D=2, n_iter=3000, burn_in=1000,
                           sigma_prop_y=0.3, sigma_prop_x=0.05,
                           sigma_y=3, sigma_x=3, sigma_c=3,
                           mu_b=1, sigma_b=1) {
  
  S <- length(rankings)       # number of participants
  R <- length(rankings[[1]])  # number of rankings per participant (assume same)
  N <- ncol(rankings[[1]][[1]]) # number of oboe multiphonics
  
  # Initialize latent locations and parameters
  Y <- lapply(1:S, function(s) lapply(1:R, function(r) rnorm(D)))  # participant rankings
  X <- matrix(rnorm(N*D), N, D)                                     # oboe multiphonics
  c <- rnorm(N)                                                      # baseline appeal
  b <- rnorm(S, mu_b, sigma_b)                                       # sensitivities
  
  # Storage
  samples_Y <- vector("list", S)
  for (s in 1:S) samples_Y[[s]] <- array(NA, c(R, D, n_iter))
  samples_X <- array(NA, c(N, D, n_iter))
  samples_c <- matrix(NA, N, n_iter)
  samples_b <- matrix(NA, S, n_iter)
  
  for (iter in 1:n_iter) {
    
    # --- Update participant locations ---
    for (s in 1:S) {
      for (r in 1:R) {
        y_prop <- Y[[s]][[r]] + rnorm(D, 0, sigma_prop_y)
        log_alpha <- log_plackett_luce(rankings[[s]][[r]], y_prop, X, c, b[s]) -
                     log_plackett_luce(rankings[[s]][[r]], Y[[s]][[r]], X, c, b[s]) +
                     sum(dnorm(y_prop, 0, sigma_y, log=TRUE)) -
                     sum(dnorm(Y[[s]][[r]], 0, sigma_y, log=TRUE))
        if (log(runif(1)) < log_alpha) Y[[s]][[r]] <- y_prop
      }
    }
    
    # --- Update oboe multiphonic locations ---
    for (j in 1:N) {
      x_prop <- X[j,] + rnorm(D, 0, sigma_prop_x)
      log_alpha <- log_posterior(Y, rbind(X[-j,], x_prop), c, b, rankings,
                                 sigma_y, sigma_x, sigma_c, mu_b, sigma_b) -
                   log_posterior(Y, X, c, b, rankings, sigma_y, sigma_x, sigma_c, mu_b, sigma_b)
      if (log(runif(1)) < log_alpha) X[j,] <- x_prop
    }
    
    # --- Update baseline appeals ---
    for (j in 1:N) {
      c_prop <- c
      c_prop[j] <- c[j] + rnorm(1, 0, sigma_prop_x)
      log_alpha <- log_posterior(Y, X, c_prop, b, rankings,
                                 sigma_y, sigma_x, sigma_c, mu_b, sigma_b) -
                   log_posterior(Y, X, c, b, rankings, sigma_y, sigma_x, sigma_c, mu_b, sigma_b)
      if (log(runif(1)) < log_alpha) c[j] <- c_prop[j]
    }
    
    # --- Update sensitivities ---
    for (s in 1:S) {
      b_prop <- b[s] + rnorm(1, 0, sigma_prop_x)
      log_alpha <- log_posterior(Y, X, c, replace(b, s, b_prop), rankings,
                                 sigma_y, sigma_x, sigma_c, mu_b, sigma_b) -
                   log_posterior(Y, X, c, b, rankings, sigma_y, sigma_x, sigma_c, mu_b, sigma_b)
      if (log(runif(1)) < log_alpha) b[s] <- b_prop
    }
    
    # --- Store samples ---
    for (s in 1:S) for (r in 1:R) samples_Y[[s]][r,,iter] <- Y[[s]][[r]]
    samples_X[,,iter] <- X
    samples_c[,iter] <- c
    samples_b[,iter] <- b
    
    if (iter %% 100 == 0) cat("Iteration:", iter, "\n")
  }
  
  list(samples_Y=samples_Y, samples_X=samples_X,
       samples_c=samples_c, samples_b=samples_b)
}









# ============================================================
# EXAMPLE: Simulated Data, Fit Model, Plot Latent Spaces
# ============================================================

set.seed(123)

S <- 5   # number of participants
R <- 14  # rankings per participant
N <- 7   # oboe multiphonics
D <- 2   # latent space dimensions

# Generate "true" latent locations
true_X <- matrix(rnorm(N * D, 0, 1), N, D)
true_c <- rnorm(N, 0, 1)
true_b <- runif(S, 0.5, 1.5)

# Function to simulate a single ranking
simulate_ranking <- function(y, X, c, b) {
  remaining <- 1:N
  ranking <- numeric(N)
  
  for (t in 1:N) {
    probs <- exp(c[remaining] - b * rowSums((X[remaining, , drop=FALSE] - matrix(y, length(remaining), D, byrow=TRUE))^2))
    probs <- probs / sum(probs)
    choice <- sample(length(remaining), 1, prob=probs)
    ranking[t] <- remaining[choice]
    remaining <- remaining[-choice]
  }
  return(ranking)
}

# Simulate participant rankings
rankings <- vector("list", S)
Y_true <- vector("list", S)

for (s in 1:S) {
  rankings[[s]] <- vector("list", R)
  Y_true[[s]] <- vector("list", R)
  for (r in 1:R) {
    # participant latent location for ranking r
    y <- rnorm(D, 0, 1)
    Y_true[[s]][[r]] <- y
    rankings[[s]][[r]] <- simulate_ranking(y, true_X, true_c, true_b[s])
  }
}

# ============================================================
# Fit the model using Metropolis-Hastings
# ============================================================
fit <- latent_rank_MH(rankings, D=D, n_iter=2000, burn_in=1000,
                      sigma_prop_y=0.3, sigma_prop_x=0.05,
                      sigma_y=1.5, sigma_x=1.5, sigma_c=1.5,
                      mu_b=1, sigma_b=0.5)

# ============================================================
# Posterior summaries for oboe multiphonics
# ============================================================
compute_candidate_stats <- function(samples_X, burn_in=1000) {
  samples <- samples_X[,,(burn_in+1):dim(samples_X)[3]]
  N <- dim(samples)[1]
  D <- dim(samples)[2]
  
  means <- matrix(0, N, D)
  covs <- vector("list", N)
  
  for (j in 1:N) {
    draws <- t(samples[j,,])
    means[j,] <- colMeans(draws)
    covs[[j]] <- cov(draws)
  }
  
  list(means=means, covs=covs)
}

stats <- compute_candidate_stats(fit$samples_X, burn_in=1000)

# ============================================================
# Plotting 95% confidence ellipses
# ============================================================
plot_ellipse <- function(mu, Sigma, level=0.95, npoints=100) {
  theta <- seq(0, 2*pi, length.out=npoints)
  circle <- cbind(cos(theta), sin(theta))
  
  eig <- eigen(Sigma)
  radius <- sqrt(qchisq(level, df=2))
  
  ellipse <- t(mu + radius * t(circle %*% diag(sqrt(eig$values)) %*% t(eig$vectors)))
  lines(ellipse)
}

plot_latent_space <- function(means, covs) {
  plot(means, pch=19, col="blue",
       xlab="Dim 1", ylab="Dim 2",
       main="Latent Space of Oboe Multiphonics with 95% Ellipses",
       xlim=range(means[,1]) + c(-1,1), ylim=range(means[,2]) + c(-1,1))
  text(means, labels=1:nrow(means), pos=3)
  for (j in 1:nrow(means)) {
    plot_ellipse(means[j,], covs[[j]])
  }
}

plot_latent_space(stats$means, stats$covs)



