# ============================================================
# LATENT SPACE MODEL FOR RANK DATA (Gormley & Murphy, 2006)
# FULL VERSION WITH PCA + CONFIDENCE ELLIPSES
# ============================================================


# ------------------------------------------------------------
# Distance + Likelihood
# ------------------------------------------------------------

sq_dist <- function(z, ZETA) {
  rowSums((ZETA - matrix(z, nrow(ZETA), length(z), byrow = TRUE))^2)
}

log_plackett_luce <- function(ranking, d_vec) {
  remaining <- ranking
  loglik <- 0
  
  for (t in seq_along(ranking)) {
    num <- exp(-d_vec[remaining[1]])
    denom <- sum(exp(-d_vec[remaining]))
    loglik <- loglik + log(num / denom)
    remaining <- remaining[-1]
    if (length(remaining) == 0) break
  }
  
  return(loglik)
}


log_posterior <- function(Z, ZETA, rankings,
                          sigma_v = 3, sigma_c = 3) {
  
  M <- nrow(Z)
  loglik <- 0
  
  for (i in 1:M) {
    d_vec <- sq_dist(Z[i, ], ZETA)
    loglik <- loglik + log_plackett_luce(rankings[[i]], d_vec)
  }
  
  logprior_Z <- sum(dnorm(Z, 0, sigma_v, log = TRUE))
  logprior_ZETA <- sum(dnorm(ZETA, 0, sigma_c, log = TRUE))
  
  return(loglik + logprior_Z + logprior_ZETA)
}


# ------------------------------------------------------------
# Local updates
# ------------------------------------------------------------

loglik_voter_i <- function(i, Z, ZETA, rankings) {
  d_vec <- sq_dist(Z[i, ], ZETA)
  log_plackett_luce(rankings[[i]], d_vec)
}


# ------------------------------------------------------------
# Procrustes alignment
# ------------------------------------------------------------

procrustes_align <- function(Z, ZETA, Z_ref, ZETA_ref) {
  C <- rbind(Z, ZETA)
  C_ref <- rbind(Z_ref, ZETA_ref)
  
  C <- scale(C, center = TRUE, scale = FALSE)
  C_ref <- scale(C_ref, center = TRUE, scale = FALSE)
  
  svd_res <- svd(t(C_ref) %*% C)
  Q <- svd_res$v %*% t(svd_res$u)
  
  C_rot <- C %*% Q
  
  M <- nrow(Z)
  list(
    Z = C_rot[1:M, ],
    ZETA = C_rot[(M+1):nrow(C_rot), ]
  )
}


# ------------------------------------------------------------
# MCMC Sampler
# ------------------------------------------------------------

latent_rank_MH <- function(rankings,
                           D = 2,
                           n_iter = 3000,
                           burn_in = 1000,
                           sigma_prop_z = 0.3,
                           sigma_prop_zeta = 0.05,
                           sigma_v = 3,
                           sigma_c = 3) {
  
  M <- length(rankings)
  N <- max(unlist(rankings))
  
  Z <- matrix(rnorm(M * D), M, D)
  ZETA <- matrix(rnorm(N * D), N, D)
  
  samples_Z <- array(NA, c(M, D, n_iter))
  samples_ZETA <- array(NA, c(N, D, n_iter))
  
  Z_ref <- Z
  ZETA_ref <- ZETA
  
  for (iter in 1:n_iter) {
    
    # --- Update voters ---
    for (i in 1:M) {
      Z_prop <- Z
      Z_prop[i, ] <- Z[i, ] + rnorm(D, 0, sigma_prop_z)
      
      log_alpha <- (
        loglik_voter_i(i, Z_prop, ZETA, rankings) -
        loglik_voter_i(i, Z, ZETA, rankings)
      ) + (
        sum(dnorm(Z_prop[i, ], 0, sigma_v, log=TRUE)) -
        sum(dnorm(Z[i, ], 0, sigma_v, log=TRUE))
      )
      
      if (log(runif(1)) < log_alpha) {
        Z <- Z_prop
      }
    }
    
    # --- Update candidates ---
    for (j in 1:N) {
      ZETA_prop <- ZETA
      ZETA_prop[j, ] <- ZETA[j, ] + rnorm(D, 0, sigma_prop_zeta)
      
      log_alpha <- log_posterior(Z, ZETA_prop, rankings) -
                   log_posterior(Z, ZETA, rankings)
      
      if (log(runif(1)) < log_alpha) {
        ZETA <- ZETA_prop
      }
    }
    
    # --- Procrustes ---
    if (iter > burn_in) {
      aligned <- procrustes_align(Z, ZETA, Z_ref, ZETA_ref)
      Z <- aligned$Z
      ZETA <- aligned$ZETA
    }
    
    samples_Z[,,iter] <- Z
    samples_ZETA[,,iter] <- ZETA
  }
  
  return(list(samples_Z = samples_Z,
              samples_ZETA = samples_ZETA))
}


# ------------------------------------------------------------
# Posterior summaries + covariance
# ------------------------------------------------------------

compute_candidate_stats <- function(samples_ZETA, burn_in=1000) {
  
  samples <- samples_ZETA[,,(burn_in+1):dim(samples_ZETA)[3]]
  
  N <- dim(samples)[1]
  D <- dim(samples)[2]
  
  means <- matrix(0, N, D)
  covs <- vector("list", N)
  
  for (j in 1:N) {
    draws <- t(samples[j,,])
    means[j,] <- colMeans(draws)
    covs[[j]] <- cov(draws)
  }
  
  list(means = means, covs = covs)
}


# ------------------------------------------------------------
# PCA for dimension selection
# ------------------------------------------------------------

run_pca <- function(ZETA_mean) {
  pca <- prcomp(ZETA_mean)
  var_exp <- pca$sdev^2 / sum(pca$sdev^2)
  
  print(var_exp)
  
  plot(var_exp, type="b", pch=19,
       main="PCA Variance Explained",
       xlab="Component", ylab="Variance")
}


# ------------------------------------------------------------
# Ellipse plotting
# ------------------------------------------------------------

plot_ellipse <- function(mu, Sigma, level=0.95, npoints=100) {
  theta <- seq(0, 2*pi, length.out=npoints)
  circle <- cbind(cos(theta), sin(theta))
  
  eig <- eigen(Sigma)
  radius <- sqrt(qchisq(level, df=2))
  
  ellipse <- t(mu + radius * t(circle %*% diag(sqrt(eig$values)) %*% t(eig$vectors)))
  
  lines(ellipse)
}


plot_latent_space <- function(means, covs) {
  
  plot(means, pch=19,
       xlab="Dim 1", ylab="Dim 2",
       main="Latent Candidate Space with 95% Ellipses")
  
  text(means, labels=1:nrow(means), pos=3)
  
  for (j in 1:nrow(means)) {
    plot_ellipse(means[j,], covs[[j]])
  }
}


# ============================================================
# EXAMPLE RUN
# ============================================================

rankings <- list(
  c(1,2,3,4),
  c(2,1,4,3),
  c(1,3,2,4),
  c(3,1,2,4)
)

fit <- latent_rank_MH(rankings, D=2, n_iter=3000)

# --- Posterior stats
stats <- compute_candidate_stats(fit$samples_ZETA, burn_in=1000)

# --- Plot latent space with uncertainty
plot_latent_space(stats$means, stats$covs)

# --- PCA diagnostic
run_pca(stats$means)
