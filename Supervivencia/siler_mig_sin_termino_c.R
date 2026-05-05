# Modelo siler con dispersión sin el parámetro "c" 
set.seed(0312)
library(rstan)
library(ggplot2)

# FUNCIONES
h_siler <- function(t, a0, a1, b0, b1){
  exp(a0 - a1*t) + exp(b0 + b1*t)
}
# exp(a0 - a1*t) riesgo inicial 
# exp(b0 + b1*t) riesgo al envejecer

# Riesgo acumulado 
H_siler <- function(t, a0, a1, b0, b1){
  (exp(a0)/a1) * (1 - exp(-a1*t)) + (exp(b0)/b1) * (exp(b1*t) - 1)
}

# Supervivencia a muerte
S_muerte <- function(t, a0, a1, b0, b1){
  exp(-H_siler(t, a0, a1, b0, b1))
}

# SIMULACIÓN DE DATOS
N    <- 600
tmax <- 20
tgrid <- seq(0, tmax, length.out = 250001)
# Parámetros (se pueden ajustar)
true <- list(
  a0 = -1.0,
  a1 = 1.0,
  b0 = -3.0,
  b1 = 0.15,
  
  # Dispersión: tau ~ Gamma(k, r), con media mu = k/r
  k_tau = 8,           #forma      
  mu_tau_hembra = 12,        # casi no dispersan las hembras
  beta_macho_logmu = -0.25   # macho dispersa "antes"
)

# Sexo
es_macho <- rbinom(N, 1, 0.5)

# Dispersión individual
mu_tau_i <- true$mu_tau_hembra * exp(true$beta_macho_logmu * es_macho)
rate_tau_i <- true$k_tau / mu_tau_i
tau <- rgamma(N, shape = true$k_tau, rate = rate_tau_i)

# Fm(t) = 1 - Sm(t)
Fmuerte <- 1 - S_muerte(tgrid, true$a0,true$a1,true$b0,true$b1)
uD <- runif(N)

T_muerte <- numeric(N)
for(i in 1:N){
  idx <- which.min(abs(Fmuerte - uD[i]))
  T_muerte[i] <- tgrid[idx]
}

# Censura
cens_time <- runif(N, 2, 10)

t_obs <- pmin(T_muerte, tau, cens_time)
status <- ifelse(
  T_muerte <= tau & T_muerte <= cens_time, 1L,
  ifelse(tau <= T_muerte & tau <= cens_time, 2L, 0L)
)

# (0=cens,1=muerte,2=migr)
print(table(status))

# Gráficas

df_plot <- data.frame(
  t = tgrid,
  hD = h_siler(tgrid, true$a0, true$a1, true$b0, true$b1),
  Sd = S_muerte(tgrid, true$a0, true$a1, true$b0, true$b1)
)

# Riesgo de muerte
ggplot(df_plot, aes(t, hD)) +
  geom_line(color = "maroon", linewidth = 1.2) +
  labs(title = "Riesgo de muerte (Siler sin c)",
       y = "hD(t)", x = "Tiempo") +
  theme_minimal()

# Supervivencia a muerte
ggplot(df_plot, aes(t, Sd)) +
  geom_line(color = "pink", linewidth = 1.2) +
  labs(title = "Supervivencia a muerte (Siler sin c)",
       y = "S_D(t)", x = "Tiempo") +
  theme_minimal()

# CODIGO STAN
stan_code <- "
functions {
  real log_h_siler(real t, real a0, real a1, real b0, real b1) {
    return log_sum_exp(a0 - a1*t, b0 + b1*t);
  }

  real H_siler(real t, real a0, real a1, real b0, real b1) {
    return (exp(a0) / a1) * (1 - exp(-a1 * t))
           + (exp(b0) / b1) * (exp(b1 * t) - 1);
  }
}

data {
  int<lower=1> N;
  vector<lower=0>[N] t;
  array[N] int<lower=0, upper=2> status; 
  array[N] int<lower=0, upper=1> es_macho;
}

parameters {
  real a0;
  real<lower=0> a1;
  real b0;
  real<lower=0> b1;

  real<lower=0> k_tau;
  real<lower=0> mu_tau_hembra;
  real beta_macho_logmu;
}

transformed parameters {
  vector<lower=0>[N] mu_tau;
  vector<lower=0>[N] rate_tau;

  for (i in 1:N) {
    mu_tau[i] = mu_tau_hembra * exp(beta_macho_logmu * es_macho[i]);
    rate_tau[i] = k_tau / mu_tau[i];
  }
}

model {
  // Priors
  a0 ~ normal(0, 2);
  a1 ~ lognormal(0, 1);
  b0 ~ normal(0, 2);
  b1 ~ lognormal(0, 1);

  k_tau ~ lognormal(log(5), 0.5);
  mu_tau_hembra ~ lognormal(log(10), 0.5);
  beta_macho_logmu ~ normal(0, 0.7);

  for (i in 1:N) {
    real loghD = log_h_siler(t[i], a0, a1, b0, b1);
    real HD = H_siler(t[i], a0, a1, b0, b1);
    real logSd = -HD;

    real logf_tau = gamma_lpdf(t[i] | k_tau, rate_tau[i]);
    real logS_tau = gamma_lccdf(t[i] | k_tau, rate_tau[i]); 

    if (status[i] == 0) target += logSd + logS_tau;
    else if (status[i] == 1) target += (loghD + logSd) + logS_tau;
    else target += logf_tau + logSd;
  }
}

generated quantities {
  vector[N] log_lik;
  for (i in 1:N) {
    real loghD = log_h_siler(t[i], a0, a1, b0, b1);
    real HD = H_siler(t[i], a0, a1, b0, b1);
    real logSd = -HD;

    real logf_tau = gamma_lpdf(t[i] | k_tau, rate_tau[i]);
    real logS_tau = gamma_lccdf(t[i] | k_tau, rate_tau[i]);

    if (status[i] == 0) log_lik[i] = logSd + logS_tau;
    else if (status[i] == 1) log_lik[i] = (loghD + logSd) + logS_tau;
    else log_lik[i] = logf_tau + logSd;
  }
}
"

sm <- stan_model(model_code = stan_code)

stan_data <- list(
  N = N,
  t = as.vector(t_obs),
  status = as.integer(status),
  es_macho = as.integer(es_macho)
)

fit <- sampling(
  sm, data = stan_data,
  chains = 4, iter = 2000, warmup = 1000,
  seed = 0312,
  control = list(adapt_delta = 0.9, max_treedepth = 12)
)

print(fit, pars = c("a0","a1","b0","b1","k_tau","mu_tau_hembra","beta_macho_logmu"),
      probs = c(0.1,0.5,0.9))

# Mediana posterior
post <- rstan::extract(fit)

est <- c(
  a0 = median(post$a0),
  a1 = median(post$a1),
  b0 = median(post$b0),
  b1 = median(post$b1),
  k_tau = median(post$k_tau),
  mu_tau_hembra = median(post$mu_tau_hembra),
  beta_macho_logmu = median(post$beta_macho_logmu)
)

# verdad y estimado
tabla_final <- rbind(
  true = c(true$a0, true$a1, true$b0, true$b1,
           true$k_tau, true$mu_tau_hembra, true$beta_macho_logmu),
  est  = est
)

colnames(tabla_final) <- c("a0","a1","b0","b1",
                           "k_tau","mu_tau_hembra","beta_macho_logmu")

print(round(tabla_final, 4))
