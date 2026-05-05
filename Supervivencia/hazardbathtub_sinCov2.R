library(cmdstanr)
library(posterior)
library(ggplot2)

# ARCHIVO DE STAN
stan_codigo <- '
functions {
   // log de la función de riesgo de Chen (h(t) = lambda * beta * t^(beta-1) * exp(t^beta))
  real chen_log_h(real t, real beta, real lambda) {
    return log(lambda) + log(beta) + (beta - 1) * log(t) + pow(t, beta);
  }

  // función de riesgo acumulado (H(t) = lambda * (exp(t^beta) - 1))
  real chen_H(real t, real beta, real lambda) {
    return lambda * (exp(pow(t, beta)) - 1);
  }
}

data {
  int<lower=1> N;
  vector<lower=0>[N] time;
  array[N] int<lower=0,upper=1> event;   // 1=falla, 0=censura
}

parameters {
  // Para hazard bathtub en Chen: 0 < beta < 1 (art.)
  real<lower=0,upper=1> beta;

  // lambda > 0
  real<lower=0> lambda;
}

model {
  // Priors (hay que mover parámetros)
  beta   ~ beta(2, 5);       // depende qué tan marcada sea
  lambda ~ lognormal(0, 1);       // >0

  // verosimilirud con censura
  for (i in 1:N) {
    real H = chen_H(time[i], beta, lambda);
    if (event[i] == 1) {
      target += chen_log_h(time[i], beta, lambda) - H;
    } else {
      target += -H;
    }
  }
}

generated quantities {
  real beta_out = beta;
  real lambda_out = lambda;
}
'
writeLines(stan_codigo, "chen_bathtub_only.stan")
#Compilarlo
mod <- cmdstan_model("chen_bathtub_only.stan")

# Generamos datos para probarlo
set.seed(123)
N <- 200

# tiempos de falla (mezcla de distribuciones o puede ser directamente dist. Chen)
time_failure <- c(
  rweibull(N/2, shape = 1.5, scale = 0.8),
  rweibull(N/2, shape = 2.5, scale = 1.5)
)

# Tiempo de censura (ej: estudio de 2 años)
censor_time <- runif(N, 1.5, 2.5)

# Tiempo observado = mínimo( tiempo de falla, tiempo de censura)
time_observed <- pmin(time_failure, censor_time)

# Evento = 1 si falló, 0 si fue censurado
event <- as.integer(time_failure <= censor_time)

# dataframe
dat <- data.frame(time = time_observed, event = event)

# Aplicación
stan_data <- list(
  N = nrow(dat),
  time = dat$time,
  event = dat$event
)

# Ajuste del modelo
fit <- mod$sample(
  data = stan_data,
  chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  seed = 123
)

print(fit$summary(c("beta_out","lambda_out")))

# Graficar hazard posterior 
draws <- fit$draws(c("beta_out","lambda_out"))
d <- posterior::as_draws_df(draws)

tgrid <- seq(0.001, max(dat$time), length.out = 400)


haz_chen <- function(t, beta, lambda) {
  lambda * beta * t^(beta - 1) * exp(t^beta)
}

set.seed(123)
idx <- sample(seq_len(nrow(d)), 200)

haz_mat <- sapply(tgrid, function(tt){
  vals <- sapply(idx, function(j){
    haz_chen(tt, d$beta_out[j], d$lambda_out[j])
  })
  c(mean(vals), quantile(vals, 0.05), quantile(vals, 0.95))
})

haz_mean <- haz_mat[1,]
haz_lo   <- haz_mat[2,]
haz_hi   <- haz_mat[3,]

df_haz <- data.frame(t = tgrid, mean = haz_mean, lo = haz_lo, hi = haz_hi)

#plot
ggplot(df_haz, aes(x = t)) +
  geom_ribbon(
    aes(ymin = lo, ymax = hi),
    fill  = "#F4A7C1",   
    alpha = 0.35
  ) +
  geom_line(
    aes(y = mean),
    color = "#C26AA0",   
    linewidth = 1
  ) +
  labs(
    x = "Tiempo (t)",
    y = "Función de riesgo h(t)",
    title = "Hazard posterior: Chen (bathtub)",
    subtitle = "Media posterior y banda confianza 90%"
  ) +
  theme_minimal(base_size = 13)


#zoom
ggplot(df_haz, aes(x = t)) +
  geom_ribbon(
    aes(ymin = lo, ymax = hi),
    fill  = "#F4A7C1",   
    alpha = 0.35
  ) +
  geom_line(
    aes(y = mean),
    color = "#C26AA0",  
    linewidth = 1
  ) +
  labs(
    x = "Tiempo (t)",
    y = "Función de riesgo h(t)",
    title = "Hazard posterior: Chen (bathtub)",
    subtitle = "Media posterior y banda confianza 90%"
  ) +
  coord_cartesian(
    xlim = c(0, 1.2),   
    ylim = c(0, 0.8)    
  ) +
  theme_minimal(base_size = 13)


