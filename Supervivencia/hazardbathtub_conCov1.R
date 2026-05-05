# CON COVARIABLES
# Covariables: sexo (0/1), edad (dist. normal)
# h(t|x) = h0(t) * exp(x' * gamma)
# H(t|x) = H0(t) * exp(x' * gamma)


library(tibble)
library(dplyr)
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

  // covariables
  int<lower=1> P;
  matrix[N, P] X;
}

parameters {
  // Para hazard bathtub en Chen: 0 < beta < 1 (art.)
  real<lower=0,upper=1> beta;

  // escala
  real<lower=0> lambda;

  // coeficientes riesgos prop
  vector[P] gamma;
}

model {
  // Priors (hay que mover parámetros)
  beta   ~ beta(2, 5);       // depende qué tan marcada sea
  lambda ~ lognormal(0, 1);  // >0

  gamma  ~ normal(0, 1);

  // verosimilitud bajo riesgos prop: delta*(log h0 + xg) - H0*exp(xg)
  for (i in 1:N) {
    real H0 = chen_H(time[i], beta, lambda);
    real logh0 = chen_log_h(time[i], beta, lambda);
    real linpred = dot_product(row(X, i), gamma);
    target += event[i] * (logh0 + linpred) - H0 * exp(linpred);
  }
}

generated quantities {
  real beta_out = beta;
  real lambda_out = lambda;
}
'
writeLines(stan_codigo, "chen_ph_cov_only.stan")

#Compilarlo
mod <- cmdstan_model("chen_ph_cov_only.stan")

# Generamos datos y covariables para probarlo
set.seed(123)
N <- 200

sexo <- rbinom(N, 1, 0.5)   # 0 = mujer, 1 = hombre (ej.)
edad <- round(rnorm(N, mean=40, sd=12)) # edad bajo una dist. normal
edad_z <- as.numeric(scale(edad))

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
dat <- data.frame(
  time = time_observed,
  event = event,
  sexo = sexo,
  edad_z = edad_z
)

# Matriz de covariables (sexo, edad, ..., +)
X <- cbind(sexo = dat$sexo, edad_z = dat$edad_z)
P <- ncol(X)

# Aplicación
stan_data <- list(
  N = nrow(dat),
  time = dat$time,
  event = dat$event,
  P = P,
  X = X
)

# Ajuste del modelo
fit <- mod$sample(
  data = stan_data,
  chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  seed = 123
)

print(fit$summary(c("beta_out","lambda_out","gamma")))


# Graficar hazard posterior 

draws <- fit$draws(c("beta_out","lambda_out","gamma"))
d <- posterior::as_draws_df(draws)

tgrid <- seq(0.001, max(dat$time), length.out = 400)

haz0_chen <- function(t, beta, lambda) {
  lambda * beta * t^(beta - 1) * exp(t^beta)
}

hazx_chen <- function(t, xrow, beta, lambda, gamma) {
  h0 <- haz0_chen(t, beta, lambda)
  lp <- sum(xrow * gamma)
  h0 * exp(lp)
}

# Selección de perfiles a comparar:
# - Mujer (sexo=0) vs Hombre (sexo=1)
# - Edad promedio (edad_z = 0)
x_mujer  <- c(sexo=0, edad_z=0)
x_hombre <- c(sexo=1, edad_z=0)

# comparación de edades: joven (edad_z=-1) vs mayor (edad_z=+1)
x_joven_mujer <- c(sexo=0, edad_z=-1)
x_mayor_mujer <- c(sexo=0, edad_z=+1)

# muestreamos 200 veces (puede ser más pero es tardado)
set.seed(123)
idx <- sample(seq_len(nrow(d)), 300)

resumen_haz <- function(xrow){
  haz_mat <- sapply(tgrid, function(tt){
    vals <- sapply(idx, function(j){
      gamma_j <- c(d$`gamma[1]`[j], d$`gamma[2]`[j])
      hazx_chen(tt, xrow, d$beta_out[j], d$lambda_out[j], gamma_j)
    })
    c(mean(vals), quantile(vals, 0.05), quantile(vals, 0.95))
  })
  list(mean=haz_mat[1,], lo=haz_mat[2,], hi=haz_mat[3,])
}


# Plot: Mujer vs Hombre

mujer  <- resumen_haz(x_mujer)
hombre <- resumen_haz(x_hombre)

df_plot <- bind_rows(
  tibble(t = tgrid, mean = mujer$mean,  lo = mujer$lo,  hi = mujer$hi,  perfil = "Mujer"),
  tibble(t = tgrid, mean = hombre$mean, lo = hombre$lo, hi = hombre$hi, perfil = "Hombre")
)

ggplot(df_plot, aes(x = t, y = mean, color = perfil, fill = perfil)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.18, color = NA) +
  geom_line(linewidth = 0.8) +
  labs(
    title = "Hazard posterior (Chen bathtub) con covariables",
    x = "Tiempo t",
    y = "Función de riesgo h(t|x)",
    color = "Perfil",
    fill  = "Perfil"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top", plot.title = element_text(face="bold"))

#zoom
ggplot(df_plot, aes(x = t, y = mean, color = perfil, fill = perfil)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.18, color = NA) +
  geom_line(linewidth = 0.8) +
  labs(
    title = "Hazard posterior (Chen bathtub) con covariables",
    x = "Tiempo t",
    y = "Función de riesgo h(t|x)"
  ) +
  coord_cartesian(
    xlim = c(0, 1.2),   
    ylim = c(0, 0.8)    
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "top",
    plot.title = element_text(face = "bold")
  )


# Mujer joven vs mayor
joven <- resumen_haz(x_joven_mujer)
mayor <- resumen_haz(x_mayor_mujer)

df_plot_edad <- bind_rows(
  tibble(t = tgrid, mean = joven$mean, lo = joven$lo, hi = joven$hi,
         perfil = "Mujer joven"),
  tibble(t = tgrid, mean = mayor$mean, lo = mayor$lo, hi = mayor$hi,
         perfil = "Mujer mayor")
)

ggplot(df_plot_edad,
       aes(x = t, y = mean, color = perfil, fill = perfil)) +
  geom_ribbon(aes(ymin = lo, ymax = hi),
              alpha = 0.18, color = NA) +
  geom_line(linewidth = 0.8) +
  labs(
    title = "Hazard posterior: mujer joven vs mujer mayor (Chen bathtub)",
    x = "Tiempo t",
    y = "Función de riesgo h(t|x)",
    color = "Perfil",
    fill  = "Perfil"
  ) +
  scale_color_manual(
    values = c(
      "Mujer joven" = "#F4A7C1",  
      "Mujer mayor" = "#C7B7E2"   
    )
  ) +
  scale_fill_manual(
    values = c(
      "Mujer joven" = "#F4A7C1",
      "Mujer mayor" = "#C7B7E2"
    )
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "top",
    plot.title = element_text(face = "bold"),
    legend.title = element_text(face = "bold")
  )

#zoom 
ggplot(df_plot_edad,
       aes(x = t, y = mean, color = perfil, fill = perfil)) +
  geom_ribbon(
    aes(ymin = lo, ymax = hi),
    alpha = 0.18,
    color = NA
  ) +
  geom_line(linewidth = 0.8) +
  labs(
    title = "Hazard posterior: mujer joven vs mujer mayor (Chen bathtub) - SIN estrés",
    subtitle = "Bandas creíbles 90% (zoom)",
    x = "Tiempo t",
    y = "Función de riesgo h(t|x)",
    color = "Perfil",
    fill  = "Perfil"
  ) +
  coord_cartesian(
    xlim = c(0, 1.2),
    ylim = c(0, 0.8)
  ) +
  scale_color_manual(
    values = c(
      "Mujer joven" = "#F4A7C1",  
      "Mujer mayor" = "#C7B7E2"   
    )
  ) +
  scale_fill_manual(
    values = c(
      "Mujer joven" = "#F4A7C1",
      "Mujer mayor" = "#C7B7E2"
    )
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "top",
    plot.title = element_text(face = "bold"),
    legend.title = element_text(face = "bold")
  )


