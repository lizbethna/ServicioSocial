# Caso 3: h(R) = e + a R^m + c(1 - R)^n
set.seed(123)
library(cmdstanr)
## cmdstanr::install_cmdstan()
library(posterior)
library(bayesplot)
library(ggplot2)
library(deSolve) #Resuelve EDO

# ARCHIVO DE STAN
stan_codigo <- '
functions {
  //h(R) = e + a R^m + c(1 - R)^n
  real h_of_R(real R, real e, real a, real m, real c, real n) {
    //R debe estar en (0,1) por ser proba
    real R_s = fmin(1.0 - 1e-12, fmax(1e-12, R));
    return e + a * pow(R_s, m) + c * pow(1.0 - R_s, n);
  }
  //La EDO dR/dt = - h(R) * R
  vector r_dot(real t, vector y,
               real e, real a, real m, real c, real n) {
    vector[1] dydt;
    real R  = y[1];
    real hR = h_of_R(R, e, a, m, c, n);
    dydt[1] = -fmax(0.0, hR) * fmax(1e-12, R);
    return dydt;
  }
}

data {
  int<lower=1> N;
  array[N] real<lower=0> time; //tiempos
  array[N] int<lower=0,upper=1> event; //falla o censura
  array[0] real x_r;
  array[0] int  x_i;
}

parameters {
  real<lower=0> e;
  real<lower=0> a;
  real<lower=0> m;
  real<lower=0> c;
  real<lower=0> n;
}

transformed parameters {
  vector[1] y0;
  real t0;
  array[N] vector[1] y_hat;

  y0[1] = 1.0;
  t0 = 0.0;
  //Para resolver la EDO en cada tiempo
  y_hat = ode_rk45_tol(
    r_dot, y0, t0, time,
    1e-8, 1e-8, 2000000,
    e, a, m, c, n
  );
}

model {
  //Los parámetros se pueden ajustar dependiendo de la forma (más o menos pronunciada)
  //Según el articulo e>0 riesgo base
  e ~ exponential(10);
  // pico inicial >0
  a ~ lognormal(log(1.0), 0.6);
  m ~ lognormal(log(2.0), 0.4);
  // desgaste
  c ~ lognormal(log(1.0), 0.4);
  n ~ lognormal(log(1.2), 0.3);

  //Verosimilitud de falla o censura
  for (i in 1:N) {
    real R_i = fmax(1e-12, fmin(1.0 - 1e-12, y_hat[i][1]));
    real h_i = h_of_R(R_i, e, a, m, c, n);
    
    // log R(t) si hay censura
    target += log(R_i);

    // + log h(t) si hay falla
    if (event[i] == 1)
      target += log(h_i);
  }
}

generated quantities {
  //log-verosimilitud
  vector[N] log_lik;
  for (i in 1:N) {
    real R_i = fmax(1e-12, fmin(1.0 - 1e-12, y_hat[i][1]));
    real h_i = h_of_R(R_i, e, a, m, c, n);
    log_lik[i] = log(R_i) + (event[i] == 1 ? log(h_i) : 0);
  }
}
'

# Compilarlo
file_stan <- write_stan_file(stan_codigo)
mod <- cmdstan_model(file_stan)

#Estos valores hacen que la hazard tenga forma "bathtub" pero se pueden ajustar
true <- c(e = 0.002, a = 1.2, m = 2.2, c = 0.5, n = 2.5)

#Para la ecuación diferencial dR/dt = -h(R)R donde h(R) = e + a R^m + c(1 - R)^n.
ode_rhs <- function(t, state, pars){
  R <- state[1]
  R_s <- min(1 - 1e-12, max(1e-12, R))
  e <- pars["e"]; a <- pars["a"]; m <- pars["m"]; c <- pars["c"]; n <- pars["n"]
  hR <- e + a * (R_s^m) + c * ((1 - R_s)^n)
  dR <- -hR * R_s
  list(c(dR))
}

# Resolvemos R(t) 
tmax_sim <- 400
tgrid_sim <- seq(0, tmax_sim, length.out = 8000)
sol_sim <- ode(y = c(R = 1), times = tgrid_sim, func = ode_rhs, parms = true, method = "rk4")
Rgrid <- sol_sim[, "R"]


inv_R <- function(u){
  idx <- which(Rgrid <= u)[1]
  if (is.na(idx)) return(tmax_sim)
  tgrid_sim[idx]
}

N_sim <- 160
U <- runif(N_sim)
time <- sort(vapply(U, inv_R, numeric(1)))

# Simulamos los datos para probarlo (falla y censura)
cens <- runif(N_sim, min = 150, max = 400)
event <- as.integer(time <= cens)
time <- pmin(time, cens)

dat <- data.frame(time = time, event = event)

stan_data <- list(
  N = nrow(dat),
  time = dat$time,               
  event = dat$event,        
  x_r = numeric(0),
  x_i = integer(0)
)

stopifnot(all(is.finite(stan_data$time)), all(stan_data$time >= 0))
stan_data$time <- sort(stan_data$time)


#Ajuste del modelo
fit <- mod$sample(
  data = stan_data,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 800,
  iter_sampling = 800,
  refresh = 200
)

#Resumen de los parametros
print(fit$summary(c("e","a","m","c","n")))

# POSTERIOR: h(t)=h(R(t))
set.seed(123)
d <- as_draws_df(fit$draws(c("e","a","m","c","n")))
idx <- sample(seq_len(nrow(d)), 250)
tgrid <- seq(0, 400, length.out = 2000)


h_of_R_R <- function(R,e,a,m,c,n){
  R <- pmin(1-1e-12, pmax(1e-12, R))
  e + a * R^m + c * (1-R)^n
}

haz_draws <- matrix(NA_real_, nrow = length(idx), ncol = length(tgrid))

for(k in seq_along(idx)){
  j <- idx[k]
  pars <- c(e=d$e[j], a=d$a[j], m=d$m[j], c=d$c[j], n=d$n[j])
  
  sol <- ode(y=c(R=1), times=tgrid, func=ode_rhs, parms=pars, method="rk4")
  R_vals <- sol[,"R"]
  
  haz_draws[k,] <- h_of_R_R(R_vals, pars["e"],pars["a"],pars["m"],pars["c"],pars["n"])
}

df_haz <- data.frame(
  t    = tgrid,
  mean = apply(haz_draws, 2, mean),
  lo   = apply(haz_draws, 2, quantile, probs=0.025),
  hi   = apply(haz_draws, 2, quantile, probs=0.975)
)

# GRÁFICAS
set.seed(123)
ggplot(df_haz, aes(x=t)) +
  geom_ribbon(aes(ymin=lo, ymax=hi), alpha=0.30, fill="pink") +
  geom_line(aes(y=mean), linewidth=1, color="maroon") +
  labs(
    title="Caso 3 - Hazard - Reliability R(t) ",
    subtitle="h(R)= e + a R^m + c(1-R)^n,  dR/dt=-h(R)R",
    x="Tiempo", y="h(t)") +
  coord_cartesian(xlim=c(0, 10), ylim = c(0,0.7))+
  theme_minimal()

# Hazard individuales
sel <- sample(seq_len(nrow(haz_draws)), 20)
H <- haz_draws[sel, , drop = FALSE]
stopifnot(ncol(H) == length(tgrid))

df_long <- data.frame(
  draw = rep(seq_along(sel), each = length(tgrid)),
  t    = rep(tgrid, times = length(sel)),
  h    = as.vector(t(H))
)

ggplot() +
  geom_line(data=df_long, aes(t, h, group=draw), alpha=0.5, color="maroon") +
  geom_ribbon(data=df_haz, aes(t, ymin=lo, ymax=hi), alpha=0.20, fill="pink") +
  labs(
    title="Caso 3 Hazard (reliability) - (individuales)",
    x="Tiempo", y="h(t)"
  ) +
  coord_cartesian(xlim=c(0, 10), ylim = c(0,0.7))+
  theme_minimal()


# h(R) en términos de R
Rgrid2 <- seq(1e-6, 1-1e-6, length.out = 400)
pars_hat <- c(e=mean(d$e), a=mean(d$a), m=mean(d$m), c=mean(d$c), n=mean(d$n))
hR_hat <- h_of_R_R(Rgrid2, pars_hat["e"],pars_hat["a"],pars_hat["m"],pars_hat["c"],pars_hat["n"])

df_hR <- data.frame(R=Rgrid2, desgaste=1-Rgrid2, h=hR_hat)

ggplot(df_hR, aes(x = desgaste, y = h)) +
    geom_line() +
    labs(x="1 - R (desgaste)", y="h(R)", title="h(R)") +
    theme_minimal()

