parameters {
  real x;
  real y;
}
model {
  x ~ normal(2.0, 3.0);
  y ~ normal(-1.0, 2.2);
}
