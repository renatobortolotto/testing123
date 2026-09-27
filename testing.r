%r

ajuste_lognormal <- modelos_tempo[["lognormal"]]

if (is.null(ajuste_lognormal) ||
    !inherits(ajuste_lognormal, "survreg")) {
  stop("O ajuste lognormal não foi encontrado em modelos_tempo.")
}

if (length(coef(ajuste_lognormal)) != 1L ||
    length(ajuste_lognormal$scale) != 1L) {
  stop("Este bloco pressupõe um modelo apenas com intercepto.")
}

mu <- unname(coef(ajuste_lognormal)[1])
sigma <- unname(ajuste_lognormal$scale)

if (!is.finite(mu) || !is.finite(sigma) || sigma <= 0) {
  stop("O ajuste retornou parâmetros inválidos.")
}

# O modelo foi ajustado com as durações convertidas para dias.
parametros_temporais <- data.frame(
  estado = "sem_acao:::classe",
  distribuicao = "lognormal",
  unidade_tempo = "dias",
  meanlog = mu,
  sdlog = sigma,
  mediana_dias = exp(mu)
)

print(parametros_temporais, row.names = FALSE)

# Compara diferentes tempos já transcorridos no silêncio.
dias_decorridos <- c(1, 5, 10, 30)
horizonte_dias <- 7

log_s_atual <- stats::plnorm(
  dias_decorridos,
  meanlog = mu,
  sdlog = sigma,
  lower.tail = FALSE,
  log.p = TRUE
)

log_s_futuro <- stats::plnorm(
  dias_decorridos + horizonte_dias,
  meanlog = mu,
  sdlog = sigma,
  lower.tail = FALSE,
  log.p = TRUE
)

if (any(!is.finite(c(log_s_atual, log_s_futuro)))) {
  stop("Não foi possível calcular as probabilidades nesses horizontes.")
}

probabilidade_saida <- -expm1(log_s_futuro - log_s_atual)

previsoes_saida <- data.frame(
  dias_ja_em_silencio = dias_decorridos,
  horizonte_dias = horizonte_dias,
  probabilidade_saida = probabilidade_saida,
  percentual_saida = round(100 * probabilidade_saida, 2)
)

print(previsoes_saida, row.names = FALSE)