%r

max_grupos <- 200000L

consulta <- sprintf(
  "
  SELECT tipo_censura, dur_min, dur_max, peso
  FROM global_temp.nba_sm_tempo_validacao
  LIMIT %d
  ",
  max_grupos + 1L
)

validacao <- dplyr::collect(
  sparklyr::sdf_sql(sc, consulta)
)

if (nrow(validacao) == 0L || nrow(validacao) > max_grupos) {
  stop("A validação ficou vazia ou excedeu o limite de grupos.")
}

if (any(!is.finite(validacao$peso)) ||
    any(validacao$peso <= 0)) {
  stop("Foram encontrados pesos inválidos.")
}

# Mesma unidade utilizada no treinamento.
validacao$inferior <- validacao$dur_min / 86400
validacao$superior <- validacao$dur_max / 86400

# Retorna log-densidade, log-CDF ou log-sobrevivência.
log_funcao <- function(t, ajuste, densidade = FALSE,
                       acumulada = FALSE) {
  if (length(coef(ajuste)) != 1L ||
      length(ajuste$scale) != 1L) {
    stop("Este bloco exige modelos apenas com intercepto.")
  }

  mu <- unname(coef(ajuste)[1])
  sigma <- unname(ajuste$scale)
  familia <- ajuste$dist

  if (!is.finite(mu) || !is.finite(sigma) || sigma <= 0) {
    stop("Parâmetros inválidos no modelo.")
  }

  if (familia %in% c("weibull", "exponential")) {
    if (densidade) {
      return(dweibull(
        t, shape = 1 / sigma, scale = exp(mu), log = TRUE
      ))
    }

    return(pweibull(
      t,
      shape = 1 / sigma,
      scale = exp(mu),
      lower.tail = acumulada,
      log.p = TRUE
    ))
  }

  if (!familia %in% c("lognormal", "loglogistic")) {
    stop(paste("Família não implementada:", familia))
  }

  z <- (log(t) - mu) / sigma

  if (densidade) {
    log_densidade <- if (familia == "lognormal") {
      dnorm(z, log = TRUE)
    } else {
      dlogis(z, log = TRUE)
    }

    return(log_densidade - log(sigma) - log(t))
  }

  if (familia == "lognormal") {
    return(pnorm(
      z, lower.tail = acumulada, log.p = TRUE
    ))
  }

  plogis(z, lower.tail = acumulada, log.p = TRUE)
}

avaliar_modelo <- function(familia) {
  ajuste <- modelos_tempo[[familia]]

  exata <- validacao$tipo_censura == "exata"
  intervalo <- validacao$tipo_censura == "intervalo"
  direita <- validacao$tipo_censura == "direita"

  inferior <- validacao$inferior
  superior <- validacao$superior
  log_l <- rep(NA_real_, nrow(validacao))

  log_l[exata] <- log_funcao(
    inferior[exata], ajuste, densidade = TRUE
  )

  log_l[direita] <- log_funcao(
    inferior[direita], ajuste
  )

  # Probabilidade intervalar: F(U) - F(L) = S(L) - S(U).
  # Escolhe a cauda mais adequada para reduzir perda de precisão.
  log_f_u <- log_funcao(
    superior[intervalo], ajuste, acumulada = TRUE
  )
  log_f_l <- log_funcao(
    inferior[intervalo], ajuste, acumulada = TRUE
  )
  log_s_l <- log_funcao(inferior[intervalo], ajuste)
  log_s_u <- log_funcao(superior[intervalo], ajuste)

  usar_cdf <- log_f_u < log(0.5)

  log_maior <- ifelse(usar_cdf, log_f_u, log_s_l)
  log_menor <- ifelse(usar_cdf, log_f_l, log_s_u)

  log_l[intervalo] <- (
    log_maior + log(-expm1(log_menor - log_maior))
  )

  if (any(!is.finite(log_l))) {
    stop(paste(
      "Contribuição inválida ou problema numérico em:", familia
    ))
  }

  data.frame(
    distribuicao = familia,
    n_permanencias = sum(validacao$peso),
    nll_media = -weighted.mean(log_l, validacao$peso)
  )
}

validacao_modelos <- do.call(
  rbind,
  lapply(names(modelos_tempo), avaliar_modelo)
)

validacao_modelos <- validacao_modelos[
  order(validacao_modelos$nll_media),
]

rownames(validacao_modelos) <- NULL
print(validacao_modelos, row.names = FALSE)