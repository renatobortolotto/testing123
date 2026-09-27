familias <- c(
  "weibull",
  "exponential",
  "lognormal",
  "loglogistic"
)

ajustar_familia <- function(familia) {
  ajuste <- withCallingHandlers(
    survival::survreg(
      survival::Surv(
        inferior,
        superior,
        type = "interval2"
      ) ~ 1,
      data = dados,
      weights = peso,
      dist = familia,
      na.action = na.fail,
      control = survival::survreg.control(maxiter = 100)
    ),
    warning = function(w) {
      stop(
        paste(familia, conditionMessage(w), sep = ": "),
        call. = FALSE
      )
    }
  )

  if (!is.null(ajuste$fail)) {
    stop(paste(familia, ajuste$fail, sep = ": "))
  }

  parametros <- c(coef(ajuste), ajuste$scale)

  if (any(!is.finite(parametros)) || ajuste$scale <= 0) {
    stop(paste("Parâmetros inválidos no ajuste:", familia))
  }

  if (!is.finite(as.numeric(logLik(ajuste)))) {
    stop(paste("Log-verossimilhança inválida:", familia))
  }

  ajuste
}

modelos_tempo <- setNames(
  lapply(familias, ajustar_familia),
  familias
)

comparacao <- do.call(
  rbind,
  lapply(names(modelos_tempo), function(familia) {
    ajuste <- modelos_tempo[[familia]]
    log_verossimilhanca <- logLik(ajuste)

    data.frame(
      distribuicao = familia,
      n_parametros = attr(log_verossimilhanca, "df"),
      log_verossimilhanca = as.numeric(log_verossimilhanca),
      AIC = AIC(ajuste)
    )
  })
)

comparacao$delta_AIC <- comparacao$AIC - min(comparacao$AIC)
comparacao <- comparacao[order(comparacao$AIC), ]
rownames(comparacao) <- NULL

print(comparacao)