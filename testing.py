    return (
        SQL_FEATURES.replace("__INICIO__", inicio.date().isoformat())
        .replace("__FIM__", fim.date().isoformat())
        .replace("__DATA_REFERENCIA__", fim.date().isoformat())
        .replace("__CAMPO_INFORMACAO__", campo)
    )