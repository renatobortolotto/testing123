# Databricks notebook source
# MAGIC %md
# MAGIC # Propensao EGV — STG Kraken
# MAGIC Versao reconstruida a partir dos notebooks fornecidos em 25/10/2026.
# MAGIC Execute as definicoes; o registro e a validacao ficam em chamadas
# MAGIC explicitas no final. Nao ha treinamento, envio de e-mail ou instalacao
# MAGIC de pacotes durante a aplicacao semanal.
# MAGIC
# MAGIC Regras: segunda-feira 05h America/Sao_Paulo; janela entre duas
# MAGIC execucoes; A/CA; CVG Leves/Pesados; todas as propostas elegiveis;
# MAGIC exclusao por CPF de quem ja fechou na concorrencia; um grupo por CPF.

# COMMAND ----------
# MAGIC %run ./DEPLOY_CONTRACT

# COMMAND ----------

from copy import deepcopy
from datetime import datetime, time, timedelta
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo
import json
import re

import joblib
import numpy as np
import pandas as pd


# CONFIGURACAO DA EXECUCAO
FUSO = ZoneInfo("America/Sao_Paulo")
HORA_CORTE = 5
SAFRA_REPROCESSAMENTO = None  # Exemplo: "20261026"; deve ser segunda-feira.
FRAC_TESTE = 0.90
THRESHOLD = 0.50
DIAS_MATURACAO = 30
BUNDLE_ARTIFACT = "propensao/egv_model_bundle.joblib"

# BigView e disponibilizada pelo Kraken. Em homologacao manual, informar
# uma tabela HISTORICA com a BIG deste modelo e seu novo schema.
BIG_HISTORICA = "BigView"

# Para coluna DATETIME (sem fuso), usa a convencao da origem. Se a origem
# armazena horas UTC em DATETIME, alterar para "UTC" antes da homologacao.
FUSO_DATETIME_ORIGINACAO = "America/Sao_Paulo"
FUSO_DATETIME_PROPOSTA = "America/Sao_Paulo"

# A foto do create tem duas siglas diferentes. Use a sigla oficial do
# componente, ou mantenha None se ela ja estiver no DEPLOY_CONTRACT completo.
SIGLA_COMPONENTE = None

# Conserva os identificadores oficiais do DEPLOY_CONTRACT existente.
# O registro ao final devolve um novo run_id e o caminho do treino alinhado.
if "CONTRACT" not in globals():
    CONTRACT = {}

FAIXAS = [
    "Top 1%", "Top 5%", "Top 10%", "Top 20%",
    "Top 30%", "Top 50%", "Bottom 50%",
]
LIMITES_FAIXAS = [0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]
COLUNA_PROPOSTA = "proposta_ultimo_egv_alvo"


# COMMAND ----------
# Referencia temporal


def resolver_safra(agora: datetime) -> str:
    """Retorna a ultima segunda-feira cujo corte das 05h ja foi atingido."""
    if agora.tzinfo is None:
        raise ValueError("A referencia deve conter fuso horario.")
    local = agora.astimezone(FUSO)
    segunda = local.date() - timedelta(days=local.weekday())
    corte = datetime.combine(segunda, time(HORA_CORTE), tzinfo=FUSO)
    if local < corte:
        segunda -= timedelta(days=7)
    return segunda.strftime("%Y%m%d")


# Uma referencia por execucao: atraso do job nao altera o limite das 05h.
_SAFRA_EXECUCAO = resolver_safra(datetime.now(FUSO))


def get_safra() -> str:
    """Identifica a segunda-feira de fechamento da janela, em YYYYMMDD."""
    safra = SAFRA_REPROCESSAMENTO or _SAFRA_EXECUCAO
    calcular_janela(safra)
    return safra


def calcular_janela(safra: str) -> tuple[datetime, datetime]:
    """Intervalo semanal [segunda anterior 05h, segunda da safra 05h)."""
    if not re.fullmatch(r"\d{8}", safra):
        raise ValueError("Safra deve ter formato YYYYMMDD.")
    dia = datetime.strptime(safra, "%Y%m%d").date()
    if dia.weekday() != 0:
        raise ValueError("A safra semanal deve corresponder a segunda-feira.")
    fim = datetime.combine(dia, time(HORA_CORTE), tzinfo=FUSO)
    return fim - timedelta(days=7), fim


# COMMAND ----------
# Preparacao, scores e grupos


def preparar_features(
    df: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """Preserva a ordem salva; valores invalidos/infinitos tornam-se NaN."""
    if len(feature_cols) != len(set(feature_cols)):
        raise ValueError("O artefato contem features duplicadas.")
    faltantes = sorted(set(feature_cols) - set(df.columns))
    if faltantes:
        raise ValueError(f"Features ausentes na base: {faltantes}")
    x = df.loc[:, feature_cols].copy()
    for coluna in feature_cols:
        x[coluna] = pd.to_numeric(x[coluna], errors="coerce")
    return x.replace([np.inf, -np.inf], np.nan).astype("float64")


def calcular_gh(scores, splits) -> np.ndarray:
    """GH crescente; igualdade com o corte fica no grupo inferior (treino)."""
    valores = np.asarray(scores, dtype=float)
    cortes = np.asarray(splits, dtype=float)
    if valores.ndim != 1 or cortes.ndim != 1:
        raise ValueError("Scores e cortes devem ser vetores.")
    if not np.isfinite(valores).all():
        raise ValueError("Score nao finito recebido.")
    if ((valores < 0) | (valores > 1)).any():
        raise ValueError("Probabilidades devem estar entre zero e um.")
    if not np.isfinite(cortes).all() or not (np.diff(cortes) > 0).all():
        raise ValueError("Cortes GH devem ser finitos e crescentes.")
    return (np.searchsorted(cortes, valores, side="left") + 1).astype(int)


def adicionar_faixas(df: pd.DataFrame) -> pd.DataFrame:
    """Replica o rank percentual do exemplo, incluindo empates medios."""
    saida = df.copy()
    saida["rank_pct"] = saida["prob1"].rank(
        pct=True, ascending=False, method="average"
    )
    saida["faixa"] = pd.cut(
        saida["rank_pct"], bins=LIMITES_FAIXAS, labels=FAIXAS,
        right=True, include_lowest=False,
    ).astype("object")
    return saida


def atribuir_grupos(
    df: pd.DataFrame, safra: str, frac_teste: float = 0.90
) -> pd.DataFrame:
    """Sorteia CPFs por faixa e devolve TODAS as propostas originais.

    A proposta de menor score determina o estrato do CPF. Logo, qualquer
    proposta Bottom 50% coloca o CPF inteiro em controle. Nas demais faixas,
    floor(frac_teste * n_cpfs) CPFs vao para teste. O hash inclui a safra e
    estabiliza o sorteio frente a mudancas na ordem de leitura das linhas.
    Nao ha promessa de manter o grupo do CPF em semanas futuras.
    """
    if not 0 <= frac_teste <= 1:
        raise ValueError("frac_teste deve estar no intervalo [0, 1].")
    exigidas = {"cpf", COLUNA_PROPOSTA, "prob1", "faixa"}
    if not exigidas.issubset(df.columns):
        faltantes = sorted(exigidas - set(df.columns))
        raise ValueError(f"Faltam colunas: {faltantes}")
    saida = df.copy()
    if saida.empty:
        for nome in ["faixa_controle", "TipoControle", "grupocontrole"]:
            saida[nome] = pd.Series(index=saida.index, dtype="object")
        saida["TesteDigito"] = pd.Series(index=saida.index, dtype="int64")
        return saida
    if saida["cpf"].isna().any() or saida["faixa"].isna().any():
        raise ValueError("CPF e faixa devem estar preenchidos.")
    if not set(saida["faixa"].unique()).issubset(FAIXAS):
        raise ValueError("Faixa de score desconhecida.")

    pessoas = (
        saida.sort_values(["cpf", "prob1", COLUNA_PROPOSTA], kind="stable")
        .drop_duplicates("cpf", keep="first")
        .loc[:, ["cpf", "faixa"]]
        .rename(columns={"faixa": "faixa_controle"})
    )
    pessoas["_sorteio"] = pessoas["cpf"].map(
        lambda cpf: sha256(f"{safra}|{cpf}|EGV".encode()).hexdigest()
    )
    pessoas = pessoas.sort_values(["faixa_controle", "_sorteio", "cpf"])
    pessoas["TesteDigito"] = 1
    pessoas["TipoControle"] = "controle_score_baixo"
    elegiveis = pessoas["faixa_controle"].ne("Bottom 50%")
    grupos = pessoas.loc[elegiveis].groupby("faixa_controle", sort=False)
    posicao = grupos.cumcount() + 1
    tamanho = grupos["cpf"].transform("size")
    teste = posicao.div(tamanho).le(frac_teste)
    pessoas.loc[elegiveis, "TesteDigito"] = np.where(teste, 0, 1)
    pessoas.loc[elegiveis, "TipoControle"] = np.where(
        teste, "teste_proporcional", "controle_proporcional"
    )
    mapa = pessoas.set_index("cpf")
    for coluna in ["faixa_controle", "TesteDigito", "TipoControle"]:
        saida[coluna] = saida["cpf"].map(mapa[coluna])
    saida["TesteDigito"] = saida["TesteDigito"].astype(int)
    saida["grupocontrole"] = saida["TesteDigito"].astype(str)
    return saida


# COMMAND ----------
# Artefatos e acesso ao ambiente


def validar_bundle(bundle: dict) -> None:
    """Confere os componentes necessarios sem substituir dados ausentes."""
    obrigatorios = {"model_pipeline", "feature_cols", "splits", "calibrator"}
    faltantes = obrigatorios - set(bundle)
    if faltantes:
        raise ValueError(f"Bundle incompleto: {sorted(faltantes)}")
    if len(bundle["feature_cols"]) != 63:
        raise ValueError("Esta versao do STG espera as 63 features estudadas.")
    calcular_gh(np.array([], dtype=float), bundle["splits"])
    classes = np.asarray(bundle["model_pipeline"].classes_)
    if np.count_nonzero(classes == 1) != 1:
        raise ValueError("Nao foi identificada a classe positiva 1 no modelo.")


@lru_cache(maxsize=2)
def _carregar_bundle_run(run_id: str) -> dict:
    from mlflow.artifacts import download_artifacts

    with TemporaryDirectory(prefix="propensao_") as pasta:
        arquivo = download_artifacts(
            run_id=run_id, artifact_path=BUNDLE_ARTIFACT, dst_path=pasta
        )
        bundle = joblib.load(arquivo)
    validar_bundle(bundle)
    return bundle


def carregar_bundle() -> dict:
    """Carrega o pacote completo, vinculado a mesma run do contrato."""
    configuracoes = CONTRACT.get("mlflow", [])
    if not configuracoes or not configuracoes[0].get("run_id"):
        raise ValueError("Defina a run MLflow do modelo no CONTRACT.")
    return _carregar_bundle_run(configuracoes[0]["run_id"])


def _spark():
    from pyspark.sql import SparkSession

    session = globals().get("spark") or SparkSession.getActiveSession()
    if session is None:
        raise RuntimeError("Execute esta etapa no Databricks com Spark ativo.")
    return session


def dataprep_read_table_bigquery(query: str):
    """Mantem o conector e os parametros de materializacao do STG original."""
    from kraken.variables import KrakenVariables

    return (
        _spark().read.format("bigquery")
        .option("parentProject", KrakenVariables.GCP_PROJECT)
        .option("query", query)
        .option("viewsEnabled", "true")
        .option("materializationProject", "bv-pdbd-prd")
        .option("materializationDataset", "stg_tecn_dataexport")
        .load()
    )


def montar_query_features(safra: str, feature_cols: list[str]) -> str:
    """Usa o nome salvo no bundle para resolver l/i ilegivel na fotografia."""
    inicio, fim = calcular_janela(safra)
    ZoneInfo(FUSO_DATETIME_ORIGINACAO)
    candidatos = {"cdtipinformacaoltlr", "cdtipinformacaoitlr"}
    encontrados = candidatos.intersection(feature_cols)
    if len(encontrados) != 1:
        raise ValueError("Conferir a feature cdtipinformacao* no bundle.")
    campo = next(iter(encontrados))
    esperadas = set(FEATURE_DESCRIPTIONS)
    esperadas.remove("cdtipinformacaoltlr")
    esperadas.add(campo)
    if set(feature_cols) != esperadas:
        raise ValueError(
            "Features do bundle diferem da SQL transcrita: "
            f"{sorted(set(feature_cols).symmetric_difference(esperadas))}"
        )
    return (
        SQL_FEATURES.replace("__INICIO__", inicio.isoformat())
        .replace("__FIM__", fim.isoformat())
        .replace("__DATA_REFERENCIA__", fim.date().isoformat())
        .replace("__CAMPO_INFORMACAO__", campo)
        .replace("__FUSO_ORIGINACAO__", FUSO_DATETIME_ORIGINACAO)
    )


def get_base_egv_apnp(feature_cols: list[str]):
    """Extrai features por proposta, excluindo CPFs com concorrencia.

    O bloqueio ocorre na SQL, antes do scoring, das faixas e do sorteio.
    A consulta de concorrencia considera os dois grupos CVG, independente
    do status, desde o inicio da janela semanal, sem limite superior.
    """
    query = montar_query_features(get_safra(), feature_cols)
    return dataprep_read_table_bigquery(query)


# COMMAND ----------
# Saida BIG


def _normalizar_chaves(df: pd.DataFrame) -> pd.DataFrame:
    saida = df.copy()
    for coluna in ["cpf", COLUNA_PROPOSTA]:
        if coluna not in saida:
            raise ValueError(f"Coluna de identificacao ausente: {coluna}")
        if saida[coluna].isna().any():
            raise ValueError(
                f"Ha propostas sem {coluna}; revisar a ligacao com as fontes."
            )
        # Propostas/CPFs chegam como STRING da SQL; nao passar por float.
        saida[coluna] = saida[coluna].astype("string").str.strip()
    if not saida["cpf"].str.fullmatch(r"\d{11}").all():
        raise ValueError("CPF fora do formato de 11 digitos.")
    if not saida[COLUNA_PROPOSTA].str.fullmatch(r"\d+").all():
        raise ValueError("Numero de proposta deve conter somente digitos.")
    if saida[COLUNA_PROPOSTA].duplicated().any():
        raise ValueError("Join gerou mais de uma linha para a mesma proposta.")
    return saida


def escorar_dataframe(
    df: pd.DataFrame, bundle: dict, safra: str, model_grc: str,
    treino: bool = False,
) -> pd.DataFrame:
    """Infere com o pipeline aprovado; o modo treino conserva o target real."""
    validar_bundle(bundle)
    if treino:
        base = df.copy()
        for coluna in ["cpf", COLUNA_PROPOSTA]:
            if coluna not in base:
                base[coluna] = None
    else:
        base = _normalizar_chaves(df)
    features = list(bundle["feature_cols"])
    x = preparar_features(base, features)
    modelo = bundle["model_pipeline"]
    indice_positivo = int(np.flatnonzero(np.asarray(modelo.classes_) == 1)[0])
    if len(x):
        scores = modelo.predict_proba(x)[:, indice_positivo]
        if bundle["calibrator"] is not None:
            scores = bundle["calibrator"].predict(scores)
    else:
        scores = np.array([], dtype=float)

    saida = base.loc[:, ["cpf", COLUNA_PROPOSTA]].copy()
    for coluna in features:
        saida[coluna] = x[coluna]
    saida["idmodelo"] = str(model_grc)
    saida["idchavemodelo"] = saida[COLUNA_PROPOSTA]
    if treino:
        # Identificador TECNICO do registro de referencia; nunca uma proposta
        # ficticia. A coluna de proposta permanece nula se nao foi exportada.
        referencia = pd.Series(
            [f"referencia_treino:{i}" for i in range(len(saida))],
            index=saida.index,
        )
        saida["idchavemodelo"] = saida["idchavemodelo"].fillna(referencia)
    saida["safra"] = safra
    saida["prob1"] = np.asarray(scores, dtype=float)
    saida["predito"] = (saida["prob1"] >= THRESHOLD).astype(int)
    saida["gh"] = calcular_gh(scores, bundle["splits"])
    if treino:
        if "target" not in base or base["target"].isna().any():
            raise ValueError("A referencia de treino precisa do target real.")
        target = pd.to_numeric(base["target"], errors="raise")
        if not target.isin([0, 1]).all():
            raise ValueError("O target de treino deve ser zero ou um.")
        saida["target"] = target.astype(int)
    else:
        saida["target"] = None

    saida = adicionar_faixas(saida)
    if treino:
        # Nao inventa uma atribuicao de experimento para registros de treino.
        saida["faixa_controle"] = None
        saida["TesteDigito"] = None
        saida["TipoControle"] = None
        saida["grupocontrole"] = None
    else:
        saida = atribuir_grupos(saida, safra, FRAC_TESTE)
    return saida


def schema_big(feature_cols: list[str]):
    """Schema unico para BIG e referencia de treino, inclusive base vazia."""
    from pyspark.sql import types as T

    strings = [
        "idmodelo", "idchavemodelo", "cpf", COLUNA_PROPOSTA, "safra",
        "grupocontrole", "faixa", "faixa_controle", "TipoControle",
    ]
    inteiros = ["target", "predito", "gh", "TesteDigito"]
    doubles = ["prob1", "rank_pct", *feature_cols]
    return T.StructType(
        [T.StructField(c, T.StringType(), True) for c in strings]
        + [T.StructField(c, T.IntegerType(), True) for c in inteiros]
        + [T.StructField(c, T.DoubleType(), True) for c in doubles]
    )


def para_spark_big(df: pd.DataFrame, feature_cols: list[str]):
    from pyspark.sql import types as T

    schema = schema_big(feature_cols)
    registros = []
    for linha in df.loc[:, schema.names].itertuples(index=False, name=None):
        valores = []
        for valor, campo in zip(linha, schema.fields):
            if pd.isna(valor):
                valores.append(None)
            elif isinstance(campo.dataType, T.StringType):
                valores.append(str(valor))
            elif isinstance(campo.dataType, T.IntegerType):
                valores.append(int(valor))
            else:
                valores.append(float(valor))
        registros.append(tuple(valores))
    return _spark().createDataFrame(registros, schema=schema)


def apply_model():
    """Hook Kraken: exclui concorrencia e devolve BIG com teste/controle."""
    if not CONTRACT.get("model_grc"):
        raise ValueError("Defina model_grc no contrato oficial.")
    bundle = carregar_bundle()
    base = get_base_egv_apnp(list(bundle["feature_cols"])).toPandas()
    saida = escorar_dataframe(
        base, bundle, get_safra(), CONTRACT["model_grc"]
    )
    # Nao sobrescrever BigView: o apply_target precisa do historico.
    return para_spark_big(saida, list(bundle["feature_cols"]))


# COMMAND ----------
# Apuracao do target


def apply_target():
    """Apura AP por CPF, preservando a regra do STG e espera minima de 30d.

    A agenda semanal apura a coorte na primeira execucao com pelo menos
    DIAS_MATURACAO dias completos. Esta regra usa pagamentos observados desde
    a escoragem ate a apuracao; 30 dias e MATURACAO MINIMA, nao uma janela
    maxima nova de conversao. Uma coorte semanal normalmente sera apurada
    aos 35 dias. Propostas diferentes do CPF recebem seu mesmo desfecho.

    A BIG historica precisa conter o schema desta versao e target nulo para
    os registros ainda nao apurados. O executor Kraken faz a persistencia.
    """
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    spark_session = _spark()
    schema = T.StructType([
        T.StructField("idchavemodelo", T.StringType(), True),
        T.StructField("safra", T.StringType(), True),
        T.StructField("target", T.IntegerType(), True),
    ])
    historico = spark_session.table(BIG_HISTORICA)
    necessarias = {"idchavemodelo", "cpf", "safra", "target", "idmodelo"}
    if not necessarias.issubset(historico.columns):
        raise ValueError(
            "BIG historica sem o schema desta versao: "
            f"{sorted(necessarias - set(historico.columns))}"
        )
    pendentes = historico.filter(
        (F.col("idmodelo") == str(CONTRACT["model_grc"]))
        & F.col("target").isNull()
    )
    _, fim_apuracao = calcular_janela(get_safra())
    referencias = []
    for linha in pendentes.select("safra").distinct().collect():
        _, origem = calcular_janela(linha["safra"])
        if origem + timedelta(days=DIAS_MATURACAO) <= fim_apuracao:
            referencias.append((linha["safra"], int(origem.timestamp())))
    if not referencias:
        return spark_session.createDataFrame([], schema)

    marcos = spark_session.createDataFrame(
        referencias, "safra string, _inicio_target long"
    )
    base = pendentes.join(marcos, "safra", "inner").select(
        F.col("idchavemodelo").cast("string"),
        F.col("cpf").cast("string"), "safra", "_inicio_target",
    )
    minimo = min(item[1] for item in referencias)
    maximo = int(fim_apuracao.timestamp())
    ZoneInfo(FUSO_DATETIME_PROPOSTA)
    query = f"""
        WITH pagamentos AS (
            SELECT
                REGEXP_REPLACE(psa.NuCpfCnpj, r'[^0-9]+', '') AS cpf_pago,
                CASE TYPEOF(prop.DtStatusProposta)
                    WHEN 'TIMESTAMP' THEN
                        CAST(prop.DtStatusProposta AS TIMESTAMP)
                    WHEN 'DATETIME' THEN TIMESTAMP(
                        CAST(prop.DtStatusProposta AS DATETIME),
                        '{FUSO_DATETIME_PROPOSTA}'
                    )
                    ELSE ERROR('DtStatusProposta precisa conter horario')
                END AS data_pagamento
            FROM `bv-pdbd-prd.trusted_dbcred.tbproposta` AS prop
            INNER JOIN `bv-pdbd-prd.trusted_dbcred.tbpessoasobanalise` AS psa
                ON prop.NuProposta = psa.NuProposta
            WHERE TRIM(UPPER(prop.StProposta)) = 'AP'
              AND TRIM(prop.TpProposta) = '1'
              AND prop.CdSubProduto IN (
                  8, 19, 59, 60, 102, 103, 13, 20, 62, 63
              )
              AND psa.CdPapel = 1
              AND TRIM(UPPER(psa.TpPessoa)) = 'F'
        )
        SELECT DISTINCT
            cpf_pago,
            UNIX_SECONDS(data_pagamento) AS instante_pagamento
        FROM pagamentos
        WHERE data_pagamento >= TIMESTAMP_SECONDS({minimo})
          AND data_pagamento < TIMESTAMP_SECONDS({maximo})
    """
    pagos = dataprep_read_table_bigquery(query)
    cruzamento = base.join(
        pagos,
        (base["cpf"] == pagos["cpf_pago"])
        & (pagos["instante_pagamento"] >= base["_inicio_target"]),
        "left",
    )
    return cruzamento.groupBy("idchavemodelo", "safra").agg(
        F.max(F.when(F.col("cpf_pago").isNotNull(), 1).otherwise(0))
        .cast("int").alias("target")
    )


# COMMAND ----------
# Preparacao unica do registro


def preparar_artefatos_deploy(
    caminho_bundle: str,
    df_treino,
    caminho_referencia_treino: str,
) -> dict:
    """Executar uma vez, no Databricks, para registrar os artefatos completos.

    df_treino deve ser SOMENTE o treino aprovado, com features e target
    reais. Nao incluir OOT/validacao. Identificadores existentes sao
    preservados. Se a exportacao excluiu a proposta, ela fica nula e apenas
    idchavemodelo recebe uma chave tecnica da referencia, conforme o uso de
    metadados auxiliares no tutorial; nao se inventa uma proposta historica.
    Nenhum fit e executado. O destino Parquet deve ser um diretorio NOVO;
    errorifexists evita substituir o treino ou uma referencia ja existente.
    Esta funcao nao e chamada por apply_model nem automaticamente.
    """
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    if not CONTRACT.get("model_grc") or not CONTRACT.get("mlflow"):
        raise ValueError("Carregue o DEPLOY_CONTRACT antes do registro.")
    if mlflow.active_run() is not None:
        raise RuntimeError("Encerre a run ativa antes deste registro.")
    bundle = joblib.load(caminho_bundle)
    validar_bundle(bundle)
    features = list(bundle["feature_cols"])
    base = df_treino.copy() if isinstance(df_treino, pd.DataFrame) else (
        df_treino.toPandas()
    )
    if base.empty:
        raise ValueError("O conjunto de treino esta vazio.")
    # Se um identificador foi exportado como inteiro/float integral, restaurar
    # sua representacao decimal sem acrescentar identificadores ficticios.
    for coluna in ["cpf", COLUNA_PROPOSTA]:
        if coluna in base and pd.api.types.is_numeric_dtype(base[coluna]):
            valores = pd.to_numeric(base[coluna], errors="raise")
            preenchidos = valores.dropna()
            if (preenchidos % 1 != 0).any():
                raise ValueError(f"Identificador invalido no treino: {coluna}")
            if (preenchidos.abs() >= 2 ** 53).any():
                raise ValueError(
                    f"Identificador pode ter perdido precisao: {coluna}"
                )
            base[coluna] = valores.astype("Int64").astype("string")
    saida = escorar_dataframe(
        base, bundle, get_safra(), CONTRACT["model_grc"], treino=True
    )
    para_spark_big(saida, features).write.mode("errorifexists").parquet(
        caminho_referencia_treino
    )
    exemplo = preparar_features(base.head(20), features)
    modelo = bundle["model_pipeline"]
    assinatura = infer_signature(exemplo, modelo.predict(exemplo))
    bibliotecas = [
        f"{pacote}=={version(pacote)}"
        for pacote in [
            "numpy", "pandas", "scikit-learn", "lightgbm", "joblib", "mlflow"
        ]
    ]
    mlflow.set_experiment(CONTRACT["mlflow"][0]["experiment_name"])
    with mlflow.start_run(run_name="propensao-egv-kraken-completo") as run:
        mlflow.sklearn.log_model(
            sk_model=modelo,
            artifact_path="model",
            signature=assinatura,
            input_example=exemplo,
            pip_requirements=bibliotecas,
        )
        with TemporaryDirectory(prefix="registro_propensao_") as pasta:
            arquivo = Path(pasta) / "egv_model_bundle.joblib"
            joblib.dump(bundle, arquivo, compress=3)
            mlflow.log_artifact(str(arquivo), artifact_path="propensao")
        mlflow.log_params({
            "project": "EGV",
            "n_features": len(features),
            "gh_interval_side": "left",
        })
        mlflow.log_dict({
            "build_libraries": bibliotecas,
            "path_to_data_train": caminho_referencia_treino,
            "gh_interval_side": "left",
            "bundle_artifact": BUNDLE_ARTIFACT,
        }, "propensao/deploy_metadata.json")
        run_id = run.info.run_id

    # Verifica a serializacao na mesma amostra antes de atualizar o contrato.
    recuperado = _carregar_bundle_run(run_id)
    esperado = escorar_dataframe(
        base.head(20), bundle, get_safra(), CONTRACT["model_grc"], treino=True
    )
    obtido = escorar_dataframe(
        base.head(20), recuperado, get_safra(), CONTRACT["model_grc"],
        treino=True,
    )
    np.testing.assert_allclose(esperado["prob1"], obtido["prob1"], rtol=1e-12)
    np.testing.assert_array_equal(esperado["gh"], obtido["gh"])

    CONTRACT["mlflow"][0]["run_id"] = run_id
    CONTRACT["path_to_data_train"] = caminho_referencia_treino
    CONTRACT["build_libraries"] = bibliotecas
    return {
        "run_id": run_id,
        "path_to_data_train": caminho_referencia_treino,
        "build_libraries": bibliotecas,
        "n_treino": len(base),
    }


# COMMAND ----------
# Testes locais


def test_janela_semanal():
    inicio, fim = calcular_janela("20261026")
    assert inicio.isoformat() == "2026-10-19T05:00:00-03:00"
    assert fim.isoformat() == "2026-10-26T05:00:00-03:00"
    assert calcular_janela("20261102")[0] == fim


def test_cortes_gh():
    np.testing.assert_array_equal(
        calcular_gh([0.0, 0.2, 0.2001, 0.8, 0.8001, 1.0], [0.2, 0.8]),
        [1, 1, 2, 2, 3, 3],
    )


def test_grupos_por_cpf():
    exemplo = pd.DataFrame({
        "cpf": ["00000000001", "00000000001", "00000000002"],
        COLUNA_PROPOSTA: ["1", "2", "3"],
        "prob1": [0.9, 0.1, 0.8],
    })
    exemplo = adicionar_faixas(exemplo)
    saida = atribuir_grupos(exemplo, "20261026")
    assert len(saida) == len(exemplo)
    assert saida.groupby("cpf")["TesteDigito"].nunique().max() == 1
    assert saida.loc[saida["cpf"] == "00000000001", "TesteDigito"].eq(1).all()




# COMMAND ----------
# SQL completa: historico do treino preservado; ancora por proposta.
# O campo __CAMPO_INFORMACAO__ e resolvido pela feature_cols do bundle.
SQL_FEATURES = r"""
WITH params AS (
    SELECT
        [8, 19, 59, 60, 102, 103, 13, 20, 62, 63] AS subprodutos_egv,
        ['78', '79', '27', '6', '28'] AS modalidades_egv_relacionamento
),
originacao AS (
    SELECT
        tbo.NuProposta,
        tbo.NuCpfCnpj,
        tbo.NmGrupoProdutoMeta,
        tbo.StProposta,
        tbo.FlConcorrente,
        CASE TYPEOF(tbo.DtEntradaProposta)
            WHEN 'TIMESTAMP' THEN CAST(tbo.DtEntradaProposta AS TIMESTAMP)
            WHEN 'DATETIME' THEN TIMESTAMP(
                CAST(tbo.DtEntradaProposta AS DATETIME), '__FUSO_ORIGINACAO__'
            )
            ELSE ERROR('DtEntradaProposta precisa de horario: TIMESTAMP/DATETIME')
        END AS entrada_referencia
    FROM `bv-pdbd-prd.trusted_nego_rodas.tboriginacao` AS tbo
),
cpfs_concorrencia AS (
    -- Mesmo bloqueio por CPF do exemplo de uso, aplicado aos dois grupos.
    -- Nao restringir a A/CA: outra proposta/status pode sinalizar concorrencia.
    -- A referencia inferior agora e o inicio da janela semanal. Como no
    -- exemplo, nao ha limite superior: consulta-se a sinalizacao atual antes
    -- da geracao da BIG, inclusive se o job estiver atrasado/reprocessando.
    -- A data e a entrada da proposta, nao a data do fechamento na concorrencia.
    SELECT DISTINCT
        REGEXP_REPLACE(CAST(NuCpfCnpj AS STRING), r'[^0-9]+', '') AS cpf
    FROM originacao
    WHERE TRIM(NmGrupoProdutoMeta) IN ('CVG Leves', 'CVG Pesados')
      AND entrada_referencia >= TIMESTAMP('__INICIO__')
      AND CAST(FlConcorrente AS INT64) = 1
),
elegiveis AS (
    SELECT DISTINCT SAFE_CAST(NuProposta AS INT64) AS nuproposta
    FROM originacao
    WHERE TRIM(NmGrupoProdutoMeta) IN ('CVG Leves', 'CVG Pesados')
      AND TRIM(UPPER(StProposta)) IN ('A', 'CA')
      AND entrada_referencia >= TIMESTAMP('__INICIO__')
      AND entrada_referencia < TIMESTAMP('__FIM__')
),
proposta_base AS (
    SELECT
        NuProposta,
        TRIM(StProposta) AS StProposta,
        DtEntradaProposta,
        DtStatusProposta,
        CdLoja,
        CdSubProduto,
        TpProposta,
        pcentrada,
        pcmargemliquida,
        qtalteracao,
        qtdiascarencia,
        qtparcelas,
        txcetcalculada,
        txfinanciamento,
        txpactuada,
        vrcomissao,
        vrcustos,
        vrentrada,
        vrfinanciamento,
        vrpresente,
        vrproposta,
        vrsegurobvmais,
        vrspread,
        vrtac
    FROM `bv-pdbd-prd.trusted_dbcred.tbproposta`
    WHERE TRIM(TpProposta) = '1'
      AND DATE(DtEntradaProposta) >= DATE_SUB(
          DATE('__DATA_REFERENCIA__'),
          INTERVAL 4 YEAR
      )
),
pessoa AS (
    SELECT
        REGEXP_REPLACE(psa.NuCpfCnpj, r'[^0-9]+', '') AS cpf,
        psa.NuProposta,
        pb.DtEntradaProposta AS data_proposta_pessoa,
        pb.DtStatusProposta AS data_status_proposta_pessoa,
        psa.cdregraunicaperdaesperada,
        psa.__CAMPO_INFORMACAO__,
        psa.cdtipinformacaospcserasa,
        psa.cdtipoconfirmacaolr,
        psa.cdvigenciaregraunicaprdaespra,
        IF(TRIM(UPPER(psa.flalertapessoa)) = 'S', 1, 0) AS flalertapessoa,
        IF(TRIM(UPPER(psa.flclienteativo)) = 'S', 1, 0) AS flclienteativo,
        IF(TRIM(UPPER(psa.flconsultabv)) = 'S', 1, 0) AS flconsultabv,
        IF(TRIM(UPPER(psa.flexcliente)) = 'S', 1, 0) AS flexcliente,
        IF(TRIM(UPPER(psa.flrisco)) = 'S', 1, 0) AS flrisco,
        IF(TRIM(UPPER(psa.flscoredefinidopolitica)) = 'S', 1, 0)
            AS flscoredefinidopolitica,
        IF(TRIM(UPPER(psa.flserasa)) = 'S', 1, 0) AS flserasa,
        IF(TRIM(UPPER(psa.flspc)) = 'S', 1, 0) AS flspc,
        psa.nuscorefim,
        psa.pcpartsocio,
        IF(psa.pcpartsocio IS NOT NULL, 1, 0) AS fltemsocio,
        psa.qtbehavior,
        psa.qtscore
    FROM `bv-pdbd-prd.trusted_dbcred.tbpessoasobanalise` AS psa
    INNER JOIN proposta_base AS pb
        ON psa.NuProposta = pb.NuProposta
    WHERE psa.CdPapel = 1
      AND TRIM(UPPER(psa.TpPessoa)) = 'F'
      AND LENGTH(REGEXP_REPLACE(psa.NuCpfCnpj, r'[^0-9]+', '')) = 11
),
ultimo_egv_alvo_por_cpf AS (
    -- Nome legado mantido; cada proposta elegivel agora e uma ancora.
    SELECT DISTINCT
        p.cpf,
        e.NuProposta AS proposta_ultimo_egv_alvo,
        e.DtEntradaProposta AS data_ultimo_egv_alvo
    FROM elegiveis AS elig
    INNER JOIN `bv-pdbd-prd.trusted_dbcred.tbproposta` AS e
        ON elig.nuproposta = e.NuProposta
    INNER JOIN (
        SELECT DISTINCT
            NuProposta,
            REGEXP_REPLACE(NuCpfCnpj, r'[^0-9]+', '') AS cpf
        FROM `bv-pdbd-prd.trusted_dbcred.tbpessoasobanalise`
        WHERE CdPapel = 1
          AND TRIM(UPPER(TpPessoa)) = 'F'
          AND LENGTH(REGEXP_REPLACE(NuCpfCnpj, r'[^0-9]+', '')) = 11
    ) AS p
        ON e.NuProposta = p.NuProposta
),
pessoa_anterior_egv_alvo AS (
    SELECT
        u.cpf,
        u.proposta_ultimo_egv_alvo,
        u.data_ultimo_egv_alvo,
        p.NuProposta AS proposta_pessoa_anterior,
        p.data_proposta_pessoa AS data_pessoa_anterior,
        p.data_status_proposta_pessoa AS data_status_pessoa_anterior,
        p.cdregraunicaperdaesperada,
        p.__CAMPO_INFORMACAO__,
        p.cdtipinformacaospcserasa,
        p.cdtipoconfirmacaolr,
        p.cdvigenciaregraunicaprdaespra,
        p.flalertapessoa,
        p.flclienteativo,
        p.flconsultabv,
        p.flexcliente,
        p.flrisco,
        p.flscoredefinidopolitica,
        p.flserasa,
        p.flspc,
        p.nuscorefim,
        p.pcpartsocio,
        p.fltemsocio,
        p.qtbehavior,
        p.qtscore,
        IF(p.NuProposta IS NULL, 1, 0) AS fl_sem_pessoa_anterior
    FROM ultimo_egv_alvo_por_cpf AS u
    LEFT JOIN pessoa AS p
        ON u.cpf = p.cpf
       AND p.NuProposta <> u.proposta_ultimo_egv_alvo
       AND p.data_status_proposta_pessoa < u.data_ultimo_egv_alvo
       AND (
           p.data_proposta_pessoa < u.data_ultimo_egv_alvo
           OR (
               p.data_proposta_pessoa = u.data_ultimo_egv_alvo
               AND p.NuProposta < u.proposta_ultimo_egv_alvo
           )
       )
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY u.cpf, u.proposta_ultimo_egv_alvo
        ORDER BY
            p.data_status_proposta_pessoa DESC,
            p.data_proposta_pessoa DESC,
            p.NuProposta DESC
    ) = 1
),
base_historica AS (
    SELECT
        u.cpf,
        u.proposta_ultimo_egv_alvo,
        u.data_ultimo_egv_alvo,
        h.NuProposta,
        h.StProposta,
        h.DtEntradaProposta,
        h.DtStatusProposta,
        h.CdLoja,
        h.CdSubProduto,
        h.DtEntradaProposta < u.data_ultimo_egv_alvo
            AS fl_antes_ultimo_egv_alvo,
        h.DtStatusProposta < u.data_ultimo_egv_alvo
            AS fl_status_conhecido_antes_alvo,
        h.CdSubProduto IN UNNEST(prm.subprodutos_egv)
            AS fl_subproduto_egv,
        CASE
            WHEN h.CdLoja IS NOT NULL
             AND NULLIF(TRIM(CAST(h.CdLoja AS STRING)), '') IS NOT NULL
             AND TRIM(CAST(h.CdLoja AS STRING)) <> '0'
            THEN TRIM(CAST(h.CdLoja AS STRING))
        END AS cdloja_tratada
    FROM ultimo_egv_alvo_por_cpf AS u
    LEFT JOIN pessoa AS p
        ON u.cpf = p.cpf
    LEFT JOIN proposta_base AS h
        ON p.NuProposta = h.NuProposta
    CROSS JOIN params AS prm
),
historico_agg AS (
    SELECT
        cpf,
        proposta_ultimo_egv_alvo,
        data_ultimo_egv_alvo,
        COUNT(DISTINCT IF(
            fl_antes_ultimo_egv_alvo, NuProposta, NULL
        )) AS qtd_propostas_antes_ultimo_egv_alvo,
        COUNT(DISTINCT IF(
            fl_antes_ultimo_egv_alvo, cdloja_tratada, NULL
        )) AS qtd_lojas_antes_ultimo_egv_alvo,
        COUNTIF(
            fl_antes_ultimo_egv_alvo
            AND fl_status_conhecido_antes_alvo
            AND StProposta = 'A'
        ) AS qtd_status_a_antes_ultimo_egv_alvo,
        COUNTIF(
            fl_antes_ultimo_egv_alvo
            AND fl_status_conhecido_antes_alvo
            AND StProposta = 'AP'
        ) AS qtd_status_ap_antes_ultimo_egv_alvo,
        COUNTIF(
            fl_antes_ultimo_egv_alvo
            AND fl_status_conhecido_antes_alvo
            AND StProposta = 'CA'
        ) AS qtd_status_ca_antes_ultimo_egv_alvo,
        COUNTIF(
            fl_antes_ultimo_egv_alvo
            AND fl_status_conhecido_antes_alvo
            AND StProposta NOT IN ('A', 'AP', 'CA')
        ) AS qtd_status_outros_antes_ultimo_egv_alvo,
        COUNTIF(
            fl_antes_ultimo_egv_alvo
            AND fl_status_conhecido_antes_alvo
            AND fl_subproduto_egv
            AND StProposta = 'AP'
        ) AS qtd_eventos_egv_ap_antes_ultimo_egv_alvo,
        COUNTIF(
            fl_antes_ultimo_egv_alvo
            AND fl_status_conhecido_antes_alvo
            AND fl_subproduto_egv
            AND StProposta = 'CA'
        ) AS qtd_eventos_egv_ca_antes_ultimo_egv_alvo,
        COUNTIF(
            fl_antes_ultimo_egv_alvo
            AND fl_subproduto_egv
        ) AS qtd_eventos_egv_antes_ultimo_egv_alvo,
        COUNTIF(
            fl_antes_ultimo_egv_alvo
            AND NOT fl_subproduto_egv
        ) AS qtd_eventos_outros_produtos_antes_ultimo_egv_alvo,
        COUNT(DISTINCT IF(
            fl_antes_ultimo_egv_alvo, CdSubProduto, NULL
        )) AS qtd_subprodutos_distintos_antes_ultimo_egv_alvo,
        DATE_DIFF(
            DATE(data_ultimo_egv_alvo),
            DATE(MAX(IF(fl_antes_ultimo_egv_alvo, DtEntradaProposta, NULL))),
            DAY
        ) AS dias_desde_evento_anterior,
        DATE_DIFF(
            DATE(data_ultimo_egv_alvo),
            DATE(MIN(IF(fl_antes_ultimo_egv_alvo, DtEntradaProposta, NULL))),
            DAY
        ) AS janela_historica_dias,
        IF(
            COUNTIF(fl_antes_ultimo_egv_alvo) = 0,
            1,
            0
        ) AS fl_sem_historico_antes_ultimo_egv_alvo,
        IF(
            COUNTIF(fl_antes_ultimo_egv_alvo) = 0,
            1,
            0
        ) AS fl_entrada_egv_alvo
    FROM base_historica
    GROUP BY cpf, proposta_ultimo_egv_alvo, data_ultimo_egv_alvo
),
relacionamento_base AS (
    SELECT
        REGEXP_REPLACE(cd_cpf_cnpj, r'[^0-9]+', '') AS cpf,
        cd_conta_contrato,
        TRIM(cd_modalidade_produto) AS cd_modalidade_produto,
        ts_inicio_conta_contrato,
        ts_fim_conta_contrato,
        qt_dias_atraso,
        ts_fim_relacionamento,
        ts_inicio_relacionamento,
        fl_cliente_relacionado,
        fl_cliente_apto,
        fl_cliente_ativo,
        fl_cliente_prejuizo,
        nm_tipo_produto
    FROM `bv-pdbd-prd.dvry_clie_crm.clientebv_relacionamento_dime`
    WHERE LENGTH(REGEXP_REPLACE(cd_cpf_cnpj, r'[^0-9]+', '')) = 11
),
contratos_anteriores_alvo AS (
    SELECT
        u.cpf,
        u.proposta_ultimo_egv_alvo,
        u.data_ultimo_egv_alvo,
        rel.cd_conta_contrato,
        rel.ts_inicio_conta_contrato,
        rel.ts_fim_conta_contrato,
        rel.qt_dias_atraso,
        rel.ts_fim_relacionamento,
        rel.ts_inicio_relacionamento,
        rel.fl_cliente_relacionado,
        rel.fl_cliente_apto,
        rel.fl_cliente_ativo,
        rel.fl_cliente_prejuizo,
        IF(LOWER(rel.nm_tipo_produto) LIKE '%cart%', 1, 0) AS fl_possui_cartao,
        IF(LOWER(rel.nm_tipo_produto) LIKE '%conta%', 1, 0) AS fl_possui_conta
    FROM ultimo_egv_alvo_por_cpf AS u
    CROSS JOIN params AS prm
    LEFT JOIN relacionamento_base AS rel
        ON u.cpf = rel.cpf
       AND rel.cd_modalidade_produto NOT IN UNNEST(
           prm.modalidades_egv_relacionamento
       )
       AND rel.ts_inicio_conta_contrato < u.data_ultimo_egv_alvo
),
relacionamento_agg AS (
    SELECT
        cpf,
        proposta_ultimo_egv_alvo,
        data_ultimo_egv_alvo,
        COUNT(cd_conta_contrato) AS nm_total_outros_contratos_banco,
        COUNTIF(
            cd_conta_contrato IS NOT NULL
            AND ts_inicio_conta_contrato < data_ultimo_egv_alvo
            AND (
                ts_fim_conta_contrato IS NULL
                OR ts_fim_conta_contrato >= data_ultimo_egv_alvo
            )
        ) AS nm_total_contratos_ativos,
        COUNTIF(
            cd_conta_contrato IS NOT NULL
            AND ts_fim_conta_contrato < data_ultimo_egv_alvo
        ) AS nm_total_contratos_inativos,
        MAX(
            CASE
                WHEN ts_fim_conta_contrato IS NOT NULL
                 AND ts_fim_conta_contrato < data_ultimo_egv_alvo
                THEN DATE_DIFF(
                    DATE(ts_fim_conta_contrato),
                    DATE(ts_inicio_conta_contrato),
                    DAY
                )
            END
        ) AS nm_idade_dias_contrato_mais_demorado,
        MAX(
            CASE
                WHEN ts_inicio_relacionamento IS NOT NULL
                 AND ts_inicio_relacionamento < data_ultimo_egv_alvo
                THEN DATE_DIFF(
                    LEAST(
                        COALESCE(
                            DATE(ts_fim_relacionamento),
                            DATE(data_ultimo_egv_alvo)
                        ),
                        DATE(data_ultimo_egv_alvo)
                    ),
                    DATE(ts_inicio_relacionamento),
                    DAY
                )
            END
        ) AS tempo_relacionamento,
        CASE
            WHEN COUNTIF(
                cd_conta_contrato IS NOT NULL
                AND ts_fim_conta_contrato < data_ultimo_egv_alvo
            ) = 0
            AND COUNT(cd_conta_contrato) > 0
            THEN 1
            ELSE 0
        END AS fl_apenas_contratos_ativos,
        MAX(
            IF(
                cd_conta_contrato IS NOT NULL
                AND qt_dias_atraso != 0
                AND ts_inicio_conta_contrato < data_ultimo_egv_alvo,
                1,
                0
            )
        ) AS fl_cliente_atrasou_algum_contrato,
        MAX(fl_possui_cartao) AS fl_possui_cartao,
        MAX(fl_possui_conta) AS fl_possui_conta
    FROM contratos_anteriores_alvo
    GROUP BY cpf, proposta_ultimo_egv_alvo, data_ultimo_egv_alvo
),
medias_propostas_anteriores AS (
    SELECT
        cpf,
        proposta_ultimo_egv_alvo,
        data_ultimo_egv_alvo,
        COUNT(DISTINCT NuProposta) AS qtd_propostas_usadas_media,
        AVG(pcentrada) AS media_pcentrada,
        AVG(pcmargemliquida) AS media_pcmargemliquida,
        AVG(qtalteracao) AS media_qtalteracao,
        AVG(qtdiascarencia) AS media_qtdiascarencia,
        AVG(qtparcelas) AS media_qtparcelas,
        AVG(txcetcalculada) AS media_txcetcalculada,
        AVG(txfinanciamento) AS media_txfinanciamento,
        AVG(txpactuada) AS media_txpactuada,
        AVG(vrcomissao) AS media_vrcomissao,
        AVG(vrcustos) AS media_vrcustos,
        AVG(vrentrada) AS media_vrentrada,
        AVG(vrfinanciamento) AS media_vrfinanciamento,
        AVG(vrpresente) AS media_vrpresente,
        AVG(vrproposta) AS media_vrproposta,
        AVG(vrsegurobvmais) AS media_vrsegurobvmais,
        AVG(vrspread) AS media_vrspread,
        AVG(vrtac) AS media_vrtac
    FROM (
        SELECT DISTINCT
            u.cpf,
            u.proposta_ultimo_egv_alvo,
            u.data_ultimo_egv_alvo,
            h.NuProposta,
            h.pcentrada,
            h.pcmargemliquida,
            h.qtalteracao,
            h.qtdiascarencia,
            h.qtparcelas,
            h.txcetcalculada,
            h.txfinanciamento,
            h.txpactuada,
            h.vrcomissao,
            h.vrcustos,
            h.vrentrada,
            h.vrfinanciamento,
            h.vrpresente,
            h.vrproposta,
            h.vrsegurobvmais,
            h.vrspread,
            h.vrtac
        FROM ultimo_egv_alvo_por_cpf AS u
        INNER JOIN pessoa AS p
            ON u.cpf = p.cpf
        INNER JOIN proposta_base AS h
            ON p.NuProposta = h.NuProposta
        WHERE h.NuProposta <> u.proposta_ultimo_egv_alvo
          AND h.DtStatusProposta < u.data_ultimo_egv_alvo
          AND (
              h.DtEntradaProposta < u.data_ultimo_egv_alvo
              OR (
                  h.DtEntradaProposta = u.data_ultimo_egv_alvo
                  AND h.NuProposta < u.proposta_ultimo_egv_alvo
              )
          )
    )
    GROUP BY cpf, proposta_ultimo_egv_alvo, data_ultimo_egv_alvo
),
final_base_alvo AS (
    SELECT
        u.cpf,
        u.data_ultimo_egv_alvo,
        CAST(elig.nuproposta AS STRING) AS proposta_ultimo_egv_alvo,
        CAST(NULL AS INT64) AS target,
        -- Features de pessoa vindas da proposta imediatamente anterior
        IFNULL(pa.cdregraunicaperdaesperada, 0) AS cdregraunicaperdaesperada,
        IFNULL(pa.__CAMPO_INFORMACAO__, 0) AS __CAMPO_INFORMACAO__,
        IFNULL(pa.cdtipinformacaospcserasa, 0) AS cdtipinformacaospcserasa,
        IFNULL(pa.cdtipoconfirmacaolr, 0) AS cdtipoconfirmacaolr,
        IFNULL(pa.cdvigenciaregraunicaprdaespra, 0)
            AS cdvigenciaregraunicaprdaespra,
        IFNULL(pa.flalertapessoa, 0) AS flalertapessoa,
        IFNULL(pa.flclienteativo, 0) AS flclienteativo,
        IFNULL(pa.flconsultabv, 0) AS flconsultabv,
        IFNULL(pa.flexcliente, 0) AS flexcliente,
        IFNULL(pa.flrisco, 0) AS flrisco,
        IFNULL(pa.flscoredefinidopolitica, 0) AS flscoredefinidopolitica,
        IFNULL(pa.flserasa, 0) AS flserasa,
        IFNULL(pa.flspc, 0) AS flspc,
        IFNULL(pa.nuscorefim, 0) AS nuscorefim,
        IFNULL(pa.pcpartsocio, 0) AS pcpartsocio,
        IFNULL(pa.fltemsocio, 0) AS fltemsocio,
        IFNULL(pa.qtbehavior, 0) AS qtbehavior,
        IFNULL(pa.qtscore, 0) AS qtscore,
        IFNULL(pa.fl_sem_pessoa_anterior, 1) AS fl_sem_pessoa_anterior,
        pa.proposta_pessoa_anterior,
        pa.data_pessoa_anterior,
        -- Features da proposta alvo: comentadas no original.
        -- u.ultimo_cdgruposcore,
        -- u.ultimo_cdtabelafinanciamento,
        -- IFNULL(u.ultimo_cdfilial, 0) AS ultimo_cdfilial,
        -- IF(u.ultimo_cdfilial IS NULL, 1, 0) AS fl_sem_cdfilial,
        -- IFNULL(u.ultimo_cdveiclegalmesacred, 0)
        --     AS ultimo_cdveiclegalmesacred,
        IFNULL(mp.qtd_propostas_usadas_media, 0) AS qtd_propostas_usadas_media,
        IFNULL(mp.media_pcentrada, 0) AS media_pcentrada,
        IFNULL(mp.media_pcmargemliquida, 0) AS media_pcmargemliquida,
        IFNULL(mp.media_qtalteracao, 0) AS media_qtalteracao,
        IFNULL(mp.media_qtdiascarencia, 0) AS media_qtdiascarencia,
        IFNULL(mp.media_qtparcelas, 0) AS media_qtparcelas,
        IFNULL(mp.media_txcetcalculada, 0) AS media_txcetcalculada,
        IFNULL(mp.media_txfinanciamento, 0) AS media_txfinanciamento,
        IFNULL(mp.media_txpactuada, 0) AS media_txpactuada,
        IFNULL(mp.media_vrcomissao, 0) AS media_vrcomissao,
        IFNULL(mp.media_vrcustos, 0) AS media_vrcustos,
        IFNULL(mp.media_vrentrada, 0) AS media_vrentrada,
        IFNULL(mp.media_vrfinanciamento, 0) AS media_vrfinanciamento,
        IFNULL(mp.media_vrpresente, 0) AS media_vrpresente,
        IFNULL(mp.media_vrproposta, 0) AS media_vrproposta,
        IFNULL(mp.media_vrsegurobvmais, 0) AS media_vrsegurobvmais,
        IFNULL(mp.media_vrspread, 0) AS media_vrspread,
        IFNULL(mp.media_vrtac, 0) AS media_vrtac,
        IF(mp.qtd_propostas_usadas_media IS NULL, 1, 0)
            AS fl_sem_propostas_anteriores_para_media,
        -- Historico de propostas antes do alvo
        IFNULL(h.qtd_propostas_antes_ultimo_egv_alvo, 0)
            AS qtd_propostas_antes_ultimo_egv_alvo,
        IFNULL(h.qtd_lojas_antes_ultimo_egv_alvo, 0)
            AS qtd_lojas_antes_ultimo_egv_alvo,
        IFNULL(h.qtd_status_a_antes_ultimo_egv_alvo, 0)
            AS qtd_status_a_antes_ultimo_egv_alvo,
        IFNULL(h.qtd_status_ap_antes_ultimo_egv_alvo, 0)
            AS qtd_status_ap_antes_ultimo_egv_alvo,
        IFNULL(h.qtd_status_ca_antes_ultimo_egv_alvo, 0)
            AS qtd_status_ca_antes_ultimo_egv_alvo,
        IFNULL(h.qtd_status_outros_antes_ultimo_egv_alvo, 0)
            AS qtd_status_outros_antes_ultimo_egv_alvo,
        IFNULL(h.qtd_eventos_egv_ap_antes_ultimo_egv_alvo, 0)
            AS qtd_eventos_egv_ap_antes_ultimo_egv_alvo,
        IFNULL(h.qtd_eventos_egv_ca_antes_ultimo_egv_alvo, 0)
            AS qtd_eventos_egv_ca_antes_ultimo_egv_alvo,
        IFNULL(h.qtd_eventos_egv_antes_ultimo_egv_alvo, 0)
            AS qtd_eventos_egv_antes_ultimo_egv_alvo,
        IFNULL(h.qtd_eventos_outros_produtos_antes_ultimo_egv_alvo, 0)
            AS qtd_eventos_outros_produtos_antes_ultimo_egv_alvo,
        IFNULL(h.qtd_subprodutos_distintos_antes_ultimo_egv_alvo, 0)
            AS qtd_subprodutos_distintos_antes_ultimo_egv_alvo,
        IFNULL(h.dias_desde_evento_anterior, 0) AS dias_desde_evento_anterior,
        IFNULL(h.janela_historica_dias, 0) AS janela_historica_dias,
        IFNULL(h.fl_sem_historico_antes_ultimo_egv_alvo, 1)
            AS fl_sem_historico_antes_ultimo_egv_alvo,
        IFNULL(h.fl_entrada_egv_alvo, 1) AS fl_entrada_egv_alvo,
        -- Historico de relacionamento antes do alvo
        IFNULL(r.nm_total_outros_contratos_banco, 0)
            AS nm_total_outros_contratos_banco,
        IFNULL(r.nm_total_contratos_ativos, 0) AS nm_total_contratos_ativos,
        IFNULL(r.nm_total_contratos_inativos, 0) AS nm_total_contratos_inativos,
        IFNULL(r.nm_idade_dias_contrato_mais_demorado, 0)
            AS nm_idade_dias_contrato_mais_demorado,
        IFNULL(r.tempo_relacionamento, 0) AS tempo_relacionamento,
        IF(r.tempo_relacionamento IS NULL, 1, 0)
            AS fl_contratos_emitidos_no_mesmo_dia_egv,
        IFNULL(r.fl_apenas_contratos_ativos, 0) AS fl_apenas_contratos_ativos,
        IFNULL(r.fl_cliente_atrasou_algum_contrato, 0)
            AS fl_cliente_atrasou_algum_contrato,
        IFNULL(r.fl_possui_cartao, 0) AS fl_possui_cartao,
        IFNULL(r.fl_possui_conta, 0) AS fl_possui_conta
    FROM elegiveis AS elig
    LEFT JOIN ultimo_egv_alvo_por_cpf AS u
        ON elig.nuproposta = u.proposta_ultimo_egv_alvo
    LEFT JOIN pessoa_anterior_egv_alvo AS pa
        ON u.cpf = pa.cpf
       AND u.proposta_ultimo_egv_alvo = pa.proposta_ultimo_egv_alvo
       AND u.data_ultimo_egv_alvo = pa.data_ultimo_egv_alvo
    LEFT JOIN historico_agg AS h
        ON u.cpf = h.cpf
       AND u.proposta_ultimo_egv_alvo = h.proposta_ultimo_egv_alvo
       AND u.data_ultimo_egv_alvo = h.data_ultimo_egv_alvo
    LEFT JOIN relacionamento_agg AS r
        ON u.cpf = r.cpf
       AND u.proposta_ultimo_egv_alvo = r.proposta_ultimo_egv_alvo
       AND u.data_ultimo_egv_alvo = r.data_ultimo_egv_alvo
    LEFT JOIN medias_propostas_anteriores AS mp
        ON u.cpf = mp.cpf
       AND u.proposta_ultimo_egv_alvo = mp.proposta_ultimo_egv_alvo
       AND u.data_ultimo_egv_alvo = mp.data_ultimo_egv_alvo
)
SELECT f.*
FROM final_base_alvo AS f
WHERE NOT EXISTS (
    SELECT 1
    FROM cpfs_concorrencia AS c
    WHERE c.cpf = f.cpf
)
"""


# COMMAND ----------
# Dicionario das features


FEATURE_DESCRIPTIONS = {
    'cdregraunicaperdaesperada': (
        'Valor de cdregraunicaperdaesperada da pessoa na proposta anterior '
        'selecionada pelas regras temporais; ausente preenchido com zero.'
    ),
    'cdtipinformacaoltlr': (
        'Valor de cdtipinformacaoltlr da pessoa na proposta anterior '
        'selecionada pelas regras temporais; ausente preenchido com zero.'
    ),
    'cdtipinformacaospcserasa': (
        'Valor de cdtipinformacaospcserasa da pessoa na proposta anterior '
        'selecionada pelas regras temporais; ausente preenchido com zero.'
    ),
    'cdtipoconfirmacaolr': (
        'Valor de cdtipoconfirmacaolr da pessoa na proposta anterior '
        'selecionada pelas regras temporais; ausente preenchido com zero.'
    ),
    'cdvigenciaregraunicaprdaespra': (
        'Valor de cdvigenciaregraunicaprdaespra da pessoa na proposta '
        'anterior selecionada pelas regras temporais; ausente preenchido com '
        'zero.'
    ),
    'flalertapessoa': (
        'Indicador derivado de flalertapessoa da proposta anterior: 1 quando '
        'o campo original normalizado vale S; 0 nos demais casos e na '
        'ausencia.'
    ),
    'flclienteativo': (
        'Indicador derivado de flclienteativo da proposta anterior: 1 quando '
        'o campo original normalizado vale S; 0 nos demais casos e na '
        'ausencia.'
    ),
    'flconsultabv': (
        'Indicador derivado de flconsultabv da proposta anterior: 1 quando o '
        'campo original normalizado vale S; 0 nos demais casos e na ausencia.'
    ),
    'flexcliente': (
        'Indicador derivado de flexcliente da proposta anterior: 1 quando o '
        'campo original normalizado vale S; 0 nos demais casos e na ausencia.'
    ),
    'flrisco': (
        'Indicador derivado de flrisco da proposta anterior: 1 quando o campo '
        'original normalizado vale S; 0 nos demais casos e na ausencia.'
    ),
    'flscoredefinidopolitica': (
        'Indicador derivado de flscoredefinidopolitica da proposta anterior: '
        '1 quando o campo original normalizado vale S; 0 nos demais casos e '
        'na ausencia.'
    ),
    'flserasa': (
        'Indicador derivado de flserasa da proposta anterior: 1 quando o '
        'campo original normalizado vale S; 0 nos demais casos e na ausencia.'
    ),
    'flspc': (
        'Indicador derivado de flspc da proposta anterior: 1 quando o campo '
        'original normalizado vale S; 0 nos demais casos e na ausencia.'
    ),
    'nuscorefim': (
        'Valor de nuscorefim da pessoa na proposta anterior selecionada pelas '
        'regras temporais; ausente preenchido com zero.'
    ),
    'pcpartsocio': (
        'Valor de pcpartsocio da pessoa na proposta anterior selecionada '
        'pelas regras temporais; ausente preenchido com zero.'
    ),
    'fltemsocio': (
        'Indicador de pcpartsocio preenchido na proposta anterior; 0 quando '
        'nao ha pessoa anterior.'
    ),
    'qtbehavior': (
        'Valor de qtbehavior da pessoa na proposta anterior selecionada pelas '
        'regras temporais; ausente preenchido com zero.'
    ),
    'qtscore': (
        'Valor de qtscore da pessoa na proposta anterior selecionada pelas '
        'regras temporais; ausente preenchido com zero.'
    ),
    'fl_sem_pessoa_anterior': (
        '1 quando nao foi encontrada pessoa em proposta anterior elegivel; 0 '
        'quando encontrada.'
    ),
    'qtd_propostas_usadas_media': (
        'Numero de propostas anteriores distintas usadas nas medias '
        'financeiras, apos filtros temporais e exclusao da proposta alvo; '
        'ausencia preenchida com zero.'
    ),
    'media_pcentrada': (
        'Media de pcentrada nas propostas anteriores elegiveis; ausencia '
        'preenchida com zero.'
    ),
    'media_pcmargemliquida': (
        'Media de pcmargemliquida nas propostas anteriores elegiveis; '
        'ausencia preenchida com zero.'
    ),
    'media_qtalteracao': (
        'Media de qtalteracao nas propostas anteriores elegiveis; ausencia '
        'preenchida com zero.'
    ),
    'media_qtdiascarencia': (
        'Media de qtdiascarencia nas propostas anteriores elegiveis; ausencia '
        'preenchida com zero.'
    ),
    'media_qtparcelas': (
        'Media de qtparcelas nas propostas anteriores elegiveis; ausencia '
        'preenchida com zero.'
    ),
    'media_txcetcalculada': (
        'Media de txcetcalculada nas propostas anteriores elegiveis; ausencia '
        'preenchida com zero.'
    ),
    'media_txfinanciamento': (
        'Media de txfinanciamento nas propostas anteriores elegiveis; '
        'ausencia preenchida com zero.'
    ),
    'media_txpactuada': (
        'Media de txpactuada nas propostas anteriores elegiveis; ausencia '
        'preenchida com zero.'
    ),
    'media_vrcomissao': (
        'Media de vrcomissao nas propostas anteriores elegiveis; ausencia '
        'preenchida com zero.'
    ),
    'media_vrcustos': (
        'Media de vrcustos nas propostas anteriores elegiveis; ausencia '
        'preenchida com zero.'
    ),
    'media_vrentrada': (
        'Media de vrentrada nas propostas anteriores elegiveis; ausencia '
        'preenchida com zero.'
    ),
    'media_vrfinanciamento': (
        'Media de vrfinanciamento nas propostas anteriores elegiveis; '
        'ausencia preenchida com zero.'
    ),
    'media_vrpresente': (
        'Media de vrpresente nas propostas anteriores elegiveis; ausencia '
        'preenchida com zero.'
    ),
    'media_vrproposta': (
        'Media de vrproposta nas propostas anteriores elegiveis; ausencia '
        'preenchida com zero.'
    ),
    'media_vrsegurobvmais': (
        'Media de vrsegurobvmais nas propostas anteriores elegiveis; ausencia '
        'preenchida com zero.'
    ),
    'media_vrspread': (
        'Media de vrspread nas propostas anteriores elegiveis; ausencia '
        'preenchida com zero.'
    ),
    'media_vrtac': (
        'Media de vrtac nas propostas anteriores elegiveis; ausencia '
        'preenchida com zero.'
    ),
    'fl_sem_propostas_anteriores_para_media': (
        '1 quando nao ha linha agregada de propostas anteriores para calcular '
        'medias; 0 quando ha.'
    ),
    'qtd_propostas_antes_ultimo_egv_alvo': (
        'Quantidade de numeros de proposta distintos com entrada anterior a '
        'entrada da proposta alvo.'
    ),
    'qtd_lojas_antes_ultimo_egv_alvo': (
        'Quantidade de lojas distintas validas em propostas com entrada '
        'anterior a entrada da proposta alvo; ignora vazio e zero.'
    ),
    'qtd_status_a_antes_ultimo_egv_alvo': (
        'Contagem de registros historicos em status A com entrada e data de '
        'status anteriores a entrada da proposta alvo.'
    ),
    'qtd_status_ap_antes_ultimo_egv_alvo': (
        'Contagem de registros historicos em status AP com entrada e data de '
        'status anteriores a entrada da proposta alvo.'
    ),
    'qtd_status_ca_antes_ultimo_egv_alvo': (
        'Contagem de registros historicos em status CA com entrada e data de '
        'status anteriores a entrada da proposta alvo.'
    ),
    'qtd_status_outros_antes_ultimo_egv_alvo': (
        'Contagem de registros historicos fora dos status A, AP e CA com '
        'entrada e data de status anteriores a entrada da proposta alvo.'
    ),
    'qtd_eventos_egv_ap_antes_ultimo_egv_alvo': (
        'Contagem de registros dos subprodutos EGV configurados em status AP '
        'com entrada e data de status anteriores a entrada da proposta alvo.'
    ),
    'qtd_eventos_egv_ca_antes_ultimo_egv_alvo': (
        'Contagem de registros dos subprodutos EGV configurados em status CA '
        'com entrada e data de status anteriores a entrada da proposta alvo.'
    ),
    'qtd_eventos_egv_antes_ultimo_egv_alvo': (
        'Contagem de registros dos subprodutos EGV configurados com entrada '
        'anterior a entrada da proposta alvo, independentemente de status.'
    ),
    'qtd_eventos_outros_produtos_antes_ultimo_egv_alvo': (
        'Contagem de registros de outros subprodutos com entrada anterior a '
        'entrada da proposta alvo, independentemente de status.'
    ),
    'qtd_subprodutos_distintos_antes_ultimo_egv_alvo': (
        'Quantidade de subprodutos distintos em propostas com entrada '
        'anterior a entrada da proposta alvo.'
    ),
    'dias_desde_evento_anterior': (
        'Dias entre a entrada da proposta alvo e a entrada mais recente de '
        'proposta anterior; ausencia preenchida com zero.'
    ),
    'janela_historica_dias': (
        'Dias entre a entrada da proposta alvo e a entrada mais antiga no '
        'historico disponivel anterior; ausencia preenchida com zero.'
    ),
    'fl_sem_historico_antes_ultimo_egv_alvo': (
        '1 quando nao ha registro de proposta com entrada anterior a proposta '
        'alvo; 0 caso contrario.'
    ),
    'fl_entrada_egv_alvo': (
        'Indicador definido no SQL como 1 quando nao ha registro de proposta '
        'com entrada anterior a proposta alvo; 0 caso contrario.'
    ),
    'nm_total_outros_contratos_banco': (
        'Contagem de identificadores de contratos nao nulos, iniciados antes '
        'da proposta alvo, excluindo modalidades EGV configuradas.'
    ),
    'nm_total_contratos_ativos': (
        'Contagem de contratos iniciados antes da proposta alvo cujo fim esta '
        'ausente ou ocorre na entrada da proposta alvo ou depois dela.'
    ),
    'nm_total_contratos_inativos': (
        'Contagem de contratos com fim anterior a entrada da proposta alvo.'
    ),
    'nm_idade_dias_contrato_mais_demorado': (
        'Maior duracao em dias entre inicio e fim dos contratos encerrados '
        'antes da entrada da proposta alvo; ausencia preenchida com zero.'
    ),
    'tempo_relacionamento': (
        'Maior duracao em dias desde o inicio do relacionamento ate o menor '
        'valor entre fim do relacionamento e entrada da proposta alvo; '
        'ausencia preenchida com zero.'
    ),
    'fl_contratos_emitidos_no_mesmo_dia_egv': (
        'Indicador definido no SQL como 1 quando tempo_relacionamento '
        'agregado esta nulo antes do preenchimento; 0 caso contrario. O nome '
        'nao descreve uma comparacao explicita de datas iguais.'
    ),
    'fl_apenas_contratos_ativos': (
        '1 quando ha pelo menos um contrato considerado e nenhum contrato com '
        'fim anterior a entrada da proposta alvo; 0 caso contrario.'
    ),
    'fl_cliente_atrasou_algum_contrato': (
        '1 quando algum contrato considerado tem qt_dias_atraso diferente de '
        'zero; 0 caso contrario.'
    ),
    'fl_possui_cartao': (
        '1 quando algum contrato considerado tem nm_tipo_produto contendo '
        'cart apos conversao para minusculas; 0 caso contrario.'
    ),
    'fl_possui_conta': (
        '1 quando algum contrato considerado tem nm_tipo_produto contendo '
        'conta apos conversao para minusculas; 0 caso contrario.'
    ),
}


# COMMAND ----------
# Dicionario da saida


EXTRA_DESCRIPTIONS = {
    "idmodelo": "Identificador GRC do modelo informado no contrato.",
    "idchavemodelo": (
        "Numero da proposta como chave da aplicacao. Na referencia de treino, "
        "pode ser chave tecnica referencia_treino quando a proposta nao foi "
        "exportada; isso nao representa numero de proposta historico."
    ),
    "cpf": "CPF do titular pessoa fisica, com 11 digitos.",
    "proposta_ultimo_egv_alvo": (
        "Numero da proposta escorada; nome legado, agora uma linha por "
        "proposta elegivel. Pode estar nulo somente na referencia de treino "
        "quando o identificador foi excluido da exportacao original."
    ),
    "safra": (
        "Segunda-feira de fechamento da janela semanal, em YYYYMMDD. "
        "Na referencia de treino identifica sua preparacao, nao uma campanha."
    ),
    "prob1": (
        "Probabilidade da classe positiva AP calculada pelo modelo aprovado, "
        "com calibrador quando salvo no bundle."
    ),
    "predito": "1 quando prob1 >= 0.5; 0 nos demais casos.",
    "target": (
        "Resultado observado AP por CPF apos maturacao minima de 30 dias; "
        "nulo antes da apuracao. No treino, usa o target real original."
    ),
    "gh": (
        "Grupo pelos cortes fixos do bundle; GH1 menor score. Igualdade com "
        "corte permanece no grupo inferior, conforme o treinamento."
    ),
    "grupocontrole": "String 0 para teste e 1 para controle; nulo no treino.",
    "TesteDigito": "Inteiro 0 para teste e 1 para controle; nulo no treino.",
    "TipoControle": (
        "teste_proporcional, controle_proporcional ou controle_score_baixo; "
        "a atribuicao vale para o CPF inteiro na safra. Nulo no treino."
    ),
    "rank_pct": (
        "Rank percentual decrescente de prob1 na populacao de propostas, "
        "com rank medio nos empates."
    ),
    "faixa": "Faixa comercial da proposta obtida do rank percentual.",
    "faixa_controle": (
        "Faixa da proposta de menor score do CPF, usada como estrato do "
        "sorteio. Nula na referencia de treino."
    ),
}


# COMMAND ----------
# Contrato Kraken


def montar_contrato(bundle: dict) -> dict:
    """Atualiza o contrato existente para a saída semanal por proposta.

    A identidade do componente, a referência da run do MLflow, os caminhos,
    as dependências e os demais metadados permanecem os do CONTRACT informado.
    """
    contrato = deepcopy(CONTRACT)
    # Defaults observados no dc.create original; identidades ja carregadas
    # prevalecem. Nenhum run_id foi copiado manualmente de fotografia.
    contrato.setdefault("author", ["Renato Netto (renato.netto@bv.com.br)"])
    contrato.setdefault("model_name", "CRMX_PROPENSAO_EGV_APNP_V1")
    contrato.setdefault("used_tables", [])
    contrato.setdefault("path_to_model", [])
    contrato.setdefault("cluster_info", {
        "spark_version": "16.4.x-cpu-ml-scala2.12",
        "node_type_id": "n2d-highmem-8",
        "driver_node_type_id": "n2d-highmem-8",
    })
    contrato.setdefault("dvry_bigquery_dataset", "dvry_anly_cienciadado")
    contrato.setdefault("version", [
        "kraken-bv==0.38.4",
        "kraken-drift-detectors==0.22.0",
        "kraken-monap==0.7.0",
        "pylib-data-base-utils-mlops==0.10.5",
        "model-version==0.1.0",
    ])
    contrato.setdefault("governance", {
        "business_unit": "ciencia_dados",
        "classification": "interno",
        "policy_tag_columns": ["cpf"],
        "selo": "bronze",
    })
    contrato.setdefault("big_dvry", {}).setdefault(
        "nome_dvry", "tbpropensaoegvapnp"
    )
    if SIGLA_COMPONENTE is not None:
        contrato["sigla"] = SIGLA_COMPONENTE
    if not contrato.get("model_grc") or not contrato.get("sigla"):
        raise ValueError(
            "Preserve model_grc e sigla oficiais do componente. "
            "Se o DEPLOY_CONTRACT for minimo, preencha SIGLA_COMPONENTE."
        )
    mapa_descricoes = dict(FEATURE_DESCRIPTIONS)
    mapa_descricoes["cdtipinformacaoitlr"] = mapa_descricoes[
        "cdtipinformacaoltlr"
    ].replace("cdtipinformacaoltlr", "cdtipinformacaoitlr")

    caminho_treino = contrato.get("path_to_data_train")
    if not isinstance(caminho_treino, str) or not caminho_treino.strip():
        raise ValueError(
            "CONTRACT['path_to_data_train'] deve apontar para a referência "
            "de treinamento já preparada e alinhada ao schema da BIG."
        )

    configuracoes_mlflow = contrato.get("mlflow")
    if not isinstance(configuracoes_mlflow, list) or not configuracoes_mlflow:
        raise ValueError("CONTRACT deve conter a configuração do MLflow.")
    if not isinstance(configuracoes_mlflow[0], dict):
        raise ValueError("CONTRACT['mlflow'][0] deve ser um dicionário.")
    if not configuracoes_mlflow[0].get("run_id"):
        raise ValueError("Defina a run do modelo em CONTRACT['mlflow'][0].")

    features = list(bundle["feature_cols"])
    sem_descricao = [
        coluna for coluna in features if coluna not in mapa_descricoes
    ]
    if sem_descricao:
        raise ValueError(
            "Faltam descrições das features no contrato: "
            + ", ".join(sem_descricao)
        )

    # A expressão observada no Kraken usa sete campos Quartz. O job deve
    # configurar America/Sao_Paulo como timezone; a API interna fornecida
    # não demonstra um campo de timezone no dicionário do contrato.
    contrato["build_schedule"] = "0 0 5 ? * MON *"

    contrato.setdefault("monap", {}).update(
        {"tpmodelo": 1, "threshold": 0.5}
    )
    contrato.setdefault("concept_drift", {}).update(
        {"predict_var_name": ["prob1"]}
    )
    contrato.setdefault("big_dvry", {}).update(
        {
            "keys": ["idchavemodelo", "safra"],
            "campos_big": "*",
            "campos_dvry": "*",
            "source": "apply_model",
            "target": "apply_target",
            "safra": "get_safra",
        }
    )

    descricoes = {
        coluna: mapa_descricoes[coluna] for coluna in features
    }
    descricoes.update(EXTRA_DESCRIPTIONS)
    contrato.setdefault("model_output_description", {}).setdefault(
        "table_description",
        "Propensao EGV por proposta e safra semanal, com grupo de teste/"
        "controle consistente por CPF, exclusao de CPFs sinalizados com "
        "concorrencia e apuracao posterior do target AP.",
    )
    contrato.setdefault("model_output_description", {})[
        "fields_description"
    ] = descricoes

    # O contrato original usa nomes de funções em listas. Mantemos esse
    # formato com as funções implementadas nesta versão. A aceitação das
    # assinaturas pelo validador será verificada no ambiente do Kraken.
    contrato["dataprep"] = [
        "calcular_janela",
        "preparar_features",
        "calcular_gh",
        "adicionar_faixas",
        "atribuir_grupos",
        "dataprep_read_table_bigquery",
        "carregar_bundle",
        "get_base_egv_apnp",
    ]
    contrato["test_dataprep"] = [
        "test_janela_semanal",
        "test_grupos_por_cpf",
        "test_cortes_gh",
    ]

    # used_tables, author, model_grc, sigla, mlflow, version, cluster_info,
    # build_libraries e demais campos não são reconstruídos por suposição.
    return contrato


def validar_contrato():
    """Executa explicitamente a validação do contrato no ambiente Kraken.

    Esta função pode executar as etapas de integração do framework. Ela
    somente será chamada pelo responsável após configurar o CONTRACT.
    """
    global CONTRACT

    from kraken.contract import DeployContract

    bundle = carregar_bundle()
    CONTRACT = montar_contrato(bundle)
    dc = DeployContract(globals())
    return dc.create(CONTRACT)


# COMMAND ----------
# MAGIC %md
# MAGIC ## Preparacao unica e primeira validacao
# MAGIC 1. Mantenha os metadados oficiais no DEPLOY_CONTRACT. Se ele estiver
# MAGIC    reduzido aos quatro campos das fotos, preencha SIGLA_COMPONENTE.
# MAGIC    Confira model_grc: as fotos contêm test202603 e mod001920.
# MAGIC    Esta versao usa o valor do contrato carregado.
# MAGIC 2. Use SOMENTE o Parquet do treino aprovado. Nunca incluir OOT.
# MAGIC 3. Chame preparar_artefatos_deploy com um diretorio de destino NOVO.
# MAGIC    A funcao registra o pipeline com assinatura, salva o bundle inteiro,
# MAGIC    alinha o schema de referencia e atualiza CONTRACT nesta sessao.
# MAGIC 4. Execute os testes, apply_model e a validacao do contrato.
# MAGIC 5. Confirme no job: segunda-feira 05h, timezone America/Sao_Paulo.
# MAGIC    build_schedule usa 0 0 5 ? * MON *. Se o job usar UTC, o horario
# MAGIC    equivalente e 08h; o fuso do job nao e inferido pela expressao cron.
# MAGIC
# MAGIC A SQL de features foi transcrita das imagens e precisa compilar contra
# MAGIC os schemas reais. A selecao usa status/data da tboriginacao, como o STG
# MAGIC anterior. DATETIME usa FUSO_DATETIME_ORIGINACAO;
# MAGIC DATE sem hora e rejeitada.
# MAGIC Concorrencia: tboriginacao.FlConcorrente = 1 exclui TODAS as propostas
# MAGIC do CPF, antes do scoring, das faixas e do sorteio. A consulta considera
# MAGIC CVG Leves/Pesados, qualquer status e entrada desde a segunda anterior
# MAGIC 05h, sem limite superior, preservando a consulta corrente do exemplo.
# MAGIC Esse recorte e pela entrada da proposta; nao e um bloqueio vitalicio
# MAGIC nem usa a data em que ocorreu o fechamento na concorrencia.
# MAGIC Reprocessar fixa janela e sorteio, mas nao congela fontes mutaveis.
# MAGIC O historico para apply_target deve vir do executor/BIG_HISTORICA.

# COMMAND ----------
# Exemplo de preparacao MANUAL (descomente e informe um destino novo):
# df_treino = spark.read.parquet(CONTRACT["path_to_data_train"])
# registro = preparar_artefatos_deploy(
#     caminho_bundle=(
#         "/Workspace/Users/renato.netto@bv.com.br/first-test/EGV/"
#         "egv_model_bundle_latest.joblib"
#     ),
#     df_treino=df_treino,
#     caminho_referencia_treino=(
#         "/Volumes/kraken_dasc/users/train_data/"
#         "propensao_egv_REFERENCIA_NOVA"
#     ),
# )
# print(json.dumps(registro, indent=2, ensure_ascii=False))

# COMMAND ----------
# Testes locais que nao acessam fontes externas:
# test_janela_semanal()
# test_cortes_gh()
# test_grupos_por_cpf()

# COMMAND ----------
# Primeira leitura e scoring no ambiente, antes do create:
# from pyspark.sql import functions as F
# df_big = apply_model()
# display(df_big)
# df_big.groupBy("faixa", "TipoControle").count().show()
# df_big.groupBy("cpf", "safra").agg(
#     F.countDistinct("TesteDigito").alias("n_grupos")
# ).filter("n_grupos > 1").show()

# COMMAND ----------
# Validacao Kraken. A chamada nao e executada automaticamente ao importar.
# resultado_validacao = validar_contrato()
# Depois do create bem-sucedido, seguir o fluxo interno de commit/promocao.
