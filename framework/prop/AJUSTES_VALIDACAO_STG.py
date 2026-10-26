# Databricks notebook source
# MAGIC %md
# MAGIC # Blocos para substituir NO STG existente
# MAGIC Nao executar este notebook por %run: o validador le o codigo do STG.
# MAGIC Copie as celulas para substituir as definicoes correspondentes no STG.
# MAGIC Preserve sua SQL e o DEPLOY_CONTRACT que ja funcionaram.
# MAGIC Remova preparar_artefatos_deploy e suas chamadas do STG; guarde-os
# MAGIC no notebook separado PREPARAR_ARTEFATOS. Remova os tres testes
# MAGIC antigos test_janela_semanal/test_cortes_gh/test_grupos_por_cpf.
# MAGIC Remova as celulas de instalacao/reinicio, mantendo as bibliotecas
# MAGIC declaradas no contrato. Nao cole funcoes duplicadas no STG.

# COMMAND ----------


FUNCOES_DATAPREP = (
    "resolver_safra",
    "calcular_janela",
    "preparar_features",
    "calcular_gh",
    "adicionar_faixas",
    "atribuir_grupos",
    "validar_bundle",
    "_carregar_bundle_run",
    "carregar_bundle",
    "_spark",
    "dataprep_read_table_bigquery",
    "montar_query_features",
    "get_base_egv_apnp",
    "_normalizar_chaves",
    "escorar_dataframe",
    "schema_big",
    "para_spark_big",
    "montar_contrato",
    "validar_contrato",
)


# COMMAND ----------
@lru_cache(maxsize=2)
def _carregar_bundle_run(run_id: str) -> dict:
    """Baixa e valida o bundle da run, reutilizando o cache local.

    Args:
        run_id: Identificador da run que contem BUNDLE_ARTIFACT.

    Returns:
        Bundle completo com pipeline, features, calibrador e cortes GH.
    """
    from mlflow.artifacts import download_artifacts

    with TemporaryDirectory(prefix="propensao_") as pasta:
        arquivo = download_artifacts(
            run_id=run_id, artifact_path=BUNDLE_ARTIFACT, dst_path=pasta
        )
        bundle = joblib.load(arquivo)
    validar_bundle(bundle)
    return bundle


# COMMAND ----------
def _spark():
    """Retorna a sessao Spark do notebook ou a sessao ativa.

    Raises:
        RuntimeError: Quando nenhuma sessao Spark esta disponivel.
    """
    from pyspark.sql import SparkSession

    session = globals().get("spark") or SparkSession.getActiveSession()
    if session is None:
        raise RuntimeError("Execute esta etapa no Databricks com Spark ativo.")
    return session


# COMMAND ----------
def _normalizar_chaves(df: pd.DataFrame) -> pd.DataFrame:
    """Valida CPF e proposta, preservando identificadores como strings.

    Rejeita identificadores ausentes, CPF sem 11 digitos e proposta
    duplicada. Retorna uma copia sem alterar o DataFrame recebido.
    """
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


# COMMAND ----------
def para_spark_big(df: pd.DataFrame, feature_cols: list[str]):
    """Converte a saida Pandas para Spark com o schema explicito da BIG.

    Preserva strings identificadoras e converte ausentes para None.
    Nao grava arquivos nem tabelas: a persistencia pertence ao Kraken.
    """
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


# COMMAND ----------
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
    if not contrato.get("used_tables"):
        contrato["used_tables"] = [
            "bigquery:bv-pdbd-prd.trusted_dbcred.tbproposta",
            "bigquery:bv-pdbd-prd.trusted_dbcred.tbpessoasobanalise",
            "bigquery:bv-pdbd-prd.trusted_nego_rodas.tboriginacao",
            "bigquery:bv-pdbd-prd.dvry_clie_crm.clientebv_relacionamento_dime",
        ]
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
    bibliotecas = list(contrato.get("build_libraries") or [])
    nomes_bibliotecas = {
        item.split("==", 1)[0].strip().lower().replace("_", "-")
        for item in bibliotecas
    }
    if "kraken-bv" not in nomes_bibliotecas:
        versao_kraken = next(
            (
                item for item in contrato["version"]
                if item.startswith("kraken-bv==")
            ),
            None,
        )
        if versao_kraken is None:
            raise ValueError("Declare a versao do kraken-bv no contrato.")
        bibliotecas.append(versao_kraken)
    contrato["build_libraries"] = bibliotecas

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
    contrato.setdefault("build_schedule", "0 0 5 ? * MON *")

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

    # Correspondencia explicita de uma funcao de dataprep para um teste.
    # Os hooks get_safra/apply_model/apply_target ficam em big_dvry.
    contrato["dataprep"] = list(FUNCOES_DATAPREP)
    contrato["test_dataprep"] = [
        f"test_{nome}" for nome in FUNCOES_DATAPREP
    ]
    for nome in (*contrato["dataprep"], *contrato["test_dataprep"]):
        funcao = globals().get(nome)
        if not callable(funcao):
            raise ValueError(f"Funcao do contrato nao definida no STG: {nome}")
        if not getattr(funcao, "__doc__", None):
            raise ValueError(f"Funcao do contrato sem docstring: {nome}")

    # Preserva identidades, run, referencia de treino e configuracoes
    # ja preenchidas. Nao registra um modelo novo durante esta validacao.
    return contrato


# COMMAND ----------
# Testes unitarios: uma funcao test_<nome> por funcao de dataprep.
# Colar no STG no lugar dos tres testes antigos, nao em paralelo a eles.
# Dependencias externas sao simuladas; nenhum teste registra um novo modelo.


def test_resolver_safra() -> None:
    """Verifica a segunda de referencia, o fuso e o corte das 05h."""
    from unittest import TestCase

    assert resolver_safra(
        datetime(2026, 10, 26, 4, 59, tzinfo=FUSO)
    ) == "20261019"
    assert resolver_safra(
        datetime(2026, 10, 26, 5, 0, tzinfo=FUSO)
    ) == "20261026"
    assert resolver_safra(
        datetime(2026, 10, 26, 8, 0, tzinfo=ZoneInfo("UTC"))
    ) == "20261026"
    with TestCase().assertRaises(ValueError):
        resolver_safra(datetime(2026, 10, 26, 5, 0))


def test_calcular_janela() -> None:
    """Valida continuidade semanal e rejeita safras fora de segunda-feira."""
    from unittest import TestCase

    inicio, fim = calcular_janela("20261026")
    assert inicio.isoformat() == "2026-10-19T05:00:00-03:00"
    assert fim.isoformat() == "2026-10-26T05:00:00-03:00"
    assert calcular_janela("20261102")[0] == fim
    with TestCase().assertRaises(ValueError):
        calcular_janela("20261027")
    with TestCase().assertRaises(ValueError):
        calcular_janela("2026-10-26")


def test_preparar_features() -> None:
    """Verifica ordem, conversao numerica, infinitos e features ausentes."""
    from unittest import TestCase

    base = pd.DataFrame({"b": ["2", "invalido"], "a": [np.inf, "1"]})
    saida = preparar_features(base, ["a", "b"])
    assert list(saida.columns) == ["a", "b"]
    assert saida.loc[0, "b"] == 2.0
    assert saida.loc[1, "a"] == 1.0
    assert pd.isna(saida.loc[0, "a"])
    assert pd.isna(saida.loc[1, "b"])
    assert all(tipo == np.dtype("float64") for tipo in saida.dtypes)
    assert base.loc[1, "b"] == "invalido"
    with TestCase().assertRaises(ValueError):
        preparar_features(base, ["ausente"])
    with TestCase().assertRaises(ValueError):
        preparar_features(base, ["a", "a"])


def test_calcular_gh() -> None:
    """Confere os limites dos GHs e rejeita cortes/valores invalidos."""
    from unittest import TestCase

    np.testing.assert_array_equal(
        calcular_gh([0.0, 0.2, 0.2001, 0.8, 0.8001, 1.0], [0.2, 0.8]),
        [1, 1, 2, 2, 3, 3],
    )
    assert len(calcular_gh([], [0.2, 0.8])) == 0
    with TestCase().assertRaises(ValueError):
        calcular_gh([0.5], [0.8, 0.2])
    with TestCase().assertRaises(ValueError):
        calcular_gh([np.nan], [0.2, 0.8])
    with TestCase().assertRaises(ValueError):
        calcular_gh([1.1], [0.2, 0.8])


def test_adicionar_faixas() -> None:
    """Confere rank medio dos empates e classificacao em faixas."""
    base = pd.DataFrame({"prob1": [0.9, 0.9, 0.7, 0.1]})
    saida = adicionar_faixas(base)
    np.testing.assert_allclose(saida["rank_pct"], [0.375, 0.375, 0.75, 1.0])
    assert saida["faixa"].tolist() == [
        "Top 50%", "Top 50%", "Bottom 50%", "Bottom 50%"
    ]
    assert "faixa" not in base.columns
    assert adicionar_faixas(base.head(0)).empty


def test_atribuir_grupos() -> None:
    """Verifica grupo unico por CPF, controle por score e proporcao 90/10."""
    base = pd.DataFrame({
        "cpf": [f"{i:011d}" for i in range(1, 101)],
        COLUNA_PROPOSTA: [str(i) for i in range(1, 101)],
        "prob1": [0.9] * 100,
        "faixa": ["Top 20%"] * 100,
    })
    saida = atribuir_grupos(base, "20261026", 0.90)
    assert saida["TesteDigito"].eq(0).sum() == 90
    assert saida["TesteDigito"].eq(1).sum() == 10
    repetido = atribuir_grupos(base.iloc[::-1], "20261026", 0.90)
    pd.testing.assert_series_equal(
        saida.set_index("cpf")["TesteDigito"].sort_index(),
        repetido.set_index("cpf")["TesteDigito"].sort_index(),
    )
    adicionais = pd.DataFrame({
        "cpf": ["00000000001"],
        COLUNA_PROPOSTA: ["101"],
        "prob1": [0.01],
        "faixa": ["Bottom 50%"],
    })
    saida = atribuir_grupos(
        pd.concat([base, adicionais], ignore_index=True), "20261026", 0.90
    )
    assert len(saida) == 101
    assert saida.groupby("cpf")["TesteDigito"].nunique().max() == 1
    assert saida.loc[saida["cpf"] == "00000000001", "TesteDigito"].eq(1).all()
    assert atribuir_grupos(base.head(0), "20261026").empty


def test_validar_bundle() -> None:
    """Valida os componentes do bundle e a identificacao da classe 1."""
    from unittest import TestCase
    from unittest.mock import Mock

    modelo = Mock()
    modelo.classes_ = np.array([0, 1])
    bundle = {
        "model_pipeline": modelo,
        "feature_cols": [f"f_{i}" for i in range(63)],
        "splits": np.array([0.2, 0.8]),
        "calibrator": None,
    }
    assert validar_bundle(bundle) is None
    with TestCase().assertRaises(ValueError):
        validar_bundle({k: v for k, v in bundle.items() if k != "splits"})
    with TestCase().assertRaises(ValueError):
        validar_bundle({**bundle, "feature_cols": ["f_0"]})
    modelo.classes_ = np.array([0, 2])
    with TestCase().assertRaises(ValueError):
        validar_bundle(bundle)


def test__carregar_bundle_run() -> None:
    """Valida o caminho do artefato e o cache sem acessar o MLflow remoto."""
    from unittest.mock import Mock, patch

    modelo = Mock()
    modelo.classes_ = np.array([0, 1])
    bundle = {
        "model_pipeline": modelo,
        "feature_cols": [f"f_{i}" for i in range(63)],
        "splits": np.array([0.2, 0.8]),
        "calibrator": None,
    }
    _carregar_bundle_run.cache_clear()
    try:
        with patch(
            "mlflow.artifacts.download_artifacts",
            return_value="/tmp/bundle_unitario.joblib",
        ) as download:
            with patch.object(joblib, "load", return_value=bundle) as carga:
                assert _carregar_bundle_run("run_unitaria") is bundle
                assert _carregar_bundle_run("run_unitaria") is bundle
                download.assert_called_once()
                assert download.call_args.kwargs["run_id"] == "run_unitaria"
                caminho = download.call_args.kwargs["artifact_path"]
                assert caminho == BUNDLE_ARTIFACT
                carga.assert_called_once_with("/tmp/bundle_unitario.joblib")
    finally:
        _carregar_bundle_run.cache_clear()


def test_carregar_bundle() -> None:
    """Confere a run do contrato e rejeita configuracao ausente."""
    from unittest import TestCase
    from unittest.mock import Mock, patch

    esperado = {"origem": "teste"}
    carga = Mock(return_value=esperado)
    with patch.dict(globals(), {
        "CONTRACT": {"mlflow": [{"run_id": "run_unitaria"}]},
        "_carregar_bundle_run": carga,
    }):
        assert carregar_bundle() is esperado
        carga.assert_called_once_with("run_unitaria")
    with patch.dict(globals(), {"CONTRACT": {}}):
        with TestCase().assertRaises(ValueError):
            carregar_bundle()


def test__spark() -> None:
    """Confere a sessao existente e o erro quando nao ha Spark ativo."""
    from unittest import TestCase
    from unittest.mock import Mock, patch
    from pyspark.sql import SparkSession

    sessao = Mock()
    with patch.dict(globals(), {"spark": sessao}):
        assert _spark() is sessao
    with patch.dict(globals(), {"spark": None}):
        with patch.object(
            SparkSession, "getActiveSession", return_value=sessao
        ):
            assert _spark() is sessao
        with patch.object(SparkSession, "getActiveSession", return_value=None):
            with TestCase().assertRaises(RuntimeError):
                _spark()


def test_dataprep_read_table_bigquery() -> None:
    """Valida as opcoes do conector com leitura externa simulada."""
    from unittest.mock import Mock, patch
    from kraken.variables import KrakenVariables

    leitor = Mock()
    leitor.format.return_value = leitor
    leitor.option.return_value = leitor
    esperado = object()
    leitor.load.return_value = esperado
    sessao = Mock()
    sessao.read = leitor
    with patch.dict(globals(), {"_spark": Mock(return_value=sessao)}):
        resultado = dataprep_read_table_bigquery("SELECT 1 AS valor")
    assert resultado is esperado
    leitor.format.assert_called_once_with("bigquery")
    opcoes = {
        chamada.args[0]: chamada.args[1]
        for chamada in leitor.option.call_args_list
    }
    assert opcoes["query"] == "SELECT 1 AS valor"
    assert opcoes["parentProject"] == KrakenVariables.GCP_PROJECT
    assert opcoes["viewsEnabled"] == "true"
    assert opcoes["materializationProject"] == "bv-pdbd-prd"
    assert opcoes["materializationDataset"] == "stg_tecn_dataexport"
    leitor.load.assert_called_once_with()


def test_montar_query_features() -> None:
    """Confere substituicoes da SQL, janela DATE e exclusao de concorrencia."""
    from unittest import TestCase

    features = list(FEATURE_DESCRIPTIONS)
    query = montar_query_features("20261026", features)
    assert "DATE('2026-10-19')" in query
    assert "DATE('2026-10-26')" in query
    assert "entrada_referencia >= DATE('2026-10-19')" in query
    assert "entrada_referencia < DATE('2026-10-26')" in query
    assert "CAST(FlConcorrente AS INT64) = 1" in query
    assert "cpfs_concorrencia" in query
    for marcador in (
        "__INICIO__", "__FIM__", "__DATA_REFERENCIA__", "__CAMPO_INFORMACAO__"
    ):
        assert marcador not in query
    alternativas = [
        "cdtipinformacaoitlr" if f == "cdtipinformacaoltlr" else f
        for f in features
    ]
    assert "psa.cdtipinformacaoitlr" in montar_query_features(
        "20261026", alternativas
    )
    with TestCase().assertRaises(ValueError):
        montar_query_features("20261026", ["feature_inexistente"])


def test_get_base_egv_apnp() -> None:
    """Valida safra, SQL e leitura sem consultar tabelas reais."""
    from unittest.mock import Mock, patch

    esperado = object()
    montar = Mock(return_value="SELECT 1 AS valor")
    ler = Mock(return_value=esperado)
    with patch.dict(globals(), {
        "get_safra": Mock(return_value="20261026"),
        "montar_query_features": montar,
        "dataprep_read_table_bigquery": ler,
    }):
        assert get_base_egv_apnp(["f_0"]) is esperado
    montar.assert_called_once_with("20261026", ["f_0"])
    ler.assert_called_once_with("SELECT 1 AS valor")


def test__normalizar_chaves() -> None:
    """Preserva zeros dos identificadores e rejeita CPF/proposta invalidos."""
    from unittest import TestCase

    base = pd.DataFrame({
        "cpf": [" 00000000001 ", "00000000002"],
        COLUNA_PROPOSTA: [" 123 ", "456"],
    })
    saida = _normalizar_chaves(base)
    assert saida["cpf"].tolist() == ["00000000001", "00000000002"]
    assert saida[COLUNA_PROPOSTA].tolist() == ["123", "456"]
    invalido = base.copy()
    invalido.loc[0, "cpf"] = "1"
    with TestCase().assertRaises(ValueError):
        _normalizar_chaves(invalido)
    invalido = base.copy()
    invalido[COLUNA_PROPOSTA] = ["123", "123"]
    with TestCase().assertRaises(ValueError):
        _normalizar_chaves(invalido)
    invalido = base.copy()
    invalido.loc[0, "cpf"] = None
    with TestCase().assertRaises(ValueError):
        _normalizar_chaves(invalido)


def test_escorar_dataframe() -> None:
    """Confere classe positiva, limiar, GH, calibracao e target de treino."""
    from unittest.mock import Mock

    features = [f"f_{i}" for i in range(63)]
    base = pd.DataFrame({nome: [1.0, 2.0] for nome in features})
    base["cpf"] = ["00000000001", "00000000002"]
    base[COLUNA_PROPOSTA] = ["123", "456"]
    base["target"] = [1, 0]
    modelo = Mock()
    modelo.classes_ = np.array([1, 0])
    modelo.predict_proba.return_value = np.array([[0.5, 0.5], [0.2, 0.8]])
    bundle = {
        "model_pipeline": modelo,
        "feature_cols": features,
        "splits": np.array([0.2, 0.8]),
        "calibrator": None,
    }
    saida = escorar_dataframe(base, bundle, "20261026", "modelo_unitario")
    np.testing.assert_allclose(saida["prob1"], [0.5, 0.2])
    np.testing.assert_array_equal(saida["predito"], [1, 0])
    np.testing.assert_array_equal(saida["gh"], [2, 1])
    assert saida["target"].isna().all()
    assert saida["idchavemodelo"].tolist() == ["123", "456"]
    assert list(modelo.predict_proba.call_args.args[0].columns) == features
    treino = escorar_dataframe(
        base, bundle, "20261026", "modelo_unitario", treino=True
    )
    assert treino["target"].tolist() == [1, 0]
    assert treino["TesteDigito"].isna().all()
    calibrador = Mock()
    calibrador.predict.return_value = np.array([0.7, 0.9])
    calibrado = escorar_dataframe(
        base, {**bundle, "calibrator": calibrador},
        "20261026", "modelo_unitario",
    )
    np.testing.assert_allclose(calibrado["prob1"], [0.7, 0.9])
    calibrador.predict.assert_called_once()


def test_schema_big() -> None:
    """Verifica nomes, tipos e nulabilidade do schema de referencia."""
    from pyspark.sql import types as T

    schema = schema_big(["f_0", "f_1"])
    assert len(schema.names) == len(set(schema.names))
    assert isinstance(schema["cpf"].dataType, T.StringType)
    assert isinstance(schema["idchavemodelo"].dataType, T.StringType)
    assert isinstance(schema["target"].dataType, T.IntegerType)
    assert isinstance(schema["gh"].dataType, T.IntegerType)
    assert isinstance(schema["prob1"].dataType, T.DoubleType)
    assert isinstance(schema["f_0"].dataType, T.DoubleType)
    assert all(campo.nullable for campo in schema.fields)


def test_para_spark_big() -> None:
    """Confere conversao de tipos/nulos usando criacao Spark simulada."""
    from unittest.mock import Mock, patch
    from pyspark.sql import types as T

    schema = schema_big(["f_0"])
    valores = {}
    for campo in schema.fields:
        if isinstance(campo.dataType, T.StringType):
            valores[campo.name] = "texto"
        elif isinstance(campo.dataType, T.IntegerType):
            valores[campo.name] = np.int64(1)
        else:
            valores[campo.name] = np.float64(0.5)
    valores.update({"cpf": "00000000001", "target": None, "f_0": np.nan})
    sessao = Mock()
    esperado = object()
    sessao.createDataFrame.return_value = esperado
    with patch.dict(globals(), {"_spark": Mock(return_value=sessao)}):
        assert para_spark_big(pd.DataFrame([valores]), ["f_0"]) is esperado
    chamada = sessao.createDataFrame.call_args
    registros = chamada.args[0]
    schema_obtido = chamada.kwargs["schema"]
    assert schema_obtido == schema
    assert len(registros) == 1
    registro = dict(zip(schema.names, registros[0]))
    assert registro["cpf"] == "00000000001"
    assert registro["target"] is None
    assert registro["f_0"] is None
    assert type(registro["gh"]) is int
    assert type(registro["prob1"]) is float


def test_montar_contrato() -> None:
    """Verifica identidade preservada, descricoes e pares de funcoes/testes."""
    from unittest.mock import patch

    original = {
        "model_grc": "modelo_unitario",
        "sigla": "sigla_unitaria",
        "mlflow": [{"run_id": "run_unitaria"}],
        "path_to_data_train": "/Volumes/teste/referencia",
        "build_schedule": "0 0 8 ? * MON *",
        "build_libraries": ["joblib==1.0.0"],
        "used_tables": ["bigquery:projeto.dataset.tabela"],
    }
    esperado = deepcopy(original)
    contexto = {"CONTRACT": original, "SIGLA_COMPONENTE": None}
    with patch.dict(globals(), contexto):
        resultado = montar_contrato(
            {"feature_cols": list(FEATURE_DESCRIPTIONS)}
        )
    assert original == esperado
    assert resultado["mlflow"] == original["mlflow"]
    assert resultado["path_to_data_train"] == original["path_to_data_train"]
    assert resultado["build_schedule"] == original["build_schedule"]
    assert resultado["used_tables"] == original["used_tables"]
    assert set(original["build_libraries"]).issubset(
        resultado["build_libraries"]
    )
    assert resultado["big_dvry"]["keys"] == ["idchavemodelo", "safra"]
    assert resultado["dataprep"] == list(FUNCOES_DATAPREP)
    assert resultado["test_dataprep"] == [
        f"test_{nome}" for nome in FUNCOES_DATAPREP
    ]
    descricoes = resultado["model_output_description"]["fields_description"]
    assert set(FEATURE_DESCRIPTIONS).issubset(descricoes)
    for nome in (*resultado["dataprep"], *resultado["test_dataprep"]):
        assert callable(globals().get(nome)), nome
        assert globals()[nome].__doc__, nome


def test_validar_contrato() -> None:
    """Valida a chamada ao Kraken com o executor simulado, sem publicar."""
    from unittest.mock import Mock, patch

    bundle = {"origem": "teste"}
    contrato_teste = {"model_grc": "modelo_unitario"}
    montar = Mock(return_value=contrato_teste)
    with patch.dict(globals(), {
        "CONTRACT": {},
        "carregar_bundle": Mock(return_value=bundle),
        "montar_contrato": montar,
    }):
        with patch("kraken.contract.DeployContract") as classe:
            classe.return_value.create.return_value = "validacao_simulada"
            assert validar_contrato() == "validacao_simulada"
            montar.assert_called_once_with(bundle)
            classe.assert_called_once()
            assert classe.call_args.args[0] is globals()
            classe.return_value.create.assert_called_once_with(contrato_teste)
